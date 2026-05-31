"""Train the CTF policy with the local PPO/MAPPO implementation."""

from __future__ import annotations

import os
import random
import sys
from typing import Optional

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_SCRIPT_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

import numpy as np
import torch

from game_field_gpu import GPUCTFVecEnv, GPUFieldConfig, VEC_OBS_DIM
from rl.config.ppo_config import PPOConfig, TrainMode
from rl.custom_ppo import CustomPPOTrainer
from rl.custom_ppo.trainer_audit import log_input_dim_contract
from rl.curriculum import CurriculumState, jacob_paper_curriculum_state, phase_from_tag
from rl.global_state import GLOBAL_STATE_DIM
from rl.latent_marl import CONTEXT_STATE_DIM
from rl.training.config_validation import (
    EVAL_ONLY_TRAINING_OPPONENT_TAGS,
    _normalize_train_mode,
    _strip_eval_only_opponents_from_training_pool,
    normalize_and_validate_training_config,
)
from rl.training.env_factory import (  # noqa: F401  -- re-exported for tools/critic_ceiling.py and other external callers
    _apply_initial_opponent_params,
    _gpu_env_reward_kwargs,
    build_training_env,
)
from rl.training.run_artifacts import (
    _RunLock,
    _acquire_run_lock,
    _find_git_root,
    _git_metadata,
    _json_safe,
    _metrics_csv_nonempty,
    _pid_is_running,
    _read_run_lock,
    _rotate_csv_aside,
    _run_config_json_path,
    write_run_config_json,
)

# Re-export so existing ``from rl.train_ppo import ...`` call sites (presets,
# tools, tests, archived log preflights) keep working without rewrites. The
# canonical homes are now ``rl.config.ppo_config``,
# ``rl.training.run_artifacts``, ``rl.training.env_factory``, and
# ``rl.training.config_validation`` -- prefer those paths in new code.
__all__ = [
    "PPOConfig",
    "TrainMode",
    "train_ppo",
    "write_run_config_json",
    "_apply_initial_opponent_params",
    "EVAL_ONLY_TRAINING_OPPONENT_TAGS",
    "_strip_eval_only_opponents_from_training_pool",
    "_normalize_train_mode",
    "normalize_and_validate_training_config",
]


def _resolve_2v2_checkpoint(filename: str) -> Optional[str]:
    """Find ``checkpoints/2v2/<filename>`` whether cwd is repo root or ``AICTFProject``."""
    cwd = os.getcwd()
    candidates = (
        os.path.join(_PARENT_DIR, "checkpoints", "2v2", filename),
        os.path.join(cwd, "checkpoints", "2v2", filename),
        os.path.join(cwd, "AICTFProject", "checkpoints", "2v2", filename),
        os.path.join(os.path.dirname(_PARENT_DIR), "AICTFProject", "checkpoints", "2v2", filename),
    )
    for raw in candidates:
        path = os.path.normpath(raw)
        if os.path.isfile(path):
            return path
    return None


def set_global_seed(seed: int, torch_seed: bool = True, deterministic: bool = False) -> None:
    """Set Python, NumPy, and Torch seeds."""
    random.seed(int(seed))
    np.random.seed(int(seed))
    if torch_seed:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def _apply_training_preset(cfg: PPOConfig, preset: str) -> PPOConfig:
    """Apply named high-level presets for repeatable training recipes."""
    from rl.presets import apply_preset

    return apply_preset(cfg, preset)


# Default ``python rl/train_ppo.py`` recipe when ``--preset`` is omitted: plan-faithful
# latent with sparse persistence and entropy. Pass ``--preset none`` to skip.
DEFAULT_CLI_TRAINING_PRESET = "plan_faithful_latent_persist_entropy"


def train_ppo(cfg: Optional[PPOConfig] = None) -> None:
    """Run the default local PPO/MAPPO training path."""
    cfg = normalize_and_validate_training_config(cfg or PPOConfig())

    set_global_seed(cfg.seed, torch_seed=True, deterministic=cfg.use_deterministic)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    max_agents = max(1, int(getattr(cfg, "max_blue_agents", 2)))
    team_size = _agents_suffix(max_agents)
    curriculum: CurriculumState | None = None
    initial_opponent_tag = str(cfg.fixed_opponent_tag).upper()
    initial_phase = phase_from_tag(initial_opponent_tag)
    if cfg.mode == TrainMode.CURRICULUM.value:
        curriculum = jacob_paper_curriculum_state(max_agents)
        initial_opponent_tag = curriculum.phase
        initial_phase = curriculum.phase
    print(f"[PPO] Agents: {max_agents} per team ({team_size}) | mode={cfg.mode} | run_tag={cfg.run_tag!r}")
    print("[PPO] Algorithm backend: custom local PPO")
    print(f"[PPO] Total timesteps: {int(cfg.total_timesteps):,}")
    base_gs_dim = GLOBAL_STATE_DIM
    temp_ctx_dim = CONTEXT_STATE_DIM if bool(getattr(cfg, "use_latent_strategy", False)) else 0
    q_phi_dim = CONTEXT_STATE_DIM if bool(getattr(cfg, "use_latent_strategy", False)) else 0
    crit_dim = CONTEXT_STATE_DIM if bool(getattr(cfg, "use_latent_strategy", False)) else GLOBAL_STATE_DIM
    actor_cnn_feat = int(getattr(cfg, "actor_cnn_feature_dim", 128))
    z_embed = int(getattr(cfg, "latent_z_embed_dim", 16))
    act_dim = (actor_cnn_feat + 20 + z_embed) if bool(getattr(cfg, "use_latent_strategy", False)) else (actor_cnn_feat + 20)
    print(
        f"[PPO] Input dims: base_global_state_dim={base_gs_dim} "
        f"temporal_context_dim={temp_ctx_dim} "
        f"q_phi_input_dim={q_phi_dim} "
        f"critic_context_dim={crit_dim} "
        f"actor_input_dim={act_dim}"
    )
    print(f"[PPO] Actor CNN feature dim: {int(getattr(cfg, 'actor_cnn_feature_dim', 128))}")
    print(f"[PPO] Map set: {str(getattr(cfg, 'map_set', 'train')).lower()}")
    if bool(getattr(cfg, "train_domain_randomization", False)):
        print(
            "[PPO] Domain randomization: ON "
            f"(sensor_noise_sigma max={float(getattr(cfg, 'dr_sensor_noise_sigma_max', 0.0)):.3f}, "
            f"sensor_dropout max={float(getattr(cfg, 'dr_sensor_dropout_max', 0.0)):.3f}, "
            f"blue_speed_jitter={float(getattr(cfg, 'dr_blue_speed_jitter', 0.0)):.3f}; "
            "blue-policy side only, slowdown-only speed scale)"
        )
    if curriculum is not None:
        print("[PPO] Training profile: curriculum baseline")
    elif bool(getattr(cfg, "use_latent_strategy", False)):
        print("[PPO] Training profile: default latent (Summer implementation)")
    else:
        print("[PPO] Training profile: no-latent baseline")
    if bool(getattr(cfg, "normalize_returns", False)):
        print("[PPO] Return normalization: enabled for critic targets/predictions; GAE uses denormalized values.")
    decay_steps = max(0, int(getattr(cfg, "reward_shaping_decay_steps", 0) or 0))
    if decay_steps > 0:
        print(
            "[PPO] Reward shaping decay: "
            f"coef {float(getattr(cfg, 'reward_shaping_coef_start', 1.0)):.3f} -> "
            f"{float(getattr(cfg, 'reward_shaping_coef_end', 1.0)):.3f} "
            f"over {decay_steps:,} steps before RewardConfig weighting/scaling."
        )
    if cfg.mode == TrainMode.OPPONENT_POOL.value or bool(getattr(cfg, "opponent_randomize", False)):
        label = "OPPONENT_POOL mode" if cfg.mode == TrainMode.OPPONENT_POOL.value else "opponent_randomize flag"
        print(
            "[PPO] Opponent randomization: enabled "
            f"({label}; uniform per completed episode over pool={list(cfg.opponent_pool)}; "
            "pre-reset hook \u2014 opponent logged for each episode is the one played during that episode)."
        )
    if curriculum is not None:
        print(
            "[PPO] Jacob paper curriculum: enabled "
            "(SCRIPTED:OP1 -> SCRIPTED:OP2 -> SCRIPTED:OP3; scripted-only curriculum)."
        )
        print(
            "[PPO] Curriculum gates: "
            f"min_episodes={curriculum.config.min_episodes}, "
            f"min_winrate={curriculum.config.min_winrate}, "
            f"windows={curriculum.config.winrate_window_by_phase}."
        )
    if bool(cfg.use_latent_strategy):
        interval = int(getattr(cfg, "latent_resample_every_n", 0) or 0)
        fixed = bool(getattr(cfg, "fixed_latent_strategy", False))
        interval_label = "fixed" if fixed else ("episode start" if interval <= 0 else f"every {interval} decision steps")
        on_flag = bool(getattr(cfg, "latent_resample_on_flag", False)) and not fixed
        lam_kl = 0.0 if fixed else float(getattr(cfg, "latent_kl_consecutive", 0.0) or 0.0)
        fixed_label = f", fixed_z={int(getattr(cfg, 'fixed_latent_strategy_id', 0) or 0)}" if fixed else ""
        h_obj = getattr(cfg, "latent_entropy_objective", "maximize") or "maximize"
        aux_head = bool(getattr(cfg, "latent_strategy_aux_return_head", False))
        episode_credit = bool(getattr(cfg, "latent_episode_strategy_ppo", False))
        print(
            "[PPO] Latent team strategy: enabled "
            f"(K={int(cfg.latent_k)}, sample={interval_label}, on_flag={on_flag}, "
            f"lambda_p={float(cfg.latent_lam_p):.4f}, lambda_H={float(cfg.latent_lam_h):.4f} "
            f"(H:{h_obj}), "
            f"lambda_KL={lam_kl:.4f}, strategy_ppo_coef={float(cfg.latent_strategy_ppo_coef):.3f}, "
            f"episode_credit={episode_credit}, episode_coef={float(getattr(cfg, 'latent_episode_strategy_coef', 0.0)):.3f}, "
            f"aux_return_head={aux_head}, aux_return_coef={float(cfg.latent_strategy_aux_return_coef):.3f}, "
            f"tau={float(cfg.latent_strategy_tau):.3f}, "
            f"GAE_reset_on_z_change={bool(getattr(cfg, 'latent_gae_reset_on_z_change', True))}, "
            f"bootstrap_z_deterministic={bool(getattr(cfg, 'latent_bootstrap_z_deterministic', True))}"
            f"{fixed_label})"
        )
        if (not fixed) and interval <= 0 and float(getattr(cfg, "latent_lam_p", 0.0) or 0.0) > 0.0:
            print(
                "[PPO] NOTE: latent_lam_p is active only on sparse mid-episode resamples; "
                "with sample=episode start it has near-zero training effect."
            )
        if fixed:
            print("[PPO] Fixed-latent baseline: q_phi sampling/losses are bypassed; actor/critic receive one z ID.")
    else:
        print("[PPO] Latent team strategy: disabled (vanilla local PPO baseline).")
    print(f"[PPO] Checkpoint dir: {cfg.checkpoint_dir}")
    if bool(getattr(cfg, "enable_metrics_csv", True)):
        if not cfg.metrics_csv_path:
            cfg.metrics_csv_path = os.path.join(cfg.checkpoint_dir, f"{cfg.run_tag}_metrics.csv")
        if not cfg.episode_csv_path:
            cfg.episode_csv_path = os.path.join(cfg.checkpoint_dir, f"{cfg.run_tag}_episodes.csv")
        strategy_experience_enabled = bool(getattr(cfg, "use_latent_strategy", False)) and bool(
            getattr(cfg, "latent_episode_strategy_ppo", False)
        )
        if strategy_experience_enabled and not cfg.strategy_experience_csv_path:
            cfg.strategy_experience_csv_path = os.path.join(
                cfg.checkpoint_dir, f"{cfg.run_tag}_strategy_experience.csv"
            )
        elif not strategy_experience_enabled:
            cfg.strategy_experience_csv_path = None
        print(f"[PPO] Update metrics CSV: {cfg.metrics_csv_path}")
        print(f"[PPO] Episode metrics CSV: {cfg.episode_csv_path}")
        if strategy_experience_enabled and cfg.strategy_experience_csv_path:
            print(f"[PPO] Strategy experience CSV: {cfg.strategy_experience_csv_path}")
        _e3p = str(getattr(cfg, "e3_step_telemetry_path", "") or "").strip()
        if _e3p:
            print(f"[PPO] E3 step telemetry CSV (per-step z, team_phase, behavior telemetry, buckets, MI-related fields): {_e3p}")
        if (not cfg.fresh_metrics_csv) and (not cfg.load_path) and (
            _metrics_csv_nonempty(cfg.metrics_csv_path)
            or _metrics_csv_nonempty(cfg.episode_csv_path)
            or _metrics_csv_nonempty(cfg.strategy_experience_csv_path)
        ):
            print(
                "[PPO] WARNING: metrics/episode/strategy-experience CSV already exists; this run will APPEND. "
                "That duplicates `timestep`/update indices if you reused --run-tag. "
                "Use --fresh-metrics-csv (rotates old files aside) or a new --run-tag."
            )
    else:
        cfg.metrics_csv_path = None
        cfg.episode_csv_path = None
        cfg.strategy_experience_csv_path = None
        print("[PPO] Metrics CSV logging disabled.")
    run_lock = _acquire_run_lock(cfg)
    if bool(getattr(cfg, "enable_metrics_csv", True)) and cfg.fresh_metrics_csv:
        _rotate_csv_aside(cfg.metrics_csv_path, label="metrics")
        _rotate_csv_aside(cfg.episode_csv_path, label="episode")
        _rotate_csv_aside(cfg.strategy_experience_csv_path, label="strategy experience")
    try:
        rc_path = write_run_config_json(cfg)
        print(f"[PPO] Run config written: {rc_path}")
    except Exception as exc:
        print(f"[PPO] WARNING: could not write run config JSON: {exc}")
    elog = int(getattr(cfg, "episode_log_every", 0) or 0)
    if elog > 0:
        mode_label = "curriculum phase" if curriculum is not None else "scripted opponent tag"
        if curriculum is not None:
            tag_label = initial_opponent_tag
        elif cfg.mode == TrainMode.OPPONENT_POOL.value or bool(getattr(cfg, "opponent_randomize", False)):
            tag_label = f"randomized pool {list(cfg.opponent_pool)}"
        else:
            tag_label = str(cfg.fixed_opponent_tag).upper()
        print(
            f"[PPO] Episode stats: every {elog} completed episode(s) print W/L/D and WR "
            f"(mode={cfg.mode}, {mode_label}={tag_label})."
        )
    else:
        print("[PPO] Episode stats logging disabled (episode_log_every=0).")

    if max_agents == 6:
        cfg.n_envs = min(int(cfg.n_envs), 1)
        cfg.n_steps = min(int(cfg.n_steps), 512)
        cfg.max_decision_steps = min(int(cfg.max_decision_steps), 400)

    if str(cfg.device).lower().startswith("cuda"):
        try:
            torch.zeros(1, device=cfg.device)
        except RuntimeError as exc:
            print(f"[PPO] CUDA unavailable for this torch build ({exc}). Falling back to CPU.")
            cfg.device = "cpu"

    env = build_training_env(
        cfg,
        initial_phase=initial_phase,
        initial_opponent_tag=initial_opponent_tag,
    )
    try:
        learning_rate = float(cfg.learning_rate)
        ent_coef = float(cfg.ent_coef)
        clip_range = float(cfg.clip_range)
        n_epochs = int(cfg.n_epochs)
        batch_size = int(cfg.batch_size)

        if bool(getattr(cfg, "use_stable_marl_ppo", False)):
            learning_rate = 1.5e-4
            ent_coef = 0.005
            clip_range = 0.10
            n_epochs = 2
            batch_size = 1024
            print(
                "[PPO] Optional stable-MARL override (use_stable_marl_ppo=True; not in Word spec): "
                "lr=1.5e-4, n_epochs=2, clip=0.10, ent=0.005, batch_size=1024."
            )
        if max_agents > 2:
            learning_rate *= 0.75
            print(f"[PPO] {team_size}: using lr={learning_rate:.2e} for stability.")

        rollout_size = max(1, int(cfg.n_steps) * max(1, int(cfg.n_envs)))
        if batch_size > rollout_size:
            batch_size = rollout_size
            print(f"[PPO] Adjusting batch_size to rollout size: {batch_size}.")

        trainer = CustomPPOTrainer(
            env,
            cfg,
            learning_rate=learning_rate,
            clip_range=clip_range,
            ent_coef=ent_coef,
            n_epochs=n_epochs,
            batch_size=batch_size,
            value_clip_range=getattr(cfg, "clip_range_vf", clip_range),
            curriculum=curriculum,
        )
        log_input_dim_contract(trainer)
        if cfg.load_path and os.path.isfile(cfg.load_path):
            print(f"[PPO] Resuming checkpoint: {cfg.load_path}")
            trainer.load(cfg.load_path)
        try:
            stats = trainer.learn(total_timesteps=int(cfg.total_timesteps))
        except KeyboardInterrupt:
            interrupt_path = os.path.join(
                cfg.checkpoint_dir,
                f"interrupt_{cfg.run_tag}_{int(getattr(trainer, 'global_step', 0))}.zip",
            )
            trainer.save(interrupt_path)
            print(f"[PPO] KeyboardInterrupt: emergency checkpoint saved to: {interrupt_path}")
            raise
        final_path = os.path.join(cfg.checkpoint_dir, f"final_{cfg.run_tag}.zip")
        trainer.save(final_path)
        if stats:
            print(
                "[PPO] Final stats: "
                f"policy_loss={stats.get('policy_loss', 0.0):.4f}, "
                f"value_loss={stats.get('value_loss', 0.0):.4f}, "
                f"approx_kl={stats.get('approx_kl', 0.0):.5f}"
            )
        print(f"[PPO] Training complete. Final checkpoint saved to: {final_path}")
    finally:
        env.close()
        run_lock.release()


def run_verify_4v4(num_episodes: int = 10) -> None:
    """Run random-action verification episodes at 4v4."""
    set_global_seed(42)
    cfg = GPUFieldConfig(n_envs=1, n_agents_per_team=4, max_decision_steps=400, device="cpu", seed=42)
    env = GPUCTFVecEnv(cfg)
    try:
        for ep in range(num_episodes):
            env.reset()
            done = False
            steps = 0
            while not done and steps < 800:
                env.step_async(np.asarray(env.action_space.sample(), dtype=np.int64)[None, :])
                _, _, done_arr, _ = env.step_wait()
                done = bool(done_arr[0])
                steps += 1
            print(f"[Verify-4v4] episode {ep + 1}/{num_episodes} steps={steps} done={done}")
    finally:
        env.close()


def run_test_vec_schema() -> None:
    """Verify GPU core observation and global-state schemas."""
    cfg = GPUFieldConfig(n_envs=1, n_agents_per_team=2, device="cpu", seed=42)
    env = GPUCTFVecEnv(cfg)
    try:
        obs = env.reset()
        vec = obs["vec"]
        state = env.state()
        assert vec.dtype == np.float32, f"vec.dtype {vec.dtype}, expected float32"
        assert vec.ndim == 3 and vec.shape[2] == VEC_OBS_DIM, (
            f"vec.shape {vec.shape}, expected (B,N,{VEC_OBS_DIM})"
        )
        assert np.all(np.isfinite(vec)), "vec has non-finite values"
        assert state.shape == (1, GLOBAL_STATE_DIM), f"state.shape {state.shape}"
        print("[test-vec-schema] obs vec and global state schemas OK.")
    finally:
        env.close()


def _agents_suffix(n_agents: int) -> str:
    n = max(1, min(int(n_agents), 16))
    return f"{n}v{n}"


def _ensure_run_tag_has_agent_suffix(run_tag: str, n_agents: int) -> str:
    suffix = _agents_suffix(n_agents)
    tag_suffix = f"_{suffix}"
    for existing in ("_2v2", "_4v4", "_6v6", "_8v8"):
        if run_tag.endswith(existing):
            run_tag = run_tag[: -len(existing)]
            break
    if not run_tag.endswith(tag_suffix):
        run_tag = run_tag.rstrip("_") + tag_suffix
    return run_tag


def _default_run_tag_for_mode(
    mode: str,
    fixed_opponent_tag: str = "OP3",
    n_agents: int = 2,
    *,
    latent: bool = True,
) -> str:
    suffix = _agents_suffix(n_agents)
    family = "ppo_latent" if latent else "ppo_custom"
    if _normalize_train_mode(mode) == TrainMode.CURRICULUM.value:
        return f"{family}_curriculum_{suffix}"
    if _normalize_train_mode(mode) == TrainMode.OPPONENT_POOL.value:
        return f"{family}_opp_pool_{suffix}"
    if _normalize_train_mode(mode) == TrainMode.FIXED_OPPONENT.value:
        return f"{family}_fixed_{fixed_opponent_tag.lower()}_{suffix}"
    return f"{family}_{suffix}"


if __name__ == "__main__":
    from rl.training.cli import main

    main()
