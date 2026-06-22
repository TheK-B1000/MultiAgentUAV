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
from rl.training.banner import print_episode_stats_banner, print_training_banner
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
from rl.training.run_artifacts import (  # noqa: F401  -- re-exported for legacy import paths from tools/tests/presets
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


def _resolve_initial_opponent_and_phase(
    cfg: PPOConfig, max_agents: int
) -> tuple[Optional[CurriculumState], str, str]:
    """Pick the initial scripted opponent tag and phase for the run.

    In ``CURRICULUM`` mode this also constructs the :class:`CurriculumState`
    that the trainer will advance between phases.

    In ``OPPONENT_POOL`` mode (or with ``opponent_randomize=True``) the
    very first env reset uses ``set_next_opponent("SCRIPTED", initial_tag)``
    to seed the env before pool sampling takes over on subsequent resets.
    If ``cfg.fixed_opponent_tag`` is not in ``cfg.opponent_pool`` (the
    common case: the v4i1/v4i3/v5* chain leaves ``fixed_opponent_tag``
    at the ``"OP3"`` default while configuring a pool like
    ``("OP5", "OP6", "OP7")``), that first episode would leak the
    out-of-pool opponent into the very first telemetry slice, producing
    a misleading "OP3 slice" diagnostic even though the audit banner
    correctly reports the configured pool. To make the first episode
    consistent with the audit, fall back to the first pool entry when
    the legacy ``fixed_opponent_tag`` is out-of-pool. An explicit in-pool
    ``fixed_opponent_tag`` still wins (so users who want to seed a
    specific opener can do so).

    Returns ``(curriculum, initial_phase, initial_opponent_tag)``;
    ``curriculum`` is ``None`` outside curriculum mode.
    """
    curriculum: Optional[CurriculumState] = None
    initial_opponent_tag = str(cfg.fixed_opponent_tag).upper()
    initial_phase = phase_from_tag(initial_opponent_tag)
    if cfg.mode == TrainMode.CURRICULUM.value:
        curriculum = jacob_paper_curriculum_state(max_agents)
        initial_opponent_tag = curriculum.phase
        initial_phase = curriculum.phase
        return curriculum, initial_phase, initial_opponent_tag
    pool_mode = cfg.mode == TrainMode.OPPONENT_POOL.value or bool(
        getattr(cfg, "opponent_randomize", False)
    )
    if pool_mode:
        pool = tuple(str(tag).upper() for tag in (getattr(cfg, "opponent_pool", ()) or ()))
        if pool and initial_opponent_tag not in pool:
            initial_opponent_tag = pool[0]
            initial_phase = phase_from_tag(initial_opponent_tag)
    return curriculum, initial_phase, initial_opponent_tag


def _resolve_metrics_csv_paths(cfg: PPOConfig) -> None:
    """Resolve metrics / episode / strategy-experience CSV paths into ``cfg``.

    When metrics CSV logging is enabled, fills in any unset path from
    ``cfg.checkpoint_dir`` + ``cfg.run_tag``. When the strategy-experience CSV
    is not applicable (latent strategy or episode-credit disabled), clears it
    to ``None``. When metrics CSV logging is disabled, clears all three.

    Mutates ``cfg`` in place. Must run *before* :func:`print_training_banner`
    so the banner prints the same paths the trainer will later write to.
    """
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
        refresh_log_enabled = bool(getattr(cfg, "latent_v3i3_refresh_log_enabled", False))
        if refresh_log_enabled and not getattr(cfg, "latent_v3i3_refresh_log_path", None):
            cfg.latent_v3i3_refresh_log_path = os.path.join(
                cfg.checkpoint_dir, f"{cfg.run_tag}_refresh_log.csv"
            )
        elif not refresh_log_enabled:
            cfg.latent_v3i3_refresh_log_path = None
    else:
        cfg.metrics_csv_path = None
        cfg.episode_csv_path = None
        cfg.strategy_experience_csv_path = None
        cfg.latent_v3i3_refresh_log_path = None


def _rotate_fresh_run_telemetry(cfg: PPOConfig) -> None:
    """Rotate telemetry files that would otherwise be appended on a fresh run."""
    if not (bool(getattr(cfg, "enable_metrics_csv", True)) and cfg.fresh_metrics_csv):
        return
    _rotate_csv_aside(cfg.metrics_csv_path, label="metrics")
    _rotate_csv_aside(cfg.episode_csv_path, label="episode")
    _rotate_csv_aside(cfg.strategy_experience_csv_path, label="strategy experience")
    _rotate_csv_aside(
        getattr(cfg, "latent_v3i3_refresh_log_path", None), label="v3i3 refresh log"
    )
    _rotate_csv_aside(
        getattr(cfg, "e3_step_telemetry_path", None), label="E3 step telemetry"
    )


def _clamp_runtime_config_for_team_size(cfg: PPOConfig, max_agents: int) -> None:
    """Reduce rollout / episode-length knobs for the 6v6 / 8v8 memory profile."""
    if max_agents == 6:
        cfg.n_envs = min(int(cfg.n_envs), 1)
        cfg.n_steps = min(int(cfg.n_steps), 512)
        cfg.max_decision_steps = min(int(cfg.max_decision_steps), 400)


def _ensure_cuda_or_fallback(cfg: PPOConfig) -> None:
    """If ``cfg.device`` is CUDA but unavailable for this torch build, fall back to CPU."""
    if not str(cfg.device).lower().startswith("cuda"):
        return
    try:
        torch.zeros(1, device=cfg.device)
    except RuntimeError as exc:
        print(f"[PPO] CUDA unavailable for this torch build ({exc}). Falling back to CPU.")
        cfg.device = "cpu"


def train_ppo(cfg: Optional[PPOConfig] = None) -> None:
    """Run the default local PPO/MAPPO training path."""
    cfg = normalize_and_validate_training_config(cfg or PPOConfig())
    if bool(getattr(cfg, "evaluation_only_preset", False)):
        runner = str(getattr(cfg, "evaluation_only_runner", "") or "the evaluation runner")
        raise ValueError(
            f"Preset {getattr(cfg, 'cli_preset', getattr(cfg, 'run_tag', 'unknown'))!r} "
            "is evaluation-only and must not start PPO training. "
            f"Use {runner} with a promoted v6i2 checkpoint."
        )

    if cfg.run_tag and "gate_open" in cfg.run_tag:
        if cfg.latent_cf_require_competence:
            raise ValueError(
                f"Ablation run {cfg.run_tag} requires --no-latent-cf-require-competence "
                f"but resolved latent_cf_require_competence is True!"
            )
        import json
        import dataclasses
        prior_config_path = os.path.join("checkpoints", "4v4_diag", "v6i5_cf_sweep_8x_150k_4v4_run_config.json")
        if not os.path.exists(prior_config_path):
            raise FileNotFoundError(f"Prior run config not found: {prior_config_path}")
        with open(prior_config_path, "r", encoding="utf-8") as f:
            prior_data = json.load(f)
        prior_resolved = prior_data.get("resolved_ppo_config", {})
        allowed_diffs = {
            "run_tag", "metrics_csv_path", "episode_csv_path",
            "latent_cf_require_competence", "checkpoint_dir",
            "utc_timestamp"
        }
        current_dict = dataclasses.asdict(cfg)
        mismatches = []
        for key, prior_val in prior_resolved.items():
            if key in allowed_diffs:
                continue
            if key not in current_dict:
                continue
            curr_val = current_dict[key]
            if isinstance(prior_val, list):
                prior_val = tuple(prior_val)
            if isinstance(curr_val, list):
                curr_val = tuple(curr_val)
            if curr_val != prior_val:
                mismatches.append(f"{key}: prior={prior_val}, current={curr_val}")
        if mismatches:
            raise ValueError(
                f"Configuration mismatch vs prior 8x sweep config:\n" + "\n".join(mismatches)
            )

    set_global_seed(cfg.seed, torch_seed=True, deterministic=cfg.use_deterministic)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    max_agents = max(1, int(getattr(cfg, "max_blue_agents", 2)))
    team_size = _agents_suffix(max_agents)
    curriculum, initial_phase, initial_opponent_tag = _resolve_initial_opponent_and_phase(cfg, max_agents)

    # Resolve CSV paths into cfg before banner so printed paths match the trainer's
    # actual write targets.
    _resolve_metrics_csv_paths(cfg)

    print_training_banner(
        cfg,
        curriculum=curriculum,
        max_agents=max_agents,
        team_size=team_size,
    )

    run_lock = _acquire_run_lock(cfg)
    _rotate_fresh_run_telemetry(cfg)
    try:
        rc_path = write_run_config_json(cfg)
        print(f"[PPO] Run config written: {rc_path}")
    except Exception as exc:
        print(f"[PPO] WARNING: could not write run config JSON: {exc}")

    print_episode_stats_banner(cfg, curriculum=curriculum, initial_opponent_tag=initial_opponent_tag)

    _clamp_runtime_config_for_team_size(cfg, max_agents)
    _ensure_cuda_or_fallback(cfg)

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
