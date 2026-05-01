"""Train the CTF policy with the local PPO/MAPPO implementation."""

from __future__ import annotations

import os
import random
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Literal, Optional

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_SCRIPT_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

import numpy as np
import torch

from game_field_gpu import GPUCTFVecEnv, GPUFieldConfig, VEC_OBS_DIM
from opponent_params import sample_batched_opponent_params
from rl.custom_ppo import CustomPPOTrainer
from rl.curriculum import CurriculumState, jacob_paper_curriculum_state, phase_from_tag
from rl.global_state import GLOBAL_STATE_DIM
from rl.stress_schedule import STRESS_BY_PHASE


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


class TrainMode(str, Enum):
    FIXED_OPPONENT = "FIXED_OPPONENT"
    CURRICULUM = "CURRICULUM"
    # Backward-compatible alias for old configs/commands.
    CURRICULUM_NO_LEAGUE = "CURRICULUM"


@dataclass
class PPOConfig:
    seed: int = 42
    total_timesteps: int = 1_000_000
    n_envs: int = 8
    n_steps: int = 2048
    batch_size: int = 1024
    n_epochs: int = 6
    gamma: float = 0.995
    gae_lambda: float = 0.99
    clip_range: float = 0.25
    clip_range_vf: Optional[float] = 0.2
    vf_coef: float = 1.0
    normalize_returns: bool = False
    ent_coef: float = 0.01
    learning_rate: float = 5e-4
    lr_floor_frac: float = 0.1
    max_grad_norm: float = 0.5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint_dir: str = "checkpoints"
    load_path: Optional[str] = None
    run_tag: str = "ppo_latent_2v2"
    enable_metrics_csv: bool = True
    metrics_csv_path: Optional[str] = None
    episode_csv_path: Optional[str] = None
    # E3: optional per-step CSV (z, H(q), argmax, switch, phase). See `rl.custom_ppo.E3_STEP_TELEMETRY_FIELDS`.
    e3_step_telemetry_path: Optional[str] = None
    # SB3-compatible: ``tqdm`` (prefer ``tqdm.rich``) during rollout, ``total=remaining`` timesteps, ``update(n_envs)`` / step.
    enable_progress_bar: bool = True
    verbose_training: bool = False
    # After this many *completed* episodes, print W/L/D and win rate (0 = disabled).
    episode_log_every: int = 1000

    max_decision_steps: int = 400
    map_set: str = "train"
    mode: str = TrainMode.FIXED_OPPONENT.value
    fixed_opponent_tag: str = "OP3"
    max_blue_agents: int = 2
    use_deterministic: bool = False
    # Not in *Summer Implementation Plan.docx*; when True, overrides several PPO fields below for a legacy "stable" profile. Default False so explicit config matches the spec numbers.
    use_stable_marl_ppo: bool = False
    target_kl: Optional[float] = 0.02
    actor_cnn_feature_dim: int = 128

    # Summer/ICRA latent team strategy is the default proposed algorithm.
    use_latent_strategy: bool = True
    latent_k: int = 4
    latent_z_embed_dim: int = 16
    latent_vf_hidden: int = 128
    latent_strategy_hidden: int = 128
    # Plan IMPLEMENTATION §6: typical λ_H ∈ [0.001, 0.01]; λ_p ∈ [0.01, 0.05] (see also §3.3 for a wider λ_p range).
    # ``maximize`` matches the plan (encourage exploratory / diverse q_phi). ``minimize`` adds +λ_H·H to the
    # minimized loss and sharpens q_phi (recommended when telemetry shows strategy_entropy≈ln K with no persistence grad).
    # ``none`` removes the H term (strategy_encoder receives no gradient from λ_H when λ_p/KL are also inactive).
    latent_entropy_objective: Literal["maximize", "minimize", "none"] = "maximize"
    latent_lam_h: float = 0.005
    latent_lam_p: float = 0.02
    # A1: clipped PPO/REINFORCE-style update for sampled z. Kept low because z operates at episode cadence.
    latent_strategy_ppo_coef: float = 0.1
    # A2 (opt-in): train q_phi from normalized episode/rollout returns through a small Q head.
    latent_strategy_q_head: bool = False
    latent_strategy_q_coef: float = 1.0
    latent_strategy_tau: float = 1.0
    # 0 = sample once at episode start (main paper default; plan Option A). N>=2 = sparse refresh (Option B).
    latent_resample_every_n: int = 0
    # Mid-episode z changes make V(s,z) discontinuous; optionally break GAE carry across z[t]!=z[t+1].
    latent_gae_reset_on_z_change: bool = True
    # Use argmax z from q_phi(s') when bootstrapping V(s') so peek matches no duplicate stochastic z draw.
    latent_bootstrap_z_deterministic: bool = True
    # Baseline: keep latent actor/critic plumbing, but clamp every rollout to one strategy ID.
    # This tests whether learned/multiple strategy selection matters beyond a single learned z embedding.
    fixed_latent_strategy: bool = False
    fixed_latent_strategy_id: int = 0
    # **Ablation / plan §12 only** — not combined with the main “episode-start z” story by default.
    # Use ``rl.config_presets.ablation_flag_resample_config`` for an explicit run.
    latent_resample_on_flag: bool = False
    # Optional §12: KL( q_\phi(s_t) || q_\phi(s_{t-1}) ) on consecutive time steps; 0 = off (ablation only).
    latent_kl_consecutive: float = 0.0

    # Episode-level domain randomization for sim robustness (sensor dropout/noise, blue speed jitter).
    # See ``GPUFieldConfig`` for numeric ranges; eval harnesses should keep this False.
    train_domain_randomization: bool = False
    dr_sensor_noise_sigma_max: float = 0.12
    dr_sensor_dropout_max: float = 0.08
    dr_blue_speed_jitter: float = 0.12


def train_ppo(cfg: Optional[PPOConfig] = None) -> None:
    """Run the default local PPO/MAPPO training path."""
    cfg = cfg or PPOConfig()

    cfg.mode = _normalize_train_mode(cfg.mode)
    supported_modes = {TrainMode.FIXED_OPPONENT.value, TrainMode.CURRICULUM.value}
    if cfg.mode not in supported_modes:
        raise ValueError(
            "The local PPO trainer currently supports FIXED_OPPONENT and "
            "CURRICULUM training."
        )
    if cfg.mode == TrainMode.CURRICULUM.value:
        if bool(getattr(cfg, "use_latent_strategy", False)) or bool(getattr(cfg, "fixed_latent_strategy", False)):
            print("[PPO] Curriculum mode is the Jacob paper baseline; forcing latent strategy OFF.")
        cfg.use_latent_strategy = False
        cfg.fixed_latent_strategy = False

    if bool(getattr(cfg, "use_latent_strategy", False)):
        k = int(getattr(cfg, "latent_k", 4))
        if k not in (4, 6):
            raise ValueError("latent_k must be 4 or 6 (Summer Implementation Plan: fixed K for all experiments).")
        if bool(getattr(cfg, "fixed_latent_strategy", False)):
            fixed_id = int(getattr(cfg, "fixed_latent_strategy_id", 0) or 0)
            if fixed_id < 0 or fixed_id >= k:
                raise ValueError(f"fixed_latent_strategy_id must be in [0, {k - 1}] when latent_k={k}.")
        res_n = int(getattr(cfg, "latent_resample_every_n", 0) or 0)
        if res_n == 1:
            raise ValueError(
                "latent_resample_every_n=1 is disallowed (do not resample z every decision step). "
                "Use 0 (sample at episode start) or N>=2 (sparse refresh)."
            )
    elif bool(getattr(cfg, "fixed_latent_strategy", False)):
        raise ValueError("fixed_latent_strategy requires use_latent_strategy=True.")

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
    print(f"[PPO] Global state dim: {GLOBAL_STATE_DIM}")
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
        q_head = bool(getattr(cfg, "latent_strategy_q_head", False))
        print(
            "[PPO] Latent team strategy: enabled "
            f"(K={int(cfg.latent_k)}, sample={interval_label}, on_flag={on_flag}, "
            f"lambda_p={float(cfg.latent_lam_p):.4f}, lambda_H={float(cfg.latent_lam_h):.4f} "
            f"(H:{h_obj}), "
            f"lambda_KL={lam_kl:.4f}, strategy_ppo_coef={float(cfg.latent_strategy_ppo_coef):.3f}, "
            f"q_head={q_head}, q_coef={float(cfg.latent_strategy_q_coef):.3f}, "
            f"tau={float(cfg.latent_strategy_tau):.3f}, "
            f"GAE_reset_on_z_change={bool(getattr(cfg, 'latent_gae_reset_on_z_change', True))}, "
            f"bootstrap_z_deterministic={bool(getattr(cfg, 'latent_bootstrap_z_deterministic', True))}"
            f"{fixed_label})"
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
        print(f"[PPO] Update metrics CSV: {cfg.metrics_csv_path}")
        print(f"[PPO] Episode metrics CSV: {cfg.episode_csv_path}")
    else:
        cfg.metrics_csv_path = None
        cfg.episode_csv_path = None
        print("[PPO] Metrics CSV logging disabled.")
    elog = int(getattr(cfg, "episode_log_every", 0) or 0)
    if elog > 0:
        mode_label = "curriculum phase" if curriculum is not None else "scripted opponent tag"
        tag_label = initial_opponent_tag if curriculum is not None else str(cfg.fixed_opponent_tag).upper()
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

    gpu_cfg = GPUFieldConfig(
        n_envs=max(1, int(cfg.n_envs)),
        n_agents_per_team=max_agents,
        map_set=str(getattr(cfg, "map_set", "train")).lower(),
        max_decision_steps=max(1, int(cfg.max_decision_steps)),
        aquaticus_profile=True,
        rules_profile="OURS",
        device=str(cfg.device),
        seed=int(cfg.seed),
        train_domain_randomization=bool(getattr(cfg, "train_domain_randomization", False)),
        dr_sensor_noise_sigma_max=float(getattr(cfg, "dr_sensor_noise_sigma_max", 0.12)),
        dr_sensor_dropout_max=float(getattr(cfg, "dr_sensor_dropout_max", 0.08)),
        dr_blue_speed_jitter=float(getattr(cfg, "dr_blue_speed_jitter", 0.12)),
    )
    env = GPUCTFVecEnv(gpu_cfg)
    try:
        env.env_method("set_stress_schedule", STRESS_BY_PHASE)
        env.env_method("set_dynamics_config", {"rules_profile": "OURS"})
        env.env_method("set_phase", initial_phase)
        env.env_method("set_next_opponent", "SCRIPTED", initial_opponent_tag)
        _apply_initial_opponent_params(env, cfg, gpu_cfg, opponent_tag=initial_opponent_tag, phase=initial_phase)

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
        if cfg.load_path and os.path.isfile(cfg.load_path):
            print(f"[PPO] Resuming checkpoint: {cfg.load_path}")
            trainer.load(cfg.load_path)
        stats = trainer.learn(total_timesteps=int(cfg.total_timesteps))
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


def _apply_initial_opponent_params(
    env: GPUCTFVecEnv,
    cfg: PPOConfig,
    gpu_cfg: GPUFieldConfig,
    *,
    opponent_tag: str | None = None,
    phase: str | None = None,
) -> None:
    try:
        key = str(opponent_tag or cfg.fixed_opponent_tag).upper()
        phase_key = str(phase or phase_from_tag(key)).upper()
        opp_params = sample_batched_opponent_params(
            kind="SCRIPTED",
            key=key,
            phase=phase_key,
            n_agents=gpu_cfg.max_red_agents,
            batch_size=gpu_cfg.n_envs,
            device=gpu_cfg.device,
        )
        dyn_cfg: dict[str, object] = {
            key: value
            for key, value in opp_params.items()
            if key in {"deception_prob", "speed_mult", "attacker_style", "defender_style", "role_switch_prob"}
        }
        if dyn_cfg:
            env.env_method("set_dynamics_config", dyn_cfg)
    except Exception as exc:
        print(f"[PPO] opponent_params sampling failed; using defaults: {exc}")


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


def _normalize_train_mode(mode: str) -> str:
    raw = str(mode).upper().strip()
    aliases = {
        "FIXED": TrainMode.FIXED_OPPONENT.value,
        "FIXED_OPPONENT": TrainMode.FIXED_OPPONENT.value,
        "PAPER": TrainMode.CURRICULUM.value,
        "NO_LEAGUE": TrainMode.CURRICULUM.value,
        "CURRICULUM": TrainMode.CURRICULUM.value,
        "CURRICULUM_NO_LEAGUE": TrainMode.CURRICULUM.value,
        "JACOB": TrainMode.CURRICULUM.value,
    }
    removed = {"LEAGUE", "CURRICULUM_LEAGUE", "SELF_PLAY"}
    if raw in removed:
        print(f"[PPO] Train mode {raw!r} is not in the local PPO audit path; using FIXED_OPPONENT.")
        return TrainMode.FIXED_OPPONENT.value
    return aliases.get(raw, raw)


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
    if _normalize_train_mode(mode) == TrainMode.FIXED_OPPONENT.value:
        return f"{family}_fixed_{fixed_opponent_tag.lower()}_{suffix}"
    return f"{family}_{suffix}"


if __name__ == "__main__":
    import argparse

    if "--verify-4v4" in sys.argv:
        run_verify_4v4(num_episodes=10)
    elif "--test-vec-schema" in sys.argv:
        run_test_vec_schema()
    else:
        parser = argparse.ArgumentParser(description="Train custom PPO/MAPPO for CTF")
        parser.add_argument("--seed", type=int, default=None)
        parser.add_argument(
            "--mode",
            type=str,
            default=None,
            help="Training mode: FIXED_OPPONENT, or CURRICULUM for OP1->OP2->OP3.",
        )
        parser.add_argument("--run-tag", type=str, default=None)
        parser.add_argument("--total-steps", type=int, default=None)
        parser.add_argument("--checkpoint-dir", type=str, default=None)
        parser.add_argument("--metrics-csv", type=str, default=None, help="Path for per-update training metrics CSV.")
        parser.add_argument("--episode-csv", type=str, default=None, help="Path for per-episode training outcome CSV.")
        parser.add_argument("--no-metrics-csv", action="store_true", help="Disable training CSV telemetry.")
        parser.add_argument("--load", type=str, default=None)
        parser.add_argument("--learning-rate", type=float, default=None, help="PPO learning rate.")
        parser.add_argument(
            "--lr-floor-frac",
            type=float,
            default=None,
            help="Minimum fraction of the base learning rate used by the linear schedule.",
        )
        parser.add_argument("--target-kl", type=float, default=None, help="PPO target KL; negative disables early stopping.")
        parser.add_argument("--n-epochs", type=int, default=None, help="Number of PPO optimization epochs per rollout.")
        parser.add_argument(
            "--clip-range-vf",
            type=float,
            default=None,
            help="Value-function clip range; negative disables value clipping.",
        )
        parser.add_argument("--vf-coef", type=float, default=None, help="Value-function loss coefficient.")
        parser.add_argument(
            "--return-normalization",
            action="store_true",
            help="Train critic on normalized returns while denormalizing values for GAE/advantages.",
        )
        parser.add_argument("--fixed-opponent", type=str, default="OP3")
        parser.add_argument("--map-set", type=str, choices=["train", "eval"], default=None)
        parser.add_argument("--agents", type=int, choices=[2, 4, 6, 8], default=None)
        parser.add_argument("--max-blue-agents", type=int, default=None)
        parser.add_argument("--device", type=str, default=None)
        parser.add_argument("--deterministic", action="store_true")
        parser.add_argument("--verbose-training", action="store_true")
        parser.add_argument(
            "--stable-marl",
            action="store_true",
            help="Enable stable-MARL PPO hyperparameter override (not defined in Summer Implementation Plan.docx).",
        )
        parser.add_argument(
            "--latent-strategy",
            action="store_true",
            help="Enable latent team strategy training (default in PPOConfig).",
        )
        parser.add_argument(
            "--no-latent-strategy",
            action="store_true",
            help="Disable latent team strategy for vanilla local PPO ablations.",
        )
        parser.add_argument("--latent-k", type=int, default=None, help="Number of discrete team strategies.")
        parser.add_argument(
            "--latent-resample-every",
            type=int,
            default=None,
            help="Sparse strategy refresh interval in decision steps; 0 samples at episode start only.",
        )
        parser.add_argument(
            "--fixed-latent-strategy",
            action="store_true",
            help="Baseline: keep latent actor/critic inputs but clamp all rollouts to one fixed z ID.",
        )
        parser.add_argument(
            "--fixed-latent-id",
            type=int,
            default=None,
            help="Strategy ID used by --fixed-latent-strategy (default: 0). Supplying this implies fixed latent.",
        )
        parser.add_argument("--latent-lam-p", type=float, default=None, help="Strategy persistence penalty weight.")
        parser.add_argument("--latent-lam-h", type=float, default=None, help="Strategy entropy weight (see --latent-entropy-objective).")
        parser.add_argument(
            "--latent-strategy-ppo-coef",
            type=float,
            default=None,
            help="Coefficient for the sampled-z clipped PPO strategy loss.",
        )
        parser.add_argument(
            "--latent-strategy-q-head",
            action="store_true",
            help="Enable A2 Q-head supervision for q_phi using normalized sampled-strategy returns.",
        )
        parser.add_argument(
            "--latent-strategy-q-coef",
            type=float,
            default=None,
            help="Coefficient for the A2 sampled-strategy Q-head MSE loss.",
        )
        parser.add_argument(
            "--latent-strategy-tau",
            type=float,
            default=None,
            help="Softmax temperature for Q-head strategy logits.",
        )
        parser.add_argument(
            "--latent-entropy-objective",
            type=str,
            choices=("maximize", "minimize", "none"),
            default=None,
            help="How λ_H shapes H(q_phi): maximize=paper bonus on entropy; minimize=penalty (sharper z); none=off.",
        )
        parser.add_argument(
            "--latent-resample-on-flag",
            action="store_true",
            help="Resample z when the global-state flag/territory features change (optional plan §12).",
        )
        parser.add_argument(
            "--latent-kl-consecutive",
            type=float,
            default=None,
            help="Weight for consecutive-step KL on q_phi logits (0=off; optional plan §12).",
        )
        parser.add_argument(
            "--no-latent-gae-z-reset",
            action="store_true",
            help="Keep legacy GAE: carry λ-returns across z switches (can smear credit when V(s,z) jumps).",
        )
        parser.add_argument(
            "--latent-bootstrap-z-stochastic",
            action="store_true",
            help="Sample z for V(s') bootstrap; default argmax to avoid RNG mismatch with the next step.",
        )
        parser.add_argument(
            "--domain-randomization",
            action="store_true",
            help="Enable episode-level domain randomization (obs noise/dropout, blue speed jitter).",
        )
        parser.add_argument(
            "--dr-sensor-noise-max",
            type=float,
            default=None,
            help="Max enemy obs position noise (map cells) per episode when domain randomization is on.",
        )
        parser.add_argument(
            "--dr-sensor-dropout-max",
            type=float,
            default=None,
            help="Max in-range enemy dropout probability per episode when domain randomization is on.",
        )
        parser.add_argument(
            "--dr-blue-speed-jitter",
            type=float,
            default=None,
            help="Blue max-speed scale draws from U(1-j,1) per episode (slowdown-only; marine speed cap unchanged).",
        )
        parser.add_argument(
            "--latent-z-embed-dim",
            type=int,
            default=None,
            help="Strategy embedding dimension used by the shared actor.",
        )
        parser.add_argument(
            "--latent-vf-hidden",
            type=int,
            default=None,
            help="Hidden width for the centralized latent critic.",
        )
        parser.add_argument(
            "--episode-log-every",
            type=int,
            default=None,
            metavar="N",
            help="Log W/L/D every N completed episodes (0=off; default from PPOConfig).",
        )
        parser.add_argument(
            "--no-progress-bar",
            action="store_true",
            help="Disable the SB3-style tqdm rollout bar (default: on; uses tqdm.rich if installed).",
        )
        args = parser.parse_args()

        cfg = PPOConfig()
        if args.mode is not None:
            cfg.mode = _normalize_train_mode(args.mode)
        if args.seed is not None:
            cfg.seed = int(args.seed)
        if args.max_blue_agents is not None:
            cfg.max_blue_agents = max(1, min(int(args.max_blue_agents), 16))
        elif args.agents is not None:
            cfg.max_blue_agents = int(args.agents)
        cfg.fixed_opponent_tag = str(args.fixed_opponent).upper()
        if args.map_set is not None:
            cfg.map_set = str(args.map_set).lower()
        if args.latent_strategy:
            cfg.use_latent_strategy = True
        if args.no_latent_strategy:
            cfg.use_latent_strategy = False
        if args.latent_k is not None:
            cfg.latent_k = max(2, int(args.latent_k))
        if args.latent_resample_every is not None:
            cfg.latent_resample_every_n = max(0, int(args.latent_resample_every))
        if args.fixed_latent_strategy:
            cfg.fixed_latent_strategy = True
        if args.fixed_latent_id is not None:
            cfg.fixed_latent_strategy = True
            cfg.fixed_latent_strategy_id = max(0, int(args.fixed_latent_id))
        if args.latent_lam_p is not None:
            cfg.latent_lam_p = max(0.0, float(args.latent_lam_p))
        if args.latent_lam_h is not None:
            cfg.latent_lam_h = max(0.0, float(args.latent_lam_h))
        if args.latent_strategy_ppo_coef is not None:
            cfg.latent_strategy_ppo_coef = max(0.0, float(args.latent_strategy_ppo_coef))
        if args.latent_strategy_q_head:
            cfg.latent_strategy_q_head = True
        if args.latent_strategy_q_coef is not None:
            cfg.latent_strategy_q_coef = max(0.0, float(args.latent_strategy_q_coef))
        if args.latent_strategy_tau is not None:
            cfg.latent_strategy_tau = max(1e-3, float(args.latent_strategy_tau))
        if args.latent_entropy_objective is not None:
            cfg.latent_entropy_objective = args.latent_entropy_objective  # type: ignore[assignment]
        if args.latent_resample_on_flag:
            cfg.latent_resample_on_flag = True
        if args.latent_kl_consecutive is not None:
            cfg.latent_kl_consecutive = max(0.0, float(args.latent_kl_consecutive))
        if args.no_latent_gae_z_reset:
            cfg.latent_gae_reset_on_z_change = False
        if args.latent_bootstrap_z_stochastic:
            cfg.latent_bootstrap_z_deterministic = False
        if args.domain_randomization:
            cfg.train_domain_randomization = True
        if args.dr_sensor_noise_max is not None:
            cfg.dr_sensor_noise_sigma_max = max(0.0, float(args.dr_sensor_noise_max))
        if args.dr_sensor_dropout_max is not None:
            cfg.dr_sensor_dropout_max = max(0.0, min(1.0, float(args.dr_sensor_dropout_max)))
        if args.dr_blue_speed_jitter is not None:
            cfg.dr_blue_speed_jitter = max(0.0, min(0.75, float(args.dr_blue_speed_jitter)))
        if args.latent_z_embed_dim is not None:
            cfg.latent_z_embed_dim = max(1, int(args.latent_z_embed_dim))
        if args.latent_vf_hidden is not None:
            cfg.latent_vf_hidden = max(1, int(args.latent_vf_hidden))
        cfg.run_tag = args.run_tag or _default_run_tag_for_mode(
            cfg.mode,
            cfg.fixed_opponent_tag,
            cfg.max_blue_agents,
            latent=bool(cfg.use_latent_strategy),
        )
        cfg.run_tag = _ensure_run_tag_has_agent_suffix(cfg.run_tag, cfg.max_blue_agents)
        cfg.checkpoint_dir = args.checkpoint_dir or os.path.join("checkpoints", _agents_suffix(cfg.max_blue_agents))
        if args.no_metrics_csv:
            cfg.enable_metrics_csv = False
        if args.metrics_csv is not None:
            cfg.metrics_csv_path = args.metrics_csv
        if args.episode_csv is not None:
            cfg.episode_csv_path = args.episode_csv
        if args.total_steps is not None:
            cfg.total_timesteps = int(args.total_steps)
        if args.load is not None:
            cfg.load_path = args.load
        if args.learning_rate is not None:
            cfg.learning_rate = max(0.0, float(args.learning_rate))
        if args.lr_floor_frac is not None:
            cfg.lr_floor_frac = max(0.0, min(float(args.lr_floor_frac), 1.0))
        if args.target_kl is not None:
            cfg.target_kl = None if float(args.target_kl) < 0.0 else max(0.0, float(args.target_kl))
        if args.n_epochs is not None:
            cfg.n_epochs = max(1, int(args.n_epochs))
        if args.clip_range_vf is not None:
            cfg.clip_range_vf = None if float(args.clip_range_vf) < 0.0 else max(0.0, float(args.clip_range_vf))
        if args.vf_coef is not None:
            cfg.vf_coef = max(0.0, float(args.vf_coef))
        if args.return_normalization:
            cfg.normalize_returns = True
        if args.device is not None:
            cfg.device = str(args.device).strip().lower()
        if args.deterministic:
            cfg.use_deterministic = True
        if args.verbose_training:
            cfg.verbose_training = True
        if args.stable_marl:
            cfg.use_stable_marl_ppo = True
        if args.episode_log_every is not None:
            cfg.episode_log_every = max(0, int(args.episode_log_every))
        if args.no_progress_bar:
            cfg.enable_progress_bar = False
        train_ppo(cfg)
