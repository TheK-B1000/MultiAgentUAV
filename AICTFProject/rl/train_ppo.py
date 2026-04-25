"""Train the CTF policy with the local PPO/MAPPO implementation."""

from __future__ import annotations

import os
import random
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Optional

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_SCRIPT_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

import numpy as np
import torch

from game_field_gpu import GPUCTFVecEnv, GPUFieldConfig
from opponent_params import sample_batched_opponent_params
from rl.custom_ppo import CustomPPOTrainer
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


@dataclass
class PPOConfig:
    seed: int = 42
    total_timesteps: int = 1_000_000
    n_envs: int = 8
    n_steps: int = 2048
    batch_size: int = 512
    n_epochs: int = 10
    gamma: float = 0.995
    gae_lambda: float = 0.99
    clip_range: float = 0.2
    clip_range_vf: Optional[float] = 0.2
    ent_coef: float = 0.01
    learning_rate: float = 3e-4
    max_grad_norm: float = 0.5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint_dir: str = "checkpoints"
    load_path: Optional[str] = None
    run_tag: str = "ppo_custom_2v2"
    enable_progress_bar: bool = True
    verbose_training: bool = False

    max_decision_steps: int = 400
    mode: str = TrainMode.FIXED_OPPONENT.value
    fixed_opponent_tag: str = "OP3"
    max_blue_agents: int = 2
    use_deterministic: bool = False
    use_stable_marl_ppo: bool = True
    target_kl: Optional[float] = 0.02

    # Reserved for the Summer/ICRA latent strategy implementation.
    use_latent_strategy: bool = False
    latent_k: int = 4
    latent_z_embed_dim: int = 16
    latent_vf_hidden: int = 128
    latent_strategy_hidden: int = 128
    latent_lam_h: float = 0.01
    latent_lam_p: float = 0.02
    latent_resample_every_n: int = 0


def train_ppo(cfg: Optional[PPOConfig] = None) -> None:
    """Run the default local PPO/MAPPO training path."""
    cfg = cfg or PPOConfig()
    if bool(getattr(cfg, "use_latent_strategy", False)):
        raise NotImplementedError(
            "Latent strategy training is reserved for the local trainer follow-up phase. "
            "Phase 4/7 should add it to rl.custom_ppo using the existing rollout fields."
        )

    cfg.mode = _normalize_train_mode(cfg.mode)
    if cfg.mode != TrainMode.FIXED_OPPONENT.value:
        raise ValueError("The local PPO trainer currently supports FIXED_OPPONENT training.")

    set_global_seed(cfg.seed, torch_seed=True, deterministic=cfg.use_deterministic)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    max_agents = max(1, int(getattr(cfg, "max_blue_agents", 2)))
    team_size = _agents_suffix(max_agents)
    print(f"[PPO] Agents: {max_agents} per team ({team_size}) | mode={cfg.mode} | run_tag={cfg.run_tag!r}")
    print("[PPO] Algorithm backend: custom local PPO")
    print(f"[PPO] Total timesteps: {int(cfg.total_timesteps):,}")
    print(f"[PPO] Global state dim: {GLOBAL_STATE_DIM}")
    print(f"[PPO] Checkpoint dir: {cfg.checkpoint_dir}")

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
        max_decision_steps=max(1, int(cfg.max_decision_steps)),
        aquaticus_profile=True,
        rules_profile="OURS",
        device=str(cfg.device),
        seed=int(cfg.seed),
    )
    env = GPUCTFVecEnv(gpu_cfg)
    try:
        env.env_method("set_stress_schedule", STRESS_BY_PHASE)
        env.env_method("set_dynamics_config", {"rules_profile": "OURS"})
        env.env_method("set_phase", str(cfg.fixed_opponent_tag).upper())
        env.env_method("set_next_opponent", "SCRIPTED", str(cfg.fixed_opponent_tag).upper())
        _apply_initial_opponent_params(env, cfg, gpu_cfg)

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
            print("[PPO] Stable MARL profile: lr=1.5e-4, n_epochs=2, clip=0.10, ent=0.005.")
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


def _apply_initial_opponent_params(env: GPUCTFVecEnv, cfg: PPOConfig, gpu_cfg: GPUFieldConfig) -> None:
    try:
        opp_params = sample_batched_opponent_params(
            kind="SCRIPTED",
            key=str(cfg.fixed_opponent_tag).upper(),
            phase=str(cfg.fixed_opponent_tag).upper(),
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
        assert vec.ndim == 3 and vec.shape[2] == 18, f"vec.shape {vec.shape}, expected (B,N,18)"
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
    aliases = {"FIXED": TrainMode.FIXED_OPPONENT.value, "FIXED_OPPONENT": TrainMode.FIXED_OPPONENT.value}
    removed = {"LEAGUE", "CURRICULUM_LEAGUE", "PAPER", "NO_LEAGUE", "CURRICULUM_NO_LEAGUE", "SELF_PLAY"}
    if raw in removed:
        print(f"[PPO] Train mode {raw!r} is not in the local PPO audit path; using FIXED_OPPONENT.")
        return TrainMode.FIXED_OPPONENT.value
    return aliases.get(raw, raw)


def _default_run_tag_for_mode(mode: str, fixed_opponent_tag: str = "OP3", n_agents: int = 2) -> str:
    suffix = _agents_suffix(n_agents)
    if _normalize_train_mode(mode) == TrainMode.FIXED_OPPONENT.value:
        return f"ppo_custom_fixed_{fixed_opponent_tag.lower()}_{suffix}"
    return f"ppo_custom_{suffix}"


if __name__ == "__main__":
    import argparse

    if "--verify-4v4" in sys.argv:
        run_verify_4v4(num_episodes=10)
    elif "--test-vec-schema" in sys.argv:
        run_test_vec_schema()
    else:
        parser = argparse.ArgumentParser(description="Train custom PPO/MAPPO for CTF")
        parser.add_argument("--mode", type=str, default=None)
        parser.add_argument("--run-tag", type=str, default=None)
        parser.add_argument("--total-steps", type=int, default=None)
        parser.add_argument("--checkpoint-dir", type=str, default=None)
        parser.add_argument("--load", type=str, default=None)
        parser.add_argument("--fixed-opponent", type=str, default="OP3")
        parser.add_argument("--agents", type=int, choices=[2, 4, 6, 8], default=None)
        parser.add_argument("--max-blue-agents", type=int, default=None)
        parser.add_argument("--device", type=str, default=None)
        parser.add_argument("--deterministic", action="store_true")
        parser.add_argument("--verbose-training", action="store_true")
        parser.add_argument(
            "--latent-strategy",
            action="store_true",
            help="Reserved. Latent strategy training will be added to the local trainer in the implementation phase.",
        )
        args = parser.parse_args()

        cfg = PPOConfig()
        if args.mode is not None:
            cfg.mode = _normalize_train_mode(args.mode)
        if args.max_blue_agents is not None:
            cfg.max_blue_agents = max(1, min(int(args.max_blue_agents), 16))
        elif args.agents is not None:
            cfg.max_blue_agents = int(args.agents)
        cfg.fixed_opponent_tag = str(args.fixed_opponent).upper()
        cfg.run_tag = args.run_tag or _default_run_tag_for_mode(cfg.mode, cfg.fixed_opponent_tag, cfg.max_blue_agents)
        cfg.run_tag = _ensure_run_tag_has_agent_suffix(cfg.run_tag, cfg.max_blue_agents)
        cfg.checkpoint_dir = args.checkpoint_dir or os.path.join("checkpoints", _agents_suffix(cfg.max_blue_agents))
        if args.total_steps is not None:
            cfg.total_timesteps = int(args.total_steps)
        if args.load is not None:
            cfg.load_path = args.load
        if args.device is not None:
            cfg.device = str(args.device).strip().lower()
        if args.deterministic:
            cfg.use_deterministic = True
        if args.verbose_training:
            cfg.verbose_training = True
        if args.latent_strategy:
            cfg.use_latent_strategy = True
        train_ppo(cfg)
