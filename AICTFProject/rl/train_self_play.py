"""
Self-play training using league-based snapshots (same as train_ppo.py SELF_PLAY mode).

This script trains blue agents against a population of opponents:
- Scripted opponents (OP3) for curriculum
- Past snapshots of the learning agent (self-play)
- Uses EloLeague for opponent management and snapshot tracking
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from game_field import make_game_field
from ctf_sb3_env import CTFGameFieldSB3Env
from rl.common import env_seed, set_global_seed
from rl.league import EloLeague
from rl.episode_result import parse_episode_result
from rl.train_ppo import TokenizedCombinedExtractor, MaskedMultiInputPolicy, SelfPlayCallback
from config import MAP_NAME, MAP_PATH


@dataclass
class SelfPlayConfig:
    """Configuration for self-play training (league-based with snapshots)."""
    seed: int = 42
    total_timesteps: int = 2_000_000
    n_envs: int = 4
    n_steps: int = 2048
    batch_size: int = 512
    n_epochs: int = 10
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    learning_rate: float = 3e-4
    max_grad_norm: float = 0.5
    device: str = "cpu"
    
    checkpoint_dir: str = "checkpoints_sb3"
    run_tag: str = "ppo_self_play_4v4"
    save_every_steps: int = 50_000
    enable_checkpoints: bool = False
    enable_tensorboard: bool = False
    
    max_decision_steps: int = 400
    max_blue_agents: int = 4
    use_tokenized_obs: bool = False
    
    # Self-play snapshot management (same as train_ppo.py)
    self_play_use_latest_snapshot: bool = True
    self_play_snapshot_every_episodes: int = 25
    self_play_max_snapshots: int = 10  # Rolling window: keep only latest N snapshots


def _make_env_fn(cfg: SelfPlayConfig, default_opponent: tuple = ("SCRIPTED", "OP3"), rank: int = 0):
    """Environment factory for self-play (same as train_ppo.py)."""
    def _fn():
        s = env_seed(cfg.seed, rank)
        import numpy as np
        np.random.seed(s)
        torch.manual_seed(s)
        env = CTFGameFieldSB3Env(
            make_game_field_fn=lambda: make_game_field(
                map_name=MAP_NAME or None,
                map_path=MAP_PATH or None,
            ),
            max_decision_steps=cfg.max_decision_steps,
            enforce_masks=True,
            seed=s,
            include_mask_in_obs=True,
            default_opponent_kind=default_opponent[0],
            default_opponent_key=default_opponent[1],
            ppo_gamma=cfg.gamma,
            action_flip_prob=0.0,
            max_blue_agents=getattr(cfg, "max_blue_agents", 4),
            print_reset_shapes=False,
            allow_missing_vec_features=False,
            reward_mode="TEAM_SUM",
            use_obs_builder=True,
            include_opponent_context=False,
            obs_debug_validate_locality=False,
            normalize_vec=False,
            auxiliary_progress_scale=0.15,
        )
        return env
    return _fn


def train_self_play(cfg: Optional[SelfPlayConfig] = None) -> None:
    """
    Train self-play using league-based snapshots (same as train_ppo.py SELF_PLAY mode).
    
    Blue learns against:
    - Scripted opponents (OP3) for curriculum
    - Past snapshots of itself (self-play)
    - Uses EloLeague for opponent management
    """
    if cfg is None:
        cfg = SelfPlayConfig()
    
    set_global_seed(cfg.seed, torch_seed=True, deterministic=False)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    
    # Setup league (same as train_ppo.py SELF_PLAY mode)
    league = EloLeague(
        seed=cfg.seed,
        k_factor=32.0,
        matchmaking_tau=200.0,
        scripted_floor=0.50,
        species_prob=0.0,  # No species in self-play
        snapshot_prob=0.0,  # Managed by callback
        anchor_op3_prob=1.0,  # Start with OP3
        species_rusher_bias=0.40,
        use_stability_mix=False,
        min_episodes_per_opponent=3,
    )
    
    # Default opponent: OP3 (same as train_ppo.py SELF_PLAY)
    default_opponent = ("SCRIPTED", "OP3")
    phase_name = "SELF_PLAY"
    
    # Create environments
    env_fns = [
        _make_env_fn(cfg, default_opponent=default_opponent, rank=i)
        for i in range(max(1, int(cfg.n_envs)))
    ]
    
    # On Windows, use DummyVecEnv to avoid multiprocessing pickling issues
    use_subproc = sys.platform != "win32"
    if use_subproc:
        try:
            venv = SubprocVecEnv(env_fns)
        except Exception:
            venv = DummyVecEnv(env_fns)
    else:
        venv = DummyVecEnv(env_fns)
    venv = VecMonitor(venv)
    
    # Set phase
    try:
        venv.env_method("set_phase", phase_name)
    except Exception:
        pass
    
    # Policy kwargs
    policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))
    use_tokenized = bool(getattr(cfg, "use_tokenized_obs", False))
    if use_tokenized:
        policy_kwargs["features_extractor_class"] = TokenizedCombinedExtractor
        policy_kwargs["features_extractor_kwargs"] = dict(cnn_output_dim=256, normalized_image=True)
    
    # Create PPO model
    model = PPO(
        policy=MaskedMultiInputPolicy,
        env=venv,
        learning_rate=cfg.learning_rate,
        n_steps=int(cfg.n_steps),
        batch_size=int(cfg.batch_size),
        n_epochs=int(cfg.n_epochs),
        gamma=float(cfg.gamma),
        gae_lambda=float(cfg.gae_lambda),
        clip_range=float(cfg.clip_range),
        ent_coef=float(cfg.ent_coef),
        vf_coef=0.5,
        max_grad_norm=float(cfg.max_grad_norm),
        tensorboard_log=(
            os.path.join(cfg.checkpoint_dir, "tb", cfg.run_tag)
            if cfg.enable_tensorboard
            else None
        ),
        policy_kwargs=policy_kwargs,
        verbose=0,
        seed=cfg.seed,
        device=cfg.device,
    )
    
    # Setup logger
    if cfg.enable_tensorboard:
        model.set_logger(configure(os.path.join(cfg.checkpoint_dir, "tb", cfg.run_tag), ["tensorboard"]))
    else:
        model.set_logger(configure(None, []))
    
    # Create initial snapshot (same as train_ppo.py)
    init_path = os.path.join(cfg.checkpoint_dir, f"{cfg.run_tag}_selfplay_init")
    try:
        model.save(init_path)
    except Exception as exc:
        print(f"[WARN] self-play init snapshot save failed: {exc}")
    else:
        league.add_snapshot(init_path + ".zip")
        init_key = league.latest_snapshot_key()
        max_snaps = max(0, int(getattr(cfg, "self_play_max_snapshots", 10)))
        if max_snaps > 0:
            while len(league.snapshots) > max_snaps:
                oldest = league.snapshots.pop(0)
                try:
                    if oldest and os.path.exists(oldest):
                        os.remove(oldest)
                except Exception as exc:
                    print(f"[WARN] snapshot cleanup failed: {exc}")
        if init_key:
            try:
                venv.env_method("set_next_opponent", "SNAPSHOT", init_key)
                venv.reset()
            except Exception:
                pass
    
    # Create PPOConfig-like object for SelfPlayCallback
    # (SelfPlayCallback expects a PPOConfig, so we create a minimal compatible object)
    class PPOConfigCompat:
        def __init__(self, cfg: SelfPlayConfig):
            self.run_tag = cfg.run_tag
            self.checkpoint_dir = cfg.checkpoint_dir
            self.self_play_snapshot_every_episodes = cfg.self_play_snapshot_every_episodes
            self.self_play_max_snapshots = cfg.self_play_max_snapshots
            self.self_play_use_latest_snapshot = cfg.self_play_use_latest_snapshot
    
    ppo_cfg_compat = PPOConfigCompat(cfg)
    
    # Callbacks
    callbacks = [SelfPlayCallback(cfg=ppo_cfg_compat, league=league)]
    
    if cfg.enable_checkpoints:
        callbacks.append(
            CheckpointCallback(
                save_freq=int(cfg.save_every_steps),
                save_path=cfg.checkpoint_dir,
                name_prefix=f"ckpt_{cfg.run_tag}",
            )
        )
    
    callbacks = CallbackList(callbacks)
    
    # Training
    print(f"[Self-Play] Starting training: {cfg.total_timesteps} total steps")
    print(f"[Self-Play] Snapshot every {cfg.self_play_snapshot_every_episodes} episodes, max {cfg.self_play_max_snapshots} snapshots")
    
    model.learn(
        total_timesteps=int(cfg.total_timesteps),
        callback=callbacks,
        reset_num_timesteps=True,
    )
    
    # Save final model
    final_path = os.path.join(cfg.checkpoint_dir, f"final_{cfg.run_tag}")
    model.save(final_path)
    print(f"[Self-Play] Training complete.")
    print(f"  Final model: {final_path}.zip")
    print(f"  Snapshots: {len(league.snapshots)}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train self-play (league-based with snapshots)")
    parser.add_argument("--run-tag", type=str, default=None, help="Run name for checkpoints")
    parser.add_argument("--total-steps", type=int, default=None, help="Total timesteps")
    parser.add_argument("--max-blue-agents", type=int, default=4, help="Number of agents per team")
    parser.add_argument("--snapshot-every", type=int, default=25, help="Episodes between snapshots")
    parser.add_argument("--max-snapshots", type=int, default=10, help="Max snapshots to keep (rolling window)")
    args = parser.parse_args()
    
    cfg = SelfPlayConfig()
    if args.run_tag is not None:
        cfg.run_tag = args.run_tag
    if args.total_steps is not None:
        cfg.total_timesteps = args.total_steps
    if args.max_blue_agents is not None:
        cfg.max_blue_agents = args.max_blue_agents
    if args.snapshot_every is not None:
        cfg.self_play_snapshot_every_episodes = args.snapshot_every
    if args.max_snapshots is not None:
        cfg.self_play_max_snapshots = args.max_snapshots
    
    train_self_play(cfg)
