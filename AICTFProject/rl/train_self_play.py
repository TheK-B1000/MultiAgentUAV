"""
Self-play training using league-based snapshots with new 4v4/8v8 observation space.

This script trains blue agents against a population of opponents:
- Scripted opponents (OP3) for curriculum
- Past snapshots of the learning agent (self-play) using new snapshot wrapper v2
- Uses new snapshot_wrapper_v2 for 4v4/8v8 compatibility
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
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from game_field import make_game_field
from ctf_sb3_env import CTFGameFieldSB3Env
from rl.common import env_seed, set_global_seed
from rl.episode_result import parse_episode_result
from rl.train_ppo import TokenizedCombinedExtractor, MaskedMultiInputPolicy
from config import MAP_NAME, MAP_PATH

# Import v2 snapshot wrapper for 4v4/8v8
try:
    from snapshot_wrapper_v2 import make_snapshot_wrapper_v2
    HAS_V2_WRAPPER = True
except ImportError:
    HAS_V2_WRAPPER = False
    print("[WARN] snapshot_wrapper_v2 not available; falling back to legacy wrapper")


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
    
    # Self-play snapshot management
    self_play_snapshot_every_episodes: int = 25
    self_play_max_snapshots: int = 10  # Rolling window: keep only latest N snapshots
    self_play_use_latest_snapshot: bool = True


class SelfPlayCallbackV2(BaseCallback):
    """
    Self-play callback using new snapshot wrapper v2 for 4v4/8v8 compatibility.
    
    Manages snapshots and switches opponents using set_red_policy_wrapper directly,
    bypassing the old league SNAPSHOT path.
    """
    
    def __init__(self, cfg: SelfPlayConfig, verbose: int = 1):
        super().__init__(verbose)
        self.cfg = cfg
        self.episode_idx = 0
        self.win_count = 0
        self.loss_count = 0
        self.draw_count = 0
        self._snapshot_paths: List[str] = []
        self._max_snapshots = max(0, int(getattr(cfg, "self_play_max_snapshots", 10)))
        self._current_snapshot_path: Optional[str] = None
        
    def _enforce_snapshot_limit(self) -> None:
        """Delete oldest snapshots when over limit."""
        if self._max_snapshots <= 0:
            return
        while len(self._snapshot_paths) > self._max_snapshots:
            old_path = self._snapshot_paths.pop(0)
            try:
                if old_path and os.path.exists(old_path):
                    os.remove(old_path)
                    if self.verbose:
                        print(f"[Self-Play] Deleted old snapshot: {os.path.basename(old_path)}")
            except Exception as exc:
                if self.verbose:
                    print(f"[WARN] Failed to delete snapshot {old_path}: {exc}")
    
    def _on_step(self) -> bool:
        from rl.episode_result import parse_episode_result
        
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        
        for i, done in enumerate(dones):
            if not done:
                continue
            info = infos[i] if i < len(infos) else {}
            summary = parse_episode_result(info)
            if summary is None:
                continue
            
            self.episode_idx += 1
            
            # Parse actual game scores to determine result
            blue_score = summary.blue_score
            red_score = summary.red_score
            
            if blue_score > red_score:
                result = "WIN"
                self.win_count += 1
            elif blue_score < red_score:
                result = "LOSS"
                self.loss_count += 1
            else:
                result = "DRAW"
                self.draw_count += 1
            
            # Save snapshot periodically
            if (self.episode_idx % int(self.cfg.self_play_snapshot_every_episodes)) == 0:
                self._enforce_snapshot_limit()
                slot = len(self._snapshot_paths) + 1
                prefix = f"{self.cfg.run_tag}_selfplay_snapshot"
                snap_path = os.path.join(self.cfg.checkpoint_dir, f"{prefix}_slot{slot:03d}")
                try:
                    self.model.save(snap_path)
                    snap_path_zip = snap_path + ".zip"
                    self._snapshot_paths.append(snap_path_zip)
                    self._enforce_snapshot_limit()
                    
                    # Switch to this new snapshot as opponent
                    self._switch_to_snapshot(snap_path_zip)
                    
                    if self.verbose:
                        print(f"[Self-Play] Saved snapshot {slot}: {os.path.basename(snap_path_zip)}")
                except Exception as exc:
                    if self.verbose:
                        print(f"[WARN] Snapshot save failed: {exc}")
            
            # Print progress
            if self.verbose:
                snapshot_count = len(self._snapshot_paths)
                print(
                    f"[PPO|SELF] ep={self.episode_idx} result={result} "
                    f"score={blue_score}:{red_score} "
                    f"snapshots={snapshot_count} "
                    f"W={self.win_count} | L={self.loss_count} | D={self.draw_count}"
                )
            
            # Log metrics
            self.logger.record("self/episode", self.episode_idx)
            self.logger.record("self/win_rate", self.win_count / max(1, self.episode_idx))
            self.logger.record("self/draw_rate", self.draw_count / max(1, self.episode_idx))
            self.logger.record("self/snapshots", float(len(self._snapshot_paths)))
        
        return True
    
    def _switch_to_snapshot(self, snapshot_path: str) -> None:
        """Switch red opponent to use the given snapshot (using v2 wrapper)."""
        if not HAS_V2_WRAPPER:
            if self.verbose:
                print("[WARN] v2 wrapper not available; cannot use snapshot opponent")
            return
        
        if not os.path.exists(snapshot_path):
            if self.verbose:
                print(f"[WARN] Snapshot not found: {snapshot_path}")
            return
        
        try:
            # Get environment to access game_field
            env = self.model.get_env()
            if env is None:
                return
            
            # Get max_agents from config
            max_agents = int(getattr(self.cfg, "max_blue_agents", 4))
            
            # Create v2 wrapper
            wrapper = make_snapshot_wrapper_v2(
                snapshot_path,
                max_agents=max_agents,
                n_macros=5,  # Default, should match training
                n_targets=8,  # Default, should match training
                include_mask_in_obs=True,
                include_opponent_context=False,
                normalize_vec=False,
            )
            
            # Set red policy wrapper directly (bypasses league SNAPSHOT path)
            try:
                env.env_method("set_red_policy_wrapper", wrapper)
            except Exception:
                # Fallback: try per-env
                for i in range(env.num_envs):
                    try:
                        env.env_method("set_red_policy_wrapper", wrapper, indices=[i])
                    except Exception:
                        pass
            
            self._current_snapshot_path = snapshot_path
            
        except Exception as e:
            if self.verbose:
                print(f"[WARN] Failed to switch to snapshot {snapshot_path}: {e}")
                import traceback
                traceback.print_exc()


def _make_env_fn(cfg: SelfPlayConfig, default_opponent: tuple = ("SCRIPTED", "OP3"), rank: int = 0):
    """Environment factory for self-play."""
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
    Train self-play using new snapshot wrapper v2 for 4v4/8v8 compatibility.
    
    Blue learns against:
    - Scripted opponents (OP3) initially
    - Past snapshots of itself (self-play) using v2 wrapper
    """
    if cfg is None:
        cfg = SelfPlayConfig()
    
    if not HAS_V2_WRAPPER:
        raise RuntimeError(
            "snapshot_wrapper_v2 is required for 4v4/8v8 self-play. "
            "Please ensure snapshot_wrapper_v2.py exists and is importable."
        )
    
    set_global_seed(cfg.seed, torch_seed=True, deterministic=False)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    
    # Default opponent: OP3
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
    
    # Create initial snapshot and set as opponent
    init_path = os.path.join(cfg.checkpoint_dir, f"{cfg.run_tag}_selfplay_init")
    try:
        model.save(init_path)
        init_path_zip = init_path + ".zip"
        
        # Switch to initial snapshot using v2 wrapper
        if HAS_V2_WRAPPER:
            wrapper = make_snapshot_wrapper_v2(
                init_path_zip,
                max_agents=cfg.max_blue_agents,
                n_macros=5,
                n_targets=8,
                include_mask_in_obs=True,
                include_opponent_context=False,
                normalize_vec=False,
            )
            try:
                venv.env_method("set_red_policy_wrapper", wrapper)
            except Exception:
                pass
        
        print(f"[Self-Play] Created initial snapshot: {os.path.basename(init_path_zip)}")
    except Exception as exc:
        print(f"[WARN] Initial snapshot creation failed: {exc}")
    
    # Callbacks
    callbacks = [SelfPlayCallbackV2(cfg=cfg)]
    
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
    print(f"[Self-Play] Using snapshot wrapper v2 for 4v4/8v8 compatibility")
    
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
    print(f"  Snapshots created: {len(callbacks[0]._snapshot_paths)}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train self-play (4v4/8v8 with snapshot wrapper v2)")
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
