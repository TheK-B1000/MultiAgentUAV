"""
Self-play training using league-based snapshots with new 4v4/8v8 observation space.

This script trains blue agents against a population of opponents:
- Scripted opponents (OP3) for curriculum
- Past snapshots of the learning agent (self-play) using new snapshot wrapper v2
- Uses new snapshot_wrapper_v2 for 4v4/8v8 compatibility
"""

from __future__ import annotations

import os
import random
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# Add parent directory to path so imports work regardless of where script is run from
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_SCRIPT_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv, VecMonitor

from game_field import make_game_field
from ctf_sb3_env import CTFGameFieldSB3Env
from rl.common import env_seed, set_global_seed
from rl.episode_result import parse_episode_result
from rl.train_ppo import TokenizedCombinedExtractor, MaskedMultiInputPolicy
from rl.curriculum import STRESS_BY_PHASE
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
    """Configuration for self-play training (league-based with snapshots). Stability preset for 4v4."""
    seed: int = 42
    total_timesteps: int = 2_000_000
    n_envs: int = 4
    n_steps: int = 2048
    batch_size: int = 1024
    n_epochs: int = 3
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_range: float = 0.1
    ent_coef: float = 0.01
    learning_rate: float = 3e-5
    max_grad_norm: float = 0.5
    target_kl: float = 0.02
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
    self_play_snapshot_every_episodes: int = 50  # 4v4: less frequent so red is slightly older, blue wins more
    self_play_max_snapshots: int = 10  # Rolling window: keep only latest N snapshots
    self_play_use_latest_snapshot: bool = True

    # 4v4 warmup: train vs scripted OP1 (then mix OP2) so Blue learns defense/offense before self-play
    warmup_opponent: str = "OP1"
    warmup_episodes: int = 300
    # Mix in scripted opponents when resampling so Blue wins more and learns defense (same rewards as League/Fixed)
    scripted_opponent_prob: float = 0.35
    scripted_op1_prob: float = 0.45  # When picking scripted, use OP1 this often (easier); else OP2
    resample_opponent_every_episodes: int = 25
    # Prefer older snapshots when sampling (so Blue beats "yesterday" and gets more wins)
    snapshot_prefer_oldest_half: bool = True


class SelfPlayCallbackV2(BaseCallback):
    """
    Self-play callback using new snapshot wrapper v2 for 4v4/8v8 compatibility.
    
    Manages snapshots and switches opponents using set_red_policy_wrapper directly,
    bypassing the old league SNAPSHOT path.
    Supports warmup: vs scripted opponent for warmup_episodes, then switch to snapshot.
    """
    
    def __init__(
        self,
        cfg: SelfPlayConfig,
        *,
        init_snapshot_path: Optional[str] = None,
        warmup_episodes: int = 0,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.cfg = cfg
        self.episode_idx = 0
        self.win_count = 0
        self.loss_count = 0
        self.draw_count = 0
        self._snapshot_paths: List[str] = []
        self._max_snapshots = max(0, int(getattr(cfg, "self_play_max_snapshots", 10)))
        self._current_snapshot_path: Optional[str] = None
        self._init_snapshot_path = init_snapshot_path
        self._warmup_episodes = max(0, int(warmup_episodes))
        self._warmup_done = False
        self._snapshot_seq = 0

    def _enforce_snapshot_limit(self) -> None:
        """Delete oldest snapshots when over limit. Never delete the current opponent."""
        if self._max_snapshots <= 0:
            return
        while len(self._snapshot_paths) > self._max_snapshots:
            old_path = self._snapshot_paths.pop(0)
            if old_path == self._current_snapshot_path:
                self._snapshot_paths.append(old_path)
                continue
            try:
                if old_path and os.path.exists(old_path):
                    os.remove(old_path)
                    if self.verbose:
                        print(f"[Self-Play] Deleted old snapshot: {os.path.basename(old_path)}")
            except Exception as exc:
                if self.verbose:
                    print(f"[WARN] Failed to delete snapshot {old_path}: {exc}")

    def _choose_snapshot(self) -> Optional[str]:
        """Pick a snapshot from the pool. Prefer older (weaker) ones so Blue wins more and learns."""
        if not self._snapshot_paths:
            return None
        paths = self._snapshot_paths
        # Exclude newest 1–2 so we don't fight today's self
        pool = paths[:-2] if len(paths) > 2 else paths
        if not pool:
            return None
        # Prefer oldest half of pool so opponent is weaker and Blue gets more wins
        if getattr(self.cfg, "snapshot_prefer_oldest_half", True) and len(pool) > 1:
            half = max(1, len(pool) // 2)
            pool = pool[:half]
        return random.choice(pool)

    def _switch_to_scripted(self, tag: str = "OP2") -> None:
        """Queue scripted opponent for next reset (so we mix in scripted, not only snapshots)."""
        try:
            env = self.model.get_env()
            if env is not None:
                env.env_method("set_next_opponent", "SCRIPTED", tag)
            self._current_snapshot_path = None
        except Exception as e:
            if self.verbose:
                print(f"[WARN] Failed to switch to scripted {tag}: {e}")

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

            # After warmup: switch from scripted opponent to initial snapshot (self-play)
            if (
                self._warmup_episodes > 0
                and not self._warmup_done
                and self.episode_idx >= self._warmup_episodes
            ):
                self._warmup_done = True
                if self._init_snapshot_path and os.path.exists(self._init_snapshot_path):
                    self._switch_to_snapshot(self._init_snapshot_path)
                    if self.verbose:
                        print(f"[Self-Play] Warmup done (ep {self.episode_idx}). Switched to self-play (initial snapshot).")
            
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
            
            # Save snapshot periodically (only after warmup; during warmup red is scripted)
            if self._warmup_done and (self.episode_idx % int(self.cfg.self_play_snapshot_every_episodes)) == 0:
                self._enforce_snapshot_limit()
                self._snapshot_seq += 1
                snap_id = f"ep{self.episode_idx:06d}_s{self._snapshot_seq:04d}"
                prefix = f"{self.cfg.run_tag}_selfplay_snapshot"
                snap_path = os.path.join(self.cfg.checkpoint_dir, f"{prefix}_{snap_id}.zip")
                try:
                    self.model.save(snap_path)
                    self._snapshot_paths.append(snap_path)
                    self._enforce_snapshot_limit()
                    # Maybe switch to this new snapshot or to an older one from the pool
                    opp = self._choose_snapshot()
                    if opp:
                        self._switch_to_snapshot(opp)
                    if self.verbose:
                        print(f"[Self-Play] Saved snapshot: {os.path.basename(snap_path)}")
                except Exception as exc:
                    if self.verbose:
                        print(f"[WARN] Snapshot save failed: {exc}")

            # Periodically resample opponent: mix in scripted (OP1/OP2) so Blue wins more and learns defense
            resample_every = max(1, int(getattr(self.cfg, "resample_opponent_every_episodes", 25)))
            scripted_prob = float(getattr(self.cfg, "scripted_opponent_prob", 0.0))
            op1_prob = float(getattr(self.cfg, "scripted_op1_prob", 0.45))
            just_saved = (
                self._warmup_done
                and (self.episode_idx % int(self.cfg.self_play_snapshot_every_episodes)) == 0
            )
            if self._warmup_done and (self.episode_idx % resample_every == 0) and not just_saved:
                if random.random() < scripted_prob:
                    tag = "OP1" if random.random() < op1_prob else "OP2"
                    self._switch_to_scripted(tag)
                else:
                    opp = self._choose_snapshot()
                    if opp:
                        self._switch_to_snapshot(opp)
            
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


class RewardScaleVecWrapper(VecEnv):
    """Scale step rewards by a constant (e.g. 1/n_agents for TEAM_SUM in 4v4 to reduce gradient scale)."""

    def __init__(self, venv: VecEnv, scale: float = 1.0):
        super().__init__(venv.num_envs, venv.observation_space, venv.action_space)
        self.venv = venv
        self._scale = float(scale)

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        rewards = rewards * self._scale
        return obs, rewards, dones, infos

    def reset(self):
        return self.venv.reset()

    def close(self):
        return self.venv.close()

    def env_is_wrapped(self, *args, **kwargs):
        return self.venv.env_is_wrapped(*args, **kwargs)

    def get_attr(self, *args, **kwargs):
        return self.venv.get_attr(*args, **kwargs)

    def set_attr(self, *args, **kwargs):
        return self.venv.set_attr(*args, **kwargs)

    def env_method(self, *args, **kwargs):
        return self.venv.env_method(*args, **kwargs)

    def step_async(self, actions):
        return self.venv.step_async(actions)


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
            auxiliary_progress_scale=0.22,
        )
        return env
    return _fn


def train_self_play(cfg: Optional[SelfPlayConfig] = None) -> None:
    """
    Train self-play using new snapshot wrapper v2 for 4v4/8v8 compatibility.
    
    Blue learns against:
    - Scripted OP1/OP2 during warmup and when resampling (same eased 4v4 params as Fixed/League)
    - Past snapshots of itself (prefer older/weaker so Blue wins more)
    
    Uses same GameManager rewards as other baselines: defense presence, escort carrier,
    progress toward carrier when defending, so Blue learns defense and smarter play.
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

    warmup_episodes = max(0, int(getattr(cfg, "warmup_episodes", 0)))
    warmup_opponent = str(getattr(cfg, "warmup_opponent", "OP1")).upper()
    # During warmup, red = scripted (OP1); after warmup, red = snapshot
    default_opponent = ("SCRIPTED", warmup_opponent) if warmup_episodes > 0 else ("SCRIPTED", "OP3")
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

    # Same stress schedule as League/Fixed so game difficulty (mines, physics) matches other baselines
    try:
        venv.env_method("set_stress_schedule", STRESS_BY_PHASE)
    except Exception:
        pass

    # Scale rewards by 1/max_blue_agents so TEAM_SUM doesn't inflate gradients in 4v4
    max_agents = max(1, int(getattr(cfg, "max_blue_agents", 4)))
    reward_scale = 1.0 / max_agents
    venv = RewardScaleVecWrapper(venv, scale=reward_scale)
    print(f"[Self-Play] Reward scale 1/{max_agents} = {reward_scale:.3f} (TEAM_SUM normalization)")
    print(f"[Self-Play] Same rewards as other baselines: defense presence, escort, progress-to-carrier when defending")

    # Set phase (uses same GameManager rewards: defense, escort, intercept)
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
    
    # Create PPO model (stability preset: lr 3e-5, n_epochs 3, clip 0.1, target_kl 0.02)
    ppo_kwargs = dict(
        policy=MaskedMultiInputPolicy,
        env=venv,
        learning_rate=float(cfg.learning_rate),
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
    if getattr(cfg, "target_kl", None) is not None and float(cfg.target_kl) > 0:
        ppo_kwargs["target_kl"] = float(cfg.target_kl)
    model = PPO(**ppo_kwargs)
    
    # Setup logger
    if cfg.enable_tensorboard:
        model.set_logger(configure(os.path.join(cfg.checkpoint_dir, "tb", cfg.run_tag), ["tensorboard"]))
    else:
        model.set_logger(configure(None, []))
    
    # Create initial snapshot; set as red only if no warmup (otherwise red stays scripted until warmup_episodes)
    init_path = os.path.join(cfg.checkpoint_dir, f"{cfg.run_tag}_selfplay_init")
    init_path_zip = None
    try:
        model.save(init_path)
        init_path_zip = init_path + ".zip"
        if warmup_episodes == 0 and HAS_V2_WRAPPER:
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
        print(f"[Self-Play] Created initial snapshot: {os.path.basename(init_path_zip)}"
              + (f" (red = scripted {warmup_opponent} for first {warmup_episodes} eps)" if warmup_episodes > 0 else " (red = this snapshot)"))
    except Exception as exc:
        print(f"[WARN] Initial snapshot creation failed: {exc}")
    
    # Callbacks
    callbacks = [
        SelfPlayCallbackV2(
            cfg=cfg,
            init_snapshot_path=init_path_zip,
            warmup_episodes=warmup_episodes,
        )
    ]
    
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
    if warmup_episodes > 0:
        print(f"[Self-Play] Warmup: first {warmup_episodes} episodes vs scripted {warmup_opponent}, then self-play")
    print(f"[Self-Play] Snapshot every {cfg.self_play_snapshot_every_episodes} episodes, max {cfg.self_play_max_snapshots} snapshots")
    print(f"[Self-Play] Scripted mix: {cfg.scripted_opponent_prob:.0%} resample prob, OP1 {cfg.scripted_op1_prob:.0%} / OP2 {1-cfg.scripted_op1_prob:.0%} (learn defense like other baselines)")
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
    parser.add_argument("--warmup-episodes", type=int, default=None, help="Episodes vs scripted before self-play (default 200)")
    parser.add_argument("--warmup-opponent", type=str, default=None, help="Scripted opponent during warmup (default OP1)")
    parser.add_argument("--scripted-prob", type=float, default=None, help="Prob of scripted when resampling (default 0.35)")
    parser.add_argument("--scripted-op1-prob", type=float, default=None, help="When scripted, use OP1 this fraction (default 0.45); else OP2")
    parser.add_argument("--resample-every", type=int, default=None, help="Episodes between opponent resample (default 25)")
    args = parser.parse_args()
    
    cfg = SelfPlayConfig()
    if args.run_tag is not None:
        cfg.run_tag = args.run_tag
    if args.total_steps is not None:
        cfg.total_timesteps = args.total_steps
    if args.max_blue_agents is not None:
        n = max(1, min(int(args.max_blue_agents), 16))
        if n != int(args.max_blue_agents):
            print(f"[Self-Play] --max-blue-agents {args.max_blue_agents} out of range; clamped to {n} (max 16).")
        cfg.max_blue_agents = n
    if args.snapshot_every is not None:
        cfg.self_play_snapshot_every_episodes = args.snapshot_every
    if args.max_snapshots is not None:
        cfg.self_play_max_snapshots = args.max_snapshots
    if args.warmup_episodes is not None:
        cfg.warmup_episodes = args.warmup_episodes
    if args.warmup_opponent is not None:
        cfg.warmup_opponent = args.warmup_opponent
    if getattr(args, "scripted_prob", None) is not None:
        cfg.scripted_opponent_prob = float(args.scripted_prob)
    if getattr(args, "scripted_op1_prob", None) is not None:
        cfg.scripted_op1_prob = float(args.scripted_op1_prob)
    if getattr(args, "resample_every", None) is not None:
        cfg.resample_opponent_every_episodes = max(1, int(args.resample_every))

    train_self_play(cfg)
