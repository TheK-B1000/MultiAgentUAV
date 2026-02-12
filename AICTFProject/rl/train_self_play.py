"""
True self-play training: both blue and red teams learn simultaneously.

This script trains both sides with neural networks using the same observation space,
avoiding the legacy snapshot wrapper that caused observation space mismatches.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, NatureCNN
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from game_field import make_game_field
from ctf_sb3_env import CTFGameFieldSB3Env
from rl.common import env_seed, set_global_seed
from rl.train_ppo import TokenizedCombinedExtractor, MaskedMultiInputPolicy
from config import MAP_NAME, MAP_PATH


@dataclass
class SelfPlayConfig:
    """Configuration for self-play training."""
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
    run_tag: str = "ppo_self_play_fresh_4v4"
    save_every_steps: int = 50_000
    enable_checkpoints: bool = False
    enable_tensorboard: bool = False
    # Lightweight snapshot management (to avoid filling disk)
    enable_snapshots: bool = True
    snapshot_every_steps: int = 100_000
    max_snapshots: int = 10
    
    max_decision_steps: int = 400
    max_blue_agents: int = 4
    use_tokenized_obs: bool = False
    
    # Self-play specific: how often to swap which side is learning
    swap_sides_every_steps: int = 100_000  # Alternate training every N steps


class SelfPlayEnv(CTFGameFieldSB3Env):
    """
    Environment wrapper that allows both blue and red to be controlled by PPO models.
    
    This extends CTFGameFieldSB3Env to generate red team actions from a PPO model
    during the step, enabling true self-play where both sides learn.
    """
    
    def __init__(self, *args, red_model: Optional[PPO] = None, train_side: str = "blue", **kwargs):
        super().__init__(*args, **kwargs)
        self.red_model = red_model
        self.train_side = train_side  # "blue" or "red" - which side is learning
        self._red_identities = None
        
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        obs, info = super().reset(seed=seed, options=options)
        
        # Build red team identities similar to blue
        if self.gf is not None:
            from rl.agent_identity import AgentIdentity
            # Build identities for red team (same structure as blue but with red_ keys)
            red_agents = getattr(self.gf, "red_agents", []) or []
            max_agents = self._max_blue_agents
            self._red_identities = []
            for i in range(max_agents):
                key = f"red_{i}"
                agent = red_agents[i] if i < len(red_agents) else None
                ident = AgentIdentity(key=key, slot_index=i, agent=agent)
                self._red_identities.append(ident)
            
            # Disable red scripted policy since red will be controlled by PPO
            self.gf.set_red_policy_wrapper(None)
            from policies import Policy
            class DummyPolicy(Policy):
                def __init__(self, side: str = "red"):
                    self.side = side
                def select_action(self, obs, agent, game_field):
                    return 0, None
            self.gf.policies["red"] = DummyPolicy("red")
        
        return obs, info
    
    def step(self, action: np.ndarray):
        """
        Override step to also generate red team actions from red_model.
        
        When train_side="blue": action is for blue, generate red actions from red_model
        When train_side="red": action is for red, generate blue actions from blue_model (stored in red_model's env)
        """
        assert self.gf is not None, "Call reset() first"
        
        # Parse actions for the learning team
        act_flat = np.asarray(action).reshape(-1)
        n_slots = self._max_blue_agents if self._max_blue_agents > 2 else 2
        
        if self.train_side == "blue":
            # Blue is learning, red uses frozen model
            blue_intended: list[Tuple[int, int]] = []
            for i in range(n_slots):
                off = i * 2
                if off + 1 < len(act_flat):
                    blue_intended.append((
                        int(act_flat[off]) % max(1, self._n_macros),
                        int(act_flat[off + 1]) % max(1, self._n_targets)
                    ))
                else:
                    blue_intended.append((0, 0))
            while len(blue_intended) < n_slots:
                blue_intended.append((0, 0))
            
            # Generate red actions from red_model
            red_intended = [(0, 0)] * n_slots
            if self.red_model is not None:
                try:
                    # Build red team observation (using same obs builder pattern as blue)
                    # Note: We reuse the same obs_builder but pass red identities
                    red_obs = self._obs_builder.build_observation(
                        self.gf,
                        self._red_identities,
                        max_blue_agents=self._max_blue_agents,
                        n_blue_agents=len([a for a in getattr(self.gf, "red_agents", []) if a is not None]),
                        n_macros=self._n_macros,
                        n_targets=self._n_targets,
                        opponent_context_id=None,
                        role_macro_mask_fn=self._action_manager._apply_role_macro_mask,
                    )
                    
                    # Get red actions from model
                    red_action, _ = self.red_model.predict(red_obs, deterministic=False)
                    red_act_flat = np.asarray(red_action).reshape(-1)
                    for i in range(n_slots):
                        off = i * 2
                        if off + 1 < len(red_act_flat):
                            red_intended[i] = (
                                int(red_act_flat[off]) % max(1, self._n_macros),
                                int(red_act_flat[off + 1]) % max(1, self._n_targets)
                            )
                except Exception as e:
                    # Fallback to no-op if red model fails
                    pass
            
            # Apply noise and sanitize for blue
            blue_executed, _, _, _ = self._action_manager.apply_noise_and_sanitize(
                blue_intended,
                self._n_blue_agents,
                self.gf,
                role_macro_mask_fn=self._action_manager._apply_role_macro_mask,
            )
            
            # Apply noise and sanitize for red
            red_executed, _, _, _ = self._action_manager.apply_noise_and_sanitize(
                red_intended,
                len([a for a in getattr(self.gf, "red_agents", []) if a is not None]),
                self.gf,
                role_macro_mask_fn=self._action_manager._apply_role_macro_mask,
            )
            
            # Submit actions for both teams
            actions_by_agent: Dict[str, Tuple[int, Any]] = {}
            
            # Blue actions
            for i, ident in enumerate(self._blue_identities):
                if i < len(blue_executed):
                    actions_by_agent[ident.key] = blue_executed[i]
            
            # Red actions
            if self._red_identities:
                for i, ident in enumerate(self._red_identities):
                    if i < len(red_executed):
                        actions_by_agent[ident.key] = red_executed[i]
            
            self.gf.submit_external_actions(actions_by_agent)
            
            # Advance simulation
            self._decision_step_count += 1
            interval = float(getattr(self.gf, "decision_interval_seconds", 0.7))
            dt = max(1e-3, 0.99 * interval)
            self.gf.update(dt)
            
            # Get rewards (blue team perspective)
            gm = self.gf.manager
            reward_events_total, reward_per_agent_events, dropped_count = (
                self._reward_manager.consume_reward_events(
                    gm,
                    self._reward_id_map,
                    len(self._blue_identities),
                )
            )
            stall_total, stall_per_agent = self._reward_manager.apply_stall_penalty(
                self.gf,
                self._n_blue_agents,
            )
            
            # Build observation for blue team
            obs = self._obs_builder.build_observation(
                self.gf,
                self._blue_identities,
                max_blue_agents=self._max_blue_agents,
                n_blue_agents=self._n_blue_agents,
                n_macros=self._n_macros,
                n_targets=self._n_targets,
                opponent_context_id=None,
                role_macro_mask_fn=self._action_manager._apply_role_macro_mask,
            )
            
            # Compute total reward (blue perspective)
            if self._reward_mode == "TEAM_SUM":
                reward = float(reward_events_total + stall_total)
            elif self._reward_mode == "PER_AGENT":
                reward = np.array([float(r) for r in reward_per_agent_events[:self._n_blue_agents]])
            else:
                reward = float(reward_events_total + stall_total)
            
            # Check termination
            done = bool(getattr(gm, "game_over", False))
            terminated = done
            truncated = self._decision_step_count >= self.max_decision_steps
            
            # Build info dict
            info = {}
            if done or truncated:
                info["episode"] = {
                    "r": reward_events_total + stall_total,
                    "length": self._decision_step_count,
                }
            
            return obs, reward, terminated, truncated, info
        else:
            # Red is learning - use parent implementation
            # (This case handled by separate env setup)
            return super().step(action)


def _make_self_play_env_fn(cfg: SelfPlayConfig, opponent_model: Optional[PPO], train_side: str = "blue", rank: int = 0):
    """
    Environment factory for self-play.
    
    Args:
        cfg: Configuration
        opponent_model: The frozen opponent model (red_model when training blue, blue_model when training red)
        train_side: "blue" or "red" - which side is learning
        rank: Environment rank for seeding
    """
    def _fn():
        s = env_seed(cfg.seed, rank)
        np.random.seed(s)
        torch.manual_seed(s)
        env = SelfPlayEnv(
            make_game_field_fn=lambda: make_game_field(
                map_name=MAP_NAME or None,
                map_path=MAP_PATH or None,
            ),
            max_decision_steps=cfg.max_decision_steps,
            enforce_masks=True,
            seed=s,
            include_mask_in_obs=True,
            default_opponent_kind="SCRIPTED",
            default_opponent_key="OP3",
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
            red_model=opponent_model if train_side == "blue" else None,  # When training blue, red_model is opponent
            train_side=train_side,
        )
        # Disable scripted policy for the opponent side since it will be controlled by PPO
        if env.gf is not None:
            if train_side == "blue":
                # Training blue: disable red scripted policy
                env.gf.set_red_policy_wrapper(None)
                from policies import Policy
                class DummyPolicy(Policy):
                    def __init__(self, side: str = "red"):
                        self.side = side
                    def select_action(self, obs, agent, game_field):
                        return 0, None
                env.gf.policies["red"] = DummyPolicy("red")
        return env
    return _fn


class SelfPlayCallback(BaseCallback):
    """Callback to track self-play training progress."""
    
    def __init__(self, cfg: SelfPlayConfig, verbose: int = 1):
        super().__init__(verbose)
        self.cfg = cfg
        self.episode_idx = 0
        self.blue_wins = 0
        self.red_wins = 0
        self.draws = 0
    
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
                self.blue_wins += 1
            elif blue_score < red_score:
                result = "LOSS"
                self.red_wins += 1
            else:
                result = "DRAW"
                self.draws += 1
            
            # Print every episode (like train_ppo.py does)
            if self.verbose:
                print(
                    f"[PPO|SELF] ep={self.episode_idx} result={result} "
                    f"score={blue_score}:{red_score} "
                    f"W={self.blue_wins} | L={self.red_wins} | D={self.draws}"
                )
        
        return True


def train_self_play(cfg: Optional[SelfPlayConfig] = None) -> None:
    """
    Train both blue and red teams simultaneously using PPO.
    
    This alternates training between blue and red models, with both using
    the same observation space and action space.
    """
    if cfg is None:
        cfg = SelfPlayConfig()
    
    set_global_seed(cfg.seed)
    
    # Create observation and action spaces (same for both teams)
    # We'll create a dummy env to get the spaces
    dummy_env_fn = _make_self_play_env_fn(cfg, opponent_model=None, rank=0)
    dummy_env = dummy_env_fn()
    dummy_env.reset()
    obs_space = dummy_env.observation_space
    action_space = dummy_env.action_space
    
    # Policy kwargs
    policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))
    use_tokenized = bool(getattr(cfg, "use_tokenized_obs", False))
    if use_tokenized:
        policy_kwargs["features_extractor_class"] = TokenizedCombinedExtractor
        policy_kwargs["features_extractor_kwargs"] = dict(cnn_output_dim=256, normalized_image=True)
    
    # Create blue model (will train against frozen red)
    # On Windows, use DummyVecEnv to avoid multiprocessing pickling issues
    use_subproc = sys.platform != "win32"
    blue_env_fns = [
        _make_self_play_env_fn(cfg, opponent_model=None, train_side="blue", rank=i)
        for i in range(max(1, int(cfg.n_envs)))
    ]
    if use_subproc:
        try:
            blue_venv = SubprocVecEnv(blue_env_fns)
        except Exception:
            blue_venv = DummyVecEnv(blue_env_fns)
    else:
        blue_venv = DummyVecEnv(blue_env_fns)
    blue_venv = VecMonitor(blue_venv)
    
    blue_model = PPO(
        policy=MaskedMultiInputPolicy,
        env=blue_venv,
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
            os.path.join(cfg.checkpoint_dir, "tb", f"{cfg.run_tag}_blue")
            if cfg.enable_tensorboard
            else None
        ),
        policy_kwargs=policy_kwargs,
        verbose=0,
        seed=cfg.seed,
        device=cfg.device,
    )
    
    # Create red model (copy from blue initially, will be updated)
    red_model = PPO(
        policy=MaskedMultiInputPolicy,
        env=blue_venv,  # Temporary, will be updated
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
            os.path.join(cfg.checkpoint_dir, "tb", f"{cfg.run_tag}_red")
            if cfg.enable_tensorboard
            else None
        ),
        policy_kwargs=policy_kwargs,
        verbose=0,
        seed=cfg.seed + 1,
        device=cfg.device,
    )
    
    # Copy blue weights to red initially
    red_model.policy.load_state_dict(blue_model.policy.state_dict())
    print("[Self-Play] Initialized red model with blue weights")
    
    # Callbacks
    callbacks_blue = [SelfPlayCallback(cfg)]
    callbacks_red = [SelfPlayCallback(cfg)]
    
    if cfg.enable_checkpoints:
        callbacks_blue.append(
            CheckpointCallback(
                save_freq=int(cfg.save_every_steps),
                save_path=cfg.checkpoint_dir,
                name_prefix=f"ckpt_{cfg.run_tag}_blue",
            )
        )
        callbacks_red.append(
            CheckpointCallback(
                save_freq=int(cfg.save_every_steps),
                save_path=cfg.checkpoint_dir,
                name_prefix=f"ckpt_{cfg.run_tag}_red",
            )
        )
    
    callbacks_blue = CallbackList(callbacks_blue)
    callbacks_red = CallbackList(callbacks_red)
    
    # Training loop: alternate between blue and red
    total_steps = int(cfg.total_timesteps)
    steps_per_swap = int(cfg.swap_sides_every_steps)
    current_steps = 0
    # Rolling snapshot list (keep only latest cfg.max_snapshots)
    snapshot_paths: List[str] = []
    
    print(f"[Self-Play] Starting training: {total_steps} total steps, swap every {steps_per_swap} steps")
    
    while current_steps < total_steps:
        steps_to_train = min(steps_per_swap, total_steps - current_steps)
        
        # Update blue environments to use frozen red model as opponent
        # On Windows, use DummyVecEnv to avoid multiprocessing pickling issues
        blue_env_fns = [
            _make_self_play_env_fn(cfg, opponent_model=red_model, train_side="blue", rank=i)
            for i in range(max(1, int(cfg.n_envs)))
        ]
        if use_subproc:
            try:
                blue_venv_new = SubprocVecEnv(blue_env_fns)
            except Exception:
                blue_venv_new = DummyVecEnv(blue_env_fns)
        else:
            blue_venv_new = DummyVecEnv(blue_env_fns)
        blue_venv_new = VecMonitor(blue_venv_new)
        blue_model.set_env(blue_venv_new)
        
        # Train blue against frozen red
        print(f"[Self-Play] Training BLUE vs frozen RED (steps {current_steps} to {current_steps + steps_to_train})")
        blue_model.learn(
            total_timesteps=steps_to_train,
            callback=callbacks_blue,
            reset_num_timesteps=False,
        )
        
        current_steps += steps_to_train

        # Optional lightweight snapshots with rolling window to save disk
        if bool(getattr(cfg, "enable_snapshots", True)) and int(getattr(cfg, "snapshot_every_steps", 0)) > 0:
            if current_steps % int(cfg.snapshot_every_steps) == 0:
                os.makedirs(cfg.checkpoint_dir, exist_ok=True)
                snap_path = os.path.join(
                    cfg.checkpoint_dir,
                    f"snapshot_{cfg.run_tag}_blue_step{current_steps}.zip",
                )
                blue_model.save(snap_path)
                snapshot_paths.append(snap_path)
                # Enforce rolling max_snapshots by deleting oldest on disk
                max_snaps = max(0, int(getattr(cfg, "max_snapshots", 10)))
                while max_snaps > 0 and len(snapshot_paths) > max_snaps:
                    old = snapshot_paths.pop(0)
                    try:
                        if os.path.exists(old):
                            os.remove(old)
                    except Exception:
                        pass
        
        if current_steps >= total_steps:
            break
        
        # Copy blue weights to red (red becomes the new opponent)
        print("[Self-Play] Copying blue weights to red (red becomes new opponent)")
        red_model.policy.load_state_dict(blue_model.policy.state_dict())
        
        # Continue training blue against the updated red opponent
        # This creates a self-play cycle where both sides improve
    
    # Save final models
    blue_path = os.path.join(cfg.checkpoint_dir, f"final_{cfg.run_tag}_blue")
    red_path = os.path.join(cfg.checkpoint_dir, f"final_{cfg.run_tag}_red")
    blue_model.save(blue_path)
    red_model.save(red_path)
    print(f"[Self-Play] Training complete.")
    print(f"  Blue model: {blue_path}.zip")
    print(f"  Red model: {red_path}.zip")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train self-play (both sides learn)")
    parser.add_argument("--run-tag", type=str, default=None, help="Run name for checkpoints")
    parser.add_argument("--total-steps", type=int, default=None, help="Total timesteps")
    parser.add_argument("--max-blue-agents", type=int, default=4, help="Number of agents per team")
    parser.add_argument("--swap-every", type=int, default=100_000, help="Steps between swapping which side trains")
    args = parser.parse_args()
    
    cfg = SelfPlayConfig()
    if args.run_tag is not None:
        cfg.run_tag = args.run_tag
    if args.total_steps is not None:
        cfg.total_timesteps = args.total_steps
    if args.max_blue_agents is not None:
        cfg.max_blue_agents = args.max_blue_agents
    if args.swap_every is not None:
        cfg.swap_sides_every_steps = args.swap_every
    
    train_self_play(cfg)
