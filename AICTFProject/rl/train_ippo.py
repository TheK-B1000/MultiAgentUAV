"""
IPPO (Independent PPO) baseline: each agent learns its own independent policy.

Training modes: CURRICULUM_LEAGUE, CURRICULUM_NO_LEAGUE, FIXED_OPPONENT, SELF_PLAY.
Uses MARLEnvWrapper to split team observations into per-agent observations.
Each agent trains its own PPO model independently.

Usage:
  python -m rl.train_ippo [--mode CURRICULUM_LEAGUE] [--total_steps 500000] ...
"""
from __future__ import annotations

import csv
import os
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, NatureCNN
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

import gymnasium as gym
from gymnasium import spaces

from game_field import make_game_field
from ctf_sb3_env import CTFGameFieldSB3Env
from rl.marl_env import MARLEnvWrapper
from rl.common import env_seed, set_global_seed
from rl.curriculum import (
    CurriculumConfig,
    CurriculumController,
    CurriculumControllerConfig,
    CurriculumState,
    STRESS_BY_PHASE,
)
from rl.league import EloLeague, OpponentSpec
from rl.episode_result import parse_episode_result, EpisodeSummary
from config import MAP_NAME, MAP_PATH


class SingleAgentExtractor(BaseFeaturesExtractor):
    """
    Feature extractor for single-agent observations.
    Processes individual agent's grid and vec features.
    Uses a simple CNN similar to NatureCNN but avoids initialization issues.
    """

    def __init__(self, observation_space, cnn_output_dim: int = 256, normalized_image: bool = True):
        import gymnasium as gym
        from gymnasium import spaces
        from game_field import NUM_CNN_CHANNELS, CNN_ROWS, CNN_COLS
        import torch.nn as nn

        assert isinstance(observation_space, gym.Space) and hasattr(observation_space, "spaces")
        spaces_dict = observation_space.spaces
        grid_space = spaces_dict.get("grid")
        vec_space = spaces_dict.get("vec")
        assert grid_space is not None and vec_space is not None
        
        grid_shape = getattr(grid_space, "shape", None)
        vec_shape = getattr(vec_space, "shape", None)
        assert len(grid_shape) == 3, f"single-agent grid must be (C, H, W), got {grid_shape}"
        assert len(vec_shape) == 1, f"single-agent vec must be (V,), got {vec_shape}"
        
        C, H, W = grid_shape
        V = vec_shape[0]
        
        # Ensure grid dimensions match expected values (safety check)
        if H < 3 or W < 3:
            # If grid is too small, use expected dimensions
            C = NUM_CNN_CHANNELS
            H = CNN_ROWS
            W = CNN_COLS
        
        self.vec_dim = V
        
        # Optional opponent context
        context_space = spaces_dict.get("context")
        self._context_dim = 0
        if context_space is not None and hasattr(context_space, "shape"):
            self._context_dim = int(np.prod(context_space.shape))
        
        # Calculate features_dim (will be set after CNN initialization)
        features_dim = cnn_output_dim + V + self._context_dim
        
        # Must call super().__init__() before creating modules
        super().__init__(observation_space, features_dim)
        
        # Build a simple CNN for 20x20 input (smaller kernels than NatureCNN)
        # Use smaller kernels and padding to avoid dimension issues
        self.cnn = nn.Sequential(
            nn.Conv2d(C, 32, kernel_size=4, stride=2, padding=1),  # 20x20 -> 10x10
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 10x10 -> 5x5
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),  # 5x5 -> 5x5
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Calculate CNN output size by doing a forward pass with dummy input
        with torch.no_grad():
            dummy_input = torch.zeros(1, C, H, W)
            cnn_out = self.cnn(dummy_input)
            cnn_features_dim = cnn_out.shape[1]
        
        # Project to desired output dimension
        self.cnn_proj = nn.Linear(cnn_features_dim, cnn_output_dim)

    def forward(self, observations):
        grid = observations["grid"]  # (B, C, H, W)
        vec = observations["vec"]    # (B, V)
        
        # Normalize grid if needed (NatureCNN does this)
        if grid.max() > 1.0:
            grid = grid / 255.0
        
        cnn_out = self.cnn(grid)  # (B, cnn_features_dim)
        cnn_out = self.cnn_proj(cnn_out)  # (B, cnn_output_dim)
        vec_flat = vec.view(vec.shape[0], -1)  # (B, V)
        out = torch.cat([cnn_out, vec_flat], dim=1)  # (B, cnn_output_dim+V)
        
        if self._context_dim > 0 and "context" in observations:
            ctx = observations["context"]
            if ctx.dim() == 1:
                ctx = ctx.unsqueeze(0)
            ctx = ctx.float()
            if ctx.shape[-1] != self._context_dim:
                ctx = ctx.reshape(ctx.shape[0], -1)[:, : self._context_dim]
            out = torch.cat([out, ctx], dim=1)
        
        return out


class TrainMode(str, Enum):
    CURRICULUM_LEAGUE = "CURRICULUM_LEAGUE"
    CURRICULUM_NO_LEAGUE = "CURRICULUM_NO_LEAGUE"
    FIXED_OPPONENT = "FIXED_OPPONENT"
    SELF_PLAY = "SELF_PLAY"


@dataclass
class IPPOConfig:
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

    checkpoint_dir: str = "checkpoints_ippo"
    run_tag: str = "ippo_league_curriculum"
    save_every_steps: int = 50_000
    eval_every_steps: int = 25_000
    eval_episodes: int = 6
    snapshot_every_episodes: int = 100
    league_max_snapshots: int = 25
    enable_tensorboard: bool = False
    enable_checkpoints: bool = False
    enable_eval: bool = False

    max_decision_steps: int = 900

    mode: str = TrainMode.CURRICULUM_LEAGUE.value
    fixed_opponent_tag: str = "OP3"
    self_play_use_latest_snapshot: bool = True
    self_play_snapshot_every_episodes: int = 25
    self_play_max_snapshots: int = 25

    action_flip_prob: float = 0.0
    use_deterministic: bool = False
    max_blue_agents: int = 2
    print_reset_shapes: bool = False
    reward_mode: str = "PER_AGENT"  # IPPO uses per-agent rewards
    use_obs_builder: bool = True
    include_opponent_context: bool = False
    obs_debug_validate_locality: bool = False


class SingleAgentEnvWrapper(gym.Env):
    """
    Wraps MARLEnvWrapper to expose a single agent's observation/reward/action space.
    Used to train independent policies per agent.
    Inherits from gym.Env to be compatible with SB3.
    """
    
    def __init__(self, marl_env: MARLEnvWrapper, agent_key: str):
        super().__init__()
        
        self.marl_env = marl_env
        self.agent_key = agent_key
        self._agent_idx = int(agent_key.split("_")[1])
        
        # Get observation and action spaces from MARL wrapper
        obs_dict, _ = marl_env.reset()
        self.observation_space = self._infer_obs_space(obs_dict[self.agent_key])
        self.action_space = marl_env.action_space[agent_key]
        
        # Gymnasium metadata
        self.metadata = {"render_modes": []}
        self.render_mode = None
        
    def _infer_obs_space(self, obs_sample: Dict[str, np.ndarray]):
        """Infer observation space from sample observation."""
        from game_field import NUM_CNN_CHANNELS, CNN_ROWS, CNN_COLS
        
        spaces_dict = {}
        for key, arr in obs_sample.items():
            if key == "grid":
                # Ensure grid has correct shape (C, H, W) = (7, 20, 20)
                arr_shape = arr.shape
                if len(arr_shape) == 3:
                    C, H, W = arr_shape
                    # Safety check: if dimensions are wrong, use expected values
                    if H < 3 or W < 3 or C != NUM_CNN_CHANNELS:
                        C, H, W = NUM_CNN_CHANNELS, CNN_ROWS, CNN_COLS
                else:
                    C, H, W = NUM_CNN_CHANNELS, CNN_ROWS, CNN_COLS
                
                spaces_dict["grid"] = spaces.Box(
                    low=0.0, high=1.0,
                    shape=(C, H, W), dtype=np.float32
                )
            elif key == "vec":
                spaces_dict["vec"] = spaces.Box(
                    low=-2.0, high=2.0,
                    shape=arr.shape, dtype=np.float32
                )
            elif key == "mask":
                spaces_dict["mask"] = spaces.Box(
                    low=0.0, high=1.0,
                    shape=arr.shape, dtype=np.float32
                )
            elif key == "context":
                spaces_dict["context"] = spaces.Box(
                    low=0.0, high=100.0,
                    shape=arr.shape, dtype=np.float32
                )
        
        return spaces.Dict(spaces_dict)
        
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        obs_dict, info = self.marl_env.reset(seed=seed, options=options)
        return obs_dict[self.agent_key], info
    
    def step(self, action):
        # Convert action to tuple if needed (handle numpy arrays, lists, etc.)
        if isinstance(action, np.ndarray):
            action = tuple(int(x) for x in action.flatten()[:2])
        elif isinstance(action, (list, tuple)):
            action = tuple(int(x) for x in action[:2])
        else:
            # Try to extract from object
            try:
                action = (int(action[0]), int(action[1]))
            except (TypeError, IndexError, KeyError):
                action = (0, 0)
        
        # Ensure we have exactly 2 values (macro, target)
        if len(action) != 2:
            action = (int(action[0]) if len(action) > 0 else 0, int(action[1]) if len(action) > 1 else 0)
        
        # Provide default actions for other agents (idle) - MARL wrapper needs all agents' actions
        actions_dict = {}
        for key in self.marl_env.agent_keys:
            if key == self.agent_key:
                actions_dict[key] = action
            else:
                actions_dict[key] = (0, 0)  # Default idle action for other agents
        
        obs_dict, rews_dict, dones_dict, infos_dict, shared_info = self.marl_env.step(actions_dict)
        obs = obs_dict[self.agent_key]
        reward = rews_dict[self.agent_key]
        terminated = dones_dict[self.agent_key]
        truncated = False  # MARL wrapper doesn't distinguish terminated/truncated
        info = infos_dict[self.agent_key]
        return obs, reward, terminated, truncated, info
    
    def close(self):
        self.marl_env.close()
    
    def render(self):
        """Not implemented."""
        return None


def _make_marl_env_fn(cfg: IPPOConfig, *, default_opponent: Tuple[str, str], rank: int) -> Any:
    """Create MARL environment factory."""
    def _fn():
        s = env_seed(cfg.seed, rank)
        np.random.seed(s)
        torch.manual_seed(s)
        
        base_env = CTFGameFieldSB3Env(
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
            action_flip_prob=getattr(cfg, "action_flip_prob", 0.0),
            max_blue_agents=getattr(cfg, "max_blue_agents", 2),
            print_reset_shapes=getattr(cfg, "print_reset_shapes", False),
            allow_missing_vec_features=getattr(cfg, "allow_missing_vec_features", False),
            reward_mode="PER_AGENT",  # IPPO needs per-agent rewards
            use_obs_builder=getattr(cfg, "use_obs_builder", True),
            include_opponent_context=getattr(cfg, "include_opponent_context", False),
            obs_debug_validate_locality=getattr(cfg, "obs_debug_validate_locality", False),
        )
        
        marl_env = MARLEnvWrapper(base_env, add_agent_id_to_vec=True)
        return marl_env
    
    return _fn


class MaskedSingleAgentPolicy(MultiInputActorCriticPolicy):
    """
    Policy for single agent with action masking.
    Handles macro + target action masking.
    """
    
    def _apply_action_mask(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if mask is None:
            return logits
        if mask.dim() == 1:
            mask = mask.unsqueeze(0)
        mask = mask.float()
        
        # Single agent: (macro, target) = 2 action components
        if hasattr(self.action_dist, "action_dims"):
            dims = list(self.action_dist.action_dims)
        else:
            dims = list(getattr(self.action_space, "nvec", []))
        if len(dims) != 2:
            return logits
        
        n_macros = int(dims[0])
        n_targets = int(dims[1])
        
        # Mask layout: [macro_mask, target_mask]
        expected = n_macros + n_targets
        if mask.shape[1] < expected:
            pad = torch.ones((mask.shape[0], expected - mask.shape[1]), device=mask.device)
            mask = torch.cat([mask, pad], dim=1)
        
        macro_mask = mask[:, :n_macros]
        target_mask = mask[:, n_macros:n_macros + n_targets]
        
        full_mask = torch.cat([macro_mask, target_mask], dim=1)
        invalid = (full_mask <= 0.0)
        return logits.masked_fill(invalid, -1e8)
    
    def get_distribution(self, obs: Dict[str, torch.Tensor]):
        features = self.extract_features(obs)
        latent_pi, _ = self.mlp_extractor(features)
        logits = self.action_net(latent_pi)
        if isinstance(obs, dict) and "mask" in obs:
            logits = self._apply_action_mask(logits, obs["mask"])
        return self.action_dist.proba_distribution(action_logits=logits)
    
    def forward(self, obs: Dict[str, torch.Tensor], deterministic: bool = False):
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        logits = self.action_net(latent_pi)
        if isinstance(obs, dict) and "mask" in obs:
            logits = self._apply_action_mask(logits, obs["mask"])
        distribution = self.action_dist.proba_distribution(action_logits=logits)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        return actions, values, log_prob


class IPPOLoggerCallback(BaseCallback):
    """Logs episode results for IPPO training."""
    
    def __init__(self, agent_key: str, verbose: int = 1):
        super().__init__(verbose)
        self.agent_key = agent_key
        self.episode_idx = 0
        self.win_count = 0
        self.loss_count = 0
        self.draw_count = 0
    
    def _on_step(self) -> bool:
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
            
            if self.verbose:
                print(
                    f"[IPPO|{self.agent_key}] ep={self.episode_idx} result={result} "
                    f"score={blue_score}:{red_score} "
                    f"W={self.win_count} | L={self.loss_count} | D={self.draw_count}"
                )
            
            self.logger.record(f"ippo_{self.agent_key}/episode", self.episode_idx)
            self.logger.record(f"ippo_{self.agent_key}/win_rate", self.win_count / max(1, self.episode_idx))
            self.logger.record(f"ippo_{self.agent_key}/draw_rate", self.draw_count / max(1, self.episode_idx))
        
        return True


def train_ippo(cfg: Optional[IPPOConfig] = None) -> None:
    cfg = cfg or IPPOConfig()
    set_global_seed(cfg.seed, torch_seed=True, deterministic=cfg.use_deterministic)
    
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    mode = str(cfg.mode).upper().strip()
    
    league = EloLeague(
        seed=cfg.seed,
        k_factor=32.0,
        matchmaking_tau=200.0,
        scripted_floor=0.50,
        species_prob=0.20,
        snapshot_prob=0.30,
    )
    
    curriculum: Optional[CurriculumState] = None
    controller: Optional[CurriculumController] = None
    if mode == TrainMode.CURRICULUM_LEAGUE.value:
        curriculum = CurriculumState(
            CurriculumConfig(
                phases=["OP1", "OP2", "OP3"],
                min_episodes={"OP1": 200, "OP2": 200, "OP3": 250},
                min_winrate={"OP1": 0.50, "OP2": 0.50, "OP3": 0.55},
                winrate_window=50,
                required_win_by={"OP1": 0, "OP2": 1, "OP3": 1},
                elo_margin=80.0,
                switch_to_league_after_op3_win=False,
            )
        )
        controller = CurriculumController(
            CurriculumControllerConfig(seed=cfg.seed),
            league=league,
        )
    elif mode == TrainMode.CURRICULUM_NO_LEAGUE.value:
        curriculum = CurriculumState(
            CurriculumConfig(
                phases=["OP1", "OP2", "OP3"],
                min_episodes={"OP1": 200, "OP2": 200, "OP3": 250},
                min_winrate={"OP1": 0.50, "OP2": 0.50, "OP3": 0.55},
                winrate_window=50,
                required_win_by={"OP1": 0, "OP2": 1, "OP3": 1},
                elo_margin=80.0,
                switch_to_league_after_op3_win=False,
            )
        )
    
    if mode == TrainMode.FIXED_OPPONENT.value:
        default_opponent = ("SCRIPTED", str(cfg.fixed_opponent_tag).upper())
        phase_name = str(cfg.fixed_opponent_tag).upper()
    elif mode == TrainMode.SELF_PLAY.value:
        default_opponent = ("SCRIPTED", "OP3")
        phase_name = "SELF_PLAY"
    elif mode == TrainMode.CURRICULUM_NO_LEAGUE.value and curriculum is not None:
        default_opponent = ("SCRIPTED", curriculum.phase)
        phase_name = curriculum.phase
    else:
        default_opponent = ("SCRIPTED", "OP1")
        phase_name = curriculum.phase if curriculum is not None else "OP1"
    
    # Create MARL environment to get agent keys
    test_marl_env_fn = _make_marl_env_fn(cfg, default_opponent=default_opponent, rank=0)
    test_marl_env = test_marl_env_fn()
    test_marl_env.reset()
    agent_keys = test_marl_env.agent_keys
    n_agents = len(agent_keys)
    test_marl_env.close()
    
    print(f"[IPPO] Training {n_agents} independent policies for agents: {agent_keys}")
    
    # Train one PPO model per agent
    models: Dict[str, Any] = {}
    
    for agent_key in agent_keys:
        print(f"[IPPO] Initializing policy for {agent_key}...")
        
        # Create per-agent environment wrapper
        def _make_agent_env_fn(agent_k: str, rank: int):
            def _fn():
                s = env_seed(cfg.seed, rank)
                np.random.seed(s)
                torch.manual_seed(s)
                marl_env = _make_marl_env_fn(cfg, default_opponent=default_opponent, rank=rank)()
                wrapper = SingleAgentEnvWrapper(marl_env, agent_k)
                return wrapper
            return _fn
        
        env_fns = [_make_agent_env_fn(agent_key, rank=i) for i in range(max(1, int(cfg.n_envs)))]
        try:
            venv = SubprocVecEnv(env_fns)
        except Exception:
            venv = DummyVecEnv(env_fns)
        venv = VecMonitor(venv)
        
        # Set phase/stress on underlying MARL envs (access through wrapper)
        try:
            for i in range(len(env_fns)):
                env_wrapper = venv.envs[i]
                if hasattr(env_wrapper, "marl_env") and hasattr(env_wrapper.marl_env, "_env"):
                    base_env = env_wrapper.marl_env._env
                    if hasattr(base_env, "set_stress_schedule"):
                        base_env.set_stress_schedule(STRESS_BY_PHASE)
                    if hasattr(base_env, "set_phase"):
                        base_env.set_phase(phase_name)
        except Exception:
            pass
        
        policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))
        obs_spaces = getattr(venv.observation_space, "spaces", None) or {}
        if "grid" in obs_spaces and "vec" in obs_spaces:
            policy_kwargs["features_extractor_class"] = SingleAgentExtractor
            policy_kwargs["features_extractor_kwargs"] = dict(cnn_output_dim=256, normalized_image=True)
        
        model = PPO(
            policy=MaskedSingleAgentPolicy,
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
                os.path.join(cfg.checkpoint_dir, "tb", f"{cfg.run_tag}_{agent_key}")
                if cfg.enable_tensorboard
                else None
            ),
            policy_kwargs=policy_kwargs,
            verbose=0,
            seed=cfg.seed + hash(agent_key) % 1000,  # Different seed per agent
            device=cfg.device,
        )
        
        if cfg.enable_tensorboard:
            model.set_logger(configure(os.path.join(cfg.checkpoint_dir, "tb", f"{cfg.run_tag}_{agent_key}"), ["tensorboard"]))
        else:
            model.set_logger(configure(None, []))
        
        models[agent_key] = model
    
    # Training loop: train all agents in parallel (true IPPO)
    print(f"[IPPO] Starting parallel training for {n_agents} agents...")
    
    total_steps_per_agent = int(cfg.total_timesteps) // n_agents
    
    callbacks_per_agent = {}
    for agent_key in agent_keys:
        callbacks = [IPPOLoggerCallback(agent_key=agent_key, verbose=1)]
        if cfg.enable_checkpoints:
            callbacks.append(
                CheckpointCallback(
                    save_freq=int(cfg.save_every_steps),
                    save_path=cfg.checkpoint_dir,
                    name_prefix=f"ckpt_{cfg.run_tag}_{agent_key}",
                )
            )
        callbacks_per_agent[agent_key] = CallbackList(callbacks)
    
    # Train all agents in parallel using threading
    threads = []
    
    def train_agent(agent_key: str):
        """Train a single agent's model."""
        print(f"[IPPO] Starting training for {agent_key}...")
        model = models[agent_key]
        callbacks = callbacks_per_agent[agent_key]
        model.learn(total_timesteps=total_steps_per_agent, callback=callbacks)
        print(f"[IPPO] Finished training {agent_key}")
    
    # Start all training threads
    for agent_key in agent_keys:
        thread = threading.Thread(target=train_agent, args=(agent_key,))
        thread.start()
        threads.append(thread)
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    print(f"[IPPO] All agents finished training")
    
    # Save final models
    for agent_key in agent_keys:
        final_path = os.path.join(cfg.checkpoint_dir, f"final_{cfg.run_tag}_{agent_key}")
        models[agent_key].save(final_path)
        print(f"[IPPO] Saved {agent_key} model to: {final_path}.zip")
    
    print(f"[IPPO] Training complete. Models saved to: {cfg.checkpoint_dir}")


if __name__ == "__main__":
    train_ippo()
