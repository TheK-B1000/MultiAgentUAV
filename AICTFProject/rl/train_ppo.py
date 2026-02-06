from __future__ import annotations

import csv
import os
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

from game_field import make_game_field
from ctf_sb3_env import CTFGameFieldSB3Env
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


class TokenizedCombinedExtractor(BaseFeaturesExtractor):
    """
    Feature extractor for tokenized/set-based obs: agents as sequence of tokens.
    grid (B, M, C, H, W), vec (B, M, V) -> run same CNN/MLP per token, flatten to (B, M*latent).
    Enables zero-shot: train 2v2 (mask 2), test 4v4 or 8v8 (mask 4 or 8).
    """

    def __init__(self, observation_space, cnn_output_dim: int = 256, normalized_image: bool = True):
        import gymnasium as gym
        from gymnasium import spaces
        from stable_baselines3.common.preprocessing import get_flattened_obs_dim

        assert isinstance(observation_space, gym.Space) and hasattr(observation_space, "spaces")
        spaces_dict = observation_space.spaces
        grid_space = spaces_dict.get("grid")
        vec_space = spaces_dict.get("vec")
        assert grid_space is not None and vec_space is not None
        grid_shape = getattr(grid_space, "shape", None)
        vec_shape = getattr(vec_space, "shape", None)
        assert len(grid_shape) == 4, f"tokenized grid must be (M, C, H, W), got {grid_shape}"
        assert len(vec_shape) == 2, f"tokenized vec must be (M, V), got {vec_shape}"
        M, C, H, W = grid_shape
        V = vec_shape[1]
        # Single-agent grid space for NatureCNN
        single_grid = spaces.Box(
            low=float(grid_space.low.min()) if hasattr(grid_space, "low") else 0.0,
            high=float(grid_space.high.max()) if hasattr(grid_space, "high") else 1.0,
            shape=(C, H, W),
            dtype=grid_space.dtype,
        )
        self._M = M
        self._V = V
        self.cnn = NatureCNN(single_grid, features_dim=cnn_output_dim, normalized_image=normalized_image)
        self.vec_dim = V
        features_dim = M * cnn_output_dim + M * V
        # Optional opponent context: shape (1,) when include_opponent_context=True
        context_space = spaces_dict.get("context")
        self._context_dim = 0
        if context_space is not None and hasattr(context_space, "shape"):
            self._context_dim = int(np.prod(context_space.shape))
            features_dim += self._context_dim
        super().__init__(observation_space, features_dim)

    def forward(self, observations):
        grid = observations["grid"]  # (B, M, C, H, W)
        vec = observations["vec"]    # (B, M, V)
        B = grid.shape[0]
        M = self._M

        # (B, M, C, H, W) -> (B*M, C, H, W)
        grid_flat = grid.view(B * M, *grid.shape[2:])
        cnn_out = self.cnn(grid_flat)  # (B*M, D)
        D = cnn_out.shape[1]
        cnn_out = cnn_out.view(B, M, D)  # (B, M, D)

        # Optional agent mask to zero padded slots (B, M)
        agent_mask = observations.get("agent_mask", None)
        if agent_mask is not None:
            if agent_mask.dim() == 1:
                agent_mask = agent_mask.unsqueeze(0)
            agent_mask = agent_mask.float().unsqueeze(-1)  # (B, M, 1)
            cnn_out = cnn_out * agent_mask
            vec = vec * agent_mask

        cnn_out = cnn_out.view(B, M * D)
        vec_flat = vec.view(B, M * self._V)
        out = torch.cat([cnn_out, vec_flat], dim=1)
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
    CURRICULUM_NO_LEAGUE = "CURRICULUM_NO_LEAGUE"  # OLD baseline: OP1 -> OP2 -> OP3, no league
    FIXED_OPPONENT = "FIXED_OPPONENT"
    SELF_PLAY = "SELF_PLAY"


@dataclass
class PPOConfig:
    seed: int = 42
    total_timesteps: int = 3_100_000
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
    run_tag: str = "ppo_league_curriculum_v3"
    save_every_steps: int = 50_000
    eval_every_steps: int = 25_000
    eval_episodes: int = 6
    snapshot_every_episodes: int = 100
    league_max_snapshots: int = 25  # cap league snapshots; delete oldest to save space
    enable_tensorboard: bool = False
    enable_checkpoints: bool = False
    enable_eval: bool = False

    max_decision_steps: int = 900
    op3_gate_tag: str = "OP3_HARD"
    # When learner Elo drops below this, substitute OP3_HARD with OP3 for more winnable games.
    league_easy_scripted_elo_threshold: float = 1200.0

    mode: str = TrainMode.CURRICULUM_NO_LEAGUE.value
    fixed_opponent_tag: str = "OP3"
    self_play_use_latest_snapshot: bool = True
    self_play_snapshot_every_episodes: int = 25
    self_play_max_snapshots: int = 25

    # Action execution noise (reliability metrics)
    action_flip_prob: float = 0.0

    # Reproducibility: deterministic PyTorch (slower, full reproducibility)
    use_deterministic: bool = False

    # Zero-shot scalability: max_blue_agents > 2 uses tokenized obs/action (train 2v2, test 4v4/8v8)
    # Sprint B 4v4 smoke test: set max_blue_agents=4, mode=FIXED_OPPONENT, fixed_opponent_tag=OP1 or OP2
    max_blue_agents: int = 2
    # Sprint A verification: print obs/action shapes once per reset
    print_reset_shapes: bool = False

    # Reward contract: TEAM_SUM (default), PER_AGENT, SHAPLEY_APPROX (future). Logged in info["reward_mode"].
    reward_mode: str = "TEAM_SUM"
    # Step 3.1: use canonical ObsBuilder (rollback: set False to keep legacy _get_obs path)
    use_obs_builder: bool = True
    # Step 4.1: add obs["context"] = opponent embedding id (stable int). Default off until ready.
    include_opponent_context: bool = False
    # Step 5.1: when True, GameField.build_continuous_features asserts no global state in vec (debug only).
    obs_debug_validate_locality: bool = False
    
    # Training stability improvements
    enable_opponent_tracking: bool = True  # Track opponent distribution and results
    opponent_tracking_window: int = 100  # Rolling window size for opponent stats
    enable_fixed_eval: bool = True  # Run fixed eval suite (no learning)
    # League species / RUSHER: increase species exposure and bias toward RUSHER to fix RUSHER weakness
    stability_species_prob: float = 0.15  # Fraction of league episodes vs species (BALANCED/RUSHER/CAMPER)
    stability_snapshot_prob: float = 0.20  # Fraction of league episodes vs snapshot opponents
    species_rusher_bias: float = 0.5  # When species is picked, probability of forcing RUSHER (0=uniform)
    fixed_eval_every_episodes: int = 500  # Run fixed eval every N episodes
    fixed_eval_episodes: int = 10  # Episodes per opponent in fixed eval
    # Phase 2 fixed-eval gating: advance only if fixed-eval WR meets threshold (None = disabled)
    fixed_eval_gate_OP1_wr: Optional[float] = 0.90  # advance to OP2 only if fixed eval vs OP1 >= this
    fixed_eval_gate_OP2_wr: Optional[float] = 0.75  # advance to OP3 only if fixed eval vs OP2 >= this
    # OP3_HARD exposure cap until fixed-eval vs OP3 is stable (avoid spicy too early)
    op3_hard_max_fraction_until_stable: float = 0.05  # max 5% OP3_HARD until OP3 fixed-eval stable
    op3_stable_fixed_eval_wr: float = 0.55  # OP3 fixed-eval WR >= this to allow full OP3_HARD
    # Reduced aggressiveness (if training unstable)
    use_reduced_aggressiveness: bool = False  # Enable gentler updates
    # Stable MARL PPO: gentler defaults for multi-agent nonstationarity (Fix 4.1)
    use_stable_marl_ppo: bool = False  # lr=1.5e-4, n_epochs=4, clip_range=0.12, ent_coef=0.005, batch_size=1024
    # KL guardrail: auto-reduce aggressiveness if approx_kl exceeds threshold repeatedly (Fix 4.2)
    approx_kl_threshold: float = 0.03  # target approx_kl ~0.01–0.03; spike above = over-updating
    kl_guardrail_consecutive: int = 3  # trigger after this many consecutive spikes
    # Explicit tokenized obs gate (Fix 5.1): use tokenized extractor when max_blue_agents>2 OR use_tokenized_obs
    use_tokenized_obs: bool = False


def _make_env_fn(cfg: PPOConfig, *, default_opponent: Tuple[str, str], rank: int) -> Any:
    """Env factory; per-env seed via env_seed so DummyVecEnv/SubprocVecEnv behave the same."""
    def _fn():
        s = env_seed(cfg.seed, rank)
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
            action_flip_prob=getattr(cfg, "action_flip_prob", 0.0),
            max_blue_agents=getattr(cfg, "max_blue_agents", 2),
            print_reset_shapes=getattr(cfg, "print_reset_shapes", False),
            allow_missing_vec_features=getattr(cfg, "allow_missing_vec_features", False),
            reward_mode=getattr(cfg, "reward_mode", "TEAM_SUM"),
            use_obs_builder=getattr(cfg, "use_obs_builder", True),
            include_opponent_context=getattr(cfg, "include_opponent_context", False),
            obs_debug_validate_locality=getattr(cfg, "obs_debug_validate_locality", False),
            auxiliary_progress_scale=getattr(cfg, "auxiliary_progress_scale", 0.1),
        )
        return env
    return _fn


class MaskedMultiInputPolicy(MultiInputActorCriticPolicy):
    """
    Apply action masks to discrete macro + target logits (MultiDiscrete).
    Mask layout: [macro0, targets0, macro1, targets1, ...] for 2 or max_blue_agents agents.
    Supports tokenized (zero-shot) when action dims length is 2*max_blue_agents.
    """

    def _apply_action_mask(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if mask is None:
            return logits
        if mask.dim() == 1:
            mask = mask.unsqueeze(0)
        mask = mask.float()

        if hasattr(self.action_dist, "action_dims"):
            dims = list(self.action_dist.action_dims)
        else:
            dims = list(getattr(self.action_space, "nvec", []))
        if not dims:
            return logits

        n_macros = int(dims[0]) if len(dims) > 0 else 0
        n_targets = int(dims[1]) if len(dims) > 1 else 0
        if n_macros <= 0:
            return logits

        # Layout: (macro, target) per agent; dims = [n_macros, n_targets, n_macros, n_targets, ...]
        num_agents = len(dims) // 2
        expected = num_agents * (n_macros + n_targets) if num_agents > 0 else 2 * (n_macros + n_targets)
        if mask.shape[1] < expected:
            pad = torch.ones((mask.shape[0], expected - mask.shape[1]), device=mask.device)
            mask = torch.cat([mask, pad], dim=1)

        full_mask = []
        offset = 0
        for i, dim in enumerate(dims):
            if i % 2 == 0:  # macro
                sz = min(n_macros, mask.shape[1] - offset)
                if sz > 0:
                    full_mask.append(mask[:, offset: offset + sz])
                    offset += sz  # ✅ Fix: Increment by sz (actual size used), not n_macros
                else:
                    full_mask.append(torch.ones((mask.shape[0], int(dim)), device=mask.device))
            else:  # target
                sz = min(n_targets, mask.shape[1] - offset)
                if sz > 0 and n_targets > 0:
                    full_mask.append(mask[:, offset: offset + sz])
                    offset += sz  # ✅ Fix: Increment by sz (actual size used), not n_targets
                else:
                    full_mask.append(torch.ones((mask.shape[0], int(dim)), device=mask.device))

        mask_cat = torch.cat(full_mask, dim=1)
        invalid = (mask_cat <= 0.0)
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


class LeagueCallback(BaseCallback):
    def __init__(
        self,
        *,
        cfg: PPOConfig,
        league: EloLeague,
        curriculum: CurriculumState,
        controller: CurriculumController,
    ) -> None:
        super().__init__(verbose=1)
        self.cfg = cfg
        self.league = league
        self.curriculum = curriculum
        self.controller = controller
        self.episode_idx = 0
        self.league_mode = False
        self.win_count = 0
        self.loss_count = 0
        self.draw_count = 0
        self._league_max_snapshots = max(0, int(getattr(cfg, "league_max_snapshots", 25)))
        
        # Step 1: Opponent distribution tracking
        self._opponent_stats: Dict[str, Dict[str, int]] = {}  # opp_key -> {wins, losses, draws, total}
        self._opponent_history: List[Tuple[str, str]] = []  # (opp_key, result) for rolling window
        self._enable_opponent_tracking = getattr(cfg, "enable_opponent_tracking", True)
        self._opponent_window = getattr(cfg, "opponent_tracking_window", 100)
        
        # Step 5: Async episode handling - batch updates
        self._pending_updates: List[Dict[str, Any]] = []  # Collect updates, apply on reset

    def _enforce_league_snapshot_limit(self) -> None:
        """Delete oldest league snapshots when over cap to save disk space."""
        if self._league_max_snapshots <= 0:
            return
        while len(self.league.snapshots) > self._league_max_snapshots:
            oldest = self.league.snapshots.pop(0)
            try:
                if oldest and os.path.exists(oldest):
                    os.remove(oldest)
                    if self.verbose:
                        print(f"[League] deleted old snapshot: {oldest}")
            except Exception as exc:
                print(f"[WARN] league snapshot cleanup failed: {exc}")

    def _select_next_opponent(self) -> OpponentSpec:
        if self.league_mode:
            # Sprint A: Pass phase to league for stability mix
            return self.league.sample_league(phase=self.curriculum.phase)
        return self.controller.select_opponent(self.curriculum.phase, league_mode=self.league_mode)
    
    def _update_opponent_stats(self, opp_key: str, result: str):
        """Track opponent distribution and results (Step 1)."""
        if not self._enable_opponent_tracking:
            return
        
        if opp_key not in self._opponent_stats:
            self._opponent_stats[opp_key] = {"wins": 0, "losses": 0, "draws": 0, "total": 0}
        
        self._opponent_stats[opp_key]["total"] += 1
        if result == "WIN":
            self._opponent_stats[opp_key]["wins"] += 1
        elif result == "LOSS":
            self._opponent_stats[opp_key]["losses"] += 1
        else:
            self._opponent_stats[opp_key]["draws"] += 1
        
        # Rolling window
        self._opponent_history.append((opp_key, result))
        if len(self._opponent_history) > self._opponent_window:
            old_opp_key, old_result = self._opponent_history.pop(0)
            if old_opp_key in self._opponent_stats:
                self._opponent_stats[old_opp_key]["total"] -= 1
                if old_result == "WIN":
                    self._opponent_stats[old_opp_key]["wins"] -= 1
                elif old_result == "LOSS":
                    self._opponent_stats[old_opp_key]["losses"] -= 1
                else:
                    self._opponent_stats[old_opp_key]["draws"] -= 1
    
    def _print_opponent_distribution(self):
        """Print opponent distribution (last N episodes)."""
        if not self._enable_opponent_tracking or not self._opponent_stats:
            return
        
        # Count recent opponents
        recent_opps: Dict[str, int] = {}
        for opp_key, _ in self._opponent_history[-self._opponent_window:]:
            recent_opps[opp_key] = recent_opps.get(opp_key, 0) + 1
        
        if not recent_opps:
            return
        
        print(f"[OpponentDist|last_{self._opponent_window}] ", end="")
        parts = []
        for opp_key, count in sorted(recent_opps.items(), key=lambda x: -x[1]):
            stats = self._opponent_stats.get(opp_key, {})
            wins = stats.get("wins", 0)
            losses = stats.get("losses", 0)
            draws = stats.get("draws", 0)
            total = stats.get("total", 0)
            # ✅ Fix: Include draws in win-rate calculation (wins / total, not wins / (wins + losses))
            wr = (wins / max(1, total)) * 100 if total > 0 else 0.0
            parts.append(f"{opp_key}:{count}({wins}W/{losses}L/{draws}D,{wr:.0f}%WR)")
        print(" | ".join(parts))

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
            win_by = summary.win_by

            if blue_score > red_score:
                result = "WIN"
                actual = 1.0
                win = True
                self.win_count += 1
            elif blue_score < red_score:
                result = "LOSS"
                actual = 0.0
                win = False
                self.loss_count += 1
            else:
                result = "DRAW"
                actual = 0.5
                win = False
                self.draw_count += 1

            opp_key = summary.opponent_key()
            self.league.update_elo(opp_key, actual)
            self.controller.record_result(opp_key, actual)
            # ✅ Fix: Actually call _update_opponent_stats to track opponent distribution
            self._update_opponent_stats(opp_key, result)

            phase = self.curriculum.phase
            self.curriculum.phase_episode_count += 1
            self.curriculum.record_result(phase, actual)

            is_scripted = opp_key.startswith("SCRIPTED:")
            if is_scripted:
                opp_rating = self.league.get_rating(opp_key)
                if self.curriculum.advance_if_ready(
                    learner_rating=self.league.learner_rating,
                    opponent_rating=opp_rating,
                    win_by=win_by,
                ):
                    phase = self.curriculum.phase

            if phase == "OP3" and is_scripted:
                min_eps = int(self.curriculum.config.min_episodes.get("OP3", 0))
                min_wr = float(self.curriculum.config.min_winrate.get("OP3", 0.0))
                req_win_by = int(self.curriculum.config.required_win_by.get("OP3", 0))
                meets_eps = self.curriculum.phase_episode_count >= min_eps
                meets_wr = self.curriculum.phase_winrate("OP3") >= min_wr
                meets_score = True if req_win_by <= 0 else (win_by >= req_win_by)
                gate_tag = str(self.cfg.op3_gate_tag).upper()
                gate_ok = opp_key.endswith(f":{gate_tag}")
                if self.curriculum.config.switch_to_league_after_op3_win and win and gate_ok:
                    self.league_mode = True
                elif meets_eps and meets_wr and meets_score and gate_ok:
                    self.league_mode = True

            if self.verbose:
                mode = "LEAGUE" if self.league_mode else "CURR"
                base = (
                    f"[PPO|{mode}] ep={self.episode_idx} result={result} "
                    f"score={blue_score}:{red_score} phase={phase} opp={opp_key} "
                    f"W={self.win_count} | L={self.loss_count} | D={self.draw_count}"
                )
                if self.league_mode:
                    base = f"{base} elo={self.league.learner_rating:.1f}"
                print(base)
                
                # Step 1: Print opponent distribution every N episodes
                if self._enable_opponent_tracking and (self.episode_idx % 50 == 0):
                    self._print_opponent_distribution()

            self.logger.record("curr/episode", self.episode_idx)
            self.logger.record("curr/win_rate", self.win_count / max(1, self.episode_idx))
            self.logger.record("curr/draw_rate", self.draw_count / max(1, self.episode_idx))
            self.logger.record("curr/phase_idx", float(self.curriculum.phase_idx))
            self.logger.record("curr/league_mode", float(self.league_mode))
            if self.league_mode:
                self.logger.record("league/elo", float(self.league.learner_rating))

            if (self.episode_idx % int(self.cfg.snapshot_every_episodes)) == 0:
                self._enforce_league_snapshot_limit()
                prefix = f"{self.cfg.run_tag}_league_snapshot"
                path = os.path.join(self.cfg.checkpoint_dir, f"{prefix}_ep{self.episode_idx:06d}")
                try:
                    self.model.save(path)
                except Exception as exc:
                    print(f"[WARN] snapshot save failed: {exc}")
                else:
                    self.league.add_snapshot(path + ".zip")
                    self._enforce_league_snapshot_limit()

            next_opp = self._select_next_opponent()
            # When learner is struggling (Elo below threshold), substitute OP3_HARD with OP3.
            threshold = float(getattr(self.cfg, "league_easy_scripted_elo_threshold", 1200.0))
            if (next_opp.kind == "SCRIPTED" and next_opp.key == "OP3_HARD" and
                    self.league.learner_rating < threshold):
                next_opp = OpponentSpec(kind="SCRIPTED", key="OP3", rating=self.league.get_rating("SCRIPTED:OP3"))
            # OP3_HARD exposure cap until fixed-eval vs OP3 is stable (Phase 2: avoid spicy too early)
            if (next_opp.kind == "SCRIPTED" and next_opp.key == "OP3_HARD"):
                op3_wr = self.curriculum._fixed_eval_wr.get("SCRIPTED:OP3", 0.0)
                stable_wr = float(getattr(self.cfg, "op3_stable_fixed_eval_wr", 0.55))
                max_frac = float(getattr(self.cfg, "op3_hard_max_fraction_until_stable", 0.05))
                if op3_wr < stable_wr and np.random.random() >= max_frac:
                    next_opp = OpponentSpec(kind="SCRIPTED", key="OP3", rating=self.league.get_rating("SCRIPTED:OP3"))
            
            # ✅ Fix: Use indices=[i] with env_method for opponent/phase/league_mode setting
            # This works for both DummyVecEnv and SubprocVecEnv (SB3 handles indices correctly)
            env = self.model.get_env()
            if env is not None:
                try:
                    env.env_method("set_next_opponent", next_opp.kind, next_opp.key, indices=[i])
                    env.env_method("set_phase", self.curriculum.phase, indices=[i])
                    env.env_method("set_league_mode", self.league_mode, indices=[i])
                except Exception:
                    # Fallback: if indices fails, try without (for compatibility)
                    try:
                        env.env_method("set_next_opponent", next_opp.kind, next_opp.key)
                        env.env_method("set_phase", self.curriculum.phase)
                        env.env_method("set_league_mode", self.league_mode)
                    except Exception:
                        pass

        return True


class CurriculumNoLeagueCallback(BaseCallback):
    """OP1 -> OP2 -> OP3 curriculum only; no league, no species, no snapshots."""

    def __init__(self, *, cfg: PPOConfig, curriculum: CurriculumState) -> None:
        super().__init__(verbose=1)
        self.cfg = cfg
        self.curriculum = curriculum
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
            win_by = summary.win_by

            if blue_score > red_score:
                result = "WIN"
                actual = 1.0
                self.win_count += 1
            elif blue_score < red_score:
                result = "LOSS"
                actual = 0.0
                self.loss_count += 1
            else:
                result = "DRAW"
                actual = 0.5
                self.draw_count += 1

            phase = self.curriculum.phase
            self.curriculum.phase_episode_count += 1
            self.curriculum.record_result(phase, actual)

            self.curriculum.advance_if_ready(
                learner_rating=1300.0,
                opponent_rating=1200.0,
                win_by=win_by,
            )
            phase = self.curriculum.phase

            if self.verbose:
                print(
                    f"[PPO|CURR_NO_LEAGUE] ep={self.episode_idx} result={result} "
                    f"score={blue_score}:{red_score} phase={phase} "
                    f"W={self.win_count} | L={self.loss_count} | D={self.draw_count}"
                )

            self.logger.record("curr_noleague/episode", self.episode_idx)
            self.logger.record("curr_noleague/win_rate", self.win_count / max(1, self.episode_idx))
            self.logger.record("curr_noleague/draw_rate", self.draw_count / max(1, self.episode_idx))
            self.logger.record("curr_noleague/phase_idx", float(self.curriculum.phase_idx))

            env = self.model.get_env()
            if env is not None:
                env.env_method("set_next_opponent", "SCRIPTED", phase)
                env.env_method("set_phase", phase)

        return True


class SelfPlayCallback(BaseCallback):
    """Self-play with rolling snapshot pool: counter resets to 1 when at max and old are deleted."""

    def __init__(self, *, cfg: PPOConfig, league: EloLeague) -> None:
        super().__init__(verbose=1)
        self.cfg = cfg
        self.league = league
        self.episode_idx = 0
        self.win_count = 0
        self.loss_count = 0
        self.draw_count = 0
        self._max_snapshots = max(0, int(getattr(cfg, "self_play_max_snapshots", 0)))
        # Rolling slot index 1..max_snapshots for snapshot filenames (resets when at max)
        self._snapshot_roll_index = 0
        # Total snapshots created over time (never resets, tracks cumulative count)
        self._total_snapshots_created = 0

    def _enforce_snapshot_limit(self) -> None:
        if self._max_snapshots <= 0:
            return
        while len(self.league.snapshots) > self._max_snapshots:
            oldest = self.league.snapshots.pop(0)
            try:
                if oldest and os.path.exists(oldest):
                    os.remove(oldest)
            except Exception as exc:
                print(f"[WARN] snapshot cleanup failed: {exc}")

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

            if (self.episode_idx % int(self.cfg.self_play_snapshot_every_episodes)) == 0:
                self._enforce_snapshot_limit()
                # Rolling counter 1..max_snapshots for filenames (goes back to 1 when at max)
                max_s = max(1, self._max_snapshots)
                self._snapshot_roll_index = (self._snapshot_roll_index % max_s) + 1
                slot = self._snapshot_roll_index
                prefix = f"{self.cfg.run_tag}_selfplay_snapshot"
                path = os.path.join(self.cfg.checkpoint_dir, f"{prefix}_slot{slot:03d}")
                try:
                    self.model.save(path)
                except Exception as exc:
                    print(f"[WARN] snapshot save failed: {exc}")
                else:
                    self.league.add_snapshot(path + ".zip")
                    self._enforce_snapshot_limit()
                    self._total_snapshots_created += 1  # Track cumulative total

            if self.verbose:
                print(
                    f"[PPO|SELF] ep={self.episode_idx} result={result} "
                    f"score={blue_score}:{red_score} "
                    f"snapshots={len(self.league.snapshots)} total_created={self._total_snapshots_created} "
                    f"W={self.win_count} | L={self.loss_count} | D={self.draw_count}"
                )

            self.logger.record("self/episode", self.episode_idx)
            self.logger.record("self/win_rate", self.win_count / max(1, self.episode_idx))
            self.logger.record("self/draw_rate", self.draw_count / max(1, self.episode_idx))
            self.logger.record("self/snapshots", float(len(self.league.snapshots)))
            self.logger.record("self/total_snapshots_created", float(self._total_snapshots_created))

            next_snapshot = None
            if bool(self.cfg.self_play_use_latest_snapshot):
                next_snapshot = self.league.latest_snapshot_key()
            else:
                spec = self.league.sample_snapshot()
                if spec.kind == "SNAPSHOT":
                    next_snapshot = spec.key

            if len(self.league.snapshots) == 0:
                fallback_path = os.path.join(
                    self.cfg.checkpoint_dir, f"{self.cfg.run_tag}_selfplay_init_fallback"
                )
                try:
                    self.model.save(fallback_path)
                except Exception as exc:
                    print(f"[WARN] self-play fallback save failed: {exc}")
                else:
                    self.league.add_snapshot(fallback_path + ".zip")
                    self._enforce_snapshot_limit()
                    next_snapshot = self.league.latest_snapshot_key()

            if next_snapshot:
                env = self.model.get_env()
                if env is not None:
                    env.env_method("set_next_opponent", "SNAPSHOT", next_snapshot)

        return True


class FixedOpponentCallback(BaseCallback):
    def __init__(self, *, cfg: PPOConfig) -> None:
        super().__init__(verbose=1)
        self.cfg = cfg
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
                opp = str(summary.scripted_tag or self.cfg.fixed_opponent_tag).upper()
                print(
                    f"[PPO|FIXED] ep={self.episode_idx} result={result} "
                    f"score={blue_score}:{red_score} opp=SCRIPTED:{opp} "
                    f"W={self.win_count} | L={self.loss_count} | D={self.draw_count}"
                )

            self.logger.record("fixed/episode", self.episode_idx)
            self.logger.record("fixed/win_rate", self.win_count / max(1, self.episode_idx))
            self.logger.record("fixed/draw_rate", self.draw_count / max(1, self.episode_idx))

        return True


class KLGuardrailCallback(BaseCallback):
    """Fix 4.2: Log approx_kl and auto-flag when it exceeds threshold repeatedly (over-updating)."""

    def __init__(
        self,
        *,
        threshold: float = 0.03,
        consecutive: int = 3,
        verbose: int = 1,
    ):
        super().__init__(verbose=verbose)
        self.threshold = float(threshold)
        self.consecutive = int(consecutive)
        self._spike_count = 0
        self._last_checked_update = -1

    def _on_step(self) -> bool:
        n_steps = getattr(self.model, "n_steps", 2048)
        if n_steps <= 0 or self.n_calls <= 1:
            return True
        # Sample approx_kl once per PPO update (first step of each new rollout, after previous learn)
        if (self.n_calls - 1) % n_steps != 0:
            return True
        update_id = (self.n_calls - 1) // n_steps
        if update_id <= self._last_checked_update:
            return True
        self._last_checked_update = update_id

        name_to_value = getattr(self.logger, "name_to_value", None) or {}
        approx_kl = float(name_to_value.get("train/approx_kl", 0.0))

        if approx_kl > self.threshold:
            self._spike_count += 1
            if self.verbose:
                self.logger.record("train/kl_guardrail_spike_count", self._spike_count)
            if self._spike_count >= self.consecutive:
                setattr(self.model, "_kl_guardrail_triggered", True)
                if self.verbose:
                    print(
                        f"[KLGuardrail] approx_kl exceeded {self.threshold} for {self._spike_count} consecutive updates "
                        f"(last approx_kl={approx_kl:.4f}). Set model._kl_guardrail_triggered=True; consider enabling use_stable_marl_ppo."
                    )
        else:
            self._spike_count = 0
        return True


class NoiseMetricsCSVCallback(BaseCallback):
    """Track action execution noise (flip rates, streaks) and log to CSV per episode.
    Uses per-env state so metrics are correct with vectorized envs (no mixing).
    episode_idx is a global monotonic counter (never reset).
    """

    def __init__(self, csv_path: str, eps: float, run_id: str, verbose: int = 0):
        super().__init__(verbose)
        self.csv_path = str(csv_path)
        self.eps = float(eps)
        self.run_id = str(run_id)
        self.episode_idx = 0
        self.curr_streak = None
        self.ep_steps = None
        self.flip_count = None
        self.macro_flip_count = None
        self.target_flip_count = None
        self.max_streak = None

        os.makedirs(os.path.dirname(os.path.abspath(self.csv_path)) or ".", exist_ok=True)
        self._ensure_header()

    def _ensure_header(self):
        if os.path.exists(self.csv_path) and os.path.getsize(self.csv_path) > 0:
            return
        headers = [
            "run_id", "phase", "episode_idx", "steps", "agents", "eps",
            "total_actions", "flip_count", "flip_rate",
            "macro_flip_count", "macro_flip_rate",
            "target_flip_count", "target_flip_rate",
            "max_flip_streak",
            "win", "score_for", "score_against",
            "collisions", "coverage",
            "mean_inter_robot_dist", "std_inter_robot_dist",
        ]
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(headers)

    def _on_training_start(self):
        n_envs = getattr(self.training_env, "num_envs", 1)
        n_envs = max(1, int(n_envs))
        self.curr_streak = np.zeros(n_envs, dtype=np.int32)
        self.ep_steps = np.zeros(n_envs, dtype=np.int64)
        self.flip_count = np.zeros(n_envs, dtype=np.int64)
        self.macro_flip_count = np.zeros(n_envs, dtype=np.int64)
        self.target_flip_count = np.zeros(n_envs, dtype=np.int64)
        self.max_streak = np.zeros(n_envs, dtype=np.int32)

    def _reset_env_ep(self, env_i: int) -> None:
        """Reset per-episode counters for a single env (after writing its row)."""
        if self.ep_steps is None or env_i < 0 or env_i >= len(self.ep_steps):
            return
        self.ep_steps[env_i] = 0
        self.flip_count[env_i] = 0
        self.macro_flip_count[env_i] = 0
        self.target_flip_count[env_i] = 0
        self.curr_streak[env_i] = 0
        self.max_streak[env_i] = 0

    def _on_step(self) -> bool:
        if self.ep_steps is None:
            return True
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        n_envs = len(self.ep_steps)

        for env_i, info in enumerate(infos):
            if env_i >= n_envs:
                break
            self.ep_steps[env_i] += 1

            flips = int(info.get("flip_count_step", 0))
            macro_flips = int(info.get("macro_flip_count_step", 0))
            target_flips = int(info.get("target_flip_count_step", 0))

            self.flip_count[env_i] += flips
            self.macro_flip_count[env_i] += macro_flips
            self.target_flip_count[env_i] += target_flips

            if flips > 0:
                self.curr_streak[env_i] += 1
                if self.curr_streak[env_i] > self.max_streak[env_i]:
                    self.max_streak[env_i] = int(self.curr_streak[env_i])
            else:
                self.curr_streak[env_i] = 0

            if env_i < len(dones) and bool(dones[env_i]):
                summary = parse_episode_result(info)
                if summary is None:
                    self._reset_env_ep(env_i)
                    continue

                phase = summary.phase_name
                agents = int(info.get("num_agents", 2))
                action_components = int(info.get("action_components", 2))
                steps_i = int(self.ep_steps[env_i])
                total_actions = steps_i * agents * action_components

                flip_rate = (self.flip_count[env_i] / total_actions) if total_actions > 0 else 0.0
                macro_rate = (self.macro_flip_count[env_i] / total_actions) if total_actions > 0 else 0.0
                target_rate = (self.target_flip_count[env_i] / total_actions) if total_actions > 0 else 0.0

                win = summary.success
                score_for = summary.blue_score
                score_against = summary.red_score
                collisions = summary.collisions_per_episode
                coverage = summary.zone_coverage if summary.zone_coverage is not None else float("nan")
                mean_dist = summary.mean_inter_robot_dist if summary.mean_inter_robot_dist is not None else float("nan")
                std_dist = summary.std_inter_robot_dist if summary.std_inter_robot_dist is not None else float("nan")

                row = [
                    self.run_id, phase, self.episode_idx, steps_i, agents, self.eps,
                    total_actions, int(self.flip_count[env_i]), flip_rate,
                    int(self.macro_flip_count[env_i]), macro_rate,
                    int(self.target_flip_count[env_i]), target_rate,
                    int(self.max_streak[env_i]),
                    win, score_for, score_against,
                    collisions, coverage,
                    mean_dist, std_dist,
                ]

                try:
                    with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                        csv.writer(f).writerow(row)
                except Exception as exc:
                    if self.verbose:
                        print(f"[NoiseMetrics] CSV write failed: {exc}")

                self.episode_idx += 1
                self._reset_env_ep(env_i)

        return True


class MetricsCSVCallback(BaseCallback):
    """Collect Top 5 IROS-style metrics per episode and write one CSV at end of training (publish-friendly)."""

    CSV_COLUMNS = [
        "episode_id",
        "success",
        "time_to_first_score",
        "time_to_game_over",
        "collisions_per_episode",
        "near_misses_per_episode",
        "collision_free_episode",
        "mean_inter_robot_dist",
        "std_inter_robot_dist",
        "zone_coverage",
        "phase_name",
        "opponent_kind",
        "scripted_tag",
        "blue_score",
        "red_score",
        "opponent_switch_count",  # Step 4.2: cumulative opponent switches so far
        "vec_schema_version",
    ]

    def __init__(self, *, save_path: str) -> None:
        super().__init__(verbose=0)
        self.save_path = str(save_path)
        self._rows: List[Dict[str, Any]] = []
        self._episode_id = 0
        self._opponent_switch_count = 0
        self._last_opponent_key: Optional[str] = None

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
            self._episode_id += 1
            opp_key = summary.opponent_key()
            if self._last_opponent_key is not None and opp_key != self._last_opponent_key:
                self._opponent_switch_count += 1
            self._last_opponent_key = opp_key

            row = {
                "episode_id": self._episode_id,
                "success": summary.success,
                "time_to_first_score": summary.time_to_first_score,
                "time_to_game_over": summary.time_to_game_over,
                "collisions_per_episode": summary.collisions_per_episode,
                "near_misses_per_episode": summary.near_misses_per_episode,
                "collision_free_episode": summary.collision_free_episode,
                "mean_inter_robot_dist": summary.mean_inter_robot_dist,
                "std_inter_robot_dist": summary.std_inter_robot_dist,
                "zone_coverage": summary.zone_coverage,
                "phase_name": summary.phase_name,
                "opponent_kind": summary.opponent_kind,
                "scripted_tag": summary.scripted_tag or "",
                "blue_score": summary.blue_score,
                "red_score": summary.red_score,
                "opponent_switch_count": self._opponent_switch_count,
                "vec_schema_version": summary.vec_schema_version,
            }
            self._rows.append(row)
        return True

    def _fmt(self, v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, float):
            return f"{v:.6g}"
        return str(v)

    def _on_training_end(self) -> None:
        if not self._rows:
            return
        path = self.save_path
        if not path.lower().endswith(".csv"):
            path = path + ".csv"
        try:
            os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=self.CSV_COLUMNS, extrasaction="ignore")
                w.writeheader()
                for row in self._rows:
                    out = {k: self._fmt(row.get(k)) for k in self.CSV_COLUMNS}
                    w.writerow(out)
            if self.verbose:
                print(f"[Metrics] Saved {len(self._rows)} rows to {path}")
        except Exception as exc:
            print(f"[WARN] Metrics CSV save failed: {exc}")


def train_ppo(cfg: Optional[PPOConfig] = None) -> None:
    cfg = cfg or PPOConfig()
    set_global_seed(cfg.seed, torch_seed=True, deterministic=cfg.use_deterministic)

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    mode = str(cfg.mode).upper().strip()

    # Sprint A: League with stability mix; use cfg consistently (Fix 3)
    _species_prob = float(getattr(cfg, "stability_species_prob", 0.15))
    _snapshot_prob = float(getattr(cfg, "stability_snapshot_prob", 0.20))
    _scripted_prob = max(0.0, 1.0 - _snapshot_prob - _species_prob)
    league = EloLeague(
        seed=cfg.seed,
        k_factor=32.0,
        matchmaking_tau=200.0,
        scripted_floor=0.50,
        species_prob=_species_prob,
        snapshot_prob=_snapshot_prob,
        stability_scripted_prob=_scripted_prob,
        stability_snapshot_prob=_snapshot_prob,
        stability_species_prob=_species_prob,
        use_stability_mix=True,
        min_episodes_per_opponent=3,  # Cap opponent switching frequency
        species_rusher_bias=float(getattr(cfg, "species_rusher_bias", 0.5)),
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
                fixed_eval_gate_OP1_wr=getattr(cfg, "fixed_eval_gate_OP1_wr", 0.90),
                fixed_eval_gate_OP2_wr=getattr(cfg, "fixed_eval_gate_OP2_wr", 0.75),
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

    # Same env_fns work with either vec env: both call env_fns[i]() with same rank i.
    env_fns = [
        _make_env_fn(cfg, default_opponent=default_opponent, rank=i)
        for i in range(max(1, int(cfg.n_envs)))
    ]
    try:
        venv = SubprocVecEnv(env_fns)
    except Exception:
        venv = DummyVecEnv(env_fns)
    venv = VecMonitor(venv)

    # Naval realism: stress + physics-by-phase (OP1 no physics, OP2 relaxed, OP3 full)
    try:
        venv.env_method("set_stress_schedule", STRESS_BY_PHASE)
    except Exception:
        pass
    try:
        venv.env_method("set_phase", phase_name)
    except Exception:
        pass

    policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))
    # Fix 0: Explicit tokenized extractor only from config (not from agent_mask in obs — env always has it)
    use_tokenized = bool(getattr(cfg, "use_tokenized_obs", False)) or (int(getattr(cfg, "max_blue_agents", 2)) > 2)
    if use_tokenized:
        policy_kwargs["features_extractor_class"] = TokenizedCombinedExtractor
        policy_kwargs["features_extractor_kwargs"] = dict(cnn_output_dim=256, normalized_image=True)

    # Step 4: Stable MARL PPO (Fix 4.1) or reduced aggressiveness
    learning_rate = float(cfg.learning_rate)
    ent_coef = float(cfg.ent_coef)
    clip_range = float(cfg.clip_range)
    n_epochs = int(cfg.n_epochs)
    batch_size = int(cfg.batch_size)

    if getattr(cfg, "use_stable_marl_ppo", False):
        learning_rate = 1.5e-4
        ent_coef = 0.005
        clip_range = 0.12
        n_epochs = 4
        batch_size = 1024
        print("[PPO] Using stable MARL PPO: lr=1.5e-4, n_epochs=4, clip_range=0.12, ent_coef=0.005, batch_size=1024")
    elif getattr(cfg, "use_reduced_aggressiveness", False):
        learning_rate = learning_rate * 0.67
        ent_coef = ent_coef * 0.5
        clip_range = clip_range * 0.75
        print(f"[PPO] Using reduced aggressiveness: LR={learning_rate:.2e}, ent_coef={ent_coef:.3f}, clip_range={clip_range:.2f}")
    
    model = PPO(
        policy=MaskedMultiInputPolicy,
        env=venv,
        learning_rate=learning_rate,
        n_steps=int(cfg.n_steps),
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=float(cfg.gamma),
        gae_lambda=float(cfg.gae_lambda),
        clip_range=clip_range,
        ent_coef=ent_coef,
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

    if cfg.enable_tensorboard:
        model.set_logger(configure(os.path.join(cfg.checkpoint_dir, "tb", cfg.run_tag), ["tensorboard"]))
    else:
        model.set_logger(configure(None, []))

    if mode == TrainMode.SELF_PLAY.value:
        init_path = os.path.join(cfg.checkpoint_dir, f"{cfg.run_tag}_selfplay_init")
        try:
            model.save(init_path)
        except Exception as exc:
            print(f"[WARN] self-play init snapshot save failed: {exc}")
        else:
            league.add_snapshot(init_path + ".zip")
            init_key = league.latest_snapshot_key()
            max_snaps = max(0, int(getattr(cfg, "self_play_max_snapshots", 0)))
            if max_snaps > 0:
                while len(league.snapshots) > max_snaps:
                    oldest = league.snapshots.pop(0)
                    try:
                        if oldest and os.path.exists(oldest):
                            os.remove(oldest)
                    except Exception as exc:
                        print(f"[WARN] snapshot cleanup failed: {exc}")
            if init_key:
                venv.env_method("set_next_opponent", "SNAPSHOT", init_key)
                venv.reset()

    callbacks = []
    if mode == TrainMode.CURRICULUM_LEAGUE.value:
        callbacks.append(LeagueCallback(cfg=cfg, league=league, curriculum=curriculum, controller=controller))
    elif mode == TrainMode.CURRICULUM_NO_LEAGUE.value and curriculum is not None:
        callbacks.append(CurriculumNoLeagueCallback(cfg=cfg, curriculum=curriculum))
    elif mode == TrainMode.SELF_PLAY.value:
        callbacks.append(SelfPlayCallback(cfg=cfg, league=league))
    elif mode == TrainMode.FIXED_OPPONENT.value:
        callbacks.append(FixedOpponentCallback(cfg=cfg))

    # Phase 2: fixed-eval suite (updates curriculum._fixed_eval_wr for gating when curriculum is set)
    if cfg.enable_fixed_eval:
        from rl.fixed_eval import FixedEvalCallback
        callbacks.append(
            FixedEvalCallback(
                cfg=cfg,
                eval_every_episodes=int(cfg.fixed_eval_every_episodes),
                episodes_per_opponent=int(cfg.fixed_eval_episodes),
                curriculum=curriculum if curriculum is not None else None,
            )
        )

    # Top 5 IROS-style metrics: CSV at end of training (simple, publish-friendly)
    metrics_csv_path = os.path.join(cfg.checkpoint_dir, f"{cfg.run_tag}_metrics")
    callbacks.append(MetricsCSVCallback(save_path=metrics_csv_path))

    # Fix 4.2: KL guardrail – log approx_kl and set model._kl_guardrail_triggered if spikes repeatedly
    if getattr(cfg, "approx_kl_threshold", 0) > 0 and getattr(cfg, "kl_guardrail_consecutive", 0) > 0:
        callbacks.append(
            KLGuardrailCallback(
                threshold=float(cfg.approx_kl_threshold),
                consecutive=int(cfg.kl_guardrail_consecutive),
                verbose=1,
            )
        )

    # Action execution noise metrics (if enabled)
    if getattr(cfg, "action_flip_prob", 0.0) > 0.0:
        noise_csv_path = os.path.join(cfg.checkpoint_dir, f"{cfg.run_tag}_noise_metrics")
        callbacks.append(
            NoiseMetricsCSVCallback(
                csv_path=noise_csv_path,
                eps=float(cfg.action_flip_prob),
                run_id=str(cfg.run_tag),
                verbose=0,
            )
        )

    if cfg.enable_checkpoints:
        callbacks.append(
            CheckpointCallback(
                save_freq=int(cfg.save_every_steps),
                save_path=cfg.checkpoint_dir,
                name_prefix=f"ckpt_{cfg.run_tag}",
            )
        )

    if cfg.enable_eval:
        eval_env = DummyVecEnv([_make_env_fn(cfg, default_opponent=("SCRIPTED", "OP3"), rank=0)])
        eval_env = VecMonitor(eval_env)
        callbacks.append(
            EvalCallback(
                eval_env,
                n_eval_episodes=int(cfg.eval_episodes),
                eval_freq=int(cfg.eval_every_steps),
                deterministic=True,
                best_model_save_path=cfg.checkpoint_dir,
            )
        )

    callbacks = CallbackList(callbacks)

    model.learn(total_timesteps=int(cfg.total_timesteps), callback=callbacks)

    final_path = os.path.join(cfg.checkpoint_dir, f"final_{cfg.run_tag}")
    model.save(final_path)
    print(f"[PPO] Training complete. Final model saved to: {final_path}.zip")


def run_verify_4v4(num_episodes: int = 10) -> None:
    """Sprint A verification: run N random-action episodes at 4v4, print shapes on reset."""
    cfg = PPOConfig(
        max_blue_agents=4,
        mode=TrainMode.FIXED_OPPONENT.value,
        fixed_opponent_tag="OP1",
        print_reset_shapes=True,
    )
    set_global_seed(cfg.seed)
    env = CTFGameFieldSB3Env(
        make_game_field_fn=lambda: make_game_field(map_name=MAP_NAME or None, map_path=MAP_PATH or None),
        max_decision_steps=cfg.max_decision_steps,
        enforce_masks=True,
        seed=cfg.seed,
        include_mask_in_obs=True,
        default_opponent_kind="SCRIPTED",
        default_opponent_key="OP1",
        ppo_gamma=cfg.gamma,
        max_blue_agents=4,
        print_reset_shapes=True,
    )
    for ep in range(num_episodes):
        obs, _ = env.reset(options={"n_agents": 4})
        done = False
        steps = 0
        while not done and steps < env.max_decision_steps * 2:
            action = env.action_space.sample()
            obs, reward, term, trunc, info = env.step(action)
            done = term or trunc
            steps += 1
        print(f"[Verify-4v4] episode {ep + 1}/{num_episodes} steps={steps} done={done}")
    env.close()
    print(f"[Verify-4v4] Done. {num_episodes} random-action 4v4 episodes completed.")


def run_test_vec_schema() -> None:
    """
    Step 2.2 verification: GameField.build_continuous_features(agent) takes an Agent instance
    (not agent_id) and returns float32 shape (12,), finite, in schema bounds.
    """
    from game_field import GameField
    from config import MAP_NAME, MAP_PATH
    V = 12
    assert hasattr(GameField, "VEC_SCHEMA_VERSION"), "GameField missing VEC_SCHEMA_VERSION"
    assert getattr(GameField, "VEC_SCHEMA_VERSION", 0) >= 1, "VEC_SCHEMA_VERSION must be >= 1"

    gf = make_game_field(map_name=MAP_NAME or None, map_path=MAP_PATH or None)
    gf.reset_default()
    blue_agents = getattr(gf, "blue_agents", []) or []
    assert blue_agents, "No blue agents after reset_default"

    for agent in blue_agents:
        if agent is None:
            continue
        # Signature: build_continuous_features(self, agent: Agent) — agent is an object, not an id
        vec = gf.build_continuous_features(agent)
        vec = np.asarray(vec, dtype=np.float32)
        assert vec.dtype == np.float32, f"dtype {vec.dtype}, expected float32"
        assert vec.shape == (V,), f"shape {vec.shape}, expected ({V},)"
        assert np.all(np.isfinite(vec)), f"non-finite values: {vec}"
        # Schema VEC_SCHEMA_VERSION==1: norm coords [0,1], heading [-1,1], speed [0,1], deltas clipped [-1,1], etc.
        assert np.all(vec >= -1.1) and np.all(vec <= 1.1), (
            f"vec values outside expected schema bounds [-1.1, 1.1]: min={vec.min():.4f} max={vec.max():.4f}"
        )
    print("[test-vec-schema] GameField.build_continuous_features(agent): dtype=float32, shape=(12,), finite, in bounds. OK.")


if __name__ == "__main__":
    import sys
    if "--verify-4v4" in sys.argv:
        run_verify_4v4(num_episodes=10)
    elif "--test-vec-schema" in sys.argv:
        run_test_vec_schema()
    else:
        train_ppo()
