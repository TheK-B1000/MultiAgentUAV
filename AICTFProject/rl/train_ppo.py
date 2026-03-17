from __future__ import annotations

import csv
import os
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

# Add parent directory to path so imports work regardless of where script is run from
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_SCRIPT_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

import numpy as np

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, NatureCNN
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecMonitor

from game_field_gpu import GPUCTFVecEnv, GPUFieldConfig
from opponent_params import sample_batched_opponent_params
from rl.curriculum import (
    CurriculumConfig,
    CurriculumController,
    CurriculumControllerConfig,
    CurriculumState,
    STRESS_BY_PHASE,
)
from rl.league import EloLeague, OpponentSpec
from rl.episode_result import parse_episode_result, EpisodeSummary, path_to_snapshot_key


def set_global_seed(seed: int, torch_seed: bool = True, deterministic: bool = False) -> None:
    """
    Set global RNG seeds for reproducibility.

    This replaces the original implementation from rl.common so that
    train_ppo.py can run without that module.
    """
    import random

    random.seed(seed)
    np.random.seed(seed)

    if torch_seed:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


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

        self._M = M
        self._V = V

        # If spatial grid is at least 3x3, use NatureCNN; otherwise fall back to simple flatten
        if H >= 3 and W >= 3:
            single_grid = spaces.Box(
                low=float(grid_space.low.min()) if hasattr(grid_space, "low") else 0.0,
                high=float(grid_space.high.max()) if hasattr(grid_space, "high") else 1.0,
                shape=(C, H, W),
                dtype=grid_space.dtype,
            )
            self.cnn = NatureCNN(single_grid, features_dim=cnn_output_dim, normalized_image=normalized_image)
            self._use_cnn = True
            cnn_latent_dim = cnn_output_dim
        else:
            # Grid is effectively a vector (e.g. 1x1); treat channels as features per token
            self.cnn = None
            self._use_cnn = False
            cnn_latent_dim = C

        self.vec_dim = V
        features_dim = M * cnn_latent_dim + M * V
        context_space = spaces_dict.get("context")
        self._context_dim = 0
        if context_space is not None and hasattr(context_space, "shape"):
            self._context_dim = int(np.prod(context_space.shape))
            features_dim += self._context_dim
        super().__init__(observation_space, features_dim)

    def forward(self, observations):
        grid = observations["grid"]
        vec = observations["vec"]
        B, M = grid.shape[0], self._M

        if self._use_cnn:
            grid_flat = grid.reshape(B * M, *grid.shape[2:])
            cnn_out = self.cnn(grid_flat)
            D = cnn_out.shape[1]
            cnn_out = cnn_out.reshape(B, M, D)
        else:
            # No CNN: just flatten channel dimension per token
            D = grid.shape[2]  # channels C
            cnn_out = grid.reshape(B, M, D)

        agent_mask = observations.get("agent_mask", None)
        if agent_mask is not None:
            if agent_mask.dim() == 1:
                agent_mask = agent_mask.unsqueeze(0)
            agent_mask = agent_mask.float().unsqueeze(-1)
            cnn_out = cnn_out * agent_mask
            vec = vec * agent_mask

        cnn_out = cnn_out.reshape(B, M * D)
        vec_flat = vec.reshape(B, M * self._V)
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
    """Training modes. Paper = curriculum (OP1→OP2→OP3) with no league/snapshots."""
    CURRICULUM_LEAGUE = "CURRICULUM_LEAGUE"   # League: curriculum then league (OP3 + snapshots)
    CURRICULUM_NO_LEAGUE = "CURRICULUM_NO_LEAGUE"  # Paper: curriculum only, no league
    FIXED_OPPONENT = "FIXED_OPPONENT"         # Fixed: 100% single scripted opponent (e.g. OP3)
    SELF_PLAY = "SELF_PLAY"                   # Self-play: vs past snapshots of self


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
    ent_coef: float = 0.01
    learning_rate: float = 3e-4
    max_grad_norm: float = 0.5
    # Default to GPU when available, otherwise CPU.
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint_dir: str = "checkpoints_sb3"
    # Distinct tag per team size so runs don't overwrite each other (default 2v2)
    run_tag: str = "ppo_league_curriculum_2v2"
    save_every_steps: int = 50_000
    eval_every_steps: int = 25_000
    eval_episodes: int = 6
    snapshot_every_episodes: int = 200
    league_max_snapshots: int = 5
    # If still in OP3 and not in league after this many timesteps, enter league anyway (80% normal promotion, 250k fallback).
    league_fallback_timesteps: int = 250_000
    # Disable TensorBoard by default to avoid dependency/version issues.
    # Re-enable (True) if you install a compatible tensorboard+protobuf pair.
    enable_tensorboard: bool = False
    enable_checkpoints: bool = False
    enable_eval: bool = False

    # Quiet training: no per-episode prints (faster, especially for 8v8). Set True to see each episode.
    verbose_training: bool = False
    # Show TQDM progress bar with ETA (set False to disable).
    enable_progress_bar: bool = True

    max_decision_steps: int = 400

    mode: str = TrainMode.CURRICULUM_LEAGUE.value
    fixed_opponent_tag: str = "OP3"
    self_play_use_latest_snapshot: bool = False
    self_play_latest_snapshot_prob: float = 0.35
    self_play_snapshot_every_episodes: int = 200
    self_play_max_snapshots: int = 5
    league_anchor_op3_prob: float = 0.60
    league_species_prob: float = 0.20
    league_snapshot_prob: float = 0.20

    action_flip_prob: float = 0.0
    use_deterministic: bool = False

    # Team size: 2=2v2 (default), 4=4v4, 8=8v8 (use --agents 4 for 4v4)
    max_blue_agents: int = 2
    print_reset_shapes: bool = False
    reward_mode: str = "TEAM_SUM"
    use_obs_builder: bool = True
    include_opponent_context: bool = False
    obs_debug_validate_locality: bool = False
    normalize_vec: bool = False

    enable_opponent_tracking: bool = True
    opponent_tracking_window: int = 100
    enable_fixed_eval: bool = False
    stability_species_prob: float = 0.15
    stability_snapshot_prob: float = 0.20
    species_rusher_bias: float = 0.5
    match_op3_exposure: bool = False
    fixed_eval_every_episodes: int = 500
    fixed_eval_episodes: int = 10
    enable_mirror_eval: bool = False
    mirror_eval_every_episodes: int = 500
    mirror_eval_episodes: int = 10
    enable_league_eval: bool = False
    league_eval_every_episodes: int = 500
    league_eval_episodes: int = 6
    use_reduced_aggressiveness: bool = False
    use_stable_marl_ppo: bool = True
    target_kl: Optional[float] = 0.02
    approx_kl_threshold: float = 0.05
    kl_guardrail_consecutive: int = 3
    # Sanity check: set True and run a few steps to verify approx_kl ~ 0 (if huge, logprob/action plumbing is broken)
    test_kl_zero_lr: bool = False
    # For now, keep standard (non-tokenized) CNN obs for 4v4 to avoid tiny 1x1 grids
    use_tokenized_obs: bool = False
    gpu_native_env: bool = True  # All training uses game_field_gpu


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

        expected = sum(dims)
        if mask.shape[1] < expected:
            pad = torch.ones((mask.shape[0], expected - mask.shape[1]), device=mask.device)
            mask = torch.cat([mask, pad], dim=1)

        full_mask = []
        offset = 0
        for dim in dims:
            d = int(dim)
            sz = min(d, mask.shape[1] - offset)
            if sz > 0:
                chunk = mask[:, offset : offset + sz]
                offset += sz
                if chunk.shape[1] < d:
                    chunk = torch.cat([chunk, torch.ones((mask.shape[0], d - chunk.shape[1]), device=mask.device)], dim=1)
                full_mask.append(chunk)
            else:
                full_mask.append(torch.ones((mask.shape[0], d), device=mask.device))

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
        # Same distribution for get_actions and log_prob (no clip after); SB3 stores as old_log_prob for PPO ratio.
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

    def evaluate_actions(self, obs: Dict[str, torch.Tensor], actions: torch.Tensor):
        # PPO training must use the same masked distribution as rollout collection,
        # otherwise old_log_prob and new log_prob are computed from different policies.
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        logits = self.action_net(latent_pi)
        if isinstance(obs, dict) and "mask" in obs:
            logits = self._apply_action_mask(logits, obs["mask"])
        distribution = self.action_dist.proba_distribution(action_logits=logits)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        entropy = distribution.entropy()
        return values, log_prob, entropy


def _tqdm_available() -> bool:
    try:
        import tqdm  # noqa: F401
        return True
    except ImportError:
        return False


def _log_line(message: str) -> None:
    """
    Write a message without losing it behind an active tqdm/rich progress bar.
    """
    text = str(message)
    try:
        from tqdm import tqdm
        tqdm.write(text)
    except Exception:
        print(text, flush=True)


class ProgressLogCallback(BaseCallback):
    """Progress bar with ETA via tqdm; falls back to print every N steps if tqdm missing."""

    def __init__(self, total_timesteps: int, interval: int = 50_000, use_tqdm: bool = True):
        super().__init__(verbose=0)
        self._total = int(total_timesteps)
        self._interval = max(1, int(interval))
        self._last_milestone = 0
        self._last_n = 0
        self._pbar = None
        self._use_tqdm = bool(use_tqdm) and _tqdm_available()

    def _init_callback(self) -> None:
        if self._use_tqdm and self._total > 0:
            try:
                from tqdm import tqdm
                self._pbar = tqdm(
                    total=self._total,
                    unit=" step",
                    unit_scale=True,
                    desc="PPO",
                    dynamic_ncols=True,
                    miniters=max(1, self._total // 500),
                )
            except Exception:
                self._pbar = None
                self._use_tqdm = False

    def _on_step(self) -> bool:
        if self._total <= 0:
            return True
        if self._pbar is not None:
            n = min(self.num_timesteps, self._total)
            delta = n - self._last_n
            self._last_n = n
            if delta > 0:
                self._pbar.update(delta)
            if n >= self._total:
                self._pbar.close()
                self._pbar = None
            return True
        milestone = self.num_timesteps // self._interval
        if milestone > self._last_milestone:
            self._last_milestone = milestone
            pct = 100.0 * self.num_timesteps / self._total
            _log_line(f"[PPO] Steps {self.num_timesteps:,}/{self._total:,} ({pct:.1f}%)")
        return True

    def _on_training_end(self) -> None:
        if self._pbar is not None:
            try:
                self._pbar.close()
            except Exception:
                pass
            self._pbar = None


class LeagueCallback(BaseCallback):
    def __init__(
        self,
        *,
        cfg: PPOConfig,
        league: EloLeague,
        curriculum: CurriculumState,
        controller: CurriculumController,
    ) -> None:
        _v = 1 if getattr(cfg, "verbose_training", False) else 0
        super().__init__(verbose=_v)
        self.cfg = cfg
        self.league = league
        self.curriculum = curriculum
        self.controller = controller
        self.episode_idx = 0
        self.league_mode = False
        self.win_count = 0
        self.loss_count = 0
        self.draw_count = 0
        self._league_max_snapshots = max(0, int(getattr(cfg, "league_max_snapshots", 5)))
        
        self._opponent_stats: Dict[str, Dict[str, int]] = {}
        self._opponent_history: List[Tuple[str, str]] = []
        self._enable_opponent_tracking = getattr(cfg, "enable_opponent_tracking", True)
        self._opponent_window = getattr(cfg, "opponent_tracking_window", 100)
        self._pending_updates: List[Dict[str, Any]] = []
        self.phase_win_count = 0
        self.phase_loss_count = 0
        self.phase_draw_count = 0

    @staticmethod
    def _curriculum_window_value(result: str) -> float:
        """Algorithm 1 uses binary curriculum windows: win=1, non-win=0."""
        return 1.0 if str(result).upper() == "WIN" else 0.0

    def _enforce_league_snapshot_limit(self) -> None:
        """Delete oldest league snapshots when over cap to save disk space."""
        if self._league_max_snapshots <= 0:
            return
        # Resolve checkpoint_dir once so we can resolve relative snapshot paths
        ckpt_abs = os.path.abspath(self.cfg.checkpoint_dir)
        while len(self.league.snapshots) > self._league_max_snapshots:
            oldest = self.league.snapshots.pop(0)
            try:
                if not oldest:
                    continue
                # Try stored path (may be relative) and absolute path so cleanup works regardless of cwd
                for p in (oldest, os.path.abspath(oldest), os.path.join(ckpt_abs, os.path.basename(oldest))):
                    if p and os.path.isfile(p):
                        os.remove(p)
                        if self.verbose:
                            _log_line(f"[League] deleted old snapshot: {os.path.basename(p)}")
                        break
            except Exception as exc:
                _log_line(f"[WARN] league snapshot cleanup failed: {exc}")

    def _get_op3_win_rate(self) -> float:
        """Get win rate against OP3 over the tracking window."""
        op3_key = "SCRIPTED:OP3"
        stats = self._opponent_stats.get(op3_key, {})
        wins = stats.get("wins", 0)
        losses = stats.get("losses", 0)
        draws = stats.get("draws", 0)
        total = wins + losses + draws
        if total < 10:  # Need at least 10 games for reliable estimate
            return 0.0
        return wins / total if total > 0 else 0.0

    def _meets_op3_gate_for_league(self) -> bool:
        """True if wins vs SCRIPTED:OP3 are sufficient to allow switching to league (before elo)."""
        min_wr = float(getattr(self.curriculum.config, "min_winrate_vs_op3", 0.0) or 0.0)
        min_games = int(getattr(self.curriculum.config, "min_games_vs_op3", 0) or 0)
        if min_games <= 0 or min_wr <= 0.0:
            return True
        op3_key = "SCRIPTED:OP3"
        stats = self._opponent_stats.get(op3_key, {})
        wins = stats.get("wins", 0)
        total = stats.get("wins", 0) + stats.get("losses", 0) + stats.get("draws", 0)
        if total < min_games:
            return False
        return (wins / total) >= min_wr
    
    def _select_next_opponent(self) -> OpponentSpec:
        if not self.league_mode:
            phase = self.curriculum.phase
            # OP3 phase: only OP3 until the curriculum gate switches into league mode.
            if phase == "OP3":
                return OpponentSpec(kind="SCRIPTED", key="OP3", rating=self.league.get_rating("SCRIPTED:OP3"))
            return OpponentSpec(
                kind="SCRIPTED",
                key=phase,
                rating=self.league.get_rating(f"SCRIPTED:{phase}"),
            )
        # League mode: after the OP3 gate, use only the league mix (OP3/species/snapshots).
        return self.league.sample_league(phase="OP3", enable_snapshots=True)
    
    def _update_opponent_stats(self, opp_key: str, result: str):
        """Track opponent distribution and rolling window stats."""
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
        
        self._opponent_history.append((opp_key, result))
        if len(self._opponent_history) > self._opponent_window:
            old_opp_key, old_result = self._opponent_history.pop(0)
            if old_opp_key in self._opponent_stats:
                self._opponent_stats[old_opp_key]["total"] = max(0, self._opponent_stats[old_opp_key]["total"] - 1)
                if old_result == "WIN":
                    self._opponent_stats[old_opp_key]["wins"] = max(0, self._opponent_stats[old_opp_key]["wins"] - 1)
                elif old_result == "LOSS":
                    self._opponent_stats[old_opp_key]["losses"] = max(0, self._opponent_stats[old_opp_key]["losses"] - 1)
                else:
                    self._opponent_stats[old_opp_key]["draws"] = max(0, self._opponent_stats[old_opp_key]["draws"] - 1)
    
    def _print_opponent_distribution(self):
        """Print opponent distribution over last N episodes."""
        if not self._enable_opponent_tracking or not self._opponent_stats:
            return
        recent_opps: Dict[str, int] = {}
        for opp_key, _ in self._opponent_history[-self._opponent_window:]:
            recent_opps[opp_key] = recent_opps.get(opp_key, 0) + 1
        
        if not recent_opps:
            return
        
        parts = []
        for opp_key, count in sorted(recent_opps.items(), key=lambda x: -x[1]):
            stats = self._opponent_stats.get(opp_key, {})
            wins = stats.get("wins", 0)
            losses = stats.get("losses", 0)
            draws = stats.get("draws", 0)
            total = stats.get("total", 0)
            wr = (wins / max(1, total)) * 100 if total > 0 else 0.0
            parts.append(f"{opp_key}:{count}({wins}W/{losses}L/{draws}D,{wr:.0f}%WR)")
        _log_line(f"[OpponentDist|last_{self._opponent_window}] " + " | ".join(parts))

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
            if self.league_mode:
                self.league.update_elo(opp_key, actual)
            self.controller.record_result(opp_key, actual)
            self._update_opponent_stats(opp_key, result)

            phase = self.curriculum.phase
            self.curriculum.phase_episode_count += 1
            self.curriculum.record_result(phase, self._curriculum_window_value(result))
            if result == "WIN":
                self.phase_win_count += 1
            elif result == "LOSS":
                self.phase_loss_count += 1
            else:
                self.phase_draw_count += 1

            is_scripted = opp_key.startswith("SCRIPTED:")
            if is_scripted:
                opp_rating = self.league.get_rating(opp_key)
                # Only check Elo once in league mode; during curriculum phase skip Elo entirely
                advanced = self.curriculum.advance_if_ready(
                    learner_rating=self.league.learner_rating,
                    opponent_rating=opp_rating,
                    win_by=win_by,
                    skip_elo_check=not self.league_mode,
                )
                if advanced:
                    old_phase = phase
                    phase_tot = self.phase_win_count + self.phase_loss_count + self.phase_draw_count
                    phase_wr = (100.0 * self.phase_win_count / phase_tot) if phase_tot > 0 else 0.0
                    _log_line(f"[PPO] Phase {old_phase} complete: W={self.phase_win_count} L={self.phase_loss_count} D={self.phase_draw_count} WR={phase_wr:.1f}%")
                    self.phase_win_count = 0
                    self.phase_loss_count = 0
                    self.phase_draw_count = 0
                    phase = self.curriculum.phase
                # Debug: log why we're not advancing when still in OP1 (every 100 eps)
                elif (
                    self.verbose
                    and phase == "OP1"
                    and self.curriculum.phase_episode_count >= 200
                    and self.episode_idx % 100 == 0
                ):
                    min_eps = self.curriculum.config.min_episodes.get("OP1", 0)
                    min_wr = self.curriculum.config.min_winrate.get("OP1", 0.0)
                    wr = self.curriculum.phase_winrate("OP1")
                    meets_eps = self.curriculum.phase_episode_count >= min_eps
                    meets_wr = wr >= min_wr
                    blocker = "min_episodes" if not meets_eps else "min_winrate" if not meets_wr else "?"
                    print(
                        f"[CURR-DEBUG] OP1 waiting: phase_eps={self.curriculum.phase_episode_count}/{min_eps} ({'ok' if meets_eps else 'need more'}), "
                        f"wr={wr:.3f}>={min_wr} ({'ok' if meets_wr else 'need higher'}) → advance when both ok (blocked by: {blocker})"
                    )

            if phase == "OP3":
                min_eps = int(self.curriculum.config.min_episodes.get("OP3", 0))
                min_wr = float(self.curriculum.config.min_winrate.get("OP3", 0.80))
                meets_eps = self.curriculum.phase_episode_count >= min_eps
                meets_wr = self.curriculum.phase_winrate("OP3") >= min_wr
                meets_op3_gate = self._meets_op3_gate_for_league()
                fallback_steps = max(0, int(getattr(self.cfg, "league_fallback_timesteps", 250_000)))
                use_fallback = fallback_steps > 0 and self.num_timesteps >= fallback_steps
                if self.curriculum.config.switch_to_league_after_op3_win:
                    if meets_eps and meets_wr and meets_op3_gate:
                        self.league_mode = True
                        if self.verbose and getattr(self.curriculum.config, "min_games_vs_op3", 0) > 0:
                            op3_stats = self._opponent_stats.get("SCRIPTED:OP3", {})
                            tw = op3_stats.get("wins", 0) + op3_stats.get("losses", 0) + op3_stats.get("draws", 0)
                            _log_line(f"[League] OP3 gate passed: {op3_stats.get('wins', 0)}W vs OP3 in last {tw} OP3 games → switching to league/elo")
                    elif use_fallback and not self.league_mode:
                        self.league_mode = True
                        _log_line(f"[League] 250k-step fallback: entering league at {self.num_timesteps:,} steps (80% normal promotion not met)")
                elif self.verbose and phase == "OP3" and not self.league_mode and self.episode_idx % 100 == 0:
                    min_g = getattr(self.curriculum.config, "min_games_vs_op3", 0)
                    min_wr_op3 = getattr(self.curriculum.config, "min_winrate_vs_op3", 0.0)
                    if min_g > 0 and min_wr_op3 > 0:
                        op3_stats = self._opponent_stats.get("SCRIPTED:OP3", {})
                        w, l, d = op3_stats.get("wins", 0), op3_stats.get("losses", 0), op3_stats.get("draws", 0)
                        tot = w + l + d
                        wr_op3 = (w / tot) if tot > 0 else 0.0
                        _log_line(f"[CURR-DEBUG] OP3→League gate: vs OP3 {w}W/{tot} games ({wr_op3:.1%}), need >={min_wr_op3:.0%} in >={min_g} games (phase_eps={meets_eps}, phase_wr={meets_wr})")

            if self.verbose:
                mode = "LEAGUE" if self.league_mode else "CURR"
                base = (
                    f"[PPO|{mode}] ep={self.episode_idx} result={result} "
                    f"score={blue_score}:{red_score} phase={phase} opp={opp_key} "
                    f"W={self.win_count} | L={self.loss_count} | D={self.draw_count}"
                )
                if self.league_mode:
                    base = f"{base} elo={self.league.learner_rating:.1f}"
                _log_line(base)
                
                if self._enable_opponent_tracking and (self.episode_idx % 50 == 0):
                    self._print_opponent_distribution()

            # Summary every 1000 episodes (always, not gated by verbose)
            if self.episode_idx > 0 and self.episode_idx % 1000 == 0:
                wr = (self.win_count / self.episode_idx) * 100
                mode = "LEAGUE" if self.league_mode else "CURR"
                opp_summary = "mixed" if self.league_mode else phase
                phase_tot = self.phase_win_count + self.phase_loss_count + self.phase_draw_count
                phase_wr = (100.0 * self.phase_win_count / phase_tot) if phase_tot > 0 else 0.0
                _log_line(f"[PPO] ep={self.episode_idx} mode={mode} phase={phase} opp={opp_summary} | total W={self.win_count} L={self.loss_count} D={self.draw_count} WR={wr:.1f}%")

            self.logger.record("curr/episode", self.episode_idx)
            self.logger.record("curr/win_rate", self.win_count / max(1, self.episode_idx))
            self.logger.record("curr/draw_rate", self.draw_count / max(1, self.episode_idx))
            self.logger.record("curr/phase_idx", float(self.curriculum.phase_idx))
            self.logger.record("curr/league_mode", float(self.league_mode))
            if self.league_mode:
                self.logger.record("league/elo", float(self.league.learner_rating))

            if self.league_mode and (self.episode_idx % int(self.cfg.snapshot_every_episodes)) == 0:
                self._enforce_league_snapshot_limit()
                prefix = f"{self.cfg.run_tag}_league_snapshot"
                path = os.path.join(self.cfg.checkpoint_dir, f"{prefix}_ep{self.episode_idx:06d}")
                try:
                    self.model.save(path)
                except Exception as exc:
                    _log_line(f"[WARN] snapshot save failed: {exc}")
                else:
                    # Store absolute path so cleanup can find the file regardless of cwd
                    path_zip = os.path.abspath(path + ".zip")
                    self.league.add_snapshot(path_zip)
                    self._enforce_league_snapshot_limit()

            next_opp = self._select_next_opponent()

            env = self.model.get_env()
            if env is not None:
                try:
                    env.env_method("set_next_opponent", next_opp.kind, next_opp.key, indices=[i])
                    env.env_method("set_phase", self.curriculum.phase, indices=[i])
                    env.env_method("set_league_mode", self.league_mode, indices=[i])
                except Exception:
                    try:
                        env.env_method("set_next_opponent", next_opp.kind, next_opp.key)
                        env.env_method("set_phase", self.curriculum.phase)
                        env.env_method("set_league_mode", self.league_mode)
                    except Exception:
                        pass

        return True


class LeagueEvalCallback(BaseCallback):
    """Deterministic league evaluation against anchors, species, snapshots, and mirror self-play."""

    def __init__(self, *, cfg: PPOConfig, league: EloLeague) -> None:
        super().__init__(verbose=0)
        self.cfg = cfg
        self.league = league
        self.episode_idx = 0
        self._eval_env: Optional[GPUCTFVecEnv] = None
        self._mirror_eval_snapshot_path = os.path.join(
            self.cfg.checkpoint_dir,
            f"{self.cfg.run_tag}_league_mirror_eval_current.zip",
        )

    def _init_callback(self) -> None:
        eval_cfg = GPUFieldConfig(
            n_envs=1,
            max_blue_agents=max(1, int(getattr(self.cfg, "max_blue_agents", 2))),
            max_red_agents=max(1, int(getattr(self.cfg, "max_blue_agents", 2))),
            max_decision_steps=max(1, int(self.cfg.max_decision_steps)),
            aquaticus_profile=True,
            rules_profile="OURS",
            device=str(self.cfg.device),
            seed=int(self.cfg.seed) + 2000,
        )
        self._eval_env = GPUCTFVecEnv(eval_cfg)
        try:
            self._eval_env.env_method("set_stress_schedule", STRESS_BY_PHASE)
        except Exception:
            pass

    def _run_eval_matchup(self, kind: str, key: str, *, episodes: int, phase: str) -> Optional[Tuple[int, int, int]]:
        env = self._eval_env
        if env is None:
            return None
        try:
            env.env_method("set_next_opponent", kind, key)
            env.env_method("set_phase", phase)
            env.env_method("set_league_mode", True)
        except Exception:
            return None

        wins = 0
        losses = 0
        draws = 0
        obs = env.reset()
        completed = 0
        while completed < episodes:
            actions, _ = self.model.predict(obs, deterministic=True)
            env.step_async(actions)
            obs, _, dones, infos = env.step_wait()
            if not bool(dones[0]):
                continue
            summary = parse_episode_result(infos[0])
            if summary is None:
                continue
            if summary.blue_score > summary.red_score:
                wins += 1
            elif summary.blue_score < summary.red_score:
                losses += 1
            else:
                draws += 1
            completed += 1
        return wins, losses, draws

    def _run_side_swapped_mirror_eval(self, episodes: int) -> Optional[Dict[str, Tuple[int, int, int]]]:
        env = self._eval_env
        if env is None:
            return None
        mirror_path_no_ext = os.path.splitext(self._mirror_eval_snapshot_path)[0]
        try:
            self.model.save(mirror_path_no_ext)
            env.env_method("set_phase", "OP3")
            env.env_method("set_league_mode", True)
        except Exception:
            return None
        snapshot_model = env.core._load_snapshot_policy(self._mirror_eval_snapshot_path)
        if snapshot_model is None:
            return None

        def _obs_numpy(side: str) -> Dict[str, np.ndarray]:
            return {
                k: v.detach().cpu().numpy().astype(np.float32)
                for k, v in env.core.get_obs_tensors(side=side).items()
            }

        def _run_pass(blue_model, red_model, *, current_side: str) -> Tuple[int, int, int]:
            wins = 0
            losses = 0
            draws = 0
            env.core.reset_all()
            completed = 0
            while completed < episodes:
                blue_obs = _obs_numpy("blue")
                red_obs = _obs_numpy("red")
                blue_actions_np, _ = blue_model.predict(blue_obs, deterministic=True)
                red_actions_np, _ = red_model.predict(red_obs, deterministic=True)
                blue_actions = torch.as_tensor(blue_actions_np, dtype=torch.int64, device=env.core.device)
                red_actions = torch.as_tensor(red_actions_np, dtype=torch.int64, device=env.core.device)
                _, _, terminated, truncated, infos = env.core.step(
                    blue_actions,
                    tensor_obs=True,
                    red_action_flat=red_actions,
                )
                done = torch.logical_or(terminated, truncated)
                if not bool(done[0].item()):
                    continue
                summary = parse_episode_result(infos[0])
                if summary is not None:
                    if current_side == "blue":
                        if summary.blue_score > summary.red_score:
                            wins += 1
                        elif summary.blue_score < summary.red_score:
                            losses += 1
                        else:
                            draws += 1
                    else:
                        if summary.red_score > summary.blue_score:
                            wins += 1
                        elif summary.red_score < summary.blue_score:
                            losses += 1
                        else:
                            draws += 1
                    completed += 1
                env.core.reset_indices(done)
            return wins, losses, draws

        blue_pass = _run_pass(self.model, snapshot_model, current_side="blue")
        red_pass = _run_pass(snapshot_model, self.model, current_side="red")
        total_w = blue_pass[0] + red_pass[0]
        total_l = blue_pass[1] + red_pass[1]
        total_d = blue_pass[2] + red_pass[2]
        return {
            "blue": blue_pass,
            "red": red_pass,
            "avg": (total_w, total_l, total_d),
        }

    def _record_matchup(self, label: str, result: Tuple[int, int, int]) -> None:
        wins, losses, draws = result
        total = max(1, wins + losses + draws)
        wr = wins / total
        _log_line(
            f"[PPO|LEAGUE_EVAL] ep={self.episode_idx} opp={label} "
            f"W={wins} L={losses} D={draws} WR={100.0 * wr:.1f}% over {total} episodes"
        )
        metric = label.lower().replace(":", "_").replace("/", "_").replace("-", "_")
        self.logger.record(f"league_eval/{metric}_win_rate", wr)
        self.logger.record(f"league_eval/{metric}_draw_rate", draws / total)

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
            if not bool(info.get("league_mode", False)):
                continue
            if not bool(getattr(self.cfg, "enable_league_eval", True)):
                continue
            every = max(1, int(getattr(self.cfg, "league_eval_every_episodes", 500)))
            if self.episode_idx % every != 0:
                continue

            episodes = max(1, int(getattr(self.cfg, "league_eval_episodes", 6)))
            # In league mode we've already passed the OP1/OP2 curriculum gates; league eval only needs OP3 + species + snapshots.
            for scripted_tag in ("OP3",):
                result = self._run_eval_matchup("SCRIPTED", scripted_tag, episodes=episodes, phase=scripted_tag)
                if result is not None:
                    self._record_matchup(f"SCRIPTED:{scripted_tag}", result)

            for species_tag in ("RUSHER", "CAMPER", "BALANCED"):
                result = self._run_eval_matchup("SPECIES", species_tag, episodes=episodes, phase="OP3")
                if result is not None:
                    self._record_matchup(f"SPECIES:{species_tag}", result)

            latest_snapshot = self.league.latest_snapshot_key()
            if latest_snapshot:
                result = self._run_eval_matchup("SNAPSHOT", latest_snapshot, episodes=episodes, phase="OP3")
                if result is not None:
                    self._record_matchup(path_to_snapshot_key(latest_snapshot), result)

            if self.league.snapshots:
                spec = self.league.sample_snapshot(target_rating=self.league.learner_rating)
                if spec.kind == "SNAPSHOT":
                    result = self._run_eval_matchup("SNAPSHOT", spec.key, episodes=episodes, phase="OP3")
                    if result is not None:
                        self._record_matchup(f"RATED/{path_to_snapshot_key(spec.key)}", result)

            if bool(getattr(self.cfg, "enable_mirror_eval", True)):
                mirror_result = self._run_side_swapped_mirror_eval(max(1, int(getattr(self.cfg, "mirror_eval_episodes", episodes))))
                if mirror_result is not None:
                    blue_w, blue_l, blue_d = mirror_result["blue"]
                    red_w, red_l, red_d = mirror_result["red"]
                    self._record_matchup("MIRROR_CURRENT_AVG", mirror_result["avg"])
                    self._record_matchup("MIRROR_CURRENT_AS_BLUE", (blue_w, blue_l, blue_d))
                    self._record_matchup("MIRROR_CURRENT_AS_RED", (red_w, red_l, red_d))
        return True

    def _on_training_end(self) -> None:
        if self._eval_env is not None:
            try:
                self._eval_env.close()
            except Exception:
                pass
            self._eval_env = None
        try:
            if os.path.exists(self._mirror_eval_snapshot_path):
                os.remove(self._mirror_eval_snapshot_path)
        except Exception:
            pass


class CurriculumNoLeagueCallback(BaseCallback):
    """OP1 -> OP2 -> OP3 curriculum only; no league, no species, no snapshots."""

    def __init__(self, *, cfg: PPOConfig, curriculum: CurriculumState) -> None:
        _v = 1 if getattr(cfg, "verbose_training", False) else 0
        super().__init__(verbose=_v)
        self.cfg = cfg
        self.curriculum = curriculum
        self.episode_idx = 0
        self.win_count = 0
        self.loss_count = 0
        self.draw_count = 0
        self.phase_win_count = 0
        self.phase_loss_count = 0
        self.phase_draw_count = 0
        self._opponent_stats: Dict[str, Dict[str, int]] = {}
        self._opponent_history: List[Tuple[str, str]] = []
        self._opponent_window = getattr(cfg, "opponent_tracking_window", 100)

    @staticmethod
    def _curriculum_window_value(result: str) -> float:
        """Paper curriculum windows count draws as non-wins."""
        return 1.0 if str(result).upper() == "WIN" else 0.0

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
                self.phase_win_count += 1
            elif blue_score < red_score:
                result = "LOSS"
                actual = 0.0
                self.loss_count += 1
                self.phase_loss_count += 1
            else:
                result = "DRAW"
                actual = 0.5
                self.draw_count += 1
                self.phase_draw_count += 1

            phase = self.curriculum.phase
            self.curriculum.phase_episode_count += 1
            self.curriculum.record_result(phase, self._curriculum_window_value(result))

            # Debug: log advancement conditions every 50 episodes (only when verbose)
            if self.verbose and self.episode_idx % 50 == 0 and phase == "OP1":
                min_eps = self.curriculum.config.min_episodes.get(phase, 0)
                min_wr = self.curriculum.config.min_winrate.get(phase, 0.0)
                winrate = self.curriculum.phase_winrate(phase)
                meets_eps = self.curriculum.phase_episode_count >= min_eps
                meets_wr = winrate >= min_wr
                # Elo check is skipped when elo_margin=0 (curriculum-only mode)
                meets_elo = (self.curriculum.config.elo_margin <= 0) or (1300.0 >= (1200.0 + self.curriculum.config.elo_margin))
                _log_line(f"[CURR-DEBUG] OP1->OP2: eps={self.curriculum.phase_episode_count}/{min_eps} ({meets_eps}), "
                          f"wr={winrate:.3f}>={min_wr} ({meets_wr}), elo_skip={self.curriculum.config.elo_margin<=0} ({meets_elo})")

            old_phase = phase
            self.curriculum.advance_if_ready(
                learner_rating=1300.0,
                opponent_rating=1200.0,
                win_by=win_by,
            )
            phase = self.curriculum.phase
            if phase != old_phase:
                phase_tot = self.phase_win_count + self.phase_loss_count + self.phase_draw_count
                phase_wr = (100.0 * self.phase_win_count / phase_tot) if phase_tot > 0 else 0.0
                _log_line(f"[PPO] Phase {old_phase} complete: W={self.phase_win_count} L={self.phase_loss_count} D={self.phase_draw_count} WR={phase_wr:.1f}%")
                self.phase_win_count = 0
                self.phase_loss_count = 0
                self.phase_draw_count = 0
                if self.verbose:
                    _log_line(f"[CURR] ADVANCED: {old_phase} -> {phase} at episode {self.episode_idx}")

            opp_key = summary.opponent_key()
            self._update_opponent_stats(opp_key, result)

            if self.verbose:
                _log_line(
                    f"[PPO|CURR_NO_LEAGUE] ep={self.episode_idx} result={result} "
                    f"score={blue_score}:{red_score} phase={phase} "
                    f"W={self.win_count} | L={self.loss_count} | D={self.draw_count}"
                )
                if self.episode_idx % 50 == 0:
                    self._print_opponent_distribution()

            # Summary every 1000 episodes (always, not gated by verbose)
            if self.episode_idx > 0 and self.episode_idx % 1000 == 0:
                wr = (self.win_count / self.episode_idx) * 100
                _log_line(f"[PPO] ep={self.episode_idx} mode=PAPER phase={phase} opp={phase} W={self.win_count} L={self.loss_count} D={self.draw_count} WR={wr:.1f}%")

            self.logger.record("curr_noleague/episode", self.episode_idx)
            self.logger.record("curr_noleague/win_rate", self.win_count / max(1, self.episode_idx))
            self.logger.record("curr_noleague/draw_rate", self.draw_count / max(1, self.episode_idx))
            self.logger.record("curr_noleague/phase_idx", float(self.curriculum.phase_idx))

            env = self.model.get_env()
            if env is not None:
                try:
                    env.env_method("set_next_opponent", "SCRIPTED", phase, indices=[i])
                    env.env_method("set_phase", phase, indices=[i])
                except Exception:
                    try:
                        env.env_method("set_next_opponent", "SCRIPTED", phase)
                        env.env_method("set_phase", phase)
                    except Exception:
                        pass

        return True
    
    def _on_training_end(self) -> None:
        """Print final opponent diet summary at end of training."""
        if self._opponent_stats:
            total_episodes = sum(stats.get("total", 0) for stats in self._opponent_stats.values())
            if total_episodes > 0:
                _log_line("\n" + "=" * 60)
                _log_line("NO-LEAGUE OPPONENT DIET SUMMARY (whole training)")
                _log_line("=" * 60)
                for opp_key in sorted(self._opponent_stats.keys()):
                    stats = self._opponent_stats[opp_key]
                    total = stats.get("total", 0)
                    wins = stats.get("wins", 0)
                    losses = stats.get("losses", 0)
                    draws = stats.get("draws", 0)
                    pct = (total / total_episodes) * 100
                    wr = (wins / max(1, total)) * 100 if total > 0 else 0.0
                    _log_line(f"  {opp_key}: {total} episodes ({pct:.1f}%) | WR={wr:.1f}% ({wins}W/{losses}L/{draws}D)")
                _log_line("=" * 60 + "\n")
    
    def _update_opponent_stats(self, opp_key: str, result: str):
        """Track opponent distribution and results (same logic as LeagueCallback)."""
        if opp_key not in self._opponent_stats:
            self._opponent_stats[opp_key] = {"wins": 0, "losses": 0, "draws": 0, "total": 0}
        
        self._opponent_stats[opp_key]["total"] += 1
        if result == "WIN":
            self._opponent_stats[opp_key]["wins"] += 1
        elif result == "LOSS":
            self._opponent_stats[opp_key]["losses"] += 1
        else:
            self._opponent_stats[opp_key]["draws"] += 1
        
        self._opponent_history.append((opp_key, result))
        if len(self._opponent_history) > self._opponent_window:
            old_opp_key, old_result = self._opponent_history.pop(0)
            if old_opp_key in self._opponent_stats:
                self._opponent_stats[old_opp_key]["total"] = max(0, self._opponent_stats[old_opp_key]["total"] - 1)
                if old_result == "WIN":
                    self._opponent_stats[old_opp_key]["wins"] = max(0, self._opponent_stats[old_opp_key]["wins"] - 1)
                elif old_result == "LOSS":
                    self._opponent_stats[old_opp_key]["losses"] = max(0, self._opponent_stats[old_opp_key]["losses"] - 1)
                else:
                    self._opponent_stats[old_opp_key]["draws"] = max(0, self._opponent_stats[old_opp_key]["draws"] - 1)
    
    def _print_opponent_distribution(self):
        """Print opponent distribution (last N episodes and whole training)."""
        if not self._opponent_stats:
            return
        
        recent_opps: Dict[str, int] = {}
        for opp_key, _ in self._opponent_history[-self._opponent_window:]:
            recent_opps[opp_key] = recent_opps.get(opp_key, 0) + 1
        
        total_episodes = sum(stats.get("total", 0) for stats in self._opponent_stats.values())
        
        if recent_opps:
            parts = []
            for opp_key, count in sorted(recent_opps.items(), key=lambda x: -x[1]):
                pct = (count / max(1, len(self._opponent_history[-self._opponent_window:]))) * 100
                parts.append(f"{opp_key}:{count}({pct:.0f}%)")
            _log_line(f"[NoLeagueDiet|last_{self._opponent_window}] " + " | ".join(parts))
        
        if total_episodes > 0:
            parts = []
            for opp_key in sorted(self._opponent_stats.keys()):
                stats = self._opponent_stats[opp_key]
                total = stats.get("total", 0)
                pct = (total / total_episodes) * 100
                parts.append(f"{opp_key}:{total}({pct:.0f}%)")
            _log_line(f"[NoLeagueDiet|total] " + " | ".join(parts))


class SelfPlayCallback(BaseCallback):
    """Self-play with rolling snapshot pool: counter resets to 1 when at max and old are deleted."""

    def __init__(self, *, cfg: PPOConfig, league: EloLeague) -> None:
        _v = 1 if getattr(cfg, "verbose_training", False) else 0
        super().__init__(verbose=_v)
        self.cfg = cfg
        self.league = league
        self.episode_idx = 0
        self.win_count = 0
        self.loss_count = 0
        self.draw_count = 0
        self._max_snapshots = max(0, int(getattr(cfg, "self_play_max_snapshots", 0)))
        self._snapshot_roll_index = 0
        self._total_snapshots_created = 0

    def _choose_training_snapshot(self) -> Optional[str]:
        if len(self.league.snapshots) == 0:
            return None
        latest_snapshot = self.league.latest_snapshot_key()
        if bool(self.cfg.self_play_use_latest_snapshot):
            return latest_snapshot
        latest_prob = max(0.0, min(1.0, float(getattr(self.cfg, "self_play_latest_snapshot_prob", 0.35))))
        if latest_snapshot and len(self.league.snapshots) > 1 and self.league.rng.random() < latest_prob:
            return latest_snapshot
        spec = self.league.sample_snapshot()
        if spec.kind == "SNAPSHOT":
            return spec.key
        return latest_snapshot

    def _enforce_snapshot_limit(self) -> None:
        if self._max_snapshots <= 0:
            return
        while len(self.league.snapshots) > self._max_snapshots:
            oldest = self.league.snapshots.pop(0)
            try:
                if oldest and os.path.exists(oldest):
                    os.remove(oldest)
            except Exception as exc:
                _log_line(f"[WARN] snapshot cleanup failed: {exc}")

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
                max_s = max(1, self._max_snapshots)
                self._snapshot_roll_index = (self._snapshot_roll_index % max_s) + 1
                slot = self._snapshot_roll_index
                prefix = f"{self.cfg.run_tag}_selfplay_snapshot"
                path = os.path.join(self.cfg.checkpoint_dir, f"{prefix}_slot{slot:03d}")
                try:
                    self.model.save(path)
                except Exception as exc:
                    _log_line(f"[WARN] snapshot save failed: {exc}")
                else:
                    self.league.add_snapshot(path + ".zip")
                    self._enforce_snapshot_limit()
                    self._total_snapshots_created += 1

            if self.verbose:
                _log_line(
                    f"[PPO|SELF] ep={self.episode_idx} result={result} "
                    f"score={blue_score}:{red_score} "
                    f"snapshots={len(self.league.snapshots)} total_created={self._total_snapshots_created} "
                    f"W={self.win_count} | L={self.loss_count} | D={self.draw_count}"
                )

            # Summary every 1000 episodes (always, not gated by verbose)
            if self.episode_idx > 0 and self.episode_idx % 1000 == 0:
                wr = (self.win_count / self.episode_idx) * 100
                _log_line(f"[PPO] ep={self.episode_idx} mode=SELF_PLAY phase=SELF_PLAY opp=self W={self.win_count} L={self.loss_count} D={self.draw_count} WR={wr:.1f}%")

            self.logger.record("self/episode", self.episode_idx)
            self.logger.record("self/win_rate", self.win_count / max(1, self.episode_idx))
            self.logger.record("self/draw_rate", self.draw_count / max(1, self.episode_idx))
            self.logger.record("self/snapshots", float(len(self.league.snapshots)))
            self.logger.record("self/total_snapshots_created", float(self._total_snapshots_created))

            next_snapshot = self._choose_training_snapshot()

            if len(self.league.snapshots) == 0:
                fallback_path = os.path.join(
                    self.cfg.checkpoint_dir, f"{self.cfg.run_tag}_selfplay_init_fallback"
                )
                try:
                    self.model.save(fallback_path)
                except Exception as exc:
                    _log_line(f"[WARN] self-play fallback save failed: {exc}")
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
        _v = 1 if getattr(cfg, "verbose_training", False) else 0
        super().__init__(verbose=_v)
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
                _log_line(
                    f"[PPO|FIXED] ep={self.episode_idx} result={result} "
                    f"score={blue_score}:{red_score} opp=SCRIPTED:{opp} "
                    f"W={self.win_count} | L={self.loss_count} | D={self.draw_count}"
                )

            # Summary every 1000 episodes (always, not gated by verbose)
            if self.episode_idx > 0 and self.episode_idx % 1000 == 0:
                wr = (self.win_count / self.episode_idx) * 100
                opp = str(summary.scripted_tag or self.cfg.fixed_opponent_tag).upper()
                _log_line(f"[PPO] ep={self.episode_idx} mode=FIXED phase=FIXED opp=SCRIPTED:{opp} W={self.win_count} L={self.loss_count} D={self.draw_count} WR={wr:.1f}%")

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
        if (self.n_calls - 1) % n_steps != 0:
            return True
        update_id = (self.n_calls - 1) // n_steps
        if update_id <= self._last_checked_update:
            return True
        is_first_check = self._last_checked_update == -1
        self._last_checked_update = update_id

        name_to_value = getattr(self.logger, "name_to_value", None) or {}
        if "train/approx_kl" not in name_to_value:
            if self.verbose and is_first_check:
                print(
                    f"[KLGuardrail] WARNING: 'train/approx_kl' not found in logger. "
                    f"Guardrail may be inactive. Check that PPO is logging approx_kl."
                )
        approx_kl = float(name_to_value.get("train/approx_kl", 0.0))

        if approx_kl > self.threshold:
            self._spike_count += 1
            if self.verbose:
                self.logger.record("train/kl_guardrail_spike_count", self._spike_count)
            if self._spike_count >= self.consecutive:
                setattr(self.model, "_kl_guardrail_triggered", True)
                if self.verbose:
                    stable_enabled = bool(getattr(getattr(self.model, "cfg", None), "use_stable_marl_ppo", False))
                    guidance = (
                        "stable MARL PPO is already enabled; lower lr / n_epochs / clip_range, and use target_kl early stopping."
                        if stable_enabled
                        else "consider enabling use_stable_marl_ppo."
                    )
                    print(
                        f"[KLGuardrail] approx_kl exceeded {self.threshold} for {self._spike_count} consecutive updates "
                        f"(last approx_kl={approx_kl:.4f}). Set model._kl_guardrail_triggered=True; {guidance}"
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
    """Stream Top 5 IROS-style metrics per episode to CSV (one row per episode, no in-memory accumulation)."""

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
        "opponent_switch_count",
        "vec_schema_version",
    ]

    def __init__(self, *, save_path: str) -> None:
        super().__init__(verbose=0)
        self.save_path = str(save_path)
        self._header_written = False
        self._episode_id = 0
        self._opponent_switch_count = 0
        self._last_opponent_key: Optional[str] = None
        self._rows_written = 0

    def _fmt(self, v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, float):
            return f"{v:.6g}"
        return str(v)

    def _write_row(self, row: Dict[str, Any]) -> None:
        path = self.save_path
        if not path.lower().endswith(".csv"):
            path = path + ".csv"
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        mode = "w" if not self._header_written else "a"
        with open(path, mode, newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self.CSV_COLUMNS, extrasaction="ignore")
            if not self._header_written:
                w.writeheader()
                self._header_written = True
            w.writerow({k: self._fmt(row.get(k)) for k in self.CSV_COLUMNS})
        self._rows_written += 1

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
            try:
                self._write_row(row)
            except Exception as exc:
                print(f"[WARN] Metrics CSV write failed: {exc}")
        return True

    def _on_training_end(self) -> None:
        if self.verbose and self._rows_written > 0:
            path = self.save_path if self.save_path.lower().endswith(".csv") else self.save_path + ".csv"
            print(f"[Metrics] Wrote {self._rows_written} rows to {path}")


def train_ppo(cfg: Optional[PPOConfig] = None) -> None:
    cfg = cfg or PPOConfig()
    set_global_seed(cfg.seed, torch_seed=True, deterministic=cfg.use_deterministic)

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    raw_mode = str(cfg.mode).upper().strip()
    mode = _normalize_train_mode(cfg.mode)
    cfg.mode = mode
    max_agents = int(getattr(cfg, "max_blue_agents", 2))
    team_size = _agents_suffix(max_agents)
    print(f"[PPO] Agents: {max_agents} per team ({team_size}) | mode={mode} | run_tag={cfg.run_tag!r}")
    if raw_mode != mode:
        print(f"[PPO] Mode alias normalized: {raw_mode} -> {mode}")
    print(f"[PPO] Total timesteps: {cfg.total_timesteps:,}")
    print(f"[PPO] Saves: final_{cfg.run_tag}.zip | snapshots/ckpts: {cfg.run_tag}_*")
    print(f"[PPO] Checkpoint dir: {cfg.checkpoint_dir}")
    if "/content/drive" in os.path.abspath(cfg.checkpoint_dir) or "MyDrive" in cfg.checkpoint_dir:
        print("[PPO] Saving to Google Drive — progress will persist if runtime disconnects.")
    if not getattr(cfg, "verbose_training", False):
        print("[PPO] Quiet mode: no per-episode logs (faster). Use --verbose-training to enable.")
    print("[PPO] Progress: steps every 50k timesteps; W/L/D summary + per-phase (OP1/OP2/OP3) every 1000 episodes.")

    # Larger team sizes need shorter episodes / smaller rollouts to keep wall-clock time reasonable.
    if max_agents == 6:
        original_n_envs = int(getattr(cfg, "n_envs", 8))
        original_n_steps = int(getattr(cfg, "n_steps", 2048))
        original_max_decision_steps = int(getattr(cfg, "max_decision_steps", 400))
        if original_n_envs > 1 or original_n_steps > 512 or original_max_decision_steps > 250:
            print(
                f"[PPO] {team_size}: using fast profile for wall-clock speed: "
                f"n_envs {original_n_envs}->1, n_steps {original_n_steps}->512, "
                f"max_decision_steps {original_max_decision_steps}->250"
            )
        cfg.n_envs = min(original_n_envs, 1)
        cfg.n_steps = min(original_n_steps, 512)
        cfg.max_decision_steps = min(original_max_decision_steps, 250)
    elif max_agents > 6:
        # 8v8 has a much larger observation tensor; keep rollout buffer size reasonable to avoid OOM.
        original_n_envs = int(getattr(cfg, "n_envs", 4))
        original_n_steps = int(getattr(cfg, "n_steps", 2048))
        original_max_decision_steps = int(getattr(cfg, "max_decision_steps", 400))
        if original_n_envs > 2 or original_n_steps > 1024 or original_max_decision_steps > 250:
            print(
                f"[PPO] {team_size}: reducing rollout/episode size for memory: "
                f"n_envs {original_n_envs}->2, n_steps {original_n_steps}->1024, "
                f"max_decision_steps {original_max_decision_steps}->250"
            )
        cfg.n_envs = min(original_n_envs, 2)
        cfg.n_steps = min(original_n_steps, 1024)
        cfg.max_decision_steps = min(original_max_decision_steps, 250)

    # 4v4/8v8: never force 100% OP3; use mix so winrate stays in learnable band (30–70%)
    # Logging:
    # - League (CURRICULUM_LEAGUE) prints the actual scripted/species/snapshot mix.
    # - Self-play uses pure self-play vs its own snapshot pool; we log that separately instead of reusing the league mix line.
    uses_league = mode == TrainMode.CURRICULUM_LEAGUE.value
    match_op3 = getattr(cfg, "match_op3_exposure", False) and (max_agents <= 2)
    if match_op3:
        anchor_op3_prob = 1.0
        species_prob = 0.0
        snapshot_prob = 0.0
        if uses_league:
            print("[League] match_op3_exposure=True: 100% OP3 (2v2 control)")
    else:
        anchor_op3_prob = float(getattr(cfg, "league_anchor_op3_prob", 0.60))
        species_prob = float(getattr(cfg, "league_species_prob", 0.20))
        snapshot_prob = float(getattr(cfg, "league_snapshot_prob", 0.20))
        if mode == TrainMode.CURRICULUM_LEAGUE.value:
            print(
                f"[League] {team_size}: using league mix "
                f"(OP3={100.0 * anchor_op3_prob:.0f}%, species={100.0 * species_prob:.0f}%, "
                f"snapshots={100.0 * snapshot_prob:.0f}%)"
            )
        elif mode == TrainMode.SELF_PLAY.value:
            max_snaps = int(getattr(cfg, "self_play_max_snapshots", 0))
            if max_snaps > 0:
                print(f"[SelfPlay] {team_size}: pure self-play vs rolling snapshot pool (max {max_snaps} snapshots)")
            else:
                print(f"[SelfPlay] {team_size}: pure self-play vs latest checkpoint (no snapshot pool configured)")
    league = EloLeague(
        seed=cfg.seed,
        k_factor=32.0,
        matchmaking_tau=200.0,
        scripted_floor=0.50,
        species_prob=species_prob,
        snapshot_prob=snapshot_prob,
        anchor_op3_prob=anchor_op3_prob,
        species_rusher_bias=float(getattr(cfg, "species_rusher_bias", 0.40)),
        use_stability_mix=False,
        min_episodes_per_opponent=int(getattr(cfg, "min_episodes_per_opponent", 3)),
    )

    curriculum: Optional[CurriculumState] = None
    controller: Optional[CurriculumController] = None

    # Elo margin only applies in league mode (after OP3 gate); set to 0 for curriculum phases
    # In CURRICULUM_LEAGUE, Elo check is skipped during curriculum phases (OP1/OP2/OP3) via dummy ratings
    elo_margin = 80.0 if mode == TrainMode.CURRICULUM_LEAGUE.value else 0.0

    # 4v4/8v8: same curriculum bar is harder (red has more agents too; blue coordination is harder).
    # Use relaxed win-rate gates, more episodes, and more total timesteps so all baselines (Paper, League, Fixed) can succeed.
    max_agents = int(getattr(cfg, "max_blue_agents", 2))
    if max_agents > 2:
        _min_episodes = {"OP1": 350, "OP2": 300, "OP3": 350}
        # Use same phase win-rate thresholds for 2v2/4v4/8v8: OP1=100%, OP2=90%, OP3=80%
        _min_winrate = {"OP1": 1.00, "OP2": 0.90, "OP3": 0.80}
        _winrate_window_by_phase = {"OP1": 80, "OP2": 80, "OP3": 120}
        # Extra OP3→League gate: require 80% vs OP3 over last _min_games_vs_op3 games
        _min_winrate_vs_op3 = 0.80
        _min_games_vs_op3 = 50
    else:
        _min_episodes = {"OP1": 200, "OP2": 200, "OP3": 250}
        _min_winrate = {"OP1": 1.00, "OP2": 0.90, "OP3": 0.80}
        _winrate_window_by_phase = {"OP1": 50, "OP2": 50, "OP3": 100}
        _min_winrate_vs_op3 = 0.0
        _min_games_vs_op3 = 0

    curriculum_config = CurriculumConfig(
        phases=["OP1", "OP2", "OP3"],
        min_episodes=_min_episodes,
        min_winrate=_min_winrate,
        winrate_window=100,
        winrate_window_by_phase=_winrate_window_by_phase,
        required_win_by={"OP1": 0, "OP2": 1, "OP3": 1},
        elo_margin=elo_margin,
        switch_to_league_after_op3_win=(mode == TrainMode.CURRICULUM_LEAGUE.value),
        min_winrate_vs_op3=_min_winrate_vs_op3,
        min_games_vs_op3=_min_games_vs_op3,
    )
    if mode == TrainMode.CURRICULUM_LEAGUE.value:
        curriculum = CurriculumState(curriculum_config)
        controller = CurriculumController(
            CurriculumControllerConfig(seed=cfg.seed),
            league=league,
        )
    elif mode == TrainMode.CURRICULUM_NO_LEAGUE.value:
        curriculum = CurriculumState(curriculum_config)

    if mode == TrainMode.FIXED_OPPONENT.value:
        default_opponent = ("SCRIPTED", str(cfg.fixed_opponent_tag).upper())
        phase_name = str(cfg.fixed_opponent_tag).upper()
    elif mode == TrainMode.SELF_PLAY.value:
        default_opponent = ("SNAPSHOT", "__SELF_PLAY_BOOTSTRAP__")
        phase_name = "SELF_PLAY"
    elif mode == TrainMode.CURRICULUM_NO_LEAGUE.value and curriculum is not None:
        default_opponent = ("SCRIPTED", curriculum.phase)
        phase_name = curriculum.phase
    else:
        default_opponent = ("SCRIPTED", "OP1")
        phase_name = curriculum.phase if curriculum is not None else "OP1"

    # If using CUDA, check that this PyTorch build supports the GPU (e.g. RTX 50-series needs nightly/sm_120)
    if str(cfg.device).lower().startswith("cuda"):
        try:
            torch.zeros(1, device=cfg.device)
        except RuntimeError as e:
            err = str(e).lower()
            if "no kernel image" in err or "not compatible" in err or "sm_" in err:
                print(f"[PPO] GPU not supported by this PyTorch build ({e}). Falling back to CPU.")
                print("[PPO] To use RTX 50-series (Blackwell), install PyTorch nightly: pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128")
                cfg.device = "cpu"

    gpu_cfg = GPUFieldConfig(
        n_envs=max(1, int(cfg.n_envs)),
        max_blue_agents=max(1, int(getattr(cfg, "max_blue_agents", 2))),
        max_red_agents=max(1, int(getattr(cfg, "max_blue_agents", 2))),
        max_decision_steps=max(1, int(cfg.max_decision_steps)),
        aquaticus_profile=True,
        rules_profile="OURS",
        device=str(cfg.device),
        seed=int(cfg.seed),
    )
    print(f"[PPO] Using GPU-native batched env: n_envs={gpu_cfg.n_envs}, agents={gpu_cfg.max_blue_agents}v{gpu_cfg.max_red_agents}, device={gpu_cfg.device}")
    venv = VecMonitor(GPUCTFVecEnv(gpu_cfg))

    try:
        venv.env_method("set_stress_schedule", STRESS_BY_PHASE)
    except Exception:
        pass
    try:
        # Project default: enforce the repo's current OURS scoring/rules profile.
        venv.env_method("set_dynamics_config", {"rules_profile": "OURS"})
    except Exception:
        pass
    try:
        venv.env_method("set_phase", phase_name)
    except Exception:
        pass
    # Initial scripted-opponent parameters (including deception and speed for red team).
    # Use actual phase/key (OP1=easier, OP2=medium, OP3=strong) so OP1/OP2 are not able to score easily.
    try:
        kind, key = default_opponent
        opp_params = sample_batched_opponent_params(
            kind=kind,
            key=key,
            phase=phase_name,
            n_agents=gpu_cfg.max_red_agents,
            batch_size=gpu_cfg.n_envs,
            device=gpu_cfg.device,
        )
        dyn_cfg: Dict[str, Any] = {}
        if "deception_prob" in opp_params:
            dyn_cfg["deception_prob"] = opp_params["deception_prob"]
        if "speed_mult" in opp_params:
            dyn_cfg["speed_mult"] = opp_params["speed_mult"]
        if "attacker_style" in opp_params:
            dyn_cfg["attacker_style"] = opp_params["attacker_style"]
        if "defender_style" in opp_params:
            dyn_cfg["defender_style"] = opp_params["defender_style"]
        if "role_switch_prob" in opp_params:
            dyn_cfg["role_switch_prob"] = opp_params["role_switch_prob"]
        if dyn_cfg:
            venv.env_method("set_dynamics_config", dyn_cfg)
    except Exception as exc:
        print(f"[PPO] opponent_params sampling failed (using defaults): {exc}")

    policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))
    use_tokenized = bool(getattr(cfg, "use_tokenized_obs", False))
    if use_tokenized:
        policy_kwargs["features_extractor_class"] = TokenizedCombinedExtractor
        policy_kwargs["features_extractor_kwargs"] = dict(cnn_output_dim=256, normalized_image=True)

    # Step 4: Stable MARL PPO (Fix 4.1) or reduced aggressiveness
    learning_rate = float(cfg.learning_rate)
    ent_coef = float(cfg.ent_coef)
    clip_range = float(cfg.clip_range)
    n_epochs = int(cfg.n_epochs)
    batch_size = int(cfg.batch_size)

    use_curriculum = mode in (TrainMode.CURRICULUM_LEAGUE.value, TrainMode.CURRICULUM_NO_LEAGUE.value)
    if use_curriculum or getattr(cfg, "use_stable_marl_ppo", False):
        learning_rate = 1.5e-4
        ent_coef = 0.005
        clip_range = 0.10
        n_epochs = 2
        batch_size = 1024
        print("[PPO] Using stable MARL PPO: lr=1.5e-4, n_epochs=2, clip_range=0.10, ent_coef=0.005, batch_size=1024, target_kl=0.02")
    elif getattr(cfg, "use_reduced_aggressiveness", False):
        learning_rate = learning_rate * 0.67
        ent_coef = ent_coef * 0.5
        clip_range = clip_range * 0.75
        print(f"[PPO] Using reduced aggressiveness: LR={learning_rate:.2e}, ent_coef={ent_coef:.3f}, clip_range={clip_range:.2f}")

    # 4v4/8v8: gentler LR to reduce KL spikes and stabilize (scripted OP3 also scaled down in opponent_params)
    if max_agents > 2:
        learning_rate = learning_rate * 0.75
        print(f"[PPO] {team_size}: using lr={learning_rate:.2e} for stability")

    # KL sanity check: run with lr=0 and verify approx_kl ~ 0 in logs; if huge, logprob/action plumbing is broken
    if getattr(cfg, "test_kl_zero_lr", False):
        learning_rate = 0.0
        print("[PPO] test_kl_zero_lr=True: learning_rate=0 — verify approx_kl ~ 0 in logs (if not, check old_logprob/action pairing)")
    
    rollout_size = max(1, int(cfg.n_steps) * max(1, int(cfg.n_envs)))
    if batch_size > rollout_size:
        adjusted_batch_size = rollout_size
        for candidate in (1024, 512, 256, 128, 64, 32):
            if candidate <= rollout_size and rollout_size % candidate == 0:
                adjusted_batch_size = candidate
                break
        print(f"[PPO] Adjusting batch_size for rollout size: {batch_size}->{adjusted_batch_size} (n_steps*n_envs={rollout_size})")
        batch_size = adjusted_batch_size

    # Optional resume from checkpoint: when cfg.load_path is set and the file exists, load PPO instead of creating a fresh model.
    load_path = getattr(cfg, "load_path", None)
    if load_path and os.path.isfile(load_path):
        from stable_baselines3 import PPO as _PPO
        print(f"[PPO] Resuming from checkpoint: {load_path}")
        model = _PPO.load(
            load_path,
            env=venv,
            device=cfg.device,
            custom_objects={
                "observation_space": venv.observation_space,
                "action_space": venv.action_space,
                "policy_class": MaskedMultiInputPolicy,
            },
        )
        # Ensure cfg/run_tag/checkpoint_dir are kept from current run, not from the checkpoint.
        model.cfg = cfg
    else:
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
            vf_coef=1.0,
            max_grad_norm=float(cfg.max_grad_norm),
            target_kl=float(cfg.target_kl) if getattr(cfg, "target_kl", None) is not None else None,
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
        model.cfg = cfg

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

    # Top 5 IROS-style metrics: CSV at end of training (simple, publish-friendly)
    metrics_csv_path = os.path.join(cfg.checkpoint_dir, f"{cfg.run_tag}_metrics")
    callbacks.append(MetricsCSVCallback(save_path=metrics_csv_path))

    # Progress:
    # - If SB3's built-in progress bar is enabled (cfg.enable_progress_bar), let SB3 handle tqdm/rich.
    # - Otherwise, fall back to our simple ProgressLogCallback (prints every 50k steps).
    if not getattr(cfg, "enable_progress_bar", False):
        callbacks.append(ProgressLogCallback(total_timesteps=int(cfg.total_timesteps), interval=50_000))

    # Fix 4.2: KL guardrail – log approx_kl and set model._kl_guardrail_triggered if spikes repeatedly
    if getattr(cfg, "approx_kl_threshold", 0) > 0 and getattr(cfg, "kl_guardrail_consecutive", 0) > 0:
        callbacks.append(
            KLGuardrailCallback(
                threshold=float(cfg.approx_kl_threshold),
                consecutive=int(cfg.kl_guardrail_consecutive),
                verbose=1,
            )
        )

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

    if cfg.enable_eval and mode not in (TrainMode.CURRICULUM_LEAGUE.value, TrainMode.SELF_PLAY.value):
        eval_env = VecMonitor(GPUCTFVecEnv(
            GPUFieldConfig(
                n_envs=1,
                max_blue_agents=max(1, int(getattr(cfg, "max_blue_agents", 2))),
                max_red_agents=max(1, int(getattr(cfg, "max_blue_agents", 2))),
                max_decision_steps=max(1, int(cfg.max_decision_steps)),
                aquaticus_profile=True,
                rules_profile="OURS",
                device=str(cfg.device),
                seed=int(cfg.seed),
            )
        ))
        
        # CRITICAL: Match training environment setup (stress schedule + phase)
        # This ensures eval environment matches training and viewer environments
        try:
            eval_env.env_method("set_stress_schedule", STRESS_BY_PHASE)
        except Exception:
            pass
        try:
            eval_env.env_method("set_phase", "OP3")  # EvalCallback always uses OP3
        except Exception:
            pass
        
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

    # Optional TQDM progress bar for ETA (if supported by this SB3 version).
    learn_kwargs: Dict[str, Any] = {}
    if getattr(cfg, "enable_progress_bar", False):
        learn_kwargs["progress_bar"] = True

    try:
        try:
            model.learn(total_timesteps=int(cfg.total_timesteps), callback=callbacks, **learn_kwargs)
        except (TypeError, ImportError) as prog_exc:
            # Older SB3 may not accept progress_bar; or tqdm/rich missing (SB3 adds ProgressBarCallback).
            if learn_kwargs.get("progress_bar") and ("progress_bar" in str(prog_exc) or "tqdm" in str(prog_exc).lower() or "rich" in str(prog_exc).lower()):
                model.learn(total_timesteps=int(cfg.total_timesteps), callback=callbacks)
            else:
                raise
    except (MemoryError, torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        # Treat both CUDA OOM and CPU/NumPy memory errors (e.g. ArrayMemoryError) as OOM so we always try to save.
        _exc_name_lower = type(exc).__name__.lower()
        _msg_lower = str(exc).lower()
        is_oom = (
            isinstance(exc, (MemoryError, torch.cuda.OutOfMemoryError))
            or "out of memory" in _msg_lower
            or "arraymemoryerror" in _exc_name_lower
        )
        if is_oom:
            crash_path = os.path.join(cfg.checkpoint_dir, f"oom_save_{cfg.run_tag}")
            try:
                model.save(crash_path)
                print(f"[PPO] OOM. Model saved to: {crash_path}.zip")
            except Exception as save_exc:
                crash_path = os.path.join(cfg.checkpoint_dir, f"crash_save_{cfg.run_tag}")
                try:
                    model.save(crash_path)
                    print(f"[PPO] OOM. Model saved to: {crash_path}.zip")
                except Exception as save_exc2:
                    print(f"[WARN] Could not save model on OOM: {save_exc2}")
            print("[PPO] To continue: restart with lower memory (e.g. --device cpu, or reduce n_envs/n_steps in code for 8v8). Load the saved .zip and train with remaining steps if your setup supports resume.")
        raise
    except Exception as exc:
        # Save current model on any other failure so progress is not lost
        crash_path = os.path.join(cfg.checkpoint_dir, f"crash_save_{cfg.run_tag}")
        try:
            model.save(crash_path)
            print(f"[PPO] Training failed. Model saved to: {crash_path}.zip")
        except Exception as save_exc:
            print(f"[WARN] Could not save model on crash: {save_exc}")
        raise

    # run_tag already includes _2v2/_4v4/_8v8 so final/checkpoints/snapshots are distinct per agent size
    final_path = os.path.join(cfg.checkpoint_dir, f"final_{cfg.run_tag}")
    model.save(final_path)
    print(f"[PPO] Training complete. Final model saved to: {final_path}.zip")


def run_verify_4v4(num_episodes: int = 10) -> None:
    """Run N random-action episodes at 4v4 on GPU env, print shapes on reset."""
    from game_field_gpu import GPUCTFVecEnv, GPUFieldConfig
    set_global_seed(42)
    cfg = GPUFieldConfig(
        n_envs=1,
        max_blue_agents=4,
        max_red_agents=4,
        max_decision_steps=400,
        aquaticus_profile=True,
        rules_profile="OURS",
        device="cpu",
        seed=42,
    )
    venv = GPUCTFVecEnv(cfg)
    for ep in range(num_episodes):
        obs = venv.reset()
        done = False
        steps = 0
        while not done and steps < 800:
            action = venv.action_space.sample()
            obs, reward, done_arr, infos = venv.step(action)
            done = bool(done_arr[0])
            steps += 1
        print(f"[Verify-4v4] episode {ep + 1}/{num_episodes} steps={steps} done={done}")
    venv.close()
    print(f"[Verify-4v4] Done. {num_episodes} random-action 4v4 episodes completed.")


def run_test_vec_schema() -> None:
    """Verify GPU core obs: vec has shape (B, N, 18), float32, finite, in bounds."""
    from game_field_gpu import BatchedCTFCore, GPUFieldConfig
    cfg = GPUFieldConfig(n_envs=1, max_blue_agents=2, max_red_agents=2, device="cpu", seed=42)
    core = BatchedCTFCore(cfg)
    core.reset_all()
    obs = core.get_obs()
    vec = obs["vec"]
    assert vec.dtype == np.float32, f"vec.dtype {vec.dtype}, expected float32"
    assert vec.ndim == 3 and vec.shape[2] == 18, f"vec.shape {vec.shape}, expected (B, N, 18)"
    assert np.all(np.isfinite(vec)), "vec has non-finite values"
    assert np.all(vec >= -1.1) and np.all(vec <= 1.1), (
        f"vec outside [-1.1, 1.1]: min={vec.min():.4f} max={vec.max():.4f}"
    )
    print("[test-vec-schema] GPU core get_obs() vec: dtype=float32, shape=(B,N,18), finite, in bounds. OK.")


def _agents_suffix(n_agents: int) -> str:
    """Return agent-size suffix for filenames: 2v2, 4v4, 8v8, or NvN."""
    n = max(1, min(int(n_agents), 16))
    return f"{n}v{n}"


def _ensure_run_tag_has_agent_suffix(run_tag: str, n_agents: int) -> str:
    """Ensure run_tag ends with _2v2, _4v4, _8v8 (or _NvN) so saves/snapshots are distinct per agent size."""
    suffix = _agents_suffix(n_agents)
    tag_suffix = f"_{suffix}"
    # Strip any existing agent suffix so we don't get ppo_league_4v4_2v2
    for existing in ("_2v2", "_4v4", "_8v8"):
        if run_tag.endswith(existing):
            run_tag = run_tag[: -len(existing)]
            break
    if not run_tag.endswith(tag_suffix):
        run_tag = run_tag.rstrip("_") + tag_suffix
    return run_tag


def _normalize_train_mode(mode: str) -> str:
    """Accept friendly CLI aliases and map them to the internal canonical mode names."""
    raw = str(mode).upper().strip()
    aliases = {
        "LEAGUE": TrainMode.CURRICULUM_LEAGUE.value,
        "CURRICULUM_LEAGUE": TrainMode.CURRICULUM_LEAGUE.value,
        "PAPER": TrainMode.CURRICULUM_NO_LEAGUE.value,
        "NO_LEAGUE": TrainMode.CURRICULUM_NO_LEAGUE.value,
        "CURRICULUM_NO_LEAGUE": TrainMode.CURRICULUM_NO_LEAGUE.value,
        "FIXED": TrainMode.FIXED_OPPONENT.value,
        "FIXED_OPPONENT": TrainMode.FIXED_OPPONENT.value,
        "SELFPLAY": TrainMode.SELF_PLAY.value,
        "SELF-PLAY": TrainMode.SELF_PLAY.value,
        "SELF_PLAY": TrainMode.SELF_PLAY.value,
    }
    return aliases.get(raw, raw)


def _default_run_tag_for_mode(mode: str, fixed_opponent_tag: str = "OP3", n_agents: int = 2) -> str:
    """Return a unique default run_tag per mode and agent size so runs don't overwrite each other."""
    m = _normalize_train_mode(mode)
    suffix = _agents_suffix(n_agents)
    if m == TrainMode.CURRICULUM_LEAGUE.value:
        return f"ppo_league_{suffix}"
    if m == TrainMode.CURRICULUM_NO_LEAGUE.value:
        return f"ppo_paper_{suffix}"
    if m == TrainMode.FIXED_OPPONENT.value:
        return f"ppo_fixed_{fixed_opponent_tag.lower()}_{suffix}"
    if m == TrainMode.SELF_PLAY.value:
        return f"ppo_self_play_{suffix}"
    return f"ppo_run_{suffix}"


if __name__ == "__main__":
    import argparse
    import sys
    if "--verify-4v4" in sys.argv:
        run_verify_4v4(num_episodes=10)
    elif "--test-vec-schema" in sys.argv:
        run_test_vec_schema()
    else:
        parser = argparse.ArgumentParser(description="Train PPO (CTF)")
        parser.add_argument("--mode", type=str, default=None,
                            help="Train mode: CURRICULUM_LEAGUE (League), CURRICULUM_NO_LEAGUE (Paper=curriculum no league), FIXED_OPPONENT, SELF_PLAY")
        parser.add_argument("--run-tag", type=str, default=None,
                            help="Run name for checkpoints (default: unique per mode)")
        parser.add_argument("--total-steps", type=int, default=None, help="Total timesteps")
        parser.add_argument("--checkpoint-dir", type=str, default=None, help="Directory for checkpoints/snapshots (e.g. /content/drive/MyDrive/ppo_checkpoints)")
        parser.add_argument("--load", type=str, default=None, help="Optional path to a .zip checkpoint to resume from")
        parser.add_argument("--fixed-opponent", type=str, default="OP3", help="For FIXED_OPPONENT mode (e.g. OP1, OP2, OP3)")
        parser.add_argument(
            "--agents",
            type=int,
            default=None,
            choices=[2, 4, 6, 8],
            help="Team size: 2=2v2, 4=4v4, 6=6v6, 8=8v8 (sets --max-blue-agents)",
        )
        parser.add_argument("--max-blue-agents", type=int, default=None, help="Agents per team (1-16). Use 2/4/8 for 2v2/4v4/8v8; overrides --agents if set.")
        parser.add_argument("--gpu-native-env", action="store_true", help="Deprecated flag; training always uses game_field_gpu.")
        parser.add_argument("--test-kl-zero-lr", action="store_true", help="Set lr=0 to verify approx_kl ~ 0 (sanity check for logprob/action plumbing)")
        parser.add_argument("--verbose-training", action="store_true", help="Print each episode result and debug logs (slower; default is quiet for speed)")
        parser.add_argument("--device", type=str, default=None, help="Device for env and PPO: cuda, cuda:0, or cpu. Default: cuda if available else cpu.")
        args = parser.parse_args()
        cfg = PPOConfig()
        if args.mode is not None:
            cfg.mode = _normalize_train_mode(args.mode)
        if args.max_blue_agents is not None:
            n = max(1, min(int(args.max_blue_agents), 16))
            if n != int(args.max_blue_agents):
                print(f"[PPO] --max-blue-agents {args.max_blue_agents} out of range; clamped to {n} (max 16).")
            cfg.max_blue_agents = n
        elif getattr(args, "agents", None) is not None:
            cfg.max_blue_agents = int(args.agents)
        if args.mode is not None:
            if args.run_tag is not None:
                cfg.run_tag = args.run_tag
            else:
                cfg.run_tag = _default_run_tag_for_mode(cfg.mode, args.fixed_opponent, cfg.max_blue_agents)
        cfg.run_tag = _ensure_run_tag_has_agent_suffix(cfg.run_tag, cfg.max_blue_agents)
        # Separate checkpoint dir per team size. On Colab, save to Drive so runs persist (no 15h loss on disconnect).
        n_agents = cfg.max_blue_agents
        suffix = _agents_suffix(n_agents)
        if os.path.exists("/content/drive/MyDrive"):
            # Colab with Drive mounted: all runs save to Drive
            base = "/content/drive/MyDrive/CTF_models"
            cfg.checkpoint_dir = os.path.join(base, suffix)
        else:
            # Local PC: save under project (checkpoints_sb3/2v2, 3v3, 4v4)
            cfg.checkpoint_dir = os.path.join("checkpoints_sb3", suffix)
        if args.total_steps is not None:
            cfg.total_timesteps = args.total_steps
        else:
            # Default total timesteps (tuned for this project):
            # All team sizes (2v2, 3v3, 4v4, 8v8) use 1.0M steps by default.
            cfg.total_timesteps = 1_000_000
        if getattr(args, "checkpoint_dir", None) is not None:
            cfg.checkpoint_dir = args.checkpoint_dir
        if getattr(args, "fixed_opponent", None) is not None and cfg.mode == TrainMode.FIXED_OPPONENT.value:
            cfg.fixed_opponent_tag = args.fixed_opponent.upper()
        if getattr(args, "test_kl_zero_lr", False):
            cfg.test_kl_zero_lr = True
        if getattr(args, "verbose_training", False):
            cfg.verbose_training = True
        if getattr(args, "load", None) is not None:
            cfg.load_path = args.load
        if getattr(args, "device", None) is not None:
            cfg.device = str(args.device).strip().lower()
        else:
            # Prefer CUDA when available (default pip torch is sometimes CPU-only; user may have installed cu118/cu121)
            if torch.cuda.is_available():
                cfg.device = "cuda"
        cfg.gpu_native_env = True  # All training uses game_field_gpu
        train_ppo(cfg)
