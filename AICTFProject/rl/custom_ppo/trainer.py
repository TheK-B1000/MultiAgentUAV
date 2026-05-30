from __future__ import annotations

import csv
import math
import os
import sys
import warnings
from collections import deque
from dataclasses import asdict
from functools import partial
from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from macro_actions import MacroAction
from rl.curriculum import phase_from_tag
from rl.discrete_mi import discrete_mi_plugin
from rl.global_state import (
    GLOBAL_STATE_DIM,
    GLOBAL_STATE_FLAG_TERRITORY_SLICE,
    coarse_game_phase_from_global_state,
)
from rl.behavior_telemetry import (
    BEHAVIOR_TELEMETRY_NAMES,
    N_ATTACK_DEFENSE_RATIO_BUCKET,
    N_ROLE_BUCKET_MI,
    N_TELEMETRY,
    bucket_ids_from_telemetry,
    compute_behavior_telemetry_batch,
)
from rl.latent_phase_labels import (
    TEAM_PHASES,
    outcome_id_from_global_state,
    outcome_label_from_global_state,
    team_phase_id_from_global_state,
    team_phase_label_from_global_state,
)
from rl.latent_marl import StrategyEncoder, paper_strategy_switch_indicator, expected_strategy_switch_penalty, TemporalStateTracker, CONTEXT_STATE_DIM
from rl.latent_losses import (
    strategy_aux_return_loss as _latent_strategy_aux_return_loss,
    strategy_entropy_loss as _latent_strategy_entropy_loss,
    strategy_kl_consecutive_loss as _latent_strategy_kl_consecutive_loss,
    strategy_persistence_loss as _latent_strategy_persistence_loss,
    strategy_phase_aux_loss as _latent_strategy_phase_aux_loss,
    strategy_ppo_loss as _latent_strategy_ppo_loss,
)
from rl.networks import CNNEncoder, CentralizedCritic
from rl.ppo_core import (
    TensorDictRolloutBuffer,
    align_next_values_to_rollout_actions,
    ppo_policy_loss,
    ppo_value_loss,
)

from rl.custom_ppo.policy import SharedActorCentralizedCritic
from rl.custom_ppo.inference import (
    _torch_load_checkpoint,
    _assert_compatible_global_state_dim,
    apply_deterministic_sampling_generators,
    _remap_legacy_strategy_aux_head_state_dict,
    _load_model_state_dict_compat,
    _model_kwargs_from_cfg,
    CUSTOM_PPO_FORMAT,
    CUSTOM_PPO_LATENT_FORMAT,
    CUSTOM_PPO_ACTOR_ARCH,
    CUSTOM_PPO_VEC_SCHEMA_VERSION,
)
from rl.custom_ppo.csv_writers import (
    _write_csv_row,
    _ensure_additive_csv_header,
    _episode_fieldnames,
    _update_fieldnames,
    _strategy_experience_fieldnames,
    _opponent_id_int_from_info,
    _opponent_id_csv_from_info,
    _opponent_legend,
    E3_STEP_TELEMETRY_FIELDS,
    _METRICS_CSV_LEGACY_COLUMN_FILL,
    SCRIPTED_OPPONENT_MI_COUNT,
)
from rl.custom_ppo.latent_diagnostics import (
    _latent_rollout_stats,
    _latent_opponent_rollout_diag,
    _behavior_diversity_stats,
    _forced_z_behavior_profile,
    _strategy_resample_advantage_stats,
    _rollout_advantage_diagnostics,
    _strategy_experience_bucket_ids,
    _write_strategy_experience_table,
    _latent_option_advantage_stats,
)
from rl.custom_ppo.curriculum_runtime import (
    _set_curriculum_opponent,
    _update_curriculum_after_episode,
    _hook_sample_training_opponent_before_reset,
)
from rl.custom_ppo.return_normalization import (
    _return_norm_std,
    _normalize_value_targets,
    _denormalize_values,
    _update_return_norm_stats,
    _update_strategy_return_stats,
    _normalize_strategy_returns,
)
from rl.custom_ppo.option_returns import compute_option_returns
from rl.custom_ppo.trainer_audit import log_decentralized_actor_contract_once
from rl.custom_ppo.training_telemetry import TrainingTelemetry


def _tqdm_for_sb3_progress() -> Any:
    """Match Stable-Baselines3 ``ProgressBarCallback``: prefer ``tqdm.rich.tqdm`` when available."""
    try:
        from tqdm import TqdmExperimentalWarning

        warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)
    except Exception:
        pass
    try:
        from tqdm.rich import tqdm  # type: ignore[import-not-found]
    except ImportError:
        from tqdm import tqdm  # type: ignore[import-not-found]
    return tqdm


def _open_sb3_style_progress(
    cfg: Any,
    *,
    total_timesteps: int,
    current_num_timesteps: int,
) -> Any:
    if not bool(getattr(cfg, "enable_progress_bar", True)):
        return None
    rem = int(total_timesteps) - int(current_num_timesteps)
    if rem <= 0:
        return None
    try:
        tqdm = _tqdm_for_sb3_progress()
    except ImportError:
        print(
            "[PPO] Install tqdm and rich for the SB3-style bar:  pip install tqdm rich",
            file=sys.stderr,
        )
        return None
    return tqdm(
        total=rem,
        dynamic_ncols=True,
        file=sys.stderr,
        mininterval=0.2,
    )


def _compose_training_reward_components(
    reward_component: dict[str, torch.Tensor],
    *,
    dense_weight: float,
    reward_scale: float,
    reward_clip: float,
    shaping_coef: float,
    stalemate: Optional[torch.Tensor] = None,
    stalemate_penalty: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Mirror GPU reward scaling for PPO targets after optional shaping decay."""
    out = dict(reward_component)
    coef = float(shaping_coef)
    if abs(coef - 1.0) > 1e-9:
        out["reward_offense"] = out["reward_offense"] * coef
        out["reward_pbrs"] = out["reward_pbrs"] * coef
        out["reward_team"] = out["reward_team"] * coef

    dense = out["reward_pbrs"] + out["reward_team"]
    raw = (
        out["reward_terminal"]
        + out["reward_sparse"]
        + out["reward_failure"]
        + out["reward_offense"]
        + float(dense_weight) * dense
    )
    if stalemate is not None:
        raw = raw + torch.where(
            stalemate.bool(),
            torch.full_like(raw, float(stalemate_penalty)),
            torch.zeros_like(raw),
        )
    scaled = torch.tanh(raw / max(1e-6, float(reward_scale)))
    out["reward_total"] = torch.clamp(scaled, -float(reward_clip), float(reward_clip))
    return out


class EpisodeStrategyRecorder:
    """Tracks sampled episode-level z actions for task-return PPO credit.

    This is not an auxiliary semantic task and it does not assign labels to z.
    It only preserves the exact sampled strategy action and old log-prob needed
    to credit q_phi from completed episode return.
    """

    def __init__(self) -> None:
        self.pending: dict[int, dict[str, Any]] = {}
        self.completed: list[dict[str, Any]] = []

    def reset(self) -> None:
        self.pending.clear()
        self.completed.clear()

    def clear_completed(self) -> None:
        self.completed.clear()

    def record_start(
        self,
        *,
        env_index: int,
        episode_id: int,
        global_state_0: torch.Tensor,
        z: torch.Tensor,
        z_logprob_old: torch.Tensor,
        bucket_id: int,
        q_phi_probs: Iterable[float],
    ) -> None:
        self.pending[int(env_index)] = {
            "episode_id": int(episode_id),
            "global_state_0": global_state_0.detach().clone(),
            "z": int(z.detach().cpu().item()),
            "z_logprob_old": float(z_logprob_old.detach().cpu().item()),
            "episode_return": None,
            "episode_win": None,
            "bucket_id": int(bucket_id),
            "q_phi_probs": [float(x) for x in q_phi_probs],
        }

    def record_outcome(
        self,
        *,
        env_index: int,
        episode_return: float,
        episode_win: int,
    ) -> Optional[dict[str, Any]]:
        record = self.pending.pop(int(env_index), None)
        if record is None:
            return None
        record["episode_return"] = float(episode_return)
        record["episode_win"] = int(episode_win)
        self.completed.append(record)
        return record


class CustomPPOTrainer:
    """Small PPO trainer that owns rollout, GAE, and update math locally."""

    def __init__(
        self,
        env,
        cfg,
        *,
        learning_rate: float,
        clip_range: float,
        ent_coef: float,
        n_epochs: int,
        batch_size: int,
        value_clip_range: Optional[float] = None,
        curriculum: Optional[Any] = None,
    ) -> None:
        self.env = env
        self.cfg = cfg
        self.curriculum = curriculum
        self.device = torch.device(str(cfg.device))
        self.use_latent_strategy = bool(getattr(cfg, "use_latent_strategy", False))
        self.latent_k = int(getattr(cfg, "latent_k", 4)) if self.use_latent_strategy else 0
        self.latent_resample_every_n = max(0, int(getattr(cfg, "latent_resample_every_n", 0) or 0))
        self.fixed_latent_strategy = self.use_latent_strategy and bool(
            getattr(cfg, "fixed_latent_strategy", False)
        )
        self.latent_gae_reset_on_z_change = bool(
            getattr(cfg, "latent_gae_reset_on_z_change", True)
        ) and (self.use_latent_strategy and not self.fixed_latent_strategy)
        self.latent_bootstrap_z_deterministic = bool(getattr(cfg, "latent_bootstrap_z_deterministic", True))
        self.fixed_latent_strategy_id = (
            max(0, min(int(getattr(cfg, "fixed_latent_strategy_id", 0) or 0), self.latent_k - 1))
            if self.use_latent_strategy
            else 0
        )
        model_kwargs: dict[str, Any] = {
            "actor_cnn_feature_dim": int(getattr(cfg, "actor_cnn_feature_dim", 128)),
        }
        if self.use_latent_strategy:
            model_kwargs.update(
                {
                    "latent_k": self.latent_k,
                    "z_embed_dim": int(getattr(cfg, "latent_z_embed_dim", 16)),
                    "strategy_hidden_dim": int(getattr(cfg, "latent_strategy_hidden", 128)),
                    "critic_hidden_dim": int(getattr(cfg, "latent_vf_hidden", 128)),
                    # Canonical attribute access. Legacy CLI / cfg-dict keys
                    # (``latent_strategy_q_head``) are folded into the
                    # canonical name at the load boundary (CLI parsing for
                    # PPOConfig, inference.canonicalize_latent_strategy_cfg
                    # for checkpoint dicts), so the trainer reads one name.
                    "use_strategy_aux_return_head": bool(
                        getattr(cfg, "latent_strategy_aux_return_head", False)
                    ),
                    "use_episode_strategy_value_head": bool(getattr(cfg, "latent_episode_strategy_ppo", False)),
                    "strategy_tau": max(1e-3, float(getattr(cfg, "latent_strategy_tau", 1.0) or 1.0)),
                }
            )
        self.model = SharedActorCentralizedCritic(env.observation_space, env.action_space, **model_kwargs).to(self.device)
        apply_deterministic_sampling_generators(
            self.model, int(getattr(cfg, "seed", 0) or 0), device=self.device
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(learning_rate), eps=1e-5)
        self.base_learning_rate = float(learning_rate)
        self.clip_range = float(clip_range)
        self.ent_coef = float(ent_coef)
        self.vf_coef = max(0.0, float(getattr(cfg, "vf_coef", 1.0) or 0.0))
        self.n_epochs = int(n_epochs)
        self.batch_size = int(batch_size)
        self.value_clip_range = None if value_clip_range is None else float(value_clip_range)
        self.normalize_returns = bool(getattr(cfg, "normalize_returns", False))
        self._return_norm_mean = 0.0
        self._return_norm_var = 1.0
        self._return_norm_count = 1e-4
        self.latent_strategy_ppo_coef = max(0.0, float(getattr(cfg, "latent_strategy_ppo_coef", 0.1) or 0.0))
        self.latent_episode_strategy_ppo = (
            self.use_latent_strategy
            and not self.fixed_latent_strategy
            and bool(getattr(cfg, "latent_episode_strategy_ppo", False))
        )
        self.latent_episode_strategy_coef = max(
            0.0, float(getattr(cfg, "latent_episode_strategy_coef", 0.0) or 0.0)
        )
        self.latent_episode_strategy_clip_eps = max(
            1e-6, float(getattr(cfg, "latent_episode_strategy_clip_eps", 0.2) or 0.2)
        )
        self.latent_episode_strategy_value_coef = max(
            0.0, float(getattr(cfg, "latent_episode_strategy_value_coef", 0.5) or 0.0)
        )
        self.latent_episode_strategy_return_norm = bool(
            getattr(cfg, "latent_episode_strategy_return_norm", True)
        )
        # Canonical attribute access only — legacy ``latent_strategy_q_*`` keys
        # are folded at the config-load boundary, see ``rl.custom_ppo.inference
        # .canonicalize_latent_strategy_cfg`` and the CLI argparse handler.
        self.latent_strategy_aux_return_coef = max(
            0.0, float(getattr(cfg, "latent_strategy_aux_return_coef", 0.0) or 0.0)
        )
        self.latent_strategy_aux_return_head = (
            self.use_latent_strategy
            and bool(getattr(cfg, "latent_strategy_aux_return_head", False))
        )
        self.latent_strategy_aux_predict_phase_coef = max(0.0, float(getattr(cfg, "latent_strategy_aux_predict_phase_coef", 0.0) or 0.0))
        self.reward_shaping_coef_start = float(getattr(cfg, "reward_shaping_coef_start", 1.0) or 1.0)
        self.reward_shaping_coef_end = float(getattr(cfg, "reward_shaping_coef_end", self.reward_shaping_coef_start))
        self.reward_shaping_decay_steps = max(0, int(getattr(cfg, "reward_shaping_decay_steps", 0) or 0))
        env_cfg = getattr(env, "cfg", None)
        self.reward_dense_weight = max(0.0, float(getattr(env_cfg, "dense_weight", 1.0) or 0.0))
        self.reward_scale = max(1e-6, float(getattr(env_cfg, "reward_scale", 1.0) or 1.0))
        self.reward_clip = max(1e-6, float(getattr(env_cfg, "reward_clip", 1.0) or 1.0))
        self.reward_stalemate_penalty = float(getattr(env_cfg, "stalemate_penalty", 0.0) or 0.0)
        self.periodic_checkpoint_steps = max(0, int(getattr(cfg, "periodic_checkpoint_steps", 0) or 0))
        self._next_periodic_checkpoint_step = (
            self.periodic_checkpoint_steps if self.periodic_checkpoint_steps > 0 else 0
        )
        self._strategy_return_mean = 0.0
        self._strategy_return_var = 1.0
        self._strategy_return_count = 1e-4
        self.global_step = 0
        self.last_stats: dict[str, float] = {}
        self.run_id = str(getattr(cfg, "run_id", "") or "")
        self.run_pid = int(getattr(cfg, "run_pid", os.getpid()) or os.getpid())
        self._updates_completed = 0
        self._ep_wins = 0
        self._ep_losses = 0
        self._ep_draws = 0
        self._episodes_completed = 0
        self._rollout_episode_records: list[dict[str, Any]] = []
        self._recent_episode_successes = deque(maxlen=200)
        self.metrics_csv_path = str(getattr(cfg, "metrics_csv_path", "") or "")
        self.episode_csv_path = str(getattr(cfg, "episode_csv_path", "") or "")
        self.strategy_experience_csv_path = str(getattr(cfg, "strategy_experience_csv_path", "") or "")
        n_envs = int(env.num_envs)
        strategy_prob_width = max(1, int(self.latent_k))
        self._episode_return_accum = torch.zeros((n_envs,), dtype=torch.float32, device=self.device)
        self._episode_strategy_state = torch.zeros(
            (n_envs, int(self.model.global_state_dim)), dtype=torch.float32, device=self.device
        )
        self._episode_strategy_z = torch.zeros((n_envs,), dtype=torch.long, device=self.device)
        self._episode_strategy_log_prob = torch.zeros((n_envs,), dtype=torch.float32, device=self.device)
        self._episode_strategy_probs = torch.zeros(
            (n_envs, strategy_prob_width), dtype=torch.float32, device=self.device
        )
        self._episode_strategy_bucket = torch.zeros((n_envs,), dtype=torch.long, device=self.device)
        self._episode_strategy_has_start = torch.zeros((n_envs,), dtype=torch.bool, device=self.device)
        self._rollout_strategy_episode_records: list[dict[str, Any]] = []
        self.episode_strategy_recorder = EpisodeStrategyRecorder()
        self._next_strategy_episode_id = 0
        self.telemetry = TrainingTelemetry(self)
        self._sb3_rollout_pbar = None
        self._last_obs = None
        self._last_global_state = None
        self._last_context_state = None
        self._current_z = None
        self._strategy_age = torch.zeros((int(env.num_envs),), dtype=torch.long, device=self.device)
        self._needs_strategy_sample = torch.ones((int(env.num_envs),), dtype=torch.bool, device=self.device)
        if self.use_latent_strategy:
            self.temporal_tracker = TemporalStateTracker(
                num_envs=int(env.num_envs),
                state_dim=GLOBAL_STATE_DIM,
                device=self.device,
            )
        else:
            self.temporal_tracker = None
        self.latent_resample_on_flag = (
            bool(getattr(cfg, "latent_resample_on_flag", False))
            and self.use_latent_strategy
            and not self.fixed_latent_strategy
        )
        self.latent_kl_consecutive = (
            max(0.0, float(getattr(cfg, "latent_kl_consecutive", 0.0) or 0.0))
            if self.use_latent_strategy and not self.fixed_latent_strategy
            else 0.0
        )
        self._z_kl_first_in_ep = None
        self._prev_z_logits = None
        self._decentralized_actor_contract_logged = False
        mode_s = str(getattr(cfg, "mode", "") or "").strip().upper()
        self._opponent_randomize_training = (
            (mode_s == "OPPONENT_POOL" or bool(getattr(cfg, "opponent_randomize", False)))
            and curriculum is None
        )
        self._opponent_pool_tags = (
            [str(x).strip().upper() for x in getattr(cfg, "opponent_pool", ())] if self._opponent_randomize_training else []
        )
        self._rng_opponent = np.random.default_rng(int(getattr(cfg, "seed", 0)) + 901)
        if self._opponent_randomize_training:
            if not self._opponent_pool_tags:
                raise ValueError(
                    "Opponent pool training (mode=OPPONENT_POOL or opponent_randomize) requires a non-empty "
                    "opponent_pool (e.g. OP1–OP3, OP5–OP7; OP4 optional with --allow-op4-in-training-pool)."
                )
            self.env._before_reset_indices_hook = partial(_hook_sample_training_opponent_before_reset, self)

    def _reward_shaping_coef(self) -> float:
        if self.reward_shaping_decay_steps <= 0:
            return float(self.reward_shaping_coef_start)
        frac = min(1.0, max(0.0, float(self.global_step) / float(self.reward_shaping_decay_steps)))
        return float(self.reward_shaping_coef_start + frac * (self.reward_shaping_coef_end - self.reward_shaping_coef_start))

    @staticmethod
    def _flag_territory_features_changed(
        pre: torch.Tensor, post: torch.Tensor, *, eps: float = 1e-4
    ) -> torch.Tensor:
        """(B, 4) pre/post flag-sector slice; return (B,) bool: min distances or capture flags changed."""
        d0 = (pre[:, 0:2] - post[:, 0:2]).abs() > float(eps)
        ch_float = d0.any(dim=-1)
        ch_cap = (pre[:, 2:4] - post[:, 2:4]).abs() > 0.5
        ch_capt = ch_cap.any(dim=-1)
        return ch_float | ch_capt

    def _on_episode_done(
        self,
        info: dict[str, Any],
        *,
        timestep: Optional[int] = None,
        rollout_step: Optional[int] = None,
        latent_z: Optional[int] = None,
        env_index: Optional[int] = None,
    ) -> None:
        er = info.get("episode_result")
        if isinstance(er, dict):
            bs = int(er.get("blue_score", 0))
            rs = int(er.get("red_score", 0))
        else:
            bs = int(info.get("blue_score", 0))
            rs = int(info.get("red_score", 0))
        success = 1 if bs > rs else 0
        if bs > rs:
            self._ep_wins += 1
        elif bs < rs:
            self._ep_losses += 1
        else:
            self._ep_draws += 1
        self._episodes_completed += 1
        self._rollout_episode_records.append(
            {
                "blue_score": int(bs),
                "red_score": int(rs),
                "win_margin": int(bs) - int(rs),
                "success": success,
                "latent_z": latent_z,
                "opponent_id": int(_opponent_id_int_from_info(self.cfg, info)),
            }
        )
        self._recent_episode_successes.append(success)
        self.telemetry.write_episode_metrics(
            info,
            blue_score=bs,
            red_score=rs,
            timestep=int(timestep or self.global_step),
            rollout_step=rollout_step,
            latent_z=latent_z,
        )
        _update_curriculum_after_episode(self, info=info, blue_score=bs, red_score=rs, env_index=env_index)
        every = int(getattr(self.cfg, "episode_log_every", 0) or 0)
        if every > 0 and self._episodes_completed % every == 0:
            self.telemetry.print_episode_progress(info)

    def _record_episode_strategy_outcome(
        self,
        env_index: int,
        info: dict[str, Any],
        *,
        episode_return: float,
    ) -> None:
        if not self.latent_episode_strategy_ppo:
            return
        env_i = int(env_index)
        if env_i < 0 or env_i >= int(self._episode_strategy_has_start.numel()):
            return
        if not bool(self._episode_strategy_has_start[env_i].detach().cpu().item()):
            return
        er = info.get("episode_result") if isinstance(info.get("episode_result"), dict) else {}
        bs = int(er.get("blue_score", info.get("blue_score", 0)) or 0)
        rs = int(er.get("red_score", info.get("red_score", 0)) or 0)
        episode_win = 1 if bs > rs else 0
        probs = self._episode_strategy_probs[env_i, : self.latent_k].detach().cpu().tolist()
        self._rollout_strategy_episode_records.append(
            {
                "episode_id": int(self._episodes_completed),
                "global_state_0": self._episode_strategy_state[env_i].detach().clone(),
                "z": int(self._episode_strategy_z[env_i].detach().cpu().item()),
                "z_logprob_old": float(self._episode_strategy_log_prob[env_i].detach().cpu().item()),
                "episode_return": float(episode_return),
                "episode_win": 1 if bs > rs else 0,
                "bucket_id": int(self._episode_strategy_bucket[env_i].detach().cpu().item()),
                "q_phi_probs": [float(x) for x in probs],
            }
        )

    def _tensor_obs(self, obs: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        return {
            "grid": torch.as_tensor(obs["grid"], dtype=torch.float32, device=self.device),
            "vec": torch.as_tensor(obs["vec"], dtype=torch.float32, device=self.device),
            "agent_mask": torch.as_tensor(obs["agent_mask"], dtype=torch.float32, device=self.device),
            "mask": torch.as_tensor(obs["mask"], dtype=torch.float32, device=self.device),
        }

    def _reset_strategy_state(self) -> None:
        if not self.use_latent_strategy:
            return
        n_envs = int(self.env.num_envs)
        z0 = self.fixed_latent_strategy_id if self.fixed_latent_strategy else 0
        self._current_z = torch.full((n_envs,), int(z0), dtype=torch.long, device=self.device)
        self._strategy_age = torch.zeros((n_envs,), dtype=torch.long, device=self.device)
        self._needs_strategy_sample = torch.full(
            (n_envs,), not self.fixed_latent_strategy, dtype=torch.bool, device=self.device
        )
        if self.latent_kl_consecutive > 0.0:
            self._z_kl_first_in_ep = torch.ones((n_envs,), dtype=torch.bool, device=self.device)
            self._prev_z_logits = None
        else:
            self._z_kl_first_in_ep = None
            self._prev_z_logits = None
        if self.temporal_tracker is not None:
            self.temporal_tracker.reset()
        self._last_context_state = None
        if hasattr(self, "_episode_return_accum"):
            self._episode_return_accum.zero_()
            self._episode_strategy_has_start.zero_()
        if hasattr(self, "episode_strategy_recorder"):
            self.episode_strategy_recorder.reset()

    def _store_episode_strategy_start(
        self,
        *,
        start_mask: torch.Tensor,
        global_state: torch.Tensor,
        z_idx: torch.Tensor,
        z_log_prob: torch.Tensor,
        z_logits: torch.Tensor,
    ) -> None:
        if not self.latent_episode_strategy_ppo or not bool(start_mask.any().item()):
            return
        idx = torch.where(start_mask)[0]
        probs = torch.softmax(z_logits.detach(), dim=-1)
        buckets = _strategy_experience_bucket_ids(global_state.index_select(0, idx)).detach()
        self._episode_strategy_state[idx] = global_state.index_select(0, idx).detach()
        self._episode_strategy_z[idx] = z_idx.index_select(0, idx).detach()
        self._episode_strategy_log_prob[idx] = z_log_prob.index_select(0, idx).detach()
        self._episode_strategy_probs[idx, : self.latent_k] = probs.index_select(0, idx)
        self._episode_strategy_bucket[idx] = buckets
        self._episode_strategy_has_start[idx] = True
        for row_i, env_i in enumerate(idx.detach().cpu().tolist()):
            self.episode_strategy_recorder.record_start(
                env_index=int(env_i),
                episode_id=int(self._next_strategy_episode_id),
                global_state_0=global_state[int(env_i)],
                z=z_idx[int(env_i)],
                z_logprob_old=z_log_prob[int(env_i)],
                bucket_id=int(buckets[row_i].detach().cpu().item()),
                q_phi_probs=probs[int(env_i), : self.latent_k].detach().cpu().tolist(),
            )
            self._next_strategy_episode_id += 1

    def _strategy_for_step(
        self,
        global_state: torch.Tensor,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], dict[str, torch.Tensor]]:
        """Return current sparse strategy and sampling metadata for one rollout step."""
        if not self.use_latent_strategy:
            return None, None, {}
        if self._current_z is None:
            self._reset_strategy_state()
        assert self._current_z is not None

        if self.fixed_latent_strategy:
            batch = int(global_state.shape[0])
            z_idx = torch.full(
                (batch,), self.fixed_latent_strategy_id, dtype=torch.long, device=self.device
            )
            prev_z = self._current_z.clone()
            self._current_z = z_idx.clone()
            fixed_logits = torch.full(
                (batch, self.latent_k), -1.0e8, dtype=torch.float32, device=self.device
            )
            fixed_logits[:, self.fixed_latent_strategy_id] = 0.0
            false_mask = torch.zeros((batch,), dtype=torch.bool, device=self.device)
            aux = {
                "z": z_idx,
                "prev_z": prev_z,
                "z_log_prob": torch.zeros((batch,), dtype=torch.float32, device=self.device),
                "z_entropy": torch.zeros((batch,), dtype=torch.float32, device=self.device),
                "z_logits": fixed_logits,
                "z_resampled": false_mask,
                "z_persist_mask": false_mask,
            }
            return z_idx, prev_z, aux

        episode_start_mask = self._needs_strategy_sample.clone()
        resample_mask = episode_start_mask.clone()
        if self.latent_resample_every_n > 0:
            resample_mask |= self._strategy_age >= self.latent_resample_every_n

        prev_z = self._current_z.clone()
        z_idx = self._current_z.clone()
        persist_mask = resample_mask & (~self._needs_strategy_sample)

        if bool(resample_mask.any().item()):
            idx = torch.where(resample_mask)[0]
            sampled_z, _, _, _ = self.model.sample_strategy(
                global_state.index_select(0, idx),
                deterministic=False,
            )
            z_idx[idx] = sampled_z
            self._current_z = z_idx.clone()
            self._strategy_age[idx] = 0
            self._needs_strategy_sample[idx] = False

        z_logits = self.model.strategy_logits(global_state)
        z_dist = Categorical(logits=z_logits)
        z_log_prob = z_dist.log_prob(z_idx)
        z_entropy = z_dist.entropy()
        self._store_episode_strategy_start(
            start_mask=episode_start_mask,
            global_state=global_state,
            z_idx=z_idx,
            z_log_prob=z_log_prob,
            z_logits=z_logits,
        )

        aux = {
            "z": z_idx,
            "prev_z": prev_z,
            "z_log_prob": z_log_prob,
            "z_entropy": z_entropy,
            "z_logits": z_logits,
            "z_resampled": resample_mask,
            "z_persist_mask": persist_mask,
        }
        return z_idx, prev_z, aux

    def _on_sb3_rollout_env_step(self) -> None:
        p = self._sb3_rollout_pbar
        if p is None:
            return
        nenv = int(self.env.num_envs)
        try:
            rest = int(p.total) - int(p.n)  # type: ignore[attr-defined]
        except Exception:
            p.update(nenv)  # type: ignore[call-arg]
            return
        p.update(int(min(nenv, max(0, rest))))  # type: ignore[call-arg]

    def _mark_strategy_step_done(self, dones: np.ndarray) -> None:
        if not self.use_latent_strategy:
            return
        done_t = torch.as_tensor(dones, dtype=torch.bool, device=self.device)
        self._strategy_age += 1
        if bool(done_t.any().item()):
            self._strategy_age[done_t] = 0
            self._needs_strategy_sample[done_t] = not self.fixed_latent_strategy

    def _obs_rows_from_next(
        self,
        next_obs: Dict[str, np.ndarray],
        infos: list[dict],
    ) -> Dict[str, np.ndarray]:
        rows: dict[str, list[np.ndarray]] = {key: [] for key in ("grid", "vec", "agent_mask", "mask")}
        for env_i, info in enumerate(infos):
            use_terminal = bool(info.get("truncated", False)) and isinstance(info.get("terminal_observation"), dict)
            terminal_obs = info.get("terminal_observation") if use_terminal else {}
            for key in rows:
                source = terminal_obs.get(key, next_obs[key][env_i]) if isinstance(terminal_obs, dict) else next_obs[key][env_i]
                rows[key].append(np.asarray(source, dtype=np.float32))
        return {key: np.stack(values, axis=0) for key, values in rows.items()}

    def _strategy_encoder_grad_norm(self) -> float:
        """Return the current q_phi gradient norm before global clipping."""
        strategy_module = getattr(self.model, "strategy_aux_return_head", None) or getattr(
            self.model, "strategy_encoder", None
        )
        if strategy_module is None:
            return 0.0
        total = torch.zeros((), dtype=torch.float32, device=self.device)
        for param in strategy_module.parameters():
            if param.grad is None:
                continue
            grad = param.grad.detach().float()
            total = total + grad.pow(2).sum()
        return float(torch.sqrt(total).detach().cpu().item())

    def _empty_latent_episode_strategy_stats(self) -> dict[str, float]:
        return {
            "latent_episode_pg_loss": 0.0,
            "latent_episode_v_loss": 0.0,
            "latent_episode_entropy": 0.0,
            "latent_episode_adv_mean": 0.0,
            "latent_episode_adv_std": 0.0,
            "latent_episode_return_mean": 0.0,
            "latent_episode_return_std": 0.0,
            "latent_episode_ratio_mean": 0.0,
            "latent_episode_ratio_max": 0.0,
            "latent_episode_ratio_min": 0.0,
            "latent_episode_approx_kl": 0.0,
            "latent_episode_clip_fraction": 0.0,
            "latent_episode_count": 0.0,
            "latent_episode_loss": 0.0,
        }

    def _episode_strategy_training_batch(self) -> Optional[dict[str, torch.Tensor]]:
        if (
            not self.latent_episode_strategy_ppo
            or self.fixed_latent_strategy
            or self.model.episode_strategy_value_head is None
        ):
            return None
        records = list(self._rollout_strategy_episode_records)
        if not records:
            return None
        states = torch.stack([r["global_state_0"].detach().float() for r in records], dim=0).to(self.device)
        z = torch.as_tensor([int(r["z"]) for r in records], dtype=torch.long, device=self.device)
        old_log_prob = torch.as_tensor(
            [float(r["z_logprob_old"]) for r in records], dtype=torch.float32, device=self.device
        )
        episode_returns = torch.as_tensor(
            [float(r["episode_return"]) for r in records], dtype=torch.float32, device=self.device
        )
        return {
            "states": states,
            "z": z,
            "old_log_prob": old_log_prob,
            "episode_returns": episode_returns,
        }

    def _apply_episode_strategy_ppo(self, *, latent_lam_h: float) -> dict[str, float]:
        stats = self._empty_latent_episode_strategy_stats()
        batch = self._episode_strategy_training_batch()
        if batch is None:
            return stats
        states = batch["states"]
        z = batch["z"]
        old_log_prob = batch["old_log_prob"]
        episode_returns = batch["episode_returns"]

        logits = self.model.strategy_logits(states)
        dist = Categorical(logits=logits)
        new_log_prob = dist.log_prob(z)
        v_z = self.model.episode_strategy_value(states, z)
        adv = episode_returns - v_z.detach()
        if self.latent_episode_strategy_return_norm and adv.numel() > 1:
            adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

        pg_loss, ppo_stats = ppo_policy_loss(
            new_log_prob,
            old_log_prob,
            adv.detach(),
            self.latent_episode_strategy_clip_eps,
        )
        v_loss = 0.5 * (episode_returns - v_z).pow(2).mean()
        z_entropy = dist.entropy().mean()
        h_goal = str(getattr(self.cfg, "latent_entropy_objective", "maximize") or "maximize").lower()
        if h_goal == "none" or latent_lam_h <= 0.0:
            entropy_term = torch.zeros((), dtype=torch.float32, device=self.device)
        elif h_goal == "minimize":
            entropy_term = float(latent_lam_h) * z_entropy
        else:
            entropy_term = -float(latent_lam_h) * z_entropy
        loss = self.latent_episode_strategy_coef * (
            pg_loss + self.latent_episode_strategy_value_coef * v_loss
        ) + entropy_term

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.cfg.max_grad_norm))
        self.optimizer.step()

        ratio = ppo_stats["ratio"].detach().float()
        with torch.no_grad():
            stats.update(
                {
                    "latent_episode_pg_loss": float(pg_loss.detach().cpu().item()),
                    "latent_episode_v_loss": float(v_loss.detach().cpu().item()),
                    "latent_episode_entropy": float(z_entropy.detach().cpu().item()),
                    "latent_episode_adv_mean": float(adv.detach().mean().cpu().item()),
                    "latent_episode_adv_std": float(
                        adv.detach().std(unbiased=False).cpu().item()
                    ) if adv.numel() > 1 else 0.0,
                    "latent_episode_return_mean": float(episode_returns.detach().mean().cpu().item()),
                    "latent_episode_return_std": float(
                        episode_returns.detach().std(unbiased=False).cpu().item()
                    ) if episode_returns.numel() > 1 else 0.0,
                    "latent_episode_ratio_mean": float(ratio.mean().cpu().item()),
                    "latent_episode_ratio_max": float(ratio.max().cpu().item()),
                    "latent_episode_ratio_min": float(ratio.min().cpu().item()),
                    "latent_episode_approx_kl": float(ppo_stats["approx_kl"].detach().cpu().item()),
                    "latent_episode_clip_fraction": float(ppo_stats["clip_fraction"].detach().cpu().item()),
                    "latent_episode_count": float(episode_returns.numel()),
                    "latent_episode_loss": float(loss.detach().cpu().item()),
                }
            )
        return stats

    def _make_buffer(self, obs: Dict[str, np.ndarray]) -> TensorDictRolloutBuffer:
        n_steps = int(self.cfg.n_steps)
        n_envs = int(self.env.num_envs)
        buffer = TensorDictRolloutBuffer(n_steps, n_envs, device=self.device)
        buffer.register_field("obs_grid", tuple(obs["grid"].shape[1:]))
        buffer.register_field("obs_vec", tuple(obs["vec"].shape[1:]))
        buffer.register_field("obs_agent_mask", tuple(obs["agent_mask"].shape[1:]))
        buffer.register_field("obs_mask", tuple(obs["mask"].shape[1:]))
        buffer.register_field("global_state", (self.model.global_state_dim,))
        buffer.register_field("actions", (len(getattr(self.env.action_space, "nvec", [])),), dtype=torch.long)
        buffer.register_field("log_probs")
        buffer.register_field("values")
        buffer.register_field("values_norm")
        buffer.register_field("next_values")
        buffer.register_field("rewards")
        buffer.register_field("reward_terminal")
        buffer.register_field("reward_offense")
        buffer.register_field("reward_pbrs")
        buffer.register_field("reward_team")
        buffer.register_field("reward_sparse")
        buffer.register_field("reward_sparse_points")
        buffer.register_field("reward_failure")
        buffer.register_field("reward_total")
        buffer.register_field("terminated", dtype=torch.bool)
        buffer.register_field("truncated", dtype=torch.bool)
        buffer.register_field("opponent_id", dtype=torch.long)
        if self.use_latent_strategy:
            buffer.register_field("z", dtype=torch.long)
            buffer.register_field("prev_z", dtype=torch.long)
            buffer.register_field("z_log_probs")
            buffer.register_field("z_logits", (self.latent_k,))
            buffer.register_field("z_resampled", dtype=torch.bool)
            buffer.register_field("z_persist_mask", dtype=torch.bool)
            buffer.register_field("phase_id", dtype=torch.long)
            buffer.register_field("outcome_id", dtype=torch.long)
            buffer.register_field("behavior_telemetry", (N_TELEMETRY,))
            buffer.register_field("spread_bucket_id", dtype=torch.long)
            buffer.register_field("role_bucket_id", dtype=torch.long)
            buffer.register_field("pressure_bucket_id", dtype=torch.long)
            buffer.register_field("attack_defense_ratio_bucket_id", dtype=torch.long)
            buffer.register_field("blue_ahead", dtype=torch.float32)
            if self.latent_kl_consecutive > 0.0:
                buffer.register_field("z_logits_prev", (self.latent_k,))
                buffer.register_field("z_kl_prev_valid")
        return buffer

    def _z_for_bootstrap(
        self,
        next_context_gs_t: torch.Tensor,
        z_t: torch.Tensor,
        dones: np.ndarray,
    ) -> torch.Tensor:
        """Strategy index for V(s', z') bootstrapping to match the start of the *next* decision."""
        if not self.use_latent_strategy:
            raise RuntimeError("_z_for_bootstrap requires latent strategy mode.")
        if self.fixed_latent_strategy:
            return torch.full_like(z_t, int(self.fixed_latent_strategy_id), dtype=torch.long)
        batch = int(z_t.shape[0])
        device = self.device
        done_t = torch.as_tensor(dones, dtype=torch.bool, device=device)
        age_next = self._strategy_age + 1
        age_next = torch.where(done_t, torch.zeros_like(age_next), age_next)
        needs_next = self._needs_strategy_sample.clone()
        if bool(done_t.any().item()):
            needs_next = needs_next.clone()
            needs_next[done_t] = bool(not self.fixed_latent_strategy)
        resample_next = needs_next.clone()
        if self.latent_resample_every_n > 0:
            resample_next = resample_next | (age_next >= int(self.latent_resample_every_n))
        resample_next = resample_next & (~done_t)
        z_next = z_t.long().clone()
        if bool(resample_next.any().item()):
            idx = torch.where(resample_next)[0]
            gs_sub = next_context_gs_t.index_select(0, idx)
            sampled_z, _, _, _ = self.model.sample_strategy(
                gs_sub,
                deterministic=bool(self.latent_bootstrap_z_deterministic),
            )
            z_next[idx] = sampled_z.long()
        return z_next

    def _next_values(
        self,
        infos: list[dict],
        next_global_state: np.ndarray,
        next_obs: Optional[Dict[str, np.ndarray]] = None,
        prev_z: Optional[torch.Tensor] = None,
        dones: Optional[np.ndarray] = None,
    ) -> torch.Tensor:
        rows = []
        for env_i, info in enumerate(infos):
            if bool(info.get("terminated", False)):
                rows.append(np.zeros((GLOBAL_STATE_DIM,), dtype=np.float32))
            elif bool(info.get("truncated", False)):
                terminal_obs = info.get("terminal_observation") or {}
                rows.append(np.asarray(terminal_obs.get("global_state", next_global_state[env_i]), dtype=np.float32))
            else:
                rows.append(np.asarray(next_global_state[env_i], dtype=np.float32))
        gs = torch.as_tensor(np.stack(rows, axis=0), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            if not self.use_latent_strategy:
                return _denormalize_values(self, self.model.values(gs))
            
            done_t = torch.as_tensor(dones, dtype=torch.bool, device=self.device) if dones is not None else None
            next_context_gs_t = self.temporal_tracker.update(gs, dones=done_t)
            self._last_context_state = next_context_gs_t

            if next_obs is None or prev_z is None:
                raise ValueError("latent next value bootstrap requires next_obs and prev_z.")
            obs_rows = self._obs_rows_from_next(next_obs, infos)
            next_obs_t = self._tensor_obs(obs_rows)
            if dones is None:
                raise ValueError("latent next value bootstrap requires dones for z lookahead.")
            next_z = self._z_for_bootstrap(
                next_context_gs_t,
                prev_z.long().reshape(-1),
                dones,
            )
            _, next_values, _, _ = self.model.act(
                next_obs_t,
                next_context_gs_t,
                deterministic=True,
                z_idx=next_z,
            )
            next_values = _denormalize_values(self, next_values)
            terminated = torch.as_tensor(
                [bool(info.get("terminated", False)) for info in infos],
                dtype=torch.bool,
                device=self.device,
            )
            return torch.where(terminated, torch.zeros_like(next_values), next_values)

    def collect_rollout(self) -> TensorDictRolloutBuffer:
        """Collect one rollout and compute advantages/returns."""
        log_decentralized_actor_contract_once(self)
        self._rollout_episode_records = []
        self._rollout_strategy_episode_records = []
        if self._last_obs is None or self._last_global_state is None:
            obs = self.env.reset()
            global_state = self.env.state().astype(np.float32)
            self._reset_strategy_state()
            if self.use_latent_strategy:
                gs_t = torch.as_tensor(global_state, dtype=torch.float32, device=self.device)
                context_state = self.temporal_tracker.update(gs_t)
            else:
                context_state = torch.as_tensor(global_state, dtype=torch.float32, device=self.device)
        else:
            obs = self._last_obs
            global_state = self._last_global_state
            if self.use_latent_strategy:
                context_state = self._last_context_state
            else:
                context_state = torch.as_tensor(global_state, dtype=torch.float32, device=self.device)
        buffer = self._make_buffer(obs)
        for step_idx in range(int(self.cfg.n_steps)):
            decision_global_state_np = np.asarray(global_state, dtype=np.float32)
            obs_t = self._tensor_obs(obs)
            with torch.no_grad():
                z_t, prev_z_t, strategy_aux = self._strategy_for_step(context_state)
                actions_t, values_norm_t, action_log_probs_t, _ = self.model.act(obs_t, context_state, z_idx=z_t)
                values_t = _denormalize_values(self, values_norm_t)
                log_probs_t = action_log_probs_t
            actions_np = actions_t.detach().cpu().numpy().astype(np.int64)
            beh_t = sb = rb = pb = adb = blue_ahead_t = None
            if self.use_latent_strategy:
                beh_t = compute_behavior_telemetry_batch(self.env.core, actions_t)
                sb, rb, pb, adb = bucket_ids_from_telemetry(beh_t, actions_t, self.env.core)
                blue_ahead_t = (self.env.core.blue_score > self.env.core.red_score).to(
                    dtype=torch.float32, device=self.device
                )
            self.env.step_async(actions_np)
            next_obs, rewards, dones, infos = self.env.step_wait()
            step_after = self.global_step + int(self.env.num_envs)
            z_np = z_t.detach().cpu().numpy() if z_t is not None else None
            for env_i, (done_i, info) in enumerate(zip(dones, infos)):
                if bool(done_i):
                    latent_z = int(z_np[env_i]) if z_np is not None else None
                    self._on_episode_done(
                        dict(info),
                        timestep=step_after,
                        rollout_step=step_idx + 1,
                        latent_z=latent_z,
                        env_index=env_i,
                    )
            next_global_state = self.env.state().astype(np.float32)
            next_values_t = self._next_values(infos, next_global_state, next_obs=next_obs, prev_z=z_t, dones=dones)
            terminated = np.asarray([bool(info.get("terminated", bool(done))) for info, done in zip(infos, dones)])
            truncated = np.asarray([bool(info.get("truncated", False)) for info in infos])
            reward_component = {
                key: torch.as_tensor(
                    [float(info.get(key, 0.0) or 0.0) for info in infos],
                    dtype=torch.float32,
                    device=self.device,
                )
                for key in (
                    "reward_terminal",
                    "reward_offense",
                    "reward_pbrs",
                    "reward_team",
                    "reward_sparse",
                    "reward_sparse_points",
                    "reward_failure",
                    "reward_total",
                )
            }
            shaping_coef = float(self._reward_shaping_coef())
            stalemate = torch.as_tensor(
                [bool(info.get("stalemate_truncated", False)) for info in infos],
                dtype=torch.bool,
                device=self.device,
            )
            reward_component = _compose_training_reward_components(
                reward_component,
                dense_weight=self.reward_dense_weight,
                reward_scale=self.reward_scale,
                reward_clip=self.reward_clip,
                shaping_coef=shaping_coef,
                stalemate=stalemate,
                stalemate_penalty=self.reward_stalemate_penalty,
            )
            if self.use_latent_strategy:
                done_t = torch.as_tensor(dones, dtype=torch.bool, device=self.device)
                self._episode_return_accum = self._episode_return_accum + reward_component["reward_total"].detach()
                if bool(done_t.any().item()):
                    if self.latent_episode_strategy_ppo:
                        for env_i, done_i in enumerate(dones):
                            if bool(done_i):
                                self._record_episode_strategy_outcome(
                                    env_i,
                                    dict(infos[env_i]),
                                    episode_return=float(self._episode_return_accum[env_i].detach().cpu().item()),
                                )
                    self._episode_return_accum[done_t] = 0.0
                    self._episode_strategy_has_start[done_t] = False

            opp_row = torch.as_tensor(
                [_opponent_id_int_from_info(self.cfg, dict(info)) for info in infos],
                dtype=torch.long,
                device=self.device,
            )

            add_items: dict[str, torch.Tensor] = dict(
                obs_grid=torch.as_tensor(obs["grid"], dtype=torch.float32, device=self.device),
                obs_vec=torch.as_tensor(obs["vec"], dtype=torch.float32, device=self.device),
                obs_agent_mask=torch.as_tensor(obs["agent_mask"], dtype=torch.float32, device=self.device),
                obs_mask=torch.as_tensor(obs["mask"], dtype=torch.float32, device=self.device),
                global_state=context_state,
                actions=actions_t,
                log_probs=log_probs_t,
                values=values_t,
                values_norm=values_norm_t,
                next_values=next_values_t,
                rewards=reward_component["reward_total"],
                reward_terminal=reward_component["reward_terminal"],
                reward_offense=reward_component["reward_offense"],
                reward_pbrs=reward_component["reward_pbrs"],
                reward_team=reward_component["reward_team"],
                reward_sparse=reward_component["reward_sparse"],
                reward_sparse_points=reward_component["reward_sparse_points"],
                reward_failure=reward_component["reward_failure"],
                reward_total=reward_component["reward_total"],
                terminated=torch.as_tensor(terminated, dtype=torch.bool, device=self.device),
                truncated=torch.as_tensor(truncated, dtype=torch.bool, device=self.device),
                opponent_id=opp_row,
            )
            if self.use_latent_strategy:
                n_e = int(self.env.num_envs)
                phase_list: list[int] = []
                outcome_list: list[int] = []
                for e in range(n_e):
                    info_e = dict(infos[e]) if e < len(infos) else {}
                    sf = float(info_e.get("stalemate_frac", 0.0) or 0.0)
                    phase_list.append(
                        int(team_phase_id_from_global_state(decision_global_state_np[e], stalemate_frac=sf))
                    )
                    outcome_list.append(int(outcome_id_from_global_state(decision_global_state_np[e])))
                add_items.update(
                    z=strategy_aux["z"],
                    prev_z=strategy_aux["prev_z"],
                    z_log_probs=strategy_aux["z_log_prob"],
                    z_logits=strategy_aux["z_logits"],
                    z_resampled=strategy_aux["z_resampled"],
                    z_persist_mask=strategy_aux["z_persist_mask"],
                    phase_id=torch.as_tensor(phase_list, dtype=torch.long, device=self.device),
                    outcome_id=torch.as_tensor(outcome_list, dtype=torch.long, device=self.device),
                    behavior_telemetry=beh_t,
                    spread_bucket_id=sb,
                    role_bucket_id=rb,
                    pressure_bucket_id=pb,
                    attack_defense_ratio_bucket_id=adb,
                    blue_ahead=blue_ahead_t,
                )
                if self.latent_kl_consecutive > 0.0 and self._z_kl_first_in_ep is not None:
                    z_logits_cur = strategy_aux["z_logits"]
                    zlp = self._prev_z_logits
                    if zlp is None:
                        zlp = torch.zeros_like(z_logits_cur)
                    add_items["z_logits_prev"] = zlp
                    add_items["z_kl_prev_valid"] = (~self._z_kl_first_in_ep).to(dtype=torch.float32)
            buffer.add(**add_items)
            probe_rows = getattr(self, "_global_state_probe_rows", None)
            if probe_rows is not None:
                score_lim = max(1, int(getattr(self.env.cfg, "score_limit", 1)))
                max_dec = max(1, int(getattr(self.env.cfg, "max_decision_steps", 400)))
                gs_np = decision_global_state_np
                for i, info in enumerate(infos):
                    bs = int(info.get("blue_score", 0) or 0)
                    rs = int(info.get("red_score", 0) or 0)
                    ds = int(info.get("decision_steps", 0) or 0)
                    probe_rows.append(
                        {
                            "global_state": np.asarray(gs_np[i], dtype=np.float32).copy(),
                            "score_diff": float(bs - rs) / float(score_lim),
                            "time_frac": float(ds) / float(max_dec),
                        }
                    )
            if self.latent_resample_on_flag:
                prev_sec = context_state[:, GLOBAL_STATE_FLAG_TERRITORY_SLICE]
                nxt_sec = torch.as_tensor(
                    next_global_state[:, GLOBAL_STATE_FLAG_TERRITORY_SLICE],
                    dtype=torch.float32,
                    device=self.device,
                )
                chg = self._flag_territory_features_changed(prev_sec, nxt_sec)
                self._needs_strategy_sample[chg] = True
            obs = next_obs
            global_state = next_global_state
            if self.use_latent_strategy:
                context_state = self._last_context_state
            else:
                context_state = torch.as_tensor(global_state, dtype=torch.float32, device=self.device)
            self.global_step += int(self.env.num_envs)
            self._on_sb3_rollout_env_step()
            if self.use_latent_strategy and self.latent_kl_consecutive > 0.0 and self._z_kl_first_in_ep is not None:
                self._prev_z_logits = strategy_aux["z_logits"].detach().clone()
                self._z_kl_first_in_ep = torch.as_tensor(dones, dtype=torch.bool, device=self.device)
            self._mark_strategy_step_done(dones)
            if self.telemetry.e3_step_telemetry_path and self.use_latent_strategy and z_t is not None and prev_z_t is not None:
                assert beh_t is not None and sb is not None and adb is not None
                self.telemetry.append_e3_step(
                    rollout_step=step_idx,
                    global_step_at_step_end=int(self.global_step),
                    decision_global_state_np=decision_global_state_np,
                    z_t=z_t,
                    prev_z=prev_z_t,
                    strategy_aux=strategy_aux,
                    infos=infos,
                    behavior_telemetry_np=beh_t.detach().cpu().numpy(),
                    spread_bucket_np=sb.detach().cpu().numpy(),
                    role_bucket_np=rb.detach().cpu().numpy(),
                    pressure_bucket_np=pb.detach().cpu().numpy(),
                    attack_defense_ratio_bucket_np=adb.detach().cpu().numpy(),
                    blue_ahead_np=blue_ahead_t.detach().cpu().numpy(),
                )

        buffer.fields["next_values"][: int(buffer.pos)].copy_(
            align_next_values_to_rollout_actions(
                buffer.fields["values"][: int(buffer.pos)],
                buffer.fields["next_values"][: int(buffer.pos)],
                buffer.fields["terminated"][: int(buffer.pos)].bool(),
                buffer.fields["truncated"][: int(buffer.pos)].bool(),
            )
        )
        gae_kw: dict[str, Any] = dict(
            gamma=float(self.cfg.gamma),
            gae_lambda=float(self.cfg.gae_lambda),
        )
        if self.latent_gae_reset_on_z_change:
            gae_kw["latent_z_field"] = "z"
            gae_kw["reset_gae_on_z_change"] = True
        buffer.compute_returns_and_advantages(**gae_kw)
        if self.use_latent_strategy:
            with torch.no_grad():
                option_returns, option_advantages = compute_option_returns(
                    rewards=buffer.fields["rewards"],
                    values=buffer.fields["values"],
                    next_values=buffer.fields["next_values"],
                    terminated=buffer.fields["terminated"],
                    truncated=buffer.fields["truncated"],
                    z_resampled=buffer.fields["z_resampled"],
                    gamma=float(self.cfg.gamma),
                )
                if "option_returns" not in buffer.fields:
                    buffer.register_field("option_returns")
                if "option_advantages" not in buffer.fields:
                    buffer.register_field("option_advantages")
                buffer.fields["option_returns"].copy_(option_returns)
                buffer.fields["option_advantages"].copy_(option_advantages)
        _update_return_norm_stats(self, buffer.fields["returns"][: int(buffer.pos)])
        self._last_obs = obs
        self._last_global_state = global_state
        return buffer

    def update(self, buffer: TensorDictRolloutBuffer, *, total_timesteps: int) -> dict[str, float]:
        """Run PPO epochs over one rollout."""
        progress_remaining = max(0.0, 1.0 - float(self.global_step) / max(1.0, float(total_timesteps)))
        lr_floor_frac = max(0.0, min(float(getattr(self.cfg, "lr_floor_frac", 0.1) or 0.0), 1.0))
        lr = self.base_learning_rate * max(progress_remaining, lr_floor_frac)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        ent_coef = self.ent_coef if progress_remaining > 0.75 else 0.5 * self.ent_coef
        latent_lam_h_start = max(0.0, float(getattr(self.cfg, "latent_lam_h", 0.0) or 0.0))
        latent_lam_h_end = min(latent_lam_h_start, 0.001)
        latent_lam_h = latent_lam_h_end + (latent_lam_h_start - latent_lam_h_end) * progress_remaining
        _update_strategy_return_stats(self, buffer)

        stats: dict[str, list[float]] = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "approx_kl": [],
            "clip_fraction": [],
            "grad_norm": [],
            "strategy_entropy": [],
            "strategy_policy_loss": [],
            "strategy_approx_kl": [],
            "strategy_clip_fraction": [],
            "strategy_ratio_std": [],
            "strategy_aux_return_loss": [],
            "strategy_persist_loss": [],
            "strategy_grad_norm": [],
            "strategy_resample_fraction": [],
            "strategy_kl": [],
            "strategy_phase_loss": [],
        }
        stop_update = False
        target_kl = getattr(self.cfg, "target_kl", None)
        for _ in range(self.n_epochs):
            for batch in buffer.iter_minibatches(self.batch_size, shuffle=True):
                obs_batch = {
                    "grid": batch["obs_grid"],
                    "vec": batch["obs_vec"],
                    "agent_mask": batch["obs_agent_mask"],
                    "mask": batch["obs_mask"],
                }
                z_idx = batch["z"] if self.use_latent_strategy else None
                values_norm, action_log_prob, entropy, aux = self.model.evaluate_actions(
                    obs_batch,
                    batch["global_state"],
                    batch["actions"],
                    z_idx=z_idx,
                )
                if self.use_latent_strategy:
                    resample = batch["z_resampled"].bool()
                    persist_mask = batch["z_persist_mask"].bool()
                    log_prob = action_log_prob
                    strategy_log_prob = aux["strategy_log_prob"]
                    strategy_entropy = aux["strategy_entropy"]
                    h_goal = str(getattr(self.cfg, "latent_entropy_objective", "maximize") or "maximize").lower()
                    strategy_entropy_loss, _ = _latent_strategy_entropy_loss(
                        strategy_entropy,
                        resample,
                        objective=h_goal,
                        lam_h=latent_lam_h,
                        device=self.device,
                    )
                    persist_term_loss, persist_stats = _latent_strategy_persistence_loss(
                        aux["strategy_logits"],
                        batch["prev_z"],
                        persist_mask,
                        lam_p=float(getattr(self.cfg, "latent_lam_p", 0.0)),
                        device=self.device,
                    )
                    if self.latent_resample_every_n == 0 and not self.latent_resample_on_flag:
                        assert persist_stats["persist_term"] == 0.0, (
                            "L_persist must be exactly 0 when no mid-episode resampling (latent_resample_every_n=0, on_flag off)"
                        )
                    persist_loss_value = persist_stats["persist_term"]
                    latent_loss = persist_term_loss + strategy_entropy_loss
                    if self.latent_kl_consecutive > 0.0:
                        kl_loss, kl_stats = _latent_strategy_kl_consecutive_loss(
                            batch["z_logits"],
                            batch["z_logits_prev"],
                            batch["z_kl_prev_valid"],
                            coef=float(self.latent_kl_consecutive),
                        )
                        latent_loss = latent_loss + kl_loss
                        stats["strategy_kl"].append(kl_stats["kl_mean"])
                    else:
                        stats["strategy_kl"].append(0.0)
                    if self.latent_strategy_aux_predict_phase_coef > 0.0:
                        phase_logits = self.model.phase_logits_from_strategy_logits(aux["strategy_logits"])
                        phase_loss_scaled, phase_stats = _latent_strategy_phase_aux_loss(
                            phase_logits,
                            batch["phase_id"],
                            coef=float(self.latent_strategy_aux_predict_phase_coef),
                        )
                        latent_loss = latent_loss + phase_loss_scaled
                        stats["strategy_phase_loss"].append(phase_stats["phase_term"])
                    else:
                        stats["strategy_phase_loss"].append(0.0)

                    if self.fixed_latent_strategy:
                        strategy_entropy = torch.zeros_like(entropy)
                        persist_loss_value = 0.0
                        latent_loss = torch.zeros((), dtype=torch.float32, device=self.device)
                else:
                    log_prob = action_log_prob
                    strategy_entropy = torch.zeros_like(entropy)
                    persist_loss_value = 0.0
                    latent_loss = torch.zeros((), dtype=torch.float32, device=self.device)
                    resample = torch.zeros_like(entropy, dtype=torch.bool)
                    stats["strategy_kl"].append(0.0)
                    stats["strategy_phase_loss"].append(0.0)

                advantages = batch["advantages"]
                if advantages.numel() > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
                if self.use_latent_strategy and not self.fixed_latent_strategy:
                    strat_adv = batch["option_advantages"] if getattr(self.cfg, "latent_q_phi_option_advantage", False) else advantages
                    strategy_policy_loss_scaled, strategy_ppo_stats = _latent_strategy_ppo_loss(
                        strategy_log_prob,
                        batch["z_log_probs"],
                        strat_adv,
                        resample,
                        clip_range=float(self.clip_range),
                        coef=float(self.latent_strategy_ppo_coef),
                        device=self.device,
                    )
                    # Default to a zero tensor so unit tests that mock
                    # ``_latent_strategy_ppo_loss`` with a minimal return value
                    # (e.g. an empty stats dict) do not KeyError. The real
                    # production path always populates this key.
                    strategy_policy_loss = strategy_ppo_stats.pop(
                        "policy_loss", torch.zeros((), dtype=torch.float32, device=self.device)
                    )
                    strategy_aux_return_loss_value = 0.0
                    if bool(resample.any().item()):
                        latent_loss = latent_loss + strategy_policy_loss_scaled
                        if self.latent_strategy_aux_return_head and self.latent_strategy_aux_return_coef > 0.0:
                            pred_all = self.model.strategy_aux_return_predictions(batch["global_state"])
                            ret_target = _normalize_strategy_returns(self, batch["returns"][resample])
                            aux_return_loss_scaled, aux_return_stats = _latent_strategy_aux_return_loss(
                                pred_all,
                                batch["z"],
                                ret_target,
                                resample,
                                latent_k=int(self.latent_k),
                                coef=float(self.latent_strategy_aux_coef if hasattr(self, 'latent_strategy_aux_coef') else self.latent_strategy_aux_return_coef),
                                device=self.device,
                            )
                            strategy_aux_return_loss_value = aux_return_stats["aux_return_term"]
                            latent_loss = latent_loss + aux_return_loss_scaled
                else:
                    strategy_policy_loss = torch.zeros((), dtype=torch.float32, device=self.device)
                    strategy_aux_return_loss_value = 0.0
                    strategy_ppo_stats = {
                        "approx_kl": torch.zeros((), dtype=torch.float32, device=self.device),
                        "clip_fraction": torch.zeros((), dtype=torch.float32, device=self.device),
                        "ratio": torch.ones((1,), dtype=torch.float32, device=self.device),
                    }
                policy_loss, ppo_stats = ppo_policy_loss(
                    log_prob,
                    batch["log_probs"],
                    advantages,
                    self.clip_range,
                )
                value_targets = _normalize_value_targets(self, batch["returns"])
                value_loss = ppo_value_loss(values_norm, batch["values_norm"], value_targets, self.value_clip_range)
                entropy_loss = -entropy.mean()
                loss = policy_loss + self.vf_coef * value_loss + ent_coef * entropy_loss + latent_loss

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                strategy_grad_norm = self._strategy_encoder_grad_norm()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.cfg.max_grad_norm))
                self.optimizer.step()

                approx_kl_value = float(ppo_stats["approx_kl"].detach().cpu().item())
                stats["policy_loss"].append(float(policy_loss.detach().cpu().item()))
                stats["value_loss"].append(float(value_loss.detach().cpu().item()))
                stats["entropy"].append(float(entropy.mean().detach().cpu().item()))
                stats["approx_kl"].append(approx_kl_value)
                stats["clip_fraction"].append(float(ppo_stats["clip_fraction"].detach().cpu().item()))
                stats["grad_norm"].append(float(torch.as_tensor(grad_norm).detach().cpu().item()))
                stats["strategy_entropy"].append(float(strategy_entropy.mean().detach().cpu().item()))
                stats["strategy_policy_loss"].append(float(strategy_policy_loss.detach().cpu().item()))
                stats["strategy_approx_kl"].append(float(strategy_ppo_stats["approx_kl"].detach().cpu().item()))
                stats["strategy_clip_fraction"].append(
                    float(strategy_ppo_stats["clip_fraction"].detach().cpu().item())
                )
                ratio_z = strategy_ppo_stats["ratio"].detach().float()
                stats["strategy_ratio_std"].append(
                    float(ratio_z.std(unbiased=False).detach().cpu().item()) if ratio_z.numel() > 1 else 0.0
                )
                stats["strategy_aux_return_loss"].append(float(strategy_aux_return_loss_value))
                stats["strategy_persist_loss"].append(float(persist_loss_value))
                stats["strategy_grad_norm"].append(strategy_grad_norm)
                stats["strategy_resample_fraction"].append(float(resample.float().mean().detach().cpu().item()))
                if target_kl is not None and approx_kl_value > 1.5 * float(target_kl):
                    stop_update = True
                    break
            if stop_update:
                break

        episode_strategy_stats = self._apply_episode_strategy_ppo(latent_lam_h=latent_lam_h)
        strategy_experience_stats = _write_strategy_experience_table(self)

        self.last_stats = {name: float(np.mean(values)) if values else 0.0 for name, values in stats.items()}
        value_losses = np.asarray(stats["value_loss"], dtype=np.float32)
        if value_losses.size > 0:
            self.last_stats.update(
                {
                    "value_loss_min": float(np.min(value_losses)),
                    "value_loss_std": float(np.std(value_losses)),
                    "value_loss_p10": float(np.percentile(value_losses, 10)),
                    "value_loss_p50": float(np.percentile(value_losses, 50)),
                    "value_loss_p90": float(np.percentile(value_losses, 90)),
                    "value_loss_max": float(np.max(value_losses)),
                }
            )
        else:
            self.last_stats.update(
                {
                    "value_loss_min": 0.0,
                    "value_loss_std": 0.0,
                    "value_loss_p10": 0.0,
                    "value_loss_p50": 0.0,
                    "value_loss_p90": 0.0,
                    "value_loss_max": 0.0,
                }
            )
        self.last_stats["learning_rate"] = float(lr)
        self.last_stats["return_norm_mean"] = float(self._return_norm_mean) if self.normalize_returns else 0.0
        self.last_stats["return_norm_std"] = float(_return_norm_std(self)) if self.normalize_returns else 0.0
        self.last_stats["return_norm_count"] = float(self._return_norm_count) if self.normalize_returns else 0.0
        self.last_stats.update(_strategy_resample_advantage_stats(self, buffer))
        self.last_stats.update(_latent_option_advantage_stats(self, buffer))
        self.last_stats.update(_rollout_advantage_diagnostics(self, buffer))
        self.last_stats.update(_latent_rollout_stats(self, buffer))
        self.last_stats.update(_latent_opponent_rollout_diag(self, buffer))
        self.last_stats.update(_behavior_diversity_stats(self, buffer))
        self.last_stats.update(_forced_z_behavior_profile(self, buffer))
        self.last_stats.update(episode_strategy_stats)
        self.last_stats.update(strategy_experience_stats)
        return self.last_stats

    def _save_periodic_checkpoint(self) -> None:
        if self.periodic_checkpoint_steps <= 0:
            return
        while self.global_step >= self._next_periodic_checkpoint_step:
            ckpt_name = f"ckpt_{str(getattr(self.cfg, 'run_tag', 'ppo'))}_{int(self._next_periodic_checkpoint_step)}.zip"
            ckpt_path = os.path.join(str(getattr(self.cfg, "checkpoint_dir", "checkpoints")), ckpt_name)
            self.save(ckpt_path)
            print(f"[PPO] Periodic checkpoint saved: {ckpt_path}")
            self._next_periodic_checkpoint_step += self.periodic_checkpoint_steps

    def learn(self, total_timesteps: int) -> dict[str, float]:
        """Train until at least ``total_timesteps`` environment transitions have been collected."""
        total = int(total_timesteps)
        self._sb3_rollout_pbar = _open_sb3_style_progress(
            self.cfg, total_timesteps=total, current_num_timesteps=self.global_step
        )
        try:
            while self.global_step < total:
                rollout = self.collect_rollout()
                stats = self.update(rollout, total_timesteps=total)
                self._updates_completed += 1
                row = self.telemetry.write_update_metrics(stats, rollout)
                self._save_periodic_checkpoint()
                if row:
                    z_wr_parts: list[str] = []
                    z_occ_parts: list[str] = []
                    if self.use_latent_strategy:
                        for i in range(self.latent_k):
                            wr = row.get(f"episode_z_{i}_win_rate", "")
                            occ = row.get(f"strategy_occupancy_{i}", "")
                            z_wr_parts.append("-" if wr == "" else f"{float(wr):.3f}")
                            z_occ_parts.append("-" if occ == "" else f"{float(occ):.3f}")
                    z_entropy = float(row.get("strategy_entropy", 0.0) or 0.0)
                    z_entropy_frac = float(row.get("strategy_entropy_frac", 0.0) or 0.0)
                    z_wr_spread = float(row.get("strategy_wr_spread", 0.0) or 0.0)
                    opp_suffix = ""
                    if self.use_latent_strategy:
                        mi_z_o = float(row.get("latent_mi_z_opponent_nats", 0.0) or 0.0)
                        mi_z_p = float(row.get("latent_mi_z_phase_nats", 0.0) or 0.0)
                        mi_z_y = float(row.get("latent_mi_z_outcome_nats", 0.0) or 0.0)
                        mi_z_f = float(row.get("latent_mi_z_flag_state_nats", 0.0) or 0.0)
                        opp_diag_bits: list[str] = []
                        for o in range(SCRIPTED_OPPONENT_MI_COUNT):
                            occ_o = [
                                float(row.get(f"strategy_occupancy_op{o}_z{k}", 0.0) or 0.0) for k in range(self.latent_k)
                            ]
                            wr_o = [row.get(f"episode_opp{o}_z{k}_win_rate", "") for k in range(self.latent_k)]
                            if sum(occ_o) < 1e-9 and all(w == "" for w in wr_o):
                                continue
                            occ_s = ",".join(f"{x:.2f}" for x in occ_o)
                            wr_s = ",".join("-" if w == "" else f"{float(w):.2f}" for w in wr_o)
                            opp_diag_bits.append(f"o{o}:z_occ=[{occ_s}] z_wr=[{wr_s}]")
                        opp_suffix = (
                            f" MI_z_o={mi_z_o:.4f} MI_z_phase={mi_z_p:.4f} "
                            f"MI_z_flag={mi_z_f:.4f} MI_z_outcome={mi_z_y:.4f} | "
                            + " ".join(opp_diag_bits)
                        )
                    print(
                        f"[PPO|diag] steps={self.global_step} "
                        f"ev={row['explained_variance']:.3f} "
                        f"v_loss={row['value_loss']:.3f} "
                        f"shape/out={row['reward_shaping_mean']:.3f}/{row['reward_outcome_mean']:.3f} "
                        f"qphi_grad={row.get('strategy_grad_norm', 0.0):.3f} "
                        f"zH={z_entropy:.3f}({z_entropy_frac:.2f}) "
                        f"z_wr_spread={z_wr_spread:.3f} "
                        f"z_aux_ret={float(row.get('strategy_aux_return_loss', row.get('strategy_q_loss', 0.0))):.3f} "
                        f"z_pi={float(row.get('strategy_policy_loss', 0.0)):.3f} "
                        f"z_ratio={float(row.get('strategy_ratio_std', 0.0)):.3f} "
                        f"z_occ=[{','.join(z_occ_parts)}] "
                        f"z_wr=[{','.join(z_wr_parts)}]"
                        f"{opp_suffix}"
                    )
                    if self.use_latent_strategy:
                        sw_cap = float(row.get("latent_switch_near_capture_frac", 0.0) or 0.0)
                        sw_kill = float(row.get("latent_switch_near_kill_frac", 0.0) or 0.0)
                        sw_ret = float(row.get("latent_switch_near_return_frac", 0.0) or 0.0)
                        div_role = float(row.get("latent_role_diversity", 0.0) or 0.0)
                        div_spread = float(row.get("latent_spread_diversity", 0.0) or 0.0)
                        div_pres = float(row.get("latent_pressure_diversity", 0.0) or 0.0)
                        div_adr = float(row.get("latent_adr_diversity", 0.0) or 0.0)
                        print(
                            f"      [Switch Near] cap={sw_cap:.3f} kill={sw_kill:.3f} ret={sw_ret:.3f} | "
                            f"div_role={div_role:.3f} div_spread={div_spread:.3f} div_pressure={div_pres:.3f} div_adr={div_adr:.3f}"
                        )
                if self.normalize_returns:
                    print(
                        "[PPO|return_norm] "
                        f"update={self._updates_completed} "
                        f"mean={stats.get('return_norm_mean', 0.0):.4f} "
                        f"std={stats.get('return_norm_std', 0.0):.4f} "
                        f"count={stats.get('return_norm_count', 0.0):.0f}"
                    )
                if bool(getattr(self.cfg, "verbose_training", False)):
                    latent_bits = ""
                    if self.use_latent_strategy:
                        latent_bits = (
                            f" z_entropy={stats.get('strategy_entropy', 0.0):.4f} "
                            f"z_persist={stats.get('strategy_persist_loss', 0.0):.4f}"
                        )
                    print(
                        "[PPO|custom] "
                        f"steps={self.global_step} policy_loss={stats['policy_loss']:.4f} "
                        f"value_loss={stats['value_loss']:.4f} approx_kl={stats['approx_kl']:.5f}"
                        f"{latent_bits}"
                    )
        finally:
            if self._sb3_rollout_pbar is not None:
                self._sb3_rollout_pbar.refresh()  # type: ignore[union-attr]
                self._sb3_rollout_pbar.close()  # type: ignore[union-attr]
                self._sb3_rollout_pbar = None
            self.telemetry.close_e3_step_telemetry()
        return self.last_stats

    def save(self, path: str) -> None:
        """Save a torch checkpoint. The project keeps the historical ``.zip`` suffix."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "global_step": self.global_step,
                "updates_completed": self._updates_completed,
                "return_norm_mean": float(self._return_norm_mean),
                "return_norm_var": float(self._return_norm_var),
                "return_norm_count": float(self._return_norm_count),
                "strategy_return_mean": float(self._strategy_return_mean),
                "strategy_return_var": float(self._strategy_return_var),
                "strategy_return_count": float(self._strategy_return_count),
                "cfg": asdict(self.cfg),
                "last_stats": self.last_stats,
                "format": CUSTOM_PPO_LATENT_FORMAT if self.use_latent_strategy else CUSTOM_PPO_FORMAT,
                "actor_arch": CUSTOM_PPO_ACTOR_ARCH,
                "actor_cnn_feature_dim": int(self.model.actor_cnn_feature_dim),
                "global_state_dim": int(self.model.global_state_dim),
                "vec_schema_version": CUSTOM_PPO_VEC_SCHEMA_VERSION,
            },
            path,
        )

    def load(self, path: str) -> None:
        """Restore a checkpoint produced by :meth:`save`."""
        payload = _torch_load_checkpoint(path, map_location=self.device)
        _assert_compatible_global_state_dim(payload, path)
        _load_model_state_dict_compat(self.model, payload["model_state_dict"])
        self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        self.global_step = int(payload.get("global_step", 0))
        self._updates_completed = int(payload.get("updates_completed", 0))
        self._return_norm_mean = float(payload.get("return_norm_mean", 0.0))
        self._return_norm_var = float(payload.get("return_norm_var", 1.0))
        self._return_norm_count = float(payload.get("return_norm_count", 1e-4))
        self._strategy_return_mean = float(payload.get("strategy_return_mean", 0.0))
        self._strategy_return_var = float(payload.get("strategy_return_var", 1.0))
        self._strategy_return_count = float(payload.get("strategy_return_count", 1e-4))
        self.last_stats = dict(payload.get("last_stats", {}))
        self._last_obs = None
        self._last_global_state = None
        self._current_z = None
        if self.use_latent_strategy:
            self._reset_strategy_state()
