"""Local PPO/MAPPO-style trainer with optional latent team strategy.

Invariant (no z supervision from opponents): strategy indices are learned only from task
reward and the plan's entropy / persistence (and optional §12 terms). Opponent kind,
curriculum phase, and scripted tag strings never feed :class:`StrategyEncoder` or
``nn.Embedding`` for ``z`` as targets.
"""

from __future__ import annotations

import csv
import math
import os
import sys
import warnings
from collections import deque
from dataclasses import asdict
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
from rl.latent_marl import StrategyEncoder, paper_strategy_switch_indicator, TemporalStateTracker, CONTEXT_STATE_DIM
from rl.networks import CNNEncoder, CentralizedCritic
from rl.ppo_core import (
    TensorDictRolloutBuffer,
    align_next_values_to_rollout_actions,
    ppo_policy_loss,
    ppo_value_loss,
)


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
    """
    Port of :class:`stable_baselines3.common.callbacks.ProgressBarCallback`.

    Bar ``total`` = remaining wall-clock budget in environment transitions (``total_timesteps`` minus
    progress so far, same as ``.learn`` when ``num_timesteps`` is already advanced).  Each
    vectorized step advances by ``n_envs`` (``ProgressBarCallback._on_step``), which feeds tqdm's
    rate/ETA (hours/minutes) while rollout collection runs.
    """
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


def _torch_load_checkpoint(path: str, *, map_location: str | torch.device):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _assert_compatible_global_state_dim(payload: dict[str, Any], path: str) -> None:
    ckpt_dim = payload.get("global_state_dim")
    if ckpt_dim is None:
        return
    cfg = payload.get("cfg") or {}
    uses_latent = bool(cfg.get("use_latent_strategy", False))
    expected_dim = CONTEXT_STATE_DIM if uses_latent else GLOBAL_STATE_DIM
    if int(ckpt_dim) != int(expected_dim):
        raise ValueError(
            f"Checkpoint {path!r} was saved with global_state_dim={int(ckpt_dim)}, "
            f"but this code expects {expected_dim}. Start a fresh run or load a "
            "checkpoint trained after the global-state expansion."
        )


def read_custom_ppo_metadata(path: str) -> dict[str, Any]:
    """Read lightweight metadata from a local PPO checkpoint."""
    payload = _torch_load_checkpoint(path, map_location="cpu")
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError("Not a custom PPO checkpoint.")
    cfg = payload.get("cfg") or {}
    fmt = str(payload.get("format", "custom_ppo_v2"))
    meta: dict[str, Any] = {
        "format": fmt,
        "model_path": path,
        "cfg": cfg,
        "actor_arch": str(payload.get("actor_arch", "flat_mlp" if fmt.endswith("_v2") else "unknown")),
        "vec_schema_version": int(payload.get("vec_schema_version", 2 if fmt.endswith("_v2") else 0)),
        "global_state_dim": int(payload.get("global_state_dim", GLOBAL_STATE_DIM)),
    }
    if isinstance(cfg, dict):
        if "max_blue_agents" in cfg:
            meta["n_blue"] = int(cfg["max_blue_agents"])
        elif "n_agents_per_team" in cfg:
            meta["n_blue"] = int(cfg["n_agents_per_team"])
        meta["use_latent_strategy"] = bool(cfg.get("use_latent_strategy", False))
        meta["fixed_latent_strategy"] = bool(cfg.get("fixed_latent_strategy", False))
        meta["fixed_latent_strategy_id"] = int(cfg.get("fixed_latent_strategy_id", 0) or 0)
        meta["actor_cnn_feature_dim"] = int(
            cfg.get("actor_cnn_feature_dim", payload.get("actor_cnn_feature_dim", 128))
        )
        if "latent_k" in cfg:
            meta["latent_k"] = int(cfg["latent_k"])
    return meta


class CustomPPOInferencePolicy:
    """Small inference wrapper with a ``predict`` method for viewer/eval code."""

    def __init__(
        self,
        model: SharedActorCentralizedCritic,
        *,
        device: str | torch.device = "cpu",
        cfg: Optional[dict[str, Any]] = None,
    ) -> None:
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()
        self._prev_z: Optional[torch.Tensor] = None
        cfg = cfg or {}
        self.strategy_interval = max(0, int(cfg.get("latent_resample_every_n", 0) or 0))
        self.fixed_latent_strategy = bool(cfg.get("fixed_latent_strategy", False))
        self.fixed_latent_strategy_id = max(0, int(cfg.get("fixed_latent_strategy_id", 0) or 0))
        self._strategy_age = 0
        self._last_strategy_z: Optional[torch.Tensor] = None
        self._last_strategy_probs: Optional[torch.Tensor] = None
        self._last_strategy_entropy: Optional[torch.Tensor] = None
        self._last_strategy_resampled = False
        self._temporal_tracker: Optional[TemporalStateTracker] = None

    def _fixed_strategy_id(self) -> int:
        if not self.model.uses_latent_strategy:
            return 0
        return min(self.fixed_latent_strategy_id, max(0, int(self.model.latent_k) - 1))

    def _fixed_strategy_tensor(self, batch: int) -> torch.Tensor:
        return torch.full((int(batch),), self._fixed_strategy_id(), dtype=torch.long, device=self.device)

    def _fixed_strategy_probs(self, batch: int) -> torch.Tensor:
        probs = torch.zeros((int(batch), int(self.model.latent_k)), dtype=torch.float32, device=self.device)
        probs[:, self._fixed_strategy_id()] = 1.0
        return probs

    def _get_temporal_tracker(self, batch_size: int) -> TemporalStateTracker:
        if self._temporal_tracker is None or self._temporal_tracker.num_envs != batch_size:
            self._temporal_tracker = TemporalStateTracker(
                num_envs=batch_size,
                state_dim=GLOBAL_STATE_DIM,
                device=self.device,
            )
        return self._temporal_tracker

    def reset_strategy(self) -> None:
        """Forget the persisted inference strategy, typically at episode reset."""
        self._prev_z = None
        self._strategy_age = 0
        self._last_strategy_z = None
        self._last_strategy_probs = None
        self._last_strategy_entropy = None
        self._last_strategy_resampled = False
        if self._temporal_tracker is not None:
            self._temporal_tracker.reset()

    def _tensor_obs(self, obs: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        return {
            "grid": torch.as_tensor(obs["grid"], dtype=torch.float32, device=self.device),
            "vec": torch.as_tensor(obs["vec"], dtype=torch.float32, device=self.device),
            "agent_mask": torch.as_tensor(obs["agent_mask"], dtype=torch.float32, device=self.device),
            "mask": torch.as_tensor(obs["mask"], dtype=torch.float32, device=self.device),
        }

    def _global_state_tensor(self, obs: Dict[str, np.ndarray], batch: int) -> torch.Tensor:
        raw = obs.get("global_state")
        if raw is None:
            return torch.zeros((batch, GLOBAL_STATE_DIM), dtype=torch.float32, device=self.device)
        arr = np.asarray(raw, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, ...]
        return torch.as_tensor(arr, dtype=torch.float32, device=self.device)

    def _batched_obs(self, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        batched: Dict[str, np.ndarray] = {}
        for key, value in obs.items():
            arr = np.asarray(value, dtype=np.float32)
            if key == "grid" and arr.ndim == 4:
                arr = arr[None, ...]
            elif key == "vec" and arr.ndim == 2:
                arr = arr[None, ...]
            elif key in {"agent_mask", "mask"} and arr.ndim == 1:
                arr = arr[None, ...]
            batched[key] = arr
        return batched

    def predict(
        self,
        obs: Dict[str, np.ndarray],
        deterministic: bool = True,
    ) -> tuple[np.ndarray, None]:
        """Return flattened MultiDiscrete actions for each batch row."""
        batched = self._batched_obs(obs)
        obs_t = self._tensor_obs(batched)
        with torch.no_grad():
            if self.model.uses_latent_strategy:
                batch = int(obs_t["grid"].shape[0])
                global_state = self._global_state_tensor(batched, batch)
                tracker = self._get_temporal_tracker(batch)
                context_gs = tracker.update(global_state)
                if self.fixed_latent_strategy:
                    z_idx = self._fixed_strategy_tensor(batch)
                    self._prev_z = z_idx.detach()
                    z_ent = torch.zeros((batch,), dtype=torch.float32, device=self.device)
                    z_probs = self._fixed_strategy_probs(batch)
                    needs_strategy = False
                else:
                    z_logits = self.model.strategy_logits(context_gs)
                    z_dist = Categorical(logits=z_logits)
                    needs_strategy = (
                        self._prev_z is None
                        or int(self._prev_z.numel()) != batch
                        or (self.strategy_interval > 0 and self._strategy_age >= self.strategy_interval)
                    )
                    if needs_strategy:
                        z_idx, _, z_ent, _ = self.model.sample_strategy(context_gs, deterministic=deterministic)
                        self._prev_z = z_idx.detach()
                        self._strategy_age = 0
                    else:
                        z_idx = self._prev_z.to(self.device)
                        z_ent = z_dist.entropy()
                    z_probs = torch.softmax(z_logits, dim=-1)
                self._last_strategy_z = z_idx.detach().cpu()
                self._last_strategy_probs = z_probs.detach().cpu()
                self._last_strategy_entropy = z_ent.detach().cpu()
                self._last_strategy_resampled = bool(needs_strategy)
                action_tensor, _, _, _ = self.model.act(
                    obs_t,
                    context_gs,
                    deterministic=deterministic,
                    z_idx=z_idx,
                )
                self._strategy_age += 1
            else:
                batch = int(obs_t["grid"].shape[0])
                global_state = self._global_state_tensor(batched, batch)
                action_tensor, _, _, _ = self.model.act(
                    obs_t, global_state, deterministic=deterministic, z_idx=None
                )
        actions_np = action_tensor.detach().cpu().numpy().astype(np.int64)
        if actions_np.shape[0] == 1:
            return actions_np[0], None
        return actions_np, None

    def entropy(self, obs: Dict[str, np.ndarray]) -> float:
        """Mean summed action-head entropy for a batch of observations."""
        batched = self._batched_obs(obs)
        obs_t = self._tensor_obs(batched)
        with torch.no_grad():
            z_idx = None
            z_entropy = torch.zeros((obs_t["grid"].shape[0],), device=self.device)
            if self.model.uses_latent_strategy:
                batch = int(obs_t["grid"].shape[0])
                if self.fixed_latent_strategy:
                    z_idx = self._fixed_strategy_tensor(batch)
                else:
                    global_state = self._global_state_tensor(batched, batch)
                    tracker = self._get_temporal_tracker(batch)
                    context_gs = tracker.get_current_context(global_state)
                    z_idx, _, z_entropy, _ = self.model.sample_strategy(context_gs, deterministic=True)
            logits = self.model._mask_logits(self.model.policy_logits(obs_t, z_idx=z_idx), obs_t.get("mask"))
            entropy = torch.stack([dist.entropy() for dist in self.model._categoricals(logits)], dim=0).sum(dim=0)
        return float((entropy + z_entropy).mean().detach().cpu().item())

    def strategy_info(self) -> dict[str, Any]:
        """Return the most recent latent strategy diagnostics for single-env evaluation."""
        if not self.model.uses_latent_strategy or self._last_strategy_z is None:
            return {}
        z = self._last_strategy_z.reshape(-1)
        probs = self._last_strategy_probs
        entropy = self._last_strategy_entropy
        out: dict[str, Any] = {
            "strategy": int(z[0].item()),
            "strategy_batch": [int(v) for v in z.tolist()],
            "strategy_resampled": bool(self._last_strategy_resampled),
        }
        if self.fixed_latent_strategy:
            out["strategy_fixed"] = True
        if probs is not None and probs.numel() > 0:
            p0 = probs.reshape(probs.shape[0], -1)[0]
            out["strategy_k"] = int(p0.numel())
            for idx, prob in enumerate(p0.tolist()):
                out[f"strategy_prob_{idx}"] = float(prob)
        if entropy is not None and entropy.numel() > 0:
            out["strategy_entropy"] = float(entropy.reshape(-1)[0].item())
        return out


def _effective_latent_aux_return_head(cfg: Any) -> bool:
    """Whether A2 auxiliary per-z return head is enabled (new or legacy checkpoint / config keys)."""
    if isinstance(cfg, dict):
        if "latent_strategy_aux_return_head" in cfg:
            return bool(cfg["latent_strategy_aux_return_head"])
        return bool(cfg.get("latent_strategy_q_head", False))
    return bool(getattr(cfg, "latent_strategy_aux_return_head", False)) or bool(
        getattr(cfg, "latent_strategy_q_head", False)
    )


def _effective_latent_aux_return_coef(cfg: Any) -> float:
    if isinstance(cfg, dict):
        if "latent_strategy_aux_return_coef" in cfg:
            return max(0.0, float(cfg["latent_strategy_aux_return_coef"] or 0.0))
        return max(0.0, float(cfg.get("latent_strategy_q_coef", 1.0) or 0.0))
    return max(
        0.0,
        float(
            getattr(
                cfg,
                "latent_strategy_aux_return_coef",
                getattr(cfg, "latent_strategy_q_coef", 1.0),
            )
            or 0.0
        ),
    )


def _remap_legacy_strategy_aux_head_state_dict(sd: Mapping[str, Any]) -> dict[str, Any]:
    """Map ``strategy_q_head`` module weights to ``strategy_aux_return_head`` (older checkpoints)."""
    out = dict(sd)
    old_prefix = "strategy_q_head"
    new_prefix = "strategy_aux_return_head"
    for k in list(out.keys()):
        if k == old_prefix:
            nk = new_prefix
        elif k.startswith(old_prefix + "."):
            nk = new_prefix + k[len(old_prefix) :]
        else:
            continue
        if nk not in out:
            out[nk] = out[k]
        del out[k]
    return out


def _model_kwargs_from_cfg(cfg: Any) -> dict[str, Any]:
    if not isinstance(cfg, dict):
        return {}
    kwargs: dict[str, Any] = {
        "actor_cnn_feature_dim": int(cfg.get("actor_cnn_feature_dim", 128)),
    }
    if bool(cfg.get("use_latent_strategy", False)):
        kwargs.update(
            {
                "latent_k": int(cfg.get("latent_k", 4)),
                "z_embed_dim": int(cfg.get("latent_z_embed_dim", 16)),
                "strategy_hidden_dim": int(cfg.get("latent_strategy_hidden", 128)),
                "critic_hidden_dim": int(cfg.get("latent_vf_hidden", 128)),
                "use_strategy_aux_return_head": _effective_latent_aux_return_head(cfg),
                "strategy_tau": float(cfg.get("latent_strategy_tau", 1.0) or 1.0),
            }
        )
    return kwargs


# Intentional, stable split from ``PPOConfig.seed`` (E3 / trace §13). Do not “tweak” without a note.
# Decimal: 268435469 (strategy) and 536870955 (action); masked with ``& 0xFFFF_FFFF``.
STRATEGY_GENERATOR_SEED_OFFSET = 0x1_0000_00D
ACTION_GENERATOR_SEED_OFFSET = 0x2_0000_02B

FORCED_Z_PROFILE_MAX_ROWS = 4096
FORCED_Z_MACRO_ACTIONS: tuple[tuple[int, str], ...] = (
    (int(MacroAction.GO_TO), "go_to"),
    (int(MacroAction.GRAB_MINE), "grab_mine"),
    (int(MacroAction.GET_FLAG), "get_flag"),
    (int(MacroAction.PLACE_MINE), "place_mine"),
    (int(MacroAction.GO_HOME), "go_home"),
)


E3_STEP_TELEMETRY_FIELDS: tuple[str, ...] = (
    "update",
    "rollout_step",
    "env_id",
    "global_step",
    "z_t",
    "q_phi_entropy",
    "q_phi_argmax",
    "switched",
    "game_phase",
    "team_phase",
    "score_outcome",
    "stalemate_frac",
    "opponent_id",
    "phase_id",
    "blue_ahead",
) + BEHAVIOR_TELEMETRY_NAMES + (
    "spread_bucket",
    "role_bucket",
    "pressure_bucket",
    "attack_defense_ratio_bucket",
)

CUSTOM_PPO_FORMAT = "custom_ppo_cnn_v1"
CUSTOM_PPO_LATENT_FORMAT = "custom_ppo_latent_cnn_v1"
CUSTOM_PPO_ACTOR_ARCH = "cnn_mlp"
CUSTOM_PPO_VEC_SCHEMA_VERSION = 1

# When renaming metrics columns, old CSV headers may still use the legacy name; see ``_write_csv_row``.
_METRICS_CSV_LEGACY_COLUMN_FILL: dict[str, str] = {"strategy_aux_return_loss": "strategy_q_loss"}

# Columns for MI(z; opponent) and episode_opp{idx}_z* (OP1 … OP5_RUSHER, OP6, OP7).
SCRIPTED_OPPONENT_MI_COUNT: int = 7


def apply_deterministic_sampling_generators(
    model: "SharedActorCentralizedCritic",
    seed: int,
    *,
    device: torch.device | str,
) -> None:
    """Attach separate :class:`torch.Generator` copies for team-strategy vs per-head action sampling.

    Extra strategy draws for ``z`` would otherwise advance the *same* default RNG that drives
    action samples. Fixed sub-seeds derived from ``seed`` keep the two streams independent
    (E3 fairness: action RNG order not stolen by ``q_phi(z|s)`` when latent is on).

    Sub-seed protocol (intentional and stable)::

        strategy: (int(seed) + STRATEGY_GENERATOR_SEED_OFFSET) & 0xFFFF_FFFF
        action:   (int(seed) + ACTION_GENERATOR_SEED_OFFSET) & 0xFFFF_FFFF
    """
    dev = torch.device(device)
    g_s = torch.Generator(device=dev)
    g_s.manual_seed((int(seed) + STRATEGY_GENERATOR_SEED_OFFSET) & 0xFFFF_FFFF)
    g_a = torch.Generator(device=dev)
    g_a.manual_seed((int(seed) + ACTION_GENERATOR_SEED_OFFSET) & 0xFFFF_FFFF)
    model.set_sampling_generators(strategy=g_s, action=g_a)


def load_custom_ppo_policy(
    path: str,
    observation_space,
    action_space,
    *,
    device: str | torch.device = "cpu",
) -> CustomPPOInferencePolicy:
    """Load a policy checkpoint produced by :class:`CustomPPOTrainer` for inference."""
    device_t = torch.device(device)
    payload = _torch_load_checkpoint(path, map_location=device_t)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError("Not a custom PPO checkpoint.")
    _assert_compatible_global_state_dim(payload, path)
    model = SharedActorCentralizedCritic(
        observation_space,
        action_space,
        **_model_kwargs_from_cfg(payload.get("cfg") or {}),
    ).to(device_t)
    model.load_state_dict(_remap_legacy_strategy_aux_head_state_dict(payload["model_state_dict"]))
    ckpt_cfg = payload.get("cfg") or {}
    if isinstance(ckpt_cfg, dict) and "seed" in ckpt_cfg:
        apply_deterministic_sampling_generators(model, int(ckpt_cfg["seed"]), device=device_t)
    return CustomPPOInferencePolicy(model, device=device_t, cfg=ckpt_cfg)


class SharedActorCentralizedCritic(nn.Module):
    """Shared decentralized actor with an optional latent team strategy."""

    def __init__(
        self,
        observation_space,
        action_space,
        *,
        actor_hidden_dim: int = 256,
        actor_cnn_feature_dim: int = 128,
        critic_hidden_dim: int = 128,
        latent_k: int = 0,
        z_embed_dim: int = 16,
        strategy_hidden_dim: int = 128,
        use_strategy_aux_return_head: bool = False,
        strategy_tau: float = 1.0,
    ) -> None:
        super().__init__()
        grid_shape = tuple(int(v) for v in observation_space.spaces["grid"].shape)
        vec_shape = tuple(int(v) for v in observation_space.spaces["vec"].shape)
        if len(grid_shape) != 4:
            raise ValueError(f"Expected tokenized grid shape (N, C, H, W), got {grid_shape!r}")
        if len(vec_shape) != 2:
            raise ValueError(f"Expected tokenized vec shape (N, V), got {vec_shape!r}")

        self.n_agents = int(grid_shape[0])
        self.vec_dim = int(vec_shape[1])
        c, h, w = int(grid_shape[1]), int(grid_shape[2]), int(grid_shape[3])
        self.grid_shape = (c, h, w)
        self.actor_cnn = CNNEncoder(self.grid_shape, feature_dim=int(actor_cnn_feature_dim))
        self.actor_cnn_feature_dim = int(self.actor_cnn.feature_dim)
        self._scalar_per_agent = self.vec_dim
        self._local_actor_in_dim = self.actor_cnn_feature_dim + self._scalar_per_agent
        self.action_dims = tuple(int(v) for v in getattr(action_space, "nvec", []))
        if len(self.action_dims) % self.n_agents != 0:
            raise ValueError("MultiDiscrete action heads must divide evenly across agents.")
        self.heads_per_agent = len(self.action_dims) // self.n_agents
        self.per_agent_action_dims = self.action_dims[: self.heads_per_agent]
        for idx in range(self.n_agents):
            start = idx * self.heads_per_agent
            end = start + self.heads_per_agent
            if self.action_dims[start:end] != self.per_agent_action_dims:
                raise ValueError("All agents must share the same macro/target action heads.")
        self.per_agent_logits = int(sum(self.per_agent_action_dims))
        self.joint_action_onehot_dim = int(sum(self.action_dims))
        self.latent_k = max(0, int(latent_k))
        self.uses_latent_strategy = self.latent_k > 0
        self.z_embed_dim = int(z_embed_dim) if self.uses_latent_strategy else 0
        self.use_strategy_aux_return_head = bool(use_strategy_aux_return_head) and self.uses_latent_strategy
        self.strategy_tau = max(1e-3, float(strategy_tau))

        self.global_state_dim = CONTEXT_STATE_DIM if self.uses_latent_strategy else GLOBAL_STATE_DIM

        if self.uses_latent_strategy:
            strategy_net = StrategyEncoder(
                state_dim=self.global_state_dim,
                latent_k=self.latent_k,
                hidden=int(strategy_hidden_dim),
            )
            if self.use_strategy_aux_return_head:
                self.strategy_aux_return_head = strategy_net
                self.strategy_encoder = None
            else:
                self.strategy_encoder = strategy_net
                self.strategy_aux_return_head = None
            # Doc IMPLEMENTATION §7: nn.Embedding(K, d_z); no special init in the spec.
            self.strategy_embedding = nn.Embedding(self.latent_k, self.z_embed_dim)
        else:
            self.strategy_encoder = None
            self.strategy_aux_return_head = None
            self.strategy_embedding = None

        # Decentralized policy: CNN(grid) is concatenated with per-agent scalar features (+ z_emb), never `GLOBAL_STATE_DIM`.
        self._decentralized_actor_in_dim = int(
            self._local_actor_in_dim + (self.z_embed_dim if self.uses_latent_strategy else 0)
        )
        actor_in = self._decentralized_actor_in_dim
        # Doc IMPLEMENTATION §7: 256–256 MLP; no custom init in the spec (default Linear init).
        self.actor_body = nn.Sequential(
            nn.Linear(int(actor_in), int(actor_hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(actor_hidden_dim), int(actor_hidden_dim)),
            nn.ReLU(),
        )
        self.actor_head = nn.Linear(int(actor_hidden_dim), self.per_agent_logits)
        critic_extra_dim = self.joint_action_onehot_dim + self.latent_k if self.uses_latent_strategy else 0
        self.critic = CentralizedCritic(
            global_state_dim=self.global_state_dim,
            hidden_dim=int(critic_hidden_dim),
            extra_dim=critic_extra_dim,
        )
        self.q_phi_input_dim = self._strategy_context_dim()
        self.critic_context_dim = int(self.critic.global_state_dim)
        self.critic_z_dim = int(self.latent_k) if self.uses_latent_strategy else 0
        self.critic_joint_action_dim = int(self.joint_action_onehot_dim) if self.uses_latent_strategy else 0
        self.actor_input_dim = int(self._decentralized_actor_in_dim)
        self._assert_input_contracts()
        # Optional: separate ``torch.Generator`` streams so q_\phi(z|s) sampling does not advance
        # the same RNG as per-head action Categoricals (fairer E3 vs no-latent; see docs).
        self._sampling_gen_strategy: Optional[torch.Generator] = None
        self._sampling_gen_action: Optional[torch.Generator] = None

    def _strategy_context_dim(self) -> int:
        if not self.uses_latent_strategy:
            return 0
        source = self.strategy_aux_return_head if self.use_strategy_aux_return_head else self.strategy_encoder
        if source is None:
            raise AssertionError("latent strategy enabled but q_phi module is missing")
        dim = getattr(source, "state_dim", None)
        if dim is not None:
            return int(dim)
        first = getattr(source, "net", [None])[0]
        if isinstance(first, nn.Linear):
            return int(first.in_features)
        raise AssertionError("could not resolve q_phi input dim")

    def _assert_input_contracts(self) -> None:
        actor_expected = int(self.actor_cnn_feature_dim) + int(self._scalar_per_agent)
        if self.uses_latent_strategy:
            actor_expected += int(self.z_embed_dim)
            if int(self.global_state_dim) != int(CONTEXT_STATE_DIM):
                raise AssertionError(
                    f"latent q_phi/critic must use temporal context dim {CONTEXT_STATE_DIM}, "
                    f"got {self.global_state_dim}"
                )
            if int(self.q_phi_input_dim) != int(CONTEXT_STATE_DIM):
                raise AssertionError(
                    f"q_phi_input_dim={self.q_phi_input_dim} does not match temporal_context_dim={CONTEXT_STATE_DIM}"
                )
            if int(self.critic.global_state_dim) != int(CONTEXT_STATE_DIM):
                raise AssertionError(
                    f"critic_context_dim={self.critic.global_state_dim} does not match temporal_context_dim={CONTEXT_STATE_DIM}"
                )
            if int(self.critic.extra_dim) != int(self.joint_action_onehot_dim + self.latent_k):
                raise AssertionError(
                    "latent critic extra input must be joint action one-hot plus z one-hot "
                    f"({self.joint_action_onehot_dim}+{self.latent_k}), got {self.critic.extra_dim}"
                )
        else:
            if int(self.global_state_dim) != int(GLOBAL_STATE_DIM):
                raise AssertionError(f"no-latent critic context dim must be {GLOBAL_STATE_DIM}, got {self.global_state_dim}")
            if int(self.q_phi_input_dim) != 0:
                raise AssertionError(f"q_phi_input_dim must be 0 when latent is disabled, got {self.q_phi_input_dim}")
            if int(self.critic.global_state_dim) != int(GLOBAL_STATE_DIM):
                raise AssertionError(
                    f"critic_context_dim={self.critic.global_state_dim} does not match base_global_state_dim={GLOBAL_STATE_DIM}"
                )
            if int(self.critic.extra_dim) != 0:
                raise AssertionError(f"no-latent critic extra dim must be 0, got {self.critic.extra_dim}")

        if int(self._decentralized_actor_in_dim) != actor_expected:
            raise AssertionError(
                f"actor_input_dim={self._decentralized_actor_in_dim} must equal local obs + z embedding width "
                f"{actor_expected}"
            )
        first_actor = self.actor_body[0]
        if not isinstance(first_actor, nn.Linear) or int(first_actor.in_features) != actor_expected:
            got = getattr(first_actor, "in_features", None)
            raise AssertionError(f"actor MLP first layer input {got} != decentralized actor input {actor_expected}")
        # Defense-in-depth: actor input must not match the *temporal* context either.
        # CONTEXT_STATE_DIM is the q_phi/critic-only width; if it accidentally lined up with
        # the actor concat width, we want to fail loudly rather than silently train a non-decentralized policy.
        if int(self._decentralized_actor_in_dim) == int(CONTEXT_STATE_DIM):
            raise AssertionError(
                f"actor_input_dim={self._decentralized_actor_in_dim} equals temporal_context_dim={CONTEXT_STATE_DIM}; "
                "actor must consume local obs + z embedding only, never the centralized temporal context."
            )

    def input_dim_contract(self) -> dict[str, int]:
        self._assert_input_contracts()
        return {
            "base_global_state_dim": int(GLOBAL_STATE_DIM),
            "temporal_context_dim": int(CONTEXT_STATE_DIM),
            "q_phi_input_dim": int(self.q_phi_input_dim),
            "critic_context_dim": int(self.critic_context_dim),
            "actor_input_dim": int(self.actor_input_dim),
            "critic_extra_dim": int(self.critic.extra_dim),
            "critic_z_dim": int(self.critic_z_dim),
            "critic_joint_action_dim": int(self.critic_joint_action_dim),
        }

    def set_sampling_generators(
        self,
        *,
        strategy: Optional[torch.Generator] = None,
        action: Optional[torch.Generator] = None,
    ) -> None:
        """Set dedicated RNGs for strategy vs. action sampling. ``None`` = PyTorch default (shared global) for that stream."""
        self._sampling_gen_strategy = strategy
        self._sampling_gen_action = action

    @staticmethod
    def _categorical_argmax_or_sample(
        dist: Categorical, *, deterministic: bool, generator: Optional[torch.Generator]
    ) -> torch.Tensor:
        if deterministic:
            return torch.argmax(dist.logits, dim=-1)
        if generator is not None:
            # ``Categorical.sample(generator=)`` is not available in all supported PyTorch versions;
            # ``torch.multinomial`` matches the same distribution and honors ``generator``.
            logits = dist.logits
            probs = torch.softmax(logits, dim=-1)
            return torch.multinomial(probs, 1, replacement=True, generator=generator).squeeze(-1)
        return dist.sample()

    def strategy_logits(self, global_state: torch.Tensor) -> torch.Tensor:
        """Return ``q_phi(z | s)`` logits for latent strategy mode."""
        if not self.uses_latent_strategy:
            raise RuntimeError("strategy_logits is only available when latent strategy is enabled.")
        if global_state.dim() != 2 or int(global_state.shape[1]) != int(self.q_phi_input_dim):
            raise AssertionError(
                f"q_phi expected context shape (B, {self.q_phi_input_dim}), got {tuple(global_state.shape)}"
            )
        if self.use_strategy_aux_return_head:
            return self.strategy_aux_return_predictions(global_state) / self.strategy_tau
        if self.strategy_encoder is None:
            raise RuntimeError("strategy encoder is not initialized.")
        return self.strategy_encoder(global_state.float())

    def strategy_aux_return_predictions(self, global_state: torch.Tensor) -> torch.Tensor:
        """A2 auxiliary: per-z scalar predictions from the shared trunk, shape ``(B, K)``.

        These are **not** a full action-value :math:`Q(s,\\mathbf{a}, z)` and are not trained with
        off-policy Bellman targets; they only supply an optional supervised signal on the **sampled**
        strategy index (see plan A2 / auxiliary return regression).
        """
        if not self.uses_latent_strategy or self.strategy_aux_return_head is None:
            raise RuntimeError(
                "strategy_aux_return_predictions is only available when the A2 auxiliary return head is enabled."
            )
        return self.strategy_aux_return_head(global_state.float())

    def sample_strategy(
        self,
        global_state: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample or greedily choose team strategy indices from ``q_phi(z | s)``."""
        logits = self.strategy_logits(global_state)
        dist = Categorical(logits=logits)
        z_idx = self._categorical_argmax_or_sample(
            dist, deterministic=deterministic, generator=self._sampling_gen_strategy
        )
        return z_idx.long(), dist.log_prob(z_idx), dist.entropy(), logits

    def policy_logits(self, obs: Dict[str, torch.Tensor], z_idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return flattened MultiDiscrete logits with shape ``(B, sum(action_dims))``."""
        grid = obs["grid"].float()
        vec = obs["vec"].float()
        if grid.dim() != 5:
            raise ValueError(f"grid must have shape (B, N, C, H, W), got {tuple(grid.shape)}")
        if vec.dim() != 3:
            raise ValueError(f"vec must have shape (B, N, V), got {tuple(vec.shape)}")
        batch = int(grid.shape[0])
        if int(grid.shape[1]) != self.n_agents or tuple(int(v) for v in grid.shape[2:]) != self.grid_shape:
            raise ValueError(
                f"grid must have shape (B, {self.n_agents}, {self.grid_shape[0]}, "
                f"{self.grid_shape[1]}, {self.grid_shape[2]}), got {tuple(grid.shape)}"
            )
        if int(vec.shape[1]) != self.n_agents or int(vec.shape[2]) != self.vec_dim:
            raise ValueError(f"vec must have shape (B, {self.n_agents}, {self.vec_dim}), got {tuple(vec.shape)}")
        cnn_features = self.actor_cnn(grid.reshape(batch * self.n_agents, *self.grid_shape))
        cnn_features = cnn_features.reshape(batch, self.n_agents, self.actor_cnn_feature_dim)
        vloc = vec.float()
        agent_mask = obs.get("agent_mask")
        if agent_mask is not None:
            if agent_mask.dim() == 1:
                agent_mask = agent_mask.unsqueeze(0)
            mask = agent_mask.float().unsqueeze(-1)
            cnn_features = cnn_features * mask
            vloc = vloc * mask
        local_obs = torch.cat([cnn_features, vloc], dim=-1)
        actor_inputs: list[torch.Tensor] = [local_obs]
        if self.uses_latent_strategy:
            if z_idx is None:
                raise ValueError("z_idx is required when latent strategy is enabled.")
            z = z_idx.long().reshape(-1).clamp(min=0, max=self.latent_k - 1)
            if z.shape[0] != batch:
                raise ValueError(f"z_idx must have shape ({batch},), got {tuple(z_idx.shape)}")
            assert self.strategy_embedding is not None
            z_emb = self.strategy_embedding(z).unsqueeze(1).expand(batch, self.n_agents, self.z_embed_dim)
            actor_inputs.append(z_emb)

        actor_in = torch.cat(actor_inputs, dim=-1)
        d_in = int(actor_in.shape[-1])
        if d_in != self._decentralized_actor_in_dim:
            raise AssertionError(
                f"decentralized actor expects concat width {self._decentralized_actor_in_dim} "
                f"(cnn_features + scalars + z), got {d_in}"
            )
        if d_in == int(GLOBAL_STATE_DIM):
            raise AssertionError("actor input width equals GLOBAL_STATE_DIM; policy must not consume global state")
        hidden = self.actor_body(actor_in.reshape(batch * self.n_agents, -1))
        per_agent_logits = self.actor_head(hidden).reshape(batch, self.n_agents, self.per_agent_logits)
        return per_agent_logits.reshape(batch, self.n_agents * self.per_agent_logits)

    def _joint_action_one_hot(self, actions: torch.Tensor) -> torch.Tensor:
        actions = actions.long()
        if actions.dim() == 1:
            actions = actions.unsqueeze(0)
        chunks = []
        for col, dim in enumerate(self.action_dims):
            action = actions[:, col].clamp(min=0, max=dim - 1)
            chunks.append(F.one_hot(action, num_classes=dim).float())
        return torch.cat(chunks, dim=-1)

    def _critic_extra(self, actions: Optional[torch.Tensor], z_idx: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if not self.uses_latent_strategy:
            return None
        if actions is None or z_idx is None:
            raise ValueError("actions and z_idx are required by the latent action-conditioned **value** critic.")
        z = z_idx.long().reshape(-1).clamp(min=0, max=self.latent_k - 1)
        z_one_hot = F.one_hot(z, num_classes=self.latent_k).float()
        extra = torch.cat([self._joint_action_one_hot(actions).to(z_one_hot.device), z_one_hot], dim=-1)
        expected = int(self.joint_action_onehot_dim + self.latent_k)
        if extra.dim() != 2 or int(extra.shape[1]) != expected:
            raise AssertionError(f"critic extra must be joint_action_onehot + z_onehot width {expected}, got {tuple(extra.shape)}")
        z_slice = extra[:, -self.latent_k :]
        z_sum = z_slice.sum(dim=-1)
        if int(z_slice.shape[1]) != int(self.latent_k) or not torch.allclose(z_sum, torch.ones_like(z_sum), atol=1e-6):
            raise AssertionError("critic input is missing the terminal z one-hot slice")
        return extra

    def values(
        self,
        global_state: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        z_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return scalar :math:`V_\\phi(s,\\mathbf{a},z)` with shape ``(B,)`` (PPO/GAE target)."""
        if global_state.dim() != 2 or int(global_state.shape[1]) != int(self.critic_context_dim):
            raise AssertionError(
                f"critic expected context shape (B, {self.critic_context_dim}), got {tuple(global_state.shape)}"
            )
        return self.critic(global_state.float(), extra=self._critic_extra(actions, z_idx)).squeeze(-1)

    def _mask_logits(self, logits: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is None:
            return logits
        if mask.dim() == 1:
            mask = mask.unsqueeze(0)
        mask = mask.float()
        masked_chunks = []
        offset = 0
        for dim in self.action_dims:
            chunk = logits[:, offset : offset + dim]
            mask_chunk = mask[:, offset : offset + dim]
            if mask_chunk.shape[1] < dim:
                pad = torch.ones((mask.shape[0], dim - mask_chunk.shape[1]), device=mask.device)
                mask_chunk = torch.cat([mask_chunk, pad], dim=1)
            all_invalid = mask_chunk.sum(dim=1, keepdim=True) <= 0.0
            safe_mask = torch.where(all_invalid, torch.ones_like(mask_chunk), mask_chunk)
            masked_chunks.append(chunk.masked_fill(safe_mask <= 0.0, -1e8))
            offset += dim
        return torch.cat(masked_chunks, dim=1)

    def _categoricals(self, logits: torch.Tensor) -> Iterable[Categorical]:
        offset = 0
        for dim in self.action_dims:
            yield Categorical(logits=logits[:, offset : offset + dim])
            offset += dim

    def _log_prob_entropy(self, logits: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        actions = actions.long()
        log_probs = []
        entropies = []
        for col, dist in enumerate(self._categoricals(logits)):
            action = actions[:, col].clamp(min=0, max=dist.logits.shape[1] - 1)
            log_probs.append(dist.log_prob(action))
            entropies.append(dist.entropy())
        return torch.stack(log_probs, dim=0).sum(dim=0), torch.stack(entropies, dim=0).sum(dim=0)

    def act(
        self,
        obs: Dict[str, torch.Tensor],
        global_state: torch.Tensor,
        *,
        deterministic: bool = False,
        z_idx: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample or greedily select actions and return values/log-probs/entropy."""
        if self.uses_latent_strategy and z_idx is None:
            z_idx, _, _, _ = self.sample_strategy(global_state, deterministic=deterministic)
        logits = self._mask_logits(self.policy_logits(obs, z_idx=z_idx), obs.get("mask"))
        actions = []
        g_act = self._sampling_gen_action
        for dist in self._categoricals(logits):
            actions.append(
                self._categorical_argmax_or_sample(
                    dist, deterministic=deterministic, generator=g_act
                )
            )
        action_tensor = torch.stack(actions, dim=1)
        log_prob, entropy = self._log_prob_entropy(logits, action_tensor)
        values = self.values(global_state, actions=action_tensor, z_idx=z_idx)
        return action_tensor, values, log_prob, entropy

    def evaluate_actions(
        self,
        obs: Dict[str, torch.Tensor],
        global_state: torch.Tensor,
        actions: torch.Tensor,
        *,
        z_idx: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Evaluate fixed actions under the current policy."""
        logits = self._mask_logits(self.policy_logits(obs, z_idx=z_idx), obs.get("mask"))
        log_prob, entropy = self._log_prob_entropy(logits, actions)
        values = self.values(global_state, actions=actions, z_idx=z_idx)
        aux: dict[str, torch.Tensor] = {}
        if self.uses_latent_strategy:
            if z_idx is None:
                raise ValueError("z_idx is required when latent strategy is enabled.")
            z_logits = self.strategy_logits(global_state)
            z_dist = Categorical(logits=z_logits)
            z = z_idx.long().reshape(-1).clamp(min=0, max=self.latent_k - 1)
            aux["strategy_logits"] = z_logits
            aux["strategy_log_prob"] = z_dist.log_prob(z)
            aux["strategy_entropy"] = z_dist.entropy()
        return values, log_prob, entropy, aux


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
                    "use_strategy_aux_return_head": _effective_latent_aux_return_head(cfg),
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
        self.latent_strategy_aux_return_coef = max(0.0, _effective_latent_aux_return_coef(cfg))
        self.latent_strategy_aux_return_head = self.use_latent_strategy and _effective_latent_aux_return_head(cfg)
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
        # Optional E3: per-env-step z / q_phi / phase rows (set before long E3 runs; see E3_STEP_TELEMETRY_FIELDS).
        self._e3_step_telemetry_path = str(getattr(cfg, "e3_step_telemetry_path", "") or "")
        # SB3-style: ``tqdm`` bar updated every ``n_envs`` sim steps during ``collect_rollout`` only.
        self._sb3_rollout_pbar: Any = None
        self._last_obs: Optional[Dict[str, np.ndarray]] = None
        self._last_global_state: Optional[np.ndarray] = None
        self._last_context_state: Optional[torch.Tensor] = None
        self._current_z: Optional[torch.Tensor] = None
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
        self._z_kl_first_in_ep: Optional[torch.Tensor] = None
        self._prev_z_logits: Optional[torch.Tensor] = None
        self._decentralized_actor_contract_logged: bool = False
        mode_s = str(getattr(cfg, "mode", "") or "").strip().upper()
        self._opponent_randomize_training = (
            (mode_s == "OPPONENT_POOL" or bool(getattr(cfg, "opponent_randomize", False)))
            and curriculum is None
        )
        self._opponent_pool_tags: list[str] = (
            [str(x).strip().upper() for x in getattr(cfg, "opponent_pool", ())] if self._opponent_randomize_training else []
        )
        self._rng_opponent = np.random.default_rng(int(getattr(cfg, "seed", 0)) + 901)
        if self._opponent_randomize_training:
            if not self._opponent_pool_tags:
                raise ValueError(
                    "Opponent pool training (mode=OPPONENT_POOL or opponent_randomize) requires a non-empty "
                    "opponent_pool (e.g. OP1–OP3, OP5–OP7; OP4 optional with --allow-op4-in-training-pool)."
                )
            self.env._before_reset_indices_hook = self._hook_sample_training_opponent_before_reset

    def _reward_shaping_coef(self) -> float:
        if self.reward_shaping_decay_steps <= 0:
            return float(self.reward_shaping_coef_start)
        frac = min(1.0, max(0.0, float(self.global_step) / float(self.reward_shaping_decay_steps)))
        return float(self.reward_shaping_coef_start + frac * (self.reward_shaping_coef_end - self.reward_shaping_coef_start))

    def _log_decentralized_actor_contract_once(self) -> None:
        """One-time training log: policy actor is CNN(grid) + scalars + optional z, not global state."""
        self.log_input_dim_contract()

    def log_input_dim_contract(self) -> None:
        """Print the startup input-dimension contract once."""
        if self._decentralized_actor_contract_logged:
            return
        m = self.model
        assert isinstance(m, SharedActorCentralizedCritic)
        dims = m.input_dim_contract()
        print(
            "[PPO] Input dims: "
            f"base_global_state_dim={dims['base_global_state_dim']} "
            f"temporal_context_dim={dims['temporal_context_dim']} "
            f"q_phi_input_dim={dims['q_phi_input_dim']} "
            f"critic_context_dim={dims['critic_context_dim']} "
            f"actor_input_dim={dims['actor_input_dim']}"
        )
        if m.uses_latent_strategy:
            print(
                "[PPO] Decentralized actor contract: per-agent MLP input dim = "
                f"{m._decentralized_actor_in_dim} "
                f"(cnn {m.actor_cnn_feature_dim} + scalars {m._scalar_per_agent} + z_emb {m.z_embed_dim}); "
                f"global_state_dim={m.global_state_dim} is for q_phi/critic only."
            )
            print(
                "[PPO] Critic z contract: "
                f"context_dim={dims['critic_context_dim']} "
                f"joint_action_onehot_dim={dims['critic_joint_action_dim']} "
                f"z_onehot_dim={dims['critic_z_dim']} "
                f"critic_extra_dim={dims['critic_extra_dim']} "
                "z_present=True"
            )
        else:
            print(
                "[PPO] Decentralized actor contract: per-agent MLP input dim = "
                f"{m._decentralized_actor_in_dim} "
                f"(cnn {m.actor_cnn_feature_dim} + scalars {m._scalar_per_agent}, no z); "
                f"global_state_dim={m.global_state_dim} not used in policy."
            )
        self._log_plan_faithful_audit()
        self._decentralized_actor_contract_logged = True

    def _log_plan_faithful_audit(self) -> None:
        """One-time Summer-plan audit: confirm forbidden objectives are absent and flag optional add-ons.

        Forbidden by the Summer Implementation Plan and the current research direction:
        supervised router labels, opponent-ID heads, Gumbel-Softmax z, VAE losses, handcrafted strategy
        labels. None are implemented in this codebase; this log is a defense-in-depth assertion plus a
        printed reminder that they remain off. Optional plan §12 add-ons (aux return head, KL-consecutive,
        flag-triggered resample, fixed-z) are reported but not blocked.
        """
        if not self.model.uses_latent_strategy:
            return
        # Sanity-check that we have not silently grown any forbidden attribute on the model.
        for forbidden_attr in (
            "opponent_id_head",
            "opponent_classifier",
            "gumbel_softmax_z",
            "vae_z_head",
            "strategy_label_head",
            "supervised_router",
        ):
            if getattr(self.model, forbidden_attr, None) is not None:
                raise AssertionError(
                    f"plan-faithful audit: forbidden module '{forbidden_attr}' is attached to the model."
                )
        cfg = self.cfg
        optional = {
            "aux_return_head": bool(getattr(cfg, "latent_strategy_aux_return_head", False)),
            "kl_consecutive": float(getattr(cfg, "latent_kl_consecutive", 0.0) or 0.0) > 0.0,
            "resample_on_flag": bool(getattr(cfg, "latent_resample_on_flag", False)),
            "fixed_latent_strategy": bool(getattr(cfg, "fixed_latent_strategy", False)),
        }
        print(
            "[PPO] Summer-plan audit (latent on): no supervised router labels, no opponent-ID heads, "
            "no Gumbel-Softmax, no VAE losses, no handcrafted strategy labels."
        )
        extras_on = [name for name, on in optional.items() if on]
        if extras_on:
            print(
                "[PPO] Summer-plan audit: optional add-ons ENABLED "
                f"{extras_on} (not plan-faithful first-run; treat as intentional ablation)."
            )
        else:
            print(
                "[PPO] Summer-plan audit: optional add-ons "
                "(aux_return_head, kl_consecutive, resample_on_flag, fixed_latent_strategy) all OFF "
                "— plan-faithful first-run."
            )

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

    def _write_csv_row(
        self,
        path: str,
        fieldnames: list[str],
        row: dict[str, Any],
        *,
        legacy_column_fill: Optional[dict[str, str]] = None,
    ) -> None:
        """Append one row with a stable header; used for long-run audit telemetry."""
        if not path:
            return
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        legacy_fill = legacy_column_fill or {}
        nonempty = os.path.isfile(path) and os.path.getsize(path) > 0
        if nonempty:
            with open(path, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                old_fields = reader.fieldnames
                if old_fields is None:
                    raise ValueError(f"CSV schema mismatch for {path!r}: empty or invalid header.")
                old_list = list(old_fields)
                old_rows = list(reader)
            if old_list != fieldnames:
                dropped = [c for c in old_list if c not in fieldnames]
                if dropped:
                    allowed_old = set(legacy_fill.values())
                    if not (legacy_fill and set(dropped).issubset(allowed_old)):
                        raise ValueError(
                            f"CSV schema mismatch for {path!r}: existing columns dropped or renamed "
                            f"{dropped!r}; existing header {old_list!r} vs expected {fieldnames!r}. "
                            "Use a new output path or migrate manually."
                        )
                print(
                    f"[PPO] Migrating CSV (additive columns): {path}\n"
                    f"      was {len(old_list)} cols -> now {len(fieldnames)} cols; "
                    f"rewriting {len(old_rows)} row(s)."
                )
                with open(path, "w", newline="", encoding="utf-8") as wf:
                    writer = csv.DictWriter(wf, fieldnames=fieldnames, extrasaction="ignore")
                    writer.writeheader()
                    for r in old_rows:
                        out_row: dict[str, Any] = {}
                        for k in fieldnames:
                            v = r.get(k, "")
                            if v == "" and k in legacy_fill:
                                v = r.get(legacy_fill[k], "")
                            out_row[k] = v
                        writer.writerow(out_row)
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            if not nonempty:
                writer.writeheader()
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    def _ensure_additive_csv_header(self, path: str, fieldnames: list[str]) -> None:
        """Rewrite CSV when new columns are appended (additive-only; never drop old columns)."""
        if not path or not (os.path.isfile(path) and os.path.getsize(path) > 0):
            return
        with open(path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                return
            old_list = list(reader.fieldnames)
            old_rows = list(reader)
        if old_list == fieldnames:
            return
        dropped = [c for c in old_list if c not in fieldnames]
        if dropped:
            raise ValueError(
                f"E3 telemetry CSV schema mismatch for {path!r}: cannot drop columns {dropped!r}. "
                f"Use a new --e3 path or migrate manually."
            )
        if len(fieldnames) <= len(old_list):
            return
        print(
            f"[PPO] Migrating E3 step CSV (additive columns): {path}\n"
            f"      was {len(old_list)} cols -> now {len(fieldnames)} cols; "
            f"rewriting {len(old_rows)} row(s).",
            flush=True,
        )
        with open(path, "w", newline="", encoding="utf-8") as wf:
            writer = csv.DictWriter(wf, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for r in old_rows:
                writer.writerow({k: r.get(k, "") for k in fieldnames})

    def _append_e3_step_telemetry(
        self,
        *,
        rollout_step: int,
        global_step_at_step_end: int,
        decision_global_state_np: np.ndarray,
        z_t: torch.Tensor,
        prev_z: torch.Tensor,
        strategy_aux: dict[str, torch.Tensor],
        infos: list[Any],
        behavior_telemetry_np: np.ndarray,
        spread_bucket_np: np.ndarray,
        role_bucket_np: np.ndarray,
        pressure_bucket_np: np.ndarray,
        attack_defense_ratio_bucket_np: np.ndarray,
        blue_ahead_np: np.ndarray,
    ) -> None:
        """One row per env for this PPO step (optional E3 / §6.3 style histograms)."""
        if not self._e3_step_telemetry_path or not self.use_latent_strategy:
            return
        path = self._e3_step_telemetry_path
        d = os.path.dirname(os.path.abspath(path)) or "."
        os.makedirs(d, exist_ok=True)
        zt = z_t.detach().cpu().numpy()
        pz = prev_z.detach().cpu().numpy()
        zH = strategy_aux["z_entropy"].detach().cpu().numpy()
        zlog = strategy_aux["z_logits"].detach().cpu().numpy()
        am = zlog.argmax(axis=-1)
        n_e = int(zt.shape[0])
        assert int(decision_global_state_np.shape[0]) == n_e, (decision_global_state_np.shape, n_e)
        fields = list(E3_STEP_TELEMETRY_FIELDS)
        self._ensure_additive_csv_header(path, fields)
        skip_header = os.path.isfile(path) and os.path.getsize(path) > 0
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            if not skip_header:
                w.writeheader()
            upd = int(self._updates_completed)
            for e in range(n_e):
                info = dict(infos[e]) if e < len(infos) else {}
                gs_e = decision_global_state_np[e]
                sf = float(info.get("stalemate_frac", 0.0) or 0.0)
                pid = int(team_phase_id_from_global_state(gs_e, stalemate_frac=sf))
                row: dict[str, Any] = {
                    "update": upd,
                    "rollout_step": int(rollout_step),
                    "env_id": e,
                    "global_step": int(global_step_at_step_end),
                    "z_t": int(zt[e]),
                    "q_phi_entropy": float(zH[e]),
                    "q_phi_argmax": int(am[e]),
                    "switched": int(bool(int(zt[e]) != int(pz[e]))),
                    "game_phase": coarse_game_phase_from_global_state(gs_e),
                    "team_phase": team_phase_label_from_global_state(gs_e, stalemate_frac=sf),
                    "score_outcome": outcome_label_from_global_state(gs_e),
                    "stalemate_frac": sf,
                    "opponent_id": int(self._opponent_id_int_from_info(info)),
                    "phase_id": pid,
                    "blue_ahead": float(blue_ahead_np[e]),
                    "spread_bucket": int(spread_bucket_np[e]),
                    "role_bucket": int(role_bucket_np[e]),
                    "pressure_bucket": int(pressure_bucket_np[e]),
                    "attack_defense_ratio_bucket": int(attack_defense_ratio_bucket_np[e]),
                }
                for j, name in enumerate(BEHAVIOR_TELEMETRY_NAMES):
                    row[name] = float(behavior_telemetry_np[e, j])
                w.writerow({key: row.get(key, "") for key in fields})

    def _episode_fieldnames(self) -> list[str]:
        return [
            "episode_id",
            "run_id",
            "run_pid",
            "timesteps",
            "policy_update",
            "rollout_step",
            "latent_z",
            "curriculum_phase",
            "mode",
            "map_set",
            "opponent",
            "opponent_id",
            "success",
            "blue_score",
            "red_score",
            "win_margin",
            "decision_steps",
            "zone_coverage",
            "collision_free_episode",
            "collision_events_per_episode",
            "near_misses_per_episode",
            "time_to_first_score",
            "mean_inter_robot_dist",
            "reward_terminal",
            "reward_offense",
            "reward_pbrs",
            "reward_team",
            "reward_sparse",
            "reward_sparse_points",
            "reward_failure",
            "reward_total",
        ]

    def _update_fieldnames(self) -> list[str]:
        fields = [
            "update",
            "run_id",
            "run_pid",
            "timesteps",
            "episodes_completed",
            "wins",
            "losses",
            "draws",
            "win_rate",
            "rolling_win_rate_50ep",
            "rolling_win_rate_200ep",
            "rollout_reward_mean",
            "rollout_reward_std",
            "rollout_return_mean",
            "rollout_return_std",
            "rollout_episodes",
            "rollout_wins",
            "rollout_losses",
            "rollout_draws",
            "rollout_win_rate",
            "rollout_win_margin_mean",
            "rollout_blue_score_mean",
            "rollout_red_score_mean",
            "explained_variance",
            "reward_terminal_mean",
            "reward_offense_mean",
            "reward_pbrs_mean",
            "reward_team_mean",
            "reward_sparse_mean",
            "reward_sparse_points_mean",
            "reward_failure_mean",
            "reward_total_mean",
            "reward_outcome_mean",
            "reward_shaping_mean",
            "reward_shaping_to_outcome_abs_ratio",
            "reward_shaping_coef",
            "reward_failure_to_outcome_abs",
            "policy_loss",
            "value_loss",
            "value_loss_min",
            "value_loss_std",
            "value_loss_p10",
            "value_loss_p50",
            "value_loss_p90",
            "value_loss_max",
            "return_norm_mean",
            "return_norm_std",
            "return_norm_count",
            "entropy",
            "approx_kl",
            "clip_fraction",
            "grad_norm",
            "learning_rate",
            "strategy_entropy",
            "strategy_entropy_frac",
            "strategy_policy_loss",
            "strategy_approx_kl",
            "strategy_clip_fraction",
            "strategy_ratio_std",
            "strategy_aux_return_loss",
            "strategy_persist_loss",
            "strategy_grad_norm",
            "strategy_resample_count",
            "strategy_resample_fraction",
            "strategy_unique_count",
            "strategy_dominant",
            "strategy_switch_count",
            "strategy_switch_fraction",
            "strategy_wr_spread",
            "strategy_resample_fraction_rollout",
            "rollout_adv_std",
            "rollout_adv_std_at_z_switch",
            "rollout_adv_std_not_z_switch",
            "curriculum_phase",
            "curriculum_phase_idx",
            "curriculum_phase_episodes",
            "curriculum_phase_win_rate",
        ]
        if self.use_latent_strategy:
            fields.append("strategy_kl")
            fields.extend(f"strategy_occupancy_{idx}" for idx in range(self.latent_k))
            for idx in range(self.latent_k):
                fields.extend(
                    [
                        f"episode_z_{idx}_count",
                        f"episode_z_{idx}_win_rate",
                        f"episode_z_{idx}_blue_score_mean",
                        f"episode_z_{idx}_red_score_mean",
                        f"episode_z_{idx}_win_margin_mean",
                    ]
                )
            for idx in range(self.latent_k):
                fields.extend(
                    [
                        f"strategy_resample_adv_mean_z{idx}",
                        f"strategy_resample_adv_std_z{idx}",
                        f"strategy_resample_adv_n_z{idx}",
                    ]
                )
            fields.append("latent_mi_z_opponent_nats")
            fields.append("latent_mi_z_phase_nats")
            fields.append("latent_mi_z_outcome_nats")
            fields.append("latent_mi_z_spread_bucket_nats")
            fields.append("latent_mi_z_role_bucket_nats")
            fields.append("latent_mi_z_pressure_bucket_nats")
            fields.append("latent_mi_z_attack_defense_ratio_bucket_nats")
            for r in range(N_ROLE_BUCKET_MI):
                for z_idx in range(self.latent_k):
                    fields.append(f"latent_role{r}_z{z_idx}_frac")
                fields.append(f"latent_role{r}_switch_mean")
            fields.extend(
                [
                    "latent_switch_rate_blue_ahead",
                    "latent_switch_rate_blue_trail",
                    "latent_reward_sum_5_after_z_switch_mean",
                ]
            )
            for p in range(len(TEAM_PHASES)):
                for z_idx in range(self.latent_k):
                    fields.append(f"latent_phase{p}_z{z_idx}_frac")
                fields.extend(
                    [
                        f"latent_phase{p}_switch_mean",
                        f"latent_phase{p}_blue_ahead_mean",
                        f"latent_phase{p}_capture_step_mean",
                        f"q_phi_phase{p}_entropy_mean",
                    ]
                )
                for z_idx in range(self.latent_k):
                    fields.append(f"q_phi_phase{p}_z{z_idx}_prob_mean")
            fields.append("latent_behavior_diversity_l2_mean")
            for z_idx in range(self.latent_k):
                for name in BEHAVIOR_TELEMETRY_NAMES:
                    fields.append(f"latent_z{z_idx}_behavior_{name}_mean")
            fields.append("forced_z_macro_jsd_mean")
            for z_idx in range(self.latent_k):
                for _action_id, action_name in FORCED_Z_MACRO_ACTIONS:
                    fields.append(f"forced_z{z_idx}_macro_{action_name}_prob")
                fields.append(f"forced_z{z_idx}_macro_entropy")
            for o_idx in range(SCRIPTED_OPPONENT_MI_COUNT):
                for z_idx in range(self.latent_k):
                    fields.append(f"strategy_occupancy_op{o_idx}_z{z_idx}")
            for o_idx in range(SCRIPTED_OPPONENT_MI_COUNT):
                for z_idx in range(self.latent_k):
                    fields.extend(
                        [
                            f"episode_opp{o_idx}_z{z_idx}_count",
                            f"episode_opp{o_idx}_z{z_idx}_win_rate",
                        ]
                    )
        return fields

    def _write_episode_metrics(
        self,
        info: dict[str, Any],
        *,
        blue_score: int,
        red_score: int,
        timestep: int,
        rollout_step: Optional[int] = None,
        latent_z: Optional[int] = None,
    ) -> None:
        if not self.episode_csv_path:
            return
        er = info.get("episode_result") if isinstance(info.get("episode_result"), dict) else {}
        row = {
            "episode_id": self._episodes_completed,
            "run_id": self.run_id,
            "run_pid": self.run_pid,
            "timesteps": int(timestep),
            "policy_update": int(self._updates_completed),
            "rollout_step": "" if rollout_step is None else int(rollout_step),
            "latent_z": "" if latent_z is None else int(latent_z),
            "curriculum_phase": str(info.get("phase", "")),
            "mode": str(getattr(self.cfg, "mode", "FIXED_OPPONENT")),
            "map_set": str(info.get("map_set", getattr(self.cfg, "map_set", "train"))).lower(),
            "opponent": self._opponent_legend(info),
            "opponent_id": self._opponent_id_csv_from_info(info),
            "success": 1 if blue_score > red_score else 0,
            "blue_score": int(blue_score),
            "red_score": int(red_score),
            "win_margin": int(blue_score) - int(red_score),
            "decision_steps": int(er.get("decision_steps", info.get("decision_steps", 0)) or 0),
            "zone_coverage": float(er.get("zone_coverage", 0.0) or 0.0),
            "collision_free_episode": int(er.get("collision_free_episode", 1) or 0),
            "collision_events_per_episode": int(er.get("collision_events_per_episode", 0) or 0),
            "near_misses_per_episode": int(er.get("near_misses_per_episode", 0) or 0),
            "time_to_first_score": er.get("time_to_first_score", ""),
            "mean_inter_robot_dist": er.get("mean_inter_robot_dist", ""),
            "reward_terminal": float(er.get("reward_terminal", info.get("reward_terminal", 0.0)) or 0.0),
            "reward_offense": float(er.get("reward_offense", info.get("reward_offense", 0.0)) or 0.0),
            "reward_pbrs": float(er.get("reward_pbrs", info.get("reward_pbrs", 0.0)) or 0.0),
            "reward_team": float(er.get("reward_team", info.get("reward_team", 0.0)) or 0.0),
            "reward_sparse": float(er.get("reward_sparse", info.get("reward_sparse", 0.0)) or 0.0),
            "reward_sparse_points": float(
                er.get("reward_sparse_points", info.get("reward_sparse_points", info.get("sparse_points", 0.0))) or 0.0
            ),
            "reward_failure": float(er.get("reward_failure", info.get("reward_failure", 0.0)) or 0.0),
            "reward_total": float(er.get("reward_total", info.get("reward_total", 0.0)) or 0.0),
        }
        self._write_csv_row(self.episode_csv_path, self._episode_fieldnames(), row)

    @staticmethod
    def _explained_variance(values: torch.Tensor, returns: torch.Tensor) -> float:
        y_pred = values.detach().float().reshape(-1)
        y_true = returns.detach().float().reshape(-1)
        if y_true.numel() <= 1:
            return 0.0
        var_y = torch.var(y_true, unbiased=False)
        if float(var_y.detach().cpu().item()) <= 1e-12:
            return 0.0
        ev = 1.0 - torch.var(y_true - y_pred, unbiased=False) / var_y
        return float(ev.detach().cpu().item())

    def _rollout_episode_summary(self) -> dict[str, Any]:
        records = list(self._rollout_episode_records)
        n = len(records)
        if n <= 0:
            base: dict[str, Any] = {
                "rollout_episodes": 0,
                "rollout_wins": 0,
                "rollout_losses": 0,
                "rollout_draws": 0,
                "rollout_win_rate": 0.0,
                "rollout_win_margin_mean": 0.0,
                "rollout_blue_score_mean": 0.0,
                "rollout_red_score_mean": 0.0,
            }
        else:
            wins = sum(int(r["success"]) for r in records)
            margins = [int(r["win_margin"]) for r in records]
            losses = sum(1 for m in margins if m < 0)
            draws = sum(1 for m in margins if m == 0)
            base = {
                "rollout_episodes": n,
                "rollout_wins": wins,
                "rollout_losses": losses,
                "rollout_draws": draws,
                "rollout_win_rate": float(wins) / float(n),
                "rollout_win_margin_mean": float(np.mean(margins)),
                "rollout_blue_score_mean": float(np.mean([int(r["blue_score"]) for r in records])),
                "rollout_red_score_mean": float(np.mean([int(r["red_score"]) for r in records])),
            }
        if self.use_latent_strategy:
            for z_idx in range(self.latent_k):
                z_records = [r for r in records if r.get("latent_z") == z_idx]
                zn = len(z_records)
                base[f"episode_z_{z_idx}_count"] = zn
                if zn <= 0:
                    base[f"episode_z_{z_idx}_win_rate"] = ""
                    base[f"episode_z_{z_idx}_blue_score_mean"] = ""
                    base[f"episode_z_{z_idx}_red_score_mean"] = ""
                    base[f"episode_z_{z_idx}_win_margin_mean"] = ""
                else:
                    base[f"episode_z_{z_idx}_win_rate"] = float(sum(int(r["success"]) for r in z_records)) / float(zn)
                    base[f"episode_z_{z_idx}_blue_score_mean"] = float(np.mean([int(r["blue_score"]) for r in z_records]))
                    base[f"episode_z_{z_idx}_red_score_mean"] = float(np.mean([int(r["red_score"]) for r in z_records]))
                    base[f"episode_z_{z_idx}_win_margin_mean"] = float(np.mean([int(r["win_margin"]) for r in z_records]))
            for o_idx in range(SCRIPTED_OPPONENT_MI_COUNT):
                for z_idx in range(self.latent_k):
                    sub = [
                        r
                        for r in records
                        if int(r.get("opponent_id", -1)) == o_idx and r.get("latent_z") == z_idx
                    ]
                    zn = len(sub)
                    base[f"episode_opp{o_idx}_z{z_idx}_count"] = zn
                    if zn <= 0:
                        base[f"episode_opp{o_idx}_z{z_idx}_win_rate"] = ""
                    else:
                        base[f"episode_opp{o_idx}_z{z_idx}_win_rate"] = float(
                            sum(int(r["success"]) for r in sub)
                        ) / float(zn)
        return base

    def _rolling_win_rate(self, window: int) -> float:
        recent = list(self._recent_episode_successes)[-max(1, int(window)):]
        if not recent:
            return 0.0
        return float(sum(recent)) / float(len(recent))

    def _write_update_metrics(self, stats: dict[str, float], buffer: TensorDictRolloutBuffer) -> dict[str, Any]:
        if not self.metrics_csv_path:
            return {}
        rewards = buffer.fields["rewards"][: int(buffer.pos)].detach().float().reshape(-1)
        returns = buffer.fields["returns"][: int(buffer.pos)].detach().float().reshape(-1)
        values = buffer.fields["values"][: int(buffer.pos)].detach().float().reshape(-1)
        games = self._ep_wins + self._ep_losses + self._ep_draws
        row: dict[str, Any] = {
            "update": self._updates_completed,
            "run_id": self.run_id,
            "run_pid": self.run_pid,
            "timesteps": int(self.global_step),
            "episodes_completed": int(self._episodes_completed),
            "wins": int(self._ep_wins),
            "losses": int(self._ep_losses),
            "draws": int(self._ep_draws),
            "win_rate": float(self._ep_wins) / float(max(1, games)),
            "rolling_win_rate_50ep": self._rolling_win_rate(50),
            "rolling_win_rate_200ep": self._rolling_win_rate(200),
            "rollout_reward_mean": float(rewards.mean().detach().cpu().item()) if rewards.numel() > 0 else 0.0,
            "rollout_reward_std": float(rewards.std(unbiased=False).detach().cpu().item()) if rewards.numel() > 1 else 0.0,
            "rollout_return_mean": float(returns.mean().detach().cpu().item()) if returns.numel() > 0 else 0.0,
            "rollout_return_std": float(returns.std(unbiased=False).detach().cpu().item()) if returns.numel() > 1 else 0.0,
            "explained_variance": self._explained_variance(values, returns),
        }
        row.update(self._rollout_episode_summary())
        if self.curriculum is not None:
            row.update(
                {
                    "curriculum_phase": str(self.curriculum.phase),
                    "curriculum_phase_idx": int(self.curriculum.phase_idx),
                    "curriculum_phase_episodes": int(self.curriculum.phase_episode_count),
                    "curriculum_phase_win_rate": float(self.curriculum.phase_winrate()),
                }
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
        ):
            vals = buffer.fields[key][: int(buffer.pos)].detach().float().reshape(-1)
            row[f"{key}_mean"] = float(vals.mean().detach().cpu().item()) if vals.numel() > 0 else 0.0
        reward_outcome = float(row.get("reward_terminal_mean", 0.0)) + float(row.get("reward_sparse_mean", 0.0))
        reward_shaping = (
            float(row.get("reward_offense_mean", 0.0))
            + self.reward_dense_weight
            * (float(row.get("reward_pbrs_mean", 0.0)) + float(row.get("reward_team_mean", 0.0)))
        )
        reward_failure = float(row.get("reward_failure_mean", 0.0))
        row["reward_outcome_mean"] = reward_outcome
        row["reward_shaping_mean"] = reward_shaping
        row["reward_shaping_to_outcome_abs_ratio"] = abs(reward_shaping) / (abs(reward_outcome) + 1e-6)
        row["reward_shaping_coef"] = float(self._reward_shaping_coef())
        row["reward_failure_to_outcome_abs"] = abs(reward_failure) / (abs(reward_outcome) + 1e-6)
        row.update(stats)
        if self.use_latent_strategy:
            entropy = float(row.get("strategy_entropy", 0.0) or 0.0)
            row["strategy_entropy_frac"] = entropy / max(1e-6, math.log(max(2, int(self.latent_k))))
            z_win_rates: list[float] = []
            for z_idx in range(self.latent_k):
                value = row.get(f"episode_z_{z_idx}_win_rate", "")
                if value == "":
                    continue
                z_win_rates.append(float(value))
            row["strategy_wr_spread"] = (
                float(max(z_win_rates) - min(z_win_rates)) if len(z_win_rates) >= 2 else 0.0
            )
        else:
            row["strategy_entropy_frac"] = 0.0
            row["strategy_wr_spread"] = 0.0
        self._write_csv_row(
            self.metrics_csv_path,
            self._update_fieldnames(),
            row,
            legacy_column_fill=_METRICS_CSV_LEGACY_COLUMN_FILL,
        )
        return row

    def _opponent_id_int_from_info(self, info: dict[str, Any]) -> int:
        """Scripted opponent index for MI telemetry: OP1→0 … OP7→6; ``-1`` if unknown / non-scripted."""
        er = info.get("episode_result") if isinstance(info.get("episode_result"), dict) else {}
        kind = str(er.get("opponent_kind", info.get("opponent_kind", "scripted")) or "scripted").lower()
        if kind != "scripted":
            return -1
        tag_raw = str(
            er.get("scripted_tag")
            or info.get("opponent_key", getattr(self.cfg, "fixed_opponent_tag", "OP3"))
            or ""
        ).strip().upper()
        tag = "OP5_RUSHER" if tag_raw == "OP5" else tag_raw
        if tag == "OP6_TURTLE":
            tag = "OP6"
        if tag == "OP7_SWITCHER":
            tag = "OP7"
        return {"OP1": 0, "OP2": 1, "OP3": 2, "OP4": 3, "OP5_RUSHER": 4, "OP6": 5, "OP7": 6}.get(tag, -1)

    def _opponent_id_csv_from_info(self, info: dict[str, Any]) -> str:
        oid = self._opponent_id_int_from_info(info)
        return str(int(oid)) if oid >= 0 else ""

    def _opponent_legend(self, info: dict[str, Any]) -> str:
        """Compact opponent string for logging (scripted:OP3, snapshot:name, ...)."""
        er = info.get("episode_result") or {}
        kind = str(er.get("opponent_kind", info.get("opponent_kind", "scripted")) or "scripted").lower()
        if kind == "scripted":
            tag = str(er.get("scripted_tag") or info.get("opponent_key", getattr(self.cfg, "fixed_opponent_tag", "OP3")))
            return f"SCRIPTED:{str(tag).upper()}"
        if kind == "snapshot":
            snap = str(er.get("opponent_snapshot", "") or info.get("opponent_key", ""))
            return f"SNAPSHOT:{snap}" if snap else "SNAPSHOT:unknown"
        return f"{kind.upper()}:?"

    def _set_curriculum_opponent(self, phase: str, env_index: Optional[int] = None) -> None:
        phase_s = str(phase).upper()
        indices = None if env_index is None else [int(env_index)]
        try:
            self.env.env_method("set_next_opponent", "SCRIPTED", phase_s, indices=indices)
            self.env.env_method("set_phase", phase_s, indices=indices)
        except Exception:
            if indices is not None:
                self.env.env_method("set_next_opponent", "SCRIPTED", phase_s)
                self.env.env_method("set_phase", phase_s)

    def _update_curriculum_after_episode(self, *, info: dict[str, Any], blue_score: int, red_score: int, env_index: Optional[int]) -> None:
        if self.curriculum is None:
            return
        episode_phase = str(info.get("phase", self.curriculum.phase)).upper()
        old_phase = str(self.curriculum.phase).upper()
        win_value = 1.0 if int(blue_score) > int(red_score) else 0.0
        if episode_phase != old_phase:
            self.curriculum.record_result(episode_phase, win_value)
            self._set_curriculum_opponent(old_phase, env_index)
            return
        self.curriculum.phase_episode_count += 1
        self.curriculum.record_result(old_phase, win_value)
        advanced = self.curriculum.advance_if_ready(win_by=int(blue_score) - int(red_score))
        new_phase = str(self.curriculum.phase).upper()
        if advanced:
            wr = 100.0 * float(self.curriculum.phase_winrate(old_phase))
            print(
                f"[PPO] Curriculum advanced: {old_phase} -> {new_phase} "
                f"after episode {self._episodes_completed} (gate_wr={wr:.1f}%)."
            )
        self._set_curriculum_opponent(new_phase, env_index)

    def _hook_sample_training_opponent_before_reset(self, done: np.ndarray, infos: list) -> None:
        """Sample the *next* episode's scripted opponent per finished sub-env (GPUCTFVecEnv hook)."""
        if self.curriculum is not None or not self._opponent_randomize_training:
            return
        for env_i, done_i in enumerate(done):
            if not bool(done_i):
                continue
            tag = str(self._rng_opponent.choice(self._opponent_pool_tags)).upper()
            phase_s = phase_from_tag(tag)
            try:
                self.env.env_method("set_next_opponent", "SCRIPTED", tag, indices=[env_i])
                self.env.env_method("set_phase", phase_s, indices=[env_i])
            except Exception:
                self.env.env_method("set_next_opponent", "SCRIPTED", tag)
                self.env.env_method("set_phase", phase_s)

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
                "opponent_id": int(self._opponent_id_int_from_info(info)),
            }
        )
        self._recent_episode_successes.append(success)
        self._write_episode_metrics(
            info,
            blue_score=bs,
            red_score=rs,
            timestep=int(timestep or self.global_step),
            rollout_step=rollout_step,
            latent_z=latent_z,
        )
        self._update_curriculum_after_episode(info=info, blue_score=bs, red_score=rs, env_index=env_index)
        every = int(getattr(self.cfg, "episode_log_every", 0) or 0)
        if every > 0 and self._episodes_completed % every == 0:
            self._print_episode_progress(info)

    def _print_episode_progress(self, info: dict[str, Any]) -> None:
        n = self._episodes_completed
        w, l, d = self._ep_wins, self._ep_losses, self._ep_draws
        wr = 100.0 * float(w) / float(max(1, w + l + d))
        mode = str(getattr(self.cfg, "mode", "FIXED_OPPONENT"))
        opp = self._opponent_legend(info)
        print(
            f"[PPO] ep={n} mode={mode} opp={opp} "
            f"W={w} L={l} D={d} WR={wr:.1f}%"
            + (f" phase={self.curriculum.phase}" if self.curriculum is not None else "")
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

        resample_mask = self._needs_strategy_sample.clone()
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
        """After ``global_step`` += ``n_envs``: mirror :meth:`ProgressBarCallback._on_step` (``update(num_envs)``)."""
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

    def _latent_rollout_stats(self, buffer: TensorDictRolloutBuffer) -> dict[str, float]:
        """Summarize strategy occupancy and switching for the latest rollout."""
        if not self.use_latent_strategy or "z" not in buffer.fields:
            return {}
        length = int(buffer.pos)
        z = buffer.fields["z"][:length].reshape(-1).long()
        prev_z = buffer.fields["prev_z"][:length].reshape(-1).long()
        if z.numel() == 0:
            return {}
        counts = torch.bincount(z.clamp(min=0, max=self.latent_k - 1), minlength=self.latent_k).float()
        occupancy = counts / counts.sum().clamp_min(1.0)
        persist_mask = buffer.fields["z_persist_mask"][:length].reshape(-1).bool()
        resampled = buffer.fields["z_resampled"][:length].reshape(-1).bool()
        switched = persist_mask & (z != prev_z)
        out = {
            "strategy_unique_count": float((counts > 0).sum().detach().cpu().item()),
            "strategy_dominant": float(torch.argmax(counts).detach().cpu().item()),
            "strategy_switch_count": float(switched.sum().detach().cpu().item()),
            "strategy_switch_fraction": float(
                (switched.float().sum() / persist_mask.float().sum().clamp_min(1.0)).detach().cpu().item()
            ),
            "strategy_resample_count": float(resampled.sum().detach().cpu().item()),
            "strategy_resample_fraction_rollout": float(resampled.float().mean().detach().cpu().item()),
        }
        for idx, value in enumerate(occupancy.detach().cpu().tolist()):
            out[f"strategy_occupancy_{idx}"] = float(value)
        return out

    def _latent_opponent_rollout_diag(self, buffer: TensorDictRolloutBuffer) -> dict[str, float]:
        """Per-opponent z occupancy plus MI(z; opponent/phase/outcome) and phase / behavior bucket rollups."""
        if not self.use_latent_strategy or "z" not in buffer.fields:
            return {}
        length = int(buffer.pos)
        if length <= 0:
            return {}
        z = buffer.fields["z"][:length].reshape(-1).long().cpu().numpy()
        K = int(self.latent_k)
        out: dict[str, float] = {}
        q_probs_np: Optional[np.ndarray] = None
        q_entropy_np: Optional[np.ndarray] = None
        if "z_logits" in buffer.fields:
            z_logits_t = buffer.fields["z_logits"][:length].reshape(-1, K).float()
            q_probs_t = torch.softmax(z_logits_t, dim=-1)
            q_entropy_t = -(q_probs_t.clamp_min(1e-8) * q_probs_t.clamp_min(1e-8).log()).sum(dim=-1)
            q_probs_np = q_probs_t.detach().cpu().numpy()
            q_entropy_np = q_entropy_t.detach().cpu().numpy()

        if "opponent_id" in buffer.fields:
            oid = buffer.fields["opponent_id"][:length].reshape(-1).long().cpu().numpy()
            joint = np.zeros((K, SCRIPTED_OPPONENT_MI_COUNT), dtype=np.float64)
            for i in range(z.size):
                zi = int(z[i])
                oi = int(oid[i])
                if 0 <= zi < K and 0 <= oi < SCRIPTED_OPPONENT_MI_COUNT:
                    joint[zi, oi] += 1.0
            out["latent_mi_z_opponent_nats"] = float(discrete_mi_plugin(joint))
            for o in range(SCRIPTED_OPPONENT_MI_COUNT):
                mask = oid == o
                if not np.any(mask):
                    for k in range(K):
                        out[f"strategy_occupancy_op{o}_z{k}"] = 0.0
                    continue
                z_sub = np.clip(z[mask], 0, K - 1)
                cnt = np.bincount(z_sub, minlength=K).astype(np.float64)
                total = float(cnt.sum())
                occ = cnt / max(total, 1.0)
                for k in range(K):
                    out[f"strategy_occupancy_op{o}_z{k}"] = float(occ[k])
        else:
            out["latent_mi_z_opponent_nats"] = 0.0

        pid_flat: Optional[np.ndarray] = None
        if "phase_id" in buffer.fields:
            pid_flat = buffer.fields["phase_id"][:length].reshape(-1).long().cpu().numpy()
            n_p = len(TEAM_PHASES)
            joint_p = np.zeros((K, n_p), dtype=np.float64)
            for i in range(z.size):
                zi = int(z[i])
                pi = int(pid_flat[i])
                if 0 <= zi < K and 0 <= pi < n_p:
                    joint_p[zi, pi] += 1.0
            out["latent_mi_z_phase_nats"] = float(discrete_mi_plugin(joint_p))
        else:
            out["latent_mi_z_phase_nats"] = 0.0

        if "outcome_id" in buffer.fields:
            yid = buffer.fields["outcome_id"][:length].reshape(-1).long().cpu().numpy()
            joint_y = np.zeros((K, 3), dtype=np.float64)
            for i in range(z.size):
                zi = int(z[i])
                yi = int(yid[i])
                if 0 <= zi < K and 0 <= yi < 3:
                    joint_y[zi, yi] += 1.0
            out["latent_mi_z_outcome_nats"] = float(discrete_mi_plugin(joint_y))
        else:
            out["latent_mi_z_outcome_nats"] = 0.0

        prev_np = buffer.fields["prev_z"][:length].reshape(-1).long().cpu().numpy()
        sw = (z != prev_np).astype(np.float64)
        ba: Optional[np.ndarray] = None
        if "blue_ahead" in buffer.fields:
            ba = buffer.fields["blue_ahead"][:length].reshape(-1).float().cpu().numpy()
        rsp_bin: Optional[np.ndarray] = None
        if "reward_sparse_points" in buffer.fields:
            rsp_bin = (
                (np.abs(buffer.fields["reward_sparse_points"][:length].reshape(-1).float().cpu().numpy()) > 1e-5)
                .astype(np.float64)
            )

        if pid_flat is not None:
            n_p = len(TEAM_PHASES)
            for p in range(n_p):
                mask = pid_flat == p
                cnt_m = float(mask.sum())
                if cnt_m <= 0.0:
                    for k in range(K):
                        out[f"latent_phase{p}_z{k}_frac"] = 0.0
                    out[f"latent_phase{p}_switch_mean"] = 0.0
                    out[f"latent_phase{p}_blue_ahead_mean"] = 0.0
                    out[f"latent_phase{p}_capture_step_mean"] = 0.0
                else:
                    z_sub = np.clip(z[mask], 0, K - 1)
                    for k in range(K):
                        out[f"latent_phase{p}_z{k}_frac"] = float((z_sub == k).mean())
                    out[f"latent_phase{p}_switch_mean"] = float(sw[mask].mean())
                    out[f"latent_phase{p}_blue_ahead_mean"] = (
                        float(ba[mask].mean()) if ba is not None else 0.0
                    )
                    out[f"latent_phase{p}_capture_step_mean"] = (
                        float(rsp_bin[mask].mean()) if rsp_bin is not None else 0.0
                    )
                if q_entropy_np is None or q_probs_np is None or not np.any(mask):
                    out[f"q_phi_phase{p}_entropy_mean"] = 0.0
                    for k in range(K):
                        out[f"q_phi_phase{p}_z{k}_prob_mean"] = 0.0
                else:
                    out[f"q_phi_phase{p}_entropy_mean"] = float(q_entropy_np[mask].mean())
                    q_phase = q_probs_np[mask]
                    for k in range(K):
                        out[f"q_phi_phase{p}_z{k}_prob_mean"] = float(q_phase[:, k].mean())

        if ba is not None:
            ahead = ba > 0.5
            trail = ~ahead
            out["latent_switch_rate_blue_ahead"] = float(sw[ahead].mean()) if bool(ahead.any()) else 0.0
            out["latent_switch_rate_blue_trail"] = float(sw[trail].mean()) if bool(trail.any()) else 0.0
        else:
            out["latent_switch_rate_blue_ahead"] = 0.0
            out["latent_switch_rate_blue_trail"] = 0.0

        Rtb = buffer.fields["rewards"][:length].detach().cpu().numpy()
        Ztb = buffer.fields["z"][:length].detach().cpu().numpy()
        Ptb = buffer.fields["prev_z"][:length].detach().cpu().numpy()
        Tn, Bn = int(Ztb.shape[0]), int(Ztb.shape[1])
        sums: list[float] = []
        for t in range(Tn):
            for b in range(Bn):
                if int(Ztb[t, b]) != int(Ptb[t, b]):
                    h = min(5, Tn - 1 - t)
                    if h > 0:
                        sums.append(float(Rtb[t + 1 : t + 1 + h, b].sum()))
        out["latent_reward_sum_5_after_z_switch_mean"] = float(np.mean(sums)) if sums else 0.0

        n_sb, n_rb, n_pb = 3, int(N_ROLE_BUCKET_MI), 3
        n_adr = int(N_ATTACK_DEFENSE_RATIO_BUCKET)

        if "spread_bucket_id" in buffer.fields:
            sb = buffer.fields["spread_bucket_id"][:length].reshape(-1).long().cpu().numpy()
            j = np.zeros((K, n_sb), dtype=np.float64)
            for i in range(z.size):
                zi, si = int(z[i]), int(sb[i])
                if 0 <= zi < K and 0 <= si < n_sb:
                    j[zi, si] += 1.0
            out["latent_mi_z_spread_bucket_nats"] = float(discrete_mi_plugin(j))
        else:
            out["latent_mi_z_spread_bucket_nats"] = 0.0

        if "role_bucket_id" in buffer.fields:
            rb = buffer.fields["role_bucket_id"][:length].reshape(-1).long().cpu().numpy()
            j = np.zeros((K, n_rb), dtype=np.float64)
            for i in range(z.size):
                zi, ri = int(z[i]), int(rb[i])
                if 0 <= zi < K and 0 <= ri < n_rb:
                    j[zi, ri] += 1.0
            out["latent_mi_z_role_bucket_nats"] = float(discrete_mi_plugin(j))
            for r in range(n_rb):
                mask = rb == r
                if not np.any(mask):
                    for k_idx in range(K):
                        out[f"latent_role{r}_z{k_idx}_frac"] = 0.0
                    out[f"latent_role{r}_switch_mean"] = 0.0
                else:
                    z_sub = np.clip(z[mask], 0, K - 1)
                    for k_idx in range(K):
                        out[f"latent_role{r}_z{k_idx}_frac"] = float((z_sub == k_idx).mean())
                    out[f"latent_role{r}_switch_mean"] = float(sw[mask].mean())
        else:
            out["latent_mi_z_role_bucket_nats"] = 0.0
            for r in range(n_rb):
                for k_idx in range(K):
                    out[f"latent_role{r}_z{k_idx}_frac"] = 0.0
                out[f"latent_role{r}_switch_mean"] = 0.0

        if "pressure_bucket_id" in buffer.fields:
            pb = buffer.fields["pressure_bucket_id"][:length].reshape(-1).long().cpu().numpy()
            j = np.zeros((K, n_pb), dtype=np.float64)
            for i in range(z.size):
                zi, pi2 = int(z[i]), int(pb[i])
                if 0 <= zi < K and 0 <= pi2 < n_pb:
                    j[zi, pi2] += 1.0
            out["latent_mi_z_pressure_bucket_nats"] = float(discrete_mi_plugin(j))
        else:
            out["latent_mi_z_pressure_bucket_nats"] = 0.0

        if "attack_defense_ratio_bucket_id" in buffer.fields:
            adb = buffer.fields["attack_defense_ratio_bucket_id"][:length].reshape(-1).long().cpu().numpy()
            j = np.zeros((K, n_adr), dtype=np.float64)
            for i in range(z.size):
                zi, ai = int(z[i]), int(adb[i])
                if 0 <= zi < K and 0 <= ai < n_adr:
                    j[zi, ai] += 1.0
            out["latent_mi_z_attack_defense_ratio_bucket_nats"] = float(discrete_mi_plugin(j))
        else:
            out["latent_mi_z_attack_defense_ratio_bucket_nats"] = 0.0

        return out

    def _behavior_diversity_stats(self, buffer: TensorDictRolloutBuffer) -> dict[str, float]:
        """Post-hoc behavior spread by sampled z; diagnostics only, no labels or losses."""
        if not self.use_latent_strategy or "z" not in buffer.fields or "behavior_telemetry" not in buffer.fields:
            return {}
        length = int(buffer.pos)
        if length <= 0:
            return {}
        z = buffer.fields["z"][:length].reshape(-1).long()
        beh = buffer.fields["behavior_telemetry"][:length].reshape(-1, N_TELEMETRY).float()
        out: dict[str, float] = {}
        means: list[torch.Tensor] = []
        for k in range(int(self.latent_k)):
            mask = z == k
            if bool(mask.any().item()):
                mean_k = beh[mask].mean(dim=0)
                means.append(mean_k)
            else:
                mean_k = torch.zeros((N_TELEMETRY,), dtype=torch.float32, device=beh.device)
            for j, name in enumerate(BEHAVIOR_TELEMETRY_NAMES):
                out[f"latent_z{k}_behavior_{name}_mean"] = float(mean_k[j].detach().cpu().item())

        if len(means) >= 2:
            pairwise: list[float] = []
            for i in range(len(means)):
                for j in range(i + 1, len(means)):
                    pairwise.append(float(torch.linalg.vector_norm(means[i] - means[j]).detach().cpu().item()))
            out["latent_behavior_diversity_l2_mean"] = float(np.mean(pairwise)) if pairwise else 0.0
        else:
            out["latent_behavior_diversity_l2_mean"] = 0.0
        return out

    def _macro_probs_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Return macro-action probabilities with shape (B, n_agents, macro_dim)."""
        macro_chunks: list[torch.Tensor] = []
        offset = 0
        for _agent_idx in range(int(self.model.n_agents)):
            for head_idx in range(int(self.model.heads_per_agent)):
                dim = int(self.model.per_agent_action_dims[head_idx])
                chunk = logits[:, offset : offset + dim]
                if head_idx == 0:
                    macro_chunks.append(torch.softmax(chunk, dim=-1))
                offset += dim
        if not macro_chunks:
            raise AssertionError("could not find macro-action heads for forced-z profiling")
        return torch.stack(macro_chunks, dim=1)

    def _forced_z_behavior_profile(self, buffer: TensorDictRolloutBuffer) -> dict[str, float]:
        """Profile actor macro preferences under every forced z on the same rollout observations."""
        if not self.use_latent_strategy:
            return {}
        length = int(buffer.pos)
        if length <= 0:
            return {}
        total = length * int(buffer.n_envs)
        if total <= 0:
            return {}
        if total > FORCED_Z_PROFILE_MAX_ROWS:
            row_idx = torch.linspace(
                0,
                total - 1,
                steps=FORCED_Z_PROFILE_MAX_ROWS,
                device=self.device,
            ).long()
        else:
            row_idx = torch.arange(total, device=self.device)
        obs_batch = {
            "grid": buffer.fields["obs_grid"][:length].reshape(total, *buffer.fields["obs_grid"].shape[2:]).index_select(0, row_idx),
            "vec": buffer.fields["obs_vec"][:length].reshape(total, *buffer.fields["obs_vec"].shape[2:]).index_select(0, row_idx),
            "agent_mask": buffer.fields["obs_agent_mask"][:length].reshape(total, *buffer.fields["obs_agent_mask"].shape[2:]).index_select(0, row_idx),
            "mask": buffer.fields["obs_mask"][:length].reshape(total, *buffer.fields["obs_mask"].shape[2:]).index_select(0, row_idx),
        }
        out: dict[str, float] = {}
        mean_macros: list[torch.Tensor] = []
        with torch.no_grad():
            for z_id in range(int(self.latent_k)):
                z_idx = torch.full((int(row_idx.numel()),), z_id, dtype=torch.long, device=self.device)
                logits = self.model.policy_logits(obs_batch, z_idx=z_idx)
                logits = self.model._mask_logits(logits, obs_batch.get("mask"))
                macro_probs = self._macro_probs_from_logits(logits)
                mean_macro = macro_probs.mean(dim=(0, 1))
                mean_macros.append(mean_macro)
                macro_entropy = -(
                    macro_probs.clamp_min(1e-8) * macro_probs.clamp_min(1e-8).log()
                ).sum(dim=-1).mean()
                for action_id, action_name in FORCED_Z_MACRO_ACTIONS:
                    if action_id < int(mean_macro.numel()):
                        out[f"forced_z{z_id}_macro_{action_name}_prob"] = float(mean_macro[action_id].detach().cpu().item())
                    else:
                        out[f"forced_z{z_id}_macro_{action_name}_prob"] = 0.0
                out[f"forced_z{z_id}_macro_entropy"] = float(macro_entropy.detach().cpu().item())

        if len(mean_macros) >= 2:
            js_vals: list[float] = []
            for i in range(len(mean_macros)):
                for j in range(i + 1, len(mean_macros)):
                    p = mean_macros[i].clamp_min(1e-8)
                    q = mean_macros[j].clamp_min(1e-8)
                    m = 0.5 * (p + q)
                    js = 0.5 * (p * (p.log() - m.log())).sum() + 0.5 * (q * (q.log() - m.log())).sum()
                    js_vals.append(float(js.detach().cpu().item()))
            out["forced_z_macro_jsd_mean"] = float(np.mean(js_vals)) if js_vals else 0.0
        else:
            out["forced_z_macro_jsd_mean"] = 0.0
        return out

    def _strategy_resample_advantage_stats(self, buffer: TensorDictRolloutBuffer) -> dict[str, float]:
        """Per-z mean/std of raw GAE advantages at z-resample steps (pre-minibatch normalization)."""
        if not self.use_latent_strategy or self.fixed_latent_strategy:
            return {}
        length = int(buffer.pos)
        if length <= 0 or "advantages" not in buffer.fields or "z" not in buffer.fields:
            return {}
        adv = buffer.fields["advantages"][:length]
        z = buffer.fields["z"][:length].long()
        rs = buffer.fields["z_resampled"][:length].bool()
        flat_adv = adv.reshape(-1).float()
        flat_z = z.reshape(-1)
        flat_rs = rs.reshape(-1)
        mask = flat_rs
        out: dict[str, float] = {}
        K = int(self.latent_k)
        for k in range(K):
            m = mask & (flat_z == k)
            n = int(m.sum().item())
            out[f"strategy_resample_adv_n_z{k}"] = float(n)
            if n > 0:
                vals = flat_adv[m]
                out[f"strategy_resample_adv_mean_z{k}"] = float(vals.mean().item())
                out[f"strategy_resample_adv_std_z{k}"] = (
                    float(vals.std(unbiased=False).item()) if n > 1 else 0.0
                )
            else:
                out[f"strategy_resample_adv_mean_z{k}"] = 0.0
                out[f"strategy_resample_adv_std_z{k}"] = 0.0
        return out

    def _rollout_advantage_diagnostics(self, buffer: TensorDictRolloutBuffer) -> dict[str, float]:
        """Raw GAE advantage scale and split at latent z-segment starts (t>0, z[t]!=z[t-1])."""
        length = int(buffer.pos)
        if length <= 0 or "advantages" not in buffer.fields:
            return {}
        adv = buffer.fields["advantages"][:length].detach().float()
        flat = adv.reshape(-1)
        out: dict[str, float] = {
            "rollout_adv_std": float(flat.std(unbiased=False).item()) if flat.numel() > 1 else 0.0,
        }
        if (
            not self.use_latent_strategy
            or self.fixed_latent_strategy
            or "z" not in buffer.fields
            or length < 2
        ):
            return out
        z = buffer.fields["z"][:length].long()
        z_switch = torch.zeros((length, z.shape[1]), dtype=torch.bool, device=z.device)
        z_switch[1:] = z[1:] != z[:-1]
        flat_sw = z_switch.reshape(-1)
        if flat_sw.any() and (~flat_sw).any():
            out["rollout_adv_std_at_z_switch"] = float(flat[flat_sw].std(unbiased=False).item())
            out["rollout_adv_std_not_z_switch"] = float(flat[~flat_sw].std(unbiased=False).item())
        else:
            out["rollout_adv_std_at_z_switch"] = float(out["rollout_adv_std"])
            out["rollout_adv_std_not_z_switch"] = float(out["rollout_adv_std"])
        return out

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

    def _update_strategy_return_stats(self, buffer: TensorDictRolloutBuffer) -> None:
        """Update running return normalization stats for sampled z targets."""
        if not self.latent_strategy_aux_return_head or "z_resampled" not in buffer.fields:
            return
        sampled = buffer.fields["z_resampled"][: int(buffer.pos)].reshape(-1).bool()
        returns = buffer.fields["returns"][: int(buffer.pos)].reshape(-1).detach().float()
        if not bool(sampled.any().item()):
            return
        values = returns[sampled]
        batch_count = float(values.numel())
        batch_mean = float(values.mean().detach().cpu().item())
        batch_var = float(values.var(unbiased=False).detach().cpu().item()) if values.numel() > 1 else 0.0

        count = float(self._strategy_return_count)
        delta = batch_mean - float(self._strategy_return_mean)
        total_count = count + batch_count
        new_mean = float(self._strategy_return_mean) + delta * batch_count / max(1e-6, total_count)
        m_a = float(self._strategy_return_var) * count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta * delta * count * batch_count / max(1e-6, total_count)
        self._strategy_return_mean = new_mean
        self._strategy_return_var = max(1e-6, m2 / max(1e-6, total_count))
        self._strategy_return_count = total_count

    def _normalize_strategy_returns(self, returns: torch.Tensor) -> torch.Tensor:
        std = max(1e-3, float(self._strategy_return_var) ** 0.5)
        return (returns.detach().float() - float(self._strategy_return_mean)) / std

    def _return_norm_std(self) -> float:
        return max(1e-3, float(self._return_norm_var) ** 0.5)

    def _normalize_value_targets(self, returns: torch.Tensor) -> torch.Tensor:
        if not self.normalize_returns:
            return returns.float()
        return (returns.float() - float(self._return_norm_mean)) / self._return_norm_std()

    def _denormalize_values(self, values: torch.Tensor) -> torch.Tensor:
        if not self.normalize_returns:
            return values.float()
        return values.float() * self._return_norm_std() + float(self._return_norm_mean)

    def _update_return_norm_stats(self, returns: torch.Tensor) -> None:
        if not self.normalize_returns:
            return
        values = returns.detach().float().reshape(-1)
        if values.numel() <= 0:
            return
        batch_count = float(values.numel())
        batch_mean = float(values.mean().detach().cpu().item())
        batch_var = float(values.var(unbiased=False).detach().cpu().item()) if values.numel() > 1 else 0.0

        count = float(self._return_norm_count)
        delta = batch_mean - float(self._return_norm_mean)
        total_count = count + batch_count
        new_mean = float(self._return_norm_mean) + delta * batch_count / max(1e-6, total_count)
        m_a = float(self._return_norm_var) * count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta * delta * count * batch_count / max(1e-6, total_count)
        self._return_norm_mean = new_mean
        self._return_norm_var = max(1e-6, m2 / max(1e-6, total_count))
        self._return_norm_count = total_count

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
        """Strategy index for V(s', z') bootstrapping to match the start of the *next* decision.

        Mirrors `_strategy_for_step` resample rules using counters *after* this env step
        (same as `_mark_strategy_step_done` would apply before the next `_strategy_for_step`).
        """
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
                return self._denormalize_values(self.model.values(gs))
            
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
            next_values = self._denormalize_values(next_values)
            terminated = torch.as_tensor(
                [bool(info.get("terminated", False)) for info in infos],
                dtype=torch.bool,
                device=self.device,
            )
            return torch.where(terminated, torch.zeros_like(next_values), next_values)

    def collect_rollout(self) -> TensorDictRolloutBuffer:
        """Collect one rollout and compute advantages/returns."""
        self._log_decentralized_actor_contract_once()
        self._rollout_episode_records = []
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
                values_t = self._denormalize_values(values_norm_t)
                # Action PPO uses action log-probs only; q_phi is trained separately
                # at actual z-sampling points in update().
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

            opp_row = torch.as_tensor(
                [self._opponent_id_int_from_info(dict(info)) for info in infos],
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
            if self._e3_step_telemetry_path and self.use_latent_strategy and z_t is not None and prev_z_t is not None:
                assert beh_t is not None and sb is not None and adb is not None
                self._append_e3_step_telemetry(
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
        self._update_return_norm_stats(buffer.fields["returns"][: int(buffer.pos)])
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
        self._update_strategy_return_stats(buffer)

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
                    # Paper default: maximize H(z)  ⇔ L += -λ_H * H(z) (minimized loss decreases as H rises).
                    # ``latent_entropy_objective=minimize`` flips sign (L += +λ_H * H(z)) so q_phi trains toward sharper z.
                    if bool(resample.any().item()):
                        h_mean = strategy_entropy[resample].mean()
                    else:
                        h_mean = torch.zeros((), dtype=torch.float32, device=self.device)
                    h_goal = str(getattr(self.cfg, "latent_entropy_objective", "maximize") or "maximize").lower()
                    if h_goal == "none" or latent_lam_h <= 0.0:
                        strategy_entropy_loss = torch.zeros((), dtype=torch.float32, device=self.device)
                    elif h_goal == "minimize":
                        strategy_entropy_loss = latent_lam_h * h_mean
                    else:
                        strategy_entropy_loss = -latent_lam_h * h_mean
                    switch = paper_strategy_switch_indicator(batch["z"], batch["prev_z"])
                    if bool(persist_mask.any().item()):
                        persist_loss = switch[persist_mask].mean()
                    else:
                        persist_loss = torch.zeros((), dtype=torch.float32, device=self.device)
                    if self.latent_resample_every_n == 0 and not self.latent_resample_on_flag:
                        assert persist_loss.item() == 0.0, (
                            "L_persist must be exactly 0 when no mid-episode resampling (latent_resample_every_n=0, on_flag off)"
                        )
                    latent_loss = float(getattr(self.cfg, "latent_lam_p", 0.0)) * persist_loss + strategy_entropy_loss
                    if self.latent_kl_consecutive > 0.0:
                        v = batch["z_kl_prev_valid"].float()
                        log_p = F.log_softmax(batch["z_logits"], -1)
                        log_q = F.log_softmax(batch["z_logits_prev"].detach(), -1)
                        p = log_p.exp()
                        kl = (p * (log_p - log_q)).sum(-1)
                        denom = v.sum().clamp_min(1.0)
                        kl_m = (kl * v).sum() / denom
                        latent_loss = latent_loss + float(self.latent_kl_consecutive) * kl_m
                        stats["strategy_kl"].append(float(kl_m.detach().cpu().item()))
                    else:
                        stats["strategy_kl"].append(0.0)
                    if self.fixed_latent_strategy:
                        strategy_entropy = torch.zeros_like(entropy)
                        persist_loss = torch.zeros((), dtype=torch.float32, device=self.device)
                        latent_loss = torch.zeros((), dtype=torch.float32, device=self.device)
                else:
                    log_prob = action_log_prob
                    strategy_entropy = torch.zeros_like(entropy)
                    persist_loss = torch.zeros((), dtype=torch.float32, device=self.device)
                    latent_loss = torch.zeros((), dtype=torch.float32, device=self.device)
                    resample = torch.zeros_like(entropy, dtype=torch.bool)
                    stats["strategy_kl"].append(0.0)

                advantages = batch["advantages"]
                if advantages.numel() > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
                if self.use_latent_strategy and not self.fixed_latent_strategy:
                    strategy_policy_loss = torch.zeros((), dtype=torch.float32, device=self.device)
                    strategy_aux_return_loss = torch.zeros((), dtype=torch.float32, device=self.device)
                    strategy_ppo_stats = {
                        "approx_kl": torch.zeros((), dtype=torch.float32, device=self.device),
                        "clip_fraction": torch.zeros((), dtype=torch.float32, device=self.device),
                        "ratio": torch.ones((1,), dtype=torch.float32, device=self.device),
                    }
                    if bool(resample.any().item()):
                        strategy_adv = advantages[resample].detach()
                        if strategy_adv.numel() > 1:
                            strategy_adv = (
                                strategy_adv - strategy_adv.mean()
                            ) / (strategy_adv.std(unbiased=False) + 1e-8)
                        strategy_policy_loss, strategy_ppo_stats = ppo_policy_loss(
                            strategy_log_prob[resample],
                            batch["z_log_probs"][resample],
                            strategy_adv,
                            self.clip_range,
                        )
                        latent_loss = latent_loss + self.latent_strategy_ppo_coef * strategy_policy_loss
                        if self.latent_strategy_aux_return_head and self.latent_strategy_aux_return_coef > 0.0:
                            pred_all = self.model.strategy_aux_return_predictions(batch["global_state"])
                            z_sel = batch["z"][resample].long().clamp(min=0, max=self.latent_k - 1)
                            pred_selected = pred_all[resample].gather(1, z_sel.reshape(-1, 1)).squeeze(1)
                            ret_target = self._normalize_strategy_returns(batch["returns"][resample])
                            strategy_aux_return_loss = F.mse_loss(pred_selected, ret_target)
                            latent_loss = (
                                latent_loss + self.latent_strategy_aux_return_coef * strategy_aux_return_loss
                            )
                else:
                    strategy_policy_loss = torch.zeros((), dtype=torch.float32, device=self.device)
                    strategy_aux_return_loss = torch.zeros((), dtype=torch.float32, device=self.device)
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
                value_targets = self._normalize_value_targets(batch["returns"])
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
                stats["strategy_aux_return_loss"].append(float(strategy_aux_return_loss.detach().cpu().item()))
                stats["strategy_persist_loss"].append(float(persist_loss.detach().cpu().item()))
                stats["strategy_grad_norm"].append(strategy_grad_norm)
                stats["strategy_resample_fraction"].append(float(resample.float().mean().detach().cpu().item()))
                if target_kl is not None and approx_kl_value > 1.5 * float(target_kl):
                    stop_update = True
                    break
            if stop_update:
                break

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
        self.last_stats["return_norm_std"] = float(self._return_norm_std()) if self.normalize_returns else 0.0
        self.last_stats["return_norm_count"] = float(self._return_norm_count) if self.normalize_returns else 0.0
        self.last_stats.update(self._strategy_resample_advantage_stats(buffer))
        self.last_stats.update(self._rollout_advantage_diagnostics(buffer))
        self.last_stats.update(self._latent_rollout_stats(buffer))
        self.last_stats.update(self._latent_opponent_rollout_diag(buffer))
        self.last_stats.update(self._behavior_diversity_stats(buffer))
        self.last_stats.update(self._forced_z_behavior_profile(buffer))
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
                row = self._write_update_metrics(stats, rollout)
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
                            f" MI_z_o={mi_z_o:.4f} MI_z_phase={mi_z_p:.4f} MI_z_outcome={mi_z_y:.4f}"
                            + (f" | {' ; '.join(opp_diag_bits)}" if opp_diag_bits else "")
                        )
                    print(
                        "[PPO|diag] "
                        f"steps={int(row.get('timesteps', self.global_step))} "
                        f"ev={float(row.get('explained_variance', 0.0)):.3f} "
                        f"v_loss={float(row.get('value_loss', 0.0)):.3f} "
                        f"shape/out={float(row.get('reward_shaping_to_outcome_abs_ratio', 0.0)):.3f} "
                        f"qphi_grad={float(row.get('strategy_grad_norm', 0.0)):.3f} "
                        f"zH={z_entropy:.3f}({z_entropy_frac:.2f}) "
                        f"z_wr_spread={z_wr_spread:.3f} "
                        f"z_aux_ret={float(row.get('strategy_aux_return_loss', row.get('strategy_q_loss', 0.0))):.3f} "
                        f"z_pi={float(row.get('strategy_policy_loss', 0.0)):.3f} "
                        f"z_ratio={float(row.get('strategy_ratio_std', 0.0)):.3f} "
                        f"z_occ=[{','.join(z_occ_parts)}] "
                        f"z_wr=[{','.join(z_wr_parts)}]"
                        f"{opp_suffix}"
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
        self.model.load_state_dict(_remap_legacy_strategy_aux_head_state_dict(payload["model_state_dict"]))
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
