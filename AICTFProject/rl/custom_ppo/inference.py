from __future__ import annotations

import os
import sys
import warnings
from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from rl.global_state import GLOBAL_STATE_DIM
from rl.latent_marl import TemporalStateTracker, CONTEXT_STATE_DIM
from rl.latent_phase_labels import TEAM_PHASES
from rl.custom_ppo.policy import (
    SharedActorCentralizedCritic,
    remap_legacy_actor_state_dict_keys,
)

from macro_actions import MacroAction

# Intentional, stable split from ``PPOConfig.seed`` (E3 / trace §13). Do not “tweak” without a note.
# Decimal: 268435469 (strategy) and 536870955 (action); masked with ``& 0xFFFF_FFFF``.
STRATEGY_GENERATOR_SEED_OFFSET = 0x1_0000_00D
# For action RNG offset:
ACTION_GENERATOR_SEED_OFFSET = 0x2_0000_02B

FORCED_Z_PROFILE_MAX_ROWS = 4096
FORCED_Z_MACRO_ACTIONS: tuple[tuple[int, str], ...] = (
    (int(MacroAction.GO_TO), "go_to"),
    (int(MacroAction.GRAB_MINE), "grab_mine"),
    (int(MacroAction.GET_FLAG), "get_flag"),
    (int(MacroAction.PLACE_MINE), "place_mine"),
    (int(MacroAction.GO_HOME), "go_home"),
)

CUSTOM_PPO_FORMAT = "custom_ppo_cnn_v1"
CUSTOM_PPO_LATENT_FORMAT = "custom_ppo_latent_cnn_v1"
CUSTOM_PPO_ACTOR_ARCH = "cnn_mlp"
CUSTOM_PPO_VEC_SCHEMA_VERSION = 1


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
    raw_cfg = payload.get("cfg") or {}
    # Canonicalize once at the read boundary so every downstream consumer of
    # the returned metadata can read ``latent_strategy_aux_return_*`` directly.
    cfg = canonicalize_latent_strategy_cfg(raw_cfg) if isinstance(raw_cfg, dict) else raw_cfg
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


# ----------------------------------------------------------------------
# Legacy config-key canonicalization for the latent strategy aux-return head.
#
# Older checkpoints and CLI flags used ``latent_strategy_q_head`` /
# ``latent_strategy_q_coef``. The canonical names are
# ``latent_strategy_aux_return_head`` / ``latent_strategy_aux_return_coef``;
# they reflect that q_phi(z|s) is **not** an action-value Q-function but an
# auxiliary per-z return regression head. All downstream code (trainer, model
# kwargs, snapshots) reads only the canonical names — legacy keys are folded
# in ONCE here, at the config-load boundary, instead of every reader
# repeatedly running ``getattr(..., "latent_strategy_q_*")`` fallbacks.
# ----------------------------------------------------------------------

_LATENT_STRATEGY_LEGACY_KEY_MAP: tuple[tuple[str, str], ...] = (
    ("latent_strategy_q_head", "latent_strategy_aux_return_head"),
    ("latent_strategy_q_coef", "latent_strategy_aux_return_coef"),
)


def canonicalize_latent_strategy_cfg(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy of ``cfg`` with legacy aux-return keys folded into canonical names.

    Idempotent: passing an already-canonical dict returns an equivalent copy.
    If both a legacy and canonical key are present the canonical key wins (i.e.
    a newer in-place fix takes precedence over a still-present legacy alias).
    """
    out: dict[str, Any] = dict(cfg)
    for legacy_key, canonical_key in _LATENT_STRATEGY_LEGACY_KEY_MAP:
        if legacy_key in out and canonical_key not in out:
            out[canonical_key] = out[legacy_key]
        out.pop(legacy_key, None)
    return out


def _effective_latent_aux_return_head(cfg: Any) -> bool:
    """Whether the aux-return head is enabled.

    Accepts canonical or legacy cfg shape (mapping or object). After Step 5
    new callers should canonicalize once with
    :func:`canonicalize_latent_strategy_cfg` and then read the canonical
    attribute directly; this wrapper exists for the boundary helpers that
    still receive an unvalidated mapping/object.
    """
    if isinstance(cfg, Mapping):
        canonical = canonicalize_latent_strategy_cfg(cfg)
        return bool(canonical.get("latent_strategy_aux_return_head", False))
    return bool(getattr(cfg, "latent_strategy_aux_return_head", False)) or bool(
        getattr(cfg, "latent_strategy_q_head", False)
    )


def _effective_latent_aux_return_coef(cfg: Any) -> float:
    if isinstance(cfg, Mapping):
        canonical = canonicalize_latent_strategy_cfg(cfg)
        return max(0.0, float(canonical.get("latent_strategy_aux_return_coef", 0.0) or 0.0))
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


def _load_model_state_dict_compat(model: nn.Module, sd: Mapping[str, Any]) -> None:
    """Load checkpoints while allowing the new opt-in episode baseline head to be absent in older files.

    Two layers of legacy compat run before ``load_state_dict``:

    1. :func:`_remap_legacy_strategy_aux_head_state_dict` — old
       ``strategy_q_head.*`` → new ``strategy_aux_return_head.*``.
    2. :func:`remap_legacy_actor_state_dict_keys` — pre-composition
       ``actor_body.*``/``actor_head.*``/``strategy_embedding.*`` → composed
       ``latent_actor.body.*``/``latent_actor.action_head.*``/``latent_actor.strategy_embedding.*``.

    Both helpers are idempotent so already-migrated state dicts pass through.
    """
    aux_remapped = _remap_legacy_strategy_aux_head_state_dict(sd)
    actor_remapped = remap_legacy_actor_state_dict_keys(aux_remapped)
    result = model.load_state_dict(actor_remapped, strict=False)
    missing = list(getattr(result, "missing_keys", []))
    unexpected = list(getattr(result, "unexpected_keys", []))
    allowed_missing = [k for k in missing if k.startswith("episode_strategy_value_head.")]
    disallowed_missing = [k for k in missing if k not in allowed_missing]
    if disallowed_missing or unexpected:
        raise RuntimeError(
            "Incompatible model state_dict: "
            f"missing={disallowed_missing!r}, unexpected={unexpected!r}"
        )


def _model_kwargs_from_cfg(cfg: Any) -> dict[str, Any]:
    if not isinstance(cfg, dict):
        return {}
    cfg = canonicalize_latent_strategy_cfg(cfg)
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
                "use_strategy_aux_return_head": bool(
                    cfg.get("latent_strategy_aux_return_head", False)
                ),
                "use_episode_strategy_value_head": bool(cfg.get("latent_episode_strategy_ppo", False)),
                "strategy_tau": float(cfg.get("latent_strategy_tau", 1.0) or 1.0),
            }
        )
    return kwargs


def apply_deterministic_sampling_generators(
    model: SharedActorCentralizedCritic,
    seed: int,
    *,
    device: torch.device | str,
) -> None:
    """Attach separate :class:`torch.Generator` copies for team-strategy vs per-head action sampling."""
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
    _load_model_state_dict_compat(model, payload["model_state_dict"])
    raw_ckpt_cfg = payload.get("cfg") or {}
    # Single canonicalization at the boundary so the inference policy + any
    # ``cfg``-key consumers see only ``latent_strategy_aux_return_*`` names.
    ckpt_cfg = (
        canonicalize_latent_strategy_cfg(raw_ckpt_cfg)
        if isinstance(raw_ckpt_cfg, dict)
        else raw_ckpt_cfg
    )
    if isinstance(ckpt_cfg, dict) and "seed" in ckpt_cfg:
        apply_deterministic_sampling_generators(model, int(ckpt_cfg["seed"]), device=device_t)
    return CustomPPOInferencePolicy(model, device=device_t, cfg=ckpt_cfg)


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
        self.latent_eval_mode: str = "normal"
        self._latent_eval_marginal: Optional[torch.Tensor] = None
        self._latent_eval_rng: Optional[torch.Generator] = None

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

    def set_latent_eval_mode(
        self,
        mode: str,
        *,
        marginal: Optional[Iterable[float]] = None,
        seed: Optional[int] = None,
    ) -> None:
        m = str(mode).strip().lower()
        if m not in {"normal", "uniform_random", "shuffled", "fixed"}:
            raise ValueError(
                f"latent_eval_mode must be one of normal|uniform_random|shuffled|fixed, got {mode!r}"
            )
        self.latent_eval_mode = m
        if marginal is not None and self.model.uses_latent_strategy:
            marg = torch.as_tensor(list(marginal), dtype=torch.float32, device=self.device)
            if int(marg.numel()) != int(self.model.latent_k):
                raise ValueError(
                    f"latent_eval_marginal must have length latent_k={self.model.latent_k}, got {int(marg.numel())}"
                )
            total = float(marg.sum().item())
            if total <= 0.0:
                raise ValueError("latent_eval_marginal must sum to > 0")
            self._latent_eval_marginal = marg / total
        elif m == "shuffled" and self._latent_eval_marginal is None:
            print(
                "[CustomPPOInferencePolicy] latent_eval_mode='shuffled' but no marginal provided; "
                "falling back to uniform marginal."
            )
            self._latent_eval_marginal = torch.full(
                (int(self.model.latent_k),), 1.0 / max(1, int(self.model.latent_k)), device=self.device
            )
        if seed is None:
            seed = 0x5EE_D + (0 if m == "normal" else 1)
        self._latent_eval_rng = torch.Generator(device=self.device)
        self._latent_eval_rng.manual_seed(int(seed) & 0xFFFFFFFF)

    def _destructive_latent_z(self, batch: int) -> torch.Tensor:
        K = max(1, int(self.model.latent_k))
        if self.latent_eval_mode == "uniform_random":
            return torch.randint(
                low=0,
                high=K,
                size=(int(batch),),
                generator=self._latent_eval_rng,
                device=self.device,
                dtype=torch.long,
            )
        if self.latent_eval_mode == "shuffled":
            probs = self._latent_eval_marginal
            if probs is None:
                probs = torch.full((K,), 1.0 / K, device=self.device)
            cat = Categorical(probs=probs.unsqueeze(0).expand(int(batch), -1))
            if self._latent_eval_rng is None:
                return cat.sample()
            return torch.multinomial(
                probs.unsqueeze(0).expand(int(batch), -1),
                num_samples=1,
                replacement=True,
                generator=self._latent_eval_rng,
            ).squeeze(-1).long()
        raise AssertionError(f"_destructive_latent_z called in mode {self.latent_eval_mode!r}")

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
                destructive = self.latent_eval_mode in ("uniform_random", "shuffled")
                if self.fixed_latent_strategy:
                    z_idx = self._fixed_strategy_tensor(batch)
                    self._prev_z = z_idx.detach()
                    z_ent = torch.zeros((batch,), dtype=torch.float32, device=self.device)
                    z_probs = self._fixed_strategy_probs(batch)
                    needs_strategy = False
                elif destructive:
                    z_logits = self.model.strategy_logits(context_gs)
                    needs_strategy = (
                        self._prev_z is None
                        or int(self._prev_z.numel()) != batch
                        or (self.strategy_interval > 0 and self._strategy_age >= self.strategy_interval)
                    )
                    if needs_strategy:
                        z_idx = self._destructive_latent_z(batch)
                        self._prev_z = z_idx.detach()
                        self._strategy_age = 0
                    else:
                        z_idx = self._prev_z.to(self.device)
                    z_ent = Categorical(logits=z_logits).entropy()
                    z_probs = torch.softmax(z_logits, dim=-1)
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
