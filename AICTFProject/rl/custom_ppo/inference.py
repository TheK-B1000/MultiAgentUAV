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
    router_context_mode = str(cfg.get("router_context_mode", "") or "")
    if uses_latent and router_context_mode == "current":
        # V6I7: q_phi uses raw global state + scheduler phase, not the EMA context stack.
        expected_dim = GLOBAL_STATE_DIM + 1
    else:
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
        meta["map_layout"] = str(cfg.get("map_layout", "map_a_open") or "map_a_open")
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


def _get_config_value(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict) or hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _is_z_adapter_active(sd: Mapping[str, Any], checkpoint_cfg: Optional[Mapping[str, Any]], target_cfg: Optional[Any]) -> bool:
    """Check if the z_adapter module in the checkpoint is effectively active."""
    # If not enabled in source run, it is a no-op
    if checkpoint_cfg is not None:
        ckpt_enabled = bool(_get_config_value(checkpoint_cfg, "latent_actor_z_adapter_enabled", False))
        ckpt_scale = float(_get_config_value(checkpoint_cfg, "latent_actor_z_adapter_scale", 0.0))
        if not ckpt_enabled or ckpt_scale <= 0.0:
            return False

    weight = sd.get("latent_actor.z_adapter.weight")
    if weight is None:
        return False

    # Get allowed latents for target router/run
    latent_k = _get_config_value(checkpoint_cfg, "latent_k")
    if latent_k is None:
        latent_k = _get_config_value(target_cfg, "latent_k", 4)

    allowed_latents = _get_config_value(target_cfg, "router_allowed_latents")
    if allowed_latents is None:
        allowed_latents = list(range(latent_k))

    # check if weight[z] is non-zero for any allowed latent
    for z in allowed_latents:
        if z < weight.shape[0]:
            if torch.max(torch.abs(weight[z])).item() > 1e-4:
                return True
    return False


def _is_actor_z_film_active(sd: Mapping[str, Any], checkpoint_cfg: Optional[Mapping[str, Any]], target_cfg: Optional[Any]) -> bool:
    """Check if the actor_z_film module in the checkpoint is effectively active."""
    # If not enabled in source run, it is a no-op
    if checkpoint_cfg is not None:
        ckpt_enabled = bool(_get_config_value(checkpoint_cfg, "enable_actor_z_film", False))
        if not ckpt_enabled:
            return False

    film_weight = sd.get("latent_actor.actor_z_film.weight")
    film_bias = sd.get("latent_actor.actor_z_film.bias")
    embed_weight = sd.get("latent_actor.strategy_embedding.weight")

    if film_weight is None:
        return False

    # Get allowed latents
    latent_k = _get_config_value(checkpoint_cfg, "latent_k")
    if latent_k is None:
        latent_k = _get_config_value(target_cfg, "latent_k", 4)

    allowed_latents = _get_config_value(target_cfg, "router_allowed_latents")
    if allowed_latents is None:
        if embed_weight is not None:
            allowed_latents = list(range(embed_weight.shape[0]))
        else:
            allowed_latents = list(range(latent_k))

    if embed_weight is not None and film_bias is not None:
        z_embed_scale = float(_get_config_value(checkpoint_cfg, "latent_actor_z_embed_scale", 1.0))
        for z in allowed_latents:
            if z < embed_weight.shape[0]:
                z_emb = embed_weight[z] * z_embed_scale
                # Compute effective output of linear layer
                out = torch.matmul(film_weight, z_emb) + film_bias
                half = out.shape[0] // 2
                gamma = out[:half]
                beta = out[half:]

                # Identity condition is gamma = 1.0 and beta = 0.0
                if torch.max(torch.abs(gamma - 1.0)).item() > 1e-4 or torch.max(torch.abs(beta)).item() > 1e-4:
                    return True
    else:
        # Fallback to simple weight check if embed weight is missing
        if torch.max(torch.abs(film_weight)).item() > 1e-4:
            return True
        if film_bias is not None:
            half = film_bias.shape[0] // 2
            gamma_bias = film_bias[:half]
            beta_bias = film_bias[half:]
            if torch.max(torch.abs(gamma_bias - 1.0)).item() > 1e-4 or torch.max(torch.abs(beta_bias)).item() > 1e-4:
                return True

    return False


def run_behavioral_equivalence_probe(
    source_model: nn.Module,
    target_model: nn.Module,
    observation_space: Any,
    allowed_latents: list[int],
    device: torch.device
) -> tuple[float, float, float, int]:
    """Run a behavioral probe check on a fixed probe bank for the specified allowed latents.
    
    Returns: (mean_kl, max_kl, max_logit_diff, argmax_disagreement)
    """
    source_model.eval()
    target_model.eval()
    
    batch_size = 5
    n_agents = getattr(source_model, "n_agents", 4)
    
    grid_shape = observation_space.spaces["grid"].shape
    vec_shape = observation_space.spaces["vec"].shape
    mask_shape = observation_space.spaces["mask"].shape
    
    grid = torch.linspace(0.0, 1.0, steps=batch_size * n_agents * grid_shape[1] * grid_shape[2] * grid_shape[3], device=device).reshape(batch_size, n_agents, *grid_shape[1:])
    vec = torch.linspace(-0.5, 0.5, steps=batch_size * n_agents * vec_shape[1], device=device).reshape(batch_size, n_agents, vec_shape[1])
    agent_mask = torch.ones((batch_size, n_agents), device=device)
    mask = torch.ones((batch_size, mask_shape[0]), device=device)
    
    obs = {
        "grid": grid,
        "vec": vec,
        "agent_mask": agent_mask,
        "mask": mask
    }
    
    all_kls = []
    all_max_logit_diffs = []
    total_argmax_disagreements = 0
    
    with torch.no_grad():
        for z in allowed_latents:
            z_idx = torch.full((batch_size,), z, dtype=torch.long, device=device)
            
            src_logits = source_model.policy_logits(obs, z_idx=z_idx)
            tgt_logits = target_model.policy_logits(obs, z_idx=z_idx)
            
            src_flat = src_logits.reshape(batch_size * n_agents, -1)
            tgt_flat = tgt_logits.reshape(batch_size * n_agents, -1)
            
            offset = 0
            for dim in source_model.per_agent_action_dims:
                src_chunk = src_flat[:, offset : offset + dim]
                tgt_chunk = tgt_flat[:, offset : offset + dim]
                
                src_dist = Categorical(logits=src_chunk)
                tgt_dist = Categorical(logits=tgt_chunk)
                
                kl = torch.distributions.kl.kl_divergence(src_dist, tgt_dist)
                all_kls.extend(kl.cpu().tolist())
                
                all_max_logit_diffs.append(torch.max(torch.abs(src_chunk - tgt_chunk)).item())
                
                src_argmax = torch.argmax(src_chunk, dim=-1)
                tgt_argmax = torch.argmax(tgt_chunk, dim=-1)
                total_argmax_disagreements += torch.sum(src_argmax != tgt_argmax).item()
                
                offset += dim
                
    mean_kl = float(np.mean(all_kls)) if all_kls else 0.0
    max_kl = float(np.max(all_kls)) if all_kls else 0.0
    max_logit_diff = float(np.max(all_max_logit_diffs)) if all_max_logit_diffs else 0.0
    
    return mean_kl, max_kl, max_logit_diff, total_argmax_disagreements


def _expand_cnn_obs_channels(sd: dict[str, Any], target_channels: int) -> dict[str, Any]:
    """Expand first-layer CNN weights when checkpoint has fewer input channels than the model.

    Called before load_state_dict when loading a 7-channel checkpoint into an 8-channel model.
    The new channel (obstacle) is zero-initialized so the policy is behaviorally unchanged at
    initialization — the warm-start contract from the audit report.
    """
    # Key may appear under the composed latent_actor namespace or the legacy flat namespace.
    candidates = (
        "latent_actor.actor_cnn.conv.0.weight",
        "actor_cnn.conv.0.weight",
    )
    for key in candidates:
        if key not in sd:
            continue
        w = sd[key]  # (out_channels, in_channels, kH, kW)
        src_ch = int(w.shape[1])
        if src_ch == target_channels:
            return sd  # already correct
        if src_ch < target_channels:
            new_w = w.new_zeros(w.shape[0], target_channels, *w.shape[2:])
            new_w[:, :src_ch] = w  # copy existing; new channels stay zero
            sd = dict(sd)
            sd[key] = new_w
            print(
                f"[checkpoint compat] CNN input channel expansion: {key} "
                f"{src_ch}→{target_channels} (new channels zero-initialized)"
            )
        return sd
    return sd


def _load_model_state_dict_compat(
    model: nn.Module,
    sd: Mapping[str, Any],
    allow_active_actor_migration: bool = False,
    checkpoint_cfg: Optional[Mapping[str, Any]] = None,
    target_cfg: Optional[Any] = None,
    observation_space: Optional[Any] = None,
    action_space: Optional[Any] = None,
) -> None:
    """Load checkpoints while allowing the new opt-in episode baseline head to be absent in older files.

    Three layers of legacy compat run before ``load_state_dict``:

    1. :func:`_remap_legacy_strategy_aux_head_state_dict` — old
       ``strategy_q_head.*`` → new ``strategy_aux_return_head.*``.
    2. :func:`remap_legacy_actor_state_dict_keys` — pre-composition
       ``actor_body.*``/``actor_head.*``/``strategy_embedding.*`` → composed
       ``latent_actor.body.*``/``latent_actor.action_head.*``/``latent_actor.strategy_embedding.*``.
    3. :func:`_expand_cnn_obs_channels` — 7→8 channel warm-start when the target
       model uses the obstacle observation channel but the checkpoint does not.

    All helpers are idempotent so already-migrated state dicts pass through.
    """
    aux_remapped = _remap_legacy_strategy_aux_head_state_dict(sd)
    actor_remapped = remap_legacy_actor_state_dict_keys(aux_remapped)

    # Detect target channel count from the model's first CNN conv layer.
    _cnn_key = "latent_actor.actor_cnn.conv.0.weight"
    _alt_key = "actor_cnn.conv.0.weight"
    _model_sd = dict(model.state_dict())
    _target_ch = None
    for _k in (_cnn_key, _alt_key):
        if _k in _model_sd:
            _target_ch = int(_model_sd[_k].shape[1])
            break
    if _target_ch is not None and _target_ch > 1:
        actor_remapped = _expand_cnn_obs_channels(actor_remapped, _target_ch)
    result = model.load_state_dict(actor_remapped, strict=False)
    missing = list(getattr(result, "missing_keys", []))
    unexpected = list(getattr(result, "unexpected_keys", []))
    _V6I7_RESIDUAL_PREFIXES = (
        "latent_actor.latent_adapters.",
        "latent_actor.latent_adapter_gates",
        "latent_actor.latent_action_biases",
    )
    allowed_missing = [k for k in missing if k.startswith("episode_strategy_value_head.")]
    allowed_missing.extend(k for k in missing if k.startswith("latent_actor.z_adapter."))
    allowed_missing.extend(k for k in missing if k.startswith("latent_actor.actor_z_film."))
    allowed_missing.extend(
        k for k in missing if any(k.startswith(p) for p in _V6I7_RESIDUAL_PREFIXES)
    )
    disallowed_missing = [k for k in missing if k not in allowed_missing]

    allowed_unexpected = [k for k in unexpected if k.startswith("episode_strategy_value_head.")]
    allowed_unexpected.extend(k for k in unexpected if k.startswith("latent_actor.z_adapter."))
    allowed_unexpected.extend(k for k in unexpected if k.startswith("latent_actor.actor_z_film."))
    allowed_unexpected.extend(
        k for k in unexpected if any(k.startswith(p) for p in _V6I7_RESIDUAL_PREFIXES)
    )
    disallowed_unexpected = [k for k in unexpected if k not in allowed_unexpected]

    # Report newly initialized parameters so the caller knows what was not loaded.
    newly_initialized = [
        k for k in missing if any(k.startswith(p) for p in _V6I7_RESIDUAL_PREFIXES)
    ]
    if newly_initialized:
        print("[checkpoint compat] Newly initialized parameters (not in checkpoint):")
        for k in sorted(newly_initialized):
            print(f"  {k}")

    if disallowed_missing or disallowed_unexpected:
        raise RuntimeError(
            "Incompatible model state_dict: "
            f"missing={disallowed_missing!r}, unexpected={disallowed_unexpected!r}"
        )

    # Categorize the ignored keys for logging
    ignored_aux = [k for k in unexpected if k.startswith("episode_strategy_value_head.")]
    ignored_actor_keys = [k for k in unexpected if k.startswith("latent_actor.z_adapter.") or k.startswith("latent_actor.actor_z_film.")]

    # Check if any ignored actor modules were active in the checkpoint
    active_prefixes = []
    if any(k.startswith("latent_actor.z_adapter.") for k in ignored_actor_keys):
        if _is_z_adapter_active(actor_remapped, checkpoint_cfg, target_cfg):
            active_prefixes.append("latent_actor.z_adapter.")
    if any(k.startswith("latent_actor.actor_z_film.") for k in ignored_actor_keys):
        if _is_actor_z_film_active(actor_remapped, checkpoint_cfg, target_cfg):
            active_prefixes.append("latent_actor.actor_z_film.")

    # Classify the loader outcome
    if not ignored_actor_keys:
        outcome = "EXACT"
    elif not active_prefixes:
        outcome = "NOOP_MODULE_ELISION"
    else:
        outcome = "ACTIVE_MIGRATION"

    migration_override_allowed = allow_active_actor_migration or os.environ.get("ALLOW_ACTIVE_COMPAT_MIGRATION") == "1"

    if ignored_aux or ignored_actor_keys:
        print(f"[checkpoint compat] Ignored auxiliary extras: {ignored_aux}")
        print(f"[checkpoint compat] Ignored actor-affecting extras: {ignored_actor_keys}")
        print(f"[checkpoint compat] Loader outcome: {outcome}")

        if outcome == "ACTIVE_MIGRATION":
            print("[checkpoint compat] WARNING: ACTIVE ACTOR MODULE MIGRATION ENABLED")
            print("[checkpoint compat] Resulting policy is not behaviorally equivalent to source checkpoint.")
            
            if not migration_override_allowed:
                raise RuntimeError(
                    f"Incompatible active actor-affecting parameters: {active_prefixes}. "
                    "These modules were active/nonzero in the checkpoint but are disabled in the target preset. "
                    "Omitting them changes the policy behavior. "
                    "To override this and proceed anyway, use --allow-active-actor-module-migration or set ALLOW_ACTIVE_COMPAT_MIGRATION=1."
                )

    # Perform behavioral-equivalence check on a fixed probe bank if observation/action spaces are available
    if observation_space is not None and action_space is not None:
        device = next(model.parameters()).device
        latent_k = _get_config_value(checkpoint_cfg, "latent_k")
        if latent_k is None:
            latent_k = _get_config_value(target_cfg, "latent_k", 4)
            
        allowed_latents = _get_config_value(target_cfg, "router_allowed_latents")
        if allowed_latents is None:
            allowed_latents = list(range(latent_k))
            
        try:
            # Reconstruct the source-compatible model strictly
            source_model = SharedActorCentralizedCritic(
                observation_space,
                action_space,
                **_model_kwargs_from_cfg(checkpoint_cfg),
            ).to(device)
            # When newly-initialized params exist, actor_remapped lacks those keys.
            # Use strict=False so the source model's new modules stay zero-initialized,
            # matching the target model — the probe will confirm outputs are identical.
            _src_strict = not bool(newly_initialized)
            source_model.load_state_dict(actor_remapped, strict=_src_strict)
            
            # Compare target model vs source-compatible model
            mean_kl, max_kl, max_logit_diff, argmax_diff = run_behavioral_equivalence_probe(
                source_model,
                model,
                observation_space,
                allowed_latents,
                device
            )
            
            # Require tight tolerance for non-override cases
            if argmax_diff > 0 or max_kl >= 1e-6:
                print(f"[checkpoint compat] Behavioral-equivalence check: FAIL (mean_kl={mean_kl:.3e}, max_kl={max_kl:.3e}, max_logit_diff={max_logit_diff:.4e}, argmax_diff={argmax_diff})")
                if not migration_override_allowed:
                    raise RuntimeError(
                        f"Behavioral equivalence check failed: argmax_diff={argmax_diff}, max_kl={max_kl:.3e}. "
                        "The policy logits differ from the source checkpoint. "
                        "To override this and proceed anyway, use --allow-active-actor-module-migration or set ALLOW_ACTIVE_COMPAT_MIGRATION=1."
                    )
            else:
                if outcome == "NOOP_MODULE_ELISION":
                    print(f"[checkpoint compat] Behavioral-equivalence check: PASS (ignored actor extras were inactive/no-op; mean_kl={mean_kl:.3e}, max_kl={max_kl:.3e}, max_logit_diff={max_logit_diff:.4e}, argmax_diff={argmax_diff})")
                else:
                    print(f"[checkpoint compat] Behavioral-equivalence check: PASS (mean_kl={mean_kl:.3e}, max_kl={max_kl:.3e}, max_logit_diff={max_logit_diff:.4e}, argmax_diff={argmax_diff})")
        except Exception as exc:
            if isinstance(exc, RuntimeError) and "Behavioral equivalence check failed" in str(exc):
                raise
            print(f"[checkpoint compat] Behavioral-equivalence check: NOT_RUN (could not reconstruct source model: {exc})")
    else:
        print("[checkpoint compat] Behavioral-equivalence check: NOT_RUN (spaces not provided)")


def _model_kwargs_from_cfg(cfg: Any) -> dict[str, Any]:
    if not isinstance(cfg, dict):
        return {}
    cfg = canonicalize_latent_strategy_cfg(cfg)
    experiment_id = str(cfg.get("experiment_id", "") or "").lower()
    router_context_mode = str(cfg.get("router_context_mode", "") or "")
    router_current_plus_delta = router_context_mode == "current_plus_delta"
    v6_staged = (
        bool(cfg.get("use_v6i1_curriculum", False))
        and str(cfg.get("training_mode", "default") or "default") == "staged_team_intent_curriculum"
        and str(cfg.get("experiment_family", "v6") or "v6") == "v6"
        and experiment_id in {"v6i1", "v6i2", "v6i3", "v6i5"}
    )
    kwargs: dict[str, Any] = {
        "actor_cnn_feature_dim": int(cfg.get("actor_cnn_feature_dim", 128)),
    }
    if bool(cfg.get("use_latent_strategy", False)):
        kwargs.update(
            {
                "latent_k": int(cfg.get("latent_k", 4)),
                "z_embed_dim": int(cfg.get("latent_z_embed_dim", 16)),
                "router_context_mode": router_context_mode,
                "router_context_dimension": int(cfg.get("router_context_dimension", 0) or 0),
                "strategy_hidden_dim": int(cfg.get("latent_strategy_hidden", 128)),
                "critic_hidden_dim": int(cfg.get("latent_vf_hidden", 128)),
                "use_strategy_aux_return_head": bool(
                    cfg.get("latent_strategy_aux_return_head", False)
                ),
                # Mirror the trainer-side gating in
                # ``rl/custom_ppo/trainer_config.py``: the
                # ``episode_strategy_value_head`` module is built when EITHER
                # episode-level strategy PPO is on, OR arc-credit is enabled
                # with the ``context_value`` baseline (which reuses this head
                # as V_phi(ctx, z)). v3i19+ runs (incl. v4i1) trip the arc-
                # credit branch; before this fix the loader was missing it
                # and rejected those checkpoints with ``unexpected=[
                # 'episode_strategy_value_head.*']``.
                "use_episode_strategy_value_head": bool(
                    cfg.get("latent_episode_strategy_ppo", False)
                    or v6_staged
                    or (
                        cfg.get("latent_arc_credit_enabled", False)
                        and str(
                            cfg.get("latent_arc_credit_baseline", "context_value")
                            or "context_value"
                        ).lower() == "context_value"
                    )
                ),
                "recurrent_selector_hidden_dim": int(
                    cfg.get("recurrent_selector_hidden_dim", None)
                    or cfg.get("v6i1_recurrent_selector_hidden", 32)
                    or 32
                ),
                "use_recurrent_selector": bool(
                    (v6_staged and not router_current_plus_delta)
                    or bool(cfg.get("use_recurrent_selector", False))
                    or (
                        router_context_mode == "current"
                        and int(cfg.get("recurrent_selector_hidden_dim", 0) or 0) > 0
                    )
                ),
                "strategy_tau": float(cfg.get("latent_strategy_tau", 1.0) or 1.0),
                "latent_actor_z_onehot_enabled": bool(
                    cfg.get("latent_actor_z_onehot_enabled", False)
                ),
                "latent_actor_z_onehot_scale": float(
                    1.0
                    if cfg.get("latent_actor_z_onehot_scale", 1.0) is None
                    else cfg.get("latent_actor_z_onehot_scale", 1.0)
                ),
                "latent_actor_z_embed_scale": float(
                    1.0
                    if cfg.get("latent_actor_z_embed_scale", 1.0) is None
                    else cfg.get("latent_actor_z_embed_scale", 1.0)
                ),
                "latent_actor_z_adapter_enabled": bool(
                    cfg.get("latent_actor_z_adapter_enabled", False)
                ),
                "latent_actor_z_adapter_scale": float(
                    cfg.get("latent_actor_z_adapter_scale", 0.0) or 0.0
                ),
                "latent_actor_z_adapter_init_std": float(
                    cfg.get("latent_actor_z_adapter_init_std", 0.02) or 0.02
                ),
                "latent_actor_z_film_layers": int(
                    cfg.get("latent_actor_z_film_layers", 1) or 1
                ),
                "enable_actor_z_film": bool(
                    cfg.get("enable_actor_z_film", False)
                ),
                "actor_z_film_init_scale": float(
                    cfg.get("actor_z_film_init_scale", 0.0) or 0.0
                ),
                "actor_z_film_layer": int(
                    cfg.get("actor_z_film_layer", 2) or 2
                ),
                # V6I7: per-latent residual adapters.
                "enable_latent_z_residual": bool(
                    cfg.get("enable_latent_z_residual", False)
                ),
                "latent_z_gate_init": float(
                    cfg.get("latent_z_gate_init", 0.01) or 0.01
                ),
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
    _load_model_state_dict_compat(
        model,
        payload["model_state_dict"],
        checkpoint_cfg=payload.get("cfg"),
        observation_space=observation_space,
        action_space=action_space,
    )
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
        if self.device.type == "cuda" and self.device.index is None:
            if torch.cuda.is_available():
                self.device = torch.device(f"cuda:{torch.cuda.current_device()}")
        self.model.to(self.device)
        self.model.eval()
        self._prev_z = None
        cfg = cfg or {}
        self.router_allowed_latents = cfg.get("router_allowed_latents", None)
        self._previous_opportunity_features = None
        self._opportunity_occurred = None
        self.strategy_interval = max(0, int(cfg.get("latent_resample_every_n", 0) or 0))
        self._original_strategy_interval = self.strategy_interval
        self.fixed_latent_strategy = bool(cfg.get("fixed_latent_strategy", False))
        self.fixed_latent_strategy_id = max(0, int(cfg.get("fixed_latent_strategy_id", 0) or 0))
        self._strategy_age = 0
        self._last_strategy_z = None
        self._last_strategy_probs = None
        self._last_strategy_entropy = None
        self._last_strategy_resampled = False
        self._last_strategy_logits = None
        self._last_context_gs = None
        self._temporal_tracker = None
        self._selector_hidden: torch.Tensor | None = None
        self.latent_eval_mode = "normal"
        self._latent_eval_marginal = None
        self._latent_eval_rng = None
        self._shuffled_mapping = None
        self._current_opponent = None
        self._current_seed = None
        self._current_episode_index = None
        self._current_eval_seed = None
        self._current_environment_seed = None
        self._current_env_index = None
        self._current_decision_step = 0
        self._opportunity_counter = 0
        self.opportunity_trace_log = []

    def set_current_episode_context(
        self,
        opponent: str,
        seed: int,
        episode_index: int,
    ) -> None:
        self._current_opponent = str(opponent).upper()
        self._current_seed = int(seed)
        self._current_episode_index = int(episode_index)
        self._current_eval_seed = int(seed)
        self._current_environment_seed = int(seed)
        self._current_env_index = int(episode_index)
        if isinstance(self._opportunity_counter, np.ndarray):
            self._opportunity_counter.fill(0)
        else:
            self._opportunity_counter = 0

    def set_eval_episode_context(
        self,
        opponent: str,
        eval_seed: int,
        environment_seed: int,
        env_index: int = 0,
    ) -> None:
        self._current_opponent = str(opponent).upper()
        self._current_eval_seed = int(eval_seed)
        self._current_environment_seed = int(environment_seed)
        self._current_env_index = int(env_index)
        self._current_seed = int(eval_seed)
        self._current_episode_index = int(env_index)
        if isinstance(self._opportunity_counter, np.ndarray):
            self._opportunity_counter.fill(0)
        else:
            self._opportunity_counter = 0

    def set_current_decision_step(self, step: int) -> None:
        self._current_decision_step = int(step)

    def inject_shuffled_mapping(self, mapping: dict) -> None:
        self._shuffled_mapping = mapping

    def clear_eval_suite_state(self) -> None:
        self._shuffled_mapping = None
        self.opportunity_trace_log = []
        self._current_opponent = None
        self._current_seed = None
        self._current_episode_index = None
        self._current_eval_seed = None
        self._current_environment_seed = None
        self._current_env_index = None
        self._current_decision_step = 0
        if isinstance(self._opportunity_counter, np.ndarray):
            self._opportunity_counter.fill(0)
        else:
            self._opportunity_counter = 0
        self.reset_strategy()

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
            allowed = self.router_allowed_latents
            if allowed is not None and len(allowed) > 0:
                print(
                    f"[CustomPPOInferencePolicy] latent_eval_mode='shuffled' fallback to uniform marginal over allowed {allowed}."
                )
                marginal = torch.zeros((int(self.model.latent_k),), device=self.device)
                val = 1.0 / len(allowed)
                for z in allowed:
                    marginal[z] = val
                self._latent_eval_marginal = marginal
            else:
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
        allowed = self.router_allowed_latents
        if allowed is not None and len(allowed) > 0:
            allowed_t = torch.tensor(allowed, dtype=torch.long, device=self.device)
            idx = torch.randint(
                low=0,
                high=len(allowed),
                size=(int(batch),),
                generator=self._latent_eval_rng,
                device=self.device,
                dtype=torch.long,
            )
            return allowed_t[idx]
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

    def reset_strategy(self, done_mask: Optional[np.ndarray | torch.Tensor] = None) -> None:
        """Forget the persisted inference strategy, typically at episode reset."""
        if done_mask is None:
            self._prev_z = None
            self._strategy_age = 0
            self._last_strategy_z = None
            self._last_strategy_probs = None
            self._last_strategy_entropy = None
            self._last_strategy_resampled = False
            self._last_strategy_logits = None
            self._last_context_gs = None
            self._selector_hidden = None
            self._opportunity_counter = 0
            self._opportunity_occurred = None
            self._previous_opportunity_features = None
            if self._temporal_tracker is not None:
                self._temporal_tracker.reset()
        else:
            mask = torch.as_tensor(done_mask, device=self.device).bool()
            batch = mask.shape[0]
            if self._prev_z is not None and self._prev_z.numel() == batch:
                if isinstance(self._strategy_age, torch.Tensor):
                    self._strategy_age[mask] = 0
                else:
                    self._strategy_age = 0
                if self._opportunity_occurred is not None:
                    self._opportunity_occurred[mask] = False
                if self._previous_opportunity_features is not None:
                    self._previous_opportunity_features[mask] = 0.0
                if self._temporal_tracker is not None:
                    self._temporal_tracker.reset(env_indices=mask)
                if isinstance(self._opportunity_counter, np.ndarray) and self._opportunity_counter.shape[0] == batch:
                    self._opportunity_counter[mask.cpu().numpy()] = 0

    def _uses_recurrent_selector(self) -> bool:
        return bool(getattr(self.model, "use_recurrent_selector", False))

    def _selector_hidden_dim(self) -> int:
        return int(getattr(self.model, "recurrent_selector_hidden_dim", 0) or 0)

    def _ensure_selector_hidden(self, batch: int) -> torch.Tensor | None:
        if not self._uses_recurrent_selector():
            return None
        hidden_dim = self._selector_hidden_dim()
        if hidden_dim <= 0:
            return None
        if self._selector_hidden is None or int(self._selector_hidden.shape[0]) != batch:
            self._selector_hidden = torch.zeros(
                (batch, hidden_dim), dtype=torch.float32, device=self.device
            )
        return self._selector_hidden

    def _strategy_logits_forward(self, context_gs: torch.Tensor) -> torch.Tensor:
        """Advance recurrent selector state and return tempered q_phi logits."""
        hidden = self._ensure_selector_hidden(int(context_gs.shape[0]))
        if hidden is None:
            return self.model.strategy_logits(context_gs)
        logits, h_new = self.model._forward_q_phi(context_gs, hidden)
        self._selector_hidden = h_new.detach()
        return logits / self.model.strategy_tau

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
                _router_mode = str(getattr(self.model, "router_context_mode", "") or "")
                if _router_mode == "current":
                    # V6I7: EMA tracker not used; pad raw 34-dim state with scheduler phase
                    # zero to produce the 35-dim input the model was trained on.
                    if global_state.shape[-1] == GLOBAL_STATE_DIM:
                        global_state = torch.cat(
                            [global_state, torch.zeros((batch, 1), dtype=torch.float32, device=self.device)],
                            dim=-1,
                        )
                    context_gs = global_state
                else:
                    tracker = self._get_temporal_tracker(batch)
                    context_gs = tracker.update(global_state)

                # Check for batch-size or device change, and resize tracking
                if (
                    self._prev_z is None
                    or self._prev_z.numel() != batch
                    or self._prev_z.device != self.device
                ):
                    self._prev_z = self._fixed_strategy_tensor(batch) if self.fixed_latent_strategy else torch.zeros((batch,), dtype=torch.long, device=self.device)
                    self._strategy_age = torch.zeros((batch,), dtype=torch.long, device=self.device)
                    self._opportunity_occurred = torch.zeros((batch,), dtype=torch.bool, device=self.device)
                    self._previous_opportunity_features = torch.zeros((batch, GLOBAL_STATE_DIM), dtype=torch.float32, device=self.device)
                    self._opportunity_counter = np.zeros((batch,), dtype=np.int64)
                
                # Build context for q_phi (the router)
                if self.model.router_current_plus_delta_enabled:
                    current = global_state[:, :GLOBAL_STATE_DIM].float()
                    previous = torch.zeros_like(current)
                    has_prev = self._opportunity_occurred
                    if has_prev.any():
                        previous[has_prev] = self._previous_opportunity_features[has_prev]
                    from rl.custom_ppo.latent.router_sampling import build_current_plus_delta_router_context
                    q_phi_context = build_current_plus_delta_router_context(global_state, previous)
                elif _router_mode == "current":
                    # V6I7: q_phi and critic both see raw global state (35-dim), not EMA stack.
                    q_phi_context = global_state
                else:
                    q_phi_context = context_gs

                if self.fixed_latent_strategy:
                    needs_strategy = torch.zeros((batch,), dtype=torch.bool, device=self.device)
                else:
                    needs_strategy = torch.zeros((batch,), dtype=torch.bool, device=self.device)
                    if self.strategy_interval > 0:
                        needs_strategy = needs_strategy | (self._strategy_age >= self.strategy_interval)
                    needs_strategy = needs_strategy | (~self._opportunity_occurred)
                
                # Retrieve z, logits, probabilities depending on modes.
                if self.fixed_latent_strategy:
                    z_idx = self._fixed_strategy_tensor(batch)
                    z_probs = self._fixed_strategy_probs(batch)
                    z_logits = torch.log(torch.clamp(z_probs, min=1e-8))
                    z_ent = torch.zeros((batch,), dtype=torch.float32, device=self.device)
                elif self.latent_eval_mode == "shuffled":
                    # Shuffled mode: enforce strict lookup
                    if self._shuffled_mapping is None:
                        raise ValueError("shuffled_mapping is not injected but mode is shuffled")
                    z_logits_full = self._strategy_logits_forward(q_phi_context)
                    if getattr(self, "_prev_logits", None) is None or self._prev_logits.shape[0] != batch:
                        self._prev_logits = torch.zeros((batch, self.model.latent_k), dtype=torch.float32, device=self.device)
                        self._prev_probs = torch.zeros((batch, self.model.latent_k), dtype=torch.float32, device=self.device)
                        self._prev_ent = torch.zeros((batch,), dtype=torch.float32, device=self.device)
                    
                    if needs_strategy.any():
                        if not isinstance(self._opportunity_counter, np.ndarray) or self._opportunity_counter.shape[0] != batch:
                            self._opportunity_counter = np.zeros((batch,), dtype=np.int64)
                        for env_idx in range(batch):
                            if needs_strategy[env_idx]:
                                opponent = self._current_opponent
                                eval_seed = getattr(self, "_current_eval_seed", None)
                                if eval_seed is None:
                                    eval_seed = self._current_seed
                                env_index = getattr(self, "_current_env_index", None)
                                if env_index is None:
                                    env_index = self._current_episode_index if self._current_episode_index is not None else 0
                                lookup_key = (
                                    opponent,
                                    eval_seed,
                                    env_index,
                                )
                                if lookup_key not in self._shuffled_mapping:
                                    raise ValueError(f"Shuffled mapping lookup failed for key: {lookup_key}")
                                decisions = self._shuffled_mapping[lookup_key]
                                opp_counter = int(self._opportunity_counter[env_idx])
                                if opp_counter >= len(decisions):
                                    raise ValueError(
                                        f"Shuffled mapping out of range for key: {lookup_key}, opportunity: {opp_counter} (max: {len(decisions)})"
                                    )
                                mapped_decision = decisions[opp_counter]
                                z_val = int(mapped_decision["selected_z"])
                                self._prev_z[env_idx] = z_val
                                self._prev_logits[env_idx] = torch.as_tensor(mapped_decision["logits"], dtype=torch.float32, device=self.device)
                                self._prev_probs[env_idx] = torch.softmax(self._prev_logits[env_idx], dim=-1)
                                self._prev_ent[env_idx] = Categorical(logits=self._prev_logits[env_idx]).entropy()
                                self._strategy_age[env_idx] = 0
                                self._opportunity_counter[env_idx] += 1
                        z_idx = self._prev_z.to(self.device)
                        z_logits = self._prev_logits.to(self.device)
                        z_probs = self._prev_probs.to(self.device)
                        z_ent = self._prev_ent.to(self.device)
                    else:
                        z_idx = self._prev_z.to(self.device)
                        z_logits = self._prev_logits.to(self.device)
                        z_probs = self._prev_probs.to(self.device)
                        z_ent = self._prev_ent.to(self.device)
                elif self.latent_eval_mode == "uniform_random":
                    z_logits = self._strategy_logits_forward(q_phi_context)
                    if needs_strategy.any():
                        z_idx_new = self._destructive_latent_z(batch)
                        self._prev_z = torch.where(needs_strategy, z_idx_new, self._prev_z)
                        self._strategy_age[needs_strategy] = 0
                    z_idx = self._prev_z.to(self.device)
                    z_probs = torch.softmax(z_logits, dim=-1)
                    z_ent = Categorical(logits=z_logits).entropy()
                else:
                    # normal or qphi_initial_only_no_switch
                    z_logits = self._strategy_logits_forward(q_phi_context)
                    z_probs = torch.softmax(z_logits, dim=-1)
                    z_ent = Categorical(logits=z_logits).entropy()
                    if needs_strategy.any():
                        hidden = self._ensure_selector_hidden(batch)
                        z_idx_sampled, _, z_ent_sampled, z_logits_sampled, h_new = self.model.sample_strategy(
                            q_phi_context,
                            deterministic=deterministic,
                            selector_hidden=hidden,
                        )
                        if h_new is not None:
                            self._selector_hidden = h_new.detach()
                        self._prev_z = torch.where(needs_strategy, z_idx_sampled, self._prev_z)
                        self._strategy_age[needs_strategy] = 0
                    z_idx = self._prev_z.to(self.device)

                if needs_strategy.any() and batch == 1:
                    prev_z_val = self.opportunity_trace_log[-1]["selected_z"] if self.opportunity_trace_log else -1
                    logit_list = z_logits.detach().cpu().numpy()[0].tolist()
                    prob_list = z_probs.detach().cpu().numpy()[0].tolist()
                    sel_z_val = int(z_idx.item())
                    
                    if self.latent_eval_mode == "shuffled":
                        opp_idx = int(self._opportunity_counter[0]) - 1
                    else:
                        opp_idx = int(self._opportunity_counter[0])
                        
                    self.opportunity_trace_log.append({
                        "opponent": self._current_opponent,
                        "seed": getattr(self, "_current_eval_seed", None) or self._current_seed,
                        "environment_seed": getattr(self, "_current_environment_seed", None) or self._current_seed,
                        "episode_index": getattr(self, "_current_env_index", None) or (self._current_episode_index if self._current_episode_index is not None else 0),
                        "opportunity_index": opp_idx,
                        "step": self._current_decision_step,
                        "logits": logit_list,
                        "probabilities": prob_list,
                        "selected_z": sel_z_val,
                        "prev_z": prev_z_val,
                        "switch_occurred": int(prev_z_val != -1 and sel_z_val != prev_z_val)
                    })
                    if self.latent_eval_mode != "shuffled":
                        if isinstance(self._opportunity_counter, np.ndarray):
                            self._opportunity_counter[0] += 1
                        else:
                            self._opportunity_counter += 1

                if needs_strategy.any() and self.model.router_current_plus_delta_enabled:
                    current = global_state[:, :GLOBAL_STATE_DIM].float()
                    self._previous_opportunity_features[needs_strategy] = current[needs_strategy].clone().detach()
                    self._opportunity_occurred[needs_strategy] = True

                self._last_strategy_z = z_idx.detach().cpu()
                self._last_strategy_probs = z_probs.detach().cpu()
                self._last_strategy_entropy = z_ent.detach().cpu()
                self._last_strategy_resampled = bool(needs_strategy.any().item())
                self._last_strategy_logits = z_logits.detach().cpu()
                self._last_context_gs = context_gs.detach().cpu()
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
                    
                    if self.model.router_current_plus_delta_enabled:
                        current = global_state[:, :GLOBAL_STATE_DIM].float()
                        previous = torch.zeros_like(current)
                        if self._opportunity_occurred is not None and self._opportunity_occurred.shape[0] == batch:
                            has_prev = self._opportunity_occurred
                            if has_prev.any():
                                previous[has_prev] = self._previous_opportunity_features[has_prev]
                        from rl.custom_ppo.latent.router_sampling import build_current_plus_delta_router_context
                        q_phi_context = build_current_plus_delta_router_context(global_state, previous)
                    else:
                        q_phi_context = context_gs
                    
                    hidden = self._ensure_selector_hidden(batch)
                    z_idx, _, z_entropy, _, h_new = self.model.sample_strategy(
                        q_phi_context,
                        deterministic=True,
                        selector_hidden=hidden,
                    )
                    if h_new is not None:
                        self._selector_hidden = h_new.detach()
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
        if self._last_strategy_logits is not None and self._last_strategy_logits.numel() > 0:
            l0 = self._last_strategy_logits.reshape(self._last_strategy_logits.shape[0], -1)[0]
            for idx, logit in enumerate(l0.tolist()):
                out[f"strategy_logit_{idx}"] = float(logit)
        if self._last_context_gs is not None and self._last_context_gs.numel() > 0:
            out["context_state"] = self._last_context_gs.reshape(self._last_context_gs.shape[0], -1)[0].numpy()
        return out
