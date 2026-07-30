from __future__ import annotations

import os
from typing import Any, Mapping, Optional

import torch
import torch.nn as nn

from rl.custom_ppo.policy import SharedActorCentralizedCritic, remap_legacy_actor_state_dict_keys

from .errors import CheckpointBehavioralEquivalenceError, CheckpointStateDictError
from .metadata import canonicalize_latent_strategy_cfg
from .models import MigrationRecord, StateDictLoadReport
from .validation import run_behavioral_equivalence_probe

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
                f"{src_ch}->{target_channels} (new channels zero-initialized)"
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
    model_sd = dict(model.state_dict())
    shape_skipped: list[str] = []
    filtered: dict[str, Any] = {}
    for key, value in actor_remapped.items():
        if key not in model_sd:
            filtered[key] = value
            continue
        if not isinstance(value, torch.Tensor):
            filtered[key] = value
            continue
        if tuple(model_sd[key].shape) != tuple(value.shape):
            shape_skipped.append(key)
            continue
        filtered[key] = value
    if shape_skipped:
        print(
            "[checkpoint compat] Skipped shape-mismatched parameters "
            f"({len(shape_skipped)} keys); target modules keep initialization."
        )
        for key in sorted(shape_skipped)[:12]:
            print(f"  {key}: checkpoint{tuple(actor_remapped[key].shape)} -> model{tuple(model_sd[key].shape)}")
        if len(shape_skipped) > 12:
            print(f"  ... and {len(shape_skipped) - 12} more")
    result = model.load_state_dict(filtered, strict=False)
    missing = list(getattr(result, "missing_keys", []))
    unexpected = list(getattr(result, "unexpected_keys", []))
    _V6I7_RESIDUAL_PREFIXES = (
        "latent_actor.latent_adapters.",
        "latent_actor.latent_adapter_gates",
        "latent_actor.latent_action_biases",
        "latent_actor.latent_action_heads.",
        "latent_actor.latent_branch_trunks.",
    )
    allowed_missing = [k for k in missing if k.startswith("episode_strategy_value_head.")]
    allowed_missing.extend(k for k in missing if k.startswith("latent_actor.z_adapter."))
    allowed_missing.extend(k for k in missing if k.startswith("latent_actor.actor_z_film."))
    allowed_missing.extend(
        k for k in missing if any(k.startswith(p) for p in _V6I7_RESIDUAL_PREFIXES)
    )
    router_reinit = bool(
        target_cfg is not None and getattr(target_cfg, "router_reinitialize_on_load", False)
    )
    if router_reinit or shape_skipped:
        allowed_missing.extend(
            k
            for k in missing
            if k.startswith("strategy_encoder.") or k.startswith("selector_gru.")
        )
    disallowed_missing = [k for k in missing if k not in allowed_missing]

    allowed_unexpected = [k for k in unexpected if k.startswith("episode_strategy_value_head.")]
    allowed_unexpected.extend(k for k in unexpected if k.startswith("latent_actor.z_adapter."))
    allowed_unexpected.extend(k for k in unexpected if k.startswith("latent_actor.actor_z_film."))
    allowed_unexpected.extend(
        k for k in unexpected if any(k.startswith(p) for p in _V6I7_RESIDUAL_PREFIXES)
    )
    if router_reinit or shape_skipped:
        allowed_unexpected.extend(
            k
            for k in unexpected
            if k.startswith("strategy_encoder.") or k.startswith("selector_gru.")
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
        # V6I23: if per-z action heads were not in the checkpoint, copy the loaded
        # shared action_head so specialists start at trunk-equivalent logits.
        if any(k.startswith("latent_actor.latent_action_heads.") for k in newly_initialized):
            la = getattr(model, "latent_actor", None)
            if la is not None and hasattr(la, "sync_per_z_action_heads_from_shared"):
                la.sync_per_z_action_heads_from_shared()
                print(
                    "[checkpoint compat] Synced latent_action_heads from loaded "
                    "shared action_head (population-birth start)."
                )
        if any(k.startswith("latent_actor.latent_branch_trunks.") for k in newly_initialized):
            la = getattr(model, "latent_actor", None)
            if la is not None and hasattr(la, "sync_latent_branch_trunks_to_identity"):
                la.sync_latent_branch_trunks_to_identity()
                print(
                    "[checkpoint compat] Initialized latent_branch_trunks as "
                    "identity transforms (LRO deep-branch start)."
                )
    if router_reinit or shape_skipped:
        router_missing = [
            k
            for k in missing
            if k.startswith("strategy_encoder.") or k.startswith("selector_gru.")
        ]
        if router_missing:
            print(
                "[checkpoint compat] Router modules left at target initialization "
                f"({len(router_missing)} keys)."
            )

    if disallowed_missing or disallowed_unexpected:
        raise CheckpointStateDictError(
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
                raise CheckpointStateDictError(
                    f"Incompatible active actor-affecting parameters: {active_prefixes}. "
                    "These modules were active/nonzero in the checkpoint but are disabled in the target preset. "
                    "Omitting them changes the policy behavior. "
                    "To override this and proceed anyway, use --allow-active-actor-module-migration or set ALLOW_ACTIVE_COMPAT_MIGRATION=1."
                )

    # Perform behavioral-equivalence check on a fixed probe bank if observation/action spaces are available
    skip_behavioral_equiv = bool(
        target_cfg is not None and getattr(target_cfg, "router_reinitialize_on_load", False)
    )
    if skip_behavioral_equiv:
        print(
            "[checkpoint compat] Skipping full-model behavioral-equivalence check "
            "(router_reinitialize_on_load=True; repertoire actor weights loaded only)."
        )
    if observation_space is not None and action_space is not None and not skip_behavioral_equiv:
        device = next(model.parameters()).device
        latent_k = _get_config_value(checkpoint_cfg, "latent_k")
        if latent_k is None:
            latent_k = _get_config_value(target_cfg, "latent_k", 4)
            
        allowed_latents = _get_config_value(target_cfg, "router_allowed_latents")
        if allowed_latents is None:
            allowed_latents = list(range(latent_k))
            
        try:
            from rl.custom_ppo.checkpoints.loader import _model_kwargs_from_cfg

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

            # V6I22E/V6I23: if adapters or per-z heads are newly initialized,
            # temporarily bypass residual + per-z heads so the equivalence check
            # confirms the shared trunk is intact.
            _adapter_bypass_set = False
            _fixed_alpha_mode = (
                bool(newly_initialized)
                and (
                    float(getattr(target_cfg, "latent_z_residual_alpha", 0.0) or 0.0) > 0
                    or bool(
                        getattr(
                            target_cfg,
                            "latent_population_birth_per_z_action_heads",
                            False,
                        )
                    )
                )
            )
            if _fixed_alpha_mode:
                la = getattr(model, "latent_actor", None)
                if la is not None:
                    la._residual_bypass_for_compat = True
                    _adapter_bypass_set = True

            # Compare target model vs source-compatible model
            mean_kl, max_kl, max_logit_diff, argmax_diff = run_behavioral_equivalence_probe(
                source_model,
                model,
                observation_space,
                allowed_latents,
                device
            )

            if _adapter_bypass_set:
                la = getattr(model, "latent_actor", None)
                if la is not None:
                    la._residual_bypass_for_compat = False
            
            # Require tight tolerance for non-override cases
            if argmax_diff > 0 or max_kl >= 1e-6:
                print(f"[checkpoint compat] Behavioral-equivalence check: FAIL (mean_kl={mean_kl:.3e}, max_kl={max_kl:.3e}, max_logit_diff={max_logit_diff:.4e}, argmax_diff={argmax_diff})")
                if not migration_override_allowed:
                    raise CheckpointStateDictError(
                        f"Behavioral equivalence check failed: argmax_diff={argmax_diff}, max_kl={max_kl:.3e}. "
                        "The policy logits differ from the source checkpoint. "
                        "To override this and proceed anyway, use --allow-active-actor-module-migration or set ALLOW_ACTIVE_COMPAT_MIGRATION=1."
                    )
            else:
                if _adapter_bypass_set:
                    print(f"[checkpoint compat] Behavioral-equivalence check: PASS (trunk-only; residual/per-z specialists bypassed; mean_kl={mean_kl:.3e}, max_kl={max_kl:.3e}, max_logit_diff={max_logit_diff:.4e}, argmax_diff={argmax_diff})")
                elif outcome == "NOOP_MODULE_ELISION":
                    print(f"[checkpoint compat] Behavioral-equivalence check: PASS (ignored actor extras were inactive/no-op; mean_kl={mean_kl:.3e}, max_kl={max_kl:.3e}, max_logit_diff={max_logit_diff:.4e}, argmax_diff={argmax_diff})")
                else:
                    print(f"[checkpoint compat] Behavioral-equivalence check: PASS (mean_kl={mean_kl:.3e}, max_kl={max_kl:.3e}, max_logit_diff={max_logit_diff:.4e}, argmax_diff={argmax_diff})")
        except Exception as exc:
            if _adapter_bypass_set:
                la = getattr(model, "latent_actor", None)
                if la is not None:
                    la._residual_bypass_for_compat = False
            if isinstance(exc, RuntimeError) and "Behavioral equivalence check failed" in str(exc):
                raise
            print(f"[checkpoint compat] Behavioral-equivalence check: NOT_RUN (could not reconstruct source model: {exc})")
    else:
        print("[checkpoint compat] Behavioral-equivalence check: NOT_RUN (spaces not provided)")


def load_model_state_dict_compat_report(model: nn.Module, sd: Mapping[str, Any], **kwargs: Any) -> StateDictLoadReport:
    _load_model_state_dict_compat(model, sd, **kwargs)
    return StateDictLoadReport(missing_keys=(), unexpected_keys=())
