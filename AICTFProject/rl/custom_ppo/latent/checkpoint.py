"""Checkpoint schema helpers for modular latent state."""

from __future__ import annotations

from typing import Any

import numpy as np

SCHEMA_VERSION = 2


def validate_latent_checkpoint(payload: dict, *, latent_k: int, n_envs: int) -> None:
    version = int(payload.get("schema_version", 0))
    if version not in (0, SCHEMA_VERSION):
        raise ValueError("Unsupported latent checkpoint schema_version")
    if version == SCHEMA_VERSION and int(payload.get("latent_k", latent_k)) != int(latent_k):
        raise ValueError("latent_k mismatch in checkpoint")


def _component_intervention(state: Any) -> dict[str, Any]:
    return {
        "cf_J": state.cf_J.copy(),
        "cf_episode_counts": state.cf_episode_counts.copy(),
        "cf_has_experience": state.cf_has_experience.copy(),
        "cf_return_mean": float(state.cf_return_mean),
        "cf_return_var": float(state.cf_return_var),
        "pair_jsd_ema": state.pair_jsd_ema.copy(),
        "jsd_gate_consecutive_updates": int(state.jsd_gate_consecutive_updates),
        "pairwise_ema_valid_updates": int(getattr(state, "pairwise_ema_valid_updates", 0)),
        "pairwise_ema_last_update_step": int(getattr(state, "pairwise_ema_last_update_step", -1)),
        "cf_pair_jsd_ema": getattr(state, "cf_pair_jsd_ema", np.zeros(6, dtype=np.float32)).copy(),
        "cf_pair_jsd_valid_updates": int(getattr(state, "cf_pair_jsd_valid_updates", 0)),
        "cf_pair_jsd_last_update_step": int(getattr(state, "cf_pair_jsd_last_update_step", -1)),
        "actor_intervention_consecutive_updates": int(
            getattr(state, "actor_intervention_consecutive_updates", 0)
        ),
        "macro_pair_jsd_ema": getattr(state, "macro_pair_jsd_ema", np.zeros(6, dtype=np.float32)).copy(),
        "macro_pair_jsd_valid_updates": int(getattr(state, "macro_pair_jsd_valid_updates", 0)),
        "macro_pair_jsd_last_update_step": int(getattr(state, "macro_pair_jsd_last_update_step", -1)),
    }


def _component_router_runtime(state: Any) -> dict[str, Any]:
    return {
        "router_optimizer_step_count": int(state.router_optimizer_step_count),
        "macro_return_running_mean": float(getattr(state, "macro_return_running_mean", 0.0)),
        "macro_return_running_count": int(getattr(state, "macro_return_running_count", 0)),
        "selector_hidden": getattr(state, "selector_hidden", None),
        "v6i1_episode_rehearsal": getattr(state, "v6i1_episode_rehearsal", None),
    }


def latent_checkpoint_payload(state: Any) -> dict[str, Any]:
    """Schema v2 aggregate for trainer checkpoint hooks."""
    trainer = getattr(state, "trainer", None)
    cfg = getattr(trainer, "cfg", None) if trainer is not None else None
    latent_k = int(getattr(trainer, "latent_k", 4) or 4)
    n_envs = int(getattr(getattr(trainer, "env", None), "num_envs", 1) or 1)
    gate_fields: dict[str, Any] = {}
    if cfg is not None:
        from rl.custom_ppo.gate_protocol import (
            gate_config_fingerprint,
            gate_lineage_audit_fields,
            resolve_gate_protocol_version,
            resolved_gate_config_dict,
        )

        gate_fields = {
            "gate_protocol_version": resolve_gate_protocol_version(cfg),
            "gate_config_fingerprint": gate_config_fingerprint(cfg),
            "resolved_gate_config": resolved_gate_config_dict(cfg),
            **gate_lineage_audit_fields(cfg),
        }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "latent_k": latent_k,
        "n_envs": n_envs,
        **gate_fields,
        "intervention": _component_intervention(state),
        "router_runtime": _component_router_runtime(state),
    }
    validate_latent_checkpoint(payload, latent_k=latent_k, n_envs=n_envs)
    return payload


def restore_latent_checkpoint_payload(state: Any, payload: dict[str, Any]) -> None:
    if not payload:
        return
    version = int(payload.get("schema_version", 0))
    if version == SCHEMA_VERSION:
        intervention = dict(payload.get("intervention", {}) or {})
        runtime = dict(payload.get("router_runtime", {}) or {})
        merged = {**intervention, **runtime, **payload}
        _restore_flat_v6i1_fields(state, merged)
        return
    _restore_flat_v6i1_fields(state, payload)


def _restore_flat_v6i1_fields(state: Any, payload: dict[str, Any]) -> None:
    """Shared flat restore path for schema v1 and v2."""
    if not payload:
        return
    trainer = getattr(state, "trainer", None)
    cfg = getattr(trainer, "cfg", None) if trainer is not None else None
    if cfg is not None:
        for key in (
            "gate_config_mismatch_override_used",
            "gate_config_fingerprint_checkpoint",
            "gate_config_fingerprint_active",
            "confirmatory_gate_lineage_valid",
        ):
            if key in payload:
                setattr(cfg, key, payload[key])
    for key, attr in (
        ("cf_J", "cf_J"),
        ("cf_episode_counts", "cf_episode_counts"),
        ("cf_has_experience", "cf_has_experience"),
        ("pair_jsd_ema", "pair_jsd_ema"),
        ("cf_pair_jsd_ema", "cf_pair_jsd_ema"),
        ("macro_pair_jsd_ema", "macro_pair_jsd_ema"),
        ("selector_hidden", "selector_hidden"),
        ("v6i1_episode_rehearsal", "v6i1_episode_rehearsal"),
    ):
        if key in payload:
            setattr(state, attr, payload[key])
    for key, attr, cast in (
        ("cf_return_mean", "cf_return_mean", float),
        ("cf_return_var", "cf_return_var", float),
        ("macro_return_running_mean", "macro_return_running_mean", float),
        ("jsd_gate_consecutive_updates", "jsd_gate_consecutive_updates", int),
        ("pairwise_ema_valid_updates", "pairwise_ema_valid_updates", int),
        ("pairwise_ema_last_update_step", "pairwise_ema_last_update_step", int),
        ("cf_pair_jsd_valid_updates", "cf_pair_jsd_valid_updates", int),
        ("cf_pair_jsd_last_update_step", "cf_pair_jsd_last_update_step", int),
        ("actor_intervention_consecutive_updates", "actor_intervention_consecutive_updates", int),
        ("macro_pair_jsd_valid_updates", "macro_pair_jsd_valid_updates", int),
        ("macro_pair_jsd_last_update_step", "macro_pair_jsd_last_update_step", int),
        ("router_optimizer_step_count", "router_optimizer_step_count", int),
        ("macro_return_running_count", "macro_return_running_count", int),
    ):
        if key in payload:
            setattr(state, attr, cast(payload[key]))
    hidden = payload.get("selector_hidden")
    if hidden is not None and getattr(state, "selector_hidden", None) is not None:
        state.selector_hidden = hidden.to(
            device=state.selector_hidden.device, dtype=state.selector_hidden.dtype
        )
    rehearsal = payload.get("v6i1_episode_rehearsal")
    if rehearsal is not None and getattr(state, "v6i1_episode_rehearsal", None) is not None:
        state.v6i1_episode_rehearsal = rehearsal.to(
            device=state.v6i1_episode_rehearsal.device,
            dtype=state.v6i1_episode_rehearsal.dtype,
        )
