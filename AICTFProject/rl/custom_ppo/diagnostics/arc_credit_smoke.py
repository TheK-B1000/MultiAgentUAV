"""One-update smoke gates for the feedforward running-mean arc-credit treatment.

The mechanism test (``v6i9_arc_credit_running_mean_feedforward_hardpool``) must
prove, before spending GPU time, that a single PPO update from the protected
repertoire anchor:

* routes credit through arc-credit (not the biased critic strategy-PPO term),
* uses the running-mean baseline,
* sees at least one valid router decision / finalized arc,
* produces a finite raw arc advantage,
* leaves the frozen actor + z-specific parameters untouched, and
* actually moves the router (q_phi) parameters.

The gate evaluation (:func:`evaluate_arc_credit_treatment_gates`) is a pure
function over already-collected numbers so it can be unit-tested without a
trainer, env, or GPU. The model-fingerprint helpers are thin wrappers around
``named_parameters`` used by the CLI.
"""
from __future__ import annotations

import hashlib
import math
from typing import Any, Callable, Mapping

import torch

# Parameter-name substrings that identify the router / q_phi stack. Everything
# else that is frozen (requires_grad=False under router_freeze_actor=True) is
# the "actor + z-specific" group that must NOT move during a router update.
ROUTER_PARAM_MARKERS: tuple[str, ...] = (
    "strategy_encoder",
    "selector_gru",
    "episode_strategy_value_head",
)


def is_router_param(name: str) -> bool:
    return any(marker in name for marker in ROUTER_PARAM_MARKERS)


def parameter_group_fingerprint(
    model: Any,
    *,
    predicate: Callable[[str, torch.nn.Parameter], bool],
) -> str:
    """SHA-256 over the (name, bytes) of every parameter matching ``predicate``."""
    h = hashlib.sha256()
    for name, param in sorted(model.named_parameters(), key=lambda kv: kv[0]):
        if not predicate(name, param):
            continue
        h.update(name.encode("utf-8"))
        h.update(param.detach().cpu().contiguous().to(torch.float32).numpy().tobytes())
    return h.hexdigest()


def frozen_actor_z_fingerprint(model: Any) -> str:
    """Fingerprint of the frozen actor + z-specific parameters.

    Uses ``requires_grad == False`` (the freeze flag set by
    ``router_freeze_actor``) intersected with "not a router parameter" so a
    frozen router param — should one ever exist — is not miscounted here.
    """
    return parameter_group_fingerprint(
        model,
        predicate=lambda name, p: (not p.requires_grad) and (not is_router_param(name)),
    )


def router_fingerprint(model: Any) -> str:
    return parameter_group_fingerprint(model, predicate=lambda name, p: is_router_param(name))


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def evaluate_arc_credit_treatment_gates(
    *,
    cfg: Any,
    arc_stats: Mapping[str, float],
    router_decision_count: int,
    frozen_hash_before: str,
    frozen_hash_after: str,
    router_hash_before: str,
    router_hash_after: str,
) -> dict[str, Any]:
    """Evaluate the one-update treatment smoke gates.

    Returns a dict with per-gate booleans, an ``all_passed`` flag, and the
    raw running-mean telemetry the operator should inspect.
    """
    arc_enabled = bool(getattr(cfg, "latent_arc_credit_enabled", False))
    strategy_ppo_coef = float(getattr(cfg, "latent_strategy_ppo_coef", 1.0) or 0.0)
    baseline_mode = str(getattr(cfg, "latent_arc_credit_baseline", "") or "")

    raw_mean = arc_stats.get("latent_arc_raw_advantage_mean")
    raw_std = arc_stats.get("latent_arc_raw_advantage_std")
    q_phi_grad = float(arc_stats.get("q_phi_grad_norm", 0.0) or 0.0)
    arc_grad = float(arc_stats.get("latent_arc_grad_norm", 0.0) or 0.0)
    finalized = float(arc_stats.get("latent_arc_finalized_count", 0.0) or 0.0)
    arc_count = float(arc_stats.get("latent_arc_count", 0.0) or 0.0)

    gates = {
        # Arc-credit is the sole router credit path (magnet removed).
        "arc_credit_source_active": arc_enabled and strategy_ppo_coef == 0.0,
        "baseline_mode_running_mean": baseline_mode == "running_mean",
        # At least one router opportunity AND one finalized/trained arc.
        "valid_router_decisions_positive": int(router_decision_count) > 0,
        "arc_records_present": arc_count > 0.0 or finalized > 0.0,
        "raw_arc_advantage_finite": _finite(raw_mean) and _finite(raw_std),
        # Frozen actor + z-specific weights must be byte-identical after update.
        "frozen_actor_z_unchanged": frozen_hash_before == frozen_hash_after,
        # Router (q_phi) must actually move: nonzero grad OR changed weights.
        "router_gradients_positive": (q_phi_grad > 0.0 or arc_grad > 0.0)
        and (router_hash_before != router_hash_after),
    }
    gates["all_passed"] = all(bool(v) for v in gates.values())

    telemetry = {
        "arc_return_mean": float(arc_stats.get("latent_arc_mean_return", 0.0) or 0.0),
        "arc_baseline_mean": float(arc_stats.get("latent_arc_baseline_mean", 0.0) or 0.0),
        "raw_arc_advantage_mean": float(raw_mean) if _finite(raw_mean) else float("nan"),
        "raw_arc_advantage_std": float(raw_std) if _finite(raw_std) else float("nan"),
        "positive_fraction": float(arc_stats.get("latent_arc_positive_fraction", 0.0) or 0.0),
        "running_mean_update_count": float(
            arc_stats.get("latent_arc_running_mean_count", 0.0) or 0.0
        ),
        "running_mean_value": float(arc_stats.get("latent_arc_running_mean_value", 0.0) or 0.0),
        "q_phi_grad_norm": q_phi_grad,
        "latent_arc_grad_norm": arc_grad,
        "router_decision_count": int(router_decision_count),
    }
    return {"gates": gates, "telemetry": telemetry}


__all__ = [
    "ROUTER_PARAM_MARKERS",
    "evaluate_arc_credit_treatment_gates",
    "frozen_actor_z_fingerprint",
    "is_router_param",
    "parameter_group_fingerprint",
    "router_fingerprint",
]
