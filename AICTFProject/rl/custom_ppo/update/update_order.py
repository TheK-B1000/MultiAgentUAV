"""Helpers for actor PPO/CF update-order diagnostics."""

from __future__ import annotations

import math
from typing import Literal


ActorCFUpdateMode = Literal["combined", "ppo_then_cf", "cf_then_ppo"]
ACTOR_CF_UPDATE_MODES: tuple[str, ...] = ("combined", "ppo_then_cf", "cf_then_ppo")


def validate_actor_cf_update_mode(mode: str) -> ActorCFUpdateMode:
    if mode not in ACTOR_CF_UPDATE_MODES:
        raise ValueError(f"actor_cf_update_mode must be one of {ACTOR_CF_UPDATE_MODES}; got {mode!r}.")
    return mode  # type: ignore[return-value]


def update_order_jsd_metrics(
    *,
    mode: str,
    before: float,
    after_ppo: float,
    after_cf: float,
    epsilon: float = 1e-12,
) -> dict[str, float | str]:
    """Compute signed JSD deltas and CF-retention diagnostics for one substep pair."""
    validate_actor_cf_update_mode(mode)
    if mode == "combined":
        return {
            "ppo_jsd_delta": math.nan,
            "cf_jsd_delta": math.nan,
            "cf_gain": math.nan,
            "retained_cf_gain": math.nan,
            "cf_retention_ratio": math.nan,
            "cf_retention_reason": "not_sequential",
        }
    if mode == "ppo_then_cf":
        ppo_delta = float(after_ppo) - float(before)
        cf_delta = float(after_cf) - float(after_ppo)
        retained = float(after_cf) - float(after_ppo)
    else:
        cf_delta = float(after_cf) - float(before)
        ppo_delta = float(after_ppo) - float(after_cf)
        retained = float(after_ppo) - float(before)
    gain = max(0.0, cf_delta)
    if gain <= float(epsilon):
        ratio = math.nan
        reason = "no_measurable_cf_gain"
    else:
        ratio = retained / max(gain, float(epsilon))
        reason = ""
    return {
        "ppo_jsd_delta": ppo_delta,
        "cf_jsd_delta": cf_delta,
        "cf_gain": gain,
        "retained_cf_gain": retained,
        "cf_retention_ratio": ratio,
        "cf_retention_reason": reason,
    }
