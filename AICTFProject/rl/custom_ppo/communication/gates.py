"""V6I3 communication gate-family evaluators."""

from __future__ import annotations

from typing import Any

from rl.config.ppo_config import PPOConfig
from rl.custom_ppo.gate_protocol import GATE_STATUS_FAIL, GATE_STATUS_NOT_RUN, GATE_STATUS_PASS, GateEvalResult


def evaluate_communication_usage(cfg: PPOConfig, telemetry: dict[str, float]) -> GateEvalResult:
    if not bool(getattr(cfg, "communication_enabled", False)):
        return GateEvalResult(status=GATE_STATUS_NOT_RUN, reason="communication_disabled")
    min_boundaries = int(getattr(cfg, "comm_min_valid_boundaries", 0) or 0)
    min_deliveries = int(getattr(cfg, "comm_min_deliveries", 0) or 0)
    min_symbols = int(getattr(cfg, "comm_min_symbols_used", 3) or 3)
    entropy_floor = float(getattr(cfg, "comm_entropy_floor", 0.0) or 0.0)
    dominance_ceiling = float(getattr(cfg, "comm_symbol_dominance_ceiling", 1.0) or 1.0)
    valid_boundaries = float(telemetry.get("comm_valid_boundaries", 0.0) or 0.0)
    deliveries = float(telemetry.get("comm_delivery_count", 0.0) or 0.0)
    symbols_used = float(telemetry.get("comm_symbols_used", 0.0) or 0.0)
    entropy_norm = float(telemetry.get("comm_symbol_entropy_normalized", 0.0) or 0.0)
    dominance = float(telemetry.get("comm_symbol_dominance", 1.0) or 1.0)
    checks = {
        "valid_boundaries": valid_boundaries >= float(min_boundaries),
        "deliveries": deliveries >= float(min_deliveries),
        "symbols_used": symbols_used >= float(min_symbols),
        "entropy_floor": entropy_norm >= entropy_floor if entropy_floor > 0.0 else True,
        "dominance_ceiling": dominance <= dominance_ceiling if dominance_ceiling < 1.0 else True,
    }
    passed = all(checks.values())
    return GateEvalResult(
        status=GATE_STATUS_PASS if passed else GATE_STATUS_FAIL,
        reason="communication_usage_checks",
        details={
            **telemetry,
            "checks": checks,
            "comm_min_valid_boundaries": min_boundaries,
            "comm_min_deliveries": min_deliveries,
            "comm_min_symbols_used": min_symbols,
            "comm_entropy_floor": entropy_floor,
            "comm_symbol_dominance_ceiling": dominance_ceiling,
        },
    )


def evaluate_listener_causal_response(cfg: PPOConfig, telemetry: dict[str, float]) -> GateEvalResult:
    if not bool(getattr(cfg, "communication_enabled", False)):
        return GateEvalResult(status=GATE_STATUS_NOT_RUN, reason="communication_disabled")
    margin = float(getattr(cfg, "comm_listener_jsd_margin", 0.0) or 0.0)
    min_pairs = int(getattr(cfg, "comm_listener_min_passing_pairs", 0) or 0)
    min_states = int(getattr(cfg, "comm_listener_min_states", 0) or 0)
    required_consecutive = int(getattr(cfg, "comm_listener_consecutive_updates", 0) or 0)
    jsd = float(telemetry.get("receiver_action_jsd_by_message_pair_mean", 0.0) or 0.0)
    pairs = float(telemetry.get("receiver_listener_pairs", 0.0) or 0.0)
    pairs_above = float(
        telemetry.get(
            "receiver_listener_pairs_above_margin",
            pairs if (margin <= 0.0 or jsd >= margin) else 0.0,
        )
        or 0.0
    )
    disagree = float(telemetry.get("receiver_argmax_disagreement_frac", 0.0) or 0.0)
    states = float(telemetry.get("receiver_listener_states", telemetry.get("comm_valid_boundaries", 0.0)) or 0.0)
    current_streak = int(
        telemetry.get(
            "listener_causal_response_consecutive_updates",
            1 if (jsd >= margin and pairs_above >= float(min_pairs) and states >= float(min_states)) else 0,
        )
        or 0
    )
    checks = {
        "jsd_margin": jsd >= margin if margin > 0.0 else True,
        "min_pairs": pairs >= float(min_pairs) if min_pairs > 0 else True,
        "min_passing_pairs": pairs_above >= float(min_pairs) if min_pairs > 0 else True,
        "min_states": states >= float(min_states) if min_states > 0 else True,
        "disagreement": disagree > 0.0 if margin > 0.0 else True,
        "consecutive_updates": current_streak >= required_consecutive if required_consecutive > 0 else True,
    }
    passed = all(checks.values())
    return GateEvalResult(
        status=GATE_STATUS_PASS if passed else GATE_STATUS_FAIL,
        reason="listener_causal_response_checks",
        details={
            **telemetry,
            "checks": checks,
            "comm_listener_jsd_margin": margin,
            "comm_listener_min_passing_pairs": min_pairs,
            "comm_listener_min_states": min_states,
            "comm_listener_consecutive_updates": required_consecutive,
            "receiver_listener_pairs_above_margin": pairs_above,
            "receiver_listener_states": states,
            "listener_causal_response_consecutive_updates": current_streak,
        },
    )


__all__ = ["evaluate_communication_usage", "evaluate_listener_causal_response"]
