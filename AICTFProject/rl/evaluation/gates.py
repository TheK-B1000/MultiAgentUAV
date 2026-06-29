"""Gate and verdict logic for V6I9 map-awareness evaluation."""
from __future__ import annotations

import argparse
from typing import Any, Mapping, Sequence

from rl.custom_ppo.probe_result import (
    CounterfactualProbeResult,
    GradientProbeResult,
    WeightProbeResult,
)
from rl.evaluation.aggregation import field_mean as _field_mean
from rl.evaluation.aggregation import policy_rows as _policy_rows

REQUIRED_GATE_KEYS: tuple[str, ...] = (
    "obstacle_weights_moved",
    "obstacle_gradient_connected",
    "obstacle_counterfactual_effect",
    "hard_pool_competence_retained",
)

DIAGNOSTIC_GATE_KEYS: tuple[str, ...] = (
    "wall_collisions_improved",
    "blocked_movement_improved",
    "stuck_behavior_improved",
    "map_dependent_routes",
    "pool_saturation",
)


def _improvement_gate(
    baseline: float | None,
    candidate: float | None,
    minimum_reduction: float,
) -> tuple[str, dict[str, Any]]:
    details: dict[str, Any] = {
        "baseline_mean": baseline,
        "candidate_mean": candidate,
        "minimum_reduction_fraction": minimum_reduction,
    }

    if baseline is None or candidate is None:
        details["reason"] = "Required telemetry is unavailable."
        return "INCONCLUSIVE", details

    if baseline <= 0:
        details["reason"] = (
            "Baseline is zero, so relative reduction is undefined."
        )
        return (
            "PASS" if candidate <= 0 else "FAIL",
            details,
        )

    reduction = (baseline - candidate) / baseline
    details["reduction_fraction"] = reduction

    return (
        "PASS"
        if reduction >= minimum_reduction
        else "FAIL",
        details,
    )


def _gate_probe_weight(
    result: WeightProbeResult,
    threshold: float,
) -> dict[str, Any]:
    """Build a gate entry from a WeightProbeResult."""
    if not result.is_success:
        return {
            "status": "ERROR",
            "error": result.error,
        }
    value = result.obstacle_weight_l2
    if value is None:
        return {"status": "INCONCLUSIVE", "error": "weight L2 not measured"}
    return {
        "status": "PASS" if value > threshold else "FAIL",
        "value": value,
        "threshold": threshold,
    }


def _gate_probe_gradient(
    result: GradientProbeResult,
    threshold: float,
) -> dict[str, Any]:
    """Build a gate entry from a GradientProbeResult."""
    if not result.is_success:
        return {
            "status": "ERROR",
            "error": result.error,
        }
    value = result.obstacle_gradient_l2
    if value is None:
        return {"status": "INCONCLUSIVE", "error": "gradient L2 not measured"}
    return {
        "status": "PASS" if value > threshold else "FAIL",
        "value": value,
        "threshold": threshold,
    }


def _gate_probe_counterfactual(
    result: CounterfactualProbeResult,
    action_threshold: float,
    kl_threshold: float,
) -> dict[str, Any]:
    """Build a gate entry from a CounterfactualProbeResult."""
    if not result.is_success:
        return {
            "status": "ERROR",
            "error": result.error,
            "states_evaluated": result.states_evaluated,
        }
    action_change = result.argmax_action_change_rate
    mean_kl = result.mean_action_kl
    mean_l2 = result.mean_logit_l2
    if action_change is None or mean_kl is None:
        return {
            "status": "INCONCLUSIVE",
            "error": "counterfactual metrics not measured",
            "states_evaluated": result.states_evaluated,
        }
    if action_change >= action_threshold or mean_kl >= kl_threshold:
        status = "PASS"
    elif mean_l2 is not None and mean_l2 > 1e-3:
        status = "WARN"
    else:
        status = "FAIL"
    return {
        "status": status,
        "argmax_change_rate": action_change,
        "argmax_change_threshold": action_threshold,
        "mean_action_kl": mean_kl,
        "kl_threshold": kl_threshold,
        "mean_logit_l2": mean_l2,
        "states_evaluated": result.states_evaluated,
    }


def _required_gate_ok(status: str, *, allow_warn: bool = False) -> bool:
    if status == "PASS":
        return True
    return bool(allow_warn and status == "WARN")


def _core_required_ready(gates: Mapping[str, Mapping[str, Any]]) -> bool:
    return all(
        _required_gate_ok(
            gates[key]["status"],
            allow_warn=(key == "obstacle_counterfactual_effect"),
        )
        for key in REQUIRED_GATE_KEYS
    )


def _resolve_verdict(
    gates: Mapping[str, Mapping[str, Any]],
    *,
    pool_saturated: bool,
    allow_saturated_pool: bool,
) -> str:
    statuses = [gate["status"] for gate in gates.values()]

    if any(s == "ERROR" for s in statuses):
        return "NOT READY FOR STAGE B — PROBE ERROR (see gate details)"

    if not _core_required_ready(gates):
        return "NOT READY FOR STAGE B"

    if pool_saturated and allow_saturated_pool:
        return "READY FOR STAGE B - SATURATED POOL"

    return "READY FOR STAGE B"


def build_summary(
    args: argparse.Namespace,
    probe: Mapping[str, Any],
    episodes: Sequence[Mapping[str, Any]],
    conditions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    candidate_weights: WeightProbeResult = probe["candidate_weights"]
    candidate_gradient: GradientProbeResult = probe["candidate_gradient"]
    candidate_counterfactual: CounterfactualProbeResult = probe[
        "candidate_counterfactual"
    ]

    gates: dict[str, Any] = {
        "obstacle_weights_moved": _gate_probe_weight(
            candidate_weights, args.obs_weight_threshold
        ),
        "obstacle_gradient_connected": _gate_probe_gradient(
            candidate_gradient, args.gradient_threshold
        ),
        "obstacle_counterfactual_effect": _gate_probe_counterfactual(
            candidate_counterfactual,
            args.counterfactual_action_threshold,
            args.counterfactual_kl_threshold,
        ),
    }

    baseline_obstacle_rows = _policy_rows(
        episodes,
        "baseline",
        obstacle_maps_only=True,
    )
    candidate_obstacle_rows = _policy_rows(
        episodes,
        "candidate",
        obstacle_maps_only=True,
    )

    obstacle_rows = baseline_obstacle_rows + candidate_obstacle_rows
    exact_wall_telemetry = bool(obstacle_rows) and all(
        row.get("collision_metric_source") == "environment_exact"
        and row.get("wall_collisions") is not None
        for row in obstacle_rows
    )

    baseline_collisions = (
        _field_mean(baseline_obstacle_rows, "wall_collisions")
        if exact_wall_telemetry
        else None
    )
    candidate_collisions = (
        _field_mean(candidate_obstacle_rows, "wall_collisions")
        if exact_wall_telemetry
        else None
    )

    if exact_wall_telemetry:
        status, details = _improvement_gate(
            baseline_collisions,
            candidate_collisions,
            args.navigation_improvement_threshold,
        )
    else:
        status = "INCONCLUSIVE"
        details = {
            "reason": (
                "No exact obstacle collision counter was found "
                "in terminal episode info."
            )
        }

    gates["wall_collisions_improved"] = {
        "status": status,
        "collision_metric_source": "environment_exact" if exact_wall_telemetry else "unavailable",
        "diagnostic_only": True,
        **details,
    }

    exact_blocked_telemetry = bool(obstacle_rows) and all(
        row.get("blocked_movement_events") is not None
        for row in obstacle_rows
    )
    if exact_blocked_telemetry:
        status, details = _improvement_gate(
            _field_mean(baseline_obstacle_rows, "blocked_movement_events"),
            _field_mean(candidate_obstacle_rows, "blocked_movement_events"),
            args.navigation_improvement_threshold,
        )
    else:
        status = "INCONCLUSIVE"
        details = {"reason": "Environment blocked-movement telemetry is unavailable."}
    gates["blocked_movement_improved"] = {
        "status": status,
        "stuck_metric_source": "environment_exact" if exact_blocked_telemetry else "unavailable",
        "diagnostic_only": True,
        **details,
    }

    exact_stuck_telemetry = bool(obstacle_rows) and all(
        row.get("stuck_metric_source") == "environment_exact"
        and row.get("stuck_steps") is not None
        for row in obstacle_rows
    )
    status, details = _improvement_gate(
        _field_mean(
            baseline_obstacle_rows,
            "stuck_steps",
        ),
        _field_mean(
            candidate_obstacle_rows,
            "stuck_steps",
        ),
        args.navigation_improvement_threshold,
    )
    gates["stuck_behavior_improved"] = {
        "status": status if exact_stuck_telemetry else "INCONCLUSIVE",
        "stuck_metric_source": "environment_exact" if exact_stuck_telemetry else "evaluator_proxy",
        "diagnostic_only": True,
        **details,
    }

    candidate_conditions = [
        row
        for row in conditions
        if row.get("policy") == "candidate"
    ]

    route_by_map: dict[str, Any] = {}
    for map_name in args.maps:
        selected = [
            row
            for row in candidate_conditions
            if row.get("map") == map_name
        ]
        route_exact = bool(selected) and all(
            row.get("route_metric_source") == "environment_exact" for row in selected
        )
        upper = _field_mean(selected, "upper_lane_use") if route_exact else None
        lower = _field_mean(selected, "lower_lane_use") if route_exact else None
        total = (upper or 0.0) + (lower or 0.0)

        route_by_map[map_name] = {
            "upper_mean": upper,
            "lower_mean": lower,
            "upper_fraction": (
                upper / total
                if total > 0
                else None
            ),
            "route_metric_source": "environment_exact" if route_exact else "unavailable",
        }

    route_fractions = [
        values["upper_fraction"]
        for values in route_by_map.values()
        if values["upper_fraction"] is not None
    ]
    route_difference = (
        max(route_fractions) - min(route_fractions)
        if len(route_fractions) >= 2
        else None
    )

    if route_difference is None:
        route_status = "INCONCLUSIVE"
    elif route_difference >= args.route_difference_threshold:
        route_status = "PASS"
    else:
        route_status = "FAIL"

    gates["map_dependent_routes"] = {
        "status": route_status,
        "route_metric_source": "environment_exact" if route_fractions else "unavailable",
        "max_upper_fraction_difference": route_difference,
        "threshold": args.route_difference_threshold,
        "per_map": route_by_map,
        "diagnostic_only": True,
    }

    baseline_win_rate = _field_mean(
        _policy_rows(episodes, "baseline"),
        "win",
    )
    candidate_win_rate = _field_mean(
        _policy_rows(episodes, "candidate"),
        "win",
    )

    competence_pass = (
        baseline_win_rate is not None
        and candidate_win_rate is not None
        and candidate_win_rate >= args.minimum_win_rate
        and candidate_win_rate
        >= baseline_win_rate
        - args.competence_retention_tolerance
    )

    gates["hard_pool_competence_retained"] = {
        "status": (
            "PASS"
            if competence_pass
            else "FAIL"
        ),
        "baseline_win_rate": baseline_win_rate,
        "candidate_win_rate": candidate_win_rate,
        "minimum_candidate_win_rate": args.minimum_win_rate,
        "maximum_allowed_drop": (
            args.competence_retention_tolerance
        ),
    }

    condition_win_rates = [
        float(row["win"])
        for row in candidate_conditions
        if row.get("win") is not None
    ]
    pool_saturated = (
        bool(condition_win_rates)
        and all(
            win_rate >= args.saturation_win_rate
            for win_rate in condition_win_rates
        )
    )

    gates["pool_saturation"] = {
        "status": "SATURATED" if pool_saturated else "INFORMATIVE",
        "condition_win_rates": condition_win_rates,
        "saturation_threshold": args.saturation_win_rate,
        "diagnostic_only": True,
    }

    # Legacy alias for downstream readers/tests.
    gates["universal_saturation_avoided"] = {
        "status": "SATURATED" if pool_saturated else "INFORMATIVE",
        "condition_win_rates": condition_win_rates,
        "saturation_threshold": args.saturation_win_rate,
        "diagnostic_only": True,
        "legacy_alias_of": "pool_saturation",
    }

    allow_saturated_pool = bool(getattr(args, "allow_saturated_pool", False))
    verdict = _resolve_verdict(
        gates,
        pool_saturated=pool_saturated,
        allow_saturated_pool=allow_saturated_pool,
    )

    collision_delta = None
    if baseline_collisions is not None and candidate_collisions is not None:
        collision_delta = float(candidate_collisions) - float(baseline_collisions)

    diagnostics = {
        "pool_saturation": gates["pool_saturation"]["status"],
        "map_route_signal": gates["map_dependent_routes"]["status"],
        "wall_collision_delta": collision_delta,
        "wall_collisions_baseline_mean": baseline_collisions,
        "wall_collisions_candidate_mean": candidate_collisions,
    }

    required_gates = {key: gates[key] for key in REQUIRED_GATE_KEYS}
    diagnostic_gates = {key: gates[key] for key in DIAGNOSTIC_GATE_KEYS}

    return {
        "verdict": verdict,
        "stage2_eligible": verdict.startswith("READY FOR STAGE B"),
        "gates": gates,
        "required_gates": required_gates,
        "diagnostic_gates": diagnostic_gates,
        "diagnostics": diagnostics,
        "episodes_per_condition": args.episodes,
        "warning": (
            "Use at least 20 episodes per map/opponent "
            "cell for a promotion decision."
            if args.episodes < 20
            else None
        ),
    }



__all__ = [
    "DIAGNOSTIC_GATE_KEYS",
    "REQUIRED_GATE_KEYS",
    "build_summary",
]
