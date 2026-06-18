"""V6 staged curriculum gate protocol — behavior only.

Resolved thresholds and timing live exclusively in ``PPOConfig``; this module
consumes those values and must not maintain a parallel default set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from rl.config.ppo_config import PPOConfig

V6I1_GATE_PROTOCOL = "v6i1_single_macro_intervention"
V6I2_GATE_PROTOCOL = "v6i2_dual_evidence"

GATE_STATUS_PASS = "PASS"
GATE_STATUS_FAIL = "FAIL"
GATE_STATUS_NOT_RUN = "NOT_RUN"
GATE_STATUS_ERROR = "ERROR"

GATE_FAMILY_NAMES_V6I1: tuple[str, ...] = (
    "coverage",
    "competence",
    "counterfactual_intervention",
    "training_integrity",
    "matched_seed_behavior",
    "selector_learnability_probe",
)

GATE_FAMILY_NAMES_V6I2: tuple[str, ...] = (
    "coverage",
    "competence",
    "actor_intervention",
    "behavioral_realization",
    "training_integrity",
    "selector_learnability_probe",
)

KNOWN_GATE_PROTOCOLS = frozenset({V6I1_GATE_PROTOCOL, V6I2_GATE_PROTOCOL})


@dataclass
class GateEvalResult:
    """Outcome of a gate-family evaluation (protocol layer)."""

    status: str
    reason: str = ""
    details: dict[str, Any] = field(default_factory=dict)


def resolve_gate_protocol_version(cfg: PPOConfig) -> str:
    version = str(getattr(cfg, "gate_protocol_version", V6I1_GATE_PROTOCOL) or V6I1_GATE_PROTOCOL)
    if version not in KNOWN_GATE_PROTOCOLS:
        raise ValueError(
            f"gate_protocol_version must be one of {sorted(KNOWN_GATE_PROTOCOLS)!r}, got {version!r}"
        )
    return version


def is_v6i2_gate_protocol(cfg: PPOConfig) -> bool:
    return resolve_gate_protocol_version(cfg) == V6I2_GATE_PROTOCOL


def get_gate_family_names(cfg: PPOConfig) -> tuple[str, ...]:
    if is_v6i2_gate_protocol(cfg):
        return GATE_FAMILY_NAMES_V6I2
    return GATE_FAMILY_NAMES_V6I1


# Backward-compatible alias.
gate_family_names = get_gate_family_names


def validate_protocol_config(cfg: PPOConfig) -> None:
    """Fail fast when protocol id disagrees with experiment row or enforce prerequisites."""
    protocol = resolve_gate_protocol_version(cfg)
    exp_id = str(getattr(cfg, "experiment_id", "v6i1"))
    if exp_id == "v6i2" and protocol != V6I2_GATE_PROTOCOL:
        raise ValueError(f"v6i2 runs must set gate_protocol_version={V6I2_GATE_PROTOCOL!r}.")
    if exp_id == "v6i1" and protocol == V6I2_GATE_PROTOCOL:
        raise ValueError("v6i1 experiment_id cannot use v6i2_dual_evidence protocol.")

    mode = str(getattr(cfg, "phase_boundary_gate_mode", "enforce")).lower()
    if mode != "enforce":
        return
    if not bool(getattr(cfg, "curriculum_gate_run_boundary_eval", False)):
        raise ValueError("V6 staged enforce mode requires matched-seed boundary evaluation.")
    if not bool(getattr(cfg, "curriculum_gate_run_probe", False)):
        raise ValueError("V6 staged enforce mode requires the selector-learnability probe.")


def is_staged_v6_team_intent_curriculum(cfg: PPOConfig) -> bool:
    """True for v6i1 and v6i2 staged team-intent rows (not repertoire ablations)."""
    return (
        bool(getattr(cfg, "use_v6i1_curriculum", False))
        and str(getattr(cfg, "training_mode", "default")) == "staged_team_intent_curriculum"
        and str(getattr(cfg, "experiment_family", "v6")) == "v6"
        and str(getattr(cfg, "experiment_id", "v6i1")) in ("v6i1", "v6i2")
    )


def is_v6_protocol(cfg: PPOConfig) -> bool:
    """Alias: staged v6 team-intent curriculum rows with gate protocols."""
    return is_staged_v6_team_intent_curriculum(cfg)


def _actor_thresholds(cfg: PPOConfig) -> tuple[float, float, int, int]:
    margin = float(cfg.actor_jsd_margin)
    floor = float(cfg.actor_jsd_floor_fraction) * margin
    min_pairs = int(cfg.actor_jsd_min_passing_pairs)
    consecutive = int(cfg.actor_jsd_consecutive_updates)
    return margin, floor, min_pairs, consecutive


def _macro_thresholds(cfg: PPOConfig) -> tuple[float, float, int]:
    margin = float(cfg.macro_jsd_margin)
    floor = float(cfg.macro_jsd_floor_fraction) * margin
    min_pairs = int(cfg.macro_jsd_min_passing_pairs)
    return margin, floor, min_pairs


def evaluate_actor_intervention(cfg: PPOConfig, latent_state: Any) -> GateEvalResult:
    """Gate A (v6i2): CF-batch actor pair JSD EMA only — never reads macro or legacy EMA."""
    margin, floor, min_pairs, required_consecutive = _actor_thresholds(cfg)
    valid_updates = int(getattr(latent_state, "cf_pair_jsd_valid_updates", 0) or 0)
    if valid_updates <= 0:
        return GateEvalResult(
            status=GATE_STATUS_NOT_RUN,
            reason="no_cf_pair_jsd_ema_updates",
            details={
                "cf_pair_jsd_valid_updates": valid_updates,
                "actor_jsd_margin": margin,
            },
        )

    cf_ema = getattr(latent_state, "cf_pair_jsd_ema", None)
    if cf_ema is None:
        return GateEvalResult(
            status=GATE_STATUS_NOT_RUN,
            reason="missing_cf_pair_jsd_ema_buffer",
        )

    pair_jsd = [float(v) for v in cf_ema.tolist()]
    if len(pair_jsd) != 6 or not all(np.isfinite(pair_jsd)):
        return GateEvalResult(
            status=GATE_STATUS_ERROR,
            reason="corrupt_cf_pair_jsd_ema",
            details={"cf_pair_jsd_ema": pair_jsd},
        )

    num_above = sum(1 for v in pair_jsd if v >= margin)
    min_ema = float(min(pair_jsd))
    update_ok = num_above >= min_pairs and min_ema >= floor
    streak = int(getattr(latent_state, "actor_intervention_consecutive_updates", 0) or 0)
    passed = update_ok and streak >= required_consecutive
    status = GATE_STATUS_PASS if passed else GATE_STATUS_FAIL

    return GateEvalResult(
        status=status,
        details={
            "cf_pair_jsd_ema": pair_jsd,
            "actor_intervention_gate_status": status,
            "cf_pair_jsd_valid_updates": valid_updates,
            "cf_pair_jsd_last_update_step": int(
                getattr(latent_state, "cf_pair_jsd_last_update_step", -1) or -1
            ),
            "actor_jsd_margin": margin,
            "actor_jsd_floor": floor,
            "actor_jsd_min_passing_pairs": min_pairs,
            "num_pairs_above_margin": int(num_above),
            "min_cf_pair_jsd_ema": min_ema,
            "single_update_ok": bool(update_ok),
            "actor_intervention_consecutive_updates": streak,
            "actor_jsd_consecutive_updates": required_consecutive,
        },
    )


def evaluate_macro_profile_support(cfg: PPOConfig, latent_state: Any) -> GateEvalResult:
    """Supporting macro-rollout profile check (finite EMA; not actor-scale)."""
    margin, floor, min_pairs = _macro_thresholds(cfg)
    valid_updates = int(getattr(latent_state, "macro_pair_jsd_valid_updates", 0) or 0)
    if valid_updates <= 0:
        return GateEvalResult(
            status=GATE_STATUS_NOT_RUN,
            reason="no_macro_pair_jsd_ema_updates",
            details={"macro_pair_jsd_valid_updates": valid_updates},
        )

    macro_ema = getattr(latent_state, "macro_pair_jsd_ema", None)
    if macro_ema is None:
        return GateEvalResult(
            status=GATE_STATUS_NOT_RUN,
            reason="missing_macro_pair_jsd_ema_buffer",
        )

    pair_jsd = [float(v) for v in macro_ema.tolist()]
    if len(pair_jsd) != 6 or not all(np.isfinite(pair_jsd)):
        return GateEvalResult(
            status=GATE_STATUS_ERROR,
            reason="corrupt_macro_pair_jsd_ema",
            details={"macro_pair_jsd_ema": pair_jsd},
        )

    num_above = sum(1 for v in pair_jsd if v >= margin)
    min_ema = float(min(pair_jsd))
    profile_ok = num_above >= min_pairs and min_ema >= floor
    status = GATE_STATUS_PASS if profile_ok else GATE_STATUS_FAIL

    return GateEvalResult(
        status=status,
        details={
            "macro_pair_jsd_ema": pair_jsd,
            "macro_profile": status,
            "macro_pair_jsd_valid_updates": valid_updates,
            "macro_pair_jsd_last_update_step": int(
                getattr(latent_state, "macro_pair_jsd_last_update_step", -1) or -1
            ),
            "macro_jsd_margin": margin,
            "macro_jsd_floor": floor,
            "macro_pairs_above_margin": int(num_above),
            "min_macro_pair_jsd_ema": min_ema,
        },
    )


def evaluate_matched_seed_semantics(
    cfg: PPOConfig,
    op_reports: dict[str, Any],
) -> GateEvalResult:
    """Mandatory matched-seed semantic component for behavioral realization."""
    if not op_reports:
        return GateEvalResult(
            status=GATE_STATUS_ERROR,
            reason="empty_matched_seed_reports",
        )

    effect_thresh = float(cfg.behavioral_realization_effect_threshold)
    adverse_thresh = float(cfg.behavioral_realization_adverse_threshold)
    min_strong = int(cfg.behavioral_realization_min_opponents_pass)
    min_seeds = int(getattr(cfg, "behavioral_matched_seed_min_seeds_per_opponent", 20))

    semantic_effects: list[float] = []
    strong_count = 0
    per_opponent: dict[str, Any] = {}

    for opp, rep in op_reports.items():
        n_seeds = int(rep.get("num_seeds", 0))
        if n_seeds < min_seeds:
            return GateEvalResult(
                status=GATE_STATUS_NOT_RUN,
                reason="insufficient_matched_seed_samples",
                details={"opponent": opp, "num_seeds": n_seeds, "min_seeds": min_seeds},
            )

        route = float(rep.get("avg_route_distance", 0.0))
        behav = float(rep.get("avg_behavior_distance", 0.0))
        wr_spread = float(rep.get("forced_z_performance_spread", 0.0))
        ci_low = float(rep.get("ci_95_low", 0.0))
        semantic = max(route, behav, float(rep.get("effect_size", 0.0)))
        strong = semantic >= effect_thresh and (
            ci_low > effect_thresh or wr_spread >= 0.03 or behav >= effect_thresh
        )
        semantic_effects.append(semantic)
        if strong:
            strong_count += 1
        per_opponent[opp] = {
            **rep,
            "semantic_effect": semantic,
            "semantic_pass": bool(strong),
        }

    aggregate = float(np.mean(semantic_effects)) if semantic_effects else 0.0
    no_adverse = all(e >= adverse_thresh for e in semantic_effects)
    passed = strong_count >= min_strong and no_adverse and aggregate >= effect_thresh
    status = GATE_STATUS_PASS if passed else GATE_STATUS_FAIL

    return GateEvalResult(
        status=status,
        details={
            "matched_seed_semantics": status,
            "opponents": per_opponent,
            "strong_opponent_count": int(strong_count),
            "aggregate_semantic_effect": aggregate,
            "no_adverse_opponent": bool(no_adverse),
            "behavioral_realization_effect_threshold": effect_thresh,
            "behavioral_realization_adverse_threshold": adverse_thresh,
        },
    )


def evaluate_behavioral_realization(
    cfg: PPOConfig,
    latent_state: Any,
    op_reports: dict[str, Any],
    *,
    boundary_eval_enabled: bool,
) -> GateEvalResult:
    """Gate B (v6i2): matched-seed semantics mandatory; macro profile is supporting only."""
    if not boundary_eval_enabled:
        return GateEvalResult(
            status=GATE_STATUS_NOT_RUN,
            reason="curriculum_gate_run_boundary_eval=false",
            details={
                "macro_profile": GATE_STATUS_NOT_RUN,
                "matched_seed_semantics": GATE_STATUS_NOT_RUN,
                "aggregate_result": GATE_STATUS_NOT_RUN,
            },
        )

    macro_result = evaluate_macro_profile_support(cfg, latent_state)
    semantics_result = evaluate_matched_seed_semantics(cfg, op_reports)

    if semantics_result.status == GATE_STATUS_ERROR:
        aggregate = GATE_STATUS_ERROR
    elif semantics_result.status == GATE_STATUS_NOT_RUN:
        aggregate = GATE_STATUS_NOT_RUN
    elif semantics_result.status == GATE_STATUS_PASS:
        aggregate = GATE_STATUS_PASS
    else:
        aggregate = GATE_STATUS_FAIL

    return GateEvalResult(
        status=aggregate,
        reason=semantics_result.reason if aggregate == GATE_STATUS_NOT_RUN else "",
        details={
            "behavioral_realization_gate_status": aggregate,
            "macro_profile": macro_result.status,
            "matched_seed_semantics": semantics_result.status,
            "aggregate_result": aggregate,
            "macro_profile_details": macro_result.details,
            "matched_seed_semantics_details": semantics_result.details,
        },
    )


__all__ = [
    "GATE_FAMILY_NAMES_V6I1",
    "GATE_FAMILY_NAMES_V6I2",
    "GATE_STATUS_ERROR",
    "GATE_STATUS_FAIL",
    "GATE_STATUS_NOT_RUN",
    "GATE_STATUS_PASS",
    "GateEvalResult",
    "KNOWN_GATE_PROTOCOLS",
    "V6I1_GATE_PROTOCOL",
    "V6I2_GATE_PROTOCOL",
    "evaluate_actor_intervention",
    "evaluate_behavioral_realization",
    "evaluate_macro_profile_support",
    "evaluate_matched_seed_semantics",
    "gate_family_names",
    "get_gate_family_names",
    "is_staged_v6_team_intent_curriculum",
    "is_v6i2_gate_protocol",
    "resolve_gate_protocol_version",
    "validate_protocol_config",
]
