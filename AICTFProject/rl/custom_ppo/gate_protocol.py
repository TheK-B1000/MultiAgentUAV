"""V6 staged curriculum gate protocol — behavior only.

Resolved thresholds and timing live exclusively in ``PPOConfig``; this module
consumes those values and must not maintain a parallel default set.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from rl.config.ppo_config import PPOConfig
from rl.gate_telemetry import (
    phase_a_actor_pair_telemetry_from_actor_gate_details,
    phase_a_matched_seed_behavioral_telemetry_from_gate_details,
)

V6I1_GATE_PROTOCOL = "v6i1_single_macro_intervention"
V6I2_GATE_PROTOCOL = "v6i2_dual_evidence"
V6I3_GATE_PROTOCOL = "v6i3_strategy_local_comm_v1"

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
)

GATE_FAMILY_NAMES_V6I2: tuple[str, ...] = (
    "coverage",
    "competence",
    "actor_intervention",
    "behavioral_realization",
    "training_integrity",
)

GATE_FAMILY_NAMES_V6I3: tuple[str, ...] = GATE_FAMILY_NAMES_V6I2 + (
    "communication_usage",
)

KNOWN_GATE_PROTOCOLS = frozenset({V6I1_GATE_PROTOCOL, V6I2_GATE_PROTOCOL, V6I3_GATE_PROTOCOL})

# Serializable gate keys — single source for checkpoint fingerprinting.
_GATE_CONFIG_KEYS_COMMON: tuple[str, ...] = (
    "gate_protocol_version",
    "phase_a_earliest_end_fraction",
    "phase_a_max_end_fraction",
    "phase_b_fixed_fraction",
    "phase_c_fixed_fraction",
    "phase_c_start_fraction",
    "curriculum_extend_terminal_on_late_promotion",
    "phase_boundary_gate_mode",
    "phase_a_gate_max_seconds",
    "phase_a_gate_progress_interval_seconds",
    "curriculum_gate_online_matched_seed_count",
    "curriculum_gate_online_matched_seed_max_steps",
    "curriculum_gate_run_boundary_eval",
    "curriculum_gate_run_probe",
    "curriculum_gate_selector_blocks_phase_a",
    "curriculum_probe_min_examples",
    "behavioral_matched_seed_min_seeds_per_opponent",
)

_GATE_CONFIG_KEYS_V6I1: tuple[str, ...] = _GATE_CONFIG_KEYS_COMMON + (
    "latent_cf_occupancy_min",
    "latent_cf_occupancy_max",
    "latent_cf_jsd_margin",
    "latent_cf_jsd_ema_alpha",
    "latent_cf_gate_consecutive_updates",
)

_GATE_CONFIG_KEYS_V6I2: tuple[str, ...] = _GATE_CONFIG_KEYS_COMMON + (
    "latent_cf_occupancy_min",
    "latent_cf_occupancy_max",
    "actor_jsd_margin",
    "actor_jsd_floor_fraction",
    "actor_jsd_min_passing_pairs",
    "actor_jsd_consecutive_updates",
    "actor_jsd_ema_decay",
    "actor_jsd_stale_gate_grace",
    "macro_jsd_margin",
    "macro_jsd_floor_fraction",
    "macro_jsd_min_passing_pairs",
    "macro_jsd_ema_decay",
    "behavioral_realization_min_opponents_pass",
    "behavioral_realization_effect_threshold",
    "behavioral_realization_adverse_threshold",
    "behavioral_route_distance_scale",
    "behavioral_task_behavior_distance_scale",
    "behavioral_performance_spread_scale",
    "behavioral_route_distance_weight",
    "behavioral_task_behavior_distance_weight",
    "behavioral_performance_spread_weight",
    "behavioral_aggregate_effect_threshold",
    "behavioral_min_task_behavior_distance",
    "behavioral_min_performance_spread",
)

_GATE_CONFIG_KEYS_V6I3: tuple[str, ...] = _GATE_CONFIG_KEYS_V6I2 + (
    "communication_enabled",
    "comm_protocol_version",
    "comm_num_symbols",
    "comm_silence_symbol",
    "comm_interval_steps",
    "comm_delivery_delay_steps",
    "comm_radius_cells",
    "comm_dropout_probability",
    "comm_entropy_coef",
    "comm_cf_include_message_head",
    "comm_min_valid_boundaries",
    "comm_min_deliveries",
    "comm_min_symbols_used",
    "comm_entropy_floor",
    "comm_symbol_dominance_ceiling",
    "comm_listener_jsd_margin",
    "comm_listener_min_passing_pairs",
    "comm_listener_min_states",
    "comm_listener_consecutive_updates",
)


def resolved_gate_config_dict(cfg: PPOConfig) -> dict[str, Any]:
    """Resolved gate configuration for checkpointing and audit (no hidden defaults)."""
    protocol = resolve_gate_protocol_version(cfg)
    if protocol == V6I3_GATE_PROTOCOL:
        keys = _GATE_CONFIG_KEYS_V6I3
    elif protocol == V6I2_GATE_PROTOCOL:
        keys = _GATE_CONFIG_KEYS_V6I2
    else:
        keys = _GATE_CONFIG_KEYS_V6I1
    resolved = {key: getattr(cfg, key) for key in keys}
    if protocol in {V6I2_GATE_PROTOCOL, V6I3_GATE_PROTOCOL}:
        resolved["actor_intervention_gate_rule"] = "batch_margin_ema_floor_v1"
    return resolved


def gate_config_fingerprint(cfg: PPOConfig) -> str:
    """Deterministic hash of resolved gate configuration."""
    payload = resolved_gate_config_dict(cfg)
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def apply_gate_config_mismatch_override(
    cfg: PPOConfig,
    *,
    checkpoint_fingerprint: str,
    active_fingerprint: str,
) -> None:
    """Mark a resume that relaxed gate-config parity as non-confirmatory."""
    cfg.gate_config_mismatch_override_used = True
    cfg.gate_config_fingerprint_checkpoint = str(checkpoint_fingerprint)
    cfg.gate_config_fingerprint_active = str(active_fingerprint)
    cfg.confirmatory_gate_lineage_valid = False
    if str(getattr(cfg, "phase_boundary_gate_mode", "enforce")).lower() == "enforce":
        cfg.phase_boundary_gate_mode = "observe_only"


def gate_lineage_audit_fields(cfg: PPOConfig) -> dict[str, Any]:
    """Serializable lineage flags for checkpoints and gate reports."""
    return {
        "gate_config_mismatch_override_used": bool(
            getattr(cfg, "gate_config_mismatch_override_used", False)
        ),
        "gate_config_fingerprint_checkpoint": str(
            getattr(cfg, "gate_config_fingerprint_checkpoint", "") or ""
        ),
        "gate_config_fingerprint_active": str(
            getattr(cfg, "gate_config_fingerprint_active", "") or ""
        ),
        "confirmatory_gate_lineage_valid": bool(
            getattr(cfg, "confirmatory_gate_lineage_valid", True)
        ),
    }


def format_gate_mismatch_override_warning(cfg: PPOConfig) -> list[str]:
    if not bool(getattr(cfg, "gate_config_mismatch_override_used", False)):
        return []
    ckpt_fp = str(getattr(cfg, "gate_config_fingerprint_checkpoint", "") or "")
    active_fp = str(getattr(cfg, "gate_config_fingerprint_active", "") or "")
    return [
        "[PPO] *** GATE CONFIG MISMATCH OVERRIDE — NOT CONFIRMATORY ***",
        f"[PPO] checkpoint fingerprint={ckpt_fp} active fingerprint={active_fp}",
        "[PPO] enforce promotion disabled; run is observe-only for gate lineage.",
    ]


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


def is_v6i3_gate_protocol(cfg: PPOConfig) -> bool:
    return resolve_gate_protocol_version(cfg) == V6I3_GATE_PROTOCOL


def is_v6i2_dual_evidence_protocol(cfg: PPOConfig) -> bool:
    """True for v6i2 and v6i3 rows that share actor-intervention + behavioral-realization gates."""
    return is_v6i2_gate_protocol(cfg) or is_v6i3_gate_protocol(cfg)


def get_gate_family_names(cfg: PPOConfig) -> tuple[str, ...]:
    if is_v6i3_gate_protocol(cfg):
        return GATE_FAMILY_NAMES_V6I3
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
    if exp_id == "v6i3" and protocol != V6I3_GATE_PROTOCOL:
        raise ValueError(f"v6i3 runs must set gate_protocol_version={V6I3_GATE_PROTOCOL!r}.")
    if exp_id == "v6i3" and not bool(getattr(cfg, "communication_enabled", False)):
        raise ValueError("v6i3 runs must set communication_enabled=True.")
    if exp_id == "v6i1" and protocol == V6I2_GATE_PROTOCOL:
        raise ValueError("v6i1 experiment_id cannot use v6i2_dual_evidence protocol.")
    if exp_id in ("v6i1", "v6i2") and protocol == V6I3_GATE_PROTOCOL:
        raise ValueError("v6i1/v6i2 experiment_id cannot use v6i3 gate protocol.")

    mode = str(getattr(cfg, "phase_boundary_gate_mode", "enforce")).lower()
    if mode != "enforce":
        return
    if not bool(getattr(cfg, "curriculum_gate_run_boundary_eval", False)):
        raise ValueError("V6 staged enforce mode requires matched-seed boundary evaluation.")
    selector_blocks = bool(getattr(cfg, "curriculum_gate_selector_blocks_phase_a", False))
    if selector_blocks and not bool(getattr(cfg, "curriculum_gate_run_probe", False)):
        raise ValueError("V6 staged enforce mode requires the selector-learnability probe.")


def is_staged_v6_team_intent_curriculum(cfg: PPOConfig) -> bool:
    """True for v6i1 and v6i2 staged team-intent rows (not repertoire ablations)."""
    return (
        bool(getattr(cfg, "use_v6i1_curriculum", False))
        and str(getattr(cfg, "training_mode", "default")) == "staged_team_intent_curriculum"
        and str(getattr(cfg, "experiment_family", "v6")) == "v6"
        and str(getattr(cfg, "experiment_id", "v6i1")) in ("v6i1", "v6i2", "v6i3")
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


def _latent_pair_labels(latent_k: int = 4) -> list[str]:
    labels: list[str] = []
    for i in range(int(latent_k)):
        for j in range(i + 1, int(latent_k)):
            labels.append(f"z{i}-z{j}")
    return labels


def _actor_pair_ledger(
    *,
    raw_pairs: list[float],
    ema_pairs: list[float],
    margin: float,
    floor: float,
    streak_before: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, label in enumerate(_latent_pair_labels(4)):
        raw = float(raw_pairs[idx]) if idx < len(raw_pairs) else float("nan")
        ema = float(ema_pairs[idx]) if idx < len(ema_pairs) else float("nan")
        rows.append(
            {
                "opponent": "ALL",
                "pair": label,
                "pair_index": idx,
                "raw_score": raw,
                "ema": ema,
                "margin": float(margin),
                "floor": float(floor),
                "raw_pass": bool(np.isfinite(raw) and raw >= float(margin)),
                "ema_pass": bool(np.isfinite(ema) and ema >= float(floor)),
                "streak": int(streak_before),
            }
        )
    return rows


def evaluate_actor_intervention(cfg: PPOConfig, latent_state: Any) -> GateEvalResult:
    """Gate A (v6i2): current CF-batch strength plus actor-CF EMA floor stability."""
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

    cf_batch = getattr(latent_state, "cf_pair_jsd_last_batch", None)
    if cf_batch is None:
        return GateEvalResult(
            status=GATE_STATUS_NOT_RUN,
            reason="missing_cf_pair_jsd_last_batch",
            details={"cf_pair_jsd_valid_updates": valid_updates},
        )
    batch_jsd = [float(v) for v in cf_batch.tolist()]
    if len(batch_jsd) != 6 or not all(np.isfinite(batch_jsd)):
        return GateEvalResult(
            status=GATE_STATUS_ERROR,
            reason="corrupt_cf_pair_jsd_last_batch",
            details={"cf_pair_jsd_last_batch": batch_jsd},
        )

    batch_pairs_above_margin = sum(1 for v in batch_jsd if v >= margin)
    ema_pairs_above_margin = sum(1 for v in pair_jsd if v >= margin)
    ema_pairs_above_floor = sum(1 for v in pair_jsd if v >= floor)
    min_ema = float(min(pair_jsd))
    batch_pass = batch_pairs_above_margin >= min_pairs
    ema_floor_pass = ema_pairs_above_floor >= min_pairs
    update_ok = batch_pass and ema_floor_pass
    streak = int(getattr(latent_state, "actor_intervention_consecutive_updates", 0) or 0)
    skipped_gate_count = int(getattr(latent_state, "actor_intervention_skipped_gate_count", 0) or 0)
    stale_grace = int(getattr(cfg, "actor_jsd_stale_gate_grace", 1) or 1)
    streak_before = max(0, streak - 1 if update_ok and streak > 0 else streak)
    ledger = _actor_pair_ledger(
        raw_pairs=batch_jsd,
        ema_pairs=pair_jsd,
        margin=margin,
        floor=floor,
        streak_before=streak_before,
    )
    weakest_pairs = [
        f"ALL:{row['pair']}"
        for row in sorted(
            ledger,
            key=lambda row: (bool(row["raw_pass"] and row["ema_pass"]), float(row["ema"])),
        )[:3]
    ]
    if skipped_gate_count > 0:
        return GateEvalResult(
            status=GATE_STATUS_NOT_RUN,
            reason="actor_pair_evidence_stale_after_skipped_gate",
            details={
                "cf_pair_jsd_ema": pair_jsd,
                "cf_pair_jsd_last_batch": batch_jsd,
                "actor_pair_ledger": ledger,
                "opponent_specific_pair_ledger": False,
                "actor_intervention_gate_status": GATE_STATUS_NOT_RUN,
                "cf_pair_jsd_valid_updates": valid_updates,
                "cf_pair_jsd_last_update_step": int(
                    getattr(latent_state, "cf_pair_jsd_last_update_step", -1) or -1
                ),
                "actor_jsd_margin": margin,
                "actor_jsd_floor": floor,
                "actor_jsd_min_passing_pairs": min_pairs,
                "passing_pairs": int(ema_pairs_above_floor),
                "total_pairs": 6,
                "required_pairs": int(min_pairs),
                "weakest_pairs": weakest_pairs,
                "behavior_eval_valid": False,
                "behavior_evidence_status": "stale_requires_fresh_actor_pair_update",
                "actor_intervention_skipped_gate_count": skipped_gate_count,
                "actor_jsd_stale_gate_grace": stale_grace,
                "actor_pair_streak_preserved": streak,
                "actor_intervention_consecutive_updates": streak,
                "actor_jsd_consecutive_updates": required_consecutive,
            },
        )
    passed = update_ok and streak >= required_consecutive
    status = GATE_STATUS_PASS if passed else GATE_STATUS_FAIL

    return GateEvalResult(
        status=status,
        details={
            "cf_pair_jsd_ema": pair_jsd,
            "cf_pair_jsd_last_batch": batch_jsd,
            "actor_pair_ledger": ledger,
            "opponent_specific_pair_ledger": False,
            "actor_intervention_gate_status": status,
            "cf_pair_jsd_valid_updates": valid_updates,
            "cf_pair_jsd_last_update_step": int(
                getattr(latent_state, "cf_pair_jsd_last_update_step", -1) or -1
            ),
            "actor_jsd_margin": margin,
            "actor_jsd_floor": floor,
            "actor_jsd_min_passing_pairs": min_pairs,
            "passing_pairs": int(ema_pairs_above_floor),
            "total_pairs": 6,
            "required_pairs": int(min_pairs),
            "weakest_pairs": weakest_pairs,
            "behavior_eval_valid": True,
            "streak_before": streak_before,
            "streak_after": streak,
            "num_pairs_above_margin": int(batch_pairs_above_margin),
            "batch_pairs_above_margin": int(batch_pairs_above_margin),
            "ema_pairs_above_margin": int(ema_pairs_above_margin),
            "ema_pairs_above_floor": int(ema_pairs_above_floor),
            "min_cf_pair_jsd_ema": min_ema,
            "batch_pass": bool(batch_pass),
            "ema_floor_pass": bool(ema_floor_pass),
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


def _opponent_semantic_verdict(
    rep: dict[str, Any],
    *,
    cfg: PPOConfig,
    adverse_thresh: float,
) -> tuple[bool, str, dict[str, float | bool]]:
    """Return (strong_pass, auditable verdict code) for one opponent."""
    route = float(rep.get("avg_route_distance", 0.0))
    behav = float(rep.get("avg_behavior_distance", 0.0))
    wr_spread = float(rep.get("forced_z_performance_spread", 0.0))
    route_scale = max(1e-8, float(getattr(cfg, "behavioral_route_distance_scale", 0.03)))
    behavior_scale = max(
        1e-8,
        float(getattr(cfg, "behavioral_task_behavior_distance_scale", 0.02)),
    )
    performance_scale = max(
        1e-8,
        float(getattr(cfg, "behavioral_performance_spread_scale", 0.03)),
    )
    route_weight = float(getattr(cfg, "behavioral_route_distance_weight", 0.25))
    behavior_weight = float(getattr(cfg, "behavioral_task_behavior_distance_weight", 0.50))
    performance_weight = float(getattr(cfg, "behavioral_performance_spread_weight", 0.25))
    normalized_route = route / route_scale
    normalized_behavior = behav / behavior_scale
    normalized_performance = wr_spread / performance_scale
    aggregate = (
        route_weight * normalized_route
        + behavior_weight * normalized_behavior
        + performance_weight * normalized_performance
    )
    aggregate_thresh = float(getattr(cfg, "behavioral_aggregate_effect_threshold", 0.75))
    min_behavior = float(getattr(cfg, "behavioral_min_task_behavior_distance", 0.01))
    min_performance = float(getattr(cfg, "behavioral_min_performance_spread", 0.01))
    behavior_floor_pass = behav >= min_behavior
    performance_floor_pass = wr_spread >= min_performance
    component_floor_pass = bool(behavior_floor_pass and performance_floor_pass)
    details: dict[str, float | bool] = {
        "route_distance": route,
        "task_behavior_distance": behav,
        "performance_spread": wr_spread,
        "normalized_route_distance": normalized_route,
        "normalized_task_behavior_distance": normalized_behavior,
        "normalized_performance_spread": normalized_performance,
        "aggregate_effect": aggregate,
        "behavioral_route_distance_scale": route_scale,
        "behavioral_task_behavior_distance_scale": behavior_scale,
        "behavioral_performance_spread_scale": performance_scale,
        "behavioral_route_distance_weight": route_weight,
        "behavioral_task_behavior_distance_weight": behavior_weight,
        "behavioral_performance_spread_weight": performance_weight,
        "behavioral_aggregate_effect_threshold": aggregate_thresh,
        "behavioral_min_task_behavior_distance": min_behavior,
        "behavioral_min_performance_spread": min_performance,
        "behavior_component_floor_pass": bool(behavior_floor_pass),
        "performance_component_floor_pass": bool(performance_floor_pass),
        "component_floor_pass": component_floor_pass,
    }

    if wr_spread < adverse_thresh:
        return False, "FAIL_ADVERSE_PERFORMANCE", details
    if not component_floor_pass:
        return False, "FAIL_COMPONENT_FLOOR", details
    if aggregate < aggregate_thresh:
        return False, "FAIL_BELOW_NORMALIZED_AGGREGATE", details
    return True, "PASS_NORMALIZED_COMPONENTS", details


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

    adverse_thresh = float(cfg.behavioral_realization_adverse_threshold)
    effect_thresh = float(cfg.behavioral_realization_effect_threshold)
    aggregate_thresh = float(getattr(cfg, "behavioral_aggregate_effect_threshold", 0.75))
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
        ci_high = float(rep.get("ci_95_high", 0.0))
        strong, verdict, component_details = _opponent_semantic_verdict(
            rep, cfg=cfg, adverse_thresh=adverse_thresh
        )
        aggregate_effect = float(component_details["aggregate_effect"])
        semantic_effects.append(aggregate_effect)
        if strong:
            strong_count += 1
        per_opponent[opp] = {
            **rep,
            "semantic_effect": aggregate_effect,
            "aggregate_effect": aggregate_effect,
            "semantic_pass": bool(strong),
            "semantic_verdict": verdict,
            "route_effect": route,
            "route_distance": route,
            "route_ci_low": ci_low,
            "route_ci_high": ci_high,
            "behavior_effect": behav,
            "task_behavior_distance": behav,
            "performance_spread": wr_spread,
            "num_seeds": n_seeds,
            **component_details,
        }

    aggregate = float(np.mean(semantic_effects)) if semantic_effects else 0.0
    no_adverse = all(
        float(row.get("performance_spread", 0.0)) >= adverse_thresh
        for row in per_opponent.values()
    )
    passed = strong_count >= min_strong and no_adverse and aggregate >= aggregate_thresh
    status = GATE_STATUS_PASS if passed else GATE_STATUS_FAIL

    return GateEvalResult(
        status=status,
        details={
            "matched_seed_semantics": status,
            "opponents": per_opponent,
            "strong_opponent_count": int(strong_count),
            "behavioral_realization_min_opponents_pass": int(min_strong),
            "aggregate_semantic_effect": aggregate,
            "aggregate_effect": aggregate,
            "no_adverse_opponent": bool(no_adverse),
            "behavioral_realization_effect_threshold": effect_thresh,
            "behavioral_aggregate_effect_threshold": aggregate_thresh,
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


def staged_latent_stdout_tag(gate_protocol: str | None) -> str:
    """Short stdout label for staged-latent rollout/gate diagnostics."""
    if gate_protocol == V6I3_GATE_PROTOCOL:
        return "V6I3"
    if gate_protocol == V6I2_GATE_PROTOCOL:
        return "V6I2"
    return "V6I1"


__all__ = [
    "GATE_FAMILY_NAMES_V6I1",
    "GATE_FAMILY_NAMES_V6I2",
    "GATE_FAMILY_NAMES_V6I3",
    "GATE_STATUS_ERROR",
    "GATE_STATUS_FAIL",
    "GATE_STATUS_NOT_RUN",
    "GATE_STATUS_PASS",
    "GateEvalResult",
    "KNOWN_GATE_PROTOCOLS",
    "V6I1_GATE_PROTOCOL",
    "V6I2_GATE_PROTOCOL",
    "V6I3_GATE_PROTOCOL",
    "apply_gate_config_mismatch_override",
    "evaluate_actor_intervention",
    "evaluate_behavioral_realization",
    "evaluate_macro_profile_support",
    "evaluate_matched_seed_semantics",
    "format_gate_mismatch_override_warning",
    "staged_latent_stdout_tag",
    "gate_config_fingerprint",
    "gate_family_names",
    "gate_lineage_audit_fields",
    "get_gate_family_names",
    "is_staged_v6_team_intent_curriculum",
    "is_v6i2_dual_evidence_protocol",
    "is_v6i2_gate_protocol",
    "is_v6i3_gate_protocol",
    "phase_a_actor_pair_telemetry_from_actor_gate_details",
    "phase_a_matched_seed_behavioral_telemetry_from_gate_details",
    "resolve_gate_protocol_version",
    "resolved_gate_config_dict",
    "validate_protocol_config",
]
