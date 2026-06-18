"""Gate report serialization and stdout formatting."""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any

from rl.config.ppo_config import PPOConfig
from rl.custom_ppo.curriculum.types import (
    GATE_STATUS_NOT_RUN,
    GateAttempt,
    GateFamilyResult,
)
from rl.custom_ppo.gate_protocol import V6I2_GATE_PROTOCOL, staged_latent_stdout_tag
from rl.forced_z_behavior_vectors import INTERVENTION_QUADRANT_GENUINE

SCHEMA_VERSION = 2


def atomic_write_json(path: str, payload: dict[str, Any], *, indent: int = 2) -> None:
    """Write JSON atomically via a temp file in the destination directory."""
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=".gate_report_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=indent)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def build_final_gate_report(
    attempt: GateAttempt,
    transition: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the machine-readable gate report for one Phase A attempt."""
    families = attempt.required_families
    gate_families_report = {
        name: attempt.gate_results[name].to_dict()
        for name in families
        if name in attempt.gate_results
    }
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "global_step": int(attempt.global_step),
        "checkpoint": attempt.checkpoint,
        "gate_protocol_version": attempt.gate_protocol_version,
        "required_families": list(families),
        "phase_boundary_gate_mode": attempt.mode,
        "curriculum_gate_run_boundary_eval": bool(attempt.boundary_enabled),
        "curriculum_gate_run_probe": bool(attempt.probe_enabled),
        "gate_families": gate_families_report,
        "gate_passed": bool(attempt.gate_passed),
        "promotion_allowed": bool(attempt.promotion_allowed),
        "overall_gate_passed": bool(attempt.overall_gate_passed),
        "promoted_to_phase_b": bool(attempt.promoted_to_phase_b),
        "nominal_transition_to_phase_b": bool(attempt.nominal_transition_to_phase_b),
        "online_report": dict(attempt.online_report),
        "matched_eval_report": dict(attempt.matched_report),
        "probe_report": dict(attempt.probe_report),
        "ranking_components": dict(attempt.ranking_components),
        "phase_a_gate_passed": bool(attempt.phase_a_gate_passed),
        "phase_a_end_step": attempt.phase_a_end_step,
    }
    if transition:
        report.update(transition)
    return report


def write_gate_report(cfg: PPOConfig, step: int, report: dict[str, Any]) -> str:
    """Persist a gate report under ``checkpoint_dir/phase_a_gate_reports``."""
    report_dir = os.path.join(cfg.checkpoint_dir, "phase_a_gate_reports")
    path = os.path.join(report_dir, f"gate_step_{int(step)}.json")
    atomic_write_json(path, report)
    print(f"[Curriculum Controller] Gate report: {path}")
    return path


def format_v6i1_gate_stdout_block(
    *,
    step: int,
    phase: str,
    overall_passed: bool,
    mode: str,
    gate_results: dict[str, GateFamilyResult],
    online_report: dict[str, Any],
    ranking_components: dict[str, Any],
    cf_coef: float,
    required_consecutive: int,
    report_path: str | None = None,
    gate_protocol: str | None = None,
) -> str:
    """Compact stdout block for a Phase A gate attempt."""
    protocol = gate_protocol or "v6i1_single_macro_intervention"
    tag = staged_latent_stdout_tag(protocol)

    def _st(name: str) -> str:
        return gate_results[name].status if name in gate_results else GATE_STATUS_NOT_RUN

    action = "PROMOTE_PHASE_B" if overall_passed and mode == "enforce" else "CONTINUE_PHASE_A"
    report_line = f"[{tag} Gate] report={report_path}" if report_path else ""

    if protocol == V6I2_GATE_PROTOCOL:
        actor = gate_results.get("actor_intervention")
        actor_details = actor.details if actor is not None else {}
        num_above = int(actor_details.get("num_pairs_above_margin", 0))
        min_ema = float(actor_details.get("min_cf_pair_jsd_ema", 0.0) or 0.0)
        actor_consec = int(actor_details.get("actor_intervention_consecutive_updates", 0) or 0)
        behav = gate_results.get("behavioral_realization")
        behav_details = behav.details if behav is not None else {}
        sem_details = behav_details.get("matched_seed_semantics_details", {})
        strong_ops = int(
            sem_details.get("strong_opponent_count", behav_details.get("strong_opponent_count", 0))
        )
        agg_effect = float(
            sem_details.get("aggregate_semantic_effect", behav_details.get("aggregate_semantic_effect", 0.0))
            or 0.0
        )
        lines = [
            f"[{tag} Gate] step={step} phase={phase} mode={mode} protocol={protocol}",
            (
                f"[{tag} Gate] coverage={_st('coverage')} competence={_st('competence')} "
                f"integrity={_st('training_integrity')}"
            ),
            (
                f"[{tag} Gate] actor_intervention={_st('actor_intervention')} "
                f"cf_pairs>=margin={num_above}/6 min_cf_ema={min_ema:.6f} "
                f"actor_consec={actor_consec}/{required_consecutive}"
            ),
            (
                f"[{tag} Gate] behavioral_realization={_st('behavioral_realization')} "
                f"strong_opponents={strong_ops}/3 aggregate_effect={agg_effect:.4f} "
                f"probe={_st('selector_learnability_probe')}"
            ),
            _format_phase_a_behavior_line(online_report),
            f"[{tag} Gate] overall={'PASS' if overall_passed else 'FAIL'} action={action} cf_coef={cf_coef:.4f}",
        ]
        if report_line:
            lines.append(report_line)
        return "\n".join(lines)

    intervention = gate_results.get("counterfactual_intervention")
    interv_details = intervention.details if intervention is not None else {}
    num_above = int(interv_details.get("num_pairs_above_margin", 0))
    min_ema = float(
        interv_details.get("min_pair_jsd_ema", online_report.get("min_pair_jsd_ema", 0.0)) or 0.0
    )
    jsd_consec = int(
        interv_details.get(
            "jsd_consecutive_updates",
            online_report.get(
                "jsd_consecutive_updates",
                online_report.get("jsd_gate_consecutive_updates", 0),
            ),
        )
        or 0
    )
    action = "PROMOTE_PHASE_B" if overall_passed and mode == "enforce" else "CONTINUE_PHASE_A"
    report_line = f"[{tag} Gate] report={report_path}" if report_path else ""
    lines = [
        f"[{tag} Gate] step={step} phase={phase} mode={mode} protocol={protocol}",
        (
            f"[{tag} Gate] coverage={_st('coverage')} competence={_st('competence')} "
            f"integrity={_st('training_integrity')}"
        ),
        (
            f"[{tag} Gate] intervention={_st('counterfactual_intervention')} "
            f"pairs>=margin={num_above}/6 min_jsd={min_ema:.5f} "
            f"jsd_consec={jsd_consec}/{required_consecutive}"
        ),
        (
            f"[{tag} Gate] matched_eval={_st('matched_seed_behavior')} "
            f"probe={_st('selector_learnability_probe')}"
        ),
        _format_phase_a_behavior_line(online_report),
        f"[{tag} Gate] overall={'PASS' if overall_passed else 'FAIL'} action={action} cf_coef={cf_coef:.4f}",
    ]
    if report_line:
        lines.append(report_line)
    return "\n".join(lines)


def _format_phase_a_behavior_line(online_report: dict[str, Any]) -> str:
    """Actor intervention vs behavioral realization 2x2 snapshot."""
    if not bool(float(online_report.get("phase_a_snapshot_usable", 0.0) or 0.0) >= 0.5):
        stale = bool(float(online_report.get("phase_a_stats_stale", 0.0) or 0.0) >= 0.5)
        reason = "stale" if stale else "invalid_or_incomplete"
        return f"[Phase A] behavior_diag=NOT_RUN reason={reason}"

    actor_jsd = float(online_report.get("phase_a_actor_jsd_mean", 0.0) or 0.0)
    beh_dist = float(online_report.get("phase_a_behavior_distance_min", 0.0) or 0.0)
    beh_mean = float(online_report.get("phase_a_behavior_distance_mean", 0.0) or 0.0)
    quad = str(online_report.get("phase_a_intervention_quadrant_label", "unknown"))
    regime = str(online_report.get("phase_a_cf_regime_label", "unknown"))
    corridor = bool(float(online_report.get("phase_a_corridor_viable", 0.0) or 0.0) >= 0.5)
    opp_fork = float(online_report.get("opportunity_fork_fraction_valid", 0.0) or 0.0)
    actor_pairs = int(online_report.get("phase_a_actor_pairs_above_margin", 0.0) or 0.0)
    beh_pairs = int(online_report.get("phase_a_behavior_pairs_above_threshold", 0.0) or 0.0)
    genuine = int(online_report.get("phase_a_intervention_quadrant", -1)) == INTERVENTION_QUADRANT_GENUINE
    return (
        f"[Phase A] actor_jsd={actor_jsd:.4f} behavior_min={beh_dist:.4f} behavior_mean={beh_mean:.4f} "
        f"quadrant={quad} cf_regime={regime} corridor_viable={corridor} "
        f"actor_pairs>margin={actor_pairs}/6 behavior_pairs>thr={beh_pairs}/6 "
        f"opp_fork_valid={opp_fork:.3f} genuine_control={genuine}"
    )


__all__ = [
    "SCHEMA_VERSION",
    "atomic_write_json",
    "build_final_gate_report",
    "format_v6i1_gate_stdout_block",
    "write_gate_report",
]
