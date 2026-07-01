"""Staged curriculum gate support (refactor of curriculum_gates)."""

from __future__ import annotations

from rl.custom_ppo.curriculum.controller import (
    V6I1CurriculumController,
    is_staged_v6i1_curriculum,
    validate_v6i1_enforce_config,
)
from rl.custom_ppo.curriculum.isolation import (
    GateIsolationBoundary,
    GateIsolationError,
    TrainingIsolationSnapshot,
    digest_all_optimizers,
    digest_module_params,
    digest_optimizer_state,
    isolated_gate_rng,
)
from rl.custom_ppo.curriculum.ranking import (
    build_lexicographic_ranking_components,
    rank_candidates_lexicographic,
)
from rl.custom_ppo.curriculum.reporting import (
    SCHEMA_VERSION,
    atomic_write_json,
    build_final_gate_report,
    format_v6i1_gate_stdout_block,
    write_gate_report,
)
from rl.custom_ppo.curriculum.schedule import (
    CurriculumSchedule,
    resolve_schedule,
    schedule_next_gate_step,
    should_run_phase_a_gate,
    should_trigger_terminal_failure,
)
from rl.custom_ppo.curriculum.types import (
    GATE_FAMILY_NAMES,
    GATE_STATUS_ERROR,
    GATE_STATUS_FAIL,
    GATE_STATUS_NOT_RUN,
    GATE_STATUS_PASS,
    LATENT_PAIR_INDEX,
    PAIR_ORDER,
    CurriculumPhase,
    GateAttempt,
    GateFamilyResult,
    GateMode,
    GateResult,
    GateStatus,
    all_required_families_passed,
    count_gate_families_measured,
    count_gate_families_passed,
    gate_family_result_from_bool,
    overall_gate_passed_for_promotion,
)

__all__ = [
    "V6I1CurriculumController",
    "GATE_FAMILY_NAMES",
    "GATE_STATUS_ERROR",
    "GATE_STATUS_FAIL",
    "GATE_STATUS_NOT_RUN",
    "GATE_STATUS_PASS",
    "LATENT_PAIR_INDEX",
    "PAIR_ORDER",
    "SCHEMA_VERSION",
    "CurriculumPhase",
    "CurriculumSchedule",
    "GateAttempt",
    "GateFamilyResult",
    "GateIsolationBoundary",
    "GateIsolationError",
    "GateMode",
    "GateResult",
    "GateStatus",
    "TrainingIsolationSnapshot",
    "all_required_families_passed",
    "atomic_write_json",
    "build_final_gate_report",
    "build_lexicographic_ranking_components",
    "count_gate_families_measured",
    "count_gate_families_passed",
    "digest_all_optimizers",
    "digest_module_params",
    "digest_optimizer_state",
    "format_v6i1_gate_stdout_block",
    "gate_family_result_from_bool",
    "is_staged_v6i1_curriculum",
    "isolated_gate_rng",
    "overall_gate_passed_for_promotion",
    "validate_v6i1_enforce_config",
    "rank_candidates_lexicographic",
    "resolve_schedule",
    "schedule_next_gate_step",
    "should_run_phase_a_gate",
    "should_trigger_terminal_failure",
    "write_gate_report",
]
