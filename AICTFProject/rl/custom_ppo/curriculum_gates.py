"""Backward-compatible re-exports for the staged curriculum gate package.

Implementation lives under ``rl.custom_ppo.curriculum``.
"""

from __future__ import annotations

from rl.custom_ppo.curriculum.controller import (
    V6I1CurriculumController,
    is_staged_v6i1_curriculum,
    validate_v6i1_enforce_config,
)
from rl.custom_ppo.curriculum.evaluators.learnability import LearnabilityClassifier
from rl.custom_ppo.curriculum.isolation import TrainingIsolationSnapshot
from rl.custom_ppo.curriculum.ranking import (
    build_lexicographic_ranking_components,
    rank_candidates_lexicographic,
)
from rl.custom_ppo.curriculum.reporting import format_v6i1_gate_stdout_block
from rl.custom_ppo.curriculum.types import (
    GATE_FAMILY_NAMES,
    GATE_STATUS_ERROR,
    GATE_STATUS_FAIL,
    GATE_STATUS_NOT_RUN,
    GATE_STATUS_PASS,
    LATENT_PAIR_INDEX,
    PAIR_ORDER,
    GateFamilyResult,
    all_required_families_passed,
    count_gate_families_measured,
    count_gate_families_passed,
    gate_family_result_from_bool,
    overall_gate_passed_for_promotion,
)

__all__ = [
    "GATE_FAMILY_NAMES",
    "GATE_STATUS_ERROR",
    "GATE_STATUS_FAIL",
    "GATE_STATUS_NOT_RUN",
    "GATE_STATUS_PASS",
    "LATENT_PAIR_INDEX",
    "PAIR_ORDER",
    "GateFamilyResult",
    "LearnabilityClassifier",
    "TrainingIsolationSnapshot",
    "V6I1CurriculumController",
    "all_required_families_passed",
    "build_lexicographic_ranking_components",
    "count_gate_families_measured",
    "count_gate_families_passed",
    "format_v6i1_gate_stdout_block",
    "gate_family_result_from_bool",
    "is_staged_v6i1_curriculum",
    "overall_gate_passed_for_promotion",
    "rank_candidates_lexicographic",
    "validate_v6i1_enforce_config",
]
