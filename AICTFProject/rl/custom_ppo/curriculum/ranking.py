"""Lexicographic candidate ranking for Phase A gate failures."""

from __future__ import annotations

from typing import Any

import numpy as np

from rl.config.ppo_config import PPOConfig
from rl.custom_ppo.curriculum.types import (
    GATE_FAMILY_NAMES,
    GateFamilyResult,
    count_gate_families_measured,
    count_gate_families_passed,
)
from rl.custom_ppo.gate_protocol import gate_family_names, is_v6i2_gate_protocol


def _ranking_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    ranking = row.get("ranking_components", {})
    return (
        -int(ranking.get("gate_families_passed", 0)),
        -int(ranking.get("gate_families_measured", 0)),
        -float(ranking.get("min_competence", 0.0)),
        -int(ranking.get("pairs_above_margin", 0)),
        -float(ranking.get("weakest_pair_normalized_separation", 0.0)),
        -float(ranking.get("matched_seed_effect_size", 0.0)),
        -float(ranking.get("probe_regret_reduction", 0.0)),
        float(ranking.get("occupancy_imbalance", 1.0)),
        int(ranking.get("global_step", 0)),
    )


def rank_candidates_lexicographic(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Lexicographic best-candidate ranking after Phase A failure.

    Returns a new list; input dicts are not mutated.
    """
    ranked = sorted(candidates, key=_ranking_sort_key)
    return [{**row, "lexicographic_rank": rank} for rank, row in enumerate(ranked)]


def build_lexicographic_ranking_components(
    *,
    gate_results: dict[str, GateFamilyResult],
    online_report: dict[str, Any],
    matched_report: dict[str, Any],
    probe_report: dict[str, Any],
    global_step: int,
    cfg: PPOConfig | None = None,
) -> dict[str, Any]:
    families = gate_family_names(cfg) if cfg is not None else GATE_FAMILY_NAMES
    comp_scores = online_report.get("competence_scores", [0.0, 0.0, 0.0, 0.0])
    if is_v6i2_gate_protocol(cfg) if cfg is not None else False:
        pair_jsd = online_report.get("cf_pair_jsd_ema", [0.0] * 6)
        margin = float(online_report.get("actor_jsd_margin", 0.001))
    else:
        pair_jsd = online_report.get("pair_jsd_ema", [0.0] * 6)
        margin = float(online_report.get("jsd_margin", 0.01))
    occupancy = online_report.get(
        "recent_z_occupancy",
        online_report.get("occupancy", [0.25, 0.25, 0.25, 0.25]),
    )
    pairs_above = sum(1 for v in pair_jsd if float(v) >= margin)
    weakest_norm = 0.0
    if pair_jsd and margin > 0:
        weakest_norm = max(min(float(v) / margin for v in pair_jsd), 0.0)
        weakest_norm = min(weakest_norm, 1.0)

    effect_sizes = [
        float(v.get("semantic_effect", v.get("effect_size", 0.0)))
        for v in matched_report.get("opponents", {}).values()
        if isinstance(v, dict)
    ]
    matched_effect = max(effect_sizes) if effect_sizes else 0.0

    fixed_regret = float(
        probe_report.get(
            "fixed_policy_regret",
            probe_report.get(
                "fixed_regret",
                probe_report.get("global_best_fixed_z_regret", 0.0),
            ),
        )
    )
    probe_regret = float(probe_report.get("probe_regret", 0.0))
    regret_reduction = float(probe_report.get("regret_reduction", fixed_regret - probe_regret))

    occ = np.asarray(occupancy, dtype=np.float64)
    occ_imbalance = float(occ.max() - occ.min()) if occ.size else 1.0

    return {
        "gate_families_passed": count_gate_families_passed(gate_results, families=families),
        "gate_families_measured": count_gate_families_measured(gate_results, families=families),
        "min_competence": float(np.min(comp_scores)) if len(comp_scores) else 0.0,
        "pairs_above_margin": int(pairs_above),
        "weakest_pair_normalized_separation": float(weakest_norm),
        "matched_seed_effect_size": float(matched_effect),
        "probe_regret_reduction": float(regret_reduction),
        "occupancy_imbalance": float(occ_imbalance),
        "global_step": int(global_step),
    }


__all__ = [
    "build_lexicographic_ranking_components",
    "rank_candidates_lexicographic",
]
