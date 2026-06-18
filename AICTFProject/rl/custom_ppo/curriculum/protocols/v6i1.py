"""V6I1 single-macro-intervention gate protocol."""

from __future__ import annotations

from typing import Any

from rl.config.ppo_config import PPOConfig
from rl.custom_ppo.curriculum.context import GateContext
from rl.custom_ppo.curriculum.evaluators.learnability import run_learnability_probe
from rl.custom_ppo.curriculum.evaluators.matched_seed import evaluate_matched_seed_behavior
from rl.custom_ppo.curriculum.evaluators.online import (
    evaluate_competence,
    evaluate_coverage,
    evaluate_training_integrity,
    v6i1_intervention,
)
from rl.custom_ppo.curriculum.ranking import build_lexicographic_ranking_components
from rl.custom_ppo.curriculum.types import GateResult
from rl.custom_ppo.gate_protocol import GATE_FAMILY_NAMES_V6I1, V6I1_GATE_PROTOCOL


class V6I1GateProtocol:
    version = V6I1_GATE_PROTOCOL

    def required_families(self) -> tuple[str, ...]:
        return GATE_FAMILY_NAMES_V6I1

    def evaluate_online(self, context: GateContext) -> dict[str, GateResult]:
        results: dict[str, GateResult] = {}
        results["coverage"] = evaluate_coverage(context)
        results["competence"] = evaluate_competence(context)
        results["counterfactual_intervention"] = v6i1_intervention(context)
        results["training_integrity"] = evaluate_training_integrity(context)
        return results

    def evaluate_boundary(self, context: GateContext) -> dict[str, GateResult]:
        return {
            "matched_seed_behavior": evaluate_matched_seed_behavior(context),
            "selector_learnability_probe": run_learnability_probe(context),
        }

    def build_ranking(
        self,
        *,
        gate_results: dict[str, GateResult],
        online_report: dict[str, Any],
        matched_report: dict[str, Any],
        probe_report: dict[str, Any],
        global_step: int,
        cfg: PPOConfig,
    ) -> dict[str, Any]:
        return build_lexicographic_ranking_components(
            gate_results=gate_results,
            online_report=online_report,
            matched_report=matched_report,
            probe_report=probe_report,
            global_step=global_step,
            cfg=cfg,
        )


__all__ = ["V6I1GateProtocol"]
