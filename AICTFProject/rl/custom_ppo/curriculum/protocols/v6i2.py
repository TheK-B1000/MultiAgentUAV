"""V6I2 dual-evidence gate protocol."""

from __future__ import annotations

from typing import Any

from rl.config.ppo_config import PPOConfig
from rl.custom_ppo.curriculum.context import GateContext
from rl.custom_ppo.curriculum.evaluators.learnability import run_learnability_probe
from rl.custom_ppo.curriculum.evaluators.matched_seed import collect_matched_seed_metrics
from rl.custom_ppo.curriculum.evaluators.online import (
    evaluate_competence,
    evaluate_coverage,
    evaluate_training_integrity,
    v6i2_actor_intervention,
)
from rl.custom_ppo.curriculum.ranking import build_lexicographic_ranking_components
from rl.custom_ppo.curriculum.types import (
    GATE_STATUS_ERROR,
    GATE_STATUS_NOT_RUN,
    GateResult,
)
from rl.custom_ppo.gate_protocol import (
    GATE_FAMILY_NAMES_V6I2,
    V6I2_GATE_PROTOCOL,
    evaluate_behavioral_realization,
)


class V6I2GateProtocol:
    version = V6I2_GATE_PROTOCOL

    def required_families(self) -> tuple[str, ...]:
        return GATE_FAMILY_NAMES_V6I2

    def evaluate_online(self, context: GateContext) -> dict[str, GateResult]:
        results: dict[str, GateResult] = {}
        results["coverage"] = evaluate_coverage(context)
        results["competence"] = evaluate_competence(context)
        results["actor_intervention"] = v6i2_actor_intervention(context)
        results["training_integrity"] = evaluate_training_integrity(context)
        return results

    def evaluate_boundary(self, context: GateContext) -> dict[str, GateResult]:
        behavioral = self._evaluate_behavioral_realization(context)
        return {
            "behavioral_realization": behavioral,
            "selector_learnability_probe": run_learnability_probe(context),
        }

    def _evaluate_behavioral_realization(self, context: GateContext) -> GateResult:
        if not bool(getattr(context.cfg, "curriculum_gate_run_boundary_eval", False)):
            return GateResult(
                status=GATE_STATUS_NOT_RUN,
                reason="curriculum_gate_run_boundary_eval=false",
                details={
                    "macro_profile": GATE_STATUS_NOT_RUN,
                    "matched_seed_semantics": GATE_STATUS_NOT_RUN,
                    "aggregate_result": GATE_STATUS_NOT_RUN,
                },
            )

        print("[Curriculum Controller] Behavioral-realization boundary evaluation...")
        op_reports, any_mismatch = collect_matched_seed_metrics(context)
        if any_mismatch:
            return GateResult(
                status=GATE_STATUS_ERROR,
                reason="matched_seed_reset_mismatch",
                details={
                    "macro_profile": GATE_STATUS_NOT_RUN,
                    "matched_seed_semantics": GATE_STATUS_ERROR,
                    "aggregate_result": GATE_STATUS_ERROR,
                    "opponents": op_reports,
                },
            )

        eval_result = evaluate_behavioral_realization(
            context.cfg,
            context.trainer.latent_state,
            op_reports,
            boundary_eval_enabled=True,
        )
        return GateResult(
            status=eval_result.status,
            reason=eval_result.reason,
            details=dict(eval_result.details),
        )

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


__all__ = ["V6I2GateProtocol"]
