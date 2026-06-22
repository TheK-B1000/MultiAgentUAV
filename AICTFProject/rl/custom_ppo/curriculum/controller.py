"""Staged V6I1 curriculum controller — orchestration only."""

from __future__ import annotations

import os
import time
from typing import Any, Optional

import torch
import torch.nn as nn

from rl.config.ppo_config import PPOConfig
from rl.custom_ppo.csv_writers import _OPPONENT_ID_TO_TAG
from rl.custom_ppo.curriculum.context import GateContext
from rl.custom_ppo.curriculum.evaluators.learnability import run_learnability_probe
from rl.custom_ppo.curriculum.evaluators.matched_seed import evaluate_matched_seed_behavior
from rl.custom_ppo.curriculum.evaluators.online import (
    evaluate_competence,
    evaluate_coverage,
    evaluate_training_integrity,
    v6i1_intervention,
    v6i2_actor_intervention,
)
from rl.custom_ppo.curriculum.isolation import GateIsolationBoundary
from rl.custom_ppo.curriculum.protocols import build_gate_protocol
from rl.custom_ppo.curriculum.ranking import rank_candidates_lexicographic
from rl.custom_ppo.curriculum.reporting import (
    SCHEMA_VERSION,
    atomic_write_json,
    build_final_gate_report,
    format_v6i1_gate_stdout_block,
    write_gate_report,
)
from rl.custom_ppo.curriculum.schedule import (
    resolve_schedule,
    schedule_next_gate_step,
    should_run_phase_a_gate,
    should_trigger_terminal_failure,
)
from rl.custom_ppo.curriculum.types import (
    CurriculumPhase,
    GateAttempt,
    GateFamilyResult,
    GateMode,
    GateResult,
    GATE_STATUS_FAIL,
    GATE_STATUS_NOT_RUN,
    all_required_families_passed,
)
from rl.custom_ppo.gate_protocol import (
    gate_family_names,
    is_v6i2_dual_evidence_protocol,
    is_v6i2_gate_protocol,
    resolve_gate_protocol_version,
    validate_protocol_config,
)
from rl.gate_telemetry import (
    phase_a_actor_pair_telemetry_from_actor_gate_details,
    phase_a_matched_seed_behavioral_telemetry_from_gate_details,
)
from rl.custom_ppo.inference import CustomPPOInferencePolicy

_STATE_SCHEMA_VERSION = 1


def is_staged_v6i1_curriculum(cfg: PPOConfig) -> bool:
    """True for staged team-intent V6 rows (v6i1 and v6i2), not repertoire ablations."""
    return (
        bool(getattr(cfg, "use_v6i1_curriculum", False))
        and str(getattr(cfg, "training_mode", "default")) == "staged_team_intent_curriculum"
        and str(getattr(cfg, "experiment_family", "v6")) == "v6"
        and str(getattr(cfg, "experiment_id", "v6i1")) in ("v6i1", "v6i2", "v6i3", "v6i5")
    )


def validate_v6i1_enforce_config(cfg: PPOConfig) -> None:
    """Fail fast when enforce mode cannot run the full Phase A gate for staged v6 rows."""
    validate_protocol_config(cfg)


def _to_family_result(result: GateResult) -> GateFamilyResult:
    return GateFamilyResult(
        status=result.status,
        reason=result.reason,
        details=dict(result.details),
    )


def _build_online_report(online_results: dict[str, GateResult]) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for name, result in online_results.items():
        report.update(result.details)
        if name == "actor_intervention":
            report.update(
                phase_a_actor_pair_telemetry_from_actor_gate_details(dict(result.details))
            )
        report[f"{name}_status"] = result.status
    return report


def _not_run(reason: str, details: dict[str, Any] | None = None) -> GateFamilyResult:
    return GateFamilyResult(
        status=GATE_STATUS_NOT_RUN,
        reason=reason,
        details=dict(details or {}),
    )


def _failed(reason: str, details: dict[str, Any] | None = None) -> GateFamilyResult:
    return GateFamilyResult(
        status=GATE_STATUS_FAIL,
        reason=reason,
        details=dict(details or {}),
    )


def _remove_file_if_exists(path: str) -> bool:
    if not path:
        return False
    if not os.path.exists(path):
        return False
    os.remove(path)
    return True


class V6I1CurriculumController:
    """Manages V6I1 staged curriculum transitions, boundary evaluations, and diagnostic probes."""

    def __init__(self, trainer: Any) -> None:
        self.trainer = trainer
        self.cfg: PPOConfig = trainer.cfg
        validate_v6i1_enforce_config(self.cfg)
        self.schedule = resolve_schedule(self.cfg)
        self.nominal_steps = self.schedule.nominal_steps
        self.phase_a_min_end = self.schedule.phase_a_min_end
        self.phase_a_max_end = self.schedule.phase_a_max_end
        self.phase_b_nominal_start = self.schedule.phase_b_nominal_start
        self.phase_c_nominal_start = self.schedule.phase_c_nominal_start
        self.phase = CurriculumPhase.A.value
        self.t_A = -1
        self.phase_a_end_step = -1
        self.phase_a_gate_passed = False
        self.protocol_version = resolve_gate_protocol_version(self.cfg)
        self.protocol = build_gate_protocol(self.cfg)
        self.active_families = gate_family_names(self.cfg)
        self.training_terminal_step = int(
            getattr(self.cfg, "total_timesteps", None) or self.nominal_steps
        )
        self.next_gate_step = self.phase_a_min_end
        self.last_gate_step_run = -1
        self.gate_check_history: list[dict[str, Any]] = []
        self.protected_candidate_checkpoints: list[str] = []
        self.best_candidate_report: Optional[dict[str, Any]] = None
        self._gate_eval_policy: CustomPPOInferencePolicy | None = None

    def _make_eval_context(self, step: int | None = None) -> GateContext:
        eval_model = getattr(self.trainer, "model", None)
        if eval_model is None:
            eval_model = nn.Module()
        return GateContext(
            trainer=self.trainer,
            cfg=self.cfg,
            step=int(step if step is not None else self.trainer.global_step),
            eval_model=eval_model,
            eval_policy=self._gate_eval_policy,
            latent_k=int(self.trainer.latent_k),
        )

    def _gate_eval_policy_wrapper(self) -> CustomPPOInferencePolicy:
        if self._gate_eval_policy is None:
            from dataclasses import asdict

            cfg_payload = (
                asdict(self.cfg)
                if hasattr(self.cfg, "__dataclass_fields__")
                else dict(vars(self.cfg))
            )
            self._gate_eval_policy = CustomPPOInferencePolicy(
                eval_model,
                device=getattr(self.trainer, "device", torch.device("cpu")),
                cfg=cfg_payload,
            )
        return self._gate_eval_policy

    def _gate_eval_configure_fixed_z(self, z_id: int) -> CustomPPOInferencePolicy:
        return self._make_eval_context().configure_fixed_z(z_id)

    def _gate_eval_predict(self, obs: dict[str, Any]) -> Any:
        return self._make_eval_context().predict(obs)

    def _evaluate_coverage_gate(self) -> GateFamilyResult:
        return _to_family_result(evaluate_coverage(self._make_eval_context()))

    def _evaluate_competence_gate(self) -> GateFamilyResult:
        return _to_family_result(evaluate_competence(self._make_eval_context()))

    def _evaluate_intervention_gate(self) -> GateFamilyResult:
        return _to_family_result(v6i1_intervention(self._make_eval_context()))

    def _evaluate_actor_intervention_gate(self) -> GateFamilyResult:
        return _to_family_result(v6i2_actor_intervention(self._make_eval_context()))

    def _evaluate_training_integrity_gate(self) -> GateFamilyResult:
        return _to_family_result(evaluate_training_integrity(self._make_eval_context()))

    def _evaluate_online_gates(
        self,
        gate_results: dict[str, GateFamilyResult],
        context: GateContext | None = None,
    ) -> dict[str, Any]:
        ctx = context or self._make_eval_context()
        online_raw = self.protocol.evaluate_online(ctx)
        for name, result in online_raw.items():
            gate_results[name] = _to_family_result(result)
        return _build_online_report(online_raw)

    def _evaluate_behavioral_realization_gate(
        self,
        context: GateContext | None = None,
        *,
        deadline_monotonic: float | None = None,
        progress_path: str | None = None,
        progress_interval_seconds: float = 60.0,
    ) -> GateFamilyResult:
        ctx = context or self._make_eval_context()
        evaluator = getattr(self.protocol, "_evaluate_behavioral_realization", None)
        if callable(evaluator):
            return _to_family_result(
                evaluator(
                    ctx,
                    deadline_monotonic=deadline_monotonic,
                    progress_path=progress_path,
                    progress_interval_seconds=progress_interval_seconds,
                )
            )
        boundary_raw = self.protocol.evaluate_boundary(ctx)
        return _to_family_result(boundary_raw["behavioral_realization"])

    def _run_matched_seed_eval(self, context: GateContext | None = None) -> GateFamilyResult:
        ctx = context or self._make_eval_context()
        return _to_family_result(evaluate_matched_seed_behavior(ctx))

    def _run_learnability_probe(self, context: GateContext | None = None) -> GateFamilyResult:
        ctx = context or self._make_eval_context()
        return _to_family_result(run_learnability_probe(ctx))

    def _phase_a_progress_path(self, step: int) -> str:
        report_dir = os.path.join(self.cfg.checkpoint_dir, "phase_a_gate_reports")
        return os.path.join(report_dir, f"gate_step_{int(step)}_progress.json")

    def _online_prerequisites_failure(
        self,
        gate_results: dict[str, GateFamilyResult],
        online_report: dict[str, Any],
    ) -> dict[str, Any] | None:
        required = ("coverage", "competence", "training_integrity")
        if not is_v6i2_dual_evidence_protocol(self.cfg):
            required = required + ("counterfactual_intervention",)

        failed = {}
        for name in required:
            result = gate_results.get(name)
            if result is None:
                failed[name] = GATE_STATUS_NOT_RUN
            elif result.status != "PASS":
                failed[name] = result.status
        if failed:
            return {"failed_online_gate_statuses": failed}

        return None

    def _mark_actor_intervention_gate_skipped(self, step: int) -> None:
        state = getattr(self.trainer, "latent_state", None)
        if state is None:
            return
        marker = getattr(state, "mark_actor_intervention_gate_skipped", None)
        if callable(marker):
            marker(int(step))
            return
        last_skip = int(getattr(state, "actor_intervention_last_skipped_gate_step", -1) or -1)
        if last_skip == int(step):
            return
        state.actor_intervention_skipped_gate_count = int(
            getattr(state, "actor_intervention_skipped_gate_count", 0) or 0
        ) + 1
        state.actor_intervention_last_skipped_gate_step = int(step)

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": _STATE_SCHEMA_VERSION,
            "gate_protocol_version": self.protocol_version,
            "phase": str(self.phase),
            "t_A": int(self.t_A),
            "phase_a_end_step": int(self.phase_a_end_step),
            "phase_a_gate_passed": bool(self.phase_a_gate_passed),
            "next_gate_step": int(self.next_gate_step),
            "last_gate_step_run": int(self.last_gate_step_run),
            "gate_check_history": list(self.gate_check_history),
            "protected_candidate_checkpoints": list(self.protected_candidate_checkpoints),
            "best_candidate_report": self.best_candidate_report,
            "training_terminal_step": int(self.training_terminal_step),
        }

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        self.phase = str(payload.get("phase", self.phase))
        self.t_A = int(payload.get("t_A", self.t_A))
        self.phase_a_end_step = int(payload.get("phase_a_end_step", self.phase_a_end_step))
        self.phase_a_gate_passed = bool(payload.get("phase_a_gate_passed", self.phase_a_gate_passed))
        self.next_gate_step = int(payload.get("next_gate_step", self.next_gate_step))
        self.last_gate_step_run = int(payload.get("last_gate_step_run", self.last_gate_step_run))
        self.gate_check_history = list(payload.get("gate_check_history", []))
        self.protected_candidate_checkpoints = list(
            payload.get("protected_candidate_checkpoints", [])
        )
        self.best_candidate_report = payload.get("best_candidate_report", self.best_candidate_report)
        self.training_terminal_step = int(
            payload.get("training_terminal_step", getattr(self, "training_terminal_step", self.nominal_steps))
        )

    def effective_training_terminal_step(self) -> int:
        return int(self.training_terminal_step)

    def _extend_terminal_on_phase_b_entry(self, step: int) -> None:
        before = int(self.training_terminal_step)
        if not bool(getattr(self.cfg, "curriculum_extend_terminal_on_late_promotion", True)):
            return
        extended = self.schedule.terminal_step_if_promoted_at(step)
        if extended > self.training_terminal_step:
            self.training_terminal_step = extended
            self.cfg.total_timesteps = extended
            from rl.custom_ppo.curriculum.schedule import format_terminal_extension_banner

            for line in format_terminal_extension_banner(
                self.schedule,
                phase_a_end_step=step,
                effective_terminal=self.training_terminal_step,
            ):
                print(line, flush=True)
            if before != extended:
                print(
                    f"[Curriculum Controller] Training terminal raised from {before:,} to {extended:,}",
                    flush=True,
                )

    def _phase_b_end_step(self) -> int:
        if self.t_A >= 0:
            return self.t_A + self.schedule.phase_b_budget_steps
        return self.phase_c_nominal_start

    def _phase_c_end_step(self) -> int:
        if self.t_A >= 0:
            return self.schedule.terminal_step_if_promoted_at(self.t_A)
        return self.nominal_steps

    def resolve_phase(self, global_step: int | None = None) -> str:
        return self.phase

    def maybe_apply_phase_transitions(self) -> bool:
        return self.maybe_apply_nominal_phase_transition()

    def maybe_apply_nominal_phase_transition(self) -> bool:
        """Apply schedule-driven phase transitions (A→B observe-only; B→C all modes)."""
        mode = GateMode.normalize(str(getattr(self.cfg, "phase_boundary_gate_mode", "enforce")))
        step = int(self.trainer.global_step)
        transitioned = False
        if mode == GateMode.OBSERVE_ONLY.value and self.phase == CurriculumPhase.A.value:
            if step >= self.phase_b_nominal_start:
                self._transition_to_phase_b(step, nominal=True)
                transitioned = True
        if self.phase == CurriculumPhase.B.value:
            phase_b_end = self._phase_b_end_step()
            if step >= phase_b_end or step >= self.phase_c_nominal_start:
                self._transition_to_phase_c(step, nominal=(mode == GateMode.OBSERVE_ONLY.value))
                transitioned = True
        return transitioned

    def should_run_phase_a_gate(self, global_step: int | None = None) -> bool:
        step = int(global_step if global_step is not None else self.trainer.global_step)
        return should_run_phase_a_gate(
            schedule=self.schedule,
            phase=self.phase,
            global_step=step,
            last_gate_step_run=self.last_gate_step_run,
            next_gate_step=self.next_gate_step,
        )

    def check_and_run_gate(self) -> bool:
        """Run a Phase A gate attempt when scheduled. Returns True if promoted to Phase B."""
        step = int(self.trainer.global_step)
        if not self.should_run_phase_a_gate(step):
            return False

        print(f"\n[Curriculum Controller] Phase A gate check at step {step}...")
        promoted = False
        with GateIsolationBoundary(self.trainer) as boundary:
            context = GateContext(
                trainer=self.trainer,
                cfg=self.cfg,
                step=step,
                eval_model=boundary.eval_model,
                eval_policy=boundary.policy(),
                latent_k=int(self.trainer.latent_k),
            )

            candidate_ckpt = os.path.join(self.cfg.checkpoint_dir, f"ckpt_candidate_{step}.zip")
            candidate_saved = False

            gate_results: dict[str, GateFamilyResult] = {}
            gate_started = time.monotonic()
            max_seconds = float(getattr(self.cfg, "phase_a_gate_max_seconds", 900) or 0.0)
            deadline = gate_started + max_seconds if max_seconds > 0.0 else None
            progress_interval = float(
                getattr(self.cfg, "phase_a_gate_progress_interval_seconds", 60) or 60
            )
            progress_path = self._phase_a_progress_path(step)
            online_report = self._evaluate_online_gates(gate_results, context)
            from rl.forced_z_behavior_vectors import phase_a_stats_snapshot

            last_stats = dict(getattr(self.trainer, "last_stats", {}) or {})
            online_report.update(phase_a_stats_snapshot(last_stats, gate_step=step))

            families = self.active_families
            mode = GateMode.normalize(str(getattr(self.cfg, "phase_boundary_gate_mode", "enforce")))
            boundary_enabled = bool(getattr(self.cfg, "curriculum_gate_run_boundary_eval", False))
            probe_enabled = bool(getattr(self.cfg, "curriculum_gate_run_probe", False))
            selector_blocks_phase_a = bool(
                getattr(self.cfg, "curriculum_gate_selector_blocks_phase_a", False)
            )
            matched_family = (
                "behavioral_realization"
                if is_v6i2_dual_evidence_protocol(self.cfg)
                else "matched_seed_behavior"
            )
            if selector_blocks_phase_a and probe_enabled:
                families = tuple(dict.fromkeys((*families, "selector_learnability_probe")))
            matched_result = _not_run("not_evaluated")
            probe_result = _not_run("selector_probe_phase_a_diagnostic_only")
            prereq_failure = (
                self._online_prerequisites_failure(gate_results, online_report)
                if boundary_enabled
                else None
            )
            if not boundary_enabled:
                matched_result = _not_run("curriculum_gate_run_boundary_eval=false")
                probe_result = _not_run(
                    "curriculum_gate_run_probe=false"
                    if not probe_enabled
                    else "selector_probe_phase_a_diagnostic_only"
                )
            elif prereq_failure is not None:
                self._mark_actor_intervention_gate_skipped(step)
                matched_result = _not_run(
                    "online_prerequisites_failed",
                    {
                        **prereq_failure,
                        "behavior_evidence_status": "paused_prerequisites_failed",
                        "resume_training": True,
                    },
                )
                probe_result = _not_run("skipped_expensive_gate_prerequisites_failed", prereq_failure)
                atomic_write_json(
                    progress_path,
                    {
                        "schema_version": SCHEMA_VERSION,
                        "gate_status": "skipped_prerequisites_failed",
                        "global_step": step,
                        "promotion": False,
                        "resume_training": True,
                        "elapsed_seconds": round(time.monotonic() - gate_started, 3),
                        **prereq_failure,
                    },
                )
            elif deadline is not None and time.monotonic() >= deadline:
                matched_result = _failed(
                    "inconclusive_timeout",
                    {
                        "timed_out": True,
                        "phase_a_gate_max_seconds": max_seconds,
                        "resume_training": True,
                    },
                )
                atomic_write_json(
                    progress_path,
                    {
                        "schema_version": SCHEMA_VERSION,
                        "gate_status": "inconclusive_timeout",
                        "global_step": step,
                        "promotion": False,
                        "resume_training": True,
                        "elapsed_seconds": round(time.monotonic() - gate_started, 3),
                    },
                )
            else:
                self.trainer.save(candidate_ckpt)
                candidate_saved = True
                self.protected_candidate_checkpoints.append(candidate_ckpt)
                print(f"[Curriculum Controller] Protected candidate: {candidate_ckpt}")

            if (
                boundary_enabled
                and prereq_failure is None
                and matched_result.reason == "not_evaluated"
                and is_v6i2_dual_evidence_protocol(self.cfg)
            ):
                matched_result = self._evaluate_behavioral_realization_gate(
                    context,
                    deadline_monotonic=deadline,
                    progress_path=progress_path,
                    progress_interval_seconds=progress_interval,
                )
            elif (
                boundary_enabled
                and prereq_failure is None
                and matched_result.reason == "not_evaluated"
            ):
                matched_result = self._run_matched_seed_eval(context)
            gate_results[matched_family] = matched_result
            matched_seed_behavioral_telemetry = (
                phase_a_matched_seed_behavioral_telemetry_from_gate_details(
                    {
                        "behavioral_realization_gate_status": matched_result.status,
                        **dict(matched_result.details or {}),
                    }
                )
                if matched_family == "behavioral_realization"
                else {}
            )
            if matched_seed_behavioral_telemetry:
                online_report.update(matched_seed_behavioral_telemetry)
                self.trainer.last_stats = {
                    **dict(getattr(self.trainer, "last_stats", {}) or {}),
                    **matched_seed_behavioral_telemetry,
                }

            if boundary_enabled and prereq_failure is None:
                if selector_blocks_phase_a and probe_enabled:
                    if deadline is not None and time.monotonic() >= deadline:
                        probe_result = _not_run(
                            "skipped_selector_probe_gate_budget_expired",
                            {
                                "timed_out": True,
                                "phase_a_gate_max_seconds": max_seconds,
                                "resume_training": True,
                            },
                        )
                    else:
                        probe_result = self._run_learnability_probe(context)
                elif probe_enabled:
                    probe_result = _not_run(
                        "selector_probe_phase_a_diagnostic_only",
                        {"curriculum_gate_selector_blocks_phase_a": False},
                    )
                else:
                    probe_result = _not_run("curriculum_gate_run_probe=false")
            gate_results["selector_learnability_probe"] = probe_result

            gate_passed = all_required_families_passed(gate_results, families=families)
            promotion_disabled = bool(getattr(self.cfg, "phase_a_disable_promotion", False))
            promotion_allowed = mode == GateMode.ENFORCE.value and not promotion_disabled
            should_promote = promotion_allowed and gate_passed
            if should_promote and not boundary_enabled:
                raise RuntimeError(
                    "Internal error: Phase A promotion requested without boundary evaluation enabled."
                )
            overall_passed = gate_passed

            ranking_components = self.protocol.build_ranking(
                gate_results=gate_results,
                online_report=online_report,
                matched_report=matched_result.details,
                probe_report=probe_result.details,
                global_step=step,
                cfg=self.cfg,
            )

            attempt = GateAttempt(
                global_step=step,
                phase=self.phase,
                checkpoint=candidate_ckpt if candidate_saved else "",
                gate_protocol_version=self.protocol_version,
                required_families=families,
                mode=mode,
                boundary_enabled=boundary_enabled,
                probe_enabled=probe_enabled,
                gate_results=gate_results,
                online_report=online_report,
                matched_report=matched_result.details,
                probe_report=probe_result.details,
                ranking_components=ranking_components,
                gate_passed=gate_passed,
                promotion_allowed=promotion_allowed,
                overall_gate_passed=overall_passed,
            )
            report = build_final_gate_report(attempt)
            from rl.custom_ppo.gate_protocol import (
                gate_config_fingerprint,
                gate_lineage_audit_fields,
                resolved_gate_config_dict,
            )

            report["gate_config_fingerprint"] = gate_config_fingerprint(self.cfg)
            report["resolved_gate_config"] = resolved_gate_config_dict(self.cfg)
            report.update(gate_lineage_audit_fields(self.cfg))
            report["promotion_disabled"] = promotion_disabled
            ranked_row = dict(report)
            self.gate_check_history.append(ranked_row)
            report_path = write_gate_report(self.cfg, step, report)

            ranked = rank_candidates_lexicographic(list(self.gate_check_history))
            if ranked:
                self.best_candidate_report = ranked[0]

            boundary.assert_unchanged()

            from rl.custom_ppo.v6i1_phase_runtime import (
                is_v6i1_staged_trainer,
                resolve_v6i1_cf_coef_current,
            )

            cf_coef = (
                float(resolve_v6i1_cf_coef_current(self.trainer))
                if is_v6i1_staged_trainer(self.trainer)
                else 0.0
            )
            req_consec = (
                int(self.cfg.actor_jsd_consecutive_updates)
                if is_v6i2_dual_evidence_protocol(self.cfg)
                else int(self.cfg.latent_cf_gate_consecutive_updates)
            )
            gate_families_report = {
                name: gate_results[name].to_dict() for name in families if name in gate_results
            }
            print(
                format_v6i1_gate_stdout_block(
                    step=step,
                    phase=self.phase,
                    overall_passed=overall_passed,
                    mode=mode,
                    gate_results=gate_results,
                    online_report=online_report,
                    ranking_components=ranking_components,
                    cf_coef=cf_coef,
                    required_consecutive=req_consec,
                    report_path=report_path,
                    gate_protocol=self.protocol_version,
                ),
                flush=True,
            )
            print(f"[Curriculum Controller] Gate families: {gate_families_report}")
            print(f"[Curriculum Controller] Overall: {'PASS' if overall_passed else 'FAIL'} (mode={mode})")

            if should_promote:
                self._transition_to_phase_b(step, nominal=False, gate_report=report)
                report["promoted_to_phase_b"] = True
                write_gate_report(self.cfg, step, report)
                promoted = True
            else:
                keep_failed = bool(
                    getattr(self.cfg, "curriculum_keep_failed_gate_candidates", False)
                )
                if candidate_saved and not keep_failed:
                    removed = _remove_file_if_exists(candidate_ckpt)
                    if candidate_ckpt in self.protected_candidate_checkpoints:
                        self.protected_candidate_checkpoints.remove(candidate_ckpt)
                    report["candidate_checkpoint_removed"] = bool(removed)
                    report["checkpoint"] = ""
                    write_gate_report(self.cfg, step, report)
                print("[Curriculum Controller] Phase A continues.")

        self.last_gate_step_run = step
        self.next_gate_step = schedule_next_gate_step(self.schedule, step=step)
        return promoted

    def _transition_to_phase_b(
        self,
        step: int,
        *,
        nominal: bool,
        gate_report: dict[str, Any] | None = None,
    ) -> None:
        if self.phase != CurriculumPhase.A.value:
            return
        self.phase = CurriculumPhase.B.value
        self.t_A = step
        self.phase_a_end_step = step
        self.phase_a_gate_passed = not nominal
        self._extend_terminal_on_phase_b_entry(step)
        boundary_ckpt = os.path.join(self.cfg.checkpoint_dir, f"ckpt_phase_a_boundary_{step}.zip")
        self.trainer.save(boundary_ckpt)
        self.protected_candidate_checkpoints.append(boundary_ckpt)
        label = "NOMINAL" if nominal else "GATE-PASSED"
        print(f"[Curriculum Controller] {label} transition to Phase B at step {step}")
        if gate_report is not None:
            gate_report["phase_a_gate_passed"] = True
            gate_report["phase_a_end_step"] = step

    def _transition_to_phase_c(self, step: int, *, nominal: bool) -> None:
        if self.phase != CurriculumPhase.B.value:
            return
        self.phase = CurriculumPhase.C.value
        label = "NOMINAL" if nominal else "SCHEDULED"
        print(f"[Curriculum Controller] {label} transition to Phase C at step {step}")

    def check_terminal_failure(self) -> None:
        mode = GateMode.normalize(str(getattr(self.cfg, "phase_boundary_gate_mode", "enforce")))
        step = int(self.trainer.global_step)
        if not should_trigger_terminal_failure(
            mode=mode,
            phase=self.phase,
            global_step=step,
            phase_a_max_end=self.phase_a_max_end,
            last_gate_step_run=self.last_gate_step_run,
            phase_a_gate_passed=self.phase_a_gate_passed,
        ):
            return
        self._handle_terminal_failure()

    def _handle_terminal_failure(self) -> None:
        print("\n[Curriculum Controller] phase_a_research_gate_failed")
        ranked = rank_candidates_lexicographic(list(self.gate_check_history))
        best_ckpt = ranked[0]["checkpoint"] if ranked else None

        terminal_ckpt = os.path.join(self.cfg.checkpoint_dir, "ckpt_phase_a_research_gate_failed.zip")
        self.trainer.save(terminal_ckpt)
        self.protected_candidate_checkpoints.append(terminal_ckpt)

        failure_report = {
            "schema_version": SCHEMA_VERSION,
            "status": "phase_a_research_gate_failed",
            "nominal_timesteps": self.nominal_steps,
            "actual_timesteps": int(self.trainer.global_step),
            "protected_candidate_checkpoints": list(self.protected_candidate_checkpoints),
            "gate_check_history": self.gate_check_history,
            "lexicographic_ranking": ranked,
            "best_candidate_checkpoint": best_ckpt,
            "opponent_id_map": dict(_OPPONENT_ID_TO_TAG),
        }
        failure_path = os.path.join(self.cfg.checkpoint_dir, "phase_a_failure_report.json")
        atomic_write_json(failure_path, failure_report)
        print(f"[Curriculum Controller] Failure report: {failure_path}")

        if best_ckpt and os.path.exists(best_ckpt):
            self.trainer.load(best_ckpt)

        raise RuntimeError("phase_a_research_gate_failed")


__all__ = [
    "V6I1CurriculumController",
    "is_staged_v6i1_curriculum",
    "validate_v6i1_enforce_config",
]
