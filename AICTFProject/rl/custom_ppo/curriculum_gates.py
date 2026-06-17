"""V6I1 staged-curriculum Phase A gate controller.

Owns phase resolution, Phase A gate scheduling, protected candidate checkpoints,
transition decisions, boundary evaluation, offline selector-learnability probes,
and machine-readable gate reports.

Gate evaluation must not mutate production model parameters, optimizer state,
or training RNG streams.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import random
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any, Iterator, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rl.config.ppo_config import PPOConfig
from rl.custom_ppo.csv_writers import _OPPONENT_ID_TO_TAG
from rl.custom_ppo.inference import CustomPPOInferencePolicy

# Fixed pair index for forced_z_pair_jsd_{i}: (0,1)=0 … (2,3)=5
LATENT_PAIR_INDEX: tuple[tuple[int, int], ...] = (
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 2),
    (1, 3),
    (2, 3),
)
PAIR_ORDER = LATENT_PAIR_INDEX

GATE_STATUS_PASS = "PASS"
GATE_STATUS_FAIL = "FAIL"
GATE_STATUS_NOT_RUN = "NOT_RUN"
GATE_STATUS_ERROR = "ERROR"

GATE_FAMILY_NAMES: tuple[str, ...] = (
    "coverage",
    "competence",
    "counterfactual_intervention",
    "training_integrity",
    "matched_seed_behavior",
    "selector_learnability_probe",
)


def is_staged_v6i1_curriculum(cfg: PPOConfig) -> bool:
    """True only for the staged team-intent V6I1 curriculum, not repertoire ablations."""
    return (
        bool(getattr(cfg, "use_v6i1_curriculum", False))
        and str(getattr(cfg, "training_mode", "default")) == "staged_team_intent_curriculum"
        and str(getattr(cfg, "experiment_family", "v6")) == "v6"
        and str(getattr(cfg, "experiment_id", "v6i1")) == "v6i1"
    )


def validate_v6i1_enforce_config(cfg: PPOConfig) -> None:
    """Fail fast when enforce mode cannot run the full six-family Phase A gate."""
    mode = str(getattr(cfg, "phase_boundary_gate_mode", "enforce")).lower()
    if mode != "enforce":
        return
    if not bool(getattr(cfg, "curriculum_gate_run_boundary_eval", False)):
        raise ValueError("V6I1 enforce mode requires matched-seed boundary evaluation.")
    if not bool(getattr(cfg, "curriculum_gate_run_probe", False)):
        raise ValueError("V6I1 enforce mode requires the selector-learnability probe.")


@contextmanager
def _preserve_model_training_mode(model: nn.Module) -> Iterator[None]:
    was_training = bool(model.training)
    try:
        yield
    finally:
        model.train(was_training)


class LearnabilityClassifier(nn.Module):
    """Temporary offline classifier; not part of production q_phi."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, num_classes: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _state_bytes(state: dict[str, Any]) -> bytes:
    buf = io.BytesIO()
    torch.save(state, buf)
    return buf.getvalue()


def _hash_module_params(module: nn.Module | None) -> str:
    if module is None:
        return ""
    hasher = hashlib.md5()
    for tensor in module.state_dict().values():
        hasher.update(tensor.detach().cpu().numpy().tobytes())
    return hasher.hexdigest()


@dataclass
class GateFamilyResult:
    """Structured gate-family outcome for reports and promotion logic."""

    status: str
    reason: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"status": self.status}
        if self.reason:
            out["reason"] = self.reason
        if self.details:
            out.update(self.details)
        return out

    @property
    def passed(self) -> bool:
        return self.status == GATE_STATUS_PASS

    @property
    def measured(self) -> bool:
        return self.status in (GATE_STATUS_PASS, GATE_STATUS_FAIL)


def gate_family_result_from_bool(passed: bool, *, details: dict[str, Any] | None = None) -> GateFamilyResult:
    return GateFamilyResult(
        status=GATE_STATUS_PASS if passed else GATE_STATUS_FAIL,
        details=dict(details or {}),
    )


def count_gate_families_passed(gate_results: dict[str, GateFamilyResult]) -> int:
    return sum(1 for name in GATE_FAMILY_NAMES if gate_results.get(name, GateFamilyResult(GATE_STATUS_NOT_RUN)).passed)


def count_gate_families_measured(gate_results: dict[str, GateFamilyResult]) -> int:
    return sum(1 for name in GATE_FAMILY_NAMES if gate_results.get(name, GateFamilyResult(GATE_STATUS_NOT_RUN)).measured)


def overall_gate_passed_for_promotion(
    gate_results: dict[str, GateFamilyResult],
    *,
    mode: str,
) -> bool:
    """In enforce mode every family must be PASS; NOT_RUN blocks promotion."""
    if str(mode).lower() != "enforce":
        return False
    for name in GATE_FAMILY_NAMES:
        result = gate_results.get(name, GateFamilyResult(GATE_STATUS_NOT_RUN))
        if result.status != GATE_STATUS_PASS:
            return False
    return True


@dataclass
class TrainingIsolationSnapshot:
    """Captured production state before an isolated gate evaluation."""

    actor_hash: str = ""
    critic_hash: str = ""
    router_hash: str = ""
    actor_opt_hash: str = ""
    critic_opt_hash: str = ""
    router_opt_hash: str = ""
    py_rng_state: Any = None
    np_rng_state: Any = None
    torch_rng_state: Any = None
    torch_cuda_rng_states: list[Any] = field(default_factory=list)
    model_was_training: bool = True
    global_step: int = 0

    @classmethod
    def capture(cls, trainer: Any) -> TrainingIsolationSnapshot:
        snap = cls()
        model = trainer.model
        snap.model_was_training = bool(model.training)
        snap.global_step = int(getattr(trainer, "global_step", 0))
        snap.actor_hash = _hash_module_params(getattr(model, "actor", None))
        snap.critic_hash = _hash_module_params(getattr(model, "critic", None))
        router_modules = [
            getattr(model, name, None)
            for name in ("strategy_encoder", "episode_strategy_value_head", "phase_predictor", "strategy_aux_return_head")
        ]
        router_hasher = hashlib.md5()
        for mod in router_modules:
            router_hasher.update(_hash_module_params(mod).encode("utf-8"))
        snap.router_hash = router_hasher.hexdigest()

        snap.actor_opt_hash = _hash_module_params_hash(trainer.optimizer)
        snap.critic_opt_hash = snap.actor_opt_hash
        router_opt = getattr(trainer, "router_optimizer", None) or getattr(trainer, "latent_router_optimizer", None)
        snap.router_opt_hash = _hash_module_params_hash(router_opt)

        snap.py_rng_state = random.getstate()
        snap.np_rng_state = np.random.get_state()
        snap.torch_rng_state = torch.get_rng_state()
        if torch.cuda.is_available():
            snap.torch_cuda_rng_states = torch.cuda.get_rng_state_all()
        return snap

    def assert_unchanged(self, trainer: Any) -> None:
        after = TrainingIsolationSnapshot.capture(trainer)
        assert self.actor_hash == after.actor_hash, "actor parameters mutated during gate evaluation"
        assert self.critic_hash == after.critic_hash, "critic parameters mutated during gate evaluation"
        assert self.router_hash == after.router_hash, "router parameters mutated during gate evaluation"
        assert self.actor_opt_hash == after.actor_opt_hash, "actor optimizer mutated during gate evaluation"
        assert self.router_opt_hash == after.router_opt_hash, "router optimizer mutated during gate evaluation"
        assert int(getattr(trainer, "global_step", 0)) == self.global_step, "global_step mutated during gate evaluation"
        assert bool(trainer.model.training) == self.model_was_training, "model.training mode mutated during gate evaluation"

    def restore_rng(self) -> None:
        if self.py_rng_state is not None:
            random.setstate(self.py_rng_state)
        if self.np_rng_state is not None:
            np.random.set_state(self.np_rng_state)
        if self.torch_rng_state is not None:
            torch.set_rng_state(self.torch_rng_state)
        if self.torch_cuda_rng_states and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(self.torch_cuda_rng_states)


def _hash_module_params_hash(optimizer: Any) -> str:
    if optimizer is None:
        return ""
    hasher = hashlib.md5()
    hasher.update(_state_bytes(optimizer.state_dict()))
    return hasher.hexdigest()


def rank_candidates_lexicographic(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Lexicographic best-candidate ranking after Phase A failure."""

    def _key(row: dict[str, Any]) -> tuple[Any, ...]:
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

    ranked = sorted(candidates, key=_key)
    for rank, row in enumerate(ranked):
        row["lexicographic_rank"] = rank
    return ranked


def build_lexicographic_ranking_components(
    *,
    gate_results: dict[str, GateFamilyResult],
    online_report: dict[str, Any],
    matched_report: dict[str, Any],
    probe_report: dict[str, Any],
    global_step: int,
) -> dict[str, Any]:
    comp_scores = online_report.get("competence_scores", [0.0, 0.0, 0.0, 0.0])
    pair_jsd = online_report.get("pair_jsd_ema", [0.0] * 6)
    margin = float(online_report.get("jsd_margin", 0.01))
    occupancy = online_report.get("occupancy", [0.25, 0.25, 0.25, 0.25])
    pairs_above = sum(1 for v in pair_jsd if float(v) >= margin)
    weakest_norm = 0.0
    if pair_jsd and margin > 0:
        weakest_norm = max(min(float(v) / margin for v in pair_jsd), 0.0)
        weakest_norm = min(weakest_norm, 1.0)

    effect_sizes = [
        float(v.get("effect_size", 0.0))
        for v in matched_report.get("opponents", {}).values()
        if isinstance(v, dict)
    ]
    matched_effect = max(effect_sizes) if effect_sizes else 0.0

    fixed_regret = float(probe_report.get("fixed_regret", 0.0))
    probe_regret = float(probe_report.get("probe_regret", 0.0))
    regret_reduction = fixed_regret - probe_regret

    occ = np.asarray(occupancy, dtype=np.float64)
    occ_imbalance = float(occ.max() - occ.min()) if occ.size else 1.0

    return {
        "gate_families_passed": count_gate_families_passed(gate_results),
        "gate_families_measured": count_gate_families_measured(gate_results),
        "min_competence": float(np.min(comp_scores)) if len(comp_scores) else 0.0,
        "pairs_above_margin": int(pairs_above),
        "weakest_pair_normalized_separation": float(weakest_norm),
        "matched_seed_effect_size": float(matched_effect),
        "probe_regret_reduction": float(regret_reduction),
        "occupancy_imbalance": float(occ_imbalance),
        "global_step": int(global_step),
    }


_GATE_FAMILY_STDOUT_LABELS: dict[str, str] = {
    "coverage": "cov",
    "competence": "comp",
    "counterfactual_intervention": "interv",
    "training_integrity": "integ",
    "matched_seed_behavior": "match",
    "selector_learnability_probe": "probe",
}


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
) -> str:
    """Compact stdout block for a Phase A gate attempt."""
    def _st(name: str) -> str:
        return gate_results[name].status if name in gate_results else GATE_STATUS_NOT_RUN

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
    report_line = f"[V6I1 Gate] report={report_path}" if report_path else ""
    lines = [
        f"[V6I1 Gate] step={step} phase={phase} mode={mode}",
        (
            f"[V6I1 Gate] coverage={_st('coverage')} competence={_st('competence')} "
            f"integrity={_st('training_integrity')}"
        ),
        (
            f"[V6I1 Gate] intervention={_st('counterfactual_intervention')} "
            f"pairs>=margin={num_above}/6 min_jsd={min_ema:.5f} "
            f"jsd_consec={jsd_consec}/{required_consecutive}"
        ),
        (
            f"[V6I1 Gate] matched_eval={_st('matched_seed_behavior')} "
            f"probe={_st('selector_learnability_probe')}"
        ),
        f"[V6I1 Gate] overall={'PASS' if overall_passed else 'FAIL'} action={action} cf_coef={cf_coef:.4f}",
    ]
    if report_line:
        lines.append(report_line)
    return "\n".join(lines)


class V6I1CurriculumController:
    """Manages V6I1 staged curriculum transitions, boundary evaluations, and diagnostic probes."""

    def __init__(self, trainer: Any):
        self.trainer = trainer
        self.cfg: PPOConfig = trainer.cfg
        validate_v6i1_enforce_config(self.cfg)
        self.phase = "A"
        self.t_A = -1
        self.phase_a_end_step = -1
        self.phase_a_gate_passed = False
        self.nominal_steps = int(self.cfg.curriculum_nominal_timesteps)
        self.phase_a_min_end = int(0.40 * self.nominal_steps)
        self.phase_a_max_end = int(0.55 * self.nominal_steps)
        self.phase_b_nominal_start = self.phase_a_min_end
        self.phase_c_nominal_start = int(0.70 * self.nominal_steps)
        self.next_gate_step = self.phase_a_min_end
        self.last_gate_step_run = -1
        self.gate_check_history: list[dict[str, Any]] = []
        self.protected_candidate_checkpoints: list[str] = []
        self.best_candidate_report: Optional[dict[str, Any]] = None
        self._gate_eval_policy: CustomPPOInferencePolicy | None = None

    def _gate_eval_policy_wrapper(self) -> CustomPPOInferencePolicy:
        if self._gate_eval_policy is None:
            cfg_payload = asdict(self.cfg) if hasattr(self.cfg, "__dataclass_fields__") else dict(vars(self.cfg))
            self._gate_eval_policy = CustomPPOInferencePolicy(
                self.trainer.model,
                device=self.trainer.device,
                cfg=cfg_payload,
            )
        return self._gate_eval_policy

    @staticmethod
    def _unwrap_vec_obs(obs: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value[0] if hasattr(value, "shape") and getattr(value, "ndim", 0) >= 2 else value
            for key, value in obs.items()
        }

    def _gate_eval_configure_fixed_z(self, z_id: int) -> CustomPPOInferencePolicy:
        policy = self._gate_eval_policy_wrapper()
        policy.fixed_latent_strategy = True
        policy.fixed_latent_strategy_id = int(z_id)
        policy.reset_strategy()
        return policy

    def _gate_eval_predict(self, obs: dict[str, Any]) -> np.ndarray:
        policy = self._gate_eval_policy_wrapper()
        act, _ = policy.predict(self._unwrap_vec_obs(obs), deterministic=True)
        return act

    def resolve_phase(self, global_step: int | None = None) -> str:
        return self.phase

    def maybe_apply_phase_transitions(self) -> bool:
        """Apply nominal Phase B→C schedule transitions."""
        return self.maybe_apply_nominal_phase_transition()

    def maybe_apply_nominal_phase_transition(self) -> bool:
        """Apply fixed schedule transitions in observe_only mode."""
        mode = str(getattr(self.cfg, "phase_boundary_gate_mode", "enforce")).lower()
        if mode != "observe_only":
            return False
        step = int(self.trainer.global_step)
        transitioned = False
        if self.phase == "A" and step >= self.phase_b_nominal_start:
            self._transition_to_phase_b(step, nominal=True)
            transitioned = True
        if self.phase == "B":
            phase_b_end = self.t_A + int(0.30 * self.nominal_steps) if self.t_A >= 0 else self.phase_c_nominal_start
            if step >= phase_b_end or step >= self.phase_c_nominal_start:
                self._transition_to_phase_c(step, nominal=True)
                transitioned = True
        return transitioned

    def should_run_phase_a_gate(self, global_step: int | None = None) -> bool:
        step = int(global_step if global_step is not None else self.trainer.global_step)
        if self.phase != "A":
            return False
        if step < self.phase_a_min_end:
            return False
        if step > self.phase_a_max_end and self.last_gate_step_run >= self.phase_a_max_end:
            return False
        if step == self.last_gate_step_run:
            return False
        interval = max(1, int(self.cfg.phase_a_gate_check_interval))
        due_by_schedule = step >= self.next_gate_step
        due_at_final_boundary = step >= self.phase_a_max_end and self.last_gate_step_run < self.phase_a_max_end
        if not (due_by_schedule or due_at_final_boundary):
            return False
        if step > self.phase_a_max_end and not due_at_final_boundary:
            return False
        return True

    def _schedule_next_gate_step(self, step: int) -> None:
        interval = max(1, int(self.cfg.phase_a_gate_check_interval))
        if step >= self.phase_a_max_end:
            self.next_gate_step = self.phase_a_max_end + 1
            return
        candidate = step + interval
        if candidate > self.phase_a_max_end:
            self.next_gate_step = self.phase_a_max_end
        else:
            self.next_gate_step = candidate

    def check_and_run_gate(self) -> bool:
        """Run a Phase A gate attempt when scheduled. Returns True if promoted to Phase B."""
        step = int(self.trainer.global_step)
        if not self.should_run_phase_a_gate(step):
            return False

        print(f"\n[Curriculum Controller] Phase A gate check at step {step}...")
        isolation = TrainingIsolationSnapshot.capture(self.trainer)
        promoted = False
        try:
            candidate_ckpt = os.path.join(self.cfg.checkpoint_dir, f"ckpt_candidate_{step}.zip")
            self.trainer.save(candidate_ckpt)
            self.protected_candidate_checkpoints.append(candidate_ckpt)
            print(f"[Curriculum Controller] Protected candidate: {candidate_ckpt}")

            gate_results: dict[str, GateFamilyResult] = {}
            online_report = self._evaluate_online_gates(gate_results)
            matched_result = self._run_matched_seed_eval()
            gate_results["matched_seed_behavior"] = matched_result
            probe_result = self._run_learnability_probe()
            gate_results["selector_learnability_probe"] = probe_result

            mode = str(getattr(self.cfg, "phase_boundary_gate_mode", "enforce")).lower()
            boundary_enabled = bool(getattr(self.cfg, "curriculum_gate_run_boundary_eval", False))
            probe_enabled = bool(getattr(self.cfg, "curriculum_gate_run_probe", False))
            overall_passed = overall_gate_passed_for_promotion(gate_results, mode=mode)
            should_promote = overall_passed and mode == "enforce"
            if mode == "enforce" and should_promote and not (boundary_enabled and probe_enabled):
                raise RuntimeError(
                    "Internal error: Phase A promotion requested without full heavy gate configuration."
                )

            ranking_components = build_lexicographic_ranking_components(
                gate_results=gate_results,
                online_report=online_report,
                matched_report=matched_result.details,
                probe_report=probe_result.details,
                global_step=step,
            )
            gate_families_report = {name: gate_results[name].to_dict() for name in GATE_FAMILY_NAMES if name in gate_results}
            report = {
                "global_step": step,
                "checkpoint": candidate_ckpt,
                "phase_boundary_gate_mode": mode,
                "curriculum_gate_run_boundary_eval": boundary_enabled,
                "curriculum_gate_run_probe": probe_enabled,
                "gate_families": gate_families_report,
                "overall_gate_passed": overall_passed,
                "promoted_to_phase_b": False,
                "nominal_transition_to_phase_b": False,
                "online_report": online_report,
                "matched_eval_report": matched_result.details,
                "probe_report": probe_result.details,
                "ranking_components": ranking_components,
                "phase_a_gate_passed": False,
                "phase_a_end_step": None,
            }
            ranked_row = dict(report)
            self.gate_check_history.append(ranked_row)
            report_path = self._write_gate_report(step, report)

            ranked = rank_candidates_lexicographic(list(self.gate_check_history))
            if ranked:
                self.best_candidate_report = ranked[0]

            isolation.assert_unchanged(self.trainer)

            from rl.custom_ppo.v6i1_phase_runtime import (
                is_v6i1_staged_trainer,
                resolve_v6i1_cf_coef_current,
            )

            cf_coef = (
                float(resolve_v6i1_cf_coef_current(self.trainer))
                if is_v6i1_staged_trainer(self.trainer)
                else 0.0
            )
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
                    required_consecutive=int(getattr(self.cfg, "latent_cf_gate_consecutive_updates", 5)),
                    report_path=report_path,
                ),
                flush=True,
            )
            print(f"[Curriculum Controller] Gate families: {gate_families_report}")
            print(f"[Curriculum Controller] Overall: {'PASS' if overall_passed else 'FAIL'} (mode={mode})")

            if should_promote:
                self._transition_to_phase_b(step, nominal=False, gate_report=report)
                report["promoted_to_phase_b"] = True
                self._write_gate_report(step, report)
                promoted = True
            else:
                print("[Curriculum Controller] Phase A continues.")
        finally:
            isolation.restore_rng()
            self.trainer.model.train(isolation.model_was_training)

        self.last_gate_step_run = step
        self._schedule_next_gate_step(step)
        return promoted

    def _transition_to_phase_b(
        self,
        step: int,
        *,
        nominal: bool,
        gate_report: dict[str, Any] | None = None,
    ) -> None:
        if self.phase != "A":
            return
        self.phase = "B"
        self.t_A = step
        self.phase_a_end_step = step
        self.phase_a_gate_passed = not nominal
        boundary_ckpt = os.path.join(self.cfg.checkpoint_dir, f"ckpt_phase_a_boundary_{step}.zip")
        self.trainer.save(boundary_ckpt)
        self.protected_candidate_checkpoints.append(boundary_ckpt)
        label = "NOMINAL" if nominal else "GATE-PASSED"
        print(f"[Curriculum Controller] {label} transition to Phase B at step {step}")
        if gate_report is not None:
            gate_report["phase_a_gate_passed"] = True
            gate_report["phase_a_end_step"] = step

    def _transition_to_phase_c(self, step: int, *, nominal: bool) -> None:
        if self.phase != "B":
            return
        self.phase = "C"
        label = "NOMINAL" if nominal else "SCHEDULED"
        print(f"[Curriculum Controller] {label} transition to Phase C at step {step}")

    def check_terminal_failure(self) -> None:
        mode = str(getattr(self.cfg, "phase_boundary_gate_mode", "enforce")).lower()
        if mode == "observe_only":
            return
        step = int(self.trainer.global_step)
        if self.phase == "A" and step >= self.phase_a_max_end and not self.phase_a_gate_passed:
            self._handle_terminal_failure()

    def _evaluate_online_gates(self, gate_results: dict[str, GateFamilyResult]) -> dict[str, Any]:
        coverage_result = self._evaluate_coverage_gate()
        competence_result = self._evaluate_competence_gate()
        intervention_result = self._evaluate_intervention_gate()
        integrity_result = self._evaluate_training_integrity_gate()

        gate_results["coverage"] = coverage_result
        gate_results["competence"] = competence_result
        gate_results["counterfactual_intervention"] = intervention_result
        gate_results["training_integrity"] = integrity_result

        report: dict[str, Any] = {}
        for name, result in (
            ("coverage", coverage_result),
            ("competence", competence_result),
            ("counterfactual_intervention", intervention_result),
            ("training_integrity", integrity_result),
        ):
            report.update(result.details)
            report[f"{name}_status"] = result.status
        return report

    def _evaluate_coverage_gate(self) -> GateFamilyResult:
        latent_state = self.trainer.latent_state
        min_eps = int(self.cfg.latent_cf_min_episodes_per_z)
        ep_counts = latent_state.cf_episode_counts.tolist()
        coverage_passed = all(int(c) >= min_eps for c in ep_counts)

        rolling_occ = [0.0] * int(self.trainer.latent_k)
        if len(latent_state.recent_z_history) > 0:
            hist = list(latent_state.recent_z_history)
            for z in hist:
                rolling_occ[int(z)] += 1.0
            rolling_occ = [c / len(hist) for c in rolling_occ]
        occupancy_passed = all(0.20 <= o <= 0.30 for o in rolling_occ)

        return gate_family_result_from_bool(
            coverage_passed and occupancy_passed,
            details={
                "cf_episode_counts": ep_counts,
                "recent_z_occupancy": rolling_occ,
                "latent_cf_min_episodes_per_z": min_eps,
            },
        )

    def _evaluate_competence_gate(self) -> GateFamilyResult:
        latent_state = self.trainer.latent_state
        comp_scores, competence_ready = latent_state.compute_competence_scores()
        competence_passed = bool(competence_ready) and all(float(s) >= 0.50 for s in comp_scores)
        J = latent_state.cf_J.tolist()
        ret_std = float(np.sqrt(max(1e-8, latent_state.cf_return_var)))
        return gate_family_result_from_bool(
            competence_passed,
            details={
                "cf_competence_ready": bool(competence_ready),
                "competence_scores": [float(s) for s in comp_scores.tolist()],
                "return_ema_z": {f"z{i}": float(J[i]) for i in range(len(J))},
                "competence_z": {f"z{i}": float(comp_scores[i]) for i in range(len(comp_scores))},
                "best_minus_worst_return": float(np.max(J) - np.min(J)) if J else 0.0,
                "return_standard_deviation": ret_std,
            },
        )

    def _evaluate_intervention_gate(self) -> GateFamilyResult:
        latent_state = self.trainer.latent_state
        margin = float(self.cfg.latent_cf_jsd_margin)
        pair_jsd = latent_state.pair_jsd_ema.tolist()
        num_valid = sum(1 for jsd in pair_jsd if float(jsd) >= margin)
        min_jsd = float(min(pair_jsd)) if pair_jsd else 0.0
        required_consecutive = int(getattr(self.cfg, "latent_cf_gate_consecutive_updates", 5))
        update_ok = num_valid >= 5 and min_jsd >= 0.5 * margin
        passed = update_ok and int(latent_state.jsd_gate_consecutive_updates) >= required_consecutive
        pair_details = {
            f"forced_z_pair_jsd_{idx}": float(pair_jsd[idx]) if idx < len(pair_jsd) else 0.0
            for idx, pair in enumerate(LATENT_PAIR_INDEX)
        }
        pair_identity = {
            f"pair_{idx}_z{pair[0]}_z{pair[1]}": pair_details[f"forced_z_pair_jsd_{idx}"]
            for idx, pair in enumerate(LATENT_PAIR_INDEX)
        }
        return gate_family_result_from_bool(
            passed,
            details={
                "pair_jsd_ema": pair_jsd,
                "pair_order": [list(pair) for pair in LATENT_PAIR_INDEX],
                **pair_details,
                **pair_identity,
                "jsd_margin": margin,
                "num_pairs_above_margin": int(num_valid),
                "min_pair_jsd_ema": min_jsd,
                "min_pair_floor": 0.5 * margin,
                "single_update_ok": bool(update_ok),
                "jsd_consecutive_updates": int(latent_state.jsd_gate_consecutive_updates),
                "latent_cf_gate_consecutive_updates": required_consecutive,
            },
        )

    def _evaluate_training_integrity_gate(self) -> GateFamilyResult:
        stats = dict(getattr(self.trainer, "last_stats", {}) or {})
        k = int(self.trainer.latent_k)
        forced_frac = float(stats.get("latent_forced_z_step_fraction", 0.0))
        router_samples = sum(
            float(stats.get(f"router_sample_count_by_z_{z}", 0.0) or 0.0) for z in range(k)
        )
        switch_count = float(stats.get("strategy_switch_count", 0.0))
        qphi_grad = float(
            stats.get(
                "q_phi_grad_norm",
                stats.get("main_loop_q_phi_grad_norm", stats.get("strategy_grad_norm", 0.0)),
            )
            or 0.0
        )
        router_opt_steps = float(getattr(self.trainer.latent_state, "router_optimizer_step_count", 0.0))

        passed = (
            abs(forced_frac - 1.0) < 1e-5
            and router_samples == 0.0
            and router_opt_steps == 0.0
            and qphi_grad < 1e-7
            and switch_count == 0.0
        )
        return gate_family_result_from_bool(
            passed,
            details={
                "forced_z_fraction": forced_frac,
                "router_sample_count": router_samples,
                "router_optimizer_step_count": router_opt_steps,
                "q_phi_grad_norm": qphi_grad,
                "strategy_switch_count": switch_count,
            },
        )

    def _run_matched_seed_eval(self) -> GateFamilyResult:
        """Boundary matched-seed evaluation (skipped when heavy eval disabled)."""
        if not bool(getattr(self.cfg, "curriculum_gate_run_boundary_eval", False)):
            return GateFamilyResult(
                status=GATE_STATUS_NOT_RUN,
                reason="curriculum_gate_run_boundary_eval=false",
            )

        from rl.training.env_factory import build_training_env

        print("[Curriculum Controller] Matched-seed boundary evaluation...")
        opponents = ["OP5", "OP6", "OP7"]
        seeds = list(range(2000, 2020))
        latent_k = int(self.trainer.latent_k)
        all_passed = True
        op_reports: dict[str, Any] = {}

        eval_cfg = PPOConfig()
        for key, value in self.cfg.__dict__.items():
            setattr(eval_cfg, key, value)
        eval_cfg.n_envs = 1

        with _preserve_model_training_mode(self.trainer.model):
            for opp in opponents:
                env = build_training_env(eval_cfg, initial_phase="PHASE1", initial_opponent_tag=opp)
                route_dists: list[float] = []
                behavior_dists: list[float] = []
                wr_by_z: dict[int, list[float]] = {z: [] for z in range(latent_k)}
                try:
                    for seed in seeds:
                        reset_states: list[dict[str, Any]] = []
                        traj_pos: dict[int, np.ndarray] = {}
                        traj_beh: dict[int, np.ndarray] = {}

                        for z_val in range(latent_k):
                            torch.manual_seed(seed)
                            np.random.seed(seed)
                            if hasattr(env, "seed"):
                                env.seed(seed)
                            obs = env.reset()
                            core = env.core
                            reset_states.append(self._capture_reset_state(core, info={}))

                            self._gate_eval_configure_fixed_z(z_val)

                            history_pos: list[tuple[float, float]] = []
                            history_beh: list[float] = []
                            done = False
                            step_i = 0
                            blue_won = False
                            while not done and step_i < 120:
                                act = self._gate_eval_predict(obs)
                                env.step_async(act)
                                obs, _, done_arr, infos = env.step_wait()
                                done = bool(done_arr[0])
                                info0 = infos[0] if infos else {}
                                bx = float(core.blue_x[0].mean().item())
                                by = float(core.blue_y[0].mean().item())
                                history_pos.append((bx, by))
                                history_beh.append(float(info0.get("dense_reward", 0.0)))
                                step_i += 1
                            er = (infos[0].get("episode_result", infos[0]) if infos else {}) or {}
                            bs = int(er.get("blue_score", core.blue_score[0].item()))
                            rs = int(er.get("red_score", core.red_score[0].item()))
                            blue_won = bs > rs
                            traj_pos[z_val] = np.asarray(history_pos, dtype=np.float64)
                            traj_beh[z_val] = np.asarray(history_beh, dtype=np.float64)
                            wr_by_z[z_val].append(1.0 if blue_won else 0.0)

                        if reset_states and any(s != reset_states[0] for s in reset_states[1:]):
                            print(f"[Curriculum Controller] WARNING: seed {seed} reset mismatch across z branches")

                        for z_a in range(latent_k):
                            for z_b in range(z_a + 1, latent_k):
                                t_len = min(len(traj_pos[z_a]), len(traj_pos[z_b]))
                                if t_len > 0:
                                    diff = traj_pos[z_a][:t_len] - traj_pos[z_b][:t_len]
                                    route_dists.append(float(np.mean(np.linalg.norm(diff, axis=-1))))
                                b_len = min(len(traj_beh[z_a]), len(traj_beh[z_b]))
                                if b_len > 0:
                                    behavior_dists.append(float(np.mean(np.abs(traj_beh[z_a][:b_len] - traj_beh[z_b][:b_len]))))

                    avg_route = float(np.mean(route_dists)) if route_dists else 0.0
                    std_route = float(np.std(route_dists)) if route_dists else 0.0
                    se = std_route / np.sqrt(len(route_dists)) if route_dists else 0.0
                    ci_low = avg_route - 1.96 * se
                    ci_high = avg_route + 1.96 * se
                    wr_spread = 0.0
                    wr_means = [float(np.mean(wr_by_z[z])) if wr_by_z[z] else 0.0 for z in range(latent_k)]
                    if wr_means:
                        wr_spread = float(max(wr_means) - min(wr_means))

                    excludes_zero = ci_low > 0.02 and avg_route > 0.02
                    op_reports[opp] = {
                        "avg_route_distance": avg_route,
                        "avg_behavior_distance": float(np.mean(behavior_dists)) if behavior_dists else 0.0,
                        "ci_95_low": float(ci_low),
                        "ci_95_high": float(ci_high),
                        "paired_ci_excludes_zero": bool(excludes_zero),
                        "forced_z_performance_spread": wr_spread,
                        "effect_size": avg_route,
                    }
                    if not (excludes_zero and wr_spread >= 0.03):
                        all_passed = False
                finally:
                    env.close()

        return gate_family_result_from_bool(
            all_passed,
            details={"opponents": op_reports, "matched_eval_passed": all_passed},
        )

    @staticmethod
    def _capture_reset_state(core: Any, info: dict[str, Any]) -> dict[str, Any]:
        return {
            "blue_score": core.blue_score.detach().cpu().tolist(),
            "red_score": core.red_score.detach().cpu().tolist(),
            "blue_flag_pos": core.blue_flag_pos.detach().cpu().tolist(),
            "red_flag_pos": core.red_flag_pos.detach().cpu().tolist(),
            "blue_x": core.blue_x.detach().cpu().tolist(),
            "blue_y": core.blue_y.detach().cpu().tolist(),
            "red_x": core.red_x.detach().cpu().tolist(),
            "red_y": core.red_y.detach().cpu().tolist(),
            "map_layout": str(getattr(core, "map_layout", "")),
        }

    def _run_learnability_probe(self) -> GateFamilyResult:
        if not bool(getattr(self.cfg, "curriculum_gate_run_probe", False)):
            return GateFamilyResult(
                status=GATE_STATUS_NOT_RUN,
                reason="curriculum_gate_run_probe=false",
            )

        print("[Curriculum Controller] Selector-learnability probe...")
        isolation = TrainingIsolationSnapshot.capture(self.trainer)
        try:
            from rl.training.env_factory import build_training_env

            opponents = ["OP5", "OP6", "OP7"]
            seeds = list(range(3000, 3050))
            latent_k = int(self.trainer.latent_k)
            gs_dim = int(self.trainer.model.global_state_dim)
            tie_margin = float(getattr(self.cfg, "probe_utility_tie_margin", 0.05))

            eval_cfg = PPOConfig()
            for key, value in self.cfg.__dict__.items():
                setattr(eval_cfg, key, value)
            eval_cfg.n_envs = 1

            contexts: list[np.ndarray] = []
            labels: list[int] = []
            all_returns: list[list[float]] = []
            ambiguous = 0

            with _preserve_model_training_mode(self.trainer.model):
                for opp in opponents:
                    env = build_training_env(eval_cfg, initial_phase="PHASE1", initial_opponent_tag=opp)
                    try:
                        for seed in seeds:
                            torch.manual_seed(seed)
                            np.random.seed(seed)
                            if hasattr(env, "seed"):
                                env.seed(seed)
                            obs = env.reset()
                            self._gate_eval_configure_fixed_z(0)
                            bootstrap_actions: list[Any] = []
                            done = False
                            step_i = 0
                            while not done and step_i < 64:
                                bootstrap_actions.append(self._gate_eval_predict(obs))
                                env.step_async(act)
                                obs, _, done_arr, _ = env.step_wait()
                                done = bool(done_arr[0])
                                step_i += 1
                            if done:
                                continue

                            context_h = env.state()[0].copy()
                            z_returns: list[float] = []
                            for z_branch in range(latent_k):
                                torch.manual_seed(seed)
                                np.random.seed(seed)
                                if hasattr(env, "seed"):
                                    env.seed(seed)
                                obs_b = env.reset()
                                for b_act in bootstrap_actions:
                                    env.step_async(b_act)
                                    obs_b, _, _, _ = env.step_wait()
                                self._gate_eval_configure_fixed_z(z_branch)
                                ret_accum = 0.0
                                b_done = False
                                b_step = 0
                                while not b_done and b_step < 64:
                                    act = self._gate_eval_predict(obs_b)
                                    env.step_async(act)
                                    obs_b, rewards, done_arr, _ = env.step_wait()
                                    ret_accum += float(rewards[0])
                                    b_done = bool(done_arr[0])
                                    b_step += 1
                                z_returns.append(ret_accum)

                            order = np.argsort(z_returns)[::-1]
                            if len(order) >= 2 and (z_returns[order[0]] - z_returns[order[1]]) < tie_margin:
                                ambiguous += 1
                                continue
                            contexts.append(context_h)
                            labels.append(int(order[0]))
                            all_returns.append(z_returns)
                    finally:
                        env.close()

            n_examples = len(contexts)
            if n_examples < 10:
                return GateFamilyResult(
                    status=GATE_STATUS_ERROR,
                    reason="insufficient_probe_examples",
                    details={"num_examples": n_examples},
                )

            train_size = max(1, int(0.80 * n_examples))
            X = torch.tensor(np.asarray(contexts, dtype=np.float32))
            y = torch.tensor(labels, dtype=torch.long)
            R = torch.tensor(all_returns, dtype=torch.float32)
            X_train, X_val = X[:train_size], X[train_size:]
            y_train, y_val = y[:train_size], y[train_size:]
            R_val = R[train_size:]

            _, val_accuracy, preds = self._train_probe_classifier(X_train, y_train, X_val, y_val, gs_dim)

            val_oracle = R_val.max(dim=-1)[0]
            oracle_mean = float(val_oracle.mean().item())
            probe_rets = R_val[torch.arange(len(y_val)), preds]
            probe_mean = float(probe_rets.mean().item())
            probe_regret = oracle_mean - probe_mean

            majority = int(torch.bincount(y_train).argmax().item())
            majority_acc = float((y_val == majority).float().mean().item())
            uniform_acc = 1.0 / float(latent_k)

            global_best_z = int(R[:train_size].sum(dim=0).argmax().item())
            fixed_mean = float(R_val[:, global_best_z].mean().item())
            fixed_regret = oracle_mean - fixed_mean

            accuracy_passed = val_accuracy >= (majority_acc + 0.05)
            regret_passed = probe_regret <= (0.90 * fixed_regret)
            gate_passed = accuracy_passed and regret_passed

            isolation.assert_unchanged(self.trainer)

            return gate_family_result_from_bool(
                gate_passed,
                details={
                    "num_examples": n_examples,
                    "ambiguous_context_fraction": float(ambiguous / max(1, len(seeds) * len(opponents))),
                    "uniform_accuracy_baseline": uniform_acc,
                    "majority_accuracy_baseline": majority_acc,
                    "probe_validation_accuracy": val_accuracy,
                    "oracle_return_mean": oracle_mean,
                    "probe_selected_return_mean": probe_mean,
                    "global_best_fixed_z_return_mean": fixed_mean,
                    "probe_regret": probe_regret,
                    "global_best_fixed_z_regret": fixed_regret,
                    "regret_reduction": fixed_regret - probe_regret,
                    "accuracy_passed": accuracy_passed,
                    "regret_passed": regret_passed,
                    "gate_passed": gate_passed,
                },
            )
        finally:
            isolation.restore_rng()

    def _train_probe_classifier(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
        input_dim: int,
    ) -> tuple[nn.Module, float, torch.Tensor]:
        model = LearnabilityClassifier(input_dim)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_train, y_train),
            batch_size=8,
            shuffle=True,
        )
        model.train()
        for _ in range(150):
            for bx, by in loader:
                optimizer.zero_grad()
                loss = criterion(model(bx), by)
                loss.backward()
                optimizer.step()
        model.eval()
        with torch.no_grad():
            preds = torch.argmax(model(X_val), dim=-1)
            accuracy = float((preds == y_val).float().mean().item())
        return model, accuracy, preds

    def _write_gate_report(self, step: int, report: dict[str, Any]) -> str:
        report_dir = os.path.join(self.cfg.checkpoint_dir, "phase_a_gate_reports")
        os.makedirs(report_dir, exist_ok=True)
        path = os.path.join(report_dir, f"gate_step_{step}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"[Curriculum Controller] Gate report: {path}")
        return path

    def _handle_terminal_failure(self) -> None:
        print("\n[Curriculum Controller] phase_a_research_gate_failed")
        ranked = rank_candidates_lexicographic(list(self.gate_check_history))
        best_ckpt = ranked[0]["checkpoint"] if ranked else None

        terminal_ckpt = os.path.join(self.cfg.checkpoint_dir, "ckpt_phase_a_research_gate_failed.zip")
        self.trainer.save(terminal_ckpt)
        self.protected_candidate_checkpoints.append(terminal_ckpt)

        failure_report = {
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
        with open(failure_path, "w", encoding="utf-8") as f:
            json.dump(failure_report, f, indent=2)
        print(f"[Curriculum Controller] Failure report: {failure_path}")

        if best_ckpt and os.path.exists(best_ckpt):
            self.trainer.load(best_ckpt)

        raise RuntimeError("phase_a_research_gate_failed")


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
