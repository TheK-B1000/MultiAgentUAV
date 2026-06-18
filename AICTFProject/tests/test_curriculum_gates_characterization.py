"""Characterization tests for staged curriculum gate scheduling and promotion.

Pins externally visible behavior before the curriculum_gates refactor (Phase 0).
"""

from __future__ import annotations

import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from rl.config.ppo_config import PPOConfig
from rl.custom_ppo.curriculum_gates import (
    GATE_STATUS_ERROR,
    GATE_STATUS_NOT_RUN,
    GATE_STATUS_PASS,
    GateFamilyResult,
    V6I1CurriculumController,
    all_required_families_passed,
    build_lexicographic_ranking_components,
    gate_family_result_from_bool,
    overall_gate_passed_for_promotion,
    rank_candidates_lexicographic,
)
from rl.custom_ppo.gate_protocol import GATE_FAMILY_NAMES_V6I1, GATE_FAMILY_NAMES_V6I2, gate_family_names
from rl.custom_ppo.v6i1_phase_runtime import load_v6i1_curriculum_state, v6i1_curriculum_state_dict
from rl.presets import apply_preset


def _make_controller(
    cfg: PPOConfig | None = None,
    *,
    relax_enforce_for_startup: bool = True,
) -> V6I1CurriculumController:
    cfg = cfg or PPOConfig()
    cfg.checkpoint_dir = tempfile.mkdtemp()
    if not getattr(cfg, "curriculum_nominal_timesteps", None):
        cfg.curriculum_nominal_timesteps = 1_000_000
    cfg.training_mode = "staged_team_intent_curriculum"
    cfg.experiment_family = "v6"
    if not getattr(cfg, "experiment_id", None):
        cfg.experiment_id = "v6i1"
    cfg.use_v6i1_curriculum = True
    if relax_enforce_for_startup and str(
        getattr(cfg, "phase_boundary_gate_mode", "enforce")
    ).lower() == "enforce":
        cfg.phase_boundary_gate_mode = "observe_only"
    trainer = SimpleNamespace(
        cfg=cfg,
        global_step=0,
        latent_k=4,
        latent_state=SimpleNamespace(
            cf_episode_counts=[60] * 4,
            recent_z_history=[0, 1, 2, 3] * 25,
            pair_jsd_ema=[0.02] * 6,
            jsd_gate_consecutive_updates=0,
            pairwise_ema_valid_updates=0,
            cf_J=[10.0] * 4,
            cf_return_var=1.0,
            router_optimizer_step_count=0,
            compute_competence_scores=lambda: (
                __import__("numpy").array([0.55, 0.52, 0.51, 0.50], dtype=__import__("numpy").float32),
                True,
            ),
        ),
        latent_episode_strategy_ppo=False,
        last_stats={},
        save=mock.Mock(),
        model=mock.Mock(training=True, train=mock.Mock()),
    )
    return V6I1CurriculumController(trainer)


class PhaseBoundaryConfigTests(unittest.TestCase):
    def test_fractions_drive_phase_boundaries(self) -> None:
        cfg = PPOConfig()
        cfg.checkpoint_dir = tempfile.mkdtemp()
        cfg.curriculum_nominal_timesteps = 100_000
        cfg.phase_a_earliest_end_fraction = 0.35
        cfg.phase_a_max_end_fraction = 0.60
        cfg.phase_c_start_fraction = 0.75
        ctrl = _make_controller(cfg)
        self.assertEqual(ctrl.phase_a_min_end, 35_000)
        self.assertEqual(ctrl.phase_a_max_end, 60_000)
        self.assertEqual(ctrl.phase_c_nominal_start, 75_000)

    def test_v6i2_preset_uses_extended_phase_a_max(self) -> None:
        cfg = apply_preset(PPOConfig(), "v6i2")
        cfg.checkpoint_dir = tempfile.mkdtemp()
        ctrl = _make_controller(cfg)
        self.assertEqual(ctrl.active_families, GATE_FAMILY_NAMES_V6I2)
        self.assertEqual(ctrl.phase_a_max_end, int(0.70 * ctrl.nominal_steps))


class GateSchedulingTests(unittest.TestCase):
    def test_no_gate_before_phase_a_min_end(self) -> None:
        ctrl = _make_controller()
        self.assertFalse(ctrl.should_run_phase_a_gate(399_999))

    def test_gate_runs_at_minimum_boundary(self) -> None:
        ctrl = _make_controller()
        self.assertTrue(ctrl.should_run_phase_a_gate(400_000))

    def test_gate_runs_on_interval(self) -> None:
        cfg = PPOConfig()
        cfg.checkpoint_dir = tempfile.mkdtemp()
        cfg.curriculum_nominal_timesteps = 1_000_000
        cfg.phase_a_gate_check_interval = 25_000
        ctrl = _make_controller(cfg)
        ctrl.last_gate_step_run = 400_001
        ctrl.next_gate_step = 425_001
        self.assertTrue(ctrl.should_run_phase_a_gate(425_500))

    def test_final_gate_runs_once_at_phase_a_max_end(self) -> None:
        cfg = PPOConfig()
        cfg.checkpoint_dir = tempfile.mkdtemp()
        cfg.curriculum_nominal_timesteps = 1_000_000
        ctrl = _make_controller(cfg)
        ctrl.phase_a_max_end = 550_000
        ctrl.next_gate_step = 600_000
        ctrl.last_gate_step_run = 525_000
        self.assertTrue(ctrl.should_run_phase_a_gate(551_000))
        ctrl.last_gate_step_run = 550_000
        self.assertFalse(ctrl.should_run_phase_a_gate(560_000))

    def test_resume_restores_phase_and_next_gate(self) -> None:
        ctrl = _make_controller()
        ctrl.phase = "A"
        ctrl.next_gate_step = 425_000
        ctrl.last_gate_step_run = 400_000
        payload = v6i1_curriculum_state_dict(ctrl)
        restored = _make_controller()
        load_v6i1_curriculum_state(restored, payload)
        self.assertEqual(restored.phase, "A")
        self.assertEqual(restored.next_gate_step, 425_000)
        self.assertEqual(restored.last_gate_step_run, 400_000)


class PromotionPolicyTests(unittest.TestCase):
    def test_all_pass_enforce_allows_promotion(self) -> None:
        results = {name: gate_family_result_from_bool(True) for name in GATE_FAMILY_NAMES_V6I1}
        self.assertTrue(all_required_families_passed(results, families=GATE_FAMILY_NAMES_V6I1))
        self.assertTrue(
            overall_gate_passed_for_promotion(results, mode="enforce", families=GATE_FAMILY_NAMES_V6I1)
        )

    def test_not_run_blocks_promotion(self) -> None:
        results = {name: gate_family_result_from_bool(True) for name in GATE_FAMILY_NAMES_V6I1}
        results["matched_seed_behavior"] = GateFamilyResult(status=GATE_STATUS_NOT_RUN)
        self.assertFalse(all_required_families_passed(results, families=GATE_FAMILY_NAMES_V6I1))
        self.assertFalse(
            overall_gate_passed_for_promotion(results, mode="enforce", families=GATE_FAMILY_NAMES_V6I1)
        )

    def test_error_blocks_promotion(self) -> None:
        results = {name: gate_family_result_from_bool(True) for name in GATE_FAMILY_NAMES_V6I1}
        results["selector_learnability_probe"] = GateFamilyResult(
            status=GATE_STATUS_ERROR, reason="dataset_invalid"
        )
        self.assertFalse(all_required_families_passed(results, families=GATE_FAMILY_NAMES_V6I1))

    def test_observe_only_gate_pass_without_promotion_eligibility(self) -> None:
        results = {name: gate_family_result_from_bool(True) for name in GATE_FAMILY_NAMES_V6I1}
        self.assertTrue(all_required_families_passed(results, families=GATE_FAMILY_NAMES_V6I1))
        self.assertFalse(
            overall_gate_passed_for_promotion(results, mode="observe_only", families=GATE_FAMILY_NAMES_V6I1)
        )

    def test_v6i2_uses_distinct_active_families(self) -> None:
        cfg = apply_preset(PPOConfig(), "v6i2")
        cfg.checkpoint_dir = tempfile.mkdtemp()
        ctrl = _make_controller(cfg)
        self.assertEqual(ctrl.active_families, GATE_FAMILY_NAMES_V6I2)
        self.assertNotEqual(GATE_FAMILY_NAMES_V6I1, GATE_FAMILY_NAMES_V6I2)


class TerminalFailureOrderingTests(unittest.TestCase):
    def test_terminal_failure_requires_final_gate_attempt(self) -> None:
        cfg = PPOConfig()
        cfg.checkpoint_dir = tempfile.mkdtemp()
        cfg.curriculum_nominal_timesteps = 1_000_000
        cfg.phase_boundary_gate_mode = "enforce"
        cfg.curriculum_gate_run_boundary_eval = True
        cfg.curriculum_gate_run_probe = True
        ctrl = _make_controller(cfg, relax_enforce_for_startup=False)
        ctrl.trainer.global_step = 560_000
        ctrl.phase_a_gate_passed = False
        ctrl.last_gate_step_run = 525_000
        with mock.patch.object(ctrl, "_handle_terminal_failure") as fail_mock:
            ctrl.check_terminal_failure()
        fail_mock.assert_not_called()

    def test_terminal_failure_after_final_gate_when_not_passed(self) -> None:
        cfg = PPOConfig()
        cfg.checkpoint_dir = tempfile.mkdtemp()
        cfg.curriculum_nominal_timesteps = 1_000_000
        cfg.phase_boundary_gate_mode = "enforce"
        cfg.curriculum_gate_run_boundary_eval = True
        cfg.curriculum_gate_run_probe = True
        ctrl = _make_controller(cfg, relax_enforce_for_startup=False)
        ctrl.trainer.global_step = 560_000
        ctrl.phase_a_gate_passed = False
        ctrl.last_gate_step_run = 550_000
        with mock.patch.object(ctrl, "_handle_terminal_failure") as fail_mock:
            ctrl.check_terminal_failure()
        fail_mock.assert_called_once()


class PhaseBCTransitionTests(unittest.TestCase):
    def test_observe_only_reaches_phase_c(self) -> None:
        cfg = PPOConfig()
        cfg.checkpoint_dir = tempfile.mkdtemp()
        cfg.curriculum_nominal_timesteps = 100_000
        cfg.phase_boundary_gate_mode = "observe_only"
        ctrl = _make_controller(cfg)
        ctrl.trainer.global_step = 40_000
        ctrl.maybe_apply_nominal_phase_transition()
        self.assertEqual(ctrl.phase, "B")
        ctrl.trainer.global_step = 70_000
        ctrl.maybe_apply_nominal_phase_transition()
        self.assertEqual(ctrl.phase, "C")

    def test_enforce_mode_reaches_phase_c_after_gate_promotion(self) -> None:
        cfg = PPOConfig()
        cfg.checkpoint_dir = tempfile.mkdtemp()
        cfg.curriculum_nominal_timesteps = 100_000
        cfg.phase_boundary_gate_mode = "enforce"
        cfg.curriculum_gate_run_boundary_eval = True
        cfg.curriculum_gate_run_probe = True
        ctrl = _make_controller(cfg, relax_enforce_for_startup=False)
        ctrl.phase = "B"
        ctrl.t_A = 40_000
        ctrl.trainer.global_step = 70_000
        ctrl.maybe_apply_nominal_phase_transition()
        self.assertEqual(ctrl.phase, "C")


class RankingMetricKeyTests(unittest.TestCase):
    def test_probe_regret_uses_canonical_fixed_policy_regret_key(self) -> None:
        gate_results = {name: gate_family_result_from_bool(True) for name in GATE_FAMILY_NAMES_V6I1}
        components = build_lexicographic_ranking_components(
            gate_results=gate_results,
            online_report={
                "competence_scores": [0.6, 0.55, 0.52, 0.51],
                "pair_jsd_ema": [0.02] * 6,
                "jsd_margin": 0.01,
                "recent_z_occupancy": [0.10, 0.30, 0.25, 0.35],
            },
            matched_report={"opponents": {"OP5": {"effect_size": 0.12}}},
            probe_report={
                "global_best_fixed_z_regret": 1.0,
                "probe_regret": 0.4,
            },
            global_step=425_000,
        )
        self.assertAlmostEqual(components["probe_regret_reduction"], 0.6)
        self.assertAlmostEqual(components["occupancy_imbalance"], 0.25)

    def test_ranking_order_changes_with_occupancy_imbalance(self) -> None:
        base = {
            "checkpoint": "a.zip",
            "ranking_components": {
                "gate_families_passed": 4,
                "gate_families_measured": 4,
                "min_competence": 0.4,
                "pairs_above_margin": 3,
                "weakest_pair_normalized_separation": 0.5,
                "matched_seed_effect_size": 0.1,
                "probe_regret_reduction": 0.0,
                "occupancy_imbalance": 0.2,
                "global_step": 500_000,
            },
        }
        better = dict(base)
        better["checkpoint"] = "b.zip"
        better["ranking_components"] = dict(base["ranking_components"])
        better["ranking_components"]["occupancy_imbalance"] = 0.05
        ranked = rank_candidates_lexicographic([base, better])
        self.assertEqual(ranked[0]["checkpoint"], "b.zip")
        self.assertNotIn("lexicographic_rank", base)


if __name__ == "__main__":
    unittest.main()
