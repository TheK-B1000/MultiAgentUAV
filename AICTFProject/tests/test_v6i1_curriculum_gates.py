"""Unit tests for V6I1 Phase A curriculum gate controller."""

from __future__ import annotations

import tempfile
import unittest
from collections import deque
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
import torch.nn as nn

from rl.config.ppo_config import PPOConfig
from rl.custom_ppo.curriculum_gates import (
    GATE_FAMILY_NAMES,
    GATE_STATUS_FAIL,
    GATE_STATUS_NOT_RUN,
    GATE_STATUS_PASS,
    LATENT_PAIR_INDEX,
    PAIR_ORDER,
    GateFamilyResult,
    TrainingIsolationSnapshot,
    V6I1CurriculumController,
    build_lexicographic_ranking_components,
    gate_family_result_from_bool,
    is_staged_v6i1_curriculum,
    overall_gate_passed_for_promotion,
    rank_candidates_lexicographic,
    validate_v6i1_enforce_config,
)
from rl.custom_ppo.latent_strategy_state import LatentStrategyState
from rl.custom_ppo.ppo_updater import set_model_requires_grad_for_phase


class LexicographicRankingTests(unittest.TestCase):
    def test_rank_prefers_more_gate_families_passed(self) -> None:
        a = {
            "checkpoint": "a.zip",
            "ranking_components": {
                "gate_families_passed": 4,
                "min_competence": 0.4,
                "pairs_above_margin": 3,
                "weakest_pair_normalized_separation": 0.5,
                "matched_seed_effect_size": 0.1,
                "probe_regret_reduction": 0.0,
                "occupancy_imbalance": 0.1,
                "global_step": 500_000,
            },
        }
        b = {
            "checkpoint": "b.zip",
            "ranking_components": {
                "gate_families_passed": 5,
                "min_competence": 0.3,
                "pairs_above_margin": 1,
                "weakest_pair_normalized_separation": 0.2,
                "matched_seed_effect_size": 0.05,
                "probe_regret_reduction": -0.1,
                "occupancy_imbalance": 0.2,
                "global_step": 600_000,
            },
        }
        ranked = rank_candidates_lexicographic([a, b])
        self.assertEqual(ranked[0]["checkpoint"], "b.zip")

    def test_tie_break_earliest_checkpoint_last(self) -> None:
        early = {
            "checkpoint": "early.zip",
            "ranking_components": {
                "gate_families_passed": 2,
                "min_competence": 0.5,
                "pairs_above_margin": 4,
                "weakest_pair_normalized_separation": 0.8,
                "matched_seed_effect_size": 0.2,
                "probe_regret_reduction": 0.1,
                "occupancy_imbalance": 0.05,
                "global_step": 400_000,
            },
        }
        late = dict(early)
        late["checkpoint"] = "late.zip"
        late["ranking_components"] = dict(early["ranking_components"])
        late["ranking_components"]["global_step"] = 450_000
        ranked = rank_candidates_lexicographic([late, early])
        self.assertEqual(ranked[0]["checkpoint"], "early.zip")

    def test_not_run_cannot_outrank_measured_passing(self) -> None:
        measured = {
            "checkpoint": "measured.zip",
            "ranking_components": {
                "gate_families_passed": 4,
                "gate_families_measured": 4,
                "min_competence": 0.4,
                "pairs_above_margin": 3,
                "weakest_pair_normalized_separation": 0.5,
                "matched_seed_effect_size": 0.1,
                "probe_regret_reduction": 0.0,
                "occupancy_imbalance": 0.1,
                "global_step": 500_000,
            },
        }
        inflated = {
            "checkpoint": "inflated.zip",
            "ranking_components": {
                "gate_families_passed": 4,
                "gate_families_measured": 2,
                "min_competence": 0.9,
                "pairs_above_margin": 6,
                "weakest_pair_normalized_separation": 1.0,
                "matched_seed_effect_size": 1.0,
                "probe_regret_reduction": 1.0,
                "occupancy_imbalance": 0.0,
                "global_step": 400_000,
            },
        }
        ranked = rank_candidates_lexicographic([inflated, measured])
        self.assertEqual(ranked[0]["checkpoint"], "measured.zip")


class StagedActivationTests(unittest.TestCase):
    def test_staged_requires_explicit_training_mode(self) -> None:
        cfg = PPOConfig()
        cfg.use_v6i1_curriculum = True
        cfg.training_mode = "staged_team_intent_curriculum"
        cfg.experiment_family = "v6"
        cfg.experiment_id = "v6i1"
        self.assertTrue(is_staged_v6i1_curriculum(cfg))

    def test_repertoire_ablation_does_not_activate_staged_controller(self) -> None:
        cfg = PPOConfig()
        cfg.use_v6i1_curriculum = False
        cfg.training_mode = "repertoire_only_ablation"
        cfg.experiment_family = "v6"
        cfg.experiment_id = "v6i1"
        self.assertFalse(is_staged_v6i1_curriculum(cfg))


class OnlineGateLogicTests(unittest.TestCase):
    def _make_controller(self, cfg: PPOConfig | None = None) -> V6I1CurriculumController:
        fresh = cfg is None
        cfg = cfg or PPOConfig()
        cfg.checkpoint_dir = tempfile.mkdtemp()
        if fresh:
            cfg.curriculum_nominal_timesteps = 1_000_000
        cfg.latent_cf_min_episodes_per_z = 50
        cfg.latent_cf_jsd_margin = 0.01
        cfg.latent_cf_gate_consecutive_updates = 5
        cfg.training_mode = "staged_team_intent_curriculum"
        cfg.experiment_family = "v6"
        cfg.experiment_id = "v6i1"
        cfg.use_v6i1_curriculum = True
        if str(getattr(cfg, "phase_boundary_gate_mode", "enforce")).lower() == "enforce":
            cfg.phase_boundary_gate_mode = "observe_only"
        latent_state = SimpleNamespace(
            cf_episode_counts=np.array([60, 60, 60, 60], dtype=np.int64),
            recent_z_history=deque([0, 1, 2, 3] * 25),
            pair_jsd_ema=np.array([0.02, 0.02, 0.02, 0.02, 0.02, 0.006], dtype=np.float32),
            jsd_gate_consecutive_updates=5,
            cf_J=np.array([10.0, 9.5, 9.0, 8.5], dtype=np.float32),
            cf_return_var=1.0,
            router_optimizer_step_count=0,
            compute_competence_scores=lambda: (
                np.array([0.55, 0.52, 0.51, 0.50], dtype=np.float32),
                True,
            ),
        )
        trainer = SimpleNamespace(
            cfg=cfg,
            global_step=400_000,
            latent_k=4,
            latent_state=latent_state,
            latent_episode_strategy_ppo=False,
            last_stats={
                "latent_forced_z_step_fraction": 1.0,
                "router_sample_count_by_z_0": 0.0,
                "router_sample_count_by_z_1": 0.0,
                "router_sample_count_by_z_2": 0.0,
                "router_sample_count_by_z_3": 0.0,
                "strategy_switch_count": 0.0,
                "q_phi_grad_norm": 0.0,
            },
        )
        return V6I1CurriculumController(trainer)

    def test_coverage_gate_passes_with_balanced_forced_z(self) -> None:
        ctrl = self._make_controller()
        result = ctrl._evaluate_coverage_gate()
        self.assertEqual(result.status, GATE_STATUS_PASS)
        self.assertTrue(all(0.20 <= o <= 0.30 for o in result.details["recent_z_occupancy"]))

    def test_competence_gate_requires_sigmoid_scores(self) -> None:
        ctrl = self._make_controller()
        result = ctrl._evaluate_competence_gate()
        self.assertEqual(result.status, GATE_STATUS_PASS)
        self.assertTrue(result.details["cf_competence_ready"])
        self.assertGreaterEqual(min(result.details["competence_scores"]), 0.50)

    def test_intervention_gate_requires_consecutive_updates(self) -> None:
        ctrl = self._make_controller()
        result = ctrl._evaluate_intervention_gate()
        self.assertEqual(result.status, GATE_STATUS_PASS)
        ctrl.trainer.latent_state.jsd_gate_consecutive_updates = 2
        result_after_reset = ctrl._evaluate_intervention_gate()
        self.assertEqual(result_after_reset.status, GATE_STATUS_FAIL)

    def test_training_integrity_gate_phase_a(self) -> None:
        ctrl = self._make_controller()
        result = ctrl._evaluate_training_integrity_gate()
        self.assertEqual(result.status, GATE_STATUS_PASS)
        self.assertEqual(result.details["forced_z_fraction"], 1.0)
        self.assertEqual(result.details["router_sample_count"], 0.0)

    def test_observe_only_transitions_on_nominal_schedule_despite_failed_gates(self) -> None:
        cfg = PPOConfig()
        cfg.checkpoint_dir = tempfile.mkdtemp()
        cfg.curriculum_nominal_timesteps = 100_000
        cfg.phase_boundary_gate_mode = "observe_only"
        cfg.curriculum_gate_run_boundary_eval = False
        cfg.curriculum_gate_run_probe = False
        ctrl = self._make_controller(cfg)
        ctrl.trainer.global_step = 40_000
        ctrl.trainer.save = mock.Mock()
        with mock.patch.object(ctrl, "check_and_run_gate", return_value=False) as gate_mock:
            transitioned = ctrl.maybe_apply_nominal_phase_transition()
        self.assertTrue(transitioned)
        self.assertEqual(ctrl.phase, "B")
        gate_mock.assert_not_called()

    def test_observe_only_nominal_phase_c_at_70_percent(self) -> None:
        cfg = PPOConfig()
        cfg.checkpoint_dir = tempfile.mkdtemp()
        cfg.curriculum_nominal_timesteps = 100_000
        cfg.phase_boundary_gate_mode = "observe_only"
        ctrl = self._make_controller(cfg)
        ctrl.phase = "B"
        ctrl.trainer.global_step = 70_000
        ctrl.trainer.save = mock.Mock()
        transitioned = ctrl.maybe_apply_nominal_phase_transition()
        self.assertTrue(transitioned)
        self.assertEqual(ctrl.phase, "C")

    def test_observe_only_gate_check_records_not_run_for_heavy_evaluators(self) -> None:
        cfg = PPOConfig()
        cfg.checkpoint_dir = tempfile.mkdtemp()
        cfg.phase_boundary_gate_mode = "observe_only"
        cfg.curriculum_gate_run_boundary_eval = False
        cfg.curriculum_gate_run_probe = False
        ctrl = self._make_controller(cfg)
        ctrl.trainer.global_step = 400_000
        ctrl.trainer.save = mock.Mock()
        snap = mock.Mock()
        snap.assert_unchanged = mock.Mock()
        snap.restore_rng = mock.Mock()
        snap.model_was_training = True
        ctrl.trainer.model = mock.Mock(training=True)
        ctrl.trainer.model.train = mock.Mock()
        with mock.patch(
            "rl.custom_ppo.curriculum_gates.TrainingIsolationSnapshot.capture",
            return_value=snap,
        ), mock.patch.object(ctrl, "_evaluate_online_gates", return_value={}):
            promoted = ctrl.check_and_run_gate()
        self.assertFalse(promoted)
        self.assertEqual(ctrl.phase, "A")
        report = ctrl.gate_check_history[-1]
        self.assertEqual(report["gate_families"]["matched_seed_behavior"]["status"], GATE_STATUS_NOT_RUN)
        self.assertEqual(report["gate_families"]["selector_learnability_probe"]["status"], GATE_STATUS_NOT_RUN)

    def test_enforce_not_run_blocks_promotion(self) -> None:
        gate_results = {
            "coverage": gate_family_result_from_bool(True),
            "competence": gate_family_result_from_bool(True),
            "counterfactual_intervention": gate_family_result_from_bool(True),
            "training_integrity": gate_family_result_from_bool(True),
            "matched_seed_behavior": GateFamilyResult(status=GATE_STATUS_NOT_RUN),
            "selector_learnability_probe": GateFamilyResult(status=GATE_STATUS_NOT_RUN),
        }
        self.assertFalse(overall_gate_passed_for_promotion(gate_results, mode="enforce"))

    def test_should_run_gate_uses_threshold_crossing(self) -> None:
        cfg = PPOConfig()
        cfg.checkpoint_dir = tempfile.mkdtemp()
        cfg.curriculum_nominal_timesteps = 1_000_000
        cfg.phase_a_gate_check_interval = 25_000
        ctrl = self._make_controller(cfg)
        self.assertFalse(ctrl.should_run_phase_a_gate(399_999))
        self.assertTrue(ctrl.should_run_phase_a_gate(400_001))
        ctrl.last_gate_step_run = 400_001
        ctrl.next_gate_step = 425_001
        self.assertTrue(ctrl.should_run_phase_a_gate(425_500))

    def test_final_gate_runs_at_or_after_55_percent_before_terminal_failure(self) -> None:
        cfg = PPOConfig()
        cfg.checkpoint_dir = tempfile.mkdtemp()
        cfg.curriculum_nominal_timesteps = 1_000_000
        cfg.phase_boundary_gate_mode = "enforce"
        cfg.curriculum_gate_run_boundary_eval = True
        cfg.curriculum_gate_run_probe = True
        ctrl = self._make_controller(cfg)
        ctrl.phase_a_max_end = 550_000
        ctrl.next_gate_step = 600_000
        ctrl.last_gate_step_run = 525_000
        self.assertTrue(ctrl.should_run_phase_a_gate(551_000))

    def test_build_ranking_components_shape(self) -> None:
        gate_results = {name: gate_family_result_from_bool(True) for name in GATE_FAMILY_NAMES}
        components = build_lexicographic_ranking_components(
            gate_results=gate_results,
            online_report={
                "competence_scores": [0.6, 0.55, 0.52, 0.51],
                "pair_jsd_ema": [0.02, 0.02, 0.02, 0.02, 0.02, 0.006],
                "jsd_margin": 0.01,
                "occupancy": [0.25, 0.25, 0.25, 0.25],
            },
            matched_report={"opponents": {"OP5": {"effect_size": 0.12}}},
            probe_report={"fixed_regret": 1.0, "probe_regret": 0.5},
            global_step=425_000,
        )
        self.assertEqual(components["gate_families_passed"], len(GATE_FAMILY_NAMES))
        self.assertEqual(components["gate_families_measured"], len(GATE_FAMILY_NAMES))
        self.assertAlmostEqual(components["probe_regret_reduction"], 0.5)


class EnforceConfigValidationTests(unittest.TestCase):
    def test_enforce_mode_requires_boundary_eval_at_startup(self) -> None:
        cfg = PPOConfig()
        cfg.phase_boundary_gate_mode = "enforce"
        cfg.curriculum_gate_run_boundary_eval = False
        cfg.curriculum_gate_run_probe = True
        with self.assertRaisesRegex(ValueError, "matched-seed boundary evaluation"):
            validate_v6i1_enforce_config(cfg)

    def test_enforce_mode_requires_probe_at_startup(self) -> None:
        cfg = PPOConfig()
        cfg.phase_boundary_gate_mode = "enforce"
        cfg.curriculum_gate_run_boundary_eval = True
        cfg.curriculum_gate_run_probe = False
        with self.assertRaisesRegex(ValueError, "selector-learnability probe"):
            validate_v6i1_enforce_config(cfg)

    def test_observe_only_allows_disabled_heavy_gates(self) -> None:
        cfg = PPOConfig()
        cfg.phase_boundary_gate_mode = "observe_only"
        cfg.curriculum_gate_run_boundary_eval = False
        cfg.curriculum_gate_run_probe = False
        validate_v6i1_enforce_config(cfg)

    def test_enforce_passes_when_both_heavy_gates_enabled(self) -> None:
        cfg = PPOConfig()
        cfg.phase_boundary_gate_mode = "enforce"
        cfg.curriculum_gate_run_boundary_eval = True
        cfg.curriculum_gate_run_probe = True
        validate_v6i1_enforce_config(cfg)


class InterventionGateProfileTests(unittest.TestCase):
    def _bare_latent_state(self) -> LatentStrategyState:
        trainer = SimpleNamespace(cfg=PPOConfig())
        state = LatentStrategyState.__new__(LatentStrategyState)
        state.trainer = trainer
        state.pair_jsd_ema = np.zeros(6, dtype=np.float32)
        state.jsd_gate_consecutive_updates = 0
        return state

    def test_weak_sixth_pair_resets_consecutive_counter(self) -> None:
        state = self._bare_latent_state()
        margin = 0.01
        profile = {f"forced_z_pair_jsd_{i}": 0.02 for i in range(5)}
        profile["forced_z_pair_jsd_5"] = 0.004
        for _ in range(5):
            LatentStrategyState.update_intervention_gate_from_profile(state, profile)
        self.assertEqual(state.jsd_gate_consecutive_updates, 0)

    def test_five_of_six_with_floor_passes_after_consecutive_updates(self) -> None:
        state = self._bare_latent_state()
        state.trainer.cfg.latent_cf_jsd_ema_alpha = 1.0
        profile = {f"forced_z_pair_jsd_{i}": 0.02 for i in range(5)}
        profile["forced_z_pair_jsd_5"] = 0.008
        for _ in range(5):
            LatentStrategyState.update_intervention_gate_from_profile(state, profile)
        self.assertEqual(state.jsd_gate_consecutive_updates, 5)
        ctrl = OnlineGateLogicTests()._make_controller()
        ctrl.trainer.latent_state = state
        result = ctrl._evaluate_intervention_gate()
        self.assertTrue(result.details["single_update_ok"])
        self.assertGreaterEqual(result.details["min_pair_jsd_ema"], result.details["min_pair_floor"])
        self.assertEqual(result.status, GATE_STATUS_PASS)

    def test_five_of_six_without_floor_fails_single_update(self) -> None:
        state = self._bare_latent_state()
        state.pair_jsd_ema = np.array([0.02, 0.02, 0.02, 0.02, 0.02, 0.004], dtype=np.float32)
        ctrl = OnlineGateLogicTests()._make_controller()
        ctrl.trainer.latent_state = state
        result = ctrl._evaluate_intervention_gate()
        self.assertFalse(result.details["single_update_ok"])
        self.assertEqual(result.status, GATE_STATUS_FAIL)

    def test_pair_order_mapping_is_stable(self) -> None:
        self.assertEqual(PAIR_ORDER, LATENT_PAIR_INDEX)
        ctrl = OnlineGateLogicTests()._make_controller()
        result = ctrl._evaluate_intervention_gate()
        for idx, pair in enumerate(LATENT_PAIR_INDEX):
            key = f"pair_{idx}_z{pair[0]}_z{pair[1]}"
            self.assertIn(key, result.details)
            self.assertEqual(result.details[f"forced_z_pair_jsd_{idx}"], result.details[key])


class TrainingIsolationSnapshotTests(unittest.TestCase):
    def test_restore_rng_reproduces_torch_draw_sequence(self) -> None:
        torch.manual_seed(123)
        expected = [float(x) for x in torch.rand(4).tolist()]
        torch.manual_seed(123)
        snap = TrainingIsolationSnapshot()
        snap.torch_rng_state = torch.get_rng_state()
        _ = torch.rand(4)
        snap.restore_rng()
        actual = [float(x) for x in torch.rand(4).tolist()]
        self.assertEqual(expected, actual)

    def test_assert_unchanged_detects_parameter_mutation(self) -> None:
        layer = nn.Linear(3, 2)
        trainer = SimpleNamespace(
            model=SimpleNamespace(
                actor=layer,
                critic=None,
                strategy_encoder=None,
                episode_strategy_value_head=None,
                phase_predictor=None,
                strategy_aux_return_head=None,
                training=True,
            ),
            optimizer=None,
            latent_router_optimizer=None,
            global_step=0,
        )
        snap = TrainingIsolationSnapshot.capture(trainer)
        with torch.no_grad():
            layer.weight.add_(1.0)
        with self.assertRaises(AssertionError):
            snap.assert_unchanged(trainer)

    def test_assert_unchanged_detects_eval_mode_mutation(self) -> None:
        layer = nn.Linear(3, 2)
        model = nn.Module()
        model.actor = layer
        trainer = SimpleNamespace(
            model=model,
            optimizer=None,
            latent_router_optimizer=None,
            global_step=0,
        )
        snap = TrainingIsolationSnapshot.capture(trainer)
        model.eval()
        with self.assertRaises(AssertionError):
            snap.assert_unchanged(trainer)

    def test_rng_restored_after_evaluator_exception(self) -> None:
        torch.manual_seed(7)
        expected = [float(x) for x in torch.rand(3).tolist()]
        torch.manual_seed(7)
        snap = TrainingIsolationSnapshot()
        snap.torch_rng_state = torch.get_rng_state()
        _ = torch.rand(3)
        probe_failed = False
        try:
            raise RuntimeError("probe failed")
        except RuntimeError:
            probe_failed = True
            snap.restore_rng()
        self.assertTrue(probe_failed)
        actual = [float(x) for x in torch.rand(3).tolist()]
        self.assertEqual(expected, actual)


class PhaseRequiresGradTests(unittest.TestCase):
    def test_critic_stays_trainable_in_all_phases(self) -> None:
        model = nn.Module()
        model.actor_cnn = nn.Linear(2, 2)
        model.latent_actor = nn.Linear(2, 2)
        model.critic = nn.Linear(2, 1)
        model.strategy_encoder = nn.Linear(2, 2)
        for p in model.parameters():
            p.requires_grad = True
        for phase in ("A", "B", "C"):
            set_model_requires_grad_for_phase(model, phase)
            self.assertTrue(model.critic.weight.requires_grad, msg=phase)

    def test_router_frozen_only_in_phase_a(self) -> None:
        model = nn.Module()
        model.actor_cnn = nn.Linear(2, 2)
        model.latent_actor = nn.Linear(2, 2)
        model.critic = nn.Linear(2, 1)
        model.strategy_encoder = nn.Linear(2, 2)
        for p in model.parameters():
            p.requires_grad = True
        set_model_requires_grad_for_phase(model, "A")
        self.assertFalse(model.strategy_encoder.weight.requires_grad)
        set_model_requires_grad_for_phase(model, "B")
        self.assertTrue(model.strategy_encoder.weight.requires_grad)


class TrainingIntegrityGateTests(unittest.TestCase):
    def test_router_optimizer_step_count_blocks_gate(self) -> None:
        ctrl = OnlineGateLogicTests()._make_controller()
        ctrl.trainer.latent_state.router_optimizer_step_count = 3
        result = ctrl._evaluate_training_integrity_gate()
        self.assertEqual(result.status, GATE_STATUS_FAIL)
        self.assertEqual(result.details["router_optimizer_step_count"], 3.0)

    def test_matched_seed_eval_reports_not_run_when_disabled(self) -> None:
        ctrl = OnlineGateLogicTests()._make_controller()
        result = ctrl._run_matched_seed_eval()
        self.assertEqual(result.status, GATE_STATUS_NOT_RUN)
        self.assertIn("curriculum_gate_run_boundary_eval=false", result.reason)


if __name__ == "__main__":
    unittest.main()
