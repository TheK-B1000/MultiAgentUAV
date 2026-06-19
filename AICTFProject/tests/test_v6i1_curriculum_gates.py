"""Unit tests for V6I1 Phase A curriculum gate controller."""

from __future__ import annotations

import tempfile
import unittest
from collections import deque
from types import SimpleNamespace
from typing import Any
from unittest import mock

import numpy as np
import torch
import torch.nn as nn

from rl.config.ppo_config import PPOConfig
from rl.custom_ppo.gate_protocol import V6I2_GATE_PROTOCOL
from rl.custom_ppo.curriculum.evaluators.matched_seed import (
    MatchedSeedEvalConfig,
    _format_matched_seed_progress,
)
from rl.custom_ppo.v6i1_cf_loss import extract_forced_z_pair_values
from rl.custom_ppo.curriculum_gates import (
    GATE_FAMILY_NAMES,
    GATE_STATUS_ERROR,
    GATE_STATUS_FAIL,
    GATE_STATUS_NOT_RUN,
    GATE_STATUS_PASS,
    LATENT_PAIR_INDEX,
    PAIR_ORDER,
    GateFamilyResult,
    TrainingIsolationSnapshot,
    V6I1CurriculumController,
    build_lexicographic_ranking_components,
    format_v6i1_gate_stdout_block,
    gate_family_result_from_bool,
    is_staged_v6i1_curriculum,
    overall_gate_passed_for_promotion,
    rank_candidates_lexicographic,
    validate_v6i1_enforce_config,
)
from rl.custom_ppo.curriculum.isolation import GateIsolationError
from rl.custom_ppo.csv_writers import V6I1_INTERVENTION_PAIR_COUNT, _update_fieldnames
from rl.custom_ppo.v6i1_phase_runtime import (
    format_v6i1_rollout_stdout_line,
    latent_state_v6i1_checkpoint,
    restore_latent_state_v6i1_checkpoint,
    v6i1_intervention_csv_stats,
)
from rl.custom_ppo.inference import CustomPPOInferencePolicy
from rl.custom_ppo.curriculum.evaluators.learnability import _format_selector_probe_progress
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
            pairwise_ema_valid_updates=5,
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
            "rl.custom_ppo.curriculum.isolation.TrainingIsolationSnapshot.capture",
            return_value=snap,
        ), mock.patch.object(ctrl, "_evaluate_online_gates", return_value={}):
            promoted = ctrl.check_and_run_gate()
        self.assertFalse(promoted)
        self.assertEqual(ctrl.phase, "A")
        report = ctrl.gate_check_history[-1]
        self.assertEqual(report["gate_families"]["matched_seed_behavior"]["status"], GATE_STATUS_NOT_RUN)
        self.assertEqual(report["probe_report"], {})

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

    def test_enforce_mode_allows_selector_probe_disabled_by_default(self) -> None:
        cfg = PPOConfig()
        cfg.phase_boundary_gate_mode = "enforce"
        cfg.curriculum_gate_run_boundary_eval = True
        cfg.curriculum_gate_run_probe = False
        cfg.curriculum_gate_selector_blocks_phase_a = False
        validate_v6i1_enforce_config(cfg)

    def test_enforce_mode_requires_probe_only_when_selector_blocks_phase_a(self) -> None:
        cfg = PPOConfig()
        cfg.phase_boundary_gate_mode = "enforce"
        cfg.curriculum_gate_run_boundary_eval = True
        cfg.curriculum_gate_run_probe = False
        cfg.curriculum_gate_selector_blocks_phase_a = True
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
        trainer = SimpleNamespace(cfg=PPOConfig(), global_step=0)
        state = LatentStrategyState.__new__(LatentStrategyState)
        state.trainer = trainer
        state.pair_jsd_ema = np.zeros(6, dtype=np.float32)
        state.jsd_gate_consecutive_updates = 0
        state.pairwise_ema_valid_updates = 0
        state.pairwise_ema_last_update_step = -1
        state.cf_J = np.zeros(4, dtype=np.float32)
        state.cf_episode_counts = np.zeros(4, dtype=np.int64)
        state.cf_has_experience = np.zeros(4, dtype=np.int64)
        state.cf_return_mean = 0.0
        state.cf_return_var = 1.0
        state.router_optimizer_step_count = 0
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
        ctrl.trainer.latent_state.pairwise_ema_valid_updates = 5
        result = ctrl._evaluate_intervention_gate()
        self.assertTrue(result.details["single_update_ok"])
        self.assertGreaterEqual(result.details["min_pair_jsd_ema"], result.details["min_pair_floor"])
        self.assertEqual(result.status, GATE_STATUS_PASS)

    def test_five_of_six_without_floor_fails_single_update(self) -> None:
        state = self._bare_latent_state()
        state.pair_jsd_ema = np.array([0.02, 0.02, 0.02, 0.02, 0.02, 0.004], dtype=np.float32)
        state.pairwise_ema_valid_updates = 3
        ctrl = OnlineGateLogicTests()._make_controller()
        ctrl.trainer.latent_state = state
        result = ctrl._evaluate_intervention_gate()
        self.assertFalse(result.details["single_update_ok"])
        self.assertEqual(result.status, GATE_STATUS_FAIL)

    def test_pair_order_mapping_is_stable(self) -> None:
        self.assertEqual(PAIR_ORDER, LATENT_PAIR_INDEX)
        ctrl = OnlineGateLogicTests()._make_controller()
        ctrl.trainer.latent_state.pairwise_ema_valid_updates = 1
        result = ctrl._evaluate_intervention_gate()
        for idx, pair in enumerate(LATENT_PAIR_INDEX):
            key = f"pair_{idx}_z{pair[0]}_z{pair[1]}"
            self.assertIn(key, result.details)
            self.assertEqual(result.details[f"pair_jsd_ema_{idx}"], result.details[key])

    def test_macro_mean_without_pairs_does_not_update_ema(self) -> None:
        state = self._bare_latent_state()
        state.trainer.cfg.latent_cf_jsd_ema_alpha = 1.0
        profile = {"forced_z_macro_jsd_mean": 0.02, "forced_z_macro_jsd": 0.02}
        updated = LatentStrategyState.update_intervention_gate_from_profile(state, profile)
        self.assertFalse(updated)
        self.assertEqual(state.pairwise_ema_valid_updates, 0)
        self.assertTrue(np.allclose(state.pair_jsd_ema, 0.0))
        self.assertEqual(state.jsd_gate_consecutive_updates, 0)

    def test_genuine_zero_pairs_update_ema(self) -> None:
        state = self._bare_latent_state()
        state.trainer.cfg.latent_cf_jsd_ema_alpha = 1.0
        profile = {f"forced_z_pair_jsd_{i}": 0.0 for i in range(6)}
        updated = LatentStrategyState.update_intervention_gate_from_profile(state, profile)
        self.assertTrue(updated)
        self.assertTrue(np.allclose(state.pair_jsd_ema, 0.0))
        self.assertEqual(state.pairwise_ema_valid_updates, 1)

    def test_intervention_gate_not_run_without_ema_updates(self) -> None:
        ctrl = OnlineGateLogicTests()._make_controller()
        ctrl.trainer.latent_state.pairwise_ema_valid_updates = 0
        result = ctrl._evaluate_intervention_gate()
        self.assertEqual(result.status, GATE_STATUS_NOT_RUN)
        self.assertIn("no_pairwise_profile_ema_updates", result.reason)

    def test_missing_one_pair_key_leaves_ema_unchanged_and_resets_streak(self) -> None:
        state = self._bare_latent_state()
        state.trainer.cfg.latent_cf_jsd_ema_alpha = 1.0
        state.pair_jsd_ema = np.array([0.03] * 6, dtype=np.float32)
        state.jsd_gate_consecutive_updates = 3
        profile = {f"forced_z_pair_jsd_{i}": 0.02 for i in range(5)}
        updated = LatentStrategyState.update_intervention_gate_from_profile(state, profile)
        self.assertFalse(updated)
        self.assertTrue(np.allclose(state.pair_jsd_ema, 0.03))
        self.assertEqual(state.pairwise_ema_valid_updates, 0)
        self.assertEqual(state.jsd_gate_consecutive_updates, 0)
        ctrl = OnlineGateLogicTests()._make_controller()
        ctrl.trainer.latent_state = state
        result = ctrl._evaluate_intervention_gate()
        self.assertEqual(result.status, GATE_STATUS_NOT_RUN)

    def test_nan_or_inf_pair_invalidates_profile(self) -> None:
        profile = {f"forced_z_pair_jsd_{i}": 0.02 for i in range(6)}
        profile["forced_z_pair_jsd_2"] = float("nan")
        self.assertIsNone(extract_forced_z_pair_values(profile))
        profile["forced_z_pair_jsd_2"] = float("inf")
        self.assertIsNone(extract_forced_z_pair_values(profile))
        state = self._bare_latent_state()
        state.jsd_gate_consecutive_updates = 2
        updated = LatentStrategyState.update_intervention_gate_from_profile(state, profile)
        self.assertFalse(updated)
        self.assertEqual(state.jsd_gate_consecutive_updates, 0)

    def test_valid_profile_after_missing_restarts_streak(self) -> None:
        state = self._bare_latent_state()
        state.trainer.cfg.latent_cf_jsd_ema_alpha = 1.0
        state.jsd_gate_consecutive_updates = 4
        incomplete = {"forced_z_macro_jsd_mean": 0.02}
        self.assertFalse(
            LatentStrategyState.update_intervention_gate_from_profile(state, incomplete)
        )
        self.assertEqual(state.jsd_gate_consecutive_updates, 0)
        complete = {f"forced_z_pair_jsd_{i}": 0.02 for i in range(6)}
        self.assertTrue(
            LatentStrategyState.update_intervention_gate_from_profile(state, complete)
        )
        self.assertEqual(state.jsd_gate_consecutive_updates, 1)

    def test_stale_ema_values_without_valid_updates_yield_not_run(self) -> None:
        ctrl = OnlineGateLogicTests()._make_controller()
        ctrl.trainer.latent_state.pair_jsd_ema = np.array(
            [0.02, 0.02, 0.02, 0.02, 0.02, 0.02], dtype=np.float32
        )
        ctrl.trainer.latent_state.jsd_gate_consecutive_updates = 5
        ctrl.trainer.latent_state.pairwise_ema_valid_updates = 0
        result = ctrl._evaluate_intervention_gate()
        self.assertEqual(result.status, GATE_STATUS_NOT_RUN)

    def test_broadcast_mean_never_mutates_pair_ema(self) -> None:
        state = self._bare_latent_state()
        state.trainer.cfg.latent_cf_jsd_ema_alpha = 1.0
        state.pair_jsd_ema = np.array([0.001] * 6, dtype=np.float32)
        profile = {
            "forced_z_macro_jsd_mean": 0.5,
            "forced_z_macro_jsd": 0.5,
            "forced_z_pair_jsd_0": 0.02,
        }
        for _ in range(3):
            LatentStrategyState.update_intervention_gate_from_profile(state, profile)
        self.assertTrue(np.allclose(state.pair_jsd_ema, 0.001))
        self.assertEqual(state.pairwise_ema_valid_updates, 0)

    def test_checkpoint_roundtrip_preserves_gate_state(self) -> None:
        state = self._bare_latent_state()
        state.cf_J = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        state.cf_episode_counts = np.array([10, 20, 30, 40], dtype=np.int64)
        state.cf_has_experience = np.array([1, 1, 0, 0], dtype=np.int64)
        state.cf_return_mean = 1.5
        state.cf_return_var = 0.25
        state.pair_jsd_ema = np.array([0.02, 0.02, 0.02, 0.02, 0.02, 0.006], dtype=np.float32)
        state.jsd_gate_consecutive_updates = 3
        state.pairwise_ema_valid_updates = 7
        state.pairwise_ema_last_update_step = 458_752
        state.router_optimizer_step_count = 0
        payload = latent_state_v6i1_checkpoint(state)
        restored = self._bare_latent_state()
        restore_latent_state_v6i1_checkpoint(restored, payload)
        self.assertTrue(np.allclose(restored.pair_jsd_ema, state.pair_jsd_ema))
        self.assertEqual(restored.jsd_gate_consecutive_updates, 3)
        self.assertEqual(restored.pairwise_ema_valid_updates, 7)
        self.assertEqual(restored.pairwise_ema_last_update_step, 458_752)


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
        with self.assertRaises(GateIsolationError):
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
        with self.assertRaises(GateIsolationError):
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
    def test_matched_seed_progress_format(self) -> None:
        self.assertEqual(
            _format_matched_seed_progress("opponent", 1, 3),
            "[Matched Seed Eval] opponent 1/3",
        )
        self.assertEqual(
            _format_matched_seed_progress("seed", 8, 20),
            "[Matched Seed Eval] seed 8/20",
        )
        self.assertEqual(
            _format_matched_seed_progress("branches", 64, 240),
            "[Matched Seed Eval] branches 64/240",
        )

    def test_selector_probe_progress_format(self) -> None:
        self.assertEqual(
            _format_selector_probe_progress("opponent", 1, 3),
            "[Selector Probe] opponent 1/3",
        )
        self.assertEqual(
            _format_selector_probe_progress("seed", 8, 20),
            "[Selector Probe] seed 8/20",
        )
        self.assertEqual(
            _format_selector_probe_progress("examples", 64, 100),
            "[Selector Probe] examples 64/100",
        )

    def test_online_matched_seed_config_caps_workload(self) -> None:
        cfg = PPOConfig()
        cfg.curriculum_gate_matched_seed_count = 20
        cfg.curriculum_gate_matched_seed_max_steps = 120
        cfg.curriculum_gate_online_matched_seed_count = 5
        cfg.curriculum_gate_online_matched_seed_max_steps = 64
        eval_config = MatchedSeedEvalConfig.online_from_cfg(cfg)
        self.assertEqual(len(eval_config.seeds), 5)
        self.assertEqual(eval_config.max_episode_steps, 64)

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

    def test_gate_eval_predict_uses_inference_wrapper(self) -> None:
        ctrl = OnlineGateLogicTests()._make_controller()
        ctrl.trainer.device = torch.device("cpu")
        policy = mock.Mock()
        policy.predict.return_value = (np.zeros(4, dtype=np.int64), None)
        ctrl._gate_eval_policy = policy
        act = ctrl._gate_eval_predict({"grid": np.zeros((1, 2, 7, 20, 20), dtype=np.float32)})
        policy.predict.assert_called_once()
        self.assertEqual(act.shape, (4,))

    def test_gate_eval_configure_fixed_z_sets_policy_state(self) -> None:
        ctrl = OnlineGateLogicTests()._make_controller()
        ctrl.trainer.device = torch.device("cpu")
        policy = mock.Mock(spec=CustomPPOInferencePolicy)
        ctrl._gate_eval_policy = policy
        ctrl._gate_eval_configure_fixed_z(2)
        self.assertTrue(policy.fixed_latent_strategy)
        self.assertEqual(policy.fixed_latent_strategy_id, 2)
        policy.reset_strategy.assert_called_once()

    def test_learnability_probe_bootstrap_passes_predicted_action_to_step_async(self) -> None:
        ctrl = OnlineGateLogicTests()._make_controller()
        ctrl.cfg.curriculum_gate_run_probe = True
        expected_act = np.arange(8, dtype=np.int64)
        step_async_actions: list[np.ndarray] = []

        class _ProbeEnv:
            def seed(self, _seed: int) -> None:
                return None

            def reset(self) -> dict[str, Any]:
                return {}

            def step_async(self, act: np.ndarray) -> None:
                step_async_actions.append(np.asarray(act, dtype=np.int64).copy())

            def step_wait(self):
                return {}, np.zeros(1), np.array([True]), [{}]

            def state(self) -> np.ndarray:
                return np.zeros((1, 170), dtype=np.float32)

            def close(self) -> None:
                return None

        ctrl.cfg.curriculum_gate_probe_seed_count = 1
        policy = mock.Mock()
        policy.predict.return_value = (expected_act, None)
        ctrl._gate_eval_policy = policy

        with mock.patch(
            "rl.training.env_factory.build_training_env",
            return_value=_ProbeEnv(),
        ), mock.patch(
            "rl.custom_ppo.curriculum.isolation.TrainingIsolationSnapshot.capture",
            return_value=mock.Mock(restore_rng=mock.Mock(), model_was_training=True),
        ), mock.patch(
            "rl.custom_ppo.curriculum.context.preserve_model_training_mode",
            return_value=mock.MagicMock(),
        ):
            ctrl.trainer.model = mock.Mock(training=True, global_state_dim=170)
            ctrl.trainer.model.train = mock.Mock()
            result = ctrl._run_learnability_probe()

        self.assertGreaterEqual(len(step_async_actions), 1)
        np.testing.assert_array_equal(step_async_actions[0], expected_act)
        self.assertEqual(result.status, GATE_STATUS_ERROR)
        self.assertEqual(result.reason, "insufficient_probe_examples")


class V6I1MetricsCsvFieldTests(unittest.TestCase):
    def test_update_csv_includes_v6i1_intervention_columns(self) -> None:
        fields = _update_fieldnames(use_latent_strategy=True, latent_k=4)
        for name in (
            "v6i1_phase_label",
            "v6i1_cf_coef_current",
            "v6i1_usage_coef_current",
            "jsd_gate_consecutive_updates",
            "cf_competence_ready",
            "cf_hinge_active",
            "cf_hinge_effective",
            "cf_valid_team_groups",
            "cf_weight_sum",
            "cf_effective_pairs",
            "cf_loss_requires_grad",
            "latent_actor_z_separation_jsd_min",
            "latent_actor_z_separation_jsd_max",
        ):
            self.assertIn(name, fields)
        for idx in range(V6I1_INTERVENTION_PAIR_COUNT):
            self.assertIn(f"forced_z_pair_jsd_{idx}", fields)
            self.assertIn(f"pair_jsd_ema_{idx}", fields)

    def test_intervention_csv_stats_exports_pair_ema(self) -> None:
        state = SimpleNamespace(
            pair_jsd_ema=np.array([0.02, 0.02, 0.02, 0.02, 0.02, 0.006], dtype=np.float32),
            jsd_gate_consecutive_updates=3,
            cf_J=np.array([10.0, 9.0, 8.0, 7.0], dtype=np.float32),
            cf_episode_counts=np.array([60, 60, 60, 60], dtype=np.int64),
            latent_k=4,
            trainer=SimpleNamespace(latent_k=4),
        )
        state.compute_competence_scores = lambda: (
            np.array([0.55, 0.52, 0.51, 0.50], dtype=np.float32),
            True,
        )
        stats = v6i1_intervention_csv_stats(
            state,
            profile_stats={"forced_z_pair_jsd_5": 0.006},
            cfg=SimpleNamespace(
                latent_cf_jsd_margin=0.01,
                latent_cf_gate_consecutive_updates=5,
            ),
        )
        self.assertEqual(stats["jsd_gate_consecutive_updates"], 3.0)
        self.assertEqual(stats["cf_competence_ready"], 1.0)
        self.assertAlmostEqual(stats["pair_jsd_ema_5"], 0.006)

    def test_rollout_stdout_line_includes_cf_and_pair_ema(self) -> None:
        row = {
            "v6i1_cf_coef_current": 0.01,
            "latent_actor_z_separation_train_active": 1.0,
            "jsd_gate_consecutive_updates": 2.0,
            "cf_competence_ready": 0.0,
            "forced_z_macro_jsd_mean": 0.000678,
            "forced_z_pair_jsd_0": 0.0005,
            "forced_z_pair_jsd_5": 0.0009,
            "pair_jsd_ema_0": 0.02,
            "pair_jsd_ema_5": 0.006,
            "actor_z_jsd_mean": 0.000678,
            "actor_z_jsd_max": 0.0473,
            "cf_hinge_active": 1.0,
            "cf_hinge_effective": 1.0,
            "cf_batch_pairs_below_margin": 4.0,
            "cf_weight_sum": 2.0,
            "cf_effective_pairs": 3.0,
            "cf_valid_team_groups": 512.0,
            "latent_actor_z_separation_jsd": 0.0012,
            "latent_actor_z_separation_jsd_min": 0.0004,
            "latent_actor_z_separation_jsd_max": 0.0031,
            "cf_loss_requires_grad": 1.0,
            "cf_actor_grad_norm": 0.00042,
            "cf_to_ppo_grad_ratio": 0.15,
        }
        text = format_v6i1_rollout_stdout_line(row, phase="A", required_consecutive=5)
        self.assertIn("[V6I1]", text)
        self.assertIn("cf_coef=0.0100", text)
        self.assertIn("sep_train=1", text)
        self.assertIn("jsd_consec=2/5", text)
        self.assertIn("macro_jsd=0.000678", text)
        self.assertIn("pair_raw=[", text)
        self.assertIn("pair_ema=[", text)
        self.assertIn("0.000500", text)
        self.assertIn("actor_jsd=0.000678/0.047300", text)
        self.assertIn("hinge=1 eff=1", text)
        self.assertIn("cf_req_grad=1", text)

    def test_rollout_stdout_line_uses_v6i2_tag_for_dual_evidence_protocol(self) -> None:
        row = {"v6i1_cf_coef_current": 0.01, "jsd_gate_consecutive_updates": 0.0}
        text = format_v6i1_rollout_stdout_line(
            row,
            phase="A",
            required_consecutive=3,
            gate_protocol=V6I2_GATE_PROTOCOL,
        )
        self.assertIn("[V6I2] phase=A", text)
        self.assertNotIn("[V6I1]", text)


class V6I1GateStdoutTests(unittest.TestCase):
    def test_gate_stdout_block_contains_family_statuses(self) -> None:
        gate_results = {
            name: GateFamilyResult(status=GATE_STATUS_PASS)
            for name in GATE_FAMILY_NAMES
        }
        gate_results["matched_seed_behavior"] = GateFamilyResult(status=GATE_STATUS_FAIL)
        gate_results["counterfactual_intervention"] = GateFamilyResult(
            GATE_STATUS_PASS,
            details={
                "num_pairs_above_margin": 5,
                "min_pair_jsd_ema": 0.006,
                "jsd_consecutive_updates": 5,
            },
        )
        text = format_v6i1_gate_stdout_block(
            step=458_752,
            phase="A",
            overall_passed=False,
            mode="enforce",
            gate_results=gate_results,
            online_report={
                "pair_jsd_ema": [0.02, 0.02, 0.02, 0.02, 0.02, 0.006],
                "jsd_consecutive_updates": 5,
                "min_pair_jsd_ema": 0.006,
                "recent_z_occupancy": [0.25, 0.25, 0.25, 0.25],
                "competence_scores": [0.55, 0.52, 0.51, 0.50],
            },
            ranking_components={
                "matched_seed_effect_size": 0.04,
                "probe_regret_reduction": 0.12,
            },
            cf_coef=0.01,
            required_consecutive=5,
        )
        self.assertIn("[V6I1 Gate]", text)
        self.assertIn("phase=A", text)
        self.assertIn("matched_eval=FAIL", text)
        self.assertIn("jsd_consec=5/5", text)
        self.assertIn("cf_coef=0.0100", text)


if __name__ == "__main__":
    unittest.main()
