"""Pins v6i2 dual-evidence gate protocol, EMA contracts, and resume safety."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from dataclasses import asdict
from types import SimpleNamespace
from typing import Any
from unittest import mock

import numpy as np

from rl.config.ppo_config import PPOConfig
from rl.custom_ppo.curriculum.evaluators.matched_seed import MatchedSeedEvalConfig
from rl.custom_ppo.curriculum_gates import (
    GATE_FAMILY_NAMES,
    GateFamilyResult,
    GATE_STATUS_ERROR,
    GATE_STATUS_FAIL,
    GATE_STATUS_NOT_RUN,
    GATE_STATUS_PASS,
    V6I1CurriculumController,
    overall_gate_passed_for_promotion,
)
from rl.custom_ppo.schedules import resolve_v6i1_cf_coef
from rl.custom_ppo.gate_protocol import (
    GATE_FAMILY_NAMES_V6I1,
    GATE_FAMILY_NAMES_V6I2,
    V6I2_GATE_PROTOCOL,
    evaluate_actor_intervention,
    evaluate_behavioral_realization,
    evaluate_matched_seed_semantics,
    evaluate_macro_profile_support,
    gate_family_names,
    is_staged_v6_team_intent_curriculum,
    is_v6i2_gate_protocol,
    phase_a_actor_pair_telemetry_from_actor_gate_details,
    phase_a_matched_seed_behavioral_telemetry_from_gate_details,
    resolve_gate_protocol_version,
    resolved_gate_config_dict,
)
from rl.custom_ppo.latent_strategy_state import LatentStrategyState
from rl.custom_ppo.v6i1_phase_runtime import (
    latent_state_v6i1_checkpoint,
    restore_latent_state_v6i1_checkpoint,
)
from rl.presets import apply_preset


def _v6i2_cfg() -> PPOConfig:
    return apply_preset(PPOConfig(), "v6i2")


def _trainer_stub(cfg: PPOConfig, latent_state: Any) -> SimpleNamespace:
    return SimpleNamespace(cfg=cfg, latent_state=latent_state, latent_k=4, global_step=1000)


def _phase_a_stats(step: int) -> dict[str, float]:
    return {
        "phase_a_stats_source_step": float(step),
        "phase_a_behavior_measurement_valid": 1.0,
        "forced_z_behavior_all_z_represented": 1.0,
        "forced_z_behavior_components_valid": 1.0,
        "online_behavior_vector_pair_gate_pass": 1.0,
        "phase_a_corridor_viable": 1.0,
        "phase_a_snapshot_usable": 1.0,
        "phase_a_actor_pairs_above_margin": 6.0,
        "online_behavior_vector_pairs_above_threshold": 6.0,
    }


class _BoundaryStub:
    def __init__(self, trainer: SimpleNamespace) -> None:
        self.trainer = trainer
        self.eval_model = trainer.model

    def __enter__(self) -> "_BoundaryStub":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        return None

    def policy(self) -> mock.Mock:
        return mock.Mock()

    def assert_unchanged(self) -> None:
        return None


def _make_controller_for_gate_attempt(
    *,
    tmpdir: str,
    step: int = 400_000,
    selector_blocks: bool = False,
) -> tuple[V6I1CurriculumController, SimpleNamespace]:
    cfg = _v6i2_cfg()
    cfg.checkpoint_dir = tmpdir
    cfg.phase_boundary_gate_mode = "enforce"
    cfg.curriculum_gate_run_boundary_eval = True
    cfg.curriculum_gate_run_probe = True
    cfg.curriculum_gate_selector_blocks_phase_a = selector_blocks
    cfg.phase_a_gate_check_interval = 1
    trainer = SimpleNamespace(
        cfg=cfg,
        global_step=step,
        latent_k=4,
        latent_state=SimpleNamespace(
            macro_pair_jsd_ema=np.full(6, 0.002, dtype=np.float32),
            macro_pair_jsd_valid_updates=3,
        ),
        model=mock.Mock(training=True),
        last_stats=_phase_a_stats(step),
        update_count=0,
    )

    def _save(path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as handle:
            handle.write(b"candidate")

    def _train_one_update() -> None:
        trainer.global_step += 128
        trainer.update_count += 1

    trainer.save = mock.Mock(side_effect=_save)
    trainer.train_one_update = mock.Mock(side_effect=_train_one_update)
    return V6I1CurriculumController(trainer), trainer


def _pass_online(
    gate_results: dict[str, GateFamilyResult],
    _context: Any,
) -> dict[str, Any]:
    for name in ("coverage", "competence", "actor_intervention", "training_integrity"):
        gate_results[name] = GateFamilyResult(status=GATE_STATUS_PASS)
    return {
        "phase_a_snapshot_usable": 1.0,
        "online_behavior_vector_pair_gate_pass": 1.0,
        "phase_a_corridor_viable": 1.0,
        "phase_a_actor_pairs_above_margin": 6.0,
        "online_behavior_vector_pairs_above_threshold": 6.0,
    }


def _fail_online(
    gate_results: dict[str, GateFamilyResult],
    _context: Any,
) -> dict[str, Any]:
    gate_results["coverage"] = GateFamilyResult(status=GATE_STATUS_FAIL)
    gate_results["competence"] = GateFamilyResult(status=GATE_STATUS_PASS)
    gate_results["actor_intervention"] = GateFamilyResult(status=GATE_STATUS_PASS)
    gate_results["training_integrity"] = GateFamilyResult(status=GATE_STATUS_PASS)
    return {
        "phase_a_snapshot_usable": 1.0,
        "online_behavior_vector_pair_gate_pass": 1.0,
        "phase_a_corridor_viable": 1.0,
    }


def _actor_fail_online(
    gate_results: dict[str, GateFamilyResult],
    _context: Any,
) -> dict[str, Any]:
    gate_results["coverage"] = GateFamilyResult(status=GATE_STATUS_PASS)
    gate_results["competence"] = GateFamilyResult(status=GATE_STATUS_PASS)
    gate_results["actor_intervention"] = GateFamilyResult(
        status=GATE_STATUS_FAIL,
        reason="insufficient_actor_pair_jsd",
        details={"pairs_above_margin": 4, "required_pairs": 5},
    )
    gate_results["training_integrity"] = GateFamilyResult(status=GATE_STATUS_PASS)
    return {
        "phase_a_snapshot_usable": 1.0,
        "online_behavior_vector_pair_gate_pass": 0.0,
        "phase_a_corridor_viable": 0.0,
        "phase_a_actor_pairs_above_margin": 4.0,
        "online_behavior_vector_pairs_above_threshold": 0.0,
    }


class V6i2PresetTests(unittest.TestCase):
    def test_v6i2_protocol_and_timing_diff_vs_v6i1(self):
        v6i1 = asdict(apply_preset(PPOConfig(), "v6i1"))
        v6i2 = asdict(apply_preset(PPOConfig(), "v6i2"))
        allowed = {
            "experiment_id",
            "gate_protocol_version",
            "latent_cf_require_competence",
            "latent_cf_coef_max",
            "latent_cf_weak_pair_boost",
            "latent_cf_worst_pair_coef",
            "phase_a_max_end_fraction",
            "run_tag",
        }
        self.assertEqual({k for k in v6i1 if v6i1[k] != v6i2[k]}, allowed)
        self.assertEqual(v6i2["gate_protocol_version"], V6I2_GATE_PROTOCOL)
        self.assertEqual(v6i2["latent_cf_coef_max"], 1.0)
        self.assertEqual(v6i2["actor_jsd_consecutive_updates"], 3)

    def test_v6i2_mounts_staged_controller(self):
        cfg = _v6i2_cfg()
        self.assertTrue(is_staged_v6_team_intent_curriculum(cfg))
        self.assertTrue(is_v6i2_gate_protocol(cfg))

    def test_v6i2_gate_lineage_records_actor_rule(self):
        resolved = resolved_gate_config_dict(_v6i2_cfg())
        self.assertEqual(
            resolved["actor_intervention_gate_rule"],
            "batch_margin_ema_floor_v1",
        )

    def test_v6i2_strong_cf_schedule_uses_coef_max_one(self):
        cfg = _v6i2_cfg()
        N = int(cfg.curriculum_nominal_timesteps)
        coef_max = float(cfg.latent_cf_coef_max)
        self.assertEqual(coef_max, 1.0)
        cases = [
            (131_000, 0.31),
            (196_000, 0.96),
            (262_000, 1.0),
        ]
        for step, expected_min in cases:
            coef = resolve_v6i1_cf_coef("A", step, 0, N, coef_max)
            self.assertAlmostEqual(coef, expected_min, places=1)


class ProtocolIsolationTests(unittest.TestCase):
    def test_v6i1_family_list_unchanged(self):
        cfg = apply_preset(PPOConfig(), "v6i1")
        self.assertEqual(gate_family_names(cfg), GATE_FAMILY_NAMES_V6I1)
        self.assertEqual(GATE_FAMILY_NAMES, GATE_FAMILY_NAMES_V6I1)

    def test_v6i2_families(self):
        cfg = _v6i2_cfg()
        self.assertEqual(gate_family_names(cfg), GATE_FAMILY_NAMES_V6I2)

    def test_actor_gate_never_reads_legacy_or_macro_ema(self):
        cfg = _v6i2_cfg()
        state = SimpleNamespace(
            cf_pair_jsd_ema=np.full(6, 0.002, dtype=np.float32),
            cf_pair_jsd_last_batch=np.full(6, 0.002, dtype=np.float32),
            cf_pair_jsd_valid_updates=3,
            cf_pair_jsd_last_update_step=500,
            actor_intervention_consecutive_updates=3,
            pair_jsd_ema=np.zeros(6, dtype=np.float32),
            macro_pair_jsd_ema=np.full(6, 0.5, dtype=np.float32),
        )
        result = evaluate_actor_intervention(cfg, state)
        self.assertEqual(result.status, GATE_STATUS_PASS)
        self.assertNotIn("pair_jsd_ema", result.details)
        self.assertNotIn("macro_pair_jsd_ema", result.details)


class ActorEmaTests(unittest.TestCase):
    def _make_state(self, cfg: PPOConfig) -> LatentStrategyState:
        trainer = _trainer_stub(cfg, None)
        state = LatentStrategyState.__new__(LatentStrategyState)
        state.trainer = trainer
        state.cf_pair_jsd_ema = np.zeros(6, dtype=np.float32)
        state.cf_pair_jsd_last_batch = np.zeros(6, dtype=np.float32)
        state.cf_pair_jsd_valid_updates = 0
        state.cf_pair_jsd_last_update_step = -1
        state.actor_intervention_consecutive_updates = 0
        state.actor_intervention_skipped_gate_count = 0
        state.actor_intervention_last_skipped_gate_step = -1
        state.cf_J = np.zeros(4, dtype=np.float32)
        state.cf_episode_counts = np.zeros(4, dtype=np.int32)
        state.cf_has_experience = np.zeros(4, dtype=np.bool_)
        state.cf_return_mean = 0.0
        state.cf_return_var = 1.0
        state.pair_jsd_ema = np.zeros(6, dtype=np.float32)
        state.jsd_gate_consecutive_updates = 0
        state.pairwise_ema_valid_updates = 0
        state.pairwise_ema_last_update_step = -1
        state.macro_pair_jsd_ema = np.zeros(6, dtype=np.float32)
        state.macro_pair_jsd_valid_updates = 0
        state.macro_pair_jsd_last_update_step = -1
        state.router_optimizer_step_count = 0
        trainer.latent_state = state
        return state

    def test_six_valid_pairs_update_once(self):
        cfg = _v6i2_cfg()
        state = self._make_state(cfg)
        vals = [0.002] * 6
        self.assertTrue(state.update_cf_pair_jsd_ema(vals, 100))
        self.assertEqual(state.cf_pair_jsd_valid_updates, 1)
        self.assertEqual(state.cf_pair_jsd_last_update_step, 100)
        np.testing.assert_allclose(state.cf_pair_jsd_ema, vals, rtol=1e-5)

    def test_missing_pair_causes_no_mutation(self):
        cfg = _v6i2_cfg()
        state = self._make_state(cfg)
        before = state.cf_pair_jsd_ema.copy()
        self.assertFalse(state.update_cf_pair_jsd_ema([0.001] * 5, 100))
        np.testing.assert_array_equal(state.cf_pair_jsd_ema, before)
        self.assertEqual(state.cf_pair_jsd_valid_updates, 0)

    def test_nan_causes_no_mutation_and_preserves_streak(self):
        cfg = _v6i2_cfg()
        state = self._make_state(cfg)
        state.actor_intervention_consecutive_updates = 2
        bad = [0.002] * 5 + [float("nan")]
        self.assertFalse(state.update_cf_pair_jsd_ema(bad, 100))
        self.assertEqual(state.actor_intervention_consecutive_updates, 2)
        self.assertEqual(state.cf_pair_jsd_valid_updates, 0)

    def test_genuine_zeros_accepted(self):
        cfg = _v6i2_cfg()
        state = self._make_state(cfg)
        self.assertTrue(state.update_cf_pair_jsd_ema([0.0] * 6, 50))
        self.assertEqual(state.cf_pair_jsd_valid_updates, 1)

    def test_consecutive_streak_increments_and_resets(self):
        cfg = _v6i2_cfg()
        state = self._make_state(cfg)
        good = [0.002] * 6
        state.update_cf_pair_jsd_ema(good, 1)
        self.assertEqual(state.actor_intervention_consecutive_updates, 1)
        state.update_cf_pair_jsd_ema(good, 2)
        self.assertEqual(state.actor_intervention_consecutive_updates, 2)
        state.cf_pair_jsd_ema[:] = 0.0001
        state.update_cf_pair_jsd_ema([0.0001] * 6, 3)
        self.assertEqual(state.actor_intervention_consecutive_updates, 0)

    def test_skipped_gate_marks_stale_without_resetting_actor_pair_state(self):
        cfg = _v6i2_cfg()
        state = self._make_state(cfg)
        good = [0.002] * 6
        state.update_cf_pair_jsd_ema(good, 1)
        state.update_cf_pair_jsd_ema(good, 2)
        before_ema = state.cf_pair_jsd_ema.copy()

        state.mark_actor_intervention_gate_skipped(400_000)

        np.testing.assert_array_equal(state.cf_pair_jsd_ema, before_ema)
        self.assertEqual(state.actor_intervention_consecutive_updates, 2)
        self.assertEqual(state.actor_intervention_skipped_gate_count, 1)
        stale = evaluate_actor_intervention(cfg, state)
        self.assertEqual(stale.status, GATE_STATUS_NOT_RUN)
        self.assertEqual(stale.reason, "actor_pair_evidence_stale_after_skipped_gate")
        self.assertEqual(stale.details["actor_pair_streak_preserved"], 2)

        state.update_cf_pair_jsd_ema(good, 400_128)
        fresh = evaluate_actor_intervention(cfg, state)
        self.assertEqual(fresh.status, GATE_STATUS_PASS)
        self.assertEqual(state.actor_intervention_skipped_gate_count, 0)

    def test_strong_current_batch_with_ema_floor_increments_streak(self):
        cfg = _v6i2_cfg()
        state = self._make_state(cfg)
        state.cf_pair_jsd_ema[:] = 0.00055
        state.cf_pair_jsd_valid_updates = 3
        state.cf_pair_jsd_last_update_step = 100

        strong_batch = [0.0012, 0.0011, 0.0013, 0.0014, 0.00105, 0.0004]
        self.assertTrue(state.update_cf_pair_jsd_ema(strong_batch, 200))

        self.assertEqual(state.actor_intervention_consecutive_updates, 1)
        np.testing.assert_allclose(state.cf_pair_jsd_last_batch, strong_batch, rtol=1e-5)
        self.assertLess(float(np.max(state.cf_pair_jsd_ema)), cfg.actor_jsd_margin)

    def test_ema_floor_blocks_spiky_current_batch(self):
        cfg = _v6i2_cfg()
        state = self._make_state(cfg)
        state.cf_pair_jsd_ema[:] = 0.0001
        state.cf_pair_jsd_valid_updates = 3
        state.cf_pair_jsd_last_update_step = 100

        strong_batch = [0.0012, 0.0011, 0.0013, 0.0014, 0.00105, 0.0012]
        self.assertTrue(state.update_cf_pair_jsd_ema(strong_batch, 200))

        self.assertEqual(state.actor_intervention_consecutive_updates, 0)

    def test_checkpoint_roundtrip(self):
        cfg = _v6i2_cfg()
        state = self._make_state(cfg)
        state.update_cf_pair_jsd_ema([0.002] * 6, 999)
        state.mark_actor_intervention_gate_skipped(1000)
        payload = latent_state_v6i1_checkpoint(state)
        restored = self._make_state(cfg)
        restore_latent_state_v6i1_checkpoint(restored, payload)
        np.testing.assert_allclose(restored.cf_pair_jsd_ema, state.cf_pair_jsd_ema)
        np.testing.assert_allclose(restored.cf_pair_jsd_last_batch, state.cf_pair_jsd_last_batch)
        self.assertEqual(restored.cf_pair_jsd_valid_updates, 1)
        self.assertEqual(restored.cf_pair_jsd_last_update_step, 999)
        self.assertEqual(restored.actor_intervention_skipped_gate_count, 1)
        self.assertEqual(restored.actor_intervention_last_skipped_gate_step, 1000)


class MacroEmaTests(unittest.TestCase):
    def _make_state(self, cfg: PPOConfig) -> LatentStrategyState:
        trainer = _trainer_stub(cfg, None)
        state = LatentStrategyState.__new__(LatentStrategyState)
        state.trainer = trainer
        state.macro_pair_jsd_ema = np.zeros(6, dtype=np.float32)
        state.macro_pair_jsd_valid_updates = 0
        state.macro_pair_jsd_last_update_step = -1
        trainer.latent_state = state
        return state

    def test_macro_ema_independent_of_cf(self):
        cfg = _v6i2_cfg()
        state = self._make_state(cfg)
        self.assertTrue(state.update_macro_pair_jsd_ema([0.0002] * 6, 200))
        self.assertEqual(state.macro_pair_jsd_valid_updates, 1)
        self.assertEqual(state.macro_pair_jsd_last_update_step, 200)

    def test_invalid_macro_input_no_mutation(self):
        cfg = _v6i2_cfg()
        state = self._make_state(cfg)
        before = state.macro_pair_jsd_ema.copy()
        self.assertFalse(state.update_macro_pair_jsd_ema([float("inf")] * 6, 1))
        np.testing.assert_array_equal(state.macro_pair_jsd_ema, before)


class ActorGateBehaviorTests(unittest.TestCase):
    def test_phase_a_actor_pair_telemetry_copies_gate_details(self):
        out = phase_a_actor_pair_telemetry_from_actor_gate_details(
            {
                "batch_pairs_above_margin": 5,
                "cf_pair_jsd_last_batch": [0.002, 0.002, 0.002, 0.002, 0.002, 0.0004],
                "min_cf_pair_jsd_ema": 0.00055,
                "single_update_ok": True,
            }
        )
        self.assertEqual(out["phase_a_actor_pairs_above_margin"], 5.0)
        self.assertEqual(out["phase_a_actor_weakest_pair_jsd"], 0.0004)
        self.assertEqual(out["phase_a_actor_pair_gate_pass"], 1.0)

    def test_matched_seed_behavioral_telemetry_copies_gate_details(self):
        out = phase_a_matched_seed_behavioral_telemetry_from_gate_details(
            {
                "behavioral_realization_gate_status": GATE_STATUS_PASS,
                "matched_seed_semantics": GATE_STATUS_PASS,
                "matched_seed_semantics_details": {
                    "strong_opponent_count": 2,
                    "behavioral_realization_min_opponents_pass": 2,
                    "aggregate_effect": 1.25,
                    "opponents": {
                        "OP5": {
                            "route_distance": 2.0,
                            "task_behavior_distance": 0.02,
                            "performance_spread": 0.4,
                            "aggregate_effect": 1.1,
                            "component_floor_pass": True,
                        },
                        "OP6": {
                            "route_distance": 3.0,
                            "task_behavior_distance": 0.03,
                            "performance_spread": 0.6,
                            "aggregate_effect": 1.4,
                            "component_floor_pass": True,
                        },
                    },
                },
            }
        )
        self.assertEqual(out["matched_seed_behavioral_gate_status"], GATE_STATUS_PASS)
        self.assertEqual(out["matched_seed_behavioral_gate_pass"], 1.0)
        self.assertEqual(out["matched_seed_behavioral_strong_opponents"], 2.0)
        self.assertEqual(out["matched_seed_behavioral_required_opponents"], 2.0)
        self.assertEqual(out["matched_seed_behavioral_component_floor_pass_count"], 2.0)
        self.assertAlmostEqual(out["matched_seed_behavioral_min_task_behavior_distance"], 0.02)
        self.assertAlmostEqual(out["matched_seed_behavioral_mean_route_distance"], 2.5)

    def test_pass_after_required_streak(self):
        cfg = _v6i2_cfg()
        state = SimpleNamespace(
            cf_pair_jsd_ema=np.full(6, 0.0015, dtype=np.float32),
            cf_pair_jsd_last_batch=np.full(6, 0.0015, dtype=np.float32),
            cf_pair_jsd_valid_updates=5,
            cf_pair_jsd_last_update_step=1000,
            actor_intervention_consecutive_updates=3,
            actor_intervention_skipped_gate_count=0,
        )
        result = evaluate_actor_intervention(cfg, state)
        self.assertEqual(result.status, GATE_STATUS_PASS)
        self.assertEqual(result.details["passing_pairs"], 6)
        self.assertEqual(result.details["required_pairs"], 5)
        self.assertEqual(len(result.details["actor_pair_ledger"]), 6)
        self.assertFalse(result.details["opponent_specific_pair_ledger"])

    def test_dual_timescale_pass_uses_batch_margin_and_ema_floor(self):
        cfg = _v6i2_cfg()
        state = SimpleNamespace(
            cf_pair_jsd_ema=np.full(6, 0.00055, dtype=np.float32),
            cf_pair_jsd_last_batch=np.array(
                [0.0012, 0.0011, 0.0013, 0.0014, 0.00105, 0.0004],
                dtype=np.float32,
            ),
            cf_pair_jsd_valid_updates=5,
            cf_pair_jsd_last_update_step=1000,
            actor_intervention_consecutive_updates=3,
            actor_intervention_skipped_gate_count=0,
        )
        result = evaluate_actor_intervention(cfg, state)
        self.assertEqual(result.status, GATE_STATUS_PASS)
        self.assertEqual(result.details["batch_pairs_above_margin"], 5)
        self.assertEqual(result.details["ema_pairs_above_floor"], 6)
        self.assertEqual(result.details["ema_pairs_above_margin"], 0)
        self.assertTrue(result.details["single_update_ok"])

    def test_four_of_six_fails(self):
        cfg = _v6i2_cfg()
        ema = np.array([0.002, 0.002, 0.002, 0.002, 0.0, 0.0], dtype=np.float32)
        state = SimpleNamespace(
            cf_pair_jsd_ema=ema,
            cf_pair_jsd_last_batch=ema,
            cf_pair_jsd_valid_updates=2,
            cf_pair_jsd_last_update_step=100,
            actor_intervention_consecutive_updates=3,
            actor_intervention_skipped_gate_count=0,
        )
        self.assertEqual(evaluate_actor_intervention(cfg, state).status, GATE_STATUS_FAIL)

    def test_insufficient_valid_updates_not_run(self):
        cfg = _v6i2_cfg()
        state = SimpleNamespace(
            cf_pair_jsd_ema=np.zeros(6, dtype=np.float32),
            cf_pair_jsd_last_batch=np.zeros(6, dtype=np.float32),
            cf_pair_jsd_valid_updates=0,
            cf_pair_jsd_last_update_step=-1,
            actor_intervention_consecutive_updates=0,
            actor_intervention_skipped_gate_count=0,
        )
        self.assertEqual(evaluate_actor_intervention(cfg, state).status, GATE_STATUS_NOT_RUN)


class BehavioralRealizationTests(unittest.TestCase):
    def test_matched_seed_mandatory_macro_supporting_only(self):
        cfg = _v6i2_cfg()
        latent = SimpleNamespace(
            macro_pair_jsd_ema=np.zeros(6, dtype=np.float32),
            macro_pair_jsd_valid_updates=0,
        )
        op_reports = {
            "OP5": {
                "avg_route_distance": 0.03,
                "avg_behavior_distance": 0.01,
                "ci_95_low": 0.025,
                "forced_z_performance_spread": 0.04,
                "effect_size": 0.03,
                "num_seeds": 20,
            },
            "OP6": {
                "avg_route_distance": 0.005,
                "avg_behavior_distance": 0.04,
                "ci_95_low": -0.01,
                "forced_z_performance_spread": 0.05,
                "effect_size": 0.04,
                "num_seeds": 20,
            },
            "OP7": {
                "avg_route_distance": 0.01,
                "avg_behavior_distance": 0.005,
                "ci_95_low": 0.0,
                "forced_z_performance_spread": 0.01,
                "effect_size": 0.01,
                "num_seeds": 20,
            },
        }
        result = evaluate_behavioral_realization(cfg, latent, op_reports, boundary_eval_enabled=True)
        self.assertEqual(result.details["matched_seed_semantics"], GATE_STATUS_PASS)
        self.assertEqual(result.details["macro_profile"], GATE_STATUS_NOT_RUN)
        self.assertEqual(result.details["aggregate_result"], GATE_STATUS_PASS)
        op5 = result.details["matched_seed_semantics_details"]["opponents"]["OP5"]
        self.assertIn("route_distance", op5)
        self.assertIn("task_behavior_distance", op5)
        self.assertIn("aggregate_effect", op5)
        self.assertTrue(op5["component_floor_pass"])

    def test_macro_fail_does_not_block_when_semantics_pass(self):
        cfg = _v6i2_cfg()
        latent = SimpleNamespace(
            macro_pair_jsd_ema=np.zeros(6, dtype=np.float32),
            macro_pair_jsd_valid_updates=5,
        )
        macro = evaluate_macro_profile_support(cfg, latent)
        self.assertEqual(macro.status, GATE_STATUS_FAIL)


class SafeguardTests(unittest.TestCase):
    def test_failed_online_prerequisite_skips_expensive_evaluators_and_candidate_save(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ctrl, trainer = _make_controller_for_gate_attempt(tmpdir=tmpdir)
            with mock.patch(
                "rl.custom_ppo.curriculum.controller.GateIsolationBoundary",
                side_effect=lambda trainer_arg: _BoundaryStub(trainer_arg),
            ), mock.patch.object(
                ctrl, "_evaluate_online_gates", side_effect=_fail_online
            ), mock.patch.object(
                ctrl, "_evaluate_behavioral_realization_gate"
            ) as matched_mock, mock.patch.object(
                ctrl, "_run_learnability_probe"
            ) as probe_mock:
                promoted = ctrl.check_and_run_gate()

            self.assertFalse(promoted)
            trainer.save.assert_not_called()
            matched_mock.assert_not_called()
            probe_mock.assert_not_called()
            report_path = os.path.join(tmpdir, "phase_a_gate_reports", "gate_step_400000.json")
            with open(report_path, "r", encoding="utf-8") as handle:
                report = json.load(handle)
            self.assertEqual(report["checkpoint"], "")
            self.assertEqual(
                report["gate_families"]["behavioral_realization"]["status"],
                GATE_STATUS_NOT_RUN,
            )
            self.assertEqual(
                report["gate_families"]["behavioral_realization"]["reason"],
                "online_prerequisites_failed",
            )
            self.assertEqual(
                report["gate_families"]["behavioral_realization"]["behavior_evidence_status"],
                "paused_prerequisites_failed",
            )
            self.assertEqual(report["probe_report"]["failed_online_gate_statuses"]["coverage"], GATE_STATUS_FAIL)

    def test_disabled_boundary_eval_status_differs_from_failed_prerequisite(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ctrl, trainer = _make_controller_for_gate_attempt(tmpdir=tmpdir)
            ctrl.cfg.curriculum_gate_run_boundary_eval = False
            with mock.patch(
                "rl.custom_ppo.curriculum.controller.GateIsolationBoundary",
                side_effect=lambda trainer_arg: _BoundaryStub(trainer_arg),
            ), mock.patch.object(ctrl, "_evaluate_online_gates", side_effect=_pass_online):
                promoted = ctrl.check_and_run_gate()

            self.assertFalse(promoted)
            report_path = os.path.join(tmpdir, "phase_a_gate_reports", "gate_step_400000.json")
            with open(report_path, "r", encoding="utf-8") as handle:
                report = json.load(handle)
            behavioral = report["gate_families"]["behavioral_realization"]
            self.assertEqual(behavioral["status"], GATE_STATUS_NOT_RUN)
            self.assertEqual(behavioral["reason"], "curriculum_gate_run_boundary_eval=false")
            self.assertNotEqual(behavioral["reason"], "online_prerequisites_failed")

    def test_actor_gate_failure_blocks_promotion_but_not_behavioral_evaluation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ctrl, trainer = _make_controller_for_gate_attempt(tmpdir=tmpdir)
            with mock.patch(
                "rl.custom_ppo.curriculum.controller.GateIsolationBoundary",
                side_effect=lambda trainer_arg: _BoundaryStub(trainer_arg),
            ), mock.patch.object(
                ctrl, "_evaluate_online_gates", side_effect=_actor_fail_online
            ), mock.patch.object(
                ctrl,
                "_evaluate_behavioral_realization_gate",
                return_value=GateFamilyResult(
                    status=GATE_STATUS_PASS,
                    details={
                        "behavioral_realization_gate_status": GATE_STATUS_PASS,
                        "matched_seed_semantics": GATE_STATUS_PASS,
                        "matched_seed_semantics_details": {
                            "strong_opponent_count": 2,
                            "behavioral_realization_min_opponents_pass": 2,
                            "aggregate_effect": 1.2,
                            "opponents": {
                                "OP5": {
                                    "route_distance": 2.0,
                                    "task_behavior_distance": 0.02,
                                    "performance_spread": 0.4,
                                    "aggregate_effect": 1.0,
                                    "component_floor_pass": True,
                                },
                                "OP6": {
                                    "route_distance": 3.0,
                                    "task_behavior_distance": 0.03,
                                    "performance_spread": 0.6,
                                    "aggregate_effect": 1.4,
                                    "component_floor_pass": True,
                                },
                            },
                        },
                    },
                ),
            ) as matched_mock, mock.patch.object(
                ctrl, "_run_learnability_probe"
            ) as probe_mock:
                promoted = ctrl.check_and_run_gate()

            self.assertFalse(promoted)
            matched_mock.assert_called_once()
            probe_mock.assert_not_called()
            trainer.save.assert_called_once()
            report_path = os.path.join(tmpdir, "phase_a_gate_reports", "gate_step_400000.json")
            with open(report_path, "r", encoding="utf-8") as handle:
                report = json.load(handle)
            self.assertEqual(
                report["gate_families"]["actor_intervention"]["status"],
                GATE_STATUS_FAIL,
            )
            self.assertEqual(
                report["gate_families"]["actor_intervention"]["reason"],
                "insufficient_actor_pair_jsd",
            )
            self.assertEqual(
                report["gate_families"]["behavioral_realization"]["status"],
                GATE_STATUS_PASS,
            )
            self.assertFalse(report["gate_passed"])
            self.assertFalse(report["promoted_to_phase_b"])
            self.assertEqual(report["checkpoint"], "")
            self.assertTrue(report["candidate_checkpoint_removed"])
            self.assertEqual(trainer.last_stats["matched_seed_behavioral_gate_pass"], 1.0)
            self.assertEqual(trainer.last_stats["matched_seed_behavioral_strong_opponents"], 2.0)
            self.assertAlmostEqual(
                trainer.last_stats["matched_seed_behavioral_min_task_behavior_distance"],
                0.02,
            )

    def test_timeout_returns_control_blocks_promotion_and_cleans_candidate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ctrl, trainer = _make_controller_for_gate_attempt(tmpdir=tmpdir)
            timeout_result = GateFamilyResult(
                status=GATE_STATUS_ERROR,
                reason="inconclusive_timeout",
                details={"timed_out": True, "resume_training": True},
            )
            with mock.patch(
                "rl.custom_ppo.curriculum.controller.GateIsolationBoundary",
                side_effect=lambda trainer_arg: _BoundaryStub(trainer_arg),
            ), mock.patch.object(
                ctrl, "_evaluate_online_gates", side_effect=_pass_online
            ), mock.patch.object(
                ctrl, "_evaluate_behavioral_realization_gate", return_value=timeout_result
            ), mock.patch.object(
                ctrl, "_run_learnability_probe"
            ) as probe_mock:
                promoted = ctrl.check_and_run_gate()

            self.assertFalse(promoted)
            self.assertEqual(ctrl.phase, "A")
            self.assertEqual(ctrl.last_gate_step_run, 400_000)
            candidate_path = os.path.join(tmpdir, "ckpt_candidate_400000.zip")
            self.assertFalse(os.path.exists(candidate_path))
            probe_mock.assert_not_called()
            report_path = os.path.join(tmpdir, "phase_a_gate_reports", "gate_step_400000.json")
            with open(report_path, "r", encoding="utf-8") as handle:
                report = json.load(handle)
            self.assertFalse(report["gate_passed"])
            self.assertFalse(report["promoted_to_phase_b"])
            self.assertTrue(report["candidate_checkpoint_removed"])
            self.assertEqual(report["checkpoint"], "")

            trainer.train_one_update()
            self.assertEqual(trainer.update_count, 1)
            self.assertEqual(trainer.global_step, 400_128)

    def test_selector_diagnostic_failure_does_not_block_promotion(self):
        families = gate_family_names(_v6i2_cfg())
        gate_results = {name: GateFamilyResult(status=GATE_STATUS_PASS) for name in families}
        gate_results["selector_learnability_probe"] = GateFamilyResult(
            status=GATE_STATUS_ERROR,
            reason="diagnostic_failed",
        )
        self.assertTrue(overall_gate_passed_for_promotion(gate_results, mode="enforce", families=families))

    def test_selector_blocks_only_when_explicitly_required_family(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ctrl, _trainer = _make_controller_for_gate_attempt(tmpdir=tmpdir, selector_blocks=True)
            blocking_probe = GateFamilyResult(
                status=GATE_STATUS_ERROR,
                reason="diagnostic_failed",
            )
            with mock.patch(
                "rl.custom_ppo.curriculum.controller.GateIsolationBoundary",
                side_effect=lambda trainer_arg: _BoundaryStub(trainer_arg),
            ), mock.patch.object(
                ctrl, "_evaluate_online_gates", side_effect=_pass_online
            ), mock.patch.object(
                ctrl,
                "_evaluate_behavioral_realization_gate",
                return_value=GateFamilyResult(status=GATE_STATUS_PASS),
            ), mock.patch.object(
                ctrl, "_run_learnability_probe", return_value=blocking_probe
            ):
                promoted = ctrl.check_and_run_gate()

            self.assertFalse(promoted)
            report_path = os.path.join(tmpdir, "phase_a_gate_reports", "gate_step_400000.json")
            with open(report_path, "r", encoding="utf-8") as handle:
                report = json.load(handle)
            self.assertIn("selector_learnability_probe", report["required_families"])
            self.assertFalse(report["gate_passed"])
            self.assertFalse(report["promoted_to_phase_b"])

    def test_full_offline_matched_seed_config_uses_larger_seed_count(self):
        cfg = _v6i2_cfg()
        cfg.curriculum_gate_matched_seed_count = 20
        cfg.curriculum_gate_online_matched_seed_count = 5
        offline = MatchedSeedEvalConfig.from_cfg(cfg)
        online = MatchedSeedEvalConfig.online_from_cfg(cfg)
        self.assertEqual(len(offline.seeds), 20)
        self.assertEqual(len(online.seeds), 5)

    def test_duplicate_timestep_cf_ema_rejected(self):
        cfg = _v6i2_cfg()
        state = ActorEmaTests()._make_state(cfg)
        vals = [0.002] * 6
        self.assertTrue(state.update_cf_pair_jsd_ema(vals, 500))
        self.assertFalse(state.update_cf_pair_jsd_ema(vals, 500))
        self.assertEqual(state.cf_pair_jsd_valid_updates, 1)

    def test_older_timestep_cf_ema_rejected(self):
        cfg = _v6i2_cfg()
        state = ActorEmaTests()._make_state(cfg)
        vals = [0.002] * 6
        self.assertTrue(state.update_cf_pair_jsd_ema(vals, 500))
        before_ema = state.cf_pair_jsd_ema.copy()
        self.assertFalse(state.update_cf_pair_jsd_ema(vals, 400))
        np.testing.assert_array_equal(state.cf_pair_jsd_ema, before_ema)
        self.assertEqual(state.cf_pair_jsd_valid_updates, 1)
        self.assertEqual(state.cf_pair_jsd_last_update_step, 500)

    def test_older_timestep_macro_ema_rejected(self):
        cfg = _v6i2_cfg()
        state = MacroEmaTests()._make_state(cfg)
        vals = [0.0002] * 6
        self.assertTrue(state.update_macro_pair_jsd_ema(vals, 300))
        before = state.macro_pair_jsd_ema.copy()
        self.assertFalse(state.update_macro_pair_jsd_ema(vals, 200))
        np.testing.assert_array_equal(state.macro_pair_jsd_ema, before)

    def test_gate_config_fingerprint_changes_with_threshold(self):
        cfg1 = _v6i2_cfg()
        cfg2 = _v6i2_cfg()
        cfg2.actor_jsd_margin = 0.002
        from rl.custom_ppo.gate_protocol import gate_config_fingerprint

        self.assertNotEqual(gate_config_fingerprint(cfg1), gate_config_fingerprint(cfg2))

    def test_fingerprint_mismatch_rejected_on_resume(self):
        cfg = _v6i2_cfg()
        state = ResumeSafetyTests()._make_minimal_state(cfg)
        payload = latent_state_v6i1_checkpoint(state)
        cfg2 = _v6i2_cfg()
        cfg2.actor_jsd_margin = 0.002
        state2 = ResumeSafetyTests()._make_minimal_state(cfg2)
        with self.assertRaises(ValueError):
            restore_latent_state_v6i1_checkpoint(state2, payload)

    def test_fingerprint_override_marks_non_confirmatory(self):
        cfg = _v6i2_cfg()
        cfg.allow_gate_config_mismatch_on_resume = True
        cfg.phase_boundary_gate_mode = "enforce"
        state = ResumeSafetyTests()._make_minimal_state(cfg)
        payload = latent_state_v6i1_checkpoint(state)
        cfg2 = _v6i2_cfg()
        cfg2.allow_gate_config_mismatch_on_resume = True
        cfg2.actor_jsd_margin = 0.002
        state2 = ResumeSafetyTests()._make_minimal_state(cfg2)
        restore_latent_state_v6i1_checkpoint(state2, payload)
        self.assertTrue(cfg2.gate_config_mismatch_override_used)
        self.assertFalse(cfg2.confirmatory_gate_lineage_valid)
        self.assertEqual(cfg2.phase_boundary_gate_mode, "observe_only")
        self.assertNotEqual(cfg2.gate_config_fingerprint_checkpoint, "")

    def test_semantics_pass_macro_fail(self):
        cfg = _v6i2_cfg()
        latent = SimpleNamespace(
            macro_pair_jsd_ema=np.zeros(6, dtype=np.float32),
            macro_pair_jsd_valid_updates=5,
        )
        op_reports = {
            "OP5": {
                "avg_route_distance": 0.03,
                "avg_behavior_distance": 0.01,
                "ci_95_low": 0.025,
                "ci_95_high": 0.04,
                "forced_z_performance_spread": 0.04,
                "effect_size": 0.03,
                "num_seeds": 20,
            },
            "OP6": {
                "avg_route_distance": 0.03,
                "avg_behavior_distance": 0.04,
                "ci_95_low": 0.02,
                "ci_95_high": 0.05,
                "forced_z_performance_spread": 0.05,
                "effect_size": 0.04,
                "num_seeds": 20,
            },
            "OP7": {
                "avg_route_distance": 0.01,
                "avg_behavior_distance": 0.005,
                "ci_95_low": 0.0,
                "ci_95_high": 0.02,
                "forced_z_performance_spread": 0.01,
                "effect_size": 0.01,
                "num_seeds": 20,
            },
        }
        result = evaluate_behavioral_realization(cfg, latent, op_reports, boundary_eval_enabled=True)
        self.assertEqual(result.details["matched_seed_semantics"], GATE_STATUS_PASS)
        self.assertEqual(result.details["macro_profile"], GATE_STATUS_FAIL)
        self.assertEqual(result.status, GATE_STATUS_PASS)

    def test_semantics_fail_macro_pass(self):
        cfg = _v6i2_cfg()
        latent = SimpleNamespace(
            macro_pair_jsd_ema=np.full(6, 0.001, dtype=np.float32),
            macro_pair_jsd_valid_updates=5,
        )
        op_reports = {
            "OP5": {
                "avg_route_distance": 0.001,
                "avg_behavior_distance": 0.001,
                "ci_95_low": -0.01,
                "ci_95_high": 0.01,
                "forced_z_performance_spread": 0.01,
                "effect_size": 0.001,
                "num_seeds": 20,
            },
            "OP6": {
                "avg_route_distance": 0.001,
                "avg_behavior_distance": 0.001,
                "ci_95_low": -0.01,
                "ci_95_high": 0.01,
                "forced_z_performance_spread": 0.01,
                "effect_size": 0.001,
                "num_seeds": 20,
            },
            "OP7": {
                "avg_route_distance": 0.001,
                "avg_behavior_distance": 0.001,
                "ci_95_low": -0.01,
                "ci_95_high": 0.01,
                "forced_z_performance_spread": 0.01,
                "effect_size": 0.001,
                "num_seeds": 20,
            },
        }
        result = evaluate_behavioral_realization(cfg, latent, op_reports, boundary_eval_enabled=True)
        self.assertEqual(result.details["macro_profile"], GATE_STATUS_PASS)
        self.assertEqual(result.details["matched_seed_semantics"], GATE_STATUS_FAIL)
        self.assertEqual(result.status, GATE_STATUS_FAIL)

    def test_insufficient_seed_count_not_run(self):
        cfg = _v6i2_cfg()
        result = evaluate_matched_seed_semantics(
            cfg,
            {
                "OP5": {
                    "avg_route_distance": 0.03,
                    "avg_behavior_distance": 0.01,
                    "ci_95_low": 0.025,
                    "forced_z_performance_spread": 0.04,
                    "effect_size": 0.03,
                    "num_seeds": 5,
                }
            },
        )
        self.assertEqual(result.status, GATE_STATUS_NOT_RUN)

    def test_route_distance_cannot_swallow_task_behavior_floor(self):
        cfg = _v6i2_cfg()
        result = evaluate_matched_seed_semantics(
            cfg,
            {
                "OP5": {
                    "avg_route_distance": 3.0,
                    "avg_behavior_distance": 0.0,
                    "ci_95_low": 2.0,
                    "ci_95_high": 4.0,
                    "forced_z_performance_spread": 0.10,
                    "effect_size": 3.0,
                    "num_seeds": 20,
                },
                "OP6": {
                    "avg_route_distance": 3.0,
                    "avg_behavior_distance": 0.0,
                    "ci_95_low": 2.0,
                    "ci_95_high": 4.0,
                    "forced_z_performance_spread": 0.10,
                    "effect_size": 3.0,
                    "num_seeds": 20,
                },
                "OP7": {
                    "avg_route_distance": 3.0,
                    "avg_behavior_distance": 0.0,
                    "ci_95_low": 2.0,
                    "ci_95_high": 4.0,
                    "forced_z_performance_spread": 0.10,
                    "effect_size": 3.0,
                    "num_seeds": 20,
                },
            },
        )
        self.assertEqual(result.status, GATE_STATUS_FAIL)
        op5 = result.details["opponents"]["OP5"]
        self.assertGreater(op5["aggregate_effect"], cfg.behavioral_aggregate_effect_threshold)
        self.assertFalse(op5["behavior_component_floor_pass"])
        self.assertEqual(op5["semantic_verdict"], "FAIL_COMPONENT_FLOOR")

    def test_checkpoint_includes_fingerprint(self):
        cfg = _v6i2_cfg()
        state = ActorEmaTests()._make_state(cfg)
        payload = latent_state_v6i1_checkpoint(state)
        self.assertIn("gate_config_fingerprint", payload)
        self.assertIn("resolved_gate_config", payload)
        self.assertIn("actor_jsd_margin", payload["resolved_gate_config"])


class ResumeSafetyTests(unittest.TestCase):
    def _make_minimal_state(self, cfg: PPOConfig) -> SimpleNamespace:
        return SimpleNamespace(
            cf_J=np.zeros(4),
            cf_episode_counts=np.zeros(4, dtype=int),
            cf_has_experience=np.zeros(4, dtype=bool),
            cf_return_mean=0.0,
            cf_return_var=1.0,
            pair_jsd_ema=np.zeros(6, dtype=np.float32),
            jsd_gate_consecutive_updates=0,
            pairwise_ema_valid_updates=0,
            pairwise_ema_last_update_step=-1,
            cf_pair_jsd_ema=np.zeros(6, dtype=np.float32),
            cf_pair_jsd_valid_updates=0,
            cf_pair_jsd_last_update_step=-1,
            actor_intervention_consecutive_updates=0,
            macro_pair_jsd_ema=np.zeros(6, dtype=np.float32),
            macro_pair_jsd_valid_updates=0,
            macro_pair_jsd_last_update_step=-1,
            router_optimizer_step_count=0,
            trainer=_trainer_stub(cfg, None),
        )

    def test_protocol_mismatch_rejected(self):
        cfg = _v6i2_cfg()
        state = self._make_minimal_state(cfg)
        payload = latent_state_v6i1_checkpoint(state)
        payload["gate_protocol_version"] = "v6i1_single_macro_intervention"
        with self.assertRaises(ValueError):
            restore_latent_state_v6i1_checkpoint(state, payload)

    def test_v6i2_enforce_resume_missing_fields_rejected(self):
        cfg = _v6i2_cfg()
        state = self._make_minimal_state(cfg)
        payload = {
            "gate_protocol_version": V6I2_GATE_PROTOCOL,
            "cf_J": state.cf_J,
            "cf_episode_counts": state.cf_episode_counts,
            "cf_has_experience": state.cf_has_experience,
            "cf_return_mean": 0.0,
            "cf_return_var": 1.0,
            "pair_jsd_ema": state.pair_jsd_ema,
        }
        with self.assertRaises(ValueError):
            restore_latent_state_v6i1_checkpoint(state, payload)


class TerminalBudgetTests(unittest.TestCase):
    def test_late_phase_a_extends_terminal_step(self):
        from rl.custom_ppo.curriculum.schedule import resolve_schedule

        cfg = _v6i2_cfg()
        schedule = resolve_schedule(cfg)
        self.assertEqual(schedule.terminal_step_if_promoted_at(700_000), 1_300_000)

    def test_startup_banner_shows_effective_terminal(self):
        from rl.custom_ppo.curriculum.schedule import format_staged_curriculum_budget_contract

        cfg = _v6i2_cfg()
        cfg.total_timesteps = 1_250_000
        lines = format_staged_curriculum_budget_contract(cfg)
        self.assertTrue(any("Current effective terminal: 1,250,000" in line for line in lines))

    def test_extension_banner_lines(self):
        from rl.custom_ppo.curriculum.schedule import (
            format_terminal_extension_banner,
            resolve_schedule,
        )

        cfg = _v6i2_cfg()
        schedule = resolve_schedule(cfg)
        lines = format_terminal_extension_banner(
            schedule, phase_a_end_step=700_000, effective_terminal=1_300_000
        )
        self.assertTrue(any("700,000" in line for line in lines))
        self.assertTrue(any("1,300,000" in line for line in lines))

    def test_overall_promotion_uses_v6i2_families(self):
        cfg = _v6i2_cfg()
        families = gate_family_names(cfg)
        results = {name: GateFamilyResult(status=GATE_STATUS_PASS) for name in families}
        self.assertTrue(overall_gate_passed_for_promotion(results, mode="enforce", families=families))


if __name__ == "__main__":
    unittest.main()
