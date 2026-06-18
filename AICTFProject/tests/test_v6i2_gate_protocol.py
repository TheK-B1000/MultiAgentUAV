"""Pins v6i2 dual-evidence gate protocol, EMA contracts, and resume safety."""

from __future__ import annotations

import unittest
from dataclasses import asdict
from types import SimpleNamespace
from typing import Any

import numpy as np

from rl.config.ppo_config import PPOConfig
from rl.custom_ppo.curriculum_gates import (
    GATE_FAMILY_NAMES,
    GateFamilyResult,
    GATE_STATUS_FAIL,
    GATE_STATUS_NOT_RUN,
    GATE_STATUS_PASS,
    V6I1CurriculumController,
    overall_gate_passed_for_promotion,
)
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
    resolve_gate_protocol_version,
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


class V6i2PresetTests(unittest.TestCase):
    def test_v6i2_protocol_and_timing_diff_vs_v6i1(self):
        v6i1 = asdict(apply_preset(PPOConfig(), "v6i1"))
        v6i2 = asdict(apply_preset(PPOConfig(), "v6i2"))
        allowed = {
            "experiment_id",
            "gate_protocol_version",
            "phase_a_max_end_fraction",
            "run_tag",
        }
        self.assertEqual({k for k in v6i1 if v6i1[k] != v6i2[k]}, allowed)
        self.assertEqual(v6i2["gate_protocol_version"], V6I2_GATE_PROTOCOL)
        self.assertEqual(v6i2["actor_jsd_consecutive_updates"], 3)

    def test_v6i2_mounts_staged_controller(self):
        cfg = _v6i2_cfg()
        self.assertTrue(is_staged_v6_team_intent_curriculum(cfg))
        self.assertTrue(is_v6i2_gate_protocol(cfg))


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
        state.cf_pair_jsd_valid_updates = 0
        state.cf_pair_jsd_last_update_step = -1
        state.actor_intervention_consecutive_updates = 0
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

    def test_nan_causes_no_mutation_and_resets_streak(self):
        cfg = _v6i2_cfg()
        state = self._make_state(cfg)
        state.actor_intervention_consecutive_updates = 2
        bad = [0.002] * 5 + [float("nan")]
        self.assertFalse(state.update_cf_pair_jsd_ema(bad, 100))
        self.assertEqual(state.actor_intervention_consecutive_updates, 0)

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

    def test_checkpoint_roundtrip(self):
        cfg = _v6i2_cfg()
        state = self._make_state(cfg)
        state.update_cf_pair_jsd_ema([0.002] * 6, 999)
        payload = latent_state_v6i1_checkpoint(state)
        restored = self._make_state(cfg)
        restore_latent_state_v6i1_checkpoint(restored, payload)
        np.testing.assert_allclose(restored.cf_pair_jsd_ema, state.cf_pair_jsd_ema)
        self.assertEqual(restored.cf_pair_jsd_valid_updates, 1)
        self.assertEqual(restored.cf_pair_jsd_last_update_step, 999)


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
    def test_pass_after_required_streak(self):
        cfg = _v6i2_cfg()
        state = SimpleNamespace(
            cf_pair_jsd_ema=np.full(6, 0.0015, dtype=np.float32),
            cf_pair_jsd_valid_updates=5,
            cf_pair_jsd_last_update_step=1000,
            actor_intervention_consecutive_updates=3,
        )
        self.assertEqual(evaluate_actor_intervention(cfg, state).status, GATE_STATUS_PASS)

    def test_four_of_six_fails(self):
        cfg = _v6i2_cfg()
        ema = np.array([0.002, 0.002, 0.002, 0.002, 0.0, 0.0], dtype=np.float32)
        state = SimpleNamespace(
            cf_pair_jsd_ema=ema,
            cf_pair_jsd_valid_updates=2,
            cf_pair_jsd_last_update_step=100,
            actor_intervention_consecutive_updates=3,
        )
        self.assertEqual(evaluate_actor_intervention(cfg, state).status, GATE_STATUS_FAIL)

    def test_insufficient_valid_updates_not_run(self):
        cfg = _v6i2_cfg()
        state = SimpleNamespace(
            cf_pair_jsd_ema=np.zeros(6, dtype=np.float32),
            cf_pair_jsd_valid_updates=0,
            cf_pair_jsd_last_update_step=-1,
            actor_intervention_consecutive_updates=0,
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

    def test_macro_fail_does_not_block_when_semantics_pass(self):
        cfg = _v6i2_cfg()
        latent = SimpleNamespace(
            macro_pair_jsd_ema=np.zeros(6, dtype=np.float32),
            macro_pair_jsd_valid_updates=5,
        )
        macro = evaluate_macro_profile_support(cfg, latent)
        self.assertEqual(macro.status, GATE_STATUS_FAIL)


class SafeguardTests(unittest.TestCase):
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
