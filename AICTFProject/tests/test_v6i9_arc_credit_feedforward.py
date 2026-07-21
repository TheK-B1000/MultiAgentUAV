"""Tests for the feedforward running-mean arc-credit treatment (v6i9 A/B).

Pins:
* The treatment differs from the feedforward control by EXACTLY the approved
  keys: ``latent_arc_credit_enabled``, ``latent_arc_credit_baseline``,
  ``latent_strategy_ppo_coef`` (magnet removal) and ``run_tag``.
* The treatment keeps the feedforward router architecture and the control's
  "keep identical" knobs.
* Alias resolution equality.
* The one-update smoke-gate evaluation logic.
* Checkpoint persistence of the arc running-mean EMA.
"""
from __future__ import annotations

import dataclasses
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from rl.config.ppo_config import PPOConfig
from rl.custom_ppo.diagnostics.arc_credit_smoke import evaluate_arc_credit_treatment_gates
from rl.custom_ppo.latent.checkpoint import _component_router_runtime, _restore_flat_v6i1_fields
from rl.presets import apply_preset

CONTROL = "v6i9_mapaware_router_feedforward_hardpool"
TREATMENT = "v6i9_arc_credit_running_mean_feedforward_hardpool"

APPROVED_DIFF_KEYS = {
    "latent_arc_credit_enabled",
    "latent_arc_credit_baseline",
    "latent_strategy_ppo_coef",
    "run_tag",
}

KEEP_IDENTICAL_KEYS = (
    "recurrent_selector_hidden_dim",
    "recurrent_seq_len",
    "recurrent_burn_in",
    "latent_resample_every_n",  # strategy interval
    "learning_rate",
    "router_ent_coef",
    "latent_k",
    "router_reward_enabled",
    "router_freeze_actor",
    "latent_arc_credit_min_len",
    "seed",
    "total_timesteps",
)


def _resolved(name: str) -> dict:
    return dataclasses.asdict(apply_preset(PPOConfig(), name))


class TreatmentPresetDiffTests(unittest.TestCase):
    def test_diff_is_exactly_approved_keys(self) -> None:
        ctrl = _resolved(CONTROL)
        trt = _resolved(TREATMENT)
        diff = {k for k in set(ctrl) | set(trt) if ctrl.get(k) != trt.get(k)}
        self.assertEqual(diff, APPROVED_DIFF_KEYS)

    def test_arc_credit_field_values(self) -> None:
        trt = _resolved(TREATMENT)
        self.assertTrue(trt["latent_arc_credit_enabled"])
        self.assertEqual(trt["latent_arc_credit_baseline"], "running_mean")
        self.assertEqual(trt["latent_strategy_ppo_coef"], 0.0)

    def test_keeps_feedforward_architecture_and_control_knobs(self) -> None:
        ctrl = _resolved(CONTROL)
        trt = _resolved(TREATMENT)
        self.assertEqual(trt["recurrent_selector_hidden_dim"], 0)
        for key in KEEP_IDENTICAL_KEYS:
            self.assertEqual(trt.get(key), ctrl.get(key), f"key changed unexpectedly: {key}")

    def test_min_len_left_at_control_default(self) -> None:
        self.assertEqual(_resolved(TREATMENT)["latent_arc_credit_min_len"], 32)

    def test_alias_resolution_equality(self) -> None:
        base = _resolved(TREATMENT)
        for alias in (
            "v6i9_arc_credit_feedforward",
            "plan_faithful_latent_v6i9_arc_credit_running_mean_feedforward_hardpool",
        ):
            self.assertEqual(_resolved(alias), base)


class SmokeGateTests(unittest.TestCase):
    def _cfg(self) -> SimpleNamespace:
        return SimpleNamespace(
            latent_arc_credit_enabled=True,
            latent_strategy_ppo_coef=0.0,
            latent_arc_credit_baseline="running_mean",
            recurrent_selector_hidden_dim=0,
        )

    def _healthy_stats(self) -> dict:
        return {
            "latent_arc_count": 6.0,
            "latent_arc_finalized_count": 6.0,
            "latent_arc_mean_return": 1.5,
            "latent_arc_baseline_mean": 1.2,
            "latent_arc_raw_advantage_mean": 0.3,
            "latent_arc_raw_advantage_std": 0.9,
            "latent_arc_positive_fraction": 0.5,
            "latent_arc_running_mean_count": 6.0,
            "latent_arc_running_mean_value": 1.2,
            "q_phi_grad_norm": 0.4,
            "latent_arc_grad_norm": 0.4,
        }

    def test_all_gates_pass_for_healthy_update(self) -> None:
        report = evaluate_arc_credit_treatment_gates(
            cfg=self._cfg(),
            arc_stats=self._healthy_stats(),
            router_decision_count=12,
            frozen_hash_before="A",
            frozen_hash_after="A",
            router_hash_before="R0",
            router_hash_after="R1",
        )
        self.assertTrue(report["gates"]["all_passed"])
        self.assertEqual(report["telemetry"]["running_mean_update_count"], 6.0)

    def test_frozen_actor_change_fails_gate(self) -> None:
        report = evaluate_arc_credit_treatment_gates(
            cfg=self._cfg(),
            arc_stats=self._healthy_stats(),
            router_decision_count=12,
            frozen_hash_before="A",
            frozen_hash_after="B",  # frozen weights moved -> fail
            router_hash_before="R0",
            router_hash_after="R1",
        )
        self.assertFalse(report["gates"]["frozen_actor_z_unchanged"])
        self.assertFalse(report["gates"]["all_passed"])

    def test_router_not_moving_fails_gate(self) -> None:
        report = evaluate_arc_credit_treatment_gates(
            cfg=self._cfg(),
            arc_stats={**self._healthy_stats(), "q_phi_grad_norm": 0.0, "latent_arc_grad_norm": 0.0},
            router_decision_count=12,
            frozen_hash_before="A",
            frozen_hash_after="A",
            router_hash_before="R0",
            router_hash_after="R0",  # router unchanged -> fail
        )
        self.assertFalse(report["gates"]["router_gradients_positive"])

    def test_magnet_still_on_fails_source_gate(self) -> None:
        cfg = self._cfg()
        cfg.latent_strategy_ppo_coef = 0.1  # biased critic channel still active
        report = evaluate_arc_credit_treatment_gates(
            cfg=cfg,
            arc_stats=self._healthy_stats(),
            router_decision_count=12,
            frozen_hash_before="A",
            frozen_hash_after="A",
            router_hash_before="R0",
            router_hash_after="R1",
        )
        self.assertFalse(report["gates"]["arc_credit_source_active"])

    def test_no_decisions_fails_gate(self) -> None:
        report = evaluate_arc_credit_treatment_gates(
            cfg=self._cfg(),
            arc_stats=self._healthy_stats(),
            router_decision_count=0,
            frozen_hash_before="A",
            frozen_hash_after="A",
            router_hash_before="R0",
            router_hash_after="R1",
        )
        self.assertFalse(report["gates"]["valid_router_decisions_positive"])


class CheckpointArcEmaTests(unittest.TestCase):
    def test_arc_running_mean_saved_and_restored(self) -> None:
        state = SimpleNamespace(
            router_optimizer_step_count=3,
            macro_return_running_mean=0.0,
            macro_return_running_count=0,
            arc_return_running_mean=1.234,
            arc_return_running_count=17,
            selector_hidden=None,
            v6i1_episode_rehearsal=None,
        )
        payload = _component_router_runtime(state)
        self.assertAlmostEqual(payload["arc_return_running_mean"], 1.234, places=6)
        self.assertEqual(payload["arc_return_running_count"], 17)

        restored = SimpleNamespace(
            trainer=None,
            arc_return_running_mean=0.0,
            arc_return_running_count=0,
            selector_hidden=None,
            v6i1_episode_rehearsal=None,
        )
        _restore_flat_v6i1_fields(restored, payload)
        self.assertAlmostEqual(restored.arc_return_running_mean, 1.234, places=6)
        self.assertEqual(restored.arc_return_running_count, 17)


if __name__ == "__main__":
    unittest.main()
