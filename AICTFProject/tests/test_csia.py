"""Tests for v5i9 Causal Strategic Impact Advantage math and gating."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from rl.csia import (
    CSIARewardModel,
    analyze_csia,
    build_payoff_matrix,
    centered_interaction,
)


def _matrix(rows: list[dict[str, object]]) -> tuple[dict[str, dict[int, float]], dict[str, dict[int, int]]]:
    return build_payoff_matrix(rows)


class CSIAMatrixTests(unittest.TestCase):
    def test_equal_latent_performance_has_zero_interaction(self) -> None:
        payoffs, _counts = _matrix(
            [
                {"opponent": "OP5", "mode": "fixed_z", "z": 0, "blue_win_rate": 0.5, "n_episodes_touched": 8},
                {"opponent": "OP5", "mode": "fixed_z", "z": 1, "blue_win_rate": 0.5, "n_episodes_touched": 8},
                {"opponent": "OP6", "mode": "fixed_z", "z": 0, "blue_win_rate": 0.5, "n_episodes_touched": 8},
                {"opponent": "OP6", "mode": "fixed_z", "z": 1, "blue_win_rate": 0.5, "n_episodes_touched": 8},
            ]
        )
        centered, strength = centered_interaction(payoffs)
        self.assertEqual(set(centered), {"OP5", "OP6"})
        self.assertAlmostEqual(strength, 0.0, places=8)

    def test_one_globally_superior_latent_has_near_zero_interaction(self) -> None:
        payoffs, _counts = _matrix(
            [
                {"opponent": "OP5", "mode": "fixed_z", "z": 0, "blue_win_rate": 0.7, "n_episodes_touched": 8},
                {"opponent": "OP5", "mode": "fixed_z", "z": 1, "blue_win_rate": 0.5, "n_episodes_touched": 8},
                {"opponent": "OP6", "mode": "fixed_z", "z": 0, "blue_win_rate": 0.7, "n_episodes_touched": 8},
                {"opponent": "OP6", "mode": "fixed_z", "z": 1, "blue_win_rate": 0.5, "n_episodes_touched": 8},
            ]
        )
        _centered, strength = centered_interaction(payoffs)
        self.assertAlmostEqual(strength, 0.0, places=8)

    def test_opponents_preferring_different_latents_has_positive_interaction(self) -> None:
        payoffs, counts = _matrix(
            [
                {"opponent": "OP5", "mode": "fixed_z", "z": 0, "blue_win_rate": 0.8, "n_episodes_touched": 8},
                {"opponent": "OP5", "mode": "fixed_z", "z": 1, "blue_win_rate": 0.2, "n_episodes_touched": 8},
                {"opponent": "OP6", "mode": "fixed_z", "z": 0, "blue_win_rate": 0.2, "n_episodes_touched": 8},
                {"opponent": "OP6", "mode": "fixed_z", "z": 1, "blue_win_rate": 0.8, "n_episodes_touched": 8},
            ]
        )
        analysis = analyze_csia(
            payoffs,
            counts,
            baselines={"OP5": 0.5, "OP6": 0.5},
            behavior_spread_by_opp={"OP5": 0.2, "OP6": 0.2},
            min_behavior_spread=0.1,
            min_interaction_strength=0.05,
            quality_floor_delta=0.4,
        )
        self.assertGreater(analysis.specialization_strength, 0.05)
        self.assertEqual(analysis.oracle_best_z_per_opponent, {"OP5": 0, "OP6": 1})
        self.assertTrue(analysis.gates.passed)

    def test_missing_cells_do_not_crash(self) -> None:
        payoffs, counts = _matrix(
            [
                {"opponent": "OP5", "mode": "fixed_z", "z": 0, "blue_win_rate": 0.7, "n_episodes_touched": 8},
                {"opponent": "OP6", "mode": "fixed_z", "z": 1, "blue_win_rate": 0.6, "n_episodes_touched": 8},
            ]
        )
        analysis = analyze_csia(payoffs, counts)
        self.assertEqual(analysis.payoff_cells, 2)
        self.assertGreaterEqual(analysis.specialization_strength, 0.0)


class CSIAGatingTests(unittest.TestCase):
    def test_csia_reward_disabled_before_gates_pass(self) -> None:
        payoffs, counts = _matrix(
            [
                {"opponent": "OP5", "mode": "fixed_z", "z": 0, "blue_win_rate": 0.5, "n_episodes_touched": 8},
                {"opponent": "OP5", "mode": "fixed_z", "z": 1, "blue_win_rate": 0.5, "n_episodes_touched": 8},
                {"opponent": "OP6", "mode": "fixed_z", "z": 0, "blue_win_rate": 0.5, "n_episodes_touched": 8},
                {"opponent": "OP6", "mode": "fixed_z", "z": 1, "blue_win_rate": 0.5, "n_episodes_touched": 8},
            ]
        )
        model = CSIARewardModel(enabled=True, reward_coef=1.0, require_gates=True)
        model.analysis = analyze_csia(
            payoffs,
            counts,
            baselines={"OP5": 0.5, "OP6": 0.5},
            behavior_spread_by_opp={"OP5": 0.2, "OP6": 0.2},
        )
        bonus = model.bonus(
            torch.tensor([4, 5]),
            torch.tensor([0, 1]),
            device="cpu",
            update=0,
        )
        self.assertFalse(model.bonus_active)
        self.assertTrue(torch.equal(bonus, torch.zeros_like(bonus)))

    def test_raw_behavior_diversity_alone_cannot_activate(self) -> None:
        payoffs, counts = _matrix(
            [
                {"opponent": "OP5", "mode": "fixed_z", "z": 0, "blue_win_rate": 0.7, "n_episodes_touched": 8},
                {"opponent": "OP5", "mode": "fixed_z", "z": 1, "blue_win_rate": 0.5, "n_episodes_touched": 8},
                {"opponent": "OP6", "mode": "fixed_z", "z": 0, "blue_win_rate": 0.7, "n_episodes_touched": 8},
                {"opponent": "OP6", "mode": "fixed_z", "z": 1, "blue_win_rate": 0.5, "n_episodes_touched": 8},
            ]
        )
        analysis = analyze_csia(
            payoffs,
            counts,
            baselines={"OP5": 0.55, "OP6": 0.55},
            behavior_spread_by_opp={"OP5": 0.3, "OP6": 0.3},
            quality_floor_delta=0.10,
        )
        self.assertTrue(analysis.gates.behavior_spread)
        self.assertFalse(analysis.gates.interaction_strength)
        self.assertFalse(analysis.gates.passed)

    def test_router_metrics_compare_router_random_and_oracle(self) -> None:
        payoffs, counts = _matrix(
            [
                {"opponent": "OP5", "mode": "fixed_z", "z": 0, "blue_win_rate": 0.8, "n_episodes_touched": 10},
                {"opponent": "OP5", "mode": "fixed_z", "z": 1, "blue_win_rate": 0.2, "n_episodes_touched": 10},
            ]
        )
        analysis = analyze_csia(
            payoffs,
            counts,
            baselines={"OP5": 0.6},
            behavior_spread_by_opp={"OP5": 0.2},
            quality_floor_delta=0.5,
        )
        self.assertAlmostEqual(analysis.router_oracle_gap, 0.2, places=8)
        self.assertAlmostEqual(analysis.routing_gain, 0.1, places=8)
        self.assertAlmostEqual(analysis.regret_weighted_routing_score, 1.0 / 3.0, places=8)

    def test_turning_csia_off_reproduces_zero_bonus(self) -> None:
        payoffs, counts = _matrix(
            [
                {"opponent": "OP5", "mode": "fixed_z", "z": 0, "blue_win_rate": 0.8, "n_episodes_touched": 8},
                {"opponent": "OP5", "mode": "fixed_z", "z": 1, "blue_win_rate": 0.2, "n_episodes_touched": 8},
                {"opponent": "OP6", "mode": "fixed_z", "z": 0, "blue_win_rate": 0.2, "n_episodes_touched": 8},
                {"opponent": "OP6", "mode": "fixed_z", "z": 1, "blue_win_rate": 0.8, "n_episodes_touched": 8},
            ]
        )
        model = CSIARewardModel(enabled=False, reward_coef=1.0)
        model.analysis = analyze_csia(
            payoffs,
            counts,
            baselines={"OP5": 0.5, "OP6": 0.5},
            behavior_spread_by_opp={"OP5": 0.2, "OP6": 0.2},
            quality_floor_delta=0.4,
        )
        bonus = model.bonus(
            torch.tensor([4, 5]),
            torch.tensor([0, 1]),
            device="cpu",
            update=0,
        )
        self.assertFalse(model.bonus_active)
        self.assertTrue(torch.equal(bonus, torch.zeros_like(bonus)))

    def test_probe_interval_zero_survives_config_resolution(self) -> None:
        cfg = SimpleNamespace(csia_enabled=True, csia_reward_coef=0.02, csia_probe_interval=0)
        model = CSIARewardModel.from_config(cfg)
        self.assertEqual(model.probe_interval, 0)

    def test_csia_bonus_uses_centered_signal_after_gates_pass(self) -> None:
        payoffs, counts = _matrix(
            [
                {"opponent": "OP5", "mode": "fixed_z", "z": 0, "blue_win_rate": 0.8, "n_episodes_touched": 8},
                {"opponent": "OP5", "mode": "fixed_z", "z": 1, "blue_win_rate": 0.2, "n_episodes_touched": 8},
                {"opponent": "OP6", "mode": "fixed_z", "z": 0, "blue_win_rate": 0.2, "n_episodes_touched": 8},
                {"opponent": "OP6", "mode": "fixed_z", "z": 1, "blue_win_rate": 0.8, "n_episodes_touched": 8},
            ]
        )
        model = CSIARewardModel(enabled=True, reward_coef=0.5)
        model.analysis = analyze_csia(
            payoffs,
            counts,
            baselines={"OP5": 0.5, "OP6": 0.5},
            behavior_spread_by_opp={"OP5": 0.2, "OP6": 0.2},
            quality_floor_delta=0.4,
        )
        bonus = model.bonus(
            torch.tensor([4, 5]),
            torch.tensor([0, 0]),
            device="cpu",
            update=0,
        )
        self.assertTrue(model.bonus_active)
        self.assertGreater(float(bonus[0]), 0.0)
        self.assertLess(float(bonus[1]), 0.0)


if __name__ == "__main__":
    unittest.main()
