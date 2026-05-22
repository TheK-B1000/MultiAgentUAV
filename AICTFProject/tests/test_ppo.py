from __future__ import annotations

import math
import unittest

import torch

from rl.custom_ppo import _compose_training_reward_components
from rl.ppo_core import (
    align_next_values_to_rollout_actions,
    compute_gae,
    ppo_policy_loss,
    ppo_value_loss,
)


class PPOCoreTests(unittest.TestCase):
    def test_gae_bootstraps_truncation_without_leaking_across_reset(self) -> None:
        rewards = torch.tensor([[1.0], [1.0]])
        values = torch.tensor([[0.0], [0.0]])
        next_values = torch.tensor([[5.0], [7.0]])
        terminated = torch.tensor([[False], [False]])
        truncated = torch.tensor([[True], [False]])

        advantages, returns = compute_gae(
            rewards,
            values,
            next_values,
            terminated,
            truncated,
            gamma=0.9,
            gae_lambda=0.95,
        )

        self.assertAlmostEqual(float(advantages[1, 0]), 1.0 + 0.9 * 7.0, places=6)
        self.assertAlmostEqual(float(advantages[0, 0]), 1.0 + 0.9 * 5.0, places=6)
        self.assertTrue(torch.allclose(advantages, returns))

    def test_gae_does_not_bootstrap_true_terminal(self) -> None:
        rewards = torch.tensor([[2.0]])
        values = torch.tensor([[0.5]])
        next_values = torch.tensor([[99.0]])
        terminated = torch.tensor([[True]])

        advantages, returns = compute_gae(rewards, values, next_values, terminated, gamma=0.9, gae_lambda=0.95)

        self.assertAlmostEqual(float(advantages[0, 0]), 1.5, places=6)
        self.assertAlmostEqual(float(returns[0, 0]), 2.0, places=6)

    def test_gae_resets_carry_across_latent_z_change(self) -> None:
        rewards = torch.zeros((2, 1))
        values = torch.ones((2, 1))
        next_values = torch.ones((2, 1))
        terminated = torch.zeros((2, 1), dtype=torch.bool)
        latent_z = torch.tensor([[0], [1]], dtype=torch.long)

        adv_reset, _ = compute_gae(
            rewards,
            values,
            next_values,
            terminated,
            gamma=0.9,
            gae_lambda=0.95,
            latent_z=latent_z,
            reset_gae_on_z_change=True,
        )
        adv_cont, _ = compute_gae(
            rewards,
            values,
            next_values,
            terminated,
            gamma=0.9,
            gae_lambda=0.95,
            latent_z=latent_z,
            reset_gae_on_z_change=False,
        )
        d = 0.9 * 1.0 - 1.0
        self.assertAlmostEqual(float(adv_reset[1, 0]), float(d), places=5)
        self.assertAlmostEqual(float(adv_reset[0, 0]), float(d), places=5)
        self.assertAlmostEqual(float(adv_cont[1, 0]), float(d), places=5)
        carry = 0.9 * 0.95 * float(d)
        self.assertAlmostEqual(float(adv_cont[0, 0]), float(d + carry), places=5)
        self.assertGreater(abs(float(adv_cont[0, 0])), abs(float(adv_reset[0, 0])))

    def test_next_values_align_to_collected_rollout_actions_for_continuations(self) -> None:
        values = torch.tensor(
            [
                [1.0, 10.0],
                [2.0, 20.0],
                [3.0, 30.0],
            ]
        )
        bootstraps = torch.tensor(
            [
                [100.0, 1000.0],
                [200.0, 2000.0],
                [300.0, 3000.0],
            ]
        )
        terminated = torch.tensor(
            [
                [False, True],
                [False, False],
                [False, False],
            ]
        )
        truncated = torch.tensor(
            [
                [False, False],
                [True, False],
                [False, False],
            ]
        )

        aligned = align_next_values_to_rollout_actions(values, bootstraps, terminated, truncated)

        expected = torch.tensor(
            [
                [2.0, 1000.0],
                [200.0, 30.0],
                [300.0, 3000.0],
            ]
        )
        self.assertTrue(torch.allclose(aligned, expected))

    def test_policy_clip_objective_ratio_clip_and_sign(self) -> None:
        old_log_prob = torch.zeros(2)
        new_log_prob = torch.log(torch.tensor([1.3, 0.7]))
        advantages = torch.tensor([1.0, -1.0])

        loss, stats = ppo_policy_loss(new_log_prob, old_log_prob, advantages, clip_range=0.2)

        self.assertAlmostEqual(float(loss), -0.2, places=6)
        self.assertTrue(torch.allclose(stats["ratio"], torch.tensor([1.3, 0.7]), atol=1e-6))
        self.assertAlmostEqual(float(stats["clip_fraction"]), 1.0, places=6)

    def test_value_clipping_uses_pessimistic_max_loss(self) -> None:
        new_values = torch.tensor([3.0])
        old_values = torch.tensor([1.0])
        returns = torch.tensor([3.0])

        loss = ppo_value_loss(new_values, old_values, returns, clip_range_vf=0.5)

        self.assertAlmostEqual(float(loss), 2.25, places=6)

    def test_unclipped_value_loss_matches_mse(self) -> None:
        new_values = torch.tensor([1.0, 3.0])
        old_values = torch.tensor([0.0, 0.0])
        returns = torch.tensor([2.0, 1.0])

        loss = ppo_value_loss(new_values, old_values, returns, clip_range_vf=None)

        self.assertAlmostEqual(float(loss), (1.0 + 4.0) / 2.0, places=6)

    def test_training_reward_composition_uses_env_weights_and_scaling(self) -> None:
        reward_component = {
            "reward_terminal": torch.tensor([1.0]),
            "reward_offense": torch.tensor([0.5]),
            "reward_pbrs": torch.tensor([2.0]),
            "reward_team": torch.tensor([1.0]),
            "reward_sparse": torch.tensor([0.5]),
            "reward_failure": torch.tensor([-0.25]),
            "reward_total": torch.tensor([99.0]),
        }

        out = _compose_training_reward_components(
            reward_component,
            dense_weight=0.25,
            reward_scale=2.0,
            reward_clip=1.0,
            shaping_coef=0.5,
            stalemate=torch.tensor([True]),
            stalemate_penalty=-0.08,
        )

        raw = 1.0 + (0.5 * 0.5) - 0.25 + 0.5 + 0.25 * ((2.0 * 0.5) + (1.0 * 0.5)) - 0.08
        expected = torch.tanh(torch.tensor(raw / 2.0))
        self.assertAlmostEqual(float(out["reward_offense"][0]), 0.25, places=6)
        self.assertAlmostEqual(float(out["reward_pbrs"][0]), 1.0, places=6)
        self.assertAlmostEqual(float(out["reward_team"][0]), 0.5, places=6)
        self.assertAlmostEqual(float(out["reward_total"][0]), float(expected), places=6)


    def test_training_reward_composition_adds_per_z_shaping(self) -> None:
        reward_component = {
            "reward_terminal": torch.tensor([1.0]),
            "reward_offense": torch.tensor([0.5]),
            "reward_pbrs": torch.tensor([2.0]),
            "reward_team": torch.tensor([1.0]),
            "reward_sparse": torch.tensor([0.5]),
            "reward_failure": torch.tensor([-0.25]),
            "reward_total": torch.tensor([99.0]),
            "reward_per_z_shaping": torch.tensor([0.15]),
        }

        out = _compose_training_reward_components(
            reward_component,
            dense_weight=0.25,
            reward_scale=2.0,
            reward_clip=1.0,
            shaping_coef=0.5,
            stalemate=torch.tensor([True]),
            stalemate_penalty=-0.08,
        )

        raw = 1.0 + (0.5 * 0.5) - 0.25 + 0.5 + 0.25 * ((2.0 * 0.5) + (1.0 * 0.5)) - 0.08 + 0.15
        expected = torch.tanh(torch.tensor(raw / 2.0))
        self.assertAlmostEqual(float(out["reward_offense"][0]), 0.25, places=6)
        self.assertAlmostEqual(float(out["reward_per_z_shaping"][0]), 0.15, places=6)
        self.assertAlmostEqual(float(out["reward_total"][0]), float(expected), places=6)


if __name__ == "__main__":
    unittest.main()
