"""Regression tests for PPO updater review fixes (recurrent replay, KL grad, separation)."""

from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from rl.custom_ppo.ppo_updater import (
    _extract_rollout_resample_subset,
    _policy_z_separation_loss,
    set_model_requires_grad_for_phase,
)
from rl.latent_losses import strategy_kl_consecutive_loss
from rl.ppo_core import TensorDictRolloutBuffer


class ConsecutiveKlGradientTests(unittest.TestCase):
    def test_kl_loss_backprops_into_current_logits(self) -> None:
        current = torch.tensor([[2.0, 0.0, -1.0], [0.5, 0.5, 0.0]], requires_grad=True)
        previous = torch.tensor([[0.0, 1.0, -1.0], [0.2, 0.3, 0.1]])
        valid = torch.tensor([True, True])
        loss, _ = strategy_kl_consecutive_loss(
            current, previous, valid, coef=0.1
        )
        loss.backward()
        self.assertIsNotNone(current.grad)
        self.assertGreater(float(current.grad.abs().sum().item()), 0.0)

    def test_rollout_logits_side_is_detached_from_graph(self) -> None:
        attached = torch.tensor([[2.0, 0.0, -1.0]], requires_grad=True)
        detached = attached.detach()
        previous = torch.tensor([[0.0, 1.0, -1.0]])
        valid = torch.tensor([True])
        attached_loss, _ = strategy_kl_consecutive_loss(attached, previous, valid, coef=0.1)
        detached_loss, _ = strategy_kl_consecutive_loss(detached, previous, valid, coef=0.1)
        attached_loss.backward()
        self.assertIsNotNone(attached.grad)
        self.assertGreater(float(attached.grad.abs().sum().item()), 0.0)
        self.assertFalse(detached_loss.requires_grad)


class SeparationHingeTests(unittest.TestCase):
    def test_partial_collapse_stays_penalized(self) -> None:
        class PartialCollapseModel(nn.Module):
            n_agents = 1
            per_agent_action_dims = (3,)

            def policy_logits(self, obs, z_idx=None):
                z = z_idx.long().reshape(-1).clamp(min=0, max=3)
                logits = torch.zeros((int(z.shape[0]), 3), dtype=torch.float32)
                # z0 and z1 differ strongly; z2 and z3 match z0.
                logits[z == 0] = torch.tensor([4.0, -4.0, -4.0])
                logits[z == 1] = torch.tensor([-4.0, 4.0, -4.0])
                logits[z == 2] = torch.tensor([4.0, -4.0, -4.0])
                logits[z == 3] = torch.tensor([4.0, -4.0, -4.0])
                return logits

            @staticmethod
            def _mask_logits(logits, mask):
                return logits

        obs = {"mask": torch.ones((4, 3), dtype=torch.float32)}
        z_idx = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        loss, stats = _policy_z_separation_loss(
            PartialCollapseModel(),
            obs,
            z_idx,
            latent_k=4,
            margin=0.02,
        )
        self.assertGreater(float(loss.item()), 0.0)
        self.assertLess(float(stats["min_jsd"].item()), 0.02)


class RolloutResampleExtractTests(unittest.TestCase):
    def test_missing_global_state_raises(self) -> None:
        buffer = TensorDictRolloutBuffer(buffer_size=4, n_envs=1)
        buffer.pos = 2
        with self.assertRaises(KeyError):
            _extract_rollout_resample_subset(buffer, require_selector_hidden=False)

    def test_no_resample_rows_reports_skip_reason(self) -> None:
        buffer = TensorDictRolloutBuffer(buffer_size=4, n_envs=1)
        buffer.register_field("global_state", (5,))
        buffer.register_field("z_resampled", (), dtype=torch.bool)
        buffer.pos = 2
        states, hidden, reason = _extract_rollout_resample_subset(
            buffer, require_selector_hidden=False
        )
        self.assertIsNone(states)
        self.assertIsNone(hidden)
        self.assertEqual(reason, "no_resample_rows")


class PhaseValidationTests(unittest.TestCase):
    def test_unknown_phase_raises(self) -> None:
        model = nn.Module()
        model.actor_cnn = nn.Linear(2, 2)
        with self.assertRaises(ValueError):
            set_model_requires_grad_for_phase(model, "B_warmup")


if __name__ == "__main__":
    unittest.main()
