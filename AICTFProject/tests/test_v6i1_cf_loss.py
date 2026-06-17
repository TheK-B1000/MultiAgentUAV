"""Tests for V6I1 competence-weighted counterfactual separation loss."""

from __future__ import annotations

import unittest

import numpy as np
import torch
import torch.nn as nn

from rl.custom_ppo.v6i1_cf_loss import v6i1_cf_separation_loss


class _CfModel(nn.Module):
    n_agents = 1
    per_agent_action_dims = (3,)

    def policy_logits(self, obs, z_idx=None):
        z = z_idx.long().reshape(-1).clamp(min=0, max=3)
        logits = torch.zeros((int(z.shape[0]), 3), dtype=torch.float32)
        logits[z == 0] = torch.tensor([4.0, -4.0, -4.0])
        logits[z == 1] = torch.tensor([-4.0, 4.0, -4.0])
        logits[z == 2] = torch.tensor([4.0, -4.0, -4.0])
        logits[z == 3] = torch.tensor([4.0, -4.0, -4.0])
        return logits

    @staticmethod
    def _mask_logits(logits, mask):
        return logits


class V6I1CfLossTests(unittest.TestCase):
    def test_competence_weights_reduce_penalty_on_collapsed_pairs(self) -> None:
        obs = {"mask": torch.ones((2, 3), dtype=torch.float32)}
        competence = np.array([1.0, 1.0, 0.01, 0.01], dtype=np.float32)
        loss_weighted, _ = v6i1_cf_separation_loss(
            _CfModel(),
            obs,
            latent_k=4,
            margin=0.02,
            competence=competence,
            competence_ready=True,
        )
        loss_uniform, _ = v6i1_cf_separation_loss(
            _CfModel(),
            obs,
            latent_k=4,
            margin=0.02,
            competence=np.ones(4, dtype=np.float32),
            competence_ready=True,
        )
        self.assertLess(float(loss_weighted.item()), float(loss_uniform.item()))

    def test_partial_collapse_stays_positive(self) -> None:
        obs = {"mask": torch.ones((2, 3), dtype=torch.float32)}
        loss, stats = v6i1_cf_separation_loss(
            _CfModel(),
            obs,
            latent_k=4,
            margin=0.02,
            competence=np.ones(4, dtype=np.float32),
            competence_ready=False,
        )
        self.assertGreater(float(loss.item()), 0.0)
        self.assertLess(float(stats["min_jsd"].item()), 0.02)


if __name__ == "__main__":
    unittest.main()
