"""Tests for per-pathway actor gradient diagnostics."""

from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from rl.custom_ppo.update.actor_pathway_diagnostics import (
    PATHWAY_GROUP_NAMES,
    actor_pathway_grad_diagnostics_for_model,
    collect_actor_pathway_parameters,
)


class _PathwayModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.actor_cnn = nn.Linear(4, 8)
        self.latent_actor = nn.Module()
        self.latent_actor.strategy_embedding = nn.Embedding(4, 16)
        self.latent_actor.body = nn.Sequential(nn.Linear(8 + 16, 32), nn.ReLU())
        self.latent_actor.action_head = nn.Linear(32, 6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        emb = self.latent_actor.strategy_embedding(z)
        h = self.latent_actor.body(torch.cat([x, emb], dim=-1))
        return self.latent_actor.action_head(h)


class ActorPathwayDiagnosticsTests(unittest.TestCase):
    def test_collect_groups_expected_modules(self) -> None:
        model = _PathwayModel()
        groups = collect_actor_pathway_parameters(model)
        self.assertTrue(groups["z_embed"])
        self.assertTrue(groups["trunk"])
        self.assertTrue(groups["action_head"])
        self.assertTrue(groups["actor_cnn"])
        self.assertEqual(groups["film"], [])

    def test_cf_and_ppo_grads_are_independent(self) -> None:
        model = _PathwayModel()
        x = torch.randn(3, 8, requires_grad=True)
        out = model(x)
        ppo_loss = out.pow(2).mean()
        cf_loss = out.sum()
        stats = actor_pathway_grad_diagnostics_for_model(
            model=model,
            scaled_cf_loss=0.5 * cf_loss,
            ppo_actor_loss=ppo_loss,
        )
        for group in PATHWAY_GROUP_NAMES:
            self.assertIn(f"{group}_grad_from_cf", stats)
            self.assertIn(f"{group}_grad_from_ppo", stats)
        self.assertGreater(stats["action_head_grad_from_cf"], 0.0)
        self.assertGreater(stats["action_head_grad_from_ppo"], 0.0)


if __name__ == "__main__":
    unittest.main()
