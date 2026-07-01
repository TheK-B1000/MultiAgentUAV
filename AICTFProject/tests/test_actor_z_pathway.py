"""Tests for actor-z causal pathway tracing and CF leverage fixes."""

from __future__ import annotations

import unittest

import torch

from rl.custom_ppo.update.actor_z_pathway import trace_actor_z_pathway
from rl.custom_ppo.v6i1_cf_loss import actor_diagnostic_grad_norm, v6i1_cf_separation_loss
from tests.test_shared_actor_composition import _build_latent_model, _fixed_obs


class ActorZPathwayTraceTests(unittest.TestCase):
    def test_concat_pathway_has_nonzero_embed_and_logit_deltas_at_init(self) -> None:
        model = _build_latent_model(seed=7)
        obs = _fixed_obs(batch=3)
        report = trace_actor_z_pathway(model, obs, z_a=0, z_b=1)
        self.assertEqual(report.conditioning_mode, "concat")
        stage_by_name = {stage.name: stage for stage in report.stages}
        self.assertGreater(stage_by_name["embed"].pair_mean_l2, 0.0)
        self.assertGreater(stage_by_name["logits"].pair_mean_l2, 0.0)
        self.assertGreater(report.logits_pairwise_jsd_mean, 0.0)

    def test_cf_loss_with_detached_locals_gradients_z_embedding(self) -> None:
        model = _build_latent_model(seed=11)
        obs = _fixed_obs(batch=4)
        obs_batch = {
            "grid": obs["grid"],
            "vec": obs["vec"],
            "agent_mask": obs["agent_mask"],
            "mask": obs["mask"],
        }
        import numpy as np

        model.train()
        for param in model.parameters():
            param.requires_grad = True
        loss, _ = v6i1_cf_separation_loss(
            model,
            obs_batch,
            latent_k=4,
            margin=0.01,
            competence=np.ones(4, dtype=np.float32),
            competence_ready=True,
        )
        self.assertTrue(bool(loss.requires_grad))
        embed_grad = actor_diagnostic_grad_norm(
            loss,
            [model.latent_actor.strategy_embedding.weight],
        )
        self.assertGreater(embed_grad, 0.0)

    def test_cf_loss_requires_grad_telemetry_uses_scaled_tensor(self) -> None:
        from rl.custom_ppo.update.loss_result import LossComponent

        live = torch.tensor(0.5, requires_grad=True)
        detached = live.detach()
        component = LossComponent(
            name="separation",
            scaled_loss=live,
            raw_value=detached,
            active=True,
        )
        self.assertFalse(bool(component.raw_value.requires_grad))
        self.assertTrue(bool(component.scaled_loss.requires_grad))


if __name__ == "__main__":
    unittest.main()
