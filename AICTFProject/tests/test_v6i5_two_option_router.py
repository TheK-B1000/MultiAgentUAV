import unittest

import torch

from rl.config.ppo_config import PPOConfig
from rl.custom_ppo.latent.router_mask import (
    apply_router_allowed_latent_mask,
    masked_uniform_logits,
    router_effective_latent_k,
)
from rl.presets import apply_preset


class V6I5TwoOptionRouterTests(unittest.TestCase):
    def test_preset_freezes_actor_and_keeps_original_latent_ids(self):
        cfg = apply_preset(PPOConfig(), "v6i5_router_z0_z3")
        self.assertEqual(cfg.latent_k, 4)
        self.assertEqual(cfg.router_allowed_latents, (0, 3))
        self.assertTrue(cfg.router_freeze_actor)
        self.assertTrue(cfg.router_reinitialize_on_load)
        self.assertEqual(cfg.router_context_mode, "current_plus_delta")
        self.assertEqual(cfg.router_context_dimension, 68)
        self.assertEqual(cfg.router_persistence_mode, "expected_switch_detached_previous")
        self.assertEqual(router_effective_latent_k(cfg, cfg.latent_k), 2)

    def test_mask_keeps_z0_z3_and_removes_z1_z2(self):
        cfg = PPOConfig(router_allowed_latents=(0, 3))
        logits = torch.tensor([[4.0, 100.0, 90.0, 3.0]])
        masked = apply_router_allowed_latent_mask(logits, cfg=cfg, latent_k=4)
        probs = torch.softmax(masked, dim=-1)
        self.assertGreater(probs[0, 0].item(), 0.0)
        self.assertEqual(probs[0, 1].item(), 0.0)
        self.assertEqual(probs[0, 2].item(), 0.0)
        self.assertGreater(probs[0, 3].item(), 0.0)

    def test_masked_uniform_is_uniform_over_allowed_original_ids(self):
        cfg = PPOConfig(router_allowed_latents=(0, 3))
        logits = masked_uniform_logits(
            2,
            cfg=cfg,
            latent_k=4,
            device=torch.device("cpu"),
        )
        probs = torch.softmax(logits, dim=-1)
        expected = torch.tensor([[0.5, 0.0, 0.0, 0.5], [0.5, 0.0, 0.0, 0.5]])
        self.assertTrue(torch.allclose(probs, expected))


if __name__ == "__main__":
    unittest.main()
