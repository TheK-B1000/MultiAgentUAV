"""Contracts for the suite Fully Shared+z distillation student."""
from __future__ import annotations

import unittest

import torch

from rl.suite_fully_shared_distill import (
    assert_fully_shared_structure,
    fully_shared_model_kwargs,
)
from rl.custom_ppo.policy import SharedActorCentralizedCritic
from tests.test_shared_actor_composition import _action_space, _fixed_obs, _obs_space


class FullySharedStructureTests(unittest.TestCase):
    def test_kwargs_force_concat_and_forbid_private_branches(self):
        kw = fully_shared_model_kwargs({"actor_cnn_feature_dim": 128, "entity_repair_enabled": False})
        self.assertEqual(kw["latent_k"], 2)
        self.assertEqual(kw["z_embed_dim"], 16)
        self.assertEqual(kw["latent_actor_conditioning"], "concat")
        self.assertFalse(kw["strategy_encoder_enabled"])
        self.assertFalse(kw["enable_actor_z_film"])
        self.assertFalse(kw["exp2c_mode_specific_action_heads"])

    def test_single_network_z_changes_logits(self):
        torch.manual_seed(0)
        kw = fully_shared_model_kwargs({"actor_cnn_feature_dim": 32, "actor_hidden_dim": 32})
        model = SharedActorCentralizedCritic(_obs_space(), _action_space(), **kw)
        assert_fully_shared_structure(model)
        obs = _fixed_obs(2)
        z0 = torch.zeros(2, dtype=torch.long)
        z1 = torch.ones(2, dtype=torch.long)
        a = model.policy_logits(obs, z_idx=z0)
        b = model.policy_logits(obs, z_idx=z1)
        self.assertGreater(float((a - b).abs().max()), 0.0)
        self.assertFalse(hasattr(model, "branch"))


if __name__ == "__main__":
    unittest.main()
