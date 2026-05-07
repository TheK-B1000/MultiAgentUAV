from __future__ import annotations

import unittest

import torch

from rl.global_state import GLOBAL_STATE_DIM
from rl.networks import CNNEncoder, CentralizedCritic, PPOPolicy


class NetworkTests(unittest.TestCase):
    def test_cnn_encoder_forward_shape(self) -> None:
        encoder = CNNEncoder((7, 20, 20), feature_dim=512)
        obs = torch.rand(4, 7, 20, 20)
        out = encoder(obs)
        self.assertEqual(tuple(out.shape), (4, 512))

    def test_policy_extra_hook_forward_shape(self) -> None:
        policy = PPOPolicy((7, 20, 20), action_dim=11, feature_dim=128, extra_dim=16)
        obs = torch.rand(4, 7, 20, 20)
        extra = torch.rand(4, 16)
        logits = policy(obs, extra=extra)
        self.assertEqual(tuple(logits.shape), (4, 11))

    def test_centralized_critic_extra_hook_forward_shape(self) -> None:
        critic = CentralizedCritic(global_state_dim=14, hidden_dim=64, extra_dim=8)
        global_state = torch.rand(4, 14)
        extra = torch.rand(4, 8)
        values = critic(global_state, extra=extra)
        self.assertEqual(tuple(values.shape), (4, 1))

    def test_centralized_critic_matches_production_global_state_dim(self) -> None:
        """The CTDE critic hook uses the documented production global state."""
        critic = CentralizedCritic(global_state_dim=GLOBAL_STATE_DIM, hidden_dim=32)
        global_state = torch.rand(2, GLOBAL_STATE_DIM)
        values = critic(global_state)
        self.assertEqual(tuple(values.shape), (2, 1))

    def test_gradients_flow_through_networks(self) -> None:
        encoder = CNNEncoder((7, 20, 20), feature_dim=64)
        policy = PPOPolicy((7, 20, 20), action_dim=5, feature_dim=64)
        critic = CentralizedCritic(global_state_dim=GLOBAL_STATE_DIM, hidden_dim=32)

        obs = torch.rand(8, 7, 20, 20) + 0.1
        global_state = torch.rand(8, GLOBAL_STATE_DIM) + 0.1
        loss = encoder(obs).sum() + policy(obs).sum() + critic(global_state).sum()
        loss.backward()

        for module in (encoder, policy, critic):
            for name, param in module.named_parameters():
                self.assertIsNotNone(param.grad, name)
                self.assertTrue(torch.isfinite(param.grad).all(), name)
                self.assertGreater(float(param.grad.abs().sum().item()), 0.0, name)


if __name__ == "__main__":
    unittest.main()
