import unittest

import torch

from rl.global_state import GLOBAL_STATE_DIM
from rl.latent_marl import (
    LatentConditionedActor,
    StrategyEncoder,
    expected_strategy_switch_penalty,
)


class LatentStrategyAlignmentTests(unittest.TestCase):
    def test_strategy_encoder_matches_paper_mlp_shape(self):
        encoder = StrategyEncoder(state_dim=GLOBAL_STATE_DIM, latent_k=4, hidden=128)
        self.assertIsInstance(encoder.net[1], torch.nn.ReLU)
        self.assertIsInstance(encoder.net[3], torch.nn.ReLU)
        x = torch.randn(3, GLOBAL_STATE_DIM)
        y = encoder(x)
        self.assertEqual(tuple(y.shape), (3, 4))

    def test_actor_is_decentralized_per_agent_given_shared_z(self):
        torch.manual_seed(7)
        actor = LatentConditionedActor((7, 20, 20), vec_dim=18, latent_k=4, action_dim=55)
        actor.eval()
        grid = torch.rand(2, 7, 20, 20)
        vec = torch.rand(2, 18)
        z_idx = torch.tensor([2, 2])
        changed_grid = grid.clone()
        changed_vec = vec.clone()
        changed_grid[1] = torch.rand(7, 20, 20)
        changed_vec[1] = torch.rand(18)

        with torch.no_grad():
            logits_a = actor(grid, vec, z_idx)
            logits_b = actor(changed_grid, changed_vec, z_idx)

        self.assertTrue(torch.allclose(logits_a[0], logits_b[0], atol=1e-6))
        self.assertFalse(torch.allclose(logits_a[1], logits_b[1], atol=1e-6))

    def test_persistence_penalty_is_low_when_previous_strategy_is_likely(self):
        logits = torch.tensor(
            [
                [5.0, -2.0, -2.0, -2.0],
                [-2.0, 5.0, -2.0, -2.0],
            ]
        )
        prev = torch.tensor([0, 1])
        penalty = expected_strategy_switch_penalty(logits, prev)
        self.assertEqual(tuple(penalty.shape), (2,))
        self.assertLess(float(penalty.max().item()), 0.01)


if __name__ == "__main__":
    unittest.main()
