import unittest

import numpy as np
import torch
from gymnasium import spaces

from rl.global_state import GLOBAL_STATE_DIM
from rl.latent_marl import (
    LatentMaskedMultiInputPolicy,
    StrategyEncoder,
    expected_strategy_switch_penalty,
)


def _lr_schedule(_: float) -> float:
    return 3e-4


def _make_obs_space(n_agents: int = 2, latent_k: int = 4) -> spaces.Dict:
    return spaces.Dict(
        {
            "grid": spaces.Box(low=0.0, high=1.0, shape=(n_agents, 7, 20, 20), dtype=np.float32),
            "vec": spaces.Box(low=-1.0, high=1.0, shape=(n_agents, 18), dtype=np.float32),
            "agent_mask": spaces.Box(low=0.0, high=1.0, shape=(n_agents,), dtype=np.float32),
            "mask": spaces.Box(low=0.0, high=1.0, shape=(n_agents * (5 + 50),), dtype=np.float32),
            "z_idx": spaces.Box(low=0.0, high=float(latent_k - 1), shape=(1,), dtype=np.float32),
            "z_prev_idx": spaces.Box(low=0.0, high=float(latent_k - 1), shape=(1,), dtype=np.float32),
            "z_onehot": spaces.Box(low=0.0, high=1.0, shape=(latent_k,), dtype=np.float32),
            "z_logits": spaces.Box(low=-50.0, high=50.0, shape=(latent_k,), dtype=np.float32),
            "z_resampled": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            "z_switch": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            "global_state": spaces.Box(low=-np.inf, high=np.inf, shape=(GLOBAL_STATE_DIM,), dtype=np.float32),
        }
    )


def _make_action_space(n_agents: int = 2) -> spaces.MultiDiscrete:
    return spaces.MultiDiscrete([5, 50] * n_agents)


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
        obs_space = _make_obs_space(n_agents=2, latent_k=4)
        act_space = _make_action_space(n_agents=2)
        policy = LatentMaskedMultiInputPolicy(obs_space, act_space, _lr_schedule)
        policy.eval()

        base_obs = {
            "grid": torch.rand(1, 2, 7, 20, 20),
            "vec": torch.rand(1, 2, 18),
            "agent_mask": torch.ones(1, 2),
            "mask": torch.ones(1, 2 * (5 + 50)),
            "z_idx": torch.tensor([[2.0]]),
            "z_prev_idx": torch.tensor([[2.0]]),
            "z_onehot": torch.tensor([[0.0, 0.0, 1.0, 0.0]]),
            "z_logits": torch.zeros(1, 4),
            "z_resampled": torch.zeros(1, 1),
            "z_switch": torch.zeros(1, 1),
            "global_state": torch.zeros(1, GLOBAL_STATE_DIM),
        }

        changed_other_agent = {k: v.clone() for k, v in base_obs.items()}
        changed_other_agent["grid"][:, 1] = torch.rand(1, 7, 20, 20)
        changed_other_agent["vec"][:, 1] = torch.rand(1, 18)

        with torch.no_grad():
            logits_a = policy._actor_logits_from_obs(base_obs)
            logits_b = policy._actor_logits_from_obs(changed_other_agent)

        per_agent_dim = 5 + 50
        self.assertTrue(torch.allclose(logits_a[:, :per_agent_dim], logits_b[:, :per_agent_dim], atol=1e-6))
        self.assertFalse(torch.allclose(logits_a[:, per_agent_dim:], logits_b[:, per_agent_dim:], atol=1e-6))

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
