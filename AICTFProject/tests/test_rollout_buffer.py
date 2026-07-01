from __future__ import annotations

import unittest

import torch

from rl.ppo_core import TensorDictRolloutBuffer


class RolloutBufferTests(unittest.TestCase):
    def test_named_registry_accepts_new_strategy_field(self) -> None:
        buffer = TensorDictRolloutBuffer(buffer_size=2, n_envs=1)
        buffer.register_field("rewards")
        buffer.register_field("values")
        buffer.register_field("next_values")
        buffer.register_field("terminated", dtype=torch.bool)
        buffer.register_field("truncated", dtype=torch.bool)
        buffer.register_field("z", (1,), dtype=torch.long)

        buffer.add(
            rewards=torch.tensor([1.0]),
            values=torch.tensor([0.0]),
            next_values=torch.tensor([0.0]),
            terminated=torch.tensor([False]),
            truncated=torch.tensor([False]),
            z=torch.tensor([[2]]),
        )
        buffer.add(
            rewards=torch.tensor([1.0]),
            values=torch.tensor([0.0]),
            next_values=torch.tensor([0.0]),
            terminated=torch.tensor([True]),
            truncated=torch.tensor([False]),
            z=torch.tensor([[3]]),
        )
        buffer.compute_returns_and_advantages(gamma=1.0, gae_lambda=1.0)

        self.assertIn("z", buffer.registry)
        self.assertIn("advantages", buffer.fields)
        self.assertEqual(tuple(buffer.fields["z"].shape), (2, 1, 1))
        self.assertAlmostEqual(float(buffer.fields["returns"][0, 0]), 2.0, places=6)
        self.assertAlmostEqual(float(buffer.fields["returns"][1, 0]), 1.0, places=6)

    def test_minibatches_flatten_time_and_env_axes(self) -> None:
        buffer = TensorDictRolloutBuffer(buffer_size=2, n_envs=2)
        buffer.register_field("actions", (4,), dtype=torch.long)
        for step in range(2):
            buffer.add(actions=torch.full((2, 4), step, dtype=torch.long))

        batches = list(buffer.iter_minibatches(batch_size=3, shuffle=False))

        self.assertEqual(len(batches), 2)
        self.assertEqual(tuple(batches[0]["actions"].shape), (3, 4))
        self.assertEqual(tuple(batches[1]["actions"].shape), (1, 4))


if __name__ == "__main__":
    unittest.main()
