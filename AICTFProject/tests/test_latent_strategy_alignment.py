import unittest

import numpy as np
import torch
from gymnasium import spaces

from game_field_gpu import VEC_OBS_DIM
from rl.global_state import GLOBAL_STATE_DIM
from rl.custom_ppo import SharedActorCentralizedCritic
from rl.latent_marl import (
    LatentConditionedActor,
    StrategyEncoder,
    expected_strategy_switch_penalty,
    TemporalStateTracker,
    CONTEXT_STATE_DIM,
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
        actor = LatentConditionedActor((7, 20, 20), vec_dim=VEC_OBS_DIM, latent_k=4, action_dim=55)
        actor.eval()
        grid = torch.rand(2, 7, 20, 20)
        vec = torch.rand(2, VEC_OBS_DIM)
        z_idx = torch.tensor([2, 2])
        changed_grid = grid.clone()
        changed_vec = vec.clone()
        changed_grid[1] = torch.rand(7, 20, 20)
        changed_vec[1] = torch.rand(VEC_OBS_DIM)

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

    def test_custom_ppo_model_conditions_actor_and_critic_on_strategy(self):
        obs_space = spaces.Dict(
            {
                "grid": spaces.Box(low=0.0, high=1.0, shape=(2, 7, 20, 20), dtype=np.float32),
                "vec": spaces.Box(low=-1.0, high=1.0, shape=(2, VEC_OBS_DIM), dtype=np.float32),
                "agent_mask": spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32),
                "mask": spaces.Box(low=0.0, high=1.0, shape=(110,), dtype=np.float32),
            }
        )
        action_space = spaces.MultiDiscrete([5, 50, 5, 50])
        model = SharedActorCentralizedCritic(obs_space, action_space, latent_k=4, z_embed_dim=8)
        dims = model.input_dim_contract()
        self.assertEqual(dims["base_global_state_dim"], GLOBAL_STATE_DIM)
        self.assertEqual(dims["temporal_context_dim"], CONTEXT_STATE_DIM)
        self.assertEqual(dims["q_phi_input_dim"], CONTEXT_STATE_DIM)
        self.assertEqual(dims["critic_context_dim"], CONTEXT_STATE_DIM)
        self.assertEqual(dims["actor_input_dim"], model.actor_cnn_feature_dim + VEC_OBS_DIM + 8)
        self.assertEqual(dims["critic_z_dim"], 4)
        obs = {
            "grid": torch.rand(3, 2, 7, 20, 20),
            "vec": torch.rand(3, 2, VEC_OBS_DIM),
            "agent_mask": torch.ones(3, 2),
            "mask": torch.ones(3, 110),
        }
        global_state = torch.rand(3, CONTEXT_STATE_DIM)
        z, z_log_prob, z_entropy, z_logits = model.sample_strategy(global_state)
        actions, values, action_log_prob, action_entropy = model.act(obs, global_state, z_idx=z)

        self.assertEqual(tuple(z.shape), (3,))
        self.assertEqual(tuple(z_logits.shape), (3, 4))
        self.assertEqual(tuple(z_log_prob.shape), (3,))
        self.assertEqual(tuple(z_entropy.shape), (3,))
        self.assertEqual(tuple(actions.shape), (3, 4))
        self.assertEqual(tuple(values.shape), (3,))
        self.assertEqual(tuple(action_log_prob.shape), (3,))
        self.assertEqual(tuple(action_entropy.shape), (3,))

        eval_values, eval_log_prob, _, aux = model.evaluate_actions(obs, global_state, actions, z_idx=z)
        self.assertEqual(tuple(eval_values.shape), (3,))
        self.assertEqual(tuple(eval_log_prob.shape), (3,))
        self.assertIn("strategy_log_prob", aux)
        extra = model._critic_extra(actions, z)
        self.assertEqual(tuple(extra[:, -4:].shape), (3, 4))
        self.assertTrue(torch.allclose(extra[:, -4:].sum(dim=-1), torch.ones(3)))
        with self.assertRaisesRegex(AssertionError, "critic expected context"):
            model.values(torch.rand(3, GLOBAL_STATE_DIM), actions=actions, z_idx=z)
        with self.assertRaisesRegex(AssertionError, "q_phi expected context"):
            model.sample_strategy(torch.rand(3, GLOBAL_STATE_DIM))

    def test_temporal_state_tracker(self):
        tracker = TemporalStateTracker(num_envs=2, state_dim=GLOBAL_STATE_DIM)
        # Test output dimension
        self.assertEqual(CONTEXT_STATE_DIM, GLOBAL_STATE_DIM * 5)
        
        # Step 0
        state0 = torch.ones((2, GLOBAL_STATE_DIM)) * 2.0
        out0 = tracker.update(state0)
        self.assertEqual(tuple(out0.shape), (2, CONTEXT_STATE_DIM))
        # Initial EMAs should equal raw state
        self.assertTrue(torch.allclose(tracker.ema_short, state0))
        self.assertTrue(torch.allclose(tracker.ema_long, state0))
        # Differences should be 0
        self.assertTrue(torch.allclose(out0[:, GLOBAL_STATE_DIM*3:GLOBAL_STATE_DIM*4], torch.zeros_like(state0)))
        
        # Step 1
        state1 = torch.ones((2, GLOBAL_STATE_DIM)) * 3.0
        out1 = tracker.update(state1)
        expected_short = 0.2 * state1 + 0.8 * state0
        expected_long = 0.05 * state1 + 0.95 * state0
        self.assertTrue(torch.allclose(tracker.ema_short, expected_short))
        self.assertTrue(torch.allclose(tracker.ema_long, expected_long))
        
        # Test reset on done
        dones = torch.tensor([True, False])
        tracker.update(state1, dones=dones)
        # First env should reset to state1 since it's done
        self.assertTrue(torch.allclose(tracker.ema_short[0], state1[0]))
        # Second env should continue update
        expected_short_env1 = 0.2 * state1[1] + 0.8 * expected_short[1]
        self.assertTrue(torch.allclose(tracker.ema_short[1], expected_short_env1))
        
        # Test passive get_current_context
        passive_out = tracker.get_current_context(state1)
        self.assertEqual(tuple(passive_out.shape), (2, CONTEXT_STATE_DIM))


if __name__ == "__main__":
    unittest.main()
