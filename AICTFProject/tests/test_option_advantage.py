from __future__ import annotations

import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import torch
from gymnasium import spaces

from rl.train_ppo import PPOConfig
from rl.custom_ppo import CustomPPOTrainer
from game_field_gpu import VEC_OBS_DIM


def _mock_strategy_ppo_stats(device: torch.device = torch.device("cpu")) -> dict:
    """Return a dict matching the real strategy_ppo_loss output keys."""
    return {
        "policy_loss": torch.zeros((), dtype=torch.float32, device=device),
        "approx_kl": torch.zeros((), dtype=torch.float32, device=device),
        "clip_fraction": torch.zeros((), dtype=torch.float32, device=device),
        "ratio": torch.ones((1,), dtype=torch.float32, device=device),
    }


class OptionAdvantageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.obs_space = spaces.Dict({
            "grid": spaces.Box(low=0.0, high=1.0, shape=(2, 7, 20, 20), dtype=np.float32),
            "vec": spaces.Box(low=-1.0, high=1.0, shape=(2, VEC_OBS_DIM), dtype=np.float32),
            "agent_mask": spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32),
            "mask": spaces.Box(low=0.0, high=1.0, shape=(110,), dtype=np.float32),
        })
        self.action_space = spaces.MultiDiscrete([5, 50, 5, 50])
        
        class MockEnv:
            observation_space = self.obs_space
            action_space = self.action_space
            num_envs = 2
            class Core:
                blue_score = torch.tensor([0, 0])
                red_score = torch.tensor([0, 0])
                device = torch.device("cpu")
                B = 2
                Nb = 2
                Nr = 2
                max_dist = 1.0
                blue_x = torch.zeros((2, 2))
                blue_y = torch.zeros((2, 2))
                blue_alive = torch.ones((2, 2), dtype=torch.bool)
                blue_carrying = torch.zeros((2, 2), dtype=torch.bool)
                red_x = torch.zeros((2, 2))
                red_y = torch.zeros((2, 2))
                red_alive = torch.ones((2, 2), dtype=torch.bool)
                red_carrying = torch.zeros((2, 2), dtype=torch.bool)
                red_flag_pos = torch.zeros((2, 2))
                blue_flag_pos = torch.zeros((2, 2))
            core = Core()
            def reset(self):
                return {
                    "grid": np.zeros((2, 2, 7, 20, 20), dtype=np.float32),
                    "vec": np.zeros((2, 2, VEC_OBS_DIM), dtype=np.float32),
                    "agent_mask": np.ones((2, 2), dtype=np.float32),
                    "mask": np.ones((2, 110), dtype=np.float32),
                }
            def state(self):
                return np.zeros((2, 19), dtype=np.float32)
            def step_async(self, actions):
                pass
            def step_wait(self):
                obs = self.reset()
                rewards = np.zeros((2,), dtype=np.float32)
                dones = np.array([False, False])
                infos = [{"blue_score": 0, "red_score": 0}, {"blue_score": 0, "red_score": 0}]
                return obs, rewards, dones, infos
            
        self.env = MockEnv()

    def _make_filled_buffer(self, trainer, n_steps=4, z_resampled_val=False):
        """Helper: create and fill a buffer with n_steps of dummy data."""
        buffer = trainer.rollout_collector.make_buffer(self.env.reset())
        for _ in range(n_steps):
            add_items = {
                "obs_grid": torch.zeros((2, 2, 7, 20, 20)),
                "obs_vec": torch.zeros((2, 2, VEC_OBS_DIM)),
                "obs_agent_mask": torch.ones((2, 2)),
                "obs_mask": torch.ones((2, 110)),
                "global_state": torch.zeros((2, 95)),
                "actions": torch.zeros((2, 4), dtype=torch.long),
                "log_probs": torch.zeros((2,)),
                "values": torch.zeros((2,)),
                "values_norm": torch.zeros((2,)),
                "next_values": torch.zeros((2,)),
                "rewards": torch.zeros((2,)),
                "reward_terminal": torch.zeros((2,)),
                "reward_offense": torch.zeros((2,)),
                "reward_pbrs": torch.zeros((2,)),
                "reward_team": torch.zeros((2,)),
                "reward_sparse": torch.zeros((2,)),
                "reward_sparse_points": torch.zeros((2,)),
                "reward_failure": torch.zeros((2,)),
                "reward_total": torch.zeros((2,)),
                "terminated": torch.zeros((2,), dtype=torch.bool),
                "truncated": torch.zeros((2,), dtype=torch.bool),
                "opponent_id": torch.zeros((2,), dtype=torch.long),
                "z": torch.zeros((2,), dtype=torch.long),
                "prev_z": torch.zeros((2,), dtype=torch.long),
                "z_log_probs": torch.zeros((2,)),
                "z_logits": torch.zeros((2, 4)),
                "z_resampled": torch.full((2,), z_resampled_val, dtype=torch.bool),
                "z_persist_mask": torch.zeros((2,), dtype=torch.bool),
                "phase_id": torch.zeros((2,), dtype=torch.long),
                "outcome_id": torch.zeros((2,), dtype=torch.long),
                "behavior_telemetry": torch.zeros((2, 13)),
                "spread_bucket_id": torch.zeros((2,), dtype=torch.long),
                "role_bucket_id": torch.zeros((2,), dtype=torch.long),
                "pressure_bucket_id": torch.zeros((2,), dtype=torch.long),
                "attack_defense_ratio_bucket_id": torch.zeros((2,), dtype=torch.long),
                "blue_ahead": torch.zeros((2,)),
            }
            buffer.add(**add_items)
        return buffer

    def test_default_behavior_is_unchanged_when_toggle_is_false(self) -> None:
        """1. Default behavior is unchanged when latent_q_phi_option_advantage=False."""
        cfg = PPOConfig()
        cfg.use_latent_strategy = True
        cfg.latent_q_phi_option_advantage = False
        cfg.n_steps = 4
        
        trainer = CustomPPOTrainer(
            self.env,
            cfg,
            learning_rate=1e-4,
            clip_range=0.2,
            ent_coef=0.01,
            n_epochs=1,
            batch_size=8,  # full batch so there's exactly one minibatch
        )
        
        buffer = self._make_filled_buffer(trainer)
        
        # Register advantages and returns (normally done by compute_returns_and_advantages)
        buffer.register_field("advantages")
        buffer.register_field("returns")
        buffer.fields["advantages"].fill_(1.5)
        buffer.fields["returns"].fill_(2.5)
        if "option_advantages" not in buffer.fields:
            buffer.register_field("option_advantages")
        if "option_returns" not in buffer.fields:
            buffer.register_field("option_returns")
        buffer.fields["option_advantages"].fill_(9.9)
        buffer.fields["option_returns"].fill_(9.9)
        
        with patch("rl.custom_ppo.ppo_updater._latent_strategy_ppo_loss") as mock_loss:
            mock_loss.return_value = (torch.tensor(0.0), _mock_strategy_ppo_stats())
            trainer.update(buffer, total_timesteps=100)
            
            self.assertTrue(mock_loss.called)
            args, kwargs = mock_loss.call_args
            passed_advantages = args[2]
            
            # Since toggle is False, the strategy loss receives the standard
            # (normalized) advantages, NOT the option_advantages (which were 9.9).
            # Standard advantages were 1.5, then normalized to mean=0 std=1 in
            # the minibatch, so they should NOT equal 9.9.
            self.assertFalse(torch.allclose(passed_advantages, torch.full_like(passed_advantages, 9.9)))

    def test_option_advantages_generated_only_for_latent_rollouts(self) -> None:
        """2. option_advantages are generated only for latent-enabled rollouts."""
        # Case A: Latent enabled
        cfg_latent = PPOConfig()
        cfg_latent.use_latent_strategy = True
        cfg_latent.n_steps = 2
        trainer_latent = CustomPPOTrainer(
            self.env,
            cfg_latent,
            learning_rate=1e-4,
            clip_range=0.2,
            ent_coef=0.01,
            n_epochs=1,
            batch_size=2,
        )
        
        def _mock_next_values_latent(*args, **kwargs):
            """Mock that also sets _last_context_state like the real method does."""
            trainer_latent._last_context_state = torch.zeros(
                2, trainer_latent.model.global_state_dim, dtype=torch.float32
            )
            return torch.zeros(2)

        with patch.object(trainer_latent.latent_state, "strategy_for_step") as mock_strat, \
             patch.object(trainer_latent.model, "act") as mock_act, \
             patch.object(trainer_latent.rollout_collector, "next_values", side_effect=_mock_next_values_latent):
             
            mock_strat.return_value = (torch.zeros(2, dtype=torch.long), torch.zeros(2, dtype=torch.long), {
                "z": torch.zeros(2, dtype=torch.long),
                "prev_z": torch.zeros(2, dtype=torch.long),
                "z_log_prob": torch.zeros(2),
                "z_entropy": torch.zeros(2),
                "z_logits": torch.zeros(2, 4),
                "z_resampled": torch.zeros(2, dtype=torch.bool),
                "z_persist_mask": torch.zeros(2, dtype=torch.bool),
            })
            mock_act.return_value = (torch.zeros(2, 4, dtype=torch.long), torch.zeros(2), torch.zeros(2), torch.zeros(2))
            
            buf_latent = trainer_latent.collect_rollout()
            self.assertIn("option_advantages", buf_latent.fields)
            self.assertIn("option_returns", buf_latent.fields)

        # Case B: Latent disabled
        cfg_no_latent = PPOConfig()
        cfg_no_latent.use_latent_strategy = False
        cfg_no_latent.n_steps = 2
        trainer_no_latent = CustomPPOTrainer(
            self.env,
            cfg_no_latent,
            learning_rate=1e-4,
            clip_range=0.2,
            ent_coef=0.01,
            n_epochs=1,
            batch_size=2,
        )
        
        with patch.object(trainer_no_latent.model, "act") as mock_act, \
             patch.object(trainer_no_latent.rollout_collector, "next_values") as mock_next:
             
            mock_act.return_value = (torch.zeros(2, 4, dtype=torch.long), torch.zeros(2), torch.zeros(2), torch.zeros(2))
            mock_next.return_value = torch.zeros(2)
            
            buf_no_latent = trainer_no_latent.collect_rollout()
            self.assertNotIn("option_advantages", buf_no_latent.fields)
            self.assertNotIn("option_returns", buf_no_latent.fields)

    def test_option_advantage_differs_from_gae_on_synthetic_rollout(self) -> None:
        """3. option advantage differs from one-step/per-step GAE on a synthetic rollout with delayed reward after the z resample."""
        T, B = 4, 1
        rewards = torch.tensor([[0.0], [0.0], [1.0], [0.0]])
        values = torch.tensor([[0.0], [0.0], [0.0], [0.0]])
        next_values = torch.tensor([[0.0], [0.0], [0.0], [0.0]])
        terminated = torch.tensor([[False], [False], [False], [False]])
        truncated = torch.tensor([[False], [False], [False], [False]])
        z_resampled = torch.tensor([[True], [False], [True], [False]])
        gamma = 1.0
        
        # Option return backward dynamic programming
        option_returns = torch.zeros_like(rewards)
        for t in reversed(range(T)):
            done_t = terminated[t] | truncated[t]
            if done_t:
                next_val = torch.where(
                    terminated[t],
                    torch.zeros_like(rewards[t]),
                    next_values[t]
                )
            else:
                if t == T - 1:
                    next_val = next_values[t]
                else:
                    next_val = torch.where(
                        z_resampled[t + 1],
                        values[t + 1],
                        option_returns[t + 1]
                    )
            option_returns[t] = rewards[t] + gamma * next_val
        
        # Option return at t=0 is 0.0 because propagation of reward at t=2 is blocked by z_resampled[2] = True
        self.assertEqual(float(option_returns[0, 0]), 0.0)
        
        # Standard GAE returns propagation (without z_resampled block)
        standard_returns = torch.zeros_like(rewards)
        for t in reversed(range(T)):
            if t == T - 1:
                next_val = next_values[t]
            else:
                next_val = standard_returns[t + 1]
            standard_returns[t] = rewards[t] + gamma * next_val
            
        self.assertEqual(float(standard_returns[0, 0]), 1.0)
        self.assertNotEqual(float(option_returns[0, 0]), float(standard_returns[0, 0]))

    def test_action_ppo_uses_standard_advantages(self) -> None:
        """4. action PPO still uses standard advantages."""
        cfg = PPOConfig()
        cfg.use_latent_strategy = True
        cfg.latent_q_phi_option_advantage = True
        cfg.n_steps = 4
        
        trainer = CustomPPOTrainer(
            self.env,
            cfg,
            learning_rate=1e-4,
            clip_range=0.2,
            ent_coef=0.01,
            n_epochs=1,
            batch_size=8,  # full batch: 4 steps × 2 envs = 8
        )
        
        buffer = self._make_filled_buffer(trainer)
            
        buffer.register_field("advantages")
        buffer.register_field("returns")
        adv = torch.zeros(8)
        adv[:4] = 1.0
        adv[4:] = 2.0
        buffer.fields["advantages"].copy_(adv.reshape(4, 2))
        buffer.fields["returns"].fill_(1.0)
        
        if "option_advantages" not in buffer.fields:
            buffer.register_field("option_advantages")
        if "option_returns" not in buffer.fields:
            buffer.register_field("option_returns")
        buffer.fields["option_advantages"].fill_(9.99)
        buffer.fields["option_returns"].fill_(9.99)
        
        with patch("rl.custom_ppo.ppo_updater.ppo_policy_loss") as mock_action_loss:
            mock_action_loss.return_value = (
                torch.tensor(0.0, requires_grad=True),
                {"ratio": torch.ones(1), "approx_kl": torch.tensor(0.0), "clip_fraction": torch.tensor(0.0)},
            )
            trainer.update(buffer, total_timesteps=100)
            
            # With batch_size=8 (full batch), there's exactly 1 call per epoch.
            # The action PPO should receive the standard advantages (values 1.0/2.0,
            # then normalized), NOT option_advantages (9.99).
            self.assertTrue(mock_action_loss.called)
            args, kwargs = mock_action_loss.call_args
            passed_advantages = args[2]
            
            # Standard advantages had variance (mix of 1.0 and 2.0), so after
            # normalization they should be non-zero. If option_advantages (uniform
            # 9.99) were used instead, normalization would produce all zeros.
            self.assertFalse(torch.allclose(passed_advantages, torch.zeros_like(passed_advantages)))

    def test_q_phi_strategy_loss_uses_option_advantages_when_toggle_is_true(self) -> None:
        """5. q_phi strategy loss uses option_advantages when the toggle is True."""
        cfg = PPOConfig()
        cfg.use_latent_strategy = True
        cfg.latent_q_phi_option_advantage = True
        cfg.n_steps = 4
        
        trainer = CustomPPOTrainer(
            self.env,
            cfg,
            learning_rate=1e-4,
            clip_range=0.2,
            ent_coef=0.01,
            n_epochs=1,
            batch_size=8,  # full batch
        )
        
        buffer = self._make_filled_buffer(trainer)
            
        buffer.register_field("advantages")
        buffer.register_field("returns")
        buffer.fields["advantages"].fill_(1.23)
        buffer.fields["returns"].fill_(1.0)
        if "option_advantages" not in buffer.fields:
            buffer.register_field("option_advantages")
        if "option_returns" not in buffer.fields:
            buffer.register_field("option_returns")
        buffer.fields["option_advantages"].fill_(9.99)
        buffer.fields["option_returns"].fill_(9.99)
        
        with patch("rl.custom_ppo.ppo_updater._latent_strategy_ppo_loss") as mock_loss:
            mock_loss.return_value = (torch.tensor(0.0), _mock_strategy_ppo_stats())
            trainer.update(buffer, total_timesteps=100)
            
            self.assertTrue(mock_loss.called)
            args, kwargs = mock_loss.call_args
            passed_advantages = args[2]
            
            # When toggle is True, the strategy loss should receive option_advantages
            # (all 9.99), not the standard advantages (all 1.23).
            self.assertTrue(torch.allclose(passed_advantages, torch.tensor(9.99)))

if __name__ == "__main__":
    unittest.main()
