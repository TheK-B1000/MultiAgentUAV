import unittest

import numpy as np
import torch
from gymnasium import spaces

from game_field_gpu import VEC_OBS_DIM
from rl.global_state import GLOBAL_STATE_DIM
from rl.custom_ppo import SharedActorCentralizedCritic
from rl.custom_ppo.csv_writers import _update_fieldnames
from rl.custom_ppo.latent_diagnostics import _latent_opponent_rollout_diag
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
        feature_dim = 7 * 20 * 20 + VEC_OBS_DIM
        actor = LatentConditionedActor(
            local_feature_dim=feature_dim,
            latent_k=4,
            action_dim=55,
        )
        actor.eval()
        grid = torch.rand(2, 7, 20, 20)
        vec = torch.rand(2, VEC_OBS_DIM)
        z_idx = torch.tensor([2, 2])
        changed_grid = grid.clone()
        changed_vec = vec.clone()
        changed_grid[1] = torch.rand(7, 20, 20)
        changed_vec[1] = torch.rand(VEC_OBS_DIM)

        def _flat_features(g, v):
            return torch.cat([g.reshape(g.shape[0], -1), v], dim=-1)

        with torch.no_grad():
            logits_a = actor(_flat_features(grid, vec), z_idx)
            logits_b = actor(_flat_features(changed_grid, changed_vec), z_idx)

        self.assertTrue(torch.allclose(logits_a[0], logits_b[0], atol=1e-6))
        self.assertFalse(torch.allclose(logits_a[1], logits_b[1], atol=1e-6))

    def test_actor_no_latent_skips_embedding_and_z(self):
        """``latent_k=0`` produces a plain MLP head and forward ignores ``z_idx``."""
        torch.manual_seed(11)
        actor = LatentConditionedActor(local_feature_dim=16, latent_k=0, action_dim=5)
        self.assertIsNone(actor.strategy_embedding)
        feats = torch.randn(3, 16)
        with torch.no_grad():
            logits = actor(feats)
        self.assertEqual(tuple(logits.shape), (3, 5))

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
        model = SharedActorCentralizedCritic(obs_space, action_space, latent_k=4, z_embed_dim=16)
        dims = model.input_dim_contract()
        self.assertEqual(dims["base_global_state_dim"], GLOBAL_STATE_DIM)
        self.assertEqual(dims["temporal_context_dim"], CONTEXT_STATE_DIM)
        self.assertEqual(dims["q_phi_input_dim"], CONTEXT_STATE_DIM)
        self.assertEqual(dims["critic_context_dim"], CONTEXT_STATE_DIM)
        self.assertEqual(dims["actor_input_dim"], model.actor_cnn_feature_dim + VEC_OBS_DIM + 16)
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

    def test_phase_aux_loss_backpropagates_into_q_phi(self):
        obs_space = spaces.Dict(
            {
                "grid": spaces.Box(low=0.0, high=1.0, shape=(2, 7, 20, 20), dtype=np.float32),
                "vec": spaces.Box(low=-1.0, high=1.0, shape=(2, VEC_OBS_DIM), dtype=np.float32),
                "agent_mask": spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32),
                "mask": spaces.Box(low=0.0, high=1.0, shape=(110,), dtype=np.float32),
            }
        )
        action_space = spaces.MultiDiscrete([5, 50, 5, 50])
        model = SharedActorCentralizedCritic(obs_space, action_space, latent_k=4, z_embed_dim=16)
        global_state = torch.rand(6, CONTEXT_STATE_DIM)
        phase_id = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.long)

        z_logits = model.strategy_logits(global_state)
        phase_logits = model.phase_logits_from_strategy_logits(z_logits)
        self.assertEqual(tuple(phase_logits.shape), (6, 6))

        model.zero_grad(set_to_none=True)
        torch.nn.functional.cross_entropy(phase_logits, phase_id).backward()
        self.assertIsNotNone(model.strategy_encoder)
        grad_norm = sum(
            float(param.grad.detach().abs().sum().item())
            for param in model.strategy_encoder.parameters()
            if param.grad is not None
        )
        self.assertGreater(grad_norm, 0.0)

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

    def test_presets_config_application(self):
        from rl.train_ppo import PPOConfig, _apply_training_preset
        
        cfg = PPOConfig()
        cfg = _apply_training_preset(cfg, "latent_recommended")
        self.assertTrue(cfg.use_latent_strategy)
        self.assertEqual(cfg.latent_k, 4)
        self.assertEqual(cfg.latent_resample_every_n, 20)
        self.assertAlmostEqual(cfg.latent_lam_p, 0.025)
        self.assertAlmostEqual(cfg.latent_lam_h, 0.003)
        self.assertEqual(cfg.run_tag, "latent_recommended_1m_2v2")
        
        cfg = PPOConfig()
        cfg = _apply_training_preset(cfg, "latent_recommended_no_persistence")
        self.assertTrue(cfg.use_latent_strategy)
        self.assertEqual(cfg.latent_k, 4)
        self.assertAlmostEqual(cfg.latent_lam_p, 0.0)
        self.assertEqual(cfg.run_tag, "latent_recommended_no_persistence_1m_2v2")

        cfg = PPOConfig()
        cfg = _apply_training_preset(cfg, "latent_recommended_no_entropy")
        self.assertTrue(cfg.use_latent_strategy)
        self.assertEqual(cfg.latent_entropy_objective, "none")
        self.assertAlmostEqual(cfg.latent_lam_h, 0.0)
        self.assertEqual(cfg.run_tag, "latent_recommended_no_entropy_1m_2v2")

        cfg = PPOConfig()
        cfg = _apply_training_preset(cfg, "latent_recommended_collapsed_k1")
        self.assertTrue(cfg.use_latent_strategy)
        self.assertEqual(cfg.latent_k, 1)
        self.assertEqual(cfg.run_tag, "latent_recommended_collapsed_k1_1m_2v2")

        cfg = PPOConfig()
        cfg = _apply_training_preset(cfg, "no_latent_baseline")
        self.assertFalse(cfg.use_latent_strategy)
        self.assertEqual(cfg.run_tag, "no_latent_baseline_1m_2v2")

    def test_assert_input_contracts_fails_on_wrong_dims(self):
        obs_space = spaces.Dict(
            {
                "grid": spaces.Box(low=0.0, high=1.0, shape=(2, 7, 20, 20), dtype=np.float32),
                "vec": spaces.Box(low=-1.0, high=1.0, shape=(2, VEC_OBS_DIM), dtype=np.float32),
                "agent_mask": spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32),
                "mask": spaces.Box(low=0.0, high=1.0, shape=(110,), dtype=np.float32),
            }
        )
        action_space = spaces.MultiDiscrete([5, 50, 5, 50])
        model = SharedActorCentralizedCritic(obs_space, action_space, latent_k=4, z_embed_dim=15)
        
        original_q_phi = model.q_phi_input_dim
        try:
            model.q_phi_input_dim = 94
            with self.assertRaises(AssertionError):
                model._assert_input_contracts()
        finally:
            model.q_phi_input_dim = original_q_phi

    def test_diagnostics_occupancies_and_diversities_and_timing(self):
        from rl.ppo_core import TensorDictRolloutBuffer
        from rl.custom_ppo import CustomPPOTrainer
        
        class MockConfig:
            device = "cpu"
            use_latent_strategy = True
            latent_k = 4
            latent_resample_every_n = 20
            fixed_latent_strategy = False
            latent_strategy_ppo_coef = 0.1
            latent_strategy_aux_return_head = False
            latent_strategy_aux_return_coef = 0.0
            latent_strategy_aux_predict_phase_coef = 0.0
            actor_cnn_feature_dim = 128
            seed = 42
        
        obs_space = spaces.Dict(
            {
                "grid": spaces.Box(low=0.0, high=1.0, shape=(2, 7, 20, 20), dtype=np.float32),
                "vec": spaces.Box(low=-1.0, high=1.0, shape=(2, VEC_OBS_DIM), dtype=np.float32),
                "agent_mask": spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32),
                "mask": spaces.Box(low=0.0, high=1.0, shape=(110,), dtype=np.float32),
            }
        )
        action_space = spaces.MultiDiscrete([5, 50, 5, 50])
        
        obs_space_ref = obs_space
        action_space_ref = action_space

        class MockEnv:
            observation_space = obs_space_ref
            action_space = action_space_ref
            num_envs = 2
            class Core:
                pass
            core = Core()
        
        trainer = CustomPPOTrainer(MockEnv(), MockConfig(), learning_rate=1e-4, clip_range=0.2, ent_coef=0.01, n_epochs=2, batch_size=2)
        
        buffer = TensorDictRolloutBuffer(buffer_size=10, n_envs=2, device="cpu")
        buffer.register_field("z", dtype=torch.long)
        buffer.register_field("prev_z", dtype=torch.long)
        buffer.register_field("z_persist_mask", dtype=torch.bool)
        buffer.register_field("z_resampled", dtype=torch.bool)
        buffer.register_field("reward_sparse_points", dtype=torch.float32)
        buffer.register_field("rewards", dtype=torch.float32)
        buffer.register_field("global_state", (CONTEXT_STATE_DIM,))
        buffer.register_field("spread_bucket_id", dtype=torch.long)
        buffer.register_field("role_bucket_id", dtype=torch.long)
        buffer.register_field("pressure_bucket_id", dtype=torch.long)
        buffer.register_field("attack_defense_ratio_bucket_id", dtype=torch.long)
        buffer.register_field("phase_id", dtype=torch.long)
        
        for step in range(10):
            gs = torch.zeros((2, CONTEXT_STATE_DIM))
            gs[0, 10] = 1.0 if step < 5 else 0.0
            gs[1, 11] = 0.0 if step < 5 else 1.0
            
            z = torch.tensor([1, 2]) if step < 5 else torch.tensor([2, 2])
            if step == 3:
                z = torch.tensor([1, 0])
                
            prev_z = torch.tensor([1, 2]) if step == 0 else (torch.tensor([1, 2]) if step < 5 else torch.tensor([2, 2]))
            if step == 3:
                prev_z = torch.tensor([1, 2])
                
            rsp = torch.tensor([0.0, 0.0])
            if step == 8:
                rsp = torch.tensor([10.0, 0.0])
            if step == 2:
                rsp = torch.tensor([0.0, 100.0])
                
            buffer.add(
                z=z,
                prev_z=prev_z,
                z_persist_mask=torch.tensor([True, True]),
                z_resampled=torch.tensor([False, False]),
                reward_sparse_points=rsp,
                rewards=torch.tensor([0.0, 0.0]),
                global_state=gs,
                spread_bucket_id=torch.tensor([1, 0]),
                role_bucket_id=torch.tensor([2, 1]),
                pressure_bucket_id=torch.tensor([0, 2]),
                attack_defense_ratio_bucket_id=torch.tensor([2, 1]),
                phase_id=torch.tensor([0, 1]),
            )
            
        buffer.pos = 10
        out = _latent_opponent_rollout_diag(trainer, buffer)
        
        self.assertIn("latent_mi_z_flag_state_nats", out)
        self.assertIn("latent_switch_near_capture_frac", out)
        self.assertIn("latent_switch_near_kill_frac", out)
        self.assertIn("latent_switch_near_return_frac", out)
        self.assertIn("latent_flag_state1_z1_frac", out)
        self.assertIn("latent_spread1_z2_frac", out)
        self.assertIn("latent_adr2_z2_frac", out)
        self.assertIn("latent_phase0_entropy", out)
        self.assertIn("latent_role_diversity", out)
        self.assertIn("latent_spread_diversity", out)
        self.assertIn("latent_pressure_diversity", out)
        self.assertIn("latent_adr_diversity", out)
        fields = _update_fieldnames(trainer.use_latent_strategy, trainer.latent_k)
        self.assertIn("latent_mi_z_flag_state_nats", fields)
        self.assertIn("latent_switch_near_capture_frac", fields)
        self.assertIn("latent_flag_state1_z1_frac", fields)


if __name__ == "__main__":
    unittest.main()
