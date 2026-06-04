from __future__ import annotations

import unittest
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F

from rl.custom_ppo.latent_strategy_state import (
    LatentStrategyState,
    _advantage_weighted_target_from_records,
)
from tests.test_latent_episode_warmup import _make_trainer


class LatentPreferenceTests(unittest.TestCase):
    def test_advantage_weighted_target_requires_clear_margin(self) -> None:
        weak_records = [
            {"z": 0, "win_loss": 1},
            {"z": 0, "win_loss": 0},
            {"z": 1, "win_loss": 1},
            {"z": 1, "win_loss": 0},
        ]
        target, stats = _advantage_weighted_target_from_records(
            weak_records,
            latent_k=4,
            min_count=4,
            min_distinct_z=2,
            temperature=0.35,
            margin_threshold=0.15,
        )
        self.assertIsNone(target)
        self.assertAlmostEqual(stats["margin"], 0.0)

        strong_records = [
            {"z": 2, "win_loss": 1},
            {"z": 2, "win_loss": 1},
            {"z": 3, "win_loss": 0},
            {"z": 3, "win_loss": 1},
        ]
        target, stats = _advantage_weighted_target_from_records(
            strong_records,
            latent_k=4,
            min_count=4,
            min_distinct_z=2,
            temperature=0.35,
            margin_threshold=0.15,
        )
        self.assertIsNotNone(target)
        self.assertEqual(int(stats["best_z"]), 2)
        self.assertGreater(stats["margin"], 0.15)
        self.assertGreater(float(target[2]), float(target[3]))

    def test_record_episode_strategy_outcome_forced_z(self) -> None:
        trainer = _make_trainer(n_envs=2, warmup=0, episode_credit=True, gs_dim=4)
        trainer.cfg = SimpleNamespace(opponent_pool=["OP3", "OP5"], opponent_pool_weights=[0.5, 0.5])
        trainer.latent_preference_coef = 0.03
        trainer.episode_stats = SimpleNamespace(episodes_completed=0)
        
        latent_state = LatentStrategyState(trainer)
        latent_state.reset()
        
        # Mark env 0 as forced-z episode
        latent_state.episode_forced_z[0] = True
        latent_state.episode_forced_z_id[0] = 2
        latent_state.episode_contrast_bucket[0] = 5
        latent_state.episode_behavior_sum[0] = torch.ones(13, dtype=torch.float32)
        latent_state.episode_behavior_count[0] = 1
        
        # Mark env 1 as regular (non-forced-z) episode with a started strategy
        latent_state.episode_forced_z[1] = False
        latent_state.episode_strategy_has_start[1] = True
        latent_state.episode_strategy_z[1] = 1
        latent_state.episode_strategy_log_prob[1] = -0.5
        latent_state.episode_strategy_bucket[1] = 6
        
        # Record outcome for env 0 (forced-z)
        info_forced = {"scripted_tag": "OP3"}
        latent_state.record_episode_strategy_outcome(0, info_forced, episode_return=5.5)
        
        # Should record in latent_preference_buffer
        self.assertEqual(len(latent_state.latent_preference_buffer), 1)
        record = latent_state.latent_preference_buffer[0]
        self.assertEqual(record["context_bucket"], 5)
        self.assertEqual(record["opponent"], 2)  # OP3 maps to index 2
        self.assertEqual(record["phase_flag_state"], 5)
        self.assertEqual(record["z"], 2)
        self.assertAlmostEqual(record["return"], 5.5)
        
        # Standard rollout records should be empty
        self.assertEqual(len(latent_state.rollout_strategy_episode_records), 0)

        # Record outcome for env 1 (regular)
        info_reg = {"scripted_tag": "OP5"}
        latent_state.record_episode_strategy_outcome(1, info_reg, episode_return=2.0)
        
        # Standard rollout records should now have 1 item
        self.assertEqual(len(latent_state.rollout_strategy_episode_records), 1)
        # Latent preference buffer should still have only 1 item (the forced-z one)
        self.assertEqual(len(latent_state.latent_preference_buffer), 1)

    def test_apply_episode_strategy_ppo_pref_loss(self) -> None:
        trainer = _make_trainer(n_envs=1, warmup=0, episode_credit=True, gs_dim=4)
        trainer.cfg = SimpleNamespace(opponent_pool=["OP3", "OP5"], opponent_pool_weights=[0.5, 0.5])
        trainer.latent_preference_coef = 0.03
        trainer.latent_preference_temperature = 1.0
        trainer.latent_preference_min_bucket_count = 3
        trainer.latent_preference_min_distinct_z = 2
        
        class MockOptimizer:
            def __init__(self):
                self.zero_grad_called = False
                self.step_called = False
                self.param_groups = [{"params": []}]
            def zero_grad(self, set_to_none=True):
                self.zero_grad_called = True
            def step(self):
                self.step_called = True
        
        trainer.optimizer = MockOptimizer()
        trainer.latent_router_optimizer = None
        trainer.cfg.max_grad_norm = 1.0
        trainer.latent_episode_strategy_coef = 0.3
        trainer.latent_episode_strategy_value_coef = 0.5
        trainer.latent_entropy_objective = "maximize"
        trainer.latent_episode_strategy_n_epochs = 1
        trainer.latent_episode_strategy_clip_eps = 0.2
        trainer.latent_episode_strategy_return_norm = False
        trainer.latent_lam_h = 0.0
        
        class MockModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.global_state_dim = 4
                self.strategy_encoder = torch.nn.Linear(4, 4)
                self.episode_strategy_value_head = torch.nn.Linear(4, 1)
            def strategy_logits(self, state):
                return self.strategy_encoder(state)
            def episode_strategy_value(self, state, z):
                return self.episode_strategy_value_head(state).squeeze(-1)
        
        mock_model = MockModel()
        trainer.model = mock_model
        
        latent_state = LatentStrategyState(trainer)
        latent_state.reset()
        
        # Add 3 records to the preference buffer for the same bucket key:
        # Opponent id = 2 (OP3), context_bucket = 5. Key = 2 * 256 + 5 = 517.
        # This satisfies min_bucket_count=3 and min_distinct_z=2 (z=0, z=1).
        latent_state.latent_preference_buffer.append({
            "context_bucket": 5,
            "opponent": 2,
            "phase_flag_state": 5,
            "z": 0,
            "return": 10.0,
            "behavior_embedding": [0.0]*13,
            "win_loss": 0,
        })
        latent_state.latent_preference_buffer.append({
            "context_bucket": 5,
            "opponent": 2,
            "phase_flag_state": 5,
            "z": 1,
            "return": 20.0,
            "behavior_embedding": [0.0]*13,
            "win_loss": 0,
        })
        latent_state.latent_preference_buffer.append({
            "context_bucket": 5,
            "opponent": 2,
            "phase_flag_state": 5,
            "z": 1,
            "return": 30.0,
            "behavior_embedding": [0.0]*13,
            "win_loss": 0,
        })
        
        # Put 1 matching episode record in standard training rollout records
        latent_state.rollout_strategy_episode_records.append({
            "episode_id": 0,
            "global_state_0": torch.zeros(4, dtype=torch.float32),
            "z": 1,
            "z_logprob_old": 0.0,
            "episode_return": 15.0,
            "bucket_id": 5,
            "opponent_id": 2,
            "q_phi_probs": [0.25]*4,
        })
        
        stats = latent_state.apply_episode_strategy_ppo(latent_lam_h=0.0)
        
        # Verify stats logged from preference update
        self.assertGreater(stats["latent_preference_loss"], 0.0)
        self.assertEqual(stats["latent_preference_active_fraction"], 1.0)
        self.assertEqual(stats["latent_preference_buffer_size"], 3)
        self.assertEqual(stats["latent_preference_num_active_buckets"], 1)
        self.assertGreater(stats["latent_preference_target_entropy"], 0.0)
        self.assertTrue(trainer.optimizer.zero_grad_called)
        self.assertTrue(trainer.optimizer.step_called)

    def test_apply_episode_strategy_ppo_opponent_balanced_telemetry(self) -> None:
        trainer = _make_trainer(n_envs=3, warmup=0, episode_credit=True, gs_dim=4)
        trainer.cfg = SimpleNamespace(opponent_pool=["OP3", "OP5", "OP6"], opponent_pool_weights=[0.33, 0.33, 0.34])
        trainer.latent_preference_coef = 0.03
        trainer.latent_preference_temperature = 1.0
        trainer.latent_preference_min_bucket_count = 3
        trainer.latent_preference_min_distinct_z = 2
        
        # Turn on opponent balanced loss and telemetry logging
        trainer.cfg.latent_preference_opponent_balanced = True
        trainer.cfg.latent_preference_log_opponent_targets = True
        
        class MockOptimizer:
            def __init__(self):
                self.zero_grad_called = False
                self.step_called = False
                self.param_groups = [{"params": []}]
            def zero_grad(self, set_to_none=True):
                self.zero_grad_called = True
            def step(self):
                self.step_called = True
        
        trainer.optimizer = MockOptimizer()
        trainer.latent_router_optimizer = None
        trainer.cfg.max_grad_norm = 1.0
        trainer.latent_episode_strategy_coef = 0.3
        trainer.latent_episode_strategy_value_coef = 0.5
        trainer.latent_entropy_objective = "maximize"
        trainer.latent_episode_strategy_n_epochs = 1
        trainer.latent_episode_strategy_clip_eps = 0.2
        trainer.latent_episode_strategy_return_norm = False
        trainer.latent_lam_h = 0.0
        
        class MockModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.global_state_dim = 4
                self.strategy_encoder = torch.nn.Linear(4, 4)
                self.episode_strategy_value_head = torch.nn.Linear(4, 1)
                
                # Setup dummy weights to yield deterministic predictions
                with torch.no_grad():
                    self.strategy_encoder.weight.zero_()
                    self.strategy_encoder.bias.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0]))
            def strategy_logits(self, state):
                return self.strategy_encoder(state)
            def episode_strategy_value(self, state, z):
                return self.episode_strategy_value_head(state).squeeze(-1)
        
        mock_model = MockModel()
        trainer.model = mock_model
        
        latent_state = LatentStrategyState(trainer)
        latent_state.reset()
        
        # Add 3 records to the preference buffer for OP5 (id 4), bucket 5
        latent_state.latent_preference_buffer.append({
            "context_bucket": 5, "opponent": 4, "phase_flag_state": 5, "z": 0, "return": 10.0, "behavior_embedding": [0.0]*13, "win_loss": 0,
        })
        latent_state.latent_preference_buffer.append({
            "context_bucket": 5, "opponent": 4, "phase_flag_state": 5, "z": 1, "return": 20.0, "behavior_embedding": [0.0]*13, "win_loss": 0,
        })
        latent_state.latent_preference_buffer.append({
            "context_bucket": 5, "opponent": 4, "phase_flag_state": 5, "z": 1, "return": 30.0, "behavior_embedding": [0.0]*13, "win_loss": 0,
        })
        
        # Add 3 records to the preference buffer for OP6 (id 5), bucket 6
        latent_state.latent_preference_buffer.append({
            "context_bucket": 6, "opponent": 5, "phase_flag_state": 6, "z": 2, "return": 40.0, "behavior_embedding": [0.0]*13, "win_loss": 0,
        })
        latent_state.latent_preference_buffer.append({
            "context_bucket": 6, "opponent": 5, "phase_flag_state": 6, "z": 3, "return": 50.0, "behavior_embedding": [0.0]*13, "win_loss": 0,
        })
        latent_state.latent_preference_buffer.append({
            "context_bucket": 6, "opponent": 5, "phase_flag_state": 6, "z": 3, "return": 60.0, "behavior_embedding": [0.0]*13, "win_loss": 0,
        })
        
        # Add 3 episodes to standard rollout records:
        # Two for OP5 (id 4), one for OP6 (id 5)
        latent_state.rollout_strategy_episode_records.append({
            "episode_id": 0, "global_state_0": torch.zeros(4, dtype=torch.float32), "z": 1, "z_logprob_old": 0.0, "episode_return": 15.0, "bucket_id": 5, "opponent_id": 4, "q_phi_probs": [0.25]*4,
        })
        latent_state.rollout_strategy_episode_records.append({
            "episode_id": 1, "global_state_0": torch.zeros(4, dtype=torch.float32), "z": 1, "z_logprob_old": 0.0, "episode_return": 25.0, "bucket_id": 5, "opponent_id": 4, "q_phi_probs": [0.25]*4,
        })
        latent_state.rollout_strategy_episode_records.append({
            "episode_id": 2, "global_state_0": torch.zeros(4, dtype=torch.float32), "z": 3, "z_logprob_old": 0.0, "episode_return": 55.0, "bucket_id": 6, "opponent_id": 5, "q_phi_probs": [0.25]*4,
        })
        
        stats = latent_state.apply_episode_strategy_ppo(latent_lam_h=0.0)
        
        # Verify specific opponent buffer count
        self.assertEqual(stats["latent_pref_op5_buffer_count"], 3.0)
        self.assertEqual(stats["latent_pref_op6_buffer_count"], 3.0)
        
        # Verify active fraction
        self.assertAlmostEqual(stats["latent_pref_op5_active_fraction"], 1.0)
        self.assertAlmostEqual(stats["latent_pref_op6_active_fraction"], 1.0)
        
        # Verify active buckets
        self.assertEqual(stats["latent_pref_op5_active_buckets"], 1.0)
        self.assertEqual(stats["latent_pref_op6_active_buckets"], 1.0)
        
        # Verify best z
        self.assertEqual(stats["latent_pref_op5_best_z"], 1.0)
        self.assertEqual(stats["latent_pref_op6_best_z"], 3.0)
        
        # Verify target entropy
        self.assertGreater(stats["latent_pref_op5_target_entropy"], 0.0)
        self.assertGreater(stats["latent_pref_op6_target_entropy"], 0.0)
        
        # Verify target distributions
        self.assertGreater(stats["latent_pref_op5_target_z1"], 0.5)
        self.assertGreater(stats["latent_pref_op6_target_z3"], 0.5)
        
        # Verify individual opponent losses
        self.assertGreater(stats["latent_pref_op5_loss"], 0.0)
        self.assertGreater(stats["latent_pref_op6_loss"], 0.0)
        
        # Since opponent_balanced = True, the overall preference loss should be the average
        # of the two individual opponent losses: (OP5_loss + OP6_loss) / 2
        expected_balanced_loss = (stats["latent_pref_op5_loss"] + stats["latent_pref_op6_loss"]) / 2.0
        self.assertAlmostEqual(stats["latent_preference_loss"], expected_balanced_loss, places=5)


    def test_v3h2_confidence_weighted_loss(self) -> None:
        trainer = _make_trainer(n_envs=1, warmup=0, episode_credit=True, gs_dim=4)
        trainer.cfg = SimpleNamespace(opponent_pool=["OP3", "OP5"], opponent_pool_weights=[0.5, 0.5])
        
        # v3h2 hyperparams
        trainer.latent_preference_coef = 0.03
        trainer.latent_preference_temperature = 1.0
        trainer.latent_preference_min_bucket_count = 2
        trainer.latent_preference_min_distinct_z = 2
        trainer.latent_preference_confidence_scale = 2.0
        trainer.latent_preference_commit_coef = 0.003
        trainer.late_entropy_floor = 0.0003
        trainer.commitment_type = "confidence_weighted_entropy"
        
        class MockOptimizer:
            def __init__(self):
                self.zero_grad_called = False
                self.step_called = False
                self.param_groups = [{"params": []}]
            def zero_grad(self, set_to_none=True):
                self.zero_grad_called = True
            def step(self):
                self.step_called = True
        
        trainer.optimizer = MockOptimizer()
        trainer.latent_router_optimizer = None
        trainer.cfg.max_grad_norm = 1.0
        trainer.latent_episode_strategy_coef = 0.3
        trainer.latent_episode_strategy_value_coef = 0.5
        trainer.latent_entropy_objective = "maximize"
        trainer.latent_episode_strategy_n_epochs = 1
        trainer.latent_episode_strategy_clip_eps = 0.2
        trainer.latent_episode_strategy_return_norm = False
        trainer.latent_lam_h = 0.0
        
        class MockModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.global_state_dim = 4
                self.strategy_encoder = torch.nn.Linear(4, 4)
                self.episode_strategy_value_head = torch.nn.Linear(4, 1)
                
                # Setup dummy weights to yield deterministic predictions
                with torch.no_grad():
                    self.strategy_encoder.weight.zero_()
                    self.strategy_encoder.bias.copy_(torch.tensor([0.0, 0.0, 0.0, 0.0]))
            def strategy_logits(self, state):
                return self.strategy_encoder(state)
            def episode_strategy_value(self, state, z):
                return self.episode_strategy_value_head(state).squeeze(-1)
        
        mock_model = MockModel()
        trainer.model = mock_model
        
        latent_state = LatentStrategyState(trainer)
        latent_state.reset()
        
        # Add 2 records to the preference buffer:
        latent_state.latent_preference_buffer.append({
            "context_bucket": 5, "opponent": 2, "phase_flag_state": 5, "z": 0, "return": 0.0, "behavior_embedding": [0.0]*13, "win_loss": 0,
        })
        latent_state.latent_preference_buffer.append({
            "context_bucket": 5, "opponent": 2, "phase_flag_state": 5, "z": 1, "return": 100.0, "behavior_embedding": [0.0]*13, "win_loss": 0,
        })
        
        # Put 1 matching episode record in standard training rollout records
        latent_state.rollout_strategy_episode_records.append({
            "episode_id": 0, "global_state_0": torch.zeros(4, dtype=torch.float32), "z": 1, "z_logprob_old": 0.0, "episode_return": 15.0, "bucket_id": 5, "opponent_id": 2, "q_phi_probs": [0.25]*4,
        })
        
        stats = latent_state.apply_episode_strategy_ppo(latent_lam_h=0.0)
        self.assertGreater(stats["latent_preference_loss"], 0.0)

    def test_apply_episode_strategy_ppo_v3i3_event_pref_normalization(self) -> None:
        trainer = _make_trainer(n_envs=1, warmup=0, episode_credit=True, gs_dim=4)
        trainer.cfg = SimpleNamespace(opponent_pool=["OP3"], opponent_pool_weights=[1.0])
        trainer.latent_v3i3_event_preference_enabled = True
        trainer.latent_v3i3_event_preference_coef = 0.5
        trainer.latent_v3i3_event_preference_temperature = 1.0
        trainer.latent_v3i3_event_preference_min_bucket_count = 3
        trainer.latent_v3i3_event_preference_min_distinct_z = 1
        trainer.latent_v3i3_event_preference_buffer_size = 1000
        trainer.latent_v3i3_event_preference_warmup_steps = 0
        trainer.latent_v3i3_event_preference_normalize = True
        trainer.global_step = 100
        trainer.latent_k = 4
        trainer.latent_episode_strategy_coef = 0.0
        trainer.latent_episode_strategy_n_epochs = 1
        trainer.cfg.max_grad_norm = 1.0
        trainer.latent_episode_strategy_return_norm = False
        trainer.latent_episode_strategy_value_coef = 0.5
        trainer.latent_entropy_objective = "maximize"
        trainer.latent_episode_strategy_clip_eps = 0.2
        trainer.latent_lam_h = 0.0
        trainer.latent_preference_coef = 0.0
        trainer.latent_event_preference_key_mode = "event_flag"

        class MockOptimizer:
            def __init__(self):
                self.zero_grad_called = False
                self.step_called = False
                self.param_groups = [{"params": []}]
            def zero_grad(self, set_to_none=True):
                self.zero_grad_called = True
            def step(self):
                self.step_called = True

        trainer.optimizer = MockOptimizer()
        trainer.latent_router_optimizer = None

        class MockModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.global_state_dim = 4
                self.strategy_encoder = torch.nn.Linear(4, 4)
                self.episode_strategy_value_head = torch.nn.Linear(4, 1)
            def strategy_logits(self, state):
                return self.strategy_encoder(state)
            def episode_strategy_value(self, state, z):
                return self.episode_strategy_value_head(state).squeeze(-1)

        trainer.model = MockModel()

        latent_state = LatentStrategyState(trainer)
        latent_state.reset()

        # Add records to refresh_preference_buffer from two different flag states:
        # Key A: opponent=2 (OP3), event=1, flag=5. Returns: z=1 -> 20.0. Baseline A = 20.0. Normalized = 0.0.
        # Key B: opponent=2 (OP3), event=1, flag=6. Returns: z=0 -> 120.0. Baseline B = 120.0. Normalized = 0.0.
        # Min bucket count = 3, so full lookup for Key A or Key B alone fails (since counts are < 3).
        # It falls back to oe level (opponent=2, event=1), combining Key A and Key B.
        # If we normalize, both resolved means for z=0 and z=1 will be 0.0, leading to a uniform resolved target.
        latent_state.refresh_preference_buffer.append({
            "opponent_id": 2, "event_type": 1, "flag_state_bucket": 5, "z": 1, "future_return": 20.0,
        })
        latent_state.refresh_preference_buffer.append({
            "opponent_id": 2, "event_type": 1, "flag_state_bucket": 5, "z": 1, "future_return": 20.0,
        })
        latent_state.refresh_preference_buffer.append({
            "opponent_id": 2, "event_type": 1, "flag_state_bucket": 6, "z": 0, "future_return": 120.0,
        })
        latent_state.refresh_preference_buffer.append({
            "opponent_id": 2, "event_type": 1, "flag_state_bucket": 6, "z": 0, "future_return": 120.0,
        })

        # Put matching records in rollout_refresh_records
        # Target lookup at (opponent=2, reason=1, flag=5) falls back to (2, 1).
        latent_state.rollout_refresh_records.append({
            "refresh_state": torch.zeros(4, dtype=torch.float32),
            "opponent_id": 2,
            "reason_id": 1,
            "flag_state_bucket": 5,
            "next_z": 1,
            "return_at_refresh": 0.0,
        })

        # Matching episode record to prevent empty check
        latent_state.rollout_strategy_episode_records.append({
            "episode_id": 0, "global_state_0": torch.zeros(4, dtype=torch.float32), "z": 1, "z_logprob_old": 0.0, "episode_return": 15.0, "bucket_id": 5, "opponent_id": 2, "q_phi_probs": [0.25]*4,
        })

        stats = latent_state.apply_episode_strategy_ppo(latent_lam_h=0.0)

        # Telemetry counts verify active records and buckets
        self.assertEqual(stats["latent_v3i3_event_pref_buffer_size"], 4.0)
        self.assertEqual(stats["latent_v3i3_event_pref_rollout_records"], 1.0)
        self.assertEqual(stats["latent_v3i3_event_pref_active_records"], 1.0)
        self.assertEqual(stats["latent_v3i3_event_pref_active_buckets"], 1.0)
        self.assertEqual(stats["latent_v3i3_event_pref_fallback_oe"], 1.0)
        self.assertEqual(stats["latent_v3i3_event_pref_fallback_full"], 0.0)

        # Target entropy for normalized target (should be close to uniform ln 4 = 1.386)
        # Because z=0 and z=1 normalized returns are both 0.0, means=[0.0, 0.0, 0.0, 0.0] -> uniform.
        self.assertAlmostEqual(stats["latent_v3i3_event_pref_target_entropy"], 1.38629436, places=4)

    def test_apply_episode_strategy_ppo_v3i4_normalizes_by_progress_key(self) -> None:
        trainer = _make_trainer(n_envs=1, warmup=0, episode_credit=True, gs_dim=4)
        trainer.cfg = SimpleNamespace(opponent_pool=["OP3"], opponent_pool_weights=[1.0])
        trainer.latent_v3i3_event_preference_enabled = True
        trainer.latent_v3i3_event_preference_coef = 0.5
        trainer.latent_v3i3_event_preference_temperature = 1.0
        trainer.latent_v3i3_event_preference_min_bucket_count = 3
        trainer.latent_v3i3_event_preference_min_distinct_z = 1
        trainer.latent_v3i3_event_preference_buffer_size = 1000
        trainer.latent_v3i3_event_preference_warmup_steps = 0
        trainer.latent_v3i3_event_preference_normalize = True
        trainer.global_step = 100
        trainer.latent_k = 4
        trainer.latent_episode_strategy_coef = 0.0
        trainer.latent_episode_strategy_n_epochs = 1
        trainer.cfg.max_grad_norm = 1.0
        trainer.latent_episode_strategy_return_norm = False
        trainer.latent_episode_strategy_value_coef = 0.5
        trainer.latent_entropy_objective = "maximize"
        trainer.latent_episode_strategy_clip_eps = 0.2
        trainer.latent_lam_h = 0.0
        trainer.latent_preference_coef = 0.0
        trainer.latent_event_preference_key_mode = "event_flag_progress"

        class MockOptimizer:
            def __init__(self):
                self.zero_grad_called = False
                self.step_called = False
                self.param_groups = [{"params": []}]
            def zero_grad(self, set_to_none=True):
                self.zero_grad_called = True
            def step(self):
                self.step_called = True

        trainer.optimizer = MockOptimizer()
        trainer.latent_router_optimizer = None

        class MockModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.global_state_dim = 4
                self.strategy_encoder = torch.nn.Linear(4, 4)
                self.episode_strategy_value_head = torch.nn.Linear(4, 1)
            def strategy_logits(self, state):
                return self.strategy_encoder(state)
            def episode_strategy_value(self, state, z):
                return self.episode_strategy_value_head(state).squeeze(-1)

        trainer.model = MockModel()

        latent_state = LatentStrategyState(trainer)
        latent_state.reset()

        # Same opponent/event/flag, different progress buckets.
        # Correct v3i4 normalization subtracts each full progress-key baseline,
        # so fallback to (opp,event,flag) sees zero advantage for both z slots.
        for _ in range(2):
            latent_state.refresh_preference_buffer.append({
                "opponent_id": 2,
                "event_type": 1,
                "flag_state_bucket": 2,
                "carrier_progress_bucket": 1,
                "z": 1,
                "future_return": 20.0,
            })
            latent_state.refresh_preference_buffer.append({
                "opponent_id": 2,
                "event_type": 1,
                "flag_state_bucket": 2,
                "carrier_progress_bucket": 3,
                "z": 0,
                "future_return": 120.0,
            })

        latent_state.rollout_refresh_records.append({
            "refresh_state": torch.zeros(4, dtype=torch.float32),
            "opponent_id": 2,
            "reason_id": 1,
            "flag_state_bucket": 2,
            "carrier_progress_bucket": 1,
            "next_z": 1,
            "return_at_refresh": 0.0,
        })
        latent_state.rollout_strategy_episode_records.append({
            "episode_id": 0,
            "global_state_0": torch.zeros(4, dtype=torch.float32),
            "z": 1,
            "z_logprob_old": 0.0,
            "episode_return": 15.0,
            "bucket_id": 5,
            "opponent_id": 2,
            "q_phi_probs": [0.25] * 4,
        })

        stats = latent_state.apply_episode_strategy_ppo(latent_lam_h=0.0)

        self.assertEqual(stats["latent_v3i3_event_pref_active_records"], 1.0)
        self.assertEqual(stats["latent_v3i3_event_pref_fallback_oef"], 1.0)
        self.assertEqual(stats["latent_v3i3_event_pref_fallback_full"], 0.0)
        self.assertAlmostEqual(
            stats["latent_v3i3_event_pref_target_entropy"], 1.38629436, places=4
        )

    def test_apply_episode_strategy_ppo_v3i7_awrd_uses_winning_z_margin(self) -> None:
        trainer = _make_trainer(n_envs=1, warmup=0, episode_credit=True, gs_dim=4)
        trainer.cfg = SimpleNamespace(opponent_pool=["OP5"], opponent_pool_weights=[1.0])
        trainer.global_step = 100
        trainer.latent_k = 4
        trainer.latent_episode_strategy_coef = 0.0
        trainer.latent_episode_strategy_n_epochs = 1
        trainer.cfg.max_grad_norm = 1.0
        trainer.latent_episode_strategy_return_norm = False
        trainer.latent_episode_strategy_value_coef = 0.5
        trainer.latent_entropy_objective = "maximize"
        trainer.latent_episode_strategy_clip_eps = 0.2
        trainer.latent_lam_h = 0.0
        trainer.latent_preference_coef = 0.0
        trainer.latent_v3i3_event_preference_enabled = False
        trainer.latent_awrd_enabled = True
        trainer.latent_awrd_coef = 0.5
        trainer.latent_awrd_temperature = 0.35
        trainer.latent_awrd_min_bucket_count = 4
        trainer.latent_awrd_min_distinct_z = 2
        trainer.latent_awrd_margin_threshold = 0.15
        trainer.latent_awrd_margin_scale = 2.0

        class MockOptimizer:
            def __init__(self):
                self.zero_grad_called = False
                self.step_called = False
                self.param_groups = [{"params": []}]
            def zero_grad(self, set_to_none=True):
                self.zero_grad_called = True
            def step(self):
                self.step_called = True

        trainer.optimizer = MockOptimizer()
        trainer.latent_router_optimizer = None

        class MockModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.global_state_dim = 4
                self.strategy_encoder = torch.nn.Linear(4, 4)
                self.episode_strategy_value_head = torch.nn.Linear(4, 1)
            def strategy_logits(self, state):
                return self.strategy_encoder(state)
            def episode_strategy_value(self, state, z):
                return self.episode_strategy_value_head(state).squeeze(-1)

        trainer.model = MockModel()

        latent_state = LatentStrategyState(trainer)
        latent_state.reset()
        for z_val, win_loss in ((2, 1), (2, 1), (3, 0), (3, 1)):
            latent_state.latent_preference_buffer.append({
                "context_bucket": 5,
                "opponent": 4,
                "phase_flag_state": 5,
                "z": z_val,
                "return": float(win_loss),
                "behavior_embedding": [0.0] * 13,
                "win_loss": win_loss,
            })

        latent_state.rollout_strategy_episode_records.append({
            "episode_id": 0,
            "global_state_0": torch.zeros(4, dtype=torch.float32),
            "z": 2,
            "z_logprob_old": 0.0,
            "episode_return": 1.0,
            "bucket_id": 5,
            "opponent_id": 4,
            "q_phi_probs": [0.25] * 4,
        })

        stats = latent_state.apply_episode_strategy_ppo(latent_lam_h=0.0)

        self.assertGreater(stats["latent_awrd_loss"], 0.0)
        self.assertEqual(stats["latent_awrd_active_fraction"], 1.0)
        self.assertEqual(stats["latent_awrd_active_buckets"], 1.0)
        self.assertAlmostEqual(stats["latent_awrd_margin_mean"], 0.5, places=5)
        self.assertAlmostEqual(stats["latent_awrd_wr_spread_mean"], 0.5, places=5)
        self.assertEqual(stats["latent_awrd_best_z_mean"], 2.0)



if __name__ == "__main__":
    unittest.main()
