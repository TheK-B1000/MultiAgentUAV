import unittest
import numpy as np
import torch
from gymnasium import spaces

from rl.custom_ppo import SharedActorCentralizedCritic
from rl.custom_ppo.inference import CustomPPOInferencePolicy
from rl.latent_marl import GLOBAL_STATE_DIM

class TestInferenceParity(unittest.TestCase):
    def setUp(self):
        self.obs_space = spaces.Dict({
            "grid": spaces.Box(low=0.0, high=1.0, shape=(2, 7, 20, 20), dtype=np.float32),
            "vec": spaces.Box(low=-1.0, high=1.0, shape=(2, 17), dtype=np.float32),
            "agent_mask": spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32),
            "mask": spaces.Box(low=0.0, high=1.0, shape=(110,), dtype=np.float32),
        })
        self.action_space = spaces.MultiDiscrete([5, 50, 5, 50])
        
    def _create_mock_policy(self, router_context_mode="current_plus_delta", fixed_latent_strategy=False):
        cfg = {
            "use_latent_strategy": True,
            "latent_k": 4,
            "latent_resample_every_n": 5,
            "router_context_mode": router_context_mode,
            "fixed_latent_strategy": fixed_latent_strategy,
            "router_allowed_latents": [0, 3]
        }
        
        # Instantiate model
        model = SharedActorCentralizedCritic(
            observation_space=self.obs_space,
            action_space=self.action_space,
            latent_k=4,
            router_context_mode=router_context_mode,
            router_context_dimension=68 if router_context_mode == "current_plus_delta" else 170
        )
        
        # Wrap in inference policy
        policy = CustomPPOInferencePolicy(model, device="cpu", cfg=cfg)
        return policy

    def test_first_opportunity_delta_zero(self):
        policy = self._create_mock_policy(router_context_mode="current_plus_delta")
        self.assertTrue(policy.model.uses_latent_strategy)
        self.assertTrue(policy.model.router_current_plus_delta_enabled)
        
        batch_size = 1
        obs = {
            "grid": np.zeros((batch_size, 2, 7, 20, 20), dtype=np.float32),
            "vec": np.zeros((batch_size, 2, 17), dtype=np.float32),
            "agent_mask": np.ones((batch_size, 2), dtype=np.float32),
            "mask": np.ones((batch_size, 110), dtype=np.float32),
        }
        
        # We manually inject a specific global state under policy._global_state_tensor
        # Let's override _global_state_tensor to return a known pattern
        gs_vals = torch.zeros((batch_size, GLOBAL_STATE_DIM), dtype=torch.float32)
        gs_vals[0, :] = torch.arange(1.0, 35.0) # 1 to 34
        
        policy._global_state_tensor = lambda batched, batch: gs_vals
        
        # Run prediction
        actions, _ = policy.predict(obs, deterministic=True)
        
        # On first step/opportunity, previous_opportunity_features should be set to current opportunity features,
        # but the delta used for prediction should be 0 because opportunity had not occurred yet.
        # Let's inspect what context was logged or used.
        # Let's check _opportunity_occurred and _previous_opportunity_features
        self.assertTrue(policy._opportunity_occurred[0])
        torch.testing.assert_close(policy._previous_opportunity_features[0], gs_vals[0, :GLOBAL_STATE_DIM])
        
        # The history tracking updated. Let's check the trace log:
        self.assertEqual(len(policy.opportunity_trace_log), 1)
        
    def test_subsequent_opportunity_delta_calculation(self):
        policy = self._create_mock_policy(router_context_mode="current_plus_delta")
        batch_size = 1
        obs = {
            "grid": np.zeros((batch_size, 2, 7, 20, 20), dtype=np.float32),
            "vec": np.zeros((batch_size, 2, 17), dtype=np.float32),
            "agent_mask": np.ones((batch_size, 2), dtype=np.float32),
            "mask": np.ones((batch_size, 110), dtype=np.float32),
        }
        
        # Step 1: First step (opportunity occurs because _opportunity_occurred is initialized to False)
        gs_vals_1 = torch.zeros((batch_size, GLOBAL_STATE_DIM), dtype=torch.float32)
        gs_vals_1[0, :] = torch.ones((GLOBAL_STATE_DIM,)) * 2.0
        policy._global_state_tensor = lambda batched, batch: gs_vals_1
        
        actions1, _ = policy.predict(obs, deterministic=True)
        self.assertTrue(policy._opportunity_occurred[0])
        torch.testing.assert_close(policy._previous_opportunity_features[0], gs_vals_1[0, :GLOBAL_STATE_DIM])
        
        # Age should start incrementing
        self.assertEqual(int(policy._strategy_age[0]), 1)
        
        # Step 2: Next steps before opportunity recurrence (resample_every_n = 5, age starts at 1, so steps 2, 3, 4 do not trigger opportunity)
        gs_vals_2 = torch.zeros((batch_size, GLOBAL_STATE_DIM), dtype=torch.float32)
        gs_vals_2[0, :] = torch.ones((GLOBAL_STATE_DIM,)) * 3.0
        policy._global_state_tensor = lambda batched, batch: gs_vals_2
        
        # Run predict (no new opportunity because age is 1 < 5)
        policy.predict(obs, deterministic=True)
        self.assertEqual(int(policy._strategy_age[0]), 2)
        # Previous opportunity features should still be the old ones
        torch.testing.assert_close(policy._previous_opportunity_features[0], gs_vals_1[0, :GLOBAL_STATE_DIM])
        
        # Step 3: Fast-forward age to trigger resampling
        policy._strategy_age[0] = 5
        # Run predict (opportunity triggers because age >= 5)
        actions2, _ = policy.predict(obs, deterministic=True)
        self.assertEqual(int(policy._strategy_age[0]), 1) # reset to 0 then incremented to 1
        # Now previous opportunity features should be updated to current ones (gs_vals_2)
        torch.testing.assert_close(policy._previous_opportunity_features[0], gs_vals_2[0, :GLOBAL_STATE_DIM])

    def test_multi_env_and_selective_reset(self):
        policy = self._create_mock_policy(router_context_mode="current_plus_delta")
        batch_size = 2
        obs = {
            "grid": np.zeros((batch_size, 2, 7, 20, 20), dtype=np.float32),
            "vec": np.zeros((batch_size, 2, 17), dtype=np.float32),
            "agent_mask": np.ones((batch_size, 2), dtype=np.float32),
            "mask": np.ones((batch_size, 110), dtype=np.float32),
        }
        
        # Opportunity 1 for both envs
        gs_vals = torch.zeros((batch_size, GLOBAL_STATE_DIM), dtype=torch.float32)
        gs_vals[0, :] = torch.ones((GLOBAL_STATE_DIM,)) * 10.0
        gs_vals[1, :] = torch.ones((GLOBAL_STATE_DIM,)) * 20.0
        policy._global_state_tensor = lambda batched, batch: gs_vals
        
        policy.predict(obs, deterministic=True)
        
        # Verify both tracked
        self.assertTrue(policy._opportunity_occurred[0])
        self.assertTrue(policy._opportunity_occurred[1])
        torch.testing.assert_close(policy._previous_opportunity_features[0], gs_vals[0, :GLOBAL_STATE_DIM])
        torch.testing.assert_close(policy._previous_opportunity_features[1], gs_vals[1, :GLOBAL_STATE_DIM])
        
        # Now perform selective reset on environment 0 only
        done_mask = [True, False]
        policy.reset_strategy(done_mask=done_mask)
        
        # Env 0 should be reset: _opportunity_occurred is False, features are 0
        self.assertFalse(policy._opportunity_occurred[0])
        torch.testing.assert_close(policy._previous_opportunity_features[0], torch.zeros((GLOBAL_STATE_DIM,)))
        # Env 1 should remain untouched
        self.assertTrue(policy._opportunity_occurred[1])
        torch.testing.assert_close(policy._previous_opportunity_features[1], gs_vals[1, :GLOBAL_STATE_DIM])

    def test_batch_size_change_safety(self):
        policy = self._create_mock_policy(router_context_mode="current_plus_delta")
        
        # First batch size = 2
        batch_size = 2
        obs_2 = {
            "grid": np.zeros((batch_size, 2, 7, 20, 20), dtype=np.float32),
            "vec": np.zeros((batch_size, 2, 17), dtype=np.float32),
            "agent_mask": np.ones((batch_size, 2), dtype=np.float32),
            "mask": np.ones((batch_size, 110), dtype=np.float32),
        }
        
        gs_vals_2 = torch.zeros((batch_size, GLOBAL_STATE_DIM), dtype=torch.float32)
        policy._global_state_tensor = lambda batched, batch: gs_vals_2
        
        policy.predict(obs_2, deterministic=True)
        self.assertEqual(policy._prev_z.shape[0], 2)
        
        # Batch size change to 3
        batch_size = 3
        obs_3 = {
            "grid": np.zeros((batch_size, 2, 7, 20, 20), dtype=np.float32),
            "vec": np.zeros((batch_size, 2, 17), dtype=np.float32),
            "agent_mask": np.ones((batch_size, 2), dtype=np.float32),
            "mask": np.ones((batch_size, 110), dtype=np.float32),
        }
        
        gs_vals_3 = torch.zeros((batch_size, GLOBAL_STATE_DIM), dtype=torch.float32)
        policy._global_state_tensor = lambda batched, batch: gs_vals_3
        
        policy.predict(obs_3, deterministic=True)
        self.assertEqual(policy._prev_z.shape[0], 3)
        self.assertEqual(policy._previous_opportunity_features.shape[0], 3)

    def test_entropy_no_history_mutation(self):
        policy = self._create_mock_policy(router_context_mode="current_plus_delta")
        batch_size = 1
        obs = {
            "grid": np.zeros((batch_size, 2, 7, 20, 20), dtype=np.float32),
            "vec": np.zeros((batch_size, 2, 17), dtype=np.float32),
            "agent_mask": np.ones((batch_size, 2), dtype=np.float32),
            "mask": np.ones((batch_size, 110), dtype=np.float32),
        }
        
        # Predict once to initialize history
        gs_vals = torch.zeros((batch_size, GLOBAL_STATE_DIM), dtype=torch.float32)
        gs_vals[0, :] = torch.ones((GLOBAL_STATE_DIM,)) * 5.0
        policy._global_state_tensor = lambda batched, batch: gs_vals
        
        policy.predict(obs, deterministic=True)
        prev_features_before = policy._previous_opportunity_features.clone()
        age_before = policy._strategy_age.clone()
        
        # Call entropy
        ent_val = policy.entropy(obs)
        
        # Check that calling entropy did NOT mutate history
        torch.testing.assert_close(policy._previous_opportunity_features, prev_features_before)
        torch.testing.assert_close(policy._strategy_age, age_before)

if __name__ == "__main__":
    unittest.main()
