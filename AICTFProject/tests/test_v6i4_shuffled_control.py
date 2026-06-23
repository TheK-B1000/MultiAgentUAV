"""Pins regression tests for logical-seed based shuffled mapping and sequence generation."""

from __future__ import annotations

import random
import unittest
import numpy as np
import torch

from rl.custom_ppo.inference import CustomPPOInferencePolicy
from rl.evaluation.router_ablation import stable_sha256_text

class MockModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.uses_latent_strategy = True
        self.latent_k = 4
        self.router_current_plus_delta_enabled = False
        self.router_allowed_latents = [0, 3]
        
    def sample_strategy(self, *args, **kwargs):
        pass

    def _strategy_logits_forward(self, context):
        return torch.zeros((context.shape[0], self.latent_k), device=context.device)

    def act(self, obs, context_gs, deterministic=True, z_idx=None, **kwargs):
        # returns action_tensor, values, action_log_probs, rnn_states
        batch_size = obs["grid"].shape[0]
        # Return dummy actions
        return torch.zeros((batch_size, 4), dtype=torch.long), torch.zeros((batch_size, 1)), torch.zeros((batch_size,)), None


class TestShuffledControlRegression(unittest.TestCase):
    def test_shuffled_sequence_regression(self) -> None:
        model = MockModel()
        # Mock policy's _strategy_logits_forward to avoid requiring full neural network forward pass
        policy = CustomPPOInferencePolicy(model, device="cpu")
        policy._strategy_logits_forward = lambda ctx: torch.zeros((ctx.shape[0], model.latent_k), device=policy.device)
        policy.latent_eval_mode = "shuffled"
        
        # 1. Create learned_t_data representing learned switching decisions
        # Simulate 10 steps of learned trajectory.
        learned_t_data = []
        for opp in ["OP5", "OP6"]:
            # logical seed = 2000, environment seed = 9919
            # seed logged is the logical seed (e.g. 2000)
            for opp_counter in range(10):
                z_val = 0 if opp_counter % 2 == 0 else 3
                learned_t_data.append({
                    "opponent": opp,
                    "seed": 2000,
                    "episode_index": 0,
                    "opportunity_index": opp_counter,
                    "selected_z": z_val,
                    "logits": [1.0, 0.0, 0.0, 2.0] if z_val == 3 else [2.0, 0.0, 0.0, 1.0],
                    "probabilities": [0.1, 0.1, 0.1, 0.7] if z_val == 3 else [0.7, 0.1, 0.1, 0.1],
                })
                
        # 2. Build shuffled mapping using our new sequence generator logic
        decisions_by_z = {}
        for t_item in learned_t_data:
            z_val = int(t_item["selected_z"])
            if z_val not in decisions_by_z:
                decisions_by_z[z_val] = []
            decisions_by_z[z_val].append({
                "logits": list(t_item["logits"]),
                "probabilities": list(t_item["probabilities"]),
                "selected_z": z_val,
            })
            
        # Count frequencies
        counts = {}
        for t_item in learned_t_data:
            z_val = int(t_item["selected_z"])
            counts[z_val] = counts.get(z_val, 0) + 1
            
        allowed = [0, 3]
        filtered_counts = {z: counts.get(z, 0) for z in allowed}
        filtered_total = sum(filtered_counts.values())
        z_probs = {z: count / filtered_total for z, count in filtered_counts.items()}
        
        max_opportunities = 100
        
        # Proportional base latents
        base_latents = []
        for z_val, p in z_probs.items():
            count = int(round(p * max_opportunities))
            base_latents.extend([z_val] * count)
            
        if len(base_latents) < max_opportunities:
            base_latents.extend([allowed[0]] * (max_opportunities - len(base_latents)))
        else:
            base_latents = base_latents[:max_opportunities]
            
        # Unique keys
        unique_keys = sorted({
            (str(t_item["opponent"]).upper(), int(t_item["seed"]), int(t_item["episode_index"]))
            for t_item in learned_t_data
        })
        
        # Generate shuffled mapping
        shuffled_mapping = {}
        for (opp, seed, env_idx) in unique_keys:
            h = stable_sha256_text(f"{opp.upper()}|{int(seed)}|{int(env_idx)}")
            local_seed = int(h[:8], 16)
            rng = random.Random(local_seed)
            
            episode_latents = list(base_latents)
            rng.shuffle(episode_latents)
            
            episode_decisions = []
            for z_val in episode_latents:
                pool = decisions_by_z.get(z_val, [])
                dec = rng.choice(pool)
                episode_decisions.append({
                    "selected_z": int(dec["selected_z"]),
                    "logits": list(dec["logits"]),
                    "probabilities": list(dec["probabilities"]),
                })
            shuffled_mapping[(opp, seed, env_idx)] = episode_decisions
            
        # Inject to policy
        policy.inject_shuffled_mapping(shuffled_mapping)
        
        # 3. Simulate prediction under logical seed 2000, environment seed 9919
        policy.set_eval_episode_context(
            opponent="OP5",
            eval_seed=2000,
            environment_seed=9919,
            env_index=0,
        )
        
        # Verify initial counter is 0 (can be scalar before predict)
        self.assertEqual(policy._opportunity_counter[0] if isinstance(policy._opportunity_counter, np.ndarray) else policy._opportunity_counter, 0)
        
        # Check lookup succeeds and returns valid decisions up to 20 steps
        # (longer than the 10 steps of learned trajectory)
        selected_zs = []
        for step in range(20):
            # mock predict inputs
            obs = {
                "grid": np.zeros((1, 10, 10, 10), dtype=np.float32),
                "vec": np.zeros((1, 10), dtype=np.float32),
                "agent_mask": np.zeros((1, 10), dtype=np.float32),
                "mask": np.zeros((1, 10), dtype=np.float32),
            }
            # force policy to sample new strategy
            policy._opportunity_occurred = torch.zeros((1,), dtype=torch.bool)
            policy._strategy_age = torch.zeros((1,), dtype=torch.long)
            
            # Predict
            act, _ = policy.predict(obs, deterministic=True)
            sel_z = int(policy._last_strategy_z.item())
            selected_zs.append(sel_z)
            
            # Verify z1 and z2 are never selected
            self.assertIn(sel_z, [0, 3])
            self.assertNotIn(sel_z, [1, 2])
            
        # Verify opportunity index is 20
        self.assertEqual(policy._opportunity_counter[0], 20)
        
        # 4. Test selective environment reset restarts opportunity index only for completed envs
        # Let's mock a batch of size 2
        # env 0 is not done, env 1 is done and reset
        from rl.latent_marl import TemporalStateTracker
        policy._temporal_tracker = TemporalStateTracker(
            num_envs=2,
            state_dim=128,
            device="cpu",
        )
        policy._prev_z = torch.zeros((2,), dtype=torch.long)
        policy._strategy_age = torch.zeros((2,), dtype=torch.long)
        policy._opportunity_occurred = torch.zeros((2,), dtype=torch.bool)
        policy._previous_opportunity_features = torch.zeros((2, 128), dtype=torch.float32)
        policy._opportunity_counter = np.array([5, 10], dtype=np.int64)
        
        done_mask = np.array([False, True])
        policy.reset_strategy(done_mask=done_mask)
        
        # Assert env 0 counter remains 5, env 1 counter is reset to 0
        self.assertEqual(policy._opportunity_counter[0], 5)
        self.assertEqual(policy._opportunity_counter[1], 0)


if __name__ == "__main__":
    unittest.main()
