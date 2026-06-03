import importlib.util
import pathlib
import sys
import types
import unittest
from types import SimpleNamespace

import numpy as np
import torch

def _stub_strategy_experience_bucket_ids(state: torch.Tensor) -> torch.Tensor:
    return torch.zeros(state.shape[0], dtype=torch.long, device=state.device)

def _load_latent_strategy_state():
    # Provide a stub that copies needed symbols from real ppo_core
    import rl.ppo_core
    ppo_core_mod = types.ModuleType("rl.ppo_core")
    ppo_core_mod.TensorDictRolloutBuffer = rl.ppo_core.TensorDictRolloutBuffer
    ppo_core_mod.ppo_policy_loss = rl.ppo_core.ppo_policy_loss
    
    sys.modules.setdefault("rl.ppo_core", ppo_core_mod)

    target = (
        pathlib.Path(__file__).resolve().parent.parent
        / "rl"
        / "custom_ppo"
        / "latent_strategy_state.py"
    )
    spec = importlib.util.spec_from_file_location(
        "rl.custom_ppo.latent_strategy_state_isolated", str(target)
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module._strategy_experience_bucket_ids = _stub_strategy_experience_bucket_ids
    if "rl.ppo_core" in sys.modules:
        del sys.modules["rl.ppo_core"]
    return module.LatentStrategyState

LatentStrategyState = _load_latent_strategy_state()

class _FakeStrategyHead(torch.nn.Module):
    def __init__(self, latent_k: int) -> None:
        super().__init__()
        self.latent_k = int(latent_k)
        self.global_state_dim = 34
        self._sampling_gen_strategy = None

    def strategy_logits(self, state: torch.Tensor) -> torch.Tensor:
        # Uniform logits
        return torch.zeros((state.shape[0], self.latent_k), dtype=torch.float32, device=state.device)

    @staticmethod
    def _categorical_argmax_or_sample(dist, *, deterministic: bool, generator):
        return torch.argmax(dist.logits, dim=-1)

def _make_trainer(
    n_envs: int,
    *,
    min_gap: int = 20,
    max_refreshes: int = 3,
    latent_k: int = 4,
) -> SimpleNamespace:
    device = torch.device("cpu")
    model = _FakeStrategyHead(latent_k=latent_k)
    env = SimpleNamespace(num_envs=n_envs)
    trainer = SimpleNamespace(
        env=env,
        device=device,
        model=model,
        use_latent_strategy=True,
        fixed_latent_strategy=False,
        fixed_latent_strategy_id=0,
        latent_k=latent_k,
        latent_kl_consecutive=0.0,
        latent_resample_every_n=0,
        latent_episode_strategy_ppo=False,
        latent_episode_strategy_warmup_decision_steps=0,
        temporal_tracker=None,
        _last_context_state=None,
        
        # event refresh configs
        latent_event_refresh_enabled=True,
        latent_event_refresh_min_gap_steps=min_gap,
        latent_event_refresh_max_per_episode=max_refreshes,
        latent_event_refresh_use_q_phi=True,
        latent_event_refresh_force_roles=False,
    )
    return trainer

class LatentEventRefreshTests(unittest.TestCase):
    def _gs(self, enemy_flag=0.0, friendly_flag=0.0, blue_score=0.0, red_score=0.0, carrier_dist=1.0) -> torch.Tensor:
        # Make a mock context state (B=1, D=170) where first 34 columns are global_state
        state = torch.zeros((1, 170), dtype=torch.float32)
        state[0, 10] = enemy_flag
        state[0, 11] = friendly_flag
        state[0, 14] = blue_score
        state[0, 15] = red_score
        state[0, 23] = carrier_dist
        return state

    def test_enemy_flag_trigger(self):
        trainer = _make_trainer(1)
        ls = LatentStrategyState(trainer)
        ls.reset()
        
        # Step 0: Episode start (sampled provisional/start z)
        state_t0 = self._gs(enemy_flag=0.0)
        z0, _, aux0 = ls.strategy_for_step(state_t0)
        self.assertTrue(bool(aux0["z_resampled_actual"].item()))
        ls.mark_strategy_step_done(np.array([False]))
        
        # Step 1: No change
        state_t1 = self._gs(enemy_flag=0.0)
        z1, _, aux1 = ls.strategy_for_step(state_t1)
        self.assertFalse(bool(aux1["z_resampled_actual"].item()))
        ls.mark_strategy_step_done(np.array([False]))
        
        # Fast forward steps to clear min gap (gap needs to be >= 20 steps)
        for _ in range(25):
            ls.mark_strategy_step_done(np.array([False]))
            
        # Step 2: Enemy flag grabbed (0.0 -> 1.0)
        # Note: steps_since_last_refresh is now > 20, so it's allowed.
        state_t2 = self._gs(enemy_flag=1.0)
        z2, _, aux2 = ls.strategy_for_step(state_t2)
        self.assertTrue(bool(aux2["z_resampled_actual"].item()))
        ls.mark_strategy_step_done(np.array([False]))
        
        # Verify telemetry
        stats = ls.event_refresh_rollout_stats()
        self.assertEqual(stats["latent_refresh_count"], 1.0)
        self.assertEqual(stats["latent_refresh_reason_enemy_flag"], 1.0)

    def test_friendly_flag_trigger(self):
        trainer = _make_trainer(1)
        ls = LatentStrategyState(trainer)
        ls.reset()
        
        # Step 0
        state_t0 = self._gs(friendly_flag=0.0)
        ls.strategy_for_step(state_t0)
        
        # Clear gap
        for _ in range(25):
            ls.mark_strategy_step_done(np.array([False]))
            
        # Trigger
        state_t1 = self._gs(friendly_flag=1.0)
        _, _, aux = ls.strategy_for_step(state_t1)
        self.assertTrue(bool(aux["z_resampled_actual"].item()))
        
        stats = ls.event_refresh_rollout_stats()
        self.assertEqual(stats["latent_refresh_count"], 1.0)
        self.assertEqual(stats["latent_refresh_reason_friendly_flag"], 1.0)

    def test_score_trigger(self):
        trainer = _make_trainer(1)
        ls = LatentStrategyState(trainer)
        ls.reset()
        
        # Step 0
        state_t0 = self._gs(blue_score=0.0, red_score=0.0)
        ls.strategy_for_step(state_t0)
        
        # Clear gap
        for _ in range(25):
            ls.mark_strategy_step_done(np.array([False]))
            
        # Trigger
        state_t1 = self._gs(blue_score=0.5, red_score=0.0)
        _, _, aux = ls.strategy_for_step(state_t1)
        self.assertTrue(bool(aux["z_resampled_actual"].item()))
        
        stats = ls.event_refresh_rollout_stats()
        self.assertEqual(stats["latent_refresh_count"], 1.0)
        self.assertEqual(stats["latent_refresh_reason_score_change"], 1.0)

    def test_enemy_carrier_near_base_trigger(self):
        trainer = _make_trainer(1)
        ls = LatentStrategyState(trainer)
        ls.reset()
        
        # Step 0: enemy flag grabbed but far from base
        state_t0 = self._gs(enemy_flag=1.0, carrier_dist=0.8)
        ls.strategy_for_step(state_t0)
        
        # Clear gap
        for _ in range(25):
            ls.mark_strategy_step_done(np.array([False]))
            
        # Trigger: moves near base
        state_t1 = self._gs(enemy_flag=1.0, carrier_dist=0.1)
        _, _, aux = ls.strategy_for_step(state_t1)
        self.assertTrue(bool(aux["z_resampled_actual"].item()))
        
        stats = ls.event_refresh_rollout_stats()
        self.assertEqual(stats["latent_refresh_count"], 1.0)
        self.assertEqual(stats["latent_refresh_reason_near_base"], 1.0)

    def test_friendly_carrier_near_base_trigger(self):
        trainer = _make_trainer(1)
        ls = LatentStrategyState(trainer)
        ls.reset()
        
        # Step 0: friendly flag grabbed but far
        state_t0 = self._gs(friendly_flag=1.0, carrier_dist=0.8)
        ls.strategy_for_step(state_t0)
        
        # Clear gap
        for _ in range(25):
            ls.mark_strategy_step_done(np.array([False]))
            
        # Trigger: moves near base
        state_t1 = self._gs(friendly_flag=1.0, carrier_dist=0.1)
        _, _, aux = ls.strategy_for_step(state_t1)
        self.assertTrue(bool(aux["z_resampled_actual"].item()))
        
        stats = ls.event_refresh_rollout_stats()
        self.assertEqual(stats["latent_refresh_count"], 1.0)
        self.assertEqual(stats["latent_refresh_reason_near_base"], 1.0)

    def test_guardrails_min_gap(self):
        trainer = _make_trainer(1, min_gap=20)
        ls = LatentStrategyState(trainer)
        ls.reset()
        
        # Step 0
        state_t0 = self._gs(enemy_flag=0.0)
        ls.strategy_for_step(state_t0)
        
        # Clear gap only 5 steps
        for _ in range(5):
            ls.mark_strategy_step_done(np.array([False]))
            
        # Try to trigger
        state_t1 = self._gs(enemy_flag=1.0)
        _, _, aux = ls.strategy_for_step(state_t1)
        # Should NOT trigger due to min gap (5 < 20)
        self.assertFalse(bool(aux["z_resampled_actual"].item()))

    def test_guardrails_max_refreshes(self):
        trainer = _make_trainer(1, min_gap=5, max_refreshes=2)
        ls = LatentStrategyState(trainer)
        ls.reset()
        
        # Step 0
        state_t0 = self._gs(enemy_flag=0.0)
        ls.strategy_for_step(state_t0)
        
        # Refresh 1
        for _ in range(10):
            ls.mark_strategy_step_done(np.array([False]))
        ls.strategy_for_step(self._gs(enemy_flag=1.0))
        
        # Refresh 2
        for _ in range(10):
            ls.mark_strategy_step_done(np.array([False]))
        # Grab resets to 0 first so we can grab again:
        ls.strategy_for_step(self._gs(enemy_flag=0.0))
        for _ in range(10):
            ls.mark_strategy_step_done(np.array([False]))
        ls.strategy_for_step(self._gs(enemy_flag=1.0))
        
        # Refresh 3 (Should fail)
        for _ in range(10):
            ls.mark_strategy_step_done(np.array([False]))
        ls.strategy_for_step(self._gs(enemy_flag=0.0))
        for _ in range(10):
            ls.mark_strategy_step_done(np.array([False]))
        _, _, aux = ls.strategy_for_step(self._gs(enemy_flag=1.0))
        self.assertFalse(bool(aux["z_resampled_actual"].item()))
        
        stats = ls.event_refresh_rollout_stats()
        self.assertEqual(stats["latent_refresh_count"], 2.0)

    def test_transition_matrix(self):
        trainer = _make_trainer(1)
        # Override sampling to return 2 initially
        trainer.model._categorical_argmax_or_sample = lambda dist, **kwargs: torch.tensor([2], dtype=torch.long)
        
        ls = LatentStrategyState(trainer)
        ls.reset()
        
        # Step 0: start. Samples z0 = 2.
        state_t0 = self._gs(enemy_flag=0.0)
        z0, _, _ = ls.strategy_for_step(state_t0)
        self.assertEqual(int(z0.item()), 2)
        ls.mark_strategy_step_done(np.array([False]))
        
        # Clear gap
        for _ in range(25):
            ls.mark_strategy_step_done(np.array([False]))
            
        # Trigger event and transition to z = 3
        trainer.model._categorical_argmax_or_sample = lambda dist, **kwargs: torch.tensor([3], dtype=torch.long)
        state_t1 = self._gs(enemy_flag=1.0)
        z1, _, _ = ls.strategy_for_step(state_t1)
        self.assertEqual(int(z1.item()), 3)
        
        stats = ls.event_refresh_rollout_stats()
        # The transition must be 2 -> 3
        self.assertEqual(stats["latent_refresh_z2_to_z3"], 1.0)
        # Other transitions must be 0.0
        self.assertEqual(stats["latent_refresh_z0_to_z0"], 0.0)
        self.assertEqual(stats["latent_refresh_z2_to_z2"], 0.0)

if __name__ == "__main__":
    unittest.main()
