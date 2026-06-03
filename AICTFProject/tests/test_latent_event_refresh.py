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

def _load_latent_strategy_module():
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
    return module


_lss_module = _load_latent_strategy_module()
LatentStrategyState = _lss_module.LatentStrategyState
_v3i3_target_from_items = _lss_module._v3i3_target_from_items
_v3i3_resolve_target = _lss_module._v3i3_resolve_target

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
    v3i3_pref_enabled: bool = False,
    v3i3_log_enabled: bool = False,
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

        # v3i3 configs (defaults: disabled; opt-in per test)
        latent_v3i3_event_preference_enabled=v3i3_pref_enabled,
        latent_v3i3_event_preference_coef=0.0 if not v3i3_pref_enabled else 0.03,
        latent_v3i3_event_preference_temperature=0.75,
        latent_v3i3_event_preference_min_bucket_count=4,
        latent_v3i3_event_preference_min_distinct_z=2,
        latent_v3i3_event_preference_buffer_size=1024,
        latent_v3i3_event_preference_warmup_steps=0,
        latent_v3i3_refresh_log_enabled=v3i3_log_enabled,
        latent_v3i3_refresh_log_path="",
        global_step=0,
        cfg=SimpleNamespace(fixed_opponent_tag="OP3"),
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

class V3i3PendingRefreshCaptureTests(unittest.TestCase):
    """v3i3 captures per-refresh (state, prev_z, next_z, event_type, flag) on each event refresh."""

    def _gs(self, enemy_flag=0.0, friendly_flag=0.0, blue_score=0.0, red_score=0.0, carrier_dist=1.0) -> torch.Tensor:
        state = torch.zeros((1, 170), dtype=torch.float32)
        state[0, 10] = enemy_flag
        state[0, 11] = friendly_flag
        state[0, 14] = blue_score
        state[0, 15] = red_score
        state[0, 23] = carrier_dist
        return state

    def test_capture_on_enemy_flag_refresh(self) -> None:
        trainer = _make_trainer(1, v3i3_log_enabled=True)
        ls = LatentStrategyState(trainer)
        ls.reset()
        ls.strategy_for_step(self._gs(enemy_flag=0.0))
        ls.mark_strategy_step_done(np.array([False]))
        for _ in range(25):
            ls.mark_strategy_step_done(np.array([False]))
        ls.strategy_for_step(self._gs(enemy_flag=1.0))

        pending = ls.pending_refresh_records.get(0, [])
        self.assertEqual(len(pending), 1)
        rec = pending[0]
        self.assertEqual(int(rec["reason_id"]), 0)  # enemy_flag
        self.assertIn("refresh_state", rec)
        self.assertEqual(rec["refresh_state"].shape, (170,))
        self.assertIn("return_at_refresh", rec)
        # flag_state_bucket = 2*enemy_has + we_have = 2*1 + 0 = 2
        self.assertEqual(int(rec["flag_state_bucket"]), 2)

    def test_event_type_priority_when_multiple_triggers_fire(self) -> None:
        """Enemy_flag wins over score_change when both fire on the same step."""
        trainer = _make_trainer(1, v3i3_log_enabled=True)
        ls = LatentStrategyState(trainer)
        ls.reset()
        ls.strategy_for_step(self._gs(enemy_flag=0.0))
        for _ in range(25):
            ls.mark_strategy_step_done(np.array([False]))
        # Simultaneously: enemy flag grabbed AND score changes.
        ls.strategy_for_step(self._gs(enemy_flag=1.0, blue_score=0.5))
        pending = ls.pending_refresh_records.get(0, [])
        self.assertEqual(len(pending), 1)
        self.assertEqual(int(pending[0]["reason_id"]), 0)  # enemy_flag priority

    def test_finalize_on_episode_done_populates_buffer(self) -> None:
        """After episode done, future_return + opponent_id land in the cumulative buffer."""
        trainer = _make_trainer(1, v3i3_pref_enabled=True, v3i3_log_enabled=True)
        ls = LatentStrategyState(trainer)
        ls.reset()
        ls.strategy_for_step(self._gs(enemy_flag=0.0))
        for _ in range(25):
            ls.mark_strategy_step_done(np.array([False]))
        # Trigger refresh; return_at_refresh = 0.0 (no reward accumulated yet)
        ls.strategy_for_step(self._gs(enemy_flag=1.0))
        self.assertEqual(len(ls.pending_refresh_records[0]), 1)
        # Finalize with a known episode return.
        info = {"opponent_kind": "scripted", "scripted_tag": "OP3"}
        ls.finalize_v3i3_refresh_records(0, info, episode_return=2.5)
        self.assertEqual(len(ls.pending_refresh_records[0]), 0)
        self.assertEqual(len(ls.rollout_refresh_records), 1)
        finalized = ls.rollout_refresh_records[0]
        self.assertAlmostEqual(float(finalized["future_return"]), 2.5, places=5)
        self.assertAlmostEqual(float(finalized["return_from_now_to_end"]), 2.5, places=5)
        self.assertEqual(int(finalized["opponent_id"]), 2)  # OP3 -> 2
        # Preference buffer received the minimal training entry.
        self.assertEqual(len(ls.refresh_preference_buffer), 1)
        b = ls.refresh_preference_buffer[0]
        self.assertEqual(int(b["opponent_id"]), 2)
        self.assertEqual(int(b["event_type"]), 0)
        self.assertEqual(int(b["flag_state_bucket"]), 2)
        self.assertAlmostEqual(float(b["future_return"]), 2.5, places=5)

    def test_finalize_no_op_when_v3i3_disabled(self) -> None:
        """Disabling v3i3 entirely should leave pending state empty and skip buffer writes."""
        trainer = _make_trainer(1)  # v3i3 disabled
        ls = LatentStrategyState(trainer)
        ls.reset()
        ls.strategy_for_step(self._gs(enemy_flag=0.0))
        for _ in range(25):
            ls.mark_strategy_step_done(np.array([False]))
        ls.strategy_for_step(self._gs(enemy_flag=1.0))
        self.assertEqual(len(ls.pending_refresh_records[0]), 0)
        ls.finalize_v3i3_refresh_records(0, {"opponent_kind": "scripted", "scripted_tag": "OP3"}, episode_return=1.0)
        self.assertEqual(len(ls.rollout_refresh_records), 0)
        self.assertEqual(len(ls.refresh_preference_buffer), 0)

    def test_episode_id_increments_on_done(self) -> None:
        trainer = _make_trainer(1, v3i3_log_enabled=True)
        ls = LatentStrategyState(trainer)
        ls.reset()
        self.assertEqual(int(ls.episode_id_per_env[0].item()), 0)
        ls.mark_strategy_step_done(np.array([True]))
        self.assertEqual(int(ls.episode_id_per_env[0].item()), 1)

    def test_clear_rollout_refresh_records_preserves_cumulative_buffer(self) -> None:
        trainer = _make_trainer(1, v3i3_pref_enabled=True)
        ls = LatentStrategyState(trainer)
        ls.reset()
        ls.strategy_for_step(self._gs(enemy_flag=0.0))
        for _ in range(25):
            ls.mark_strategy_step_done(np.array([False]))
        ls.strategy_for_step(self._gs(enemy_flag=1.0))
        ls.finalize_v3i3_refresh_records(0, {"opponent_kind": "scripted", "scripted_tag": "OP3"}, episode_return=1.0)
        self.assertEqual(len(ls.rollout_refresh_records), 1)
        self.assertEqual(len(ls.refresh_preference_buffer), 1)
        ls.clear_rollout_refresh_records()
        self.assertEqual(len(ls.rollout_refresh_records), 0)
        # The teacher's library survives across rollouts.
        self.assertEqual(len(ls.refresh_preference_buffer), 1)


class V3i3HierarchicalFallbackTests(unittest.TestCase):
    """The v3i3 target-lookup falls through (opp,event,flag) -> (opp,event) -> (opp)."""

    def _make_buffer_with_items(self, items: list) -> tuple[dict, dict, dict]:
        """``items`` is a list of (opp, event, flag, z, future_return) tuples."""
        by_full, by_oe, by_o = {}, {}, {}
        for opp, ev, fl, z, fr in items:
            pair = (int(z), float(fr))
            by_full.setdefault((int(opp), int(ev), int(fl)), []).append(pair)
            by_oe.setdefault((int(opp), int(ev)), []).append(pair)
            by_o.setdefault((int(opp),), []).append(pair)
        return by_full, by_oe, by_o

    def test_full_bucket_match_when_sufficient_evidence(self) -> None:
        by_full, by_oe, by_o = self._make_buffer_with_items([
            (2, 0, 2, 0, 1.0),
            (2, 0, 2, 0, 1.2),
            (2, 0, 2, 1, -0.5),
            (2, 0, 2, 1, -0.3),
        ])
        target, level = _v3i3_resolve_target(
            opponent_id=2, event_type=0, flag_state_bucket=2,
            by_full=by_full, by_oe=by_oe, by_o=by_o,
            latent_k=4, min_count=4, min_distinct_z=2, temperature=0.75,
            target_cache={},
        )
        self.assertEqual(level, "full")
        self.assertIsNotNone(target)
        # z0 has higher avg return than z1, so target[0] > target[1].
        self.assertGreater(float(target[0]), float(target[1]))
        # All probs sum to 1.
        self.assertAlmostEqual(float(target.sum()), 1.0, places=5)

    def test_falls_back_to_oe_when_flag_undersampled(self) -> None:
        by_full, by_oe, by_o = self._make_buffer_with_items([
            # Sparse data at (opp=2, event=0, flag=3) -- only 1 record
            (2, 0, 3, 0, 1.0),
            # Plenty at (opp=2, event=0) overall, across mixed flags
            (2, 0, 0, 0, 0.5),
            (2, 0, 1, 1, -0.5),
            (2, 0, 2, 0, 0.8),
            (2, 0, 2, 1, -0.2),
        ])
        target, level = _v3i3_resolve_target(
            opponent_id=2, event_type=0, flag_state_bucket=3,
            by_full=by_full, by_oe=by_oe, by_o=by_o,
            latent_k=4, min_count=4, min_distinct_z=2, temperature=0.75,
            target_cache={},
        )
        self.assertEqual(level, "oe")
        self.assertIsNotNone(target)

    def test_falls_back_to_o_when_event_undersampled(self) -> None:
        by_full, by_oe, by_o = self._make_buffer_with_items([
            (2, 0, 0, 0, 0.5),
            (2, 1, 0, 0, 0.6),
            (2, 2, 0, 1, -0.5),
            (2, 3, 0, 1, -0.4),
        ])
        # Asking for (opp=2, event=0, flag=2): not enough at full, not enough at (opp=2, event=0).
        target, level = _v3i3_resolve_target(
            opponent_id=2, event_type=0, flag_state_bucket=2,
            by_full=by_full, by_oe=by_oe, by_o=by_o,
            latent_k=4, min_count=4, min_distinct_z=2, temperature=0.75,
            target_cache={},
        )
        self.assertEqual(level, "o")
        self.assertIsNotNone(target)

    def test_returns_none_when_no_level_has_enough(self) -> None:
        by_full, by_oe, by_o = self._make_buffer_with_items([
            (2, 0, 0, 0, 0.5),  # only one record total for opp=2
        ])
        target, level = _v3i3_resolve_target(
            opponent_id=2, event_type=0, flag_state_bucket=2,
            by_full=by_full, by_oe=by_oe, by_o=by_o,
            latent_k=4, min_count=4, min_distinct_z=2, temperature=0.75,
            target_cache={},
        )
        self.assertIsNone(target)
        self.assertIsNone(level)

    def test_target_cache_avoids_recomputation(self) -> None:
        """Cache key includes the level so resolving twice doesn't recompute."""
        by_full, by_oe, by_o = self._make_buffer_with_items([
            (2, 0, 2, 0, 1.0),
            (2, 0, 2, 0, 1.2),
            (2, 0, 2, 1, -0.5),
            (2, 0, 2, 1, -0.3),
        ])
        cache: dict = {}
        t1, lvl1 = _v3i3_resolve_target(
            opponent_id=2, event_type=0, flag_state_bucket=2,
            by_full=by_full, by_oe=by_oe, by_o=by_o,
            latent_k=4, min_count=4, min_distinct_z=2, temperature=0.75,
            target_cache=cache,
        )
        # Cache should have at least the resolved level's entry.
        self.assertIn(("full", (2, 0, 2)), cache)
        # Second lookup hits the cache (target arrays are equal).
        t2, lvl2 = _v3i3_resolve_target(
            opponent_id=2, event_type=0, flag_state_bucket=2,
            by_full=by_full, by_oe=by_oe, by_o=by_o,
            latent_k=4, min_count=4, min_distinct_z=2, temperature=0.75,
            target_cache=cache,
        )
        self.assertEqual(lvl1, lvl2)
        self.assertTrue(np.allclose(t1, t2))

    def test_undersampled_returns_none_on_min_distinct(self) -> None:
        """All records concentrated on one z -> reject even if count is high."""
        by_full, by_oe, by_o = self._make_buffer_with_items([
            (2, 0, 2, 0, 1.0),
            (2, 0, 2, 0, 1.2),
            (2, 0, 2, 0, 0.8),
            (2, 0, 2, 0, 0.9),
        ])
        # Only one distinct z observed; min_distinct_z=2 should reject.
        t = _v3i3_target_from_items(
            by_full[(2, 0, 2)],
            latent_k=4, min_count=4, min_distinct_z=2, temperature=0.75,
        )
        self.assertIsNone(t)


if __name__ == "__main__":
    unittest.main()
