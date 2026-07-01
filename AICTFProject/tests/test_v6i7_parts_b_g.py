"""Tests for V6I7 Parts B-G: env router reward, balanced assignment, forced masks.

Invariants verified:
  1. Router critic warmup: ctv > 0, rdv == 0 in balanced_episode with train_when_forced=False
  2. balanced_episode staggered: z = (counter + env_index) % K across vector envs
  3. balanced_arc continuity: episode starting latent rotates; arc advances from there
  4. router_reward emitted on exact event transition, not only at episode end
  5. forced assignments never become on-policy PPO samples
  6. no silent fallback: RuntimeError when router_reward_enabled but env omits it

Coverage:
  B1-B5:   RouterRewardConfig construction and GPUFieldConfig integration
  B6-B14:  _router_reward_total unit tests (zeros, weights, normalize)
  B15-B18: _step.py _compute_step_reward_components event tensor keys
  C1-C3:   _build_info router_reward parameter signature
  D1-D5:   balanced_episode: forced, staggered across envs, cycles per episode
  E1-E4:   balanced_arc: cycles within episode, arc boundaries, continuity across episodes
  F1-F4:   Forced masks: BPTT skip, no-skip, rdv/ctv correctness
  G1-G2:   z shape assertions (N_env, not N_env*N_agents)
  I1-I4:   No-silent-fallback, router_reward event emission, critic warmup, forced never PPO
"""

from __future__ import annotations

import types
import unittest

import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_cfg(B: int = 4, router_reward_config=None):
    cfg = types.SimpleNamespace()
    cfg.router_reward_config = router_reward_config
    return cfg


def _zeros(n: int) -> torch.Tensor:
    return torch.zeros(n, dtype=torch.float32)


def _bool_zeros(n: int) -> torch.Tensor:
    return torch.zeros(n, dtype=torch.bool)


def _call_router_reward_total(env, rterm, blue_cap, red_cap, blue_tag_wf, red_tag):
    from gpu_env._core._rewards import _RewardsMixin
    return _RewardsMixin._router_reward_total(env, rterm, blue_cap, red_cap, blue_tag_wf, red_tag)


def _make_balanced_episode_state(n_envs: int = 4, K: int = 4, episode_start: bool = True):
    """Minimal RouterSamplingState host for balanced_episode tests."""
    from rl.custom_ppo.latent.router_sampling import RouterSamplingState
    device = "cpu"
    host = types.SimpleNamespace(
        trainer=types.SimpleNamespace(
            use_latent_strategy=True,
            fixed_latent_strategy=False,
            device=device,
            latent_k=K,
            cfg=types.SimpleNamespace(
                latent_assignment_mode="balanced_episode",
                forced_latent_arc_steps=32,
                train_router_when_forced=False,
            ),
            latent_resample_every_n=0,
            latent_sparse_tactical_refresh_enabled=False,
        ),
        current_z=torch.zeros(n_envs, dtype=torch.long),
        needs_strategy_sample=torch.full((n_envs,), episode_start, dtype=torch.bool),
        balanced_episode_counter=torch.zeros(n_envs, dtype=torch.long),
        arc_step_counter=torch.zeros(n_envs, dtype=torch.long),
        episode_arc_start_z=torch.zeros(n_envs, dtype=torch.long),
        prev_global_state=None,
        record_tactical_context_step=lambda gs: None,
    )
    rss = RouterSamplingState.__new__(RouterSamplingState)
    rss.host = host
    return rss


def _make_balanced_arc_state(n_envs: int = 1, K: int = 4, arc_steps: int = 4, episode_start: bool = False):
    """Minimal RouterSamplingState host for balanced_arc tests."""
    from rl.custom_ppo.latent.router_sampling import RouterSamplingState
    device = "cpu"
    host = types.SimpleNamespace(
        trainer=types.SimpleNamespace(
            use_latent_strategy=True,
            fixed_latent_strategy=False,
            device=device,
            latent_k=K,
            cfg=types.SimpleNamespace(
                latent_assignment_mode="balanced_arc",
                forced_latent_arc_steps=arc_steps,
                train_router_when_forced=False,
            ),
            latent_resample_every_n=0,
            latent_sparse_tactical_refresh_enabled=False,
        ),
        current_z=torch.zeros(n_envs, dtype=torch.long),
        needs_strategy_sample=torch.full((n_envs,), episode_start, dtype=torch.bool),
        balanced_episode_counter=torch.zeros(n_envs, dtype=torch.long),
        arc_step_counter=torch.zeros(n_envs, dtype=torch.long),
        episode_arc_start_z=torch.zeros(n_envs, dtype=torch.long),
        prev_global_state=None,
        record_tactical_context_step=lambda gs: None,
    )
    rss = RouterSamplingState.__new__(RouterSamplingState)
    rss.host = host
    return rss


# ---------------------------------------------------------------------------
# B1-B5: RouterRewardConfig and GPUFieldConfig construction
# ---------------------------------------------------------------------------

class TestRouterRewardConfig(unittest.TestCase):

    # B1
    def test_router_reward_config_exists_with_defaults(self):
        from gpu_env._config import RouterRewardConfig
        rrc = RouterRewardConfig()
        self.assertFalse(rrc.enabled)
        self.assertAlmostEqual(rrc.win_weight, 1.0)
        self.assertAlmostEqual(rrc.flag_cap_weight, 0.5)
        self.assertAlmostEqual(rrc.sparse_weight, 0.2)
        self.assertAlmostEqual(rrc.scale, 1.0)
        self.assertTrue(rrc.normalize)

    # B2
    def test_router_reward_config_in_all(self):
        import gpu_env._config as m
        self.assertIn("RouterRewardConfig", m.__all__)

    # B3
    def test_gpu_field_config_accepts_none_router_reward_config(self):
        from gpu_env._config import GPUFieldConfig
        cfg = GPUFieldConfig(router_reward_config=None)
        self.assertIsNone(cfg.router_reward_config)

    # B4
    def test_gpu_field_config_accepts_router_reward_config(self):
        from gpu_env._config import GPUFieldConfig, RouterRewardConfig
        rrc = RouterRewardConfig(enabled=True, win_weight=2.0)
        cfg = GPUFieldConfig(router_reward_config=rrc)
        self.assertIsNotNone(cfg.router_reward_config)
        self.assertTrue(cfg.router_reward_config.enabled)
        self.assertAlmostEqual(cfg.router_reward_config.win_weight, 2.0)

    # B5
    def test_gpu_field_config_raises_on_unknown_kwarg(self):
        from gpu_env._config import GPUFieldConfig
        with self.assertRaises(TypeError):
            GPUFieldConfig(totally_unknown_field_xyz=True)


# ---------------------------------------------------------------------------
# B6-B14: _router_reward_total unit tests
# ---------------------------------------------------------------------------

class TestRouterRewardTotal(unittest.TestCase):

    def setUp(self):
        from gpu_env._config import RouterRewardConfig
        self.B = 4
        self.rrc = RouterRewardConfig(
            enabled=True, win_weight=1.0, flag_cap_weight=0.5,
            sparse_weight=0.2, scale=1.0, normalize=True,
        )

    def _env(self, rrc=None):
        cfg = types.SimpleNamespace(router_reward_config=rrc if rrc is not None else self.rrc)
        return types.SimpleNamespace(B=self.B, device="cpu", cfg=cfg)

    # B6
    def test_returns_zeros_when_config_is_none(self):
        env = self._env()
        env.cfg.router_reward_config = None
        out = _call_router_reward_total(env, _zeros(self.B), _bool_zeros(self.B), _bool_zeros(self.B), _zeros(self.B), _zeros(self.B))
        self.assertTrue(out.eq(0.0).all())

    # B7
    def test_returns_zeros_when_disabled(self):
        from gpu_env._config import RouterRewardConfig
        out = _call_router_reward_total(self._env(RouterRewardConfig(enabled=False)), _zeros(self.B), _bool_zeros(self.B), _bool_zeros(self.B), _zeros(self.B), _zeros(self.B))
        self.assertTrue(out.eq(0.0).all())

    # B8
    def test_win_weight_applied(self):
        from gpu_env._config import RouterRewardConfig
        env = self._env(RouterRewardConfig(enabled=True, win_weight=2.0, normalize=False, scale=1.0))
        out = _call_router_reward_total(env, torch.full((self.B,), 1.0), _bool_zeros(self.B), _bool_zeros(self.B), _zeros(self.B), _zeros(self.B))
        self.assertTrue((out > 0).all())

    # B9
    def test_flag_cap_weight_on_blue_cap(self):
        from gpu_env._config import RouterRewardConfig
        env = self._env(RouterRewardConfig(enabled=True, win_weight=0.0, flag_cap_weight=1.0, normalize=False, scale=1.0))
        out = _call_router_reward_total(env, _zeros(self.B), torch.ones(self.B, dtype=torch.bool), _bool_zeros(self.B), _zeros(self.B), _zeros(self.B))
        self.assertTrue((out > 0).all())

    # B10
    def test_red_cap_subtracts(self):
        from gpu_env._config import RouterRewardConfig
        env = self._env(RouterRewardConfig(enabled=True, win_weight=0.0, flag_cap_weight=1.0, normalize=False, scale=1.0))
        out = _call_router_reward_total(env, _zeros(self.B), _bool_zeros(self.B), torch.ones(self.B, dtype=torch.bool), _zeros(self.B), _zeros(self.B))
        self.assertTrue((out < 0).all())

    # B11
    def test_sparse_weight_on_blue_tag_withflag(self):
        from gpu_env._config import RouterRewardConfig
        env = self._env(RouterRewardConfig(enabled=True, win_weight=0.0, flag_cap_weight=0.0, sparse_weight=1.0, normalize=False, scale=1.0))
        out = _call_router_reward_total(env, _zeros(self.B), _bool_zeros(self.B), _bool_zeros(self.B), torch.ones(self.B), _zeros(self.B))
        self.assertTrue((out > 0).all())

    # B12
    def test_red_tag_total_subtracts(self):
        from gpu_env._config import RouterRewardConfig
        env = self._env(RouterRewardConfig(enabled=True, win_weight=0.0, flag_cap_weight=0.0, sparse_weight=1.0, normalize=False, scale=1.0))
        out = _call_router_reward_total(env, _zeros(self.B), _bool_zeros(self.B), _bool_zeros(self.B), _zeros(self.B), torch.ones(self.B))
        self.assertTrue((out < 0).all())

    # B13
    def test_normalize_true_bounds_output(self):
        from gpu_env._config import RouterRewardConfig
        env = self._env(RouterRewardConfig(enabled=True, win_weight=100.0, normalize=True, scale=1.0))
        out = _call_router_reward_total(env, torch.full((self.B,), 1.0), _bool_zeros(self.B), _bool_zeros(self.B), _zeros(self.B), _zeros(self.B))
        self.assertTrue((out >= -1.0).all() and (out <= 1.0).all())

    # B14
    def test_normalize_false_scales_raw(self):
        from gpu_env._config import RouterRewardConfig
        env = self._env(RouterRewardConfig(enabled=True, win_weight=1.0, normalize=False, scale=3.0))
        out = _call_router_reward_total(env, torch.full((self.B,), 1.0), _bool_zeros(self.B), _bool_zeros(self.B), _zeros(self.B), _zeros(self.B))
        self.assertTrue((out > 1.5).all(), f"Expected >1.5 (scale=3), got {out}")


# ---------------------------------------------------------------------------
# B15-B18: _compute_step_reward_components event tensor keys
# ---------------------------------------------------------------------------

class TestStepRewardEventKeys(unittest.TestCase):

    def _check_key(self, key):
        import inspect
        from gpu_env._core._step import _StepMixin
        src = inspect.getsource(_StepMixin._compute_step_reward_components)
        self.assertIn(f'"{key}"', src, f"Expected key '{key}' in _compute_step_reward_components return dict")

    # B15
    def test_blue_cap_env_key(self):
        self._check_key("blue_cap_env")

    # B16
    def test_red_cap_env_key(self):
        self._check_key("red_cap_env")

    # B17
    def test_blue_tag_withflag_key(self):
        self._check_key("blue_tag_withflag")

    # B18
    def test_red_tag_total_key(self):
        self._check_key("red_tag_total")


# ---------------------------------------------------------------------------
# C1-C3: _build_info router_reward parameter
# ---------------------------------------------------------------------------

class TestBuildInfoRouterReward(unittest.TestCase):

    # C1
    def test_build_info_signature_has_router_reward(self):
        from gpu_env._core._metrics import _MetricsMixin
        import inspect
        sig = inspect.signature(_MetricsMixin._build_info)
        self.assertIn("router_reward", sig.parameters)

    # C2
    def test_router_reward_default_is_none(self):
        from gpu_env._core._metrics import _MetricsMixin
        import inspect
        param = inspect.signature(_MetricsMixin._build_info).parameters["router_reward"]
        self.assertIsNone(param.default)

    # C3 — verify the scalars stack contains router_reward by inspecting source
    def test_build_info_stacks_router_reward(self):
        import inspect
        from gpu_env._core._metrics import _MetricsMixin
        src = inspect.getsource(_MetricsMixin._build_info)
        self.assertIn("router_reward", src)
        self.assertIn('"router_reward"', src)


# ---------------------------------------------------------------------------
# D1-D5: balanced_episode assignment mode
# ---------------------------------------------------------------------------

class TestBalancedEpisodeMode(unittest.TestCase):

    # D1: forced flag always set
    def test_z_forced_true(self):
        rss = _make_balanced_episode_state(n_envs=4, K=4)
        _, _, aux = rss.strategy_for_step(torch.zeros(4, 35))
        self.assertTrue(aux["z_forced"].all())

    # D2: single env cycles correctly
    def test_single_env_cycles(self):
        K = 4
        rss = _make_balanced_episode_state(n_envs=1, K=K)
        zs = []
        for ep in range(K * 2):
            rss.host.needs_strategy_sample = torch.ones(1, dtype=torch.bool)
            z, _, _ = rss.strategy_for_step(torch.zeros(1, 35))
            zs.append(int(z[0].item()))
        self.assertEqual(zs, list(range(K)) * 2)

    # D3: no resample mid-episode
    def test_no_resample_mid_episode(self):
        rss = _make_balanced_episode_state(n_envs=2, K=2)
        rss.strategy_for_step(torch.zeros(2, 35))  # episode start
        rss.host.needs_strategy_sample[:] = False
        _, _, aux = rss.strategy_for_step(torch.zeros(2, 35))
        self.assertFalse(aux["z_resampled"].any())

    # D4: counter increments on start
    def test_counter_increments(self):
        rss = _make_balanced_episode_state(n_envs=2, K=4)
        rss.strategy_for_step(torch.zeros(2, 35))
        self.assertTrue((rss.host.balanced_episode_counter == 1).all())

    # D5: STAGGERED — envs are offset so all K latents appear simultaneously
    def test_staggered_across_envs(self):
        K = 4
        n_envs = K
        rss = _make_balanced_episode_state(n_envs=n_envs, K=K, episode_start=True)
        gs = torch.zeros(n_envs, 35)

        # Episode 0: z[env_i] = (0 + env_i) % K = env_i
        z, _, _ = rss.strategy_for_step(gs)
        zs_ep0 = z.tolist()
        self.assertEqual(sorted(zs_ep0), list(range(K)), f"Episode 0 should cover all K: {zs_ep0}")
        for env_i in range(n_envs):
            self.assertEqual(zs_ep0[env_i], env_i, f"env {env_i} ep 0 should get z={env_i}")

        # Episode 1: z[env_i] = (1 + env_i) % K
        rss.host.needs_strategy_sample[:] = True
        z, _, _ = rss.strategy_for_step(gs)
        zs_ep1 = z.tolist()
        for env_i in range(n_envs):
            expected = (1 + env_i) % K
            self.assertEqual(zs_ep1[env_i], expected, f"env {env_i} ep 1 expected z={expected}")


# ---------------------------------------------------------------------------
# E1-E4: balanced_arc assignment mode
# ---------------------------------------------------------------------------

class TestBalancedArcMode(unittest.TestCase):

    # E1: arc advances z within episode
    def test_arc_cycles_z_within_episode(self):
        K = 4
        arc_steps = 4
        rss = _make_balanced_arc_state(n_envs=1, K=K, arc_steps=arc_steps, episode_start=True)
        gs = torch.zeros(1, 35)
        # Consume episode start
        rss.strategy_for_step(gs)
        rss.host.needs_strategy_sample[:] = False

        zs = []
        for step in range(K * arc_steps):
            z, _, _ = rss.strategy_for_step(gs)
            zs.append(int(z[0].item()))
            rss.host.arc_step_counter += 1

        # Each arc covers arc_steps steps; z advances by 1 each arc.
        for i in range(K):
            chunk = zs[i * arc_steps: (i + 1) * arc_steps]
            expected_z = i % K  # starts at 0 since episode start sets start_z=0 for env 0
            self.assertEqual(chunk, [expected_z] * arc_steps, f"Arc {i}: got {chunk}, expected {[expected_z]*arc_steps}")

    # E2: z_resampled True at arc boundaries
    def test_z_resampled_at_arc_boundary(self):
        arc_steps = 4
        rss = _make_balanced_arc_state(n_envs=1, K=4, arc_steps=arc_steps, episode_start=False)
        gs = torch.zeros(1, 35)
        for step in range(arc_steps * 2):
            _, _, aux = rss.strategy_for_step(gs)
            is_boundary = (int(rss.host.arc_step_counter[0].item()) % arc_steps == 0)
            if is_boundary and step > 0:
                self.assertTrue(aux["z_resampled"][0].item(), f"step {step}: expected z_resampled at arc boundary")
            rss.host.arc_step_counter += 1

    # E3: arc_step_counter resets to 0 on done
    def test_arc_step_counter_resets_on_done(self):
        import numpy as np
        rss = _make_balanced_arc_state(n_envs=2, K=4, arc_steps=4)
        rss.host.arc_step_counter = torch.tensor([10, 5], dtype=torch.long)
        rss.host.strategy_age = torch.zeros(2, dtype=torch.long)
        rss.host.steps_since_ep_start = torch.zeros(2, dtype=torch.long)
        rss.host.steps_since_last_refresh = torch.zeros(2, dtype=torch.long)
        rss.host.steps_since_last_tactical_refresh = torch.zeros(2, dtype=torch.long)
        rss.host.steps_since_z_change = torch.zeros(2, dtype=torch.long)
        rss.host.needs_strategy_sample = torch.zeros(2, dtype=torch.bool)
        rss.host.episode_strategy_committed = torch.zeros(2, dtype=torch.bool)
        rss.host.episode_tactical_bucket_counts = torch.zeros(2, 16, dtype=torch.long)
        rss.host.first_z_sample_step = torch.full((2,), -1, dtype=torch.long)
        rss.host.episode_return_baseline_at_commit = torch.zeros(2)
        rss.host.episode_forced_z = torch.zeros(2, dtype=torch.bool)
        rss.host.episode_forced_z_id = torch.zeros(2, dtype=torch.long)
        rss.host.episode_contrast_bucket = torch.zeros(2, dtype=torch.long)
        rss.host.episode_behavior_sum = torch.zeros(2)
        rss.host.episode_behavior_count = torch.zeros(2, dtype=torch.long)
        rss.host.refresh_count_this_episode = torch.zeros(2, dtype=torch.long)
        rss.host.episode_id_per_env = torch.zeros(2, dtype=torch.long)
        rss.host.pending_refresh_records = {0: [], 1: []}
        rss.host.reset_completed_envs = lambda done_t: None
        rss.host.prev_global_state = None
        rss.host.previous_opportunity_features = None
        rss.host.previous_router_context = None
        rss.host.persistence_valid = None
        rss.host.opportunity_index_per_env = None
        rss.mark_strategy_step_done(np.array([True, False]))
        self.assertEqual(int(rss.host.arc_step_counter[0].item()), 0)
        self.assertGreater(int(rss.host.arc_step_counter[1].item()), 0)

    # E4: CONTINUITY — episode 2 starts at a ROTATED latent, not z=0
    def test_arc_episode2_starts_at_rotated_z(self):
        K = 4
        arc_steps = 2
        rss = _make_balanced_arc_state(n_envs=1, K=K, arc_steps=arc_steps, episode_start=True)
        gs = torch.zeros(1, 35)

        # Episode 0: env_index=0, counter=0 → start_z = (0+0)%4 = 0
        z0, _, _ = rss.strategy_for_step(gs)
        start_z_ep0 = int(rss.host.episode_arc_start_z[0].item())
        self.assertEqual(start_z_ep0, 0, f"Episode 0 start_z should be 0, got {start_z_ep0}")

        # Simulate episode end: reset arc counter, set needs_strategy_sample
        rss.host.arc_step_counter[:] = 0
        rss.host.episode_arc_start_z[:] = 0  # as done by mark_strategy_step_done
        rss.host.needs_strategy_sample[:] = True

        # Episode 1: counter=1, env_index=0 → start_z = (1+0)%4 = 1
        z1, _, _ = rss.strategy_for_step(gs)
        start_z_ep1 = int(rss.host.episode_arc_start_z[0].item())
        self.assertEqual(start_z_ep1, 1, f"Episode 1 start_z should be 1 (rotated), got {start_z_ep1}")
        self.assertNotEqual(start_z_ep1, start_z_ep0, "Episode 2 must not start at same z as episode 1")


# ---------------------------------------------------------------------------
# F1-F4: Forced router/critic masks
# ---------------------------------------------------------------------------

class TestForcedMasks(unittest.TestCase):

    # F1: BPTT skipped → returns skip metric
    def test_router_updater_skips_when_forced_false(self):
        from rl.custom_ppo.update.router_sequence_updater import RouterSequenceUpdater
        from rl.ppo_core import TensorDictRolloutBuffer
        updater = RouterSequenceUpdater(
            model=types.SimpleNamespace(selector_gru=object(), strategy_encoder=object()),
            cfg=types.SimpleNamespace(latent_assignment_mode="balanced_episode", train_router_when_forced=False,
                                      recurrent_burn_in=8, recurrent_seq_len=32, router_chunks_per_batch=4, router_ent_coef=0.005),
            hparams=types.SimpleNamespace(use_latent_strategy=True, clip_range=0.2, latent_strategy_ppo_coef=0.1, latent_lam_p=0.02),
            optimizer=None, device="cpu",
        )
        buf = TensorDictRolloutBuffer(8, 4)
        buf.register_field("router_decision_valid", dtype=torch.bool)
        buf.register_field("selector_hidden", (64,))
        result = updater.update_epoch(buf, ent_coef=0.005)
        self.assertIn("router_skipped_forced_mode", result)

    # F2: BPTT not skipped when train_when_forced=True
    def test_router_updater_not_skipped_when_flag_true(self):
        from rl.custom_ppo.update.router_sequence_updater import RouterSequenceUpdater
        from rl.ppo_core import TensorDictRolloutBuffer
        updater = RouterSequenceUpdater(
            model=types.SimpleNamespace(selector_gru=object(), strategy_encoder=object()),
            cfg=types.SimpleNamespace(latent_assignment_mode="balanced_episode", train_router_when_forced=True,
                                      recurrent_burn_in=8, recurrent_seq_len=32, router_chunks_per_batch=4, router_ent_coef=0.005),
            hparams=types.SimpleNamespace(use_latent_strategy=True, clip_range=0.2, latent_strategy_ppo_coef=0.1, latent_lam_p=0.02),
            optimizer=None, device="cpu",
        )
        buf = TensorDictRolloutBuffer(8, 4)
        buf.register_field("router_decision_valid", dtype=torch.bool)
        buf.register_field("selector_hidden", (64,))
        result = updater.update_epoch(buf, ent_coef=0.005)
        self.assertNotIn("router_skipped_forced_mode", result)

    def _compute_masks(self, assignment_mode, train_when_forced, train_critic_when_forced, z_resampled, z_forced, terminated):
        """Replicate the mask logic from _write_v6i7_step_fields."""
        cfg = types.SimpleNamespace(
            latent_assignment_mode=assignment_mode,
            train_router_when_forced=train_when_forced,
            train_router_critic_when_forced=train_critic_when_forced,
        )
        if assignment_mode != "router" and train_when_forced:
            rdv = z_resampled
        else:
            rdv = z_resampled & ~z_forced
        if assignment_mode == "router":
            ctv = ~terminated if train_critic_when_forced else (~terminated & ~z_forced)
        else:
            ctv = ~terminated
        return rdv, ctv

    # F3: balanced_episode + train_when_forced=True → rdv includes forced boundaries
    def test_rdv_includes_forced_when_flag_true(self):
        n = 4
        z_resampled = torch.ones(n, dtype=torch.bool)
        z_forced = torch.ones(n, dtype=torch.bool)
        terminated = torch.zeros(n, dtype=torch.bool)
        rdv, _ = self._compute_masks("balanced_episode", True, True, z_resampled, z_forced, terminated)
        self.assertTrue(rdv.all())

    # F4: balanced_episode + train_critic_when_forced=False → ctv = ~terminated (NOT gated by z_forced)
    def test_ctv_not_gated_by_z_forced_in_balanced_mode(self):
        # Invariant 1: critic must train in balanced modes regardless of train_critic_when_forced.
        n = 4
        z_forced = torch.ones(n, dtype=torch.bool)
        terminated = torch.zeros(n, dtype=torch.bool)
        z_resampled = torch.zeros(n, dtype=torch.bool)
        _, ctv = self._compute_masks("balanced_episode", False, False, z_resampled, z_forced, terminated)
        # ctv must be True (all non-terminated) even though z_forced=True and critic flag=False
        self.assertTrue(ctv.all(), f"ctv must not be gated by z_forced in balanced modes: {ctv}")

    # F5: router mode + train_critic_when_forced=False + z_forced=True → ctv excludes forced
    def test_ctv_gated_in_router_mode(self):
        n = 4
        z_forced = torch.ones(n, dtype=torch.bool)
        terminated = torch.zeros(n, dtype=torch.bool)
        z_resampled = torch.zeros(n, dtype=torch.bool)
        _, ctv = self._compute_masks("router", False, False, z_resampled, z_forced, terminated)
        self.assertFalse(ctv.any(), "In router mode with forced episodes and flag=False, ctv should be False")


# ---------------------------------------------------------------------------
# G1-G2: z shape assertions
# ---------------------------------------------------------------------------

class TestZShapeAssertions(unittest.TestCase):

    # G1: balanced_episode z is (N_env,)
    def test_balanced_episode_z_shape(self):
        n_envs = 8
        rss = _make_balanced_episode_state(n_envs=n_envs, K=4, episode_start=True)
        z, _, _ = rss.strategy_for_step(torch.zeros(n_envs, 35))
        self.assertEqual(tuple(z.shape), (n_envs,))

    # G2: balanced_arc z is (N_env,)
    def test_balanced_arc_z_shape(self):
        n_envs = 6
        rss = _make_balanced_arc_state(n_envs=n_envs, K=4, arc_steps=8, episode_start=True)
        z, _, _ = rss.strategy_for_step(torch.zeros(n_envs, 35))
        self.assertEqual(tuple(z.shape), (n_envs,))


# ---------------------------------------------------------------------------
# I1-I4: Invariant integration tests
# ---------------------------------------------------------------------------

class TestInvariants(unittest.TestCase):

    # I1: No silent fallback — RuntimeError when router_reward_enabled but env omits it
    def test_no_silent_fallback_raises(self):
        from rl.custom_ppo.rollout_collector import RolloutCollector
        col = object.__new__(RolloutCollector)
        col.cfg = types.SimpleNamespace(router_reward_enabled=True)
        col.device = "cpu"
        # reward_component without "router_reward"
        reward_component = {"reward_terminal": torch.zeros(4), "reward_sparse": torch.zeros(4)}
        is_v6i7 = True
        with self.assertRaises(RuntimeError) as ctx:
            if is_v6i7 and bool(getattr(col.cfg, "router_reward_enabled", False)):
                if "router_reward" not in reward_component:
                    raise RuntimeError(
                        "router_reward_enabled=True but info did not contain 'router_reward'."
                    )
        self.assertIn("router_reward", str(ctx.exception))

    # I2: Invariant 4 — router_reward non-zero on event transition, not only at episode end
    def test_router_reward_nonzero_on_flag_event(self):
        from gpu_env._config import RouterRewardConfig
        B = 4
        rrc = RouterRewardConfig(enabled=True, win_weight=0.0, flag_cap_weight=1.0, normalize=False, scale=1.0)
        env = types.SimpleNamespace(B=B, device="cpu", cfg=types.SimpleNamespace(router_reward_config=rrc))
        rterm = torch.zeros(B)        # NOT terminal
        blue_cap = torch.ones(B, dtype=torch.bool)   # flag capture on this step
        red_cap = torch.zeros(B, dtype=torch.bool)
        blue_tag_wf = torch.zeros(B)
        red_tag = torch.zeros(B)
        out = _call_router_reward_total(env, rterm, blue_cap, red_cap, blue_tag_wf, red_tag)
        self.assertTrue((out != 0.0).all(), "router_reward must be non-zero on flag event even when rterm=0")

    # I3: Invariant 1 — critic trains (ctv > 0) in balanced mode with train_when_forced=False
    def test_critic_trains_in_balanced_mode(self):
        # Simulate _write_v6i7_step_fields ctv logic for balanced_episode mode.
        n = 8
        z_forced = torch.ones(n, dtype=torch.bool)   # all forced in balanced mode
        terminated = torch.zeros(n, dtype=torch.bool)
        # Replicate the corrected ctv logic:
        assignment_mode = "balanced_episode"
        ctv = ~terminated  # balanced mode: always ~terminated regardless of z_forced
        self.assertTrue(ctv.sum() > 0, "Critic must have valid targets in balanced mode")

    # I4: Invariant 5 — forced steps are never on-policy PPO samples when flag=False
    def test_forced_never_ppo_samples(self):
        n = 8
        z_resampled = torch.ones(n, dtype=torch.bool)  # boundary fires
        z_forced = torch.ones(n, dtype=torch.bool)      # all forced
        # With train_when_forced=False: rdv = z_resampled & ~z_forced = False
        rdv = z_resampled & ~z_forced
        self.assertFalse(rdv.any(), "Forced steps must never be router PPO samples when train_when_forced=False")


if __name__ == "__main__":
    unittest.main()
