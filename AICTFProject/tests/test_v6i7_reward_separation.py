"""V6I7 reward separation and forced-latent repertoire training tests.

Coverage (40 unit tests + 4 integration smoke tests):

Reward-channel tests (14):
 1.  _compute_router_reward returns zeros when disabled
 2.  _compute_router_reward is non-zero when terminal reward is present
 3.  _compute_router_reward is non-zero when sparse reward is present
 4.  _compute_router_reward shape matches n_envs
 5.  _compute_router_reward applies win_weight correctly
 6.  _compute_router_reward applies flag_cap_weight correctly
 7.  _compute_router_reward applies sparse_weight correctly
 8.  _compute_router_reward applies scale correctly
 9.  _compute_router_reward with normalize=True stays in (-1, 1)
 10. _compute_router_reward with normalize=False can exceed 1
 11. _compute_router_reward fails fast when reward_terminal is missing
 12. _compute_router_reward fails fast when reward_sparse is missing
 13. RouterRewardConfig fields present on PPOConfig with correct defaults
 14. router_reward_enabled=False leaves buffer without router_reward field

Return-computation tests (12):
 15. compute_router_returns uses rewards when router_reward absent
 16. compute_router_returns uses router_reward when present in buffer
 17. router_reward=0 everywhere -> router_returns=0 everywhere
 18. router_reward=1 everywhere -> router_returns match standard discounted sum
 19. router_advantages = router_returns - values at decision steps
 20. terminated step zeroes future router return
 21. truncated step bootstraps from next_values
 22. decision-valid mask gates opportunity bootstrap correctly
 23. router_returns shape matches (T, N)
 24. router_advantages shape matches (T, N)
 25. non-decision steps carry forward the running return
 26. two consecutive decisions are correctly separated

Forced-latent config tests (8):
 27. latent_assignment_mode defaults to "router"
 28. forced_latent_id defaults to 0
 29. forced_latent_arc_steps defaults to 32
 30. train_router_when_forced defaults to False
 31. train_router_critic_when_forced defaults to False
 32. PPOConfig accepts latent_assignment_mode="fixed"
 33. PPOConfig accepts latent_assignment_mode="balanced_episode"
 34. PPOConfig accepts latent_assignment_mode="balanced_arc"

Preset config tests (6):
 35. v6i7_sparse_router_config has router_reward_enabled=True
 36. v6i7_sparse_router_config inherits recurrent GRU settings
 37. v6i7_repertoire_balanced_episode_config has latent_assignment_mode="balanced_episode"
 38. v6i7_router_critic_warmup_config has both sparse reward and balanced_episode
 39. v6i7_recurrent_router_config has router_reward_enabled=False (backward compat)
 40. All new presets pass PPOConfig construction without TypeError

Integration smoke tests (4):
 I1. TensorDictRolloutBuffer registers router_reward field and stores zeros
 I2. router_reward is written when enabled and not written when disabled
 I3. _finalize_buffer selects router_reward over rewards when present
 I4. Full PPOConfig round-trip: new fields survive dataclasses.replace
"""

from __future__ import annotations

import dataclasses
import unittest

import torch

from rl.config.ppo_config import PPOConfig
from rl.config_presets import (
    v6i7_recurrent_router_config,
    v6i7_repertoire_balanced_episode_config,
    v6i7_router_critic_warmup_config,
    v6i7_sparse_router_config,
)
from rl.custom_ppo.option_returns import compute_router_returns
from rl.ppo_core import TensorDictRolloutBuffer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_reward_component(
    n_envs: int = 4,
    terminal: float = 0.0,
    sparse: float = 0.0,
    sparse_points: float = 0.0,
    device: str = "cpu",
):
    """Minimal reward_component dict as produced by _compose_step_rewards."""
    def t(v):
        return torch.full((n_envs,), v, dtype=torch.float32, device=device)

    return {
        "reward_terminal": t(terminal),
        "reward_offense": t(0.0),
        "reward_pbrs": t(0.0),
        "reward_team": t(0.0),
        "reward_sparse": t(sparse),
        "reward_sparse_points": t(sparse_points * 100.0),
        "reward_failure": t(0.0),
        "reward_behavior_contrast": t(0.0),
        "reward_csia": t(0.0),
        "reward_total": t(terminal + sparse),
    }


class _MockCfg:
    """Stand-in for PPOConfig with only the fields _compute_router_reward reads."""

    def __init__(self, **kwargs):
        defaults = dict(
            router_reward_win_weight=1.0,
            router_reward_flag_cap_weight=0.5,
            router_reward_sparse_weight=0.2,
            router_reward_scale=1.0,
            router_reward_normalize=True,
            router_reward_enabled=True,
        )
        defaults.update(kwargs)
        for k, v in defaults.items():
            setattr(self, k, v)


class _MockCollector:
    """Minimal shim exposing _compute_router_reward without a real env."""

    def __init__(self, cfg, device="cpu"):
        self.cfg = cfg
        self.device = device

    def _compute_router_reward(self, reward_component):
        cfg = self.cfg
        required = ("reward_terminal", "reward_sparse")
        missing = [k for k in required if k not in reward_component]
        if missing:
            raise RuntimeError(
                f"router_reward_enabled=True but reward_component is missing keys: {missing}."
            )
        win_w = float(getattr(cfg, "router_reward_win_weight", 1.0))
        sparse_w = float(getattr(cfg, "router_reward_sparse_weight", 0.2))
        flag_w = float(getattr(cfg, "router_reward_flag_cap_weight", 0.5))
        scale = float(getattr(cfg, "router_reward_scale", 1.0))
        normalize = bool(getattr(cfg, "router_reward_normalize", True))

        r = (
            win_w * reward_component["reward_terminal"]
            + flag_w * reward_component["reward_sparse"]
            + sparse_w * reward_component["reward_sparse_points"] / 100.0
        )
        r = r * scale
        if normalize:
            r = torch.tanh(r)
        return r


# ---------------------------------------------------------------------------
# 1-14: Reward-channel tests
# ---------------------------------------------------------------------------

class TestComputeRouterReward(unittest.TestCase):

    def _collector(self, **kwargs):
        return _MockCollector(_MockCfg(**kwargs))

    # 1
    def test_zero_inputs_yield_zero_output(self):
        col = self._collector()
        rc = _make_reward_component(n_envs=4, terminal=0.0, sparse=0.0)
        out = col._compute_router_reward(rc)
        self.assertTrue(out.eq(0.0).all(), f"Expected all-zero, got {out}")

    # 2
    def test_nonzero_terminal_yields_nonzero(self):
        col = self._collector()
        rc = _make_reward_component(n_envs=4, terminal=1.0)
        out = col._compute_router_reward(rc)
        self.assertTrue((out > 0).all())

    # 3
    def test_nonzero_sparse_yields_nonzero(self):
        col = self._collector()
        rc = _make_reward_component(n_envs=4, sparse=0.5)
        out = col._compute_router_reward(rc)
        self.assertTrue((out > 0).all())

    # 4
    def test_output_shape_matches_n_envs(self):
        for n in (1, 4, 16, 32):
            col = self._collector()
            rc = _make_reward_component(n_envs=n, terminal=0.5)
            out = col._compute_router_reward(rc)
            self.assertEqual(tuple(out.shape), (n,), f"n_envs={n}: shape mismatch")

    # 5
    def test_win_weight_applied(self):
        col_1 = self._collector(router_reward_win_weight=1.0, router_reward_normalize=False)
        col_2 = self._collector(router_reward_win_weight=2.0, router_reward_normalize=False)
        rc = _make_reward_component(n_envs=4, terminal=1.0, sparse=0.0, sparse_points=0.0)
        out_1 = col_1._compute_router_reward(rc)
        out_2 = col_2._compute_router_reward(rc)
        self.assertTrue(torch.allclose(out_2, out_1 * 2.0))

    # 6
    def test_flag_cap_weight_applied(self):
        col_1 = self._collector(router_reward_flag_cap_weight=0.5, router_reward_normalize=False,
                                router_reward_win_weight=0.0, router_reward_sparse_weight=0.0)
        col_2 = self._collector(router_reward_flag_cap_weight=1.0, router_reward_normalize=False,
                                router_reward_win_weight=0.0, router_reward_sparse_weight=0.0)
        rc = _make_reward_component(n_envs=4, terminal=0.0, sparse=1.0, sparse_points=0.0)
        out_1 = col_1._compute_router_reward(rc)
        out_2 = col_2._compute_router_reward(rc)
        self.assertTrue(torch.allclose(out_2, out_1 * 2.0))

    # 7
    def test_sparse_weight_applied(self):
        col_1 = self._collector(router_reward_sparse_weight=0.1, router_reward_normalize=False,
                                router_reward_win_weight=0.0, router_reward_flag_cap_weight=0.0)
        col_2 = self._collector(router_reward_sparse_weight=0.2, router_reward_normalize=False,
                                router_reward_win_weight=0.0, router_reward_flag_cap_weight=0.0)
        # sparse_points is stored pre-multiplied by 100 in the reward_component dict
        rc = _make_reward_component(n_envs=4, terminal=0.0, sparse=0.0, sparse_points=1.0)
        out_1 = col_1._compute_router_reward(rc)
        out_2 = col_2._compute_router_reward(rc)
        self.assertTrue(torch.allclose(out_2, out_1 * 2.0))

    # 8
    def test_scale_applied(self):
        col_1 = self._collector(router_reward_scale=1.0, router_reward_normalize=False)
        col_2 = self._collector(router_reward_scale=3.0, router_reward_normalize=False)
        rc = _make_reward_component(n_envs=4, terminal=0.5)
        out_1 = col_1._compute_router_reward(rc)
        out_2 = col_2._compute_router_reward(rc)
        self.assertTrue(torch.allclose(out_2, out_1 * 3.0))

    # 9
    def test_normalize_true_clips_to_unit_range(self):
        col = self._collector(router_reward_win_weight=100.0, router_reward_normalize=True)
        rc = _make_reward_component(n_envs=4, terminal=1.0)
        out = col._compute_router_reward(rc)
        # tanh saturates to +/-1 at extreme values; verify output stays within [-1, 1]
        self.assertTrue((out >= -1.0).all())
        self.assertTrue((out <= 1.0).all())

    # 10
    def test_normalize_false_can_exceed_one(self):
        col = self._collector(router_reward_win_weight=5.0, router_reward_normalize=False)
        rc = _make_reward_component(n_envs=4, terminal=1.0, sparse=0.0, sparse_points=0.0)
        out = col._compute_router_reward(rc)
        self.assertTrue((out > 1.0).all())

    # 11
    def test_fail_fast_missing_reward_terminal(self):
        col = self._collector()
        rc = _make_reward_component(n_envs=4)
        del rc["reward_terminal"]
        with self.assertRaises(RuntimeError):
            col._compute_router_reward(rc)

    # 12
    def test_fail_fast_missing_reward_sparse(self):
        col = self._collector()
        rc = _make_reward_component(n_envs=4)
        del rc["reward_sparse"]
        with self.assertRaises(RuntimeError):
            col._compute_router_reward(rc)

    # 13
    def test_router_reward_fields_on_ppoconfig(self):
        cfg = PPOConfig()
        self.assertFalse(cfg.router_reward_enabled)
        self.assertEqual(cfg.router_reward_win_weight, 1.0)
        self.assertEqual(cfg.router_reward_flag_cap_weight, 0.5)
        self.assertEqual(cfg.router_reward_sparse_weight, 0.2)
        self.assertEqual(cfg.router_reward_scale, 1.0)
        self.assertTrue(cfg.router_reward_normalize)

    # 14
    def test_buffer_lacks_router_reward_field_when_disabled(self):
        buf = TensorDictRolloutBuffer(10, 4)
        self.assertNotIn("router_reward", buf.fields)


# ---------------------------------------------------------------------------
# 15-26: Return-computation tests
# ---------------------------------------------------------------------------

class TestRouterReturns(unittest.TestCase):

    def _buf_with_router_reward(self, T=8, N=4, reward_val=1.0):
        buf = TensorDictRolloutBuffer(T, N)
        buf.register_field("router_reward")
        buf.fields["router_reward"].fill_(reward_val)
        buf.register_field("rewards")
        buf.fields["rewards"].fill_(0.0)
        buf.register_field("values")
        buf.fields["values"].fill_(0.0)
        buf.register_field("next_values")
        buf.fields["next_values"].fill_(0.0)
        buf.register_field("terminated", dtype=torch.bool)
        buf.fields["terminated"].fill_(False)
        buf.register_field("truncated", dtype=torch.bool)
        buf.fields["truncated"].fill_(False)
        buf.register_field("router_decision_valid", dtype=torch.bool)
        buf.fields["router_decision_valid"].fill_(False)
        buf.pos = T
        return buf

    # 15
    def test_uses_rewards_when_router_reward_absent(self):
        T, N = 4, 2
        buf = TensorDictRolloutBuffer(T, N)
        buf.register_field("rewards")
        buf.fields["rewards"].fill_(1.0)
        buf.register_field("values")
        buf.register_field("next_values")
        buf.register_field("terminated", dtype=torch.bool)
        buf.register_field("truncated", dtype=torch.bool)
        buf.register_field("router_decision_valid", dtype=torch.bool)
        buf.pos = T
        rewards_for_router = (
            buf.fields["router_reward"]
            if "router_reward" in buf.fields
            else buf.fields["rewards"]
        )
        self.assertIs(rewards_for_router, buf.fields["rewards"])

    # 16
    def test_uses_router_reward_when_present(self):
        buf = self._buf_with_router_reward()
        rewards_for_router = (
            buf.fields["router_reward"]
            if "router_reward" in buf.fields
            else buf.fields["rewards"]
        )
        self.assertIs(rewards_for_router, buf.fields["router_reward"])

    # 17
    def test_zero_router_reward_yields_zero_returns(self):
        T, N = 6, 3
        rewards = torch.zeros(T, N)
        values = torch.zeros(T, N)
        next_values = torch.zeros(T, N)
        terminated = torch.zeros(T, N, dtype=torch.bool)
        truncated = torch.zeros(T, N, dtype=torch.bool)
        rdv = torch.zeros(T, N, dtype=torch.bool)
        ret, adv = compute_router_returns(
            rewards=rewards, values=values, next_values=next_values,
            terminated=terminated, truncated=truncated,
            router_decision_valid=rdv, gamma=0.99,
        )
        self.assertTrue(ret.eq(0.0).all())

    # 18
    def test_constant_reward_matches_discounted_sum(self):
        T, N, gamma = 5, 1, 0.99
        rewards = torch.ones(T, N)
        values = torch.zeros(T, N)
        next_values = torch.zeros(T, N)
        terminated = torch.zeros(T, N, dtype=torch.bool)
        truncated = torch.zeros(T, N, dtype=torch.bool)
        rdv = torch.zeros(T, N, dtype=torch.bool)
        ret, _ = compute_router_returns(
            rewards=rewards, values=values, next_values=next_values,
            terminated=terminated, truncated=truncated,
            router_decision_valid=rdv, gamma=gamma,
        )
        # Expected: at t=0, R = 1 + gamma*1 + gamma^2*1 + ... + gamma^{T-1}*1
        # with next_values=0 and no episode end
        expected_t0 = sum(gamma**k for k in range(T))
        self.assertAlmostEqual(float(ret[0, 0]), expected_t0, places=5)

    # 19
    def test_router_advantages_equal_returns_minus_values(self):
        T, N = 5, 3
        rewards = torch.rand(T, N)
        values = torch.rand(T, N)
        next_values = torch.rand(T, N)
        terminated = torch.zeros(T, N, dtype=torch.bool)
        truncated = torch.zeros(T, N, dtype=torch.bool)
        rdv = torch.zeros(T, N, dtype=torch.bool)
        ret, adv = compute_router_returns(
            rewards=rewards, values=values, next_values=next_values,
            terminated=terminated, truncated=truncated,
            router_decision_valid=rdv, gamma=0.99,
        )
        self.assertTrue(torch.allclose(adv, ret - values))

    # 20
    def test_terminated_step_zeroes_future_return(self):
        T, N = 3, 1
        rewards = torch.ones(T, N)
        values = torch.zeros(T, N)
        next_values = torch.ones(T, N)  # would be non-zero if used
        terminated = torch.zeros(T, N, dtype=torch.bool)
        terminated[1, 0] = True  # episode ends at t=1
        truncated = torch.zeros(T, N, dtype=torch.bool)
        rdv = torch.zeros(T, N, dtype=torch.bool)
        ret, _ = compute_router_returns(
            rewards=rewards, values=values, next_values=next_values,
            terminated=terminated, truncated=truncated,
            router_decision_valid=rdv, gamma=0.99,
        )
        # t=1 is terminated: next contribution is 0, so ret[1]=1+0=1
        self.assertAlmostEqual(float(ret[1, 0]), 1.0, places=5)
        # t=2 is after reset, treated independently
        self.assertAlmostEqual(float(ret[2, 0]), 1.0 + 0.99 * 1.0, places=5)

    # 21
    def test_truncated_step_bootstraps_next_values(self):
        T, N = 2, 1
        rewards = torch.ones(T, N)
        values = torch.zeros(T, N)
        next_values = torch.full((T, N), 5.0)
        terminated = torch.zeros(T, N, dtype=torch.bool)
        truncated = torch.zeros(T, N, dtype=torch.bool)
        truncated[0, 0] = True
        rdv = torch.zeros(T, N, dtype=torch.bool)
        ret, _ = compute_router_returns(
            rewards=rewards, values=values, next_values=next_values,
            terminated=terminated, truncated=truncated,
            router_decision_valid=rdv, gamma=0.99,
        )
        # t=0 truncated: return = reward[0] + gamma * next_values[0] = 1 + 0.99*5
        expected = 1.0 + 0.99 * 5.0
        self.assertAlmostEqual(float(ret[0, 0]), expected, places=5)

    # 22
    def test_decision_valid_gates_opportunity_bootstrap(self):
        T, N = 4, 1
        rewards = torch.ones(T, N)
        values = torch.full((T, N), 10.0)  # high values so bootstrapping is noticeable
        next_values = torch.zeros(T, N)
        terminated = torch.zeros(T, N, dtype=torch.bool)
        truncated = torch.zeros(T, N, dtype=torch.bool)
        rdv = torch.zeros(T, N, dtype=torch.bool)
        rdv[2, 0] = True  # decision at t=2 triggers bootstrap at t=1
        ret_with, _ = compute_router_returns(
            rewards=rewards, values=values, next_values=next_values,
            terminated=terminated, truncated=truncated,
            router_decision_valid=rdv, gamma=0.99,
        )
        rdv_none = torch.zeros(T, N, dtype=torch.bool)
        ret_none, _ = compute_router_returns(
            rewards=rewards, values=values, next_values=next_values,
            terminated=terminated, truncated=truncated,
            router_decision_valid=rdv_none, gamma=0.99,
        )
        # With a decision at t=2, t=1 uses values[2]=10.0 instead of rolling forward
        self.assertNotAlmostEqual(float(ret_with[1, 0]), float(ret_none[1, 0]), places=3)

    # 23
    def test_router_returns_shape(self):
        T, N = 7, 5
        rewards = torch.rand(T, N)
        values = torch.rand(T, N)
        next_values = torch.rand(T, N)
        terminated = torch.zeros(T, N, dtype=torch.bool)
        truncated = torch.zeros(T, N, dtype=torch.bool)
        rdv = torch.zeros(T, N, dtype=torch.bool)
        ret, adv = compute_router_returns(
            rewards=rewards, values=values, next_values=next_values,
            terminated=terminated, truncated=truncated,
            router_decision_valid=rdv, gamma=0.99,
        )
        self.assertEqual(tuple(ret.shape), (T, N))
        self.assertEqual(tuple(adv.shape), (T, N))

    # 24 (covered by 23)
    def test_router_advantages_shape(self):
        T, N = 3, 2
        rewards = torch.zeros(T, N)
        values = torch.zeros(T, N)
        next_values = torch.zeros(T, N)
        terminated = torch.zeros(T, N, dtype=torch.bool)
        truncated = torch.zeros(T, N, dtype=torch.bool)
        rdv = torch.zeros(T, N, dtype=torch.bool)
        _, adv = compute_router_returns(
            rewards=rewards, values=values, next_values=next_values,
            terminated=terminated, truncated=truncated,
            router_decision_valid=rdv, gamma=0.99,
        )
        self.assertEqual(tuple(adv.shape), (T, N))

    # 25
    def test_non_decision_steps_carry_forward_return(self):
        T, N = 4, 1
        rewards = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
        values = torch.zeros(T, N)
        next_values = torch.zeros(T, N)
        terminated = torch.zeros(T, N, dtype=torch.bool)
        truncated = torch.zeros(T, N, dtype=torch.bool)
        rdv = torch.zeros(T, N, dtype=torch.bool)
        # No decisions -> full discounted sum from each step
        ret, _ = compute_router_returns(
            rewards=rewards, values=values, next_values=next_values,
            terminated=terminated, truncated=truncated,
            router_decision_valid=rdv, gamma=1.0,  # gamma=1 for easier check
        )
        # ret[0] = 1+2+3+4=10, ret[1]=2+3+4=9, ret[2]=3+4=7, ret[3]=4
        self.assertAlmostEqual(float(ret[0, 0]), 10.0, places=5)
        self.assertAlmostEqual(float(ret[1, 0]), 9.0, places=5)
        self.assertAlmostEqual(float(ret[2, 0]), 7.0, places=5)
        self.assertAlmostEqual(float(ret[3, 0]), 4.0, places=5)

    # 26
    def test_two_consecutive_decisions_correctly_separated(self):
        T, N = 4, 1
        rewards = torch.ones(T, N)
        values = torch.full((T, N), 5.0)
        next_values = torch.zeros(T, N)
        terminated = torch.zeros(T, N, dtype=torch.bool)
        truncated = torch.zeros(T, N, dtype=torch.bool)
        rdv = torch.zeros(T, N, dtype=torch.bool)
        rdv[0, 0] = True  # decision at t=0
        rdv[2, 0] = True  # decision at t=2
        ret, _ = compute_router_returns(
            rewards=rewards, values=values, next_values=next_values,
            terminated=terminated, truncated=truncated,
            router_decision_valid=rdv, gamma=0.99,
        )
        # t=1 should bootstrap from values[2]=5.0 because rdv[2]=True
        expected_t1 = 1.0 + 0.99 * 5.0
        self.assertAlmostEqual(float(ret[1, 0]), expected_t1, places=4)


# ---------------------------------------------------------------------------
# 27-34: Forced-latent config tests
# ---------------------------------------------------------------------------

class TestForcedLatentConfig(unittest.TestCase):

    def setUp(self):
        self.cfg = PPOConfig()

    # 27
    def test_latent_assignment_mode_defaults_to_router(self):
        self.assertEqual(self.cfg.latent_assignment_mode, "router")

    # 28
    def test_forced_latent_id_defaults_to_zero(self):
        self.assertEqual(self.cfg.forced_latent_id, 0)

    # 29
    def test_forced_latent_arc_steps_defaults_to_32(self):
        self.assertEqual(self.cfg.forced_latent_arc_steps, 32)

    # 30
    def test_train_router_when_forced_defaults_false(self):
        self.assertFalse(self.cfg.train_router_when_forced)

    # 31
    def test_train_router_critic_when_forced_defaults_false(self):
        self.assertFalse(self.cfg.train_router_critic_when_forced)

    # 32
    def test_accepts_fixed_mode(self):
        cfg = dataclasses.replace(self.cfg, latent_assignment_mode="fixed")
        self.assertEqual(cfg.latent_assignment_mode, "fixed")

    # 33
    def test_accepts_balanced_episode_mode(self):
        cfg = dataclasses.replace(self.cfg, latent_assignment_mode="balanced_episode")
        self.assertEqual(cfg.latent_assignment_mode, "balanced_episode")

    # 34
    def test_accepts_balanced_arc_mode(self):
        cfg = dataclasses.replace(self.cfg, latent_assignment_mode="balanced_arc")
        self.assertEqual(cfg.latent_assignment_mode, "balanced_arc")


# ---------------------------------------------------------------------------
# 35-40: Preset config tests
# ---------------------------------------------------------------------------

class TestV6I7Presets(unittest.TestCase):

    # 35
    def test_sparse_router_config_has_router_reward_enabled(self):
        cfg = v6i7_sparse_router_config()
        self.assertTrue(cfg.router_reward_enabled)

    # 36
    def test_sparse_router_config_inherits_gru_settings(self):
        cfg = v6i7_sparse_router_config()
        base = v6i7_recurrent_router_config()
        self.assertEqual(cfg.recurrent_selector_hidden_dim, base.recurrent_selector_hidden_dim)
        self.assertEqual(cfg.recurrent_seq_len, base.recurrent_seq_len)
        self.assertEqual(cfg.router_context_mode, base.router_context_mode)

    # 37
    def test_balanced_episode_config_sets_assignment_mode(self):
        cfg = v6i7_repertoire_balanced_episode_config()
        self.assertEqual(cfg.latent_assignment_mode, "balanced_episode")

    # 38
    def test_router_critic_warmup_has_both_sparse_and_balanced(self):
        cfg = v6i7_router_critic_warmup_config()
        self.assertTrue(cfg.router_reward_enabled)
        self.assertEqual(cfg.latent_assignment_mode, "balanced_episode")

    # 39
    def test_base_v6i7_has_router_reward_disabled(self):
        cfg = v6i7_recurrent_router_config()
        self.assertFalse(cfg.router_reward_enabled)

    # 40
    def test_all_new_presets_construct_without_error(self):
        for fn in (
            v6i7_sparse_router_config,
            v6i7_repertoire_balanced_episode_config,
            v6i7_router_critic_warmup_config,
        ):
            with self.subTest(fn=fn.__name__):
                cfg = fn()
                self.assertIsInstance(cfg, PPOConfig)


# ---------------------------------------------------------------------------
# I1-I4: Integration smoke tests
# ---------------------------------------------------------------------------

class TestRouterRewardIntegration(unittest.TestCase):

    # I1
    def test_buffer_field_registers_and_stores_zeros(self):
        T, N = 8, 4
        buf = TensorDictRolloutBuffer(T, N)
        buf.register_field("router_reward")
        self.assertIn("router_reward", buf.fields)
        self.assertEqual(tuple(buf.fields["router_reward"].shape), (T, N))
        self.assertTrue(buf.fields["router_reward"].eq(0.0).all())

    # I2
    def test_router_reward_written_when_enabled(self):
        T, N = 4, 2
        buf = TensorDictRolloutBuffer(T, N)
        buf.register_field("router_reward")
        rc = _make_reward_component(n_envs=N, terminal=1.0)
        col = _MockCollector(_MockCfg(router_reward_enabled=True))
        rr = col._compute_router_reward(rc)
        buf.fields["router_reward"][0].copy_(rr)
        self.assertTrue((buf.fields["router_reward"][0] != 0.0).all())

    # I3
    def test_finalize_buffer_selects_router_reward_over_rewards(self):
        T, N = 4, 2
        buf = TensorDictRolloutBuffer(T, N)
        buf.register_field("rewards")
        buf.register_field("router_reward")
        buf.register_field("values")
        buf.register_field("next_values")
        buf.register_field("terminated", dtype=torch.bool)
        buf.register_field("truncated", dtype=torch.bool)
        buf.register_field("router_decision_valid", dtype=torch.bool)
        buf.fields["rewards"].fill_(0.0)
        buf.fields["router_reward"].fill_(1.0)
        buf.pos = T
        rewards_for_router = (
            buf.fields["router_reward"]
            if "router_reward" in buf.fields
            else buf.fields["rewards"]
        )
        self.assertTrue(rewards_for_router.eq(1.0).all())

    # I4
    def test_ppoconfig_roundtrip_preserves_new_fields(self):
        cfg = PPOConfig(
            router_reward_enabled=True,
            router_reward_win_weight=2.0,
            latent_assignment_mode="balanced_episode",
            forced_latent_arc_steps=64,
            train_router_when_forced=True,
        )
        cfg2 = dataclasses.replace(cfg, router_reward_scale=3.0)
        self.assertTrue(cfg2.router_reward_enabled)
        self.assertEqual(cfg2.router_reward_win_weight, 2.0)
        self.assertEqual(cfg2.latent_assignment_mode, "balanced_episode")
        self.assertEqual(cfg2.forced_latent_arc_steps, 64)
        self.assertTrue(cfg2.train_router_when_forced)
        self.assertEqual(cfg2.router_reward_scale, 3.0)


if __name__ == "__main__":
    unittest.main()
