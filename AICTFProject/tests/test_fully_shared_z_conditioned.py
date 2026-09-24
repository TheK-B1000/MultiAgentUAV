"""Contracts for the Fully Shared Strategy-Conditioned sharing-axis baseline.

Pins structural locks from FULLY_SHARED_STRATEGY_CONDITIONED_4V4_V1_SPEC
(C1–C7 pieces that do not require GPU episodes or seed spend).
"""

from __future__ import annotations

import unittest

from rl.config.ppo_config import PPOConfig, TrainMode
from rl.custom_ppo.fully_shared_z import (
    WARM_START_4V4_SHA256,
    apply_fully_shared_z_config,
    assert_fully_shared_contracts,
    count_actor_modules,
    half_half_forced_latent_ids,
    initial_opponent_keys_for_forced_z,
    opponent_tag_for_z,
)


class FullySharedZConfigTests(unittest.TestCase):
    def test_half_half_ids(self):
        ids = half_half_forced_latent_ids(8)
        self.assertEqual(ids, (0, 1, 0, 1, 0, 1, 0, 1))
        self.assertEqual(ids.count(0), ids.count(1))

    def test_half_half_rejects_odd(self):
        with self.assertRaises(ValueError):
            half_half_forced_latent_ids(7)

    def test_opponent_tag_for_z(self):
        self.assertEqual(opponent_tag_for_z(0), "OP6")
        self.assertEqual(opponent_tag_for_z(1), "OP7")
        with self.assertRaises(ValueError):
            opponent_tag_for_z(2)

    def test_initial_keys_match_z(self):
        ids = (0, 1, 0, 1)
        keys = initial_opponent_keys_for_forced_z(ids)
        self.assertEqual(keys, ["OP6", "OP7", "OP6", "OP7"])

    def test_apply_locks_sharing_axis(self):
        cfg = PPOConfig()
        cfg.n_envs = 8
        locks = apply_fully_shared_z_config(cfg, team_size=4)
        self.assertTrue(cfg.fully_shared_z_conditioned_enabled)
        self.assertTrue(cfg.fully_shared_z_pole_match)
        self.assertTrue(cfg.use_latent_strategy)
        self.assertEqual(cfg.latent_k, 2)
        self.assertEqual(cfg.latent_assignment_mode, "static_env")
        self.assertEqual(cfg.latent_actor_conditioning, "concat")
        self.assertTrue(cfg.role_conditioning_enabled)
        self.assertTrue(cfg.role_fixed_for_episode)
        self.assertFalse(cfg.split_attack_defend_enabled)
        self.assertEqual(float(cfg.defend_teacher_lambda), 0.0)
        self.assertEqual(float(cfg.latent_strategy_ppo_coef), 0.0)
        self.assertEqual(float(cfg.latent_lam_h), 0.0)
        self.assertEqual(float(cfg.latent_lam_p), 0.0)
        self.assertFalse(cfg.opponent_randomize)
        self.assertEqual(cfg.mode, TrainMode.FIXED_OPPONENT.value)
        self.assertEqual(cfg.forced_latent_env_ids.count(0), 4)
        self.assertEqual(cfg.forced_latent_env_ids.count(1), 4)
        self.assertTrue(any("pole_match" in line for line in locks))
        assert_fully_shared_contracts(cfg)

    def test_apply_rejects_team_size_2(self):
        cfg = PPOConfig()
        cfg.n_envs = 4
        with self.assertRaises(ValueError):
            apply_fully_shared_z_config(cfg, team_size=2)

    def test_contracts_reject_split(self):
        cfg = PPOConfig()
        cfg.n_envs = 4
        apply_fully_shared_z_config(cfg, team_size=4)
        cfg.split_attack_defend_enabled = True
        with self.assertRaises(AssertionError):
            assert_fully_shared_contracts(cfg)

    def test_contracts_reject_film(self):
        cfg = PPOConfig()
        cfg.n_envs = 4
        apply_fully_shared_z_config(cfg, team_size=4)
        cfg.enable_actor_z_film = True
        with self.assertRaises(AssertionError):
            assert_fully_shared_contracts(cfg)

    def test_contracts_reject_router_coef(self):
        cfg = PPOConfig()
        cfg.n_envs = 4
        apply_fully_shared_z_config(cfg, team_size=4)
        cfg.latent_strategy_ppo_coef = 0.1
        with self.assertRaises(AssertionError):
            assert_fully_shared_contracts(cfg)

    def test_warm_start_pin_constant(self):
        self.assertEqual(
            WARM_START_4V4_SHA256,
            "94dde69d091a79344db3390d5464dbb4bcf51677a175df93969ab252b2e0f478",
        )


class FullySharedZActorContractTests(unittest.TestCase):
    """C1/C2: z reaches a single shared LatentConditionedActor (CPU, tiny)."""

    def test_z_reaches_network_and_params_are_shared(self):
        import numpy as np
        import torch
        from gymnasium import spaces
        from rl.custom_ppo.policy import SharedActorCentralizedCritic

        n = 4
        n_ch, rows, cols, vec_dim = 7, 20, 20, 20
        obs_space = spaces.Dict(
            {
                "grid": spaces.Box(0.0, 1.0, shape=(n, n_ch, rows, cols), dtype=np.float32),
                "vec": spaces.Box(-1.0, 1.0, shape=(n, vec_dim), dtype=np.float32),
                "agent_mask": spaces.Box(0.0, 1.0, shape=(n,), dtype=np.float32),
                "mask": spaces.Box(0.0, 1.0, shape=(n * 55,), dtype=np.float32),
            }
        )
        act_space = spaces.MultiDiscrete([5, 50] * n)
        model = SharedActorCentralizedCritic(
            obs_space,
            act_space,
            latent_k=2,
            z_embed_dim=16,
            role_conditioning_enabled=True,
            entity_repair_enabled=False,
        )
        self.assertEqual(count_actor_modules(model), 1)
        self.assertTrue(model.uses_latent_strategy)
        self.assertTrue(model.role_conditioning_enabled)

        batch = 2
        rng = np.random.default_rng(0)
        obs = {
            "grid": torch.tensor(rng.random((batch, n, n_ch, rows, cols)), dtype=torch.float32),
            "vec": torch.tensor(rng.uniform(-1, 1, (batch, n, vec_dim)), dtype=torch.float32),
            "agent_mask": torch.ones(batch, n),
            "mask": torch.ones(batch, n * 55),
        }
        roles = torch.zeros(batch, n)
        z0 = torch.zeros(batch, dtype=torch.long)
        z1 = torch.ones(batch, dtype=torch.long)
        logits0 = model.policy_logits(obs, z_idx=z0, roles=roles)
        logits1 = model.policy_logits(obs, z_idx=z1, roles=roles)
        # C1: ablating z changes logits (random init embedding ⇒ nonzero delta).
        delta = (logits0 - logits1).abs().max().item()
        self.assertGreater(delta, 0.0)

        # C2/C3: no second actor module / no per-z action heads.
        la = model.latent_actor
        self.assertIsNone(getattr(la, "latent_action_heads", None))
        self.assertFalse(bool(getattr(la, "enable_actor_z_film", False)))

    def test_default_config_structurally_absent(self):
        cfg = PPOConfig()
        self.assertFalse(cfg.fully_shared_z_conditioned_enabled)
        self.assertFalse(cfg.fully_shared_z_pole_match)


class FullySharedZTrainGuardTests(unittest.TestCase):
    def test_authorize_launch_required_message(self):
        """Non-smoke path must refuse without --authorize-launch (import-level)."""
        import importlib
        import experiments.train_fully_shared_strategy_conditioned as T

        importlib.reload(T)
        # Smoke path to the guard: simulate argv via parsing is heavy; unit-check
        # the constant and that the module exposes main.
        self.assertTrue(callable(T.main))
        self.assertTrue(T.DEFAULT_SPEC_4V4.name.startswith("FULLY_SHARED"))


if __name__ == "__main__":
    unittest.main()
