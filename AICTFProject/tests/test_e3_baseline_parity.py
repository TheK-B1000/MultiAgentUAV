"""E3 baseline parity: identical PPO setup with only ``use_latent_strategy`` toggled (fair ablation)."""

from __future__ import annotations

import unittest
from dataclasses import asdict

import numpy as np
import torch
from gymnasium import spaces

from game_field_gpu import VEC_OBS_DIM
from rl.config_presets import paper_default_latent_config, paper_default_no_latent_config
from rl.custom_ppo import SharedActorCentralizedCritic


def _spaces_2v2():
    obs_space = spaces.Dict(
        {
            "grid": spaces.Box(low=0.0, high=1.0, shape=(2, 7, 20, 20), dtype=np.float32),
            "vec": spaces.Box(low=-1.0, high=1.0, shape=(2, VEC_OBS_DIM), dtype=np.float32),
            "agent_mask": spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32),
            "mask": spaces.Box(low=0.0, high=1.0, shape=(110,), dtype=np.float32),
        }
    )
    action_space = spaces.MultiDiscrete([5, 50, 5, 50])
    return obs_space, action_space


class E3BaselineParityTests(unittest.TestCase):
    def test_paper_default_configs_differ_only_by_latent_flag(self) -> None:
        latent = paper_default_latent_config()
        plain = paper_default_no_latent_config()
        self.assertTrue(latent.use_latent_strategy)
        self.assertFalse(plain.use_latent_strategy)
        a, b = asdict(latent), asdict(plain)
        self.assertEqual(
            {k for k in a if a[k] != b[k]},
            {"use_latent_strategy"},
            "E3 requires identical PPO/hyperparam/env *defaults* with only use_latent_strategy flipped",
        )

    def test_actor_and_critic_widths_only_differ_by_z_plumbing(self) -> None:
        obs, act = _spaces_2v2()
        k = 4
        d_z = 16
        m_l = SharedActorCentralizedCritic(
            obs, act, latent_k=k, z_embed_dim=d_z, strategy_hidden_dim=128, critic_hidden_dim=128
        )
        m0 = SharedActorCentralizedCritic(
            obs, act, latent_k=0, z_embed_dim=16, strategy_hidden_dim=128, critic_hidden_dim=128
        )
        self.assertTrue(m_l.uses_latent_strategy)
        self.assertFalse(m0.uses_latent_strategy)

        self.assertEqual(
            m_l._decentralized_actor_in_dim - m0._decentralized_actor_in_dim,
            d_z,
            "Actor MLP input differs only by the strategy embedding width d_z",
        )
        self.assertEqual(m0.critic.extra_dim, 0, "No-latent critic: global state only (no joint/z extra)")
        self.assertEqual(
            m_l.critic.extra_dim,
            k,
            "Latent critic extra = z one-hot (K) only; PPO baseline is V(s, z)",
        )

    def test_same_optimizer_hyperparam_surface(self) -> None:
        """Configs share all PPO fields; only Boolean latent flag changes (pair with test_paper_default_*)."""
        a, b = asdict(paper_default_latent_config()), asdict(paper_default_no_latent_config())
        for k, va in a.items():
            if k == "use_latent_strategy":
                continue
            self.assertEqual(
                va,
                b[k],
                f"Field {k!r} must match for E3 baseline parity (got {va!r} vs {b[k]!r})",
            )
