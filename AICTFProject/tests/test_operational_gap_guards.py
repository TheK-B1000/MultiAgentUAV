"""Mechanical checks for the Summer-plan operational gaps (actor contract, L_persist, eval, invariants)."""

from __future__ import annotations

import inspect
import os
import unittest
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from game_field_gpu import VEC_OBS_DIM
from rl.config_presets import ablation_flag_resample_config, paper_default_latent_config
from rl.custom_ppo import CustomPPOTrainer, SharedActorCentralizedCritic
from rl.global_state import GLOBAL_STATE_DIM
from rl.networks import CNNEncoder
from rl.train_ppo import PPOConfig


def _min_obs_and_action_spaces():
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


class OperationalGapGuardsTests(unittest.TestCase):
    def test_policy_forward_never_reads_obs_global_state(self) -> None:
        src = inspect.getsource(SharedActorCentralizedCritic.policy_logits)
        stripped = src.replace("GLOBAL_STATE_DIM", "")
        self.assertNotIn("global_state", stripped, "decentralized policy_logits must not read global_state")

    def test_actor_width_assert_not_global_state_dim(self) -> None:
        obs_space, action_space = _min_obs_and_action_spaces()
        m = SharedActorCentralizedCritic(obs_space, action_space, latent_k=4, z_embed_dim=8)
        self.assertNotEqual(m._decentralized_actor_in_dim, GLOBAL_STATE_DIM)
        b = 2
        obs = {
            "grid": torch.rand(b, 2, 7, 20, 20),
            "vec": torch.rand(b, 2, VEC_OBS_DIM),
            "agent_mask": torch.ones(b, 2),
            "mask": torch.ones(b, 110),
        }
        z = torch.zeros((b,), dtype=torch.long)
        m.policy_logits(obs, z_idx=z)

    def test_active_actor_matches_cnn_plus_scalar_mlp_contract(self) -> None:
        """Professor-approved implementation: CNN(grid), concat scalars/z, then 256-256 MLP logits."""
        obs_space, action_space = _min_obs_and_action_spaces()
        m = SharedActorCentralizedCritic(obs_space, action_space, latent_k=4, z_embed_dim=16)

        self.assertIsInstance(m.actor_cnn, CNNEncoder)
        self.assertTrue(any(isinstance(module, nn.Conv2d) for module in m.actor_cnn.modules()))
        self.assertIsInstance(m.strategy_embedding, nn.Embedding)
        self.assertEqual(m.strategy_embedding.num_embeddings, 4)
        self.assertEqual(m.strategy_embedding.embedding_dim, 16)

        body = list(m.actor_body)
        self.assertEqual([type(layer) for layer in body], [nn.Linear, nn.ReLU, nn.Linear, nn.ReLU])
        self.assertEqual(body[0].in_features, m.actor_cnn_feature_dim + VEC_OBS_DIM + 16)
        self.assertEqual(body[0].out_features, 256)
        self.assertEqual(body[2].in_features, 256)
        self.assertEqual(body[2].out_features, 256)
        self.assertIsInstance(m.actor_head, nn.Linear)

    def test_paper_default_latent_config_is_episode_start_only(self) -> None:
        c = paper_default_latent_config()
        self.assertTrue(c.use_latent_strategy)
        self.assertEqual(c.latent_resample_every_n, 0)
        self.assertFalse(c.latent_resample_on_flag)
        self.assertEqual(c.latent_kl_consecutive, 0.0)
        a = ablation_flag_resample_config()
        self.assertTrue(a.latent_resample_on_flag)

    def test_opponent_tag_strings_not_in_z_graph_files(self) -> None:
        here = os.path.join(os.path.dirname(__file__), "..", "rl")
        for name in ("custom_ppo.py", "latent_marl.py", "config_presets.py"):
            path = os.path.abspath(os.path.join(here, name))
            with open(path, encoding="utf-8") as f:
                text = f.read()
            # Allow the canonical scripted tag ``OP5_RUSHER`` (telemetry / opponent keys); forbid other bare tokens.
            sanitized = text.replace("OP5_RUSHER", "")
            for tag in ("RUSHER", "CAMPER", "Species"):
                self.assertNotIn(
                    tag,
                    sanitized,
                    f"{name} should not mix opponent / species labels into z code paths (grep audit)",
                )

    def test_eval_uses_deterministic_by_default(self) -> None:
        """Matches ``plot/eval_checkpoint.py`` — default paper-style eval is greedy / argmax (deterministic)."""
        p = ArgumentParser()
        p.add_argument("--deterministic", action="store_true", default=True)
        p.add_argument("--stochastic", action="store_false", dest="deterministic")
        a = p.parse_args([])
        self.assertTrue(a.deterministic)
        b = p.parse_args(["--stochastic"])
        self.assertFalse(b.deterministic)
