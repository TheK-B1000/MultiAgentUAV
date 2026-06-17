"""Checkpoint migration matrix for SharedActorCentralizedCritic policy loads."""

from __future__ import annotations

import unittest

import numpy as np
import torch
from gymnasium import spaces

from game_field_gpu import VEC_OBS_DIM
from rl.custom_ppo.policy import (
    SharedActorCentralizedCritic,
    _migrate_action_conditioned_critic_weights,
    _migrate_legacy_aliased_strategy_modules,
    remap_legacy_actor_state_dict_keys,
)
from rl.latent_marl import CONTEXT_STATE_DIM


def _obs_space() -> spaces.Dict:
    return spaces.Dict(
        {
            "grid": spaces.Box(low=0.0, high=1.0, shape=(2, 7, 20, 20), dtype=np.float32),
            "vec": spaces.Box(low=-1.0, high=1.0, shape=(2, VEC_OBS_DIM), dtype=np.float32),
            "agent_mask": spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32),
            "mask": spaces.Box(low=0.0, high=1.0, shape=(110,), dtype=np.float32),
        }
    )


def _action_space() -> spaces.MultiDiscrete:
    return spaces.MultiDiscrete([5, 50, 5, 50])


def _latent_model(
    *,
    recurrent: bool = False,
    aux_head: bool = False,
    episode_value: bool = True,
) -> SharedActorCentralizedCritic:
    return SharedActorCentralizedCritic(
        _obs_space(),
        _action_space(),
        latent_k=4,
        z_embed_dim=16,
        strategy_hidden_dim=64,
        critic_hidden_dim=64,
        use_recurrent_selector=recurrent,
        recurrent_selector_hidden_dim=16,
        use_strategy_aux_return_head=aux_head,
        use_episode_strategy_value_head=episode_value,
    )


class PolicyCheckpointMigrationTests(unittest.TestCase):
    def test_z_only_critic_strict_roundtrip(self) -> None:
        model = _latent_model()
        sd = model.state_dict()
        target = _latent_model()
        missing, unexpected = target.load_state_dict(sd, strict=False)
        self.assertEqual(missing, [])
        self.assertEqual(unexpected, [])
        target2 = _latent_model()
        target2.load_state_dict(sd, strict=True)

    def test_action_conditioned_critic_migrates_to_z_only(self) -> None:
        model = _latent_model()
        gs_dim = int(model.global_state_dim)
        joint_dim = int(model.joint_action_onehot_dim)
        k = int(model.latent_k)
        key = "critic.net.0.weight"
        old_in = gs_dim + joint_dim + k
        migrated_sd = dict(model.state_dict())
        migrated_sd[key] = torch.randn(int(model.critic.net[0].out_features), old_in)
        out = _migrate_action_conditioned_critic_weights(
            migrated_sd,
            prefix="",
            global_state_dim=gs_dim,
            joint_action_dim=joint_dim,
            latent_k=k,
        )
        self.assertEqual(int(out[key].shape[1]), gs_dim + k)
        target = _latent_model()
        missing, unexpected = target.load_state_dict(out, strict=False)
        self.assertEqual(unexpected, [])
        self.assertTrue(all(not m.startswith("critic.") for m in missing))

    def test_legacy_actor_keys_remap_under_nested_prefix(self) -> None:
        legacy = {
            "wrapper.actor_body.0.weight": torch.randn(256, 148),
            "wrapper.actor_head.weight": torch.randn(110, 256),
        }
        remapped = remap_legacy_actor_state_dict_keys(legacy, prefix="wrapper.")
        self.assertIn("wrapper.latent_actor.body.0.weight", remapped)
        self.assertIn("wrapper.latent_actor.action_head.weight", remapped)

    def test_legacy_aux_stripped_when_canonical_encoder_present_and_head_off(self) -> None:
        sd = {
            "strategy_encoder.net.0.weight": torch.randn(4, CONTEXT_STATE_DIM),
            "strategy_aux_return_head.net.0.weight": torch.randn(4, CONTEXT_STATE_DIM),
        }
        out = _migrate_legacy_aliased_strategy_modules(
            sd,
            has_strategy_encoder=True,
            has_strategy_aux_return_head=False,
        )
        self.assertIn("strategy_encoder.net.0.weight", out)
        self.assertNotIn("strategy_aux_return_head.net.0.weight", out)

    def test_incompatible_recurrent_aux_mirror_is_dropped(self) -> None:
        recurrent_in = CONTEXT_STATE_DIM + 16
        sd = {
            "strategy_encoder.net.0.weight": torch.randn(4, recurrent_in),
            "strategy_aux_return_head.net.0.weight": torch.randn(4, CONTEXT_STATE_DIM),
        }
        out = _migrate_legacy_aliased_strategy_modules(
            sd,
            has_strategy_encoder=True,
            has_strategy_aux_return_head=True,
        )
        self.assertIn("strategy_encoder.net.0.weight", out)
        self.assertNotIn("strategy_aux_return_head.net.0.weight", out)

    def test_recurrent_selector_strict_load(self) -> None:
        model = _latent_model(recurrent=True)
        sd = model.state_dict()
        target = _latent_model(recurrent=True)
        missing, unexpected = target.load_state_dict(sd, strict=True)
        self.assertEqual(missing, [])
        self.assertEqual(unexpected, [])

    def test_non_recurrent_into_recurrent_shape_mismatch_fails(self) -> None:
        src = _latent_model(recurrent=False)
        dst = _latent_model(recurrent=True)
        with self.assertRaises(RuntimeError):
            dst.load_state_dict(src.state_dict(), strict=False)


if __name__ == "__main__":
    unittest.main()
