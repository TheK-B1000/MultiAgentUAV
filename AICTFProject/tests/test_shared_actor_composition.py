"""Composition + state-dict compatibility tests for ``SharedActorCentralizedCritic``.

After Step 4 of the architecture refactor, ``SharedActorCentralizedCritic``
delegates its decentralized actor body, output head, and strategy embedding to
a single composed :class:`rl.latent_marl.LatentConditionedActor` submodule
exposed as ``model.latent_actor``. These tests pin three contracts:

* **Property shims** point into ``latent_actor`` so external code written
  against the legacy attribute names (``model.actor_body``,
  ``model.actor_head``, ``model.strategy_embedding``) keeps working.
* **Output parity**: ``model.policy_logits`` produces bit-identical logits to
  a manual reference path built out of those legacy aliases plus the CNN
  encoder.
* **Legacy state-dict compatibility**: a state dict whose keys still use the
  pre-composition layout loads into the new model without error and yields
  identical forward output. Both the latent and no-latent variants are
  covered.
"""
from __future__ import annotations

import unittest
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces

from game_field_gpu import VEC_OBS_DIM
from rl.custom_ppo.policy import (
    SharedActorCentralizedCritic,
    remap_legacy_actor_state_dict_keys,
)
from rl.latent_marl import CONTEXT_STATE_DIM, LatentConditionedActor


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


def _fixed_obs(batch: int = 3) -> Dict[str, torch.Tensor]:
    grid = torch.linspace(
        0.0, 1.0, steps=batch * 2 * 7 * 20 * 20, dtype=torch.float32
    ).reshape(batch, 2, 7, 20, 20)
    vec = torch.linspace(
        -0.75, 0.75, steps=batch * 2 * VEC_OBS_DIM, dtype=torch.float32
    ).reshape(batch, 2, VEC_OBS_DIM)
    return {
        "grid": grid,
        "vec": vec,
        "agent_mask": torch.ones((batch, 2), dtype=torch.float32),
        "mask": torch.ones((batch, 110), dtype=torch.float32),
    }


def _build_latent_model(seed: int = 1234) -> SharedActorCentralizedCritic:
    torch.manual_seed(seed)
    model = SharedActorCentralizedCritic(
        _obs_space(),
        _action_space(),
        actor_cnn_feature_dim=128,
        latent_k=4,
        z_embed_dim=16,
        strategy_hidden_dim=128,
        critic_hidden_dim=128,
    )
    model.eval()
    return model


def _build_no_latent_model(seed: int = 1234) -> SharedActorCentralizedCritic:
    torch.manual_seed(seed)
    model = SharedActorCentralizedCritic(
        _obs_space(),
        _action_space(),
        actor_cnn_feature_dim=128,
        latent_k=0,
        critic_hidden_dim=128,
    )
    model.eval()
    return model


def _legacy_rename(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Rewrite NEW composed paths back to LEGACY pre-composition paths.

    Inverse of :func:`remap_legacy_actor_state_dict_keys`; used to simulate an
    on-disk checkpoint saved by the pre-composition trainer.
    """
    renames = {
        "latent_actor.body.": "actor_body.",
        "latent_actor.action_head.": "actor_head.",
        "latent_actor.strategy_embedding.": "strategy_embedding.",
    }
    out: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = key
        for new, old in renames.items():
            if key.startswith(new):
                new_key = old + key[len(new):]
                break
        out[new_key] = value
    return out


def _reference_policy_logits(
    model: SharedActorCentralizedCritic,
    obs: Dict[str, torch.Tensor],
    z_idx: torch.Tensor | None,
) -> torch.Tensor:
    """Compute policy logits via the legacy attribute-access path.

    Uses ``model.actor_cnn``, ``model.actor_body``, ``model.actor_head``, and
    ``model.strategy_embedding`` (the latter via property shim). Asserting
    this matches ``model.policy_logits`` proves the composed implementation
    produces the same outputs as the legacy hand-written path.
    """
    grid = obs["grid"].float()
    vec = obs["vec"].float()
    batch = int(grid.shape[0])
    cnn_features = model.actor_cnn(grid.reshape(batch * model.n_agents, *model.grid_shape))
    cnn_features = cnn_features.reshape(batch, model.n_agents, model.actor_cnn_feature_dim)
    vloc = vec.float()
    mask = obs["agent_mask"].float().unsqueeze(-1)
    cnn_features = cnn_features * mask
    vloc = vloc * mask
    local_obs = torch.cat([cnn_features, vloc], dim=-1)
    if model.uses_latent_strategy:
        assert z_idx is not None
        z = z_idx.long().reshape(-1).clamp(min=0, max=model.latent_k - 1)
        assert model.strategy_embedding is not None
        z_emb = model.strategy_embedding(z).unsqueeze(1).expand(batch, model.n_agents, model.z_embed_dim)
        actor_in = torch.cat([local_obs, z_emb], dim=-1)
    else:
        actor_in = local_obs
    hidden = model.actor_body(actor_in.reshape(batch * model.n_agents, -1))
    return model.actor_head(hidden).reshape(batch, model.n_agents * model.per_agent_logits)


class PropertyShimTests(unittest.TestCase):
    def test_latent_path_exposes_actor_body_head_embedding(self) -> None:
        model = _build_latent_model()
        self.assertIs(model.actor_body, model.latent_actor.body)
        self.assertIs(model.actor_head, model.latent_actor.action_head)
        self.assertIs(model.strategy_embedding, model.latent_actor.strategy_embedding)
        self.assertIsInstance(model.actor_body, nn.Sequential)
        self.assertIsInstance(model.actor_head, nn.Linear)
        self.assertIsInstance(model.strategy_embedding, nn.Embedding)

    def test_no_latent_path_returns_none_strategy_embedding(self) -> None:
        model = _build_no_latent_model()
        self.assertIs(model.actor_body, model.latent_actor.body)
        self.assertIs(model.actor_head, model.latent_actor.action_head)
        self.assertIsNone(model.strategy_embedding)

    def test_composed_latent_actor_is_a_real_submodule(self) -> None:
        model = _build_latent_model()
        names = {name for name, _ in model.named_modules()}
        self.assertIn("latent_actor", names)
        self.assertIn("latent_actor.body", names)
        self.assertIn("latent_actor.action_head", names)
        self.assertIn("latent_actor.strategy_embedding", names)


class OutputParityTests(unittest.TestCase):
    def test_latent_policy_logits_match_legacy_reference_path(self) -> None:
        model = _build_latent_model()
        obs = _fixed_obs()
        z_idx = torch.tensor([0, 2, 3], dtype=torch.long)
        with torch.no_grad():
            produced = model.policy_logits(obs, z_idx=z_idx)
            reference = _reference_policy_logits(model, obs, z_idx)
        self.assertTrue(
            torch.equal(produced, reference),
            f"composed forward must equal legacy path bitwise; max diff = "
            f"{(produced - reference).abs().max().item():.3e}",
        )

    def test_no_latent_policy_logits_match_legacy_reference_path(self) -> None:
        model = _build_no_latent_model()
        obs = _fixed_obs()
        with torch.no_grad():
            produced = model.policy_logits(obs, z_idx=None)
            reference = _reference_policy_logits(model, obs, z_idx=None)
        self.assertTrue(
            torch.equal(produced, reference),
            "no-latent composed forward must equal legacy path bitwise",
        )


class LegacyStateDictCompatTests(unittest.TestCase):
    def _assert_param_values_match(
        self,
        legacy_sd: Dict[str, torch.Tensor],
        target: SharedActorCentralizedCritic,
    ) -> None:
        target_sd = target.state_dict()
        canonical_legacy = remap_legacy_actor_state_dict_keys(legacy_sd)
        self.assertEqual(
            set(canonical_legacy.keys()),
            set(target_sd.keys()),
            "legacy state_dict (after remap) must cover every param in the new model",
        )
        for k in target_sd:
            self.assertTrue(
                torch.equal(canonical_legacy[k], target_sd[k]),
                f"param {k!r} mismatch after legacy-state-dict round trip",
            )

    def test_latent_legacy_state_dict_loads_and_forward_matches(self) -> None:
        ref = _build_latent_model(seed=2024)
        legacy_sd = _legacy_rename(ref.state_dict())
        self.assertIn("actor_body.0.weight", legacy_sd, "rename helper must produce legacy keys")
        self.assertIn("actor_head.weight", legacy_sd)
        self.assertIn("strategy_embedding.weight", legacy_sd)
        self.assertNotIn("latent_actor.body.0.weight", legacy_sd)

        # Build a *fresh* model with different init and force it to adopt
        # the legacy-shaped state dict via the override.
        torch.manual_seed(99)
        target = SharedActorCentralizedCritic(
            _obs_space(),
            _action_space(),
            actor_cnn_feature_dim=128,
            latent_k=4,
            z_embed_dim=16,
            strategy_hidden_dim=128,
            critic_hidden_dim=128,
        )
        target.eval()
        # ``strict=True`` is the strong contract: every legacy key must land
        # somewhere in the new layout and nothing must be missing.
        target.load_state_dict(legacy_sd, strict=True)

        self._assert_param_values_match(legacy_sd, target)

        obs = _fixed_obs()
        z_idx = torch.tensor([1, 0, 3], dtype=torch.long)
        with torch.no_grad():
            ref_logits = ref.policy_logits(obs, z_idx=z_idx)
            target_logits = target.policy_logits(obs, z_idx=z_idx)
        self.assertTrue(
            torch.equal(ref_logits, target_logits),
            "model loaded from a legacy state_dict must produce identical logits",
        )

    def test_no_latent_legacy_state_dict_loads_and_forward_matches(self) -> None:
        ref = _build_no_latent_model(seed=2024)
        legacy_sd = _legacy_rename(ref.state_dict())
        self.assertIn("actor_body.0.weight", legacy_sd)
        self.assertIn("actor_head.weight", legacy_sd)
        # No strategy_embedding in the no-latent baseline.
        self.assertNotIn("strategy_embedding.weight", legacy_sd)

        torch.manual_seed(99)
        target = SharedActorCentralizedCritic(
            _obs_space(),
            _action_space(),
            actor_cnn_feature_dim=128,
            latent_k=0,
            critic_hidden_dim=128,
        )
        target.eval()
        target.load_state_dict(legacy_sd, strict=True)

        self._assert_param_values_match(legacy_sd, target)

        obs = _fixed_obs()
        with torch.no_grad():
            ref_logits = ref.policy_logits(obs, z_idx=None)
            target_logits = target.policy_logits(obs, z_idx=None)
        self.assertTrue(torch.equal(ref_logits, target_logits))

    def test_remap_helper_is_idempotent(self) -> None:
        ref = _build_latent_model(seed=7)
        new_sd = ref.state_dict()
        # Calling the remap on already-new keys must be a no-op.
        twice = remap_legacy_actor_state_dict_keys(remap_legacy_actor_state_dict_keys(new_sd))
        self.assertEqual(set(twice.keys()), set(new_sd.keys()))
        for k in new_sd:
            self.assertTrue(torch.equal(twice[k], new_sd[k]))


class LatentConditionedActorContractTests(unittest.TestCase):
    """The composed sub-module must accept the trainer's per-token layout."""

    def test_latent_actor_first_layer_in_features_matches_local_plus_z(self) -> None:
        model = _build_latent_model()
        first = model.latent_actor.body[0]
        self.assertIsInstance(first, nn.Linear)
        expected = int(model._local_actor_in_dim) + int(model.z_embed_dim)
        self.assertEqual(int(first.in_features), expected)

    def test_no_latent_first_layer_in_features_excludes_z(self) -> None:
        model = _build_no_latent_model()
        first = model.latent_actor.body[0]
        self.assertEqual(int(first.in_features), int(model._local_actor_in_dim))
        self.assertIsNone(model.latent_actor.strategy_embedding)

    def test_strategy_embedding_dim_matches_z_embed_dim(self) -> None:
        model = _build_latent_model()
        assert model.strategy_embedding is not None
        self.assertEqual(int(model.strategy_embedding.num_embeddings), int(model.latent_k))
        self.assertEqual(int(model.strategy_embedding.embedding_dim), int(model.z_embed_dim))


if __name__ == "__main__":
    unittest.main()
