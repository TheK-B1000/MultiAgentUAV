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

import inspect
import unittest
from types import SimpleNamespace
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
from rl.custom_ppo.latent_diagnostics import (
    _jsd_from_logits,
    _policy_z_sensitivity_kl,
)
from rl.custom_ppo.ppo_updater import (
    _policy_z_separation_loss,
    _warmup_ramp_value,
    _z_separation_gate_mask,
)
from rl.latent_marl import CONTEXT_STATE_DIM, LatentConditionedActor
from rl.presets import apply_preset
from rl.train_ppo import PPOConfig


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


def _build_latent_adapter_model(seed: int = 1234) -> SharedActorCentralizedCritic:
    torch.manual_seed(seed)
    model = SharedActorCentralizedCritic(
        _obs_space(),
        _action_space(),
        actor_cnn_feature_dim=128,
        latent_k=4,
        z_embed_dim=16,
        strategy_hidden_dim=128,
        critic_hidden_dim=128,
        latent_actor_z_adapter_enabled=True,
        latent_actor_z_adapter_scale=0.35,
        latent_actor_z_adapter_init_std=0.03,
    )
    model.eval()
    return model


def _build_film_only_model(
    seed: int = 1234,
    *,
    film_layers: int = 2,
) -> SharedActorCentralizedCritic:
    torch.manual_seed(seed)
    model = SharedActorCentralizedCritic(
        _obs_space(),
        _action_space(),
        actor_cnn_feature_dim=128,
        latent_k=4,
        z_embed_dim=0,
        strategy_hidden_dim=128,
        critic_hidden_dim=128,
        latent_actor_z_adapter_enabled=True,
        latent_actor_z_adapter_scale=0.5,
        latent_actor_z_adapter_init_std=0.03,
        latent_actor_z_film_layers=film_layers,
    )
    model.eval()
    return model


def _build_latent_onehot_model(seed: int = 1234) -> SharedActorCentralizedCritic:
    torch.manual_seed(seed)
    model = SharedActorCentralizedCritic(
        _obs_space(),
        _action_space(),
        actor_cnn_feature_dim=128,
        latent_k=4,
        z_embed_dim=16,
        strategy_hidden_dim=128,
        critic_hidden_dim=128,
        latent_actor_z_onehot_enabled=True,
        latent_actor_z_onehot_scale=1.0,
        latent_actor_z_embed_scale=1.25,
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
        z_emb = (
            model.strategy_embedding(z)
            * float(getattr(model.latent_actor, "z_embed_scale", 1.0))
        ).unsqueeze(1).expand(batch, model.n_agents, model.z_embed_dim)
        pieces = [local_obs, z_emb]
        if bool(getattr(model.latent_actor, "z_onehot_enabled", False)):
            z_onehot = F.one_hot(z, num_classes=model.latent_k).float()
            z_onehot = z_onehot * float(getattr(model.latent_actor, "z_onehot_scale", 1.0))
            pieces.append(
                z_onehot.unsqueeze(1).expand(batch, model.n_agents, model.latent_k)
            )
        actor_in = torch.cat(pieces, dim=-1)
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

    def test_z_adapter_preserves_actor_shapes(self) -> None:
        model = _build_latent_adapter_model()
        first = model.latent_actor.body[0]
        self.assertIsInstance(first, nn.Linear)
        self.assertEqual(int(first.in_features), int(model._local_actor_in_dim + model.z_embed_dim))
        self.assertEqual(int(model.actor_input_dim), int(model._local_actor_in_dim + model.z_embed_dim))
        self.assertIsNotNone(model.latent_actor.z_adapter)
        assert model.latent_actor.z_adapter is not None
        self.assertEqual(
            tuple(model.latent_actor.z_adapter.weight.shape),
            (int(model.latent_k), int(model.latent_actor.hidden_dim) * 2),
        )

    def test_z_adapter_changes_logits_when_forced_z_changes(self) -> None:
        model = _build_latent_adapter_model()
        obs = _fixed_obs(batch=2)
        z0 = torch.zeros((2,), dtype=torch.long)
        z1 = torch.ones((2,), dtype=torch.long)
        with torch.no_grad():
            logits_z0 = model.policy_logits(obs, z_idx=z0)
            logits_z1 = model.policy_logits(obs, z_idx=z1)
        self.assertEqual(tuple(logits_z0.shape), tuple(logits_z1.shape))
        self.assertEqual(tuple(logits_z0.shape), (2, 110))
        self.assertGreater(float((logits_z0 - logits_z1).abs().max().item()), 1e-6)

    def test_two_layer_film_keeps_v3i13_parameter_contract(self) -> None:
        one_layer = _build_film_only_model(film_layers=1)
        two_layer = _build_film_only_model(film_layers=2)
        self.assertEqual(set(one_layer.state_dict()), set(two_layer.state_dict()))
        two_layer.load_state_dict(one_layer.state_dict(), strict=True)
        self.assertEqual(two_layer.latent_actor.z_film_layers, 2)
        self.assertEqual(
            int(two_layer.actor_input_dim),
            int(two_layer._local_actor_in_dim),
        )

    def test_v3i14_tuned_actor_contract_stays_film_only_and_local(self) -> None:
        cfg = apply_preset(PPOConfig(), "latent_v3i14_tuned")
        model = SharedActorCentralizedCritic(
            _obs_space(),
            _action_space(),
            actor_cnn_feature_dim=cfg.actor_cnn_feature_dim,
            actor_hidden_dim=cfg.actor_hidden_dim,
            latent_k=cfg.latent_k,
            z_embed_dim=cfg.latent_z_embed_dim,
            strategy_hidden_dim=cfg.latent_strategy_hidden,
            critic_hidden_dim=cfg.latent_vf_hidden,
            latent_actor_z_onehot_enabled=(
                cfg.latent_actor_z_onehot_enabled
            ),
            latent_actor_z_adapter_enabled=(
                cfg.latent_actor_z_adapter_enabled
            ),
            latent_actor_z_adapter_scale=cfg.latent_actor_z_adapter_scale,
            latent_actor_z_adapter_init_std=(
                cfg.latent_actor_z_adapter_init_std
            ),
            latent_actor_z_film_layers=cfg.latent_actor_z_film_layers,
        )

        contract = model.input_dim_contract()
        self.assertEqual(contract["actor_input_dim"], 148)
        self.assertEqual(contract["q_phi_input_dim"], CONTEXT_STATE_DIM)
        self.assertEqual(contract["actor_z_embed_dim"], 0)
        self.assertEqual(contract["actor_z_onehot_dim"], 0)
        self.assertIsNone(model.strategy_embedding)
        self.assertIsNotNone(model.latent_actor.z_adapter)
        self.assertEqual(
            set(inspect.signature(model.policy_logits).parameters),
            {"obs", "z_idx"},
        )

    def test_v3i15_sparse_refresh_keeps_v3i14_actor_contract(self) -> None:
        cfg = apply_preset(PPOConfig(), "latent_v3i15_sparse_tactical_refresh")
        model = SharedActorCentralizedCritic(
            _obs_space(),
            _action_space(),
            actor_cnn_feature_dim=cfg.actor_cnn_feature_dim,
            actor_hidden_dim=cfg.actor_hidden_dim,
            latent_k=cfg.latent_k,
            z_embed_dim=cfg.latent_z_embed_dim,
            strategy_hidden_dim=cfg.latent_strategy_hidden,
            critic_hidden_dim=cfg.latent_vf_hidden,
            latent_actor_z_onehot_enabled=cfg.latent_actor_z_onehot_enabled,
            latent_actor_z_adapter_enabled=cfg.latent_actor_z_adapter_enabled,
            latent_actor_z_adapter_scale=cfg.latent_actor_z_adapter_scale,
            latent_actor_z_adapter_init_std=cfg.latent_actor_z_adapter_init_std,
            latent_actor_z_film_layers=cfg.latent_actor_z_film_layers,
        )

        contract = model.input_dim_contract()
        self.assertEqual(contract["actor_input_dim"], 148)
        self.assertEqual(contract["q_phi_input_dim"], CONTEXT_STATE_DIM)
        self.assertEqual(contract["actor_z_embed_dim"], 0)
        self.assertEqual(contract["actor_z_onehot_dim"], 0)
        self.assertIsNone(model.strategy_embedding)
        self.assertIsNotNone(model.latent_actor.z_adapter)
        self.assertEqual(
            set(inspect.signature(model.policy_logits).parameters),
            {"obs", "z_idx"},
        )

    def test_v3i16_direct_z_embedding_expands_local_actor_only(self) -> None:
        cfg = apply_preset(PPOConfig(), "latent_v3i16_policy_z_embedding")
        model = SharedActorCentralizedCritic(
            _obs_space(),
            _action_space(),
            actor_cnn_feature_dim=cfg.actor_cnn_feature_dim,
            actor_hidden_dim=cfg.actor_hidden_dim,
            latent_k=cfg.latent_k,
            z_embed_dim=cfg.latent_z_embed_dim,
            strategy_hidden_dim=cfg.latent_strategy_hidden,
            critic_hidden_dim=cfg.latent_vf_hidden,
            latent_actor_z_onehot_enabled=cfg.latent_actor_z_onehot_enabled,
            latent_actor_z_embed_scale=cfg.latent_actor_z_embed_scale,
            latent_actor_z_adapter_enabled=cfg.latent_actor_z_adapter_enabled,
            latent_actor_z_adapter_scale=cfg.latent_actor_z_adapter_scale,
            latent_actor_z_adapter_init_std=cfg.latent_actor_z_adapter_init_std,
            latent_actor_z_film_layers=cfg.latent_actor_z_film_layers,
        )
        model.eval()

        contract = model.input_dim_contract()
        self.assertEqual(contract["actor_input_dim"], 164)
        self.assertEqual(contract["actor_z_embed_dim"], 16)
        self.assertEqual(contract["actor_z_onehot_dim"], 0)
        self.assertEqual(contract["q_phi_input_dim"], CONTEXT_STATE_DIM)
        self.assertEqual(contract["critic_context_dim"], CONTEXT_STATE_DIM)
        self.assertEqual(contract["critic_z_dim"], 4)
        self.assertIsNotNone(model.strategy_embedding)
        assert model.strategy_embedding is not None
        self.assertEqual(model.strategy_embedding.embedding_dim, 16)
        self.assertIsNone(model.latent_actor.z_adapter)
        self.assertEqual(model.latent_actor.z_film_layers, 0)
        self.assertEqual(
            set(inspect.signature(model.policy_logits).parameters),
            {"obs", "z_idx"},
        )
        actions = torch.zeros((4, len(model.action_dims)), dtype=torch.long)
        critic_extra = model._critic_extra(
            actions,
            torch.arange(4, dtype=torch.long),
        )
        assert critic_extra is not None
        torch.testing.assert_close(critic_extra[:, -4:], torch.eye(4))

        obs = _fixed_obs(batch=4)
        buffer = SimpleNamespace(
            pos=1,
            n_envs=4,
            fields={
                "obs_grid": obs["grid"].unsqueeze(0),
                "obs_vec": obs["vec"].unsqueeze(0),
                "obs_agent_mask": obs["agent_mask"].unsqueeze(0),
                "obs_mask": obs["mask"].unsqueeze(0),
            },
        )
        trainer = SimpleNamespace(
            use_latent_strategy=True,
            latent_k=4,
            device=torch.device("cpu"),
            model=model,
            cfg=SimpleNamespace(batch_size=1024),
        )
        sensitivity = _policy_z_sensitivity_kl(trainer, buffer)
        self.assertGreater(sensitivity["policy_z_sensitivity_KL"], 0.0)
        self.assertGreater(sensitivity["actor_z_jsd_mean"], 0.0)
        self.assertGreaterEqual(
            sensitivity["actor_z_jsd_max"],
            sensitivity["actor_z_jsd_mean"],
        )
        self.assertGreater(sensitivity["actor_z_logit_l2"], 0.0)
        self.assertGreaterEqual(sensitivity["actor_z_argmax_disagree"], 0.0)
        self.assertLessEqual(sensitivity["actor_z_argmax_disagree"], 1.0)
        self.assertEqual(
            len(sensitivity["actor_z_jsd_per_head"].split(",")),
            model.heads_per_agent,
        )
        self.assertEqual(
            len(sensitivity["actor_z_entropy_by_z"].split(",")),
            model.latent_k,
        )
        for z_id in range(model.latent_k):
            self.assertIn(f"actor_z_entropy_z{z_id}", sensitivity)

    def test_jsd_from_logits_is_symmetric_and_zero_for_identical_logits(
        self,
    ) -> None:
        logits_a = torch.tensor([[3.0, -1.0], [0.5, 0.5]])
        logits_b = torch.tensor([[-1.0, 3.0], [0.5, 0.5]])
        same = _jsd_from_logits(logits_a, logits_a)
        ab = _jsd_from_logits(logits_a, logits_b)
        ba = _jsd_from_logits(logits_b, logits_a)

        torch.testing.assert_close(same, torch.zeros_like(same), atol=1e-7, rtol=0.0)
        torch.testing.assert_close(ab, ba)
        self.assertGreater(float(ab[0].item()), 0.0)
        self.assertAlmostEqual(float(ab[1].item()), 0.0, places=7)

    def test_two_layer_film_changes_logits_without_concat_z(self) -> None:
        model = _build_film_only_model(film_layers=2)
        self.assertIsNone(model.strategy_embedding)
        self.assertEqual(model.z_onehot_dim, 0)
        obs = _fixed_obs(batch=2)
        with torch.no_grad():
            logits_z0 = model.policy_logits(
                obs, z_idx=torch.zeros((2,), dtype=torch.long)
            )
            logits_z1 = model.policy_logits(
                obs, z_idx=torch.ones((2,), dtype=torch.long)
            )
        self.assertGreater(float((logits_z0 - logits_z1).abs().max().item()), 1e-6)

    def test_z_onehot_extends_shared_actor_input_without_adapter(self) -> None:
        model = _build_latent_onehot_model()
        first = model.latent_actor.body[0]
        self.assertIsInstance(first, nn.Linear)
        expected = int(model._local_actor_in_dim + model.z_embed_dim + model.latent_k)
        self.assertEqual(int(first.in_features), expected)
        self.assertEqual(int(model.actor_input_dim), expected)
        self.assertEqual(int(model.z_onehot_dim), int(model.latent_k))
        self.assertIsNone(model.latent_actor.z_adapter)

    def test_z_onehot_changes_logits_when_embedding_is_zeroed(self) -> None:
        model = _build_latent_onehot_model()
        assert model.strategy_embedding is not None
        with torch.no_grad():
            model.strategy_embedding.weight.zero_()
        obs = _fixed_obs(batch=2)
        z0 = torch.zeros((2,), dtype=torch.long)
        z1 = torch.ones((2,), dtype=torch.long)
        with torch.no_grad():
            logits_z0 = model.policy_logits(obs, z_idx=z0)
            logits_z1 = model.policy_logits(obs, z_idx=z1)
        self.assertEqual(tuple(logits_z0.shape), tuple(logits_z1.shape))
        self.assertGreater(float((logits_z0 - logits_z1).abs().max().item()), 1e-6)


class ZSeparationLossTests(unittest.TestCase):
    def test_warmup_ramp_value_supports_nonzero_specialization_start(self) -> None:
        kwargs = dict(
            warmup_steps=100_000,
            ramp_steps=300_000,
            start_value=0.005,
            target_value=0.02,
        )
        self.assertEqual(_warmup_ramp_value(global_step=99_999, **kwargs), 0.0)
        self.assertAlmostEqual(
            _warmup_ramp_value(global_step=100_000, **kwargs), 0.005
        )
        self.assertAlmostEqual(
            _warmup_ramp_value(global_step=250_000, **kwargs), 0.0125
        )
        self.assertAlmostEqual(
            _warmup_ramp_value(global_step=400_000, **kwargs), 0.02
        )

    def test_z_separation_gate_rejects_weak_reset_and_high_entropy_rows(self) -> None:
        global_state = torch.zeros((4, 34), dtype=torch.float32)
        global_state[:, 17] = torch.tensor([0.2, 0.01, 0.2, 0.2])
        mask = _z_separation_gate_mask(
            advantages=torch.tensor([0.1, 0.6, 0.7, 0.8]),
            action_entropy=torch.tensor([1.0, 1.0, 9.0, 1.0]),
            global_state=global_state,
            max_action_entropy=10.0,
            min_abs_advantage=0.5,
            min_decision_frac=0.05,
            max_entropy_frac=0.8,
        )
        self.assertEqual(mask.tolist(), [False, False, False, True])

    def test_z_separation_loss_penalizes_identical_logits(self) -> None:
        class ZBlindModel(nn.Module):
            n_agents = 1
            per_agent_action_dims = (3,)

            def policy_logits(self, obs, z_idx=None):
                return torch.zeros((int(z_idx.shape[0]), 3), dtype=torch.float32)

            @staticmethod
            def _mask_logits(logits, mask):
                return logits

        obs = {"mask": torch.ones((4, 3), dtype=torch.float32)}
        z_idx = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        loss, stats = _policy_z_separation_loss(
            ZBlindModel(),
            obs,
            z_idx,
            latent_k=4,
            margin=0.02,
        )
        self.assertAlmostEqual(float(stats["jsd"].item()), 0.0)
        self.assertAlmostEqual(float(loss.item()), 0.02)
        self.assertEqual(float(stats["active"].item()), 1.0)

    def test_z_separation_loss_reports_gate_active_fraction(self) -> None:
        class ZBlindModel(nn.Module):
            n_agents = 1
            per_agent_action_dims = (3,)

            def policy_logits(self, obs, z_idx=None):
                return torch.zeros((int(z_idx.shape[0]), 3), dtype=torch.float32)

            @staticmethod
            def _mask_logits(logits, mask):
                return logits

        obs = {"mask": torch.ones((4, 3), dtype=torch.float32)}
        loss, stats = _policy_z_separation_loss(
            ZBlindModel(),
            obs,
            torch.tensor([0, 1, 2, 3], dtype=torch.long),
            latent_k=4,
            margin=0.02,
            active_mask=torch.tensor([False, False, False, True]),
        )
        self.assertAlmostEqual(float(loss.item()), 0.02)
        self.assertAlmostEqual(float(stats["active"].item()), 0.25)

    def test_z_separation_loss_is_zero_when_logits_are_distinct(self) -> None:
        class ZSeparatedModel(nn.Module):
            n_agents = 1
            per_agent_action_dims = (4,)

            def policy_logits(self, obs, z_idx=None):
                z = z_idx.long().reshape(-1).clamp(min=0, max=3)
                logits = torch.full((int(z.shape[0]), 4), -4.0, dtype=torch.float32)
                logits.scatter_(1, z.unsqueeze(-1), 4.0)
                return logits

            @staticmethod
            def _mask_logits(logits, mask):
                return logits

        obs = {"mask": torch.ones((4, 4), dtype=torch.float32)}
        z_idx = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        loss, stats = _policy_z_separation_loss(
            ZSeparatedModel(),
            obs,
            z_idx,
            latent_k=4,
            margin=0.02,
        )
        self.assertGreater(float(stats["jsd"].item()), 0.02)
        self.assertAlmostEqual(float(loss.item()), 0.0)


if __name__ == "__main__":
    unittest.main()
