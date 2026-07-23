"""Pinning tests for V6I23 Summer-compatible population birth."""
from __future__ import annotations

import dataclasses
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from gymnasium import spaces

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from game_field_gpu import VEC_OBS_DIM
from rl.config.ppo_config import PPOConfig
from rl.custom_ppo import SharedActorCentralizedCritic
from rl.custom_ppo.trainer_optimizers import (
    freeze_shared_trunk_train_z_only,
    is_z_specific_actor_param,
)
from rl.presets import apply_preset
from rl.training.config_validation import normalize_and_validate_training_config

LATENT_K = 4


def _obs_space() -> spaces.Dict:
    return spaces.Dict({
        "grid": spaces.Box(0.0, 1.0, (2, 7, 20, 20), dtype=np.float32),
        "vec": spaces.Box(-1.0, 1.0, (2, VEC_OBS_DIM), dtype=np.float32),
        "agent_mask": spaces.Box(0.0, 1.0, (2,), dtype=np.float32),
        "mask": spaces.Box(0.0, 1.0, (110,), dtype=np.float32),
    })


def _action_space() -> spaces.MultiDiscrete:
    return spaces.MultiDiscrete([5, 50, 5, 50])


def _make_model(
    *,
    active_z_only: bool = False,
    per_z_heads: bool = False,
    actor_hidden_dim: int = 64,
) -> SharedActorCentralizedCritic:
    torch.manual_seed(0)
    return SharedActorCentralizedCritic(
        _obs_space(),
        _action_space(),
        latent_k=LATENT_K,
        z_embed_dim=16,
        actor_hidden_dim=actor_hidden_dim,
        strategy_hidden_dim=64,
        use_recurrent_selector=True,
        recurrent_selector_hidden_dim=64,
        use_episode_strategy_value_head=True,
        router_context_mode="current",
        enable_latent_z_residual=True,
        latent_z_residual_alpha=0.1,
        latent_population_birth_active_z_only=active_z_only,
        latent_population_birth_per_z_action_heads=per_z_heads,
    )


class V6i23PresetInheritanceTests(unittest.TestCase):
    def test_aliases_resolve_identically(self) -> None:
        aliases = [
            "v6i23",
            "v6i23_population_birth",
            "latent_v6i23_population_birth",
            "plan_faithful_latent_v6i23_population_birth",
        ]
        base = dataclasses.asdict(apply_preset(PPOConfig(), aliases[0]))
        for alias in aliases[1:]:
            self.assertEqual(base, dataclasses.asdict(apply_preset(PPOConfig(), alias)))

    def test_minimal_diff_vs_v6i22e(self) -> None:
        parent_cfg = apply_preset(PPOConfig(), "v6i22e")
        normalize_and_validate_training_config(parent_cfg)
        parent = dataclasses.asdict(parent_cfg)
        cfg_obj = apply_preset(PPOConfig(), "v6i23")
        normalize_and_validate_training_config(cfg_obj)
        cfg = dataclasses.asdict(cfg_obj)
        changed = {k for k in parent if parent[k] != cfg[k]}
        self.assertEqual(
            changed,
            {
                "experiment_id",
                "latent_population_birth_active_z_only",
                "latent_population_birth_per_z_action_heads",
                "run_tag",
            },
        )
        self.assertTrue(cfg_obj.latent_population_birth_active_z_only)
        self.assertTrue(cfg_obj.latent_population_birth_per_z_action_heads)
        self.assertEqual(cfg_obj.latent_z_residual_alpha, 0.1)
        self.assertEqual(cfg_obj.experiment_id, "v6i23")
        self.assertEqual(
            cfg_obj.run_tag,
            "v6i23_population_birth_OP8_OP9_OP10_OP11_OP12",
        )

    def test_preserves_birth_scaffold_and_blocks_router(self) -> None:
        cfg = apply_preset(PPOConfig(), "v6i23")
        normalize_and_validate_training_config(cfg)
        self.assertEqual(cfg.latent_assignment_mode, "balanced_episode")
        self.assertFalse(cfg.train_router_when_forced)
        self.assertFalse(cfg.train_router_critic_when_forced)
        self.assertEqual(cfg.v6i9_training_stage, "repertoire")
        self.assertFalse(cfg.latent_contract_specialist_enabled)
        self.assertEqual(float(cfg.latent_behavior_contrast_coef), 0.0)
        self.assertEqual(float(cfg.latent_outcome_diversity_coef), 0.0)


class V6i23ArchitectureTests(unittest.TestCase):
    def test_per_z_heads_exist_and_biases_absent(self) -> None:
        model = _make_model(active_z_only=True, per_z_heads=True)
        actor = model.latent_actor
        self.assertIsNotNone(actor.latent_action_heads)
        self.assertEqual(len(actor.latent_action_heads), LATENT_K)
        self.assertIsNone(actor.latent_action_biases)
        self.assertIsNone(actor.latent_adapter_gates)
        self.assertTrue(actor._population_birth_active_z_only)

    def test_per_z_heads_are_stage2_trainable(self) -> None:
        model = _make_model(active_z_only=True, per_z_heads=True)
        freeze_shared_trunk_train_z_only(model)
        head_names = [n for n, _ in model.named_parameters() if "latent_action_heads" in n]
        self.assertTrue(head_names)
        named = dict(model.named_parameters())
        for name in head_names:
            self.assertTrue(is_z_specific_actor_param(name), name)
            self.assertTrue(named[name].requires_grad, name)
        for name, param in model.named_parameters():
            if name.startswith("latent_actor.action_head"):
                self.assertFalse(param.requires_grad, name)

    def test_forced_z_logits_diverge_when_heads_differ(self) -> None:
        model = _make_model(active_z_only=True, per_z_heads=True)
        actor = model.latent_actor
        with torch.no_grad():
            actor.latent_action_heads[0].bias.zero_()
            actor.latent_action_heads[1].bias.fill_(5.0)
        local = torch.randn(2, actor.local_feature_dim)
        logits0 = actor(local, torch.zeros(2, dtype=torch.long))
        logits1 = actor(local, torch.ones(2, dtype=torch.long))
        self.assertGreater(float((logits0 - logits1).abs().max().detach()), 1.0)

    def test_active_z_only_matches_stack_all_numerically(self) -> None:
        """Same adapters + shared-equivalent heads → active-z-only == stack-all."""
        model_active = _make_model(active_z_only=True, per_z_heads=True)
        model_stack = _make_model(active_z_only=False, per_z_heads=False)
        with torch.no_grad():
            for k in range(LATENT_K):
                model_stack.latent_actor.latent_adapters[k].load_state_dict(
                    model_active.latent_actor.latent_adapters[k].state_dict()
                )
            model_stack.latent_actor.action_head.load_state_dict(
                model_active.latent_actor.action_head.state_dict()
            )
            model_stack.latent_actor.strategy_embedding.load_state_dict(
                model_active.latent_actor.strategy_embedding.state_dict()
            )
            model_stack.latent_actor.body.load_state_dict(
                model_active.latent_actor.body.state_dict()
            )
            model_active.latent_actor.sync_per_z_action_heads_from_shared()
            model_stack.latent_actor.latent_action_biases.zero_()

        local = torch.randn(4, model_active.latent_actor.local_feature_dim)
        z = torch.tensor([0, 1, 0, 2], dtype=torch.long)
        logits_a = model_active.latent_actor(local, z)
        logits_b = model_stack.latent_actor(local, z)
        self.assertTrue(torch.allclose(logits_a, logits_b, atol=1e-5, rtol=1e-5))

    def test_compat_bypass_uses_shared_action_head(self) -> None:
        model = _make_model(active_z_only=True, per_z_heads=True)
        actor = model.latent_actor
        with torch.no_grad():
            for head in actor.latent_action_heads:
                head.bias.fill_(4.0)
        local = torch.randn(2, actor.local_feature_dim)
        z = torch.zeros(2, dtype=torch.long)
        logits_full = actor(local, z)
        actor._residual_bypass_for_compat = True
        logits_bypass = actor(local, z)
        actor._residual_bypass_for_compat = False
        self.assertFalse(torch.allclose(logits_full, logits_bypass, atol=1e-3))
        # Bypass must match shared head on residual-free trunk features.
        # Reconstruct trunk without residual by temporary bypass + action_head.
        actor._residual_bypass_for_compat = True
        # Forward through body only: use logits_from_hidden after residual skip.
        pieces = [local.float(), actor.strategy_embedding(z) * actor.z_embed_scale]
        hidden = actor.body(torch.cat(pieces, dim=-1))
        expected = actor.action_head(hidden)
        actor._residual_bypass_for_compat = False
        self.assertTrue(torch.allclose(logits_bypass, expected, atol=1e-5, rtol=1e-5))


if __name__ == "__main__":
    unittest.main()
