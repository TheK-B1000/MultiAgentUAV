"""Pinning tests for V6I16 capacity + sharp-contract diagnostics."""

from __future__ import annotations

import dataclasses
import unittest

import torch

from rl.config.ppo_config import PPOConfig
from rl.custom_ppo.contract_specialists import contract_specialist_reward
from rl.custom_ppo.trainer_optimizers import is_z_specific_actor_param
from rl.presets import apply_preset


class V6i16PresetTests(unittest.TestCase):
    def test_default_aliases_resolve_to_combined_arm(self) -> None:
        aliases = [
            "v6i16",
            "v6i16_capacity_feature_ablation",
            "v6i16_capacity_sharp_contracts",
            "latent_v6i16_capacity_sharp_contracts",
            "plan_faithful_latent_v6i16_capacity_sharp_contracts",
        ]
        base = dataclasses.asdict(apply_preset(PPOConfig(), aliases[0]))
        for alias in aliases[1:]:
            self.assertEqual(base, dataclasses.asdict(apply_preset(PPOConfig(), alias)))

    def test_arms_are_exact_matrix_over_v6i15_3x(self) -> None:
        parent = dataclasses.asdict(apply_preset(PPOConfig(), "v6i15_contract_pressure_3x"))
        expected = {
            "v6i16_sharp_contracts": {
                "experiment_id",
                "latent_contract_specialist_variant",
                "run_tag",
            },
            "v6i16_capacity": {
                "experiment_id",
                "latent_actor_z_adapter_enabled",
                "latent_actor_z_adapter_init_std",
                "latent_actor_z_adapter_scale",
                "latent_z_gate_init",
                "run_tag",
            },
            "v6i16_capacity_sharp_contracts": {
                "experiment_id",
                "latent_actor_z_adapter_enabled",
                "latent_actor_z_adapter_init_std",
                "latent_actor_z_adapter_scale",
                "latent_contract_specialist_variant",
                "latent_z_gate_init",
                "run_tag",
            },
        }
        for preset, changed_fields in expected.items():
            with self.subTest(preset=preset):
                cfg_obj = apply_preset(PPOConfig(), preset)
                cfg = dataclasses.asdict(cfg_obj)
                changed = {k for k in parent if parent[k] != cfg[k]}
                self.assertEqual(changed, changed_fields)
                self.assertEqual(cfg_obj.experiment_id, "v6i16")
                self.assertTrue(cfg_obj.latent_contract_specialist_enabled)
                self.assertAlmostEqual(float(cfg_obj.latent_contract_specialist_coef), 0.75)
                self.assertEqual(cfg_obj.latent_assignment_mode, "balanced_episode")
                self.assertFalse(cfg_obj.train_router_when_forced)
                self.assertFalse(cfg_obj.train_router_critic_when_forced)
                self.assertEqual(cfg_obj.v6i9_training_stage, "repertoire")

    def test_capacity_arms_enable_stronger_z_pathway(self) -> None:
        for preset in ["v6i16_capacity", "v6i16_capacity_sharp_contracts"]:
            with self.subTest(preset=preset):
                cfg = apply_preset(PPOConfig(), preset)
                self.assertTrue(cfg.enable_latent_z_residual)
                self.assertAlmostEqual(float(cfg.latent_z_gate_init), 0.08)
                self.assertTrue(cfg.latent_actor_z_adapter_enabled)
                self.assertAlmostEqual(float(cfg.latent_actor_z_adapter_scale), 0.10)
                self.assertAlmostEqual(float(cfg.latent_actor_z_adapter_init_std), 0.05)

    def test_z_adapter_is_trainable_in_repertoire_freeze_allowlist(self) -> None:
        self.assertTrue(is_z_specific_actor_param("latent_actor.z_adapter.weight"))
        self.assertTrue(is_z_specific_actor_param("latent_actor.strategy_embedding.weight"))


class V6i16SharpContractRewardTests(unittest.TestCase):
    def _states(self) -> tuple[torch.Tensor, torch.Tensor]:
        prev = torch.zeros((4, 34), dtype=torch.float32)
        nxt = torch.zeros((4, 34), dtype=torch.float32)

        # z0: opening pressure / enemy-flag threat.
        nxt[0, 17] = 0.0
        nxt[0, 20] = 1.0
        nxt[0, 28] = 1.0

        # z1: friendly-carrier escort plus conversion progress.
        prev[1, 23] = 0.8
        nxt[1, 11] = 1.0
        nxt[1, 23] = 0.3
        nxt[1, 24] = 1.0
        nxt[1, 25] = 1.0

        # z2: home defense / return denial.
        prev[2, 19] = 1.0
        nxt[2, 10] = 1.0
        nxt[2, 19] = 0.2
        nxt[2, 21] = 1.0
        nxt[2, 30] = 1.0

        # z3: lane control / split pressure outside carrier context.
        prev[3, 20] = 0.1
        nxt[3, 20] = 0.5
        nxt[3, 32] = 1.0
        nxt[3, 33] = 1.0
        return prev, nxt

    def test_sharp_contract_rewards_selected_roles(self) -> None:
        prev, nxt = self._states()
        cfg = apply_preset(PPOConfig(), "v6i16_sharp_contracts")
        z = torch.tensor([0, 1, 2, 3])
        out = contract_specialist_reward(prev, nxt, z, cfg)
        expected = torch.tensor([0.75, 0.675, 0.7125, 0.6075], dtype=torch.float32)
        self.assertTrue(torch.allclose(out, expected, atol=1e-5), out)

    def test_base_contract_variant_stays_default_for_v6i15(self) -> None:
        cfg = apply_preset(PPOConfig(), "v6i15_contract_pressure_3x")
        self.assertEqual(cfg.latent_contract_specialist_variant, "base")


if __name__ == "__main__":
    unittest.main()
