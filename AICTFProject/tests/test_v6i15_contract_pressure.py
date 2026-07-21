"""Pinning tests for V6I15 contract-pressure diagnostics."""

from __future__ import annotations

import dataclasses
import unittest

import torch

from rl.config.ppo_config import PPOConfig
from rl.custom_ppo.contract_specialists import contract_specialist_reward
from rl.presets import apply_preset


class V6i15ContractPressurePresetTests(unittest.TestCase):
    def test_3x_aliases_resolve_equal(self) -> None:
        aliases = [
            "v6i15",
            "v6i15_contract_pressure",
            "v6i15_contract_pressure_3x",
            "latent_v6i15_contract_pressure_3x",
            "plan_faithful_latent_v6i15_contract_pressure_3x",
        ]
        base = dataclasses.asdict(apply_preset(PPOConfig(), aliases[0]))
        for alias in aliases[1:]:
            self.assertEqual(base, dataclasses.asdict(apply_preset(PPOConfig(), alias)))

    def test_pressure_arms_are_exact_coef_sweep_over_v6i14(self) -> None:
        parent = dataclasses.asdict(apply_preset(PPOConfig(), "v6i14"))
        expected = {
            "v6i15_contract_pressure_3x": (0.75, "v6i15_contract_pressure_3x_OP8_OP9_OP10"),
            "v6i15_contract_pressure_6x": (1.50, "v6i15_contract_pressure_6x_OP8_OP9_OP10"),
            "v6i15_contract_pressure_10x": (2.50, "v6i15_contract_pressure_10x_OP8_OP9_OP10"),
        }
        for preset, (coef, run_tag) in expected.items():
            with self.subTest(preset=preset):
                cfg_obj = apply_preset(PPOConfig(), preset)
                cfg = dataclasses.asdict(cfg_obj)
                changed = {k for k in parent if parent[k] != cfg[k]}
                self.assertEqual(
                    changed,
                    {
                        "experiment_id",
                        "latent_contract_specialist_coef",
                        "run_tag",
                    },
                )
                self.assertEqual(cfg_obj.experiment_id, "v6i15")
                self.assertEqual(cfg_obj.run_tag, run_tag)
                self.assertTrue(cfg_obj.latent_contract_specialist_enabled)
                self.assertAlmostEqual(float(cfg_obj.latent_contract_specialist_coef), coef)
                self.assertAlmostEqual(float(cfg_obj.latent_contract_specialist_clip), 1.0)
                self.assertEqual(cfg_obj.latent_assignment_mode, "balanced_episode")
                self.assertFalse(cfg_obj.train_router_when_forced)
                self.assertFalse(cfg_obj.train_router_critic_when_forced)
                self.assertEqual(cfg_obj.v6i9_training_stage, "repertoire")


class V6i15ContractPressureRewardTests(unittest.TestCase):
    def _states(self) -> tuple[torch.Tensor, torch.Tensor]:
        prev = torch.zeros((1, 34), dtype=torch.float32)
        nxt = torch.zeros((1, 34), dtype=torch.float32)
        nxt[0, 17] = 0.0
        nxt[0, 20] = 1.0
        nxt[0, 28] = 1.0
        return prev, nxt

    def test_reward_scales_with_pressure_coef(self) -> None:
        prev, nxt = self._states()
        z = torch.tensor([0])
        expected = {
            "v6i14": 0.25,
            "v6i15_contract_pressure_3x": 0.75,
            "v6i15_contract_pressure_6x": 1.50,
            "v6i15_contract_pressure_10x": 2.50,
        }
        for preset, value in expected.items():
            with self.subTest(preset=preset):
                cfg = apply_preset(PPOConfig(), preset)
                out = contract_specialist_reward(prev, nxt, z, cfg)
                self.assertTrue(
                    torch.allclose(out, torch.tensor([value], dtype=torch.float32), atol=1e-6),
                    out,
                )

    def test_10x_aliases_resolve_equal(self) -> None:
        cfg_a = dataclasses.asdict(apply_preset(PPOConfig(), "v6i15_contract_pressure_10x"))
        cfg_b = dataclasses.asdict(
            apply_preset(PPOConfig(), "plan_faithful_latent_v6i15_contract_pressure_10x")
        )
        self.assertEqual(cfg_a, cfg_b)


if __name__ == "__main__":
    unittest.main()
