"""Tests for OP6-OP12 adaptive memory flags under strategic niches."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gpu_env._core._bt_adaptive import _BTAdaptiveMixin
from gpu_env._core._bt_profiles import (
    LRO_AUDITED_OPPONENT_POOL,
    build_profile_tensors,
    profile_for_level,
)
from game_field_gpu import GPUCTFVecEnv, GPUFieldConfig
from opponent_params import sample_batched_opponent_params


class BTAdaptiveProfileTests(unittest.TestCase):
    def test_only_exploiter_and_converter_enable_adaptive(self) -> None:
        for lvl in range(5, 11):
            self.assertFalse(profile_for_level(lvl).adaptive_enabled, f"level {lvl}")
        self.assertTrue(profile_for_level(11).adaptive_enabled)
        self.assertTrue(profile_for_level(12).adaptive_enabled)

    def test_profile_tensors_export_adaptive_flag_for_audited_pool(self) -> None:
        keys = list(LRO_AUDITED_OPPONENT_POOL)
        prof = build_profile_tensors(keys, device=torch.device("cpu"), batch_size=len(keys))
        # Indices: OP6..OP10 False; OP11/OP12 True
        self.assertFalse(bool(prof["adaptive_enabled"][:5].any().item()))
        self.assertTrue(bool(prof["adaptive_enabled"][5:].all().item()))

    def test_matched_2v2_speed_bands_across_niches(self) -> None:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(123)
        for key in LRO_AUDITED_OPPONENT_POOL:
            params = sample_batched_opponent_params(
                "SCRIPTED",
                key,
                n_agents=2,
                batch_size=32,
                device="cpu",
                generator=gen,
            )
            self.assertGreaterEqual(float(params["speed_mult"].min().item()), 0.90, key)
            self.assertLessEqual(float(params["speed_mult"].max().item()), 1.05, key)

    def test_adaptive_hardpool_keys_include_audited_tags(self) -> None:
        for tag in LRO_AUDITED_OPPONENT_POOL:
            self.assertIn(tag, _BTAdaptiveMixin._ADAPTIVE_HARDPOOL_KEYS)

    def test_speed_overdrive_mask_allows_hardpool_red_to_exceed_base_cap(self) -> None:
        env = GPUCTFVecEnv(
            GPUFieldConfig(
                n_envs=1,
                max_blue_agents=2,
                max_red_agents=2,
                map_layout="map_b",
                max_decision_steps=16,
                aquaticus_profile=True,
                rules_profile="OURS",
                device="cpu",
                seed=123,
            )
        )
        try:
            x = torch.full((1, 2), 5.0)
            y = torch.full((1, 2), 5.0)
            heading = torch.zeros((1, 2))
            speed = torch.full((1, 2), float(env.cfg.max_speed_cps))
            alive = torch.ones((1, 2), dtype=torch.bool)
            tx = torch.full((1, 2), 20.0)
            ty = torch.full((1, 2), 5.0)
            cap = torch.full((1, 2), float(env.cfg.max_speed_cps) * 1.20)
            mask = torch.ones((1, 2), dtype=torch.bool)
            _, _, _, speed2, _, _ = env.core._integrate_side(
                x,
                y,
                heading,
                speed,
                alive,
                tx,
                ty,
                speed_cap=cap,
                speed_overdrive_mask=mask,
            )
            self.assertGreater(float(speed2.max().item()), float(env.cfg.max_speed_cps))
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
