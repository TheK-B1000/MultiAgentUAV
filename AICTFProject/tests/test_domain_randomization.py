"""Tests for episode-level training domain randomization."""

from __future__ import annotations

import unittest

import torch

from game_field_gpu import GPUCTFVecEnv, GPUFieldConfig


class TestDomainRandomization(unittest.TestCase):
    def test_runtime_tensors_resample_when_dr_on(self) -> None:
        cfg = GPUFieldConfig(
            n_envs=4,
            n_agents_per_team=2,
            device="cpu",
            seed=0,
            train_domain_randomization=True,
            dr_sensor_noise_sigma_max=0.2,
            dr_sensor_dropout_max=0.1,
            dr_blue_speed_jitter=0.15,
        )
        env = GPUCTFVecEnv(cfg)
        try:
            core = env.core
            a = core.rt_sensor_noise_sigma_cells.clone()
            d = core.rt_sensor_dropout_prob.clone()
            b = core.rt_blue_speed_scale.clone()
            core.reset_indices(torch.ones((core.B,), dtype=torch.bool, device=core.device))
            self.assertFalse(torch.allclose(a, core.rt_sensor_noise_sigma_cells))
            self.assertFalse(torch.allclose(d, core.rt_sensor_dropout_prob))
            self.assertFalse(torch.allclose(b, core.rt_blue_speed_scale))
            lo_spd = max(0.5, 1.0 - 0.15)
            self._assert_dr_bounds(core, noise_hi=0.2, dropout_hi=0.1, speed_lo=lo_spd, speed_hi=1.0)
        finally:
            env.close()

    def test_custom_config_bounds_not_hardcoded(self) -> None:
        """Sampler must read GPUFieldConfig limits (catch wiring to wrong constants)."""
        cfg = GPUFieldConfig(
            n_envs=3,
            n_agents_per_team=2,
            device="cpu",
            seed=7,
            train_domain_randomization=True,
            dr_sensor_noise_sigma_max=0.04,
            dr_sensor_dropout_max=0.015,
            dr_blue_speed_jitter=0.25,
        )
        lo_spd = max(0.5, 1.0 - 0.25)
        env = GPUCTFVecEnv(cfg)
        try:
            core = env.core
            core.reset_indices(torch.ones((core.B,), dtype=torch.bool, device=core.device))
            self._assert_dr_bounds(core, noise_hi=0.04, dropout_hi=0.015, speed_lo=lo_spd, speed_hi=1.0)
        finally:
            env.close()

    def test_many_resets_remain_within_bounds(self) -> None:
        cfg = GPUFieldConfig(
            n_envs=8,
            n_agents_per_team=2,
            device="cpu",
            seed=99,
            train_domain_randomization=True,
            dr_sensor_noise_sigma_max=0.12,
            dr_sensor_dropout_max=0.08,
            dr_blue_speed_jitter=0.12,
        )
        lo_spd = max(0.5, 1.0 - 0.12)
        env = GPUCTFVecEnv(cfg)
        try:
            core = env.core
            mask = torch.ones((core.B,), dtype=torch.bool, device=core.device)
            for _ in range(48):
                core.reset_indices(mask)
                self._assert_dr_bounds(core, noise_hi=0.12, dropout_hi=0.08, speed_lo=lo_spd, speed_hi=1.0)
        finally:
            env.close()

    def _assert_dr_bounds(
        self,
        core: object,
        *,
        noise_hi: float,
        dropout_hi: float,
        speed_lo: float,
        speed_hi: float,
    ) -> None:
        n = core.rt_sensor_noise_sigma_cells
        p = core.rt_sensor_dropout_prob
        s = core.rt_blue_speed_scale
        self.assertGreaterEqual(float(n.min()), -1e-6)
        self.assertLessEqual(float(n.max()), noise_hi + 1e-5)
        self.assertGreaterEqual(float(p.min()), -1e-6)
        self.assertLessEqual(float(p.max()), dropout_hi + 1e-5)
        self.assertGreaterEqual(float(s.min()), speed_lo - 1e-5)
        self.assertLessEqual(float(s.max()), speed_hi + 1e-5)

    def test_no_dr_matches_unit_speed_scale(self) -> None:
        cfg = GPUFieldConfig(
            n_envs=2,
            n_agents_per_team=2,
            device="cpu",
            seed=1,
            train_domain_randomization=False,
        )
        env = GPUCTFVecEnv(cfg)
        try:
            core = env.core
            core.reset_indices(torch.ones((core.B,), dtype=torch.bool, device=core.device))
            self.assertTrue(torch.allclose(core.rt_blue_speed_scale, torch.ones_like(core.rt_blue_speed_scale)))
            self.assertTrue(torch.allclose(core.rt_sensor_noise_sigma_cells, torch.zeros_like(core.rt_sensor_noise_sigma_cells)))
            self.assertTrue(torch.allclose(core.rt_sensor_dropout_prob, torch.zeros_like(core.rt_sensor_dropout_prob)))
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
