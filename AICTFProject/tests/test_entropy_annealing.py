from __future__ import annotations

import unittest

from rl.config.ppo_config import PPOConfig
from rl.custom_ppo.csv_writers import _update_fieldnames
from rl.custom_ppo.schedules import linear_anneal, resolve_latent_lam_h
from rl.presets import apply_preset


class EntropyAnnealingTests(unittest.TestCase):
    def test_linear_anneal_before_window(self) -> None:
        self.assertAlmostEqual(linear_anneal(100_000, 0.003, 0.0005, 200_000, 700_000), 0.003)

    def test_linear_anneal_after_window(self) -> None:
        self.assertAlmostEqual(linear_anneal(800_000, 0.003, 0.0005, 200_000, 700_000), 0.0005)

    def test_linear_anneal_midpoint(self) -> None:
        value = linear_anneal(450_000, 0.003, 0.0005, 200_000, 700_000)
        self.assertAlmostEqual(value, 0.00175)

    def test_linear_anneal_zero_width_window(self) -> None:
        self.assertAlmostEqual(linear_anneal(700_000, 0.003, 0.0005, 700_000, 700_000), 0.0005)
        self.assertAlmostEqual(linear_anneal(699_999, 0.003, 0.0005, 700_000, 700_000), 0.003)

    def test_omitted_entropy_schedule_fields_preserve_constant_lam_h(self) -> None:
        cfg = PPOConfig(latent_lam_h=0.003)

        self.assertAlmostEqual(resolve_latent_lam_h(cfg, global_step=0, total_timesteps=1_000), 0.003)
        self.assertAlmostEqual(resolve_latent_lam_h(cfg, global_step=500, total_timesteps=1_000), 0.003)
        self.assertAlmostEqual(resolve_latent_lam_h(cfg, global_step=1_200, total_timesteps=1_000), 0.003)

    def test_custom_entropy_schedule_window(self) -> None:
        cfg = PPOConfig(
            latent_lam_h=0.005,
            latent_lam_h_start=0.003,
            latent_lam_h_end=0.0005,
            latent_entropy_anneal_start=200_000,
            latent_entropy_anneal_end=700_000,
        )

        self.assertAlmostEqual(resolve_latent_lam_h(cfg, global_step=0, total_timesteps=1_000_000), 0.003)
        self.assertAlmostEqual(resolve_latent_lam_h(cfg, global_step=100_000, total_timesteps=1_000_000), 0.003)
        self.assertAlmostEqual(resolve_latent_lam_h(cfg, global_step=450_000, total_timesteps=1_000_000), 0.00175)
        self.assertAlmostEqual(resolve_latent_lam_h(cfg, global_step=700_000, total_timesteps=1_000_000), 0.0005)
        self.assertAlmostEqual(resolve_latent_lam_h(cfg, global_step=800_000, total_timesteps=1_000_000), 0.0005)

    def test_strategy_episode_preset_sets_explicit_entropy_schedule(self) -> None:
        cfg = PPOConfig()
        apply_preset(cfg, "latent_episode_strategic")

        self.assertAlmostEqual(cfg.latent_lam_h, 0.003)
        self.assertAlmostEqual(cfg.latent_lam_h_start, 0.003)
        self.assertAlmostEqual(cfg.latent_lam_h_end, 0.0005)
        self.assertEqual(cfg.latent_entropy_anneal_start, 200_000)
        self.assertEqual(cfg.latent_entropy_anneal_end, 700_000)

    def test_v3h2_entropy_schedule(self) -> None:
        cfg = PPOConfig(
            run_tag="latent_v3h2_balanced_preference_1m_4v4",
            late_entropy_floor=0.0003,
        )
        self.assertAlmostEqual(resolve_latent_lam_h(cfg, global_step=0, total_timesteps=1_000_000), 0.003)
        self.assertAlmostEqual(resolve_latent_lam_h(cfg, global_step=100_000, total_timesteps=1_000_000), 0.003)
        self.assertAlmostEqual(resolve_latent_lam_h(cfg, global_step=299_999, total_timesteps=1_000_000), 0.003)
        self.assertAlmostEqual(resolve_latent_lam_h(cfg, global_step=300_000, total_timesteps=1_000_000), 0.003)
        self.assertAlmostEqual(resolve_latent_lam_h(cfg, global_step=450_000, total_timesteps=1_000_000), 0.002)
        self.assertAlmostEqual(resolve_latent_lam_h(cfg, global_step=600_000, total_timesteps=1_000_000), 0.001)
        self.assertAlmostEqual(resolve_latent_lam_h(cfg, global_step=800_000, total_timesteps=1_000_000), 0.00065)
        self.assertAlmostEqual(resolve_latent_lam_h(cfg, global_step=1_000_000, total_timesteps=1_000_000), 0.0003)

    def test_metrics_csv_includes_current_entropy_coefficient(self) -> None:
        self.assertIn("latent_lam_h", _update_fieldnames(use_latent_strategy=True, latent_k=4))


if __name__ == "__main__":
    unittest.main()
