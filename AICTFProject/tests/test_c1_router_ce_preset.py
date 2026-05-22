from __future__ import annotations

import unittest

from rl.train_ppo import PPOConfig, TrainMode, _apply_training_preset


class C1RouterCePresetTests(unittest.TestCase):
    def test_c1_router_ce_preset_is_ce_only_non_oracle_freeze(self) -> None:
        cfg = _apply_training_preset(PPOConfig(), "latent_c1_router_ce")
        self.assertTrue(cfg.use_latent_strategy)
        self.assertEqual(cfg.mode, TrainMode.OPPONENT_POOL.value)
        self.assertEqual(cfg.opponent_pool, ("OP3", "OP5", "OP6", "OP7"))
        self.assertTrue(cfg.freeze_actor_critic)
        self.assertEqual(cfg.qphi_oracle_mode, "none")
        self.assertEqual(cfg.forced_z_mode, "none")
        self.assertFalse(cfg.use_per_z_shaping)
        self.assertFalse(cfg.latent_strategy_aux_return_head)
        self.assertAlmostEqual(cfg.latent_strategy_aux_return_coef, 0.0)
        self.assertAlmostEqual(cfg.latent_strategy_tau, 1.0)
        self.assertEqual(cfg.latent_entropy_objective, "none")
        self.assertAlmostEqual(cfg.latent_lam_h, 0.0)
        self.assertAlmostEqual(cfg.latent_lam_p, 0.0)
        self.assertAlmostEqual(cfg.latent_strategy_ppo_coef, 0.0)
        self.assertEqual(cfg.latent_resample_every_n, 0)
        self.assertAlmostEqual(cfg.router_ce_coef, 1.0)
        self.assertEqual(cfg.router_ce_mode, "soft")


if __name__ == "__main__":
    unittest.main()
