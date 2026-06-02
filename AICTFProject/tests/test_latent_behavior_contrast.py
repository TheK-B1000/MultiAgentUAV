from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
import torch

from rl.custom_ppo.latent_behavior_contrast import BehaviorContrastMemory
from rl.custom_ppo.latent_strategy_state import LatentStrategyState
from tests.test_latent_episode_warmup import _make_trainer


class BehaviorContrastMemoryTests(unittest.TestCase):
    def test_scores_only_against_other_latents_in_same_bucket(self) -> None:
        memory = BehaviorContrastMemory(
            latent_k=4,
            telemetry_dim=3,
            ema=0.5,
            margin=0.25,
            device="cpu",
        )

        first = memory.score_and_update(
            bucket_id=7,
            z=0,
            embedding=torch.tensor([0.0, 0.0, 0.0]),
            coef=0.05,
        )
        self.assertAlmostEqual(float(first.bonus.item()), 0.0)
        self.assertEqual(first.active, 0)

        second = memory.score_and_update(
            bucket_id=7,
            z=1,
            embedding=torch.tensor([1.0, 0.0, 0.0]),
            coef=0.05,
        )
        self.assertEqual(second.active, 1)
        self.assertGreater(second.distance, 0.0)
        self.assertAlmostEqual(float(second.bonus.item()), 0.0125, places=6)

        other_bucket = memory.score_and_update(
            bucket_id=8,
            z=1,
            embedding=torch.tensor([1.0, 0.0, 0.0]),
            coef=0.05,
        )
        self.assertAlmostEqual(float(other_bucket.bonus.item()), 0.0)
        self.assertEqual(other_bucket.active, 0)


class ForcedZBehaviorContrastRuntimeTests(unittest.TestCase):
    def test_forced_z_episode_skips_qphi_credit_and_records_bonus_field(self) -> None:
        trainer = _make_trainer(n_envs=3, warmup=0, episode_credit=True, gs_dim=34)
        trainer.env.core = SimpleNamespace(Nb=2)
        trainer.global_step = 0
        trainer.latent_forced_z_episode_frac = 1.0
        trainer.latent_behavior_contrast_coef = 0.05
        trainer.latent_behavior_contrast_anneal_after_steps = 0
        trainer.latent_behavior_contrast_anneal_to = 0.0
        trainer.latent_behavior_contrast = BehaviorContrastMemory(
            latent_k=4,
            telemetry_dim=13,
            margin=0.25,
            device="cpu",
        )

        state = torch.zeros((3, 34), dtype=torch.float32)
        latent_state = LatentStrategyState(trainer)
        latent_state.reset()

        z, _, aux = latent_state.strategy_for_step(state)

        self.assertTrue(bool(aux["z_forced"].all().item()))
        self.assertFalse(bool(aux["z_resampled"].any().item()))
        self.assertFalse(bool(latent_state.episode_strategy_has_start.any().item()))

        behavior = torch.ones((3, 13), dtype=torch.float32)
        bonus = latent_state.record_behavior_contrast_step(
            behavior_telemetry=behavior,
            z_idx=z,
            dones=np.ones((3,), dtype=bool),
        )
        self.assertEqual(tuple(bonus.shape), (3,))
        stats = latent_state.behavior_contrast_rollout_stats()
        self.assertAlmostEqual(stats["latent_forced_z_episode_fraction"], 1.0)
        self.assertEqual(stats["latent_behavior_contrast_coef"], 0.05)


if __name__ == "__main__":
    unittest.main()
