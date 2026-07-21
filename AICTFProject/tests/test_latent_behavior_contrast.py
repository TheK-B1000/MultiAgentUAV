from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
import torch

from rl.custom_ppo.latent_behavior_contrast import BehaviorContrastMemory, OutcomeDiversityMemory
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


class OutcomeDiversityMemoryTests(unittest.TestCase):
    def test_scores_only_against_other_latents_in_same_context_bucket(self) -> None:
        memory = OutcomeDiversityMemory(latent_k=4, ema=0.5, margin=1.0, device="cpu")

        first = memory.score_and_update(bucket_id=9, z=0, outcome=2.0, coef=0.03)
        self.assertAlmostEqual(float(first.bonus.item()), 0.0)
        self.assertEqual(first.active, 0)

        second = memory.score_and_update(bucket_id=9, z=1, outcome=0.5, coef=0.03)
        self.assertEqual(second.active, 1)
        self.assertAlmostEqual(second.distance, 1.5, places=6)
        self.assertAlmostEqual(float(second.bonus.item()), 0.03, places=6)

        other_bucket = memory.score_and_update(bucket_id=10, z=1, outcome=0.5, coef=0.03)
        self.assertAlmostEqual(float(other_bucket.bonus.item()), 0.0)
        self.assertEqual(other_bucket.active, 0)


class ForcedZBehaviorContrastRuntimeTests(unittest.TestCase):
    def test_forced_z_episode_skips_qphi_credit_and_records_bonus_field(self) -> None:
        trainer = _make_trainer(n_envs=3, warmup=0, episode_credit=True, gs_dim=34)
        trainer.env.core = SimpleNamespace(Nb=2)
        trainer.global_step = 0
        # Set the constant on cfg so resolve_latent_forced_z_frac picks it up
        # via the legacy fallback (all four schedule fields stay None).
        trainer.cfg.latent_forced_z_episode_frac = 1.0
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

    def test_balanced_episode_assignment_feeds_success_gated_contrast(self) -> None:
        trainer = _make_trainer(n_envs=2, warmup=0, episode_credit=False, gs_dim=34)
        trainer.env.core = SimpleNamespace(Nb=2)
        trainer.cfg.latent_assignment_mode = "balanced_episode"
        trainer.latent_behavior_contrast_coef = 0.05
        trainer.latent_behavior_contrast_anneal_after_steps = 0
        trainer.latent_behavior_contrast_anneal_to = 0.0
        trainer.latent_behavior_contrast = BehaviorContrastMemory(
            latent_k=4,
            telemetry_dim=13,
            margin=0.25,
            device="cpu",
        )

        state = torch.zeros((2, 34), dtype=torch.float32)
        latent_state = LatentStrategyState(trainer)
        latent_state.reset()
        z, _, aux = latent_state.strategy_for_step(state)

        self.assertTrue(bool(aux["z_forced"].all().item()))
        self.assertTrue(bool(latent_state.episode_forced_z.all().item()))

        first_behavior = torch.zeros((2, 13), dtype=torch.float32)
        first_success = torch.tensor([True, False], dtype=torch.bool)
        first_bucket = torch.tensor([80_000, 80_000], dtype=torch.long)
        first_bonus = latent_state.record_behavior_contrast_step(
            behavior_telemetry=first_behavior,
            z_idx=z,
            dones=np.ones((2,), dtype=bool),
            success_mask=first_success,
            context_bucket_ids=first_bucket,
        )
        self.assertTrue(torch.equal(first_bonus, torch.zeros_like(first_bonus)))
        first_stats = latent_state.behavior_contrast_rollout_stats()
        self.assertEqual(first_stats["latent_behavior_contrast_active_frac"], 0.0)

        latent_state.episode_forced_z[:] = True
        latent_state.episode_forced_z_id[:] = z
        second_behavior = torch.ones((2, 13), dtype=torch.float32)
        second_bonus = latent_state.record_behavior_contrast_step(
            behavior_telemetry=second_behavior,
            z_idx=torch.tensor([1, 2], dtype=torch.long),
            dones=np.ones((2,), dtype=bool),
            success_mask=torch.tensor([True, True], dtype=torch.bool),
            context_bucket_ids=first_bucket,
        )
        self.assertGreater(float(second_bonus[0].item() + second_bonus[1].item()), 0.0)
        stats = latent_state.behavior_contrast_rollout_stats()
        self.assertGreater(stats["latent_behavior_contrast_active_frac"], 0.0)

    def test_balanced_episode_assignment_feeds_success_gated_outcome_diversity(self) -> None:
        trainer = _make_trainer(n_envs=2, warmup=0, episode_credit=False, gs_dim=34)
        trainer.cfg.latent_assignment_mode = "balanced_episode"
        trainer.latent_outcome_diversity_coef = 0.03
        trainer.latent_outcome_diversity_success_only = True
        trainer.latent_outcome_diversity = OutcomeDiversityMemory(
            latent_k=4,
            margin=1.0,
            device="cpu",
        )

        state = torch.zeros((2, 34), dtype=torch.float32)
        latent_state = LatentStrategyState(trainer)
        latent_state.reset()
        z, _, aux = latent_state.strategy_for_step(state)

        self.assertTrue(bool(aux["z_forced"].all().item()))

        first_bonus = latent_state.record_outcome_diversity_step(
            z_idx=z,
            dones=np.ones((2,), dtype=bool),
            outcome_scores=torch.tensor([2.0, 1.0], dtype=torch.float32),
            success_mask=torch.tensor([True, False], dtype=torch.bool),
            context_bucket_ids=torch.tensor([80_000, 80_000], dtype=torch.long),
        )
        self.assertTrue(torch.equal(first_bonus, torch.zeros_like(first_bonus)))
        first_stats = latent_state.outcome_diversity_rollout_stats()
        self.assertEqual(first_stats["latent_outcome_diversity_active_frac"], 0.0)
        self.assertEqual(first_stats["latent_outcome_diversity_skipped_count"], 1.0)

        latent_state.episode_forced_z[:] = True
        second_bonus = latent_state.record_outcome_diversity_step(
            z_idx=torch.tensor([1, 2], dtype=torch.long),
            dones=np.ones((2,), dtype=bool),
            outcome_scores=torch.tensor([0.5, 3.0], dtype=torch.float32),
            success_mask=torch.tensor([True, True], dtype=torch.bool),
            context_bucket_ids=torch.tensor([80_000, 80_000], dtype=torch.long),
        )
        self.assertGreater(float(second_bonus[0].item() + second_bonus[1].item()), 0.0)
        stats = latent_state.outcome_diversity_rollout_stats()
        self.assertGreater(stats["latent_outcome_diversity_active_frac"], 0.0)
        self.assertAlmostEqual(stats["latent_outcome_diversity_coef"], 0.03)


if __name__ == "__main__":
    unittest.main()
