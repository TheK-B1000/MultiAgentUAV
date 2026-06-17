"""Tests for v5i8_repertoire_uniform_z Stage-1 diagnostic preset."""

from __future__ import annotations

import sys
import unittest
from dataclasses import asdict
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rl.config.ppo_config import PPOConfig
from rl.custom_ppo.latent_strategy_state import LatentStrategyState
from rl.custom_ppo.schedules import resolve_latent_forced_z_frac
from rl.presets import apply_preset
from rl.presets.plan_faithful import (
    apply_plan_faithful_latent_v5i8_repertoire_uniform_z,
    apply_plan_faithful_latent_v5i8_split_lane_v2_task_pressure,
)


class V5i8RepertoireUniformZPresetTests(unittest.TestCase):
    def test_minimal_diff_vs_v5i8(self) -> None:
        base = asdict(apply_plan_faithful_latent_v5i8_split_lane_v2_task_pressure(PPOConfig()))
        abl = asdict(apply_plan_faithful_latent_v5i8_repertoire_uniform_z(PPOConfig()))
        delta = {k: (base[k], abl[k]) for k in base if base[k] != abl[k]}
        self.assertEqual(
            set(delta),
            {
                "latent_forced_z_episode_frac",
                "latent_forced_z_episode_frac_start",
                "latent_forced_z_episode_frac_end",
                "latent_forced_z_anneal_start",
                "latent_forced_z_anneal_end",
                "run_tag",
            },
        )
        self.assertAlmostEqual(float(abl["latent_forced_z_episode_frac_start"]), 1.0)
        self.assertAlmostEqual(float(abl["latent_forced_z_episode_frac_end"]), 1.0)
        self.assertAlmostEqual(float(abl["latent_forced_z_episode_frac"]), 1.0)

    def test_forced_frac_stays_one_for_entire_run(self) -> None:
        cfg = apply_plan_faithful_latent_v5i8_repertoire_uniform_z(PPOConfig())
        for step in (0, 250_000, 500_000, 750_000, 1_000_000):
            self.assertAlmostEqual(
                resolve_latent_forced_z_frac(cfg, global_step=step),
                1.0,
            )

    def test_preset_resolves_from_registry(self) -> None:
        cfg = apply_preset(PPOConfig(), "v5i8_repertoire_uniform_z")
        self.assertIn("repertoire_uniform_z", cfg.run_tag)
        self.assertAlmostEqual(float(cfg.latent_forced_z_episode_frac_start), 1.0)


class V5i8RepertoireUniformZRuntimeTests(unittest.TestCase):
    def test_forced_z_works_without_behavior_contrast(self) -> None:
        from tests.test_forced_z_anneal import _make_trainer

        trainer = _make_trainer(n_envs=8, warmup=0, episode_credit=True, gs_dim=4)
        trainer.cfg = apply_plan_faithful_latent_v5i8_repertoire_uniform_z(trainer.cfg)
        latent_state = LatentStrategyState(trainer)
        latent_state.reset()

        gs = torch.zeros((8, 4), dtype=torch.float32)
        _, _, aux = latent_state.strategy_for_step(gs)

        self.assertTrue(bool(aux["z_forced"].all().item()))
        self.assertFalse(bool(latent_state.episode_strategy_has_start.any().item()))

    def test_forced_episode_opp_z_counter_increments(self) -> None:
        from tests.test_forced_z_anneal import _make_trainer

        trainer = _make_trainer(n_envs=1, warmup=0, episode_credit=True, gs_dim=4)
        trainer.cfg = apply_plan_faithful_latent_v5i8_repertoire_uniform_z(trainer.cfg)
        latent_state = LatentStrategyState(trainer)
        latent_state.reset()
        gs = torch.zeros((1, 4), dtype=torch.float32)
        _, z_idx, _ = latent_state.strategy_for_step(gs)
        latent_state.episode_forced_z[0] = True
        latent_state.episode_forced_z_id[0] = int(z_idx[0].item())
        latent_state.record_episode_strategy_outcome(
            0,
            {
                "opponent_kind": "scripted",
                "opponent_key": "OP5_RUSHER",
                "blue_score": 1,
                "red_score": 0,
            },
            episode_return=1.0,
        )
        z_val = int(z_idx[0].item())
        self.assertGreater(int(latent_state.rollout_forced_z_episode_count_by_z[z_val]), 0)
        self.assertGreater(int(latent_state.rollout_forced_episode_count_by_opp_z[4, z_val]), 0)

    def test_forced_episode_counting_works_when_episode_credit_disabled(self) -> None:
        from tests.test_forced_z_anneal import _make_trainer

        trainer = _make_trainer(n_envs=1, warmup=0, episode_credit=False, gs_dim=4)
        trainer.cfg = apply_plan_faithful_latent_v5i8_repertoire_uniform_z(trainer.cfg)
        latent_state = LatentStrategyState(trainer)
        latent_state.reset()
        gs = torch.zeros((1, 4), dtype=torch.float32)
        _, z_idx, _ = latent_state.strategy_for_step(gs)
        latent_state.episode_forced_z[0] = True
        latent_state.episode_forced_z_id[0] = int(z_idx[0].item())
        latent_state.record_episode_strategy_outcome(
            0,
            {
                "opponent_kind": "scripted",
                "opponent_key": "OP6",
                "blue_score": 1,
                "red_score": 0,
            },
            episode_return=1.0,
        )
        z_val = int(z_idx[0].item())
        stats = latent_state.behavior_contrast_rollout_stats()
        self.assertAlmostEqual(stats["forced_sample_count_by_z_{}".format(z_val)], 1.0)
        self.assertAlmostEqual(stats["forced_episode_opp5_z{}_count".format(z_val)], 1.0)


if __name__ == "__main__":
    unittest.main()
