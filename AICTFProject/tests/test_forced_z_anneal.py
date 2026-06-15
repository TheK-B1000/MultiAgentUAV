"""Tests for the v5i3 forced-z anneal schedule and per-z router telemetry.

The v5i3 design layers a linearly annealed ``latent_forced_z_episode_frac``
on top of v5i2 to repair the v5i2 router collapse without changing the loss
objective. These tests pin:

1. Schedule resolution before / during / after the anneal window.
2. Resume-safety: the resolver is a pure function of ``cfg`` + ``global_step``.
3. Zero-config reproduces v5i2 (the legacy constant path stays bit-stable).
4. Forced episodes never enter ``rollout_strategy_episode_records`` -- so
   q_phi's on-policy PPO update is structurally insulated from forced episodes.
5. v5i3 preset config matches the documented schedule.
6. Per-z router telemetry keys exist in the metrics CSV fieldnames.

The unit-level path uses the lightweight ``SimpleNamespace`` trainer harness
from ``test_latent_episode_warmup``; the schedule path is exercised directly
on ``rl.custom_ppo.schedules.resolve_latent_forced_z_frac``.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
import torch

from rl.config.ppo_config import PPOConfig
from rl.custom_ppo.csv_writers import _update_fieldnames
from rl.custom_ppo.schedules import resolve_latent_forced_z_frac
from rl.presets import apply_preset

from tests.test_latent_episode_warmup import (
    LatentStrategyState,
    _make_trainer,
)


class ForcedZScheduleResolverTests(unittest.TestCase):
    def test_zero_config_returns_legacy_constant(self) -> None:
        """v5i2 + every pre-v5i3 preset must reproduce exactly: no schedule fields."""
        cfg = PPOConfig(latent_forced_z_episode_frac=0.0)
        for step in (0, 100_000, 500_000, 999_999, 10_000_000):
            self.assertAlmostEqual(
                resolve_latent_forced_z_frac(cfg, global_step=step), 0.0
            )

        cfg2 = PPOConfig(latent_forced_z_episode_frac=0.45)
        for step in (0, 100_000, 500_000, 999_999, 10_000_000):
            self.assertAlmostEqual(
                resolve_latent_forced_z_frac(cfg2, global_step=step), 0.45
            )

    def test_partial_schedule_falls_back_to_legacy(self) -> None:
        """If any of the four anneal fields is None, the resolver must use the legacy constant."""
        cfg = PPOConfig(
            latent_forced_z_episode_frac=0.20,
            latent_forced_z_episode_frac_start=0.30,
            latent_forced_z_episode_frac_end=0.00,
            latent_forced_z_anneal_start=200_000,
            latent_forced_z_anneal_end=None,
        )
        self.assertAlmostEqual(
            resolve_latent_forced_z_frac(cfg, global_step=300_000), 0.20
        )

    def test_full_schedule_anneal_values(self) -> None:
        """v5i3's documented 0.30 -> 0.00 across 200k -> 500k."""
        cfg = PPOConfig(
            latent_forced_z_episode_frac=0.30,
            latent_forced_z_episode_frac_start=0.30,
            latent_forced_z_episode_frac_end=0.00,
            latent_forced_z_anneal_start=200_000,
            latent_forced_z_anneal_end=500_000,
        )
        self.assertAlmostEqual(
            resolve_latent_forced_z_frac(cfg, global_step=0), 0.30
        )
        self.assertAlmostEqual(
            resolve_latent_forced_z_frac(cfg, global_step=199_999), 0.30
        )
        self.assertAlmostEqual(
            resolve_latent_forced_z_frac(cfg, global_step=200_000), 0.30
        )
        self.assertAlmostEqual(
            resolve_latent_forced_z_frac(cfg, global_step=350_000), 0.15
        )
        self.assertAlmostEqual(
            resolve_latent_forced_z_frac(cfg, global_step=500_000), 0.00
        )
        self.assertAlmostEqual(
            resolve_latent_forced_z_frac(cfg, global_step=1_000_000), 0.00
        )

    def test_resolver_clamps_to_unit_interval(self) -> None:
        """Mis-configured schedules must not produce probabilities > 1 or < 0."""
        cfg = PPOConfig(
            latent_forced_z_episode_frac_start=1.5,
            latent_forced_z_episode_frac_end=-0.5,
            latent_forced_z_anneal_start=0,
            latent_forced_z_anneal_end=100,
        )
        for step in (-10, 0, 25, 50, 75, 100, 200):
            value = resolve_latent_forced_z_frac(cfg, global_step=step)
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)

    def test_resume_uses_passed_global_step_not_internal_state(self) -> None:
        """Resume safety: re-entering at mid-anneal step picks up correct fraction."""
        cfg = PPOConfig(
            latent_forced_z_episode_frac=0.30,
            latent_forced_z_episode_frac_start=0.30,
            latent_forced_z_episode_frac_end=0.00,
            latent_forced_z_anneal_start=200_000,
            latent_forced_z_anneal_end=500_000,
        )
        # Imagine a checkpoint restores trainer.global_step = 350_000 mid-run.
        # The resolver must return the schedule's mid-anneal value, NOT the
        # start value, because it has no hidden "anneal start time" state.
        resumed = resolve_latent_forced_z_frac(cfg, global_step=350_000)
        self.assertAlmostEqual(resumed, 0.15)

        # And re-issuing the same call must be deterministic (no hidden state).
        for _ in range(3):
            self.assertAlmostEqual(
                resolve_latent_forced_z_frac(cfg, global_step=350_000), 0.15
            )


class V5i3PresetTests(unittest.TestCase):
    def test_v5i3_inherits_v5i2_and_sets_anneal_window(self) -> None:
        cfg = PPOConfig()
        apply_preset(cfg, "v5i3")

        # v5i2 traits inherited unchanged.
        self.assertTrue(cfg.enable_actor_z_film)
        self.assertEqual(cfg.actor_z_film_layer, 2)
        self.assertAlmostEqual(cfg.actor_z_film_init_scale, 0.02)
        self.assertTrue(cfg.latent_episode_strategy_ppo)
        self.assertAlmostEqual(cfg.latent_episode_strategy_lr, 5e-3)

        # New v5i3 anneal fields.
        self.assertAlmostEqual(cfg.latent_forced_z_episode_frac_start, 0.30)
        self.assertAlmostEqual(cfg.latent_forced_z_episode_frac_end, 0.00)
        self.assertEqual(cfg.latent_forced_z_anneal_start, 200_000)
        self.assertEqual(cfg.latent_forced_z_anneal_end, 500_000)
        # Safety: legacy field set to start value so anything that reads it
        # directly (bypassing the resolver) still sees a sane warmup value.
        self.assertAlmostEqual(cfg.latent_forced_z_episode_frac, 0.30)

        # Resolver agrees with the schedule at key steps.
        self.assertAlmostEqual(
            resolve_latent_forced_z_frac(cfg, global_step=0), 0.30
        )
        self.assertAlmostEqual(
            resolve_latent_forced_z_frac(cfg, global_step=350_000), 0.15
        )
        self.assertAlmostEqual(
            resolve_latent_forced_z_frac(cfg, global_step=600_000), 0.00
        )

        # Forbidden channels stay off.
        self.assertAlmostEqual(cfg.latent_behavior_contrast_coef, 0.0)
        self.assertAlmostEqual(cfg.latent_actor_z_separation_coef, 0.0)
        self.assertAlmostEqual(cfg.latent_preference_coef, 0.0)
        self.assertFalse(cfg.latent_awrd_enabled)
        self.assertFalse(cfg.latent_v3i3_event_preference_enabled)
        self.assertFalse(cfg.latent_router_distill_enabled)
        self.assertFalse(cfg.latent_specialist_router_enabled)

        self.assertEqual(
            cfg.run_tag, "v5i3_balanced_warmup_OP5_OP6_OP7_2m_4v4"
        )

    def test_v5i2_unchanged_by_v5i3_introduction(self) -> None:
        """Zero-config: v5i2 still resolves to 0 forced fraction at every step."""
        cfg = PPOConfig()
        apply_preset(cfg, "v5i2")

        self.assertIsNone(
            getattr(cfg, "latent_forced_z_episode_frac_start", None)
        )
        self.assertIsNone(
            getattr(cfg, "latent_forced_z_episode_frac_end", None)
        )
        self.assertAlmostEqual(cfg.latent_forced_z_episode_frac, 0.0)
        for step in (0, 100_000, 500_000, 1_000_000):
            self.assertAlmostEqual(
                resolve_latent_forced_z_frac(cfg, global_step=step), 0.0
            )


class ForcedZRuntimeRoutingTests(unittest.TestCase):
    """Pin the v5i3 forced-z runtime contract via the lightweight trainer harness.

    Contract under test:
    * The fraction used at episode start comes from ``resolve_latent_forced_z_frac``.
    * Forced episodes never set ``episode_strategy_has_start`` (so they cannot
      contribute to ``rollout_strategy_episode_records`` or to q_phi's PPO update).
    * With forced fraction = 0 the legacy router-only behavior is reproduced.
    """

    def _state(self, n_envs: int, gs_dim: int) -> torch.Tensor:
        return torch.zeros((n_envs, gs_dim), dtype=torch.float32)

    def test_forced_episodes_never_set_episode_strategy_has_start(self) -> None:
        trainer = _make_trainer(n_envs=4, warmup=0, episode_credit=True, gs_dim=4)
        # behavior_contrast must be wired in or strategy_for_step will
        # short-circuit the forced-z path (see contrast_on check at ~line 1875).
        trainer.latent_behavior_contrast = SimpleNamespace()
        trainer.cfg.latent_forced_z_episode_frac_start = 1.0
        trainer.cfg.latent_forced_z_episode_frac_end = 1.0
        trainer.cfg.latent_forced_z_anneal_start = 0
        trainer.cfg.latent_forced_z_anneal_end = 1_000_000
        trainer.latent_behavior_contrast_coef = 0.05
        trainer.latent_behavior_contrast_anneal_after_steps = 0
        trainer.latent_behavior_contrast_anneal_to = 0.0

        latent_state = LatentStrategyState(trainer)
        latent_state.reset()

        _, _, aux = latent_state.strategy_for_step(self._state(4, 4))

        # Every env should be flagged as forced; none should snapshot start.
        self.assertTrue(bool(aux["z_forced"].all().item()))
        self.assertFalse(bool(latent_state.episode_strategy_has_start.any().item()))

    def test_zero_fraction_disables_forcing(self) -> None:
        trainer = _make_trainer(n_envs=4, warmup=0, episode_credit=True, gs_dim=4)
        trainer.latent_behavior_contrast = SimpleNamespace()
        # All schedule fields None and constant = 0.0 (default cfg shim) -->
        # the resolver returns 0.0 --> forced flag stays False on every env.
        trainer.latent_behavior_contrast_coef = 0.05
        trainer.latent_behavior_contrast_anneal_after_steps = 0
        trainer.latent_behavior_contrast_anneal_to = 0.0

        latent_state = LatentStrategyState(trainer)
        latent_state.reset()

        _, _, aux = latent_state.strategy_for_step(self._state(4, 4))

        self.assertFalse(bool(aux["z_forced"].any().item()))

    def test_resume_at_mid_anneal_resolves_correctly(self) -> None:
        """Trainer.global_step restoration drives the schedule deterministically."""
        trainer = _make_trainer(n_envs=2, warmup=0, episode_credit=True, gs_dim=4)
        trainer.latent_behavior_contrast = SimpleNamespace()
        trainer.cfg.latent_forced_z_episode_frac_start = 0.30
        trainer.cfg.latent_forced_z_episode_frac_end = 0.00
        trainer.cfg.latent_forced_z_anneal_start = 200_000
        trainer.cfg.latent_forced_z_anneal_end = 500_000
        trainer.latent_behavior_contrast_coef = 0.05
        trainer.latent_behavior_contrast_anneal_after_steps = 0
        trainer.latent_behavior_contrast_anneal_to = 0.0

        # Simulate three resume points and verify the resolver tracks global_step.
        for step, expected in (
            (0, 0.30),
            (350_000, 0.15),
            (600_000, 0.00),
        ):
            trainer.global_step = step
            self.assertAlmostEqual(
                resolve_latent_forced_z_frac(trainer.cfg, global_step=trainer.global_step),
                expected,
            )


class PerZRouterTelemetryTests(unittest.TestCase):
    """Per-z router telemetry must appear in the metrics CSV fieldnames."""

    def test_per_z_fields_registered_for_latent_runs(self) -> None:
        fields = _update_fieldnames(use_latent_strategy=True, latent_k=4)
        for z in range(4):
            self.assertIn(f"router_sample_count_by_z_{z}", fields)
            self.assertIn(f"forced_sample_count_by_z_{z}", fields)
            self.assertIn(f"episode_count_by_z_{z}", fields)
            self.assertIn(f"mean_episode_advantage_by_z_{z}", fields)
            self.assertIn(f"std_episode_advantage_by_z_{z}", fields)
            self.assertIn(f"mean_return_by_z_{z}", fields)
            self.assertIn(f"mean_logprob_ratio_by_z_{z}", fields)
            self.assertIn(f"clip_fraction_by_z_{z}", fields)
        self.assertIn("latent_forced_z_episode_frac_current", fields)

    def test_per_z_fields_absent_for_non_latent_runs(self) -> None:
        fields = _update_fieldnames(use_latent_strategy=False, latent_k=4)
        for z in range(4):
            self.assertNotIn(f"router_sample_count_by_z_{z}", fields)
            self.assertNotIn(f"forced_sample_count_by_z_{z}", fields)


if __name__ == "__main__":
    unittest.main()
