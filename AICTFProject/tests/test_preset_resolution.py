"""Resolve every preset in the registry and verify it matches the saved snapshot.

The snapshot at ``tests/preset_snapshots.json`` is the source of truth for
training behavior across all named presets. Any code change that intentionally
shifts a preset's resolved config must regenerate the snapshot via::

    python tools/snapshot_presets.py

If this test fails after a change you did not intend to make to training
recipes, you have probably broken a preset. Do **not** blindly regenerate
the snapshot.
"""
from __future__ import annotations

import json
import os
import unittest
from dataclasses import asdict
from typing import Any

from rl.presets import PRESET_REGISTRY, apply_preset
from rl.train_ppo import PPOConfig

_HERE = os.path.dirname(os.path.abspath(__file__))
SNAPSHOT_PATH = os.path.join(_HERE, "preset_snapshots.json")


def _resolve_preset_to_dict(key: str) -> dict[str, Any]:
    """Apply a preset to a fresh ``PPOConfig`` and return a JSON-safe dict."""
    cfg = PPOConfig()
    apply_preset(cfg, key)
    cfg_dict = asdict(cfg)
    if isinstance(cfg_dict.get("opponent_pool"), tuple):
        cfg_dict["opponent_pool"] = list(cfg_dict["opponent_pool"])
    if isinstance(cfg_dict.get("opponent_pool_weights"), tuple):
        cfg_dict["opponent_pool_weights"] = list(cfg_dict["opponent_pool_weights"])
    return cfg_dict


def resolve_all_presets() -> dict[str, dict[str, Any]]:
    """Resolve every preset key in :data:`PRESET_REGISTRY` to a JSON-safe dict.

    Exported so ``tools/snapshot_presets.py`` can reuse the exact same
    resolution path the test uses; the two cannot drift.
    """
    return {key: _resolve_preset_to_dict(key) for key in sorted(PRESET_REGISTRY.keys())}


class PresetResolutionTests(unittest.TestCase):
    def test_every_registered_preset_resolves(self) -> None:
        """Smoke test: every preset key applies cleanly to a fresh PPOConfig."""
        for key in sorted(PRESET_REGISTRY.keys()):
            with self.subTest(preset=key):
                cfg = PPOConfig()
                try:
                    apply_preset(cfg, key)
                except Exception as exc:
                    self.fail(f"preset {key!r} failed to resolve: {exc!r}")
                self.assertTrue(
                    isinstance(cfg.run_tag, str) and cfg.run_tag.strip(),
                    f"preset {key!r} left run_tag empty",
                )

    def test_resolved_configs_match_snapshot(self) -> None:
        """Resolved PPOConfig must exactly match the committed snapshot."""
        if not os.path.isfile(SNAPSHOT_PATH):
            self.fail(
                f"preset snapshot missing at {SNAPSHOT_PATH!r}. "
                "Regenerate it intentionally with: "
                "python tools/snapshot_presets.py"
            )

        resolved = resolve_all_presets()
        with open(SNAPSHOT_PATH, "r", encoding="utf-8") as f:
            snapshot = json.load(f)

        missing_in_snapshot = sorted(set(resolved.keys()) - set(snapshot.keys()))
        extra_in_snapshot = sorted(set(snapshot.keys()) - set(resolved.keys()))
        self.assertFalse(
            missing_in_snapshot,
            f"presets added without snapshot regen: {missing_in_snapshot}. "
            "Run: python tools/snapshot_presets.py",
        )
        self.assertFalse(
            extra_in_snapshot,
            f"snapshot contains stale presets no longer in registry: {extra_in_snapshot}. "
            "Run: python tools/snapshot_presets.py",
        )

        for key in sorted(resolved.keys()):
            with self.subTest(preset=key):
                self.assertEqual(
                    resolved[key],
                    snapshot[key],
                    f"preset {key!r} resolved config differs from snapshot. "
                    "If this change is intentional, run: python tools/snapshot_presets.py",
                )

    def test_only_episode_credit_presets_enable_episode_strategy_ppo(self) -> None:
        """Old presets must keep episode-level q_phi PPO disabled by default."""
        episode_credit_presets = {
            "plan_faithful_latent_option_a_episode_credit",
            "latent_option_a_episode_credit",
            "plan_faithful_latent_episode_credit",
            "plan_faithful_latent_episode_strategic",
            "latent_episode_strategic",
            "plan_faithful_latent_intent_credit",
            # v3b inherits from episode_strategic and only flips the q_phi advantage
            # baseline from "z-conditioned V" to "policy-marginal V". Episode-credit
            # stays ON; the marginal baseline lives inside apply_episode_strategy_ppo.
            "plan_faithful_latent_v3b_marginal",
            "latent_v3b_marginal",
            # v3c inherits from v3b and only changes router update strength
            # (n_epochs=6, dedicated LR=5e-3) + entropy floor 0.001. Episode-credit
            # path stays ON.
            "plan_faithful_latent_v3c_router_lr",
            "latent_v3c_router_lr",
            "plan_faithful_latent_v3c",
            "latent_v3c",
            # v3d inherits from v3c and only swaps the q_phi baseline source
            # (V-marginal -> empirical per-opponent bucket mean). Episode-credit
            # path stays ON.
            "plan_faithful_latent_v3d_smart_router",
            "latent_v3d_smart_router",
            "plan_faithful_latent_v3d",
            "latent_v3d",
            "plan_faithful_latent_v3d_delay",
            "latent_v3d_delay",
            "plan_faithful_latent_v3d_delayed_anneal",
            "latent_v3d_delayed_anneal",
            "plan_faithful_latent_v3e_strong_z_actor",
            "latent_v3e_strong_z_actor",
            "plan_faithful_latent_v3e",
            "latent_v3e",
            "plan_faithful_latent_v3f_behavior_contrast",
            "latent_v3f_behavior_contrast",
            "plan_faithful_latent_v3f",
            "latent_v3f",
            "plan_faithful_latent_v3g_preference",
            "latent_v3g_preference",
            "plan_faithful_latent_v3g",
            "latent_v3g",
            "plan_faithful_latent_v3h_balanced_preference",
            "latent_v3h_balanced_preference",
            "plan_faithful_latent_v3h",
            "latent_v3h",
            "plan_faithful_latent_v3h2_balanced_preference",
            "latent_v3h2_balanced_preference",
            "plan_faithful_latent_v3h2",
            "latent_v3h2",
            "plan_faithful_latent_v3i_event_refresh",
            "latent_v3i_event_refresh",
            "plan_faithful_latent_v3i",
            "latent_v3i",
            "plan_faithful_latent_v3i2_router_signal",
            "latent_v3i2_router_signal",
            "plan_faithful_latent_v3i2",
            "latent_v3i2",
            "plan_faithful_latent_v3i3_event_conditioned_preference",
            "latent_v3i3_event_conditioned_preference",
            "plan_faithful_latent_v3i3",
            "latent_v3i3",
            "plan_faithful_latent_v3i4_event_progress_preference",
            "latent_v3i4_event_progress_preference",
            "plan_faithful_latent_v3i4",
            "latent_v3i4",
            "plan_faithful_latent_v3i5_crisp_router",
            "latent_v3i5_crisp_router",
            "plan_faithful_latent_v3i5",
            "latent_v3i5",
            "plan_faithful_latent_v3i6_stronger_actor_contrast",
            "latent_v3i6_stronger_actor_contrast",
            "plan_faithful_latent_v3i6",
            "latent_v3i6",
            "plan_faithful_latent_v3i7_advantage_weighted_router_distill",
            "latent_v3i7_advantage_weighted_router_distill",
            "plan_faithful_latent_v3i7",
            "latent_v3i7",
            "plan_faithful_latent_v3i8_commander_lockin",
            "latent_v3i8_commander_lockin",
            "plan_faithful_latent_v3i8",
            "latent_v3i8",
            "plan_faithful_latent_v3i9_specialist_router",
            "latent_v3i9_specialist_router",
            "plan_faithful_latent_v3i9_context_specialist",
            "latent_v3i9_context_specialist",
            "plan_faithful_latent_v3i9",
            "latent_v3i9",
            "plan_faithful_latent_v3i10_role_phase_specialist",
            "latent_v3i10_role_phase_specialist",
            "plan_faithful_latent_v3i10",
            "latent_v3i10",
            "plan_faithful_latent_v3i11_z_reactive_actor_adapters",
            "latent_v3i11_z_reactive_actor_adapters",
            "plan_faithful_latent_v3i11",
            "latent_v3i11",
            "plan_faithful_latent_v3i12_faithful_z_pressure",
            "latent_v3i12_faithful_z_pressure",
            "plan_faithful_latent_v3i12",
            "latent_v3i12",
            "plan_faithful_latent_v3i13_strict_faithful_z",
            "latent_v3i13_strict_faithful_z",
            "plan_faithful_latent_v3i13",
            "latent_v3i13",
            "plan_faithful_latent_v3i14_specialized_faithful_z",
            "latent_v3i14_specialized_faithful_z",
            "plan_faithful_latent_v3i14",
            "latent_v3i14",
            "plan_faithful_latent_v3i14_tuned",
            "latent_v3i14_tuned",
            "latent_v3i14b",
            "latent_v3i14_tactical_specialist_tuned",
        }
        resolved = resolve_all_presets()
        for key, cfg in resolved.items():
            with self.subTest(preset=key):
                if key in episode_credit_presets:
                    self.assertTrue(cfg["latent_episode_strategy_ppo"])
                    self.assertTrue(cfg["latent_episode_strategy_coef"] in {0.25, 0.30})
                else:
                    self.assertFalse(cfg["latent_episode_strategy_ppo"])
                    self.assertAlmostEqual(cfg["latent_episode_strategy_coef"], 0.0)

    def test_plan_faithful_latent_step6_config(self) -> None:
        """Verify that step 6 has the expected option-style q_phi advantage and warmup configuration."""
        resolved = resolve_all_presets()
        for key in ("plan_faithful_latent_step6", "latent_step6"):
            with self.subTest(preset=key):
                cfg = resolved[key]
                self.assertTrue(cfg["latent_q_phi_option_advantage"])
                self.assertFalse(cfg["latent_episode_strategy_ppo"])
                self.assertAlmostEqual(cfg["latent_strategy_ppo_coef"], 0.40)
                self.assertEqual(cfg["latent_episode_strategy_warmup_decision_steps"], 5)
                self.assertAlmostEqual(cfg["latent_lam_h"], 0.003)
                self.assertAlmostEqual(cfg["latent_lam_h_start"], 0.003)
                self.assertAlmostEqual(cfg["latent_lam_h_end"], 0.0005)
                self.assertEqual(cfg["latent_entropy_anneal_start"], 200_000)
                self.assertEqual(cfg["latent_entropy_anneal_end"], 700_000)
                self.assertAlmostEqual(cfg["latent_lam_p"], 0.0)

    def test_latent_v3f_behavior_contrast_is_summer_faithful(self) -> None:
        """v3f separates options without supervised router labels or semantic z roles."""
        resolved = resolve_all_presets()
        for key in (
            "plan_faithful_latent_v3f_behavior_contrast",
            "latent_v3f_behavior_contrast",
            "plan_faithful_latent_v3f",
            "latent_v3f",
        ):
            with self.subTest(preset=key):
                cfg = resolved[key]
                self.assertTrue(cfg["use_latent_strategy"])
                self.assertTrue(cfg["latent_episode_strategy_ppo"])
                self.assertEqual(cfg["latent_resample_every_n"], 0)
                self.assertAlmostEqual(cfg["latent_forced_z_episode_frac"], 0.30)
                self.assertAlmostEqual(cfg["latent_behavior_contrast_coef"], 0.05)
                self.assertEqual(cfg["latent_behavior_contrast_anneal_after_steps"], 800_000)
                self.assertAlmostEqual(cfg["latent_behavior_contrast_anneal_to"], 0.005)
                self.assertAlmostEqual(cfg["latent_usage_balance_coef"], 0.01)
                self.assertEqual(cfg["latent_q_phi_train_after_steps"], 100_000)
                self.assertFalse(cfg["latent_strategy_aux_return_head"])
                self.assertAlmostEqual(cfg["latent_strategy_aux_return_coef"], 0.0)
                self.assertAlmostEqual(cfg["latent_strategy_aux_predict_phase_coef"], 0.0)
                self.assertFalse(cfg["fixed_latent_strategy"])

    def test_latent_v3g_preference_is_summer_faithful(self) -> None:
        """v3g is faithful to the Summer spec and has correct preference hyperparams."""
        resolved = resolve_all_presets()
        for key in (
            "plan_faithful_latent_v3g_preference",
            "latent_v3g_preference",
            "plan_faithful_latent_v3g",
            "latent_v3g",
        ):
            with self.subTest(preset=key):
                cfg = resolved[key]
                self.assertTrue(cfg["use_latent_strategy"])
                self.assertTrue(cfg["latent_episode_strategy_ppo"])
                self.assertAlmostEqual(cfg["latent_preference_coef"], 0.03)
                self.assertAlmostEqual(cfg["latent_preference_temperature"], 0.75)
                self.assertEqual(cfg["latent_preference_min_bucket_count"], 8)
                self.assertEqual(cfg["latent_preference_min_distinct_z"], 2)
                self.assertFalse(cfg.get("latent_preference_opponent_balanced", False))
                self.assertFalse(cfg.get("latent_preference_log_opponent_targets", False))

    def test_latent_v3i3_event_conditioned_preference_is_summer_faithful(self) -> None:
        """v3i3 inherits v3i2 + enables event-conditioned preference & refresh log.

        Plan-faithful invariants:
            * event_refresh stays ON (the audible system v3i3 trains on)
            * episode-credit path stays ON (v3i3 hooks into the same update)
            * latent_v3i3_event_preference_enabled = True with coef > 0
            * latent_v3i3_refresh_log_enabled = True (proof-layer log mandatory)
            * No new policy inputs / role labels / scripted z assignments
              (v3i3 only changes the preference *target* keying, the actor's
              input remains pi(z | state)).
        """
        resolved = resolve_all_presets()
        for key in (
            "plan_faithful_latent_v3i3_event_conditioned_preference",
            "latent_v3i3_event_conditioned_preference",
            "plan_faithful_latent_v3i3",
            "latent_v3i3",
        ):
            with self.subTest(preset=key):
                cfg = resolved[key]
                self.assertTrue(cfg["use_latent_strategy"])
                self.assertTrue(cfg["latent_episode_strategy_ppo"])
                self.assertTrue(cfg["latent_event_refresh_enabled"])
                self.assertTrue(cfg["latent_v3i3_event_preference_enabled"])
                self.assertGreater(cfg["latent_v3i3_event_preference_coef"], 0.0)
                self.assertTrue(cfg["latent_v3i3_refresh_log_enabled"])
                self.assertEqual(cfg["latent_v3i3_event_preference_min_bucket_count"], 4)
                self.assertEqual(cfg["latent_v3i3_event_preference_min_distinct_z"], 2)
                self.assertGreaterEqual(cfg["latent_v3i3_event_preference_buffer_size"], 1000)
                self.assertFalse(cfg["fixed_latent_strategy"])
                self.assertFalse(cfg.get("latent_event_refresh_force_roles", False))

    def test_latent_v3h_preference_balanced_is_summer_faithful(self) -> None:
        """v3h has correct opponent balanced preference distillation config and target telemetry."""
        resolved = resolve_all_presets()
        for key in (
            "plan_faithful_latent_v3h_balanced_preference",
            "latent_v3h_balanced_preference",
            "plan_faithful_latent_v3h",
            "latent_v3h",
        ):
            with self.subTest(preset=key):
                cfg = resolved[key]
                self.assertTrue(cfg["use_latent_strategy"])
                self.assertTrue(cfg["latent_episode_strategy_ppo"])
                self.assertAlmostEqual(cfg["latent_preference_coef"], 0.03)
                self.assertTrue(cfg.get("latent_preference_opponent_balanced", False))
                self.assertTrue(cfg.get("latent_preference_log_opponent_targets", False))

    def test_latent_v3i4_event_progress_preference_is_summer_faithful(self) -> None:
        """v3i4 inherits v3i3 + sets key_mode to event_flag_progress and warmup steps to 50_000."""
        resolved = resolve_all_presets()
        for key in (
            "plan_faithful_latent_v3i4_event_progress_preference",
            "latent_v3i4_event_progress_preference",
            "plan_faithful_latent_v3i4",
            "latent_v3i4",
        ):
            with self.subTest(preset=key):
                cfg = resolved[key]
                self.assertTrue(cfg["use_latent_strategy"])
                self.assertTrue(cfg["latent_episode_strategy_ppo"])
                self.assertTrue(cfg["latent_event_refresh_enabled"])
                self.assertTrue(cfg["latent_v3i3_event_preference_enabled"])
                self.assertEqual(cfg["latent_event_preference_key_mode"], "event_flag_progress")
                self.assertTrue(cfg["latent_v3i3_event_preference_normalize"])
                self.assertEqual(cfg["latent_v3i3_event_preference_warmup_steps"], 50_000)
                self.assertFalse(cfg["fixed_latent_strategy"])

    def test_latent_v3i5_crisp_router_is_summer_faithful(self) -> None:
        """v3i5 inherits v3i4 + removes entropy pressure on router, reduces usage balance, sharpens event preference temperature and coef."""
        resolved = resolve_all_presets()
        for key in (
            "plan_faithful_latent_v3i5_crisp_router",
            "latent_v3i5_crisp_router",
            "plan_faithful_latent_v3i5",
            "latent_v3i5",
        ):
            with self.subTest(preset=key):
                cfg = resolved[key]
                self.assertTrue(cfg["use_latent_strategy"])
                self.assertTrue(cfg["latent_episode_strategy_ppo"])
                self.assertTrue(cfg["latent_event_refresh_enabled"])
                self.assertTrue(cfg["latent_v3i3_event_preference_enabled"])
                self.assertEqual(cfg["latent_event_preference_key_mode"], "event_flag_progress")
                self.assertTrue(cfg["latent_v3i3_event_preference_normalize"])
                self.assertEqual(cfg["latent_v3i3_event_preference_warmup_steps"], 50_000)
                self.assertEqual(cfg["latent_entropy_objective"], "none")
                self.assertAlmostEqual(cfg["latent_usage_balance_coef"], 0.05)
                self.assertAlmostEqual(cfg["latent_v3i3_event_preference_temperature"], 0.35)
                self.assertAlmostEqual(cfg["latent_v3i3_event_preference_coef"], 0.05)
                self.assertFalse(cfg["fixed_latent_strategy"])

    def test_latent_v3i6_stronger_actor_contrast_is_summer_faithful(self) -> None:
        """v3i6 inherits v3i4 + sets behavior contrast coef to 0.10 and margin to 0.35, keeps latent_entropy_objective maximize."""
        resolved = resolve_all_presets()
        for key in (
            "plan_faithful_latent_v3i6_stronger_actor_contrast",
            "latent_v3i6_stronger_actor_contrast",
            "plan_faithful_latent_v3i6",
            "latent_v3i6",
        ):
            with self.subTest(preset=key):
                cfg = resolved[key]
                self.assertTrue(cfg["use_latent_strategy"])
                self.assertTrue(cfg["latent_episode_strategy_ppo"])
                self.assertTrue(cfg["latent_event_refresh_enabled"])
                self.assertTrue(cfg["latent_v3i3_event_preference_enabled"])
                self.assertEqual(cfg["latent_event_preference_key_mode"], "event_flag_progress")
                self.assertTrue(cfg["latent_v3i3_event_preference_normalize"])
                self.assertEqual(cfg["latent_v3i3_event_preference_warmup_steps"], 50_000)
                self.assertEqual(cfg["latent_entropy_objective"], "maximize")
                self.assertAlmostEqual(cfg["latent_behavior_contrast_coef"], 0.10)
                self.assertAlmostEqual(cfg["latent_behavior_contrast_margin"], 0.35)
                self.assertFalse(cfg["fixed_latent_strategy"])

    def test_latent_v3i7_advantage_weighted_router_distill_is_summer_faithful(self) -> None:
        """v3i7 inherits v3i6 + adds margin-gated advantage-weighted router distillation."""
        resolved = resolve_all_presets()
        for key in (
            "plan_faithful_latent_v3i7_advantage_weighted_router_distill",
            "latent_v3i7_advantage_weighted_router_distill",
            "plan_faithful_latent_v3i7",
            "latent_v3i7",
        ):
            with self.subTest(preset=key):
                cfg = resolved[key]
                self.assertTrue(cfg["use_latent_strategy"])
                self.assertTrue(cfg["latent_episode_strategy_ppo"])
                self.assertTrue(cfg["latent_event_refresh_enabled"])
                self.assertTrue(cfg["latent_v3i3_event_preference_enabled"])
                self.assertEqual(cfg["latent_event_preference_key_mode"], "event_flag_progress")
                self.assertTrue(cfg["latent_v3i3_event_preference_normalize"])
                self.assertAlmostEqual(cfg["latent_behavior_contrast_coef"], 0.10)
                self.assertAlmostEqual(cfg["latent_behavior_contrast_margin"], 0.35)
                self.assertTrue(cfg["latent_awrd_enabled"])
                self.assertAlmostEqual(cfg["latent_awrd_coef"], 0.04)
                self.assertAlmostEqual(cfg["latent_awrd_temperature"], 0.35)
                self.assertEqual(cfg["latent_awrd_min_bucket_count"], 8)
                self.assertEqual(cfg["latent_awrd_min_distinct_z"], 2)
                self.assertAlmostEqual(cfg["latent_awrd_margin_threshold"], 0.15)
                self.assertAlmostEqual(cfg["latent_awrd_margin_scale"], 2.0)
                self.assertFalse(cfg["fixed_latent_strategy"])

    def test_latent_v3i8_commander_lockin_is_summer_faithful(self) -> None:
        """v3i8 inherits v3i4 + adds custom AWRD router bridge config and soft margin gating."""
        resolved = resolve_all_presets()
        for key in (
            "plan_faithful_latent_v3i8_commander_lockin",
            "latent_v3i8_commander_lockin",
            "plan_faithful_latent_v3i8",
            "latent_v3i8",
        ):
            with self.subTest(preset=key):
                cfg = resolved[key]
                self.assertTrue(cfg["use_latent_strategy"])
                self.assertTrue(cfg["latent_episode_strategy_ppo"])
                self.assertTrue(cfg["latent_event_refresh_enabled"])
                self.assertTrue(cfg["latent_v3i3_event_preference_enabled"])
                self.assertEqual(cfg["latent_event_preference_key_mode"], "event_flag_progress")
                self.assertTrue(cfg["latent_v3i3_event_preference_normalize"])
                self.assertAlmostEqual(cfg["latent_behavior_contrast_coef"], 0.05)
                self.assertAlmostEqual(cfg["latent_behavior_contrast_margin"], 0.25)
                self.assertTrue(cfg["latent_awrd_enabled"])
                self.assertAlmostEqual(cfg["latent_awrd_coef"], 0.05)
                self.assertAlmostEqual(cfg["latent_awrd_temperature"], 0.35)
                self.assertAlmostEqual(cfg["latent_awrd_min_margin"], 0.08)
                self.assertAlmostEqual(cfg["latent_awrd_margin_scale"], 3.0)
                self.assertTrue(cfg["latent_awrd_soft_margin_gating"])
                self.assertFalse(cfg["fixed_latent_strategy"])

    def test_latent_v3i9_specialist_router_is_summer_faithful(self) -> None:
        """v3i9 keeps global latent usage while making q_phi decisive inside context buckets."""
        resolved = resolve_all_presets()
        for key in (
            "plan_faithful_latent_v3i9_specialist_router",
            "latent_v3i9_specialist_router",
            "plan_faithful_latent_v3i9_context_specialist",
            "latent_v3i9_context_specialist",
            "plan_faithful_latent_v3i9",
            "latent_v3i9",
        ):
            with self.subTest(preset=key):
                cfg = resolved[key]
                self.assertTrue(cfg["use_latent_strategy"])
                self.assertTrue(cfg["latent_episode_strategy_ppo"])
                self.assertTrue(cfg["latent_event_refresh_enabled"])
                self.assertTrue(cfg["latent_v3i3_event_preference_enabled"])
                self.assertEqual(cfg["latent_event_preference_key_mode"], "event_flag_progress")
                self.assertTrue(cfg["latent_v3i3_event_preference_normalize"])
                self.assertTrue(cfg["latent_awrd_enabled"])
                self.assertAlmostEqual(cfg["latent_awrd_coef"], 0.06)
                self.assertTrue(cfg["latent_awrd_soft_margin_gating"])
                self.assertTrue(cfg["latent_specialist_router_enabled"])
                self.assertAlmostEqual(cfg["latent_marginal_balance_coef"], 0.02)
                self.assertAlmostEqual(cfg["latent_conditional_entropy_min_coef"], 0.015)
                self.assertAlmostEqual(cfg["latent_context_mi_coef"], 0.04)
                self.assertEqual(cfg["latent_specialist_warmup_steps"], 100_000)
                self.assertEqual(cfg["latent_specialist_ramp_steps"], 400_000)
                self.assertEqual(cfg["latent_specialist_min_bucket_count"], 4)
                self.assertEqual(cfg["latent_entropy_anneal_start"], 100_000)
                self.assertEqual(cfg["latent_entropy_anneal_end"], 500_000)
                self.assertAlmostEqual(cfg["latent_lam_h_end"], 0.0003)
                self.assertFalse(cfg["fixed_latent_strategy"])

    def test_latent_v3i10_role_phase_specialist_is_summer_faithful(self) -> None:
        """v3i10 specializes by phase/flag/progress with opponent as secondary context."""
        resolved = resolve_all_presets()
        for key in (
            "plan_faithful_latent_v3i10_role_phase_specialist",
            "latent_v3i10_role_phase_specialist",
            "plan_faithful_latent_v3i10",
            "latent_v3i10",
        ):
            with self.subTest(preset=key):
                cfg = resolved[key]
                self.assertTrue(cfg["use_latent_strategy"])
                self.assertTrue(cfg["latent_episode_strategy_ppo"])
                self.assertEqual(cfg["mode"], "FIXED_OPPONENT")
                self.assertEqual(cfg["fixed_opponent_tag"], "OP3")
                self.assertTrue(cfg["latent_specialist_router_enabled"])
                self.assertEqual(
                    cfg["latent_specialist_context_key_mode"],
                    "role_phase_progress_opponent",
                )
                self.assertAlmostEqual(cfg["latent_marginal_balance_coef"], 0.015)
                self.assertAlmostEqual(cfg["latent_conditional_entropy_min_coef"], 0.035)
                self.assertAlmostEqual(cfg["latent_context_mi_coef"], 0.08)
                self.assertEqual(cfg["latent_specialist_min_bucket_count"], 3)
                self.assertEqual(cfg["latent_event_refresh_min_gap_steps"], 80)
                self.assertEqual(cfg["latent_event_refresh_max_per_episode"], 1)
                self.assertAlmostEqual(cfg["latent_behavior_contrast_coef"], 0.12)
                self.assertAlmostEqual(cfg["latent_behavior_contrast_margin"], 0.35)
                self.assertAlmostEqual(cfg["latent_forced_z_episode_frac"], 0.40)
                self.assertFalse(cfg["fixed_latent_strategy"])

    def test_latent_v3i11_z_reactive_actor_adapters_is_summer_faithful(self) -> None:
        """v3i11 keeps v3i10 routing and makes z directly bend actor logits."""
        resolved = resolve_all_presets()
        for key in (
            "plan_faithful_latent_v3i11_z_reactive_actor_adapters",
            "latent_v3i11_z_reactive_actor_adapters",
            "plan_faithful_latent_v3i11",
            "latent_v3i11",
        ):
            with self.subTest(preset=key):
                cfg = resolved[key]
                self.assertTrue(cfg["use_latent_strategy"])
                self.assertTrue(cfg["latent_episode_strategy_ppo"])
                self.assertEqual(cfg["mode"], "OPPONENT_POOL")
                self.assertTrue(cfg["opponent_randomize"])
                self.assertEqual(cfg["opponent_pool"], ["OP3", "OP5", "OP6"])
                self.assertTrue(cfg["latent_actor_z_adapter_enabled"])
                self.assertAlmostEqual(cfg["latent_actor_z_adapter_scale"], 0.35)
                self.assertAlmostEqual(cfg["latent_actor_z_adapter_init_std"], 0.03)
                self.assertEqual(
                    cfg["latent_specialist_context_key_mode"],
                    "role_phase_progress_opponent",
                )
                self.assertTrue(cfg["latent_specialist_router_enabled"])
                self.assertAlmostEqual(cfg["latent_usage_balance_coef"], 0.015)
                self.assertAlmostEqual(cfg["latent_conditional_entropy_min_coef"], 0.05)
                self.assertAlmostEqual(cfg["latent_context_mi_coef"], 0.12)
                self.assertAlmostEqual(cfg["latent_awrd_coef"], 0.10)
                self.assertEqual(cfg["latent_awrd_warmup_steps"], 100_000)
                self.assertEqual(cfg["latent_awrd_ramp_steps"], 250_000)
                self.assertAlmostEqual(cfg["latent_forced_z_episode_frac"], 0.45)
                self.assertFalse(cfg["fixed_latent_strategy"])

    def test_latent_v3i12_faithful_z_pressure_is_summer_faithful(self) -> None:
        """v3i12 makes z reactive through shared actor input pressure only."""
        resolved = resolve_all_presets()
        for key in (
            "plan_faithful_latent_v3i12_faithful_z_pressure",
            "latent_v3i12_faithful_z_pressure",
            "plan_faithful_latent_v3i12",
            "latent_v3i12",
        ):
            with self.subTest(preset=key):
                cfg = resolved[key]
                self.assertTrue(cfg["use_latent_strategy"])
                self.assertTrue(cfg["latent_episode_strategy_ppo"])
                self.assertEqual(cfg["mode"], "OPPONENT_POOL")
                self.assertTrue(cfg["opponent_randomize"])
                self.assertEqual(cfg["opponent_pool"], ["OP3", "OP5", "OP6"])
                self.assertTrue(cfg["latent_actor_z_onehot_enabled"])
                self.assertAlmostEqual(cfg["latent_actor_z_onehot_scale"], 1.0)
                self.assertAlmostEqual(cfg["latent_actor_z_embed_scale"], 1.25)
                self.assertFalse(cfg["latent_actor_z_adapter_enabled"])
                self.assertAlmostEqual(cfg["latent_actor_z_adapter_scale"], 0.0)
                self.assertAlmostEqual(cfg["latent_actor_z_separation_coef"], 0.015)
                self.assertAlmostEqual(cfg["latent_actor_z_separation_margin"], 0.02)
                self.assertEqual(
                    cfg["latent_specialist_context_key_mode"],
                    "role_phase_progress_opponent",
                )
                self.assertTrue(cfg["latent_specialist_router_enabled"])
                self.assertAlmostEqual(cfg["latent_usage_balance_coef"], 0.015)
                self.assertAlmostEqual(cfg["latent_marginal_balance_coef"], 0.02)
                self.assertAlmostEqual(cfg["latent_conditional_entropy_min_coef"], 0.05)
                self.assertAlmostEqual(cfg["latent_context_mi_coef"], 0.12)
                self.assertEqual(cfg["latent_specialist_warmup_steps"], 100_000)
                self.assertEqual(cfg["latent_specialist_ramp_steps"], 300_000)
                self.assertTrue(cfg["latent_awrd_enabled"])
                self.assertAlmostEqual(cfg["latent_awrd_coef"], 0.10)
                self.assertEqual(cfg["latent_awrd_warmup_steps"], 100_000)
                self.assertEqual(cfg["latent_awrd_ramp_steps"], 300_000)
                self.assertAlmostEqual(cfg["latent_forced_z_episode_frac"], 0.45)
                self.assertFalse(cfg["fixed_latent_strategy"])

    def test_latent_v3i14_specialized_faithful_z_owns_tactical_niches(self) -> None:
        resolved = resolve_all_presets()
        for key in (
            "plan_faithful_latent_v3i14_specialized_faithful_z",
            "latent_v3i14_specialized_faithful_z",
            "plan_faithful_latent_v3i14",
            "latent_v3i14",
        ):
            with self.subTest(preset=key):
                cfg = resolved[key]
                self.assertTrue(cfg["use_latent_strategy"])
                self.assertTrue(cfg["latent_episode_strategy_ppo"])
                self.assertEqual(cfg["mode"], "OPPONENT_POOL")
                self.assertEqual(cfg["opponent_pool"], ["OP3", "OP5", "OP6"])
                self.assertEqual(
                    cfg["opponent_pool_weights"],
                    [0.15, 0.40, 0.45],
                )
                self.assertEqual(
                    cfg["latent_q_phi_bucket_baseline"],
                    "tactical_context_opponent",
                )
                self.assertEqual(
                    cfg["latent_specialist_context_key_mode"],
                    "tactical_phase_flags_score_opponent",
                )
                self.assertEqual(
                    cfg["latent_specialist_conditional_entropy_scope"],
                    "context_bucket",
                )
                self.assertTrue(cfg["latent_specialist_use_rollout_states"])
                self.assertEqual(
                    cfg["latent_specialist_rollout_max_samples"], 8192
                )
                self.assertAlmostEqual(
                    cfg["latent_conditional_entropy_min_coef_start"], 0.01
                )
                self.assertAlmostEqual(
                    cfg["latent_conditional_entropy_min_coef"], 0.05
                )
                self.assertAlmostEqual(cfg["latent_marginal_balance_coef"], 0.02)
                self.assertAlmostEqual(cfg["latent_lam_h"], 0.0001)
                self.assertFalse(cfg["latent_actor_z_onehot_enabled"])
                self.assertEqual(cfg["latent_z_embed_dim"], 0)
                self.assertTrue(cfg["latent_actor_z_adapter_enabled"])
                self.assertAlmostEqual(cfg["latent_actor_z_adapter_scale"], 0.5)
                self.assertEqual(cfg["latent_actor_z_film_layers"], 2)
                self.assertAlmostEqual(
                    cfg["latent_actor_z_separation_start_coef"], 0.005
                )
                self.assertAlmostEqual(
                    cfg["latent_actor_z_separation_coef"], 0.02
                )
                self.assertAlmostEqual(
                    cfg["latent_actor_z_separation_min_abs_advantage"], 0.5
                )
                self.assertAlmostEqual(
                    cfg["latent_actor_z_separation_min_decision_frac"], 0.05
                )
                self.assertAlmostEqual(
                    cfg["latent_actor_z_separation_max_entropy_frac"], 0.90
                )
                self.assertFalse(cfg["fixed_latent_strategy"])

    def test_latent_v3i14_tuned_strengthens_tactical_specialization(self) -> None:
        resolved = resolve_all_presets()
        base = resolved["latent_v3i14"]
        for key in (
            "plan_faithful_latent_v3i14_tuned",
            "latent_v3i14_tuned",
            "latent_v3i14b",
            "latent_v3i14_tactical_specialist_tuned",
        ):
            with self.subTest(preset=key):
                cfg = resolved[key]
                self.assertEqual(
                    cfg["latent_q_phi_bucket_baseline"],
                    base["latent_q_phi_bucket_baseline"],
                )
                self.assertEqual(
                    cfg["latent_specialist_context_key_mode"],
                    base["latent_specialist_context_key_mode"],
                )
                self.assertTrue(cfg["latent_specialist_use_rollout_states"])
                self.assertEqual(
                    cfg["latent_specialist_rollout_max_samples"],
                    base["latent_specialist_rollout_max_samples"],
                )
                self.assertGreater(
                    cfg["latent_conditional_entropy_min_coef"],
                    base["latent_conditional_entropy_min_coef"],
                )
                self.assertGreater(
                    cfg["latent_actor_z_separation_coef"],
                    base["latent_actor_z_separation_coef"],
                )
                self.assertLessEqual(
                    cfg["latent_marginal_balance_coef"],
                    base["latent_marginal_balance_coef"],
                )
                self.assertLessEqual(cfg["latent_lam_h"], base["latent_lam_h"])
                self.assertEqual(
                    cfg["latent_specialist_warmup_steps"],
                    base["latent_specialist_warmup_steps"],
                )
                self.assertEqual(
                    cfg["latent_specialist_ramp_steps"],
                    base["latent_specialist_ramp_steps"],
                )
                self.assertEqual(
                    cfg["latent_specialist_min_bucket_count"],
                    base["latent_specialist_min_bucket_count"],
                )
                self.assertEqual(cfg["learning_rate"], base["learning_rate"])
                self.assertEqual(cfg["n_epochs"], base["n_epochs"])
                self.assertEqual(cfg["opponent_pool"], ["OP3", "OP5", "OP6"])
                self.assertEqual(
                    cfg["opponent_pool_weights"],
                    [0.15, 0.40, 0.45],
                )
                self.assertFalse(cfg["allow_op4_in_training_pool"])
                self.assertFalse(cfg["latent_actor_z_onehot_enabled"])
                self.assertEqual(cfg["latent_z_embed_dim"], 0)
                self.assertTrue(cfg["latent_actor_z_adapter_enabled"])
                self.assertEqual(cfg["latent_actor_z_film_layers"], 2)

    def test_latent_v3i11_remains_unchanged_by_v3i14_tuning(self) -> None:
        cfg = resolve_all_presets()["latent_v3i11"]
        self.assertFalse(cfg["latent_specialist_use_rollout_states"])
        self.assertEqual(
            cfg["latent_specialist_context_key_mode"],
            "role_phase_progress_opponent",
        )
        self.assertAlmostEqual(
            cfg["latent_conditional_entropy_min_coef"], 0.05
        )
        self.assertAlmostEqual(cfg["latent_actor_z_adapter_scale"], 0.35)
        self.assertAlmostEqual(cfg["latent_lam_h"], 0.003)




if __name__ == "__main__":
    unittest.main()
