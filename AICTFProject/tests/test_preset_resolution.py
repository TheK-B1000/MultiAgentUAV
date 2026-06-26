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
    """Apply a preset to a fresh ``PPOConfig`` and return a JSON-safe dict.

    JSON has no tuple type, so any ``tuple`` field on PPOConfig has to be
    normalised to a list before comparison: otherwise a freshly resolved
    config (tuple) would never equal a snapshot loaded from JSON (list).
    Add every tuple-typed PPOConfig field that ships in the registry here.
    """
    cfg = PPOConfig()
    apply_preset(cfg, key)
    cfg_dict = asdict(cfg)
    for tuple_field in (
        "opponent_pool",
        "opponent_pool_weights",
        "latent_router_distill_opponents",
        "router_allowed_latents",
        "router_ablation_conditions",
        "router_ablation_oracle_conditions",
        "router_ablation_primary_metrics",
        "router_ablation_diagnostic_metrics",
        "router_ablation_opponents",
        "v6i6_anchor_latents",
        "v6i6_dormant_latents",
    ):
        if isinstance(cfg_dict.get(tuple_field), tuple):
            cfg_dict[tuple_field] = list(cfg_dict[tuple_field])
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
            "plan_faithful_latent_v3i15_sparse_tactical_refresh",
            "latent_v3i15_sparse_tactical_refresh",
            "latent_v3i15",
            "latent_v3i15_sparse_refresh",
            "plan_faithful_latent_v3i15_strong_separation",
            "latent_v3i15_strong_separation",
            # v3i17: consequence-only family. Inherits v3i16's architecture but
            # turns episode-credit PPO back on as the SOLE q_phi gradient.
            "plan_faithful_latent_v3i17_episode_arc",
            "latent_v3i17_episode_arc",
            "latent_v3i17",
            "v3i17_episode_arc",
            "plan_faithful_latent_v3i17_long_arc",
            "latent_v3i17_long_arc",
            "latent_v3i17b",
            "v3i17_long_arc",
            # v5i1: additive reward-credit repair for the strict-Summer router.
            "plan_faithful_latent_v5i1_reward_credit_router",
            "latent_v5i1_reward_credit_router",
            "v5i1_reward_credit_router",
            "v5i1",
            # v5i2 inherits v5i1's router and changes actor conditioning only.
            "plan_faithful_latent_v5i2_stronger_z_conditioning",
            "latent_v5i2_stronger_z_conditioning",
            "v5i2_stronger_z_conditioning",
            "v5i2",
            # v5i3 inherits v5i2 and layers a forced-z anneal on top -- the
            # episode-credit PPO path is unchanged.
            "plan_faithful_latent_v5i3_balanced_warmup",
            "latent_v5i3_balanced_warmup",
            "v5i3_balanced_warmup",
            "v5i3",
            "balanced_warmup",
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

    def test_latent_v3i15_sparse_tactical_refresh_is_isolated_timescale_variant(
        self,
    ) -> None:
        resolved = resolve_all_presets()
        base = resolved["latent_v3i14_tuned"]
        aliases = (
            "plan_faithful_latent_v3i15_sparse_tactical_refresh",
            "latent_v3i15_sparse_tactical_refresh",
            "latent_v3i15",
            "latent_v3i15_sparse_refresh",
        )
        first = resolved[aliases[0]]
        for key in aliases:
            with self.subTest(preset=key):
                cfg = resolved[key]
                self.assertEqual(cfg, first)
                self.assertTrue(cfg["latent_sparse_tactical_refresh_enabled"])
                self.assertEqual(
                    cfg["latent_sparse_tactical_refresh_interval_steps"],
                    32,
                )
                self.assertEqual(
                    cfg["latent_sparse_tactical_refresh_min_dwell_steps"],
                    16,
                )
                self.assertFalse(cfg["latent_event_refresh_enabled"])
                self.assertEqual(
                    cfg["latent_episode_strategy_warmup_decision_steps"],
                    5,
                )
                self.assertTrue(cfg["latent_gae_reset_on_z_change"])
                self.assertAlmostEqual(cfg["latent_lam_p"], 0.02)
                self.assertAlmostEqual(cfg["latent_lam_h"], 0.000025)
                self.assertAlmostEqual(
                    cfg["latent_conditional_entropy_min_coef"],
                    0.09,
                )
                self.assertAlmostEqual(
                    cfg["latent_actor_z_separation_coef"],
                    0.028,
                )
                self.assertAlmostEqual(
                    cfg["latent_marginal_balance_coef"],
                    0.015,
                )
                self.assertEqual(
                    cfg["latent_q_phi_bucket_baseline"],
                    base["latent_q_phi_bucket_baseline"],
                )
                self.assertEqual(
                    cfg["latent_specialist_context_key_mode"],
                    base["latent_specialist_context_key_mode"],
                )
                self.assertTrue(cfg["latent_specialist_use_rollout_states"])
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

        self.assertFalse(base["latent_sparse_tactical_refresh_enabled"])
        self.assertAlmostEqual(base["latent_lam_p"], 0.0)
        self.assertAlmostEqual(base["latent_lam_h"], 0.00005)
        self.assertIn("latent_v3i15_strong_separation", resolved)

    def test_latent_v3i16_is_strict_plan_faithful_z_embedding(self) -> None:
        resolved = resolve_all_presets()
        aliases = (
            "plan_faithful_latent_v3i16_policy_z_embedding",
            "latent_v3i16_policy_z_embedding",
            "latent_v3i16",
            "latent_v3i16_summer_z_embed",
            "plan_faithful_latent_v3i16_z_embed",
            "v3i16_plan_faithful_z_embed",
        )
        first = resolved[aliases[0]]
        for key in aliases:
            with self.subTest(preset=key):
                cfg = resolved[key]
                self.assertEqual(cfg, first)
                self.assertTrue(cfg["use_latent_strategy"])
                self.assertEqual(cfg["latent_z_embed_dim"], 16)
                self.assertAlmostEqual(
                    cfg["latent_actor_z_embed_scale"],
                    1.0,
                )
                self.assertFalse(cfg["latent_actor_z_onehot_enabled"])
                self.assertFalse(cfg["latent_actor_z_adapter_enabled"])
                self.assertAlmostEqual(cfg["latent_actor_z_adapter_scale"], 0.0)
                self.assertEqual(cfg["latent_actor_z_film_layers"], 1)
                self.assertEqual(cfg["latent_resample_every_n"], 64)
                self.assertFalse(cfg["latent_event_refresh_enabled"])
                self.assertFalse(cfg["latent_sparse_tactical_refresh_enabled"])
                self.assertTrue(cfg["latent_gae_reset_on_z_change"])
                self.assertAlmostEqual(cfg["latent_lam_p"], 0.02)
                self.assertAlmostEqual(cfg["latent_lam_h"], 0.001)
                self.assertAlmostEqual(cfg["latent_lam_h_start"], 0.001)
                self.assertAlmostEqual(cfg["latent_lam_h_end"], 0.001)
                self.assertEqual(cfg["mode"], "FIXED_OPPONENT")
                self.assertEqual(cfg["fixed_opponent_tag"], "OP3")
                self.assertFalse(cfg["latent_episode_strategy_ppo"])
                self.assertFalse(cfg["latent_strategy_aux_return_head"])
                self.assertAlmostEqual(
                    cfg["latent_strategy_aux_return_coef"], 0.0
                )
                self.assertAlmostEqual(
                    cfg["latent_strategy_aux_predict_phase_coef"], 0.0
                )
                self.assertAlmostEqual(cfg["latent_forced_z_episode_frac"], 0.0)
                self.assertAlmostEqual(cfg["latent_behavior_contrast_coef"], 0.0)
                self.assertAlmostEqual(
                    cfg["latent_actor_z_separation_coef"], 0.0
                )
                self.assertAlmostEqual(cfg["latent_preference_coef"], 0.0)
                self.assertFalse(cfg["latent_awrd_enabled"])
                self.assertAlmostEqual(cfg["latent_awrd_coef"], 0.0)
                self.assertFalse(cfg["latent_specialist_router_enabled"])
                self.assertAlmostEqual(
                    cfg["latent_marginal_balance_coef"], 0.0
                )
                self.assertAlmostEqual(
                    cfg["latent_conditional_entropy_min_coef"], 0.0
                )
                self.assertAlmostEqual(cfg["latent_context_mi_coef"], 0.0)
                self.assertFalse(
                    cfg["latent_v3i3_event_preference_enabled"]
                )
                self.assertEqual(
                    cfg["run_tag"],
                    "v3i16_plan_faithful_z_embed_1m_4v4",
                )

    def test_latent_v3i17_is_consequence_only_episode_arc(self) -> None:
        """v3i17 episode_arc: episode-level z, lam_h anneal -> 0, episode-credit only.

        Plan-faithful invariants:
            * One z per episode (latent_resample_every_n == 0)
            * No mid-episode refresh machinery
            * Existence pressure annealed to exactly 0 (latent_lam_h_end == 0.0)
            * Sole consequence channel: latent_episode_strategy_ppo with coef > 0
            * No supervised heads / preference / specialist / behavior-contrast
        """
        resolved = resolve_all_presets()
        aliases = (
            "plan_faithful_latent_v3i17_episode_arc",
            "latent_v3i17_episode_arc",
            "latent_v3i17",
            "v3i17_episode_arc",
        )
        first = resolved[aliases[0]]
        for key in aliases:
            with self.subTest(preset=key):
                cfg = resolved[key]
                self.assertEqual(cfg, first)
                # Duration: episode-level z
                self.assertEqual(cfg["latent_resample_every_n"], 0)
                self.assertFalse(cfg["latent_resample_on_flag"])
                self.assertFalse(cfg["latent_event_refresh_enabled"])
                self.assertFalse(cfg["latent_sparse_tactical_refresh_enabled"])
                self.assertAlmostEqual(cfg["latent_lam_p"], 0.0)
                # Existence pressure annealed to exactly 0.0 by end of run
                self.assertAlmostEqual(cfg["latent_lam_h_start"], 0.003)
                self.assertAlmostEqual(cfg["latent_lam_h_end"], 0.0)
                self.assertEqual(cfg["latent_entropy_anneal_start"], 200_000)
                self.assertEqual(cfg["latent_entropy_anneal_end"], 700_000)
                # Consequence channel: episode-credit PPO only
                self.assertTrue(cfg["latent_episode_strategy_ppo"])
                self.assertAlmostEqual(cfg["latent_episode_strategy_coef"], 0.30)
                self.assertAlmostEqual(cfg["latent_strategy_ppo_coef"], 0.0)
                self.assertEqual(
                    cfg["latent_episode_strategy_warmup_decision_steps"], 5
                )
                # No supervised / existence channels
                self.assertFalse(cfg["latent_v3i3_event_preference_enabled"])
                self.assertFalse(cfg["latent_awrd_enabled"])
                self.assertFalse(cfg["latent_specialist_router_enabled"])
                self.assertAlmostEqual(cfg["latent_preference_coef"], 0.0)
                self.assertAlmostEqual(cfg["latent_behavior_contrast_coef"], 0.0)
                self.assertAlmostEqual(
                    cfg["latent_actor_z_separation_coef"], 0.0
                )
                self.assertAlmostEqual(cfg["latent_usage_balance_coef"], 0.0)
                self.assertAlmostEqual(
                    cfg["latent_marginal_balance_coef"], 0.0
                )
                self.assertAlmostEqual(
                    cfg["latent_strategy_aux_return_coef"], 0.0
                )
                self.assertAlmostEqual(
                    cfg["latent_strategy_aux_predict_phase_coef"], 0.0
                )
                # Architecture inherited from v3i16
                self.assertEqual(cfg["latent_z_embed_dim"], 16)
                self.assertFalse(cfg["latent_actor_z_onehot_enabled"])
                self.assertFalse(cfg["latent_actor_z_adapter_enabled"])
                self.assertEqual(
                    cfg["run_tag"], "v3i17_episode_arc_1m_4v4"
                )

    def test_latent_v3i18_is_v3i16_with_only_resample_interval_changed(self) -> None:
        """v3i18 = v3i16 with exactly ONE knob changed (resample_every_n: 64 -> 128).

        Plan-faithful guarantees:

        * Inherits v3i16's actor z-embedding architecture verbatim.
        * The resolved config differs from v3i16 in exactly two keys:
          ``latent_resample_every_n`` and ``run_tag``.
        * No new supervised heads, preference channels, or existence rewards.
        """
        resolved = resolve_all_presets()
        aliases = (
            "plan_faithful_latent_v3i18_v3i16_plus_128",
            "latent_v3i18_v3i16_plus_128",
            "latent_v3i18",
            "v3i18_v3i16_plus_128",
        )
        first = resolved[aliases[0]]
        v3i16 = resolved["latent_v3i16"]

        # Exact minimal-delta property: only resample interval + run_tag differ.
        differing_keys = {k for k in first if first[k] != v3i16.get(k)}
        differing_keys |= {k for k in v3i16 if k not in first}
        self.assertEqual(
            differing_keys,
            {"latent_resample_every_n", "run_tag"},
            f"v3i18 must differ from v3i16 only in (resample_every_n, run_tag); "
            f"unexpected diff: {differing_keys}",
        )
        self.assertEqual(v3i16["latent_resample_every_n"], 64)
        self.assertEqual(first["latent_resample_every_n"], 128)

        for key in aliases:
            with self.subTest(preset=key):
                cfg = resolved[key]
                self.assertEqual(cfg, first)

                # Architecture inherited from v3i16
                self.assertTrue(cfg["use_latent_strategy"])
                self.assertEqual(cfg["latent_k"], 4)
                self.assertEqual(cfg["latent_z_embed_dim"], 16)
                self.assertAlmostEqual(cfg["latent_actor_z_embed_scale"], 1.0)
                self.assertFalse(cfg["latent_actor_z_onehot_enabled"])
                self.assertFalse(cfg["latent_actor_z_adapter_enabled"])
                self.assertAlmostEqual(cfg["latent_actor_z_adapter_scale"], 0.0)
                self.assertEqual(cfg["latent_actor_z_film_layers"], 1)

                # Only the resample interval changed.
                self.assertEqual(cfg["latent_resample_every_n"], 128)
                self.assertFalse(cfg["latent_resample_on_flag"])
                self.assertFalse(cfg["latent_event_refresh_enabled"])
                self.assertFalse(cfg["latent_sparse_tactical_refresh_enabled"])
                self.assertTrue(cfg["latent_gae_reset_on_z_change"])

                # v3i16 knobs preserved bit-for-bit (the user's explicit list).
                self.assertAlmostEqual(cfg["latent_strategy_ppo_coef"], 0.30)
                self.assertAlmostEqual(cfg["latent_lam_p"], 0.02)
                self.assertAlmostEqual(cfg["latent_lam_h"], 0.001)
                self.assertAlmostEqual(cfg["latent_lam_h_start"], 0.001)
                self.assertAlmostEqual(cfg["latent_lam_h_end"], 0.001)

                # No episode-credit PPO (per the user's explicit "keep unchanged" list).
                self.assertFalse(cfg["latent_episode_strategy_ppo"])
                self.assertAlmostEqual(cfg["latent_episode_strategy_coef"], 0.0)

                # Summer-faithful audit: no supervised / existence / preference channels.
                self.assertFalse(cfg["latent_strategy_aux_return_head"])
                self.assertAlmostEqual(
                    cfg["latent_strategy_aux_return_coef"], 0.0
                )
                self.assertAlmostEqual(
                    cfg["latent_strategy_aux_predict_phase_coef"], 0.0
                )
                self.assertFalse(cfg["latent_v3i3_event_preference_enabled"])
                self.assertFalse(cfg["latent_awrd_enabled"])
                self.assertAlmostEqual(cfg["latent_awrd_coef"], 0.0)
                self.assertAlmostEqual(cfg["latent_preference_coef"], 0.0)
                self.assertAlmostEqual(cfg["latent_behavior_contrast_coef"], 0.0)
                self.assertAlmostEqual(
                    cfg["latent_actor_z_separation_coef"], 0.0
                )
                self.assertAlmostEqual(cfg["latent_usage_balance_coef"], 0.0)
                self.assertAlmostEqual(
                    cfg["latent_marginal_balance_coef"], 0.0
                )
                self.assertFalse(cfg["latent_specialist_router_enabled"])
                self.assertAlmostEqual(
                    cfg["latent_conditional_entropy_min_coef"], 0.0
                )
                self.assertAlmostEqual(cfg["latent_context_mi_coef"], 0.0)
                self.assertAlmostEqual(cfg["latent_forced_z_episode_frac"], 0.0)

                self.assertEqual(
                    cfg["run_tag"], "v3i18_v3i16_plus_128_1m_4v4"
                )

    def test_latent_v3i17_long_arc_uses_256_step_persistence(self) -> None:
        """v3i17 long_arc: 256-step dwell, same consequence-only contract.

        The only differences vs v3i17_episode_arc are:
            * latent_resample_every_n == 256 (was 0)
            * latent_lam_p == 0.01 (small switch cost within the long arc)
        """
        resolved = resolve_all_presets()
        aliases = (
            "plan_faithful_latent_v3i17_long_arc",
            "latent_v3i17_long_arc",
            "latent_v3i17b",
            "v3i17_long_arc",
        )
        first = resolved[aliases[0]]
        for key in aliases:
            with self.subTest(preset=key):
                cfg = resolved[key]
                self.assertEqual(cfg, first)
                self.assertEqual(cfg["latent_resample_every_n"], 256)
                self.assertAlmostEqual(cfg["latent_lam_p"], 0.01)
                # Same consequence-only contract as episode_arc
                self.assertAlmostEqual(cfg["latent_lam_h_end"], 0.0)
                self.assertTrue(cfg["latent_episode_strategy_ppo"])
                self.assertAlmostEqual(
                    cfg["latent_episode_strategy_coef"], 0.30
                )
                self.assertAlmostEqual(
                    cfg["latent_strategy_ppo_coef"], 0.0
                )
                self.assertFalse(cfg["latent_v3i3_event_preference_enabled"])
                self.assertFalse(cfg["latent_event_refresh_enabled"])
                self.assertFalse(cfg["latent_sparse_tactical_refresh_enabled"])
                # No supervised / preference / specialist
                self.assertFalse(cfg["latent_awrd_enabled"])
                self.assertFalse(cfg["latent_specialist_router_enabled"])
                self.assertAlmostEqual(cfg["latent_preference_coef"], 0.0)
                self.assertEqual(cfg["run_tag"], "v3i17_long_arc_1m_4v4")

    def test_latent_v3i19_summer_consequence_arc_credit_contract(self) -> None:
        """v3i19: Summer-faithful per-arc consequence credit.

        Asserts the locked design from the user spec:

        * Sampling: ``latent_resample_every_n == 64`` + flag-event refresh ON.
        * Persistence: ``latent_lam_p == 0.03``.
        * Entropy: anneals 0.003 -> 0.0002 over 0 -> 300_000 steps.
        * Credit assignment: arc credit ON (coef 1.0, baseline context_value,
          min_len 32, return_norm), per-step PPO OFF, episode-credit OFF.
        * Actor conditioning: FiLM (1 layer) + onehot concat, z_embed_dim=16.
        * No supervised / preference / existence channels.
        * Inherits ``latent_k=4`` and critic-z conditioning from v3i16 lineage.
        """
        resolved = resolve_all_presets()
        aliases = (
            "plan_faithful_latent_v3i19_summer_consequence",
            "latent_v3i19_summer_consequence",
            "latent_v3i19",
            "v3i19_summer_consequence",
        )
        first = resolved[aliases[0]]
        for key in aliases:
            with self.subTest(preset=key):
                cfg = resolved[key]
                self.assertEqual(cfg, first)

                # K, shared actor, critic z (inherited from v3i16 lineage).
                self.assertTrue(cfg["use_latent_strategy"])
                self.assertEqual(cfg["latent_k"], 4)

                # Sparse interval-only sampling. ``latent_resample_on_flag``
                # is OFF because its current implementation (distance-delta
                # triggers in flag-territory slice) fires every step in 4v4
                # and would collapse arc lengths to ~1, dropping 100% of
                # arcs under ``min_len=32`` and starving q_phi of gradient.
                # See plan_faithful.py preset docstring for context.
                self.assertEqual(cfg["latent_resample_every_n"], 64)
                self.assertFalse(cfg["latent_resample_on_flag"])
                self.assertFalse(cfg["latent_event_refresh_enabled"])
                self.assertFalse(cfg["latent_sparse_tactical_refresh_enabled"])
                self.assertTrue(cfg["latent_gae_reset_on_z_change"])

                # Persistence.
                self.assertAlmostEqual(cfg["latent_lam_p"], 0.03)

                # Entropy schedule.
                self.assertAlmostEqual(cfg["latent_lam_h_start"], 0.003)
                self.assertAlmostEqual(cfg["latent_lam_h_end"], 0.0002)
                self.assertEqual(cfg["latent_entropy_anneal_start"], 0)
                self.assertEqual(cfg["latent_entropy_anneal_end"], 300_000)

                # Credit assignment: arc credit is the SOLE q_phi gradient.
                self.assertTrue(cfg["latent_arc_credit_enabled"])
                self.assertAlmostEqual(cfg["latent_arc_credit_coef"], 1.0)
                self.assertEqual(cfg["latent_arc_credit_baseline"], "context_value")
                self.assertEqual(cfg["latent_arc_credit_min_len"], 32)
                self.assertTrue(cfg["latent_arc_credit_return_norm"])
                self.assertEqual(cfg["latent_arc_credit_n_epochs"], 4)
                self.assertAlmostEqual(cfg["latent_arc_credit_clip_eps"], 0.2)
                self.assertAlmostEqual(cfg["latent_strategy_ppo_coef"], 0.0)
                self.assertFalse(cfg["latent_episode_strategy_ppo"])
                self.assertAlmostEqual(cfg["latent_episode_strategy_coef"], 0.0)

                # Actor: FiLM + onehot concat at embed_dim 16.
                self.assertEqual(cfg["latent_z_embed_dim"], 16)
                self.assertEqual(cfg["latent_actor_z_film_layers"], 1)
                self.assertTrue(cfg["latent_actor_z_onehot_enabled"])
                self.assertAlmostEqual(cfg["latent_actor_z_onehot_scale"], 1.0)
                self.assertFalse(cfg["latent_actor_z_adapter_enabled"])

                # Summer-faithful audit: no supervised / preference / existence
                # / role-label / aux-prediction channels.
                self.assertFalse(cfg["latent_strategy_aux_return_head"])
                self.assertAlmostEqual(
                    cfg["latent_strategy_aux_return_coef"], 0.0
                )
                self.assertAlmostEqual(
                    cfg["latent_strategy_aux_predict_phase_coef"], 0.0
                )
                self.assertFalse(cfg["latent_v3i3_event_preference_enabled"])
                self.assertAlmostEqual(
                    cfg["latent_v3i3_event_preference_coef"], 0.0
                )
                self.assertFalse(cfg["latent_v3i3_refresh_log_enabled"])
                self.assertFalse(cfg["latent_awrd_enabled"])
                self.assertAlmostEqual(cfg["latent_awrd_coef"], 0.0)
                self.assertAlmostEqual(cfg["latent_preference_coef"], 0.0)
                self.assertAlmostEqual(cfg["latent_preference_commit_coef"], 0.0)
                self.assertAlmostEqual(cfg["latent_behavior_contrast_coef"], 0.0)
                self.assertAlmostEqual(
                    cfg["latent_actor_z_separation_coef"], 0.0
                )
                self.assertAlmostEqual(
                    cfg["latent_actor_z_separation_start_coef"], 0.0
                )
                self.assertAlmostEqual(cfg["latent_usage_balance_coef"], 0.0)
                self.assertAlmostEqual(
                    cfg["latent_marginal_balance_coef"], 0.0
                )
                self.assertFalse(cfg["latent_specialist_router_enabled"])
                self.assertAlmostEqual(
                    cfg["latent_conditional_entropy_min_coef"], 0.0
                )
                self.assertAlmostEqual(cfg["latent_context_mi_coef"], 0.0)
                self.assertAlmostEqual(cfg["latent_forced_z_episode_frac"], 0.0)

                self.assertEqual(
                    cfg["run_tag"], "v3i19_summer_consequence_1m_4v4"
                )

    def test_v5i1_reward_credit_router_contract(self) -> None:
        resolved = resolve_all_presets()
        aliases = (
            "plan_faithful_latent_v5i1_reward_credit_router",
            "latent_v5i1_reward_credit_router",
            "v5i1_reward_credit_router",
            "v5i1",
        )
        first = resolved[aliases[0]]
        for key in aliases:
            with self.subTest(preset=key):
                cfg = resolved[key]
                self.assertEqual(cfg, first)

                self.assertTrue(cfg["use_latent_strategy"])
                self.assertEqual(cfg["latent_k"], 4)
                self.assertEqual(cfg["latent_z_embed_dim"], 16)
                self.assertFalse(cfg["latent_actor_z_onehot_enabled"])
                self.assertFalse(cfg["latent_actor_z_adapter_enabled"])

                self.assertEqual(cfg["latent_resample_every_n"], 0)
                self.assertFalse(cfg["latent_resample_on_flag"])
                self.assertAlmostEqual(cfg["latent_lam_p"], 0.0)
                self.assertAlmostEqual(cfg["latent_lam_h_start"], 0.003)
                self.assertAlmostEqual(cfg["latent_lam_h_end"], 0.001)
                self.assertEqual(cfg["latent_entropy_anneal_start"], 200_000)
                self.assertEqual(cfg["latent_entropy_anneal_end"], 700_000)

                self.assertTrue(cfg["latent_episode_strategy_ppo"])
                self.assertAlmostEqual(cfg["latent_episode_strategy_coef"], 0.30)
                self.assertEqual(
                    cfg["latent_episode_strategy_warmup_decision_steps"], 5
                )
                self.assertEqual(cfg["latent_episode_strategy_n_epochs"], 6)
                self.assertAlmostEqual(cfg["latent_episode_strategy_lr"], 5e-3)
                self.assertTrue(cfg["latent_q_phi_marginal_baseline"])
                self.assertFalse(cfg["latent_arc_credit_enabled"])
                self.assertAlmostEqual(cfg["latent_strategy_ppo_coef"], 0.0)

                self.assertFalse(cfg["latent_strategy_aux_return_head"])
                self.assertAlmostEqual(
                    cfg["latent_strategy_aux_predict_phase_coef"], 0.0
                )
                self.assertAlmostEqual(cfg["latent_preference_coef"], 0.0)
                self.assertFalse(cfg["latent_awrd_enabled"])
                self.assertFalse(cfg["latent_specialist_router_enabled"])
                self.assertFalse(cfg["latent_router_distill_enabled"])
                self.assertEqual(
                    cfg["run_tag"],
                    "v5i1_reward_credit_router_OP5_OP6_OP7_2m_4v4",
                )

    def test_v5i2_stronger_z_conditioning_contract(self) -> None:
        resolved = resolve_all_presets()
        aliases = (
            "plan_faithful_latent_v5i2_stronger_z_conditioning",
            "latent_v5i2_stronger_z_conditioning",
            "v5i2_stronger_z_conditioning",
            "v5i2",
        )
        first = resolved[aliases[0]]
        for key in aliases:
            with self.subTest(preset=key):
                self.assertEqual(resolved[key], first)

        v5i1 = resolved["v5i1"]
        ignored = {
            "enable_actor_z_film",
            "actor_z_film_init_scale",
            "actor_z_film_layer",
            "run_tag",
        }
        self.assertEqual(
            {k: v for k, v in first.items() if k not in ignored},
            {k: v for k, v in v5i1.items() if k not in ignored},
        )
        self.assertFalse(v5i1["enable_actor_z_film"])
        self.assertAlmostEqual(v5i1["actor_z_film_init_scale"], 0.0)
        self.assertEqual(v5i1["actor_z_film_layer"], 2)
        self.assertTrue(first["enable_actor_z_film"])
        self.assertAlmostEqual(first["actor_z_film_init_scale"], 0.02)
        self.assertEqual(first["actor_z_film_layer"], 2)
        self.assertAlmostEqual(first["latent_actor_z_separation_coef"], 0.0)
        self.assertAlmostEqual(first["latent_behavior_contrast_coef"], 0.0)
        self.assertAlmostEqual(first["latent_forced_z_episode_frac"], 0.0)
        self.assertEqual(
            first["run_tag"],
            "v5i2_stronger_z_conditioning_OP5_OP6_OP7_2m_4v4",
        )

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
