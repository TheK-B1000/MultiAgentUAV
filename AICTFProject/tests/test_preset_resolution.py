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


if __name__ == "__main__":
    unittest.main()
