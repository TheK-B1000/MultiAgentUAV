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


if __name__ == "__main__":
    unittest.main()
