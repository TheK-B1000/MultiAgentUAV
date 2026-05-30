import json
import os
import unittest
from dataclasses import asdict

from rl.train_ppo import PPOConfig
from rl.presets import apply_preset, PRESET_REGISTRY


class PresetResolutionTests(unittest.TestCase):
    def test_all_presets_resolve_and_match_snapshots(self) -> None:
        """Resolve every preset in the registry and verify it matches the saved snapshot."""
        here = os.path.dirname(os.path.abspath(__file__))
        snapshot_path = os.path.join(here, "preset_snapshots.json")

        # Resolve all presets to JSON-serializable dicts
        resolved = {}
        for key in sorted(PRESET_REGISTRY.keys()):
            cfg = PPOConfig()
            apply_preset(cfg, key)
            
            # Convert config to a dict, converting any non-serializable types (like tuples) to lists/strings
            cfg_dict = asdict(cfg)
            # Make sure tuples inside dict are lists for JSON consistency
            if "opponent_pool" in cfg_dict and isinstance(cfg_dict["opponent_pool"], tuple):
                cfg_dict["opponent_pool"] = list(cfg_dict["opponent_pool"])
                
            resolved[key] = cfg_dict

        # If snapshot does not exist, write the current values as the baseline
        if not os.path.isfile(snapshot_path):
            with open(snapshot_path, "w", encoding="utf-8") as f:
                json.dump(resolved, f, indent=2)
            print(f"\n[Presets Test] Created reference snapshots file at: {snapshot_path}")

        # Load snapshot and compare
        with open(snapshot_path, "r", encoding="utf-8") as f:
            snapshot = json.load(f)

        # Assert keys match exactly
        self.assertEqual(
            set(resolved.keys()),
            set(snapshot.keys()),
            "The set of keys in the preset registry does not match the saved snapshot keys.",
        )

        # Assert config fields match exactly for each preset
        for key in resolved:
            self.assertEqual(
                resolved[key],
                snapshot[key],
                f"Preset '{key}' configuration differs from saved snapshot.",
            )


if __name__ == "__main__":
    unittest.main()
