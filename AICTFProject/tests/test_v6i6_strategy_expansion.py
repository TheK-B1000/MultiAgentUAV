"""Pins v6i6 evidence-gated repertoire expansion wiring."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from dataclasses import asdict

from rl.config.ppo_config import PPOConfig
from rl.presets import apply_preset
from rl.training.cli import cfg_from_args, parse_train_args
from rl.training.config_validation import normalize_and_validate_training_config


class V6I6PresetTests(unittest.TestCase):
    def test_v6i6_aliases_resolve_identically_and_do_not_select_latent_ids(self) -> None:
        aliases = (
            "plan_faithful_latent_v6i6_strategy_expansion",
            "latent_v6i6_strategy_expansion",
            "v6i6_strategy_expansion",
            "v6i6",
        )
        resolved = [asdict(apply_preset(PPOConfig(), alias)) for alias in aliases]
        first = resolved[0]
        for cfg in resolved:
            self.assertEqual(cfg, first)

        self.assertEqual(first["experiment_id"], "v6i6")
        self.assertEqual(first["v6i6_expansion_stage"], "E1")
        self.assertEqual(first["v6i6_expansion_protocol_version"], "v6i6_repertoire_expansion_e1_v1")
        self.assertTrue(first["use_v6i6_expansion"])
        self.assertTrue(first["v6i6_require_validated_anchors"])
        self.assertIsNone(first["v6i6_anchor_validation_manifest"])
        self.assertEqual(first["v6i6_anchor_latents"], ())
        self.assertEqual(first["v6i6_target_latent"], -1)
        self.assertEqual(first["v6i6_dormant_latents"], ())
        self.assertEqual(first["latent_resample_every_n"], 0)
        self.assertTrue(first["v6i6_fixed_z_episode_attribution"])
        self.assertEqual(first["v6i6_trainable_scope"], "target_embedding_gate_adapter_only")
        self.assertTrue(first["v6i6_use_reference_critic_for_opportunity"])
        self.assertTrue(first["v6i6_restore_masked_latent_rows_after_step"])
        self.assertTrue(first["v6i6_assert_anchor_bitwise_invariant"])
        self.assertEqual(first["v6i6_count_draw_as"], 0.5)
        self.assertTrue(first["latent_actor_z_adapter_enabled"])
        self.assertEqual(first["latent_actor_z_adapter_scale"], 0.05)
        self.assertEqual(first["latent_actor_z_adapter_init_std"], 0.0)

    def test_v6i6_diff_vs_v6i5_is_expansion_contract_only(self) -> None:
        v6i5 = asdict(apply_preset(PPOConfig(), "v6i5"))
        v6i6 = asdict(apply_preset(PPOConfig(), "v6i6"))
        diff = {key for key in v6i6 if v6i6[key] != v6i5[key]}
        self.assertEqual(
            diff,
            {
                "experiment_id",
                "gate_protocol_version",
                "use_v6i6_expansion",
                "v6i6_expansion_protocol_version",
                "v6i6_expansion_stage",
                "v6i6_trainable_scope",
                "latent_actor_z_adapter_enabled",
                "latent_actor_z_adapter_scale",
                "latent_actor_z_adapter_init_std",
                "latent_resample_every_n",
                "run_tag",
            },
        )

    def test_training_validation_rejects_missing_anchor_manifest(self) -> None:
        cfg = apply_preset(PPOConfig(), "v6i6")
        with self.assertRaisesRegex(ValueError, "validated anchor manifest"):
            normalize_and_validate_training_config(cfg)

    def test_training_validation_hydrates_valid_manifest(self) -> None:
        cfg = apply_preset(PPOConfig(), "v6i6")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "v6i6_anchor_manifest.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "checkpoint_hash": "abc123",
                        "verdict": "VALIDATED",
                        "anchors": [0, 3],
                        "expansion_target": 1,
                        "dormant": [2],
                        "evidence": {
                            "forced_z_report_hash": "forced",
                            "branch_report_hash": "branch",
                        },
                    },
                    f,
                )
            cfg.v6i6_anchor_validation_manifest = path
            normalized = normalize_and_validate_training_config(cfg)
        self.assertEqual(normalized.v6i6_anchor_latents, (0, 3))
        self.assertEqual(normalized.v6i6_target_latent, 1)
        self.assertEqual(normalized.v6i6_dormant_latents, (2,))

    def test_training_validation_rejects_unvalidated_or_overlapping_manifest(self) -> None:
        for manifest in (
            {"verdict": "PENDING", "anchors": [0, 3], "expansion_target": 1, "dormant": [2]},
            {"verdict": "VALIDATED", "anchors": [0, 1], "expansion_target": 1, "dormant": [2]},
        ):
            cfg = apply_preset(PPOConfig(), "v6i6")
            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, "bad_manifest.json")
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(manifest, f)
                cfg.v6i6_anchor_validation_manifest = path
                with self.assertRaises(ValueError):
                    normalize_and_validate_training_config(cfg)

    def test_cli_accepts_manifest_path_without_renaming_real_flags(self) -> None:
        parsed = parse_train_args(
            [
                "--preset",
                "v6i6",
                "--v6i6-anchor-validation-manifest",
                "reports/v6i6_anchor_manifest.json",
            ]
        )
        cfg = cfg_from_args(parsed)
        self.assertEqual(cfg.v6i6_anchor_validation_manifest, "reports/v6i6_anchor_manifest.json")
        self.assertEqual(cfg.load_path, None)
        self.assertEqual(cfg.total_timesteps, PPOConfig().total_timesteps)


if __name__ == "__main__":
    unittest.main()
