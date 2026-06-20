"""Pins the v6i4 evaluation-only router ablation protocol."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from rl.evaluation.router_ablation import (
    POSTHOC_ORACLE_SELECTIONS,
    PRIMARY_SELECTIONS,
    V6I4_CLASSIFICATION,
    V6I4_PROTOCOL_VERSION,
    aggregate_condition_summary,
    default_conditions,
    deterministic_cross_context_permutation,
    model_parameter_sha256,
    paired_comparisons,
    paired_episode_key,
    select_calibrated_fixed_latents,
    validate_seed_split,
    validate_promoted_v6i2_checkpoint_metadata,
    write_artifacts,
)
from rl.config.ppo_config import PPOConfig
from rl.presets import PRESET_REGISTRY, apply_preset


class V6I4RouterAblationTests(unittest.TestCase):
    def test_v6i4_preset_is_evaluation_only_metadata(self):
        self.assertIn("v6i4", PRESET_REGISTRY)
        cfg = apply_preset(PPOConfig(), "v6i4")
        self.assertTrue(cfg.evaluation_only_preset)
        self.assertTrue(cfg.evaluation_only_requires_checkpoint)
        self.assertEqual(cfg.evaluation_only_checkpoint_family, "promoted_v6i2")
        self.assertIn("router_ablation", cfg.router_ablation_protocol_version)

    def test_seed_split_rejects_leakage(self):
        with self.assertRaises(ValueError):
            validate_seed_split([1, 2, 3], [3, 4, 5])
        validate_seed_split([1, 2], [3, 4])

    def test_condition_contract_contains_required_baselines(self):
        names = {c.name for c in default_conditions(4)}
        for name in PRIMARY_SELECTIONS:
            self.assertIn(name, names)
        for name in POSTHOC_ORACLE_SELECTIONS:
            self.assertIn(name, names)
        self.assertTrue(any(c.identity_assisted for c in default_conditions(4)))
        self.assertTrue(any(c.posthoc_only for c in default_conditions(4)))
        online = {c.name for c in default_conditions(4) if c.online_rollout and not c.posthoc_only}
        posthoc = {c.name for c in default_conditions(4) if c.posthoc_only}
        self.assertIn("preselected_global_fixed_z", online)
        self.assertIn("preselected_per_opponent_fixed_z", online)
        self.assertTrue(posthoc.isdisjoint(online))

    def test_promoted_checkpoint_metadata_requires_lineage_gate_and_step(self):
        good = {
            "cfg": {
                "experiment_id": "v6i2",
                "phase_a_gate_passed": True,
                "gate_config_fingerprint_active": "224f1aea9ab36319",
                "phase_a_end_step": 640000,
                "confirmatory_gate_lineage_valid": True,
            }
        }
        evidence = validate_promoted_v6i2_checkpoint_metadata(good, checkpoint_sha256="abc")
        self.assertEqual(evidence["experiment_lineage"], "v6i2")
        self.assertEqual(evidence["phase_a_promotion"], "PASS")
        bad = {"cfg": {**good["cfg"], "phase_a_gate_passed": False, "promoted_to_phase_b": False}}
        with self.assertRaisesRegex(ValueError, "Phase A promotion"):
            validate_promoted_v6i2_checkpoint_metadata(bad, checkpoint_sha256="abc")

    def test_cross_context_permutation_is_deterministic_and_breaks_identity(self):
        p1 = deterministic_cross_context_permutation(8, seed=17)
        p2 = deterministic_cross_context_permutation(8, seed=17)
        self.assertEqual(p1, p2)
        self.assertEqual(sorted(p1), list(range(8)))
        self.assertTrue(any(i != p for i, p in enumerate(p1)))

    def test_parameter_hash_changes_when_weights_change(self):
        module = torch.nn.Linear(2, 2)
        h1 = model_parameter_sha256(module)
        with torch.no_grad():
            module.weight[0, 0] += 1.0
        h2 = model_parameter_sha256(module)
        self.assertNotEqual(h1, h2)

    def test_calibrated_latents_use_only_calibration_rows(self):
        rows = [
            {"condition": "fixed_z0", "fixed_latent_id": 0, "opponent": "OP5", "return": 1.0},
            {"condition": "fixed_z1", "fixed_latent_id": 1, "opponent": "OP5", "return": 3.0},
            {"condition": "fixed_z0", "fixed_latent_id": 0, "opponent": "OP6", "return": 4.0},
            {"condition": "fixed_z1", "fixed_latent_id": 1, "opponent": "OP6", "return": 2.0},
        ]
        global_z, per_opp = select_calibrated_fixed_latents(rows, latent_k=2)
        self.assertEqual(global_z, 0)
        self.assertEqual(per_opp["OP5"], 1)
        self.assertEqual(per_opp["OP6"], 0)

    def test_paired_comparisons_align_same_seed_opponent_and_map(self):
        rows = [
            {"condition": "learned_qphi_switching", "map_set": "eval", "opponent": "OP5", "seed": 1, "test_seed": 1, "episode_index": 1, "initial_state_hash": "a", "return": 3.0, "success": 1},
            {"condition": "uniform_random_at_router_opportunities", "map_set": "eval", "opponent": "OP5", "seed": 1, "test_seed": 1, "episode_index": 1, "initial_state_hash": "a", "return": 1.0, "success": 0},
            {"condition": "learned_qphi_switching", "map_set": "eval", "opponent": "OP5", "seed": 2, "test_seed": 2, "episode_index": 1, "initial_state_hash": "b", "return": 2.0, "success": 1},
            {"condition": "uniform_random_at_router_opportunities", "map_set": "eval", "opponent": "OP5", "seed": 2, "test_seed": 2, "episode_index": 1, "initial_state_hash": "b", "return": 1.0, "success": 1},
            {"condition": "uniform_random_at_router_opportunities", "map_set": "eval", "opponent": "OP5", "seed": 999, "test_seed": 999, "episode_index": 1, "initial_state_hash": "x", "return": 100.0, "success": 1},
        ]
        self.assertEqual(paired_episode_key(rows[0]), ("OP5", 1, 1, "a"))
        comps = paired_comparisons(rows, baselines=["uniform_random_at_router_opportunities"], n_bootstrap=0)
        self.assertEqual(len(comps), 1)
        self.assertEqual(comps[0].n_pairs, 2)
        self.assertAlmostEqual(comps[0].mean_delta_return, 1.5)
        self.assertAlmostEqual(comps[0].mean_delta_success, 0.5)

    def test_artifact_writer_uses_frozen_names(self):
        rows = [
            {"condition": "learned_qphi_switching", "map_set": "eval", "opponent": "OP5", "seed": 1, "return": 2.0, "success": 1, "win_margin": 1},
            {"condition": "uniform_random_at_router_opportunities", "map_set": "eval", "opponent": "OP5", "seed": 1, "return": 1.0, "success": 0, "win_margin": -1},
        ]
        manifest = {
            "protocol_version": V6I4_PROTOCOL_VERSION,
            "classification": V6I4_CLASSIFICATION,
            "parameter_hash_before": "a",
            "parameter_hash_after": "a",
            "parameters_unchanged": True,
        }
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_artifacts(tmp, manifest=manifest, episode_rows=rows, n_bootstrap=0)
            for path in paths.values():
                self.assertTrue(Path(path).exists())
            payload = json.loads(Path(paths["final_report"]).read_text(encoding="utf-8"))
        self.assertEqual(payload["protocol_version"], V6I4_PROTOCOL_VERSION)

    def test_summary_groups_by_condition_opponent_and_map(self):
        rows = [
            {"condition": "learned_qphi_switching", "map_set": "eval", "opponent": "OP5", "return": 2.0, "success": 1, "win_margin": 1},
            {"condition": "learned_qphi_switching", "map_set": "eval", "opponent": "OP5", "return": 4.0, "success": 0, "win_margin": -1},
        ]
        summary = aggregate_condition_summary(rows)
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary[0]["episodes"], 2)
        self.assertAlmostEqual(summary[0]["success_rate"], 0.5)
        self.assertAlmostEqual(summary[0]["return_mean"], 3.0)


if __name__ == "__main__":
    unittest.main()
