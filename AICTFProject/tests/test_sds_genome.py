"""CPU tests for Strategic Demand Searcher genomes and profile overlays."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gpu_env._core._bt_profiles import BT_PROFILES, build_profile_tensors, profile_for_opponent_key
from experiments.sds_genome import (
    ANCHOR_B,
    SDSGenome,
    canonical_parent,
    degeneracy_penalty,
    mutate,
    overlay_profile,
    recombine,
)


class OverlayDoesNotMutateRegistryTests(unittest.TestCase):
    def test_none_overrides_match_default(self) -> None:
        keys = ["OP6", "OP7", "OP10", "OP12"]
        a = build_profile_tensors(keys, device=torch.device("cpu"), batch_size=4)
        b = build_profile_tensors(
            keys, device=torch.device("cpu"), batch_size=4, overrides=None)
        self.assertTrue(torch.equal(a["threat_radius"], b["threat_radius"]))
        self.assertTrue(torch.equal(a["bt_level"], b["bt_level"]))

    def test_override_changes_knob_not_level(self) -> None:
        from dataclasses import replace
        parent = profile_for_opponent_key("OP6")
        ov = replace(parent, threat_radius=12.5)
        keys = ["OP6"]
        prof = build_profile_tensors(
            keys, device=torch.device("cpu"), batch_size=1, overrides=[ov])
        self.assertEqual(int(prof["bt_level"][0].item()), 6)
        self.assertAlmostEqual(float(prof["threat_radius"][0].item()), 12.5)
        self.assertEqual(float(BT_PROFILES[6].threat_radius), 0.0)

    def test_canonical_parent_is_identity_overlay(self) -> None:
        g = canonical_parent("OP6")
        self.assertEqual(g.overlay, {})
        self.assertEqual(g.opening_hold_steps, 0)
        self.assertEqual(overlay_profile(g).threat_radius, BT_PROFILES[6].threat_radius)


class GenomeMutationTests(unittest.TestCase):
    def test_mutate_stays_legal(self) -> None:
        rng = np.random.default_rng(0)
        g = canonical_parent("OP6")
        for i in range(40):
            g = mutate(g, rng, new_id=f"m{i}")
            self.assertNotEqual(g.base_opponent, ANCHOR_B)
            SDSGenome.from_dict(g.to_dict())

    def test_op7_allowed_only_as_explicit_anchor(self) -> None:
        SDSGenome(
            genome_id="B", derived_from="OP7", base_opponent="OP7",
            overlay={}, opening_hold_steps=0)
        with self.assertRaises(ValueError):
            SDSGenome(
                genome_id="bad", derived_from="OP5", base_opponent="OP5",
                overlay={})

    def test_degeneracy_penalty_zero_on_healthy_games(self) -> None:
        self.assertEqual(degeneracy_penalty(0.0, 1.5), 0.0)
        self.assertGreater(degeneracy_penalty(0.9, 0.0), 0.0)

    def test_recombine_stays_legal_and_new_id(self) -> None:
        rng = np.random.default_rng(1)
        a = canonical_parent("OP6")
        b = canonical_parent("OP12")
        b.opening_hold_steps = 40
        c = recombine(a, b, rng, new_id="mix1")
        self.assertEqual(c.genome_id, "mix1")
        self.assertIn(c.base_opponent, ("OP6", "OP12"))
        self.assertIn(c.opening_hold_steps, (0, 40))
        self.assertNotEqual(c.genome_id, a.genome_id)


if __name__ == "__main__":
    unittest.main()
