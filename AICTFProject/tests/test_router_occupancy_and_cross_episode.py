"""Tests for router collapse telemetry and the cross-episode shuffle regrouping.

Covers two fixes:

1. Decision-point selected-z occupancy (``router_selected_z_occupancy_z*``)
   surfaced by ``_latent_rollout_stats`` so router collapse (all opportunities
   pick one z) is visible during training.
2. Cross-episode histogram-preserving shuffle cells keyed by (opponent, map)
   instead of (opponent, seed), so a unique-seed-per-episode eval protocol no
   longer degenerates into singleton cells with ``can_reassign=False``.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from rl.custom_ppo.diagnostics.aggregation import _latent_rollout_stats
from rl.evaluation.router_ablation import (
    build_cross_episode_shuffled_mapping_from_learned_traces,
)


def _fake_buffer(z_seq: list[int], resampled: list[int], latent_k: int) -> SimpleNamespace:
    t = len(z_seq)
    z = torch.tensor(z_seq, dtype=torch.long).reshape(t, 1)
    prev = torch.tensor([0] + z_seq[:-1], dtype=torch.long).reshape(t, 1)
    rs = torch.tensor(resampled, dtype=torch.bool).reshape(t, 1)
    fields = {
        "z": z,
        "prev_z": prev,
        "z_persist_mask": rs.clone(),
        "z_resampled": rs.clone(),
    }
    return SimpleNamespace(pos=t, fields=fields)


class TestSelectedZOccupancyTelemetry(unittest.TestCase):
    def test_decision_point_occupancy_and_collapse(self) -> None:
        trainer = SimpleNamespace(use_latent_strategy=True, latent_k=4)
        # 6 opportunities: z=3 five times, z=1 once. Continuation steps carry z=3
        # but are NOT decision points, so decision occupancy must ignore them.
        z_seq = [3, 3, 3, 3, 3, 3, 3, 1, 1, 1]
        resampled = [1, 0, 1, 0, 1, 0, 1, 1, 0, 1]  # 6 decision points
        stats = _latent_rollout_stats(trainer, _fake_buffer(z_seq, resampled, 4))
        occ = [stats[f"router_selected_z_occupancy_z{k}"] for k in range(4)]
        # Decision z's are [3,3,3,3,1,1] -> z3=4/6, z1=2/6, z0=z2=0.
        self.assertAlmostEqual(occ[3], 4.0 / 6.0, places=6)
        self.assertAlmostEqual(occ[1], 2.0 / 6.0, places=6)
        self.assertEqual(occ[0], 0.0)
        self.assertEqual(occ[2], 0.0)
        self.assertAlmostEqual(sum(occ), 1.0, places=6)
        self.assertEqual(stats["router_selected_z_dominant"], 3.0)
        self.assertEqual(stats["router_selected_z_unique_count"], 2.0)
        self.assertAlmostEqual(stats["router_selected_z_occupancy_max"], 4.0 / 6.0, places=6)
        self.assertEqual(stats["router_selected_z_decision_count"], 6.0)

    def test_full_collapse_single_z(self) -> None:
        trainer = SimpleNamespace(use_latent_strategy=True, latent_k=4)
        z_seq = [2, 2, 2, 2]
        resampled = [1, 1, 1, 1]
        stats = _latent_rollout_stats(trainer, _fake_buffer(z_seq, resampled, 4))
        self.assertEqual(stats["router_selected_z_occupancy_z2"], 1.0)
        self.assertEqual(stats["router_selected_z_unique_count"], 1.0)
        self.assertEqual(stats["router_selected_z_occupancy_max"], 1.0)
        self.assertEqual(stats["router_selected_z_dominant"], 2.0)


def _cell_traces(
    opponent: str, map_name: str, seed: int, seq: list[int]
) -> list[dict]:
    rows = []
    for opp_idx, z in enumerate(seq):
        rows.append(
            {
                "opponent": opponent,
                "map": map_name,
                "seed": seed,
                "episode_index": 0,
                "opportunity_index": opp_idx,
                "selected_z": z,
                "logits": [0.1, 0.2, 0.3, 0.4],
                "probabilities": [0.1, 0.2, 0.3, 0.4],
            }
        )
    return rows


class TestCrossEpisodeMapGrouping(unittest.TestCase):
    def test_unique_seed_per_episode_groups_by_map_not_singleton(self) -> None:
        # 4 episodes share (OP8, map_b) but each has a unique seed (the real eval
        # protocol). Before the fix these became 4 singleton cells; now they form
        # one (opponent, map) cell with 4 permutable episodes.
        traces: list[dict] = []
        traces += _cell_traces("OP8", "map_b", 15000, [3, 3])
        traces += _cell_traces("OP8", "map_b", 15001, [1, 1])
        traces += _cell_traces("OP8", "map_b", 15002, [3, 1])
        traces += _cell_traces("OP8", "map_b", 15003, [1, 3])
        _mapping, meta = build_cross_episode_shuffled_mapping_from_learned_traces(
            traces, latent_k=4, switch_cadence=32
        )
        self.assertEqual(meta["cell_count"], 1)
        self.assertTrue(meta["can_reassign"])
        self.assertGreater(meta["reassigned_episode_count"], 0)
        self.assertTrue(meta["episode_histogram_preserved"])

    def test_distinct_maps_form_distinct_cells(self) -> None:
        traces: list[dict] = []
        traces += _cell_traces("OP8", "map_b", 15000, [3, 3])
        traces += _cell_traces("OP8", "map_b", 15001, [1, 1])
        traces += _cell_traces("OP8", "map_b_split", 15100, [3, 3])
        traces += _cell_traces("OP8", "map_b_split", 15101, [1, 1])
        _mapping, meta = build_cross_episode_shuffled_mapping_from_learned_traces(
            traces, latent_k=4, switch_cadence=32
        )
        self.assertEqual(meta["cell_count"], 2)
        self.assertTrue(meta["can_reassign"])

    def test_missing_map_falls_back_to_opponent_grouping(self) -> None:
        # No "map" key -> fall back to (opponent, "") grouping, still non-singleton.
        traces: list[dict] = []
        for seed, seq in ((200, [3, 3]), (201, [1, 1]), (202, [3, 1])):
            for opp_idx, z in enumerate(seq):
                traces.append(
                    {
                        "opponent": "OP9",
                        "seed": seed,
                        "episode_index": 0,
                        "opportunity_index": opp_idx,
                        "selected_z": z,
                        "logits": [0.25, 0.25, 0.25, 0.25],
                        "probabilities": [0.25, 0.25, 0.25, 0.25],
                    }
                )
        _mapping, meta = build_cross_episode_shuffled_mapping_from_learned_traces(
            traces, latent_k=4, switch_cadence=32
        )
        self.assertEqual(meta["cell_count"], 1)
        self.assertTrue(meta["can_reassign"])

    def test_collapsed_router_cannot_reassign(self) -> None:
        # Every episode in the cell has identical signature (collapse) -> the
        # cross-episode shuffle is legitimately a no-op (can_reassign False).
        traces: list[dict] = []
        for seed in (300, 301, 302):
            traces += _cell_traces("OP8", "map_b", seed, [3, 3])
        _mapping, meta = build_cross_episode_shuffled_mapping_from_learned_traces(
            traces, latent_k=4, switch_cadence=32
        )
        self.assertEqual(meta["cell_count"], 1)
        self.assertFalse(meta["can_reassign"])
        self.assertEqual(meta["reassigned_episode_count"], 0)


if __name__ == "__main__":
    unittest.main()
