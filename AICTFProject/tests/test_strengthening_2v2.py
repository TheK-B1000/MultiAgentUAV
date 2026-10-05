"""Pure-logic tests for the frozen 2v2 strengthening runners (no env, no checkpoints)."""
import json
import unittest
from pathlib import Path

import numpy as np

import experiments.strengthening_2v2_common as C
import experiments.run_behavior_signatures_2v2 as B
import experiments.run_z_intervention_2v2 as Z


def _snap(pos, carrying=(0, 0), alive=(1, 1), tagged=(0, 0)):
    return {"pos": np.asarray(pos, np.float32), "alive": np.asarray(alive, bool),
            "tagged": np.asarray(tagged, bool), "carrying": np.asarray(carrying, bool),
            "own_home": np.array([2.0, 5.0]), "enemy_flag": np.array([38.0, 5.0]),
            "enemy_home": np.array([38.0, 5.0]), "cols": np.float32(40)}


class StatsTests(unittest.TestCase):
    def test_holm_step_down_stops_at_first_non_rejection(self):
        h = C.holm({"a": 0.001, "b": 0.02, "c": 0.04})
        self.assertTrue(h["a"]["reject"])
        self.assertTrue(h["b"]["reject"])      # 0.02 <= 0.05/2
        self.assertTrue(h["c"]["reject"])      # last step compares 0.04 with 0.05/1
        h2 = C.holm({"a": 0.03, "b": 0.03, "c": 0.001})
        self.assertTrue(h2["c"]["reject"])     # 0.001 <= 0.05/3
        self.assertFalse(h2["a"]["reject"])    # 0.03 > 0.05/2 -> stop
        self.assertFalse(h2["b"]["reject"])    # nothing after a stop is rejected

    def test_holm_monotone_adjusted(self):
        h = C.holm({str(i): p for i, p in enumerate([0.01, 0.2, 0.011, 0.5])})
        adj = sorted(v["p_holm"] for v in h.values())
        self.assertEqual(adj, sorted(adj))

    def test_sign_flip_symmetric_and_extreme(self):
        self.assertAlmostEqual(C.sign_flip_p(np.r_[np.ones(10), -np.ones(10)]), 1.0, places=6)
        self.assertLess(C.sign_flip_p(np.full(20, 0.3)), 1e-3)
        self.assertAlmostEqual(C.sign_flip_p(np.array([1.0])), C.sign_flip_p(np.array([-1.0])))

    def test_shards_partition_work(self):
        n = 3
        seen = [i for s in range(n) for i in range(20) if C.in_shard(i, f"{s}/{n}")]
        self.assertEqual(sorted(seen), list(range(20)))
        with self.assertRaises(SystemExit):
            C.in_shard(0, "3/3")


class RecorderTests(unittest.TestCase):
    def test_metrics_and_horizon_rule(self):
        r = B.EpisodeRecorder()
        r.tick(_snap([[2, 5], [30, 5]]), np.array([2, 0]), np.array([1, 1], bool), 0)
        v = r.values()
        self.assertEqual(v["M5_first_grab_tick"], float(B.HORIZON))   # no grab -> horizon
        self.assertEqual(v["grab_occurred"], 0)
        self.assertAlmostEqual(v["M1_home_share"], 0.5)
        self.assertAlmostEqual(v["M2_forward_share"], 0.5)
        self.assertAlmostEqual(v["M4_spacing"], 28.0)

    def test_spacing_undefined_without_two_active(self):
        r = B.EpisodeRecorder()
        r.tick(_snap([[2, 5], [30, 5]], tagged=(0, 1)), np.array([0, 0]), np.array([1, 0], bool), 0)
        self.assertIsNone(r.values()["M4_spacing"])

    def test_first_grab_tick(self):
        r = B.EpisodeRecorder()
        r.tick(_snap([[2, 5], [30, 5]]), np.array([0, 0]), np.array([0, 0], bool), 0)
        r.tick(_snap([[2, 5], [37, 5]], carrying=(0, 1)), np.array([0, 0]), np.array([0, 0], bool), 1)
        self.assertEqual(r.values()["M5_first_grab_tick"], 1.0)


class AnalysisTests(unittest.TestCase):
    def _rows(self, eff_a, eff_b, n=40, noise=0.01):
        rng = np.random.default_rng(0)
        rows = []
        for system in B.SYSTEMS:
            for pole, eff in (("A", eff_a), ("B", eff_b)):
                for s in range(n):
                    base = {m: float(rng.normal(0, noise)) for m in B.TESTED}
                    for strat in ("A", "B"):
                        row = {"system": system, "strategy": strat, "pole": pole, "seed": s,
                               "grab_occurred": 1, **{f"M6_macro{i}_share": 0.2 for i in range(5)}}
                        for m in B.TESTED:
                            row[m] = base[m] + (eff if strat == "A" else 0.0)
                        rows.append(row)
        return rows

    def test_signature_needs_same_sign_on_both_poles(self):
        a = B.analyse(self._rows(0.5, 0.5))
        self.assertEqual(a["Ours"]["n_signatures"], 5)
        self.assertTrue(a["preserved_by_Fully_Shared_z_r"]["M1_home_share"])
        b = B.analyse(self._rows(0.5, -0.5))
        self.assertEqual(b["Ours"]["n_signatures"], 0)

    def test_family_is_exactly_ten_tests_per_system(self):
        a = B.analyse(self._rows(0.5, 0.5))
        self.assertEqual(len(a["Ours"]["tests"]), 10)


class ZTests(unittest.TestCase):
    def test_contrasts_use_seed_as_unit(self):
        rows = []
        for seed in range(10):
            for k in range(1 + seed):          # unequal agent-state counts per seed
                rows.append({"seed": seed, "collect_z": "A", "pole": "A", "role": "ATTACK",
                             "dA_z0": 0.1, "dA_z1": 0.3, "dB_z0": 0.4, "dB_z1": 0.1,
                             "jsd_z0_z1": 0.05, "argmax_flip": 0, "kl_TA_TB": 1.0, "kl_TB_TA": 1.0})
        a = Z.analyse(rows)
        self.assertAlmostEqual(a["primary"]["C_A"]["mean"], 0.2)
        self.assertAlmostEqual(a["primary"]["C_B"]["mean"], 0.3)
        self.assertEqual(a["primary"]["C_A"]["n"], 10)
        self.assertTrue(a["primary"]["z_steers_toward_matching_teacher"])


class SpecTests(unittest.TestCase):
    def test_specs_frozen_and_blocks_match_registry(self):
        from experiments.seed_registry import load
        reg = {b["experiment_id"]: b for b in load()["blocks"]}
        b = C.load_frozen_spec("BEHAVIOR_SIGNATURES_2V2_V1")["design"]["seeds"]["block"]
        self.assertEqual((reg["BEHAVIOR_SIGNATURES_2V2_V1"]["lo"], reg["BEHAVIOR_SIGNATURES_2V2_V1"]["hi"]), tuple(b))
        z = C.load_frozen_spec("Z_INTERVENTION_2V2_V1")["state_collection"]["block"]
        self.assertEqual((reg["Z_INTERVENTION_2V2_V1"]["lo"], reg["Z_INTERVENTION_2V2_V1"]["hi"]), tuple(z))
        r = C.load_frozen_spec("REPLICATION_2V2_DUAL_BRANCH_V1")
        e = reg["REPLICATION_2V2_DUAL_BRANCH_V1_EVAL"]
        self.assertEqual((e["lo"], e["hi"]), tuple(r["evaluation_frozen"]["block"]))
        for rep, sides in r["replicates"]["training_seeds"].items():
            for side, seed in sides.items():
                blk = reg[f"REPLICATION_2V2_{rep.upper()}_{side}_TRAIN"]
                self.assertEqual(blk["lo"], seed)

    def test_smoke_check_refuses_real_seeds(self):
        with self.assertRaises(SystemExit):
            C.check_block("BEHAVIOR_SIGNATURES_2V2_V1", 31000001, 31000002, smoke=True)


if __name__ == "__main__":
    unittest.main()
