"""Symmetric-role baseline suite: collector, audit, trainer/evaluator naming and the prepare step.

The four collector properties the PI asked for:
  1. legacy collection is unchanged (asymmetric specs, paths, k, Pole-B action);
  2. symmetric Pole B really goes through the role splice (pi_DB on DEFEND, frozen pi_B on ATTACK);
  3. a missing/wrong pi_DB, wrong k or a defender equal to a teacher fails closed;
  4. the symmetric dataset has its own path and cannot land on (or overwrite) an asymmetric one.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import collect_suite_distillation_states as C  # noqa: E402
from experiments import prepare_symmetric_baselines as P  # noqa: E402

SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
LEGACY_SPECS = {2: "SUITE_DISTILLATION_2V2_SPEC.json", 4: "SUITE_DISTILLATION_4V4_V2_SPEC.json",
                6: "SUITE_DISTILLATION_6V6_SPEC.json"}


class _Model:
    def __init__(self, n, h):
        self.n_agents, self.heads_per_agent = n, h


class _Fake:
    """Deterministic policy that proposes a constant action per head and counts its calls."""

    def __init__(self, value, n=6, h=3):
        self.value, self.calls, self.model = value, 0, _Model(n, h)

    def predict(self, obs, deterministic=True):
        assert deterministic
        self.calls += 1
        return np.full((1, self.model.n_agents * self.model.heads_per_agent), self.value, dtype=np.int64), None


def _pin(c):
    return {"path": f"x/{c}.zip", "sha256": c * 64}


def _sym_act():
    return {"construction": "symmetric",
            "Pole_A": {"pi_D": _pin("a"), "frozen_attack_pi_A": _pin("1")},
            "Pole_B": {"pi_D": _pin("b"), "frozen_attack_pi_B": _pin("2")}}


KL = {"pi_A": _pin("1"), "pi_B": _pin("2")}


class LegacyUnchangedTests(unittest.TestCase):
    def test_existing_specs_are_asymmetric_and_keep_their_k(self):
        for n, name in LEGACY_SPECS.items():
            spec = json.loads((SD / name).read_text(encoding="utf-8"))
            self.assertFalse(C.spec_is_symmetric(spec), name)
            self.assertEqual(tuple(C.check_tag_matches_construction(spec, "V2" if n == 4 else "")), (False, False))
            self.assertEqual(int(spec["ALLOCATOR_locked"]["k_defend"]), C.K_DEFEND_BY_SCALE[n])
        self.assertEqual(C.K_DEFEND_BY_SCALE, {2: 1, 4: 2, 6: 1})

    def test_legacy_paths_unchanged(self):
        self.assertEqual(C._out_dir(6, False), SD / "suite_distillation_6v6" / "states")
        self.assertEqual(C._out_dir(4, False, "V2"), SD / "suite_distillation_4v4_v2" / "states")
        self.assertEqual(C._manifest(6, False).name, "SUITE_DISTILLATION_6V6_DATASET.json")

    def test_legacy_pole_b_is_plain_pi_b(self):
        pi_D, atk, pi_B = _Fake(7), _Fake(5), _Fake(3)
        obs = {"roles": np.array([[0, 1, 1, 1, 1, 1]], dtype=np.float32)}
        act = C.acting_action("B", obs, pi_D, atk, pi_B, None)
        self.assertTrue((np.asarray(act) == 3).all())
        self.assertEqual((pi_D.calls, atk.calls, pi_B.calls), (0, 0, 1))


class SymmetricPoleBSpliceTests(unittest.TestCase):
    def test_pole_b_splices_pi_db_on_defend_and_frozen_pi_b_on_attack(self):
        pi_D, atk, pi_B, pi_DB = _Fake(7), _Fake(5), _Fake(3), _Fake(9)
        roles = np.array([[0, 1, 1, 0, 1, 1]], dtype=np.float32)          # k=2 defenders: slots 0, 3
        act = np.asarray(C.acting_action("B", {"roles": roles}, pi_D, atk, pi_B, pi_DB)).reshape(6, 3)
        for i in range(6):
            self.assertTrue((act[i] == (9 if roles[0, i] < 0.5 else 3)).all(), i)
        self.assertEqual((pi_D.calls, atk.calls, pi_B.calls, pi_DB.calls), (0, 0, 1, 1))

    def test_pole_a_unchanged_by_symmetric_mode(self):
        pi_D, atk, pi_B, pi_DB = _Fake(7), _Fake(5), _Fake(3), _Fake(9)
        roles = np.array([[1, 0, 1, 1, 0, 1]], dtype=np.float32)
        act = np.asarray(C.acting_action("A", {"roles": roles}, pi_D, atk, pi_B, pi_DB)).reshape(6, 3)
        for i in range(6):
            self.assertTrue((act[i] == (7 if roles[0, i] < 0.5 else 5)).all(), i)
        self.assertEqual((pi_B.calls, pi_DB.calls), (0, 0))


class FailClosedTests(unittest.TestCase):
    def test_valid_symmetric_acting_passes(self):
        C.check_symmetric_acting(_sym_act(), KL)

    def test_missing_pi_db_refuses(self):
        a = _sym_act()
        del a["Pole_B"]["pi_D"]
        with self.assertRaises(SystemExit):
            C.check_symmetric_acting(a, KL)

    def test_legacy_shaped_pole_b_refuses_in_symmetric_mode(self):
        a = _sym_act()
        a["Pole_B"] = _pin("2")
        with self.assertRaises(SystemExit):
            C.check_symmetric_acting(a, KL)

    def test_malformed_sha_refuses(self):
        a = _sym_act()
        a["Pole_B"]["pi_D"]["sha256"] = "abc"
        with self.assertRaises(SystemExit):
            C.check_symmetric_acting(a, KL)

    def test_attacker_must_be_the_teacher(self):
        a = _sym_act()
        a["Pole_B"]["frozen_attack_pi_B"] = _pin("3")
        with self.assertRaises(SystemExit):
            C.check_symmetric_acting(a, KL)

    def test_defender_can_never_be_a_teacher(self):
        a = _sym_act()
        a["Pole_B"]["pi_D"] = _pin("2")
        with self.assertRaises(SystemExit):
            C.check_symmetric_acting(a, KL)
        a = _sym_act()
        a["Pole_B"]["pi_D"] = _pin("a")                                   # same as pi_DA
        with self.assertRaises(SystemExit):
            C.check_symmetric_acting(a, KL)

    def test_k_must_be_ceil_n_over_3(self):
        for n, k in ((2, 1), (4, 2), (6, 2)):
            self.assertEqual(C.symmetric_k_defend({"ALLOCATOR_locked": {"k_defend": k}}, n), k)
        with self.assertRaises(SystemExit):
            C.symmetric_k_defend({"ALLOCATOR_locked": {"k_defend": 1}}, 6)

    def test_unknown_construction_refuses(self):
        with self.assertRaises(SystemExit):
            C.spec_is_symmetric({"ACTING_DEPLOYMENT_locked": {"construction": "mirror"}})


class FreshPathTests(unittest.TestCase):
    def test_symmetric_tree_is_separate(self):
        for n in (2, 4, 6):
            sym = C._out_dir(n, False, "SYM")
            self.assertEqual(sym, SD / "suite_distillation_symmetric" / f"{n}v{n}" / "states")
            self.assertNotEqual(sym, C._out_dir(n, False))
            self.assertNotEqual(C._manifest(n, False, "SYM"), C._manifest(n, False))
            self.assertNotEqual(C._spec_path(n, "SYM"), C._spec_path(n))

    def test_tag_and_construction_are_locked_together(self):
        sym = {"ACTING_DEPLOYMENT_locked": {"construction": "symmetric"}}
        with self.assertRaises(SystemExit):                     # symmetric spec on a legacy path
            C.check_tag_matches_construction(sym, "")
        with self.assertRaises(SystemExit):                     # legacy spec on the symmetric path
            C.check_tag_matches_construction({"ACTING_DEPLOYMENT_locked": {}}, "SYM")
        self.assertEqual(tuple(C.check_tag_matches_construction(sym, "SYM")), (True, False))

    def test_student_and_eval_families_are_separate(self):
        from experiments import eval_suite_sharing_crossover as E
        from experiments import run_suite_sharing_distillation as T
        for n in (2, 4, 6):
            self.assertNotEqual(T.spec_path(n, "SYM"), T.spec_path(n))
            self.assertNotEqual(T._paths("generalist", n, "SYM")["out"], T._paths("generalist", n)["out"])
            self.assertEqual(T._paths("generalist", n)["out"], SD / "suite_sharing_std" / f"{n}v{n}" / "generalist")
            self.assertNotEqual(E._spec_path(n, "SYM"), E._spec_path(n))

    def test_audit_families(self):
        from experiments import audit_suite_datasets_cross_scale as AU
        for n in (2, 4, 6):
            self.assertEqual(AU.DATASETS[f"{n}v{n}_sym"][1], C._manifest(n, False, "SYM").name)
            self.assertEqual(AU.K_BY_SCALE[f"{n}v{n}_sym"], P.k_sym(n))
        self.assertEqual({k: AU.K_BY_SCALE[k] for k in ("2v2", "4v4", "6v6")}, {"2v2": 1, "4v4": 2, "6v6": 1})


class PrepareTests(unittest.TestCase):
    def test_blocks_are_disjoint_and_sized(self):
        spans = []
        for n in (2, 4, 6):
            b = P.blocks(n)
            self.assertEqual(b["collection_A"][2] - b["collection_A"][1] + 1, 96)
            self.assertEqual(b["share_encoder"][2] - b["share_encoder"][1] + 1, 2)
            spans += [(lo, hi) for _e, lo, hi in b.values()]
        spans.sort()
        for (a0, a1), (b0, _b1) in zip(spans, spans[1:]):
            self.assertLess(a1, b0)

    def test_blocks_do_not_collide_with_other_registered_blocks(self):
        from experiments import seed_registry as SR
        ours = {e for n in (2, 4, 6) for e, _lo, _hi in P.blocks(n).values()}
        for n in (2, 4, 6):
            for eid, lo, hi in P.blocks(n).values():
                for b in SR.load()["blocks"]:
                    if b["experiment_id"] in ours:
                        continue
                    self.assertTrue(hi < b["lo"] or lo > b["hi"], (eid, b["experiment_id"]))

    def test_frozen_spec_is_write_once(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "S.json"
            P._write_frozen(p, {"a": 1, "utc": "t0"})
            P._write_frozen(p, {"a": 1, "utc": "t1"})           # identical content: accepted
            with self.assertRaises(SystemExit):
                P._write_frozen(p, {"a": 2, "utc": "t2"})

    def test_labels_are_distinct_and_top50_prefixed(self):
        for n in (2, 4, 6):
            lab = P.labels(n)
            self.assertEqual(len(set(lab.values())), 6)
            self.assertTrue(all(v.startswith(f"TOP50_{n}V{n}_SYM_") for v in lab.values()))


class SharingEvalPostHocSeedsTests(unittest.TestCase):
    """The sharing evaluator's post-hoc mode runs exactly the frozen list on a SPENT block."""

    def _spec(self, n=6, tamper=None):
        from experiments.eval_suite_sharing_crossover import _sha
        d = json.loads((SD / "SYMMETRIC_ROLE_TOP50_DIAGNOSTIC_SPEC.json").read_text(encoding="utf-8"))
        e = d["POST_HOC_MATCHED_ROLE_ABLATIONS"][f"TOP50_{n}V{n}_SYMMETRIC_OURS"]
        f = ROOT / e["seed_ids_file"]
        ids = sorted(int(s) for s in json.loads(f.read_text(encoding="utf-8")))
        entry = {"registry_experiment_id": e["registry_experiment_id"], "block": e["block"],
                 "primary_record": e["primary_record"], "seed_ids": ids}
        spec = {"SEEDS": {"registry_experiment_id": e["registry_experiment_id"], "block": e["block"],
                          "n": len(ids), "seed_class": "post_hoc", "seed_ids_file": e["seed_ids_file"],
                          "seed_ids_sha256": _sha(f)},
                "POST_HOC_MATCHED_ROLE_ABLATIONS": {"L": entry}}
        if tamper:
            tamper(spec)
        return spec, ids

    def test_resolves_the_frozen_list(self):
        from experiments.eval_suite_sharing_crossover import resolve_seeds
        spec, ids = self._spec()
        r = resolve_seeds(spec, "L")
        self.assertEqual(r["seeds"], ids)
        self.assertIsNotNone(r["post_hoc"])
        self.assertEqual(len(ids), 50)

    def test_refuses_unlisted_label_and_tampered_list(self):
        from experiments.eval_suite_sharing_crossover import resolve_seeds
        spec, _ = self._spec()
        with self.assertRaises(SystemExit):
            resolve_seeds(spec, "NOT_AUTHORIZED")
        spec, _ = self._spec(tamper=lambda s: s["SEEDS"].update(seed_ids_sha256="0" * 64))
        with self.assertRaises(SystemExit):
            resolve_seeds(spec, "L")
        spec, _ = self._spec(tamper=lambda s: s["POST_HOC_MATCHED_ROLE_ABLATIONS"]["L"]["seed_ids"].pop())
        with self.assertRaises(SystemExit):
            resolve_seeds(spec, "L")


if __name__ == "__main__":
    unittest.main()


class SuiteReadoutTests(unittest.TestCase):
    """The readout's arithmetic on synthetic rows with known answers (no GPU, no real artifacts)."""

    def test_crossovers_generalist_robustness_margin(self):
        import csv as _csv
        import tempfile
        from unittest import mock
        from experiments import symmetric_baseline_suite as S
        seeds = list(range(1, 11))
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)

            def spec_rows(name, a_on_a, b_on_a, a_on_b, b_on_b):
                p = d / f"{name}.csv"
                with p.open("w", newline="", encoding="utf-8") as fh:
                    w = _csv.writer(fh)
                    w.writerow(["policy", "pole", "seed", "blue", "red", "win", "margin"])
                    for pol, pole, wr in (("pi_A", "A", a_on_a), ("pi_B", "A", b_on_a), ("pi_A", "B", a_on_b),
                                          ("pi_B", "B", b_on_b)):
                        for s in seeds:
                            win = int(s <= wr * 10)
                            w.writerow([pol, pole, s, win * 2, 0, win, win * 2])
                return p

            def z_rows(name, cells):
                p = d / f"{name}.csv"
                with p.open("w", newline="", encoding="utf-8") as fh:
                    w = _csv.writer(fh)
                    w.writerow(["z", "pole", "seed", "blue", "red", "win", "margin"])
                    for (z, pole), wr in cells.items():
                        for s in seeds:
                            win = int(s <= wr * 10)
                            w.writerow([z, pole, s, win, 0, win, win])
                return p

            rows = {"specialists_no_role": spec_rows("nr", .5, .5, .5, .5),
                    "ours_asymmetric_old": spec_rows("old", .8, .3, .2, 1.0),
                    "ours_symmetric": spec_rows("sym", .9, .3, .2, .8),
                    "share_encoder": z_rows("se", {(0, "A"): .6, (1, "A"): .4, (0, "B"): .4, (1, "B"): .6}),
                    "fully_shared_z": z_rows("fs", {(0, "A"): .5, (1, "A"): .5, (0, "B"): .5, (1, "B"): .5}),
                    "generalist": z_rows("g", {(0, "A"): .7, (0, "B"): .6})}
            for k in P.ROBUSTNESS:
                rows[f"ours_symmetric_{k.lower()}"] = spec_rows(f"r_{k}", .9, .3, .2, .8)
            suite = S.Suite(6, d / "out", None, 1)
            with mock.patch.object(S.Suite, "rows", lambda self: rows), \
                    mock.patch.object(S.Suite, "seeds", lambda self: seeds):
                ro = suite.write_readout()
            x = ro["crossovers_win"]
            self.assertAlmostEqual(x["ours_symmetric"]["Delta_A"]["mean"], 0.6)
            self.assertAlmostEqual(x["ours_symmetric"]["Delta_B"]["mean"], 0.6)
            self.assertAlmostEqual(x["ours_asymmetric_old"]["Delta_B"]["mean"], 0.8)
            self.assertAlmostEqual(x["share_encoder"]["Delta_A"]["mean"], 0.2)
            self.assertAlmostEqual(x["fully_shared_z"]["Delta_B"]["mean"], 0.0)
            self.assertAlmostEqual(x["generalist"]["Delta_G_A_vs_ours_symmetric"]["mean"], 0.9 - 0.7)
            self.assertAlmostEqual(x["generalist"]["Delta_G_B_vs_ours_symmetric"]["mean"], 0.8 - 0.6)
            for k in P.ROBUSTNESS:                     # identical to nominal -> zero change
                self.assertAlmostEqual(ro["robustness_ours_symmetric"][k.lower()]["change_vs_nominal_Delta_A"]["mean"], 0.0)
            self.assertAlmostEqual(ro["score_margin"]["ours_symmetric"]["Delta_A"]["mean"], 1.2)
            self.assertTrue((d / "out" / "BASELINE_READOUT.md").is_file())


class SymmetricEvaluatorFailClosedTests(unittest.TestCase):
    """PI 2026-10-01: a symmetric evaluation must splice both sides (pi_DB never evaluated alone)."""

    def _args(self, a="x/a.zip", sa="1" * 64, b="x/b.zip", sb="2" * 64):
        from types import SimpleNamespace
        return SimpleNamespace(frozen_attack_path=a, frozen_attack_path_sha256=sa,
                               frozen_attack_path_b=b, frozen_attack_path_b_sha256=sb)

    def test_detects_symmetric_labels_and_entries(self):
        from experiments.eval_specialist_crossover_scaled import is_symmetric_evaluation as S
        self.assertTrue(S("TOP50_2V2_SYMMETRIC_OURS", None))
        self.assertTrue(S("TOP50_6V6_SYM_OURS_DELAY_MEDIUM", None))
        self.assertTrue(S("X", {"frozen_attack_B": {"sha256": "2" * 64}}))
        self.assertTrue(S("X", {"system": "symmetric Ours"}))
        for legacy in ("TOP50_2V2_NOROLE", "STANDARDIZED_6V6_SPLIT_K1_CONFIRMATORY", "STANDARDIZED_6V6_NOISE_DELAY_MEDIUM"):
            self.assertFalse(S(legacy, None), legacy)

    def test_symmetric_requires_both_frozen_attackers(self):
        from experiments.eval_specialist_crossover_scaled import require_symmetric_split as R
        R("TOP50_2V2_SYMMETRIC_OURS", None, self._args())
        for bad in (self._args(b=""), self._args(sb=""), self._args(a=""), self._args(sa="")):
            with self.assertRaises(SystemExit):
                R("TOP50_2V2_SYMMETRIC_OURS", None, bad)

    def test_hashes_must_match_the_spec_entry(self):
        from experiments.eval_specialist_crossover_scaled import require_symmetric_split as R
        e = {"frozen_attack_A": {"sha256": "1" * 64}, "frozen_attack_B": {"sha256": "2" * 64}}
        R("L", e, self._args())
        with self.assertRaises(SystemExit):
            R("L", e, self._args(sb="3" * 64))

    def test_legacy_labels_untouched(self):
        from experiments.eval_specialist_crossover_scaled import require_symmetric_split as R
        R("TOP50_2V2_NOROLE", {"pi_A": {}, "pi_B": {}}, self._args(a="", sa="", b="", sb=""))

    def test_live_diagnostic_spec_entries(self):
        from experiments.eval_specialist_crossover_scaled import is_symmetric_evaluation as S
        d = json.loads((SD / "SYMMETRIC_ROLE_TOP50_DIAGNOSTIC_SPEC.json").read_text(encoding="utf-8"))
        for label, e in d["POST_HOC_MATCHED_ROLE_ABLATIONS"].items():
            self.assertEqual(S(label, e), "SYMMETRIC" in label, label)

    def test_frozen_policy_named_in_split_log(self):
        src = (ROOT / "rl" / "training" / "orchestrator.py").read_text(encoding="utf-8")
        self.assertNotIn('"[SPLIT-ATTACK-DEFEND] frozen pi_A ATTACHED', src)
        # the frozen policy's name is computed from the checkpoint (variable name may change)
        import re
        self.assertRegex(src, r"\[SPLIT-ATTACK-DEFEND\] frozen \{_\w+\} ATTACHED")
