"""Generic specialist-qualification pipeline: scientific rules, state machine, integrity, configs.

All pipeline tests run in a temporary root with a fake executor (no training, no evaluation)."""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import specialist_qualification as SQ  # noqa: E402


def _cfg(tmp: Path) -> dict:
    """A FROZEN 4v4-shaped config pointing at fake files inside tmp."""
    c = copy.deepcopy(SQ.load_config(4))
    for side in "AB":
        for sect in (c["existing_specialists"][side], c["candidates"]["parents"][side]):
            p = tmp / sect["path"]
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(f"{sect['path']}".encode())
            sect["sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
    c["existing_pair_qualification"] = {"source": "fresh_block", "label": "SQ_4V4_EXISTING_PAIR", "block": [1, 8]}
    c["blocks"] = {"d_select": [11, 18], "d_gate": [21, 28], "role_dev": [31, 38], "final_confirmation": [41, 48]}
    return c


class FakePipeline(SQ.Pipeline):
    """execute() fabricates outputs: trainings write a final checkpoint; evaluations write sealed rows whose win
    rates come from self.script[label] = {"A@A":..,"B@A":..,"A@B":..,"B@B":..} (win=1 for the first round(p*n) seeds)."""

    def __init__(self, cfg, root, script, workers=3):
        super().__init__(cfg, root=root, workers=workers)
        self.script, self.calls = script, []

    def git_head(self):
        return "test"

    def execute(self, jobs):
        for tag, argv, done in jobs:
            if done():
                continue
            self.calls.append((tag, argv))
            if argv[0].endswith("train_specialist_scale.py"):
                suffix = argv[argv.index("--run-label-suffix") + 1]
                side = argv[argv.index("--policy") + 1]
                f = self.root / f"artifacts/scale_{self.n}v{self.n}_specialists/pi_{side}_specialist_{self.n}v{self.n}{suffix}/ckpts/final_pi_{side}_specialist_{self.n}v{self.n}{suffix}.zip"
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_bytes(f"{tag}".encode())
            else:
                label = argv[argv.index("--label") + 1]
                lo, n = int(argv[argv.index("--seed-base") + 1]), int(argv[argv.index("--n-seeds") + 1])
                self.sd.mkdir(parents=True, exist_ok=True)
                with self.rows_path(label).open("w", newline="", encoding="utf-8") as fh:
                    w = csv.writer(fh)
                    w.writerow(["policy", "pole", "seed", "blue", "red", "win", "margin"])
                    for pol in ("pi_A", "pi_B"):
                        for pole in "AB":
                            p = self.script[label][f"{pol[-1]}@{pole}"]
                            for j, s in enumerate(range(lo, lo + n)):
                                win = int(j < round(p * n))
                                w.writerow([pol, pole, s, win, 0, win, win])
                self.result_path(label).write_text(json.dumps({"status": "SEALED"}), encoding="utf-8")


def _good(a=1.0, b=1.0):
    return {"A@A": a, "B@A": 0.25, "A@B": 0.25, "B@B": b}


class RuleTests(unittest.TestCase):
    def test_select_highest_q(self):
        s = {"A1": {"side": "A", "own": .8, "off": .7, "seed": 1}, "A2": {"side": "A", "own": .7, "off": .3, "seed": 2},
             "B1": {"side": "B", "own": .6, "off": .5, "seed": 3}, "B2": {"side": "B", "own": .9, "off": .2, "seed": 4}}
        self.assertEqual(SQ.select_candidates(s), {"A": "A2", "B": "B2"})

    def test_first_tie_break_intended_pole(self):
        s = {"A1": {"side": "A", "own": .6, "off": .2, "seed": 1}, "A2": {"side": "A", "own": .8, "off": .4, "seed": 2},
             "B1": {"side": "B", "own": .5, "off": .1, "seed": 3}}
        self.assertEqual(SQ.select_candidates(s)["A"], "A2")          # equal Q 0.4 -> higher own

    def test_seed_tie_break(self):
        s = {"A1": {"side": "A", "own": .6, "off": .2, "seed": 9}, "A2": {"side": "A", "own": .6, "off": .2, "seed": 5},
             "B1": {"side": "B", "own": .5, "off": .1, "seed": 3}}
        self.assertEqual(SQ.select_candidates(s)["A"], "A2")          # full tie -> lowest seed

    def test_shared_k_is_one_value_for_both(self):
        for intended in ({0: {"A": .8, "B": .6}, 1: {"A": .7, "B": .9}}, {0: {"A": .5, "B": .5}, 1: {"A": .5, "B": .5}}):
            k = SQ.shared_k(intended, [0, 1])
            self.assertIsInstance(k, int)                               # one k: k_A = k_B by construction
        self.assertEqual(SQ.shared_k({0: {"A": .5, "B": .5}, 1: {"A": .5, "B": .5}}, [0, 1]), 0)   # tie -> smaller

    def test_overlapping_blocks_fail_validation(self):
        c = SQ.load_config(4)
        c = copy.deepcopy(c)
        c["blocks"]["d_gate"] = [29500010, 29500070]
        self.assertTrue(any("overlap" in p for p in SQ.validate_config(c, for_run=False)))


class ConfigTests(unittest.TestCase):
    def test_all_configs_pass_schema(self):
        for n in (2, 4, 6):
            self.assertEqual(SQ.validate_config(SQ.load_config(n), for_run=False), [], n)

    def test_only_4v4_can_run(self):
        self.assertEqual(SQ.validate_config(SQ.load_config(4), for_run=True), [])
        for n in (2, 6):
            probs = SQ.validate_config(SQ.load_config(n), for_run=True)
            self.assertTrue(any("FROZEN" in p for p in probs) and any("menu" in p for p in probs), n)

    def test_4v4_config_equals_frozen_spec(self):
        c = SQ.load_config(4)
        spec = json.loads((ROOT / c["frozen_spec"]).read_text(encoding="utf-8"))
        cp = spec["CANDIDATE_PROCEDURE"]
        self.assertEqual(c["candidates"]["training_seeds"], cp["training_seeds"])
        self.assertEqual(c["candidates"]["parents"], cp["recipe_locked"]["parents"])
        for side in "AB":
            self.assertEqual(["train_specialist_scale.py", *c["candidates"]["recipe"][side]], cp["recipe_locked"][side].split())
        self.assertEqual(c["blocks"]["d_select"], cp["selection_on_D_select"]["block"])
        self.assertEqual(c["blocks"]["d_gate"], cp["gate_on_D_gate"]["block"])
        rs = spec["ROLE_STAGE_unchanged"]
        self.assertEqual(c["blocks"]["role_dev"], rs["role_dev_block"])
        self.assertEqual(c["blocks"]["final_confirmation"], rs["final_confirmation"]["block"])
        self.assertEqual(c["role"]["defender_seeds"], rs["defenders"]["training_seeds"])
        self.assertEqual(c["role"]["menu"], [0, 1])

    def test_6v6_menu_not_inferred(self):
        self.assertIsNone(SQ.load_config(6)["role"]["menu"])


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmpd = tempfile.TemporaryDirectory()
        self.tmp = Path(self.tmpd.name)
        self.cfg = _cfg(self.tmp)

    def tearDown(self):
        self.tmpd.cleanup()

    def _pl(self, script, workers=3):
        return FakePipeline(self.cfg, self.tmp, script, workers=workers)

    def test_existing_pair_passes_skips_candidates(self):
        pl = self._pl({"SQ_4V4_EXISTING_PAIR": _good(), "SQ_4V4_ROLE_DEV_K0": _good(), "SQ_4V4_ROLE_DEV_K1": _good(.5, .5),
                       "SQ_4V4_CONFIRM": _good()})
        pl.run()
        self.assertFalse(any(t.startswith("train_") for t, _ in pl.calls))
        self.assertTrue(pl.rec("CONFIRM_READOUT.json").is_file())

    def test_existing_pair_fails_trains_exactly_3a_3b(self):
        script = {"SQ_4V4_EXISTING_PAIR": {"A@A": .8, "B@A": .4, "A@B": .7, "B@B": .5},
                  **{f"SQ_4V4_SELECT_C{i}": _good() for i in (1, 2, 3)}, "SQ_4V4_GATE": _good(),
                  "SQ_4V4_ROLE_DEV_K0": _good(), "SQ_4V4_ROLE_DEV_K1": _good(.5, .5), "SQ_4V4_CONFIRM": _good()}
        pl = self._pl(script)
        pl.run()
        trains = sorted(t for t, _ in pl.calls if t.startswith("train_"))
        self.assertEqual(trains, ["train_A1", "train_A2", "train_A3", "train_B1", "train_B2", "train_B3"])

    def test_gate_failure_stops(self):
        script = {"SQ_4V4_EXISTING_PAIR": {"A@A": .8, "B@A": .4, "A@B": .7, "B@B": .5},
                  **{f"SQ_4V4_SELECT_C{i}": _good() for i in (1, 2, 3)}, "SQ_4V4_GATE": {"A@A": .8, "B@A": .4, "A@B": .7, "B@B": .5}}
        pl = self._pl(script)
        pl.run()
        self.assertTrue(pl.rec("STOPPED.txt").is_file())
        self.assertFalse(any(t.startswith(("def_", "role_dev", "confirm")) for t, _ in pl.calls))

    def test_gate_pass_enters_role_stage_and_shared_k(self):
        script = {"SQ_4V4_EXISTING_PAIR": {"A@A": .8, "B@A": .4, "A@B": .7, "B@B": .5},
                  **{f"SQ_4V4_SELECT_C{i}": _good() for i in (1, 2, 3)}, "SQ_4V4_GATE": _good(),
                  "SQ_4V4_ROLE_DEV_K0": _good(.5, .5), "SQ_4V4_ROLE_DEV_K1": _good(), "SQ_4V4_CONFIRM": _good()}
        pl = self._pl(script)
        pl.run()
        self.assertIn("def_A", [t for t, _ in pl.calls])
        rec = json.loads(pl.rec("ROLE_SELECTION_SEALED.json").read_text(encoding="utf-8"))
        self.assertEqual(rec["selected_k"], 1)
        confirm = [a for t, a in pl.calls if t == "confirm"][0]
        self.assertIn("--frozen-attack-path", confirm)               # k=1 for A ...
        self.assertIn("--frozen-attack-path-b", confirm)             # ... and the same k=1 for B

    def test_selected_pair_immutable_and_sealed_not_overwritten(self):
        p = self.tmp / "x" / "R.json"
        SQ.write_sealed(p, {"a": 1})
        SQ.write_sealed(p, {"a": 1})                                 # identical: accepted
        with self.assertRaises(SystemExit):
            SQ.write_sealed(p, {"a": 2})

    def test_resume_skips_completed_stages(self):
        script = {"SQ_4V4_EXISTING_PAIR": {"A@A": .8, "B@A": .4, "A@B": .7, "B@B": .5},
                  **{f"SQ_4V4_SELECT_C{i}": _good() for i in (1, 2, 3)}, "SQ_4V4_GATE": _good(),
                  "SQ_4V4_ROLE_DEV_K0": _good(), "SQ_4V4_ROLE_DEV_K1": _good(.5, .5), "SQ_4V4_CONFIRM": _good()}
        self._pl(script).run()
        again = self._pl(script)
        again.run()
        self.assertEqual(again.calls, [])                            # every stage reused, nothing recomputed

    def test_missing_pinned_checkpoint_fails_closed(self):
        (self.tmp / self.cfg["existing_specialists"]["A"]["path"]).write_bytes(b"changed")
        with self.assertRaises(SystemExit):
            self._pl({"SQ_4V4_EXISTING_PAIR": _good()}).run()

    def test_worker_count_does_not_change_identity(self):
        a, b = self._pl({}, workers=1), self._pl({}, workers=8)
        self.assertEqual({k: (v["seed"], v["final"], v["eid"]) for k, v in a.cands.items()},
                         {k: (v["seed"], v["final"], v["eid"]) for k, v in b.cands.items()})
        self.assertEqual(a.train_argv(a.cands["B2"]), b.train_argv(b.cands["B2"]))

    def test_unfrozen_config_refuses_run(self):
        c = copy.deepcopy(self.cfg)
        c["status"] = "PREPARED_NOT_FROZEN"
        with self.assertRaises(SystemExit):
            FakePipeline(c, self.tmp, {}).run()


class FourV4CompatibilityTests(unittest.TestCase):
    def test_real_4v4_sealed_stages_detected(self):
        pl = SQ.Pipeline(SQ.load_config(4))
        st = pl.status()
        for s in ("EXISTING_PAIR_QUALIFICATION", "CANDIDATE_TRAINING", "D_SELECT", "D_GATE", "ROLE_STAGE", "SHARED_K_SELECTION"):
            self.assertTrue(st[s], s)
        sel = json.loads(pl.rec("SELECTION_SEALED.json").read_text(encoding="utf-8"))
        self.assertEqual(sel["selected"], {"A": "A3", "B": "B2"})

    def test_4v4_names_unchanged(self):
        pl = SQ.Pipeline(SQ.load_config(4))
        self.assertEqual(pl.label("CONFIRM"), "SQ_4V4_CONFIRM")
        self.assertEqual(pl.cands["A3"]["final"], "artifacts/scale_4v4_specialists/pi_A_specialist_4v4_sq_candA3/ckpts/final_pi_A_specialist_4v4_sq_candA3.zip")
        self.assertEqual(pl.def_dir("B")["final"], "artifacts/scale_4v4_specialists/pi_B_specialist_4v4_sq_def_k1/ckpts/final_pi_B_specialist_4v4_sq_def_k1.zip")


if __name__ == "__main__":
    unittest.main()
