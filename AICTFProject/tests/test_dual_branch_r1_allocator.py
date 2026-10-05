"""R1 rescue (DUAL_BRANCH_ROLE_COMPOSITE_V1_R1_SPEC.json): spec-gated allocator rule and output isolation."""
from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl.custom_ppo import split_attack_defend as S  # noqa: E402

SD = ROOT / "artifacts" / "strategic_demand" / "sppo"


class AllocatorRuleTests(unittest.TestCase):
    def test_nearest_third_matches_the_math_definition(self):
        import math
        for n in range(1, 25):
            self.assertEqual(S.nearest_third_k(n), max(1, math.floor(n / 3 + 0.5)), n)
        self.assertEqual([S.nearest_third_k(n) for n in (2, 4, 6, 8)], [1, 1, 2, 3])

    def test_ceil_rule_unchanged(self):
        self.assertEqual([S.ceil_n_over_3(n) for n in (2, 4, 6, 8)], [1, 2, 2, 3])

    def test_spec_without_rule_is_v1(self):
        v1 = json.loads((SD / "DUAL_BRANCH_ROLE_COMPOSITE_V1_SPEC.json").read_text(encoding="utf-8"))
        self.assertNotIn("ALLOCATOR_RULE_locked", v1)
        self.assertEqual([S.dual_branch_k(v1, n) for n in (2, 4, 6)], [1, 2, 2])
        self.assertEqual(S.dual_branch_k({}, 4), 2)

    def test_r1_spec_gives_k1_only_at_4v4(self):
        r1 = json.loads((SD / "DUAL_BRANCH_ROLE_COMPOSITE_V1_R1_SPEC.json").read_text(encoding="utf-8"))
        self.assertIn("FROZEN", r1["status"])
        self.assertEqual([S.dual_branch_k(r1, n) for n in (2, 4, 6, 8)], [1, 1, 2, 3])

    def test_unknown_rule_fails_closed(self):
        with self.assertRaises(ValueError):
            S.dual_branch_k({"ALLOCATOR_RULE_locked": {"rule": "N/2"}}, 4)
        with self.assertRaises(ValueError):
            S.k_for_rule("anything", 4)


class R1RunnerIsolationTests(unittest.TestCase):
    """Importing the R1 runner overrides the V1 driver module only; every output is an R1 path."""

    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("r1_runner", ROOT / "4v4" / "run_dual_branch_4v4_r1.py")
        cls.M = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.M)
        cls.D = cls.M.D

    def test_k_and_spec(self):
        self.assertEqual(self.D.K, 1)
        self.assertTrue(self.D.SPEC.endswith("DUAL_BRANCH_ROLE_COMPOSITE_V1_R1_SPEC.json"))

    def test_no_v1_output_paths(self):
        D = self.D
        for p in (D.LOG, D.STATE, D.MANIFESTS, D.SEAL, D.STAGE4_TEACHERS, D.LEGACY_DONE, D.LOCAL_DONE, D.OWNER):
            self.assertIn("r1", str(p).replace("\\", "/").split("/4v4/")[-1], p)
        self.assertTrue(D.MANIFEST.startswith("4v4/r1/"))
        self.assertNotIn("STAGE4_4V4_TEACHERS_SEALED.json", str(D.STAGE4_TEACHERS).split("r1")[0][-40:])
        for pol, r in D.RUNS.items():
            for key in ("run_dir", "final", "attack"):
                self.assertIn("_dual_branch_r1_k1", r[key])
                self.assertNotIn("_dual_branch_v1", r[key])
            self.assertEqual(r["spec_ck"], getattr(D, pol))          # same sealed 1M foundations as V1

    def test_training_args_differ_from_v1_only_in_k_spec_seed_suffix(self):
        D = self.D
        a = D.train_args("B", 200_000, 27_600_001, "X", "_dual_branch_r1_k1", smoke=False)
        kv = {a[i]: a[i + 1] for i in range(len(a) - 1) if a[i].startswith("--") and not a[i + 1].startswith("--")}
        self.assertEqual(kv["--role-k-defend"], "1")
        self.assertTrue(kv["--dual-branch-spec"].endswith("_R1_SPEC.json"))
        self.assertEqual(kv["--total-timesteps"], "200000")
        self.assertEqual(kv["--defend-teacher-lambda"], "0.1")
        self.assertEqual(kv["--defend-teacher-decay-end-step"], "150000")
        self.assertEqual(kv["--load-path"], D.B)

    def test_eval_uses_fresh_block_and_k1(self):
        a = self.M.eval_args.__code__.co_consts
        self.assertEqual(self.M.SEED_BASE, 27_700_001)
        self.assertEqual(self.M.N_SEEDS, 128)


if __name__ == "__main__":
    unittest.main()
