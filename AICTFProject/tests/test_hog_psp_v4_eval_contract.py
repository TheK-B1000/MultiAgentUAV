from __future__ import annotations

import hashlib
import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "artifacts" / "strategic_demand" / "sppo" / "HOG_PSP_V4_EVAL_SPEC.json"
FROZEN_PATH = ROOT / "artifacts" / "strategic_demand" / "sppo" / "HOG_PSP_V4_MODEL_FROZEN.json"
EVALUATOR_PATH = ROOT / "experiments" / "eval_hog_psp_v4.py"


def _load_evaluator():
    spec = importlib.util.spec_from_file_location("eval_hog_psp_v4_contract", EVALUATOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class HogPspV4EvalContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
        cls.frozen = json.loads(FROZEN_PATH.read_text(encoding="utf-8"))
        cls.evaluator = _load_evaluator()

    def test_frozen_seed_gate_and_bootstrap_match_v3(self) -> None:
        self.assertEqual(self.spec["status"], "FROZEN_BEFORE_EVAL_IS_OPENED")
        self.assertEqual(self.spec["SEEDS"]["block"], "11400101..11400132")
        self.assertEqual(self.evaluator.EVAL_SEEDS, list(range(11_400_101, 11_400_133)))
        gate = self.spec["PRIMARY_GATE_CROSSOVER"]
        self.assertEqual((gate["n_boot"], gate["alpha"], gate["rng_seed"]),
                         (20_000, 0.05, 7))
        self.assertEqual(gate["criterion"],
                         "BOTH delta_A.lcb95 > 0 AND delta_B.lcb95 > 0")

    def test_terminal_checkpoint_identity_is_exact(self) -> None:
        expected = "e65d701bee2d10cae98220630b62d9a3bfe539bc1630eaadaedadc009574c2f0"
        self.assertEqual(self.spec["MODEL_UNDER_TEST"]["sha256"], expected)
        self.assertEqual(self.frozen["TERMINAL_CHECKPOINT"]["sha256"], expected)
        checkpoint = ROOT / self.frozen["TERMINAL_CHECKPOINT"]["path"]
        self.assertEqual(hashlib.sha256(checkpoint.read_bytes()).hexdigest(), expected)

    def test_mechanism_and_payoff_claims_are_separate(self) -> None:
        claims = self.spec["CLAIMS_SEPARATED_BEFORE_EVAL"]
        self.assertIn("TRAJECTORY_IDENTITY_PARTIAL", claims["trajectory_identity"])
        self.assertEqual(claims["strategic_payoff_crossover"],
                         "UNEVALUATED until this one-shot protocol completes")
        self.assertEqual(self.evaluator.OUT.name, "HOG_PSP_V4_EVAL_RESULT.json")
        self.assertEqual(self.evaluator.ROWS_CSV.name, "hog_psp_v4_eval_rows.csv")

    def test_preflight_accepts_only_frozen_v4_terminal(self) -> None:
        spec, frozen, mechanism, checkpoint = self.evaluator._preflight()
        self.assertEqual(spec["record_id"], "HOG_PSP_V4_EVAL_SPEC")
        self.assertEqual(mechanism["READING"], "TRAJECTORY_IDENTITY_PARTIAL")
        self.assertEqual(mechanism["PRESERVATION_LABEL"], "IDENTITY_DEGRADED")
        self.assertEqual(frozen["TERMINAL_CHECKPOINT"]["sha256"],
                         spec["MODEL_UNDER_TEST"]["sha256"])
        self.assertEqual(checkpoint.name, "final_hog_psp_v4_production.zip")


if __name__ == "__main__":
    unittest.main()
