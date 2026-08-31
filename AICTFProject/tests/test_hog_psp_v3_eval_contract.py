from __future__ import annotations

import hashlib
import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "artifacts" / "strategic_demand" / "sppo" / "HOG_PSP_V3_EVAL_SPEC.json"
FROZEN_PATH = ROOT / "artifacts" / "strategic_demand" / "sppo" / "HOG_PSP_V3_MODEL_FROZEN.json"
EVALUATOR_PATH = ROOT / "experiments" / "eval_hog_psp_v3.py"


def _load_evaluator():
    module_spec = importlib.util.spec_from_file_location("eval_hog_psp_v3_contract", EVALUATOR_PATH)
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


class HogPspV3EvalContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
        cls.frozen = json.loads(FROZEN_PATH.read_text(encoding="utf-8"))
        cls.evaluator = _load_evaluator()

    def test_frozen_seed_and_bootstrap_contract(self) -> None:
        self.assertEqual(self.spec["status"], "FROZEN_BEFORE_EVAL_IS_OPENED")
        self.assertEqual(self.spec["SEEDS"]["block"], "11300101..11300132")
        self.assertEqual(self.evaluator.EVAL_SEEDS, list(range(11_300_101, 11_300_133)))
        gate = self.spec["PRIMARY_GATE_CROSSOVER"]
        self.assertEqual((gate["n_boot"], gate["alpha"], gate["rng_seed"]), (20_000, 0.05, 7))
        self.assertEqual(gate["criterion"], "BOTH delta_A.lcb95 > 0 AND delta_B.lcb95 > 0")

    def test_terminal_checkpoint_identity_is_exact(self) -> None:
        expected = "9f705eaed43e83ee48662dd95449819d0f239b5733e64184fa00cabd12885a69"
        self.assertEqual(self.spec["MODEL_UNDER_TEST"]["sha256"], expected)
        self.assertEqual(self.frozen["TERMINAL_CHECKPOINT"]["sha256"], expected)
        checkpoint = ROOT / self.frozen["TERMINAL_CHECKPOINT"]["path"]
        self.assertEqual(hashlib.sha256(checkpoint.read_bytes()).hexdigest(), expected)

    def test_claims_and_outputs_are_separate(self) -> None:
        claims = self.spec["CLAIMS_SEPARATED_BEFORE_EVAL"]
        self.assertIn("TRAJECTORY_IDENTITY_CONFIRMED", claims["trajectory_identity"])
        self.assertEqual(claims["strategic_payoff_crossover"],
                         "UNEVALUATED until this one-shot protocol completes")
        self.assertEqual(self.evaluator.OUT.name, "HOG_PSP_V3_EVAL_RESULT.json")
        self.assertEqual(self.evaluator.ROWS_CSV.name, "hog_psp_v3_eval_rows.csv")

    def test_preflight_accepts_only_the_frozen_terminal(self) -> None:
        spec, frozen, checkpoint = self.evaluator._preflight()
        self.assertEqual(spec["record_id"], "HOG_PSP_V3_EVAL_SPEC")
        self.assertEqual(frozen["TERMINAL_CHECKPOINT"]["sha256"],
                         spec["MODEL_UNDER_TEST"]["sha256"])
        self.assertEqual(checkpoint.name, "final_hog_psp_v3_production.zip")


if __name__ == "__main__":
    unittest.main()
