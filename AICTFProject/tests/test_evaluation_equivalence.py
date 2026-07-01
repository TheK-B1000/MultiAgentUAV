from __future__ import annotations

import json
import unittest
from pathlib import Path


class EvaluationEquivalenceTests(unittest.TestCase):
    def test_recorded_phase10_equivalence_reports_pass(self) -> None:
        paths = [
            Path("artifacts/phase10_equivalence_config_loading_preflight/equivalence_report.json"),
            Path("artifacts/phase10_equivalence_obstacle_probes/equivalence_report.json"),
            Path("artifacts/phase10_equivalence_episode_matched_seed/equivalence_report.json"),
            Path("artifacts/phase10_equivalence_aggregation_gates/equivalence_report.json"),
            Path("artifacts/phase10_equivalence_artifact_writer/equivalence_report.json"),
        ]
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            self.skipTest("Phase 10 equivalence artifacts not present: " + ", ".join(missing))
        for path in paths:
            self.assertTrue(json.loads(path.read_text(encoding="utf-8"))["equivalent"], str(path))


if __name__ == "__main__":
    unittest.main()
