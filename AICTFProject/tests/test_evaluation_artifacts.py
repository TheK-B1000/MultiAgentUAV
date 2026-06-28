from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rl.evaluation.artifact_writer import report_text, write_csv


class EvaluationArtifactTests(unittest.TestCase):
    def test_write_csv_preserves_first_seen_field_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows.csv"
            write_csv(path, [{"b": 1, "a": 2}, {"c": 3, "a": 4}])
            header = path.read_text(encoding="utf-8").splitlines()[0]
        self.assertEqual(header, "b,a,c")

    def test_report_text_preserves_verdict_line(self) -> None:
        summary = {"verdict": "INCONCLUSIVE", "gates": {key: {"status": "PASS"} for key in ["obstacle_weights_moved", "obstacle_gradient_connected", "obstacle_counterfactual_effect", "wall_collisions_improved", "blocked_movement_improved", "stuck_behavior_improved", "map_dependent_routes", "hard_pool_competence_retained", "universal_saturation_avoided"]}}
        self.assertIn("VERDICT: INCONCLUSIVE", report_text(summary))


if __name__ == "__main__":
    unittest.main()
