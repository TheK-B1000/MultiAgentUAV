from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rl.evaluation.config import MapAwarenessEvaluationConfig
from rl.evaluation.errors import EvaluationManifestError
from rl.evaluation.manifest import (
    ManifestStatus,
    begin_manifest,
    fail_manifest,
    interrupt_manifest,
)


class EvaluationManifestTests(unittest.TestCase):
    def _config(self, root: Path) -> MapAwarenessEvaluationConfig:
        baseline = root / "baseline.zip"
        candidate = root / "candidate.zip"
        baseline.write_bytes(b"baseline")
        candidate.write_bytes(b"candidate")
        return MapAwarenessEvaluationConfig(
            baseline_checkpoint=baseline,
            candidate_checkpoint=candidate,
            maps=("map_a_open",),
            opponents=("OP8",),
            episodes_per_cell=1,
            seed_start=7000,
            device="cpu",
            output_dir=root / "out",
            max_decision_steps=4,
            counterfactual_steps=1,
            obs_weight_threshold=1e-4,
            gradient_threshold=0.0,
            counterfactual_kl_threshold=1e-5,
            counterfactual_action_threshold=0.01,
            navigation_improvement_threshold=0.1,
            route_difference_threshold=0.1,
            minimum_win_rate=0.6,
            competence_retention_tolerance=0.05,
            saturation_win_rate=0.95,
        )

    def test_running_manifest_is_written_before_completion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = begin_manifest(
                self._config(root),
                command=["eval.py"],
                project_root=Path.cwd(),
                baseline_metadata={"b": 1},
                candidate_metadata={"c": 2},
                n_agents=2,
            )
            data = json.loads(manifest.path.read_text(encoding="utf-8"))
            self.assertEqual(data["status"], "in_progress")
            self.assertEqual(manifest.status, ManifestStatus.RUNNING)
            self.assertEqual(manifest.write_count, 1)

            manifest.complete(artifact_paths=[root / "out" / "final_report.json"])
            final = json.loads(manifest.path.read_text(encoding="utf-8"))
            self.assertEqual(final["status"], "completed")
            self.assertIn("completed_at", final)
            self.assertEqual(manifest.terminal_write_count, 1)
            with self.assertRaises(EvaluationManifestError):
                manifest.complete()

    def test_failure_and_interruption_are_distinct_terminal_states(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = begin_manifest(
                self._config(root),
                command=["eval.py"],
                project_root=Path.cwd(),
                baseline_metadata={},
                candidate_metadata={},
                n_agents=2,
            )
            fail_manifest(manifest, RuntimeError("boom"))
            failed = json.loads(manifest.path.read_text(encoding="utf-8"))
            self.assertEqual(failed["status"], "failed")
            self.assertEqual(failed["error"], "RuntimeError: boom")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = begin_manifest(
                self._config(root),
                command=["eval.py"],
                project_root=Path.cwd(),
                baseline_metadata={},
                candidate_metadata={},
                n_agents=2,
            )
            interrupt_manifest(manifest)
            interrupted = json.loads(manifest.path.read_text(encoding="utf-8"))
            self.assertEqual(interrupted["status"], "interrupted")
            self.assertNotIn("error", interrupted)

    def test_checkpoint_hashes_are_retained(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = begin_manifest(
                self._config(root),
                command=["eval.py"],
                project_root=Path.cwd(),
                baseline_metadata={},
                candidate_metadata={},
                n_agents=2,
            )
            data = json.loads(manifest.path.read_text(encoding="utf-8"))
            self.assertEqual(len(data["baseline_sha256"]), 64)
            self.assertEqual(len(data["candidate_sha256"]), 64)

    def test_manifest_module_does_not_perform_scientific_calculation(self) -> None:
        source = Path("rl/evaluation/manifest.py").read_text(encoding="utf-8")
        forbidden = ["build_summary(", "aggregate_conditions(", "gradient_probe(", "run_episode("]
        for token in forbidden:
            self.assertNotIn(token, source)


if __name__ == "__main__":
    unittest.main()
