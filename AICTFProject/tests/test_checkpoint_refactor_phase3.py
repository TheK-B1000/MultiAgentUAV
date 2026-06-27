from __future__ import annotations

import ast
import json
import unittest
from dataclasses import asdict
from pathlib import Path

import torch

from rl.custom_ppo import inference
from rl.custom_ppo.checkpoints.migrations import SevenToEightChannelCNNMigration
from rl.custom_ppo.checkpoints.models import (
    CheckpointDescriptor,
    CheckpointLoadReport,
    CheckpointMetadata,
    PolicyArchitecture,
)
from rl.custom_ppo.inference import (
    CustomPPOInferencePolicy,
    load_custom_ppo_checkpoint,
    load_custom_ppo_policy,
    read_custom_ppo_metadata,
)


ROOT = Path(__file__).resolve().parents[1]


def _imports_for(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    return imports


class CheckpointRefactorPhase3Tests(unittest.TestCase):
    def test_existing_inference_imports_still_work(self) -> None:
        self.assertIs(CustomPPOInferencePolicy, inference.CustomPPOInferencePolicy)
        self.assertIs(load_custom_ppo_policy, inference.load_custom_ppo_policy)
        self.assertIs(read_custom_ppo_metadata, inference.read_custom_ppo_metadata)
        self.assertIs(load_custom_ppo_checkpoint, inference.load_custom_ppo_checkpoint)

    def test_facade_stays_thin(self) -> None:
        source = (ROOT / "rl" / "custom_ppo" / "inference.py").read_text(encoding="utf-8")
        self.assertLess(len(source.splitlines()), 250)
        self.assertNotIn("load_state_dict(", source)
        self.assertNotIn("torch.zeros", source)
        self.assertNotIn("ZipFile(", source)

    def test_inference_policy_does_not_import_checkpoint_loader_or_migrations(self) -> None:
        imports = _imports_for(ROOT / "rl" / "custom_ppo" / "inference_policy.py")
        forbidden = {
            "rl.custom_ppo.checkpoints.archive",
            "rl.custom_ppo.checkpoints.loader",
            "rl.custom_ppo.checkpoints.migrations",
        }
        self.assertTrue(forbidden.isdisjoint(imports))

    def test_policy_does_not_import_checkpoint_modules(self) -> None:
        imports = _imports_for(ROOT / "rl" / "custom_ppo" / "policy.py")
        self.assertFalse(any(name.startswith("rl.custom_ppo.checkpoints") for name in imports))

    def test_checkpoint_modules_do_not_import_evaluation_modules(self) -> None:
        for path in (ROOT / "rl" / "custom_ppo" / "checkpoints").glob("*.py"):
            imports = _imports_for(path)
            self.assertFalse(any(name.startswith("plot.") or ".evaluation" in name for name in imports), path.name)

    def test_seven_to_eight_migration_is_deterministic_and_reports_changed_key(self) -> None:
        key = "latent_actor.actor_cnn.conv.0.weight"
        weight = torch.arange(4 * 7 * 3 * 3, dtype=torch.float32).reshape(4, 7, 3, 3)
        state = {
            key: weight.clone(),
            "latent_actor.actor_cnn.conv.0.bias": torch.arange(4, dtype=torch.float32),
            "unrelated.weight": torch.ones(2, 2),
        }
        metadata = CheckpointMetadata(
            format="custom_ppo_latent_cnn_v1",
            model_path=Path("checkpoint.zip"),
            cfg={"use_latent_strategy": True, "latent_k": 4},
            actor_arch="cnn_mlp",
            vec_schema_version=1,
            global_state_dim=170,
            observation_channels=7,
            n_agents=4,
            n_macros=5,
            n_targets=50,
            latent_count=4,
        )
        target = PolicyArchitecture(
            observation_channels=8,
            n_agents=4,
            n_macros=5,
            n_targets=50,
            latent_count=4,
            model_kwargs={},
        )
        migration = SevenToEightChannelCNNMigration()
        self.assertTrue(migration.applies_to(metadata, state, target))

        migrated_a, record_a = migration.apply(metadata, state, target)
        migrated_b, record_b = migration.apply(metadata, state, target)

        self.assertEqual(record_a, record_b)
        self.assertEqual(record_a.changed_keys, (key,))
        self.assertTrue(torch.equal(migrated_a[key], migrated_b[key]))
        self.assertTrue(torch.equal(migrated_a[key][:, :7], weight))
        self.assertTrue(torch.equal(migrated_a[key][:, 7], torch.zeros_like(migrated_a[key][:, 7])))
        self.assertTrue(torch.equal(migrated_a["latent_actor.actor_cnn.conv.0.bias"], state["latent_actor.actor_cnn.conv.0.bias"]))
        self.assertTrue(torch.equal(migrated_a["unrelated.weight"], state["unrelated.weight"]))
        self.assertTrue(torch.equal(state[key], weight))

    def test_load_report_serializes_to_json_safe_dict(self) -> None:
        descriptor = CheckpointDescriptor(
            path=Path("checkpoint.zip"),
            sha256="0" * 64,
            size_bytes=123,
            schema_version=None,
            policy_version="test",
            observation_channels=8,
            n_agents=4,
            n_macros=5,
            n_targets=50,
            latent_count=4,
        )
        report = CheckpointLoadReport(
            descriptor=descriptor,
            migrations=(),
            missing_keys=(),
            unexpected_keys=(),
            behavioral_equivalence=None,
            device="cpu",
            loaded_at="2026-06-27T00:00:00+00:00",
            torch_version="test",
        )
        encoded = json.dumps(report.to_json_dict(), sort_keys=True)
        self.assertIn('"sha256"', encoded)
        self.assertEqual(asdict(report)["descriptor"]["sha256"], "0" * 64)


if __name__ == "__main__":
    unittest.main()
