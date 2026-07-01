"""Manifest lifecycle for evaluation runs.

The manifest module only serializes run facts supplied by the orchestrator.  It
intentionally does not calculate probes, episode metrics, gates, or verdicts.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from gpu_env._navigation_telemetry import (
    BLOCKED_DISPLACEMENT_THRESHOLD_CELLS,
    MAP_ROUTE_METADATA_VERSION,
    NAVIGATION_TELEMETRY_VERSION,
    ROUTE_CLASSIFIER_VERSION,
    STUCK_CONSECUTIVE_STEP_WINDOW,
    STUCK_DISPLACEMENT_EPSILON_CELLS,
)
from rl.evaluation.config import MapAwarenessEvaluationConfig
from rl.evaluation.errors import EvaluationManifestError


class ManifestStatus(str, Enum):
    CREATED = "CREATED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    INTERRUPTED = "INTERRUPTED"


_TERMINAL = {ManifestStatus.COMPLETED, ManifestStatus.FAILED, ManifestStatus.INTERRUPTED}
_LEGACY_STATUS = {
    ManifestStatus.CREATED: "created",
    ManifestStatus.RUNNING: "in_progress",
    ManifestStatus.COMPLETED: "completed",
    ManifestStatus.FAILED: "failed",
    ManifestStatus.INTERRUPTED: "interrupted",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_metadata(project_root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty_out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=project_root,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return {"git_commit": commit, "git_dirty": len(dirty_out.strip()) > 0}
    except Exception:
        return {"git_commit": None, "git_dirty": None}


def runtime_metadata() -> dict[str, Any]:
    cuda_version: str | None = None
    if torch.cuda.is_available():
        try:
            cuda_version = torch.version.cuda
        except AttributeError:
            pass
    return {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": cuda_version,
    }


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
        text=True,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(json_safe(payload), handle, indent=2)
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


@dataclass
class EvaluationManifest:
    path: Path
    data: dict[str, Any]
    status: ManifestStatus = ManifestStatus.CREATED
    terminal_written: bool = False
    write_count: int = 0
    terminal_write_count: int = 0
    artifact_inventory: list[str] = field(default_factory=list)

    def write(self) -> None:
        atomic_write_json(self.path, self.data)
        self.write_count += 1

    def set_running(self) -> None:
        self._set_status(ManifestStatus.RUNNING)
        self.write()

    def complete(self, *, artifact_paths: Sequence[Path] = ()) -> None:
        self._terminal(ManifestStatus.COMPLETED, artifact_paths=artifact_paths)

    def fail(self, exc: BaseException, *, artifact_paths: Sequence[Path] = ()) -> None:
        self.data["error"] = f"{type(exc).__name__}: {exc}"
        self._terminal(ManifestStatus.FAILED, artifact_paths=artifact_paths)

    def interrupt(self, *, artifact_paths: Sequence[Path] = ()) -> None:
        self._terminal(ManifestStatus.INTERRUPTED, artifact_paths=artifact_paths)

    def _set_status(self, status: ManifestStatus) -> None:
        self.status = status
        self.data["status"] = _LEGACY_STATUS[status]
        if status in _TERMINAL:
            self.data["completed_at"] = utc_now()

    def _terminal(self, status: ManifestStatus, *, artifact_paths: Sequence[Path]) -> None:
        if self.terminal_written:
            raise EvaluationManifestError("Manifest terminal status was already written.")
        self._set_status(status)
        if artifact_paths:
            self.artifact_inventory = [str(path) for path in artifact_paths]
            self.data["artifact_inventory"] = self.artifact_inventory
        self.write()
        self.terminal_written = True
        self.terminal_write_count += 1


def build_manifest_payload(
    config: MapAwarenessEvaluationConfig,
    *,
    command: Sequence[str],
    project_root: Path,
    baseline_metadata: Mapping[str, Any],
    candidate_metadata: Mapping[str, Any],
    n_agents: int,
) -> dict[str, Any]:
    return {
        "schema_version": 3,
        "telemetry_implementation_version": NAVIGATION_TELEMETRY_VERSION,
        "collision_metric_source": "environment_exact_required",
        "stuck_metric_source": "environment_exact_preferred",
        "route_metric_source": "environment_exact_preferred",
        "stuck_epsilon": STUCK_DISPLACEMENT_EPSILON_CELLS,
        "stuck_consecutive_step_window": STUCK_CONSECUTIVE_STEP_WINDOW,
        "blocked_displacement_threshold": BLOCKED_DISPLACEMENT_THRESHOLD_CELLS,
        "route_classifier_version": ROUTE_CLASSIFIER_VERSION,
        "map_route_metadata_version": MAP_ROUTE_METADATA_VERSION,
        "run_id": str(uuid.uuid4()),
        "started_at": utc_now(),
        "completed_at": None,
        "status": _LEGACY_STATUS[ManifestStatus.RUNNING],
        "command": list(command),
        "baseline": str(config.baseline_checkpoint),
        "candidate": str(config.candidate_checkpoint),
        "baseline_sha256": sha256_file(config.baseline_checkpoint),
        "candidate_sha256": sha256_file(config.candidate_checkpoint),
        "baseline_cnn_channels": config.baseline_cnn_channels,
        "candidate_cnn_channels": config.candidate_cnn_channels,
        "n_agents": n_agents,
        "maps": list(config.maps),
        "opponents": list(config.opponents),
        "episodes": config.episodes_per_cell,
        "seed_start": config.seed_start,
        "max_decision_steps": config.max_decision_steps,
        "device": config.device,
        "baseline_metadata": baseline_metadata,
        "candidate_metadata": candidate_metadata,
        **git_metadata(project_root),
        **runtime_metadata(),
    }


def begin_manifest(
    config: MapAwarenessEvaluationConfig,
    *,
    command: Sequence[str],
    project_root: Path,
    baseline_metadata: Mapping[str, Any],
    candidate_metadata: Mapping[str, Any],
    n_agents: int,
) -> EvaluationManifest:
    payload = build_manifest_payload(
        config,
        command=command,
        project_root=project_root,
        baseline_metadata=baseline_metadata,
        candidate_metadata=candidate_metadata,
        n_agents=n_agents,
    )
    manifest = EvaluationManifest(
        path=config.output_dir / "evaluation_manifest.json",
        data=payload,
        status=ManifestStatus.RUNNING,
    )
    manifest.write()
    return manifest


def complete_manifest(manifest: EvaluationManifest, *, artifact_paths: Sequence[Path] = ()) -> None:
    manifest.complete(artifact_paths=artifact_paths)


def fail_manifest(manifest: EvaluationManifest, exc: BaseException, *, artifact_paths: Sequence[Path] = ()) -> None:
    manifest.fail(exc, artifact_paths=artifact_paths)


def interrupt_manifest(manifest: EvaluationManifest, *, artifact_paths: Sequence[Path] = ()) -> None:
    manifest.interrupt(artifact_paths=artifact_paths)


__all__ = [
    "EvaluationManifest",
    "ManifestStatus",
    "atomic_write_json",
    "begin_manifest",
    "build_manifest_payload",
    "complete_manifest",
    "fail_manifest",
    "git_metadata",
    "interrupt_manifest",
    "json_safe",
    "runtime_metadata",
    "sha256_file",
]
