from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .models import CheckpointDescriptor, CheckpointMetadata


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def describe_checkpoint(path: str | Path, metadata: CheckpointMetadata) -> CheckpointDescriptor:
    p = Path(path).resolve()
    return CheckpointDescriptor(
        path=p,
        sha256=sha256_file(p),
        size_bytes=int(p.stat().st_size),
        schema_version=metadata.schema_version,
        policy_version=metadata.policy_version,
        observation_channels=metadata.observation_channels,
        n_agents=metadata.n_agents,
        n_macros=metadata.n_macros,
        n_targets=metadata.n_targets,
        latent_count=metadata.latent_count,
    )
