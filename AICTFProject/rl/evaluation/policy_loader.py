"""Policy loading facade for Phase 10 evaluation orchestration.

This module owns the public Phase 10 name.  The implementation delegates to
``rl.evaluation.checkpoint`` so existing behavior and loader diagnostics stay
unchanged while the monolithic evaluator is decomposed in small slices.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from rl.evaluation.checkpoint import (
    get_conv0_weight,
    get_model,
    load_policy,
    policy_device,
    read_checkpoint_dimensions,
)
from rl.evaluation.errors import EvaluationCheckpointError


@dataclass(frozen=True)
class LoadedEvaluationPolicy:
    """A loaded policy plus the checkpoint facts needed by the evaluator."""

    label: str
    checkpoint_path: str
    policy: Any
    metadata: Mapping[str, Any]
    n_agents: int
    n_macros: int
    n_targets: int
    cnn_channels: int


def load_evaluation_policy(
    label: str,
    checkpoint_path: str,
    *,
    device: str,
    cnn_channels: int,
) -> LoadedEvaluationPolicy:
    """Load one policy and preserve its metadata for manifests/preflight."""
    try:
        metadata, n_agents, n_macros, n_targets = read_checkpoint_dimensions(
            checkpoint_path
        )
        policy = load_policy(
            checkpoint_path,
            device=device,
            num_cnn_channels=cnn_channels,
        )
    except Exception as exc:  # pragma: no cover - tested through concrete cases.
        raise EvaluationCheckpointError(
            f"Failed to load {label} policy from {checkpoint_path}: {exc}"
        ) from exc

    return LoadedEvaluationPolicy(
        label=label,
        checkpoint_path=checkpoint_path,
        policy=policy,
        metadata=metadata,
        n_agents=n_agents,
        n_macros=n_macros,
        n_targets=n_targets,
        cnn_channels=cnn_channels,
    )


__all__ = [
    "LoadedEvaluationPolicy",
    "get_conv0_weight",
    "get_model",
    "load_evaluation_policy",
    "load_policy",
    "policy_device",
    "read_checkpoint_dimensions",
]
