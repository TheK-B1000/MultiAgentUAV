"""Runtime context for an in-progress training run.

:class:`RunContext` is a thin container created at the start of
:func:`rl.training.orchestrator.orchestrate_training_run` and threaded
through lifecycle calls so every lifecycle helper has access to the
run-lock, sidecar paths, and the frozen :class:`RunIdentity` without
reconstructing them.

The dataclass is intentionally mutable (not frozen): the ``rc_path``
field may be filled in after the run-config JSON write.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class RunContext:
    """Holds the mutable state acquired during training run setup.

    Attributes
    ----------
    run_lock:
        The ``_RunLock`` returned by :func:`rl.training.run_artifacts._acquire_run_lock`.
        ``run_lock.release()`` must be called in the training finally-block.
    run_identity:
        The single frozen :class:`RunIdentity` resolved from the live environment
        before any artifact is written. Mandatory for formal production runs.
    rc_path:
        Filesystem path of the ``run_config.json`` sidecar written at startup,
        or ``None`` when not yet written.
    training_manifest_path:
        Path of ``training_manifest.json``, or ``None`` when not yet written.
    """

    run_lock: Any
    run_identity: Any
    rc_path: Optional[str] = field(default=None)
    training_manifest_path: Optional[str] = field(default=None)
