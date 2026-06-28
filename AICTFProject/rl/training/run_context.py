"""Runtime context for an in-progress training run.

:class:`RunContext` is a thin container created at the start of
:func:`rl.training.orchestrator.orchestrate_training_run` and threaded
through lifecycle calls so every lifecycle helper has access to the
run-lock and sidecar paths without them needing to reconstruct them.

The dataclass is intentionally mutable (not frozen): the ``rc_path``
field may be filled in asynchronously after the run-config JSON write
attempt, even if that write fails gracefully.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    pass  # _RunLock is typed as Any to avoid a hard import from run_artifacts at module level


@dataclass
class RunContext:
    """Holds the mutable state acquired during training run setup.

    Attributes
    ----------
    run_lock:
        The ``_RunLock`` returned by :func:`rl.training.run_artifacts._acquire_run_lock`.
        ``run_lock.release()`` must be called in the training finally-block.
    rc_path:
        Filesystem path of the ``run_config.json`` sidecar written at startup,
        or ``None`` when the write attempt raised an exception (gracefully handled).
    """

    run_lock: Any
    rc_path: Optional[str] = field(default=None)
