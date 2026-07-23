"""V6I26 phase-pod training runtime: inject exclusive scenarios after reset."""
from __future__ import annotations

from typing import Any

import numpy as np

from experiments.v6i26_phase_pods import PHASE_POD_IDS, apply_phase_pod_scenario


def attach_phase_pod_hooks(env: Any, trainer: Any) -> None:
    """Install after-reset scenario injection when ``cfg.phase_pod_id`` is set."""
    pod_id = str(getattr(getattr(trainer, "cfg", None), "phase_pod_id", "") or "").strip().lower()
    if not pod_id:
        return
    if pod_id not in PHASE_POD_IDS:
        raise ValueError(f"phase_pod_id={pod_id!r} not in {PHASE_POD_IDS}")

    core = getattr(env, "core", None)
    if core is not None and not hasattr(core, "apply_phase_pod_scenario"):

        def _method(pod: str, *, env_indices=None):
            apply_phase_pod_scenario(core, pod, env_indices=env_indices)

        core.apply_phase_pod_scenario = _method  # type: ignore[attr-defined]

    prev_after = getattr(env, "_after_reset_indices_hook", None)

    def _after(done: np.ndarray, infos: list, *, _pod: str = pod_id) -> None:
        idxs = [i for i, d in enumerate(done) if bool(d)]
        if idxs:
            apply_phase_pod_scenario(env.core, _pod, env_indices=idxs)
        if callable(prev_after):
            prev_after(done, infos)

    env._after_reset_indices_hook = _after
    trainer._phase_pod_id = pod_id


__all__ = ["attach_phase_pod_hooks"]
