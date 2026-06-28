"""Composable BatchedCTFCore assembled from focused mixins."""
from __future__ import annotations

from ._core._dynamics import _DynamicsMixin
from ._core._metrics import _MetricsMixin
from ._core._mines import _MinesMixin
from ._core._observations import _ObservationsMixin
from ._core._rewards import _RewardsMixin
from ._core._rules import _RulesMixin
from ._core._scripted_red import _ScriptedRedMixin
from ._core._state import _StateMixin
from ._core._step import _StepMixin
from ._config import GPUFieldConfig


class BatchedCTFCore(
    _StateMixin,
    _DynamicsMixin,
    _ScriptedRedMixin,
    _MinesMixin,
    _RulesMixin,
    _RewardsMixin,
    _MetricsMixin,
    _ObservationsMixin,
    _StepMixin,
):
    """GPU-vectorized CTF core with Aquaticus-profile option."""

    def __init__(
        self,
        cfg: GPUFieldConfig | None = None,
        *,
        n_envs: int | None = None,
        device: str | None = None,
    ) -> None:
        """Create the vectorized core.

        ``BatchedCTFCore(cfg)`` is the canonical runtime path.  The keyword
        overrides keep older tests/tools that called ``BatchedCTFCore(n_envs=...,
        device=...)`` working after the state decomposition.
        """
        cfg = cfg or GPUFieldConfig()
        if n_envs is not None:
            cfg.n_envs = int(n_envs)
        if device is not None:
            cfg.device = str(device)
        super().__init__(cfg)
