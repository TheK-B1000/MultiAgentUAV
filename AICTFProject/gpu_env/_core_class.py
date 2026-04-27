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

    pass
