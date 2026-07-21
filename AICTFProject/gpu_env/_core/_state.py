"""State mixin for BatchedCTFCore — backward-compatibility composition facade.

All implementation lives in :mod:`gpu_env.state.*` sub-mixins.  This module
composes them into the single ``_StateMixin`` class that ``BatchedCTFCore``
inherits, preserving the original MRO and public API without changes.

Canonical import paths for new code:

* ``gpu_env.state.models._CoreStateMixin``          — ``__init__``, RNG helpers
* ``gpu_env.state.allocation._AllocationMixin``     — ``_alloc_state``, macro targets
* ``gpu_env.state.agent_state._AgentStateMixin``    — agent tensors, ``_respawn_side``
* ``gpu_env.state.team_state._TeamStateMixin``      — ``_side_tensors``, mirroring
* ``gpu_env.state.flag_state._FlagStateMixin``      — flag/score tensors
* ``gpu_env.state.episode_state._EpisodeStateMixin``— episode bookkeeping, reset
* ``gpu_env.state.map_state._MapStateMixin``        — obstacle geometry
* ``gpu_env.state.opponent_state._OpponentStateMixin`` — opponent/dynamics API
* ``gpu_env.state.telemetry_state._TelemetryStateMixin`` — metric/nav buffers
* ``gpu_env.state.scratch._ScratchStateMixin``      — runtime buffers, mine state
* ``gpu_env.state.validation._ValidationMixin``     — index/phase utilities
* ``gpu_env.state.snapshots._SnapshotsMixin``       — snapshot policy cache
"""
from __future__ import annotations

from gpu_env.state.agent_state import _AgentStateMixin
from gpu_env.state.allocation import _AllocationMixin
from gpu_env.state.episode_state import _EpisodeStateMixin
from gpu_env.state.flag_state import _FlagStateMixin
from gpu_env.state.map_pool_state import _MapPoolStateMixin
from gpu_env.state.map_state import _MapStateMixin
from gpu_env.state.models import _CoreStateMixin
from gpu_env.state.opponent_state import _OpponentStateMixin
from gpu_env.state.scratch import _ScratchStateMixin
from gpu_env.state.snapshots import _SnapshotsMixin
from gpu_env.state.team_state import _TeamStateMixin
from gpu_env.state.telemetry_state import _TelemetryStateMixin
from gpu_env.state.validation import _ValidationMixin


class _StateMixin(
    _CoreStateMixin,
    _MapPoolStateMixin,
    _AllocationMixin,
    _AgentStateMixin,
    _TeamStateMixin,
    _FlagStateMixin,
    _EpisodeStateMixin,
    _MapStateMixin,
    _OpponentStateMixin,
    _TelemetryStateMixin,
    _ScratchStateMixin,
    _ValidationMixin,
    _SnapshotsMixin,
):
    """GPU CTF state mixin — composed from 12 focused sub-mixins in gpu_env.state.*."""

    pass
