"""Team-level game phase labels for latent strategy telemetry and MI(z; phase).

Phases summarize *situation* from the fixed-size global state (same signal as q_phi / critic),
not opponent identity. See Summer Plan: z encodes team coordination intent; MI(z; phase)
tests that more directly than MI(z; opponent_id).
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

# Order defines integer ids 0..n-1 used in rollout buffers and joint count matrices.
TEAM_PHASES: tuple[str, ...] = (
    "neutral",
    "attacking_enemy_flag",
    "carrying_flag_home",
    "defending_own_flag",
    "enemy_carrying_our_flag",
    "stalemate",
)

# Running score pressure from normalized global state (indices 14–15).
OUTCOME_CLASSES: tuple[str, ...] = ("behind", "tied", "ahead")

_STALEMATE_FRAC_THRESHOLD = 0.75
_NEAR_FLAG_FRAC = 0.22


def team_phase_id_from_global_state(state: object, *, stalemate_frac: float = 0.0) -> int:
    """Map ``GLOBAL_STATE_DIM`` vector + stalemate pressure to :data:`TEAM_PHASES` index.

    Uses the same flag / distance semantics as :func:`rl.global_state.build_global_state_batch`.
    ``stalemate_frac`` should be in ``[0, 1]`` (e.g. ``stalemate_steps / stalemate_max_steps``).
    """
    arr = np.asarray(state, dtype=np.float64).reshape(-1)
    _neutral, _attack, _carry_home, _defend, _enemy_carry, _stale = 0, 1, 2, 3, 4, 5
    if arr.size < 12:
        return _neutral
    sf = float(max(0.0, min(1.0, stalemate_frac)))
    if sf >= _STALEMATE_FRAC_THRESHOLD:
        return _stale

    min_b_rf = float(arr[8])
    min_r_bf = float(arr[9])
    blue_flag_captured = bool(arr[10] > 0.5)
    red_flag_captured = bool(arr[11] > 0.5)

    if blue_flag_captured and red_flag_captured:
        return _neutral
    if blue_flag_captured and not red_flag_captured:
        return _enemy_carry
    if red_flag_captured and not blue_flag_captured:
        if min_b_rf < _NEAR_FLAG_FRAC:
            return _attack
        return _carry_home
    if (not blue_flag_captured) and (not red_flag_captured) and min_r_bf < _NEAR_FLAG_FRAC:
        return _defend
    return _neutral


def team_phase_label_from_global_state(state: object, *, stalemate_frac: float = 0.0) -> str:
    return TEAM_PHASES[team_phase_id_from_global_state(state, stalemate_frac=stalemate_frac)]


def outcome_id_from_global_state(state: object) -> int:
    """Ternary running score state from normalized blue/red score slots (14–15)."""
    arr = np.asarray(state, dtype=np.float64).reshape(-1)
    if arr.size < 16:
        return 1
    b = float(arr[14])
    r = float(arr[15])
    eps = 1e-3
    if b > r + eps:
        return 2
    if b < r - eps:
        return 0
    return 1


def outcome_label_from_global_state(state: object) -> str:
    return OUTCOME_CLASSES[outcome_id_from_global_state(state)]


def joint_counts_z_discrete(
    z: Iterable[int],
    y: Iterable[int],
    *,
    n_z: int,
    n_y: int,
) -> np.ndarray:
    """Accumulate a ``(n_z, n_y)`` joint histogram for plug-in MI."""
    joint = np.zeros((int(n_z), int(n_y)), dtype=np.float64)
    for zi, yi in zip(z, y):
        i = int(zi)
        j = int(yi)
        if 0 <= i < n_z and 0 <= j < n_y:
            joint[i, j] += 1.0
    return joint


__all__ = [
    "TEAM_PHASES",
    "OUTCOME_CLASSES",
    "team_phase_id_from_global_state",
    "team_phase_label_from_global_state",
    "outcome_id_from_global_state",
    "outcome_label_from_global_state",
    "joint_counts_z_discrete",
]
