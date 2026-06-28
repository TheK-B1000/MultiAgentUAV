"""Tactical behavior-tree profiles for scripted opponents OP5 through OP12.

Each profile tunes which BT roles fire, how aggressively routes are scored,
and hysteresis / commitment — without duplicating the vectorized BT engine in
``_bt_red.py``.  Higher opponent numbers unlock more coordination features.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, Optional, Tuple

import torch


# Canonical level 5..12 for each opponent alias.
_BT_KEY_TO_LEVEL: Dict[str, int] = {
    "OP5": 5,
    "OP5_RUSHER": 5,
    "OP6": 6,
    "OP6_TURTLE": 6,
    "OP7": 7,
    "OP7_SWITCHER": 7,
    "OP8": 8,
    "OP8_INTERCEPTOR": 8,
    "OP9": 9,
    "OP9_FORTRESS": 9,
    "OP10": 10,
    "OP10_ESCORT": 10,
    "OP11": 11,
    "OP11_BT_BALANCED": 11,
    "OP12": 12,
    "OP12_COUNTER": 12,
}

BT_OPPONENT_KEYS: FrozenSet[str] = frozenset(_BT_KEY_TO_LEVEL.keys())


def normalize_bt_level(opponent_key: str) -> Optional[int]:
    """Return curriculum level 5..12 for a scripted opponent key, or None."""
    return _BT_KEY_TO_LEVEL.get(str(opponent_key or "").strip().upper())


def is_bt_opponent(opponent_key: str) -> bool:
    return normalize_bt_level(opponent_key) is not None


@dataclass(frozen=True)
class BTProfile:
    """Per-opponent tactical configuration consumed by ``_BTRedMixin``."""

    level: int
    name: str

    # Role gates (False = role assignment slot skipped).
    enable_flag_retr: bool = True
    enable_escort: bool = False
    enable_intercept: bool = True
    enable_counter: bool = True
    counter_always: bool = False
    counter_when_trailing: bool = True
    enable_defender: bool = True
    enable_2v1: bool = False

    # Role commitment (ticks before reassignment).
    lock_flag_retr: int = 8
    lock_escort: int = 6
    lock_intercept: int = 6
    lock_counter: int = 10
    lock_defender: int = 8
    lock_2v1: int = 4
    lock_attacker: int = 5

    # Route / threat tuning.
    threat_radius: float = 8.0
    lane_amplitude_frac: float = 0.30
    intercept_block_base: float = 0.50
    intercept_block_trailing_bonus: float = 0.20
    defender_zone_frac: float = 0.25
    defender_orbit_radius: float = 0.0
    escort_perpendicular_fallback: bool = True
    escort_interpose: bool = False
    late_game_evasion_unlock: bool = False
    intercept_feasibility_ratio: float = 0.90
    min_alive_for_defender: int = 2


def _profile(
    level: int,
    name: str,
    **kwargs,
) -> BTProfile:
    return BTProfile(level=level, name=name, **kwargs)


# Curriculum: OP5 simplest situational awareness → OP12 full coordination.
BT_PROFILES: Dict[int, BTProfile] = {
    5: _profile(
        5,
        "OP5_RUSHER",
        enable_escort=False,
        enable_intercept=True,
        enable_counter=True,
        counter_always=False,
        counter_when_trailing=True,
        enable_2v1=False,
        lock_intercept=4,
        lock_attacker=3,
        lane_amplitude_frac=0.22,
        intercept_feasibility_ratio=0.85,
    ),
    6: _profile(
        6,
        "OP6_TURTLE",
        enable_escort=False,
        enable_intercept=True,
        enable_counter=False,
        counter_always=False,
        counter_when_trailing=False,
        enable_2v1=False,
        lock_defender=12,
        lock_intercept=8,
        defender_zone_frac=0.15,
        defender_orbit_radius=1.5,
        threat_radius=7.0,
        lane_amplitude_frac=0.18,
    ),
    7: _profile(
        7,
        "OP7_SWITCHER",
        enable_escort=True,
        enable_intercept=True,
        enable_counter=True,
        counter_always=False,
        counter_when_trailing=True,
        enable_2v1=True,
        lock_escort=5,
        lock_attacker=4,
        lane_amplitude_frac=0.28,
    ),
    8: _profile(
        8,
        "OP8_INTERCEPTOR",
        enable_escort=False,
        enable_intercept=True,
        enable_counter=True,
        counter_always=False,
        counter_when_trailing=True,
        enable_2v1=True,
        lock_intercept=10,
        intercept_block_base=0.50,
        intercept_block_trailing_bonus=0.20,
        intercept_feasibility_ratio=0.88,
    ),
    9: _profile(
        9,
        "OP9_FORTRESS",
        enable_escort=False,
        enable_intercept=True,
        enable_counter=False,
        counter_always=False,
        counter_when_trailing=False,
        enable_2v1=False,
        lock_defender=14,
        defender_orbit_radius=1.0,
        defender_zone_frac=0.10,
        late_game_evasion_unlock=True,
        threat_radius=7.5,
    ),
    10: _profile(
        10,
        "OP10_ESCORT",
        enable_escort=True,
        enable_intercept=True,
        enable_counter=True,
        counter_always=False,
        counter_when_trailing=True,
        enable_2v1=False,
        lock_escort=10,
        escort_interpose=True,
        escort_perpendicular_fallback=False,
    ),
    11: _profile(
        11,
        "OP11_BT_BALANCED",
        enable_escort=True,
        enable_intercept=True,
        enable_counter=True,
        counter_always=False,
        counter_when_trailing=True,
        enable_2v1=True,
        escort_interpose=True,
    ),
    12: _profile(
        12,
        "OP12_COUNTER",
        enable_escort=True,
        enable_intercept=True,
        enable_counter=True,
        counter_always=True,
        counter_when_trailing=True,
        enable_2v1=True,
        lock_counter=14,
        escort_interpose=True,
    ),
}


def profile_for_level(level: int) -> BTProfile:
    return BT_PROFILES[int(level)]


def resolve_bt_levels(opponent_keys: Tuple[str, ...] | list[str]) -> list[int]:
    """Map opponent key strings to profile levels; non-BT keys → 0."""
    out: list[int] = []
    for key in opponent_keys:
        lvl = normalize_bt_level(key)
        out.append(int(lvl) if lvl is not None else 0)
    return out


def build_profile_tensors(
    opponent_keys: Tuple[str, ...] | list[str],
    *,
    device: torch.device,
    batch_size: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """Vectorize profile scalars/bools to ``[B]`` tensors for batched BT."""
    keys = list(opponent_keys)
    B = int(batch_size if batch_size is not None else len(keys))
    if len(keys) != B:
        raise ValueError(f"Expected {B} opponent keys, got {len(keys)}")

    levels = resolve_bt_levels(keys)
    f32 = torch.float32

    def _scalar(field: str, default: float = 0.0) -> torch.Tensor:
        vals = []
        for lvl in levels:
            if lvl <= 0:
                vals.append(default)
            else:
                vals.append(float(getattr(BT_PROFILES[lvl], field)))
        return torch.tensor(vals, dtype=f32, device=device)

    def _bool(field: str) -> torch.Tensor:
        vals = []
        for lvl in levels:
            if lvl <= 0:
                vals.append(False)
            else:
                vals.append(bool(getattr(BT_PROFILES[lvl], field)))
        return torch.tensor(vals, dtype=torch.bool, device=device)

    def _int(field: str, default: int = 0) -> torch.Tensor:
        vals = []
        for lvl in levels:
            if lvl <= 0:
                vals.append(default)
            else:
                vals.append(int(getattr(BT_PROFILES[lvl], field)))
        return torch.tensor(vals, dtype=torch.int32, device=device)

    level_t = torch.tensor(levels, dtype=torch.int32, device=device)
    is_op12 = level_t == 12

    return dict(
        bt_level=level_t,
        is_op12=is_op12,
        enable_flag_retr=_bool("enable_flag_retr"),
        enable_escort=_bool("enable_escort"),
        enable_intercept=_bool("enable_intercept"),
        enable_counter=_bool("enable_counter"),
        counter_always=_bool("counter_always"),
        counter_when_trailing=_bool("counter_when_trailing"),
        enable_defender=_bool("enable_defender"),
        enable_2v1=_bool("enable_2v1"),
        lock_flag_retr=_int("lock_flag_retr", 8),
        lock_escort=_int("lock_escort", 6),
        lock_intercept=_int("lock_intercept", 6),
        lock_counter=_int("lock_counter", 10),
        lock_defender=_int("lock_defender", 8),
        lock_2v1=_int("lock_2v1", 4),
        lock_attacker=_int("lock_attacker", 5),
        threat_radius=_scalar("threat_radius", 8.0),
        lane_amplitude_frac=_scalar("lane_amplitude_frac", 0.30),
        intercept_block_base=_scalar("intercept_block_base", 0.50),
        intercept_block_trailing_bonus=_scalar("intercept_block_trailing_bonus", 0.20),
        defender_zone_frac=_scalar("defender_zone_frac", 0.25),
        defender_orbit_radius=_scalar("defender_orbit_radius", 0.0),
        escort_perpendicular_fallback=_bool("escort_perpendicular_fallback"),
        escort_interpose=_bool("escort_interpose"),
        late_game_evasion_unlock=_bool("late_game_evasion_unlock"),
        intercept_feasibility_ratio=_scalar("intercept_feasibility_ratio", 0.90),
        min_alive_for_defender=_int("min_alive_for_defender", 2),
    )


__all__ = [
    "BT_OPPONENT_KEYS",
    "BT_PROFILES",
    "BTProfile",
    "build_profile_tensors",
    "is_bt_opponent",
    "normalize_bt_level",
    "profile_for_level",
    "resolve_bt_levels",
]
