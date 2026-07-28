"""Tactical behavior-tree profiles for scripted opponents OP5 through OP12.

OP6..OP12 are **strategic niches**: incompatible role structures that create
distinct best responses for blue. Physical capabilities stay matched; niches
come from decision structure, not speed dials.

Tag discipline
--------------
* Short tags ``OP6``..``OP12`` are aliases only - they resolve through
  :data:`OPPONENT_ALIASES` to one canonical long name.
* The audited LRO / Summer pool is :data:`LRO_AUDITED_OPPONENT_POOL`
  (exactly one tag per niche). Do not dump every registry key into a pool.
* Extra style names (e.g. ``OP6_DUAL_RUSH``) are separate variants only when
  they have their own profile entry; they are not registered as silent
  duplicates of the audited tags.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, Optional, Tuple

import torch


# Short ID -> single documented long name (no ambiguous plain OP8).
OPPONENT_ALIASES: Dict[str, str] = {
    "OP5": "OP5_RUSHER",
    "OP6": "OP6_IMMEDIATE_DUAL_RUSH",
    "OP7": "OP7_DEEP_FORTRESS",
    "OP8": "OP8_PROTECTED_CARRIER_ESCORT",
    "OP9": "OP9_SPLIT_LANE_FEINT",
    "OP10": "OP10_AGGRESSIVE_INTERCEPTOR",
    "OP11": "OP11_ADAPTIVE_EXPLOITER",
    "OP12": "OP12_LATE_CONVERTER",
}

# Historical long names that share an audited niche (same BT profile).
# Prefer the right-hand tag in new experiments.
OPPONENT_SYNONYMS: Dict[str, str] = {
    "OP6_TURTLE": "OP6_IMMEDIATE_DUAL_RUSH",
    "OP6_DUAL_RUSH": "OP6_IMMEDIATE_DUAL_RUSH",
    "OP7_SWITCHER": "OP7_DEEP_FORTRESS",
    "OP7_FORTRESS": "OP7_DEEP_FORTRESS",
    "OP8_INTERCEPTOR": "OP8_PROTECTED_CARRIER_ESCORT",
    "OP8_ESCORT": "OP8_PROTECTED_CARRIER_ESCORT",
    "OP9_FORTRESS": "OP9_SPLIT_LANE_FEINT",
    "OP9_FEINT": "OP9_SPLIT_LANE_FEINT",
    "OP10_ESCORT": "OP10_AGGRESSIVE_INTERCEPTOR",
    "OP10_INTERCEPTOR": "OP10_AGGRESSIVE_INTERCEPTOR",
    "OP11_BT_BALANCED": "OP11_ADAPTIVE_EXPLOITER",
    "OP11_EXPLOITER": "OP11_ADAPTIVE_EXPLOITER",
    "OP12_COUNTER": "OP12_LATE_CONVERTER",
    "OP12_CONVERTER": "OP12_LATE_CONVERTER",
}

# One clear tag per strategic niche for Summer / LRO payoff matrices.
LRO_AUDITED_OPPONENT_POOL: Tuple[str, ...] = (
    "OP6_IMMEDIATE_DUAL_RUSH",
    "OP7_DEEP_FORTRESS",
    "OP8_PROTECTED_CARRIER_ESCORT",
    "OP9_SPLIT_LANE_FEINT",
    "OP10_AGGRESSIVE_INTERCEPTOR",
    "OP11_ADAPTIVE_EXPLOITER",
    "OP12_LATE_CONVERTER",
)

# Canonical long name -> curriculum level (unique identity per level).
_CANONICAL_TO_LEVEL: Dict[str, int] = {
    "OP5_RUSHER": 5,
    "OP6_IMMEDIATE_DUAL_RUSH": 6,
    "OP7_DEEP_FORTRESS": 7,
    "OP8_PROTECTED_CARRIER_ESCORT": 8,
    "OP9_SPLIT_LANE_FEINT": 9,
    "OP10_AGGRESSIVE_INTERCEPTOR": 10,
    "OP11_ADAPTIVE_EXPLOITER": 11,
    "OP12_LATE_CONVERTER": 12,
}


def canonicalize_opponent_key(opponent_key: str) -> str:
    """Resolve short aliases and historical synonyms to the audited long name."""
    key = str(opponent_key or "").strip().upper()
    if not key:
        return key
    key = OPPONENT_ALIASES.get(key, key)
    key = OPPONENT_SYNONYMS.get(key, key)
    return key


def normalize_bt_level(opponent_key: str) -> Optional[int]:
    """Return curriculum level 5..12 for a scripted opponent key, or None."""
    canon = canonicalize_opponent_key(opponent_key)
    return _CANONICAL_TO_LEVEL.get(canon)


def is_bt_opponent(opponent_key: str) -> bool:
    return normalize_bt_level(opponent_key) is not None


BT_OPPONENT_KEYS: FrozenSet[str] = frozenset(
    set(OPPONENT_ALIASES.keys())
    | set(OPPONENT_ALIASES.values())
    | set(OPPONENT_SYNONYMS.keys())
    | set(_CANONICAL_TO_LEVEL.keys())
)


@dataclass(frozen=True)
class BTProfile:
    """Per-opponent tactical configuration consumed by ``_BTRedMixin``."""

    level: int
    name: str

    enable_flag_retr: bool = True
    enable_escort: bool = False
    enable_intercept: bool = True
    enable_counter: bool = True
    counter_always: bool = False
    counter_when_trailing: bool = True
    enable_defender: bool = True
    enable_2v1: bool = False

    lock_flag_retr: int = 8
    lock_escort: int = 6
    lock_intercept: int = 6
    lock_counter: int = 10
    lock_defender: int = 8
    lock_2v1: int = 4
    lock_attacker: int = 5

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

    enable_mines: bool = False
    mine_defender_lane_frac: float = 0.40
    mine_cooldown_steps: int = 50
    mine_approach_lead_steps: int = 15
    mine_place_radius: float = 1.5
    mine_min_spacing: float = 3.0
    mine_lock_ticks: int = 20

    adaptive_enabled: bool = False


def _profile(level: int, name: str, **kwargs) -> BTProfile:
    return BTProfile(level=level, name=name, **kwargs)


# Level profiles: pairwise-incompatible role gates (decision structure).
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
    # OP6 - Immediate dual assault (TURTLE niche host).
    # Relentless two-lane offense: both reds stay ATTACKER on opposite lanes.
    # Punishes blues that abandon home (RUSH/SPLIT/ESCORT). TURTLE anchors
    # one agent and counters into the empty red rear. No intercept/counter
    # peel — those softens dual-rush scoring and helped SPLIT (dev14-19).
    6: _profile(
        6,
        "OP6_IMMEDIATE_DUAL_RUSH",
        enable_escort=False,
        enable_intercept=False,
        enable_counter=False,
        counter_always=False,
        counter_when_trailing=False,
        enable_defender=True,
        enable_2v1=False,
        enable_mines=False,
        # Defender never fires with Nr=2; kept only for profile shape.
        min_alive_for_defender=3,
        lock_attacker=28,
        lock_intercept=4,
        lock_defender=4,
        lock_counter=4,
        lock_flag_retr=5,
        defender_zone_frac=0.35,
        # No chase-collapse onto one blue; pure flag dual-assault.
        threat_radius=0.0,
        # Wide opposite-lane corridors (see _bt_route_target OP6 branch).
        lane_amplitude_frac=0.42,
        intercept_block_base=0.40,
        intercept_feasibility_ratio=0.45,
        adaptive_enabled=False,
    ),
    # OP7_SWITCHER - Deep fortress (historical tag; identity = fortress).
    7: _profile(
        7,
        "OP7_DEEP_FORTRESS",
        enable_escort=False,
        enable_intercept=True,
        enable_counter=False,
        counter_always=False,
        counter_when_trailing=False,
        enable_defender=True,
        enable_2v1=False,
        enable_mines=True,
        lock_defender=28,
        lock_intercept=18,
        lock_flag_retr=20,
        lock_attacker=10,
        defender_zone_frac=0.05,
        defender_orbit_radius=1.4,
        threat_radius=12.0,
        lane_amplitude_frac=0.16,
        intercept_block_base=0.84,
        intercept_block_trailing_bonus=0.30,
        intercept_feasibility_ratio=0.72,
        late_game_evasion_unlock=True,
        mine_defender_lane_frac=0.55,
        mine_cooldown_steps=24,
        mine_approach_lead_steps=28,
        adaptive_enabled=False,
    ),
    # OP8_INTERCEPTOR - Protected carrier escort (RUSH niche host, Contract A).
    # Matchup contract: brief early window while carrier/protector formation
    # deploys (home defense incomplete); after legal trigger, sticky escort /
    # intercept / counter. Opening gate is OP8-only in _bt_assign_roles
    # (bt_level==8); does not touch OP9. Profile fingerprint unchanged.
    8: _profile(
        8,
        "OP8_PROTECTED_CARRIER_ESCORT",
        enable_escort=True,
        enable_intercept=True,
        enable_counter=True,
        counter_always=False,
        counter_when_trailing=True,
        enable_defender=True,
        enable_2v1=True,
        enable_mines=False,
        escort_interpose=True,
        escort_perpendicular_fallback=True,
        lock_escort=18,
        lock_2v1=12,
        lock_intercept=10,
        lock_counter=12,
        lock_defender=10,
        lock_flag_retr=14,
        threat_radius=9.0,
        lane_amplitude_frac=0.28,
        intercept_block_base=0.55,
        intercept_block_trailing_bonus=0.22,
        defender_zone_frac=0.18,
        adaptive_enabled=False,
    ),
    # OP9_FEINT - Split-lane feint (OP9_FORTRESS is a synonym).
    9: _profile(
        9,
        "OP9_SPLIT_LANE_FEINT",
        enable_escort=False,
        enable_intercept=True,
        enable_counter=True,
        counter_always=False,
        counter_when_trailing=True,
        enable_defender=True,
        enable_2v1=False,
        enable_mines=False,
        lock_attacker=3,
        lock_intercept=3,
        lock_defender=4,
        lock_counter=5,
        lock_flag_retr=4,
        lock_2v1=4,
        threat_radius=8.5,
        lane_amplitude_frac=0.55,
        intercept_block_base=0.48,
        intercept_block_trailing_bonus=0.18,
        defender_zone_frac=0.22,
        intercept_feasibility_ratio=0.88,
        adaptive_enabled=False,
    ),
    # OP10_ESCORT - Aggressive interceptor (historical tag; identity = interceptor).
    10: _profile(
        10,
        "OP10_AGGRESSIVE_INTERCEPTOR",
        enable_escort=False,
        enable_intercept=True,
        enable_counter=False,
        counter_always=False,
        counter_when_trailing=False,
        enable_defender=True,
        enable_2v1=False,
        enable_mines=False,
        lock_intercept=28,
        lock_defender=14,
        lock_flag_retr=16,
        lock_attacker=8,
        threat_radius=11.0,
        lane_amplitude_frac=0.24,
        intercept_block_base=0.88,
        intercept_block_trailing_bonus=0.36,
        intercept_feasibility_ratio=0.70,
        defender_zone_frac=0.12,
        defender_orbit_radius=1.0,
        late_game_evasion_unlock=True,
        adaptive_enabled=False,
    ),
    # OP11_EXPLOITER - Adaptive exploiter.
    11: _profile(
        11,
        "OP11_ADAPTIVE_EXPLOITER",
        enable_escort=True,
        enable_intercept=True,
        enable_counter=True,
        counter_always=True,
        counter_when_trailing=True,
        enable_defender=True,
        enable_2v1=True,
        enable_mines=False,
        escort_interpose=True,
        escort_perpendicular_fallback=True,
        lock_escort=6,
        lock_intercept=6,
        lock_counter=8,
        lock_2v1=5,
        lock_attacker=4,
        lock_defender=6,
        lock_flag_retr=8,
        threat_radius=10.0,
        lane_amplitude_frac=0.36,
        intercept_block_base=0.62,
        intercept_block_trailing_bonus=0.28,
        defender_zone_frac=0.16,
        adaptive_enabled=True,
    ),
    # OP12_COUNTER - Late converter (sticky locks; no 2v1, distinct from OP11).
    12: _profile(
        12,
        "OP12_LATE_CONVERTER",
        enable_escort=True,
        enable_intercept=True,
        enable_counter=True,
        counter_always=True,
        counter_when_trailing=True,
        enable_defender=True,
        enable_2v1=False,  # distinct from OP11 exploiter (2v1 ON)
        enable_mines=False,
        escort_interpose=True,
        escort_perpendicular_fallback=False,
        lock_counter=30,
        lock_escort=14,
        lock_intercept=16,
        lock_flag_retr=18,
        lock_defender=12,
        lock_attacker=8,
        threat_radius=11.0,
        lane_amplitude_frac=0.30,
        intercept_block_base=0.70,
        intercept_block_trailing_bonus=0.34,
        defender_zone_frac=0.14,
        late_game_evasion_unlock=True,
        adaptive_enabled=True,
    ),
}


def profile_for_level(level: int) -> BTProfile:
    return BT_PROFILES[int(level)]


def profile_for_opponent_key(opponent_key: str) -> BTProfile:
    lvl = normalize_bt_level(opponent_key)
    if lvl is None:
        raise KeyError(f"Not a BT opponent key: {opponent_key!r}")
    return profile_for_level(lvl)


def resolve_bt_levels(opponent_keys: Tuple[str, ...] | list[str]) -> list[int]:
    """Map opponent key strings to profile levels; non-BT keys -> 0."""
    out: list[int] = []
    for key in opponent_keys:
        lvl = normalize_bt_level(key)
        out.append(int(lvl) if lvl is not None else 0)
    return out


def role_gate_fingerprint(level: int) -> Tuple[bool, bool, bool, bool, bool, bool]:
    """Structural niche fingerprint (decision structure, not physics)."""
    p = profile_for_level(level)
    return (
        bool(p.enable_escort),
        bool(p.enable_counter),
        bool(p.counter_always),
        bool(p.enable_mines),
        bool(p.enable_2v1),
        bool(p.enable_intercept),
    )


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
        enable_mines=_bool("enable_mines"),
        mine_defender_lane_frac=_scalar("mine_defender_lane_frac", 0.40),
        mine_cooldown_steps=_int("mine_cooldown_steps", 50),
        mine_approach_lead_steps=_int("mine_approach_lead_steps", 15),
        mine_place_radius=_scalar("mine_place_radius", 1.5),
        mine_min_spacing=_scalar("mine_min_spacing", 3.0),
        mine_lock_ticks=_int("mine_lock_ticks", 20),
        adaptive_enabled=_bool("adaptive_enabled"),
    )


__all__ = [
    "BT_OPPONENT_KEYS",
    "BT_PROFILES",
    "BTProfile",
    "LRO_AUDITED_OPPONENT_POOL",
    "OPPONENT_ALIASES",
    "OPPONENT_SYNONYMS",
    "build_profile_tensors",
    "canonicalize_opponent_key",
    "is_bt_opponent",
    "normalize_bt_level",
    "profile_for_level",
    "profile_for_opponent_key",
    "resolve_bt_levels",
    "role_gate_fingerprint",
]
