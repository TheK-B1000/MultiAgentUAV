"""Strategic Demand Searcher genomes: legal 2v2 opponent overlays.

Canonical OP6-OP12 registry entries are never written. A genome names a
legal dispatch key plus a BTProfile overlay and an optional opening-hold
(delayed intent). Default overlay + hold=0 is the parent opponent.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any, Dict

from gpu_env._core._bt_profiles import BTProfile, profile_for_opponent_key

LEGAL_A_BASES = ("OP6", "OP8", "OP9", "OP10", "OP11", "OP12")
ANCHOR_B = "OP7"

OVERLAY_BOOL = ("enable_flag_retr", "enable_intercept", "enable_defender")
OVERLAY_INT = ("lock_attacker", "lock_defender", "min_alive_for_defender")
OVERLAY_FLOAT = ("threat_radius", "defender_zone_frac", "lane_amplitude_frac")
INT_BOUNDS = {
    "lock_attacker": (4, 40),
    "lock_defender": (4, 40),
    "min_alive_for_defender": (2, 3),
    "opening_hold_steps": (0, 80),
}
FLOAT_BOUNDS = {
    "threat_radius": (0.0, 16.0),
    "defender_zone_frac": (0.05, 0.40),
    "lane_amplitude_frac": (0.10, 0.40),
}
HOLD_STEPS = (0, 20, 40, 60, 80)


@dataclass
class SDSGenome:
    genome_id: str
    derived_from: str
    base_opponent: str
    overlay: Dict[str, Any]
    opening_hold_steps: int = 0

    def __post_init__(self) -> None:
        self.base_opponent = str(self.base_opponent).upper()
        self.derived_from = str(self.derived_from).upper()
        self.opening_hold_steps = int(self.opening_hold_steps)
        if self.base_opponent not in LEGAL_A_BASES and self.base_opponent != ANCHOR_B:
            raise ValueError(f"illegal dispatch key {self.base_opponent!r}")
        lo, hi = INT_BOUNDS["opening_hold_steps"]
        if not (lo <= self.opening_hold_steps <= hi):
            raise ValueError("opening_hold_steps out of range")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "genome_id": self.genome_id,
            "derived_from": self.derived_from,
            "base_opponent": self.base_opponent,
            "overlay": dict(self.overlay),
            "opening_hold_steps": int(self.opening_hold_steps),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SDSGenome":
        return cls(
            genome_id=str(d["genome_id"]),
            derived_from=str(d.get("derived_from") or d["base_opponent"]),
            base_opponent=str(d["base_opponent"]),
            overlay=dict(d.get("overlay") or {}),
            opening_hold_steps=int(d.get("opening_hold_steps") or 0),
        )


def canonical_parent(base: str) -> SDSGenome:
    return SDSGenome(
        genome_id=f"SDS_PARENT_{base.upper()}",
        derived_from=base.upper(),
        base_opponent=base.upper(),
        overlay={},
        opening_hold_steps=0,
    )


def overlay_profile(genome: SDSGenome) -> BTProfile:
    parent = profile_for_opponent_key(genome.base_opponent)
    if not genome.overlay:
        return parent
    allowed = {f.name for f in fields(BTProfile)}
    kw = {k: v for k, v in genome.overlay.items() if k in allowed}
    return replace(parent, **kw)


def apply_genome_to_core(core, genome: SDSGenome) -> None:
    """Attach overlay + hold. No-op relative to parent when both are default."""
    has_overlay = bool(genome.overlay)
    hold = int(genome.opening_hold_steps)
    if has_overlay:
        core._bt_profile_override = overlay_profile(genome)
    else:
        core._bt_profile_override = None
    core._sds_opening_hold_steps = hold


def mutate(parent: SDSGenome, rng, *, new_id: str) -> SDSGenome:
    """One-axis mutation. Does not invent illegal dispatch keys.

    Canonical OP6-OP12 registry entries are never written. The child gets a
    new genome_id; ``derived_from`` is the parent genome_id.
    """
    child = SDSGenome.from_dict(parent.to_dict())
    child.genome_id = new_id
    child.derived_from = parent.genome_id
    axis = rng.choice(
        ["base", "hold", "bool", "int", "float"],
        p=[0.15, 0.25, 0.20, 0.20, 0.20],
    )
    overlay = dict(child.overlay)
    if axis == "base":
        child.base_opponent = str(rng.choice(LEGAL_A_BASES))
        overlay = {}
    elif axis == "hold":
        child.opening_hold_steps = int(rng.choice(HOLD_STEPS))
    elif axis == "bool":
        k = str(rng.choice(OVERLAY_BOOL))
        parent_p = profile_for_opponent_key(child.base_opponent)
        cur = overlay[k] if k in overlay else bool(getattr(parent_p, k))
        overlay[k] = (not cur)
    elif axis == "int":
        k = str(rng.choice(OVERLAY_INT))
        lo, hi = INT_BOUNDS[k]
        overlay[k] = int(rng.integers(lo, hi + 1))
    else:
        k = str(rng.choice(OVERLAY_FLOAT))
        lo, hi = FLOAT_BOUNDS[k]
        overlay[k] = float(rng.uniform(lo, hi))
    child.overlay = overlay
    return child


def recombine(a: SDSGenome, b: SDSGenome, rng, *, new_id: str) -> SDSGenome:
    """Mix two legal parents. Base from one, opening-hold from the other."""
    if rng.random() < 0.5:
        src_base, src_hold = a, b
    else:
        src_base, src_hold = b, a
    return SDSGenome(
        genome_id=new_id,
        derived_from=f"{a.genome_id}+{b.genome_id}",
        base_opponent=src_base.base_opponent,
        overlay=dict(src_base.overlay),
        opening_hold_steps=int(src_hold.opening_hold_steps),
    )


def degeneracy_penalty(frac_00: float, mean_total_score: float) -> float:
    return float(
        0.5 * max(0.0, float(frac_00) - 0.25)
        + 0.25 * max(0.0, 0.5 - float(mean_total_score))
    )


def development_eligible(rec: Dict[str, Any], *, promote: float = 0.05) -> bool:
    """Frozen pieces only. Does not authorize confirmation block 2500001."""
    return bool(
        float(rec.get("delta_G", -9.0)) > float(promote)
        and bool(rec.get("precommitment_uncertain"))
        and float(rec.get("J", -9.0)) > 0.0
    )
