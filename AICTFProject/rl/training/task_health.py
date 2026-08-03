"""TASK_HEALTH: is the policy still playing CTF?

SYSTEM_HEALTH answers "is PPO numerically alive?" -- finite losses, nonzero
gradients, valid identities, reloadable checkpoints. Two G0-v2 seeds passed
every one of those checks across 33 checkpoints while having stopped playing the
game entirely: zero flag pickups, zero forward commitment, 100% of decisions
spent on their own half.

Numerical integrity is not competence. This module supplies the second,
independent verdict so a policy that has stopped attacking can never again be
reported as merely HEALTHY.

The panel is deliberately tiny -- a handful of fixed episodes -- because it runs
inside training at every checkpoint. Its validation seeds are kept disjoint from
both training and any evaluation set so it never contaminates a formal result.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Sequence

# Disjoint from training seeds (2,500,00x - 3,200,00x), discovery evaluation
# (9,100,00x) and the collapse diagnostic (9,200,00x).
VALIDATION_SEED_BASE = 9_300_000
VALIDATION_SEEDS = tuple(VALIDATION_SEED_BASE + i for i in range(3))
# All seven admitted opponents, not a subset. The previous 3-opponent panel
# (9 episodes) SATURATED: V5 seed 3100001 won 9/9 from 51k onward and produced
# byte-identical panels for the rest of the run, leaving the attractor check
# blind on its strongest seed. Covering the full admitted mixture restores
# headroom, because the harder opponents keep some episodes winnable-but-lost.
VALIDATION_OPPONENTS = ("OP6", "OP7", "OP8", "OP9", "OP10", "OP11", "OP12")

# A policy that never picks the flag up or never crosses the midline is not
# playing, regardless of how healthy its gradients look.
MIN_PICKUPS = 1
MIN_OFFENSIVE_COMMITMENT = 1e-6
MAX_DEFENSIVE_COMMITMENT = 1.0 - 1e-6


@dataclass
class TaskHealthPanel:
    """Outcome of one validation panel at a single checkpoint."""

    global_step: int
    episodes: int
    pickups: int
    captures_blue: int
    captures_red: int
    drops: int
    wins: int
    offensive_commitment: float
    defensive_commitment: float
    capture_conversion: float | None
    win_rate: float
    net_captures: int
    reasons: list[str] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        return "PASS" if not self.reasons else "FAIL"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["verdict"] = self.verdict
        return d


def evaluate_task_health(rows: Sequence[dict], *, global_step: int) -> TaskHealthPanel:
    """Judge a set of validation episode rows.

    ``rows`` use the same schema the G0-v2 evaluation produces, so the panel and
    the formal evaluation cannot drift apart in how they count a pickup or a
    capture.
    """
    n = max(len(rows), 1)
    pickups = int(sum(r["pickups"] for r in rows))
    cap_b = int(sum(r["captures_blue"] for r in rows))
    cap_r = int(sum(r["captures_red"] for r in rows))
    drops = int(sum(r["drops"] for r in rows))
    wins = int(sum(r["win"] for r in rows))
    off = sum(r["both_forward_frac"] for r in rows) / n
    dfn = sum(r["none_forward_frac"] for r in rows) / n

    reasons: list[str] = []
    if pickups < MIN_PICKUPS:
        reasons.append(f"no flag pickups in {len(rows)} validation episodes")
    if off <= MIN_OFFENSIVE_COMMITMENT:
        reasons.append("offensive_commitment is zero: no agent advanced together")
    if dfn >= MAX_DEFENSIVE_COMMITMENT:
        reasons.append("defensive_commitment is 1.0: no agent ever crossed the midline")

    return TaskHealthPanel(
        global_step=int(global_step),
        episodes=len(rows),
        pickups=pickups,
        captures_blue=cap_b,
        captures_red=cap_r,
        drops=drops,
        wins=wins,
        offensive_commitment=round(off, 6),
        defensive_commitment=round(dfn, 6),
        # Null rather than 0: "never held the flag" is not "never converted it".
        capture_conversion=(round(cap_b / pickups, 4) if pickups > 0 else None),
        win_rate=round(wins / n, 4),
        net_captures=cap_b - cap_r,
        reasons=reasons,
    )


def combined_verdict(system_health_ok: bool, panel: TaskHealthPanel) -> dict[str, str]:
    """Report the two dimensions separately -- never collapse them into one word."""
    return {
        "SYSTEM_HEALTH": "PASS" if system_health_ok else "FAIL",
        "TASK_HEALTH": panel.verdict,
    }


__all__ = [
    "MAX_DEFENSIVE_COMMITMENT",
    "MIN_OFFENSIVE_COMMITMENT",
    "MIN_PICKUPS",
    "VALIDATION_OPPONENTS",
    "VALIDATION_SEEDS",
    "VALIDATION_SEED_BASE",
    "TaskHealthPanel",
    "combined_verdict",
    "evaluate_task_health",
]
