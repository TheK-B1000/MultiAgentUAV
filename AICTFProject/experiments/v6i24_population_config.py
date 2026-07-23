"""V6I24 fixed opponent×map cell pressures from V6I21J calibration evidence.

Pressures are computed once, logged, and held fixed through the 25u diagnostic.
No PFSP, Nash, or rotation.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from gpu_env._maps import MAP_B_SPLIT_LANE, MAP_B_SPLIT_LANE_V2, normalize_map_layout

DEFAULT_CALIBRATION_REPORT = Path(
    "artifacts/v6i21J_hardpool_balance_calibration/calibration_report.json"
)

# Training map pool for V6I21J lineage (map_b aliases to map_b_split_lane).
DEFAULT_MAPS = (MAP_B_SPLIT_LANE, MAP_B_SPLIT_LANE_V2)
DEFAULT_OPPONENTS = ("OP8", "OP9", "OP10", "OP11", "OP12")

STEPS_PER_UPDATE = 4 * 256  # n_envs * n_steps
PROBE_UPDATES = (5, 10, 25)
# Cheap rejection filter: hardest cells only, tiny train budget.
MICRO_PROBE_UPDATES = 2
MICRO_OPPONENTS = ("OP11", "OP12")
MICRO_MAPS = (MAP_B_SPLIT_LANE, MAP_B_SPLIT_LANE_V2)
MICRO_EPISODES_PER_CELL = 8


@dataclass(frozen=True)
class CellStat:
    opponent: str
    map_layout: str
    win_rate: float
    blue_score_mean: float
    red_score_mean: float
    episodes: int

    @property
    def key(self) -> tuple[str, str]:
        return (self.opponent, self.map_layout)


@dataclass(frozen=True)
class MemberPressure:
    member_id: int
    label: str
    description: str
    cell_weights: tuple[tuple[str, str, float], ...]  # (opp, map, weight)
    seed_offset: int

    def as_training_cell_distribution(self) -> tuple[tuple[str, str, float], ...]:
        return self.cell_weights


def _normalize(weights: Sequence[float], eps: float = 1e-12) -> list[float]:
    clipped = [max(0.0, float(w)) for w in weights]
    total = sum(clipped)
    if total <= eps:
        n = len(clipped)
        return [1.0 / n] * n if n else []
    return [w / total for w in clipped]


def _ensure_both_maps(
    cells: Sequence[CellStat],
    weights: list[float],
    maps: Sequence[str] = DEFAULT_MAPS,
    floor_frac: float = 0.05,
) -> list[float]:
    """Guarantee each map retains at least floor_frac mass."""
    w = list(weights)
    for m in maps:
        idxs = [i for i, c in enumerate(cells) if c.map_layout == m]
        mass = sum(w[i] for i in idxs)
        if mass < floor_frac and idxs:
            need = floor_frac - mass
            add = need / len(idxs)
            for i in idxs:
                w[i] += add
    return _normalize(w)


def load_calibration_cells(report_path: Path | str = DEFAULT_CALIBRATION_REPORT) -> list[CellStat]:
    path = Path(report_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    cells: list[CellStat] = []
    for row in payload.get("cells", []):
        opp = str(row["opponent"]).upper()
        layout = normalize_map_layout(str(row["map"]))
        cells.append(
            CellStat(
                opponent=opp,
                map_layout=layout,
                win_rate=float(row.get("win_rate", 0.0) or 0.0),
                blue_score_mean=float(row.get("blue_score_mean", 0.0) or 0.0),
                red_score_mean=float(row.get("red_score_mean", 0.0) or 0.0),
                episodes=int(row.get("episodes", 0) or 0),
            )
        )
    if not cells:
        raise ValueError(f"No cells in calibration report: {path}")
    return cells


def build_member_pressures(
    cells: Sequence[CellStat] | None = None,
    *,
    report_path: Path | str = DEFAULT_CALIBRATION_REPORT,
) -> list[MemberPressure]:
    """Fixed pressures π0..π3 from baseline WR / Bernoulli variance proxies."""
    cells = list(cells) if cells is not None else load_calibration_cells(report_path)
    n = len(cells)

    # π0: balanced / uniform
    w0 = _ensure_both_maps(cells, [1.0] * n)

    # π1: failure cells — weight ∝ (1 - WR)^2
    raw1 = [(1.0 - c.win_rate) ** 2 for c in cells]
    w1 = _ensure_both_maps(cells, _normalize(raw1))

    # π2: high-variance / hard cells — Bernoulli var WR*(1-WR) + red-score pressure
    raw2 = [
        max(0.0, c.win_rate * (1.0 - c.win_rate)) + 0.15 * max(0.0, c.red_score_mean)
        for c in cells
    ]
    w2 = _ensure_both_maps(cells, _normalize(raw2))

    # π3: complementary to π1+π2 (still both maps)
    raw3 = [1.0 / (1e-3 + w1[i] + w2[i]) for i in range(n)]
    w3 = _ensure_both_maps(cells, _normalize(raw3))

    def _pack(weights: list[float]) -> tuple[tuple[str, str, float], ...]:
        return tuple((c.opponent, c.map_layout, float(weights[i])) for i, c in enumerate(cells))

    return [
        MemberPressure(
            member_id=0,
            label="balanced",
            description="Uniform over OP8-OP12 x both maps",
            cell_weights=_pack(w0),
            seed_offset=0,
        ),
        MemberPressure(
            member_id=1,
            label="failure_cells",
            description="Weight cells where V6I21J baseline WR is lowest",
            cell_weights=_pack(w1),
            seed_offset=100,
        ),
        MemberPressure(
            member_id=2,
            label="high_variance",
            description="Weight high Bernoulli-variance / high red-score cells",
            cell_weights=_pack(w2),
            seed_offset=200,
        ),
        MemberPressure(
            member_id=3,
            label="complementary",
            description="Complement of failure + high-variance weights",
            cell_weights=_pack(w3),
            seed_offset=300,
        ),
    ]


def pressures_manifest(pressures: Sequence[MemberPressure], *, source: str) -> dict[str, Any]:
    return {
        "protocol": "v6i24_full_policy_population_fixed_cell_pressures",
        "classification": "DIAGNOSTIC",
        "path": "C_fallback_independent_teachers",
        "calibration_source": source,
        "fixed_through_updates": 25,
        "probe_updates": list(PROBE_UPDATES),
        "steps_per_update": STEPS_PER_UPDATE,
        "members": [asdict(p) for p in pressures],
    }


__all__ = [
    "CellStat",
    "MemberPressure",
    "DEFAULT_CALIBRATION_REPORT",
    "DEFAULT_MAPS",
    "DEFAULT_OPPONENTS",
    "PROBE_UPDATES",
    "STEPS_PER_UPDATE",
    "load_calibration_cells",
    "build_member_pressures",
    "pressures_manifest",
]
