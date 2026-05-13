#!/usr/bin/env python3
"""Recompute plug-in MI(z; phase), MI(z; opponent), MI(z; outcome) from E3 step telemetry CSV.

New training runs write ``team_phase``, ``opponent_id``, and ``score_outcome`` per row.
Older CSVs may only have ``game_phase``; we map those labels onto the team-phase taxonomy
for a coarse MI(z; phase) proxy.

Example::

    python plot/analyze_e3_latent_mi.py experiments/hypothesis_runs/.../research_*_e3_steps.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rl.discrete_mi import discrete_mi_plugin  # noqa: E402
from rl.latent_phase_labels import (  # noqa: E402
    OUTCOME_CLASSES,
    TEAM_PHASES,
)

_LEGACY_PHASE_TO_TEAM: dict[str, str] = {
    "neutral": "neutral",
    "blue_attack": "attacking_enemy_flag",
    "blue_defense": "defending_own_flag",
    "both_flags": "neutral",
    "unknown": "neutral",
}


def _z_ids(rows: list[dict[str, str]], k: int) -> np.ndarray:
    out: list[int] = []
    for r in rows:
        zt = int(float(r.get("z_t", -1)))
        if 0 <= zt < k:
            out.append(zt)
        else:
            out.append(-1)
    return np.asarray(out, dtype=np.int64)


def _phase_ids(rows: list[dict[str, str]]) -> np.ndarray:
    out: list[int] = []
    for r in rows:
        label = (r.get("team_phase") or "").strip()
        if not label:
            gp = (r.get("game_phase") or "").strip()
            label = _LEGACY_PHASE_TO_TEAM.get(gp, "neutral")
        try:
            out.append(TEAM_PHASES.index(label))  # type: ignore[arg-type]
        except ValueError:
            out.append(0)
    return np.asarray(out, dtype=np.int64)


def _opponent_ids(rows: list[dict[str, str]]) -> np.ndarray | None:
    if not rows or "opponent_id" not in rows[0]:
        return None
    out: list[int] = []
    for r in rows:
        try:
            out.append(int(float(r.get("opponent_id", -1))))
        except ValueError:
            out.append(-1)
    return np.asarray(out, dtype=np.int64)


def _outcome_ids(rows: list[dict[str, str]]) -> np.ndarray | None:
    if not rows or "score_outcome" not in rows[0]:
        return None
    out: list[int] = []
    for r in rows:
        lab = (r.get("score_outcome") or "").strip().lower()
        try:
            out.append(OUTCOME_CLASSES.index(lab))  # type: ignore[arg-type]
        except ValueError:
            out.append(1)
    return np.asarray(out, dtype=np.int64)


def _infer_k(rows: list[dict[str, str]]) -> int:
    m = 0
    for r in rows:
        try:
            m = max(m, int(float(r["z_t"])) + 1)
        except (KeyError, ValueError):
            continue
    return max(2, m)


def _mi_z_bucket_from_rows(
    rows: list[dict[str, str]],
    z_valid: np.ndarray,
    valid_mask: np.ndarray,
    *,
    key: str,
    n_bucket: int,
    k: int,
) -> float:
    col = np.asarray([int(float(r.get(key, -1))) for r in rows], dtype=np.int64)[valid_mask]
    good = (col >= 0) & (col < n_bucket)
    zz = z_valid[good]
    bb = col[good]
    if zz.size == 0:
        return float("nan")
    joint = np.zeros((k, n_bucket), dtype=np.float64)
    for i in range(zz.size):
        joint[int(zz[i]), int(bb[i])] += 1.0
    return float(discrete_mi_plugin(joint))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("e3_csv", type=Path, help="Path to *_e3_steps.csv (or any E3 telemetry CSV).")
    ap.add_argument("--max-rows", type=int, default=0, help="Optional cap on rows (0 = all).")
    args = ap.parse_args()
    path = args.e3_csv
    if not path.is_file():
        raise SystemExit(f"not a file: {path}")

    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if args.max_rows and len(rows) > args.max_rows:
        rows = rows[: int(args.max_rows)]
    if not rows:
        print("no rows")
        return

    k = _infer_k(rows)
    z = _z_ids(rows, k)
    valid = z >= 0
    z = z[valid]
    if z.size == 0:
        print("no valid z_t rows")
        return

    phase = _phase_ids(rows)[valid]
    if rows and "phase_id" in rows[0] and (rows[0].get("phase_id", "") != ""):
        try:
            phase = np.asarray([int(float(r.get("phase_id", 0))) for r in rows], dtype=np.int64)[valid]
            phase = np.clip(phase, 0, len(TEAM_PHASES) - 1)
        except ValueError:
            pass
    joint_p = np.zeros((k, len(TEAM_PHASES)), dtype=np.float64)
    for i in range(z.size):
        joint_p[int(z[i]), int(phase[i])] += 1.0
    mi_phase = discrete_mi_plugin(joint_p)

    mi_o = float("nan")
    oid = _opponent_ids(rows)
    n_opp_mi = 5
    if oid is not None:
        oid = oid[valid]
        joint_o = np.zeros((k, n_opp_mi), dtype=np.float64)
        for i in range(z.size):
            oi = int(oid[i])
            if 0 <= oi < n_opp_mi:
                joint_o[int(z[i]), oi] += 1.0
        mi_o = discrete_mi_plugin(joint_o)

    mi_y = float("nan")
    yid = _outcome_ids(rows)
    if yid is not None:
        yid = yid[valid]
        joint_y = np.zeros((k, len(OUTCOME_CLASSES)), dtype=np.float64)
        for i in range(z.size):
            yi = int(yid[i])
            if 0 <= yi < len(OUTCOME_CLASSES):
                joint_y[int(z[i]), yi] += 1.0
        mi_y = discrete_mi_plugin(joint_y)

    print(f"file={path}")
    print(f"rows_used={z.size} latent_k={k}")
    print(f"MI(z; phase)_nats={mi_phase:.6f}  (ceiling log {len(TEAM_PHASES)} = {np.log(len(TEAM_PHASES)):.4f})")
    if np.isfinite(mi_o):
        print(f"MI(z; opponent)_nats={mi_o:.6f}  (ceiling log {n_opp_mi} = {np.log(float(n_opp_mi)):.4f})")
    else:
        print("MI(z; opponent)_nats=nan  (missing opponent_id column)")
    if np.isfinite(mi_y):
        print(f"MI(z; outcome)_nats={mi_y:.6f}  (ceiling log 3 = {np.log(3.0):.4f})")
    else:
        print("MI(z; outcome)_nats=nan  (missing score_outcome column)")

    mi_sb = _mi_z_bucket_from_rows(rows, z, valid, key="spread_bucket", n_bucket=3, k=k)
    mi_rb = _mi_z_bucket_from_rows(rows, z, valid, key="role_bucket", n_bucket=7, k=k)
    mi_pb = _mi_z_bucket_from_rows(rows, z, valid, key="pressure_bucket", n_bucket=3, k=k)
    mi_adr = _mi_z_bucket_from_rows(rows, z, valid, key="attack_defense_ratio_bucket", n_bucket=3, k=k)
    if np.isfinite(mi_sb):
        print(f"MI(z; spread_bucket)_nats={mi_sb:.6f}")
    else:
        print("MI(z; spread_bucket)_nats=nan  (missing or empty spread_bucket)")
    if np.isfinite(mi_rb):
        print(f"MI(z; role_bucket)_nats={mi_rb:.6f}")
    else:
        print("MI(z; role_bucket)_nats=nan  (missing or empty role_bucket)")
    if np.isfinite(mi_pb):
        print(f"MI(z; pressure_bucket)_nats={mi_pb:.6f}")
    else:
        print("MI(z; pressure_bucket)_nats=nan  (missing or empty pressure_bucket)")
    if np.isfinite(mi_adr):
        print(f"MI(z; attack_defense_ratio_bucket)_nats={mi_adr:.6f}")
    else:
        print(
            "MI(z; attack_defense_ratio_bucket)_nats=nan  (missing or empty attack_defense_ratio_bucket)"
        )

    if np.isfinite(mi_o) and np.isfinite(mi_phase) and mi_phase > mi_o + 1e-6:
        print(
            "interpretation: MI(z; phase) > MI(z; opponent) — z aligns more with "
            "game-phase / coordination context than opponent identity (paper-safe summary)."
        )


if __name__ == "__main__":
    main()
