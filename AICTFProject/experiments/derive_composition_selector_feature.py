"""Outcome-blind Phase-2 feature derivation for
PYQUATICUS_4V4_STATE_CONDITIONED_COMPOSITION_SELECTOR_V1_SPEC.json.

Collects the candidate feature (count of live enemies within a declared
radius of blue's own flag, averaged over a trailing window) under BOTH poles
with an IDENTICAL fixed blue composition (2A_2D), on the certified
opponents. No score, win, reward or return field is read, computed, or
printed anywhere in this module -- this is execution-logic/state-distribution
derivation, not an outcome measurement.

Uses the smoke seed family (999xxxxx) deliberately: this is diagnostic
feature derivation, not evidence entering the frozen spec's outcome arms, so
it needs no Rule-9 registry allocation.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_pyquaticus_4v4_role_composition_sweep import (  # noqa: E402
    _action_for_roles,
    composition_roles,
)
from experiments.run_pyquaticus_4v4_team_evaluation import _make_env  # noqa: E402

HORIZON = 240
SMOKE_SEEDS = list(range(99_900_101, 99_900_101 + 16))
RADII_CANDIDATES = (4.0, 6.0, 8.0, 10.0)
COMPOSITION = "2A_2D"  # fixed, identical across poles, to avoid deriving the
# feature from the very composition choice the selector will later make


def local_enemy_pressure(core, radius: float) -> int:
    """Count of alive red agents within `radius` cells of blue's own flag.
    Every input here is already part of blue's legal observation builder
    (gpu_env/_core/_observations.py reads enemy_x/enemy_y and own flag
    position for its grid/distance channels)."""
    fx = float(core.blue_flag_pos[0, 0].item())
    fy = float(core.blue_flag_pos[0, 1].item())
    count = 0
    for i in range(core.red_x.shape[1]):
        if not bool(core.red_alive[0, i].item()):
            continue
        dx = float(core.red_x[0, i].item()) - fx
        dy = float(core.red_y[0, i].item()) - fy
        if (dx * dx + dy * dy) ** 0.5 <= radius:
            count += 1
    return count


def collect_one(pole: str, seed: int) -> dict[str, Any]:
    env, core, genome, live = _make_env(pole, seed)
    try:
        roles = composition_roles(COMPOSITION)
        per_tick: dict[float, list[int]] = {r: [] for r in RADII_CANDIDATES}
        for _ in range(HORIZON):
            for r in RADII_CANDIDATES:
                per_tick[r].append(local_enemy_pressure(core, r))
            env.step_async(_action_for_roles(core, roles))
            _obs, _rew, done, _infos = env.step_wait()
            if bool(np.asarray(done).any()):
                break
        return {
            "pole": pole,
            "seed": seed,
            **{f"mean_pressure_r{int(r)}": float(np.mean(v)) for r, v in per_tick.items()},
            **{f"max_pressure_r{int(r)}": int(np.max(v)) for r, v in per_tick.items()},
        }
    finally:
        env.close()


def _min_misclassification_threshold(a_vals: np.ndarray, b_vals: np.ndarray) -> dict[str, Any]:
    """Pre-declared mechanical criterion: the threshold minimizing pooled
    misclassification over the combined sample (standard two-class empirical
    decision-boundary estimate). Candidate cut points are every midpoint
    between adjacent sorted pooled values -- exhaustive, not hand-picked."""
    a_higher = np.median(a_vals) > np.median(b_vals)
    pooled = np.sort(np.unique(np.concatenate([a_vals, b_vals])))
    candidates = (pooled[:-1] + pooled[1:]) / 2.0
    best_t, best_err = None, np.inf
    for t in candidates:
        if a_higher:
            err = np.mean(a_vals <= t) * len(a_vals) + np.mean(b_vals > t) * len(b_vals)
        else:
            err = np.mean(a_vals >= t) * len(a_vals) + np.mean(b_vals < t) * len(b_vals)
        err /= (len(a_vals) + len(b_vals))
        if err < best_err:
            best_err, best_t = err, float(t)
    return {"threshold": best_t, "pooled_misclassification_rate": float(best_err), "pole_A_higher": bool(a_higher)}


def main() -> int:
    rows = []
    for pole in ("A", "B"):
        for seed in SMOKE_SEEDS:
            rows.append(collect_one(pole, seed))

    raw_path = ROOT / "artifacts/strategic_demand/sppo/composition_selector_feature_derivation_raw_rows.csv"
    import csv as _csv
    with raw_path.open("w", newline="", encoding="utf-8") as fh:
        w = _csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    report: dict[str, Any] = {
        "composition_held_fixed": COMPOSITION,
        "n_seeds_per_pole": len(SMOKE_SEEDS),
        "raw_rows": raw_path.name,
        "threshold_criterion_predeclared": (
            "minimize pooled misclassification rate over the combined A+B sample "
            "(exhaustive scan over midpoints between adjacent sorted pooled values); "
            "declared before this scan was run, applies identically at every radius"
        ),
        "radii": {},
    }
    for r in RADII_CANDIDATES:
        key = f"mean_pressure_r{int(r)}"
        a_vals = np.array([row[key] for row in rows if row["pole"] == "A"])
        b_vals = np.array([row[key] for row in rows if row["pole"] == "B"])
        naive_midpoint = float((np.median(a_vals) + np.median(b_vals)) / 2.0)
        optimal = _min_misclassification_threshold(a_vals, b_vals)
        report["radii"][r] = {
            "A_median": float(np.median(a_vals)), "A_mean": float(a_vals.mean()), "A_std": float(a_vals.std(ddof=1)),
            "B_median": float(np.median(b_vals)), "B_mean": float(b_vals.mean()), "B_std": float(b_vals.std(ddof=1)),
            "A_min": float(a_vals.min()), "A_max": float(a_vals.max()),
            "B_min": float(b_vals.min()), "B_max": float(b_vals.max()),
            "naive_midpoint_threshold": naive_midpoint,
            "optimal_threshold": optimal["threshold"],
            "optimal_pooled_misclassification_rate": optimal["pooled_misclassification_rate"],
            "pole_A_higher_pressure": optimal["pole_A_higher"],
        }
    out = ROOT / "artifacts/strategic_demand/sppo/COMPOSITION_SELECTOR_FEATURE_DERIVATION_TRACE.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
