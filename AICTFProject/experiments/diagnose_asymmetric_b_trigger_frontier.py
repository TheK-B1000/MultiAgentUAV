"""Diagnostic follow-up to COMPOSITION_SELECTOR_ASYMMETRIC_B_TRIGGER_CALIBRATION_RESULT.json.

Not a retuning attempt: max_A_FP_tick_rate=0.10 and B_dominant_pass_fraction=0.50
stay exactly as frozen, and this script changes no decision. Its only purpose
is to report the achievable (A_FP, B_TP) trade-off frontier across the same
already-scanned grid, so NO_CONSERVATIVE_B_TRIGGER can be read correctly: is
0.10 narrowly missed or wildly infeasible for this single feature? Reuses the
same deterministic calibration seeds (99900101-99900116) -- no new seeds.

Outcome-blind: no score/win/reward/return/draw touched.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.calibrate_asymmetric_b_trigger import (  # noqa: E402
    CALIB_SEEDS, DWELL_GRID, HYST_GRID, R_BLUE, WINDOW_GRID,
    collect_p_blue, run_state_machine, score, windowed,
)

OUT = ROOT / "artifacts/strategic_demand/sppo/COMPOSITION_SELECTOR_ASYMMETRIC_B_TRIGGER_FRONTIER_DIAGNOSTIC.json"
FP_CAPS = [round(x, 2) for x in np.arange(0.0, 0.55, 0.02)]


def main() -> int:
    print("Recollecting calibration traces (deterministic, same seeds)...", flush=True)
    calib: dict[tuple[str, int], list[int]] = {}
    for pole in ("A", "B"):
        for seed in CALIB_SEEDS:
            calib[(pole, seed)] = collect_p_blue(pole, seed)

    quantiles = np.arange(1.0, 100.0, 1.0)
    all_rows = []
    for w in WINDOW_GRID:
        win = {k: windowed(v, w) for k, v in calib.items()}
        pooled = np.concatenate(list(win.values()))
        candidates = np.unique(np.percentile(pooled, quantiles))
        for m in HYST_GRID:
            for d in DWELL_GRID:
                for thr in candidates:
                    preds = {k: run_state_machine(v, float(thr), m, d) for k, v in win.items()}
                    s = score(preds)
                    all_rows.append({"window": w, "hysteresis": m, "dwell": d, "threshold": float(thr), **s})
        print(f"  W={w} done, {len(all_rows)} rows so far", flush=True)

    frontier = []
    for cap in FP_CAPS:
        survivors = [r for r in all_rows if r["A_FP_tick_rate"] <= cap]
        if not survivors:
            frontier.append({"A_FP_cap": cap, "n_survivors": 0, "max_B_TP": None, "config": None})
            continue
        best = max(survivors, key=lambda r: r["B_TP_tick_rate"])
        frontier.append({
            "A_FP_cap": cap, "n_survivors": len(survivors),
            "max_B_TP": best["B_TP_tick_rate"],
            "B_fraction_episodes_dominant_4A0D": best["B_fraction_episodes_dominant_4A0D"],
            "config": {k: best[k] for k in ("window", "hysteresis", "dwell", "threshold")},
        })

    # also: unconstrained best (ignore A entirely) as an upper-bound sanity check
    global_best = max(all_rows, key=lambda r: r["B_TP_tick_rate"])

    report = {
        "purpose": "trade-off frontier only; changes no frozen decision",
        "frozen_constraint_unaffected": 0.10,
        "n_grid_rows": len(all_rows),
        "frontier_A_FP_cap_vs_max_B_TP": frontier,
        "global_best_B_TP_ignoring_A_entirely": global_best,
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for row in frontier:
        print(f"A_FP<={row['A_FP_cap']:.2f}  n_survivors={row['n_survivors']:4d}  max_B_TP={row['max_B_TP']}")
    print("global best B_TP (A unconstrained):", json.dumps(global_best, indent=2))
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
