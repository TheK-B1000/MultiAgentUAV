"""Stages A-C of COMPOSITION_SELECTOR_ONLINE_CALIBRATION_AMENDMENT.json.

Outcome-blind. Never reads, computes, or prints score/win/reward/return/draw.
Only the declared pressure-within-4-cells feature, tick index, true pole
label (used only to score classification error, never as a selector input),
and switch counts are touched.
"""
from __future__ import annotations

import csv
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
RADIUS = 4.0  # locked: PYQUATICUS_4V4_STATE_CONDITIONED... radius selection
COMPOSITION = "2A_2D"
CALIB_SEEDS = list(range(99_900_101, 99_900_101 + 16))
HOLDOUT_SEEDS = list(range(99_900_201, 99_900_201 + 16))

WINDOW_GRID = (20, 40, 60, 80)
HYST_GRID = (0.0, 0.05, 0.10, 0.15)
DWELL_GRID = (4, 10, 20, 40)  # 4 = macro_commit_go_to_ticks
TIE_BAND = 0.02


def collect_tick_series(pole: str, seed: int) -> list[int]:
    env, core, genome, live = _make_env(pole, seed)
    try:
        roles = composition_roles(COMPOSITION)
        series = []
        for _ in range(HORIZON):
            fx = float(core.blue_flag_pos[0, 0].item())
            fy = float(core.blue_flag_pos[0, 1].item())
            count = 0
            for i in range(core.red_x.shape[1]):
                if not bool(core.red_alive[0, i].item()):
                    continue
                dx = float(core.red_x[0, i].item()) - fx
                dy = float(core.red_y[0, i].item()) - fy
                if (dx * dx + dy * dy) ** 0.5 <= RADIUS:
                    count += 1
            series.append(count)
            env.step_async(_action_for_roles(core, roles))
            _obs, _rew, done, _infos = env.step_wait()
            if bool(np.asarray(done).any()):
                break
        return series
    finally:
        env.close()


def windowed(series: list[int], w: int) -> np.ndarray:
    arr = np.asarray(series, dtype=np.float64)
    out = np.empty_like(arr)
    csum = np.concatenate([[0.0], np.cumsum(arr)])
    for t in range(len(arr)):
        lo = max(0, t - w + 1)
        out[t] = (csum[t + 1] - csum[lo]) / (t + 1 - lo)
    return out


def min_misclass_threshold(pooled_a: np.ndarray, pooled_b: np.ndarray) -> tuple[float, float]:
    a_higher = np.median(pooled_a) > np.median(pooled_b)
    pooled = np.sort(np.unique(np.concatenate([pooled_a, pooled_b])))
    if len(pooled) < 2:
        return float(pooled[0]) if len(pooled) else 0.0, 1.0
    candidates = (pooled[:-1] + pooled[1:]) / 2.0
    best_t, best_err = None, np.inf
    for t in candidates:
        if a_higher:
            err = (np.sum(pooled_a <= t) + np.sum(pooled_b > t)) / (len(pooled_a) + len(pooled_b))
        else:
            err = (np.sum(pooled_a >= t) + np.sum(pooled_b < t)) / (len(pooled_a) + len(pooled_b))
        if err < best_err:
            best_err, best_t = err, float(t)
    return best_t, float(best_err)


def apply_hysteresis_dwell(
    values: np.ndarray, threshold: float, a_higher: bool, m: float, d: int,
) -> np.ndarray:
    """Returns per-tick predicted label array (1=A-like/2A2D, 0=B-like/4A0D)."""
    upper = threshold + m
    lower = threshold - m
    state = 1 if (values[0] > threshold) == a_higher else 0
    out = np.empty(len(values), dtype=np.int64)
    last_switch = -10_000
    for t, v in enumerate(values):
        want_a = (v > upper) if a_higher else (v < lower)
        want_b = (v < lower) if a_higher else (v > upper)
        if state == 1 and want_b and (t - last_switch) >= d:
            state = 0
            last_switch = t
        elif state == 0 and want_a and (t - last_switch) >= d:
            state = 1
            last_switch = t
        out[t] = state
    return out


def score_config(
    calib: dict[tuple[str, int], list[int]], w: int, m: float, d: int,
) -> dict[str, Any]:
    windows = {k: windowed(v, w) for k, v in calib.items()}
    pooled_a = np.concatenate([windows[k] for k in windows if k[0] == "A"])
    pooled_b = np.concatenate([windows[k] for k in windows if k[0] == "B"])
    threshold, _ = min_misclass_threshold(pooled_a, pooled_b)
    a_higher = bool(np.median(pooled_a) > np.median(pooled_b))

    errs, switches_per_ep = [], []
    for (pole, _seed), wv in windows.items():
        pred = apply_hysteresis_dwell(wv, threshold, a_higher, m, d)
        true_label = 1 if pole == "A" else 0
        errs.append(np.mean(pred != true_label))
        switches_per_ep.append(int(np.sum(np.diff(pred) != 0)))
    return {
        "window": w, "hysteresis": m, "dwell": d,
        "threshold": threshold, "a_higher": a_higher,
        "per_tick_error": float(np.mean(errs)),
        "mean_switches_per_episode": float(np.mean(switches_per_ep)),
    }


def select_config(grid_results: list[dict[str, Any]]) -> dict[str, Any]:
    min_err = min(r["per_tick_error"] for r in grid_results)
    tied = [r for r in grid_results if r["per_tick_error"] <= min_err + TIE_BAND]
    min_switch = min(r["mean_switches_per_episode"] for r in tied)
    tied2 = [r for r in tied if r["mean_switches_per_episode"] <= min_switch + 1e-9]
    tied2.sort(key=lambda r: (r["dwell"] != 4, r["window"]))
    return tied2[0]


def main() -> int:
    print("Stage A: collecting calibration tick series (32 episodes)...", flush=True)
    calib = {}
    for pole in ("A", "B"):
        for seed in CALIB_SEEDS:
            calib[(pole, seed)] = collect_tick_series(pole, seed)
            print(f"  calib {pole} {seed} done", flush=True)

    print("Scoring grid...", flush=True)
    grid_results = []
    for w in WINDOW_GRID:
        for m in HYST_GRID:
            for d in DWELL_GRID:
                grid_results.append(score_config(calib, w, m, d))

    selected = select_config(grid_results)
    print("Selected config:", json.dumps(selected, indent=2))

    print("Stage C: collecting held-out tick series (32 episodes)...", flush=True)
    holdout = {}
    for pole in ("A", "B"):
        for seed in HOLDOUT_SEEDS:
            holdout[(pole, seed)] = collect_tick_series(pole, seed)
            print(f"  holdout {pole} {seed} done", flush=True)

    w, m, d = selected["window"], selected["hysteresis"], selected["dwell"]
    threshold, a_higher = selected["threshold"], selected["a_higher"]
    ho_errs, ho_switches, ho_any_switch, ho_dominant_correct = [], [], [], []
    for (pole, _seed), series in holdout.items():
        wv = windowed(series, w)
        pred = apply_hysteresis_dwell(wv, threshold, a_higher, m, d)
        true_label = 1 if pole == "A" else 0
        ho_errs.append(float(np.mean(pred != true_label)))
        n_switch = int(np.sum(np.diff(pred) != 0))
        ho_switches.append(n_switch)
        ho_any_switch.append(n_switch > 0)
        dominant = int(np.mean(pred) > 0.5)
        ho_dominant_correct.append(dominant == true_label)

    report = {
        "stage_A_grid": grid_results,
        "selected_config": selected,
        "stage_C_held_out": {
            "n_episodes": len(holdout),
            "per_tick_classification_error": float(np.mean(ho_errs)),
            "mean_switches_per_episode": float(np.mean(ho_switches)),
            "fraction_episodes_with_switch": float(np.mean(ho_any_switch)),
            "dominant_composition_accuracy": float(np.mean(ho_dominant_correct)),
        },
    }
    out = ROOT / "artifacts/strategic_demand/sppo/COMPOSITION_SELECTOR_ONLINE_CALIBRATION_RESULT.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["stage_C_held_out"], indent=2))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
