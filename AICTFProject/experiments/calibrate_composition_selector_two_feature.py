"""COMPOSITION_SELECTOR_TWO_FEATURE_AMENDMENT.json, all steps.

Outcome-blind. Never reads, computes, or prints score/win/reward/return/draw.
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
COMPOSITION = "2A_2D"
R_BLUE = 4.0
RED_RADII = (4.0, 6.0, 8.0, 10.0)
CALIB_SEEDS = list(range(99_900_101, 99_900_101 + 16))
HOLDOUT2_SEEDS = list(range(99_900_301, 99_900_301 + 16))

WINDOW_GRID = (20, 40, 60, 80)
HYST_GRID = (0.0, 0.05, 0.10, 0.15)
DWELL_GRID = (4, 10, 20, 40)
TIE_BAND = 0.02
KILL_THRESHOLD = 0.80


def count_near(core, cx: float, cy: float, radius: float) -> int:
    count = 0
    for i in range(core.red_x.shape[1]):
        if not bool(core.red_alive[0, i].item()):
            continue
        dx = float(core.red_x[0, i].item()) - cx
        dy = float(core.red_y[0, i].item()) - cy
        if (dx * dx + dy * dy) ** 0.5 <= radius:
            count += 1
    return count


def collect_two_feature_series(pole: str, seed: int) -> dict[str, list[int]]:
    env, core, genome, live = _make_env(pole, seed)
    try:
        roles = composition_roles(COMPOSITION)
        series: dict[str, list[int]] = {"p_blue": []}
        for r in RED_RADII:
            series[f"p_red_r{int(r)}"] = []
        for _ in range(HORIZON):
            bfx = float(core.blue_flag_pos[0, 0].item())
            bfy = float(core.blue_flag_pos[0, 1].item())
            rfx = float(core.red_flag_pos[0, 0].item())
            rfy = float(core.red_flag_pos[0, 1].item())
            series["p_blue"].append(count_near(core, bfx, bfy, R_BLUE))
            for r in RED_RADII:
                series[f"p_red_r{int(r)}"].append(count_near(core, rfx, rfy, r))
            env.step_async(_action_for_roles(core, roles))
            _obs, _rew, done, _infos = env.step_wait()
            if bool(np.asarray(done).any()):
                break
        return series
    finally:
        env.close()


def min_misclass_threshold(pooled_a: np.ndarray, pooled_b: np.ndarray) -> tuple[float, float, bool]:
    a_higher = np.median(pooled_a) > np.median(pooled_b)
    pooled = np.sort(np.unique(np.concatenate([pooled_a, pooled_b])))
    if len(pooled) < 2:
        return float(pooled[0]) if len(pooled) else 0.0, 1.0, a_higher
    candidates = (pooled[:-1] + pooled[1:]) / 2.0
    best_t, best_err = None, np.inf
    for t in candidates:
        if a_higher:
            err = (np.sum(pooled_a <= t) + np.sum(pooled_b > t)) / (len(pooled_a) + len(pooled_b))
        else:
            err = (np.sum(pooled_a >= t) + np.sum(pooled_b < t)) / (len(pooled_a) + len(pooled_b))
        if err < best_err:
            best_err, best_t = err, float(t)
    return best_t, float(best_err), a_higher


def windowed(series: list[float], w: int) -> np.ndarray:
    arr = np.asarray(series, dtype=np.float64)
    out = np.empty_like(arr)
    csum = np.concatenate([[0.0], np.cumsum(arr)])
    for t in range(len(arr)):
        lo = max(0, t - w + 1)
        out[t] = (csum[t + 1] - csum[lo]) / (t + 1 - lo)
    return out


def apply_hysteresis_dwell(values: np.ndarray, threshold: float, a_higher: bool, m: float, d: int) -> np.ndarray:
    upper, lower = threshold + m, threshold - m
    state = 1 if (values[0] > threshold) == a_higher else 0
    out = np.empty(len(values), dtype=np.int64)
    last_switch = -10_000
    for t, v in enumerate(values):
        want_a = (v > upper) if a_higher else (v < lower)
        want_b = (v < lower) if a_higher else (v > upper)
        if state == 1 and want_b and (t - last_switch) >= d:
            state, last_switch = 0, t
        elif state == 0 and want_a and (t - last_switch) >= d:
            state, last_switch = 1, t
        out[t] = state
    return out


def score_config(d_series: dict[tuple[str, int], list[float]], w: int, m: float, d: int) -> dict[str, Any]:
    windows = {k: windowed(v, w) for k, v in d_series.items()}
    pooled_a = np.concatenate([windows[k] for k in windows if k[0] == "A"])
    pooled_b = np.concatenate([windows[k] for k in windows if k[0] == "B"])
    threshold, _, a_higher = min_misclass_threshold(pooled_a, pooled_b)
    errs, switches = [], []
    for (pole, _seed), wv in windows.items():
        pred = apply_hysteresis_dwell(wv, threshold, a_higher, m, d)
        true_label = 1 if pole == "A" else 0
        errs.append(np.mean(pred != true_label))
        switches.append(int(np.sum(np.diff(pred) != 0)))
    return {
        "window": w, "hysteresis": m, "dwell": d, "threshold": threshold, "a_higher": a_higher,
        "per_tick_error": float(np.mean(errs)), "mean_switches_per_episode": float(np.mean(switches)),
    }


def select_config(grid_results: list[dict[str, Any]]) -> dict[str, Any]:
    min_err = min(r["per_tick_error"] for r in grid_results)
    tied = [r for r in grid_results if r["per_tick_error"] <= min_err + TIE_BAND]
    min_switch = min(r["mean_switches_per_episode"] for r in tied)
    tied2 = [r for r in tied if r["mean_switches_per_episode"] <= min_switch + 1e-9]
    tied2.sort(key=lambda r: (r["dwell"] != 4, r["window"]))
    return tied2[0]


def main() -> int:
    print("Step 1: re-collecting calibration two-feature series (32 episodes)...", flush=True)
    calib: dict[tuple[str, int], dict[str, list[int]]] = {}
    for pole in ("A", "B"):
        for seed in CALIB_SEEDS:
            calib[(pole, seed)] = collect_two_feature_series(pole, seed)
            print(f"  calib {pole} {seed} done", flush=True)

    # sanity: P_blue full-episode means should reproduce the original derivation
    pb_a = np.mean([np.mean(calib[("A", s)]["p_blue"]) for s in CALIB_SEEDS])
    pb_b = np.mean([np.mean(calib[("B", s)]["p_blue"]) for s in CALIB_SEEDS])
    print(f"  determinism check: P_blue mean A={pb_a:.4f} B={pb_b:.4f} (expect ~1.08 / ~0.37)")

    print("Step 2: selecting r_red* by full-episode minimize-pooled-misclassification...", flush=True)
    red_selection = {}
    for r in RED_RADII:
        key = f"p_red_r{int(r)}"
        a_vals = np.array([np.mean(calib[("A", s)][key]) for s in CALIB_SEEDS])
        b_vals = np.array([np.mean(calib[("B", s)][key]) for s in CALIB_SEEDS])
        thr, err, a_higher = min_misclass_threshold(a_vals, b_vals)
        red_selection[r] = {
            "A_mean": float(a_vals.mean()), "B_mean": float(b_vals.mean()),
            "threshold": thr, "pooled_misclassification_rate": err, "a_higher": a_higher,
        }
        print(f"  r_red={r}: A_mean={a_vals.mean():.4f} B_mean={b_vals.mean():.4f} err={err:.4f}")
    r_red_star = min(RED_RADII, key=lambda r: red_selection[r]["pooled_misclassification_rate"])
    print(f"  selected r_red* = {r_red_star}  (err={red_selection[r_red_star]['pooled_misclassification_rate']:.4f})")

    print("Step 3-4: forming D_t and re-running the frozen grid...", flush=True)
    key_red = f"p_red_r{int(r_red_star)}"
    d_series: dict[tuple[str, int], list[float]] = {}
    for (pole, seed), s in calib.items():
        d_series[(pole, seed)] = [pb - pr for pb, pr in zip(s["p_blue"], s[key_red])]

    grid_results = []
    for w in WINDOW_GRID:
        for m in HYST_GRID:
            for d in DWELL_GRID:
                grid_results.append(score_config(d_series, w, m, d))
    selected = select_config(grid_results)
    print("  selected config:", json.dumps(selected, indent=2))

    print("Step 5: collecting fresh held-out block (99900301-316)...", flush=True)
    holdout: dict[tuple[str, int], dict[str, list[int]]] = {}
    for pole in ("A", "B"):
        for seed in HOLDOUT2_SEEDS:
            holdout[(pole, seed)] = collect_two_feature_series(pole, seed)
            print(f"  holdout2 {pole} {seed} done", flush=True)

    w, m, d = selected["window"], selected["hysteresis"], selected["dwell"]
    threshold, a_higher = selected["threshold"], selected["a_higher"]
    ho_errs, ho_switches, ho_any_switch, ho_dominant_correct = [], [], [], []
    for (pole, _seed), s in holdout.items():
        dt = [pb - pr for pb, pr in zip(s["p_blue"], s[key_red])]
        wv = windowed(dt, w)
        pred = apply_hysteresis_dwell(wv, threshold, a_higher, m, d)
        true_label = 1 if pole == "A" else 0
        ho_errs.append(float(np.mean(pred != true_label)))
        n_switch = int(np.sum(np.diff(pred) != 0))
        ho_switches.append(n_switch)
        ho_any_switch.append(n_switch > 0)
        ho_dominant_correct.append(int(np.mean(pred) > 0.5) == true_label)

    dominant_accuracy = float(np.mean(ho_dominant_correct))
    kill_pass = dominant_accuracy >= KILL_THRESHOLD

    report = {
        "determinism_check_p_blue_means": {"A": float(pb_a), "B": float(pb_b)},
        "r_red_selection": {str(r): v for r, v in red_selection.items()},
        "r_red_star": r_red_star,
        "stage_A_grid_on_D_t": grid_results,
        "selected_config": selected,
        "stage_C_held_out": {
            "n_episodes": len(holdout),
            "per_tick_classification_error": float(np.mean(ho_errs)),
            "mean_switches_per_episode": float(np.mean(ho_switches)),
            "fraction_episodes_with_switch": float(np.mean(ho_any_switch)),
            "dominant_composition_accuracy": dominant_accuracy,
        },
        "KILL_RULE": {
            "threshold": KILL_THRESHOLD,
            "observed": dominant_accuracy,
            "result": "PASS" if kill_pass else "FAIL",
        },
    }
    out = ROOT / "artifacts/strategic_demand/sppo/COMPOSITION_SELECTOR_TWO_FEATURE_RESULT.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["KILL_RULE"], indent=2))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
