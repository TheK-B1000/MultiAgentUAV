"""COMPOSITION_SELECTOR_ASYMMETRIC_B_TRIGGER_V1_SPEC.json -- calibration only.

Outcome-blind: no score/win/reward/return/draw is read, computed, printed or
serialized. Pole labels are used ONLY for FP/TP accounting, never as a
selector input (the selector sees the windowed P_blue statistic alone).

Sign convention, documented because the spec's prose says "exceeds the frozen
high threshold": the measured feature P_blue (live Reds within 4 cells of
BLUE's flag) is HIGHER on Pole A (~1.08) than Pole B (~0.37). Strong B-like
evidence is therefore a LOW windowed P_blue. The state machine triggers to
4A_0D when the statistic falls BELOW the trigger bound and returns to the
2A_2D default when it rises back above the release bound. "Exceeds the high
threshold" is read as evidence strength, not raw feature magnitude.
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

SPEC = ROOT / "artifacts/strategic_demand/sppo/COMPOSITION_SELECTOR_ASYMMETRIC_B_TRIGGER_V1_SPEC.json"
OUT = ROOT / "artifacts/strategic_demand/sppo/COMPOSITION_SELECTOR_ASYMMETRIC_B_TRIGGER_CALIBRATION_RESULT.json"

HORIZON = 240
R_BLUE = 4.0
COMPOSITION = "2A_2D"
CALIB_SEEDS = list(range(99_900_101, 99_900_101 + 16))
HOLDOUT_SEEDS = list(range(99_900_401, 99_900_401 + 16))

WINDOW_GRID = (20, 40, 60, 80, 120)
HYST_GRID = (0.0, 0.05, 0.10, 0.15, 0.20)
DWELL_GRID = (4, 10, 20, 40)
MAX_A_FP_TICK_RATE = 0.10          # frozen in the spec, not retunable here
B_DOMINANT_PASS_FRACTION = 0.50    # frozen in the spec
THRESHOLD_QUANTILE_STEP = 1.0      # pre-declared candidate resolution

DEFAULT_STATE = 0   # 0 = 2A_2D (default/guardrail), 1 = 4A_0D (triggered)


def collect_p_blue(pole: str, seed: int) -> list[int]:
    env, core, genome, live = _make_env(pole, seed)
    try:
        roles = composition_roles(COMPOSITION)
        series: list[int] = []
        for _ in range(HORIZON):
            fx = float(core.blue_flag_pos[0, 0].item())
            fy = float(core.blue_flag_pos[0, 1].item())
            count = 0
            for i in range(core.red_x.shape[1]):
                if not bool(core.red_alive[0, i].item()):
                    continue
                dx = float(core.red_x[0, i].item()) - fx
                dy = float(core.red_y[0, i].item()) - fy
                if (dx * dx + dy * dy) ** 0.5 <= R_BLUE:
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


def run_state_machine(values: np.ndarray, threshold: float, m: float, d: int) -> np.ndarray:
    """Default 2A_2D at episode start (C2). Trigger to 4A_0D when the
    statistic drops below threshold-m; release back when it rises above
    threshold+m; both subject to the dwell minimum."""
    trigger_bound = threshold - m
    release_bound = threshold + m
    state = DEFAULT_STATE
    last_switch = -10_000
    out = np.empty(len(values), dtype=np.int8)
    for t in range(len(values)):
        v = values[t]
        if state == 0 and v < trigger_bound and (t - last_switch) >= d:
            state, last_switch = 1, t
        elif state == 1 and v > release_bound and (t - last_switch) >= d:
            state, last_switch = 0, t
        out[t] = state
    return out


def score(preds: dict[tuple[str, int], np.ndarray]) -> dict[str, float]:
    a_ticks = np.concatenate([p for (pole, _), p in preds.items() if pole == "A"])
    b_ticks = np.concatenate([p for (pole, _), p in preds.items() if pole == "B"])
    a_switches = [int(np.sum(np.diff(p) != 0)) for (pole, _), p in preds.items() if pole == "A"]
    b_switches = [int(np.sum(np.diff(p) != 0)) for (pole, _), p in preds.items() if pole == "B"]
    a_any = [bool(np.any(p == 1)) for (pole, _), p in preds.items() if pole == "A"]
    b_dom = [bool(np.mean(p) > 0.5) for (pole, _), p in preds.items() if pole == "B"]
    return {
        "A_FP_tick_rate": float(np.mean(a_ticks == 1)),
        "B_TP_tick_rate": float(np.mean(b_ticks == 1)),
        "A_mean_switches_per_episode": float(np.mean(a_switches)),
        "B_mean_switches_per_episode": float(np.mean(b_switches)),
        "A_fraction_episodes_with_any_4A0D_tick": float(np.mean(a_any)),
        "B_fraction_episodes_dominant_4A0D": float(np.mean(b_dom)),
    }


def main() -> int:
    if not SPEC.is_file():
        raise SystemExit(f"REFUSING: frozen spec missing: {SPEC}")
    if OUT.exists():
        raise SystemExit(f"REFUSING: calibration result already exists: {OUT}")

    print("Collecting calibration traces (99900101-116, both poles)...", flush=True)
    calib: dict[tuple[str, int], list[int]] = {}
    for pole in ("A", "B"):
        for seed in CALIB_SEEDS:
            calib[(pole, seed)] = collect_p_blue(pole, seed)
    print(f"  calibration episodes: {len(calib)}", flush=True)

    print("Scanning frozen grid under the asymmetric criterion...", flush=True)
    quantiles = np.arange(THRESHOLD_QUANTILE_STEP, 100.0, THRESHOLD_QUANTILE_STEP)
    survivors: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    for w in WINDOW_GRID:
        win = {k: windowed(v, w) for k, v in calib.items()}
        pooled = np.concatenate(list(win.values()))
        candidates = np.unique(np.percentile(pooled, quantiles))
        for m in HYST_GRID:
            for d in DWELL_GRID:
                for thr in candidates:
                    preds = {k: run_state_machine(v, float(thr), m, d) for k, v in win.items()}
                    s = score(preds)
                    row = {"window": w, "hysteresis": m, "dwell": d, "threshold": float(thr), **s}
                    all_rows.append(row)
                    if s["A_FP_tick_rate"] <= MAX_A_FP_TICK_RATE:
                        survivors.append(row)
        print(f"  W={w} done ({len(survivors)} survivors so far)", flush=True)

    if not survivors:
        report = {
            "record_id": "COMPOSITION_SELECTOR_ASYMMETRIC_B_TRIGGER_CALIBRATION_RESULT",
            "status": "COMPLETE",
            "implements": str(SPEC.relative_to(ROOT)),
            "calibration_seeds": [CALIB_SEEDS[0], CALIB_SEEDS[-1]],
            "max_A_FP_tick_rate": MAX_A_FP_TICK_RATE,
            "n_grid_rows": len(all_rows),
            "n_survivors": 0,
            "DECISION": "NO_CONSERVATIVE_B_TRIGGER",
            "reason": "No (window, hysteresis, dwell, threshold) configuration kept the Pole-A false-positive tick rate at or below 0.10 on calibration traces.",
            "best_A_FP_achieved": float(min(r["A_FP_tick_rate"] for r in all_rows)),
            "outcome_blind": True,
        }
        OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({k: report[k] for k in ("DECISION", "n_survivors", "best_A_FP_achieved")}, indent=2))
        return 2

    # frozen tie-break: max B_TP; then min A switches; then prefer d=4; then smallest W
    best_tp = max(r["B_TP_tick_rate"] for r in survivors)
    tied = [r for r in survivors if r["B_TP_tick_rate"] >= best_tp - 1e-12]
    min_sw = min(r["A_mean_switches_per_episode"] for r in tied)
    tied = [r for r in tied if r["A_mean_switches_per_episode"] <= min_sw + 1e-12]
    tied.sort(key=lambda r: (r["dwell"] != 4, r["window"], r["threshold"]))
    selected = tied[0]
    print("Selected:", json.dumps(selected, indent=2), flush=True)

    print("Collecting held-out traces (99900401-416, both poles)...", flush=True)
    holdout: dict[tuple[str, int], list[int]] = {}
    for pole in ("A", "B"):
        for seed in HOLDOUT_SEEDS:
            holdout[(pole, seed)] = collect_p_blue(pole, seed)

    w, m, d, thr = selected["window"], selected["hysteresis"], selected["dwell"], selected["threshold"]
    ho_preds = {k: run_state_machine(windowed(v, w), thr, m, d) for k, v in holdout.items()}
    ho = score(ho_preds)

    pass_a = ho["A_FP_tick_rate"] <= MAX_A_FP_TICK_RATE
    pass_b = ho["B_fraction_episodes_dominant_4A0D"] >= B_DOMINANT_PASS_FRACTION
    decision = "CONSERVATIVE_B_TRIGGER_CALIBRATED" if (pass_a and pass_b) else "NO_CONSERVATIVE_B_TRIGGER"

    report = {
        "record_id": "COMPOSITION_SELECTOR_ASYMMETRIC_B_TRIGGER_CALIBRATION_RESULT",
        "status": "COMPLETE",
        "implements": str(SPEC.relative_to(ROOT)),
        "sign_convention": "P_blue is higher on Pole A; strong B evidence is a LOW windowed value. Trigger to 4A_0D below threshold-m, release to 2A_2D above threshold+m.",
        "calibration_seeds": f"{CALIB_SEEDS[0]}-{CALIB_SEEDS[-1]}",
        "holdout_seeds": f"{HOLDOUT_SEEDS[0]}-{HOLDOUT_SEEDS[-1]}",
        "frozen_constraints": {
            "max_A_FP_tick_rate": MAX_A_FP_TICK_RATE,
            "B_dominant_pass_fraction": B_DOMINANT_PASS_FRACTION,
        },
        "grid": {"window": list(WINDOW_GRID), "hysteresis": list(HYST_GRID), "dwell": list(DWELL_GRID),
                 "threshold_candidates": "percentiles of pooled calibration windowed statistic, 1% steps"},
        "n_grid_rows": len(all_rows),
        "n_survivors_under_A_FP_constraint": len(survivors),
        "selected_config": selected,
        "held_out": ho,
        "calibration_pass_rule": {
            "held_out_A_FP_tick_rate<=0.10": pass_a,
            "held_out_B_fraction_episodes_dominant_4A0D>=0.50": pass_b,
        },
        "DECISION": decision,
        "outcome_blind": True,
        "claim_boundary": "Calibration only. No outcome episode was run. PASS earns the frozen outcome step; it does not establish any win-rate result.",
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"DECISION": decision, "held_out": ho, "selected": selected}, indent=2))
    return 0 if decision == "CONSERVATIVE_B_TRIGGER_CALIBRATED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
