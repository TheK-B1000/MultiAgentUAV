"""COMPOSITION_SELECTOR_TWO_FEATURE_ASYMMETRIC_REVIVAL_SPEC.json -- calibration only.

One bounded pass. The ONLY change versus
`calibrate_asymmetric_b_trigger.py` is the statistic fed to the state machine:

    P_blue_t(r=4.0)   ->   D_t = P_blue_t(r=4.0) - P_red_t(r_red*)

The window/hysteresis/dwell grid, the threshold candidate rule, the A-side
false-positive budget, the B dominance bar, the tie-break order and the state
machine itself are imported from the single-feature module so they cannot
drift. Nothing about dwell changes.

Outcome-blind: no score/win/reward/return/draw is read, computed, printed or
serialized. Pole labels are used ONLY for FP/TP accounting, never as a
selector input.
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

from experiments.calibrate_asymmetric_b_trigger import (  # noqa: E402
    B_DOMINANT_PASS_FRACTION,
    DWELL_GRID,
    HYST_GRID,
    MAX_A_FP_TICK_RATE,
    THRESHOLD_QUANTILE_STEP,
    WINDOW_GRID,
    run_state_machine,
    score,
    windowed,
)
from experiments.run_pyquaticus_4v4_role_composition_sweep import (  # noqa: E402
    _action_for_roles,
    composition_roles,
)
from experiments.run_pyquaticus_4v4_team_evaluation import _make_env  # noqa: E402

ART = ROOT / "artifacts/strategic_demand/sppo"
SPEC = ART / "COMPOSITION_SELECTOR_TWO_FEATURE_ASYMMETRIC_REVIVAL_SPEC.json"
DERIVATION = ART / "COMPOSITION_SELECTOR_FEATURE_DERIVATION_TRACE.json"
OUT = ART / "COMPOSITION_SELECTOR_TWO_FEATURE_ASYMMETRIC_RESULT.json"
FRONTIER = ART / "COMPOSITION_SELECTOR_TWO_FEATURE_ASYMMETRIC_FRONTIER_DIAGNOSTIC.json"
ROWS_CSV = ART / "composition_selector_two_feature_asymmetric_raw_ticks.csv"

HORIZON = 240
COMPOSITION = "2A_2D"
R_BLUE = 4.0
RED_RADII = (4.0, 6.0, 8.0, 10.0)
CALIB_SEEDS = list(range(99_900_101, 99_900_101 + 16))
HOLDOUT_SEEDS = list(range(99_900_301, 99_900_301 + 16))
DETERMINISM_TOL = 1e-6
FRONTIER_CAPS = [round(0.02 * i, 2) for i in range(0, 28)]

Key = tuple[str, int]


# ---------------------------------------------------------------- collection

def _count_red_near(core, cx: float, cy: float, radius: float) -> int:
    count = 0
    for i in range(core.red_x.shape[1]):
        if not bool(core.red_alive[0, i].item()):
            continue
        dx = float(core.red_x[0, i].item()) - cx
        dy = float(core.red_y[0, i].item()) - cy
        if (dx * dx + dy * dy) ** 0.5 <= radius:
            count += 1
    return count


def collect_series(pole: str, seed: int) -> dict[str, list[int]]:
    """Per-tick P_blue (reds near BLUE's flag = forward pressure) and P_red at
    every candidate radius (reds near RED's OWN flag = home occupancy)."""
    env, core, _genome, _live = _make_env(pole, seed)
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
            series["p_blue"].append(_count_red_near(core, bfx, bfy, R_BLUE))
            for r in RED_RADII:
                series[f"p_red_r{int(r)}"].append(_count_red_near(core, rfx, rfy, r))
            env.step_async(_action_for_roles(core, roles))
            _obs, _rew, done, _infos = env.step_wait()
            if bool(np.asarray(done).any()):
                break
        return series
    finally:
        env.close()


def collect_block(seeds: list[int], label: str) -> dict[Key, dict[str, list[int]]]:
    out: dict[Key, dict[str, list[int]]] = {}
    for pole in ("A", "B"):
        for seed in seeds:
            out[(pole, seed)] = collect_series(pole, seed)
            print(f"  {label} {pole} {seed}: {len(out[(pole, seed)]['p_blue'])} ticks", flush=True)
    return out


def persist_rows(blocks: dict[str, dict[Key, dict[str, list[int]]]]) -> None:
    fields = ["split", "pole", "seed", "tick", "p_blue"] + [f"p_red_r{int(r)}" for r in RED_RADII]
    with ROWS_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for split, block in blocks.items():
            for (pole, seed), s in sorted(block.items()):
                for t in range(len(s["p_blue"])):
                    row = {"split": split, "pole": pole, "seed": seed, "tick": t,
                           "p_blue": s["p_blue"][t]}
                    for r in RED_RADII:
                        row[f"p_red_r{int(r)}"] = s[f"p_red_r{int(r)}"][t]
                    writer.writerow(row)


# ------------------------------------------------------- feature construction

def min_misclass(a_vals: np.ndarray, b_vals: np.ndarray) -> tuple[float, float, bool]:
    """Identical criterion to the sealed r_blue derivation: exhaustive scan over
    midpoints between adjacent sorted pooled values, minimizing pooled error."""
    a_higher = bool(np.median(a_vals) > np.median(b_vals))
    pooled = np.sort(np.unique(np.concatenate([a_vals, b_vals])))
    if len(pooled) < 2:
        return (float(pooled[0]) if len(pooled) else 0.0), 1.0, a_higher
    n = len(a_vals) + len(b_vals)
    best_t, best_err = None, np.inf
    for t in (pooled[:-1] + pooled[1:]) / 2.0:
        if a_higher:
            err = (np.sum(a_vals <= t) + np.sum(b_vals > t)) / n
        else:
            err = (np.sum(a_vals >= t) + np.sum(b_vals < t)) / n
        if err < best_err:
            best_err, best_t = float(err), float(t)
    return best_t, best_err, a_higher


def gate_determinism(calib: dict[Key, dict[str, list[int]]]) -> dict[str, float]:
    """Absence or drift of the sealed anchor is an error state, never a default."""
    if not DERIVATION.is_file():
        raise SystemExit(f"ABORT: sealed derivation anchor missing: {DERIVATION}")
    trace = json.loads(DERIVATION.read_text(encoding="utf-8"))
    anchor = trace["radii"]["4.0"]
    for field in ("A_mean", "B_mean"):
        if field not in anchor:
            raise SystemExit(f"ABORT: sealed anchor lacks {field}")
    got_a = float(np.mean([np.mean(calib[("A", s)]["p_blue"]) for s in CALIB_SEEDS]))
    got_b = float(np.mean([np.mean(calib[("B", s)]["p_blue"]) for s in CALIB_SEEDS]))
    da = abs(got_a - float(anchor["A_mean"]))
    db = abs(got_b - float(anchor["B_mean"]))
    if da > DETERMINISM_TOL or db > DETERMINISM_TOL:
        raise SystemExit(
            "ABORT: re-collected P_blue does not reproduce the sealed derivation.\n"
            f"  A: got {got_a!r} expected {anchor['A_mean']!r} (delta {da:.3e})\n"
            f"  B: got {got_b!r} expected {anchor['B_mean']!r} (delta {db:.3e})"
        )
    print(f"  determinism gate PASS (A {got_a:.10f}, B {got_b:.10f})", flush=True)
    return {"A": got_a, "B": got_b, "max_abs_delta": max(da, db)}


def build_d_series(block: dict[Key, dict[str, list[int]]], key_red: str) -> dict[Key, list[float]]:
    return {k: [float(pb - pr) for pb, pr in zip(s["p_blue"], s[key_red])]
            for k, s in block.items()}


def gate_direction(d_calib: dict[Key, list[float]]) -> dict[str, float]:
    """The predeclared convention is D higher on A. Verify; never flip to fit."""
    a = np.concatenate([np.asarray(v) for (pole, _), v in d_calib.items() if pole == "A"])
    b = np.concatenate([np.asarray(v) for (pole, _), v in d_calib.items() if pole == "B"])
    stats = {"A_mean": float(a.mean()), "B_mean": float(b.mean()),
             "A_median": float(np.median(a)), "B_median": float(np.median(b))}
    if not (stats["A_mean"] > stats["B_mean"] and stats["A_median"] >= stats["B_median"]):
        raise SystemExit(
            "ABORT: predeclared sign convention violated (D must be higher on Pole A). "
            f"{json.dumps(stats)}"
        )
    print(f"  direction gate PASS ({json.dumps(stats)})", flush=True)
    return stats


# ------------------------------------------------------------- independent pass

def rederive_from_csv(split: str, key_red: str, w: int, thr: float, m: float, d: int) -> dict[str, float]:
    """Second implementation, reading only the persisted CSV. Deliberately naive
    (O(n*w) means, plainly written state machine) so it shares no code path with
    the scoring pass."""
    by_ep: dict[Key, list[tuple[int, float]]] = {}
    with ROWS_CSV.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["split"] != split:
                continue
            k = (row["pole"], int(row["seed"]))
            val = float(row["p_blue"]) - float(row[key_red])
            by_ep.setdefault(k, []).append((int(row["tick"]), val))

    a_ticks: list[int] = []
    b_ticks: list[int] = []
    a_sw: list[int] = []
    b_sw: list[int] = []
    a_any: list[bool] = []
    b_dom: list[bool] = []
    for (pole, _seed), pairs in by_ep.items():
        vals = [v for _t, v in sorted(pairs)]
        means = []
        for t in range(len(vals)):
            lo = max(0, t - w + 1)
            chunk = vals[lo:t + 1]
            means.append(sum(chunk) / len(chunk))
        state, last = 0, -10_000
        preds: list[int] = []
        for t, v in enumerate(means):
            if state == 0 and v < thr - m and (t - last) >= d:
                state, last = 1, t
            elif state == 1 and v > thr + m and (t - last) >= d:
                state, last = 0, t
            preds.append(state)
        switches = sum(1 for i in range(1, len(preds)) if preds[i] != preds[i - 1])
        if pole == "A":
            a_ticks.extend(preds)
            a_sw.append(switches)
            a_any.append(any(p == 1 for p in preds))
        else:
            b_ticks.extend(preds)
            b_sw.append(switches)
            b_dom.append((sum(preds) / len(preds)) > 0.5)
    return {
        "A_FP_tick_rate": sum(a_ticks) / len(a_ticks),
        "B_TP_tick_rate": sum(b_ticks) / len(b_ticks),
        "A_mean_switches_per_episode": sum(a_sw) / len(a_sw),
        "B_mean_switches_per_episode": sum(b_sw) / len(b_sw),
        "A_fraction_episodes_with_any_4A0D_tick": sum(a_any) / len(a_any),
        "B_fraction_episodes_dominant_4A0D": sum(b_dom) / len(b_dom),
    }


# --------------------------------------------------------------------- main

def main() -> int:
    if not SPEC.is_file():
        raise SystemExit(f"REFUSING: frozen spec missing: {SPEC}")
    if OUT.exists():
        raise SystemExit(f"REFUSING: result already exists (one bounded pass): {OUT}")

    print("Collecting calibration traces (99900101-116, both poles)...", flush=True)
    calib = collect_block(CALIB_SEEDS, "calib")
    print("Collecting held-out traces (99900301-316, both poles)...", flush=True)
    holdout = collect_block(HOLDOUT_SEEDS, "holdout")
    persist_rows({"calibration": calib, "holdout": holdout})
    print(f"  persisted per-tick rows -> {ROWS_CSV.name}", flush=True)

    determinism = gate_determinism(calib)

    print("Selecting r_red* by predeclared pooled-misclassification...", flush=True)
    red_selection: dict[str, dict[str, Any]] = {}
    for r in RED_RADII:
        key = f"p_red_r{int(r)}"
        a_vals = np.array([np.mean(calib[("A", s)][key]) for s in CALIB_SEEDS])
        b_vals = np.array([np.mean(calib[("B", s)][key]) for s in CALIB_SEEDS])
        thr, err, a_higher = min_misclass(a_vals, b_vals)
        red_selection[str(r)] = {
            "A_mean": float(a_vals.mean()), "B_mean": float(b_vals.mean()),
            "threshold": thr, "pooled_misclassification_rate": err,
            "pole_A_higher": a_higher,
        }
        print(f"  r_red={r}: A={a_vals.mean():.4f} B={b_vals.mean():.4f} err={err:.4f}", flush=True)
    r_red_star = min(RED_RADII, key=lambda r: red_selection[str(r)]["pooled_misclassification_rate"])
    key_red = f"p_red_r{int(r_red_star)}"
    print(f"  r_red* = {r_red_star}", flush=True)

    d_calib = build_d_series(calib, key_red)
    direction = gate_direction(d_calib)

    print("Scanning the frozen grid on D_t under the asymmetric criterion...", flush=True)
    quantiles = np.arange(THRESHOLD_QUANTILE_STEP, 100.0, THRESHOLD_QUANTILE_STEP)
    all_rows: list[dict[str, Any]] = []
    survivors: list[dict[str, Any]] = []
    for w in WINDOW_GRID:
        win = {k: windowed(v, w) for k, v in d_calib.items()}
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
        print(f"  W={w} done ({len(all_rows)} rows, {len(survivors)} survivors)", flush=True)

    # descriptive-only frontier, emitted regardless of the decision
    frontier = []
    for cap in FRONTIER_CAPS:
        elig = [r for r in all_rows if r["A_FP_tick_rate"] <= cap]
        if not elig:
            frontier.append({"A_FP_cap": cap, "n_survivors": 0, "max_B_TP": None, "config": None})
            continue
        best = max(elig, key=lambda r: r["B_TP_tick_rate"])
        frontier.append({
            "A_FP_cap": cap, "n_survivors": len(elig),
            "max_B_TP": best["B_TP_tick_rate"],
            "B_fraction_episodes_dominant_4A0D": best["B_fraction_episodes_dominant_4A0D"],
            "config": {k: best[k] for k in ("window", "hysteresis", "dwell", "threshold")},
        })
    global_best = max(all_rows, key=lambda r: r["B_TP_tick_rate"])
    FRONTIER.write_text(json.dumps({
        "record_id": "COMPOSITION_SELECTOR_TWO_FEATURE_ASYMMETRIC_FRONTIER_DIAGNOSTIC",
        "purpose": "trade-off frontier only; DESCRIPTIVE, changes no frozen decision. A favourable frontier above the 0.10 cap is NOT a pass.",
        "statistic": f"D_t = P_blue(4.0) - P_red({r_red_star})",
        "frozen_constraint_unaffected": MAX_A_FP_TICK_RATE,
        "n_grid_rows": len(all_rows),
        "frontier_A_FP_cap_vs_max_B_TP": frontier,
        "global_best_B_TP_ignoring_A_entirely": global_best,
        "comparable_single_feature_record": "COMPOSITION_SELECTOR_ASYMMETRIC_B_TRIGGER_FRONTIER_DIAGNOSTIC.json",
    }, indent=2) + "\n", encoding="utf-8")
    print(f"  frontier -> {FRONTIER.name}", flush=True)

    base = {
        "record_id": "COMPOSITION_SELECTOR_TWO_FEATURE_ASYMMETRIC_RESULT",
        "status": "COMPLETE",
        "implements": str(SPEC.relative_to(ROOT)).replace("\\", "/"),
        "statistic": f"D_t = P_blue(r=4.0) - P_red(r={r_red_star})",
        "sign_convention": "D is higher on Pole A; strong B evidence is a LOW windowed D. Trigger to 4A_0D below threshold-m, release to 2A_2D above threshold+m.",
        "calibration_seeds": f"{CALIB_SEEDS[0]}-{CALIB_SEEDS[-1]}",
        "holdout_seeds": f"{HOLDOUT_SEEDS[0]}-{HOLDOUT_SEEDS[-1]}",
        "determinism_gate": determinism,
        "direction_gate": direction,
        "r_red_selection": red_selection,
        "r_red_star": r_red_star,
        "frozen_constraints": {
            "max_A_FP_tick_rate": MAX_A_FP_TICK_RATE,
            "B_dominant_pass_fraction": B_DOMINANT_PASS_FRACTION,
        },
        "grid": {
            "window": list(WINDOW_GRID), "hysteresis": list(HYST_GRID), "dwell": list(DWELL_GRID),
            "threshold_candidates": "percentiles of pooled calibration windowed statistic, 1% steps",
        },
        "n_grid_rows": len(all_rows),
        "n_survivors_under_A_FP_constraint": len(survivors),
        "raw_ticks_csv": ROWS_CSV.name,
        "frontier_diagnostic": FRONTIER.name,
        "outcome_blind": True,
        "claim_boundary": "Calibration only. No outcome episode was run. PASS earns the frozen outcome step; it does not establish any win-rate result.",
    }

    max_b_tp = max((r["B_TP_tick_rate"] for r in survivors), default=0.0)
    if not survivors or max_b_tp <= 0.0:
        base.update({
            "selected_config": None,
            "degenerate_no_fire": True,
            "max_B_TP_among_A_FP_survivors": max_b_tp,
            "best_A_FP_achieved": float(min(r["A_FP_tick_rate"] for r in all_rows)),
            "DECISION": "NO_CONSERVATIVE_B_TRIGGER_TWO_FEATURE",
            "reason": (
                "No configuration fired at all while keeping the Pole-A false-positive tick rate "
                f"at or below {MAX_A_FP_TICK_RATE}. The A-side budget was satisfiable only by a "
                "never-firing selector, so no operating point is reported."
            ) if survivors else (
                "No configuration kept the Pole-A false-positive tick rate at or below "
                f"{MAX_A_FP_TICK_RATE}."
            ),
            "held_out": None,
            "held_out_note": "Held-out block collected but NOT scored: there is no non-degenerate configuration to score. The block is unspent for selection purposes.",
        })
        OUT.write_text(json.dumps(base, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({k: base[k] for k in ("DECISION", "n_survivors_under_A_FP_constraint",
                                               "max_B_TP_among_A_FP_survivors",
                                               "best_A_FP_achieved")}, indent=2))
        return 2

    # frozen tie-break: max B_TP; then min A switches; then prefer d=4; then smallest W
    tied = [r for r in survivors if r["B_TP_tick_rate"] >= max_b_tp - 1e-12]
    min_sw = min(r["A_mean_switches_per_episode"] for r in tied)
    tied = [r for r in tied if r["A_mean_switches_per_episode"] <= min_sw + 1e-12]
    tied.sort(key=lambda r: (r["dwell"] != 4, r["window"], r["threshold"]))
    selected = tied[0]
    print("Selected:", json.dumps(selected, indent=2), flush=True)

    d_hold = build_d_series(holdout, key_red)
    w, m, d, thr = selected["window"], selected["hysteresis"], selected["dwell"], selected["threshold"]
    ho = score({k: run_state_machine(windowed(v, w), thr, m, d) for k, v in d_hold.items()})

    check = rederive_from_csv("holdout", key_red, w, thr, m, d)
    mismatches = {k: [ho[k], check[k]] for k in ho if abs(ho[k] - check[k]) > 1e-9}
    if mismatches:
        raise SystemExit(f"ABORT: held-out statistics disagree with independent CSV re-derivation: {mismatches}")
    print("  independent CSV re-derivation PASS (6/6 exact)", flush=True)

    pass_a = ho["A_FP_tick_rate"] <= MAX_A_FP_TICK_RATE
    pass_b = ho["B_fraction_episodes_dominant_4A0D"] >= B_DOMINANT_PASS_FRACTION
    decision = ("TWO_FEATURE_CONSERVATIVE_B_TRIGGER_CALIBRATED" if (pass_a and pass_b)
                else "NO_CONSERVATIVE_B_TRIGGER_TWO_FEATURE")

    base.update({
        "selected_config": selected,
        "degenerate_no_fire": False,
        "max_B_TP_among_A_FP_survivors": max_b_tp,
        "held_out": ho,
        "independent_csv_rederivation": {"checks": len(ho), "mismatches": 0, "source": ROWS_CSV.name},
        "calibration_pass_rule": {
            f"held_out_A_FP_tick_rate<={MAX_A_FP_TICK_RATE}": pass_a,
            f"held_out_B_fraction_episodes_dominant_4A0D>={B_DOMINANT_PASS_FRACTION}": pass_b,
        },
        "DECISION": decision,
    })
    OUT.write_text(json.dumps(base, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"DECISION": decision, "selected": selected, "held_out": ho}, indent=2))
    return 0 if decision == "TWO_FEATURE_CONSERVATIVE_B_TRIGGER_CALIBRATED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
