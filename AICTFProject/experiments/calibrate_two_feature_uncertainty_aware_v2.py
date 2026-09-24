"""COMPOSITION_SELECTOR_TWO_FEATURE_UNCERTAINTY_AWARE_V2_SPEC.json -- one pass.

The ONLY change versus `calibrate_two_feature_asymmetric_b_trigger.py` is how a
calibration configuration becomes eligible:

    A_FP_hat <= 0.10            ->      UCB95_episode(A_FP) <= 0.10

The statistic, grid, dwell, hysteresis, windows, threshold rule, tie-break
order, state machine and the held-out pass rule are unchanged; the grid
constants and the state machine are imported so they cannot drift.

The held-out rule stays a POINT rule. The bootstrap makes calibration
selection conservative; it does not redefine the operational gate.

Outcome-blind: no score/win/reward/return/draw is read, computed, printed or
serialized. Pole labels are used ONLY for FP/TP accounting.
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
from experiments.calibrate_two_feature_asymmetric_b_trigger import (  # noqa: E402
    RED_RADII,
    collect_block,
    gate_direction,
    min_misclass,
    persist_rows,
)
import experiments.calibrate_two_feature_asymmetric_b_trigger as v1  # noqa: E402

ART = ROOT / "artifacts/strategic_demand/sppo"
SPEC = ART / "COMPOSITION_SELECTOR_TWO_FEATURE_UNCERTAINTY_AWARE_V2_SPEC.json"
DERIVATION = ART / "COMPOSITION_SELECTOR_FEATURE_DERIVATION_TRACE.json"
V1_ROWS = ART / "composition_selector_two_feature_asymmetric_raw_ticks.csv"
OUT = ART / "COMPOSITION_SELECTOR_TWO_FEATURE_UNCERTAINTY_AWARE_V2_RESULT.json"
FRONTIER = ART / "COMPOSITION_SELECTOR_TWO_FEATURE_UNCERTAINTY_AWARE_V2_FRONTIER_DIAGNOSTIC.json"
ROWS_CSV = ART / "composition_selector_two_feature_v2_raw_ticks.csv"

R_BLUE = 4.0
R_RED_FROZEN = 4.0
KEY_RED = f"p_red_r{int(R_RED_FROZEN)}"
CALIB_SEEDS = list(range(99_900_101, 99_900_101 + 16))
HOLDOUT_SEEDS = list(range(99_900_501, 99_900_501 + 16))
DETERMINISM_TOL = 1e-6

# frozen before scoring, see the spec's BOOTSTRAP_SPECIFICATION block
N_BOOT = 20_000
BOOT_SEED = 20_260_919
UCB_PERCENTILE = 95.0
UCB_CAP = MAX_A_FP_TICK_RATE
CHUNK = 512
UCB_FRONTIER_CAPS = [round(0.02 * i, 2) for i in range(0, 28)]

Key = tuple[str, int]


# ------------------------------------------------------------ persisted input

def load_calibration_from_v1_rows() -> dict[Key, dict[str, list[int]]]:
    """Reuse the V1 per-tick rows rather than re-simulating: identical data, and
    it removes any chance of the two runs disagreeing about the calibration set."""
    if not V1_ROWS.is_file():
        raise SystemExit(f"ABORT: persisted V1 calibration rows missing: {V1_ROWS}")
    block: dict[Key, dict[str, list[int]]] = {}
    ticks: dict[Key, list[int]] = {}
    with V1_ROWS.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["split"] != "calibration":
                continue
            k = (row["pole"], int(row["seed"]))
            s = block.setdefault(k, {"p_blue": []})
            for r in RED_RADII:
                s.setdefault(f"p_red_r{int(r)}", [])
            ticks.setdefault(k, []).append(int(row["tick"]))
            s["p_blue"].append(int(float(row["p_blue"])))
            for r in RED_RADII:
                s[f"p_red_r{int(r)}"].append(int(float(row[f"p_red_r{int(r)}"])))
    if not block:
        raise SystemExit("ABORT: no calibration rows found -- absence is an error state")
    expected = {("A", s) for s in CALIB_SEEDS} | {("B", s) for s in CALIB_SEEDS}
    if set(block) != expected:
        raise SystemExit(f"ABORT: calibration episode set mismatch: {sorted(set(block) ^ expected)}")
    for k, tk in ticks.items():
        if tk != sorted(tk) or tk != list(range(len(tk))):
            raise SystemExit(f"ABORT: {k} tick index is not a dense ordered range")
    return block


def gate_determinism(calib: dict[Key, dict[str, list[int]]]) -> dict[str, float]:
    if not DERIVATION.is_file():
        raise SystemExit(f"ABORT: sealed derivation anchor missing: {DERIVATION}")
    anchor = json.loads(DERIVATION.read_text(encoding="utf-8"))["radii"]["4.0"]
    got_a = float(np.mean([np.mean(calib[("A", s)]["p_blue"]) for s in CALIB_SEEDS]))
    got_b = float(np.mean([np.mean(calib[("B", s)]["p_blue"]) for s in CALIB_SEEDS]))
    da, db = abs(got_a - float(anchor["A_mean"])), abs(got_b - float(anchor["B_mean"]))
    if da > DETERMINISM_TOL or db > DETERMINISM_TOL:
        raise SystemExit(f"ABORT: calibration rows do not reproduce the sealed derivation "
                         f"(A delta {da:.3e}, B delta {db:.3e})")
    print(f"  determinism gate PASS (A {got_a:.10f}, B {got_b:.10f})", flush=True)
    return {"A": got_a, "B": got_b, "max_abs_delta": max(da, db)}


def gate_r_red(calib: dict[Key, dict[str, list[int]]]) -> dict[str, Any]:
    """Consistency check only. r_red is frozen at 4.0 by the V1 result; this
    re-derivation must agree, and an disagreement aborts rather than re-selects."""
    sel: dict[str, Any] = {}
    for r in RED_RADII:
        key = f"p_red_r{int(r)}"
        a = np.array([np.mean(calib[("A", s)][key]) for s in CALIB_SEEDS])
        b = np.array([np.mean(calib[("B", s)][key]) for s in CALIB_SEEDS])
        thr, err, a_higher = min_misclass(a, b)
        sel[str(r)] = {"A_mean": float(a.mean()), "B_mean": float(b.mean()),
                       "threshold": thr, "pooled_misclassification_rate": err,
                       "pole_A_higher": a_higher}
    rederived = min(RED_RADII, key=lambda r: sel[str(r)]["pooled_misclassification_rate"])
    if rederived != R_RED_FROZEN:
        raise SystemExit(f"ABORT: r_red re-derivation returned {rederived}, frozen value is {R_RED_FROZEN}")
    print(f"  r_red consistency gate PASS ({R_RED_FROZEN})", flush=True)
    return sel


# ---------------------------------------------------------------- bootstrap

def make_boot_index(n_ep: int) -> np.ndarray:
    """One common-random-number resample matrix, drawn once from the frozen
    seed and reused for every configuration (declared in the spec)."""
    rng = np.random.default_rng(BOOT_SEED)
    return rng.integers(0, n_ep, size=(N_BOOT, n_ep), dtype=np.int64)


def ucb95_batch(rates: np.ndarray, boot_idx: np.ndarray) -> np.ndarray:
    """rates: (n_configs, n_episodes) per-episode Pole-A false-positive rates.
    Returns the one-sided 95th-percentile bootstrap upper bound per config.
    Uses a counts matrix so the whole batch is a single BLAS matmul."""
    n_ep = rates.shape[1]
    counts = np.zeros((boot_idx.shape[0], n_ep), dtype=np.float64)
    for col in range(n_ep):
        counts[:, col] = (boot_idx == col).sum(axis=1)
    counts /= n_ep
    out = np.empty(rates.shape[0], dtype=np.float64)
    for lo in range(0, rates.shape[0], CHUNK):
        hi = min(lo + CHUNK, rates.shape[0])
        means = counts @ rates[lo:hi].T          # (n_boot, chunk)
        out[lo:hi] = np.percentile(means, UCB_PERCENTILE, axis=0)
    return out


# --------------------------------------------------------------------- main

def main() -> int:
    if not SPEC.is_file():
        raise SystemExit(f"REFUSING: frozen spec missing: {SPEC}")
    if OUT.exists():
        raise SystemExit(f"REFUSING: result already exists (one pass only): {OUT}")

    print("Loading persisted calibration rows (99900101-116, both poles)...", flush=True)
    calib = load_calibration_from_v1_rows()
    determinism = gate_determinism(calib)
    r_red_selection = gate_r_red(calib)

    d_calib = {k: [float(pb - pr) for pb, pr in zip(s["p_blue"], s[KEY_RED])]
               for k, s in calib.items()}
    direction = gate_direction(d_calib)

    a_keys = sorted(k for k in d_calib if k[0] == "A")
    b_keys = sorted(k for k in d_calib if k[0] == "B")
    boot_idx = make_boot_index(len(a_keys))
    print(f"  bootstrap: {N_BOOT} resamples of {len(a_keys)} Pole-A episodes, "
          f"seed {BOOT_SEED}, one-sided p{UCB_PERCENTILE:g}", flush=True)

    print("Scanning the frozen grid with uncertainty-aware eligibility...", flush=True)
    quantiles = np.arange(THRESHOLD_QUANTILE_STEP, 100.0, THRESHOLD_QUANTILE_STEP)
    rows: list[dict[str, Any]] = []
    a_rate_rows: list[np.ndarray] = []
    for w in WINDOW_GRID:
        win = {k: windowed(v, w) for k, v in d_calib.items()}
        pooled = np.concatenate(list(win.values()))
        for m in HYST_GRID:
            for d in DWELL_GRID:
                for thr in np.unique(np.percentile(pooled, quantiles)):
                    preds = {k: run_state_machine(v, float(thr), m, d) for k, v in win.items()}
                    s = score(preds)
                    per_ep_a = np.array([float(np.mean(preds[k] == 1)) for k in a_keys])
                    # the two definitions must coincide; they only do at equal lengths
                    if abs(float(per_ep_a.mean()) - s["A_FP_tick_rate"]) > 1e-12:
                        raise SystemExit(
                            "ABORT: mean of per-episode A rates != pooled A tick rate "
                            f"({per_ep_a.mean()!r} vs {s['A_FP_tick_rate']!r}); episode "
                            "lengths must be equal for the constraint to keep its meaning"
                        )
                    rows.append({"window": w, "hysteresis": m, "dwell": d,
                                 "threshold": float(thr), **s})
                    a_rate_rows.append(per_ep_a)
        print(f"  W={w} done ({len(rows)} rows)", flush=True)

    print("Computing episode-level UCB95 for every configuration...", flush=True)
    ucb = ucb95_batch(np.vstack(a_rate_rows), boot_idx)
    for row, u in zip(rows, ucb):
        row["A_FP_UCB95_episode"] = float(u)
    eligible = [r for r in rows if r["A_FP_UCB95_episode"] <= UCB_CAP]
    print(f"  eligible under UCB95 <= {UCB_CAP}: {len(eligible)} of {len(rows)}", flush=True)

    # descriptive-only frontier
    frontier = []
    for cap in UCB_FRONTIER_CAPS:
        elig = [r for r in rows if r["A_FP_UCB95_episode"] <= cap]
        if not elig:
            frontier.append({"UCB95_cap": cap, "n_eligible": 0, "max_B_TP": None, "config": None})
            continue
        best = max(elig, key=lambda r: r["B_TP_tick_rate"])
        frontier.append({
            "UCB95_cap": cap, "n_eligible": len(elig),
            "max_B_TP": best["B_TP_tick_rate"],
            "B_fraction_episodes_dominant_4A0D": best["B_fraction_episodes_dominant_4A0D"],
            "point_A_FP_of_that_config": best["A_FP_tick_rate"],
            "config": {k: best[k] for k in ("window", "hysteresis", "dwell", "threshold")},
        })
    FRONTIER.write_text(json.dumps({
        "record_id": "COMPOSITION_SELECTOR_TWO_FEATURE_UNCERTAINTY_AWARE_V2_FRONTIER_DIAGNOSTIC",
        "purpose": "DESCRIPTIVE ONLY; changes no frozen decision and may not be used to pick an operating point.",
        "eligibility_statistic": f"episode-level bootstrap UCB{UCB_PERCENTILE:g} of Pole-A false-positive rate",
        "frozen_cap_unaffected": UCB_CAP,
        "n_grid_rows": len(rows),
        "frontier_UCB95_cap_vs_max_B_TP": frontier,
        "comparable_point_estimate_record": "COMPOSITION_SELECTOR_TWO_FEATURE_ASYMMETRIC_FRONTIER_DIAGNOSTIC.json",
    }, indent=2) + "\n", encoding="utf-8")
    print(f"  frontier -> {FRONTIER.name}", flush=True)

    base: dict[str, Any] = {
        "record_id": "COMPOSITION_SELECTOR_TWO_FEATURE_UNCERTAINTY_AWARE_V2_RESULT",
        "status": "COMPLETE",
        "implements": str(SPEC.relative_to(ROOT)).replace("\\", "/"),
        "statistic": f"D_t = P_blue(r={R_BLUE}) - P_red(r={R_RED_FROZEN})",
        "eligibility_rule": f"episode-level bootstrap UCB{UCB_PERCENTILE:g}(A_FP) <= {UCB_CAP}",
        "held_out_rule_unchanged": f"A_FP <= {UCB_CAP} AND B_dominant >= {B_DOMINANT_PASS_FRACTION} (point rule, no bootstrap)",
        "calibration_seeds": f"{CALIB_SEEDS[0]}-{CALIB_SEEDS[-1]}",
        "holdout_seeds": f"{HOLDOUT_SEEDS[0]}-{HOLDOUT_SEEDS[-1]}",
        "bootstrap": {"n_boot": N_BOOT, "rng_seed": BOOT_SEED, "percentile": UCB_PERCENTILE,
                      "resampling_unit": "Pole-A calibration episode", "n_episodes": len(a_keys),
                      "common_random_numbers": True},
        "determinism_gate": determinism,
        "direction_gate": direction,
        "r_red_frozen": R_RED_FROZEN,
        "r_red_consistency_recheck": r_red_selection,
        "grid": {"window": list(WINDOW_GRID), "hysteresis": list(HYST_GRID),
                 "dwell": list(DWELL_GRID),
                 "threshold_candidates": "percentiles of pooled calibration windowed statistic, 1% steps"},
        "n_grid_rows": len(rows),
        "n_eligible_under_UCB": len(eligible),
        "n_would_be_eligible_under_V1_point_rule": sum(1 for r in rows if r["A_FP_tick_rate"] <= UCB_CAP),
        "frontier_diagnostic": FRONTIER.name,
        "calibration_reuse_disclosure": "Calibration block 99900101-116 informed selection twice (V1 point rule, V2 UCB rule). Calibration-side numbers are not independent evidence. The fresh held-out block is the only clean evidence here.",
        "outcome_blind": True,
        "claim_boundary": "Calibration only. No outcome episode was run. PASS earns the frozen outcome step; it does not establish any win-rate result.",
    }

    best_b = max((r["B_TP_tick_rate"] for r in eligible), default=0.0)
    if not eligible or best_b <= 0.0:
        base.update({
            "selected_config": None, "degenerate_or_empty": True,
            "max_B_TP_among_eligible": best_b,
            "min_UCB95_achieved": float(min(r["A_FP_UCB95_episode"] for r in rows)),
            "DECISION": "NO_UNCERTAINTY_ELIGIBLE_B_TRIGGER",
            "reason": ("No configuration is demonstrably safe at the 0.10 operational limit: "
                       f"no grid row achieved UCB95 <= {UCB_CAP} with any B detection."),
            "held_out": None,
            "held_out_note": "Held-out block NOT collected or scored: there is no eligible configuration to score. 99900501-516 remains unspent.",
        })
        OUT.write_text(json.dumps(base, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({k: base[k] for k in ("DECISION", "n_eligible_under_UCB",
                                               "max_B_TP_among_eligible", "min_UCB95_achieved")}, indent=2))
        return 2

    tied = [r for r in eligible if r["B_TP_tick_rate"] >= best_b - 1e-12]
    min_sw = min(r["A_mean_switches_per_episode"] for r in tied)
    tied = [r for r in tied if r["A_mean_switches_per_episode"] <= min_sw + 1e-12]
    tied.sort(key=lambda r: (r["dwell"] != 4, r["window"], r["threshold"]))
    selected = tied[0]
    print("Selected:", json.dumps(selected, indent=2), flush=True)

    print("Collecting FRESH held-out traces (99900501-516, both poles)...", flush=True)
    holdout = collect_block(HOLDOUT_SEEDS, "holdout")
    # persist_rows and rederive_from_csv both address the V1 module global. Point
    # it at THIS run's file so the V1 artifact is never overwritten.
    if ROWS_CSV == V1_ROWS:
        raise SystemExit("ABORT: V2 rows path collides with the sealed V1 rows artifact")
    v1.ROWS_CSV = ROWS_CSV
    persist_rows({"calibration": calib, "holdout": holdout})
    print(f"  persisted per-tick rows -> {ROWS_CSV.name}", flush=True)

    d_hold = {k: [float(pb - pr) for pb, pr in zip(s["p_blue"], s[KEY_RED])]
              for k, s in holdout.items()}
    w, m, d, thr = selected["window"], selected["hysteresis"], selected["dwell"], selected["threshold"]
    ho_preds = {k: run_state_machine(windowed(v, w), thr, m, d) for k, v in d_hold.items()}
    ho = score(ho_preds)

    check = v1.rederive_from_csv("holdout", KEY_RED, w, thr, m, d)
    mismatches = {k: [ho[k], check[k]] for k in ho if abs(ho[k] - check[k]) > 1e-9}
    if mismatches:
        raise SystemExit(f"ABORT: held-out statistics disagree with independent CSV re-derivation: {mismatches}")
    print("  independent CSV re-derivation PASS (6/6 exact)", flush=True)

    ho_a = np.array([float(np.mean(ho_preds[k] == 1)) for k in sorted(k for k in ho_preds if k[0] == "A")])
    ho_b = np.array([float(np.mean(ho_preds[k] == 1)) for k in sorted(k for k in ho_preds if k[0] == "B")])
    ho_boot = (np.zeros(1) if len(ho_a) == 0 else
               np.percentile(ho_a[make_boot_index(len(ho_a))].mean(axis=1), [2.5, 50.0, 97.5]))

    pass_a = ho["A_FP_tick_rate"] <= UCB_CAP
    pass_b = ho["B_fraction_episodes_dominant_4A0D"] >= B_DOMINANT_PASS_FRACTION
    decision = ("CONSERVATIVE_B_TRIGGER_CALIBRATED_V2" if (pass_a and pass_b)
                else "NO_CONSERVATIVE_B_TRIGGER_V2")

    base.update({
        "selected_config": selected,
        "degenerate_or_empty": False,
        "max_B_TP_among_eligible": best_b,
        "selected_config_margin": {
            "calibration_point_A_FP": selected["A_FP_tick_rate"],
            "calibration_UCB95_A_FP": selected["A_FP_UCB95_episode"],
            "margin_of_point_estimate_below_cap": UCB_CAP - selected["A_FP_tick_rate"],
            "v1_comparison": "V1's selected config had a point A_FP of 0.09635416666666667, margin 0.0036.",
        },
        "held_out": ho,
        "held_out_descriptive_non_gating": {
            "A_per_episode": [round(float(x), 6) for x in np.sort(ho_a)],
            "B_per_episode": [round(float(x), 6) for x in np.sort(ho_b)],
            "A_episode_bootstrap_p2.5_p50_p97.5": [round(float(x), 6) for x in np.atleast_1d(ho_boot)],
            "note": "Descriptive only. The held-out gate is the unchanged point rule; no bound here can convert a pass or a fail.",
        },
        "independent_csv_rederivation": {"checks": len(ho), "mismatches": 0, "source": ROWS_CSV.name},
        "held_out_pass_rule": {
            f"held_out_A_FP_tick_rate<={UCB_CAP}": pass_a,
            f"held_out_B_fraction_episodes_dominant_4A0D>={B_DOMINANT_PASS_FRACTION}": pass_b,
        },
        "DECISION": decision,
    })
    OUT.write_text(json.dumps(base, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"DECISION": decision, "selected": selected, "held_out": ho}, indent=2))
    return 0 if decision == "CONSERVATIVE_B_TRIGGER_CALIBRATED_V2" else 2


if __name__ == "__main__":
    raise SystemExit(main())
