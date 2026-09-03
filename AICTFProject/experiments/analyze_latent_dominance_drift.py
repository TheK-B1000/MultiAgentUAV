"""Exploratory: does continued training systematically favour one latent over the other?

MOTIVATION. Across the ladder, delta_A = V(z0,A) - V(z1,A) and delta_B = V(z1,B) - V(z0,B)
have moved in a suspiciously coordinated way. Note that a FALL in delta_A and a RISE in
delta_B both mean the same underlying thing: z1 gaining ground on z0. If fine-tuning
systematically shifts advantage toward z1 on BOTH poles, that is latent DOMINANCE drift, not
"pole inversion" -- and it is a different, more tractable phenomenon than a crossover failure.

This script tests that reading against every frozen eval row file this program has produced.
It runs on already-collected data only: no training, no new rollouts, no GPU.

STATUS: EXPLORATORY / HYPOTHESIS-GENERATING. This analysis was designed AFTER seeing the
RSCFT result, so it is explicitly NOT a preregistered test and nothing here may be reported
as a confirmatory finding. It exists to characterise a pattern precisely enough that a
FUTURE preregistered test could be written against it.

IMPORTANT LIMITATION, stated up front: every method was evaluated on its OWN disjoint sealed
seed block (32 or 64 seeds). Cross-method comparisons are therefore UNPAIRED and differ in
seed difficulty as well as in method. That is why per-method specialist references (pi_A on
Pole A, pi_B on Pole B, scored on the SAME block) are reported alongside -- they give a
per-block difficulty anchor that the latent numbers can be read against.

Run:  python experiments/analyze_latent_dominance_drift.py
"""
from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.eval_hog_psp_v3 import _mean_ci

SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
OUT = SD / "LATENT_DOMINANCE_DRIFT_ANALYSIS.json"

# (label, file, arm-or-None). Ladder order: earliest method -> latest.
SOURCES = [
    ("V1 (oracle-gated K=2)", "oracle_gated_k2_eval_rows.csv", None),
    ("OG-PSP", "og_psp_eval_rows.csv", None),
    ("H-OG-PSP V3", "hog_psp_v3_eval_rows.csv", None),
    ("H-OG-PSP V4", "hog_psp_v4_eval_rows.csv", None),
    ("CCP successor (incumbent)", "ccp_successor_eval_rows.csv", None),
    ("CCP-S2 CONTROL", "ccp_s2_eval_rows.csv", "CONTROL"),
    ("CCP-S2 TREATMENT", "ccp_s2_eval_rows.csv", "TREATMENT"),
    ("RSCFT CONTROL", "rscft_eval_rows.csv", "CONTROL"),
    ("RSCFT TREATMENT", "rscft_eval_rows.csv", "TREATMENT"),
]


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_cells(fname: str, arm: str | None) -> dict:
    """-> {(policy_label, pole): {seed: win}} for one method/arm."""
    rows = list(csv.DictReader((SD / fname).open(encoding="utf-8")))
    out: dict = {}
    for r in rows:
        if arm is not None:
            if r.get("arm") != arm:
                continue
            label = r["z"]                       # 'z0' / 'z1'
        else:
            label = r["policy"]                  # 'z0' / 'z1' / 'pi_A' / 'pi_B'
        out.setdefault((label, r["pole"]), {})[int(r["seed"])] = int(r["win"])
    return out


def wins(cells: dict, label: str, pole: str) -> np.ndarray | None:
    d = cells.get((label, pole))
    if not d:
        return None
    return np.array([d[s] for s in sorted(d)], dtype=np.float64)


def main() -> int:
    if OUT.is_file():
        raise SystemExit(f"REFUSING: {OUT.name} exists; delete it to re-run this analysis")

    print(f"LATENT DOMINANCE DRIFT -- EXPLORATORY  {_now()}")
    print("  (post-hoc analysis of frozen rows; NOT a preregistered test)\n")
    header = (f"  {'method':28s} {'V(z0,A)':>8s} {'V(z1,A)':>8s} {'V(z0,B)':>8s} "
              f"{'V(z1,B)':>8s} {'dA':>8s} {'dB':>8s} {'z1-z0':>8s}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    records = []
    for label, fname, arm in SOURCES:
        cells = load_cells(fname, arm)
        w = {k: wins(cells, *k) for k in (("z0", "A"), ("z1", "A"), ("z0", "B"), ("z1", "B"))}
        if any(v is None for v in w.values()):
            print(f"  {label:28s}  (missing cells, skipped)")
            continue
        v = {k: float(x.mean()) for k, x in w.items()}
        d_a = _mean_ci(w[("z0", "A")] - w[("z1", "A")])
        d_b = _mean_ci(w[("z1", "B")] - w[("z0", "B")])
        # the unified quantity: mean z1 advantage over z0 ACROSS BOTH POLES.
        # positive => z1 is the dominant latent overall, regardless of intended assignment.
        z1_minus_z0 = _mean_ci(
            0.5 * ((w[("z1", "A")] - w[("z0", "A")]) + (w[("z1", "B")] - w[("z0", "B")])))

        spec = {}
        for sp, pole in (("pi_A", "A"), ("pi_B", "B")):
            sw = wins(cells, sp, pole)
            if sw is not None:
                spec[f"{sp}@{pole}"] = float(sw.mean())

        n_seeds = len(w[("z0", "A")])
        records.append({
            "method": label, "n_seeds": n_seeds,
            "V": {f"{k[0]}@{k[1]}": val for k, val in v.items()},
            "delta_A": d_a, "delta_B": d_b,
            "z1_minus_z0_both_poles": z1_minus_z0,
            "specialist_reference_same_block": spec,
        })
        print(f"  {label:28s} {v[('z0','A')]:8.4f} {v[('z1','A')]:8.4f} {v[('z0','B')]:8.4f} "
              f"{v[('z1','B')]:8.4f} {d_a['mean']:+8.4f} {d_b['mean']:+8.4f} "
              f"{z1_minus_z0['mean']:+8.4f}")

    # Did the fine-tuned arms move toward z1 relative to their own warm start?
    inc = next((r for r in records if r["method"].startswith("CCP successor")), None)
    finetuned = [r for r in records if r["method"].startswith(("CCP-S2", "RSCFT"))]
    drift = None
    if inc and finetuned:
        base = inc["z1_minus_z0_both_poles"]["mean"]
        moves = [{"method": r["method"],
                  "z1_minus_z0": r["z1_minus_z0_both_poles"]["mean"],
                  "shift_vs_incumbent": r["z1_minus_z0_both_poles"]["mean"] - base}
                 for r in finetuned]
        n_toward_z1 = sum(1 for m in moves if m["shift_vs_incumbent"] > 0)
        drift = {"incumbent_z1_minus_z0": base, "arms": moves,
                 "n_arms_shifted_toward_z1": n_toward_z1, "n_arms": len(moves)}
        print(f"\n  incumbent z1-z0 (both poles): {base:+.4f}")
        for m in moves:
            print(f"    {m['method']:26s} {m['z1_minus_z0']:+.4f}  "
                  f"shift {m['shift_vs_incumbent']:+.4f}")
        print(f"  arms shifted toward z1: {n_toward_z1}/{len(moves)}")

    OUT.write_text(json.dumps({
        "record": "Latent dominance drift -- exploratory analysis",
        "status": "EXPLORATORY_NOT_PREREGISTERED", "utc": _now(),
        "designed_after_seeing_results": True,
        "may_not_be_reported_as_confirmatory": True,
        "hypothesis": "A fall in delta_A and a rise in delta_B both mean z1 gaining on z0. "
                      "If continued fine-tuning shifts advantage toward z1 on BOTH poles, the "
                      "phenomenon is latent dominance drift, not pole inversion.",
        "limitation": "every method used its OWN disjoint sealed seed block (32 or 64 seeds), "
                      "so cross-method comparison is UNPAIRED and confounds seed difficulty "
                      "with method. Per-method specialist references on the same block are "
                      "reported as a difficulty anchor.",
        "methods": records,
        "drift_vs_incumbent": drift,
    }, indent=2), encoding="utf-8")
    print(f"\n  -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
