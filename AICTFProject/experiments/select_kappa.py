"""Select kappa on CALIB, by the procedure frozen in AMENDMENT_3.

    1. Sweep candidate kappa using CALIB predictions only.
    2. Require per-cell P(commit | resolvable) >= rho in every cell WHERE
       RESOLVABLE MASS EXISTS.
    3. Require unresolved over-commitment o <= o_max.
    4. Among qualifying kappa, maximise accuracy on committed resolvable examples.
    5. If tied, choose the HIGHER, more conservative kappa.
    6. If no kappa satisfies the constraints, CALIBRATION FAILS. Do not relax them.

Step 6 is a real outcome, not a formality. It means the probe cannot commit often
enough on resolvable states without over-committing on unresolved ones -- a genuine
negative finding about the method. It must be reported as such and must NOT be
repaired by moving rho, o_max, or the architecture.

CALIB only. EVAL is never touched. This program does not choose tau, rho or o_max;
all three were frozen before the probe existed.

Run:  python experiments/select_kappa.py --device cuda
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.fit_k2_preference_probe import (                # noqa: E402
    CALIB_HI, CALIB_LO, HEAD_HIDDEN, OUT_DIR, TRUNK_SHA, build, features, load_split,
)
from rl.launch_gate import LaunchGateError                        # noqa: E402

SD = ROOT / "artifacts" / "strategic_demand"
PROTOCOL = SD / "sppo" / "ABSTAINING_SUPERVISION_PROTOCOL_DESIGN.json"
FIT_RECORD = OUT_DIR / "K2_PROBE_FIT.json"
OUT = SD / "sppo" / "KAPPA_SELECTION.json"
THRESHOLDS = SD / "sppo" / "ABSTENTION_THRESHOLDS.json"

# kappa is a max-softmax probability, so it lives in [0.5, 1.0].
KAPPA_GRID = np.round(np.arange(0.50, 0.9951, 0.005), 4)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _frozen_constraints() -> tuple[float, float, float]:
    spec = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    g = spec["GATES_JOINTLY_FROZEN"]
    rho = g["gate_2_resolved_coverage"].get("rho")
    o_max = g["gate_3_over_commitment_ceiling"].get("o_max")
    tau = g["gate_1_strategic_accuracy"].get("tau")
    if not all(isinstance(v, (int, float)) for v in (rho, o_max, tau)):
        raise LaunchGateError(
            f"REFUSING: constraints not frozen (rho={rho}, o_max={o_max}, tau={tau})")
    if "AMENDMENT_3_KAPPA_SEPARATED_FROM_TAU" not in spec:
        raise LaunchGateError("REFUSING: AMENDMENT_3 absent; the kappa procedure is not frozen")
    return float(rho), float(o_max), float(tau)


def evaluate(kappa: float, conf, correct, resolvable, cells, rho, o_max):
    """Coverage per resolvable cell, over-commitment, accuracy on committed."""
    commit = conf >= kappa
    per_cell, exempt = {}, []
    for cell in sorted(set(cells)):
        m = cells == cell
        res_m = m & resolvable
        if not res_m.any():
            exempt.append(cell)                    # ZERO_RESOLVABLE_MASS_RULE
            continue
        per_cell[cell] = float(commit[res_m].mean())
    failing = sorted(c for c, v in per_cell.items() if v < rho)

    tied = ~resolvable
    over = float(commit[tied].mean()) if tied.any() else 0.0
    committed_res = commit & resolvable
    acc = float(correct[committed_res].mean()) if committed_res.any() else float("nan")
    return {
        "kappa": float(kappa),
        "per_cell_coverage": per_cell,
        "exempt_cells": exempt,
        "cells_below_rho": failing,
        "min_cell_coverage": min(per_cell.values()) if per_cell else float("nan"),
        "over_commitment": over,
        "n_committed_resolvable": int(committed_res.sum()),
        "accuracy_on_committed_resolvable": acc,
        "qualifies": (not failing) and over <= o_max and committed_res.any(),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import torch
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"

    if OUT.is_file():
        raise SystemExit(f"REFUSING: {OUT} exists; kappa selection is one-shot")
    if not FIT_RECORD.is_file():
        raise SystemExit("REFUSING: the probe has not been fitted")
    rho, o_max, tau = _frozen_constraints()

    fit_rec = json.loads(FIT_RECORD.read_text(encoding="utf-8"))
    if fit_rec["architecture"]["trunk_sha256"] != TRUNK_SHA:
        raise LaunchGateError("REFUSING: fit record trunk hash differs from this module's")
    if fit_rec["data"]["EVAL_touched"]:
        raise LaunchGateError("REFUSING: the fit record admits EVAL access")

    qpsi, head, _, _ = build(device)
    ck = torch.load(OUT_DIR / "k2_probe_head.pt", map_location=device, weights_only=False)
    head.load_state_dict(ck["state_dict"])
    head.eval()

    calib = load_split(CALIB_LO, CALIB_HI, resolvable_only=False)
    with torch.no_grad():
        probs = torch.softmax(head(features(qpsi, calib, device)), dim=1)
        conf = probs.max(dim=1).values.cpu().numpy()
        pred = probs.argmax(dim=1).cpu().numpy()

    resolvable = calib["delta"] != 0
    correct = pred == calib["label"]
    cells = calib["cell"]

    print(f"KAPPA SELECTION  {_now()}")
    print(f"  CALIB {CALIB_LO}..{CALIB_HI}: {len(conf)} states, {int(resolvable.sum())} resolvable")
    print(f"  frozen constraints: rho={rho}  o_max={o_max}   (tau={tau} is an EVAL gate, not used here)")
    print(f"  confidence range [{conf.min():.3f}, {conf.max():.3f}]\n")

    rows = [evaluate(k, conf, correct, resolvable, cells, rho, o_max) for k in KAPPA_GRID]
    qualifying = [r for r in rows if r["qualifies"]]

    print(f"  {'kappa':>6s} {'min cell cov':>13s} {'over-commit':>12s} {'n cmt res':>10s} {'acc':>7s}  qualifies")
    for r in rows[::10]:
        acc = r["accuracy_on_committed_resolvable"]
        print(f"  {r['kappa']:6.3f} {r['min_cell_coverage']:13.3f} {r['over_commitment']:12.3f} "
              f"{r['n_committed_resolvable']:10d} {acc:7.3f}  {'YES' if r['qualifies'] else ''}")

    if not qualifying:
        best = max(rows, key=lambda r: (not r["cells_below_rho"], -r["over_commitment"]))
        rec = {
            "record": "kappa selection on CALIB", "status": "FROZEN_RESULT", "utc": _now(),
            "VERDICT": "CALIBRATION_FAILED",
            "meaning": (
                "No kappa satisfies the frozen constraints simultaneously. The probe "
                "cannot commit often enough on resolvable states without over-committing "
                "on unresolved ones. This is a genuine negative finding about the method "
                "as configured, NOT a plumbing failure."),
            "prohibited_repairs": [
                "lowering rho or raising o_max",
                "changing the probe architecture after seeing this result",
                "re-fitting with cell weighting or oversampling",
                "selecting kappa on any split other than CALIB"],
            "must_be_read_with": fit_rec["DOCUMENTED_LIMITATION"],
            "frozen_constraints": {"rho": rho, "o_max": o_max},
            "closest_candidate": best,
            "sweep": rows,
        }
        OUT.write_text(json.dumps(rec, indent=2), encoding="utf-8")
        print(f"\n  VERDICT: CALIBRATION_FAILED -- no kappa satisfies both constraints")
        print(f"  -> {OUT}")
        return 1

    # steps 4 and 5: maximise accuracy, break ties toward the higher kappa
    best_acc = max(r["accuracy_on_committed_resolvable"] for r in qualifying)
    tied_best = [r for r in qualifying
                 if abs(r["accuracy_on_committed_resolvable"] - best_acc) < 1e-12]
    chosen = max(tied_best, key=lambda r: r["kappa"])

    rec = {
        "record": "kappa selection on CALIB", "status": "FROZEN_RESULT", "utc": _now(),
        "VERDICT": "KAPPA_SELECTED",
        "kappa": chosen["kappa"],
        "procedure": "AMENDMENT_3, frozen before the probe existed",
        "selected_on": "CALIB only; EVAL untouched",
        "frozen_constraints": {"rho": rho, "o_max": o_max},
        "chosen_operating_point": chosen,
        "n_qualifying_kappa": len(qualifying),
        "tie_break_applied": len(tied_best) > 1,
        "tie_break_rule": "higher, more conservative kappa",
        "calib_accuracy_is_not_the_gate": (
            f"Accuracy here is a SELECTION criterion on CALIB. Gate 1 (tau={tau}) is "
            "scored on EVAL, which has not been touched."),
        "must_be_read_with": fit_rec["DOCUMENTED_LIMITATION"],
        "sweep": rows,
    }
    OUT.write_text(json.dumps(rec, indent=2), encoding="utf-8")

    THRESHOLDS.write_text(json.dumps({
        "record": "Frozen abstention thresholds",
        "status": "FROZEN", "utc": _now(),
        "calibrated_on": "CALIB",
        "thresholds": {"tau": tau, "rho": rho, "o_max": o_max, "kappa": chosen["kappa"]},
        "provenance": {
            "tau": "frozen 2026-08-29 before any probe existed (AMENDMENT_4); EVAL gate",
            "rho": "frozen 2026-08-29 (AMENDMENT_2)",
            "o_max": "frozen 2026-08-29 (AMENDMENT_2)",
            "kappa": "selected on CALIB by the AMENDMENT_3 procedure, frozen before the probe was fitted"},
        "EVAL_touched": False,
    }, indent=2), encoding="utf-8")

    print(f"\n  VERDICT: KAPPA_SELECTED   kappa = {chosen['kappa']:.3f}")
    print(f"    min cell coverage {chosen['min_cell_coverage']:.3f} (rho {rho})")
    print(f"    over-commitment   {chosen['over_commitment']:.3f} (o_max {o_max})")
    print(f"    CALIB accuracy on committed resolvable {chosen['accuracy_on_committed_resolvable']:.3f}")
    print(f"  -> {OUT}\n  -> {THRESHOLDS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
