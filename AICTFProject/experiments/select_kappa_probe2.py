"""Select kappa for probe 2, thresholding P(resolvable | s).

Identical procedure to probe 1 (AMENDMENT_3, frozen before either probe existed).
The ONLY difference is the quantity kappa cuts:

    probe 1   P(my A/B guess is right)      -- measured AUC 0.415, the wrong quantity
    probe 2   P(preference is resolvable)   -- explicitly supervised

Direction predictions still come from the frozen probe-1 head. CALIB only; EVAL is
never touched.

Run:  python experiments/select_kappa_probe2.py --device cuda
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.fit_k2_preference_probe import (                 # noqa: E402
    CALIB_HI, CALIB_LO, HEAD_HIDDEN, OUT_DIR, build, features, load_split,
)
from experiments.select_kappa import _frozen_constraints, evaluate  # noqa: E402
from rl.launch_gate import LaunchGateError                        # noqa: E402

SD = ROOT / "artifacts" / "strategic_demand"
OUT = SD / "sppo" / "KAPPA_SELECTION_PROBE2.json"
THRESHOLDS = SD / "sppo" / "ABSTENTION_THRESHOLDS.json"
KAPPA_GRID = np.round(np.arange(0.02, 0.981, 0.005), 4)   # sigmoid output spans [0,1]


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _auc(score, label):
    order = np.argsort(score)
    ranks = np.empty(len(score), dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1)
    n1 = float(label.sum()); n0 = float(len(label) - n1)
    return float((ranks[label == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import torch
    from torch import nn
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"

    if OUT.is_file():
        raise SystemExit(f"REFUSING: {OUT} exists; kappa selection is one-shot")
    fit_rec = OUT_DIR / "PROBE2_FIT.json"
    if not fit_rec.is_file():
        raise SystemExit("REFUSING: probe 2 has not been fitted")
    if json.loads(fit_rec.read_text(encoding="utf-8"))["data"]["EVAL_touched"]:
        raise LaunchGateError("REFUSING: probe-2 fit record admits EVAL access")
    rho, o_max, tau = _frozen_constraints()

    qpsi, dir_head, _, _ = build(device)
    dir_head.load_state_dict(torch.load(OUT_DIR / "k2_probe_head.pt", map_location=device,
                                        weights_only=False)["state_dict"])
    dir_head.eval()
    res_head = nn.Sequential(nn.Linear(256, HEAD_HIDDEN), nn.ReLU(),
                             nn.Linear(HEAD_HIDDEN, 1)).to(device)
    res_head.load_state_dict(torch.load(OUT_DIR / "resolvability_head.pt", map_location=device,
                                        weights_only=False)["state_dict"])
    res_head.eval()

    calib = load_split(CALIB_LO, CALIB_HI, resolvable_only=False)
    with torch.no_grad():
        h = features(qpsi, calib, device)
        p_res = torch.sigmoid(res_head(h).squeeze(-1)).cpu().numpy()
        pred = torch.softmax(dir_head(h), dim=1).argmax(1).cpu().numpy()

    resolvable = calib["delta"] != 0
    correct = pred == calib["label"]
    auc = _auc(p_res, resolvable.astype(int))

    print(f"KAPPA SELECTION -- PROBE 2  {_now()}")
    print(f"  CALIB: {len(p_res)} states, {int(resolvable.sum())} resolvable")
    print(f"  frozen constraints rho={rho}  o_max={o_max}   (tau={tau} is an EVAL gate)")
    print(f"  P(resolvable) range [{p_res.min():.3f}, {p_res.max():.3f}]")
    print(f"  AUC on CALIB = {auc:.4f}   (probe 1 direction-confidence achieved 0.4150)\n")

    rows = [evaluate(k, p_res, correct, resolvable, calib["cell"], rho, o_max)
            for k in KAPPA_GRID]
    qualifying = [r for r in rows if r["qualifies"]]

    print(f"  {'kappa':>6s} {'min cell cov':>13s} {'over-commit':>12s} {'n cmt res':>10s} {'acc':>7s}  ok")
    for r in rows[::12]:
        a = r["accuracy_on_committed_resolvable"]
        print(f"  {r['kappa']:6.3f} {r['min_cell_coverage']:13.3f} {r['over_commitment']:12.3f} "
              f"{r['n_committed_resolvable']:10d} {a:7.3f}  {'YES' if r['qualifies'] else ''}")

    base = {
        "record": "kappa selection on CALIB -- probe 2 (resolvability head)",
        "status": "FROZEN_RESULT", "utc": _now(),
        "implements": "TWO_HEAD_RESOLVABILITY_PROBE_SPEC.json + AMENDMENT_3 procedure",
        "quantity_thresholded": "P(preference is resolvable | s)",
        "calib_auc_resolvability": round(auc, 4),
        "probe1_comparison": {"probe1_auc": 0.4150, "probe1_verdict": "CALIBRATION_FAILED",
                              "only_change": "resolvability supervised explicitly"},
        "frozen_constraints": {"rho": rho, "o_max": o_max},
        "selected_on": "CALIB only; EVAL untouched",
        "sweep": rows,
    }

    if not qualifying:
        base.update({
            "VERDICT": "CALIBRATION_FAILED",
            "meaning": (
                "Even with resolvability explicitly supervised, no kappa satisfies both "
                "frozen constraints. This is a STRONGER negative than probe 1: the model "
                "was trained for exactly this task and still cannot support abstention at "
                "rho=0.50 / o_max=0.10. Evidence that resolvability is not sufficiently "
                "predictable from the state under this target."),
            "prohibited_repairs": [
                "relaxing rho or o_max", "re-fitting after seeing this result",
                "class weighting", "selecting kappa off CALIB", "touching EVAL"],
        })
        OUT.write_text(json.dumps(base, indent=2), encoding="utf-8")
        print("\n  VERDICT: CALIBRATION_FAILED -- no kappa satisfies both constraints")
        print(f"  -> {OUT}")
        return 1

    best_acc = max(r["accuracy_on_committed_resolvable"] for r in qualifying)
    tied_best = [r for r in qualifying
                 if abs(r["accuracy_on_committed_resolvable"] - best_acc) < 1e-12]
    chosen = max(tied_best, key=lambda r: r["kappa"])

    base.update({
        "VERDICT": "KAPPA_SELECTED", "kappa": chosen["kappa"],
        "chosen_operating_point": chosen,
        "n_qualifying_kappa": len(qualifying),
        "tie_break_applied": len(tied_best) > 1,
        "tie_break_rule": "higher, more conservative kappa",
        "calib_accuracy_is_not_the_gate": (
            f"Accuracy here is a CALIB selection criterion. Gate 1 (tau={tau}) is scored "
            "on EVAL, which remains untouched."),
    })
    OUT.write_text(json.dumps(base, indent=2), encoding="utf-8")
    THRESHOLDS.write_text(json.dumps({
        "record": "Frozen abstention thresholds",
        "status": "FROZEN", "utc": _now(), "calibrated_on": "CALIB",
        "thresholds": {"tau": tau, "rho": rho, "o_max": o_max, "kappa": chosen["kappa"]},
        "kappa_thresholds": "P(preference is resolvable | s), from the probe-2 resolvability head",
        "provenance": {
            "tau": "AMENDMENT_4, frozen before any probe existed; EVAL gate",
            "rho": "AMENDMENT_2", "o_max": "AMENDMENT_2",
            "kappa": "selected on CALIB by the AMENDMENT_3 procedure, frozen before either probe was fitted"},
        "EVAL_touched": False,
    }, indent=2), encoding="utf-8")

    print(f"\n  VERDICT: KAPPA_SELECTED   kappa = {chosen['kappa']:.3f}")
    print(f"    min cell coverage {chosen['min_cell_coverage']:.3f}  (rho {rho})")
    print(f"    over-commitment   {chosen['over_commitment']:.3f}  (o_max {o_max})")
    print(f"    CALIB accuracy on committed resolvable {chosen['accuracy_on_committed_resolvable']:.3f}")
    print(f"  -> {OUT}\n  -> {THRESHOLDS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
