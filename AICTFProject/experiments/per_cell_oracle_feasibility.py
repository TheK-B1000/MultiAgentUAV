"""Per-cell ORACLE threshold feasibility -- diagnostic only, CALIB only.

Answers one question before any per-cell machinery is built:

    If every cell were allowed its own kappa_c, chosen with full knowledge of the
    CALIB labels, is there ANY combination that satisfies the frozen gates?

    o*_min = minimum achievable GLOBAL over-commitment, subject to
             coverage_c >= rho in every cell that has resolvable mass.

This is an ORACLE and therefore an UPPER BOUND on what per-cell calibration could
ever deliver. Thresholds are chosen using the very labels they are scored against,
so any honest procedure does strictly worse. That is the point: if the oracle cannot
clear o_max, no real per-cell scheme can, and the problem is representation rather
than calibration.

The optimisation decomposes exactly. Cells partition the states, and global
over-commitment is (total tied commits) / (total tied states) with a FIXED
denominator, so minimising the sum means minimising each cell's tied-commit count
independently under its own coverage constraint. No search over combinations is
needed.

No retraining. No new model. No EVAL. Existing probe-2 predictions only.

Run:  python experiments/per_cell_oracle_feasibility.py --device cuda
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
from experiments.select_kappa import _frozen_constraints          # noqa: E402

SD = ROOT / "artifacts" / "strategic_demand"
OUT = SD / "sppo" / "PER_CELL_ORACLE_FEASIBILITY.json"


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
        raise SystemExit(f"REFUSING: {OUT} exists; this diagnostic is one-shot")
    rho, o_max, tau = _frozen_constraints()

    qpsi, dir_head, _, _ = build(device)
    dir_head.load_state_dict(torch.load(OUT_DIR / "k2_probe_head.pt", map_location=device,
                                        weights_only=False)["state_dict"])
    res_head = nn.Sequential(nn.Linear(256, HEAD_HIDDEN), nn.ReLU(),
                             nn.Linear(HEAD_HIDDEN, 1)).to(device)
    res_head.load_state_dict(torch.load(OUT_DIR / "resolvability_head.pt", map_location=device,
                                        weights_only=False)["state_dict"])
    dir_head.eval(); res_head.eval()

    calib = load_split(CALIB_LO, CALIB_HI, resolvable_only=False)
    with torch.no_grad():
        h = features(qpsi, calib, device)
        p_res = torch.sigmoid(res_head(h).squeeze(-1)).cpu().numpy()
        pred = torch.softmax(dir_head(h), dim=1).argmax(1).cpu().numpy()

    resolvable = calib["delta"] != 0
    correct = pred == calib["label"]
    cells = calib["cell"]
    n_tied_total = int((~resolvable).sum())

    print(f"PER-CELL ORACLE FEASIBILITY  {_now()}")
    print(f"  CALIB {len(p_res)} states, {int(resolvable.sum())} resolvable, {n_tied_total} tied")
    print(f"  frozen: rho={rho}  o_max={o_max}   global AUC {_auc(p_res, resolvable.astype(int)):.4f}")
    print("  ORACLE: kappa_c chosen with knowledge of CALIB labels -> UPPER BOUND only\n")

    per_cell, tied_commit_total, infeasible = {}, 0, []
    print(f"  {'cell':18s} {'res':>4s} {'tied':>5s} {'AUC':>6s} {'kappa*':>7s} {'cov':>6s} {'tied cmt':>9s}")
    for cell in sorted(set(cells)):
        m = cells == cell
        s, r = p_res[m], resolvable[m]
        n_res, n_tie = int(r.sum()), int((~r).sum())
        cell_auc = _auc(s, r.astype(int)) if n_res and n_tie else float("nan")

        # candidate thresholds: every distinct score, plus one above the max
        cands = np.unique(np.concatenate([s, [s.max() + 1e-6]]))
        best = None
        for k in cands:
            commit = s >= k
            cov = float(commit[r].mean()) if n_res else None
            if n_res and cov < rho:
                continue                          # violates this cell's coverage floor
            tied_commits = int(commit[~r].sum())
            if best is None or tied_commits < best[1]:
                best = (float(k), tied_commits, cov)
        if best is None:
            infeasible.append(cell)               # cannot reach rho at any threshold
            best = (float(cands[0]), int((s >= cands[0])[~r].sum()),
                    float((s >= cands[0])[r].mean()) if n_res else None)
        kappa_c, tied_commits, cov = best
        tied_commit_total += tied_commits
        per_cell[cell] = {
            "n_resolvable": n_res, "n_tied": n_tie,
            "auc": None if np.isnan(cell_auc) else round(cell_auc, 4),
            "oracle_kappa": round(kappa_c, 4),
            "coverage_at_oracle": None if cov is None else round(cov, 4),
            "tied_commits_at_oracle": tied_commits,
            "exempt_no_resolvable_mass": n_res == 0,
            "infeasible_at_any_threshold": cell in infeasible,
        }
        cov_s = "exempt" if n_res == 0 else f"{cov:.3f}"
        auc_s = "  n/a" if np.isnan(cell_auc) else f"{cell_auc:.3f}"
        print(f"  {cell:18s} {n_res:4d} {n_tie:5d} {auc_s:>6s} {kappa_c:7.3f} {cov_s:>6s} {tied_commits:9d}")

    o_star = tied_commit_total / n_tied_total if n_tied_total else float("nan")
    feasible = (not infeasible) and o_star <= o_max

    print(f"\n  o*_min (oracle minimum global over-commitment) = {o_star:.4f}")
    print(f"  frozen ceiling o_max = {o_max}")
    if infeasible:
        print(f"  cells that cannot reach rho at ANY threshold: {infeasible}")

    verdict = ("PER_CELL_CALIBRATION_COULD_SUFFICE" if feasible
               else "PER_CELL_CALIBRATION_CANNOT_SUFFICE")
    if feasible:
        reading = (
            f"o*_min = {o_star:.4f} clears the {o_max} ceiling. The resolvability signal "
            "is useful but its calibration is regime-dependent, so a cell-conditioned "
            "scheme is scientifically justified. NOTE this is an ORACLE bound -- an "
            "honest procedure that does not see the labels will do worse, and some cells "
            "have very little CALIB support, so a low-dimensional cell-conditioned "
            "calibration is preferable to 16 free thresholds.")
    else:
        reading = (
            f"o*_min = {o_star:.4f} exceeds the {o_max} ceiling even with oracle "
            "knowledge of the labels. Per-cell thresholds CANNOT fix this. The problem "
            "is not that cells sit on different score scales -- the resolvability "
            "representation does not separate ties sharply enough within cells. The next "
            "problem is REPRESENTATION, not calibration, and adding threshold machinery "
            "would be wasted effort.")
    print(f"\n  VERDICT: {verdict}\n  {reading}")

    OUT.write_text(json.dumps({
        "record": "Per-cell oracle threshold feasibility (diagnostic)",
        "status": "DIAGNOSTIC -- CALIB only, no retraining, no EVAL, no new model",
        "utc": _now(),
        "question": ("If every cell had its own kappa_c chosen with full knowledge of the "
                     "CALIB labels, could the frozen gates be satisfied?"),
        "ORACLE_CAVEAT": (
            "Thresholds are chosen using the labels they are scored against. This is an "
            "UPPER BOUND on achievable per-cell performance; any honest procedure does "
            "strictly worse."),
        "decomposition_note": (
            "Exact, not approximate: cells partition the states and the over-commitment "
            "denominator is fixed, so minimising total tied commits means minimising each "
            "cell independently under its own coverage floor."),
        "frozen_constraints": {"rho": rho, "o_max": o_max, "tau": tau},
        "global_auc": round(_auc(p_res, resolvable.astype(int)), 4),
        "o_star_min": round(o_star, 4),
        "clears_o_max": bool(o_star <= o_max),
        "cells_infeasible_at_any_threshold": infeasible,
        "per_cell": per_cell,
        "VERDICT": verdict,
        "reading": reading,
        "does_not_alter": "the frozen CALIBRATION_FAILED verdicts of probe 1 or probe 2",
        "EVAL_touched": False,
    }, indent=2), encoding="utf-8")
    print(f"  -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
