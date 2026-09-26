"""Row-level integrity audit of the 4v4 exploratory sharing crossovers that wrote
*_INTEGRITY_REQUIRED.json (Delta_B <= 0).

Read-only. For each arm it checks that the flag is not produced by a mechanical fault:

  1. rows: 256 = 4 cells x 64 seeds, seed set == the frozen spec block, no duplicates
  2. row consistency: win == (blue > red) and margin == blue - red on every row
  3. checkpoint on disk still has the sha the spec pinned
  4. forced z reaches the policy: on the same seed and pole, z0 and z1 must NOT
     produce identical episodes (a policy that ignored z would make them identical)
  5. z-label mapping: holdout agreement z0<->pi_A and z1<->pi_B from STUDENT_FROZEN
     (a swapped mapping would show low agreement, not ~0.9)
  6. Delta_A / Delta_B re-derived with the evaluator's own _mean_ci
  7. context: the teacher pair's own sealed 4v4 crossover, for comparison

It does NOT certify the design. Writes SUITE_4V4_SHARING_FLAGGED_ARMS_ROW_AUDIT.json.

Run: python experiments/audit_suite_sharing_4v4_flagged_arms.py [--arms share_encoder fully_shared_z ...]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.eval_hog_psp_v3 import _mean_ci  # noqa: E402

SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
SPEC = SD / "SUITE_SHARING_4V4_CROSSOVER_EVAL_SPEC.json"
TEACHER_READING = SD / "4V4_ENTITY_REPAIR_CORRECTED_CROSSOVER_READING.json"
OUT = SD / "SUITE_4V4_SHARING_FLAGGED_ARMS_ROW_AUDIT.json"

# spec arm key -> (rows csv, flag json, suite_sharing dir)
ARMS = {
    "fully_shared_z": ("suite_fully_shared_z_4v4_exploratory_crossover_eval_rows.csv",
                       "SUITE_FULLY_SHARED_Z_4V4_EXPLORATORY_CROSSOVER_EVAL_INTEGRITY_REQUIRED.json",
                       "fully_shared_z"),
    "share_encoder": ("suite_share_encoder_4v4_exploratory_crossover_eval_rows.csv",
                      "SUITE_SHARE_ENCODER_4V4_EXPLORATORY_CROSSOVER_EVAL_INTEGRITY_REQUIRED.json",
                      "share_encoder"),
    "share_backbone": ("suite_share_backbone_4v4_exploratory_crossover_eval_rows.csv",
                       "SUITE_SHARE_BACKBONE_4V4_EXPLORATORY_CROSSOVER_EVAL_INTEGRITY_REQUIRED.json",
                       "share_backbone"),
    "share_macro": ("suite_share_macro_4v4_exploratory_crossover_eval_rows.csv",
                    "SUITE_SHARE_MACRO_4V4_EXPLORATORY_CROSSOVER_EVAL_INTEGRITY_REQUIRED.json",
                    "share_macro"),
}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def audit_arm(key: str, spec: dict) -> dict:
    rows_name, flag_name, dirname = ARMS[key]
    rows_p, flag_p = SD / rows_name, SD / flag_name
    frozen_p = SD / "suite_sharing" / "4v4" / dirname / "STUDENT_FROZEN.json"
    checks: list[dict] = []

    def add(name: str, ok: bool, detail: str, **data) -> None:
        checks.append({"name": name, "result": "PASS" if ok else "FAIL", "detail": detail, **data})

    for p in (rows_p, frozen_p):
        if not p.is_file():
            raise SystemExit(f"FAIL-CLOSED: required evidence missing for {key}: {p}")
    rows = list(csv.DictReader(rows_p.open(encoding="utf-8")))
    for r in rows:
        for f in ("z", "seed", "blue", "red", "win", "margin"):
            r[f] = int(r[f])

    lo, hi = (int(x) for x in str(spec["SEEDS"][f"{key}_exploratory"]).split(".."))
    want = set(range(lo, lo + int(spec["SEEDS"]["n_exploratory"])))
    cells: dict = {}
    for r in rows:
        cells.setdefault((r["z"], r["pole"]), []).append(r["seed"])
    add("row_count", len(rows) == 256, f"{len(rows)} rows, expected 256")
    add("cell_seed_block_exact", len(cells) == 4 and all(set(v) == want and len(v) == len(set(v)) for v in cells.values()),
        f"{len(cells)} cells, each the frozen block {min(want)}..{max(want)} exactly once")
    bad = [r for r in rows if r["win"] != int(r["blue"] > r["red"]) or r["margin"] != r["blue"] - r["red"]]
    add("win_margin_consistent_with_scores", not bad, f"{len(bad)} inconsistent rows of {len(rows)}")

    frozen = json.loads(frozen_p.read_text(encoding="utf-8"))
    ck_spec = spec["ARMS"][key]
    ck_p = ROOT / ck_spec["checkpoint"]
    add("checkpoint_sha_matches_spec_pin", ck_p.is_file() and _sha(ck_p) == ck_spec["sha256"],
        f"on-disk sha {(_sha(ck_p)[:12] if ck_p.is_file() else 'MISSING')} vs pin {ck_spec['sha256'][:12]}")
    add("student_frozen_sha_matches_pin", frozen.get("sha256") == ck_spec["sha256"],
        f"STUDENT_FROZEN sha {str(frozen.get('sha256'))[:12]} vs pin {ck_spec['sha256'][:12]}")

    by = {(r["z"], r["pole"], r["seed"]): r for r in rows}
    ident = {}
    for pole in ("A", "B"):
        same = sum(1 for s in want if (by[(0, pole, s)]["blue"], by[(0, pole, s)]["red"]) ==
                   (by[(1, pole, s)]["blue"], by[(1, pole, s)]["red"]))
        ident[pole] = same / len(want)
    add("forced_z_changes_episodes", ident["A"] < 0.95 and ident["B"] < 0.95,
        f"z0/z1 give identical final scores on the same seed: Pole A {ident['A']:.3f}, Pole B {ident['B']:.3f} "
        f"(a policy ignoring z would give 1.000)", identical_score_fraction=ident)

    h = frozen["final_holdout"]
    add("z_label_mapping_consistent", h["holdout_agree_z0_vs_piA"] > 0.8 and h["holdout_agree_z1_vs_piB"] > 0.8,
        f"holdout agreement z0<->pi_A {h['holdout_agree_z0_vs_piA']:.3f}, z1<->pi_B {h['holdout_agree_z1_vs_piB']:.3f}",
        agree_z0_piA=h["holdout_agree_z0_vs_piA"], agree_z1_piB=h["holdout_agree_z1_vs_piB"])

    def wins(z, pole):
        return np.array([by[(z, pole, s)]["win"] for s in sorted(want)], dtype=np.float64)

    dA, dB = _mean_ci(wins(0, "A") - wins(1, "A")), _mean_ci(wins(1, "B") - wins(0, "B"))
    flag = json.loads(flag_p.read_text(encoding="utf-8")) if flag_p.is_file() else None
    if flag is not None:
        pe = flag["point_estimates"]
        add("rederived_deltas_match_flag_record",
            abs(pe["delta_A"] - dA["mean"]) < 1e-9 and abs(pe["delta_B"] - dB["mean"]) < 1e-9,
            f"flag delta_A {pe['delta_A']:+.5f} / delta_B {pe['delta_B']:+.5f} vs re-derived "
            f"{dA['mean']:+.5f} / {dB['mean']:+.5f}")
    cell_rates = {f"z{z}@Pole{p}": float(wins(z, p).mean()) for z in (0, 1) for p in ("A", "B")}
    gating = [c for c in checks]
    return {
        "arm": key, "n_per_cell": len(want), "rows_csv": rows_name,
        "all_mechanical_checks_pass": all(c["result"] == "PASS" for c in gating),
        "checks": checks, "cell_win_rates": cell_rates,
        "delta_A": dA, "delta_B": dB,
        "student_fit": {"agree_z0_piA": h["holdout_agree_z0_vs_piA"], "agree_z1_piB": h["holdout_agree_z1_vs_piB"]},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["fully_shared_z", "share_encoder", "share_backbone"],
                    choices=list(ARMS))
    args = ap.parse_args()
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    if not str(spec.get("status", "")).startswith("FROZEN"):
        raise SystemExit("REFUSING: eval spec not frozen")
    tr = json.loads(TEACHER_READING.read_text(encoding="utf-8"))
    teacher = {"source": TEACHER_READING.name, "status": tr["status"],
               "delta_A": tr["PRIMARY_GATE_OUTCOME"]["delta_A"], "delta_B": tr["PRIMARY_GATE_OUTCOME"]["delta_B"],
               "raw_cells": tr["raw_cells"],
               "note": "n=128 seeds 18300001-128; teachers are the KL teachers pinned in SUITE_DISTILLATION_4V4_SPEC "
                       "(pi_A3 = z0, corrected pi_B3 = z1). Different seeds, n and possibly pole overlay than the "
                       "student evals -- context, not a matched comparison."}
    results = [audit_arm(k, spec) for k in args.arms]

    print(f"ROW-LEVEL AUDIT  {_now()}")
    for r in results:
        print(f"\n  {r['arm']}: mechanical checks {'ALL PASS' if r['all_mechanical_checks_pass'] else 'FAILURES'}")
        for c in r["checks"]:
            print(f"    [{c['result']}] {c['name']}: {c['detail']}")
        cr = r["cell_win_rates"]
        print(f"    cells  z0@A {cr['z0@PoleA']:.3f}  z1@A {cr['z1@PoleA']:.3f}  z0@B {cr['z0@PoleB']:.3f}  z1@B {cr['z1@PoleB']:.3f}")
        print(f"    delta_A {r['delta_A']['mean']:+.4f} [{r['delta_A']['lcb95']:+.4f}, {r['delta_A']['ucb95']:+.4f}]   "
              f"delta_B {r['delta_B']['mean']:+.4f} [{r['delta_B']['lcb95']:+.4f}, {r['delta_B']['ucb95']:+.4f}]")
    print(f"\n  teacher pair (context): delta_A {teacher['delta_A']['mean']:+.4f}  delta_B {teacher['delta_B']['mean']:+.4f} "
          f"[{teacher['delta_B']['lcb95']:+.4f}, {teacher['delta_B']['ucb95']:+.4f}]  cells {teacher['raw_cells']}")

    OUT.write_text(json.dumps({
        "record": "SUITE 4V4 sharing flagged arms -- row-level integrity audit", "utc": _now(),
        "scope": "read-only mechanical audit of exploratory n=64 crossovers that wrote INTEGRITY_REQUIRED",
        "what_a_pass_means": "rows are well-formed, seeds/checkpoints match the frozen spec, z forcing changes episodes, "
                             "and the z-label mapping is consistent with the imitation fit. It does not certify the "
                             "design and it does not seal any result.",
        "arms": results, "teacher_pair_context": teacher,
    }, indent=2), encoding="utf-8")
    print(f"\n  -> {OUT}")
    return 0 if all(r["all_mechanical_checks_pass"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
