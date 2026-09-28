r"""Collect the evaluation-only baseline rows (specialists, Separated) across scales.

    python experiments/collect_baseline_results.py [--write]

Reporting rule (PI, 2026-09-27): the headline is the MEAN crossover effect, delta_A / delta_B, with
one yardstick at every scale; 95% intervals go to figures and records, and pass/fail wording is not
the narrative. Sealed gate verdicts are carried as provenance only.

Every filled cell is RE-DERIVED from the rows CSV on disk with the evaluator's own bootstrap
(eval_hog_psp_v3._mean_ci: paired percentile, n_boot=20000, alpha=0.05, rng_seed=7) and must equal
the governing record's recorded means; its pole identity must be attested. A row whose source is
absent is reported PENDING with the reason -- never filled from memory or a default.

    delta_A = V(pi_A, Pole A) - V(pi_B, Pole A);   delta_B = V(pi_B, Pole B) - V(pi_A, Pole B)
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

from experiments.eval_hog_psp_v3 import _mean_ci  # noqa: E402  (the evaluator's own bootstrap)

SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
OUT_JSON = ROOT / "paper" / "data" / "baseline_results_mean_effects.json"
OUT_MD = ROOT / "paper" / "data" / "BASELINE_RESULTS_MEAN_EFFECTS.md"

#: The standardized suite's evaluation-only baseline rows. Each filled source names its rows CSV,
#: its governing record, and how the record states pole identity.
ROWS = [
    {"scale": "2v2", "row": "Specialists (repaired pi_A vs repaired pi_B, no roles)",
     "label": "STANDARDIZED_2V2_DIAG_PRESPLIT", "kind": "sealed_result",
     "pending_reason": "the PRESPLIT arm of STANDARDIZED_2V2_POLE_A_LOCALIZATION_DIAG (running; ETA 2026-09-27 ~23:30)"},
    {"scale": "2v2", "row": "Separated (frozen pi_A ATTACK + pi_D DEFEND, CLOSEST_DEFENDS k=1)",
     "label": "STANDARDIZED_2V2_SPLIT_K1_CONFIRMATORY", "kind": "sealed_result",
     "note": "fresh-seed confirmatory values are the 2v2 numbers; the earlier n=64 exploratory draw (+0.281/+0.500) is not reported as the 2v2 value"},
    {"scale": "4v4", "row": "Specialists (repaired pi_A3 vs repaired pi_B3-corrected, no roles)",
     "rows_csv": "confirmatory_b3_entity_repair_corrected_specialist_crossover_eval_rows.csv",
     "record": "4V4_ENTITY_REPAIR_CORRECTED_CROSSOVER_READING.json", "kind": "audited_reading",
     "note": "no RESULT json (the eval wrote INTEGRITY_REQUIRED on the delta_B reversal); the reading rests on the row-level TIE_REVERSAL audit and pi_B3's live pole attestation against certified B3-3"},
    {"scale": "4v4", "row": "Separated (frozen pi_A ATTACK + pi_D DEFEND, CLOSEST_DEFENDS k=2)",
     "label": "DEFEND_ATTACK_SPLIT_POLICY_A_V1_CONFIRMATORY_V1", "kind": "sealed_result"},
    {"scale": "6v6", "row": "Specialists (repaired c2 pair, no roles)", "kind": "pending",
     "pending_reason": "suite checkpoints are produced by SCHOOL_PC_6V6_LOCKED_PIPELINE.json and are not on this machine (c2 dirs hold only stale run locks); the historical 6v6 specialists are explicitly not suite teachers"},
    {"scale": "6v6", "row": "Separated (CLOSEST_DEFENDS k=1)", "kind": "pending",
     "pending_reason": "the 6v6 split pi_D (split_defend_k1_v1) is produced on the school PC; not on this machine"},
    {"scale": "all", "row": "Generalist", "kind": "pending",
     "pending_reason": "no standardized generalist checkpoint exists at any scale; needs a design decision"},
    {"scale": "all", "row": "Distilled / sharing arms", "kind": "pending",
     "pending_reason": "need the standardized CLOSEST_DEFENDS datasets per scale (4v4 must be recollected on certified B3-3; the old 4v4 student rows are invalidated)"},
]


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _derive(rows_csv: Path) -> dict:
    by: dict = {}
    for r in csv.DictReader(rows_csv.open(encoding="utf-8")):
        by.setdefault((r["policy"], r["pole"]), {})[int(r["seed"])] = float(r["win"])
    seeds = sorted(by[("pi_A", "A")])
    for k in (("pi_A", "B"), ("pi_B", "A"), ("pi_B", "B")):
        if sorted(by[k]) != seeds:
            raise SystemExit(f"FAIL-CLOSED: {rows_csv.name}: cell {k} seed set differs from pi_A@A")
    v = {k: np.array([by[k][s] for s in seeds], dtype=np.float64) for k in by}
    return {
        "n_seeds": len(seeds), "seed_block": [seeds[0], seeds[-1]],
        "win_rates": {f"{p}@{q}": float(v[(p, q)].mean()) for p in ("pi_A", "pi_B") for q in ("A", "B")},
        "delta_A": _mean_ci(v[("pi_A", "A")] - v[("pi_B", "A")]),
        "delta_B": _mean_ci(v[("pi_B", "B")] - v[("pi_A", "B")]),
    }


def collect() -> dict:
    out_rows = []
    for spec in ROWS:
        row = {k: spec[k] for k in ("scale", "row")}
        if spec["kind"] == "pending":
            row.update(status="PENDING", reason=spec["pending_reason"])
            out_rows.append(row)
            continue
        if spec["kind"] == "sealed_result":
            rec_p = SD / f"{spec['label']}_SPECIALIST_CROSSOVER_EVAL_RESULT.json"
            csv_p = SD / f"{spec['label'].lower()}_specialist_crossover_eval_rows.csv"
        else:
            rec_p, csv_p = SD / spec["record"], SD / spec["rows_csv"]
        if not rec_p.is_file() or not csv_p.is_file():
            row.update(status="PENDING", reason=spec.get("pending_reason", f"missing {rec_p.name} or {csv_p.name}"))
            out_rows.append(row)
            continue
        rec = json.loads(rec_p.read_text(encoding="utf-8"))
        d = _derive(csv_p)
        if spec["kind"] == "sealed_result":
            if rec.get("status") not in ("SEALED", "FROZEN_RESULT"):
                raise SystemExit(f"FAIL-CLOSED: {rec_p.name} status {rec.get('status')!r}")
            att = rec.get("pole_attestations") or {}
            if not att or not all(att[p].get("hashes_match") is True for p in ("A", "B")):
                raise SystemExit(f"FAIL-CLOSED: {rec_p.name} lacks a matching pole attestation for both poles")
            rg = rec["PRIMARY_GATE"]
            recorded = {"delta_A": rg["delta_A"], "delta_B": rg["delta_B"]}
            provenance = {"record": rec_p.name, "record_status": rec["status"], "arm": rec.get("arm"),
                          "seed_class": (rec.get("seeds") or {}).get("seed_class")
                          or ("sealed_confirmatory" if rec.get("confirmatory") else "exploratory"),
                          "checkpoints_sha256": rec.get("checkpoints"),
                          "split_frozen_attack_sha256": (rec.get("split_policy_pi_A") or {}).get("frozen_attack_sha256"),
                          "pole_config_hash": {p: att[p]["live_config_hash"] for p in ("A", "B")},
                          "frozen_gate_verdict_provenance_only": "PASS" if rg.get("passes") else "FAIL"}
        else:
            recorded = {"delta_A": rec["PRIMARY_GATE_OUTCOME"]["delta_A"], "delta_B": rec["PRIMARY_GATE_OUTCOME"]["delta_B"]}
            provenance = {"record": rec_p.name, "record_status": rec["status"], "rests_on": rec.get("rests_on"),
                          "frozen_gate_verdict_provenance_only": rec["PRIMARY_GATE_OUTCOME"].get("joint_gate")}
        for k in ("delta_A", "delta_B"):
            if abs(d[k]["mean"] - float(recorded[k]["mean"])) > 1e-3:
                raise SystemExit(f"FAIL-CLOSED: {rec_p.name} {k}: rows give {d[k]['mean']:+.4f}, "
                                 f"record says {float(recorded[k]['mean']):+.4f}")
        provenance["rows_csv"] = csv_p.name
        provenance["rows_csv_sha256"] = _sha(csv_p)
        row.update(status="FILLED", **d, provenance=provenance)
        if spec.get("note"):
            row["note"] = spec["note"]
        out_rows.append(row)
    return {
        "record_id": "BASELINE_RESULTS_MEAN_EFFECTS",
        "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "reporting_rule": "headline = mean delta_A / delta_B at every scale (PI 2026-09-27); 95% intervals for figures; frozen gate verdicts carried as provenance only",
        "delta_convention": "delta_A = V(pi_A,A) - V(pi_B,A); delta_B = V(pi_B,B) - V(pi_A,B)",
        "bootstrap": "eval_hog_psp_v3._mean_ci -- paired percentile over seeds, n_boot=20000, alpha=0.05, rng_seed=7",
        "rows": out_rows,
    }


def _md(doc: dict) -> str:
    L = ["# Baseline results: mean specialization effect (Δ_A / Δ_B)", "",
         "Generated by `experiments/collect_baseline_results.py` from the rows on disk (every cell re-derived and "
         "checked against its governing record). Headline = mean effect; 95% intervals are for figures.", "",
         "Δ_A = V(π_A, A) − V(π_B, A); Δ_B = V(π_B, B) − V(π_A, B).", "",
         "| Scale | Baseline | Mean Δ_A / Δ_B | 95% CI (figures) | Win rates π_A@A, π_A@B, π_B@A, π_B@B | n | Source |",
         "|---|---|---|---|---|---:|---|"]
    for r in doc["rows"]:
        if r["status"] != "FILLED":
            L.append(f"| {r['scale']} | {r['row']} | PENDING | — | — | — | {r['reason']} |")
            continue
        a, b, w = r["delta_A"], r["delta_B"], r["win_rates"]
        L.append(f"| {r['scale']} | {r['row']} | **{a['mean']:+.3f} / {b['mean']:+.3f}** | "
                 f"[{a['lcb95']:+.3f}, {a['ucb95']:+.3f}] / [{b['lcb95']:+.3f}, {b['ucb95']:+.3f}] | "
                 f"{w['pi_A@A']:.3f}, {w['pi_A@B']:.3f}, {w['pi_B@A']:.3f}, {w['pi_B@B']:.3f} | {r['n_seeds']} | "
                 f"`{r['provenance']['record']}` ({r['provenance'].get('seed_class') or r['provenance']['record_status']}) |")
    notes = [f"- **{r['scale']} {r['row'].split(' (')[0]}**: {r['note']}" for r in doc["rows"] if r.get("note")]
    if notes:
        L += ["", "Notes:"] + notes
    L += ["", "Frozen gate verdicts (lower bound above zero on both Δ) are kept in the records as provenance and are "
          "not the headline: " + "; ".join(f"{r['scale']} {r['row'].split(' (')[0]} = {r['provenance']['frozen_gate_verdict_provenance_only']}"
                                            for r in doc["rows"] if r["status"] == "FILLED") + "."]
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # the table uses Greek letters
    doc = collect()
    md = _md(doc)
    print(md)
    if a.write:
        OUT_JSON.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
        OUT_MD.write_text(md, encoding="utf-8")
        print(f"-> {OUT_JSON}\n-> {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
