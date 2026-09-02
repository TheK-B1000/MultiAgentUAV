"""Figure 2 data extractor: frozen eval artifacts -> paper/data/fig2_crossover.csv.

Reads delta_A/delta_B (mean, lcb95, ucb95) directly out of each method's frozen
*_EVAL_RESULT.json / summary.json -- never hand-types a number. Source commit provenance is
derived at run time via `git log -1 -- <path>`, not hardcoded, so it stays correct even if a
file is ever touched again (it should not be; every source here is FROZEN_RESULT/FROZEN).

Public-facing names replace internal repo codenames (V1, OG-PSP, V3, V4, CCP-S2) per the PI's
naming decision: a reviewer should be able to follow the method progression without decoding
lab call signs. The internal name is kept in the manifest for provenance, never in the figure.

CCP-S2 itself has no eval yet (still in the causal-collection stage) -- deliberately left out
of the CSV rather than stubbed with a placeholder row; extract_final_eval.py (not yet built)
will append it once CCP-S2's own EVAL_RESULT exists.

Run:  python paper/data/extract_fig2_data.py
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "fig2_crossover.csv"
MANIFEST = Path(__file__).resolve().parent / "fig2_crossover_manifest.json"

# (public_name, internal_name, file path relative to ROOT, verdict_key)
# Order is the paper narrative order: reference -> compression family (ascending
# sophistication) -> incumbent baseline. CCP-S2 (Causal Commitment Distillation) is appended
# separately once it has a real eval result.
METHODS = [
    ("Specialists",        "SAPPO",
     "artifacts/strategic_demand/sappo_crossover/summary.json", "verdict"),
    ("One-Sided Latent",   "V1 (Oracle-Gated K=2)",
     "artifacts/strategic_demand/sppo/ORACLE_GATED_K2_EVAL_RESULT.json", "VERDICT"),
    ("Paired Latent",      "OG-PSP",
     "artifacts/strategic_demand/sppo/OG_PSP_EVAL_RESULT.json", "VERDICT"),
    ("Trajectory-Guided",  "H-OG-PSP V3",
     "artifacts/strategic_demand/sppo/HOG_PSP_V3_EVAL_RESULT.json", "VERDICT"),
    ("Private-Critic",     "H-OG-PSP V4",
     "artifacts/strategic_demand/sppo/HOG_PSP_V4_EVAL_RESULT.json", "VERDICT"),
    ("Final Latent Baseline", "CCP successor (CCP-S2's warm-start incumbent)",
     "artifacts/strategic_demand/sppo/CCP_SUCCESSOR_EVAL_RESULT.json", "VERDICT"),
]


def _commit_for(rel_path: str) -> str:
    out = subprocess.run(["git", "log", "-1", "--format=%H", "--", rel_path],
                         cwd=ROOT, capture_output=True, text=True, check=True)
    sha = out.stdout.strip()
    if not sha:
        raise SystemExit(f"REFUSING: no git history found for {rel_path}")
    return sha


def main() -> int:
    rows, manifest_entries = [], []
    for public_name, internal_name, rel_path, verdict_key in METHODS:
        p = ROOT / rel_path
        if not p.is_file():
            raise SystemExit(f"REFUSING: source file missing: {rel_path}")
        d = json.loads(p.read_text(encoding="utf-8"))
        status = d.get("status") or d.get("record") or "UNKNOWN"
        verdict = d.get(verdict_key, "MISSING_VERDICT_FIELD")
        commit = _commit_for(rel_path)
        for pole, key in (("A", "delta_A"), ("B", "delta_B")):
            if key not in d:
                raise SystemExit(f"REFUSING: {rel_path} has no {key!r} field")
            dv = d[key]
            for req in ("mean", "lcb95", "ucb95"):
                if req not in dv:
                    raise SystemExit(f"REFUSING: {rel_path}::{key} missing {req!r}")
            rows.append({
                "method": public_name, "pole": pole,
                "delta": dv["mean"], "ci_low": dv["lcb95"], "ci_high": dv["ucb95"],
                "passes": dv.get("passes"),
                "verdict": verdict, "internal_name": internal_name,
                "source_path": rel_path, "source_commit": commit,
            })
        manifest_entries.append({
            "public_name": public_name, "internal_name": internal_name,
            "source_path": rel_path, "source_commit": commit,
            "status": status, "verdict": verdict,
        })
        print(f"  {public_name:22s} ({internal_name:38s}) <- {rel_path}  @ {commit[:8]}  "
              f"{verdict}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["method", "pole", "delta", "ci_low", "ci_high",
                                           "passes", "verdict", "internal_name",
                                           "source_path", "source_commit"])
        w.writeheader()
        w.writerows(rows)
    MANIFEST.write_text(json.dumps({
        "record": "Figure 2 data provenance manifest", "generator": __file__,
        "note": "CCP-S2 (Causal Commitment Distillation) not yet included -- no eval result "
                "exists yet. extract_final_eval.py appends it once CCP-S2's own "
                "EVAL_RESULT.json is frozen.",
        "methods": manifest_entries,
    }, indent=2), encoding="utf-8")
    print(f"\n  -> {OUT}")
    print(f"  -> {MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
