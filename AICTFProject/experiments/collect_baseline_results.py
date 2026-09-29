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
OUT_TEX_DIR = ROOT / "paper" / "aamas2027" / "generated"

#: The standardized suite's evaluation-only baseline rows. Each filled source names its rows CSV,
#: its governing record, and how the record states pole identity.
#:
#: table_role (PI, 2026-09-28) -- consumers (LaTeX tables) select rows by role, never by position:
#:   "main"                      the scale's reported value for that baseline (fresh confirmatory where one exists)
#:   "role_allocation_paired"    the Separated arm evaluated on EXACTLY the seeds of its no-role partner
#:                               (paired_with), for the role-allocation comparison only; not the main value
#: A "main" row may also be the no-role side of a paired comparison (it names no partner itself).
ROWS = [
    {"scale": "2v2", "row": "Specialists (repaired pi_A vs repaired pi_B, no roles)",
     "label": "STANDARDIZED_2V2_DIAG_PRESPLIT", "kind": "sealed_result", "table_role": "main",
     "note": "also the no-role side of the paired role-allocation comparison (same 128 seeds as the paired SPLIT arm)"},
    {"scale": "2v2", "row": "Separated (frozen pi_A ATTACK + pi_D DEFEND, CLOSEST_DEFENDS k=1)",
     "label": "STANDARDIZED_2V2_SPLIT_K1_CONFIRMATORY", "kind": "sealed_result", "table_role": "main",
     "note": "fresh-seed confirmatory values are the 2v2 main numbers; the earlier n=64 exploratory draw (+0.281/+0.500) is not reported as the 2v2 value"},
    {"scale": "2v2", "row": "Separated, paired with no-role Specialists (same seeds; CLOSEST_DEFENDS k=1)",
     "label": "STANDARDIZED_2V2_DIAG_SPLIT", "kind": "sealed_result", "table_role": "role_allocation_paired",
     "paired_with": "STANDARDIZED_2V2_DIAG_PRESPLIT",
     "note": "role-allocation comparison only: evaluated on exactly the no-role Specialists' seeds, so the difference between the two rows is the effect of role allocation on matched seeds; the main 2v2 Separated value is the confirmatory row"},
    {"scale": "2v2", "row": "Share-Encoder (shared CNN encoder; private per-z body and heads)",
     "label": "STANDARDIZED_2V2_SHARE_ENCODER", "kind": "sharing_result",
     "eval_spec": "STANDARDIZED_2V2_SHARING_EVAL_SPEC.json", "arm_key": "share_encoder",
     "frozen": "suite_sharing_std/2v2/share_encoder/STUDENT_FROZEN.json"},
    {"scale": "2v2", "row": "Fully Shared+z (one network, concat strategy ID)",
     "label": "STANDARDIZED_2V2_FULLY_SHARED_Z", "kind": "sharing_result",
     "eval_spec": "STANDARDIZED_2V2_SHARING_EVAL_SPEC.json", "arm_key": "fully_shared_z",
     "frozen": "suite_sharing_std/2v2/fully_shared_z/STUDENT_FROZEN.json"},
    {"scale": "2v2", "row": "Generalist (one network, no strategy ID)",
     "label": "STANDARDIZED_2V2_GENERALIST", "kind": "generalist_result",
     "eval_spec": "STANDARDIZED_2V2_SHARING_EVAL_SPEC.json", "arm_key": "generalist",
     "frozen": "suite_sharing_std/2v2/generalist/STUDENT_FROZEN.json",
     "reference": "STANDARDIZED_2V2_SEPARATED_GENERALIST_REF",
     "note": "GENERALIST_DEFINITION_V1: a single policy has no crossover Delta; reported as V(pi_G, A), V(pi_G, B) and Delta_G = V(ours) - V(pi_G), paired by seed against the Separated system re-scored on the same 128 seeds"},
    {"scale": "4v4", "row": "Specialists (repaired pi_A3 vs repaired pi_B3-corrected, no roles)",
     "rows_csv": "confirmatory_b3_entity_repair_corrected_specialist_crossover_eval_rows.csv",
     "record": "4V4_ENTITY_REPAIR_CORRECTED_CROSSOVER_READING.json", "kind": "audited_reading", "table_role": "main",
     "note": "no RESULT json (the eval wrote INTEGRITY_REQUIRED on the delta_B reversal); the reading rests on the row-level TIE_REVERSAL audit and pi_B3's live pole attestation against certified B3-3"},
    {"scale": "4v4", "row": "Separated (frozen pi_A ATTACK + pi_D DEFEND, CLOSEST_DEFENDS k=2)",
     "label": "DEFEND_ATTACK_SPLIT_POLICY_A_V1_CONFIRMATORY_V1", "kind": "sealed_result", "table_role": "main"},
    {"scale": "6v6", "row": "Specialists (repaired c2 pair, no roles)", "kind": "pending",
     "pending_reason": "suite checkpoints are produced by SCHOOL_PC_6V6_LOCKED_PIPELINE.json and are not on this machine (c2 dirs hold only stale run locks); the historical 6v6 specialists are explicitly not suite teachers"},
    {"scale": "6v6", "row": "Separated (CLOSEST_DEFENDS k=1)", "kind": "pending",
     "pending_reason": "the 6v6 split pi_D (split_defend_k1_v1) is produced on the school PC; not on this machine"},
    *({"scale": sc, "row": arm, "kind": "pending",
       "pending_reason": f"{sc} is opened only after the previous scale locks (PI operating rule 2026-09-28); "
                         f"same frozen definitions as 2v2 (GENERALIST_DEFINITION_V1 for the Generalist)"}
      for sc in ("4v4", "6v6") for arm in ("Share-Encoder", "Fully Shared+z", "Generalist")),
]
for _s in ROWS:
    _s.setdefault("table_role", "main")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _cells(rows_csv: Path, field: str) -> dict:
    """(value of ``field``, pole) -> {seed: win} from a sealed rows CSV."""
    by: dict = {}
    for r in csv.DictReader(rows_csv.open(encoding="utf-8")):
        by.setdefault((r[field], r["pole"]), {})[int(r["seed"])] = float(r["win"])
    return by


def _derive(rows_csv: Path, field: str = "policy") -> dict:
    """Crossover cells. Sharing arms record ``z`` (0 plays pi_A's role, 1 plays pi_B's) instead of
    ``policy``; the deltas are the same definition either way."""
    by = _cells(rows_csv, field)
    if field == "z":
        by = {({"0": "pi_A", "1": "pi_B"}[k[0]], k[1]): v for k, v in by.items()}
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


def _sealed(rec_p: Path) -> dict:
    rec = json.loads(rec_p.read_text(encoding="utf-8"))
    if rec.get("status") not in ("SEALED", "FROZEN_RESULT"):
        raise SystemExit(f"FAIL-CLOSED: {rec_p.name} status {rec.get('status')!r}")
    return rec


def _check_recorded(name: str, rederived: dict, recorded: dict) -> None:
    if abs(rederived["mean"] - float(recorded["mean"])) > 1e-3:
        raise SystemExit(f"FAIL-CLOSED: {name}: rows give {rederived['mean']:+.4f}, "
                         f"record says {float(recorded['mean']):+.4f}")


def _collect_suite_arm(spec: dict) -> dict | None:
    """A distilled sharing arm or the Generalist, from its sealed record under the scale's eval spec.
    None while the record does not exist yet; any disagreement with the spec fails closed."""
    label = spec["label"]
    rec_p = SD / f"{label}_CROSSOVER_EVAL_RESULT.json"
    csv_p = SD / f"{label.lower()}_crossover_eval_rows.csv"
    if not rec_p.is_file():
        return None
    rec = _sealed(rec_p)
    es = json.loads((SD / spec["eval_spec"]).read_text(encoding="utf-8"))
    arm = es["ARMS"][spec["arm_key"]]
    if rec.get("checkpoint_sha256") != arm["sha256"] or _sha(ROOT / arm["checkpoint"]) != arm["sha256"]:
        raise SystemExit(f"FAIL-CLOSED: {label} checkpoint differs from {spec['eval_spec']}")
    for p in ("A", "B"):
        if (rec.get("poles") or {}).get(p, {}).get("pole_config_hash") != es["POLES"][p]["pole_config_hash"]:
            raise SystemExit(f"FAIL-CLOSED: {label} pole {p} identity differs from {spec['eval_spec']}")
    frozen = json.loads((SD / spec["frozen"]).read_text(encoding="utf-8"))
    provenance = {"record": rec_p.name, "record_status": rec["status"], "seed_class": rec["seeds"]["seed_class"],
                  "checkpoint_sha256": arm["sha256"], "rows_csv": csv_p.name, "rows_csv_sha256": _sha(csv_p),
                  "unique_actor_params": frozen.get("unique_actor_params"),
                  "holdout_fidelity": {k: frozen["final_holdout"][k] for k in
                                       ("holdout_agree_z0_vs_piA", "holdout_agree_z1_vs_piB")},
                  "pole_config_hash": {p: rec["poles"][p]["pole_config_hash"] for p in ("A", "B")}}
    if spec["kind"] == "sharing_result":
        d = _derive(csv_p, field="z")
        for k in ("delta_A", "delta_B"):
            _check_recorded(f"{rec_p.name} {k}", d[k], rec["PRIMARY_GATE"][k])
        provenance["frozen_gate_verdict_provenance_only"] = "PASS" if rec["PRIMARY_GATE"].get("passes") else "FAIL"
        return {**d, "provenance": provenance}

    # Generalist: V(pi_G, pole) and Delta_G against the Separated reference on the same seeds
    g = _cells(csv_p, "z")
    ref_label = spec["reference"]
    ref_p = SD / f"{ref_label}_SPECIALIST_CROSSOVER_EVAL_RESULT.json"
    ref_csv = SD / f"{ref_label.lower()}_specialist_crossover_eval_rows.csv"
    if not ref_p.is_file():
        return None
    ref_rec, ref = _sealed(ref_p), _cells(ref_csv, "policy")
    seeds = sorted(g[("0", "A")])
    for cell, src in ((g[("0", "B")], "pi_G@B"), (ref[("pi_A", "A")], "ref pi_A@A"), (ref[("pi_B", "B")], "ref pi_B@B")):
        if sorted(cell) != seeds:
            raise SystemExit(f"FAIL-CLOSED: {label}: {src} seed set differs from pi_G@A -- Delta_G must be paired")
    vec = lambda c: np.array([c[s] for s in seeds], dtype=np.float64)   # noqa: E731
    v_a, v_b = _mean_ci(vec(g[("0", "A")])), _mean_ci(vec(g[("0", "B")]))
    _check_recorded(f"{rec_p.name} V_pole_A", v_a, rec["PRIMARY_GATE"]["V_pole_A"])
    _check_recorded(f"{rec_p.name} V_pole_B", v_b, rec["PRIMARY_GATE"]["V_pole_B"])
    provenance.update(reference_record=ref_p.name, reference_status=ref_rec["status"],
                      reference_rows_csv_sha256=_sha(ref_csv))
    return {"n_seeds": len(seeds), "seed_block": [seeds[0], seeds[-1]],
            "V_pole_A": v_a, "V_pole_B": v_b,
            "ours_win_rates": {"pi_A+D@A": float(vec(ref[("pi_A", "A")]).mean()),
                               "pi_B@B": float(vec(ref[("pi_B", "B")]).mean())},
            "delta_G_A": _mean_ci(vec(ref[("pi_A", "A")]) - vec(g[("0", "A")])),
            "delta_G_B": _mean_ci(vec(ref[("pi_B", "B")]) - vec(g[("0", "B")])),
            "provenance": provenance}


#: Deployment-noise suite per scale: the final system on ONE matched seed block, nominal first.
#: (label, perturbation family, macro stem)
NOISE = {
    "2v2": {"spec": "STANDARDIZED_2V2_NOISE_SPEC.json", "conditions": [
        ("STANDARDIZED_2V2_NOISE_NOMINAL", "nominal", "TwoNoiseNom"),
        ("STANDARDIZED_2V2_NOISE_LOCALIZATION_MEDIUM", "localization_noise", "TwoNoiseLoc"),
        ("STANDARDIZED_2V2_NOISE_MOTION_MEDIUM", "motion_error", "TwoNoiseMot"),
        ("STANDARDIZED_2V2_NOISE_DELAY_MEDIUM", "control_delay", "TwoNoiseDel"),
    ]},
}


def collect_noise(scale: str) -> dict | None:
    """Per-condition crossover and paired degradation from nominal, seed by seed.

    None until every condition has sealed. Fails closed if a record's perturbation is not the
    condition's family, if its poles are not attested, or if the seed sets differ (the
    degradation is only defined on matched seeds)."""
    cfg = NOISE.get(scale)
    if cfg is None:
        return None
    recs = {}
    for label, family, _stem in cfg["conditions"]:
        rec_p = SD / f"{label}_SPECIALIST_CROSSOVER_EVAL_RESULT.json"
        if not rec_p.is_file():
            return None
        rec = _sealed(rec_p)
        if (rec.get("perturbation") or {}).get("family") != family:
            raise SystemExit(f"FAIL-CLOSED: {rec_p.name} perturbation {rec.get('perturbation')} is not {family}")
        att = rec.get("pole_attestations") or {}
        if not all(att.get(p, {}).get("hashes_match") is True for p in ("A", "B")):
            raise SystemExit(f"FAIL-CLOSED: {rec_p.name} lacks a matching pole attestation for both poles")
        csv_p = SD / f"{label.lower()}_specialist_crossover_eval_rows.csv"
        recs[label] = (rec, csv_p, _cells(csv_p, "policy"))
    nom_label = cfg["conditions"][0][0]
    seeds = sorted(recs[nom_label][2][("pi_A", "A")])
    out = {"spec": cfg["spec"], "n_seeds": len(seeds), "seed_block": [seeds[0], seeds[-1]], "conditions": []}

    def vecs(cells):
        for k, c in cells.items():
            if sorted(c) != seeds:
                raise SystemExit(f"FAIL-CLOSED: noise cell {k} seed set differs from nominal -- not matched")
        v = {k: np.array([c[s] for s in seeds], dtype=np.float64) for k, c in cells.items()}
        return v[("pi_A", "A")] - v[("pi_B", "A")], v[("pi_B", "B")] - v[("pi_A", "B")], v

    nom_a, nom_b, _ = vecs(recs[nom_label][2])
    for label, family, stem in cfg["conditions"]:
        rec, csv_p, cells = recs[label]
        da, db, v = vecs(cells)
        d = {"label": label, "family": family, "stem": stem,
             "perturbation": rec["perturbation"],
             "delta_A": _mean_ci(da), "delta_B": _mean_ci(db),
             "own_pole_win": {"pi_A+D@A": float(v[("pi_A", "A")].mean()), "pi_B@B": float(v[("pi_B", "B")].mean())},
             "record": f"{label}_SPECIALIST_CROSSOVER_EVAL_RESULT.json", "rows_csv_sha256": _sha(csv_p)}
        for k in ("delta_A", "delta_B"):
            _check_recorded(f"{d['record']} {k}", d[k], rec["PRIMARY_GATE"][k])
        if family != "nominal":
            d["paired_change_vs_nominal"] = {"delta_A": _mean_ci(da - nom_a), "delta_B": _mean_ci(db - nom_b)}
        out["conditions"].append(d)
    return out


def collect() -> dict:
    out_rows = []
    for spec in ROWS:
        row = {k: spec[k] for k in ("scale", "row", "table_role")}
        if spec.get("paired_with"):
            row["paired_with"] = spec["paired_with"]
        if spec.get("label"):
            row["label"] = spec["label"]
        if spec["kind"] == "pending":
            row.update(status="PENDING", reason=spec["pending_reason"])
            out_rows.append(row)
            continue
        if spec["kind"] in ("sharing_result", "generalist_result"):
            filled = _collect_suite_arm(spec)
            if filled is None:
                row.update(status="PENDING", reason=f"{spec['label']} not sealed yet")
            else:
                row.update(status="FILLED", **filled)
                if spec.get("note"):
                    row["note"] = spec["note"]
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
    by_label = {r["label"]: r for r in out_rows if r.get("label")}
    for r in out_rows:
        if not r.get("paired_with") or r["status"] != "FILLED":
            continue
        p = by_label.get(r["paired_with"])
        if p is None or p["status"] != "FILLED":
            raise SystemExit(f"FAIL-CLOSED: {r['label']} is paired with {r['paired_with']}, which is not filled")
        if (p["n_seeds"], p["seed_block"]) != (r["n_seeds"], r["seed_block"]):
            raise SystemExit(f"FAIL-CLOSED: {r['label']} seeds {r['seed_block']} (n={r['n_seeds']}) differ from "
                             f"its pair {p['label']} {p['seed_block']} (n={p['n_seeds']})")
    return {
        "record_id": "BASELINE_RESULTS_MEAN_EFFECTS",
        "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "reporting_rule": "headline = mean delta_A / delta_B at every scale (PI 2026-09-27); 95% intervals for figures; frozen gate verdicts carried as provenance only",
        "delta_convention": "delta_A = V(pi_A,A) - V(pi_B,A); delta_B = V(pi_B,B) - V(pi_A,B)",
        "bootstrap": "eval_hog_psp_v3._mean_ci -- paired percentile over seeds, n_boot=20000, alpha=0.05, rng_seed=7",
        "rows": out_rows,
        "noise": {scale: collect_noise(scale) for scale in NOISE},
    }


def _md(doc: dict) -> str:
    L = ["# Baseline results: mean specialization effect (Δ_A / Δ_B)", "",
         "Generated by `experiments/collect_baseline_results.py` from the rows on disk (every cell re-derived and "
         "checked against its governing record). Headline = mean effect; 95% intervals are for figures.", "",
         "Δ_A = V(π_A, A) − V(π_B, A); Δ_B = V(π_B, B) − V(π_A, B).", "",
         "Use: `main` = the scale's reported value; `role_allocation_paired` = same-seed arm for the role-allocation "
         "comparison only (select rows by this field, never by position).", "",
         "| Scale | Baseline | Use | Mean Δ_A / Δ_B | 95% CI (figures) | Win rates π_A@A, π_A@B, π_B@A, π_B@B | n | Source |",
         "|---|---|---|---|---|---|---:|---|"]
    for r in doc["rows"]:
        if r["status"] != "FILLED":
            L.append(f"| {r['scale']} | {r['row']} | {r['table_role']} | PENDING | — | — | — | {r['reason']} |")
            continue
        if "delta_G_A" in r:          # Generalist: no crossover Delta (GENERALIST_DEFINITION_V1)
            a, b, va, vb = r["delta_G_A"], r["delta_G_B"], r["V_pole_A"], r["V_pole_B"]
            L.append(f"| {r['scale']} | {r['row']} | {r['table_role']} | Δ_G **{a['mean']:+.3f} / {b['mean']:+.3f}** | "
                     f"[{a['lcb95']:+.3f}, {a['ucb95']:+.3f}] / [{b['lcb95']:+.3f}, {b['ucb95']:+.3f}] | "
                     f"π_G@A {va['mean']:.3f}, π_G@B {vb['mean']:.3f} | {r['n_seeds']} | "
                     f"`{r['provenance']['record']}` ({r['provenance']['seed_class']}) |")
            continue
        a, b, w = r["delta_A"], r["delta_B"], r["win_rates"]
        L.append(f"| {r['scale']} | {r['row']} | {r['table_role']} | **{a['mean']:+.3f} / {b['mean']:+.3f}** | "
                 f"[{a['lcb95']:+.3f}, {a['ucb95']:+.3f}] / [{b['lcb95']:+.3f}, {b['ucb95']:+.3f}] | "
                 f"{w['pi_A@A']:.3f}, {w['pi_A@B']:.3f}, {w['pi_B@A']:.3f}, {w['pi_B@B']:.3f} | {r['n_seeds']} | "
                 f"`{r['provenance']['record']}` ({r['provenance'].get('seed_class') or r['provenance']['record_status']}) |")
    notes = [f"- **{r['scale']} {r['row'].split(' (')[0]}**: {r['note']}" for r in doc["rows"] if r.get("note")]
    if notes:
        L += ["", "Notes:"] + notes
    L += ["", "Frozen gate verdicts (lower bound above zero on both Δ) are kept in the records as provenance and are "
          "not the headline: " + "; ".join(f"{r['scale']} {r['row'].split(' (')[0]} = {r['provenance']['frozen_gate_verdict_provenance_only']}"
                                            for r in doc["rows"] if r["status"] == "FILLED"
                                            and "frozen_gate_verdict_provenance_only" in r["provenance"]) + "."]
    for scale, noise in (doc.get("noise") or {}).items():
        if not noise:
            L += ["", f"Deployment noise {scale}: PENDING (not every condition has sealed)."]
            continue
        L += ["", f"## Deployment noise, {scale} (final system, {noise['n_seeds']} matched seeds, `{noise['spec']}`)", "",
              "| Condition | Mean Δ_A / Δ_B | 95% CI | Own-pole win π_A+D@A, π_B@B | Paired change vs nominal Δ_A / Δ_B |",
              "|---|---|---|---|---|"]
        for c in noise["conditions"]:
            a, b, w = c["delta_A"], c["delta_B"], c["own_pole_win"]
            ch = c.get("paired_change_vs_nominal")
            chs = "—" if ch is None else f"{ch['delta_A']['mean']:+.3f} / {ch['delta_B']['mean']:+.3f}"
            L.append(f"| {c['family']} | **{a['mean']:+.3f} / {b['mean']:+.3f}** | "
                     f"[{a['lcb95']:+.3f}, {a['ucb95']:+.3f}] / [{b['lcb95']:+.3f}, {b['ucb95']:+.3f}] | "
                     f"{w['pi_A+D@A']:.3f}, {w['pi_B@B']:.3f} | {chs} |")
    return "\n".join(L) + "\n"


#: Macro stem per sealed row (letters only, as LaTeX requires). experiments.tex references only
#: these macros, so every number in the paper is regenerated from the sealed rows, never typed.
TEX_STEM = {
    "STANDARDIZED_2V2_DIAG_PRESPLIT": "TwoSpec",
    "STANDARDIZED_2V2_SPLIT_K1_CONFIRMATORY": "TwoSep",
    "STANDARDIZED_2V2_DIAG_SPLIT": "TwoSepPaired",
    "STANDARDIZED_2V2_SHARE_ENCODER": "TwoShareEnc",
    "STANDARDIZED_2V2_FULLY_SHARED_Z": "TwoFullZ",
    "STANDARDIZED_2V2_GENERALIST": "TwoGen",
}


def _tex(doc: dict, scale: str) -> str:
    """\\newcommand macros for one scale's FILLED rows. A row that is not filled defines no macro, so a
    paper that cites it fails to compile instead of printing a stale number."""
    s = lambda x: f"{x:+.3f}"            # noqa: E731  signed; cite inside math mode ($\TwoSepDA$) for a true minus
    u = lambda x: f"{x:.3f}"                                            # noqa: E731
    out = [f"% Generated by experiments/collect_baseline_results.py -- do not edit by hand.",
           f"% Source: paper/data/baseline_results_mean_effects.json ({doc['utc']})."]

    def cmd(name, val):
        out.append(f"\\newcommand{{\\{name}}}{{{val}}}")

    for r in doc["rows"]:
        stem = TEX_STEM.get(r.get("label", ""))
        if r["scale"] != scale or r["status"] != "FILLED" or stem is None:
            continue
        cmd(f"{stem}N", r["n_seeds"])
        pairs = (("DGA", "delta_G_A"), ("DGB", "delta_G_B")) if "delta_G_A" in r else (("DA", "delta_A"), ("DB", "delta_B"))
        for short, key in pairs:
            cmd(f"{stem}{short}", s(r[key]["mean"]))
            cmd(f"{stem}{short}Lo", s(r[key]["lcb95"]))
            cmd(f"{stem}{short}Hi", s(r[key]["ucb95"]))
        if "delta_G_A" in r:
            cmd(f"{stem}VA", u(r["V_pole_A"]["mean"]))
            cmd(f"{stem}VB", u(r["V_pole_B"]["mean"]))
            cmd(f"{stem}OursA", u(r["ours_win_rates"]["pi_A+D@A"]))
            cmd(f"{stem}OursB", u(r["ours_win_rates"]["pi_B@B"]))
        else:
            for k, name in (("pi_A@A", "WAA"), ("pi_A@B", "WAB"), ("pi_B@A", "WBA"), ("pi_B@B", "WBB")):
                cmd(f"{stem}{name}", u(r["win_rates"][k]))
        prov = r["provenance"]
        if prov.get("unique_actor_params"):
            cmd(f"{stem}Params", f"{prov['unique_actor_params'] / 1e6:.2f}M")
        if prov.get("holdout_fidelity"):
            f = prov["holdout_fidelity"]
            cmd(f"{stem}AgreeA", u(f["holdout_agree_z0_vs_piA"]))
            cmd(f"{stem}AgreeB", u(f["holdout_agree_z1_vs_piB"]))
    noise = (doc.get("noise") or {}).get(scale)
    if noise:
        for c in noise["conditions"]:
            st = c["stem"]
            cmd(f"{st}N", noise["n_seeds"])
            for short, key in (("DA", "delta_A"), ("DB", "delta_B")):
                cmd(f"{st}{short}", s(c[key]["mean"]))
                cmd(f"{st}{short}Lo", s(c[key]["lcb95"]))
                cmd(f"{st}{short}Hi", s(c[key]["ucb95"]))
            cmd(f"{st}WAA", u(c["own_pole_win"]["pi_A+D@A"]))
            cmd(f"{st}WBB", u(c["own_pole_win"]["pi_B@B"]))
            for short, key in (("DDA", "delta_A"), ("DDB", "delta_B")):
                if "paired_change_vs_nominal" in c:
                    ch = c["paired_change_vs_nominal"][key]
                    cmd(f"{st}{short}", s(ch["mean"]))
                    cmd(f"{st}{short}Lo", s(ch["lcb95"]))
                    cmd(f"{st}{short}Hi", s(ch["ucb95"]))
    return "\n".join(out) + "\n"


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
        for scale in ("2v2", "4v4", "6v6"):
            if any(r["scale"] == scale and r["status"] == "FILLED" and r.get("label") in TEX_STEM for r in doc["rows"]) \
                    or (doc.get("noise") or {}).get(scale):
                p = OUT_TEX_DIR / f"results_{scale}.tex"
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(_tex(doc, scale), encoding="utf-8")
                print(f"-> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
