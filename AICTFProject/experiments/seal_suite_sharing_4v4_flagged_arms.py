"""Post-hoc seal of the four 4v4 distilled-arm exploratory crossovers, ALL OR NOTHING.

The four arms (Fully Shared+z, Share-Encoder, Share-Backbone, Share-Macro) ran through the
pre-Rule-7 runner ``eval_suite_sharing_crossover_4v4.py`` (last at ae89f636), which
hand-wrote status and, on a Delta_B reversal, wrote only an INTEGRITY_REQUIRED flag. Their
row-level audits now pass. This routes all four through ``experiments.run_state.seal``,
which re-derives every sealed statistic from the rows on disk and runs the full audit.

Three rules this file enforces:

* **All or nothing.** Every arm is audited first (``run_audit``, no side effects). If any
  audit fails, NOTHING is sealed -- four experiments run through the same path must not end
  up with an arbitrary provenance difference.
* **Sealed integrity is not historical Rule-9 compliance.** The seed blocks were registered
  AFTER they were spent. ``registration_origin`` and ``historically_pre_registered`` are
  READ from each block's registry entry and carried into the seal, not asserted here.
* **No fabricated timeline.** No RunState is created: a RunState begun now would claim the
  run started now. The actual eval-finished time is read from each run's flag file.

Default is a dry run. ``--seal`` performs the one-shot seal.

Run: python experiments/seal_suite_sharing_4v4_flagged_arms.py [--seal]
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
import experiments.run_state as rs  # noqa: E402
from experiments import seed_registry as sr  # noqa: E402

SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
SPEC_PATH = SD / "SUITE_SHARING_4V4_CROSSOVER_EVAL_SPEC.json"
TEACHER_READING = SD / "4V4_ENTITY_REPAIR_CORRECTED_CROSSOVER_READING.json"
ARMS = ("fully_shared_z", "share_encoder", "share_backbone", "share_macro")
N_BOOT, ALPHA, RNG = 20_000, 0.05, 7
SEED_CLASS = "exploratory"
PRODUCING_RUNNER = {"path": "experiments/eval_suite_sharing_crossover_4v4.py",
                    "last_commit": "ae89f636",
                    "status_mechanism": "hand-written (pre-Rule-7); flag-only on Delta_B reversal"}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def build(arm_key: str, spec: dict) -> dict:
    arm = spec["ARMS"][arm_key]
    label = str(arm["label_exploratory"])
    rows_csv = SD / f"{label.lower()}_crossover_eval_rows.csv"
    flag = SD / f"{label}_CROSSOVER_EVAL_INTEGRITY_REQUIRED.json"
    out = SD / f"{label}_CROSSOVER_EVAL_RESULT.json"
    audit_path = SD / f"{label}_CROSSOVER_EVAL_AUDIT.json"
    ck = ROOT / arm["checkpoint"]
    lo, hi = (int(x) for x in str(spec["SEEDS"][f"{arm_key}_exploratory"]).split(".."))
    seeds = list(range(lo, lo + int(spec["SEEDS"]["n_exploratory"])))

    for need in (rows_csv, flag, ck):
        if not need.is_file():
            raise SystemExit(f"FAIL-CLOSED: {arm_key}: required evidence missing: {need.name}")

    # Registration provenance is READ from the registry; absence fails closed.
    reg = next((b for b in sr.load().get("blocks", []) if b.get("experiment_id") == label), None)
    if reg is None:
        raise SystemExit(f"FAIL-CLOSED: {arm_key}: no registry entry for {label}")
    for k in ("registration_origin", "historically_pre_registered"):
        if k not in reg:
            raise SystemExit(f"FAIL-CLOSED: {arm_key}: registry entry lacks {k!r}; "
                             f"cannot state registration provenance")
    if (int(reg["lo"]), int(reg["hi"])) != (seeds[0], seeds[-1]):
        raise SystemExit(f"FAIL-CLOSED: {arm_key}: registry block {reg['lo']}..{reg['hi']} "
                         f"!= spec block {seeds[0]}..{seeds[-1]}")

    rows = list(csv.DictReader(rows_csv.open(encoding="utf-8")))
    by = {(int(r["z"]), r["pole"], int(r["seed"])): int(r["win"]) for r in rows}

    def wins(z, pole):
        return np.array([by[(z, pole, s)] for s in seeds], dtype=np.float64)

    dA = _mean_ci(wins(0, "A") - wins(1, "A"))
    dB = _mean_ci(wins(1, "B") - wins(0, "B"))
    for d in (dA, dB):
        d["passes"] = bool(d["mean"] > 0 and d["lcb95"] > 0)
    gate = bool(dA["passes"] and dB["passes"])
    reversal = [k for k, d in (("delta_A", dA), ("delta_B", dB)) if d["mean"] <= 0.0]
    cells = {f"z{z}_pole{p}": float(wins(z, p).mean()) for z in (0, 1) for p in ("A", "B")}
    flag_doc = json.loads(flag.read_text(encoding="utf-8"))

    plan = rs.AuditPlan(
        rows_csv=rows_csv, expected_rows=4 * len(seeds), expected_seeds=seeds,
        group_by=("z", "pole"), seed_field="seed",
        int_fields=("z", "seed", "blue", "red", "margin"), binary_fields=("win",),
        derived={
            "win": rs.Derived("int(blue > red)", lambda r: int(int(r["blue"]) > int(r["red"]))),
            "margin": rs.Derived("blue - red", lambda r: int(r["blue"]) - int(r["red"])),
        },
        checkpoints={arm_key: (ck, str(arm["sha256"]))}, spec_path=SPEC_PATH,
        claims=[
            rs.Claim("delta_A", {k: dA[k] for k in ("mean", "lcb95", "ucb95")},
                     minuend={"z": 0, "pole": "A"}, subtrahend={"z": 1, "pole": "A"}),
            rs.Claim("delta_B", {k: dB[k] for k in ("mean", "lcb95", "ucb95")},
                     minuend={"z": 1, "pole": "B"}, subtrahend={"z": 0, "pole": "B"}),
        ],
        n_boot=N_BOOT, alpha=ALPHA, rng_seed=RNG,
        seed_class=SEED_CLASS, experiment_id=label,
    )
    # status is owned by seal(); it must not appear here.
    payload = {
        "record": f"{label} crossover EVAL (post-hoc seal)",
        "one_shot": True,
        "utc": _now(),
        "implements": "SUITE_SHARING_4V4_CROSSOVER_EVAL_SPEC.json#EVALUATION",
        "suite_arm": arm_key, "team_size": 4, "arm": "EXPLORATORY", "confirmatory": False,
        "checkpoint": arm["checkpoint"], "checkpoint_sha256": _sha(ck),
        "seeds": {"block": [seeds[0], seeds[-1]], "n": len(seeds),
                  "shared_across_z_and_poles": True},
        "cell_win_rates": cells,
        "PRIMARY_GATE": {"delta_A": dA, "delta_B": dB, "passes": gate},
        "scientific_verdict": "FLAG" if reversal else ("PASS" if gate else "FAIL"),
        "reversal_on": reversal,
        "PROVENANCE": {
            "sealed_post_hoc": True,
            "sealed_integrity_is_not_historical_rule9_compliance": True,
            "seed_registration_origin": reg["registration_origin"],
            "historically_pre_registered": reg["historically_pre_registered"],
            "registry_entry": label,
            "registry_timestamps_note": (
                f"the registry's allocated_utc/spent_utc ({reg.get('spent_utc')}) record when the "
                "block was RECONCILED, not when its seeds were spent"),
            "eval_finished_utc_from_flag": flag_doc.get("utc"),
            "rows_produced_by": PRODUCING_RUNNER,
            "flag_record": flag.name,
            "no_run_state": "no RunState was created: one begun at seal time would falsely "
                            "claim the run started then",
        },
        "INTERPRETATION": (
            "The distilled students reproduce the 4v4 teacher pair's behavioural asymmetry: the "
            "KL teachers (entity-repair pi_A3 -> z0, corrected pi_B3 -> z1) themselves fail "
            "Delta_B at 4v4 (see 4V4_ENTITY_REPAIR_CORRECTED_CROSSOVER_READING.json). These "
            "reversals are not, by themselves, evidence that sharing destroyed specialization."),
        "bootstrap": {"procedure": "paired percentile bootstrap over evaluation seeds",
                      "samples": N_BOOT, "alpha": ALPHA, "rng_seed": RNG},
        "no_model_selection_occurred": True,
        "total_episodes": len(rows),
        "claim_boundary": "EXPLORATORY n=64; not confirmatory; not PAPER-FAITHFUL",
    }
    return {"arm": arm_key, "label": label, "out": out, "audit_path": audit_path,
            "plan": plan, "payload": payload, "dA": dA, "dB": dB, "reg": reg}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seal", action="store_true", help="perform the one-shot seal")
    args = ap.parse_args()

    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    if not str(spec.get("status", "")).startswith("FROZEN"):
        raise SystemExit(f"REFUSING: {SPEC_PATH.name} not frozen")

    built = [build(a, spec) for a in ARMS]
    existing = [b["out"].name for b in built if b["out"].exists()] + \
               [b["audit_path"].name for b in built if b["audit_path"].exists()]
    if existing:
        raise SystemExit(f"REFUSING: one-shot outputs already exist: {existing}")

    print(f"POST-HOC SEAL, 4v4 distilled arms  {_now()}  mode={'SEAL' if args.seal else 'DRY-RUN'}\n")
    audits = {}
    for b in built:
        a = rs.run_audit(b["plan"])                     # no side effects
        audits[b["arm"]] = a
        failed = a["failed_checks"]
        print(f"  {b['arm']:<16} audit {'PASS' if a['passed'] else 'FAIL'} "
              f"({a['n_gating'] - a['n_failed']}/{a['n_gating']} gating)  "
              f"dA {b['dA']['mean']:+.4f} [{b['dA']['lcb95']:+.4f},{b['dA']['ucb95']:+.4f}]  "
              f"dB {b['dB']['mean']:+.4f} [{b['dB']['lcb95']:+.4f},{b['dB']['ucb95']:+.4f}]  "
              f"verdict {b['payload']['scientific_verdict']}  "
              f"reg {b['reg']['registration_origin']}/pre_registered={b['reg']['historically_pre_registered']}"
              + (f"  FAILED={failed}" if failed else ""))
    all_pass = all(a["passed"] for a in audits.values())

    if not all_pass:
        print("\n  NOT ALL FOUR AUDITS PASS -> sealing NONE (all-or-nothing). The flagged, "
              "row-audited records stand as they are.")
        return 1
    if not args.seal:
        print("\n  DRY RUN: all four audits pass. Re-run with --seal to seal all four.")
        return 0

    for b in built:
        rs.seal(out_path=b["out"], payload=b["payload"], plan=b["plan"],
                state=None, strict=True, audit_path=b["audit_path"])
    print("\n  sealed all four.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
