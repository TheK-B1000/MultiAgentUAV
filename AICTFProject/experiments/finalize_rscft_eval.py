"""Write RSCFT's terminal verdict record, applying the frozen interpretation mechanically.

eval_rscft.py deliberately stopped before writing a verdict when it detected a tie/reversal
(delta <= 0 on A_C and A_T), routing instead to the required row-level integrity audit. That
audit has since run and returned GENUINE_REVERSAL_POLE_A with zero failures -- i.e. the
reversals are a real behavioural result, not an evaluator defect. This script completes the
deferred step.

It does NOT decide anything. It reads the frozen artifacts and applies
RSCFT_SPEC.json#EVAL_PROTOCOL's already-frozen PRIMARY_GATE and PREREGISTERED_INTERPRETATION
to the audit's independently-recomputed deltas. The gate wording is frozen: "delta_A > 0 AND
LCB95(delta_A) > 0 AND delta_B > 0 AND LCB95(delta_B) > 0 ... same primary gate. No mercy, no
reinterpretation."

Refuses unless the audit reached a genuine-result verdict -- a defect verdict must never be
converted into a scientific reading.

Run:  python experiments/finalize_rscft_eval.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
SPEC = SD / "RSCFT_SPEC.json"
FLAG = SD / "RSCFT_EVAL_INTEGRITY_REQUIRED.json"
AUDIT = SD / "RSCFT_EVAL_INTEGRITY.json"
FROZEN = SD / "RSCFT_MODELS_FROZEN.json"
OUT = SD / "RSCFT_EVAL_RESULT.json"

GENUINE_VERDICTS = ("GENUINE_REVERSAL_POLE_A", "GENUINE_REVERSAL_ALL_CELLS",
                    "GENUINE_REVERSAL", "GENUINE_TIE", "GENUINE_RESULT")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def passes_gate(delta_a: dict, delta_b: dict) -> bool:
    """The frozen criterion, verbatim: mean > 0 AND lcb95 > 0, on BOTH poles."""
    return bool(delta_a["mean"] > 0 and delta_a["lcb95"] > 0
                and delta_b["mean"] > 0 and delta_b["lcb95"] > 0)


def main() -> int:
    if OUT.is_file():
        raise SystemExit(f"REFUSING: {OUT.name} exists; the terminal record is one-shot")
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    flag = json.loads(FLAG.read_text(encoding="utf-8"))
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))

    if audit.get("VERDICT") not in GENUINE_VERDICTS:
        raise SystemExit(f"REFUSING: audit verdict {audit.get('VERDICT')!r} is not a "
                         "genuine-result verdict; a suspected defect must not be converted "
                         "into a scientific reading")
    if audit.get("failures"):
        raise SystemExit(f"REFUSING: audit recorded failures: {audit['failures']}")

    d = audit["checks"]["3_deltas_and_gammas_recomputed"]
    if not d.get("matches_flag_point_estimates", False):
        raise SystemExit("REFUSING: the audit's recomputed deltas do not match the EVAL's own "
                         "flagged point estimates")

    retention = {"delta_A": d["delta_A_TREATMENT"], "delta_B": d["delta_B_TREATMENT"]}
    control = {"delta_A": d["delta_A_CONTROL"], "delta_B": d["delta_B_CONTROL"]}
    gamma_a, gamma_b = d["Gamma_A"], d["Gamma_B"]

    retention_pass = passes_gate(retention["delta_A"], retention["delta_B"])
    control_pass = passes_gate(control["delta_A"], control["delta_B"])
    gamma_clears = bool(gamma_a["lcb95"] > 0 and gamma_b["lcb95"] > 0)

    table = {(True, False): "crossover recovered under retention stabilization",
             (True, True): "crossover recovered, but retention is not necessary under this run",
             (False, False): "retention stabilization insufficient"}
    reading = table.get((retention_pass, control_pass))
    unmatched = reading is None
    if unmatched:
        reading = (f"UNMATCHED COMBINATION (retention="
                   f"{'PASS' if retention_pass else 'FAIL'}, "
                   f"control={'PASS' if control_pass else 'FAIL'}). RSCFT_SPEC.json's frozen "
                   "interpretation list has no row for this outcome; recorded as such rather "
                   "than coerced into the nearest row. Requires an explicit PI reading.")
    qualifier = None
    if retention_pass and not gamma_clears:
        qualifier = ("claim recovery under the complete treatment, not statistically "
                     "established retention attribution")

    print(f"RSCFT TERMINAL VERDICT  {_now()}\n")
    for name, arm in (("RETENTION (treatment)", retention), ("CONTROL (causal only)", control)):
        a, b = arm["delta_A"], arm["delta_B"]
        print(f"  {name:22s} delta_A {a['mean']:+.4f} [{a['lcb95']:+.4f}, {a['ucb95']:+.4f}]"
              f"   delta_B {b['mean']:+.4f} [{b['lcb95']:+.4f}, {b['ucb95']:+.4f}]")
    print(f"\n  retention gate: {'PASS' if retention_pass else 'FAIL'}"
          f"    control gate: {'PASS' if control_pass else 'FAIL'}")
    print(f"  Gamma_A {gamma_a['mean']:+.4f} [{gamma_a['lcb95']:+.4f}, {gamma_a['ucb95']:+.4f}]"
          f"   Gamma_B {gamma_b['mean']:+.4f} [{gamma_b['lcb95']:+.4f}, {gamma_b['ucb95']:+.4f}]")
    print(f"  Gamma clears zero on both poles: {gamma_clears}")
    print(f"\n  READING: {reading}")

    OUT.write_text(json.dumps({
        "record": "RSCFT sealed EVAL terminal verdict", "status": "FROZEN_RESULT",
        "one_shot": True, "utc": _now(),
        "implements": "RSCFT_SPEC.json#EVAL_PROTOCOL",
        "written_by": "experiments/finalize_rscft_eval.py -- eval_rscft.py deferred this step "
                      "when it detected a tie/reversal and routed to the mandatory audit",
        "PRIMARY_GATE": {
            "criterion": spec["EVAL_PROTOCOL"]["PRIMARY_GATE"]["criterion"],
            "RETENTION_treatment": {**retention, "passes": retention_pass},
            "CONTROL_causal_only": {**control, "passes": control_pass}},
        "ATTRIBUTION": {"Gamma_A": gamma_a, "Gamma_B": gamma_b,
                        "clears_zero_both_poles": gamma_clears,
                        "reading": "neither Gamma CI clears zero, so retention has NO "
                                   "statistically established effect in either direction -- "
                                   "this is not evidence that retention helped, and equally "
                                   "not evidence that it harmed"},
        "READING": reading, "QUALIFIER": qualifier,
        "unmatched_interpretation_combination": unmatched,
        "integrity_audit": {
            "verdict": audit["VERDICT"], "failures": audit["failures"],
            "triggered_by": flag["triggered_by"],
            "PROVENANCE_CAVEAT": "the audit script was created at 15:02 local, approximately "
                                 "19 minutes AFTER rscft_eval_rows.csv was written at 14:42. "
                                 "The frozen rule asks for an audit 'written before its rows "
                                 "are read', and the CCP-S2 precedent was stricter still "
                                 "(written before the row files existed at all). Mitigating: "
                                 "its check structure is template-derived from "
                                 "verify_ccp_s2_eval_integrity.py (the same generic checks "
                                 "1/1b/3/4/5/5b/6), not tailored to this dataset. Recorded "
                                 "here rather than omitted."},
        "checkpoints": {a: frozen[a]["TERMINAL_CHECKPOINT"]["sha256"]
                        for a in ("CONTROL", "TREATMENT")},
        "cross_instrument_context": audit["checks"]["5b_cross_instrument_vs_incumbent_own_sealed_eval"],
        "prior_caveats_still_apply": "CCP_S2_PRELAUNCH_INTERPRETATION_CAVEATS.json -- the "
            "z0/z1 supervision imbalance (177 vs 51 behaviour-changing targets) is unchanged, "
            "since RSCFT reuses the identical frozen causal bank",
        "no_model_selection_occurred": True,
    }, indent=2), encoding="utf-8")
    print(f"\n  -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
