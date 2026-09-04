"""Write trunk-freeze's terminal verdict record, applying the frozen interpretation mechanically.

eval_trunk_freeze.py stopped before writing a verdict when it detected CONTROL's Pole-A
reversal (A_C <= 0), routing to the required row-level integrity audit. That audit has since
run and returned GENUINE_ASYMMETRIC_CONTROL_REVERSAL_POLE_A with zero failures -- the CONTROL
reversal is real, and specifically NOT a shared evaluator defect (TREATMENT's own Pole-A point
estimate, computed on the same seeds through the same code path, is positive).

This script does not decide anything new. It reads the frozen artifacts and applies
TRUNK_FREEZE_SPEC.json#EVAL_PROTOCOL's already-frozen PRIMARY_GATE and
PREREGISTERED_INTERPRETATION to the audit's independently-recomputed deltas -- same gate
wording as CCP-S2 and RSCFT: "delta_A > 0 AND LCB95(delta_A) > 0 AND delta_B > 0 AND
LCB95(delta_B) > 0 ... No mercy, no reinterpretation."

Refuses unless the audit reached a genuine-result verdict.

Run:  python experiments/finalize_trunk_freeze_eval.py
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
SPEC = SD / "TRUNK_FREEZE_SPEC.json"
FLAG = SD / "TRUNK_FREEZE_EVAL_INTEGRITY_REQUIRED.json"
AUDIT = SD / "TRUNK_FREEZE_EVAL_INTEGRITY.json"
FROZEN = SD / "TRUNK_FREEZE_MODELS_FROZEN.json"
OUT = SD / "TRUNK_FREEZE_EVAL_RESULT.json"

GENUINE_VERDICTS = ("GENUINE_REVERSAL_POLE_A", "GENUINE_ASYMMETRIC_CONTROL_REVERSAL_POLE_A",
                    "GENUINE_REVERSAL_ALL_CELLS", "GENUINE_REVERSAL", "GENUINE_TIE",
                    "GENUINE_RESULT", "GENUINE_ROWS_NO_POLE_A_REVERSAL")


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

    treatment = {"delta_A": d["delta_A_TREATMENT"], "delta_B": d["delta_B_TREATMENT"]}
    control = {"delta_A": d["delta_A_CONTROL"], "delta_B": d["delta_B_CONTROL"]}
    gamma_a, gamma_b = d["Gamma_A"], d["Gamma_B"]

    treatment_pass = passes_gate(treatment["delta_A"], treatment["delta_B"])
    control_pass = passes_gate(control["delta_A"], control["delta_B"])
    gamma_clears = bool(gamma_a["lcb95"] > 0 and gamma_b["lcb95"] > 0)

    table = {(True, False): "crossover recovered under trunk freezing",
             (True, True): "crossover recovered, but trunk freezing is not necessary "
                           "under this run",
             (False, False): "trunk freezing insufficient"}
    reading = table.get((treatment_pass, control_pass))
    unmatched = reading is None
    if unmatched:
        reading = (f"UNMATCHED COMBINATION (treatment="
                   f"{'PASS' if treatment_pass else 'FAIL'}, "
                   f"control={'PASS' if control_pass else 'FAIL'}). "
                   "TRUNK_FREEZE_SPEC.json's frozen interpretation list has no row for this "
                   "outcome; recorded as such rather than coerced into the nearest row. "
                   "Requires an explicit PI reading.")
    qualifier = None
    if treatment_pass and not gamma_clears:
        qualifier = ("claim recovery under the complete treatment, not statistically "
                     "established trunk-freeze attribution")

    print(f"TRUNK-FREEZE TERMINAL VERDICT  {_now()}\n")
    for name, arm in (("TREATMENT (trunk-frozen+causal)", treatment),
                      ("CONTROL (trunk-frozen only)", control)):
        a, b = arm["delta_A"], arm["delta_B"]
        print(f"  {name:32s} delta_A {a['mean']:+.4f} [{a['lcb95']:+.4f}, {a['ucb95']:+.4f}]"
              f"   delta_B {b['mean']:+.4f} [{b['lcb95']:+.4f}, {b['ucb95']:+.4f}]")
    print(f"\n  treatment gate: {'PASS' if treatment_pass else 'FAIL'}"
          f"    control gate: {'PASS' if control_pass else 'FAIL'}")
    print(f"  Gamma_A {gamma_a['mean']:+.4f} [{gamma_a['lcb95']:+.4f}, {gamma_a['ucb95']:+.4f}]"
          f"   Gamma_B {gamma_b['mean']:+.4f} [{gamma_b['lcb95']:+.4f}, {gamma_b['ucb95']:+.4f}]")
    print(f"  Gamma clears zero on both poles: {gamma_clears}")
    print(f"\n  READING: {reading}")

    OUT.write_text(json.dumps({
        "record": "Trunk-freeze sealed EVAL terminal verdict", "status": "FROZEN_RESULT",
        "one_shot": True, "utc": _now(),
        "implements": "TRUNK_FREEZE_SPEC.json#EVAL_PROTOCOL",
        "written_by": "experiments/finalize_trunk_freeze_eval.py -- eval_trunk_freeze.py "
                      "deferred this step when it detected CONTROL's Pole-A reversal and "
                      "routed to the mandatory audit",
        "PRIMARY_GATE": {
            "criterion": spec["EVAL_PROTOCOL"]["PRIMARY_GATE"]["criterion"],
            "TREATMENT_trunk_frozen_plus_causal": {**treatment, "passes": treatment_pass},
            "CONTROL_trunk_frozen_only": {**control, "passes": control_pass}},
        "ATTRIBUTION": {"Gamma_A": gamma_a, "Gamma_B": gamma_b,
                        "clears_zero_both_poles": gamma_clears},
        "READING": reading, "QUALIFIER": qualifier,
        "unmatched_interpretation_combination": unmatched,
        "integrity_audit": {
            "verdict": audit["VERDICT"], "failures": audit["failures"],
            "triggered_by": flag["triggered_by"],
            "asymmetric_pattern_check": audit["checks"]["2_asymmetric_pattern"],
            "PROVENANCE": "audit was written and run before any interpretive reading of "
                         "trunk_freeze_eval_rows.csv; a fresh script authored specifically "
                         "for this trigger pattern rather than reused as-is",
        },
        "checkpoints": {a: frozen[a]["TERMINAL_CHECKPOINT"]["sha256"]
                        for a in ("CONTROL", "TREATMENT")},
        "cross_instrument_context": audit["checks"]["5b_cross_instrument_vs_incumbent_own_sealed_eval"],
        "prior_caveats_still_apply": "CCP_S2_PRELAUNCH_INTERPRETATION_CAVEATS.json -- the "
            "z0/z1 supervision imbalance in the causal bank is unchanged, since TREATMENT "
            "reuses the identical frozen causal bank",
        "no_model_selection_occurred": True,
    }, indent=2), encoding="utf-8")
    print(f"\n  -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
