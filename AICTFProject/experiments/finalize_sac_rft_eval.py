"""Write SAC-RFT's terminal verdict record, applying the frozen interpretation mechanically.

eval_sac_rft.py stopped before writing a verdict when it detected an off-diagonal reversal
pattern (CONTROL fails Pole A, TREATMENT fails Pole B), routing to the required row-level
integrity audit. That audit has since run and returned GENUINE_CROSS_REVERSAL_OFF_DIAGONAL
with zero failures -- the pattern is real, and specifically not a shared evaluator defect (see
the audit's check 2: a code-path bug could not selectively hit exactly the two off-diagonal
cells while sparing both on-diagonal cells).

This script does not decide anything new. It reads the frozen artifacts and applies
SAC_RFT_SPEC.json#EVAL_PROTOCOL's already-frozen PRIMARY_GATE and PREREGISTERED_INTERPRETATION
to the audit's independently-recomputed deltas -- same gate wording as every prior EVAL in this
program: "delta_A > 0 AND LCB95(delta_A) > 0 AND delta_B > 0 AND LCB95(delta_B) > 0 ... No
mercy, no reinterpretation."

Refuses unless the audit reached a genuine-result verdict.

Run:  python experiments/finalize_sac_rft_eval.py
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
SPEC = SD / "SAC_RFT_SPEC.json"
FLAG = SD / "SAC_RFT_EVAL_INTEGRITY_REQUIRED.json"
AUDIT = SD / "SAC_RFT_EVAL_INTEGRITY.json"
FROZEN = SD / "SAC_RFT_MODELS_FROZEN.json"
OUT = SD / "SAC_RFT_EVAL_RESULT.json"

GENUINE_VERDICTS = ("GENUINE_REVERSAL_POLE_A", "GENUINE_ASYMMETRIC_CONTROL_REVERSAL_POLE_A",
                    "GENUINE_CROSS_REVERSAL_OFF_DIAGONAL", "GENUINE_REVERSAL_ALL_CELLS",
                    "GENUINE_REVERSAL", "GENUINE_TIE", "GENUINE_RESULT",
                    "GENUINE_ROWS_NO_REVERSAL", "GENUINE_ROWS_NO_POLE_A_REVERSAL")


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

    anchor = {"delta_A": d["delta_A_TREATMENT"], "delta_B": d["delta_B_TREATMENT"]}
    ema_control = {"delta_A": d["delta_A_CONTROL"], "delta_B": d["delta_B_CONTROL"]}
    gamma_a, gamma_b = d["Gamma_A"], d["Gamma_B"]

    anchor_pass = passes_gate(anchor["delta_A"], anchor["delta_B"])
    control_pass = passes_gate(ema_control["delta_A"], ema_control["delta_B"])
    gamma_clears = bool(gamma_a["lcb95"] > 0 and gamma_b["lcb95"] > 0)

    table = {(True, False): "frozen strategic anchor recovered crossover where EMA "
                            "retention did not",
             (True, True): "crossover recovered under both teachers; frozen anchor not "
                           "necessary under this run",
             (False, False): "frozen strategic anchor also insufficient"}
    reading = table.get((anchor_pass, control_pass))
    unmatched = reading is None
    if unmatched:
        reading = (f"UNMATCHED COMBINATION (anchor={'PASS' if anchor_pass else 'FAIL'}, "
                   f"ema_control={'PASS' if control_pass else 'FAIL'}). "
                   "SAC_RFT_SPEC.json's frozen interpretation list has no row for this "
                   "outcome; recorded as such rather than coerced into the nearest row. "
                   "Requires an explicit PI reading.")
    qualifier = None
    if anchor_pass and not gamma_clears:
        qualifier = ("claim recovery under the complete treatment, not statistically "
                     "established anchor attribution")

    print(f"SAC-RFT TERMINAL VERDICT  {_now()}\n")
    for name, arm in (("ANCHOR (frozen reference)", anchor), ("EMA_CONTROL (moving teacher)",
                      ema_control)):
        a, b = arm["delta_A"], arm["delta_B"]
        print(f"  {name:28s} delta_A {a['mean']:+.4f} [{a['lcb95']:+.4f}, {a['ucb95']:+.4f}]"
              f"   delta_B {b['mean']:+.4f} [{b['lcb95']:+.4f}, {b['ucb95']:+.4f}]")
    print(f"\n  anchor gate: {'PASS' if anchor_pass else 'FAIL'}"
          f"    ema_control gate: {'PASS' if control_pass else 'FAIL'}")
    print(f"  Gamma_A {gamma_a['mean']:+.4f} [{gamma_a['lcb95']:+.4f}, {gamma_a['ucb95']:+.4f}]"
          f"   Gamma_B {gamma_b['mean']:+.4f} [{gamma_b['lcb95']:+.4f}, {gamma_b['ucb95']:+.4f}]")
    print(f"  Gamma clears zero on both poles: {gamma_clears}")
    print(f"\n  READING: {reading}")

    OUT.write_text(json.dumps({
        "record": "SAC-RFT sealed EVAL terminal verdict", "status": "FROZEN_RESULT",
        "one_shot": True, "utc": _now(),
        "implements": "SAC_RFT_SPEC.json#EVAL_PROTOCOL",
        "written_by": "experiments/finalize_sac_rft_eval.py -- eval_sac_rft.py deferred this "
                      "step when it detected the off-diagonal reversal pattern and routed to "
                      "the mandatory audit",
        "PRIMARY_GATE": {
            "criterion": spec["EVAL_PROTOCOL"]["PRIMARY_GATE"]["criterion"],
            "ANCHOR_TREATMENT_frozen": {**anchor, "passes": anchor_pass},
            "EMA_CONTROL": {**ema_control, "passes": control_pass}},
        "ATTRIBUTION": {"Gamma_A": gamma_a, "Gamma_B": gamma_b,
                        "clears_zero_both_poles": gamma_clears},
        "READING": reading, "QUALIFIER": qualifier,
        "unmatched_interpretation_combination": unmatched,
        "integrity_audit": {
            "verdict": audit["VERDICT"], "failures": audit["failures"],
            "triggered_by": flag["triggered_by"],
            "off_diagonal_pattern_check": audit["checks"]["2_off_diagonal_cross_pattern"],
            "PROVENANCE": "audit was written and run before any interpretive reading of "
                         "sac_rft_eval_rows.csv; a fresh script authored specifically for "
                         "this trigger pattern rather than reused as-is",
        },
        "checkpoints": {a: frozen[a]["TERMINAL_CHECKPOINT"]["sha256"]
                        for a in ("CONTROL", "TREATMENT")},
        "cross_instrument_context": audit["checks"]["5b_cross_instrument_vs_incumbent_own_sealed_eval"],
        "prior_caveats_still_apply": "CCP_S2_PRELAUNCH_INTERPRETATION_CAVEATS.json -- the "
            "z0/z1 supervision imbalance in the causal bank is unchanged, since both SAC-RFT "
            "arms reuse the identical frozen causal bank",
        "no_model_selection_occurred": True,
    }, indent=2), encoding="utf-8")
    print(f"\n  -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
