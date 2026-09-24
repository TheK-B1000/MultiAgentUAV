"""DESCRIPTIVE_ONE_DEFENDER_GOAL_VOLUME_ADDENDUM_SPEC.json.

DESCRIPTIVE. NON-GATING. NO LABEL. Reads the sealed causal rows only (no simulation) and reports,
for each of the four policy x pole cells, the paired Blue-goal difference and the paired score-margin
difference (+1 defender minus native) with paired bootstrap 95% intervals, beside the native and +1D
means. Motivation is post-hoc (see the spec). It does not replace the sealed win-rate verdict.
"""
from __future__ import annotations

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

from experiments.run_routed_composition_outcome import _bootstrap  # noqa: E402

SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
SPEC_PATH = SD / "DESCRIPTIVE_ONE_DEFENDER_GOAL_VOLUME_ADDENDUM_SPEC.json"
CAUSAL_ROWS = SD / "DEFENDER_INJECTION_CAUSAL_BRIDGE_ROWS.csv"
CAUSAL_RESULT = SD / "DEFENDER_INJECTION_CAUSAL_BRIDGE_RESULT.json"
OUT = SD / "DESCRIPTIVE_ONE_DEFENDER_GOAL_VOLUME_ADDENDUM_RESULT.json"

SEEDS = list(range(20_900_001, 20_900_001 + 96))
POLICIES, POLES, ARMS = ("pi_A", "pi_B"), ("A", "B"), ("native", "plus_one_defender")


def main() -> int:
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    if not str(spec.get("status", "")).startswith("FROZEN"):
        raise SystemExit(f"REFUSING: addendum spec is not frozen ({spec.get('status')!r})")

    # Bind to the real sealed object: the result must be SEALED and the rows file must be the audited one.
    sealed = json.loads(CAUSAL_RESULT.read_text(encoding="utf-8"))
    if sealed.get("status") != "SEALED":
        raise SystemExit(f"REFUSING: causal result status is {sealed.get('status')!r}, not SEALED")
    want_sha = sealed["AUDIT"]["rows_sha256"]
    got_sha = hashlib.sha256(CAUSAL_ROWS.read_bytes()).hexdigest()
    if got_sha != want_sha:
        raise SystemExit(f"ABORT: rows sha256 {got_sha} != sealed audit rows_sha256 {want_sha}")

    with CAUSAL_ROWS.open(encoding="utf-8") as fh:
        rows = [{**r, "seed": int(r["seed"]), "blue": int(r["blue"]), "red": int(r["red"]),
                 "margin": int(r["margin"])} for r in csv.DictReader(fh)]
    if len(rows) != 768:
        raise SystemExit(f"ABORT: expected 768 rows, found {len(rows)}")
    bad_margin = [r for r in rows if r["margin"] != r["blue"] - r["red"]]
    if bad_margin:
        raise SystemExit(f"ABORT: {len(bad_margin)} row(s) where margin != blue - red, e.g. {bad_margin[0]}")

    def cell(policy: str, pole: str, arm: str, field: str) -> np.ndarray:
        by_seed = {r["seed"]: float(r[field]) for r in rows
                   if r["policy"] == policy and r["pole"] == pole and r["arm"] == arm}
        if sorted(by_seed) != SEEDS:
            raise SystemExit(f"ABORT: {policy}/{pole}/{arm} does not carry exactly the frozen 96 seeds")
        return np.asarray([by_seed[s] for s in SEEDS], dtype=np.float64)

    per_cell: dict[str, dict] = {}
    for pole in POLES:
        for policy in POLICIES:
            entry = {}
            for name, field in (("blue_goals", "blue"), ("margin", "margin")):
                nat, plus = cell(policy, pole, "native", field), cell(policy, pole, "plus_one_defender", field)
                entry[name] = {"native_mean": round(float(nat.mean()), 6), "plus_one_defender_mean": round(float(plus.mean()), 6),
                               "delta_plus1D_minus_native_paired": _bootstrap(plus - nat)}
            per_cell[f"{policy}_pole{pole}"] = entry

    payload = {
        "record_id": "DESCRIPTIVE_ONE_DEFENDER_GOAL_VOLUME_ADDENDUM_RESULT", "implements": SPEC_PATH.name,
        "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "classification": "DESCRIPTIVE ADDENDUM. NON-GATING. NO TERMINAL LABEL. Post-hoc motivation disclosed in the spec.",
        "source": {"rows": CAUSAL_ROWS.name, "rows_sha256": got_sha, "equals_sealed_audit_rows_sha256": True,
                   "n_rows": len(rows), "margin_equals_blue_minus_red_on_every_row": True},
        "PER_CELL": per_cell,
        "claim_boundary": "Descriptive only, on the sealed rows. Does not replace or re-score the sealed win-rate "
                          "verdict ONE_DEFENDER_HARM_ONLY, which stands under its frozen binary-win criterion. No "
                          "mechanism claim, no gate, no label.",
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(f"source rows bound to the sealed audit (sha256 {got_sha[:16]}...); margin == blue - red on all {len(rows)} rows\n")
    print(f"{'cell':<14}{'endpoint':<12}{'native':>9}{'+1D':>9}   {'+1D - native, paired [95% CI]'}")
    for pole in POLES:
        for policy in POLICIES:
            e = per_cell[f"{policy}_pole{pole}"]
            for name in ("blue_goals", "margin"):
                d = e[name]["delta_plus1D_minus_native_paired"]
                print(f"{policy} pole {pole}  {name:<12}{e[name]['native_mean']:>9.3f}{e[name]['plus_one_defender_mean']:>9.3f}"
                      f"   {d['mean']:>+7.3f} [{d['lcb95']:>+7.3f}, {d['ucb95']:>+7.3f}]  n={d['n']}")
    print(f"\n-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
