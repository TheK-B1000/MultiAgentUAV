r"""Pre-registered readings for STANDARDIZED_2V2_POLE_A_LOCALIZATION_DIAG_SPEC.json.

    python experiments/analyze_pole_a_localization_2v2.py [--write]

Reads the two SEALED arm records and their rows CSVs and computes exactly the quantities the
spec pre-registered (READINGS_preregistered), with the spec's bootstrap (paired percentile over
seeds, n=20000, alpha=0.05, rng_seed=7):

  delta_A_presplit   = V(pi_A_repaired, A) - V(pi_B, A)
  delta_A_split      = V(pi_A+D, A)        - V(pi_B, A)
  split_effect_on_A  = V(pi_A_repaired, A) - V(pi_A+D, A)      paired by seed across arms

and reports pi_B's per-seed agreement across the two arms (evaluation stochasticity). It refuses
unless both records are SEALED, carry the pinned checkpoint hashes, and cover the same seeds.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
SPEC = SD / "STANDARDIZED_2V2_POLE_A_LOCALIZATION_DIAG_SPEC.json"
ARMS = {"presplit": "STANDARDIZED_2V2_DIAG_PRESPLIT", "split": "STANDARDIZED_2V2_DIAG_SPLIT"}
N_BOOT, ALPHA, RNG = 20000, 0.05, 7


def _ci(x: np.ndarray) -> dict:
    rng = np.random.default_rng(RNG)
    idx = rng.integers(0, len(x), size=(N_BOOT, len(x)))
    bs = x[idx].mean(axis=1)
    return {"mean": float(x.mean()), "lcb95": float(np.percentile(bs, 100 * ALPHA / 2)),
            "ucb95": float(np.percentile(bs, 100 * (1 - ALPHA / 2))), "n": int(len(x))}


def _arm(label: str, want_pi_a: str, pins: dict) -> dict:
    rec = json.loads((SD / f"{label}_SPECIALIST_CROSSOVER_EVAL_RESULT.json").read_text(encoding="utf-8"))
    if rec.get("status") != "SEALED":
        raise SystemExit(f"REFUSING: {label} record status is {rec.get('status')!r}, not SEALED")
    if rec["checkpoints"]["pi_A"] != pins[want_pi_a]["sha256"] or rec["checkpoints"]["pi_B"] != pins["pi_B_repaired"]["sha256"]:
        raise SystemExit(f"REFUSING: {label} checkpoint hashes differ from the spec pins")
    rows = list(csv.DictReader((SD / f"{label.lower()}_specialist_crossover_eval_rows.csv").open(encoding="utf-8")))
    by: dict = {}
    for r in rows:
        by.setdefault((r["policy"], r["pole"]), {})[int(r["seed"])] = float(r["win"])
    return {"record": rec, "by": by}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    pins = spec["CHECKPOINTS_locked"]
    pre = _arm(ARMS["presplit"], "pi_A_repaired", pins)
    spl = _arm(ARMS["split"], "pi_D", pins)
    if spl["record"]["split_policy_pi_A"]["frozen_attack_sha256"] != pins["pi_A_repaired"]["sha256"]:
        raise SystemExit("REFUSING: split arm frozen-attack hash differs from the pin")
    seeds = sorted(pre["by"][("pi_A", "A")])
    if seeds != sorted(spl["by"][("pi_A", "A")]) or len(seeds) != spec["SEEDS"]["n"]:
        raise SystemExit("REFUSING: the two arms do not cover the same seed set")

    def v(arm, pol, pole):
        return np.array([arm["by"][(pol, pole)][s] for s in seeds])

    out = {
        "record_id": "STANDARDIZED_2V2_POLE_A_LOCALIZATION_DIAG_READINGS",
        "spec": SPEC.name,
        "seeds": {"block": [seeds[0], seeds[-1]], "n": len(seeds)},
        "win_rates": {
            "presplit": {f"{p}@{q}": float(v(pre, p, q).mean()) for p in ("pi_A", "pi_B") for q in ("A", "B")},
            "split": {f"{p}@{q}": float(v(spl, p, q).mean()) for p in ("pi_A", "pi_B") for q in ("A", "B")},
        },
        "delta_A_presplit": _ci(v(pre, "pi_A", "A") - v(pre, "pi_B", "A")),
        "delta_A_split": _ci(v(spl, "pi_A", "A") - v(spl, "pi_B", "A")),
        "split_effect_on_A": _ci(v(pre, "pi_A", "A") - v(spl, "pi_A", "A")),
        "delta_B_presplit": _ci(v(pre, "pi_B", "B") - v(pre, "pi_A", "B")),
        "delta_B_split": _ci(v(spl, "pi_B", "B") - v(spl, "pi_A", "B")),
        "pi_B_cross_arm_agreement": {q: float((v(pre, "pi_B", q) == v(spl, "pi_B", q)).mean()) for q in ("A", "B")},
        "bootstrap": {"procedure": "paired percentile bootstrap over seeds", "samples": N_BOOT, "alpha": ALPHA, "rng_seed": RNG},
    }
    se = out["split_effect_on_A"]
    if se["mean"] > 0 and se["lcb95"] > 0:
        reading = "SPLIT_WEAKENS_A"
    elif se["mean"] < 0 and se["ucb95"] < 0:
        reading = "SPLIT_HELPS_A"
    else:
        # split_effect's interval straddles 0: delta_A_presplit is not clearly larger than
        # delta_A_split, which is exactly the spec's WEAKNESS_PREDATES_SPLIT condition.
        reading = "WEAKNESS_PREDATES_SPLIT"
    out["reading"] = reading
    out["reading_rule"] = spec["READINGS_preregistered"]["interpretation"]
    print(json.dumps({k: out[k] for k in ("win_rates", "delta_A_presplit", "delta_A_split", "split_effect_on_A",
                                         "delta_B_presplit", "delta_B_split", "pi_B_cross_arm_agreement", "reading")}, indent=1))
    if a.write:
        p = SD / f"{out['record_id']}.json"
        if p.exists():
            raise SystemExit(f"REFUSING to overwrite {p.name}")
        p.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
        print("->", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
