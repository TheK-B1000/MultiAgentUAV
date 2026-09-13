r"""Rule-12 hard contract for ACTION_INTERFACE_DECOMP.

Fail-closed: DECOMP ORIGINAL must match ORACLE ORIGINAL, and DECOMP FULL must
match ORACLE PROJECTED, seed-by-seed on GUARD@A for every overlapping scale.

If any win bit disagrees, write INTEGRITY_REQUIRED and refuse binder reading.
If the contract passes, write the sealed paired-loss reading (L_S, L_C, L_F).

    python -m experiments.verify_action_interface_decomp_oracle_contract
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
DECOMP_ROWS = SD / "action_interface_decomp_rows.csv"
ORACLE_ROWS = SD / "projected_teacher_oracle_rows.csv"
AMEND = SD / "ACTION_INTERFACE_DECOMP_RULE12_ORACLE_CONTRACT_AMENDMENT.json"
OUT_OK = SD / "ACTION_INTERFACE_DECOMP_SEALED_READING.json"
OUT_BAD = SD / "ACTION_INTERFACE_DECOMP_INTEGRITY_REQUIRED.json"


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _boot(d: np.ndarray, n_boot=20000, alpha=0.05, seed=7) -> dict:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    b = d[idx].mean(axis=1)
    lo, hi = np.percentile(b, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"mean": round(float(d.mean()), 6), "lcb95": round(float(lo), 6),
            "ucb95": round(float(hi), 6), "n": int(d.size)}


def _load(path: Path, *, strategy="GUARD", pole="A") -> dict:
    out = {}
    with path.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r.get("strategy") != strategy or r.get("pole") != pole:
                continue
            key = (r["scale"], r["arm"], int(r["seed"]))
            out[key] = {
                "win": int(r["win"]),
                "blue": int(r["blue"]) if "blue" in r else None,
                "red": int(r["red"]) if "red" in r else None,
            }
    return out


def main() -> int:
    if not AMEND.is_file():
        raise SystemExit(f"REFUSING: missing {AMEND.name}")
    if not DECOMP_ROWS.is_file():
        raise SystemExit(f"REFUSING: missing {DECOMP_ROWS.name} -- collect not finished")
    if not ORACLE_ROWS.is_file():
        raise SystemExit(f"REFUSING: missing {ORACLE_ROWS.name}")

    decomp = _load(DECOMP_ROWS)
    oracle = _load(ORACLE_ROWS)
    # Map decomp arm -> oracle arm
    pairs = (("ORIGINAL", "ORIGINAL"), ("FULL", "PROJECTED"))
    mismatches = []
    compared = 0
    scales = sorted({k[0] for k in decomp})
    seeds_by_scale = {}
    for scale in scales:
        seeds = sorted({k[2] for k in decomp if k[0] == scale and k[1] == "ORIGINAL"})
        seeds_by_scale[scale] = seeds
        for seed in seeds:
            for d_arm, o_arm in pairs:
                dk = (scale, d_arm, seed)
                ok = (scale, o_arm, seed)
                if dk not in decomp:
                    mismatches.append(f"MISSING decomp {dk}")
                    continue
                if ok not in oracle:
                    mismatches.append(f"MISSING oracle {ok}")
                    continue
                compared += 1
                if decomp[dk]["win"] != oracle[ok]["win"]:
                    mismatches.append(
                        f"{scale} seed={seed}: decomp.{d_arm} win={decomp[dk]['win']} "
                        f"!= oracle.{o_arm} win={oracle[ok]['win']}")
                # Prefer score equality too when both present
                if (decomp[dk]["blue"] is not None and oracle[ok]["blue"] is not None
                        and (decomp[dk]["blue"], decomp[dk]["red"])
                        != (oracle[ok]["blue"], oracle[ok]["red"])):
                    mismatches.append(
                        f"{scale} seed={seed}: decomp.{d_arm} "
                        f"score=({decomp[dk]['blue']},{decomp[dk]['red']}) "
                        f"!= oracle.{o_arm} "
                        f"score=({oracle[ok]['blue']},{oracle[ok]['red']})")

    if mismatches:
        rec = {
            "record": "ACTION_INTERFACE_DECOMP Rule-12 oracle contract",
            "status": "INTEGRITY_REQUIRED",
            "utc": _now(),
            "implements": AMEND.name,
            "compared": compared,
            "n_mismatch_lines": len(mismatches),
            "mismatches": mismatches[:50],
            "reading_refused": True,
            "note": ("Overlapping arms disagree with the sealed oracle. "
                     "Do NOT interpret L_S/L_C/L_F. Audit the harness."),
        }
        if OUT_BAD.exists():
            raise SystemExit(f"REFUSING: {OUT_BAD.name} already exists")
        OUT_BAD.write_text(json.dumps(rec, indent=2), encoding="utf-8")
        print(f"CONTRACT FAIL -- {len(mismatches)} issues; -> {OUT_BAD}")
        return 2

    # Contract passed: compute paired losses.
    results = {}
    for scale in scales:
        seeds = seeds_by_scale[scale]

        def V(arm: str) -> np.ndarray:
            return np.array([decomp[(scale, arm, s)]["win"] for s in seeds], dtype=float)

        vo, vs, vc, vf = V("ORIGINAL"), V("SPATIAL_ONLY"), V("COMMIT_ONLY"), V("FULL")
        Ls, Lc, Lf = vo - vs, vo - vc, vo - vf
        results[scale] = {
            "n": len(seeds),
            "V": {
                "ORIGINAL": round(float(vo.mean()), 4),
                "SPATIAL_ONLY": round(float(vs.mean()), 4),
                "COMMIT_ONLY": round(float(vc.mean()), 4),
                "FULL": round(float(vf.mean()), 4),
            },
            "L_S": _boot(Ls),
            "L_C": _boot(Lc),
            "L_F": _boot(Lf),
        }
        print(f"{scale} GUARD@A")
        print(f"  V: ORIG={results[scale]['V']['ORIGINAL']:.4f}  "
              f"SPATIAL={results[scale]['V']['SPATIAL_ONLY']:.4f}  "
              f"COMMIT={results[scale]['V']['COMMIT_ONLY']:.4f}  "
              f"FULL={results[scale]['V']['FULL']:.4f}")
        for tag, L in (("L_S", results[scale]["L_S"]),
                       ("L_C", results[scale]["L_C"]),
                       ("L_F", results[scale]["L_F"])):
            print(f"  {tag}={L['mean']:+.4f} [{L['lcb95']:+.4f},{L['ucb95']:+.4f}]")

    rec = {
        "record": "ACTION_INTERFACE_DECOMP sealed paired-loss reading",
        "status": "FROZEN_RESULT",
        "utc": _now(),
        "implements": [AMEND.name, "ACTION_INTERFACE_DECOMP_SPEC.json"],
        "study_class": "MECHANISTIC_FOLLOW_UP",
        "not": "INDEPENDENT_CONFIRMATION",
        "rule12_oracle_contract": {
            "passed": True, "compared": compared, "mismatches": 0,
            "equalities": [
                "DECOMP.ORIGINAL == ORACLE.ORIGINAL",
                "DECOMP.FULL == ORACLE.PROJECTED",
            ],
        },
        "primary": "L_S, L_C, L_F with paired CIs; binder label is secondary",
        "results": results,
    }
    if OUT_OK.exists():
        raise SystemExit(f"REFUSING: {OUT_OK.name} already exists")
    OUT_OK.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(f"CONTRACT PASS -- -> {OUT_OK}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
