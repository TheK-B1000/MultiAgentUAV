"""Recover Gate 3 for Confirmation A from a deterministic replay.

Implements artifacts/vgc_specialists/CONFIRMATION_A_RECOVERY_FROZEN.json.

Order of operations is the point of this script:
  1. REPLAY_EQUIVALENCE -- all 35 replayed cells must EXACTLY match the original
     aggregates. Deterministic evaluation on identical inputs must reproduce
     identical numbers, so any difference means the persistence patch was not
     write-only or an input was not what we believe. Zero tolerance.
  2. Only if that passes, compute Gate 3, SELECTED_POLICY_PER_OPPONENT and
     SPECIALIST_INCREMENTAL_VALUE from the recovered per-episode rows.

Gate 1 and Gate 2 are NOT recomputed. They were scored on the original run from
per-cell proportions, which were never missing.

Run:  python experiments/recover_confirmation_a_gate3.py
"""
from __future__ import annotations

import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "artifacts/vgc_specialists/CONFIRMATION_A_RECOVERY_FROZEN.json"
ORIG = ROOT / "artifacts/vgc_diversity/crossplay/specialist_pilot_summary.json"
REPLAY = ROOT / "artifacts/vgc_diversity/crossplay/specialist_pilot_recovery_summary.json"
ROWS = ROOT / "artifacts/vgc_diversity/crossplay/specialist_pilot_recovery_episode_rows.csv"
OUT = ROOT / "artifacts/vgc_specialists/CONFIRMATION_A_GATE3_RESULT.json"

S_OP7 = "vgc_s_op7_seed3900007"
S_OP8 = "vgc_s_op8_seed3900008"
OPPONENTS = ("OP6", "OP7", "OP8", "OP9", "OP10", "OP11", "OP12")


def cells_of(summary: dict) -> dict[str, dict[str, dict]]:
    return {s["policy_id"]: s["per_opponent"] for s in summary["summaries"]}


def _wr(d, pol, opp) -> float:
    v = d.get(pol, {}).get(opp, [])
    return sum(v) / len(v) if v else float("nan")


def main() -> int:
    for p in (ORIG, REPLAY, ROWS):
        if not p.is_file():
            print(f"BLOCKED: missing {p}", file=sys.stderr)
            return 2

    orig = json.loads(ORIG.read_text(encoding="utf-8"))
    rep = json.loads(REPLAY.read_text(encoding="utf-8"))
    co, cr = cells_of(orig), cells_of(rep)

    # ---- 1. REPLAY_EQUIVALENCE, zero tolerance --------------------------
    mismatches = []
    checked = 0
    for pol in co:
        if pol not in cr:
            mismatches.append({"policy": pol, "issue": "missing from replay"})
            continue
        for opp in OPPONENTS:
            a, b = co[pol].get(opp), cr[pol].get(opp)
            if a is None or b is None:
                mismatches.append({"policy": pol, "opponent": opp,
                                   "issue": "cell missing"})
                continue
            checked += 1
            if float(a["win_rate"]) != float(b["win_rate"]) or int(b["n"]) != 60:
                mismatches.append({"policy": pol, "opponent": opp,
                                   "original_wr": a["win_rate"],
                                   "replay_wr": b["win_rate"],
                                   "replay_n": b["n"]})

    equiv = {
        "cells_checked": checked,
        "cells_expected": 35,
        "tolerance": 0.0,
        "seed_base_orig": orig.get("eval_seed_base"),
        "seed_base_replay": rep.get("eval_seed_base"),
        "seed_base_match": orig.get("eval_seed_base") == rep.get("eval_seed_base") == 9300000,
        "mismatches": mismatches,
        "verdict": "PASS" if (not mismatches and checked == 35
                              and orig.get("eval_seed_base") == rep.get("eval_seed_base") == 9300000)
                   else "BLOCKED_REPLAY_MISMATCH",
    }

    if equiv["verdict"] != "PASS":
        OUT.write_text(json.dumps({
            "gate": "CONFIRMATION_A_GATE3",
            "verdict": "BLOCKED_REPLAY_MISMATCH",
            "REPLAY_EQUIVALENCE": equiv,
            "meaning": ("deterministic replay did not reproduce the original "
                        "aggregates. Do NOT score gate3, do NOT widen the "
                        "tolerance. Either the persistence patch was not "
                        "write-only or an input differed from what was recorded; "
                        "diagnose before using any number from this replay."),
        }, indent=2), encoding="utf-8")
        print("BLOCKED_REPLAY_MISMATCH", json.dumps(mismatches[:5], indent=2),
              file=sys.stderr)
        return 3

    # ---- 2. Gate 3 from recovered rows ----------------------------------
    rows = list(csv.DictReader(open(ROWS, encoding="utf-8")))
    ha: dict = defaultdict(lambda: defaultdict(list))
    hb: dict = defaultdict(lambda: defaultdict(list))
    for r in rows:
        tgt = ha if int(r["eval_seed"]) % 2 == 0 else hb
        tgt[r["policy_id"]][r["opponent"]].append(int(r["win"]))

    all_pol = list(co.keys())
    incumbents = [p for p in all_pol if p not in (S_OP7, S_OP8)]

    def selective(pols):
        chosen = {o: max(pols, key=lambda p: _wr(ha, p, o)) for o in OPPONENTS}
        val = sum(_wr(hb, chosen[o], o) for o in OPPONENTS) / len(OPPONENTS)
        return val, chosen

    # AMENDMENT_1: the fixed comparator's argmax is chosen on half A too.
    fixed_pol = max(all_pol, key=lambda p: sum(_wr(ha, p, o) for o in OPPONENTS))
    v_fixed = sum(_wr(hb, fixed_pol, o) for o in OPPONENTS) / len(OPPONENTS)
    v_sel, chosen = selective(all_pol)
    v_inc, chosen_inc = selective(incumbents)
    delta = v_sel - v_fixed

    rng = random.Random(20260815)
    boots = []
    for _ in range(10000):
        diffs = []
        for o in OPPONENTS:
            sp, fp = hb[chosen[o]][o], hb[fixed_pol][o]
            if not sp or not fp:
                continue
            k = min(len(sp), len(fp))
            idx = [rng.randrange(k) for _ in range(k)]
            diffs.append(sum(sp[i] - fp[i] for i in idx) / k)
        if diffs:
            boots.append(sum(diffs) / len(diffs))
    boots.sort()
    lcb = boots[int(0.05 * len(boots))] if boots else float("nan")

    specialists_selected = sorted({v for v in chosen.values() if v in (S_OP7, S_OP8)})
    result = {
        "gate": "CONFIRMATION_A_GATE3",
        "protocol": "artifacts/vgc_specialists/CONFIRMATION_A_RECOVERY_FROZEN.json",
        "estimator": "SPLIT_HALF_ORACLE; both the adaptive mapping and V_fixed selected on half A, scored on half B (AMENDMENT_1)",
        "REPLAY_EQUIVALENCE": equiv,
        "half_sizes": {"half_A_by_even_eval_seed": 30, "half_B_by_odd_eval_seed": 30},
        "gate3": {
            "V_selective_halfB": round(v_sel, 4),
            "V_fixed_halfB": round(v_fixed, 4),
            "fixed_policy_selected_on_halfA": fixed_pol,
            "delta_pool": round(delta, 4),
            "bootstrap_LCB95": round(lcb, 4),
            "floor": 0.05,
            "verdict": "PASS" if (delta >= 0.05 and lcb > 0) else "FAIL",
        },
        "SELECTED_POLICY_PER_OPPONENT": chosen,
        "SPECIALIST_INCREMENTAL_VALUE": {
            "V_all5_halfB": round(v_sel, 4),
            "V_incumbents_only_halfB": round(v_inc, 4),
            "incremental": round(v_sel - v_inc, 4),
            "incumbents_only_selection_halfA": chosen_inc,
            "specialists_ever_selected": bool(specialists_selected),
            "which_specialists_selected": specialists_selected,
            "interpretation_rule": (
                "if no specialist is selected, the specialists contributed no "
                "repertoire value regardless of whether gate3 passed"),
        },
        "gate1_and_gate2_not_recomputed": {
            "source": "artifacts/vgc_specialists/CONFIRMATION_A_RESULT.json",
            "gate1": "FAIL", "gate2": "FAIL",
            "why": "both need only per-cell proportions, which were never missing",
        },
    }
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"REPLAY_EQUIVALENCE": equiv["verdict"],
                      "gate3": result["gate3"],
                      "SELECTED_POLICY_PER_OPPONENT": chosen,
                      "specialists_ever_selected":
                          result["SPECIALIST_INCREMENTAL_VALUE"]["specialists_ever_selected"]},
                     indent=2))
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
