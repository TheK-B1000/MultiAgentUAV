"""Score the frozen specialist pilot (Confirmation A).

Implements artifacts/vgc_specialists/SPECIALIST_PILOT_FROZEN.json and its
AMENDMENT_1 exactly. Computes nothing the protocol does not authorise, and
refuses to substitute a permitted-looking estimator for a prohibited one.

Gate 1 / Gate 2 need only per-cell proportions and n, so they are computable
from the summary. Gate 3, SPECIALIST_INCREMENTAL_VALUE and the paired bootstrap
need PER-EPISODE outcomes, because the split-half rule selects on half the
episodes and scores on the other half. If the episode rows are absent those
outputs are reported BLOCKED -- never approximated with the naive same-episode
oracle, which AMENDMENT_1 prohibits.

Run:  python experiments/score_specialist_pilot.py
"""
from __future__ import annotations

import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "artifacts/vgc_specialists/SPECIALIST_PILOT_FROZEN.json"
AMEND = ROOT / "artifacts/vgc_specialists/SPECIALIST_PILOT_AMENDMENT_1.json"
SUMMARY = ROOT / "artifacts/vgc_diversity/crossplay/specialist_pilot_summary.json"
ROWS = ROOT / "artifacts/vgc_diversity/crossplay/specialist_pilot_episode_rows.csv"
OUT = ROOT / "artifacts/vgc_specialists/CONFIRMATION_A_RESULT.json"

S_OP7 = "vgc_s_op7_seed3900007"
S_OP8 = "vgc_s_op8_seed3900008"
OPPONENTS = ("OP6", "OP7", "OP8", "OP9", "OP10", "OP11", "OP12")


def wald_lcb(p_a: float, p_b: float, n: int) -> tuple[float, float]:
    """Two-proportion Wald 95% lower bound on p_a - p_b. The frozen estimator."""
    d = p_a - p_b
    se = math.sqrt(p_a * (1 - p_a) / n + p_b * (1 - p_b) / n)
    return d, d - 1.96 * se


def load_cells(summary: dict) -> dict[str, dict[str, float]]:
    """per_opponent is {op: {win_rate, n, seen}}; flatten to {op: win_rate}."""
    return {s["policy_id"]: {op: float(v["win_rate"])
                             for op, v in s["per_opponent"].items()}
            for s in summary["summaries"]}


def split_half(rows: list[dict]) -> tuple[dict, dict]:
    """Partition episodes by eval_seed parity, per AMENDMENT_1."""
    half_a: dict = defaultdict(lambda: defaultdict(list))
    half_b: dict = defaultdict(lambda: defaultdict(list))
    for r in rows:
        tgt = half_a if int(r["eval_seed"]) % 2 == 0 else half_b
        tgt[r["policy_id"]][r["opponent"]].append(int(r["win"]))
    return half_a, half_b


def _wr(d, pol, opp) -> float:
    v = d.get(pol, {}).get(opp, [])
    return sum(v) / len(v) if v else float("nan")


def selective_value(half_a, half_b, policies: list[str]) -> tuple[float, dict]:
    """Choose argmax per opponent on half A; score that frozen choice on half B."""
    chosen, scored = {}, []
    for opp in OPPONENTS:
        best = max(policies, key=lambda p: _wr(half_a, p, opp))
        chosen[opp] = best
        scored.append(_wr(half_b, best, opp))
    return sum(scored) / len(scored), chosen


def fixed_value(half_a, half_b, policies: list[str]) -> tuple[float, str]:
    """Choose the single best fixed policy on half A; score it on half B.

    AMENDMENT_1: the fixed comparator's argmax is selected on half A too, not
    on the scoring half. Selecting it on half B would give the baseline a
    winner's-curse advantage the adaptive side has been debiased of.
    """
    best = max(policies,
               key=lambda p: sum(_wr(half_a, p, o) for o in OPPONENTS) / len(OPPONENTS))
    return sum(_wr(half_b, best, o) for o in OPPONENTS) / len(OPPONENTS), best


def main() -> int:
    if not SUMMARY.is_file():
        print(f"BLOCKED: missing {SUMMARY}", file=sys.stderr)
        return 2
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))

    # ---- provenance invariants (fail closed) ----------------------------
    inv: dict[str, object] = {}
    inv["eval_seed_base_is_9300000"] = summary.get("eval_seed_base") == 9300000
    inv["not_default_block"] = summary.get("eval_seed_block_is_default") is False
    inv["episodes_per_cell_is_60"] = summary.get("episodes_per_cell") == 60
    inv["five_policies"] = len(summary.get("summaries", [])) == 5
    cells = load_cells(summary)
    inv["all_35_cells"] = all(
        len(cells[p]) == 7 for p in cells) and len(cells) == 5
    inv["specialists_present"] = S_OP7 in cells and S_OP8 in cells
    inv["checkpoint_sha_recorded"] = all(
        s.get("checkpoint_sha256") for s in summary["summaries"])
    if not all(inv.values()):
        OUT.write_text(json.dumps({
            "gate": "CONFIRMATION_A", "verdict": "BLOCKED",
            "reason": "provenance invariant failed",
            "invariants": inv,
        }, indent=2), encoding="utf-8")
        print("BLOCKED: provenance invariant failed", inv, file=sys.stderr)
        return 3

    n = int(summary["episodes_per_cell"])

    # ---- Gate 1 / Gate 2 -------------------------------------------------
    d1, l1 = wald_lcb(cells[S_OP7]["OP7"], cells[S_OP8]["OP7"], n)
    d2, l2 = wald_lcb(cells[S_OP8]["OP8"], cells[S_OP7]["OP8"], n)
    gate1 = {"claim": "S_OP7 > S_OP8 on OP7", "wr_a": cells[S_OP7]["OP7"],
             "wr_b": cells[S_OP8]["OP7"], "diff": round(d1, 4),
             "LCB95": round(l1, 4), "verdict": "PASS" if l1 > 0 else "FAIL"}
    gate2 = {"claim": "S_OP8 > S_OP7 on OP8", "wr_a": cells[S_OP8]["OP8"],
             "wr_b": cells[S_OP7]["OP8"], "diff": round(d2, 4),
             "LCB95": round(l2, 4), "verdict": "PASS" if l2 > 0 else "FAIL"}

    # descriptive: does either specialist dominate the whole board?
    dom = {"S_OP7_beats_or_ties_S_OP8_on": sum(
        1 for o in OPPONENTS if cells[S_OP7][o] >= cells[S_OP8][o]),
        "of_opponents": len(OPPONENTS)}

    result = {
        "gate": "CONFIRMATION_A_SPECIALIST_PILOT",
        "protocol": "artifacts/vgc_specialists/SPECIALIST_PILOT_FROZEN.json",
        "amendment": "artifacts/vgc_specialists/SPECIALIST_PILOT_AMENDMENT_1.json",
        "eval_seed_base": summary["eval_seed_base"],
        "episodes_per_cell": n,
        "git_commit_of_evaluation": summary.get("git_commit"),
        "provenance_invariants": inv,
        "gate1": gate1,
        "gate2": gate2,
        "descriptive_dominance": dom,
        "per_cell_win_rates": cells,
    }

    # ---- Gate 3 + attribution: require per-episode rows ------------------
    if not ROWS.is_file():
        result["gate3"] = {
            "verdict": "BLOCKED",
            "reason": "MISSING_EPISODE_ROWS",
            "detail": (
                "SPECIALIST_PILOT_FROZEN.delta_pool_definition.SPLIT_HALF_ORACLE "
                "requires per-episode outcomes keyed by eval_seed: the "
                "per-opponent argmax and the fixed comparator are both selected "
                "on half A and scored on half B. run_crossplay_eval.py built "
                "`all_rows` in memory but did not persist it, so only per-cell "
                "aggregates survive for block 9300000. Split-half cannot be "
                "computed from aggregates."),
            "prohibited_alternative": (
                "the naive same-episode oracle IS computable from aggregates and "
                "is explicitly prohibited -- it is positively biased by "
                "construction and cannot return a non-positive gap. Reporting it "
                "as delta_pool would reintroduce the exact bias the gate exists "
                "to exclude."),
            "remedy": (
                "evaluation is deterministic on a fixed seed block, so re-running "
                "Confirmation A with the now-added row persistence reproduces the "
                "identical per-cell win rates AND yields the rows. It is a replay, "
                "not a re-roll. Cost ~15h GPU. Requires human authorisation."),
        }
        result["SPECIALIST_INCREMENTAL_VALUE"] = {"verdict": "BLOCKED",
                                                  "reason": "MISSING_EPISODE_ROWS"}
        result["SELECTED_POLICY_PER_OPPONENT"] = {
            "verdict": "BLOCKED", "reason": "MISSING_EPISODE_ROWS",
            "note": ("a full-cell argmax is computable but is NOT the frozen "
                     "quantity, which is the half-A selection. Not reported, to "
                     "avoid a differently-defined number being read as the gate.")}
        result["verdict"] = "PARTIAL_GATE3_BLOCKED"
    else:
        rows = list(csv.DictReader(open(ROWS, encoding="utf-8")))
        ha, hb = split_half(rows)
        all_pol = list(cells.keys())
        incumbents = [p for p in all_pol if p not in (S_OP7, S_OP8)]

        v_sel, chosen = selective_value(ha, hb, all_pol)
        v_fix, fixed_pol = fixed_value(ha, hb, all_pol)
        delta = v_sel - v_fix
        v_sel_inc, chosen_inc = selective_value(ha, hb, incumbents)

        rng = random.Random(12345)
        boots = []
        for _ in range(10000):
            # paired resample over half-B episodes
            s = []
            for opp in OPPONENTS:
                pool = hb[chosen[opp]][opp]
                fpool = hb[fixed_pol][opp]
                if not pool or not fpool:
                    continue
                idx = [rng.randrange(len(pool)) for _ in pool]
                s.append(sum(pool[i] for i in idx) / len(idx)
                         - sum(fpool[i % len(fpool)] for i in idx) / len(idx))
            if s:
                boots.append(sum(s) / len(s))
        boots.sort()
        lcb = boots[int(0.05 * len(boots))] if boots else float("nan")

        result["gate3"] = {
            "V_selective_halfB": round(v_sel, 4),
            "V_fixed_halfB": round(v_fix, 4),
            "fixed_policy_selected_on_halfA": fixed_pol,
            "delta_pool": round(delta, 4),
            "bootstrap_LCB95": round(lcb, 4),
            "floor": 0.05,
            "verdict": "PASS" if (delta >= 0.05 and lcb > 0) else "FAIL",
        }
        result["SELECTED_POLICY_PER_OPPONENT"] = chosen
        result["SPECIALIST_INCREMENTAL_VALUE"] = {
            "V_all5_halfB": round(v_sel, 4),
            "V_incumbents_only_halfB": round(v_sel_inc, 4),
            "incremental": round(v_sel - v_sel_inc, 4),
            "incumbents_selection_halfA": chosen_inc,
            "specialists_ever_selected": any(
                v in (S_OP7, S_OP8) for v in chosen.values()),
        }
        result["verdict"] = "SCORED"

    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in
                      ("verdict", "gate1", "gate2", "descriptive_dominance")}, indent=2))
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
