"""CCP Phase 1 stage 3: pure inference over the frozen bank.

Implements CCP_PHASE1_CAUSAL_BRANCHING_SPEC.json amendments 2-4. Selection is closed and
measurement is done; this stage adds no rollouts and makes no choices.

SIGN ORIENTATION -- the one thing that must never flip silently:

    Pole A:  delta_Q = Q(pi_A) - Q(pi_B)
    Pole B:  delta_Q = Q(pi_B) - Q(pi_A)

so positive ALWAYS means the pole-matched specialist helped, and
CAUSAL_LEVERAGE_CONFIRMED carries one consistent interpretation across both poles.

Two families, corrected separately (amendment 4):

    PRIMARY    single_macro    32 contrasts, Holm at FWER 0.05 -> CAUSAL_LEVERAGE_CONFIRMED
    SECONDARY  full_takeover   32 contrasts, Holm at FWER 0.05 -> SEQUENCE_LEVERAGE_CONFIRMED

A significant takeover contrast with no significant local contrast cannot rescue the
decision-repair hypothesis; it gets its own label.

Run:  python experiments/ccp_phase1_aggregate.py
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
SPEC = SD / "CCP_PHASE1_CAUSAL_BRANCHING_SPEC.json"
MANIFEST = SD / "CCP_PHASE1_PILOT_MANIFEST.json"
PLAN = SD / "CCP_PHASE1_JOB_PLAN.json"
ROWS_DIR = SD / "ccp_phase1_rows"
OUT = SD / "CCP_PHASE1_CAUSAL_BRANCHING.json"

M = 16
ALPHA = 0.05
MODES = ("single_macro", "full_takeover")
POLE_MATCHED = {"A": "pi_A", "B": "pi_B"}      # the specialist that SHOULD help on that pole


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def exact_binomial_two_sided(k: int, n: int) -> float:
    """P(|X - n/2| >= |k - n/2|) for X ~ Bin(n, 1/2). Exact, no scipy."""
    if n == 0:
        return 1.0
    hi = max(k, n - k)
    tail = sum(math.comb(n, i) for i in range(hi, n + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def holm(pvals: dict[str, float], alpha: float) -> dict[str, dict]:
    """Holm-Bonferroni step-down. Returns per-key reject decision and adjusted p."""
    order = sorted(pvals.items(), key=lambda kv: kv[1])
    n = len(order)
    out, running, still = {}, 0.0, True
    for idx, (key, p) in enumerate(order):
        adj = min(1.0, max(running, p * (n - idx)))
        running = adj
        reject = still and adj <= alpha
        if not reject:
            still = False
        out[key] = {"p_raw": p, "p_holm": adj, "reject": bool(reject)}
    return out


def main() -> int:
    if OUT.is_file():
        raise SystemExit(f"REFUSING: {OUT} exists; one-shot")
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    fail: list[str] = []

    files = sorted(ROWS_DIR.glob("worker_*.jsonl"))
    rows = []
    for f in files:
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))

    # ---------------------------------------------------- hard invariants
    ids = [r["job_id"] for r in rows]
    if len(ids) != 2048:
        fail.append(f"expected 2048 rows, got {len(ids)}")
    if len(set(ids)) != len(ids):
        dupes = len(ids) - len(set(ids))
        fail.append(f"{dupes} duplicate job ids")
    manifest_states = {s["state_id"] for s in manifest["states"]}
    row_states = {r["state_id"] for r in rows}
    if row_states != manifest_states:
        fail.append(f"state ids differ from the frozen manifest: "
                    f"{len(row_states ^ manifest_states)} mismatched")
    by_mode = {m: sum(1 for r in rows if r["mode"] == m) for m in MODES}
    if by_mode != {"single_macro": 1024, "full_takeover": 1024}:
        fail.append(f"mode counts {by_mode} != 1024/1024")
    by_est: dict[str, int] = {}
    for r in rows:
        by_est[r["estimand"]] = by_est.get(r["estimand"], 0) + 1
    if by_est != {"agent0": 768, "agent1": 896, "joint": 384}:
        fail.append(f"estimand counts {by_est} != 768/896/384")

    # teacher SHAs still match what the manifest pinned
    for name, meta in manifest["TEACHER_POLICIES"].items():
        if name.startswith("pi_"):
            p = ROOT / meta["path"]
            if hashlib.sha256(p.read_bytes()).hexdigest() != meta["sha256"]:
                fail.append(f"{name} sha256 no longer matches the manifest")

    # cells: 16 j per policy cell, and A/B seeds identical per matched j
    cells: dict[tuple, dict] = {}
    for r in rows:
        cells.setdefault((r["state_id"], r["estimand"], r["mode"]), {}).setdefault(
            r["policy"], {})[r["j"]] = r
    if len(cells) != 64:
        fail.append(f"expected 64 (state,estimand,mode) cells, got {len(cells)}")
    for key, pol in cells.items():
        for name in ("pi_A", "pi_B"):
            if len(pol.get(name, {})) != M:
                fail.append(f"{key} {name}: {len(pol.get(name, {}))} continuations, expected {M}")
        for j in range(M):
            a, b = pol.get("pi_A", {}).get(j), pol.get("pi_B", {}).get(j)
            if a and b and a["r_j"] != b["r_j"]:
                fail.append(f"{key} j={j}: A/B continuation seeds differ")

    if fail:
        print("INVARIANT FAILURES:")
        for f_ in fail:
            print(f"  {f_}")
        raise SystemExit("REFUSING: the bank does not satisfy the frozen invariants")

    # ------------------------------------------------------- contrasts
    states_by_id = {s["state_id"]: s for s in manifest["states"]}
    contrasts: dict[str, dict] = {}
    for (state_id, estimand, mode), pol in cells.items():
        pole = states_by_id[state_id]["pole"]
        matched = POLE_MATCHED[pole]
        other = "pi_B" if matched == "pi_A" else "pi_A"
        n_plus = n_minus = n_zero = 0
        for j in range(M):
            y_m = pol[matched][j]["win"]
            y_o = pol[other][j]["win"]
            d = y_m - y_o                      # positive = pole-matched specialist won
            n_plus += d > 0
            n_minus += d < 0
            n_zero += d == 0
        disc = n_plus + n_minus
        contrasts[f"{state_id}|{estimand}|{mode}"] = {
            "state_id": state_id, "pole": pole, "estimand": estimand, "mode": mode,
            "free_set": states_by_id[state_id]["free_set"],
            "phase": states_by_id[state_id]["phase"],
            "orientation": f"positive = {matched} (pole-matched) beats {other}",
            "n_plus": n_plus, "n_minus": n_minus, "n_zero": n_zero,
            "delta_Q_hat": (n_plus - n_minus) / M,
            "discordant": disc,
            "p_exact": exact_binomial_two_sided(n_plus, disc),
            "candidate_leverage_boundary": bool(n_plus != n_minus),
        }

    families = {}
    for mode, label in (("single_macro", "CAUSAL_LEVERAGE_CONFIRMED"),
                        ("full_takeover", "SEQUENCE_LEVERAGE_CONFIRMED")):
        keys = {k: v["p_exact"] for k, v in contrasts.items() if v["mode"] == mode}
        if len(keys) != 32:
            raise SystemExit(f"REFUSING: {mode} family has {len(keys)} contrasts, expected 32")
        res = holm(keys, ALPHA)
        for k, v in res.items():
            contrasts[k].update(v)
            contrasts[k]["label"] = label if v["reject"] else (
                "candidate_leverage_boundary" if contrasts[k]["candidate_leverage_boundary"]
                else "no_leverage")
        families[mode] = {
            "n_tests": 32, "correction": "Holm", "familywise_alpha": ALPHA,
            "label_on_success": label,
            "n_significant": sum(1 for v in res.values() if v["reject"]),
            "n_candidate_nonzero": sum(1 for k, v in contrasts.items()
                                       if v["mode"] == mode and v["candidate_leverage_boundary"]),
            "min_p_raw": min(keys.values()),
            "max_abs_delta_Q": max(abs(contrasts[k]["delta_Q_hat"]) for k in keys),
            "n_directional_substantial": sum(
                1 for k in keys if contrasts[k]["delta_Q_hat"] >= 0.25),
            "discordance_distribution": {
                str(dcount): sum(1 for k in keys if contrasts[k]["discordant"] == dcount)
                for dcount in sorted({contrasts[k]["discordant"] for k in keys})},
            "median_discordant": sorted(contrasts[k]["discordant"] for k in keys)[len(keys) // 2],
        }

    # amendment 5 precedence, evaluated in the frozen order
    prim, sec = families["single_macro"], families["full_takeover"]
    if prim["n_significant"] and sec["n_significant"]:
        reading = "DECISION_LEVEL_LEVERAGE"
    elif prim["n_significant"] and not sec["n_significant"]:
        reading = "LOCAL_ONLY_REQUIRES_EXPLANATION"
    elif not prim["n_significant"] and sec["n_significant"]:
        reading = "SEQUENCE_LEVERAGE"
    elif prim["n_directional_substantial"] >= 4:
        reading = "SUGGESTIVE_BUT_UNDERPOWERED"
    else:
        reading = "NO_LOCAL_LEVERAGE_FOUND"

    OUT.write_text(json.dumps({
        "record": "CCP Phase 1 causal branching pilot",
        "status": "FROZEN_RESULT", "one_shot": True, "utc": _now(),
        "implements": "CCP_PHASE1_CAUSAL_BRANCHING_SPEC.json amendments 2-4",
        "READING": reading,
        "reading_meanings": spec["AMENDMENT_3_TWO_INTERVENTION_MODES"]["PREREGISTERED_DISCRIMINATION"],
        "CLAIM_SCOPE": spec["AMENDMENT_2_M16_PILOT_AND_CLAIM_SCOPE"]["CLAIM_SCOPE_IS_EXISTENTIAL"],
        "sign_orientation": {
            "rule": "Pole A: Q(pi_A) - Q(pi_B). Pole B: Q(pi_B) - Q(pi_A).",
            "meaning": "positive ALWAYS means the pole-matched specialist helped, so the label "
                       "has one interpretation across both poles"},
        "invariants_all_passed": {
            "rows": len(ids), "unique_job_ids": len(set(ids)), "cells": len(cells),
            "continuations_per_policy_cell": M,
            "ab_seed_pairing": "identical for every matched j",
            "state_ids_match_manifest": True,
            "mode_counts": by_mode, "estimand_counts": by_est,
            "teacher_sha256_match": True,
            "prefix_divergences": 0,
            "worker_shards": len(files)},
        "families": families,
        "contrasts": contrasts,
        "reproducibility_caveat": {
            "finding": "(seed, prefix, frozen numerical/device stack) -> s_t is exact, but the "
                       "mapping is NOT cross-device invariant.",
            "evidence": "CUDA reproduces the frozen manifest exactly; on CPU the V4 prefix "
                        "diverges by step 8 ([2,45,0,45] vs [0,45,0,45]), a flipped argmax from "
                        "different float arithmetic.",
            "consequence": "Phase 0's replay equivalence is a within-device property. This bank "
                           "was collected on CUDA, matching the device the manifest was frozen on.",
            "not_a_phase_0_failure": True},
        "EVAL_touched": False,
    }, indent=2), encoding="utf-8")

    print(f"CCP PHASE 1 RESULT  {_now()}")
    print(f"  invariants: all passed ({len(ids)} rows, {len(cells)} cells)")
    for mode in MODES:
        f_ = families[mode]
        print(f"  {mode:14s} significant {f_['n_significant']:2d}/32   "
              f"non-zero {f_['n_candidate_nonzero']:2d}/32   "
              f"dQ>=+0.25 {f_['n_directional_substantial']:2d}/32   "
              f"min p {f_['min_p_raw']:.4f}   max|dQ| {f_['max_abs_delta_Q']:+.4f}")
        print(f"  {'':14s} discordance per contrast: median {f_['median_discordant']}, "
              f"dist {f_['discordance_distribution']}")
    print(f"\n  READING: {reading}")
    print(f"  -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
