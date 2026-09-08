"""Retrospective decision-boundary audit of the FIT/CALIB teacher bank.

Phase 0 established that this environment is temporally abstracted: a macro command is held
for several sim steps (macro_commit_*_ticks, 2-4 depending on the macro) and the environment
accepts a new command only when blue_commit_ticks_left <= 0, per agent.

The observation mask ENCODES that commitment. _observations.py collapses a committed agent's
mask to exactly one legal macro and one legal target:

    committed = (blue_commit_ticks_left > 0) & blue_alive
    mask[macro]  <- one-hot(commit_macro)   where committed
    mask[target] <- one-hot(commit_target)  where committed

so the boundary predicate is recoverable from the STORED bank alone, with no environment and
no GPU:

    d_i = NOT (macro slice is one-hot AND target slice is one-hot)

This is descriptive only. It computes exactly the quantities the PI prespecified and no
others. It does not touch EVAL, does not reinterpret any frozen result, and has no verdict,
threshold or gate that could be shopped. V1-V4 were evaluated through the real environment
and stand exactly as recorded.

Mask layout (validated by teacher-action legality, 145/145):
    [agent0_macro(5), agent0_target(50), agent1_macro(5), agent1_target(50)]

Run:  python experiments/ccp_boundary_audit.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.diagnose_oracle_gated_k2_fit_calib as DG

OUT = ROOT / "artifacts" / "strategic_demand" / "sppo" / "CCP_BOUNDARY_AUDIT.json"

N_MACROS, N_TARGETS = 5, 50
SPLITS = {
    "CALIB": (10_700_097, 10_700_128),
    "BANK_10700": (10_700_001, 10_700_096),
    "BANK_11000": (11_000_001, 11_000_320),
}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _slices(agent: int) -> tuple[slice, slice]:
    base = agent * (N_MACROS + N_TARGETS)
    return slice(base, base + N_MACROS), slice(base + N_MACROS, base + N_MACROS + N_TARGETS)


def audit(split: dict) -> dict:
    m = np.asarray(split["mask"])
    pa, pb = np.asarray(split["pi_a"]), np.asarray(split["pi_b"])
    n = len(m)
    if m.shape[1] != 2 * (N_MACROS + N_TARGETS):
        raise SystemExit(f"REFUSING: mask width {m.shape[1]}, expected {2*(N_MACROS+N_TARGETS)}")

    # legality check -- the decode must be right before anything else is believed
    legal = np.ones(n, dtype=bool)
    for agent in range(2):
        ms, ts = _slices(agent)
        for teacher in (pa, pb):
            mac, tar = teacher[:, agent * 2], teacher[:, agent * 2 + 1]
            legal &= m[np.arange(n), ms.start + mac] > 0
            legal &= m[np.arange(n), ts.start + tar] > 0
    if not legal.all():
        raise SystemExit(f"REFUSING: mask decode wrong -- {int((~legal).sum())} teacher "
                         "actions are illegal under the assumed layout")

    free = np.zeros((n, 2), dtype=bool)
    for agent in range(2):
        ms, ts = _slices(agent)
        committed = (m[:, ms].sum(1) == 1) & (m[:, ts].sum(1) == 1)
        free[:, agent] = ~committed

    any_free = free.any(1)
    both_free = free.all(1)
    none_free = ~any_free

    # teacher disagreement, whole joint action and per agent
    dis_joint = (pa != pb).any(1)
    dis_agent = np.stack([(pa[:, i * 2:i * 2 + 2] != pb[:, i * 2:i * 2 + 2]).any(1)
                          for i in range(2)], axis=1)

    def rate(mask_: np.ndarray, values: np.ndarray) -> dict:
        k = int(mask_.sum())
        return {"n": k, "rate": (float(values[mask_].mean()) if k else None)}

    # structural prediction: a COMMITTED agent's action is forced, so the teachers cannot
    # disagree about it. Checked, not assumed.
    forced_violations = int((dis_agent & ~free).sum())

    return {
        "n_states": n,
        "mask_decode_validated": "all teacher actions legal under the assumed layout",
        "boundary_rates": {
            "P_agent0_free": float(free[:, 0].mean()),
            "P_agent1_free": float(free[:, 1].mean()),
            "P_at_least_one_free": float(any_free.mean()),
            "P_both_free": float(both_free.mean()),
            "P_both_mid_hold": float(none_free.mean()),
        },
        "teacher_disagreement": {
            "overall": float(dis_joint.mean()),
            "given_at_least_one_free": rate(any_free, dis_joint),
            "given_both_free": rate(both_free, dis_joint),
            "given_both_mid_hold": rate(none_free, dis_joint),
        },
        "per_agent_disagreement": {
            f"agent{i}": {
                "overall": float(dis_agent[:, i].mean()),
                "given_this_agent_free": rate(free[:, i], dis_agent[:, i]),
                "given_this_agent_mid_hold": rate(~free[:, i], dis_agent[:, i]),
            } for i in range(2)
        },
        "structural_check": {
            "claim": "a committed agent's mask is one-hot, so its action is forced and the "
                     "teachers cannot disagree about it",
            "violations": forced_violations,
            "holds": forced_violations == 0,
        },
    }


def main() -> int:
    if OUT.is_file():
        raise SystemExit(f"REFUSING: {OUT} exists")
    print(f"CCP RETROSPECTIVE BOUNDARY AUDIT  {_now()}\n")

    results, pooled = {}, []
    for name, (lo, hi) in SPLITS.items():
        try:
            split = DG.load_split(lo, hi)
        except Exception as exc:
            results[name] = {"unavailable": f"{type(exc).__name__}: {exc}"}
            print(f"  {name:12s} unavailable: {exc}")
            continue
        r = audit(split)
        results[name] = r
        pooled.append(split)
        b, t = r["boundary_rates"], r["teacher_disagreement"]
        print(f"  {name:12s} n={r['n_states']:5d}   "
              f"one-free {b['P_at_least_one_free']:.3f}  both-free {b['P_both_free']:.3f}  "
              f"both-hold {b['P_both_mid_hold']:.3f}")
        print(f"  {'':12s} teacher disagreement: overall {t['overall']:.4f}   "
              f"| >=1 free {t['given_at_least_one_free']['rate']}   "
              f"| both mid-hold {t['given_both_mid_hold']['rate']}")
        print(f"  {'':12s} forced-action violations: {r['structural_check']['violations']}\n")

    combined = None
    if pooled:
        merged = {k: np.concatenate([np.asarray(s[k]) for s in pooled])
                  for k in ("mask", "pi_a", "pi_b")}
        combined = audit(merged)
        b, t = combined["boundary_rates"], combined["teacher_disagreement"]
        print(f"  {'POOLED':12s} n={combined['n_states']:5d}   "
              f"one-free {b['P_at_least_one_free']:.3f}  both-hold {b['P_both_mid_hold']:.3f}")
        print(f"  {'':12s} teacher disagreement: overall {t['overall']:.4f}   "
              f"| >=1 free {t['given_at_least_one_free']['rate']}   "
              f"| both mid-hold {t['given_both_mid_hold']['rate']}")

    OUT.write_text(json.dumps({
        "record": "CCP retrospective decision-boundary audit of the teacher bank",
        "status": "FROZEN_RESULT", "utc": _now(),
        "program": "CAUSAL_CROSSOVER_PROGRAM_OPENING.json",
        "DESCRIPTIVE_ONLY": ("No verdict, threshold or gate. Computes exactly the quantities "
                             "the PI prespecified. Does not reinterpret any frozen result: "
                             "V1-V4 were evaluated through the real environment and stand as "
                             "recorded."),
        "boundary_predicate": {
            "runtime_source": "_step.py: new_commit = blue_commit_ticks_left <= 0, per agent",
            "recovered_from_the_bank_via": ("_observations.py collapses a committed agent's mask "
                                            "to one-hot(commit_macro) and one-hot(commit_target), "
                                            "so d_i = NOT(macro one-hot AND target one-hot)"),
            "mask_layout": "[a0_macro(5), a0_target(50), a1_macro(5), a1_target(50)]",
            "commit_durations": {"go_to": 4, "grab": 3, "get_flag": 4, "place": 2, "go_home": 4},
        },
        "splits": results,
        "pooled": combined,
        "EVAL_touched": False,
    }, indent=2), encoding="utf-8")
    print(f"\n  -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
