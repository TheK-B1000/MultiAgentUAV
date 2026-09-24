r"""Where do ORIGINAL and PROJECTED actually diverge? (4v4, pole A)

Execution-path trace, per the standing rule: trace first, repair second.

Two competing hypotheses for why delta_A collapsed under projection
(0.417 -> 0.000) in REPAIRED_GO_TO_H1_PROJECTED:

  H1  COMMITMENT BLOCKING. The teacher wants to switch macro this tick but the
      engine's commitment holds the previous one, so the executed behaviour is
      stale. Predicted signature: many ticks where adapted_macro != committed
      macro while commit_ticks_left > 0, concentrated in GUARD.

  H2  CARRIER SUBSTITUTION. On the ORIGINAL path a carrying agent receives
      `_carrier_evasion_target` (multi-threat tangent evasion toward home). On
      the PROJECTED path `_build_targets_from_action` forces a carrying agent
      to EXACT own_flag_home regardless of macro. If those differ materially,
      projection is not reproducing the teacher for carriers -- it substitutes a
      different carrier policy. Predicted signature: large teacher-vs-executed
      residual on carrying ticks, concentrated in BREACH (20.8% carrying at 4v4).

The sealed evidence favours H2: GUARD@A barely moved (0.708 -> 0.750) while
BREACH@A improved sharply (0.292 -> 0.750, CI excludes zero). H1 alone cannot
explain an IMPROVEMENT in BREACH.

    python -m experiments.trace_projection_divergence --scale 4 --n-seeds 8
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
LABEL = "PROJECTION_DIVERGENCE_TRACE"

FIELDS = ["scale", "strategy", "pole", "seed", "tick", "agent", "is_defender",
          "carrying", "tagged", "teacher_tx", "teacher_ty",
          "adapt_category", "adapt_macro", "adapt_residual",
          "committed_macro", "commit_ticks_left", "at_boundary",
          "wants_switch", "switch_blocked", "exec_tx", "exec_ty", "exec_residual"]


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def trace_cell(scale, strategy, pole, seeds, device, go_to_horizon=None):
    import experiments.strategic_demand_searcher as S0
    from experiments.eval_projected_teacher_oracle import _make_env
    from experiments.teacher_action_adapter import adapt, executed_target
    from experiments.tqdm_loop import set_postfix, tqdm_iter

    style = S0.GUARD if strategy == "GUARD" else S0.BREACH
    n_def = (scale + 1) // 2
    def_lo = scale - n_def
    rows = []
    bar = tqdm_iter(seeds, desc=f"trace {scale}v{scale} {strategy}@{pole}", unit="seed")
    for seed in bar:
        set_postfix(bar, f"rows={len(rows)}")
        env, core, S = _make_env(scale, pole, style, seed, device,
                                 go_to_horizon=go_to_horizon)
        try:
            core.blue_scripted = False          # PROJECTED arm
            W = core._macro_targets.detach().cpu().numpy()
            action = np.zeros((scale, 2), dtype=np.int64)
            have = False
            for tick in range(S.MAX_STEPS):
                nc = (core.blue_commit_ticks_left[0] <= 0).detach().cpu().numpy()
                cm = core.blue_commit_macro[0].detach().cpu().numpy()
                ctl = core.blue_commit_ticks_left[0].detach().cpu().numpy()
                btx, bty = core._get_scripted_targets("blue")
                for i in range(scale):
                    tx, ty = float(btx[0, i]), float(bty[0, i])
                    a = adapt(core, tx, ty, i, W)
                    at_boundary = bool(nc[i]) or not have
                    # what the teacher WANTS this tick vs what is committed
                    wants = a.macro if a.macro is not None else 0
                    wants_switch = bool(wants != int(cm[i]))
                    switch_blocked = bool(wants_switch and not at_boundary)
                    # what the engine WOULD actually execute under the currently
                    # committed action (this is the real behaviour this tick)
                    ex, ey = executed_target(core, int(cm[i]),
                                             int(core.blue_commit_target[0, i]), i)
                    rows.append({
                        "scale": f"{scale}v{scale}", "strategy": strategy, "pole": pole,
                        "seed": seed, "tick": tick, "agent": i,
                        "is_defender": int(i >= def_lo),
                        "carrying": int(bool(core.blue_carrying[0, i])),
                        "tagged": int(bool(core.blue_tagged[0, i])),
                        "teacher_tx": round(tx, 4), "teacher_ty": round(ty, 4),
                        "adapt_category": a.category, "adapt_macro": wants,
                        "adapt_residual": round(a.residual, 4),
                        "committed_macro": int(cm[i]),
                        "commit_ticks_left": int(ctl[i]),
                        "at_boundary": int(at_boundary),
                        "wants_switch": int(wants_switch),
                        "switch_blocked": int(switch_blocked),
                        "exec_tx": round(ex, 4), "exec_ty": round(ey, 4),
                        "exec_residual": round(float(np.hypot(tx - ex, ty - ey)), 4),
                    })
                    if at_boundary:
                        if a.uses_waypoint:
                            action[i] = (a.macro, a.target_idx)
                        elif a.macro is not None:
                            action[i] = (a.macro, 0)
                        else:
                            action[i] = (0, 0)
                have = True
                env.step_async(action.reshape(-1))
                _o, _r, done, _i = env.step_wait()
                if bool(np.asarray(done).any()):
                    break
        finally:
            env.close()
    return rows


def summarize(rows):
    MACRO = {0: "GO_TO", 1: "GRAB_MINE", 2: "GET_FLAG", 3: "PLACE_MINE", 4: "GO_HOME"}
    by = defaultdict(list)
    for r in rows:
        by[(r["strategy"], r["pole"])].append(r)
    out = {}
    for key, rs in by.items():
        n = len(rs)
        blocked = [r for r in rs if r["switch_blocked"]]
        carrying = [r for r in rs if r["carrying"]]
        noncarry = [r for r in rs if not r["carrying"] and not r["tagged"]]
        trans = Counter((MACRO.get(r["committed_macro"], "?"),
                         MACRO.get(r["adapt_macro"], "?")) for r in blocked)
        er_carry = np.array([r["exec_residual"] for r in carrying]) if carrying else np.array([])
        er_non = np.array([r["exec_residual"] for r in noncarry]) if noncarry else np.array([])
        out[f"{key[0]}@{key[1]}"] = {
            "n_agent_ticks": n,
            "H1_commitment_blocking": {
                "frac_ticks_switch_blocked": round(len(blocked) / max(1, n), 4),
                "top_blocked_transitions": [
                    {"from": a, "to": b, "n": c, "frac_of_all_ticks": round(c / max(1, n), 4)}
                    for (a, b), c in trans.most_common(6)],
            },
            "H2_carrier_substitution": {
                "frac_ticks_carrying": round(len(carrying) / max(1, n), 4),
                "exec_residual_on_CARRYING_ticks": {
                    "n": int(er_carry.size),
                    "median": round(float(np.median(er_carry)), 4) if er_carry.size else None,
                    "p90": round(float(np.percentile(er_carry, 90)), 4) if er_carry.size else None,
                    "max": round(float(er_carry.max()), 4) if er_carry.size else None,
                    "frac_gt_1_cell": round(float((er_carry > 1.0).mean()), 4) if er_carry.size else None,
                },
                "exec_residual_on_NONCARRYING_untagged_ticks": {
                    "n": int(er_non.size),
                    "median": round(float(np.median(er_non)), 4) if er_non.size else None,
                    "p90": round(float(np.percentile(er_non, 90)), 4) if er_non.size else None,
                    "frac_gt_1_cell": round(float((er_non > 1.0).mean()), 4) if er_non.size else None,
                },
            },
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--n-seeds", type=int, default=8)
    ap.add_argument("--pole", default="A")
    ap.add_argument("--go-to-horizon", type=int, default=None)
    a = ap.parse_args()
    seeds = list(range(17600001, 17600001 + a.n_seeds))

    print(f"{LABEL}  {_now()}")
    print(f"  {a.scale}v{a.scale} pole{a.pole}, seeds {seeds[0]}..{seeds[-1]}, "
          f"go_to_horizon={a.go_to_horizon}")
    print(f"  H1 = commitment blocks an intended macro switch")
    print(f"  H2 = carrier target substituted (evasion -> exact home)\n", flush=True)

    rows = []
    for strat in ("GUARD", "BREACH"):
        rows += trace_cell(a.scale, strat, a.pole, seeds, a.device, a.go_to_horizon)

    ROWS = SD / f"{LABEL.lower()}_rows.csv"
    with ROWS.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS); w.writeheader()
        for r in rows:
            w.writerow(r);
        fh.flush(); os.fsync(fh.fileno())

    summ = summarize(rows)
    print(json.dumps(summ, indent=2))
    out = SD / f"{LABEL}_RESULT.json"
    out.write_text(json.dumps({
        "record": f"{LABEL} execution-path divergence trace",
        "status": "FROZEN_RESULT", "utc": _now(), "arm": "DIAGNOSTIC",
        "confirmatory": False, "study_class": "MECHANISTIC_FOLLOW_UP",
        "scale": f"{a.scale}v{a.scale}", "pole": a.pole, "seeds": [seeds[0], seeds[-1]],
        "go_to_horizon": a.go_to_horizon,
        "hypotheses": {"H1": "commitment blocks an intended macro switch",
                       "H2": "carrier target substituted: evasion -> exact home"},
        "summary": summ,
    }, indent=2), encoding="utf-8")
    print(f"\n  -> {out}\n  -> {ROWS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
