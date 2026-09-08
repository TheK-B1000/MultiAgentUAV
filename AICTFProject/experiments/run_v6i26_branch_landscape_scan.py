#!/usr/bin/env python3
"""V6I26 branch-vs-branch landscape scan: does ANY existing latent branch
(z0, z1, z2, z3) beat any other across the full map x opponent surface?

Distinct from experiments/run_v6i26_strategic_landscape_scan.py, which is
the historical Stage-0 gate comparing archived V6I23/V6I24 policy CHECKPOINTS
(no z-branch awareness at all -- that scan already ran and justified
launching LRO training in the first place; its G_available_effective<0
result is old, resolved business, not new information about z0-z3).

This script asks the NEW question directly: across the canonical 3-map x
7-opponent audited surface, does the best-performing branch ever change
(real payoff crossover), using MARGINS (mean win_margin), not just win rate?

Two subcommands:

  scan     Broad, cheap survey (default 4 episodes/cell, matching the
           existing Stage-0 scan's convention) across all 3 maps x 7
           opponents x 4 branches. Reuses _collect_branch_outcomes from
           run_v6i26_usable_selector_eval.py (already validated: env/model
           built once per cell, matched-seed episodes). Reports, per cell,
           each branch's mean win_margin and which branch is best; flags
           cells where the best branch differs from the surface-wide modal
           best branch as crossover candidates.

  confirm  Cheap oracle screen (default 16 episodes/cell) for one specific
           (map, opponent, branch_a, branch_b) cell flagged by `scan`.
           Computes V_a, V_b, best_fixed, hindsight oracle, and delta_oracle
           via a paired bootstrap (reusing _paired_bootstrap) -- the exact
           promotion gate: nontrivial wins for both branches, meaningful
           oracle gain, LCB > 0, not tie-dominated.

Read-only with respect to training: loads checkpoints for inference only.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(line_buffering=True)

from experiments.run_v6i26_usable_selector_eval import _collect_branch_outcomes  # noqa: E402
from experiments.v6i26_lro_core import write_json  # noqa: E402
from gpu_env._core._bt_profiles import LRO_AUDITED_OPPONENT_POOL  # noqa: E402


def _default_branches(checkpoint_z0: str, checkpoint_z3: str) -> list[tuple[str, str, int]]:
    return [
        ("z0", checkpoint_z0, 0),
        ("z1", checkpoint_z0, 1),
        ("z2", checkpoint_z0, 2),
        ("z3", checkpoint_z3, 3),
    ]


def cmd_scan(args: argparse.Namespace) -> int:
    maps = list(args.maps)
    opponents = list(args.opponents)
    branches = _default_branches(args.checkpoint_z0, args.checkpoint_z3)
    print(f"Branches: {[b[0] for b in branches]}")
    print(f"Maps ({len(maps)}): {maps}")
    print(f"Opponents ({len(opponents)}): {opponents}")
    print(f"episodes_per_cell={args.episodes_per_cell}  "
          f"-> {len(maps) * len(opponents) * len(branches)} (cell,branch) collections planned", flush=True)

    cells: dict[tuple[str, str], dict[str, float]] = {}
    t_start = time.time()
    n_done = 0
    n_total = len(maps) * len(opponents)
    for map_idx, map_name in enumerate(maps):
        for opp_idx, opponent in enumerate(opponents):
            cell_seed = int(args.base_seed) + 1000 * (map_idx * len(opponents) + opp_idx)
            branch_means: dict[str, float] = {}
            for branch_name, checkpoint, branch_id in branches:
                outcomes = _collect_branch_outcomes(
                    checkpoint=checkpoint, fixed_z=branch_id, opponent=opponent, map_name=map_name,
                    cell_seed=cell_seed, episodes_per_context=int(args.episodes_per_cell),
                    device=args.device, max_decision_steps=int(args.max_decision_steps),
                )
                vals = np.array(list(outcomes.values()), dtype=np.float64)
                branch_means[branch_name] = float(vals.mean())
            cells[(opponent, map_name)] = branch_means
            n_done += 1
            elapsed = time.time() - t_start
            eta = (elapsed / n_done) * (n_total - n_done) if n_done > 0 else 0.0
            best = max(branch_means, key=branch_means.get)
            print(f"  [{opponent}|{map_name}] {branch_means}  best={best}  "
                  f"({n_done}/{n_total} cells, elapsed={elapsed:.0f}s, ETA={eta:.0f}s)", flush=True)

    best_per_cell = {k: max(v, key=v.get) for k, v in cells.items()}
    from collections import Counter
    modal_best, modal_count = Counter(best_per_cell.values()).most_common(1)[0]

    # A cell only counts as a crossover CANDIDATE if some branch beats the
    # modal-best branch's OWN score in that cell by a real margin -- ties
    # (common given how tie-dominated this surface already is) must not be
    # reported as crossover just because dict/max() insertion-order tie-
    # breaking happened to pick a different key.
    crossover_cells: list[tuple[str, str]] = []
    for k, v in cells.items():
        best_branch, best_val = max(v.items(), key=lambda kv: kv[1])
        margin_over_modal = best_val - v[modal_best]
        if best_branch != modal_best and margin_over_modal > float(args.margin_threshold):
            crossover_cells.append(k)

    print()
    print("=" * 72)
    print(f"Modal best branch across {len(cells)} cells: {modal_best} ({modal_count}/{len(cells)} cells)")
    print(f"Crossover candidate cells (best branch beats modal-best by margin > {args.margin_threshold}): "
          f"{len(crossover_cells)}")
    for opponent, map_name in crossover_cells:
        bm = cells[(opponent, map_name)]
        sorted_b = sorted(bm.items(), key=lambda kv: -kv[1])
        margin = sorted_b[0][1] - bm[modal_best]
        print(f"  [{opponent}|{map_name}] best={sorted_b[0][0]} ({sorted_b[0][1]:.4f}) "
              f"vs modal_best={modal_best} ({bm[modal_best]:.4f})  margin={margin:.4f}  full={bm}")

    report = {
        "protocol": "v6i26_branch_landscape_scan",
        "branches": [b[0] for b in branches],
        "maps": maps, "opponents": opponents, "episodes_per_cell": int(args.episodes_per_cell),
        "cells": {f"{k[0]}|{k[1]}": v for k, v in cells.items()},
        "best_per_cell": {f"{k[0]}|{k[1]}": v for k, v in best_per_cell.items()},
        "modal_best_branch": modal_best, "modal_best_count": modal_count, "n_cells": len(cells),
        "crossover_candidate_cells": [f"{o}|{m}" for o, m in crossover_cells],
    }
    write_json(Path(args.output), report)
    print(f"\nwrote {args.output}")
    return 0


def cmd_confirm(args: argparse.Namespace) -> int:
    from experiments.run_v6i26_usable_selector_eval import _paired_bootstrap

    branches = {b[0]: (b[1], b[2]) for b in _default_branches(args.checkpoint_z0, args.checkpoint_z3)}
    if args.branch_a not in branches or args.branch_b not in branches:
        raise ValueError(f"branches must be one of {sorted(branches)}, got {args.branch_a!r}/{args.branch_b!r}")
    ckpt_a, id_a = branches[args.branch_a]
    ckpt_b, id_b = branches[args.branch_b]

    print(f"Confirming [{args.opponent}|{args.map_name}]: {args.branch_a} (id={id_a}) vs {args.branch_b} (id={id_b}), "
          f"{args.episodes_per_cell} episodes", flush=True)
    outcomes_a = _collect_branch_outcomes(
        checkpoint=ckpt_a, fixed_z=id_a, opponent=args.opponent, map_name=args.map_name,
        cell_seed=int(args.cell_seed), episodes_per_context=int(args.episodes_per_cell),
        device=args.device, max_decision_steps=int(args.max_decision_steps),
    )
    outcomes_b = _collect_branch_outcomes(
        checkpoint=ckpt_b, fixed_z=id_b, opponent=args.opponent, map_name=args.map_name,
        cell_seed=int(args.cell_seed), episodes_per_context=int(args.episodes_per_cell),
        device=args.device, max_decision_steps=int(args.max_decision_steps),
    )
    keys = sorted(set(outcomes_a) & set(outcomes_b))
    a_arr = np.array([outcomes_a[k] for k in keys])
    b_arr = np.array([outcomes_b[k] for k in keys])
    n = len(keys)
    adv = a_arr - b_arr
    print(f"n_matched_units={n}  V_{args.branch_a}={a_arr.mean():.4f}  V_{args.branch_b}={b_arr.mean():.4f}  "
          f"frac_tie={(adv == 0).mean():.3f}  frac_{args.branch_a}_better={(adv > 0).mean():.3f}  "
          f"frac_{args.branch_b}_better={(adv < 0).mean():.3f}")

    picks_b_hindsight = b_arr > a_arr
    result = _paired_bootstrap(outcomes_z0=a_arr, outcomes_z3=b_arr, selector_picks_z3=picks_b_hindsight,
                                n_boot=int(args.bootstrap_samples), seed=int(args.seed))
    print(f"best_fixed={result['best_fixed']:.4f}  V_hindsight_oracle={result['V_hindsight_oracle']:.4f}  "
          f"delta_oracle={result['delta_oracle']:.4f}  CI95={result['delta_oracle_CI95']}  "
          f"LCB={result['delta_oracle_LCB']:.4f} > 0 ? {result['delta_oracle_LCB_gt_0']}")

    both_nontrivial = min((a_arr > 0).sum(), (b_arr > 0).sum()) >= max(3, int(0.1 * n))
    verdict = "PROMOTE" if (result["delta_oracle_LCB_gt_0"] and both_nontrivial) else "HOLD_OR_FAIL"
    print(f"nontrivial_wins_both_branches={both_nontrivial}  verdict={verdict}")

    report = {
        "protocol": "v6i26_branch_landscape_scan_confirm", "opponent": args.opponent, "map": args.map_name,
        "branch_a": args.branch_a, "branch_b": args.branch_b, "n_matched_units": n,
        f"V_{args.branch_a}": float(a_arr.mean()), f"V_{args.branch_b}": float(b_arr.mean()),
        "frac_tie": float((adv == 0).mean()), **result, "nontrivial_wins_both_branches": bool(both_nontrivial),
        "verdict": verdict,
    }
    write_json(Path(args.output), report)
    print(f"wrote {args.output}")
    return 0 if verdict == "PROMOTE" else 1


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("scan")
    ps.add_argument("--checkpoint-z0", required=True, help="Also supplies z1, z2 (frozen branches in this file)")
    ps.add_argument("--checkpoint-z3", required=True)
    ps.add_argument("--maps", nargs="+", default=["map_a_open", "map_b_split_lane", "map_b_split_lane_v2"])
    ps.add_argument("--opponents", nargs="+", default=list(LRO_AUDITED_OPPONENT_POOL))
    ps.add_argument("--episodes-per-cell", type=int, default=4)
    ps.add_argument("--margin-threshold", type=float, default=0.05,
                     help="Minimum margin over the modal-best branch's own score in a cell to count as a "
                          "crossover candidate (filters out tie-breaking artifacts, not just exact ties)")
    ps.add_argument("--base-seed", type=int, default=88001)
    ps.add_argument("--device", default="cuda")
    ps.add_argument("--max-decision-steps", type=int, default=240)
    ps.add_argument("--output", required=True)
    ps.set_defaults(func=cmd_scan)

    pc = sub.add_parser("confirm")
    pc.add_argument("--checkpoint-z0", required=True)
    pc.add_argument("--checkpoint-z3", required=True)
    pc.add_argument("--opponent", required=True)
    pc.add_argument("--map-name", required=True)
    pc.add_argument("--branch-a", required=True, choices=["z0", "z1", "z2", "z3"])
    pc.add_argument("--branch-b", required=True, choices=["z0", "z1", "z2", "z3"])
    pc.add_argument("--episodes-per-cell", type=int, default=16)
    pc.add_argument("--cell-seed", type=int, default=99001)
    pc.add_argument("--bootstrap-samples", type=int, default=2000)
    pc.add_argument("--seed", type=int, default=0)
    pc.add_argument("--device", default="cuda")
    pc.add_argument("--max-decision-steps", type=int, default=240)
    pc.add_argument("--output", required=True)
    pc.set_defaults(func=cmd_confirm)

    return p.parse_args()


def main() -> int:
    args = _parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
