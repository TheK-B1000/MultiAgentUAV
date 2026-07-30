#!/usr/bin/env python3
"""G0 learned-incumbent weakness sweep (NEXT after cancelled 300k replication).

Promote the three completed 1M πR policies to frozen incumbent family G0 and
search for contexts that defeat the *learned* generalist — not scripted probes.

    G0 = {s901001, s901002, s901003}

Discovery design (locked in research-progress-tracker, 2026-07-30):

    Opponents:  OP6–OP12 only
    Map:        map_a  (recorded in every row)
    Policies:   all three G0 seeds
    Horizon:    240
    Eval:       deterministic, no DR, n_envs=1
    Budget:     32 fresh episodes per (policy × opponent) cell

Select a context only if it challenges the entire incumbent family, ideally:
all three G0 policies have negative mean margins, family-level upper CI < 0,
and low saturation. Then train the next response oracle against that weakness.

This script is the executable scaffold. Default is --dry-run. Do not launch
until the discovery behavior audit has released the GPU and the operator
passes --force-run.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "plot"))

G0_SEEDS = [901001, 901002, 901003]
OPPONENTS = [
    "OP6_IMMEDIATE_DUAL_RUSH",
    "OP7_DEEP_FORTRESS",
    "OP8_PROTECTED_CARRIER_ESCORT",
    "OP9_SPLIT_LANE_FEINT",
    "OP10_AGGRESSIVE_INTERCEPTOR",
    "OP11_ADAPTIVE_EXPLOITER",
    "OP12_LATE_CONVERTER",
]
MAP = "map_a"
MAX_DECISION_STEPS = 240
AGENTS = 2
EPISODES = 32
# Fresh eval base, disjoint from all prior blocks.
SEED_BASE = 1_210_001


def ckpt_for(seed: int) -> Path:
    final = PROJECT_ROOT / "checkpoints" / "k2v2_piR" / f"final_k2v2_piR_op11_mapb_s{seed}_2v2.zip"
    if final.exists():
        return final
    return PROJECT_ROOT / "checkpoints" / "k2v2_piR" / f"ckpt_k2v2_piR_op11_mapb_s{seed}_2v2_1000000.zip"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=EPISODES)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out-dir", default="artifacts/g0_weakness_sweep")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force-run", action="store_true",
                   help="Actually run the sweep (default is dry-run / manifest only).")
    args = p.parse_args()

    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cells = []
    missing = []
    for seed in G0_SEEDS:
        path = ckpt_for(seed)
        if not path.exists():
            missing.append(str(path))
        for i, opp in enumerate(OPPONENTS):
            cells.append({
                "g0_seed": seed,
                "checkpoint": str(path),
                "opponent": opp,
                "map": MAP,
                "seed_base": SEED_BASE + i * 1000 + (seed % 1000),
                "episodes": int(args.episodes),
            })

    manifest = {
        "experiment": "g0_learned_incumbent_weakness_sweep",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PREDECLARED" if not args.force_run else "RUNNING",
        "G0_seeds": G0_SEEDS,
        "opponents": OPPONENTS,
        "map": MAP,
        "max_decision_steps": MAX_DECISION_STEPS,
        "agents": AGENTS,
        "episodes_per_cell": int(args.episodes),
        "n_cells": len(cells),
        "selection_rule": {
            "prefer": [
                "all three G0 policies have negative mean margins",
                "family-level upper CI < 0",
                "low saturation",
            ],
            "purpose": "train next response oracle against actual learned weakness",
        },
        "precedent": {
            "k2v3_300k_replication": "CANCELLED_PRELAUNCH",
            "k2v2_1m_formal": "FAIL — piR dominant generalist",
        },
        "cells": cells,
        "missing_checkpoints": missing,
    }
    man_path = out_dir / "manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[manifest] {man_path}  cells={len(cells)}")
    if missing:
        print("[abort] missing G0 checkpoints:", file=sys.stderr)
        for m in missing:
            print("   ", m, file=sys.stderr)
        return 1

    if args.dry_run or not args.force_run:
        for c in cells:
            print(f"  [dry] G0 s{c['g0_seed']} vs {c['opponent']} "
                  f"seeds {c['seed_base']}..{c['seed_base']+c['episodes']-1}")
        print("[dry] pass --force-run to execute (after discovery audit frees GPU).")
        return 0

    from game_field_gpu import GPUCTFVecEnv, GPUFieldConfig
    from eval_rollout import run_eval_episodes

    rows_path = out_dir / "episode_rows.csv"
    fields = ["g0_seed", "opponent", "map", "episode_index", "episode_seed",
              "success", "blue_score", "red_score", "win_margin", "steps", "return"]
    wrote = rows_path.exists()
    with open(rows_path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        if not wrote:
            w.writeheader()
        for idx, cell in enumerate(cells, 1):
            print(f"[{idx}/{len(cells)}] G0 s{cell['g0_seed']} -> {cell['opponent']}",
                  flush=True)
            cfg = GPUFieldConfig(
                n_envs=1, max_blue_agents=AGENTS, max_red_agents=AGENTS,
                map_set="train", map_layout=MAP,
                max_decision_steps=MAX_DECISION_STEPS,
                aquaticus_profile=True, rules_profile="OURS",
                device=args.device, seed=int(cell["seed_base"]),
            )
            env = GPUCTFVecEnv(cfg)
            try:
                eps = run_eval_episodes(
                    cell["checkpoint"], env, int(cell["episodes"]), args.device,
                    cell["opponent"], deterministic=True, coordination_metrics=False,
                    latent_eval_seed=int(cell["seed_base"]), progress_every=0,
                )
            finally:
                env.close()
            for ep_i, ep in enumerate(eps):
                bs = int(ep.get("blue_score", 0) or 0)
                rs = int(ep.get("red_score", 0) or 0)
                w.writerow({
                    "g0_seed": cell["g0_seed"], "opponent": cell["opponent"], "map": MAP,
                    "episode_index": ep_i,
                    "episode_seed": int(cell["seed_base"]) + ep_i,
                    "success": int(bool(ep.get("success", bs > rs))),
                    "blue_score": bs, "red_score": rs, "win_margin": bs - rs,
                    "steps": int(ep.get("steps", 0) or 0),
                    "return": float(ep.get("return", 0.0) or 0.0),
                })
            fh.flush()

    manifest["status"] = "COMPLETE"
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    man_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[done] {rows_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
