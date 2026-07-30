#!/usr/bin/env python3
"""K=2 LRO step-5: paired cross-evaluation of both specialist families on
both frozen contexts.

Reuses the project's established rollout path
(``plot/eval_rollout.py::run_eval_episodes``) -- no new rollout logic.

Frozen contexts (docs/research-progress-tracker.md, 2026-07-29):
    C_RUSH  = OP11_ADAPTIVE_EXPLOITER | map_b_split_lane
    C_SPLIT = OP9_SPLIT_LANE_FEINT    | map_b_split_lane

Evaluation invariants held fixed for every cell:
    map_b_split_lane, 2v2, max_decision_steps=240 (matches the episode
    length both contexts were confirmed at), no domain randomization,
    deterministic action selection, n_envs=1.

Paired seeds: ``run_eval_episodes``'s legacy path uses
``seed = latent_eval_seed + episode_index``, which is independent of the
policy and checkpoint -- so passing one fixed base per context gives the
same 32 episode seeds to every policy, as required for paired CIs.

Eval seed blocks are disjoint from all training seeds (901xxx/902xxx),
all context-confirmation blocks (620001/681001/691001/701001/711001),
every earlier development/held-out block, and the pilots (821001/831001).

CHECKPOINT NOTE: the predeclared trajectory points were 250k/500k/1M, but
training saved every 100k, so no 250k checkpoint exists. Rather than
approximate one, this script brackets it with the real 200k and 300k
checkpoints. 1M remains the sole formal gate.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "plot"))

from eval_rollout import run_eval_episodes  # noqa: E402

CONTEXTS = {
    "C_RUSH": {"opponent": "OP11_ADAPTIVE_EXPLOITER", "seed_base": 1_010_001},
    "C_SPLIT": {"opponent": "OP9_SPLIT_LANE_FEINT", "seed_base": 1_020_001},
}
MAP = "map_b_split_lane"
MAX_DECISION_STEPS = 240
AGENTS = 2

FAMILIES = {
    "piR": {"dir": "checkpoints/k2v2_piR", "stem": "k2v2_piR_op11_mapb_s{seed}_2v2",
            "seeds": [901001, 901002, 901003]},
    "piS": {"dir": "checkpoints/k2v2_piS", "stem": "k2v2_piS_op9_mapb_s{seed}_2v2",
            "seeds": [902001, 902002, 902003]},
}


def ckpt_path(family: str, seed: int, step: int) -> Path:
    spec = FAMILIES[family]
    stem = spec["stem"].format(seed=seed)
    return PROJECT_ROOT / spec["dir"] / f"ckpt_{stem}_{step}.zip"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=32)
    p.add_argument("--checkpoints", type=int, nargs="+", default=[1_000_000, 500_000, 300_000, 200_000])
    p.add_argument("--out-dir", default="artifacts/k2v2_specialist_cross_eval")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "episode_rows.csv"
    manifest_path = out_dir / "manifest.json"

    cells = []
    for step in args.checkpoints:
        for family, spec in FAMILIES.items():
            for seed in spec["seeds"]:
                for ctx_name in CONTEXTS:
                    cells.append((step, family, seed, ctx_name))

    missing = sorted({str(ckpt_path(f, s, st)) for st, f, s, _ in cells
                      if not ckpt_path(f, s, st).exists()})
    if missing:
        print("[abort] missing checkpoints:", file=sys.stderr)
        for m in missing:
            print("   ", m, file=sys.stderr)
        return 1

    prev = {}
    if manifest_path.exists():
        try:
            prev = json.loads(manifest_path.read_text())
        except json.JSONDecodeError:
            prev = {}
    prior_ckpts = list(prev.get("checkpoints") or [])
    merged_ckpts = sorted(set(int(x) for x in prior_ckpts) | set(int(x) for x in args.checkpoints))

    manifest = {
        "experiment": "k2v2_specialist_cross_eval",
        "created_utc": prev.get("created_utc") or datetime.now(timezone.utc).isoformat(),
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "contexts": {k: f"{v['opponent']}|{MAP}" for k, v in CONTEXTS.items()},
        "eval_seed_blocks": {
            k: [v["seed_base"], v["seed_base"] + args.episodes - 1] for k, v in CONTEXTS.items()
        },
        "episodes_per_cell": args.episodes,
        "checkpoints": merged_ckpts,
        "checkpoints_this_launch": list(args.checkpoints),
        "formal_gate_checkpoint": 1_000_000,
        "invariants": {
            "map": MAP, "agents": AGENTS, "max_decision_steps": MAX_DECISION_STEPS,
            "deterministic": True, "domain_randomization": False, "n_envs": 1,
        },
        "families": {k: v["seeds"] for k, v in FAMILIES.items()},
        "checkpoint_note": (
            "Predeclared trajectory points were 250k/500k/1M; training saved every "
            "100k so no 250k checkpoint exists. 250k is bracketed by real 200k and "
            "300k checkpoints instead of being approximated. 1M is the sole formal gate; "
            "500k/300k/200k are trajectory diagnostics only."
        ),
        "n_cells_this_launch": len(cells),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[manifest] {manifest_path}  ({len(cells)} cells x {args.episodes} eps)")

    if args.dry_run:
        for st, f, s, c in cells:
            print(f"  [dry] step={st:>8} {f} s{s} {c}")
        return 0

    from game_field_gpu import GPUCTFVecEnv, GPUFieldConfig

    fieldnames = ["checkpoint_step", "family", "train_seed", "context", "opponent", "map",
                  "episode_index", "episode_seed", "success", "blue_score", "red_score",
                  "win_margin", "steps", "return"]
    wrote_header = rows_path.exists()
    fh = open(rows_path, "a", newline="")
    writer = csv.DictWriter(fh, fieldnames=fieldnames)
    if not wrote_header:
        writer.writeheader()

    try:
        for idx, (step, family, seed, ctx_name) in enumerate(cells, 1):
            ctx = CONTEXTS[ctx_name]
            path = ckpt_path(family, seed, step)
            cfg = GPUFieldConfig(
                n_envs=1, max_blue_agents=AGENTS, max_red_agents=AGENTS,
                map_set="train", map_layout=MAP,
                max_decision_steps=MAX_DECISION_STEPS,
                aquaticus_profile=True, rules_profile="OURS",
                device=args.device, seed=int(ctx["seed_base"]),
            )
            env = GPUCTFVecEnv(cfg)
            try:
                print(f"[{idx}/{len(cells)}] step={step} {family} s{seed} -> {ctx_name} "
                      f"({ctx['opponent']})", flush=True)
                eps = run_eval_episodes(
                    str(path), env, int(args.episodes), args.device, ctx["opponent"],
                    deterministic=True, coordination_metrics=False,
                    latent_eval_seed=int(ctx["seed_base"]),
                    progress_every=0,
                )
            finally:
                env.close()
            for ep_i, ep in enumerate(eps):
                bs = int(ep.get("blue_score", 0) or 0)
                rs = int(ep.get("red_score", 0) or 0)
                writer.writerow({
                    "checkpoint_step": step, "family": family, "train_seed": seed,
                    "context": ctx_name, "opponent": ctx["opponent"], "map": MAP,
                    "episode_index": ep_i,
                    "episode_seed": int(ctx["seed_base"]) + ep_i,
                    "success": int(bool(ep.get("success", bs > rs))),
                    "blue_score": bs, "red_score": rs, "win_margin": bs - rs,
                    "steps": int(ep.get("steps", 0) or 0),
                    "return": float(ep.get("return", 0.0) or 0.0),
                })
            fh.flush()
    finally:
        fh.close()

    print(f"\n[done] rows -> {rows_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
