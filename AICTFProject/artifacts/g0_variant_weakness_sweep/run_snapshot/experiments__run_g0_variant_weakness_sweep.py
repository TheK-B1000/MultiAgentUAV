#!/usr/bin/env python3
"""G0 weakness sweep — SECOND pass, all 21 existing OP6-OP12 VARIANTS on map_a.

The base-key sweep is CLOSED:

    G0 competence on map_a:  CONFIRMED (0/7 opponents negative)
    Base OP6-OP12 weakness:  NONE (no context cleared the strict gate)
    Response oracle O1:      no valid target yet

That result does NOT show map_a is too easy. It shows the seven BASE opponent
configurations sit below G0's current capability. The existing variants are the
correct next test, and ALL of them are run -- not a promising-looking subset,
which would reintroduce selection bias.

This is a NEW file rather than an edit of run_g0_weakness_sweep.py so that the
completed base sweep's recorded provenance stays valid forever; its integrity
machinery is imported rather than copied.

Design (locked before running):
    variants    21 non-base OP6-OP12 keys, all of them
    policies    all three frozen G0 seeds; no best-seed selection
    map         map_a (canonical), resolved map_a_open, recorded per row
    episodes    32 paired discovery seeds, block 1_400_001..1_400_032
    totals      21 x 3 x 32 = 2016 discovery episodes
    fixed       2v2, 240 steps, deterministic, no DR, n_envs=1,
                obstacle_obs_channel=True (8-channel contract G0 was trained on)

Confirmation (separate run, fresh block 1_410_001..1_410_064) is required
before any variant becomes C1 -- discovery over 21 candidates is even more
selection-biased than over 7.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "plot") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "plot"))

# Reuse the base sweep's integrity machinery verbatim.
from experiments.run_g0_weakness_sweep import (  # noqa: E402
    AGENTS,
    BEHAVIOR_FIELDS,
    FIELDS,
    G0_SEEDS,
    G0_STEP,
    MAP,
    MAX_DECISION_STEPS,
    RESOLVED_MAP,
    ROW_KEY,
    ExclusiveLock,
    ckpt_for,
    load_keys,
)

# All 21 non-base variants. Every one is run; no subsetting.
VARIANTS = [
    "OP6_DUAL_RUSH", "OP6_IMMEDIATE_DUAL_RUSH", "OP6_TURTLE",
    "OP7_DEEP_FORTRESS", "OP7_FORTRESS", "OP7_SWITCHER",
    "OP8_ESCORT", "OP8_INTERCEPTOR", "OP8_PROTECTED_CARRIER_ESCORT",
    "OP9_FEINT", "OP9_FORTRESS", "OP9_SPLIT_LANE_FEINT",
    "OP10_AGGRESSIVE_INTERCEPTOR", "OP10_ESCORT", "OP10_INTERCEPTOR",
    "OP11_ADAPTIVE_EXPLOITER", "OP11_BT_BALANCED", "OP11_EXPLOITER",
    "OP12_CONVERTER", "OP12_COUNTER", "OP12_LATE_CONVERTER",
]

DISCOVERY_SEED_BASE = 1_400_001
CONFIRMATION_SEED_BASE = 1_410_001
EPISODES = 32


def verify(path: Path, episodes: int, variants: list[str]) -> int:
    if not path.exists():
        print(f"[verify] FAIL: {path} does not exist", file=sys.stderr)
        return 1
    keys, cells, maps, rmaps = Counter(), Counter(), set(), set()
    n = 0
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            n += 1
            keys[tuple(str(r[k]) for k in ROW_KEY)] += 1
            cells[(r["opponent"], r["g0_seed"])] += 1
            maps.add(r["map"])
            rmaps.add(r.get("resolved_map", ""))
    expected_cells = len(variants) * len(G0_SEEDS)
    expected = expected_cells * episodes
    dups = {k: v for k, v in keys.items() if v > 1}
    bad = {c: v for c, v in cells.items() if v != episodes}
    ok = (n == expected and not dups and not bad
          and len(cells) == expected_cells
          and maps == {MAP} and rmaps == {RESOLVED_MAP})
    print(f"[verify] rows              : {n} (expected {expected})")
    print(f"[verify] duplicate keys    : {len(dups)} (expected 0)")
    print(f"[verify] cells             : {len(cells)} (expected {expected_cells})")
    print(f"[verify] cells != {episodes} eps  : {len(bad)} (expected 0)")
    print(f"[verify] map values        : {sorted(maps)} (expected ['{MAP}'])")
    print(f"[verify] resolved_map      : {sorted(rmaps)} (expected ['{RESOLVED_MAP}'])")
    for k in list(dups)[:5]:
        print(f"   duplicate: {k} x{dups[k]}", file=sys.stderr)
    for c in list(bad)[:5]:
        print(f"   short/long cell: {c} -> {bad[c]}", file=sys.stderr)
    print(f"[verify] {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episodes", type=int, default=EPISODES)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out-dir", default="artifacts/g0_variant_weakness_sweep")
    p.add_argument("--seed-base", type=int, default=DISCOVERY_SEED_BASE,
                   help=f"Discovery {DISCOVERY_SEED_BASE}; confirmation "
                        f"{CONFIRMATION_SEED_BASE}.")
    p.add_argument("--variants", nargs="+", default=VARIANTS,
                   help="Default is ALL 21; subsetting reintroduces selection bias.")
    p.add_argument("--force-run", action="store_true")
    p.add_argument("--verify", action="store_true")
    args = p.parse_args()

    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "episode_rows.csv"

    if args.verify:
        return verify(rows_path, args.episodes, args.variants)

    unknown = [v for v in args.variants if v not in VARIANTS]
    if unknown:
        print(f"[abort] not in the declared variant list: {unknown}", file=sys.stderr)
        return 1

    eval_seeds = [args.seed_base + i for i in range(args.episodes)]
    cells = [(opp, seed) for opp in args.variants for seed in G0_SEEDS]
    missing = [str(ckpt_for(s)) for s in G0_SEEDS if not ckpt_for(s).exists()]

    phase = ("discovery" if args.seed_base == DISCOVERY_SEED_BASE
             else "confirmation" if args.seed_base == CONFIRMATION_SEED_BASE else "custom")
    manifest = {
        "experiment": "g0_variant_weakness_sweep",
        "pass": f"second (21 existing OP6-OP12 variants) -- {phase}",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PREDECLARED" if not args.force_run else "RUNNING",
        "predecessor": {
            "experiment": "g0_weakness_sweep (base keys)",
            "result": "G0 COMPETENT on map_a (0/7 negative); NO base context cleared "
                      "the strict weakness gate; no valid C1.",
        },
        "incumbent": {"name": "G0", "seeds": G0_SEEDS, "checkpoint_step": G0_STEP,
                      "best_seed_selection": "prohibited"},
        "variants": args.variants,
        "all_variants_run": args.variants == VARIANTS,
        "subsetting_note": "All 21 variants are run; selecting only promising-looking "
                           "ones would reintroduce selection bias.",
        "map": MAP, "resolved_map": RESOLVED_MAP,
        "map_recording_rule": "every row records map=map_a; resolved_map is provenance only",
        "map_pooling": "map_a must never be pooled with map_b",
        "phase": phase,
        "eval_seed_block": [eval_seeds[0], eval_seeds[-1]],
        "discovery_seed_block": [DISCOVERY_SEED_BASE, DISCOVERY_SEED_BASE + 31],
        "confirmation_seed_block": [CONFIRMATION_SEED_BASE, CONFIRMATION_SEED_BASE + 63],
        "episodes_per_cell": args.episodes,
        "n_cells": len(cells),
        "expected_rows": len(cells) * args.episodes,
        "invariants": {"agents": AGENTS, "max_decision_steps": MAX_DECISION_STEPS,
                       "deterministic": True, "domain_randomization": False, "n_envs": 1,
                       "obstacle_obs_channel": True},
        "obs_contract_note": (
            "obstacle_obs_channel pinned True: G0 trained on map_b where it defaults True "
            "(8-channel CNN input); on map_a it would default False (7 channels) and the "
            "checkpoint would fail to load. On the open arena the channel is all zeros."
        ),
        "row_key": list(ROW_KEY),
        "integrity": "exclusive lock file; duplicate keys rejected, never appended",
        "weakness_gate": {
            "required": ["all three G0 seeds have negative mean margin",
                         "family-level UCB95 < 0"],
            "rank_if_multiple": ["lowest strongest-member margin",
                                 "lowest family mean margin",
                                 "lower saturation and tie rate"],
            "if_none_pass": (
                "Do NOT select the smallest positive score. Conclusion becomes: no existing "
                "opponent behavior creates a robust strategic weakness for G0 on map_a. "
                "Next declared step is to ENGINEER one new OP6-OP12 variant on map_a around "
                "an unavoidable trade-off (delay until G0 commits forward; counterattack its "
                "exposed home flag; use the second RED agent to block the return path)."
            ),
            "confirmation_required": (
                f"Every discovery qualifier must reconfirm on the fresh 64-seed block "
                f"{CONFIRMATION_SEED_BASE}..{CONFIRMATION_SEED_BASE + 63} before becoming C1."
            ),
        },
        "missing_checkpoints": missing,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[manifest] {out_dir / 'manifest.json'}  cells={len(cells)} "
          f"expected_rows={len(cells) * args.episodes}  phase={phase}")

    if missing:
        print("[abort] missing G0 checkpoints:", file=sys.stderr)
        for m in missing:
            print("   ", m, file=sys.stderr)
        return 1

    if not args.force_run:
        for opp, seed in cells:
            print(f"  [dry] G0/s{seed} vs {opp:32s} map={MAP} "
                  f"seeds {eval_seeds[0]}..{eval_seeds[-1]}")
        print("[dry] pass --force-run to execute.")
        return 0

    from game_field_gpu import GPUCTFVecEnv, GPUFieldConfig
    from eval_rollout import run_eval_episodes

    with ExclusiveLock(out_dir / ".sweep.lock"):
        seen = load_keys(rows_path)
        if seen:
            print(f"[resume] {len(seen)} existing rows; duplicate keys will be rejected")
        wrote = rows_path.exists()
        with open(rows_path, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
            if not wrote:
                w.writeheader()

            for idx, (opp, seed) in enumerate(cells, 1):
                print(f"[{idx}/{len(cells)}] G0/s{seed} -> {opp} (map={MAP})", flush=True)
                cfg = GPUFieldConfig(
                    n_envs=1, max_blue_agents=AGENTS, max_red_agents=AGENTS,
                    map_set="train", map_layout=MAP,
                    max_decision_steps=MAX_DECISION_STEPS,
                    aquaticus_profile=True, rules_profile="OURS",
                    device=args.device, seed=int(args.seed_base),
                    obstacle_obs_channel=True,
                )
                env = GPUCTFVecEnv(cfg)
                try:
                    eps = run_eval_episodes(
                        str(ckpt_for(seed)), env, int(args.episodes), args.device, opp,
                        deterministic=True, coordination_metrics=False,
                        collect_behavior_mean=True,
                        latent_eval_seed=int(args.seed_base), progress_every=0,
                    )
                    actual = (env.env_method("get_opponent_key")[0] or "").strip().upper()
                    if actual != opp.strip().upper():
                        print(f"[abort] opponent mismatch: core has {actual!r}, "
                              f"requested {opp!r}", file=sys.stderr)
                        return 1
                finally:
                    env.close()

                n_dup = 0
                for ep_i, ep in enumerate(eps):
                    bs = int(ep.get("blue_score", 0) or 0)
                    rs = int(ep.get("red_score", 0) or 0)
                    row = {
                        "checkpoint_step": G0_STEP, "g0_seed": seed, "opponent": opp,
                        "map": MAP, "resolved_map": RESOLVED_MAP,
                        "episode_index": ep_i, "episode_seed": eval_seeds[ep_i],
                        "success": int(bool(ep.get("success", bs > rs))),
                        "blue_score": bs, "red_score": rs, "win_margin": bs - rs,
                        "steps": int(ep.get("steps", 0) or 0),
                        "return": float(ep.get("return", 0.0) or 0.0),
                    }
                    for b in BEHAVIOR_FIELDS:
                        v = ep.get(f"behavior_{b}")
                        row[f"behavior_{b}"] = "" if v is None else float(v)
                    key = tuple(str(row[k]) for k in ROW_KEY)
                    if key in seen:
                        n_dup += 1
                        continue
                    seen.add(key)
                    w.writerow(row)
                if n_dup:
                    print(f"    rejected {n_dup} duplicate keys", flush=True)
                fh.flush()

    manifest["status"] = "COMPLETE"
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\n[done] {rows_path}")
    print("Next: --verify, then experiments/analyze_g0_weakness.py from the analysis worktree")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
