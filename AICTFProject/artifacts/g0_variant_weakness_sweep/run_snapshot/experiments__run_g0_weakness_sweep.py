#!/usr/bin/env python3
"""G0 learned-incumbent weakness sweep — first pass, seven BASE opponents.

The OP11/OP9 specialist attempt is CLOSED_FAILED. Its lesson:

    Context diversity is not policy diversity. Scripted strategy labels are
    not learned strategy branches.

Contexts were previously chosen by asking which *scripted probe* won. That did
not predict what independently trained PPO policies would do. This sweep asks
instead: which context defeats the actual LEARNED incumbent?

G0 is the frozen 1M piR family — NOT "the RUSH specialist". It learned
something broader than that label.

LOCKED SEMANTICS
----------------
Opponents   the seven BASE keys only: OP6..OP12. Variants such as
            OP11_ADAPTIVE_EXPLOITER and OP9_SPLIT_LANE_FEINT are deliberately
            EXCLUDED from this first pass; they belong to a separately declared
            expansion, and only if the base sweep finds no robust weakness.
Map         experiment-facing canonical name is `map_a` and that is what every
            row records. `map_a_open` is carried only as `resolved_map`
            provenance. Normalization must never overwrite the row's `map`.
Policies    all three G0 seeds; no best-seed selection.
Eval seeds  ONE fresh block of 32 seeds, paired across every policy AND every
            opponent, so any cell can be differenced against any other.
Totals      3 policies x 7 opponents x 32 episodes = 672 unique rows.
Fixed       2v2, 240 decision steps, deterministic, no domain randomization,
            n_envs=1.

INTEGRITY (after the duplicate-writer incident)
-----------------------------------------------
Unique row key:  (checkpoint, g0_training_seed, opponent, map, evaluation_seed)

* an exclusive lock file prevents a second writer from touching the output;
* existing keys are loaded up front and duplicates are REJECTED, not appended;
* `--verify` checks 672 rows, 0 duplicates, exactly 32 episodes per cell.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "plot") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "plot"))

# Seven BASE keys only. Variants excluded from the first pass by design.
OPPONENTS = ["OP6", "OP7", "OP8", "OP9", "OP10", "OP11", "OP12"]

# Experiment-facing canonical map name. The env resolves map_a -> map_a_open;
# that resolution is provenance only and must not replace `map` in any row.
MAP = "map_a"
RESOLVED_MAP = "map_a_open"

G0_SEEDS = [901001, 901002, 901003]
G0_STEP = 1_000_000
MAX_DECISION_STEPS = 240
AGENTS = 2
EPISODES = 32

# ONE fresh block, shared by every (policy x opponent) cell so all cells are
# paired. Disjoint from every prior block.
EVAL_SEED_BASE = 1_300_001

BEHAVIOR_FIELDS = [
    "team_spread", "num_attackers", "num_defenders", "carrier_escort_count",
    "avg_blue_to_enemy_flag", "avg_blue_to_own_flag",
    "intercept_pressure", "defense_pressure", "attack_defense_ratio",
]

ROW_KEY = ("checkpoint_step", "g0_seed", "opponent", "map", "episode_seed")

FIELDS = ["checkpoint_step", "g0_seed", "opponent", "map", "resolved_map",
          "episode_index", "episode_seed", "success", "blue_score", "red_score",
          "win_margin", "steps", "return"] + [f"behavior_{b}" for b in BEHAVIOR_FIELDS]


def ckpt_for(seed: int) -> Path:
    return (PROJECT_ROOT / "checkpoints" / "k2v2_piR"
            / f"ckpt_k2v2_piR_op11_mapb_s{seed}_2v2_{G0_STEP}.zip")


class ExclusiveLock:
    """Fail-fast lock so two processes can never write the same output."""

    def __init__(self, path: Path):
        self.path = path
        self.fd = None

    def __enter__(self):
        try:
            self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            raise SystemExit(
                f"[abort] output is locked by another process: {self.path}\n"
                f"        If no sweep is running, delete the lock file and retry."
            )
        os.write(self.fd, f"pid={os.getpid()} {datetime.now(timezone.utc).isoformat()}\n"
                          .encode())
        return self

    def __exit__(self, *exc):
        if self.fd is not None:
            os.close(self.fd)
        self.path.unlink(missing_ok=True)


def load_keys(path: Path) -> set:
    keys = set()
    if not path.exists():
        return keys
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            keys.add(tuple(str(r[k]) for k in ROW_KEY))
    return keys


def verify(path: Path, episodes: int) -> int:
    if not path.exists():
        print(f"[verify] FAIL: {path} does not exist", file=sys.stderr)
        return 1
    keys = Counter()
    cells = Counter()
    maps = set()
    n = 0
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            n += 1
            keys[tuple(str(r[k]) for k in ROW_KEY)] += 1
            cells[(r["opponent"], r["g0_seed"])] += 1
            maps.add(r["map"])
    expected = len(OPPONENTS) * len(G0_SEEDS) * episodes
    dups = {k: v for k, v in keys.items() if v > 1}
    bad_cells = {c: v for c, v in cells.items() if v != episodes}
    ok = True
    print(f"[verify] rows              : {n} (expected {expected})")
    print(f"[verify] duplicate keys    : {len(dups)} (expected 0)")
    print(f"[verify] cells             : {len(cells)} (expected "
          f"{len(OPPONENTS) * len(G0_SEEDS)})")
    print(f"[verify] cells != {episodes} eps  : {len(bad_cells)} (expected 0)")
    print(f"[verify] distinct map vals : {sorted(maps)} (expected ['{MAP}'])")
    if n != expected or dups or bad_cells or maps != {MAP}:
        ok = False
        for k in list(dups)[:5]:
            print(f"   duplicate: {k} x{dups[k]}", file=sys.stderr)
        for c in list(bad_cells)[:5]:
            print(f"   short/long cell: {c} -> {bad_cells[c]}", file=sys.stderr)
    print(f"[verify] {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episodes", type=int, default=EPISODES)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out-dir", default="artifacts/g0_weakness_sweep")
    p.add_argument("--force-run", action="store_true",
                   help="Actually run. Default is dry-run / manifest only.")
    p.add_argument("--verify", action="store_true",
                   help="Check row count, duplicate keys, and per-cell counts; exit.")
    args = p.parse_args()

    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "episode_rows.csv"

    if args.verify:
        return verify(rows_path, args.episodes)

    eval_seeds = [EVAL_SEED_BASE + i for i in range(args.episodes)]
    cells = [(opp, seed) for opp in OPPONENTS for seed in G0_SEEDS]
    missing = [str(ckpt_for(s)) for s in G0_SEEDS if not ckpt_for(s).exists()]

    manifest = {
        "experiment": "g0_learned_incumbent_weakness_sweep",
        "pass": "first (seven BASE opponent keys only)",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PREDECLARED" if not args.force_run else "RUNNING",
        "incumbent": {
            "name": "G0", "seeds": G0_SEEDS, "checkpoint_step": G0_STEP,
            "description": "frozen 1M piR family, incumbent generalist "
                           "(NOT 'the RUSH specialist')",
            "best_seed_selection": "prohibited",
        },
        "opponents": OPPONENTS,
        "opponent_variants_excluded": (
            "OP11_ADAPTIVE_EXPLOITER, OP9_SPLIT_LANE_FEINT and all other variants are "
            "excluded from this first pass. They belong to a separately declared "
            "expansion, run only if the base sweep finds no robust weakness."
        ),
        "map": MAP,
        "resolved_map": RESOLVED_MAP,
        "map_recording_rule": "every row records map=map_a; resolved_map is provenance only",
        "map_pooling": "map_a must never be pooled with map_b",
        "eval_seed_block": [eval_seeds[0], eval_seeds[-1]],
        "eval_seed_pairing": "ONE block shared by every policy AND opponent",
        "episodes_per_cell": args.episodes,
        "n_cells": len(cells),
        "expected_rows": len(cells) * args.episodes,
        "invariants": {"agents": AGENTS, "max_decision_steps": MAX_DECISION_STEPS,
                       "deterministic": True, "domain_randomization": False, "n_envs": 1,
                       "obstacle_obs_channel": True},
        "obs_contract_note": (
            "obstacle_obs_channel is pinned True. G0 was trained on map_b where it "
            "defaults True (8-channel CNN input); on map_a it would default False "
            "(7 channels) and the checkpoint would fail to load with a shape mismatch on "
            "actor_cnn.conv.0.weight. On the open arena the obstacle channel is all zeros. "
            "This is the documented cross-map observation contract."
        ),
        "row_key": list(ROW_KEY),
        "integrity": "exclusive lock file; duplicate keys rejected, never appended",
        "weakness_gate": {
            "required": ["all three G0 seeds have negative mean margin",
                         "family-level UCB95 < 0"],
            "rank_if_multiple": ["lowest strongest-member margin",
                                 "lowest family mean margin",
                                 "lower saturation and tie rate"],
            "if_none_pass": ("close the base sweep as 'no robust weakness'; do NOT pick "
                             "the least-good opponent. Stage a separately declared "
                             "existing-variant sweep within OP6-OP12."),
        },
        "precedent": {"k2v3_300k_replication": "CANCELLED_PRELAUNCH",
                      "k2v2_1m_formal": "FAIL - piR dominant generalist"},
        "missing_checkpoints": missing,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[manifest] {out_dir / 'manifest.json'}  cells={len(cells)} "
          f"expected_rows={len(cells) * args.episodes}")

    if missing:
        print("[abort] missing G0 checkpoints:", file=sys.stderr)
        for m in missing:
            print("   ", m, file=sys.stderr)
        return 1

    if not args.force_run:
        for opp, seed in cells:
            print(f"  [dry] G0/s{seed} vs {opp:5s} map={MAP} "
                  f"seeds {eval_seeds[0]}..{eval_seeds[-1]}")
        print("[dry] pass --force-run to execute (after the GPU audit/provenance chain).")
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
                    device=args.device, seed=int(EVAL_SEED_BASE),
                    # G0 was trained on map_b, where obstacle_obs_channel defaults
                    # True -> an 8-channel CNN input. On map_a the same flag would
                    # default False (7 channels) and the checkpoint would fail to
                    # load. Pin the 8-channel contract: on the open arena the
                    # obstacle channel is simply all zeros. This is the documented
                    # cross-map contract, not a workaround -- see
                    # experiments/diagnose_v6i26_map_a_obs_compat.py.
                    obstacle_obs_channel=True,
                )
                env = GPUCTFVecEnv(cfg)
                try:
                    eps = run_eval_episodes(
                        str(ckpt_for(seed)), env, int(args.episodes), args.device, opp,
                        deterministic=True, coordination_metrics=False,
                        collect_behavior_mean=True,
                        latent_eval_seed=int(EVAL_SEED_BASE), progress_every=0,
                    )
                    actual = (env.env_method("get_opponent_key")[0] or "").strip().upper()
                    if actual != opp.strip().upper():
                        print(f"[abort] opponent mismatch: core has {actual!r}, "
                              f"requested {opp!r}", file=sys.stderr)
                        return 1
                finally:
                    env.close()

                n_written = n_dup = 0
                for ep_i, ep in enumerate(eps):
                    bs = int(ep.get("blue_score", 0) or 0)
                    rs = int(ep.get("red_score", 0) or 0)
                    row = {
                        "checkpoint_step": G0_STEP, "g0_seed": seed, "opponent": opp,
                        "map": MAP, "resolved_map": RESOLVED_MAP,
                        "episode_index": ep_i,
                        "episode_seed": eval_seeds[ep_i],
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
                    n_written += 1
                if n_dup:
                    print(f"    rejected {n_dup} duplicate keys", flush=True)
                fh.flush()

    manifest["status"] = "COMPLETE"
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\n[done] {rows_path}")
    print("Next: --verify, then experiments/analyze_g0_weakness.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
