"""Launch the C5 discovery scan as independent per-cell worker processes.

The scan is embarrassingly parallel and was NOT parallel: C4 spent 37,609s
(10.4h) walking 21 (policy, opponent) cells serially at ~60s per episode.

This changes ONLY which process computes which cell. It does not touch the
experiment: every frozen parameter -- 30 seeds of block 9840000, exhaustive
response enumeration, the 30-step carrier-survival horizon, delta, support
minima, replication -- is untouched. collect_states builds a fresh env seeded
per (opponent, seed), so a cell's states do not depend on which other cells ran,
which is verified empirically by the order-independence probe before this is
used in anger.

Shards are merged in a FIXED cell order, not in completion order, so the merged
states file is byte-identical regardless of which worker finishes first.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv/Scripts/python.exe")
G0_SEEDS = [3200001, 3200002, 3200003]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--workers", type=int, default=11)
    ap.add_argument("--shard-dir", default=str(ROOT / "artifacts/c5_discovery/shards"))
    ap.add_argument("--states-out", default=str(ROOT / "artifacts/c5_discovery/states.json"))
    args = ap.parse_args()

    from experiments.run_g0_v2_seed import OPPONENTS
    opponents = list(OPPONENTS)

    shard_dir = Path(args.shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)

    # One cell = one (policy, opponent). Fixed order defines the merge order.
    cells = [(p, o) for p in G0_SEEDS for o in opponents]
    print(f"C5 PARALLEL SCAN: {len(cells)} cells, {args.workers} concurrent, "
          f"{args.episodes} episodes/cell", flush=True)

    pending = list(cells)
    running: list[tuple] = []
    started = time.time()
    done = 0

    while pending or running:
        while pending and len(running) < args.workers:
            pseed, opp = pending.pop(0)
            sf = shard_dir / f"states_{pseed}_{opp}.json"
            if sf.exists():          # resume: a completed shard is never recomputed
                print(f"  [skip] {pseed}/{opp} already present", flush=True)
                done += 1
                continue
            log = shard_dir / f"{pseed}_{opp}.log"
            cmd = [PY, "-u", str(ROOT / "experiments/run_c4_opportunity_cost.py"),
                   "--episodes", str(args.episodes),
                   "--policies", str(pseed), "--opponents", opp,
                   "--partition-mode", "opponent",
                   "--out", str(shard_dir / f"res_{pseed}_{opp}.json"),
                   "--states-out", str(sf)]
            fh = open(log, "w", encoding="utf-8")
            # Each worker drives a single-env sim, so intra-process BLAS threading
            # buys nothing and many unpinned workers would thrash the 16 cores
            # against each other. One thread each, parallelism across processes.
            env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
            proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=fh,
                                    stderr=subprocess.STDOUT, env=env)
            running.append((proc, pseed, opp, fh))
            print(f"  [start] {pseed}/{opp}", flush=True)

        time.sleep(5)
        still = []
        for proc, pseed, opp, fh in running:
            if proc.poll() is None:
                still.append((proc, pseed, opp, fh))
                continue
            fh.close()
            done += 1
            el = time.time() - started
            ok = proc.returncode == 0
            print(f"  [{'done' if ok else 'FAIL'}] {pseed}/{opp}  rc={proc.returncode}  "
                  f"{done}/{len(cells)}  elapsed {el/60:.1f}m", flush=True)
            if not ok:
                print(f"    see {shard_dir / f'{pseed}_{opp}.log'}", flush=True)
        running = still

    # Merge in FIXED cell order so the result is independent of completion order.
    merged: dict[str, list] = {str(p): [] for p in G0_SEEDS}
    missing = []
    for pseed, opp in cells:
        sf = shard_dir / f"states_{pseed}_{opp}.json"
        if not sf.exists():
            missing.append(f"{pseed}/{opp}")
            continue
        shard = json.loads(sf.read_text(encoding="utf-8"))
        merged[str(pseed)].extend(shard.get(str(pseed), []))

    if missing:
        print(f"\nABORT: {len(missing)} cells missing: {missing}", file=sys.stderr)
        print("Re-run this launcher; completed shards are reused.", file=sys.stderr)
        return 1

    Path(args.states_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.states_out).write_text(json.dumps(merged), encoding="utf-8")
    el = time.time() - started
    print(f"\nmerged {sum(len(v) for v in merged.values())} states -> {args.states_out}")
    print(f"wall {el/60:.1f} min (C4 serial baseline: 626.8 min)")
    for p in G0_SEEDS:
        print(f"  policy {p}: {len(merged[str(p)])} states")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
