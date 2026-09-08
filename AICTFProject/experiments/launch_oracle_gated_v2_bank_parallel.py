"""Launch four disjoint V2 bank collectors (parallel shard collection).

Each worker writes ONLY to its own directory under workers/. After all finish,
run merge_oracle_gated_v2_bank_workers.py.

Run:  python experiments/launch_oracle_gated_v2_bank_parallel.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
COLLECTOR = ROOT / "experiments" / "collect_oracle_gated_v2_bank_data.py"
OUT = ROOT / "artifacts" / "strategic_demand" / "sppo" / "oracle_gated_k2_v2_bank_data"
PLAN = OUT / "PARALLEL_COLLECTION_PLAN.json"

BASE, N_SEEDS = 11_000_001, 320
FROZEN = list(range(BASE, BASE + N_SEEDS))
N_WORKERS = 4


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _serial_completed() -> list[int]:
    manifest = OUT / "collection_manifest.json"
    if not manifest.is_file():
        return []
    return [int(s) for s in json.loads(manifest.read_text(encoding="utf-8")).get("completed_seeds", [])]


def _partition(remaining: list[int], n_workers: int) -> list[tuple[str, int, int]]:
    chunks = []
    per = (len(remaining) + n_workers - 1) // n_workers
    for i in range(n_workers):
        start = i * per
        if start >= len(remaining):
            break
        block = remaining[start:start + per]
        chunks.append((f"w{i + 1}", block[0], block[-1]))
    return chunks


def main() -> int:
    serial = sorted(_serial_completed())
    remaining = [s for s in FROZEN if s not in serial]
    if not remaining:
        print("REFUSING: no remaining seeds to partition")
        return 1
    workers = _partition(remaining, N_WORKERS)
    PLAN.write_text(json.dumps({
        "utc": _now(),
        "frozen_block": [FROZEN[0], FROZEN[-1], len(FROZEN)],
        "serial_prefix_completed": serial,
        "remaining_seeds": len(remaining),
        "workers": [{"id": w, "lo": lo, "hi": hi, "count": hi - lo + 1} for w, lo, hi in workers],
    }, indent=2), encoding="utf-8")
    print(json.dumps(json.loads(PLAN.read_text()), indent=2))

    procs = []
    for wid, lo, hi in workers:
        wout = OUT / "workers" / wid
        wout.mkdir(parents=True, exist_ok=True)
        log_out = wout / "collector.stdout.log"
        log_err = wout / "collector.stderr.log"
        cmd = [
            PY, str(COLLECTOR),
            "--device", "cuda",
            "--seed-lo", str(lo),
            "--seed-hi", str(hi),
            "--out-dir", str(wout),
            "--worker-id", wid,
        ]
        print(f"launch {wid} {lo}..{hi} pid pending -> {wout.relative_to(ROOT)}")
        with log_out.open("w", encoding="utf-8") as fo, log_err.open("w", encoding="utf-8") as fe:
            procs.append(subprocess.Popen(cmd, cwd=str(ROOT), stdout=fo, stderr=fe))
    print(f"started {len(procs)} workers: {[p.pid for p in procs]}")
    print(f"plan -> {PLAN}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
