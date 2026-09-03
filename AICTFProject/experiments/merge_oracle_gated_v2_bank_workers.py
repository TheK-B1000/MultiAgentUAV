"""Merge parallel V2 bank collection workers into the canonical shard tree.

Run:  python experiments/merge_oracle_gated_v2_bank_workers.py
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SD = ROOT / "artifacts" / "strategic_demand"
OUT = SD / "sppo" / "oracle_gated_k2_v2_bank_data"
WORKERS_ROOT = OUT / "workers"
PLAN = OUT / "PARALLEL_COLLECTION_PLAN.json"
COMPLETE = OUT / "COLLECTION_COMPLETE.json"
MERGE_RECORD = OUT / "V2_BANK_MERGE_MANIFEST.json"

BASE, N_SEEDS = 11_000_001, 320
FROZEN = list(range(BASE, BASE + N_SEEDS))


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_plan() -> tuple[list[int], dict[str, tuple[int, int]]]:
    if not PLAN.is_file():
        raise SystemExit(f"REFUSING: {PLAN} missing")
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    serial = [int(s) for s in plan["serial_prefix_completed"]]
    workers = {w["id"]: (int(w["lo"]), int(w["hi"])) for w in plan["workers"]}
    return serial, workers


def _sources_for_seed(seed: int, serial: list[int], workers: dict[str, tuple[int, int]]) -> list[Path]:
    paths = []
    if seed in serial:
        paths.append(OUT / "seed_shards" / f"seed_{seed}.npz")
    for wid, (lo, hi) in workers.items():
        if lo <= seed <= hi:
            paths.append(WORKERS_ROOT / wid / "seed_shards" / f"seed_{seed}.npz")
    return paths


def main() -> int:
    if COMPLETE.is_file():
        raise SystemExit(f"REFUSING: {COMPLETE} already exists")
    if MERGE_RECORD.is_file():
        raise SystemExit(f"REFUSING: {MERGE_RECORD} exists; merge is one-shot")

    serial, workers = _load_plan()
    shards_dir = OUT / "seed_shards"
    summaries_dir = OUT / "seed_summaries"
    shards_dir.mkdir(parents=True, exist_ok=True)
    summaries_dir.mkdir(parents=True, exist_ok=True)

    copied, shard_hashes = [], {}
    for seed in FROZEN:
        sources = [p for p in _sources_for_seed(seed, serial, workers) if p.is_file()]
        if not sources:
            continue
        if len(sources) > 1:
            hashes = {_sha256(p) for p in sources}
            if len(hashes) > 1:
                raise SystemExit(f"REFUSING: conflicting shards for seed {seed}")
        src = sources[0]
        dst = shards_dir / f"seed_{seed}.npz"
        if not dst.is_file():
            shutil.copy2(src, dst)
            copied.append(seed)
        elif _sha256(dst) != _sha256(src):
            raise SystemExit(f"REFUSING: canonical shard for {seed} disagrees with source")
        shard_hashes[str(seed)] = _sha256(dst)

        for wid, (lo, hi) in workers.items():
            if lo <= seed <= hi:
                wsum = WORKERS_ROOT / wid / "seed_summaries" / f"seed_{seed}.json"
                dsum = summaries_dir / f"seed_{seed}.json"
                if wsum.is_file() and not dsum.is_file():
                    shutil.copy2(wsum, dsum)
                break

    present = {int(p.stem.split("seed_")[-1]) for p in shards_dir.glob("seed_*.npz")}
    missing = sorted(set(FROZEN) - present)
    extra = sorted(present - set(FROZEN))
    if missing:
        raise SystemExit(f"REFUSING: missing {len(missing)} seeds (e.g. {missing[:5]})")
    if extra:
        raise SystemExit(f"REFUSING: extra seeds {extra[:5]}")

    for wid in workers:
        marker = WORKERS_ROOT / wid / "WORKER_RANGE_COMPLETE.json"
        if not marker.is_file():
            raise SystemExit(f"REFUSING: worker {wid} not complete")

    record = {
        "record": "V2 parallel bank merge",
        "utc": _now(),
        "plan": str(PLAN.relative_to(ROOT)),
        "frozen_block": [FROZEN[0], FROZEN[-1], len(FROZEN)],
        "serial_prefix": serial,
        "shard_sha256": shard_hashes,
        "block_digest": hashlib.sha256(
            "".join(shard_hashes[str(s)] for s in sorted(shard_hashes, key=int)).encode()
        ).hexdigest(),
    }
    MERGE_RECORD.write_text(json.dumps(record, indent=2), encoding="utf-8")
    COMPLETE.write_text(json.dumps({
        "verdict": "COLLECTION_COMPLETE",
        "utc": _now(),
        "seed_block": [FROZEN[0], FROZEN[-1], len(FROZEN)],
        "completed_seeds": len(FROZEN),
        "parallel_merge": str(MERGE_RECORD.relative_to(ROOT)),
        "next_step": "experiments/audit_oracle_gated_v2_bank.py",
    }, indent=2), encoding="utf-8")
    print(f"MERGE OK  {len(FROZEN)} seeds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
