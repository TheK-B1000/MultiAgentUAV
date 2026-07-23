#!/usr/bin/env python3
"""Chain: wait for Stage-0 JSON → launch Stage-1 LRO manufacture/refine.

No more archive fishing. G_available=0 is an explicit go for manufacture.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage-0 → Stage-1 auto-chain")
    p.add_argument(
        "--landscape-scan",
        default="artifacts/v6i26_landscape_scan_op8_12_seed1/landscape_scan.json",
    )
    p.add_argument(
        "--checkpoint",
        default="artifacts/v6i23_population_birth_5u_seed1/final_v6i23_population_birth_5u_seed1_2v2.zip",
    )
    p.add_argument("--output-dir", default="artifacts/v6i26_lro_round1_seed1")
    p.add_argument("--updates", type=int, default=25)
    p.add_argument("--device", default="cuda")
    p.add_argument("--poll-seconds", type=float, default=30.0)
    p.add_argument("--timeout-seconds", type=float, default=14400.0)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    scan_path = Path(args.landscape_scan)
    t0 = time.time()
    print(f"Waiting for Stage-0: {scan_path}", flush=True)
    while not scan_path.is_file():
        if time.time() - t0 > float(args.timeout_seconds):
            print("ERROR: timed out waiting for landscape_scan.json")
            return 2
        time.sleep(float(args.poll_seconds))

    # Wait until file is stable (scan still writing).
    last_size = -1
    stable = 0
    while stable < 2:
        size = scan_path.stat().st_size
        if size > 0 and size == last_size:
            stable += 1
        else:
            stable = 0
        last_size = size
        time.sleep(2.0)

    scan = json.loads(scan_path.read_text(encoding="utf-8"))
    decision = str(scan.get("decision") or "")
    g = scan.get("G_available_effective")
    if g is None:
        g = (scan.get("summary") or {}).get("G_available_point")
    try:
        g_f = float(g) if g is not None else float("nan")
    except (TypeError, ValueError):
        g_f = float("nan")

    # Remap legacy Stage-0 labels: G<=0 is manufacture, not more fishing.
    if decision in {"INCONCLUSIVE_EXPAND_SCAN", "INCONCLUSIVE_EXTEND_SCAN", "FAIL_TASK_DISTRIBUTION"} or (
        decision == "" and not (g_f > 0.0)
    ):
        decision = "MANUFACTURE_VIA_LRO_STAGE1"
        scan["decision"] = decision
        scan["note"] = (
            "Remapped: archives have no harvestable repertoire. "
            "Stage-1 manufactures; first success is G_after > G_before."
        )
        scan["G_available_effective"] = g_f
        scan_path.write_text(json.dumps(scan, indent=2), encoding="utf-8")

    print(f"Stage-0 ready: decision={decision} G_available={g_f}", flush=True)
    print(f"note={scan.get('note')}", flush=True)

    cmd = [
        "uv",
        "run",
        "python",
        "experiments/run_v6i26_lro_oracle_round.py",
        "--landscape-scan",
        str(scan_path),
        "--checkpoint",
        str(args.checkpoint),
        "--output-dir",
        str(args.output_dir),
        "--updates",
        str(int(args.updates)),
        "--device",
        str(args.device),
    ]
    if args.dry_run:
        cmd.append("--dry-run")
    print("Launching Stage-1:", " ".join(cmd), flush=True)
    return int(subprocess.call(cmd, cwd=str(PROJECT_ROOT)))


if __name__ == "__main__":
    raise SystemExit(main())
