#!/usr/bin/env python3
"""
Parallelizes experiments/run_forced_z_eval.py ACROSS (opponent) conditions,
without touching its seeding contract.

Why this exists: run_forced_z_eval.py runs every (opponent, z, map) condition
sequentially through a single n_envs=1 environment. The seeding scheme in
plot/eval_rollout.py::run_eval_episodes reseeds *global* RNG state once per
episode (random.seed/np.random.seed/torch.manual_seed/env.seed) rather than
using per-env-slot generators, so simply raising n_envs would make every
parallel slot replay the *same* episode instead of collecting independent
samples -- that would silently corrupt the matched-seed comparison the whole
forced-z protocol is built on. See experiments/forced_z_eval/protocol.py's
ForcedZProtocol.cell_seed: cell_seed = base_seed + 1000*opponent_index +
100*map_index, where opponent_index is the position of that opponent in
*that invocation's* --opponents list.

The safe axis is process-level parallelism, one opponent per subprocess: each
subprocess is an unmodified `run_forced_z_eval.py` invocation, given a single
opponent and a --base-seed compensated so cell_seed comes out numerically
identical to what the canonical single full-sweep run would have produced for
that opponent's TRUE index in the canonical opponent list. No changes to
protocol.py / runner.py / run_forced_z_eval.py are needed or made.

After all per-opponent partitions finish, this script:
  1. Concatenates their episode_results.csv files (same schema, disjoint
     opponent rows) into one merged run directory.
  2. Writes a run_manifest.json for the FULL canonical protocol (same
     checkpoint/base_seed/opponents/maps as an unpartitioned run would have
     produced -- env_reward_kwargs/max_decision_steps are checkpoint-derived,
     not opponent-list-derived, so they match automatically).
  3. Calls `run_forced_z_eval.py --from-run <merged_dir> --analyze-only` to
     regenerate stage_c/oracle/behavior/complementarity reports from the
     complete merged dataset -- unmodified analysis code, just fed the
     parallel-collected data.

Usage (mirrors the run_v6i26_phase2_confirm.py "after" eval call):
    python experiments/run_forced_z_eval_parallel.py \
        --checkpoint artifacts/v6i26_lro_niches_round1_seed1/final_v6i26_lro_z3_r1_25u_seed1.zip \
        --out-dir artifacts/v6i26_lro_niches_round1_seed1/phase2_confirm/forced_z_after_32 \
        --inherit-training-config --episodes 32 --device cuda --base-seed 1 \
        --oracle-metric win_margin --max-decision-steps 240 --progress-every 8 \
        --opponents OP6_IMMEDIATE_DUAL_RUSH OP7_DEEP_FORTRESS OP8_PROTECTED_CARRIER_ESCORT \
                    OP9_SPLIT_LANE_FEINT OP10_AGGRESSIVE_INTERCEPTOR OP11_ADAPTIVE_EXPLOITER \
                    OP12_LATE_CONVERTER \
        --maps map_b_split_lane map_b_split_lane_v2 \
        --max-concurrent 4
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.forced_z_eval.env_overrides import resolve_forced_z_env_overrides  # noqa: E402
from experiments.forced_z_eval.io import write_manifest  # noqa: E402
from experiments.forced_z_eval.protocol import DEFAULT_LATENTS, ForcedZProtocol  # noqa: E402

_RUN_FORCED_Z = str(SCRIPT_DIR / "run_forced_z_eval.py")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out-dir", required=True, help="Final merged run directory")
    p.add_argument("--episodes", type=int, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--opponents", nargs="+", required=True, help="Canonical order -- must match the order used for G_before")
    p.add_argument("--maps", nargs="+", required=True)
    p.add_argument("--base-seed", type=int, required=True)
    p.add_argument("--oracle-metric", choices=("return", "win_margin", "success"), default="win_margin")
    p.add_argument("--max-decision-steps", type=int, default=None)
    p.add_argument("--inherit-training-config", action="store_true")
    p.add_argument("--progress-every", type=int, default=8)
    p.add_argument("--max-concurrent", type=int, default=4, help="How many opponent partitions to run at once")
    p.add_argument("--python", default=None, help="Override interpreter (default: current sys.executable)")
    return p.parse_args()


def _partition_dir(out_dir: Path, opp_idx: int, opponent: str) -> Path:
    safe = opponent.replace("/", "_")
    return out_dir / "_parallel_partitions" / f"opp{opp_idx:02d}_{safe}"


def _build_partition_cmd(
    *,
    python_exe: str,
    checkpoint: str,
    partition_dir: Path,
    opponent: str,
    maps: list[str],
    episodes: int,
    compensated_base_seed: int,
    device: str,
    oracle_metric: str,
    max_decision_steps: Optional[int],
    inherit_training_config: bool,
    progress_every: int,
) -> list[str]:
    cmd = [
        python_exe,
        _RUN_FORCED_Z,
        "--checkpoint", checkpoint,
        "--out-dir", str(partition_dir),
        "--episodes", str(int(episodes)),
        "--device", device,
        "--base-seed", str(int(compensated_base_seed)),
        "--oracle-metric", oracle_metric,
        "--progress-every", str(int(progress_every)),
        "--opponents", opponent,
        "--maps", *maps,
    ]
    if max_decision_steps is not None:
        cmd.extend(["--max-decision-steps", str(int(max_decision_steps))])
    if inherit_training_config:
        cmd.append("--inherit-training-config")
    return cmd


def _run_partitions(jobs: list[tuple[str, list[str]]], *, max_concurrent: int) -> dict[str, int]:
    """Run (label, cmd) jobs with bounded concurrency. Returns {label: returncode}."""
    results: dict[str, int] = {}
    running: dict[str, subprocess.Popen] = {}
    pending = list(jobs)
    while pending or running:
        while pending and len(running) < max_concurrent:
            label, cmd = pending.pop(0)
            print(f"[parallel] launching {label}: {' '.join(cmd)}", flush=True)
            running[label] = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT))
        finished = [label for label, proc in running.items() if proc.poll() is not None]
        for label in finished:
            proc = running.pop(label)
            results[label] = int(proc.returncode)
            status = "OK" if proc.returncode == 0 else f"FAILED rc={proc.returncode}"
            print(f"[parallel] {label} finished: {status}", flush=True)
        if running:
            time.sleep(5)
    return results


def _merge_episode_csvs(partition_dirs: list[Path], merged_csv: Path) -> int:
    merged_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    all_rows: list[dict] = []
    for pdir in partition_dirs:
        csv_path = pdir / "episode_results.csv"
        if not csv_path.is_file():
            raise RuntimeError(f"Missing episode_results.csv for partition: {pdir}")
        with csv_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for key in reader.fieldnames or []:
                if key not in fieldnames:
                    fieldnames.append(key)
            all_rows.extend(reader)
    with merged_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)
    return len(all_rows)


def main() -> int:
    args = _parse_args()
    python_exe = args.python or sys.executable
    out_dir = Path(args.out_dir)
    opponents = list(args.opponents)
    maps = list(args.maps)

    max_steps, env_reward_kwargs, run_config_source = resolve_forced_z_env_overrides(
        checkpoint=str(args.checkpoint),
        run_config_path=None,
        inherit_training_config=bool(args.inherit_training_config),
        max_decision_steps=args.max_decision_steps,
    )

    jobs: list[tuple[str, list[str]]] = []
    partition_dirs: list[Path] = []
    for opp_idx, opponent in enumerate(opponents):
        pdir = _partition_dir(out_dir, opp_idx, opponent)
        partition_dirs.append(pdir)
        compensated_seed = int(args.base_seed) + 1000 * opp_idx
        cmd = _build_partition_cmd(
            python_exe=python_exe,
            checkpoint=str(args.checkpoint),
            partition_dir=pdir,
            opponent=opponent,
            maps=maps,
            episodes=int(args.episodes),
            compensated_base_seed=compensated_seed,
            device=str(args.device),
            oracle_metric=str(args.oracle_metric),
            max_decision_steps=args.max_decision_steps,
            inherit_training_config=bool(args.inherit_training_config),
            progress_every=int(args.progress_every),
        )
        jobs.append((f"opp{opp_idx:02d}_{opponent}", cmd))

    print(f"[parallel] {len(jobs)} opponent partitions, max_concurrent={args.max_concurrent}")
    print(f"[parallel] canonical opponent order (index -> seed offset): "
          f"{[(i, o, int(args.base_seed) + 1000 * i) for i, o in enumerate(opponents)]}")

    results = _run_partitions(jobs, max_concurrent=int(args.max_concurrent))
    failures = {label: rc for label, rc in results.items() if rc != 0}
    if failures:
        print(f"[parallel] FAILED partitions: {failures}")
        return 1

    merged_csv = out_dir / "episode_results.csv"
    n_rows = _merge_episode_csvs(partition_dirs, merged_csv)
    print(f"[parallel] merged {n_rows} episode rows -> {merged_csv}")

    full_protocol = ForcedZProtocol(
        checkpoint=str(args.checkpoint),
        opponents=tuple(opponents),
        maps=tuple(maps),
        latents=DEFAULT_LATENTS,
        episodes_per_cell=int(args.episodes),
        base_seed=int(args.base_seed),
        deterministic_actions=True,
        max_decision_steps=int(max_steps),
        env_reward_kwargs=dict(env_reward_kwargs),
        training_run_config=run_config_source,
        device=str(args.device),
        collect_behavior_mean=True,
        progress_every=int(args.progress_every),
    )
    write_manifest(out_dir, protocol=full_protocol, status="running", episode_count=n_rows)

    analyze_cmd = [
        python_exe,
        _RUN_FORCED_Z,
        "--from-run", str(out_dir),
        "--analyze-only",
        "--oracle-metric", str(args.oracle_metric),
    ]
    print(f"[parallel] regenerating reports: {' '.join(analyze_cmd)}", flush=True)
    rc = subprocess.call(analyze_cmd, cwd=str(PROJECT_ROOT))
    if rc != 0:
        print(f"[parallel] analyze-only step FAILED rc={rc}")
        return 1

    print(f"[parallel] done. Merged run ready at: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
