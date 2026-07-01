#!/usr/bin/env python3
"""Repeatable benchmark tooling for Phase 6.1 telemetry performance profiling."""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Ensure AICTFProject is in sys.path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_SCRIPT_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

from rl.config.ppo_config import PPOConfig
from rl.custom_ppo import CustomPPOTrainer
from rl.train_ppo import (
    _clamp_runtime_config_for_team_size,
    _resolve_initial_opponent_and_phase,
)
from rl.training.env_factory import build_training_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Training Pipeline Performance")
    parser.add_argument(
        "--env-counts",
        nargs="+",
        default=["16,64,256"],
        help="Environment counts. Accepts comma-separated or space-separated values.",
    )
    parser.add_argument(
        "--telemetry-modes",
        nargs="+",
        default=["off,basic,full"],
        help="Telemetry modes. Accepts comma-separated or space-separated values.",
    )
    parser.add_argument(
        "--warmup-rollouts",
        type=int,
        default=2,
        help="Number of warm-up rollouts.",
    )
    parser.add_argument(
        "--measured-rollouts",
        type=int,
        default=10,
        help="Number of measured rollouts.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run benchmark on (cpu, cuda, cuda:0, etc.).",
    )
    parser.add_argument(
        "--checkpoint-path",
        "--checkpoint",
        type=str,
        default=None,
        dest="checkpoint_path",
        help="Optional path to a checkpoint zip file to load.",
    )
    parser.add_argument(
        "--map",
        "--map-layout",
        type=str,
        default="map_a_open",
        dest="map_layout",
        help="Map layout to benchmark.",
    )
    parser.add_argument(
        "--opponent",
        type=str,
        default="OP3",
        help="Fixed scripted opponent tag to benchmark against.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_results",
        help="Directory to save benchmark outputs.",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=64,
        help="Number of steps per rollout per environment.",
    )
    parser.add_argument(
        "--rollout-only",
        action="store_true",
        help="Only measure rollouts (skip PPO updates).",
    )
    return parser.parse_args()


def _parse_values(values: list[str]) -> list[str]:
    parsed: list[str] = []
    for value in values:
        parsed.extend(item.strip() for item in str(value).split(",") if item.strip())
    return parsed


def _ensure_device_available(cfg: PPOConfig) -> None:
    if not str(cfg.device).lower().startswith("cuda"):
        return
    try:
        torch.zeros(1, device=cfg.device)
    except RuntimeError as exc:
        print(f"[benchmark] CUDA unavailable ({exc}). Falling back to CPU.")
        cfg.device = "cpu"


def _apply_checkpoint_config(cfg: PPOConfig, checkpoint_path: str | None) -> None:
    if not checkpoint_path:
        return
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        print(f"[benchmark] checkpoint config hydration skipped: {exc}")
        return
    raw_cfg = payload.get("cfg") if isinstance(payload, dict) else None
    if not isinstance(raw_cfg, dict):
        return
    for key, value in raw_cfg.items():
        if hasattr(cfg, key):
            try:
                setattr(cfg, key, value)
            except Exception:
                pass


def run_benchmark_matrix(args: argparse.Namespace) -> None:
    env_counts = [int(x) for x in _parse_values(args.env_counts)]
    telemetry_modes = _parse_values(args.telemetry_modes)
    device = args.device

    print("=" * 60)
    print(f"STARTING TELEMETRY BENCHMARK MATRIX")
    print(f"  Telemetry modes: {telemetry_modes}")
    print(f"  Env counts:      {env_counts}")
    print(f"  Device:          {device}")
    print(f"  Map layout:      {args.map_layout}")
    print(f"  Opponent:        {args.opponent}")
    print(f"  Rollout steps:   {args.n_steps}")
    print(f"  Measured/Warmup: {args.measured_rollouts}/{args.warmup_rollouts}")
    print("=" * 60)

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    raw_samples_file = output_path / "raw_samples.csv"
    raw_headers = [
        "timestamp",
        "telemetry_mode",
        "env_count",
        "rollout_index",
        "rollout_duration_seconds",
        "rollout_transitions",
        "rollout_transitions_per_second",
        "optimization_duration_seconds",
        "optimization_samples",
        "optimization_samples_per_second",
        "peak_allocated_cuda_bytes",
        "peak_reserved_cuda_bytes",
    ]
    with open(raw_samples_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(raw_headers)

    results = []

    for mode in telemetry_modes:
        for env_count in env_counts:
            print(f"\n--- Running telemetry_mode={mode}, env_count={env_count} ---")
            
            # Setup Config
            cfg = PPOConfig()
            _apply_checkpoint_config(cfg, args.checkpoint_path)
            cfg.seed = 42
            cfg.device = device
            cfg.n_envs = env_count
            cfg.n_steps = args.n_steps
            cfg.batch_size = max(1024, env_count * args.n_steps)
            cfg.n_epochs = 1
            cfg.training_telemetry_mode = mode
            cfg.load_path = args.checkpoint_path
            cfg.checkpoint_dir = str(output_path)
            cfg.mode = "FIXED_OPPONENT"
            cfg.fixed_opponent_tag = str(args.opponent).upper()
            cfg.opponent_randomize = False
            cfg.opponent_pool = (str(args.opponent).upper(),)
            cfg.opponent_pool_weights = ()
            cfg.map_layout = str(args.map_layout).lower()
            cfg.use_latent_strategy = True  # Enable latent mode for telemetry coverage
            cfg.enable_metrics_csv = False
            cfg.gpu_native_env = True
            
            # Temporary JSONL output paths to avoid mixing metrics
            cfg.training_events_jsonl_path = str(output_path / f"events_{mode}_{env_count}.jsonl")
            cfg.performance_samples_path = str(output_path / f"perf_{mode}_{env_count}.csv")
            cfg.performance_summary_path = str(output_path / f"summary_{mode}_{env_count}.json")

            max_agents = max(1, int(getattr(cfg, "max_blue_agents", 2)))
            curriculum, initial_phase, initial_opponent_tag = _resolve_initial_opponent_and_phase(cfg, max_agents)
            _clamp_runtime_config_for_team_size(cfg, max_agents)
            _ensure_device_available(cfg)

            # Build Environment
            env = build_training_env(
                cfg,
                initial_phase=initial_phase,
                initial_opponent_tag=initial_opponent_tag,
            )

            # Build Trainer
            trainer = CustomPPOTrainer(
                env=env,
                cfg=cfg,
                learning_rate=3e-4,
                clip_range=0.2,
                ent_coef=0.01,
                n_epochs=1,
                batch_size=cfg.batch_size,
                value_clip_range=0.2,
                curriculum=curriculum,
            )

            # Load checkpoint if requested
            if args.checkpoint_path:
                print(f"Loading checkpoint: {args.checkpoint_path}")
                trainer.load(args.checkpoint_path)

            # Warm-up
            print("Running warm-up rollouts...")
            for _ in range(args.warmup_rollouts):
                rollout = trainer.collect_rollout()
                if not args.rollout_only:
                    trainer.update(rollout, total_timesteps=1000000)
                    trainer._updates_completed += 1

            # Measured rollouts
            print("Running measured rollouts...")
            rollout_durs = []
            opt_durs = []
            allocated_peaks = []
            reserved_peaks = []

            for r_idx in range(args.measured_rollouts):
                if device == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                
                t_start = time.perf_counter()
                rollout = trainer.collect_rollout()
                if device == "cuda":
                    torch.cuda.synchronize()
                t_end = time.perf_counter()
                rollout_dur = t_end - t_start
                rollout_durs.append(rollout_dur)
                
                transitions = env_count * args.n_steps
                rollout_tps = transitions / rollout_dur

                opt_dur = 0.0
                opt_sps = 0.0
                if not args.rollout_only:
                    if device == "cuda":
                        torch.cuda.synchronize()
                    t_start_opt = time.perf_counter()
                    trainer.update(rollout, total_timesteps=1000000)
                    trainer._updates_completed += 1
                    if device == "cuda":
                        torch.cuda.synchronize()
                    t_end_opt = time.perf_counter()
                    opt_dur = t_end_opt - t_start_opt
                    opt_sps = transitions / opt_dur
                opt_durs.append(opt_dur)

                # CUDA Memory snapshot
                allocated_peak = None
                reserved_peak = None
                if device == "cuda":
                    allocated_peak = torch.cuda.max_memory_allocated()
                    reserved_peak = torch.cuda.max_memory_reserved()
                    allocated_peaks.append(allocated_peak)
                    reserved_peaks.append(reserved_peak)

                # Write to raw samples CSV
                with open(raw_samples_file, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        time.time(),
                        mode,
                        env_count,
                        r_idx,
                        rollout_dur,
                        transitions,
                        rollout_tps,
                        opt_dur,
                        transitions,
                        opt_sps,
                        allocated_peak or 0,
                        reserved_peak or 0,
                    ])

                print(
                    f"  Rollout {r_idx}: duration={rollout_dur:.4f}s "
                    f"({rollout_tps:.1f} transitions/s)"
                    + (f", Opt: duration={opt_dur:.4f}s ({opt_sps:.1f} samples/s)" if not args.rollout_only else "")
                )

            # Close environment
            try:
                env.close()
            except Exception:
                pass
            trainer.telemetry.close_e3_step_telemetry()

            # Compile matrix cell stats
            rollout_durs = np.array(rollout_durs)
            opt_durs = np.array(opt_durs)
            transitions = env_count * args.n_steps

            rollout_tps_list = transitions / rollout_durs
            median_rollout_tps = float(np.median(rollout_tps_list))
            p95_rollout_tps = float(np.percentile(rollout_tps_list, 95))

            median_opt_sps = None
            p95_opt_sps = None
            if not args.rollout_only:
                opt_sps_list = transitions / opt_durs
                median_opt_sps = float(np.median(opt_sps_list))
                p95_opt_sps = float(np.percentile(opt_sps_list, 95))

            peak_alloc = int(np.max(allocated_peaks)) if allocated_peaks else None
            peak_res = int(np.max(reserved_peaks)) if reserved_peaks else None

            results.append({
                "telemetry_mode": mode,
                "env_count": env_count,
                "rollout": {
                    "median_duration_seconds": float(np.median(rollout_durs)),
                    "p95_duration_seconds": float(np.percentile(rollout_durs, 95)),
                    "median_transitions_per_second": median_rollout_tps,
                    "p95_transitions_per_second": p95_rollout_tps,
                },
                "optimization": {
                    "median_duration_seconds": float(np.median(opt_durs)) if not args.rollout_only else 0.0,
                    "p95_duration_seconds": float(np.percentile(opt_durs, 95)) if not args.rollout_only else 0.0,
                    "median_samples_per_second": median_opt_sps,
                    "p95_samples_per_second": p95_opt_sps,
                },
                "peak_allocated_cuda_bytes": peak_alloc,
                "peak_reserved_cuda_bytes": peak_res,
            })

    # Save summary JSON
    summary_file = output_path / "benchmark_summary.json"
    summary_data = {
        "benchmark_timestamp": time.time(),
        "device": device,
        "map_layout": args.map_layout,
        "opponent": args.opponent,
        "checkpoint_path": args.checkpoint_path,
        "rollout_steps": args.n_steps,
        "measured_rollouts": args.measured_rollouts,
        "warmup_rollouts": args.warmup_rollouts,
        "results": results,
    }
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    # Save manifest JSON (Track M schema)
    manifest_file = output_path / "benchmark_manifest.json"
    manifest_data = {
        "manifest_version": 1,
        "device_name": device,
        "map_layout": args.map_layout,
        "opponent": args.opponent,
        "checkpoint_path": args.checkpoint_path,
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "timestamp": time.time(),
        "matrix": [
            {
                "telemetry_mode": res["telemetry_mode"],
                "env_count": res["env_count"],
                "median_rollout_tps": res["rollout"]["median_transitions_per_second"],
                "median_opt_sps": res["optimization"]["median_samples_per_second"],
                "peak_allocated_cuda_bytes": res["peak_allocated_cuda_bytes"],
            }
            for res in results
        ]
    }
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    # Save human-readable summary TXT
    summary_txt = output_path / "benchmark_summary.txt"
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("TELEMETRY OBSERVABILITY BENCHMARK RESULTS SUMMARY\n")
        f.write("=" * 70 + "\n")
        f.write(f"Timestamp:  {time.ctime()}\n")
        f.write(f"Device:     {device}\n")
        f.write(f"Steps:      {args.n_steps}\n\n")

        f.write(f"{'Mode':<10} | {'Envs':<6} | {'Rollout TPS (Med)':<20} | {'Opt SPS (Med)':<15} | {'Peak CUDA (MB)':<15}\n")
        f.write("-" * 70 + "\n")
        for res in results:
            m_bytes = (res["peak_allocated_cuda_bytes"] / (1024 * 1024)) if res["peak_allocated_cuda_bytes"] else 0.0
            opt_str = f"{res['optimization']['median_samples_per_second']:.1f}" if res['optimization']['median_samples_per_second'] else "N/A"
            f.write(
                f"{res['telemetry_mode']:<10} | "
                f"{res['env_count']:<6} | "
                f"{res['rollout']['median_transitions_per_second']:<20.1f} | "
                f"{opt_str:<15} | "
                f"{m_bytes:<15.1f}\n"
            )
        f.write("=" * 70 + "\n")

    print("\nBenchmark completed successfully!")
    print(f"Results written to: {output_path.resolve()}")


if __name__ == "__main__":
    run_benchmark_matrix(parse_args())

