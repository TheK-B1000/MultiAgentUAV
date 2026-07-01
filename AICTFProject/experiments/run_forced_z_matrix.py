#!/usr/bin/env python3
"""Run evaluation matrix for 4v4 latent checkpoint.
Evaluates the checkpoint against OP3, OP4, OP5_RUSHER, and OP6_TURTLE
under normal routing and forced/clamped latent strategies (z=0, 1, 2, 3).
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
import time
from typing import Any

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DEFAULT_CHECKPOINT = os.path.join(
    "checkpoints",
    "4v4",
    "final_latent_v3d_delayedanneal_300k_800k_bucketopp_1m_4v4.zip"
)


def parse_progress_from_log(log_path: str) -> tuple[str, str] | None:
    """Parse current opponent and episode progress from a log file.
    
    Returns:
        (opponent, progress_str) or None
    """
    if not os.path.exists(log_path):
        return None
        
    opponent = "Unknown"
    progress = "0/0"
    
    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
            
        # Look for the last run line to identify the opponent
        # Format: [eval_checkpoint] ... map=eval vs OP3: 50 episode(s)
        for line in reversed(lines):
            m = re.search(r"vs\s+([A-Za-z0-9_]+):", line)
            if m:
                opponent = m.group(1)
                break
                
        # Look for the last episode progress line
        # Format:   episode 12/50
        for line in reversed(lines):
            m = re.search(r"episode\s+(\d+/\d+)", line)
            if m:
                progress = m.group(1)
                break
                
        return opponent, progress
    except Exception:
        return None


def run_process_matrix(
    checkpoint: str,
    opponents: list[str],
    episodes: int,
    device: str,
    parallel_limit: int,
) -> list[dict[str, Any]]:
    # Create logs directory
    log_dir = os.path.join(PROJECT_ROOT, "logs", "eval_z_matrix")
    os.makedirs(log_dir, exist_ok=True)
    
    # 5 configurations: normal, z0, z1, z2, z3
    configs = [
        {"label": "v3d_normal", "fixed_z": None},
        {"label": "v3d_z0", "fixed_z": 0},
        {"label": "v3d_z1", "fixed_z": 1},
        {"label": "v3d_z2", "fixed_z": 2},
        {"label": "v3d_z3", "fixed_z": 3},
    ]
    
    # Python executable path
    python_exe = sys.executable or "python"
    
    # Construct subprocesses details
    active_jobs = []
    pending_jobs = []
    
    for cfg in configs:
        label = cfg["label"]
        fixed_z = cfg["fixed_z"]
        
        cmd = [
            python_exe,
            "plot/eval_checkpoint.py",
            "--checkpoint", checkpoint,
            "--opponents", *opponents,
            "--map-sets", "eval",
            "--episodes", str(episodes),
            "--device", device,
            "--deterministic",
            "--label", label,
        ]
        if fixed_z is not None:
            cmd.extend(["--fixed-latent-id", str(fixed_z)])
            
        log_file = os.path.join(log_dir, f"eval_{label}.log")
        
        pending_jobs.append({
            "label": label,
            "fixed_z": fixed_z,
            "cmd": cmd,
            "log_file": log_file,
            "proc": None,
            "status": "pending",
            "start_time": None,
            "duration": 0.0
        })
        
    print(f"Starting forced-z matrix evaluation (Parallel Limit: {parallel_limit})")
    print(f"Checkpoint: {checkpoint}")
    print(f"Opponents: {opponents}")
    print(f"Episodes per configuration/opponent: {episodes}")
    print(f"Logs will be written to: {log_dir}\n")
    
    completed_jobs = []
    
    while pending_jobs or active_jobs:
        # Check active jobs and update their progress
        still_active = []
        for job in active_jobs:
            proc = job["proc"]
            ret = proc.poll()
            
            if ret is not None:
                # Job completed
                job["status"] = "completed" if ret == 0 else "failed"
                job["duration"] = time.time() - job["start_time"]
                completed_jobs.append(job)
                print(f"\n[DONE] Configuration {job['label']} finished in {job['duration']:.1f}s with status: {job['status']}")
            else:
                # Job still running
                still_active.append(job)
                
        active_jobs = still_active
        
        # Start new jobs if we have capacity
        while len(active_jobs) < parallel_limit and pending_jobs:
            job = pending_jobs.pop(0)
            job["start_time"] = time.time()
            job["status"] = "running"
            
            log_f = open(job["log_file"], "w", encoding="utf-8")
            job["proc"] = subprocess.Popen(
                job["cmd"],
                cwd=PROJECT_ROOT,
                stdout=log_f,
                stderr=subprocess.STDOUT
            )
            active_jobs.append(job)
            print(f"[START] Configuration {job['label']} (PID: {job['proc'].pid}) -> {job['log_file']}")
            
        # Display progress of running jobs
        if active_jobs:
            prog_strs = []
            for job in active_jobs:
                prog = parse_progress_from_log(job["log_file"])
                if prog:
                    opp, ep_prog = prog
                    prog_strs.append(f"{job['label']}: {opp} ({ep_prog})")
                else:
                    prog_strs.append(f"{job['label']}: starting...")
            sys.stdout.write("\rProgress: " + " | ".join(prog_strs) + "   ")
            sys.stdout.flush()
            
        time.sleep(5)
        
    print("\nAll subprocesses finished. Loading CSV files and summarizing results...\n")
    return completed_jobs


def parse_results(opponents: list[str]) -> list[dict[str, Any]]:
    # Find CSV directory
    csv_dir = os.path.join(PROJECT_ROOT, "csv")
    results = []
    
    # 5 configurations
    configs = [
        {"label": "v3d_normal", "display_name": "Normal (Router)"},
        {"label": "v3d_z0", "display_name": "Forced z=0"},
        {"label": "v3d_z1", "display_name": "Forced z=1"},
        {"label": "v3d_z2", "display_name": "Forced z=2"},
        {"label": "v3d_z3", "display_name": "Forced z=3"},
    ]
    
    for cfg in configs:
        label = cfg["label"]
        display_name = cfg["display_name"]
        
        # CSV filename template: eval_{label}_4v4_aggregate.csv
        csv_file = os.path.join(csv_dir, f"eval_{label}_4v4_aggregate.csv")
        
        if not os.path.exists(csv_file):
            print(f"Warning: Aggregate file not found: {csv_file}")
            continue
            
        try:
            with open(csv_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Filter for 'eval' map_set and check if opponent is in our target opponents list
                    # (Note: eval_checkpoint.py normalizes opponent names to UPPERCASE)
                    opponent = row.get("opponent", "").strip().upper()
                    map_set = row.get("map_set", "").strip().lower()
                    
                    # Match normalized opponent name
                    matched_opponent = None
                    for o in opponents:
                        if o.upper() == opponent or (o == "OP5_RUSHER" and opponent == "OP5"):
                            matched_opponent = o
                            break
                            
                    if matched_opponent and map_set == "eval":
                        results.append({
                            "config": label,
                            "config_display": display_name,
                            "opponent": matched_opponent,
                            "episodes": int(row.get("episodes", 0)),
                            "wins": int(row.get("wins", 0)),
                            "losses": int(row.get("losses", 0)),
                            "draws": int(row.get("draws", 0)),
                            "win_rate": float(row.get("success_rate", 0.0)),
                            "win_rate_std": float(row.get("success_rate_std", 0.0))
                        })
        except Exception as e:
            print(f"Error parsing CSV {csv_file}: {e}")
            
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate 4v4 checkpoint under forced z configurations.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="Path to checkpoint zip file")
    parser.add_argument("--episodes", type=int, default=50, help="Number of episodes per configuration")
    parser.add_argument("--parallel", type=int, default=3, help="Number of concurrent subprocesses (default 3 to fit memory)")
    parser.add_argument("--device", default="cuda", help="Device (cuda or cpu)")
    parser.add_argument("--opponents", nargs="+", default=["OP3", "OP4", "OP5_RUSHER", "OP6_TURTLE"], help="List of opponents to evaluate")
    args = parser.parse_args()
    
    checkpoint_path = os.path.abspath(args.checkpoint)
    if not os.path.exists(checkpoint_path):
        print(f"[ERROR] Checkpoint file does not exist: {checkpoint_path}")
        sys.exit(1)
        
    start_time = time.time()
    
    # Run the evaluations
    run_process_matrix(
        checkpoint=checkpoint_path,
        opponents=args.opponents,
        episodes=args.episodes,
        device=args.device,
        parallel_limit=args.parallel
    )
    
    # Parse the output CSVs
    results = parse_results(args.opponents)
    
    if not results:
        print("No evaluation results were successfully parsed. Check the logs under logs/eval_z_matrix/.")
        sys.exit(1)
        
    # Generate the Markdown output table
    markdown_lines = []
    markdown_lines.append("# Forced Latent Strategy (z) Evaluation Matrix (4v4)")
    markdown_lines.append(f"\n* **Checkpoint:** `{os.path.basename(checkpoint_path)}`")
    markdown_lines.append(f"* **Episodes:** {args.episodes} per condition (held-out `eval` maps)")
    markdown_lines.append(f"* **Device:** `{args.device}`")
    markdown_lines.append(f"* **Run completed at:** {time.strftime('%Y-%m-%d %H:%M:%S')}")
    markdown_lines.append(f"* **Total Execution Time:** {time.time() - start_time:.1f}s\n")
    
    # Group results by opponent
    by_opp = {}
    for r in results:
        by_opp.setdefault(r["opponent"], []).append(r)
        
    # Generate markdown table for each opponent
    markdown_lines.append("| Opponent | Latent Configuration | Wins | Losses | Draws | Win Rate (%) | Std Error (%) |")
    markdown_lines.append("|---|---|---|---|---|---|---|")
    
    # Ordering of configs in the table
    config_order = ["v3d_normal", "v3d_z0", "v3d_z1", "v3d_z2", "v3d_z3"]
    
    for opp in args.opponents:
        opp_results = by_opp.get(opp, [])
        # Sort by config order
        opp_results_sorted = sorted(
            opp_results,
            key=lambda x: config_order.index(x["config"]) if x["config"] in config_order else 99
        )
        
        # Mark the maximum win rate among forced-z options
        forced_z_results = [r for r in opp_results_sorted if r["config"] != "v3d_normal"]
        max_forced_wr = max([r["win_rate"] for r in forced_z_results]) if forced_z_results else -1.0
        
        for r in opp_results_sorted:
            config_str = r["config_display"]
            wr_str = f"{r['win_rate']:.1f}%"
            
            # Format row
            is_normal = r["config"] == "v3d_normal"
            is_best_forced = not is_normal and r["win_rate"] == max_forced_wr and max_forced_wr > -1.0
            
            if is_best_forced:
                wr_str = f"**{wr_str}** (Best Forced)"
            elif is_normal:
                wr_str = f"*{wr_str}* (Router)"
                
            # Binomial standard error: sqrt(p*(1-p)/N) * 100
            p = r["win_rate"] / 100.0
            se = 100.0 * ((p * (1.0 - p)) / r["episodes"]) ** 0.5 if r["episodes"] > 0 else 0.0
            
            markdown_lines.append(
                f"| {opp} | {config_str} | {r['wins']} | {r['losses']} | {r['draws']} | {wr_str} | ±{se:.1f}% |"
            )
            
    markdown_text = "\n".join(markdown_lines)
    
    # Write to file
    out_md_path = os.path.join(PROJECT_ROOT, "csv", "forced_z_matrix_results.md")
    with open(out_md_path, "w", encoding="utf-8") as f:
        f.write(markdown_text)
        
    print("=" * 80)
    print("EVALUATION RESULTS SUMMARY")
    print("=" * 80)
    print(markdown_text)
    print("=" * 80)
    print(f"Full markdown report saved to: {out_md_path}")


if __name__ == "__main__":
    main()
