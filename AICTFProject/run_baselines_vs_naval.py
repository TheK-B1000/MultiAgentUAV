"""
Run all baselines vs naval opponents to see which performs best.

Naval opponents (NAVAL_DEFENDER, NAVAL_RUSHER, NAVAL_BALANCED) are held-out scripted
opponents with naval-style behavior; eval uses OP3 stress (physics, sensors) so the
test is realistic. Not used in training, so win rates separate baselines.

Usage:
  python run_baselines_vs_naval.py --episodes 30 --headless
  python run_baselines_vs_naval.py --baselines noleague selfplay fixed league --episodes 50 --out-csv results.csv
"""

from __future__ import annotations

import argparse
import os
import sys
import csv
from typing import Dict, List, Any, Optional

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

from ctfviewer import CTFViewer, BASELINE_MODEL_PATHS


# Which baselines to run (keys into BASELINE_MODEL_PATHS or label -> path)
DEFAULT_BASELINES = ["curriculum_no_league", "self_play", "fixed_op3", "curriculum_league"]


def run_baselines_vs_naval(
    baseline_paths: Dict[str, str],
    num_episodes_per_opp: int = 30,
    headless: bool = True,
    out_csv: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    For each baseline (label -> model path), run evaluate_naval_opponents.
    Returns results[baseline_label][opponent] = {win_rate, wins, losses, draws}.
    """
    results: Dict[str, Dict[str, Any]] = {}
    for label, path in baseline_paths.items():
        if not path or not os.path.isfile(path):
            print(f"[SKIP] {label}: model not found at {path!r}")
            results[label] = {}
            continue
        print(f"\n{'='*60}\nBaseline: {label}\n{'='*60}")
        try:
            viewer = CTFViewer(ppo_model_path=path, viewer_use_obs_builder=True)
            if not getattr(viewer.blue_ppo_team, "model_loaded", False):
                print(f"[ERROR] {label}: failed to load model")
                results[label] = {}
                continue
            viewer._apply_blue_mode("PPO")
            out = viewer.evaluate_naval_opponents(
                num_episodes_per_opp=num_episodes_per_opp,
                eval_model="ppo",
                headless=headless,
                save_dir=None,
            )
            results[label] = out.get("results_per_opponent", {})
        except Exception as e:
            print(f"[ERROR] {label}: {e}")
            import traceback
            traceback.print_exc()
            results[label] = {}

    # Print comparison table
    naval_opps = ("NAVAL_DEFENDER", "NAVAL_RUSHER", "NAVAL_BALANCED")
    print("\n" + "=" * 70)
    print("BASELINES VS NAVAL OPPONENTS (held-out, physics on)")
    print("=" * 70)
    col_w = max(10, max(len(l) for l in results.keys()))
    header = "Baseline".ljust(col_w) + "".join(o.rjust(col_w) for o in naval_opps)
    print(header)
    print("-" * len(header))
    for label in results:
        row = label.ljust(col_w)
        for opp in naval_opps:
            r = results.get(label, {}).get(opp, {})
            wr = r.get("win_rate", 0.0)
            row += f"{wr:.0%}".rjust(col_w)
        print(row)
    print("=" * 70)

    if out_csv:
        os.makedirs(os.path.dirname(os.path.abspath(out_csv)) or ".", exist_ok=True)
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["baseline", "opponent", "win_rate", "wins", "losses", "draws"])
            for label in results:
                for opp in naval_opps:
                    r = results.get(label, {}).get(opp, {})
                    w.writerow([
                        label, opp,
                        r.get("win_rate", ""),
                        r.get("wins", ""),
                        r.get("losses", ""),
                        r.get("draws", ""),
                    ])
        print(f"Wrote {out_csv}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run all baselines vs naval opponents")
    parser.add_argument(
        "--baselines",
        type=str,
        nargs="+",
        default=DEFAULT_BASELINES,
        help=f"Baseline keys (default: {DEFAULT_BASELINES}). Keys: {list(BASELINE_MODEL_PATHS.keys())}",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=30,
        help="Episodes per naval opponent per baseline (default: 30)",
    )
    parser.add_argument("--headless", action="store_true", help="Run without display")
    parser.add_argument("--out-csv", type=str, default=None, help="Save results to CSV")
    args = parser.parse_args()

    baseline_paths = {}
    for key in args.baselines:
        k = key.lower().strip()
        if k in BASELINE_MODEL_PATHS:
            baseline_paths[key] = BASELINE_MODEL_PATHS[k]
        else:
            # Assume it's a path
            if os.path.isfile(k):
                baseline_paths[key] = k
            else:
                print(f"[WARN] Unknown baseline key or missing file: {key!r}; skipping.")
    if not baseline_paths:
        print("[ERROR] No valid baselines. Use --baselines with keys from BASELINE_MODEL_PATHS or paths.")
        sys.exit(1)

    run_baselines_vs_naval(
        baseline_paths=baseline_paths,
        num_episodes_per_opp=args.episodes,
        headless=args.headless,
        out_csv=args.out_csv,
    )


if __name__ == "__main__":
    main()
