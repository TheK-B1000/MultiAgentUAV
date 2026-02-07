"""
Cross-play tournament: round-robin where each blue model plays each other model as red.

Answers: does league+Elo produce a policy that beats other learned policies, not just scripts?

Usage:
  python cross_play_tournament.py --models path/noleague.zip path/selfplay.zip path/fixed.zip path/league.zip --labels noleague selfplay fixed league --episodes 20
  python cross_play_tournament.py --models checkpoints_sb3/final_ppo_noleague.zip checkpoints_sb3/final_ppo_selfplay.zip --episodes 30 --headless

Outputs:
  - Head-to-head win rate matrix (blue rows, red columns)
  - Elo ratings derived from the matrix
  - Optional CSV of matrix and ratings
"""

from __future__ import annotations

import argparse
import os
import sys
import csv
from typing import List, Optional, Tuple, Dict, Any

# Project root
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

import numpy as np

from ctfviewer import CTFViewer


def run_match(
    blue_path: str,
    red_path: str,
    num_episodes: int,
    base_seed: int,
    headless: bool = True,
) -> Tuple[int, int, int]:
    """
    Run num_episodes with blue=blue_path, red=red_path. Use distinct seeds per episode.
    Returns (wins, losses, draws).
    """
    episode_seeds = [base_seed + k for k in range(num_episodes)]
    viewer = CTFViewer(ppo_model_path=blue_path, viewer_use_obs_builder=True)
    if not getattr(viewer.blue_ppo_team, "model_loaded", False):
        raise RuntimeError(f"Failed to load blue model: {blue_path}")
    summary = viewer.evaluate_model(
        num_episodes=num_episodes,
        headless=headless,
        red_model_path=red_path,
        episode_seeds=episode_seeds,
    )
    if not summary:
        return 0, 0, num_episodes
    return (
        int(summary.get("wins", 0)),
        int(summary.get("losses", 0)),
        int(summary.get("draws", 0)),
    )


def compute_elo(
    win_rate_matrix: np.ndarray,
    K: float = 32.0,
    max_iters: int = 100,
    initial_elo: float = 1500.0,
) -> np.ndarray:
    """
    Compute Elo ratings from head-to-head win rate matrix.
    win_rate_matrix[i,j] = fraction of games where blue i beat red j (0..1).
    Returns 1D array of Elo ratings (same order as rows/cols).
    """
    n = win_rate_matrix.shape[0]
    r = np.full(n, initial_elo, dtype=np.float64)
    for _ in range(max_iters):
        dr = np.zeros(n, dtype=np.float64)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                exp_ij = 1.0 / (1.0 + 10.0 ** ((r[j] - r[i]) / 400.0))
                actual = float(win_rate_matrix[i, j])
                diff = K * (actual - exp_ij)
                dr[i] += diff
                dr[j] -= diff
        r += dr
        if np.max(np.abs(dr)) < 0.01:
            break
    r = r - np.mean(r) + initial_elo
    return r


def run_tournament(
    model_paths: List[str],
    labels: Optional[List[str]] = None,
    num_episodes_per_match: int = 20,
    base_seed: int = 42,
    headless: bool = True,
    out_csv: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Round-robin: for each ordered pair (i,j) with i != j, blue=model_i, red=model_j.
    Uses 10--30 distinct seeds per matchup (num_episodes_per_match) and shuffled spawns via seed.
    """
    n = len(model_paths)
    if n < 2:
        raise ValueError("Need at least 2 models for tournament")
    if labels is None:
        labels = [os.path.basename(p).replace(".zip", "") for p in model_paths]
    if len(labels) != n:
        labels = labels[:n] if len(labels) > n else labels + [f"model_{i}" for i in range(len(labels), n)]

    wins = np.zeros((n, n), dtype=np.int64)
    losses = np.zeros((n, n), dtype=np.int64)
    draws = np.zeros((n, n), dtype=np.int64)

    total_matches = n * (n - 1)
    current = 0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            current += 1
            print(f"[{current}/{total_matches}] Blue={labels[i]} vs Red={labels[j]} ({num_episodes_per_match} episodes, seeds {base_seed}..{base_seed + num_episodes_per_match - 1})")
            try:
                w, l, d = run_match(
                    model_paths[i],
                    model_paths[j],
                    num_episodes=num_episodes_per_match,
                    base_seed=base_seed + (i * n + j) * 1000,
                    headless=headless,
                )
                wins[i, j] = w
                losses[i, j] = l
                draws[i, j] = d
                print(f"    -> {w}W / {l}L / {d}D")
            except Exception as e:
                print(f"    -> ERROR: {e}")
                draws[i, j] = num_episodes_per_match

    # Win rate matrix (P(blue i beats red j))
    total_games = wins + losses + draws
    win_rate = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if i == j:
                win_rate[i, j] = 0.5
            elif total_games[i, j] > 0:
                win_rate[i, j] = wins[i, j] / total_games[i, j]
            else:
                win_rate[i, j] = 0.5

    elo = compute_elo(win_rate)

    # Print results
    print("\n" + "=" * 70)
    print("CROSS-PLAY TOURNAMENT: Head-to-Head Win Rate (Blue rows, Red columns)")
    print("=" * 70)
    col_width = max(8, max(len(l) for l in labels))
    header = "Blue \\ Red".ljust(col_width) + "".join(labels[k].rjust(col_width) for k in range(n))
    print(header)
    print("-" * len(header))
    for i in range(n):
        row = labels[i].ljust(col_width)
        for j in range(n):
            if i == j:
                cell = "  --  "
            else:
                wr = win_rate[i, j]
                cell = f"{wr:.0%}".rjust(col_width)
            row += cell
        print(row)

    print("\n" + "=" * 70)
    print("Elo ratings (mean 1500)")
    print("=" * 70)
    order = np.argsort(-elo)
    for idx in order:
        print(f"  {labels[idx]:{col_width}}  {elo[idx]:.1f}")
    print()

    if out_csv:
        os.makedirs(os.path.dirname(os.path.abspath(out_csv)) or ".", exist_ok=True)
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["blue_label", "red_label", "wins", "losses", "draws", "win_rate"])
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    w.writerow([
                        labels[i], labels[j],
                        int(wins[i, j]), int(losses[i, j]), int(draws[i, j]),
                        f"{win_rate[i, j]:.4f}",
                    ])
            w.writerow([])
            w.writerow(["label", "elo"])
            for i in range(n):
                w.writerow([labels[i], f"{elo[i]:.2f}"])
        print(f"Wrote {out_csv}")

    return {
        "labels": labels,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": win_rate,
        "elo": elo,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Cross-play tournament: round-robin blue vs red (learned policies)"
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        required=True,
        help="Paths to PPO .zip models (order = index in matrix)",
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs="+",
        default=None,
        help="Short names for each model (default: basename without .zip)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
        help="Episodes per matchup (use 10--30 for distinct seeds and varied spawns; default 20)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base RNG seed; each matchup gets distinct seeds",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without display",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default=None,
        help="Save win matrix and Elo to CSV",
    )
    args = parser.parse_args()

    for p in args.models:
        if not os.path.isfile(p):
            print(f"[ERROR] Model not found: {p}")
            sys.exit(1)

    run_tournament(
        model_paths=args.models,
        labels=args.labels,
        num_episodes_per_match=args.episodes,
        base_seed=args.seed,
        headless=args.headless,
        out_csv=args.out_csv,
    )


if __name__ == "__main__":
    main()
