"""
Evaluates a trained adapted-ROA-Star checkpoint (PFSP or PFSP+exploiter, see
rl/train_ppo_roastar.py) against the same scripted opponents (OP3, held-out
OP4) and seed convention used for the existing Ours/Jacob et al./Self-play
comparison, writing rows in the exact CSV schema plot_eval_metrics.py already
consumes via --metrics-csv (see csv/eval_metrics_2v2_3v3_4v4_OP34.csv). No
changes to plot_eval_metrics.py are needed to visualize the result -- just:

    python plot/plot_eval_metrics.py --metrics-csv <out.csv> --modes 2v2

Usage:
    python plot/eval_roastar.py --checkpoint checkpoints_sb3/2v2/final_ppo_roastar_pfsp_2v2_seed42.zip \
        --method-label "ROA-Star (PFSP)" --setting 2v2 --agents 2 --episodes 100 \
        --out csv/eval_roastar_pfsp_2v2.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Any

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from eval_rollout import compute_aggregates, run_eval_episodes

# Must match plot_eval_metrics.py's load_metrics_csv()/table_rows schema exactly.
CSV_FIELDNAMES = [
    "setting", "method", "opponent",
    "success_rate_mean", "success_rate_std",
    "mean_steps_mean", "mean_steps_std",
    "collision_free_mean", "collision_free_std",
    "return_variance_mean", "return_variance_std",
    "coverage_efficiency_mean", "coverage_efficiency_std",
    "win_margin_mean", "win_margin_std",
    "time_to_first_score_mean", "time_to_first_score_std",
    "mean_inter_robot_dist_mean", "mean_inter_robot_dist_std",
]


def _row_from_aggregates(setting: str, method: str, opponent: str, agg: dict) -> dict:
    return {
        "setting": setting,
        "method": method,
        "opponent": opponent,
        "success_rate_mean": agg.get("success_rate", 0.0),
        "success_rate_std": agg.get("success_rate_std", 0.0),
        "mean_steps_mean": agg.get("mean_steps", 0.0),
        "mean_steps_std": agg.get("mean_steps_std", 0.0),
        "collision_free_mean": agg.get("collision_free_rate", 0.0),
        "collision_free_std": agg.get("collision_free_rate_std", 0.0),
        "return_variance_mean": agg.get("return_var", 0.0),
        "return_variance_std": agg.get("return_var_std", 0.0),
        "coverage_efficiency_mean": agg.get("coverage_efficiency", 0.0),
        "coverage_efficiency_std": agg.get("coverage_efficiency_std", 0.0),
        "win_margin_mean": agg.get("win_margin_mean", 0.0),
        "win_margin_std": agg.get("win_margin_std", 0.0),
        "time_to_first_score_mean": agg.get("time_to_first_score_mean", float("nan")),
        "time_to_first_score_std": agg.get("time_to_first_score_std", 0.0),
        "mean_inter_robot_dist_mean": agg.get("mean_inter_robot_dist_mean", float("nan")),
        "mean_inter_robot_dist_std": agg.get("mean_inter_robot_dist_std", 0.0),
    }


def evaluate_checkpoint(
    *,
    checkpoint_path: str,
    method_label: str,
    setting: str,
    n_agents: int,
    opponents: list[str],
    episodes: int,
    device: str,
    seed_base: int = 42,
) -> list[dict]:
    """
    Evaluate one checkpoint against each opponent in `opponents`, returning
    CSV-schema rows (one per opponent). Mirrors plot_eval_metrics.py's own
    live-eval loop -- same seed convention (seed_base, +1 for OP4) -- so
    numbers are directly comparable to the existing Ours/Jacob et al./
    Self-play rows without any extra normalization.
    """
    from game_field_gpu import GPUCTFVecEnv, GPUFieldConfig

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    rows: list[dict] = []
    for opp in opponents:
        opp_clean = str(opp).strip().upper()
        seed = seed_base + (1 if opp_clean == "OP4" else 0)
        cfg = GPUFieldConfig(
            n_envs=1,
            max_blue_agents=n_agents,
            max_red_agents=n_agents,
            max_decision_steps=400,
            aquaticus_profile=True,
            rules_profile="OURS",
            device=device,
            seed=seed,
        )
        env = GPUCTFVecEnv(cfg)
        try:
            print(f"[eval_roastar] {method_label} vs {opp_clean} ({episodes} episodes, seed={seed})...")
            episode_dicts = run_eval_episodes(checkpoint_path, env, episodes, device, opp_clean)
            agg = compute_aggregates(episode_dicts)
        finally:
            env.close()
        rows.append(_row_from_aggregates(setting, method_label, opp_clean, agg))
    return rows


def write_rows(rows: list[dict[str, Any]], out_path: str, *, append: bool) -> None:
    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    file_exists = os.path.isfile(out_path)
    mode = "a" if append and file_exists else "w"
    with open(out_path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        if mode == "w":
            writer.writeheader()
        writer.writerows(rows)
    verb = "Appended" if mode == "a" else "Wrote"
    print(f"[eval_roastar] {verb} {len(rows)} row(s) -> {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="Path to the trained .zip checkpoint")
    parser.add_argument("--method-label", required=True, help='e.g. "ROA-Star (PFSP)" or "ROA-Star (PFSP+Exploiter)"')
    parser.add_argument("--setting", required=True, help="Team-size label, e.g. 2v2, matching --agents")
    parser.add_argument("--agents", type=int, required=True)
    parser.add_argument("--opponents", nargs="+", default=["OP3", "OP4"])
    parser.add_argument("--episodes", type=int, default=100, help="Match the episode count used for Ours/Jacob/Self-play")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed-base", type=int, default=42)
    parser.add_argument("--out", required=True, help="CSV path to write/append (schema matches plot_eval_metrics.py)")
    parser.add_argument("--append", action="store_true", help="Append to --out if it exists, instead of overwriting")
    args = parser.parse_args()

    rows = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        method_label=args.method_label,
        setting=args.setting,
        n_agents=args.agents,
        opponents=args.opponents,
        episodes=args.episodes,
        device=args.device,
        seed_base=args.seed_base,
    )
    write_rows(rows, args.out, append=args.append)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
