#!/usr/bin/env python3
"""
Plot evaluation metrics by category: Performance, Coordination, Robustness, Stability, Specialization, Robotics.

Uses the same default checkpoints as plot_*_winrate.py (Ours / Jacob et al. / Self-play).
Runs 2v2, 3v3, 4v4, 5v5, and 8v8 eval with GPUCTFVecEnv (optional OP3/OP4) and collects:
  - Performance: success rate, mean steps to completion
  - Coordination: coverage efficiency (zone_coverage), coordination proxy (collision-free)
  - Robustness: generalization (success vs OP3 vs OP4); default eval uses both opponents
  - Stability: return variance across episodes
  - Specialization: placeholder (requires action logging)
  - Robotics: safety (collision-free rate); energy/path N/A

Usage:
  python plot_eval_metrics.py [--league PATH] [--paper PATH] [--selfplay PATH] [--episodes N]
  python plot_eval_metrics.py [--league-4v4 PATH] [--paper-4v4 PATH] [--selfplay-4v4 PATH]  # 4v4 checkpoints
  python plot_eval_metrics.py [--league-5v5 PATH] [--paper-5v5 PATH] [--selfplay-5v5 PATH]  # 5v5 (defaults: final_ppo_*_5v5.zip)
  python plot_eval_metrics.py   # default: OP3 + OP4 (all metrics + robustness)
  python plot_eval_metrics.py --opponents OP4 --episodes 50   # single opponent
  python plot_eval_metrics.py --table-out eval_table.csv --out eval_metrics.png
  python plot_eval_metrics.py --table-out eval_table.csv --table-opponent OP4   # table/CSV and plots use OP4
  python plot_eval_metrics.py --training-csv logs/behavior_ppo.csv   # add AUC learning curve if CSV has episode_id, success

  Robustness (OP3 vs OP4, success rate): default panels 2v2 & 4v4 (paper-style). Add 5v5 panel:
    python plot_eval_metrics.py --robustness-modes 2v2 4v4 5v5

  Produces PNGs: Performance, Coordination, Robustness_OP3_OP4 (when both opponents), Stability, Robotics.
  Base name from --out (e.g. eval_metrics -> eval_metrics_Performance_OP4.png).
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from typing import Any

import numpy as np

warnings.filterwarnings("ignore", message=".*render_mode.*")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
# Imports resolve from AICTFProject (rl/, game_field_gpu.py), not plot/
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from eval_rollout import compute_aggregates, run_eval_episodes


def _safe_float(val: str | float) -> float:
    """Parse CSV value; 'nan' -> np.nan."""
    if val is None or (isinstance(val, str) and val.strip().lower() in ("", "nan", "none")):
        return float("nan")
    try:
        f = float(val)
        return f if np.isfinite(f) else float("nan")
    except (ValueError, TypeError):
        return float("nan")


def load_metrics_csv(csv_path: str) -> tuple[dict[str, dict[tuple[str, str], dict]], list[str]]:
    """
    Load results_by_mode and opponents from a saved eval table CSV.
    Returns (results_by_mode, opponents).
    CSV must have: setting, method, opponent, success_rate_mean, success_rate_std, ...
    """
    import csv as csv_module

    results_by_mode: dict[str, dict[tuple[str, str], dict]] = {}
    opponents_set: set[str] = set()

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv_module.DictReader(f)
        rows = list(reader)

    if not rows:
        return results_by_mode, []

    for row in rows:
        setting = str(row.get("setting", "")).strip()
        method = str(row.get("method", "")).strip()
        opponent = str(row.get("opponent", "")).strip().upper()
        if not setting or not method or not opponent:
            continue
        opponents_set.add(opponent)

        agg = {
            "success_rate": _safe_float(row.get("success_rate_mean", 0)),
            "success_rate_std": _safe_float(row.get("success_rate_std", 0)),
            "mean_steps": _safe_float(row.get("mean_steps_mean", 0)),
            "mean_steps_std": _safe_float(row.get("mean_steps_std", 0)),
            "collision_free_rate": _safe_float(row.get("collision_free_mean", 0)),
            "collision_free_rate_std": _safe_float(row.get("collision_free_std", 0)),
            "return_var": _safe_float(row.get("return_variance_mean", 0)),
            "return_var_std": _safe_float(row.get("return_variance_std", 0)),
            "coverage_efficiency": _safe_float(row.get("coverage_efficiency_mean", 0)),
            "coverage_efficiency_std": _safe_float(row.get("coverage_efficiency_std", 0)),
            "win_margin_mean": _safe_float(row.get("win_margin_mean", 0)),
            "win_margin_std": _safe_float(row.get("win_margin_std", 0)),
            "time_to_first_score_mean": _safe_float(row.get("time_to_first_score_mean", "nan")),
            "time_to_first_score_std": _safe_float(row.get("time_to_first_score_std", 0)),
            "mean_inter_robot_dist_mean": _safe_float(row.get("mean_inter_robot_dist_mean", "nan")),
            "mean_inter_robot_dist_std": _safe_float(row.get("mean_inter_robot_dist_std", 0)),
        }

        if setting not in results_by_mode:
            results_by_mode[setting] = {}
        results_by_mode[setting][(method, opponent)] = agg

    opponents = sorted(opponents_set)
    return results_by_mode, opponents


def load_training_success_auc(csv_path: str) -> float | None:
    """Load CSV with episode_id, success; return trapezoidal AUC of success curve, or None."""
    try:
        import csv
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if not rows or "episode_id" not in rows[0] or "success" not in rows[0]:
            return None
        ep_ids = []
        success = []
        for r in rows:
            try:
                ep_ids.append(int(float(r["episode_id"])))
                success.append(float(r["success"]))
            except (ValueError, KeyError):
                continue
        if len(ep_ids) < 2:
            return None
        ep_ids = np.array(ep_ids)
        success = np.array(success)
        order = np.argsort(ep_ids)
        x = ep_ids[order]
        y = success[order]
        auc = float(np.trapz(y, x)) / (x[-1] - x[0]) if x[-1] > x[0] else float(np.mean(y))
        return auc
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot evaluation metrics by category (2v2–5v5, 8v8)")
    parser.add_argument("--league", type=str, default=None, help="2v2 League model .zip")
    parser.add_argument("--paper", type=str, default=None, help="2v2 Paper model .zip")
    parser.add_argument("--selfplay", type=str, default=None, help="2v2 Self-play model .zip")
    parser.add_argument("--league-3v3", type=str, default=None, help="3v3 League model .zip")
    parser.add_argument("--paper-3v3", type=str, default=None, help="3v3 Paper model .zip")
    parser.add_argument("--selfplay-3v3", type=str, default=None, help="3v3 Self-play model .zip")
    parser.add_argument("--league-4v4", type=str, default=None, help="4v4 League model .zip")
    parser.add_argument("--paper-4v4", type=str, default=None, help="4v4 Paper model .zip")
    parser.add_argument("--selfplay-4v4", type=str, default=None, help="4v4 Self-play model .zip")
    parser.add_argument("--league-5v5", type=str, default=None, help="5v5 League model .zip")
    parser.add_argument("--paper-5v5", type=str, default=None, help="5v5 Paper model .zip")
    parser.add_argument("--selfplay-5v5", type=str, default=None, help="5v5 Self-play model .zip")
    parser.add_argument("--league-8v8", type=str, default=None, help="8v8 League model .zip")
    parser.add_argument("--paper-8v8", type=str, default=None, help="8v8 Paper model .zip")
    parser.add_argument("--selfplay-8v8", type=str, default=None, help="8v8 Self-play model .zip")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--opponent", type=str, default=None, help="Single opponent (used only if --opponents not set)")
    parser.add_argument(
        "--opponents",
        type=str,
        nargs="+",
        default=None,
        help="Opponents to evaluate (default: OP3 OP4 for full metrics + robustness). E.g. --opponents OP4 for single opponent.",
    )
    parser.add_argument(
        "--robustness-modes",
        type=str,
        nargs="+",
        default=None,
        metavar="MODE",
        help=(
            "Team sizes for eval_metrics_*_Robustness_OP3_OP4.png (default: 2v2 4v4, paper-style). "
            "Example: --robustness-modes 2v2 4v4 5v5 for a third panel."
        ),
    )
    parser.add_argument("--out", type=str, default="eval_metrics.png", help="Output plot path (default: figures/eval_metrics.png)")
    parser.add_argument("--table-out", type=str, default=None, help="If set, write paper-ready metrics table to this CSV (default: csv/eval_table.csv)")
    parser.add_argument("--table-opponent", type=str, default=None, help="Opponent for table/CSV and printed metrics (default: first in --opponents). E.g. --table-opponent OP4 to get OP4 results.")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--training-csv", type=str, default=None, help="Optional training CSV for AUC learning curve")
    parser.add_argument(
        "--from-csv",
        "--metrics-csv",
        type=str,
        default=None,
        dest="metrics_csv",
        help="Read metrics from this CSV instead of running evaluation. Ensures reproducibility: plots use frozen numbers.",
    )
    args = parser.parse_args()

    # Send plots to AICTFProject/figures/, CSV tables to AICTFProject/csv/
    project_root = os.path.dirname(SCRIPT_DIR)
    figures_dir = os.path.join(project_root, "figures")
    csv_dir = os.path.join(project_root, "csv")
    if not os.path.dirname(os.path.abspath(args.out)):
        args.out = os.path.join(figures_dir, args.out)
    if args.table_out and not os.path.dirname(os.path.abspath(args.table_out)):
        args.table_out = os.path.join(csv_dir, args.table_out)
    os.makedirs(figures_dir, exist_ok=True)
    if args.table_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.table_out)) or ".", exist_ok=True)

    def path_or_default(name: str | None, default_name: str, subdir: str) -> str:
        if name is not None:
            p = name
        else:
            p = os.path.join(project_root, "checkpoints_sb3", subdir, default_name)
        if not p.endswith(".zip"):
            p = p + ".zip"
        return os.path.abspath(p)

    def path_or_default_candidates(name: str | None, candidates: list[str], subdir: str) -> str:
        """Like path_or_default but tries several filenames under subdir (5v5 resumed/oom names)."""
        if name is not None:
            p = name
        else:
            base = os.path.join(project_root, "checkpoints_sb3", subdir)
            chosen: str | None = None
            for cand in candidates:
                bn = cand if cand.endswith(".zip") else cand + ".zip"
                full = os.path.join(base, os.path.basename(bn))
                if os.path.isfile(full):
                    chosen = full
                    break
            if chosen is None:
                first = candidates[0]
                first = first if first.endswith(".zip") else first + ".zip"
                chosen = os.path.join(base, os.path.basename(first))
            p = chosen
        if not p.endswith(".zip"):
            p = p + ".zip"
        return os.path.abspath(p)

    # Defaults: same as plot_2v2_winrate.py / plot_3v3_winrate.py / plot_4v4_winrate.py / plot_8v8_winrate.py
    # (checkpoints_sb3/<NxN>/ under AICTFProject, not under plot/)
    model_paths_2v2 = [
        ("Ours", path_or_default(args.league, "final_ppo_league_2v2.zip", "2v2")),
        ("Jacob et al.", path_or_default(args.paper, "final_ppo_paper_2v2.zip", "2v2")),
        ("Self-play", path_or_default(args.selfplay, "final_ppo_self_play_2v2.zip", "2v2")),
    ]
    # 3v3: matches plot_3v3_winrate.py
    model_paths_3v3 = [
        ("Ours", path_or_default(args.league_3v3, "final_ppo_league_3v3.zip", "3v3")),
        ("Jacob et al.", path_or_default(args.paper_3v3, "final_ppo_paper_3v3.zip", "3v3")),
        ("Self-play", path_or_default(args.selfplay_3v3, "final_ppo_self_play_3v3.zip", "3v3")),
    ]
    # 4v4: same defaults as (modern) plot_4v4_winrate.py
    model_paths_4v4 = [
        ("Ours", path_or_default(args.league_4v4, "final_ppo_league_4v4.zip", "4v4")),
        ("Jacob et al.", path_or_default(args.paper_4v4, "final_ppo_paper_4v4.zip", "4v4")),
        ("Self-play", path_or_default(args.selfplay_4v4, "final_ppo_self_play_4v4.zip", "4v4")),
    ]
    # 5v5: same fallbacks as plot_5v5_winrate.py (final → resumed → oom_save)
    model_paths_5v5 = [
        (
            "Ours",
            path_or_default_candidates(
                args.league_5v5,
                [
                    "final_ppo_league_5v5.zip",
                    "final_ppo_league_5v5_resumed_5v5.zip",
                    "oom_save_ppo_league_5v5.zip",
                ],
                "5v5",
            ),
        ),
        (
            "Jacob et al.",
            path_or_default_candidates(
                args.paper_5v5,
                [
                    "final_ppo_paper_5v5.zip",
                    "final_ppo_paper_5v5_resumed_5v5.zip",
                    "oom_save_ppo_paper_5v5.zip",
                ],
                "5v5",
            ),
        ),
        (
            "Self-play",
            path_or_default_candidates(
                args.selfplay_5v5,
                ["final_ppo_self_play_5v5.zip", "oom_save_ppo_self_play_5v5.zip"],
                "5v5",
            ),
        ),
    ]
    # 8v8: matches plot_8v8_winrate.py
    model_paths_8v8 = [
        ("Ours", path_or_default(args.league_8v8, "final_ppo_league_8v8.zip", "8v8")),
        ("Jacob et al.", path_or_default(args.paper_8v8, "final_ppo_paper_8v8.zip", "8v8")),
        ("Self-play", path_or_default(args.selfplay_8v8, "final_ppo_self_play_8v8.zip", "8v8")),
    ]
    use_metrics_csv = args.metrics_csv and os.path.isfile(args.metrics_csv)

    if use_metrics_csv:
        print(f"Loading metrics from {args.metrics_csv} (no evaluation run).")
        results_by_mode, opponents = load_metrics_csv(args.metrics_csv)
        if not results_by_mode:
            sys.exit(f"[ERROR] No valid rows in {args.metrics_csv}")
        print(f"  Opponents in CSV: {opponents}")
    else:
        if args.metrics_csv:
            print(f"[WARN] --metrics-csv={args.metrics_csv} not found or not a file; running evaluation.")
        for _label, p in model_paths_2v2 + model_paths_3v3 + model_paths_4v4 + model_paths_5v5 + model_paths_8v8:
            if not os.path.isfile(p):
                print(f"[WARN] Not found: {p}")
                sys.exit(1)

        from game_field_gpu import GPUCTFVecEnv, GPUFieldConfig

        opponents = args.opponents if args.opponents else ([args.opponent] if args.opponent is not None else ["OP3", "OP4"])
        n_episodes = args.episodes

        results_by_mode = {}
        for mode, n_agents, model_paths in [
            ("2v2", 2, model_paths_2v2),
            ("3v3", 3, model_paths_3v3),
            ("4v4", 4, model_paths_4v4),
            ("5v5", 5, model_paths_5v5),
            ("8v8", 8, model_paths_8v8),
        ]:
            results = {}
            for opp in opponents:
                opp_clean = str(opp).strip().upper()
                seed = 42 + (1 if opp_clean == "OP4" else 0)
                cfg = GPUFieldConfig(
                    n_envs=1,
                    max_blue_agents=n_agents,
                    max_red_agents=n_agents,
                    max_decision_steps=400,
                    aquaticus_profile=True,
                    rules_profile="OURS",
                    device=args.device,
                    seed=seed,
                )
                env = GPUCTFVecEnv(cfg)
                for label, model_path in model_paths:
                    print(f"[{mode}] Evaluating {label} vs {opp_clean} ({n_episodes} episodes, seed={seed})...")
                    episodes = run_eval_episodes(model_path, env, n_episodes, args.device, opp_clean)
                    results[(label, opp_clean)] = compute_aggregates(episodes)
                env.close()
            results_by_mode[mode] = results

    # Optional training AUC (per-run, not per-model; we don't have per-model CSVs by default)
    training_auc: float | None = None
    if args.training_csv and os.path.isfile(args.training_csv):
        training_auc = load_training_success_auc(args.training_csv)
        if training_auc is not None:
            print(f"Training AUC (success curve): {training_auc:.4f}")

    # Paper-ready table: mean ± std per method per setting (2v2–5v5, 8v8); use --table-opponent to pick OP4
    main_opp = opponents[0]
    table_opp = (args.table_opponent or "").strip().upper() or main_opp
    if table_opp not in opponents:
        table_opp = main_opp
    table_rows: list[dict] = []
    for mode, model_paths in [
        ("2v2", model_paths_2v2),
        ("3v3", model_paths_3v3),
        ("4v4", model_paths_4v4),
        ("5v5", model_paths_5v5),
        ("8v8", model_paths_8v8),
    ]:
        results = results_by_mode.get(mode, {})
        for label, _ in model_paths:
            r = results.get((label, table_opp), {})
            table_rows.append({
                "setting": mode,
                "method": label,
                "opponent": table_opp,
                "success_rate_mean": r.get("success_rate", 0),
                "success_rate_std": r.get("success_rate_std", 0),
                "mean_steps_mean": r.get("mean_steps", 0),
                "mean_steps_std": r.get("mean_steps_std", 0),
                "collision_free_mean": r.get("collision_free_rate", 0),
                "collision_free_std": r.get("collision_free_rate_std", 0),
                "return_variance_mean": r.get("return_var", 0),
                "return_variance_std": r.get("return_var_std", 0),
                "coverage_efficiency_mean": r.get("coverage_efficiency", 0),
                "coverage_efficiency_std": r.get("coverage_efficiency_std", 0),
                "win_margin_mean": r.get("win_margin_mean", 0),
                "win_margin_std": r.get("win_margin_std", 0),
                "time_to_first_score_mean": r.get("time_to_first_score_mean", float("nan")),
                "time_to_first_score_std": r.get("time_to_first_score_std", 0),
                "mean_inter_robot_dist_mean": r.get("mean_inter_robot_dist_mean", float("nan")),
                "mean_inter_robot_dist_std": r.get("mean_inter_robot_dist_std", 0),
            })
    # Print compact table to console
    print("\n--- Paper-ready metrics (mean ± std over episodes, opponent=%s) ---" % table_opp)
    for mode in ("2v2", "3v3", "4v4", "5v5", "8v8"):
        print(f"\n  [{mode}]")
        for row in table_rows:
            if row["setting"] != mode:
                continue
            m = row["method"]
            print(f"    {m}:")
            print(f"      success_rate:        {row['success_rate_mean']:.2f} ± {row['success_rate_std']:.2f} %")
            print(f"      win_margin:           {row['win_margin_mean']:.2f} ± {row['win_margin_std']:.2f} (blue - red)")
            print(f"      mean_steps:          {row['mean_steps_mean']:.1f} ± {row['mean_steps_std']:.1f}")
            print(f"      collision_free:      {row['collision_free_mean']:.2f} ± {row['collision_free_std']:.2f} %")
            print(f"      return_variance:     {row['return_variance_mean']:.4f} ± {row['return_variance_std']:.4f}")
            print(f"      coverage_efficiency: {row['coverage_efficiency_mean']:.2f} ± {row['coverage_efficiency_std']:.2f} %")
            ttfs = row.get('time_to_first_score_mean', float('nan'))
            if ttfs is not None and (isinstance(ttfs, (int, float)) and np.isfinite(ttfs)):
                print(f"      time_to_first_score: {ttfs:.2f} ± {row.get('time_to_first_score_std', 0):.2f} (lower = faster offense)")
            midist = row.get('mean_inter_robot_dist_mean', float('nan'))
            if midist is not None and (isinstance(midist, (int, float)) and np.isfinite(midist)):
                print(f"      mean_inter_robot_dist: {midist:.3f} ± {row.get('mean_inter_robot_dist_std', 0):.3f} (higher = more spread)")
    if args.table_out and table_rows:
        import csv
        fieldnames = list(table_rows[0].keys())
        with open(args.table_out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(table_rows)
        print(f"\nTable saved: {args.table_out}")

    _valid_rob_modes = {"2v2", "3v3", "4v4", "5v5", "8v8"}
    if args.robustness_modes:
        robustness_modes: list[str] = []
        for m in args.robustness_modes:
            key = str(m).strip().lower()
            if key not in _valid_rob_modes:
                sys.exit(f"[ERROR] --robustness-modes: unknown {m!r} (use {sorted(_valid_rob_modes)})")
            robustness_modes.append(key)
    else:
        # Paper-style figure: 2v2 & 4v4 only (match OP3/OP4 robustness bar layout).
        robustness_modes = ["2v2", "4v4"]

    import matplotlib.pyplot as plt
    plt.rc("font", size=16)
    # Match 2v2_winrate.png: one clean bar chart per figure, well spaced, agent count + opponent in title
    team_bar_colors = ["#2ecc71", "#3498db", "#9b59b6", "#e67e22"]  # 2v2, 3v3, 4v4, 5v5
    bar_kw = dict(edgecolor="black", linewidth=1.2)

    base_out = args.out
    if base_out.endswith(".png"):
        base_out = base_out[:-4]
    plot_opp = table_opp
    method_labels = [m[0] for m in model_paths_2v2]  # Ours, Jacob et al., Self-play (same for 2v2/4v4)

    def _metric_at(res: dict, label: str, opp: str, key: str, default: float = 0.0) -> float:
        return float(res.get((label, opp), {}).get(key, default))

    def _save_single(
        title: str,
        ylabel: str,
        values_2v2: list[float],
        values_3v3: list[float],
        values_4v4: list[float],
        values_5v5: list[float],
        suffix: str,
        fmt: str = "{:.1f}%",
        ylim: tuple[float, float] | None = (0, 105),
        draw_zero: bool = False,
    ) -> None:
        """Bar chart: 2v2, 3v3, 4v4, 5v5 grouped per method."""
        fig, ax = plt.subplots(figsize=(11, 6))
        n = len(method_labels)
        x = np.arange(n)
        width = 0.18
        offs = (-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width)
        bars1 = ax.bar(x + offs[0], values_2v2, width, label="2v2", color=team_bar_colors[0], **bar_kw)
        bars2 = ax.bar(x + offs[1], values_3v3, width, label="3v3", color=team_bar_colors[1], **bar_kw)
        bars3 = ax.bar(x + offs[2], values_4v4, width, label="4v4", color=team_bar_colors[2], alpha=0.9, **bar_kw)
        bars4 = ax.bar(x + offs[3], values_5v5, width, label="5v5", color=team_bar_colors[3], alpha=0.9, **bar_kw)
        ax.set_xticks(x)
        ax.set_xticklabels(method_labels, fontsize=18)
        ax.tick_params(axis="y", labelsize=18)
        ax.set_ylabel(ylabel, fontsize=20)
        ax.set_title(title, fontsize=22)
        ax.legend(fontsize=14, ncol=2)
        if draw_zero:
            ax.axhline(0, color="gray", linestyle="--", linewidth=1)
        if ylim:
            ax.set_ylim(ylim[0], ylim[1])
        all_vals = values_2v2 + values_3v3 + values_4v4 + values_5v5
        text_offset = 1.5 if ylim else (max(abs(v) for v in all_vals) * 0.05 + 0.1 if all_vals else 0.5)
        for bars, vals in ((bars1, values_2v2), (bars2, values_3v3), (bars3, values_4v4), (bars4, values_5v5)):
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + text_offset, fmt.format(val), ha="center", fontsize=14)
        plt.tight_layout()
        path = f"{base_out}_{suffix}.png"
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"Saved: {path}")

    r2 = results_by_mode.get("2v2", {})
    r3 = results_by_mode.get("3v3", {})
    r4 = results_by_mode.get("4v4", {})
    r5 = results_by_mode.get("5v5", {})
    labels = [m[0] for m in model_paths_2v2]

    # PNGs: Performance, Win margin, Coordination, Robustness, Stability, Robotics (2v2–5v5)
    sr_2 = [_metric_at(r2, l, plot_opp, "success_rate") for l in labels]
    sr_3 = [_metric_at(r3, l, plot_opp, "success_rate") for l in labels]
    sr_4 = [_metric_at(r4, l, plot_opp, "success_rate") for l in labels]
    sr_5 = [_metric_at(r5, l, plot_opp, "success_rate") for l in labels]
    _save_single(
        f"Performance (2v2–5v5 vs {plot_opp})",
        f"Success rate vs {plot_opp} (%)",
        sr_2,
        sr_3,
        sr_4,
        sr_5,
        f"Performance_{plot_opp}",
    )

    # Win margin (blue - red): higher = dominance
    wm_2 = [_metric_at(r2, l, plot_opp, "win_margin_mean") for l in labels]
    wm_3 = [_metric_at(r3, l, plot_opp, "win_margin_mean") for l in labels]
    wm_4 = [_metric_at(r4, l, plot_opp, "win_margin_mean") for l in labels]
    wm_5 = [_metric_at(r5, l, plot_opp, "win_margin_mean") for l in labels]
    _save_single(
        f"Win margin (2v2–5v5 vs {plot_opp})",
        "Win margin (blue - red)",
        wm_2,
        wm_3,
        wm_4,
        wm_5,
        f"WinMargin_{plot_opp}",
        fmt="{:.2f}",
        ylim=None,
        draw_zero=True,
    )

    cf_2 = [_metric_at(r2, l, plot_opp, "collision_free_rate") for l in labels]
    cf_3 = [_metric_at(r3, l, plot_opp, "collision_free_rate") for l in labels]
    cf_4 = [_metric_at(r4, l, plot_opp, "collision_free_rate") for l in labels]
    cf_5 = [_metric_at(r5, l, plot_opp, "collision_free_rate") for l in labels]
    _save_single(
        f"Coordination (2v2–5v5 vs {plot_opp})",
        "Collision-free (%)",
        cf_2,
        cf_3,
        cf_4,
        cf_5,
        f"Coordination_{plot_opp}",
    )

    if {"OP3", "OP4"}.issubset({str(o).strip().upper() for o in opponents}):
        # One panel per --robustness-modes; each panel = 3 methods × 2 bars (OP3 solid, OP4 alpha).
        method_cols = ["#2ecc71", "#3498db", "#9b59b6"]
        r8 = results_by_mode.get("8v8", {})
        mode_to_res = {"2v2": r2, "3v3": r3, "4v4": r4, "5v5": r5, "8v8": r8}
        n_p = len(robustness_modes)
        fig, axes = plt.subplots(1, n_p, figsize=(6 * n_p, 5), squeeze=False)
        ax_list = list(axes[0])

        def _robustness_suptitle(modes: list[str]) -> str:
            if len(modes) == 1:
                body = modes[0]
            elif len(modes) == 2:
                body = f"{modes[0]} & {modes[1]}"
            else:
                body = ", ".join(modes[:-1]) + f" & {modes[-1]}"
            return f"Robustness ({body} vs OP3 / OP4)"

        for ax, mode in zip(ax_list, robustness_modes):
            res = mode_to_res.get(mode, {})
            x = np.arange(len(labels))
            w = 0.35
            v3 = [res.get((l, "OP3"), {}).get("success_rate", 0) for l in labels]
            v4 = [res.get((l, "OP4"), {}).get("success_rate", 0) for l in labels]
            b3 = ax.bar(x - w / 2, v3, w, label="vs OP3", color=method_cols, **bar_kw)
            b4 = ax.bar(x + w / 2, v4, w, label="vs OP4", color=method_cols, alpha=0.75, **bar_kw)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=18)
            ax.set_ylabel("Success rate (%)", fontsize=20)
            ax.set_title(f"Robustness ({mode})", fontsize=20)
            ax.legend(fontsize=14)
            ax.set_ylim(0, 105)
            for b, val in zip(b3, v3):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5, f"{val:.1f}%", ha="center", fontsize=14)
            for b, val in zip(b4, v4):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5, f"{val:.1f}%", ha="center", fontsize=14)
        plt.suptitle(_robustness_suptitle(robustness_modes), fontsize=22)
        plt.tight_layout()
        plt.savefig(f"{base_out}_Robustness_OP3_OP4.png", dpi=150)
        plt.close()
        print(f"Saved: {base_out}_Robustness_OP3_OP4.png")
    else:
        _save_single(
            f"Robustness (2v2–5v5 vs {plot_opp})",
            f"Success rate vs {plot_opp} (%)",
            sr_2,
            sr_3,
            sr_4,
            sr_5,
            f"Robustness_{plot_opp}",
        )

    rvar_2 = [_metric_at(r2, l, plot_opp, "return_var") for l in labels]
    rvar_3 = [_metric_at(r3, l, plot_opp, "return_var") for l in labels]
    rvar_4 = [_metric_at(r4, l, plot_opp, "return_var") for l in labels]
    rvar_5 = [_metric_at(r5, l, plot_opp, "return_var") for l in labels]
    _save_single(
        f"Stability (2v2–5v5 vs {plot_opp})",
        "Variance of episode return",
        rvar_2,
        rvar_3,
        rvar_4,
        rvar_5,
        f"Stability_{plot_opp}",
        fmt="{:.3f}",
        ylim=None,
    )

    safety_2 = [_metric_at(r2, l, plot_opp, "collision_free_rate") for l in labels]
    safety_3 = [_metric_at(r3, l, plot_opp, "collision_free_rate") for l in labels]
    safety_4 = [_metric_at(r4, l, plot_opp, "collision_free_rate") for l in labels]
    safety_5 = [_metric_at(r5, l, plot_opp, "collision_free_rate") for l in labels]
    _save_single(
        f"Robotics (2v2–5v5 vs {plot_opp})",
        "Safety (collision-free %)",
        safety_2,
        safety_3,
        safety_4,
        safety_5,
        f"Robotics_{plot_opp}",
    )

    return


if __name__ == "__main__":
    main()
