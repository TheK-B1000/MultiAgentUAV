#!/usr/bin/env python3
"""
Plot evaluation metrics by category: Performance, Coordination, Robustness, Stability, Specialization, Robotics.

Uses the same models as plot_2v2_winrate and plot_4v4_winrate (Ours / Jacob et al. / Self-play).
Runs both 2v2 and 4v4 eval with GPUCTFVecEnv (optional OP3/OP4) and collects:
  - Performance: success rate, mean steps to completion
  - Coordination: coverage efficiency (zone_coverage), coordination proxy (collision-free)
  - Robustness: generalization (success vs OP3 vs OP4); default eval uses both opponents
  - Stability: return variance across episodes
  - Specialization: placeholder (requires action logging)
  - Robotics: safety (collision-free rate); energy/path N/A

Usage:
  python plot_eval_metrics.py [--league PATH] [--paper PATH] [--selfplay PATH] [--episodes N]
  python plot_eval_metrics.py [--league-4v4 PATH] [--paper-4v4 PATH] [--selfplay-4v4 PATH]  # 4v4 checkpoints
  python plot_eval_metrics.py   # default: OP3 + OP4 (all metrics + robustness)
  python plot_eval_metrics.py --opponents OP4 --episodes 50   # single opponent
  python plot_eval_metrics.py --table-out eval_table.csv --out eval_metrics.png
  python plot_eval_metrics.py --table-out eval_table.csv --table-opponent OP4   # table/CSV and plots use OP4
  python plot_eval_metrics.py --training-csv logs/behavior_ppo.csv   # add AUC learning curve if CSV has episode_id, success

  Produces 5 clean, well-spaced PNGs (agent count + OP3/OP4 in title): Performance, Coordination,
  Robustness, Stability, Robotics. Base name from --out (e.g. eval_metrics -> eval_metrics_Performance_OP4.png).
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
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)


def _numpy_compat_shim() -> None:
    if "numpy._core.numeric" not in sys.modules:
        try:
            import numpy.core as _core
            import numpy.core.numeric
            import numpy.core.multiarray
            import numpy.core.umath
            sys.modules["numpy._core"] = _core
            sys.modules["numpy._core.numeric"] = _core.numeric
            sys.modules["numpy._core.multiarray"] = _core.multiarray
            sys.modules["numpy._core.umath"] = _core.umath
        except Exception:
            pass
    try:
        import numpy.random._pickle as _np_pickle
        _orig_bg_ctor = _np_pickle.__bit_generator_ctor

        def _patched_bg_ctor(bit_generator_name: Any = "MT19937") -> Any:
            if isinstance(bit_generator_name, type):
                bit_generator_name = bit_generator_name.__name__
            return _orig_bg_ctor(bit_generator_name)

        _np_pickle.__bit_generator_ctor = _patched_bg_ctor
    except Exception:
        pass


def run_eval_episodes(
    model_path: str,
    env: Any,
    n_episodes: int,
    device: str,
    opponent: str,
) -> list[dict]:
    """Run n_episodes, return list of per-episode dicts: success, steps, return, zone_coverage, collision_free."""
    from stable_baselines3 import PPO
    from rl.train_ppo import MaskedMultiInputPolicy

    _numpy_compat_shim()
    custom = {
        "observation_space": env.observation_space,
        "action_space": env.action_space,
        "policy_class": MaskedMultiInputPolicy,
    }
    model = PPO.load(model_path, device=device, custom_objects=custom)
    model.policy.set_training_mode(False)

    try:
        env.env_method("set_phase", opponent)
        env.env_method("set_next_opponent", "SCRIPTED", opponent)
        # Verify core actually has this opponent (so eval is truly vs OP3/OP4)
        out = env.env_method("get_opponent_key")
        actual = (out[0] if out else "").strip().upper()
        requested = str(opponent).strip().upper()
        if actual != requested:
            import warnings
            warnings.warn(
                f"Opponent mismatch: requested {requested!r}, core has {actual!r}. "
                "Eval may not be against the intended opponent."
            )
    except Exception as e:
        import warnings
        warnings.warn(
            f"Failed to set opponent to {opponent!r}: {e}. "
            "Red team may still be using the previous opponent — OP3 vs OP4 results can look identical."
        )

    episodes: list[dict] = []
    obs = env.reset()

    for _ in range(n_episodes):
        ep_return = 0.0
        steps = 0
        while True:
            single = {
                k: v[0] if hasattr(v, "shape") and len(v.shape) > 1 and v.shape[0] == 1 else v
                for k, v in obs.items()
            }
            act, _ = model.predict(single, deterministic=True)
            env.step_async(act)
            obs, rew, done, infos = env.step_wait()
            steps += 1
            ep_return += float(rew[0])
            if done.any():
                for i in range(len(done)):
                    if done[i]:
                        info = infos[i] if i < len(infos) else {}
                        ep_res = info.get("episode_result", info)
                        bs = int(ep_res.get("blue_score", 0))
                        rs = int(ep_res.get("red_score", 0))
                        success = 1 if bs > rs else 0
                        decision_steps = int(ep_res.get("decision_steps", info.get("decision_steps", 0)))
                        zone_cov = float(ep_res.get("zone_coverage", 0.0))
                        collision_free = int(ep_res.get("collision_free_episode", 1))
                        ttfs = ep_res.get("time_to_first_score")
                        try:
                            ttfs_f = float(ttfs) if ttfs is not None and ttfs != "" else np.nan
                        except (TypeError, ValueError):
                            ttfs_f = np.nan
                        mean_dist = ep_res.get("mean_inter_robot_dist")
                        try:
                            mean_dist_f = float(mean_dist) if mean_dist is not None and mean_dist != "" else np.nan
                        except (TypeError, ValueError):
                            mean_dist_f = np.nan
                        episodes.append({
                            "success": success,
                            "steps": decision_steps,
                            "return": ep_return,
                            "zone_coverage": zone_cov,
                            "collision_free": collision_free,
                            "win_margin": bs - rs,
                            "time_to_first_score": ttfs_f,
                            "mean_inter_robot_dist": mean_dist_f,
                        })
                        ep_return = 0.0
                break

    return episodes


def compute_aggregates(episodes: list[dict]) -> dict:
    """Compute mean and std (over episodes) for paper-ready tables."""
    base = {
        "success_rate": 0.0,
        "success_rate_std": 0.0,
        "mean_steps": 0.0,
        "mean_steps_std": 0.0,
        "mean_return": 0.0,
        "return_var": 0.0,
        "return_var_std": 0.0,
        "coverage_efficiency": 0.0,
        "coverage_efficiency_std": 0.0,
        "collision_free_rate": 0.0,
        "collision_free_rate_std": 0.0,
        "win_margin_mean": 0.0,
        "win_margin_std": 0.0,
        "time_to_first_score_mean": float("nan"),
        "time_to_first_score_std": 0.0,
        "mean_inter_robot_dist_mean": float("nan"),
        "mean_inter_robot_dist_std": 0.0,
    }
    if not episodes:
        return base
    arr = np.array([
        [e["success"], e["steps"], e["return"], e["zone_coverage"], e["collision_free"],
        e["win_margin"],
        e.get("time_to_first_score", np.nan),
        e.get("mean_inter_robot_dist", np.nan),
    ]
        for e in episodes
    ])
    n = arr.shape[0]
    ddof = 1 if n > 1 else 0
    success_rate = float(np.mean(arr[:, 0])) * 100.0
    success_rate_std = float(np.std(arr[:, 0], ddof=ddof)) * 100.0
    mean_steps = float(np.mean(arr[:, 1]))
    mean_steps_std = float(np.std(arr[:, 1], ddof=ddof))
    mean_return = float(np.mean(arr[:, 2]))
    return_var = float(np.var(arr[:, 2], ddof=ddof))
    return_var_std = 0.0
    coverage_efficiency = float(np.mean(arr[:, 3])) * 100.0
    coverage_efficiency_std = float(np.std(arr[:, 3], ddof=ddof)) * 100.0
    collision_free_rate = float(np.mean(arr[:, 4])) * 100.0
    collision_free_rate_std = float(np.std(arr[:, 4], ddof=ddof)) * 100.0
    win_margin_mean = float(np.mean(arr[:, 5]))
    win_margin_std = float(np.std(arr[:, 5], ddof=ddof))
    ttfs = arr[:, 6]
    ttfs_valid = ttfs[np.isfinite(ttfs)]
    time_to_first_score_mean = float(np.mean(ttfs_valid)) if len(ttfs_valid) > 0 else np.nan
    time_to_first_score_std = float(np.std(ttfs_valid, ddof=1)) if len(ttfs_valid) > 1 else 0.0
    midist = arr[:, 7]
    midist_valid = midist[np.isfinite(midist)]
    mean_inter_robot_dist_mean = float(np.mean(midist_valid)) if len(midist_valid) > 0 else np.nan
    mean_inter_robot_dist_std = float(np.std(midist_valid, ddof=1)) if len(midist_valid) > 1 else 0.0
    return {
        "success_rate": success_rate,
        "success_rate_std": success_rate_std,
        "mean_steps": mean_steps,
        "mean_steps_std": mean_steps_std,
        "mean_return": mean_return,
        "return_var": return_var,
        "return_var_std": return_var_std,
        "coverage_efficiency": coverage_efficiency,
        "coverage_efficiency_std": coverage_efficiency_std,
        "collision_free_rate": collision_free_rate,
        "collision_free_rate_std": collision_free_rate_std,
        "win_margin_mean": win_margin_mean,
        "win_margin_std": win_margin_std,
        "time_to_first_score_mean": time_to_first_score_mean,
        "time_to_first_score_std": time_to_first_score_std,
        "mean_inter_robot_dist_mean": mean_inter_robot_dist_mean,
        "mean_inter_robot_dist_std": mean_inter_robot_dist_std,
    }


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
    parser = argparse.ArgumentParser(description="Plot evaluation metrics by category (2v2, 3v3, 4v4)")
    parser.add_argument("--league", type=str, default=None, help="2v2 League model .zip")
    parser.add_argument("--paper", type=str, default=None, help="2v2 Paper model .zip")
    parser.add_argument("--selfplay", type=str, default=None, help="2v2 Self-play model .zip")
    parser.add_argument("--league-3v3", type=str, default=None, help="3v3 League model .zip")
    parser.add_argument("--paper-3v3", type=str, default=None, help="3v3 Paper model .zip")
    parser.add_argument("--selfplay-3v3", type=str, default=None, help="3v3 Self-play model .zip")
    parser.add_argument("--league-4v4", type=str, default=None, help="4v4 League model .zip")
    parser.add_argument("--paper-4v4", type=str, default=None, help="4v4 Paper model .zip")
    parser.add_argument("--selfplay-4v4", type=str, default=None, help="4v4 Self-play model .zip")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--opponent", type=str, default=None, help="Single opponent (used only if --opponents not set)")
    parser.add_argument(
        "--opponents",
        type=str,
        nargs="+",
        default=None,
        help="Opponents to evaluate (default: OP3 OP4 for full metrics + robustness). E.g. --opponents OP4 for single opponent.",
    )
    parser.add_argument("--out", type=str, default="eval_metrics.png")
    parser.add_argument("--table-out", type=str, default=None, help="If set, write paper-ready metrics table to this CSV (e.g. eval_table.csv)")
    parser.add_argument("--table-opponent", type=str, default=None, help="Opponent for table/CSV and printed metrics (default: first in --opponents). E.g. --table-opponent OP4 to get OP4 results.")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--training-csv", type=str, default=None, help="Optional training CSV for AUC learning curve")
    args = parser.parse_args()

    default_dir = os.path.join(SCRIPT_DIR, "checkpoints_sb3")
    def path_or_default(name: str | None, default_name: str) -> str:
        p = name if name is not None else os.path.join(default_dir, default_name)
        if not p.endswith(".zip"):
            p = p + ".zip"
        return os.path.abspath(p)

    # 2v2: same defaults as plot_2v2_winrate.py
    model_paths_2v2 = [
        ("Ours", path_or_default(args.league, "final_ppo_league_2v2_colab.zip")),
        ("Jacob et al.", path_or_default(args.paper, "final_weekend_paper_2v2.zip")),
        ("Self-play", path_or_default(args.selfplay, "final_weekend_selfplay_2v2.zip")),
    ]
    # 3v3: matches plot_3v3_winrate.py
    model_paths_3v3 = [
        ("Ours", path_or_default(args.league_3v3, "final_weekend_league_3v3.zip")),
        ("Jacob et al.", path_or_default(args.paper_3v3, "final_weekend_paper_3v3.zip")),
        ("Self-play", path_or_default(args.selfplay_3v3, "final_weekend_selfplay_3v3.zip")),
    ]
    # 4v4: same defaults as plot_4v4_winrate.py
    model_paths_4v4 = [
        ("Ours", path_or_default(args.league_4v4, "final_ppo_league_4v4_colab.zip")),
        ("Jacob et al.", path_or_default(args.paper_4v4, "final_weekend_paper_4v4.zip")),
        ("Self-play", path_or_default(args.selfplay_4v4, "final_ppo_selfplay_4v4_colab.zip")),
    ]
    for _label, p in model_paths_2v2 + model_paths_3v3 + model_paths_4v4:
        if not os.path.isfile(p):
            print(f"[WARN] Not found: {p}")
            sys.exit(1)

    from game_field_gpu import GPUCTFVecEnv, GPUFieldConfig

    # Default: OP3 + OP4 so one run gives all metrics and robustness; use --opponents OP4 for single opponent
    opponents = args.opponents if args.opponents else ([args.opponent] if args.opponent is not None else ["OP3", "OP4"])
    n_episodes = args.episodes

    # results_by_mode[mode][(label, opponent)] = aggregate dict
    results_by_mode: dict[str, dict[tuple[str, str], dict]] = {}

    for mode, n_agents, model_paths in [
        ("2v2", 2, model_paths_2v2),
        ("3v3", 3, model_paths_3v3),
        ("4v4", 4, model_paths_4v4),
    ]:
        # Match winrate scripts: use a fresh env (and seed) per opponent, so numbers are directly comparable
        results: dict[tuple[str, str], dict] = {}
        for opp in opponents:
            opp_clean = str(opp).strip().upper()
            seed = 42 + (1 if opp_clean == "OP4" else 0)
            cfg = GPUFieldConfig(
                n_envs=1,
                max_blue_agents=n_agents,
                max_red_agents=n_agents,
                max_decision_steps=400,
                aquaticus_profile=True,
                rules_profile="AQUATICUS_2024",
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

    # Paper-ready table: mean ± std per method per setting (2v2, 3v3, 4v4); use --table-opponent to pick OP4
    main_opp = opponents[0]
    table_opp = (args.table_opponent or "").strip().upper() or main_opp
    if table_opp not in opponents:
        table_opp = main_opp
    table_rows: list[dict] = []
    for mode, model_paths in [("2v2", model_paths_2v2), ("3v3", model_paths_3v3), ("4v4", model_paths_4v4)]:
        results = results_by_mode[mode]
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
    for mode in ("2v2", "3v3", "4v4"):
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

    import matplotlib.pyplot as plt
    plt.rc("font", size=16)
    # Match 2v2_winrate.png: one clean bar chart per figure, well spaced, agent count + opponent in title
    bar_colors = ["#2ecc71", "#3498db", "#9b59b6"]  # Ours, Jacob et al., Self-play
    bar_kw = dict(edgecolor="black", linewidth=1.2)

    base_out = args.out
    if base_out.endswith(".png"):
        base_out = base_out[:-4]
    plot_opp = table_opp
    mode_configs = [("2v2", model_paths_2v2), ("4v4", model_paths_4v4)]

    method_labels = [m[0] for m in model_paths_2v2]  # Ours, Jacob et al., Self-play (same for 2v2/4v4)

    def _save_single(
        title: str,
        ylabel: str,
        values_2v2: list[float],
        values_4v4: list[float],
        suffix: str,
        fmt: str = "{:.1f}%",
        ylim: tuple[float, float] | None = (0, 105),
        draw_zero: bool = False,
    ) -> None:
        """One clean bar chart with 2v2 and 4v4 grouped (like 2v2_winrate style), agent count + opponent in title."""
        fig, ax = plt.subplots(figsize=(10, 6))
        n = len(method_labels)
        x = np.arange(n)
        width = 0.35
        bars1 = ax.bar(x - width / 2, values_2v2, width, label="2v2", color=bar_colors, **bar_kw)
        bars2 = ax.bar(x + width / 2, values_4v4, width, label="4v4", color=bar_colors, alpha=0.75, **bar_kw)
        ax.set_xticks(x)
        ax.set_xticklabels(method_labels, fontsize=18)
        ax.tick_params(axis="y", labelsize=18)
        ax.set_ylabel(ylabel, fontsize=20)
        ax.set_title(title, fontsize=22)
        ax.legend(fontsize=16)
        if draw_zero:
            ax.axhline(0, color="gray", linestyle="--", linewidth=1)
        if ylim:
            ax.set_ylim(ylim[0], ylim[1])
        all_vals = values_2v2 + values_4v4
        text_offset = 1.5 if ylim else (max(abs(v) for v in all_vals) * 0.05 + 0.1 if all_vals else 0.5)
        for bar, val in zip(bars1, values_2v2):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + text_offset, fmt.format(val), ha="center", fontsize=16)
        for bar, val in zip(bars2, values_4v4):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + text_offset, fmt.format(val), ha="center", fontsize=16)
        plt.tight_layout()
        path = f"{base_out}_{suffix}.png"
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"Saved: {path}")

    r2 = results_by_mode["2v2"]
    r4 = results_by_mode["4v4"]
    labels = [m[0] for m in model_paths_2v2]

    # 5 clean PNGs + Offense metrics (Performance, Win margin, Coordination, Robustness, Stability, Robotics)
    sr_2 = [r2[(l, plot_opp)]["success_rate"] for l in labels]
    sr_4 = [r4[(l, plot_opp)]["success_rate"] for l in labels]
    _save_single(f"Performance (2v2 & 4v4 vs {plot_opp})", f"Success rate vs {plot_opp} (%)", sr_2, sr_4, f"Performance_{plot_opp}")

    # Win margin (blue - red): higher = dominance
    wm_2 = [r2[(l, plot_opp)]["win_margin_mean"] for l in labels]
    wm_4 = [r4[(l, plot_opp)]["win_margin_mean"] for l in labels]
    _save_single(
        f"Win margin (2v2 & 4v4 vs {plot_opp})",
        "Win margin (blue - red)",
        wm_2,
        wm_4,
        f"WinMargin_{plot_opp}",
        fmt="{:.2f}",
        ylim=None,
        draw_zero=True,
    )

    cf_2 = [r2[(l, plot_opp)]["collision_free_rate"] for l in labels]
    cf_4 = [r4[(l, plot_opp)]["collision_free_rate"] for l in labels]
    _save_single(f"Coordination (2v2 & 4v4 vs {plot_opp})", "Collision-free (%)", cf_2, cf_4, f"Coordination_{plot_opp}")

    if "OP4" in opponents and "OP3" in opponents:
        # Robustness: show both OP3 and OP4 for 2v2 and 4v4 (one chart: 2v2 vs OP3, 2v2 vs OP4, 4v4 vs OP3, 4v4 vs OP4 as 4 series would cramp; do two charts)
        # Single Robustness chart: 2v2 vs OP3, 4v4 vs OP3 in one; then 2v2 vs OP4, 4v4 vs OP4 in another. That's 2 charts. User asked 5. So one "Robustness" with 4 series is cramped. Use one Robustness chart with 2v2 (vs OP3 + OP4) and 4v4 (vs OP3 + OP4): 2 groups of 2 bars per method = 6 groups of 2 = 12 bars. Simpler: Robustness shows success vs OP3 (2v2 and 4v4) and vs OP4 (2v2 and 4v4) as grouped bars: for each method, 4 bars (2v2-OP3, 2v2-OP4, 4v4-OP3, 4v4-OP4). That's 3 methods × 4 = 12 bars - cramped. Keep it to 5 graphs: Robustness one figure with 2v2 vs OP3, 4v4 vs OP3, 2v2 vs OP4, 4v4 vs OP4 as four lines of data - we can do 2v2 (OP3, OP4) and 4v4 (OP3, OP4) with width 0.2 so 4 bars per method. Actually for "5 nice graphs" just do one Robustness figure: y = success rate, x = Ours, Jacob, Self-play, with 2 bars per method (2v2 vs plot_opp, 4v4 vs plot_opp). So same as Performance but we're labeling "Robustness". That duplicates Performance. So Robustness should show generalization: e.g. 2v2 vs OP3 and 2v2 vs OP4. So one chart: for each method, two bars (vs OP3, vs OP4) for 2v2; then same for 4v4. That's 2 subplots (2v2, 4v4) in one figure. So one Robustness.png with 2 panels (2v2 and 4v4), each panel has 3 methods × 2 bars (OP3, OP4). Clean.
        # Simpler: 5 figures. Robustness = one figure with two panels (2v2, 4v4), each panel 3 methods × 2 bars (OP3, OP4). So we need a small multi-panel for Robustness only, or we do Robustness_2v2 and Robustness_4v4 as separate files and that makes 6 files (Perf, Coord, Robust_2v2, Robust_4v4, Stability, Robotics). User said 5. So: Performance, Coordination, Stability, Robotics (each 2v2 & 4v4 grouped), and Robustness (one figure: 2v2 with OP3/OP4 bars, 4v4 with OP3/OP4 bars - two panels in one figure, each panel clean and spacious). So Robustness is one file with 2 subplots (2v2, 4v4), each subplot 3 methods × 2 bars (OP3, OP4). That gives 5 files and agent count + OP in titles.
        fig, (ax2, ax4) = plt.subplots(1, 2, figsize=(12, 5))
        for ax, mode, res in [(ax2, "2v2", r2), (ax4, "4v4", r4)]:
            x = np.arange(len(labels))
            w = 0.35
            v3 = [res.get((l, "OP3"), {}).get("success_rate", 0) for l in labels]
            v4 = [res.get((l, "OP4"), {}).get("success_rate", 0) for l in labels]
            b3 = ax.bar(x - w / 2, v3, w, label="vs OP3", color=bar_colors, **bar_kw)
            b4 = ax.bar(x + w / 2, v4, w, label="vs OP4", color=bar_colors, alpha=0.75, **bar_kw)
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
        plt.suptitle(f"Robustness (2v2 & 4v4 vs OP3 / OP4)", fontsize=22)
        plt.tight_layout()
        plt.savefig(f"{base_out}_Robustness_OP3_OP4.png", dpi=150)
        plt.close()
        print(f"Saved: {base_out}_Robustness_OP3_OP4.png")
    else:
        _save_single(f"Robustness (2v2 & 4v4 vs {plot_opp})", f"Success rate vs {plot_opp} (%)", sr_2, sr_4, f"Robustness_{plot_opp}")

    rvar_2 = [r2[(l, plot_opp)]["return_var"] for l in labels]
    rvar_4 = [r4[(l, plot_opp)]["return_var"] for l in labels]
    _save_single(f"Stability (2v2 & 4v4 vs {plot_opp})", "Variance of episode return", rvar_2, rvar_4, f"Stability_{plot_opp}", fmt="{:.3f}", ylim=None)

    safety_2 = [r2[(l, plot_opp)]["collision_free_rate"] for l in labels]
    safety_4 = [r4[(l, plot_opp)]["collision_free_rate"] for l in labels]
    _save_single(f"Robotics (2v2 & 4v4 vs {plot_opp})", "Safety (collision-free %)", safety_2, safety_4, f"Robotics_{plot_opp}")

    return


if __name__ == "__main__":
    main()
