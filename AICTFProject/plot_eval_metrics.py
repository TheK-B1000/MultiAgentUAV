#!/usr/bin/env python3
"""
Plot evaluation metrics by category: Performance, Coordination, Robustness, Stability, Specialization, Robotics.

Uses the same models as plot_2v2_winrate and plot_4v4_winrate (Ours / Jacob et al. / Self-play).
Runs both 2v2 and 4v4 eval with GPUCTFVecEnv (optional OP3/OP4) and collects:
  - Performance: success rate, mean steps to completion
  - Coordination: coverage efficiency (zone_coverage), coordination proxy (collision-free)
  - Robustness: generalization (success vs OP4 vs OP3 if --opponents OP3 OP4)
  - Stability: return variance across episodes
  - Specialization: placeholder (requires action logging)
  - Robotics: safety (collision-free rate); energy/path N/A

Usage:
  python plot_eval_metrics.py [--league PATH] [--paper PATH] [--selfplay PATH] [--episodes N]
  python plot_eval_metrics.py [--league-4v4 PATH] [--paper-4v4 PATH] [--selfplay-4v4 PATH]  # 4v4 checkpoints
  python plot_eval_metrics.py --opponents OP3 OP4 --episodes 50 --out eval_metrics.png
  python plot_eval_metrics.py --training-csv logs/behavior_ppo.csv   # add AUC learning curve if CSV has episode_id, success

  Produces eval_metrics_2v2.png and eval_metrics_4v4.png (or base of --out + _2v2.png / _4v4.png).
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
    except Exception:
        pass

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
                        episodes.append({
                            "success": success,
                            "steps": decision_steps,
                            "return": ep_return,
                            "zone_coverage": zone_cov,
                            "collision_free": collision_free,
                        })
                        ep_return = 0.0
                break

    return episodes


def compute_aggregates(episodes: list[dict]) -> dict:
    if not episodes:
        return {
            "success_rate": 0.0,
            "mean_steps": 0.0,
            "mean_return": 0.0,
            "return_var": 0.0,
            "coverage_efficiency": 0.0,
            "collision_free_rate": 0.0,
        }
    arr = np.array([
        [e["success"], e["steps"], e["return"], e["zone_coverage"], e["collision_free"]]
        for e in episodes
    ])
    success_rate = float(np.mean(arr[:, 0])) * 100.0
    mean_steps = float(np.mean(arr[:, 1]))
    mean_return = float(np.mean(arr[:, 2]))
    return_var = float(np.var(arr[:, 2]))
    coverage_efficiency = float(np.mean(arr[:, 3])) * 100.0
    collision_free_rate = float(np.mean(arr[:, 4])) * 100.0
    return {
        "success_rate": success_rate,
        "mean_steps": mean_steps,
        "mean_return": mean_return,
        "return_var": return_var,
        "coverage_efficiency": coverage_efficiency,
        "collision_free_rate": collision_free_rate,
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
    parser = argparse.ArgumentParser(description="Plot evaluation metrics by category (2v2 and 4v4)")
    parser.add_argument("--league", type=str, default=None, help="2v2 League model .zip")
    parser.add_argument("--paper", type=str, default=None, help="2v2 Paper model .zip")
    parser.add_argument("--selfplay", type=str, default=None, help="2v2 Self-play model .zip")
    parser.add_argument("--league-4v4", type=str, default=None, help="4v4 League model .zip")
    parser.add_argument("--paper-4v4", type=str, default=None, help="4v4 Paper model .zip")
    parser.add_argument("--selfplay-4v4", type=str, default=None, help="4v4 Self-play model .zip")
    parser.add_argument("--episodes", type=int, default=25)
    parser.add_argument("--opponent", type=str, default="OP4", help="Single opponent for main eval")
    parser.add_argument(
        "--opponents",
        type=str,
        nargs="+",
        default=None,
        help="E.g. OP3 OP4 to compute generalization (success on OP4 vs OP3)",
    )
    parser.add_argument("--out", type=str, default="eval_metrics.png")
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
    # 4v4: same defaults as plot_4v4_winrate.py
    model_paths_4v4 = [
        ("Ours", path_or_default(args.league_4v4, "final_ppo_league_4v4_colab.zip")),
        ("Jacob et al.", path_or_default(args.paper_4v4, "final_weekend_paper_4v4.zip")),
        ("Self-play", path_or_default(args.selfplay_4v4, "final_ppo_selfplay_4v4_colab.zip")),
    ]
    for _label, p in model_paths_2v2 + model_paths_4v4:
        if not os.path.isfile(p):
            print(f"[WARN] Not found: {p}")
            sys.exit(1)

    from game_field_gpu import GPUCTFVecEnv, GPUFieldConfig

    opponents = args.opponents if args.opponents else [args.opponent]
    n_episodes = args.episodes

    # results_by_mode[mode][(label, opponent)] = aggregate dict
    results_by_mode: dict[str, dict[tuple[str, str], dict]] = {}

    for mode, n_agents, model_paths in [
        ("2v2", 2, model_paths_2v2),
        ("4v4", 4, model_paths_4v4),
    ]:
        cfg = GPUFieldConfig(
            n_envs=1,
            max_blue_agents=n_agents,
            max_red_agents=n_agents,
            max_decision_steps=400,
            aquaticus_profile=True,
            rules_profile="AQUATICUS_2024",
            device=args.device,
            seed=42,
        )
        env = GPUCTFVecEnv(cfg)
        results: dict[tuple[str, str], dict] = {}
        for label, model_path in model_paths:
            for opp in opponents:
                print(f"[{mode}] Evaluating {label} vs {opp} ({n_episodes} episodes)...")
                episodes = run_eval_episodes(model_path, env, n_episodes, args.device, opp)
                results[(label, opp)] = compute_aggregates(episodes)
        results_by_mode[mode] = results
        env.close()

    # Optional training AUC (per-run, not per-model; we don't have per-model CSVs by default)
    training_auc: float | None = None
    if args.training_csv and os.path.isfile(args.training_csv):
        training_auc = load_training_success_auc(args.training_csv)
        if training_auc is not None:
            print(f"Training AUC (success curve): {training_auc:.4f}")

    import matplotlib.pyplot as plt
    plt.rc("font", size=16)

    # Output path: eval_metrics.png -> eval_metrics_2v2.png, eval_metrics_4v4.png
    base_out = args.out
    if base_out.endswith(".png"):
        base_out = base_out[:-4]

    mode_configs = [
        ("2v2", model_paths_2v2),
        ("4v4", model_paths_4v4),
    ]

    for mode, model_paths in mode_configs:
        results = results_by_mode[mode]
        labels = [m[0] for m in model_paths]
        x = np.arange(len(labels))
        width = 0.25
        main_opp = opponents[0]

        fig, axes = plt.subplots(2, 3, figsize=(14, 9))
        fig.suptitle(f"Evaluation metrics ({mode})", fontsize=22)

        # 1. Performance
        ax = axes[0, 0]
        ax.set_title("Performance", fontsize=20)
        sr = [results[(l, main_opp)]["success_rate"] for l in labels]
        ax.bar(x - width / 2, sr, width, label="Success rate (%)", color="#2ecc71")
        ax.set_ylabel("Success rate (%)", fontsize=18)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=18)
        ax.tick_params(axis="y", labelsize=18)
        ax.set_ylim(0, 105)
        ax2 = ax.twinx()
        steps = [results[(l, main_opp)]["mean_steps"] for l in labels]
        ax2.bar(x + width / 2, steps, width, label="Mean steps to done", color="#3498db", alpha=0.7)
        ax2.set_ylabel("Mean steps (lower better)", fontsize=18)
        ax.legend(loc="upper left", fontsize=14)
        ax2.legend(loc="upper right", fontsize=14)

        # 2. Coordination
        ax = axes[0, 1]
        ax.set_title("Coordination", fontsize=20)
        cov = [results[(l, main_opp)]["coverage_efficiency"] for l in labels]
        cf = [results[(l, main_opp)]["collision_free_rate"] for l in labels]
        ax.bar(x - width / 2, cov, width, label="Coverage efficiency (%)", color="#9b59b6")
        ax.bar(x + width / 2, cf, width, label="Collision-free rate (%)", color="#e74c3c", alpha=0.7)
        ax.set_ylabel("%", fontsize=18)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=18)
        ax.tick_params(axis="y", labelsize=18)
        ax.set_ylim(0, 105)
        ax.legend(fontsize=14)

        # 3. Robustness
        ax = axes[0, 2]
        ax.set_title("Robustness", fontsize=20)
        if "OP4" in opponents and "OP3" in opponents:
            sr_op3 = [results.get((l, "OP3"), {}).get("success_rate", 0) for l in labels]
            sr_op4 = [results.get((l, "OP4"), {}).get("success_rate", 0) for l in labels]
            ax.bar(x - width / 2, sr_op3, width, label="Success vs OP3 (%)", color="#3498db")
            ax.bar(x + width / 2, sr_op4, width, label="Success vs OP4 (%)", color="#e67e22")
            ax.set_ylabel("Success rate (%)", fontsize=18)
            ax.legend(fontsize=14)
        else:
            ax.text(0.5, 0.5, "Use --opponents OP3 OP4\nfor generalization", ha="center", va="center", transform=ax.transAxes, fontsize=16)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=18)
        ax.tick_params(axis="y", labelsize=18)
        ax.set_ylim(0, 105)

        # 4. Stability
        ax = axes[1, 0]
        ax.set_title("Stability (return variance)", fontsize=20)
        rvar = [results[(l, main_opp)]["return_var"] for l in labels]
        ax.bar(x, rvar, color=["#2ecc71", "#3498db", "#9b59b6"])
        ax.set_ylabel("Variance of episode return", fontsize=18)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=18)
        ax.tick_params(axis="y", labelsize=18)
        if training_auc is not None:
            ax.text(0.02, 0.98, f"Training AUC: {training_auc:.3f}", transform=ax.transAxes, fontsize=14, va="top")

        # 5. Specialization
        ax = axes[1, 1]
        ax.set_title("Specialization", fontsize=20)
        ax.text(0.5, 0.5, "Role index / tactical diversity\nrequire action logging", ha="center", va="center", transform=ax.transAxes, fontsize=16)
        ax.set_axis_off()

        # 6. Robotics
        ax = axes[1, 2]
        ax.set_title("Robotics metrics", fontsize=20)
        safety = [results[(l, main_opp)]["collision_free_rate"] for l in labels]
        ax.bar(x, safety, color=["#2ecc71", "#3498db", "#9b59b6"])
        ax.set_ylabel("Safety (collision-free %)", fontsize=18)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=18)
        ax.tick_params(axis="y", labelsize=18)
        ax.set_ylim(0, 105)
        ax.text(0.02, 0.02, "Energy / path optimality: N/A", transform=ax.transAxes, fontsize=14)

        plt.tight_layout()
        out_path = f"{base_out}_{mode}.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved: {out_path}")
    return


if __name__ == "__main__":
    main()
