#!/usr/bin/env python3
"""
Plot evaluation metrics by category: Performance, Coordination, Robustness, Stability, Specialization, Robotics.

Runs the same eval as plot_2v2_winrate (GPUCTFVecEnv, optional OP3/OP4) and collects:
  - Performance: success rate, mean steps to completion
  - Coordination: coverage efficiency (zone_coverage), coordination proxy (collision-free)
  - Robustness: generalization (success vs OP4 vs OP3 if --opponents OP3 OP4)
  - Stability: return variance across episodes
  - Specialization: placeholder (requires action logging)
  - Robotics: safety (collision-free rate); energy/path N/A

Usage:
  python plot_eval_metrics.py [--league PATH] [--paper PATH] [--selfplay PATH] [--episodes N]
  python plot_eval_metrics.py --opponents OP3 OP4 --episodes 50 --out eval_metrics.png
  python plot_eval_metrics.py --training-csv logs/behavior_ppo.csv   # add AUC learning curve if CSV has episode_id, success
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

    _numpy_compat_shim()
    custom = {"observation_space": env.observation_space, "action_space": env.action_space}
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
    parser = argparse.ArgumentParser(description="Plot evaluation metrics by category")
    parser.add_argument("--league", type=str, default=None)
    parser.add_argument("--paper", type=str, default=None)
    parser.add_argument("--selfplay", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--opponent", type=str, default="OP3", help="Single opponent for main eval")
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

    league_path = path_or_default(args.league, "final_ppo_league_2v2_colab.zip")
    paper_path = path_or_default(args.paper, "final_ppo_paper_2v2_colab.zip")
    selfplay_path = path_or_default(args.selfplay, "final_ppo_selfplay_2v2_colab.zip")

    model_paths = [
        ("League", league_path),
        ("Paper", paper_path),
        ("Self-play", selfplay_path),
    ]
    for _label, p in model_paths:
        if not os.path.isfile(p):
            print(f"[WARN] Not found: {p}")
            sys.exit(1)

    from game_field_gpu import GPUCTFVecEnv, GPUFieldConfig

    cfg = GPUFieldConfig(
        n_envs=1,
        max_blue_agents=2,
        max_red_agents=2,
        max_decision_steps=400,
        aquaticus_profile=True,
        rules_profile="AQUATICUS_2024",
        device=args.device,
        seed=42,
    )
    env = GPUCTFVecEnv(cfg)

    opponents = args.opponents if args.opponents else [args.opponent]
    n_episodes = args.episodes

    # results[(label, opponent)] = aggregate dict
    results: dict[tuple[str, str], dict] = {}
    for label, model_path in model_paths:
        for opp in opponents:
            print(f"Evaluating {label} vs {opp} ({n_episodes} episodes)...")
            episodes = run_eval_episodes(model_path, env, n_episodes, args.device, opp)
            results[(label, opp)] = compute_aggregates(episodes)

    # Optional training AUC (per-run, not per-model; we don't have per-model CSVs by default)
    training_auc: float | None = None
    if args.training_csv and os.path.isfile(args.training_csv):
        training_auc = load_training_success_auc(args.training_csv)
        if training_auc is not None:
            print(f"Training AUC (success curve): {training_auc:.4f}")

    # Plot: 2x3 grid by category
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle(f"Evaluation metrics ({n_episodes} episodes each)", fontsize=14)

    labels = [m[0] for m in model_paths]
    x = np.arange(len(labels))
    width = 0.25

    # Main opponent for single-metric panels (use first in list)
    main_opp = opponents[0]

    # 1. Performance: success rate, mean steps to completion
    ax = axes[0, 0]
    ax.set_title("Performance")
    sr = [results[(l, main_opp)]["success_rate"] for l in labels]
    bars1 = ax.bar(x - width / 2, sr, width, label="Success rate (%)", color="#2ecc71")
    ax.set_ylabel("Success rate (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 105)
    ax2 = ax.twinx()
    steps = [results[(l, main_opp)]["mean_steps"] for l in labels]
    bars2 = ax2.bar(x + width / 2, steps, width, label="Mean steps to done", color="#3498db", alpha=0.7)
    ax2.set_ylabel("Mean steps (lower better)")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")

    # 2. Coordination: coverage efficiency, collision-free rate
    ax = axes[0, 1]
    ax.set_title("Coordination")
    cov = [results[(l, main_opp)]["coverage_efficiency"] for l in labels]
    cf = [results[(l, main_opp)]["collision_free_rate"] for l in labels]
    ax.bar(x - width / 2, cov, width, label="Coverage efficiency (%)", color="#9b59b6")
    ax.bar(x + width / 2, cf, width, label="Collision-free rate (%)", color="#e74c3c", alpha=0.7)
    ax.set_ylabel("%")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 105)
    ax.legend()

    # 3. Robustness: generalization (OP4 vs OP3) if we have both opponents
    ax = axes[0, 2]
    ax.set_title("Robustness")
    if "OP4" in opponents and "OP3" in opponents:
        sr_op3 = [results.get((l, "OP3"), {}).get("success_rate", 0) for l in labels]
        sr_op4 = [results.get((l, "OP4"), {}).get("success_rate", 0) for l in labels]
        ax.bar(x - width / 2, sr_op3, width, label="Success vs OP3 (%)", color="#3498db")
        ax.bar(x + width / 2, sr_op4, width, label="Success vs OP4 (%)", color="#e67e22")
        ax.set_ylabel("Success rate (%)")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "Use --opponents OP3 OP4\nfor generalization", ha="center", va="center", transform=ax.transAxes)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 105)

    # 4. Stability: return variance
    ax = axes[1, 0]
    ax.set_title("Stability (return variance)")
    rvar = [results[(l, main_opp)]["return_var"] for l in labels]
    ax.bar(x, rvar, color=["#2ecc71", "#3498db", "#9b59b6"])
    ax.set_ylabel("Variance of episode return")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    if training_auc is not None:
        ax.text(0.02, 0.98, f"Training AUC: {training_auc:.3f}", transform=ax.transAxes, fontsize=8, va="top")

    # 5. Specialization: placeholder
    ax = axes[1, 1]
    ax.set_title("Specialization")
    ax.text(0.5, 0.5, "Role index / tactical diversity\nrequire action logging", ha="center", va="center", transform=ax.transAxes)
    ax.set_axis_off()

    # 6. Robotics: safety (collision-free), energy/path N/A
    ax = axes[1, 2]
    ax.set_title("Robotics metrics")
    safety = [results[(l, main_opp)]["collision_free_rate"] for l in labels]
    ax.bar(x, safety, color=["#2ecc71", "#3498db", "#9b59b6"])
    ax.set_ylabel("Safety (collision-free %)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 105)
    ax.text(0.02, 0.02, "Energy / path optimality: N/A", transform=ax.transAxes, fontsize=7)

    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    plt.close()
    print(f"Saved: {args.out}")
    return


if __name__ == "__main__":
    main()
