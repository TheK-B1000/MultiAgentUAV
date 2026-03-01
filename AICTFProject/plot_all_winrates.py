#!/usr/bin/env python3
"""
Plot win rate for 2v2, 3v3, and 4v4: League vs Paper vs Self-play vs OP3.

Evaluates 9 models total (3 per team size) and produces one figure with 3 panels.

Usage:
  python plot_all_winrates.py [--episodes N] [--out plot.png] [--checkpoint-dir DIR]

Defaults (under checkpoints_sb3/):
  2v2: final_ppo_league_2v2_colab.zip, final_ppo_paper_2v2_colab.zip, final_ppo_selfplay_2v2_colab.zip
  3v3: final_ppo_league_3v3_colab.zip, final_ppo_paper_3v3_colab.zip, final_ppo_selfplay_3v3_colab.zip
  4v4: final_ppo_league_4v4_colab.zip, final_ppo_paper_4v4_colab.zip, final_ppo_selfplay_4v4_colab.zip
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore", message=".*render_mode.*")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# Default filenames per team size (league, paper, selfplay)
DEFAULTS = {
    2: (
        "final_ppo_league_2v2_colab.zip",
        "final_ppo_paper_2v2_colab.zip",
        "final_ppo_selfplay_2v2_colab.zip",
    ),
    3: (
        "final_ppo_league_3v3_colab.zip",
        "final_ppo_paper_3v3_colab.zip",
        "final_ppo_selfplay_3v3_colab.zip",
    ),
    4: (
        "final_ppo_league_4v4_colab.zip",
        "final_ppo_paper_4v4_colab.zip",
        "final_ppo_selfplay_4v4_colab.zip",
    ),
}
METHOD_LABELS = ("League", "Paper", "Self-play")


def path_ensure_zip(path: str) -> str:
    return path if path.endswith(".zip") else path + ".zip"


def main():
    parser = argparse.ArgumentParser(description="Plot 2v2, 3v3, 4v4 win rates: League vs Paper vs Self-play")
    parser.add_argument("--episodes", type=int, default=100, help="Evaluation episodes per model")
    parser.add_argument("--opponent", type=str, default="OP3", help="Scripted opponent (OP1, OP2, OP3)")
    parser.add_argument("--out", type=str, default="all_winrates.png", help="Output plot path")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Directory containing .zip files (default: checkpoints_sb3)")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    args = parser.parse_args()

    ckpt_dir = args.checkpoint_dir or os.path.join(SCRIPT_DIR, "checkpoints_sb3")
    ckpt_dir = os.path.abspath(ckpt_dir)

    from stable_baselines3 import PPO
    from game_field_gpu import GPUCTFVecEnv, GPUFieldConfig

    device = args.device
    n_episodes = args.episodes
    opponent = args.opponent.upper()

    # results[team_size][method] = {"wins": w, "losses": l, "draws": d, "total": t}
    results = {2: {}, 3: {}, 4: {}}

    for n_agents in (2, 3, 4):
        league_name, paper_name, selfplay_name = DEFAULTS[n_agents]
        paths = {
            "League": os.path.join(ckpt_dir, path_ensure_zip(league_name)),
            "Paper": os.path.join(ckpt_dir, path_ensure_zip(paper_name)),
            "Self-play": os.path.join(ckpt_dir, path_ensure_zip(selfplay_name)),
        }
        for method, p in paths.items():
            if not os.path.isfile(p):
                print(f"[WARN] Not found: {p}")
                results[n_agents][method] = {"wins": 0, "losses": 0, "draws": 0, "total": 0}
                continue

        cfg = GPUFieldConfig(
            n_envs=1,
            max_blue_agents=n_agents,
            max_red_agents=n_agents,
            max_decision_steps=400,
            aquaticus_profile=True,
            rules_profile="AQUATICUS_2024",
            device=device,
            seed=42,
        )
        env = GPUCTFVecEnv(cfg)
        try:
            env.env_method("set_phase", opponent)
            env.env_method("set_next_opponent", "SCRIPTED", opponent)
        except Exception:
            pass

        def _numpy_compat_shim():
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

                def _patched_bg_ctor(bit_generator_name="MT19937"):
                    if isinstance(bit_generator_name, type):
                        bit_generator_name = bit_generator_name.__name__
                    return _orig_bg_ctor(bit_generator_name)

                _np_pickle.__bit_generator_ctor = _patched_bg_ctor
            except Exception:
                pass

        def run_eval(model_path: str) -> tuple[int, int, int]:
            _numpy_compat_shim()
            custom = {"observation_space": env.observation_space, "action_space": env.action_space}
            model = PPO.load(model_path, device=device, custom_objects=custom)
            model.policy.set_training_mode(False)
            wins, losses, draws = 0, 0, 0
            obs = env.reset()
            for ep in range(n_episodes):
                while True:
                    single = {
                        k: v[0] if hasattr(v, "shape") and len(v.shape) > 1 and v.shape[0] == 1 else v
                        for k, v in obs.items()
                    }
                    act, _ = model.predict(single, deterministic=True)
                    env.step_async(act)
                    obs, _, done, infos = env.step_wait()
                    if done.any():
                        for i in range(len(done)):
                            if done[i]:
                                info = infos[i] if i < len(infos) else {}
                                ep_res = info.get("episode_result", info)
                                bs = int(ep_res.get("blue_score", 0))
                                rs = int(ep_res.get("red_score", 0))
                                if bs > rs:
                                    wins += 1
                                elif bs < rs:
                                    losses += 1
                                else:
                                    draws += 1
                        break
            return wins, losses, draws

        suffix = f"{n_agents}v{n_agents}"
        print(f"--- {suffix} ---")
        for method in METHOD_LABELS:
            p = paths[method]
            if not os.path.isfile(p):
                continue
            print(f"  Evaluating {method}: {os.path.basename(p)} ...")
            w, l, d = run_eval(p)
            results[n_agents][method] = {"wins": w, "losses": l, "draws": d, "total": w + l + d}
            t = w + l + d
            wr = (w / t * 100) if t > 0 else 0.0
            print(f"    {method}: W={w} L={l} D={d} WR={wr:.1f}%")

        env.close()

    # Plot: 3 subplots (2v2, 3v3, 4v4)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot. Install with: pip install matplotlib")
        return

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    colors = ["#2ecc71", "#3498db", "#9b59b6"]

    for idx, n_agents in enumerate((2, 3, 4)):
        ax = axes[idx]
        suffix = f"{n_agents}v{n_agents}"
        method_order = [m for m in METHOD_LABELS if results[n_agents].get(m) and results[n_agents][m]["total"] > 0]
        if not method_order:
            ax.set_title(suffix)
            ax.set_ylim(0, 105)
            continue
        win_rates = [
            results[n_agents][m]["wins"] / max(1, results[n_agents][m]["total"]) * 100
            for m in method_order
        ]
        x = np.arange(len(method_order))
        bars = ax.bar(x, win_rates, color=colors[: len(method_order)], edgecolor="black", linewidth=1.2)
        ax.set_xticks(x)
        ax.set_xticklabels(method_order)
        ax.set_ylabel("Win rate vs " + opponent + " (%)")
        ax.set_title(suffix)
        ax.set_ylim(0, 105)
        for bar, wr in zip(bars, win_rates):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5, f"{wr:.1f}%", ha="center", fontsize=10)

    plt.suptitle(f"Win rate ({n_episodes} episodes each)")
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
