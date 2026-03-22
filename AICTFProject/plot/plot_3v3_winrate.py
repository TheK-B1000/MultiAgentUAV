#!/usr/bin/env python3
"""
Plot 3v3 win rate: Ours (league), Jacob et al. (paper), Self-play vs a scripted opponent.

Default opponent: OP3 (in-training opponent). Use --opponent OP4 for held-out generalization.

Uses the same evaluation environment as training: GPUCTFVecEnv (game_field_gpu.py),
with max_blue_agents = max_red_agents = 3. Training is done with rl/train_ppo.py.

Usage:
  python plot_3v3_winrate.py [--league PATH] [--paper PATH] [--selfplay PATH] [--episodes N] [--out plot.png]

Defaults (under checkpoints_sb3/3v3/):
  --league   final_ppo_league_3v3.zip
  --paper    final_ppo_paper_3v3.zip
  --selfplay final_ppo_self_play_3v3.zip
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore", message=".*render_mode.*")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from eval_rollout import count_wld, run_eval_episodes


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot 3v3 win rate: Ours vs Jacob et al. vs Self-play")
    parser.add_argument("--league", type=str, default=None, help="Path to League model .zip")
    parser.add_argument("--paper", type=str, default=None, help="Path to Paper model .zip")
    parser.add_argument("--selfplay", type=str, default=None, help="Path to Self-play model .zip")
    parser.add_argument("--episodes", type=int, default=100, help="Evaluation episodes per model")
    parser.add_argument(
        "--opponent",
        type=str,
        default="OP3",
        help="Scripted opponent (OP1, OP2, OP3, OP4). Use OP4 for held-out eval (never in training).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed for eval env (OP4 uses seed+1 to avoid identical streams).",
    )
    parser.add_argument("--match-eval", action="store_true", help="Use OP4, 100 episodes, seed=42 to match plot_eval_metrics paper numbers.")
    parser.add_argument("--match-eval-op3", action="store_true", help="Use OP3 (training-time opponent), 100 episodes, seed=42.")
    parser.add_argument("--out", type=str, default="3v3_winrate.png", help="Output plot path")
    parser.add_argument("--device", type=str, default="cuda", help="Device for eval (cpu or cuda)")
    args = parser.parse_args()
    if args.match_eval:
        args.opponent = "OP4"
        args.episodes = 100
        args.seed = 42
    elif args.match_eval_op3:
        args.opponent = "OP3"
        args.episodes = 100
        args.seed = 42
    # Default output filename includes difficulty (opponent) and episode count
    if args.out == "3v3_winrate.png":
        args.out = f"3v3_winrate_{args.opponent.upper()}_{args.episodes}ep.png"

    # Send plots to AICTFProject/figures/ when --out is a bare filename
    if not os.path.dirname(os.path.abspath(args.out)):
        figures_dir = os.path.join(PROJECT_ROOT, "figures")
        os.makedirs(figures_dir, exist_ok=True)
        args.out = os.path.join(figures_dir, os.path.basename(args.out))

    default_dir = os.path.join(PROJECT_ROOT, "checkpoints_sb3", "3v3")

    def path_or_default(name: str | None, default_name: str) -> str:
        p = name if name is not None else os.path.join(default_dir, default_name)
        if not p.endswith(".zip"):
            p = p + ".zip"
        return os.path.abspath(p)

    league_path = path_or_default(args.league, "final_ppo_league_3v3.zip")
    paper_path = path_or_default(args.paper, "final_ppo_paper_3v3.zip")
    selfplay_path = path_or_default(args.selfplay, "final_ppo_self_play_3v3.zip")

    for label, p in [("Ours", league_path), ("Jacob et al.", paper_path), ("Self-play", selfplay_path)]:
        if not os.path.isfile(p):
            print(f"[WARN] Not found: {p}")
            sys.exit(1)

    from stable_baselines3 import PPO
    from game_field_gpu import GPUCTFVecEnv, GPUFieldConfig
    from rl.train_ppo import MaskedMultiInputPolicy

    device = args.device
    n_episodes = args.episodes
    opponent = args.opponent.upper()
    base_seed = int(args.seed)
    seed = base_seed + (1 if opponent == "OP4" else 0)
    print(f"3v3 win rate vs {opponent} ({n_episodes} episodes per model, seed={seed})")

    cfg = GPUFieldConfig(
        n_envs=1,
        max_blue_agents=3,
        max_red_agents=3,
        max_decision_steps=400,
        aquaticus_profile=True,
        rules_profile="OURS",
        device=device,
        seed=seed,
    )
    env = GPUCTFVecEnv(cfg)
    try:
        env.env_method("set_phase", opponent)
        env.env_method("set_next_opponent", "SCRIPTED", opponent)
        out = env.env_method("get_opponent_key")
        actual = (out[0] if out else "").strip().upper()
        print(f"Opponent: {actual} (requested {opponent})")
        # Log red params so we can confirm OP3 vs OP4 differ (defender_style, role_switch, speed)
        try:
            core = getattr(env, "core", None) or (getattr(env, "vec", None) and getattr(env.vec, "core", None))
            if core is not None:
                ds = int(core.red_defender_style[0].item())
                rs = float(core.red_role_switch_prob[0].item())
                sp = float(core.red_speed_mult[0].item())
                print(f"  red params: defender_style={ds}, role_switch_prob={rs:.2f}, speed_mult={sp:.2f}")
        except Exception:
            pass
        if actual != opponent:
            import warnings

            warnings.warn(
                f"Opponent mismatch: core has {actual!r}, requested {opponent!r}. "
                "Eval may not be vs intended opponent."
            )
    except Exception as e:
        import warnings

        warnings.warn(
            f"Failed to set opponent to {opponent!r}: {e}. "
            "Red team may still be previous opponent."
        )

    # Evaluate each method
    results: dict[str, dict[str, int]] = {}
    for label, path in [("Ours", league_path), ("Jacob et al.", paper_path), ("Self-play", selfplay_path)]:
        print(f"Evaluating {label}: {path} ...")
        episodes = run_eval_episodes(path, env, n_episodes, device, opponent)
        w, l, d = count_wld(episodes)
        total = w + l + d
        results[label] = {"wins": w, "losses": l, "draws": d, "total": total}
        wr = (w / total * 100.0) if total > 0 else 0.0
        print(f"  {label}: W={w} L={l} D={d} WR={wr:.1f}%")

    env.close()

    # Plot
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot. Install with: pip install matplotlib")
        return

    labels = list(results.keys())
    win_rates = [(results[l]["wins"] / max(1, results[l]["total"]) * 100.0) for l in labels]
    colors = ["#2ecc71", "#3498db", "#9b59b6"]
    x = np.arange(len(labels))
    plt.rc("font", size=16)
    bars = plt.bar(x, win_rates, color=colors, edgecolor="black", linewidth=1.2)
    plt.xticks(x, labels, fontsize=18)
    plt.yticks(fontsize=18)
    plt.ylabel(f"Win rate vs {opponent} (%)", fontsize=20)
    plt.title("3v3 Win rate", fontsize=22)
    plt.ylim(0, 105)
    for bar, wr in zip(bars, win_rates):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5, f"{wr:.1f}%", ha="center", fontsize=18)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()

