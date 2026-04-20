#!/usr/bin/env python3
"""
Plot 4v4 win rate: Ours (league), Jacob et al. (paper), Self-play vs a scripted opponent.

Default opponent: OP3 (in-training opponent). Use --opponent OP4 for held-out generalization.

Uses the same evaluation environment as training: GPUCTFVecEnv (game_field_gpu.py).
Training is done with rl/train_ppo.py.

Usage:
  python plot_4v4_winrate.py [--league PATH] [--paper PATH] [--selfplay PATH] [--episodes N] [--out plot.png]
  python plot_4v4_winrate.py --opponents OP3 OP4 [--episodes 100 --seed 42]
  python plot_4v4_winrate.py --match-eval-op34   # OP3 then OP4, 100 ep, seed 42 (two figures)

Defaults (under checkpoints_sb3/4v4/):
  --league   final_ppo_league_4v4.zip
  --paper    final_ppo_paper_4v4.zip
  --selfplay final_ppo_self_play_4v4.zip
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

from eval_rollout import binomial_se, count_wld, run_eval_episodes


def main():
    parser = argparse.ArgumentParser(description="Plot 4v4 win rate: Ours vs Jacob et al. vs Self-play")
    parser.add_argument("--league", type=str, default=None, help="Path to League model .zip")
    parser.add_argument("--paper", type=str, default=None, help="Path to Paper model .zip")
    parser.add_argument("--selfplay", type=str, default=None, help="Path to Self-play model .zip")
    parser.add_argument("--episodes", type=int, default=100, help="Evaluation episodes per model")
    parser.add_argument(
        "--opponents",
        nargs="+",
        default=None,
        metavar="OP",
        help="Evaluate vs each opponent in order (e.g. OP3 OP4). Overrides --opponent.",
    )
    parser.add_argument("--opponent", type=str, default="OP3", help="Scripted opponent (OP1, OP2, OP3, OP4). Use OP4 for held-out eval (never in training).")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed (OP4 uses seed+1). Use --seed 42 to match plot_eval_metrics.")
    parser.add_argument("--match-eval", action="store_true", help="Use OP4, 100 episodes, seed=42 to match plot_eval_metrics paper numbers.")
    parser.add_argument("--match-eval-op3", action="store_true", help="Use OP3 (training-time opponent), 100 episodes, seed=42.")
    parser.add_argument(
        "--match-eval-op34",
        action="store_true",
        help="Run vs OP3 then OP4 (100 episodes each, seed=42). Writes two figures under figures/.",
    )
    parser.add_argument("--out", type=str, default="4v4_winrate.png", help="Output plot path")
    parser.add_argument("--device", type=str, default="cuda", help="Device for eval (cpu or cuda)")
    args = parser.parse_args()
    if args.match_eval_op34:
        if args.opponents is None:
            args.opponents = ["OP3", "OP4"]
        args.episodes = 100
        args.seed = 42
    if args.opponents:
        opponents_list = [o.strip().upper() for o in args.opponents]
    elif args.match_eval:
        args.opponent = "OP4"
        args.episodes = 100
        args.seed = 42
        opponents_list = [args.opponent.upper()]
    elif args.match_eval_op3:
        args.opponent = "OP3"
        args.episodes = 100
        args.seed = 42
        opponents_list = [args.opponent.upper()]
    else:
        opponents_list = [args.opponent.upper()]

    raw_out = args.out
    is_default_out = raw_out == "4v4_winrate.png"
    figures_dir = os.path.join(PROJECT_ROOT, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    def out_path_for(op: str) -> str:
        if is_default_out:
            fname = f"4v4_winrate_{op}_{args.episodes}ep.png"
        elif len(opponents_list) > 1:
            stem, ext = os.path.splitext(os.path.basename(raw_out))
            fname = f"{stem}_{op}{ext}"
        else:
            fname = os.path.basename(raw_out)
        d = os.path.dirname(raw_out)
        if not d:
            return os.path.join(figures_dir, fname)
        return os.path.join(os.path.abspath(d), fname)

    default_dir = os.path.join(PROJECT_ROOT, "checkpoints_sb3", "4v4")

    def path_or_default(name: str, default_name: str) -> str:
        p = name if name is not None else os.path.join(default_dir, default_name)
        if not p.endswith(".zip"):
            p = p + ".zip"
        return os.path.abspath(p)

    league_path = path_or_default(args.league, "final_ppo_league_4v4.zip")
    paper_path = path_or_default(args.paper, "final_ppo_paper_4v4.zip")
    selfplay_path = path_or_default(args.selfplay, "final_ppo_self_play_4v4.zip")

    for label, p in [("Ours", league_path), ("Jacob et al.", paper_path), ("Self-play", selfplay_path)]:
        if not os.path.isfile(p):
            print(f"[WARN] Not found: {p}")
            sys.exit(1)

    from game_field_gpu import GPUCTFVecEnv, GPUFieldConfig

    device = args.device
    n_episodes = args.episodes
    base_seed = int(args.seed)

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

    for opponent in opponents_list:
        seed = base_seed + (1 if opponent == "OP4" else 0)
        print(f"4v4 win rate vs {opponent} ({n_episodes} episodes per model, seed={seed})")

        cfg = GPUFieldConfig(
            n_envs=1,
            max_blue_agents=4,
            max_red_agents=4,
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
                    f"Opponent mismatch: core has {actual!r}, requested {opponent!r}. Eval may not be vs intended opponent."
                )
        except Exception as e:
            import warnings

            warnings.warn(f"Failed to set opponent to {opponent!r}: {e}. Red team may still be previous opponent.")

        results: dict = {}
        for label, path in [("Ours", league_path), ("Jacob et al.", paper_path), ("Self-play", selfplay_path)]:
            print(f"Evaluating {label}: {path} ...")
            episodes = run_eval_episodes(path, env, n_episodes, device, opponent)
            w, l, d = count_wld(episodes)
            results[label] = {"wins": w, "losses": l, "draws": d, "total": w + l + d}
            wr = (w / (w + l + d) * 100) if (w + l + d) > 0 else 0.0
            print(f"  {label}: W={w} L={l} D={d} WR={wr:.1f}%")

        env.close()

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed; skipping plot. Install with: pip install matplotlib")
            return

        labels = list(results.keys())
        win_rates = [(results[l]["wins"] / max(1, results[l]["total"]) * 100) for l in labels]
        ses = [binomial_se(results[l]["wins"], results[l]["total"]) for l in labels]
        colors = ["#2ecc71", "#3498db", "#9b59b6"]
        x = np.arange(len(labels))
        plt.rc("font", size=16)
        plt.figure()
        bars = plt.bar(
            x, win_rates, color=colors, edgecolor="black", linewidth=1.2,
            yerr=ses, capsize=6, error_kw={"elinewidth": 1.6, "ecolor": "black"},
        )
        plt.xticks(x, labels, fontsize=18)
        plt.yticks(fontsize=18)
        plt.ylabel("Win rate vs " + opponent + " (%)", fontsize=20)
        plt.title("4v4 Win rate", fontsize=22)
        plt.ylim(0, 115)
        for bar, wr, se in zip(bars, win_rates, ses):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + se + 2.0,
                f"{wr:.1f}% \u00b1 {se:.1f}",
                ha="center", fontsize=16,
            )
        plt.tight_layout()
        out_file = out_path_for(opponent)
        plt.savefig(out_file, dpi=150)
        plt.close()
        print(f"Saved: {out_file}")


if __name__ == "__main__":
    main()
