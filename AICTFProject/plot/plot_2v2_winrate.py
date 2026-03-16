#!/usr/bin/env python3
"""
Plot 2v2 win rate: Ours (league), Jacob et al. (paper), Self-play vs a scripted opponent (default OP3).

Use --opponent OP4 to evaluate vs a held-out opponent (OP4 is never used in training).

Uses the same evaluation environment as training: GPUCTFVecEnv (game_field_gpu.py),
i.e. the same BatchedCTFCore backend. ctfviewer.py is for visual playback only (pygame);
training is done with rl/train_ppo.py.

Usage:
  python plot_2v2_winrate.py [--league PATH] [--paper PATH] [--selfplay PATH] [--episodes N] [--out plot.png]

Defaults (under project checkpoints_sb3/2v2/):
  --league   final_ppo_league_2v2.zip
  --paper    final_ppo_paper_2v2.zip
  --selfplay final_ppo_self_play_2v2.zip
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings

import numpy as np

# Suppress SB3 warning about render_mode (our env does not define it; not needed for eval).
warnings.filterwarnings("ignore", message=".*render_mode.*")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def main():
    parser = argparse.ArgumentParser(description="Plot 2v2 win rate: Ours vs Jacob et al. vs Self-play")
    parser.add_argument("--league", type=str, default=None, help="Path to League model .zip")
    parser.add_argument("--paper", type=str, default=None, help="Path to Paper model .zip")
    parser.add_argument("--selfplay", type=str, default=None, help="Path to Self-play model .zip")
    parser.add_argument("--episodes", type=int, default=100, help="Evaluation episodes per model")
    parser.add_argument("--opponent", type=str, default="OP3", help="Scripted opponent (OP1, OP2, OP3, OP4). Use OP4 for held-out eval (never in training) to compare League vs Paper generalization.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed for eval env (OP4 uses seed+1 to avoid identical streams).")
    parser.add_argument("--match-eval", action="store_true", help="Use OP4, 100 episodes, seed=42 to match plot_eval_metrics paper numbers.")
    parser.add_argument("--match-eval-op3", action="store_true", help="Use OP3 (training-time opponent), 100 episodes, seed=42.")
    parser.add_argument("--out", type=str, default="2v2_winrate.png", help="Output plot path")
    parser.add_argument("--device", type=str, default="cpu", help="Device for eval (cpu or cuda)")
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
    if args.out == "2v2_winrate.png":
        args.out = f"2v2_winrate_{args.opponent.upper()}_{args.episodes}ep.png"

    # Send plots to AICTFProject/figures/ when --out is a bare filename
    if not os.path.dirname(os.path.abspath(args.out)):
        figures_dir = os.path.join(PROJECT_ROOT, "figures")
        os.makedirs(figures_dir, exist_ok=True)
        args.out = os.path.join(figures_dir, os.path.basename(args.out))

    default_dir = os.path.join(PROJECT_ROOT, "checkpoints_sb3", "2v2")
    def path_or_default(name: str, default_name: str) -> str:
        p = name if name is not None else os.path.join(default_dir, default_name)
        if not p.endswith(".zip"):
            p = p + ".zip"
        return os.path.abspath(p)

    league_path = path_or_default(args.league, "final_ppo_league_2v2.zip")
    paper_path = path_or_default(args.paper, "final_ppo_paper_2v2.zip")
    selfplay_path = path_or_default(args.selfplay, "final_ppo_self_play_2v2.zip")

    for label, p in [("Ours", league_path), ("Jacob et al.", paper_path), ("Self-play", selfplay_path)]:
        if not os.path.isfile(p):
            print(f"[WARN] Not found: {p}")
            sys.exit(1)

    from stable_baselines3 import PPO
    from game_field_gpu import GPUCTFVecEnv, GPUFieldConfig

    device = args.device
    n_episodes = args.episodes
    opponent = args.opponent.upper()
    # Different seed per opponent so OP3 vs OP4 runs don't replay the same RNG stream
    base_seed = int(args.seed)
    seed = base_seed + (1 if opponent == "OP4" else 0)
    print(f"2v2 win rate vs {opponent} ({n_episodes} episodes per model, seed={seed})")

    cfg = GPUFieldConfig(
        n_envs=1,
        max_blue_agents=2,
        max_red_agents=2,
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
            warnings.warn(f"Opponent mismatch: core has {actual!r}, requested {opponent!r}. Eval may not be vs intended opponent.")
    except Exception as e:
        import warnings
        warnings.warn(f"Failed to set opponent to {opponent!r}: {e}. Red team may still be previous opponent.")

    from rl.train_ppo import MaskedMultiInputPolicy

    def _numpy_compat_shim():
        """Allow loading models saved on Colab (NumPy 2.x) when running on NumPy 1.x. Register only at load time."""
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
        # NumPy 2.x pickles pass BitGenerator class (e.g. numpy.random._pcg64.PCG64); 1.x expects string name.
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
        """Run n_episodes, return (wins, losses, draws)."""
        _numpy_compat_shim()
        # Use eval env's spaces and policy class so checkpoints that fail to deserialize (e.g. different Python) still load.
        custom = {
            "observation_space": env.observation_space,
            "action_space": env.action_space,
            "policy_class": MaskedMultiInputPolicy,
        }
        model = PPO.load(model_path, device=device, custom_objects=custom)
        model.policy.set_training_mode(False)
        wins, losses, draws = 0, 0, 0
        obs = env.reset()
        for ep in range(n_episodes):
            while True:
                # VecEnv obs: dict of (1, ...); SB3 predict accepts single-env dict
                single = {k: v[0] if hasattr(v, "shape") and len(v.shape) > 1 and v.shape[0] == 1 else v for k, v in obs.items()}
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
                    # step_wait already reset done envs and returned new obs
                    break
        return wins, losses, draws

    # Run evaluation for each model
    results = {}
    for label, path in [("Ours", league_path), ("Jacob et al.", paper_path), ("Self-play", selfplay_path)]:
        print(f"Evaluating {label}: {path} ...")
        w, l, d = run_eval(path)
        results[label] = {"wins": w, "losses": l, "draws": d, "total": w + l + d}
        wr = (w / (w + l + d) * 100) if (w + l + d) > 0 else 0.0
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
    win_rates = [(results[l]["wins"] / max(1, results[l]["total"]) * 100) for l in labels]
    colors = ["#2ecc71", "#3498db", "#9b59b6"]
    x = np.arange(len(labels))
    plt.rc("font", size=16)
    bars = plt.bar(x, win_rates, color=colors, edgecolor="black", linewidth=1.2)
    plt.xticks(x, labels, fontsize=18)
    plt.yticks(fontsize=18)
    plt.ylabel("Win rate vs " + opponent + " (%)", fontsize=20)
    plt.title("2v2 Win rate", fontsize=22)
    plt.ylim(0, 105)
    for i, (bar, wr) in enumerate(zip(bars, win_rates)):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5, f"{wr:.1f}%", ha="center", fontsize=18)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
