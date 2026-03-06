#!/usr/bin/env python3
"""
Plot NvN win rate: Ours (league), Jacob et al. (paper), Self-play vs a scripted opponent.

Checkpoint and env must match team size: 4v4 models work only in 4v4 env, 6v6 models only in 6v6 env.
Use --agents 4 with 4v4 checkpoints (default). Use --agents 6 only if you have 6v6-trained checkpoints.

Default opponent: OP4. Use --opponent OP3 for in-training opponent.
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot NvN win rate. Checkpoint team size must match --agents (4v4 checkpoints require --agents 4)."
    )
    parser.add_argument("--agents", type=int, default=4, choices=[4, 6], help="Team size (4 or 6). Default 4 for 4v4 checkpoints.")
    parser.add_argument("--league", type=str, default=None, help="Path to League model .zip")
    parser.add_argument("--paper", type=str, default=None, help="Path to Paper model .zip")
    parser.add_argument("--selfplay", type=str, default=None, help="Path to Self-play model .zip")
    parser.add_argument("--episodes", type=int, default=25, help="Evaluation episodes per model")
    parser.add_argument(
        "--opponent",
        type=str,
        default="OP4",
        help="Scripted opponent (OP1, OP2, OP3, OP4). Default OP4 (harder, held-out).",
    )
    parser.add_argument("--out", type=str, default="6v6_winrate.png", help="Output plot path")
    parser.add_argument("--device", type=str, default="cpu", help="Device for eval (cpu or cuda)")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed (OP4 uses seed+1).")
    args = parser.parse_args()

    # Send plots to AICTFProject/figures/ when --out is a bare filename
    project_root = os.path.dirname(SCRIPT_DIR)
    if not os.path.dirname(os.path.abspath(args.out)):
        figures_dir = os.path.join(project_root, "figures")
        os.makedirs(figures_dir, exist_ok=True)
        args.out = os.path.join(figures_dir, os.path.basename(args.out))

    default_dir = os.path.join(SCRIPT_DIR, "checkpoints_sb3")

    def path_or_default(name: str | None, default_name: str) -> str:
        p = name if name is not None else os.path.join(default_dir, default_name)
        if not p.endswith(".zip"):
            p = p + ".zip"
        return os.path.abspath(p)

    n_agents = args.agents
    suffix = "4v4_colab" if n_agents == 4 else "6v6"
    league_path = path_or_default(args.league, f"final_ppo_league_{suffix}.zip")
    paper_path = path_or_default(args.paper, f"final_weekend_paper_{'4v4' if n_agents == 4 else '6v6'}.zip")
    selfplay_path = path_or_default(args.selfplay, f"final_ppo_selfplay_{suffix}.zip")

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
    print(f"{n_agents}v{n_agents} win rate vs {opponent} ({n_episodes} episodes per model, seed={seed})")

    cfg = GPUFieldConfig(
        n_envs=1,
        max_blue_agents=n_agents,
        max_red_agents=n_agents,
        max_decision_steps=400,
        aquaticus_profile=True,
        rules_profile="AQUATICUS_2024",
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
        # Log red params for sanity
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

    def _numpy_compat_shim() -> None:
        """Allow loading models saved on Colab (NumPy 2.x) when running on NumPy 1.x."""
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

            def _patched_bg_ctor(bit_generator_name: object = "MT19937") -> object:
                if isinstance(bit_generator_name, type):
                    bit_generator_name = bit_generator_name.__name__
                return _orig_bg_ctor(bit_generator_name)

            _np_pickle.__bit_generator_ctor = _patched_bg_ctor
        except Exception:
            pass

    def run_eval(model_path: str) -> tuple[int, int, int]:
        _numpy_compat_shim()
        custom = {
            "observation_space": env.observation_space,
            "action_space": env.action_space,
            "policy_class": MaskedMultiInputPolicy,
        }
        try:
            model = PPO.load(model_path, device=device, custom_objects=custom)
        except RuntimeError as e:
            if "size mismatch" in str(e) or "shape" in str(e).lower():
                sys.exit(
                    f"Checkpoint {model_path!r} was trained for a different team size.\n"
                    f"4v4 models cannot run in {n_agents}v{n_agents} env (obs/action dims differ).\n"
                    f"Use --agents 4 with 4v4 checkpoints, or train {n_agents}v{n_agents} models."
                )
            raise
        model.policy.set_training_mode(False)
        wins, losses, draws = 0, 0, 0
        obs = env.reset()
        for _ in range(n_episodes):
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

    results: dict[str, dict[str, int]] = {}
    for label, path in [("Ours", league_path), ("Jacob et al.", paper_path), ("Self-play", selfplay_path)]:
        print(f"Evaluating {label}: {path} ...")
        w, l, d = run_eval(path)
        total = w + l + d
        results[label] = {"wins": w, "losses": l, "draws": d, "total": total}
        wr = (w / total * 100.0) if total > 0 else 0.0
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
    win_rates = [(results[l]["wins"] / max(1, results[l]["total"]) * 100.0) for l in labels]
    colors = ["#2ecc71", "#3498db", "#9b59b6"]
    x = np.arange(len(labels))
    plt.rc("font", size=16)
    bars = plt.bar(x, win_rates, color=colors, edgecolor="black", linewidth=1.2)
    plt.xticks(x, labels, fontsize=18)
    plt.yticks(fontsize=18)
    plt.ylabel(f"Win rate vs {opponent} (%)", fontsize=20)
    plt.title(f"{n_agents}v{n_agents} Win rate", fontsize=22)
    plt.ylim(0, 105)
    for bar, wr in zip(bars, win_rates):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5, f"{wr:.1f}%", ha="center", fontsize=18)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()

