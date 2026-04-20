#!/usr/bin/env python3
"""
Testing-only robustness experiment: 10 randomly-placed mines, unknown at deployment.

NO TRAINING. We load the already-trained PPO checkpoints (Ours vs Jacob et al.)
and evaluate them in the SAME envs used for the paper's win-rate plots, but
every episode starts with 10 extra live mines sprinkled randomly across the
map:
    * 5 RED mines, random positions in the RED half       (hazard to BLUE as it
                                                            advances to grab the
                                                            red flag)
    * 5 BLUE mines, random positions in the BLUE half     (hazard to RED as it
                                                            advances to grab the
                                                            blue flag)

By default we sweep 2v2, 3v3, 4v4 (per user request) and produce one combined
3-panel figure + per-size figures, all with ±1 binomial-SE bars.

Why no retraining is needed
---------------------------
The policy's observation is built with per-agent CNN channels (mines are
scattered onto a single channel) and a scalar `nearest own-mine distance`
feature (see `_build_grid_obs` / `_build_vec_obs` in game_field_gpu.py). Both
reduce over the mine dimension, so observation SHAPE does not depend on
`max_mines_per_team`. We only need to allocate enough slots (5 per team) and
write random positions after each reset.

Usage (from project root)
-------------------------
    python plot/test_mines_robustness.py --episodes 100 --opponent OP3 \
        --device cuda --seed 42

    # Subset of sizes
    python plot/test_mines_robustness.py --team-sizes 2 3 --episodes 100

    # Single size still supported
    python plot/test_mines_robustness.py --team-sizes 2 --episodes 100

Outputs:
  * Console W/L/D and win rate per (team size, model).
  * figures/mines_robustness_<OP>_<N>ep_<NvN>.png   per-size bar chart
  * figures/mines_robustness_<OP>_<N>ep_all.png     combined 3-panel chart
  * csv/mines_robustness_<OP>_<N>ep.csv             per-episode results
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from typing import Callable

import numpy as np
import torch

warnings.filterwarnings("ignore", message=".*render_mode.*")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from eval_rollout import binomial_se, count_wld, ppo_load_custom_objects  # noqa: E402


def _install_random_mine_hook(
    core,
    *,
    n_red_mines: int,
    n_blue_mines: int,
    red_x_range: tuple[float, float],
    blue_x_range: tuple[float, float],
    y_range: tuple[float, float],
    rng: torch.Generator,
) -> Callable[[], None]:
    """Monkey-patch ``core.reset_indices`` to seed random mines after every reset.

    Returns an 'uninstaller' closure that restores the original method.
    """
    device = core.device
    orig_reset_indices = core.reset_indices
    Nm = int(core.Nm)
    if n_red_mines > Nm or n_blue_mines > Nm:
        raise RuntimeError(
            f"Not enough mine slots: need {max(n_red_mines, n_blue_mines)} per team but "
            f"env was built with max_mines_per_team={Nm}. Rebuild env with a higher value."
        )

    def _uniform(shape, low: float, high: float) -> torch.Tensor:
        return torch.rand(shape, generator=rng, device=device) * (high - low) + low

    def patched_reset_indices(env_mask: torch.Tensor) -> None:
        orig_reset_indices(env_mask)
        idx = torch.where(env_mask)[0]
        if idx.numel() == 0:
            return
        ne = int(idx.numel())
        for slot in range(n_red_mines):
            core.red_mine_x[idx, slot] = _uniform((ne,), *red_x_range)
            core.red_mine_y[idx, slot] = _uniform((ne,), *y_range)
            core.red_mine_active[idx, slot] = True
        for slot in range(n_blue_mines):
            core.blue_mine_x[idx, slot] = _uniform((ne,), *blue_x_range)
            core.blue_mine_y[idx, slot] = _uniform((ne,), *y_range)
            core.blue_mine_active[idx, slot] = True

    core.reset_indices = patched_reset_indices

    def _uninstall() -> None:
        core.reset_indices = orig_reset_indices

    return _uninstall


def _run_eval(
    model_path: str,
    env,
    n_episodes: int,
    device: str,
    opponent: str,
    *,
    deterministic: bool = True,
    progress_every: int = 10,
) -> list[dict]:
    """Like eval_rollout.run_eval_episodes but also force-seeds mines on ep 1."""
    from stable_baselines3 import PPO

    model = PPO.load(model_path, device=device, custom_objects=ppo_load_custom_objects(env))
    model.policy.set_training_mode(False)
    if progress_every > 0:
        print(f"  checkpoint loaded; {n_episodes} episodes (deterministic={deterministic})", flush=True)

    env.env_method("set_phase", opponent)
    env.env_method("set_next_opponent", "SCRIPTED", opponent)
    obs = env.reset()
    core = env.core
    all_mask = torch.ones((core.B,), dtype=torch.bool, device=core.device)
    # env.reset() called core.reset_all (no mines seeded). Re-run our patched
    # reset_indices on every env so the FIRST episode starts with mines.
    core.reset_indices(all_mask)
    obs = core.get_obs()

    episodes: list[dict] = []
    for _ in range(n_episodes):
        ep_return = 0.0
        steps = 0
        while True:
            single = {
                k: v[0] if hasattr(v, "shape") and len(v.shape) > 1 and v.shape[0] == 1 else v
                for k, v in obs.items()
            }
            act, _ = model.predict(single, deterministic=deterministic)
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
                        episodes.append(
                            {
                                "blue_score": bs,
                                "red_score": rs,
                                "success": 1 if bs > rs else 0,
                                "decision_steps": int(ep_res.get("decision_steps", steps)),
                                "return": ep_return,
                            }
                        )
                        if progress_every > 0:
                            le = len(episodes)
                            if le == 1 or le % progress_every == 0 or le == n_episodes:
                                print(f"  episode {le}/{n_episodes}", flush=True)
                break
    return episodes


def _run_for_team_size(
    team_size: int,
    *,
    league_path: str,
    paper_path: str,
    opponent: str,
    n_episodes: int,
    device: str,
    seed: int,
    n_red_mines: int,
    n_blue_mines: int,
    max_mines_per_team: int,
    progress_every: int,
    deterministic: bool,
) -> dict[str, dict]:
    """Build env for ``NvN``, install mine hook, evaluate Ours vs Jacob, return results."""
    from game_field_gpu import GPUCTFVecEnv, GPUFieldConfig  # noqa: E402

    cfg = GPUFieldConfig(
        n_envs=1,
        max_blue_agents=team_size,
        max_red_agents=team_size,
        max_decision_steps=400,
        aquaticus_profile=True,
        rules_profile="OURS",
        device=device,
        seed=seed,
        max_mines_per_team=max(max_mines_per_team, n_red_mines, n_blue_mines),
    )
    env = GPUCTFVecEnv(cfg)
    core = env.core
    cols = float(core.cols)
    rows = float(core.rows)
    half = cols / 2.0
    margin = 1.0
    red_x_range = (half + margin, cols - 1 - margin)
    blue_x_range = (margin, half - margin)
    y_range = (margin, rows - 1 - margin)

    rng = torch.Generator(device=core.device)
    rng.manual_seed(int(seed) * 7919 + 1)

    print(
        f"\n[test] {team_size}v{team_size} vs {opponent}  "
        f"| episodes={n_episodes} seed={seed} device={device}\n"
        f"[test] random mines per episode: "
        f"{n_red_mines} RED in x in [{red_x_range[0]:.1f},{red_x_range[1]:.1f}]  + "
        f"{n_blue_mines} BLUE in x in [{blue_x_range[0]:.1f},{blue_x_range[1]:.1f}]"
    )

    uninstall = _install_random_mine_hook(
        core,
        n_red_mines=n_red_mines,
        n_blue_mines=n_blue_mines,
        red_x_range=red_x_range,
        blue_x_range=blue_x_range,
        y_range=y_range,
        rng=rng,
    )

    results: dict[str, dict] = {}
    try:
        for label, path in [("Ours", league_path), ("Jacob et al.", paper_path)]:
            rng.manual_seed(int(seed) * 7919 + 1)  # same minefields for both models
            print(f"\n=== {team_size}v{team_size}  Evaluating {label}: {path} ===")
            episodes = _run_eval(
                path, env, n_episodes, device, opponent,
                deterministic=deterministic,
                progress_every=progress_every,
            )
            w, l, d = count_wld(episodes)
            results[label] = {
                "wins": w, "losses": l, "draws": d,
                "total": w + l + d, "episodes": episodes,
            }
            wr = (w / max(1, w + l + d)) * 100
            print(
                f"  {team_size}v{team_size} {label}: W={w} L={l} D={d}  "
                f"WR={wr:.1f}%  \u00b1{binomial_se(w, w+l+d):.1f}% (SE)"
            )
    finally:
        uninstall()
        env.close()

    return results


def _plot_single(ax, team_size: int, results: dict[str, dict], opponent: str, n_red: int, n_blue: int) -> None:
    labels = list(results.keys())
    wrs = [100.0 * results[l]["wins"] / max(1, results[l]["total"]) for l in labels]
    ses = [binomial_se(results[l]["wins"], results[l]["total"]) for l in labels]
    colors = ["#2ecc71", "#3498db"]
    x = np.arange(len(labels))
    bars = ax.bar(
        x, wrs,
        color=colors[: len(labels)],
        edgecolor="black", linewidth=1.2,
        yerr=ses, capsize=6,
        error_kw={"elinewidth": 1.6, "ecolor": "black"},
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=16)
    ax.tick_params(axis="y", labelsize=16)
    ax.set_ylabel(f"Win rate vs {opponent} (%)", fontsize=17)
    ax.set_title(
        f"{team_size}v{team_size}  (+{n_red}R/{n_blue}B random mines)",
        fontsize=16,
    )
    ax.set_ylim(0, 115)
    for bar, wr, se in zip(bars, wrs, ses):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + se + 2.5,
            f"{wr:.1f}% \u00b1 {se:.1f}",
            ha="center", fontsize=14,
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="10-random-mines robustness test (testing only, no training).")
    ap.add_argument(
        "--team-sizes", type=int, nargs="+", default=[2, 3, 4],
        help="Team sizes to sweep (default: 2 3 4).",
    )
    ap.add_argument("--league", type=str, default=None, help="Override 'Ours' ckpt path (applies to ALL sizes).")
    ap.add_argument("--paper", type=str, default=None, help="Override 'Jacob et al.' ckpt path (applies to ALL sizes).")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--opponent", type=str, default="OP3")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--n-red-mines", type=int, default=5, help="Random RED mines per episode (hazard to BLUE).")
    ap.add_argument("--n-blue-mines", type=int, default=5, help="Random BLUE mines per episode (hazard to RED).")
    ap.add_argument("--max-mines-per-team", type=int, default=5, help="Env slot capacity; must be >= max(n_red, n_blue).")
    ap.add_argument("--progress-every", type=int, default=10)
    ap.add_argument("--stochastic", action="store_true", help="Sample from policy (default deterministic argmax).")
    ap.add_argument(
        "--out-combined", type=str, default=None,
        help="Output path for the combined N-panel PNG (default: figures/mines_robustness_<OP>_<N>ep_all.png).",
    )
    args = ap.parse_args()

    opponent = args.opponent.upper()
    n_ep = int(args.episodes)
    deterministic = not args.stochastic
    progress_every = max(0, int(args.progress_every))
    team_sizes = sorted({int(n) for n in args.team_sizes})
    if not team_sizes:
        print("[ERR] --team-sizes must include at least one value.")
        sys.exit(2)

    figures_dir = os.path.join(PROJECT_ROOT, "figures")
    csv_dir = os.path.join(PROJECT_ROOT, "csv")
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)

    sizes_tag = "".join(f"{n}v{n}_" for n in team_sizes).rstrip("_")
    out_combined = args.out_combined or os.path.join(
        figures_dir, f"mines_robustness_{opponent}_{n_ep}ep_{sizes_tag}.png"
    )
    out_csv = os.path.join(csv_dir, f"mines_robustness_{opponent}_{n_ep}ep_{sizes_tag}.csv")

    # Resolve per-size checkpoints and validate up front.
    per_size_paths: dict[int, tuple[str, str]] = {}
    for n in team_sizes:
        ckpt_dir = os.path.join(PROJECT_ROOT, "checkpoints_sb3", f"{n}v{n}")
        def _resolve(explicit: str | None, default_name: str) -> str:
            if explicit is not None:
                p = explicit if explicit.endswith(".zip") else explicit + ".zip"
                return os.path.abspath(p)
            return os.path.abspath(os.path.join(ckpt_dir, default_name))

        league_path = _resolve(args.league, f"final_ppo_league_{n}v{n}.zip")
        paper_path = _resolve(args.paper, f"final_ppo_paper_{n}v{n}.zip")
        for label, p in [("Ours", league_path), ("Jacob et al.", paper_path)]:
            if not os.path.isfile(p):
                print(f"[ERR] {n}v{n} {label} checkpoint not found: {p}")
                sys.exit(1)
        per_size_paths[n] = (league_path, paper_path)
        print(f"[paths] {n}v{n}: Ours={os.path.basename(league_path)}  Jacob={os.path.basename(paper_path)}")

    # Evaluate each team size sequentially (builds a fresh env per size).
    all_results: dict[int, dict[str, dict]] = {}
    for n in team_sizes:
        league_path, paper_path = per_size_paths[n]
        all_results[n] = _run_for_team_size(
            n,
            league_path=league_path,
            paper_path=paper_path,
            opponent=opponent,
            n_episodes=n_ep,
            device=args.device,
            seed=int(args.seed),
            n_red_mines=int(args.n_red_mines),
            n_blue_mines=int(args.n_blue_mines),
            max_mines_per_team=int(args.max_mines_per_team),
            progress_every=progress_every,
            deterministic=deterministic,
        )

    # CSV dump: team_size, model, episode, scores, success, steps, return.
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("team_size,model,episode,blue_score,red_score,success,decision_steps,return\n")
        for n, results in all_results.items():
            for label, info in results.items():
                for ep_i, e in enumerate(info["episodes"]):
                    f.write(
                        f"{n}v{n},{label},{ep_i},{e['blue_score']},{e['red_score']},"
                        f"{e['success']},{e['decision_steps']},{e['return']:.4f}\n"
                    )
    print(f"\nSaved per-episode CSV: {out_csv}")

    # Plot.
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot.")
        return

    plt.rc("font", size=14)

    # Per-size PNGs.
    for n in team_sizes:
        fig, ax = plt.subplots(figsize=(6.2, 5.2))
        _plot_single(ax, n, all_results[n], opponent, int(args.n_red_mines), int(args.n_blue_mines))
        plt.tight_layout()
        out_png = os.path.join(
            figures_dir, f"mines_robustness_{opponent}_{n_ep}ep_{n}v{n}.png"
        )
        plt.savefig(out_png, dpi=150)
        plt.close(fig)
        print(f"Saved: {out_png}")

    # Combined N-panel figure.
    n_panels = len(team_sizes)
    fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 5.2), squeeze=False)
    for ax, n in zip(axes[0], team_sizes):
        _plot_single(ax, n, all_results[n], opponent, int(args.n_red_mines), int(args.n_blue_mines))
    plt.suptitle(
        f"Mine-robustness test: +{args.n_red_mines}R / +{args.n_blue_mines}B random mines per episode  "
        f"(n={n_ep}, error bars = \u00b11 binomial SE)",
        fontsize=17,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    plt.savefig(out_combined, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_combined}")


if __name__ == "__main__":
    main()
