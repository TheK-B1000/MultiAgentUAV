"""Vectorized fixed-z evaluation for SLSR Stage 1.

``plot/eval_checkpoint.py`` hard-codes ``n_envs=1``; a full SLSR fixed-z sweep
(K=4 z * 4 opponents * ~200 episodes) takes ~11 hours under that constraint.
This tool runs ``--n-envs`` GPUCTFVecEnv environments in parallel and
accumulates episodes per (z, opponent, map_set) cell, producing per-z aggregate
CSVs whose filenames match what ``tools/derive_best_z_labels.py`` expects:

    eval_<tag>_fix_z<z>_op5_bite_v3_<setting>_aggregate.csv

so the downstream label-derivation pipeline picks them up unchanged.

Example::

    python tools/eval_fixed_z_parallel.py \\
        --checkpoint checkpoints/4v4/final_latent_slsr_stage1_forced_z_seed1_4v4.zip \\
        --tag slsr_stage1 \\
        --latent-k 4 \\
        --episodes 200 \\
        --opponents OP3 OP5_RUSHER OP6 OP7 \\
        --map-sets eval \\
        --n-envs 16 \\
        --device cuda

Only fixed-z deployment is supported (q_phi is bypassed by clamping
``model.fixed_latent_strategy_id`` per z). Train/eval map split is selected via
``--map-sets``; the derive tool consumes the ``eval`` rows.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from game_field_gpu import GPUCTFVecEnv, GPUFieldConfig  # noqa: E402
from plot.eval_rollout import binomial_se, compute_aggregates, count_wld  # noqa: E402
from rl.custom_ppo import load_custom_ppo_policy, read_custom_ppo_metadata  # noqa: E402


def _set_opponent_and_stress(env: GPUCTFVecEnv, opponent: str) -> None:
    env.env_method("set_phase", opponent)
    env.env_method("set_next_opponent", "SCRIPTED", opponent)
    try:
        from rl.stress_schedule import STRESS_BY_PHASE

        env.env_method("set_stress_schedule", STRESS_BY_PHASE)
    except Exception:
        pass


def _build_env(*, n_envs: int, agents: int, map_set: str, device: str, seed: int) -> GPUCTFVecEnv:
    cfg = GPUFieldConfig(
        n_envs=n_envs,
        max_blue_agents=agents,
        max_red_agents=agents,
        map_set=map_set,
        max_decision_steps=400,
        aquaticus_profile=True,
        rules_profile="OURS",
        device=device,
        seed=int(seed),
    )
    return GPUCTFVecEnv(cfg)


def _run_setting(
    *,
    model: Any,
    env: GPUCTFVecEnv,
    n_episodes: int,
    progress_label: str = "",
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """Run vectorized rollout until ``n_episodes`` complete; deterministic actions."""
    episodes: List[Dict[str, Any]] = []
    n_envs = int(env.num_envs)
    if verbose:
        print(f"  [{progress_label}] env.reset() ...", flush=True)
    obs = env.reset()
    if verbose:
        sample_shapes = {k: getattr(v, "shape", None) for k, v in obs.items() if hasattr(v, "shape")}
        print(f"  [{progress_label}] reset done, obs shapes={sample_shapes}", flush=True)
    if hasattr(model, "reset_strategy"):
        model.reset_strategy()

    ep_return = np.zeros(n_envs, dtype=np.float64)
    ep_steps = np.zeros(n_envs, dtype=np.int64)

    last_print = time.time()
    step_count = 0
    while len(episodes) < n_episodes:
        if verbose and step_count == 0:
            print(f"  [{progress_label}] first predict() ...", flush=True)
        act, _ = model.predict(obs, deterministic=True)
        if verbose and step_count == 0:
            print(
                f"  [{progress_label}] first predict ok, act shape={getattr(act, 'shape', None)}",
                flush=True,
            )
        if act.ndim == 1:
            act = act[None, :]
        if verbose and step_count == 0:
            print(f"  [{progress_label}] first env.step_async ...", flush=True)
        env.step_async(act)
        obs, rew, done, infos = env.step_wait()
        if verbose and step_count == 0:
            print(
                f"  [{progress_label}] first env.step done, rew shape={getattr(rew, 'shape', None)}, "
                f"done shape={getattr(done, 'shape', None)}",
                flush=True,
            )
        step_count += 1
        ep_steps += 1
        # rew can be shape [n_envs] or [n_envs, n_agents]; sum along last axis if needed.
        rew_arr = np.asarray(rew)
        if rew_arr.ndim == 2:
            rew_arr = rew_arr.sum(axis=-1)
        ep_return += rew_arr.astype(np.float64)

        done_arr = np.asarray(done)
        if done_arr.any():
            for i in range(n_envs):
                if not bool(done_arr[i]):
                    continue
                info = infos[i] if i < len(infos) else {}
                ep_res = info.get("episode_result", info) if isinstance(info, dict) else {}
                if not isinstance(ep_res, dict):
                    ep_res = {}
                bs = int(ep_res.get("blue_score", 0))
                rs = int(ep_res.get("red_score", 0))
                steps_i = int(ep_res.get("decision_steps", ep_steps[i]))
                row: Dict[str, Any] = {
                    "success": 1 if bs > rs else 0,
                    "blue_score": bs,
                    "red_score": rs,
                    "steps": steps_i,
                    "return": float(ep_return[i]),
                    "zone_coverage": float(ep_res.get("zone_coverage", 0.0) or 0.0),
                    "collision_free": int(ep_res.get("collision_free_episode", 1) or 0),
                    "win_margin": bs - rs,
                }
                episodes.append(row)
                ep_return[i] = 0.0
                ep_steps[i] = 0
                if len(episodes) >= n_episodes:
                    break

        now = time.time()
        if now - last_print >= 5.0:
            print(
                f"  [{progress_label}] {len(episodes)}/{n_episodes} done, steps={step_count}",
                flush=True,
            )
            last_print = now

    return episodes[:n_episodes]


def _write_aggregate_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main() -> int:
    parser = argparse.ArgumentParser(description="Vectorized fixed-z eval for SLSR.")
    parser.add_argument("--checkpoint", required=True, help="Path to custom PPO checkpoint .zip")
    parser.add_argument("--tag", required=True, help="Run tag embedded in label, e.g. 'slsr_stage1'.")
    parser.add_argument("--latent-k", type=int, default=4)
    parser.add_argument("--episodes", type=int, default=200, help="Episodes per (z, opp, map_set) cell.")
    parser.add_argument(
        "--opponents",
        nargs="+",
        default=["OP3", "OP5_RUSHER", "OP6", "OP7"],
        help="Opponents to evaluate against (canonical names).",
    )
    parser.add_argument(
        "--map-sets",
        nargs="+",
        default=["eval"],
        choices=["train", "eval"],
        help="Map splits to evaluate; derive tool consumes eval.",
    )
    parser.add_argument("--n-envs", type=int, default=16)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--agents", type=int, default=None, help="Agents per team; defaults to checkpoint metadata.")
    parser.add_argument("--out-dir", default=None, help="CSV output dir (default AICTFProject/csv).")
    args = parser.parse_args()

    checkpoint = os.path.abspath(args.checkpoint if args.checkpoint.endswith(".zip") else args.checkpoint + ".zip")
    if not os.path.isfile(checkpoint):
        print(f"[ERROR] checkpoint not found: {checkpoint}", file=sys.stderr)
        return 2

    meta = read_custom_ppo_metadata(checkpoint)
    agents = int(args.agents or meta.get("n_blue", 2))
    mode = f"{agents}v{agents}"
    out_dir = Path(args.out_dir) if args.out_dir else (PROJECT_ROOT / "csv")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[parallel-eval] checkpoint={checkpoint}")
    print(f"[parallel-eval] agents={agents} mode={mode} n_envs={args.n_envs} device={args.device}")
    print(
        f"[parallel-eval] z=0..{args.latent_k - 1} opponents={args.opponents} map_sets={args.map_sets} "
        f"episodes_per_cell={args.episodes}"
    )

    t_total = time.time()
    # Load policy once on a probe env so observation/action spaces are bound; we'll rebind to per-cell
    # envs in the loop (spaces match because n_envs differs but obs/act spec is identical).
    print("[parallel-eval] probing env to load policy ...", flush=True)
    probe_env = _build_env(
        n_envs=1, agents=agents, map_set=args.map_sets[0], device=args.device, seed=int(args.seed)
    )
    try:
        model = load_custom_ppo_policy(
            checkpoint, probe_env.observation_space, probe_env.action_space, device=args.device
        )
    finally:
        probe_env.close()
    print(f"[parallel-eval] policy loaded ({type(model).__name__}); starting sweep", flush=True)

    for z in range(int(args.latent_k)):
        label = f"{args.tag}_fix_z{z}"
        per_z_rows: List[Dict[str, Any]] = []
        if hasattr(model, "model") and bool(getattr(model.model, "uses_latent_strategy", False)):
            model.fixed_latent_strategy = True
            model.fixed_latent_strategy_id = int(z)
            print(f"[parallel-eval] z={z}: model.fixed_latent_strategy_id set", flush=True)
        for map_set in args.map_sets:
            for opp in args.opponents:
                opp_canon = str(opp).strip().upper()
                t_cell = time.time()
                env_seed = int(args.seed) + z * 10_000 + (abs(hash(opp_canon)) % 1000)
                print(
                    f"[parallel-eval] building env n_envs={args.n_envs} map={map_set} opp={opp_canon} "
                    f"seed={env_seed} ...",
                    flush=True,
                )
                env = _build_env(
                    n_envs=int(args.n_envs),
                    agents=agents,
                    map_set=map_set,
                    device=args.device,
                    seed=env_seed,
                )
                try:
                    print(f"[parallel-eval]   set opponent={opp_canon} ...", flush=True)
                    _set_opponent_and_stress(env, opp_canon)
                    # Rebind action sampling generator (matches new env.num_envs) if applicable.
                    if hasattr(model, "_prev_z"):
                        model._prev_z = None
                    print(
                        f"[parallel-eval] z={z} {map_set} vs {opp_canon}: target={args.episodes} eps",
                        flush=True,
                    )
                    eps = _run_setting(
                        model=model,
                        env=env,
                        n_episodes=int(args.episodes),
                        progress_label=f"z={z} {map_set}/{opp_canon}",
                        verbose=True,
                    )
                finally:
                    env.close()

                w, l, d = count_wld(eps)
                wr = 100.0 * w / max(1, len(eps))
                agg = compute_aggregates(eps)
                row: Dict[str, Any] = {
                    "label": label,
                    "setting": mode,
                    "map_set": map_set,
                    "opponent": opp_canon,
                    "checkpoint": checkpoint,
                    "episodes": len(eps),
                    "wins": w,
                    "losses": l,
                    "draws": d,
                    "success_rate": wr,
                    "success_rate_std": binomial_se(w, len(eps)),
                }
                # Override success_rate via compute_aggregates if available to match existing CSVs;
                # keep the binomial SE we computed (compute_aggregates uses std-over-trials).
                for k, v in agg.items():
                    if k in ("success_rate", "success_rate_std"):
                        continue
                    row[k] = v
                per_z_rows.append(row)
                dt = time.time() - t_cell
                print(
                    f"  [done] z={z} {map_set}/{opp_canon}: W={w} L={l} D={d} "
                    f"WR={wr:.1f}% n={len(eps)} in {dt:.1f}s",
                    flush=True,
                )

        agg_path = out_dir / f"eval_{label}_op5_bite_v3_{mode}_aggregate.csv"
        _write_aggregate_csv(agg_path, per_z_rows)
        print(f"  [csv] {agg_path}")

    print(f"[parallel-eval] DONE in {time.time() - t_total:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
