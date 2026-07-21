#!/usr/bin/env python3
"""Episode-persistent router cadence ablation for V6I9 repertoire checkpoint.

Tests whether the routing problem is cadence/switching or context-mapping:

  current_switching_learned   : learned q_phi at cadence 32 (trained behaviour)
  learned_initial_no_switch   : q_phi once at episode start, held for episode
  uniform_episode_persistent  : uniform random z once at episode start, held
  fixed_z2                    : clamp z=2 entire episode (best single-z baseline)
  shuffled_episode_assignment : learned initial-z histogram permuted across episodes

Grid: OP8 + OP10 x map_a + map_b x 10 seeds per cell = 200 episodes per condition.

Usage
-----
    uv run python experiments/run_cadence_ablation.py \\
        --checkpoint checkpoints/2v2/final_v6i9-mapaware-repertoire-hardpool-refactor-r1-seed1_2v2.zip \\
        --out-dir experiments/cadence_ablation_runs/<stamp> \\
        [--device cpu] [--seeds 10] [--base-seed 3000]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from game_field_gpu import GPUCTFVecEnv, GPUFieldConfig
from rl.custom_ppo.checkpoints.archive import read_checkpoint_payload
from rl.custom_ppo.checkpoints.loader import load_custom_ppo_checkpoint
from rl.custom_ppo.communication.observation import extend_observation_space_if_needed
from rl.evaluation.router_ablation import (
    EvalCondition,
    configure_condition,
    deterministic_cross_context_permutation,
)
from plot.eval_rollout import reset_eval_runtime_state


# ---------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------

CONDITIONS = [
    EvalCondition(
        name="current_switching_learned",
        selection_rule="qphi",
        strategy_interval=32,
        allow_switching=True,
        description="Learned q_phi at trained cadence-32.",
    ),
    EvalCondition(
        name="learned_initial_no_switch",
        selection_rule="qphi",
        strategy_interval=0,
        allow_switching=False,
        description="q_phi runs once at episode start; z held for entire episode.",
    ),
    EvalCondition(
        name="uniform_episode_persistent",
        selection_rule="uniform",
        strategy_interval=0,
        allow_switching=False,
        description="Uniform random z sampled once at episode start; held for episode.",
    ),
    EvalCondition(
        name="fixed_z2",
        selection_rule="fixed_z2",
        strategy_interval=0,
        allow_switching=False,
        fixed_latent_id=2,
        description="Clamp z=2 for entire episode.",
    ),
    # shuffled_episode_assignment is handled separately after learned_initial run
]


# ---------------------------------------------------------------------------
# Single-episode rollout
# ---------------------------------------------------------------------------

def _run_one_episode(
    policy: object,
    env: "GPUCTFVecEnv",
    *,
    opponent: str,
    seed: int,
    max_steps: int = 400,
) -> dict:
    """Roll out one episode, return metrics dict."""
    from plot.eval_rollout import reset_eval_runtime_state

    reset_eval_runtime_state(policy)
    core_reset_attr = "reseed"
    if hasattr(env.core, core_reset_attr):
        env.core.reseed(seed)

    obs = env.reset()

    total_reward = 0.0
    steps = 0
    done = False
    success = 0
    win_margin = 0.0

    while not done and steps < max_steps:
        # Build action via policy.predict (manages internal z state).
        # Inject global state so the router sees real context (not zeros).
        obs_np = {k: np.asarray(v) for k, v in obs.items()}
        if hasattr(env, "state"):
            obs_np["global_state"] = np.asarray(env.state(), dtype=np.float32)
        actions, _ = policy.predict(obs_np, deterministic=True)
        obs, rews, dones, infos = env.step(actions)
        total_reward += float(np.asarray(rews).mean())
        steps += 1
        if np.any(dones):
            done = True
            info = infos[0] if isinstance(infos, (list, tuple)) else infos
            success = int(info.get("blue_win", info.get("success", 0)))
            win_margin = float(info.get("win_margin", 0.0))

    # Extract opportunity trace if present
    trace = list(getattr(policy, "opportunity_trace_log", []))

    return {
        "return": total_reward,
        "steps": steps,
        "success": success,
        "win_margin": win_margin,
        "_trace": trace,
    }


def _run_condition_grid(
    ckpt_path: str,
    condition: "EvalCondition",
    *,
    opponents: list[str],
    maps: list[tuple[str, str]],   # (map_layout, map_set)
    seeds: list[int],
    n_agents: int,
    obs_space: object,
    act_space: object,
    device: str,
    fixed_z_override: int | None = None,
    shuffled_assignments: dict | None = None,
) -> list[dict]:
    """Run condition across full grid; return episode rows."""
    policy = load_custom_ppo_checkpoint(ckpt_path, obs_space, act_space, device=device).policy
    configure_condition(policy, condition)

    if fixed_z_override is not None:
        policy.fixed_latent_strategy = True
        policy.fixed_latent_strategy_id = int(fixed_z_override)

    rows: list[dict] = []
    for map_layout, map_set in maps:
        for opponent in opponents:
            for seed in seeds:
                # Per-episode override for shuffled condition
                if shuffled_assignments is not None:
                    ep_key = (opponent.upper(), map_layout, seed)
                    assigned_z = shuffled_assignments.get(ep_key)
                    if assigned_z is None:
                        raise RuntimeError(f"Missing shuffled assignment for {ep_key}")
                    policy.fixed_latent_strategy = True
                    policy.fixed_latent_strategy_id = int(assigned_z)

                env = GPUCTFVecEnv(GPUFieldConfig(
                    n_envs=1,
                    n_agents_per_team=n_agents,
                    max_blue_agents=n_agents,
                    max_red_agents=n_agents,
                    map_layout=map_layout,
                    map_set=map_set,
                    device=device,
                    seed=seed,
                    max_decision_steps=400,
                    aquaticus_profile=True,
                    rules_profile="OURS",
                ))
                try:
                    env.env_method("set_phase", opponent)
                    env.env_method("set_next_opponent", "SCRIPTED", opponent)
                    result = _run_one_episode(policy, env, opponent=opponent, seed=seed)
                finally:
                    env.close()

                trace = result.pop("_trace")
                # Record initial z from trace
                initial_z = int(trace[0]["selected_z"]) if trace else -1
                switch_count = sum(1 for t in trace if t.get("switch_occurred", False))

                rows.append({
                    "condition": condition.name,
                    "opponent": opponent.upper(),
                    "map_layout": map_layout,
                    "map_set": map_set,
                    "seed": seed,
                    "return": result["return"],
                    "steps": result["steps"],
                    "success": result["success"],
                    "win_margin": result["win_margin"],
                    "initial_z": initial_z,
                    "strategy_switches": switch_count,
                    "n_opportunities": len(trace),
                })
    del policy
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


# ---------------------------------------------------------------------------
# Shuffled assignment builder
# ---------------------------------------------------------------------------

def _build_shuffled_assignments(
    learned_initial_rows: list[dict],
) -> dict[tuple[str, str, int], int]:
    """Permute the initial-z histogram across matched episode keys.

    For each (map_layout, opponent) cell, permute the learned initial-z
    sequence across seeds so that each seed receives a different seed's z.
    Derangement: no seed keeps its own learned z when alternatives exist.

    Returns
    -------
    dict mapping (opponent, map_layout, seed) → shuffled_z
    """
    # Group by cell
    cells: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for row in learned_initial_rows:
        key = (row["opponent"].upper(), row["map_layout"])
        cells.setdefault(key, []).append((int(row["seed"]), int(row["initial_z"])))

    assignments: dict[tuple[str, str, int], int] = {}
    for (opp, map_layout), items in cells.items():
        items_sorted = sorted(items, key=lambda x: x[0])
        seeds_list = [s for s, _ in items_sorted]
        z_list = [z for _, z in items_sorted]
        n = len(seeds_list)

        # Rotate by 1 (deterministic derangement when n > 1)
        perm = deterministic_cross_context_permutation(n, seed=abs(hash((opp, map_layout))) % (2**31))
        shuffled_z = [z_list[perm[i]] for i in range(n)]

        # Sanity check: histogram preserved
        assert Counter(shuffled_z) == Counter(z_list), (
            f"Histogram mismatch for ({opp}, {map_layout})"
        )
        if n > 1 and len(set(z_list)) > 1:
            # At least one must differ when multiple distinct z values exist
            assert any(sz != oz for sz, oz in zip(shuffled_z, z_list)), (
                f"Derangement produced identity for ({opp}, {map_layout})"
            )

        for seed, sz in zip(seeds_list, shuffled_z):
            assignments[(opp, map_layout, seed)] = sz

    return assignments


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _summarize(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str, str], list[dict]] = {}
    for r in rows:
        key = (r["condition"], r["opponent"], r["map_layout"])
        groups.setdefault(key, []).append(r)

    out = []
    for (cond, opp, mlay), g in sorted(groups.items()):
        returns = [r["return"] for r in g]
        wins = [r["success"] for r in g]
        init_z_counts = Counter(r["initial_z"] for r in g)
        out.append({
            "condition": cond,
            "opponent": opp,
            "map_layout": mlay,
            "n_episodes": len(g),
            "return_mean": float(np.mean(returns)),
            "return_std": float(np.std(returns)),
            "success_rate": float(np.mean(wins)),
            "initial_z_dist": dict(sorted(init_z_counts.items())),
        })
    return out


def _print_table(summary: list[dict]) -> None:
    print(f"\n{'Condition':<35} {'Opponent':<6} {'Map':<12} {'Return':>9} {'Success':>9} {'z-dist'}")
    print("-" * 90)
    for r in summary:
        zd = "  ".join(f"z{k}:{v}" for k, v in sorted(r["initial_z_dist"].items()) if k >= 0)
        print(
            f"{r['condition']:<35} {r['opponent']:<6} {r['map_layout']:<12} "
            f"{r['return_mean']:9.3f} {r['success_rate']:9.3f}  {zd}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cadence ablation for V6I9 repertoire checkpoint")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seeds", type=int, default=10, help="Seeds per cell")
    p.add_argument("--base-seed", type=int, default=3000)
    p.add_argument("--opponents", nargs="+", default=["OP8", "OP10"])
    p.add_argument(
        "--maps",
        nargs="+",
        default=["map_a_open", "map_b_split_lane"],
        help="Map layouts to test",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    ckpt_path = str(Path(args.checkpoint).resolve())
    seeds = list(range(args.base_seed, args.base_seed + args.seeds))

    # Output dir
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out_dir or f"experiments/cadence_ablation_runs/{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Checkpoint config
    payload = read_checkpoint_payload(ckpt_path, map_location="cpu")
    ckpt_cfg = payload.get("cfg", {})
    n_agents = int(ckpt_cfg.get("n_agents_per_team") or 2)
    trained_cadence = int(ckpt_cfg.get("strategy_interval") or ckpt_cfg.get("latent_resample_every_n") or 32)
    map_layout_default = str(ckpt_cfg.get("map_layout") or "map_b_split_lane")

    # Update condition 0's interval from checkpoint
    CONDITIONS[0] = EvalCondition(
        name="current_switching_learned",
        selection_rule="qphi",
        strategy_interval=trained_cadence,
        allow_switching=True,
        description=f"Learned q_phi at trained cadence-{trained_cadence}.",
    )

    # Obs/act space from a probe env
    probe_env = GPUCTFVecEnv(GPUFieldConfig(
        n_envs=1, n_agents_per_team=n_agents,
        max_blue_agents=n_agents, max_red_agents=n_agents,
        device=args.device, seed=seeds[0], map_layout=map_layout_default,
    ))
    obs_space = extend_observation_space_if_needed(probe_env.observation_space, ckpt_cfg)
    act_space = probe_env.action_space
    probe_env.close()

    maps: list[tuple[str, str]] = []
    for ml in args.maps:
        # map_set defaults to "eval"
        maps.append((ml, "eval"))

    opponents = [o.upper() for o in args.opponents]

    print(f"Cadence ablation  ckpt={Path(ckpt_path).name}")
    print(f"  opponents={opponents}  maps={[m for m, _ in maps]}  seeds={seeds[:3]}..+{len(seeds)}")
    print(f"  trained_cadence={trained_cadence}  n_agents={n_agents}")
    print(f"  output={out_dir}")

    all_rows: list[dict] = []

    # ── 1. Run the 4 standard conditions ─────────────────────────────────────
    for cond in CONDITIONS:
        print(f"\nRunning condition: {cond.name}")
        cond_rows = _run_condition_grid(
            ckpt_path, cond,
            opponents=opponents, maps=maps, seeds=seeds,
            n_agents=n_agents, obs_space=obs_space, act_space=act_space,
            device=args.device,
        )
        all_rows.extend(cond_rows)
        n_ep = len(cond_rows)
        ret_mean = np.mean([r["return"] for r in cond_rows])
        suc = np.mean([r["success"] for r in cond_rows])
        print(f"  {n_ep} episodes  return={ret_mean:.3f}  success={suc:.3f}")

    # ── 2. Build shuffled_episode_assignment from learned_initial_no_switch ──
    learned_initial_rows = [r for r in all_rows if r["condition"] == "learned_initial_no_switch"]
    print(f"\nBuilding shuffled_episode_assignment from {len(learned_initial_rows)} learned-initial episodes...")
    shuffled_assignments = _build_shuffled_assignments(learned_initial_rows)

    # Verify histogram per cell
    for (opp, mlay), items in {
        (r["opponent"], r["map_layout"]): None for r in learned_initial_rows
    }.items():
        original_z = [r["initial_z"] for r in learned_initial_rows
                      if r["opponent"] == opp and r["map_layout"] == mlay]
        shuf_z = [shuffled_assignments[(opp, mlay, r["seed"])]
                  for r in learned_initial_rows
                  if r["opponent"] == opp and r["map_layout"] == mlay]
        assert Counter(original_z) == Counter(shuf_z), f"Histogram mismatch for ({opp},{mlay})"
    print("  Histogram preserved: OK")

    shuffled_cond = EvalCondition(
        name="shuffled_episode_assignment",
        selection_rule="fixed_z0",   # overridden per-episode; rule just silences configure_condition
        strategy_interval=0,
        allow_switching=False,
        fixed_latent_id=0,
        description=(
            "Learned initial-z histogram permuted across matched episode seeds; "
            "z held for entire episode. Tests whether initial-z context correlates "
            "with useful assignment."
        ),
    )

    print(f"\nRunning condition: {shuffled_cond.name}")
    shuffled_rows = _run_condition_grid(
        ckpt_path, shuffled_cond,
        opponents=opponents, maps=maps, seeds=seeds,
        n_agents=n_agents, obs_space=obs_space, act_space=act_space,
        device=args.device,
        shuffled_assignments=shuffled_assignments,
    )
    # Patch condition name on rows
    for r in shuffled_rows:
        r["condition"] = "shuffled_episode_assignment"
    all_rows.extend(shuffled_rows)
    ret_mean = np.mean([r["return"] for r in shuffled_rows])
    suc = np.mean([r["success"] for r in shuffled_rows])
    print(f"  {len(shuffled_rows)} episodes  return={ret_mean:.3f}  success={suc:.3f}")

    # ── 3. Write outputs ───────────────────────────────────────────────────────
    summary = _summarize(all_rows)
    _print_table(summary)

    # Paired comparison table: condition vs uniform_episode_persistent
    print("\n=== Paired return deltas vs uniform_episode_persistent ===")
    by_key = {}
    for r in all_rows:
        k = (r["condition"], r["opponent"], r["map_layout"], r["seed"])
        by_key[k] = r

    ref_cond = "uniform_episode_persistent"
    test_conds = [
        "current_switching_learned",
        "learned_initial_no_switch",
        "fixed_z2",
        "shuffled_episode_assignment",
    ]
    for tc in test_conds:
        deltas: list[float] = []
        for r in all_rows:
            if r["condition"] != tc:
                continue
            ref = by_key.get((ref_cond, r["opponent"], r["map_layout"], r["seed"]))
            if ref is None:
                continue
            deltas.append(r["return"] - ref["return"])
        if deltas:
            arr = np.array(deltas)
            print(f"  {tc:<35}  d_return vs uniform_ep: {arr.mean():.4f}  std={arr.std():.4f}  "
                  f"n={len(arr)}")

    # Also compare learned_initial vs shuffled (the key diagnostic comparison)
    print("\n=== learned_initial vs shuffled_episode (key diagnostic) ===")
    deltas_init_vs_shuf: list[float] = []
    for r in all_rows:
        if r["condition"] != "learned_initial_no_switch":
            continue
        shuf = by_key.get(("shuffled_episode_assignment", r["opponent"], r["map_layout"], r["seed"]))
        if shuf is None:
            continue
        deltas_init_vs_shuf.append(r["return"] - shuf["return"])
    if deltas_init_vs_shuf:
        arr = np.array(deltas_init_vs_shuf)
        print(f"  learned_initial - shuffled:  mean={arr.mean():.4f}  std={arr.std():.4f}  n={len(arr)}")
        positive_frac = float(np.mean(arr > 0))
        print(f"  learned > shuffled on {positive_frac:.1%} of episodes")

    # CSV output
    fields: list[str] = []
    for r in all_rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    ep_csv = out_dir / "cadence_ablation_episodes.csv"
    with ep_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)

    summary_csv = out_dir / "cadence_ablation_summary.csv"
    sfields = list(summary[0].keys()) if summary else []
    with summary_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=sfields)
        w.writeheader()
        for row in summary:
            row_copy = dict(row)
            row_copy["initial_z_dist"] = json.dumps(row_copy["initial_z_dist"])
            w.writerow(row_copy)

    manifest = {
        "checkpoint": ckpt_path,
        "conditions": [c.name for c in CONDITIONS] + ["shuffled_episode_assignment"],
        "opponents": opponents,
        "maps": [m for m, _ in maps],
        "seeds": seeds,
        "n_episodes": len(all_rows),
        "trained_cadence": trained_cadence,
        "shuffled_assignment_summary": {
            f"{opp}|{ml}": {
                "original": [
                    r["initial_z"] for r in learned_initial_rows
                    if r["opponent"] == opp and r["map_layout"] == ml
                ],
                "shuffled": [
                    shuffled_assignments[(opp, ml, r["seed"])]
                    for r in learned_initial_rows
                    if r["opponent"] == opp and r["map_layout"] == ml
                ],
            }
            for opp in opponents
            for _, ml in [(None, r["map_layout"]) for r in learned_initial_rows
                          if r["opponent"] == opp][:1]  # unique maps per opp
        },
    }
    (out_dir / "cadence_ablation_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print(f"\nOutputs written to {out_dir}/")
    print(f"  {ep_csv.name}")
    print(f"  {summary_csv.name}")
    print(f"  cadence_ablation_manifest.json")


if __name__ == "__main__":
    main()
