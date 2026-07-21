#!/usr/bin/env python3
"""Per-cell counterfactual action-JSD probe on a forced-z repertoire checkpoint.

Collects states under one forced z, then evaluates all z on those SAME states
and reports pairwise action-distribution JSD. This is the right test for
context-conditioned behavioral separation (not stochastic trajectory distance).
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _make_env(checkpoint: Path, map_name: str, seed: int, device: str, max_steps: int):
    from game_field_gpu import GPUCTFVecEnv, GPUFieldConfig
    from rl.custom_ppo.inference import read_custom_ppo_metadata

    meta = read_custom_ppo_metadata(str(checkpoint))
    agents = int(meta.get("n_blue", 2))
    cfg = GPUFieldConfig(
        n_envs=1,
        max_blue_agents=agents,
        max_red_agents=agents,
        map_layout=map_name,
        max_decision_steps=int(max_steps),
        aquaticus_profile=True,
        rules_profile="OURS",
        device=device,
        seed=seed,
    )
    return GPUCTFVecEnv(cfg)


def _collect_obs(
    policy,
    env,
    *,
    opponent: str,
    fixed_z: int,
    n_steps: int,
    seed: int,
    deterministic: bool,
) -> list[dict[str, np.ndarray]]:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if hasattr(policy, "reset_strategy"):
        policy.reset_strategy()
    policy.fixed_latent_strategy = True
    policy.fixed_latent_strategy_id = int(fixed_z)

    try:
        env.env_method("set_phase", opponent)
        env.env_method("set_next_opponent", "SCRIPTED", opponent)
    except Exception:
        pass

    obs = env.reset()
    collected: list[dict[str, np.ndarray]] = []
    for _ in range(n_steps):
        # Store a shallow copy of array fields
        snap = {k: np.array(v, copy=True) for k, v in obs.items() if isinstance(v, np.ndarray)}
        collected.append(snap)
        actions, _ = policy.predict(obs, deterministic=deterministic)
        step_out = env.step(actions)
        if len(step_out) == 5:
            obs, _rew, dones, _trunc, _info = step_out
        else:
            obs, _rew, dones, _info = step_out
        done = bool(np.any(dones))
        if done:
            obs = env.reset()
            if hasattr(policy, "reset_strategy"):
                policy.reset_strategy()
            policy.fixed_latent_strategy = True
            policy.fixed_latent_strategy_id = int(fixed_z)
    return collected


def _obs_batch(snaps: list[dict[str, np.ndarray]], device: torch.device) -> dict[str, torch.Tensor]:
    keys = ["grid", "vec", "agent_mask", "mask"]
    out: dict[str, torch.Tensor] = {}
    for k in keys:
        stacked = np.concatenate([s[k] for s in snaps if k in s], axis=0)
        out[k] = torch.as_tensor(stacked, device=device)
    return out


def _pairwise_jsd_report(model, obs_t: dict[str, torch.Tensor], latent_k: int) -> dict[str, Any]:
    from rl.custom_ppo.diagnostics.counterfactual import jsd_from_logits

    n = int(obs_t["grid"].shape[0])
    logits_by_z: list[torch.Tensor] = []
    with torch.no_grad():
        for z in range(latent_k):
            z_idx = torch.full((n,), z, dtype=torch.long, device=obs_t["grid"].device)
            logits = model.policy_logits(obs_t, z_idx=z_idx)
            logits = model._mask_logits(logits, obs_t.get("mask"))
            logits_by_z.append(logits.float())

    pair_rows = []
    for i, j in combinations(range(latent_k), 2):
        dists_i = list(model._categoricals(logits_by_z[i]))
        dists_j = list(model._categoricals(logits_by_z[j]))
        # Sum head JSDs (same aggregation as training KL sensitivity), then mean over rows.
        jsd_sum = torch.zeros((n,), device=logits_by_z[i].device)
        for di, dj in zip(dists_i, dists_j):
            jsd_sum = jsd_sum + jsd_from_logits(di.logits, dj.logits)
        jsd_mean = float(jsd_sum.mean().item())
        disagree = float(
            (dists_i[0].probs.argmax(-1) != dists_j[0].probs.argmax(-1)).float().mean().item()
        )
        pair_rows.append(
            {
                "z_i": i,
                "z_j": j,
                "action_jsd_mean": jsd_mean,
                "head0_argmax_disagree": disagree,
            }
        )
    jsds = [r["action_jsd_mean"] for r in pair_rows]
    return {
        "n_states": n,
        "pair_jsd_mean": float(np.mean(jsds)) if jsds else float("nan"),
        "pair_jsd_max": float(np.max(jsds)) if jsds else float("nan"),
        "pair_jsd_min": float(np.min(jsds)) if jsds else float("nan"),
        "pairs_above_0_05": int(sum(1 for v in jsds if v > 0.05)),
        "pairs": pair_rows,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--opponents", nargs="+", default=["OP8", "OP11", "OP12"])
    p.add_argument("--maps", nargs="+", default=["map_b", "map_b_split_lane_v2"])
    p.add_argument("--steps-per-cell", type=int, default=240)
    p.add_argument("--collect-z", type=int, default=0, help="Forced z used while collecting states")
    p.add_argument("--max-decision-steps", type=int, default=240)
    p.add_argument("--base-seed", type=int, default=42)
    args = p.parse_args()

    from rl.custom_ppo import load_custom_ppo_policy

    ckpt = Path(args.checkpoint)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build a dummy env to get spaces
    env0 = _make_env(ckpt, args.maps[0], args.base_seed, args.device, args.max_decision_steps)
    policy = load_custom_ppo_policy(
        str(ckpt),
        env0.observation_space,
        env0.action_space,
        device=args.device,
    )
    model = policy.model
    model.eval()
    latent_k = int(getattr(model, "latent_k", 4))

    # Inherit reward surface if run_config present (env already created without; CF JSD is on states)
    report: dict[str, Any] = {
        "checkpoint": str(ckpt),
        "collect_z": int(args.collect_z),
        "steps_per_cell": int(args.steps_per_cell),
        "hypothesis": "per-cell counterfactual action JSD on shared states",
        "cells": {},
    }

    opp_idx = {o: i for i, o in enumerate(args.opponents)}
    map_idx = {m: i for i, m in enumerate(args.maps)}

    for opponent in args.opponents:
        for map_name in args.maps:
            cell_seed = int(args.base_seed) + 1000 * opp_idx[opponent] + 100 * map_idx[map_name]
            print(f"=== {opponent} | {map_name} seed={cell_seed} ===")
            env = _make_env(ckpt, map_name, cell_seed, args.device, args.max_decision_steps)
            snaps = _collect_obs(
                policy,
                env,
                opponent=opponent,
                fixed_z=int(args.collect_z),
                n_steps=int(args.steps_per_cell),
                seed=cell_seed,
                deterministic=True,
            )
            obs_t = _obs_batch(snaps, torch.device(args.device))
            cell_report = _pairwise_jsd_report(model, obs_t, latent_k)
            key = f"{opponent}|{map_name}"
            report["cells"][key] = cell_report
            print(
                f"  n={cell_report['n_states']} jsd_mean={cell_report['pair_jsd_mean']:.4f} "
                f"jsd_max={cell_report['pair_jsd_max']:.4f} pairs>0.05={cell_report['pairs_above_0_05']}"
            )
            for pr in cell_report["pairs"]:
                print(
                    f"    z{pr['z_i']}-z{pr['z_j']}: JSD={pr['action_jsd_mean']:.4f} "
                    f"head0_disagree={pr['head0_argmax_disagree']:.3f}"
                )
            try:
                env.close()
            except Exception:
                pass

    # Aggregate
    all_means = [c["pair_jsd_mean"] for c in report["cells"].values()]
    all_maxes = [c["pair_jsd_max"] for c in report["cells"].values()]
    report["summary"] = {
        "cells": len(report["cells"]),
        "mean_of_cell_jsd_means": float(np.mean(all_means)) if all_means else float("nan"),
        "max_of_cell_jsd_maxes": float(np.max(all_maxes)) if all_maxes else float("nan"),
        "cells_with_any_pair_above_0_05": int(
            sum(1 for c in report["cells"].values() if c["pairs_above_0_05"] > 0)
        ),
        "gate_any_pair_jsd_gt_0_05": bool(
            any(c["pairs_above_0_05"] > 0 for c in report["cells"].values())
        ),
    }
    print()
    print("SUMMARY", json.dumps(report["summary"], indent=2))
    out_path = out_dir / "action_jsd_probe_report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    try:
        env0.close()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
