#!/usr/bin/env python3
"""RUSH_PROBE_ROOT_CAUSE_AUDIT -- Phase 1 (baseline).

Diagnostic-only, read-only: does not modify BLUE_PROBES_V2 or any red
opponent/profile. Runs the FROZEN, official BLUE_RUSH and BLUE_SPLIT
controllers (unchanged) against OP6, OP7, OP8, OP10, OP12 and records the
metrics needed to check whether _blue_rush_targets expresses a competent,
distinct rush strategy before any further opponent redesign is attempted.

Metrics recorded per episode:
  - time to first enemy-territory entry (either blue agent crosses midline)
  - time to first flag pickup
  - path length traveled by the eventual carrier, start -> first pickup
  - number of target changes for either agent before pickup
  - steps (pre-pickup) where both agents' targets coincide (within 1.0 cell)
  - carrier return time (pickup -> first score, or None)
  - post-pickup mean distance from non-carrier's target to the carrier
    (small = escort-like support, large = independent objective)
  - min inter-agent distance pre-pickup (crowding/collision proxy)
  - whether blue's own flag was lost (red picked it up) before blue's
    first score
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gpu_env import GPUCTFVecEnv, GPUFieldConfig  # noqa: E402

OPPONENTS = (
    "OP6_IMMEDIATE_DUAL_RUSH",
    "OP7_DEEP_FORTRESS",
    "OP8_PROTECTED_CARRIER_ESCORT",
    "OP10_AGGRESSIVE_INTERCEPTOR",
    "OP12_LATE_CONVERTER",
)
STYLES = ("BLUE_RUSH", "BLUE_SPLIT")
TARGET_CHANGE_EPS = 0.5
SAME_TARGET_EPS = 1.0


def run_episode(style: str, red: str, seed: int, *, steps: int, device: str, map_layout: str) -> dict:
    cfg = GPUFieldConfig(
        n_envs=1, max_blue_agents=2, max_red_agents=2, map_layout=map_layout,
        max_decision_steps=steps + 10, aquaticus_profile=True, rules_profile="OURS",
        device=device, seed=seed,
    )
    env = GPUCTFVecEnv(cfg)
    core = env.core
    try:
        env.env_method("set_phase", red)
        env.env_method("set_next_opponent", "SCRIPTED", red)
        core.blue_scripted = True
        core.set_blue_style(style)
        env.reset()

        midline = float(core.cols) * 0.5
        first_entry_step = None
        first_pickup_step = None
        carrier_idx = None
        target_changes = 0
        same_target_steps = 0
        first_score_step = None
        return_time = None
        own_flag_lost_before_score = False
        own_flag_lost_step = None
        min_inter_agent_dist_pre_pickup = 1e9
        post_pickup_noncarrier_dist_sum = 0.0
        post_pickup_steps = 0
        prev_target_x = None
        prev_target_y = None
        prev_bx = None
        prev_by = None
        cum_dist = [0.0, 0.0]
        path_len_to_pickup = None
        n_steps = 0

        for t in range(steps):
            act = env.action_space.sample() * 0
            env.step_async(act)
            obs, rew, done, infos = env.step_wait()
            n_steps += 1

            bx = core.blue_x[0].detach().cpu()
            by = core.blue_y[0].detach().cpu()
            carrying = core.blue_carrying[0].detach().cpu()
            red_carrying = core.red_carrying[0].detach().cpu()
            blue_score = int(core.blue_score[0].item())
            tx = core._debug_blue_target_x[0].detach().cpu() if hasattr(core, "_debug_blue_target_x") else None
            ty = core._debug_blue_target_y[0].detach().cpu() if hasattr(core, "_debug_blue_target_y") else None

            if first_entry_step is None and bool((bx > midline).any()):
                first_entry_step = t

            inter_dist = float(((bx[0] - bx[1]) ** 2 + (by[0] - by[1]) ** 2) ** 0.5)

            if first_pickup_step is None:
                min_inter_agent_dist_pre_pickup = min(min_inter_agent_dist_pre_pickup, inter_dist)
                if prev_bx is not None:
                    for a in range(2):
                        cum_dist[a] += float(((bx[a] - prev_bx[a]) ** 2 + (by[a] - prev_by[a]) ** 2) ** 0.5)
                if tx is not None and prev_target_x is not None:
                    if (tx - prev_target_x).abs().max() > TARGET_CHANGE_EPS or (ty - prev_target_y).abs().max() > TARGET_CHANGE_EPS:
                        target_changes += 1
                    if float(((tx[0] - tx[1]) ** 2 + (ty[0] - ty[1]) ** 2) ** 0.5) < SAME_TARGET_EPS:
                        same_target_steps += 1

            if carrying.any() and first_pickup_step is None:
                first_pickup_step = t
                carrier_idx = int(torch.argmax(carrying.to(torch.int64)).item())
                path_len_to_pickup = cum_dist[carrier_idx]

            if red_carrying.any() and not own_flag_lost_before_score and blue_score == 0:
                if own_flag_lost_step is None:
                    own_flag_lost_step = t

            if blue_score > 0 and first_score_step is None:
                first_score_step = t
                if first_pickup_step is not None:
                    return_time = t - first_pickup_step
                if own_flag_lost_step is not None and own_flag_lost_step < t:
                    own_flag_lost_before_score = True

            if first_pickup_step is not None and carrier_idx is not None and tx is not None:
                noncarrier = 1 - carrier_idx
                cx, cy = bx[carrier_idx], by[carrier_idx]
                ntx, nty = tx[noncarrier], ty[noncarrier]
                post_pickup_noncarrier_dist_sum += float(((ntx - cx) ** 2 + (nty - cy) ** 2) ** 0.5)
                post_pickup_steps += 1

            prev_target_x, prev_target_y = tx, ty
            prev_bx, prev_by = bx, by

            if done.any():
                break

        return {
            "seed": seed, "steps": n_steps,
            "first_entry_step": first_entry_step,
            "first_pickup_step": first_pickup_step,
            "path_len_to_pickup": path_len_to_pickup,
            "target_changes_pre_pickup": target_changes,
            "same_target_steps_pre_pickup": same_target_steps,
            "first_score_step": first_score_step,
            "return_time": return_time,
            "own_flag_lost_before_score": own_flag_lost_before_score,
            "min_inter_agent_dist_pre_pickup": min_inter_agent_dist_pre_pickup,
            "post_pickup_mean_noncarrier_to_carrier_dist": (
                post_pickup_noncarrier_dist_sum / post_pickup_steps if post_pickup_steps else None
            ),
        }
    finally:
        env.close()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-seed", type=int, default=591001)
    p.add_argument("--n-episodes", type=int, default=4)
    p.add_argument("--steps", type=int, default=240)
    p.add_argument("--device", default="cuda")
    p.add_argument("--map", default="map_a", help="Canonical niche map (default map_a per locked rule).")
    args = p.parse_args()

    print(f"map={args.map}")
    all_results = {}
    for red in OPPONENTS:
        for style in STYLES:
            key = (red, style)
            results = [
                run_episode(style, red, args.base_seed + i, steps=args.steps, device=args.device, map_layout=args.map)
                for i in range(args.n_episodes)
            ]
            all_results[key] = results

    for red in OPPONENTS:
        print(f"=== {red} (map={args.map}) ===")
        for style in STYLES:
            results = all_results[(red, style)]
            n = len(results)

            def mean_of(field, results=results):
                vals = [r[field] for r in results if r[field] is not None]
                return statistics.mean(vals) if vals else None

            entry = mean_of("first_entry_step")
            pickup = mean_of("first_pickup_step")
            plen = mean_of("path_len_to_pickup")
            tchg = mean_of("target_changes_pre_pickup")
            same = mean_of("same_target_steps_pre_pickup")
            ret = mean_of("return_time")
            mind = mean_of("min_inter_agent_dist_pre_pickup")
            noncarr = mean_of("post_pickup_mean_noncarrier_to_carrier_dist")
            flag_lost = sum(1 for r in results if r["own_flag_lost_before_score"])
            n_pickup = sum(1 for r in results if r["first_pickup_step"] is not None)
            n_score = sum(1 for r in results if r["first_score_step"] is not None)
            mind_s = f"{mind:.2f}" if mind is not None else "n/a"
            plen_s = f"{plen:.2f}" if plen is not None else "n/a"
            print(f"  {style:12s} first_entry={entry} first_pickup={pickup} (n_pickup={n_pickup}/{n}) "
                  f"path_len={plen_s} "
                  f"target_changes={tchg} same_target_steps={same} "
                  f"return_time={ret} (n_score={n_score}/{n}) "
                  f"min_inter_agent_dist={mind_s} "
                  f"post_pickup_noncarrier_dist={noncarr} own_flag_lost_first={flag_lost}/{n}")
            for r in results:
                plen_r = f"{r['path_len_to_pickup']:.2f}" if r['path_len_to_pickup'] is not None else "n/a"
                print(f"    seed={r['seed']} entry={r['first_entry_step']} pickup={r['first_pickup_step']} "
                      f"path_len={plen_r} "
                      f"tchg={r['target_changes_pre_pickup']} same={r['same_target_steps_pre_pickup']} "
                      f"score={r['first_score_step']} return={r['return_time']} "
                      f"flag_lost_first={r['own_flag_lost_before_score']} "
                      f"min_dist={r['min_inter_agent_dist_pre_pickup']:.2f} "
                      f"noncarr_dist={r['post_pickup_mean_noncarrier_to_carrier_dist']}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
