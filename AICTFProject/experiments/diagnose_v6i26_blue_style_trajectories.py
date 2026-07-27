#!/usr/bin/env python3
"""Trajectory diagnostic for the four scripted blue probe styles.

Required gate, per the pool-admissibility protocol, BEFORE building the full
payoff-matrix runner (run_scripted_style_payoff_matrix.py): confirm each
style actually expresses its intended behavior, not just that it runs
without crashing.

  rush   shows strong early forward pressure
  turtle has the highest home-half occupancy
  split  has the greatest average agent-to-agent y-separation
  escort has the smallest carrier-to-teammate distance (while carrying)

Each check is comparative but tolerant. In particular, RUSH is not required
to cross midfield first on every seed; it is validated by early forward
progress and low home-defense occupancy.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gpu_env import GPUCTFVecEnv, GPUFieldConfig  # noqa: E402
from gpu_env._core._scripted_blue_styles import BLUE_STYLE_NAMES  # noqa: E402


def _run_trajectory(style: str, *, opponent: str, map_name: str, steps: int, seed: int, device: str) -> dict:
    cfg = GPUFieldConfig(
        n_envs=1, max_blue_agents=2, max_red_agents=2, map_layout=map_name,
        max_decision_steps=steps + 10, aquaticus_profile=True, rules_profile="OURS",
        device=device, seed=seed,
    )
    env = GPUCTFVecEnv(cfg)
    core = env.core
    try:
        env.env_method("set_phase", opponent)
        env.env_method("set_next_opponent", "SCRIPTED", opponent)
        core.blue_scripted = True
        core.set_blue_style(style)
        env.reset()

        midline = float(core.cols) * 0.5
        cross_step = None
        home_half_steps = 0
        y_sep_sum = 0.0
        pre_pickup_y_sep_sum = 0.0
        pre_pickup_steps = 0
        pre_pickup_home_half_steps = 0
        pre_pickup_forward_sum = 0.0
        pre_pickup_cross_agent_steps = 0
        simultaneous_lane_penetration_steps = 0
        carrying_dist_sum = 0.0
        carrying_steps = 0
        post_pickup_y_sep_sum = 0.0
        post_pickup_return_progress_sum = 0.0
        first_pickup_step = None
        early_progress_sum = 0.0
        early_cross_agent_steps = 0
        n_steps_taken = 0

        for t in range(steps):
            act = env.action_space.sample() * 0
            env.step_async(act)
            obs, rew, done, infos = env.step_wait()
            n_steps_taken += 1

            bx = core.blue_x[0].detach().cpu().numpy()
            by = core.blue_y[0].detach().cpu().numpy()
            carrying = core.blue_carrying[0].detach().cpu().numpy()
            before_pickup = first_pickup_step is None and not carrying.any()

            if cross_step is None and np.any(bx > midline):
                cross_step = t
            if t < min(30, steps):
                early_progress_sum += float(np.mean(bx))
                early_cross_agent_steps += int(np.sum(bx > midline))
            if np.all(bx < midline):
                home_half_steps += 1
            y_sep_sum += abs(float(by[0] - by[1]))
            if before_pickup:
                pre_pickup_steps += 1
                pre_pickup_forward_sum += float(np.mean(bx))
                pre_pickup_cross_agent_steps += int(np.sum(bx > midline))
                pre_pickup_y_sep_sum += abs(float(by[0] - by[1]))
                pre_pickup_home_half_steps += int(np.all(bx < midline))
                lane_sides = np.sign(by - (float(core.rows) * 0.5))
                simultaneous_lane_penetration_steps += int(
                    np.all(bx > midline) and lane_sides[0] * lane_sides[1] < 0
                )
            if carrying.any():
                if first_pickup_step is None:
                    first_pickup_step = t
                carrier_idx = int(np.argmax(carrying))
                other_idx = 1 - carrier_idx
                carrying_dist_sum += float(np.hypot(bx[carrier_idx] - bx[other_idx], by[carrier_idx] - by[other_idx]))
                post_pickup_y_sep_sum += abs(float(by[0] - by[1]))
                post_pickup_return_progress_sum += float(core.red_flag_pos[0, 0].item() - bx[carrier_idx])
                carrying_steps += 1

            if done.any():
                break

        return {
            "style": style,
            "steps_taken": n_steps_taken,
            "midfield_cross_step": cross_step if cross_step is not None else n_steps_taken,
            "early_forward_progress": early_progress_sum / max(1, min(30, n_steps_taken)),
            "early_cross_agent_steps": early_cross_agent_steps,
            "home_half_occupancy": home_half_steps / max(1, n_steps_taken),
            "mean_y_separation": y_sep_sum / max(1, n_steps_taken),
            "first_pickup_step": first_pickup_step if first_pickup_step is not None else n_steps_taken,
            "pre_pickup_steps": pre_pickup_steps,
            "pre_pickup_forward_progress": pre_pickup_forward_sum / max(1, pre_pickup_steps),
            "pre_pickup_cross_agent_steps": pre_pickup_cross_agent_steps,
            "pre_pickup_home_half_occupancy": pre_pickup_home_half_steps / max(1, pre_pickup_steps),
            "pre_pickup_mean_y_separation": pre_pickup_y_sep_sum / max(1, pre_pickup_steps),
            "pre_pickup_simultaneous_lane_penetration_steps": simultaneous_lane_penetration_steps,
            "carrying_steps": carrying_steps,
            "mean_carrier_teammate_dist": (carrying_dist_sum / carrying_steps) if carrying_steps > 0 else float("nan"),
            "post_pickup_mean_y_separation": (post_pickup_y_sep_sum / carrying_steps) if carrying_steps > 0 else float("nan"),
            "post_pickup_mean_return_progress": (
                post_pickup_return_progress_sum / carrying_steps
            ) if carrying_steps > 0 else float("nan"),
        }
    finally:
        env.close()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--opponent", default="OP9_SPLIT_LANE_FEINT")
    p.add_argument("--map-name", default="map_a_open")
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    results = {}
    for style in BLUE_STYLE_NAMES:
        r = _run_trajectory(style, opponent=args.opponent, map_name=args.map_name,
                             steps=args.steps, seed=args.seed, device=args.device)
        results[style] = r
        print(f"{style:12s} steps={r['steps_taken']:4d}  midfield_cross_step={r['midfield_cross_step']:4d}  "
              f"early_x={r['early_forward_progress']:.3f}  early_cross_agent_steps={r['early_cross_agent_steps']:3d}  "
              f"home_half_occ={r['home_half_occupancy']:.3f}  mean_y_sep={r['mean_y_separation']:.3f}  "
              f"pre_y_sep={r['pre_pickup_mean_y_separation']:.3f}  pre_lane_pen={r['pre_pickup_simultaneous_lane_penetration_steps']:3d}  "
              f"carrying_steps={r['carrying_steps']:4d}  mean_carrier_teammate_dist={r['mean_carrier_teammate_dist']:.3f}  "
              f"post_y_sep={r['post_pickup_mean_y_separation']:.3f}")

    print()
    checks = []

    best_early_x = max(results, key=lambda s: results[s]["pre_pickup_forward_progress"])
    best_cross_steps = max(results, key=lambda s: results[s]["pre_pickup_cross_agent_steps"])
    lowest_home = min(results, key=lambda s: results[s]["pre_pickup_home_half_occupancy"])
    rush_cross_tol = results["BLUE_RUSH"]["midfield_cross_step"] <= min(
        r["midfield_cross_step"] for r in results.values()
    ) + 2
    ok = (
        best_early_x == "BLUE_RUSH"
        or best_cross_steps == "BLUE_RUSH"
        or (lowest_home == "BLUE_RUSH" and rush_cross_tol)
        or (
            rush_cross_tol
            and results["BLUE_RUSH"]["early_forward_progress"]
            >= results["BLUE_ESCORT"]["early_forward_progress"] - 0.25
            and results["BLUE_RUSH"]["early_cross_agent_steps"]
            >= results["BLUE_ESCORT"]["early_cross_agent_steps"] - 2
        )
    )
    checks.append(("rush shows early offensive pressure", ok,
                    f"best_early_x={best_early_x}, best_cross_agent_steps={best_cross_steps}, "
                    f"lowest_home={lowest_home}, rush_cross={results['BLUE_RUSH']['midfield_cross_step']}, "
                    f"rush_early_x={results['BLUE_RUSH']['early_forward_progress']:.3f}, "
                    f"escort_early_x={results['BLUE_ESCORT']['early_forward_progress']:.3f}"))

    highest_home_occ = max(results, key=lambda s: results[s]["pre_pickup_home_half_occupancy"])
    ok = highest_home_occ == "BLUE_TURTLE"
    checks.append(("turtle has highest home-half occupancy", ok,
                    f"highest={highest_home_occ} (turtle={results['BLUE_TURTLE']['pre_pickup_home_half_occupancy']:.3f})"))

    highest_y_sep = max(results, key=lambda s: results[s]["pre_pickup_mean_y_separation"])
    ok = highest_y_sep == "BLUE_SPLIT"
    checks.append(("split has greatest pre-pickup y-separation", ok,
                    f"highest={highest_y_sep} (split={results['BLUE_SPLIT']['pre_pickup_mean_y_separation']:.3f})"))

    most_lane_pen = max(results, key=lambda s: results[s]["pre_pickup_simultaneous_lane_penetration_steps"])
    ok = most_lane_pen == "BLUE_SPLIT"
    checks.append(("split has strongest pre-pickup simultaneous lane penetration", ok,
                    f"highest={most_lane_pen} "
                    f"(split={results['BLUE_SPLIT']['pre_pickup_simultaneous_lane_penetration_steps']})"))

    with_carry = {s: r for s, r in results.items() if r["carrying_steps"] > 0}
    if with_carry:
        smallest_dist = min(with_carry, key=lambda s: with_carry[s]["mean_carrier_teammate_dist"])
        ok = smallest_dist == "BLUE_ESCORT"
        checks.append(("escort has smallest carrier-teammate distance", ok,
                        f"smallest={smallest_dist} among {list(with_carry)} "
                        f"(escort={results['BLUE_ESCORT']['mean_carrier_teammate_dist']:.3f} "
                        f"if carrying occurred)"))
        split_less_clustered_than_escort = (
            "BLUE_SPLIT" in with_carry
            and "BLUE_ESCORT" in with_carry
            and results["BLUE_SPLIT"]["mean_carrier_teammate_dist"]
            > results["BLUE_ESCORT"]["mean_carrier_teammate_dist"]
        )
        checks.append(("split remains less clustered than escort while carrying", split_less_clustered_than_escort,
                        f"split={results['BLUE_SPLIT']['mean_carrier_teammate_dist']:.3f}, "
                        f"escort={results['BLUE_ESCORT']['mean_carrier_teammate_dist']:.3f}"))
    else:
        checks.append(("escort has smallest carrier-teammate distance", False,
                        "no style ever obtained the flag in this run -- increase --steps or retry with a weaker opponent"))

    print("=" * 72)
    all_pass = True
    for name, ok, detail in checks:
        status = "PASS" if ok else "FAIL"
        all_pass = all_pass and ok
        print(f"[{status}] {name}: {detail}")
    print("=" * 72)
    print(f"Overall: {'PASS -- styles express their intended behavior' if all_pass else 'FAIL -- do not trust the payoff matrix yet'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
