#!/usr/bin/env python3
"""Cross-opponent SPLIT-dominance root-cause audit -- Test 3 of the
diagnostic suite requested after OP6, OP7, OP8, OP9, OP10, and OP12 all
independently turned out BLUE_SPLIT-dominant (see docs/research-progress-
tracker.md, "OP6 unmodified development screen" section). Diagnostic-only,
read-only: does not modify any opponent, profile, or BT default. Freezes
every red preset and runs the existing BLUE_SPLIT probe unchanged.

For each opponent, logs per step across N episodes:
  - which blue agent (0/1) is "uncovered" (min distance to any alive red
    agent exceeds a fixed threshold) and the longest uncovered streak any
    single blue agent achieves
  - red role-churn rate (fraction of steps where either red agent's role
    differs from the previous step)
  - first pickup step per blue agent, first score, and whether the scoring
    carrier was the agent with the longer uncovered streak up to that point
    ("scored through the ignored lane")
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gpu_env import GPUCTFVecEnv, GPUFieldConfig  # noqa: E402

UNCOVERED_DIST = 3.0  # roughly 2x tag range; a blue agent farther than this
                       # from BOTH red agents has no immediate tag threat.

OPPONENTS = (
    "OP6_IMMEDIATE_DUAL_RUSH",
    "OP7_DEEP_FORTRESS",
    "OP8_PROTECTED_CARRIER_ESCORT",
    "OP9_SPLIT_LANE_FEINT",
    "OP10_AGGRESSIVE_INTERCEPTOR",
    "OP12_LATE_CONVERTER",
)


def run_episode(red: str, seed: int, *, steps: int, device: str) -> dict:
    cfg = GPUFieldConfig(
        n_envs=1, max_blue_agents=2, max_red_agents=2, map_layout="map_b_split_lane",
        max_decision_steps=steps + 10, aquaticus_profile=True, rules_profile="OURS",
        device=device, seed=seed,
    )
    env = GPUCTFVecEnv(cfg)
    core = env.core
    try:
        env.env_method("set_phase", red)
        env.env_method("set_next_opponent", "SCRIPTED", red)
        core.blue_scripted = True
        core.set_blue_style("BLUE_SPLIT")
        env.reset()

        uncovered_streak = [0, 0]
        max_uncovered_streak = [0, 0]
        first_pickup_step = [-1, -1]
        first_score_step = -1
        scored_via_ignored_lane = None
        role_changes = 0
        prev_roles = None
        prev_carrying = None
        n_steps = 0

        for t in range(steps):
            act = env.action_space.sample() * 0
            env.step_async(act)
            obs, rew, done, infos = env.step_wait()
            n_steps += 1

            rx = core.red_x[0].detach().cpu()
            ry = core.red_y[0].detach().cpu()
            bx = core.blue_x[0].detach().cpu()
            by = core.blue_y[0].detach().cpu()
            red_alive = core.red_alive[0].detach().cpu()
            roles = core.bt_red_role[0].detach().cpu().tolist()
            carrying = core.blue_carrying[0].detach().cpu()
            blue_score = int(core.blue_score[0].item())

            for b in range(2):
                dists = []
                for r in range(2):
                    if bool(red_alive[r]):
                        dists.append(float(((rx[r] - bx[b]) ** 2 + (ry[r] - by[b]) ** 2) ** 0.5))
                min_d = min(dists) if dists else 1e9
                if min_d > UNCOVERED_DIST:
                    uncovered_streak[b] += 1
                    max_uncovered_streak[b] = max(max_uncovered_streak[b], uncovered_streak[b])
                else:
                    uncovered_streak[b] = 0

            if prev_roles is not None and roles != prev_roles:
                role_changes += 1
            prev_roles = roles

            for b in range(2):
                if bool(carrying[b]) and (prev_carrying is None or not bool(prev_carrying[b])):
                    if first_pickup_step[b] < 0:
                        first_pickup_step[b] = t

            if blue_score > 0 and first_score_step < 0:
                first_score_step = t
                carrier_b = int(torch.argmax(carrying.to(torch.int64)).item()) if carrying.any() else -1
                other_b = 1 - carrier_b if carrier_b >= 0 else -1
                if carrier_b >= 0 and other_b >= 0:
                    scored_via_ignored_lane = max_uncovered_streak[other_b] > max_uncovered_streak[carrier_b]

            prev_carrying = carrying

            if done.any():
                break

        return {
            "seed": seed,
            "steps": n_steps,
            "max_uncovered_streak": max_uncovered_streak,
            "role_change_rate": role_changes / max(1, n_steps),
            "first_pickup_step": first_pickup_step,
            "first_score_step": first_score_step,
            "scored_via_ignored_lane": scored_via_ignored_lane,
        }
    finally:
        env.close()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-seed", type=int, default=571001)
    p.add_argument("--n-episodes", type=int, default=4)
    p.add_argument("--steps", type=int, default=240)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    for red in OPPONENTS:
        print(f"=== {red} ===")
        results = []
        for i in range(args.n_episodes):
            seed = args.base_seed + i
            r = run_episode(red, seed, steps=args.steps, device=args.device)
            results.append(r)
            print(
                f"  seed={seed} steps={r['steps']:3d} "
                f"max_uncov=[{r['max_uncovered_streak'][0]:3d},{r['max_uncovered_streak'][1]:3d}] "
                f"role_chg_rate={r['role_change_rate']:.3f} "
                f"first_pickup={r['first_pickup_step']} first_score={r['first_score_step']} "
                f"scored_via_ignored_lane={r['scored_via_ignored_lane']}"
            )
        n = len(results)
        mean_max_uncov = sum(max(r["max_uncovered_streak"]) for r in results) / n
        mean_churn = sum(r["role_change_rate"] for r in results) / n
        ignored_lane_true = sum(1 for r in results if r["scored_via_ignored_lane"] is True)
        ignored_lane_known = sum(1 for r in results if r["scored_via_ignored_lane"] is not None)
        print(f"  -- summary: mean_max_uncovered_streak={mean_max_uncov:.1f} steps, "
              f"mean_role_change_rate={mean_churn:.3f}, "
              f"scored_via_ignored_lane={ignored_lane_true}/{ignored_lane_known}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
