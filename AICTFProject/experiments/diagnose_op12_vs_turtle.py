#!/usr/bin/env python3
"""Event-level diagnostic for OP12_LATE_CONVERTER vs BLUE_TURTLE.

Round-1 Stage 3 (punish TURTLE via late conversion) failed twice on
code-reading hypotheses (blue passivity; shared-attacker-lane pinching --
see docs/research-progress-tracker.md's "OP12 RUSH-niche redesign" section
for both write-ups). This script replaces guessing with the same
trace/event-data diagnosis already used for OP7/8/9/dev26: for each of the
8 dev seeds (base_seed=556001, matching every other OP12 dev screen this
session), log every red-side TAG, PICKUP, DROP, and SCORE event with the
step it happened at and which agent, so the actual failure stage (getting
tagged en route, getting tagged while carrying, dropping the flag, or
something else) is read off real data instead of assumed.
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


def run_episode(seed: int, *, steps: int, device: str) -> dict:
    cfg = GPUFieldConfig(
        n_envs=1, max_blue_agents=2, max_red_agents=2, map_layout="map_b_split_lane",
        max_decision_steps=steps + 10, aquaticus_profile=True, rules_profile="OURS",
        device=device, seed=seed,
    )
    env = GPUCTFVecEnv(cfg)
    core = env.core
    events = []
    try:
        env.env_method("set_phase", "OP12_LATE_CONVERTER")
        env.env_method("set_next_opponent", "SCRIPTED", "OP12_LATE_CONVERTER")
        core.blue_scripted = True
        core.set_blue_style("BLUE_TURTLE")
        env.reset()

        prev_red_tagged = core.red_tagged[0].detach().cpu().clone()
        prev_red_carrying = core.red_carrying[0].detach().cpu().clone()
        prev_red_score = int(core.red_score[0].item())
        prev_blue_score = int(core.blue_score[0].item())
        prev_roles = list(core.bt_red_role[0].detach().cpu().tolist())

        n_steps = 0
        for t in range(steps):
            act = env.action_space.sample() * 0
            env.step_async(act)
            obs, rew, done, infos = env.step_wait()
            n_steps += 1

            red_tagged = core.red_tagged[0].detach().cpu()
            red_carrying = core.red_carrying[0].detach().cpu()
            red_score = int(core.red_score[0].item())
            blue_score = int(core.blue_score[0].item())
            roles = list(core.bt_red_role[0].detach().cpu().tolist())
            rx = core.red_x[0].detach().cpu().tolist()
            ry = core.red_y[0].detach().cpu().tolist()
            bx = core.blue_x[0].detach().cpu().tolist()
            by = core.blue_y[0].detach().cpu().tolist()

            for j in range(red_tagged.shape[0]):
                if bool(red_tagged[j]) and not bool(prev_red_tagged[j]):
                    was_carrying = bool(prev_red_carrying[j])
                    events.append(
                        f"t={t:3d} TAG   red[{j}] role={prev_roles[j]} was_carrying={was_carrying} "
                        f"red_pos=({rx[j]:.1f},{ry[j]:.1f})"
                    )
                if bool(prev_red_carrying[j]) and not bool(red_carrying[j]) and not bool(red_tagged[j]):
                    events.append(f"t={t:3d} DROP-NOTAG red[{j}] pos=({rx[j]:.1f},{ry[j]:.1f}) (unexpected)")
                if bool(red_carrying[j]) and not bool(prev_red_carrying[j]):
                    events.append(f"t={t:3d} PICKUP red[{j}] role={roles[j]} pos=({rx[j]:.1f},{ry[j]:.1f})")

            if red_score != prev_red_score:
                events.append(f"t={t:3d} SCORE red  {prev_red_score}->{red_score}")
            if blue_score != prev_blue_score:
                events.append(f"t={t:3d} SCORE blue {prev_blue_score}->{blue_score}")

            prev_red_tagged = red_tagged
            prev_red_carrying = red_carrying
            prev_red_score = red_score
            prev_blue_score = blue_score
            prev_roles = roles

            if done.any():
                events.append(f"t={t:3d} EPISODE_END")
                break

        return {
            "seed": seed,
            "steps": n_steps,
            "final_red_score": prev_red_score,
            "final_blue_score": prev_blue_score,
            "events": events,
        }
    finally:
        env.close()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-seed", type=int, default=556001)
    p.add_argument("--n-episodes", type=int, default=8)
    p.add_argument("--steps", type=int, default=240)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    all_tag_while_carrying = 0
    all_tag_while_not_carrying = 0
    all_pickups = 0
    all_scores = 0

    for i in range(args.n_episodes):
        seed = args.base_seed + i
        result = run_episode(seed, steps=args.steps, device=args.device)
        print(f"=== episode {i} seed={seed} steps={result['steps']} "
              f"final red={result['final_red_score']} blue={result['final_blue_score']} ===")
        for e in result["events"]:
            print("  " + e)
            if "TAG" in e and "was_carrying=True" in e:
                all_tag_while_carrying += 1
            elif e.split()[1] == "TAG":
                all_tag_while_not_carrying += 1
            elif "PICKUP" in e:
                all_pickups += 1
            elif "SCORE red" in e:
                all_scores += 1
        print()

    print("=" * 72)
    print(f"totals across {args.n_episodes} episodes:")
    print(f"  red pickups (of blue's flag): {all_pickups}")
    print(f"  red scores:                   {all_scores}")
    print(f"  red tagged WHILE carrying:     {all_tag_while_carrying}  (pickup wasted -> dropped)")
    print(f"  red tagged NOT carrying:       {all_tag_while_not_carrying}  (blocked en route)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
