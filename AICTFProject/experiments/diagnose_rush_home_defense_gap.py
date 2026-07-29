#!/usr/bin/env python3
"""Rank OP6-OP12 (map_a, canonical) on the metrics needed to find a
TURTLE-engineering candidate: a context where RUSH's abandoned-home
weakness is punished hard, and a home-anchoring style could plausibly stop
that punishment.

Diagnostic-only: does not modify any opponent, profile, or blue probe.
Uses the corrected episode-setup pattern (re-apply phase/opponent/style
after env.reset(); read final score from infos[0]["episode_result"], not
the core score tensors directly) verified against
run_scripted_style_payoff_matrix.py's _run_one_episode in this session.

Metrics per (red, blue_style) cell, averaged over episodes:
  - red score rate while both blue agents are simultaneously away from
    home (x > home-side threshold) -- proxy for "abandoned base punished"
  - blue's own flag lost (red picked it up) before blue's first score
  - red pickup-to-score conversion rate (red grabs blue's flag -> red
    eventually scores)
  - simultaneous-carry frequency: fraction of steps where BOTH teams have
    a carrier at once (mutual-aggression / race proxy)
  - mean margin, win rate (for reference against the already-collected
    RUSH/TURTLE payoff numbers)
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
    "OP9_SPLIT_LANE_FEINT",
    "OP10_AGGRESSIVE_INTERCEPTOR",
    "OP11_ADAPTIVE_EXPLOITER",
    "OP12_LATE_CONVERTER",
)
STYLES = ("BLUE_RUSH", "BLUE_TURTLE")


def run_episode(style: str, red: str, seed: int, *, steps: int, device: str) -> dict:
    cfg = GPUFieldConfig(
        n_envs=1, max_blue_agents=2, max_red_agents=2, map_layout="map_a",
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
        env.env_method("set_phase", red)
        env.env_method("set_next_opponent", "SCRIPTED", red)
        core.blue_scripted = True
        core.set_blue_style(style)

        midline = float(core.cols) * 0.5
        away_steps = 0
        away_red_score_events = 0
        prev_red_score = 0
        first_blue_score_step = None
        own_flag_lost_step = None
        own_flag_lost_before_score = False
        red_pickup_ever = False
        red_pickup_to_score = False
        first_red_pickup_seen = False
        both_carry_steps = 0
        n_steps = 0
        act = env.action_space.sample() * 0
        last_info: dict = {}

        for t in range(steps):
            env.step_async(act)
            obs, rew, done, infos = env.step_wait()
            n_steps += 1
            last_info = infos[0] if infos else {}

            bx = core.blue_x[0].detach().cpu()
            blue_carrying = core.blue_carrying[0].detach().cpu()
            red_carrying = core.red_carrying[0].detach().cpu()
            blue_score = int(core.blue_score[0].item())
            red_score = int(core.red_score[0].item())

            both_away = bool((bx > midline).all())
            if both_away:
                away_steps += 1
                if red_score > prev_red_score:
                    away_red_score_events += 1
            prev_red_score = red_score

            if red_carrying.any():
                if not first_red_pickup_seen:
                    first_red_pickup_seen = True
                red_pickup_ever = True

            if blue_score > 0 and first_blue_score_step is None:
                first_blue_score_step = t
            if red_carrying.any() and blue_score == 0 and own_flag_lost_step is None:
                own_flag_lost_step = t
            if first_blue_score_step is not None and own_flag_lost_step is not None and own_flag_lost_step < first_blue_score_step:
                own_flag_lost_before_score = True

            if blue_carrying.any() and red_carrying.any():
                both_carry_steps += 1

            if done.any():
                break

        ep_res = last_info.get("episode_result", last_info)
        blue_final = int(ep_res.get("blue_score", 0))
        red_final = int(ep_res.get("red_score", 0))
        if red_pickup_ever and red_final > 0:
            red_pickup_to_score = True

        return {
            "seed": seed, "steps": n_steps, "margin": blue_final - red_final,
            "win": blue_final > red_final,
            "away_steps_frac": away_steps / max(1, n_steps),
            "red_score_events_while_away": away_red_score_events,
            "red_final_score": red_final,
            "own_flag_lost_before_score": own_flag_lost_before_score,
            "red_pickup_ever": red_pickup_ever,
            "red_pickup_to_score": red_pickup_to_score,
            "both_carry_frac": both_carry_steps / max(1, n_steps),
        }
    finally:
        env.close()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-seed", type=int, default=661001)
    p.add_argument("--n-episodes", type=int, default=8)
    p.add_argument("--steps", type=int, default=240)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    print("map=map_a")
    for red in OPPONENTS:
        print(f"=== {red} ===")
        for style in STYLES:
            results = [
                run_episode(style, red, args.base_seed + i, steps=args.steps, device=args.device)
                for i in range(args.n_episodes)
            ]
            n = len(results)
            margin = statistics.mean(r["margin"] for r in results)
            wins = sum(1 for r in results if r["win"])
            red_scored_while_away_rate = sum(r["red_score_events_while_away"] for r in results) / n
            flag_lost_first = sum(1 for r in results if r["own_flag_lost_before_score"])
            n_red_pickup = sum(1 for r in results if r["red_pickup_ever"])
            n_red_conv = sum(1 for r in results if r["red_pickup_to_score"])
            conv_rate = (n_red_conv / n_red_pickup) if n_red_pickup else None
            both_carry = statistics.mean(r["both_carry_frac"] for r in results)
            conv_s = f"{conv_rate:.2f}" if conv_rate is not None else "n/a"
            print(f"  {style:12s} margin={margin:+.3f} WR={wins}/{n} "
                  f"red_score_while_both_away={red_scored_while_away_rate:.2f}/ep "
                  f"own_flag_lost_first={flag_lost_first}/{n} "
                  f"red_conv={conv_s} ({n_red_conv}/{n_red_pickup}) "
                  f"both_carry_frac={both_carry:.3f}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
