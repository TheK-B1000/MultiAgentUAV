#!/usr/bin/env python3
"""Test 1 of the SPLIT-dominance root-cause audit: does BLUE_SPLIT dominate
even against a red opponent with NO sophisticated behavior-tree logic?

Uses OP1 as the neutral baseline -- confirmed (via code research) to be the
simplest legacy-brain opponent: attacker_style=0, defender_style=0,
role_switch_prob=0.05, zero deception, zero coordinated_attack
(opponent_params.py). OP1 never dispatches through the adaptive _bt_*
pipeline at all (bt_dispatch_level_for_opponent_key returns None for
level<6) -- it always runs the same legacy `_assign_scripted_targets_by_role`
brain as every non-BT opponent, with none of OP6-OP12's per-role lock
counters, threat-radius, or split-lane-pressure detection.

Important gotcha this script works around: _rules.py's red_score_allowed
gate is `~phase_tensor_equals(("OP1","OP2"))`, keyed on the CURRICULUM
PHASE (self._phase), not the opponent-parameter key (_opponent_key) -- if
phase is left at "OP1", red can NEVER score regardless of behavior, which
would make BLUE_SPLIT trivially "dominate" for a reason that has nothing
to do with tactical competence. This script decouples them: phase is set
to a neutral, non-gated value ("OP3") while set_next_opponent still points
at OP1's actual parameter set, so red can score normally on its own merits.

Diagnostic-only: does not modify any opponent, profile, or BT default.
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gpu_env import GPUCTFVecEnv, GPUFieldConfig  # noqa: E402
from gpu_env._core._scripted_blue_styles import BLUE_STYLE_NAMES  # noqa: E402

NEUTRAL_PHASE = "OP3"  # any phase other than OP1/OP2 avoids the score gate


def run_episode(style: str, seed: int, *, steps: int, device: str) -> dict:
    cfg = GPUFieldConfig(
        n_envs=1, max_blue_agents=2, max_red_agents=2, map_layout="map_b_split_lane",
        max_decision_steps=steps + 10, aquaticus_profile=True, rules_profile="OURS",
        device=device, seed=seed,
    )
    env = GPUCTFVecEnv(cfg)
    core = env.core
    try:
        env.env_method("set_phase", NEUTRAL_PHASE)
        env.env_method("set_next_opponent", "SCRIPTED", "OP1")
        core.blue_scripted = True
        core.set_blue_style(style)
        env.reset()

        n_steps = 0
        for t in range(steps):
            act = env.action_space.sample() * 0
            env.step_async(act)
            obs, rew, done, infos = env.step_wait()
            n_steps += 1
            if done.any():
                break

        blue_score = int(core.blue_score[0].item())
        red_score = int(core.red_score[0].item())
        return {"seed": seed, "steps": n_steps, "blue_score": blue_score,
                 "red_score": red_score, "win_margin": blue_score - red_score}
    finally:
        env.close()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-seed", type=int, default=581001)
    p.add_argument("--n-episodes", type=int, default=8)
    p.add_argument("--steps", type=int, default=240)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    all_results = {}
    for style in BLUE_STYLE_NAMES:
        results = [run_episode(style, args.base_seed + i, steps=args.steps, device=args.device)
                   for i in range(args.n_episodes)]
        all_results[style] = results
        margins = [r["win_margin"] for r in results]
        wins = sum(1 for m in margins if m > 0)
        print(f"{style:12s} mean_margin={statistics.mean(margins):+.3f}  "
              f"WR={wins}/{args.n_episodes}  margins={margins}")

    print()
    print("paired per-seed win_margin (all styles, same 8 seeds):")
    print(f"{'ep':>3} {'seed':>8}  " + "  ".join(f"{s:>12}" for s in BLUE_STYLE_NAMES))
    for i in range(args.n_episodes):
        row = [str(all_results[s][i]["win_margin"]) for s in BLUE_STYLE_NAMES]
        print(f"{i:>3} {args.base_seed+i:>8}  " + "  ".join(f"{v:>12}" for v in row))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
