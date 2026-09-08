#!/usr/bin/env python3
"""OP6 mutual-carry race denial diagnostic (map_a) — RUSH + TURTLE control.

Logs race-mode activation, interceptor target, ETAs, first contact, blue
delay, and who scores first. Focused gates:

  RUSH blue carrier interrupted/delayed: ≥5/8
  RUSH red-first:                        ≥5/8
  TURTLE race-mode activation:           ≤1/8
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.run_scripted_style_payoff_matrix import (  # noqa: E402
    _episode_seed,
    _make_env,
    _zero_action,
)
from gpu_env._core._bt_red import ROLE_INTERCEPTOR, _BTRedMixin  # noqa: E402

BLUE_STYLES = ("BLUE_RUSH", "BLUE_TURTLE")
RED_STYLE = "OP6_IMMEDIATE_DUAL_RUSH"
DEFAULT_MAP = "map_a"
CAP_R = 1.2
TAG_FALLBACK = 2.5


def _eta(dist: float, speed: float, contact_r: float = 0.0) -> float:
    return max(0.0, float(dist) - float(contact_r)) / max(float(speed), 1e-6)


def _run_episode(
    *,
    blue_style: str,
    episode_index: int,
    episode_seed: int,
    map_name: str,
    max_decision_steps: int,
    device: str,
) -> dict[str, Any]:
    env = _make_env(
        map_name=map_name,
        seed=episode_seed,
        max_decision_steps=max_decision_steps,
        device=device,
    )
    try:
        core = env.core
        env.env_method("set_phase", RED_STYLE)
        env.env_method("set_next_opponent", "SCRIPTED", RED_STYLE)
        core.blue_scripted = True
        core.set_blue_style(blue_style)
        env.reset()
        env.env_method("set_phase", RED_STYLE)
        env.env_method("set_next_opponent", "SCRIPTED", RED_STYLE)
        core.blue_scripted = True
        core.set_blue_style(blue_style)

        max_speed = float(core.cfg.max_speed_cps)
        tag_r = float(getattr(core.cfg, "tag_range_cells", TAG_FALLBACK))
        bhome_x = float(core.blue_flag_home[0, 0].item())
        bhome_y = float(core.blue_flag_home[0, 1].item())

        prev_race_act = int(core.bt_op6_race_activations[0].item())
        prev_blue_carry = bool(
            (core.blue_carrying[0] & core.blue_alive[0] & (~core.blue_tagged[0]))
            .any()
            .item()
        )
        prev_blue_score = 0

        race_activate_step = -1
        intercept_eta_at_arm = float("nan")
        blue_score_eta_at_arm = float("nan")
        race_target = -1
        first_contact_step = -1
        blue_interrupted = 0
        blue_delay_steps = 0
        blue_score_eta_at_arm_int = -1
        expected_blue_score_step = -1
        red_first_score_step = -1
        blue_first_score_step = -1
        steps = 0
        last_info: dict[str, Any] = {}

        for _ in range(int(max_decision_steps) + 5):
            action = _zero_action(env)
            env.step_async(action)
            _, _, done, infos = env.step_wait()
            last_info = infos[0] if infos else {}
            sim = steps + 1

            ep_res = last_info.get("episode_result", last_info)
            blue_score_now = int(ep_res.get("blue_score", core.blue_score[0].item()))
            red_score_now = int(ep_res.get("red_score", core.red_score[0].item()))
            if red_first_score_step < 0 and red_score_now > 0:
                red_first_score_step = sim
            if blue_first_score_step < 0 and blue_score_now > 0:
                blue_first_score_step = sim
                if expected_blue_score_step >= 0 and blue_first_score_step > expected_blue_score_step:
                    blue_delay_steps = blue_first_score_step - expected_blue_score_step
                    blue_interrupted = 1
                elif expected_blue_score_step >= 0 and blue_first_score_step < 0:
                    pass

            race_act = int(core.bt_op6_race_activations[0].item())
            race_on = int(core.bt_op6_race_ticks[0].item()) > 0
            if race_act > prev_race_act and race_activate_step < 0:
                race_activate_step = sim
                race_target = int(core.bt_op6_race_target_idx[0].item())
                # Interceptor = non-carrier red.
                red_carry = core.red_carrying[0] & core.red_alive[0] & (~core.red_tagged[0])
                rc = int(torch.argmax(red_carry.to(torch.int64)).item()) if bool(red_carry.any()) else 0
                ri = 1 - rc
                if race_target < 0:
                    race_target = int(torch.argmax(core.blue_carrying[0].to(torch.int64)).item())
                rx = float(core.red_x[0, ri].item())
                ry = float(core.red_y[0, ri].item())
                bx = float(core.blue_x[0, race_target].item())
                by = float(core.blue_y[0, race_target].item())
                dist_i = math.hypot(rx - bx, ry - by)
                dist_b = math.hypot(bx - bhome_x, by - bhome_y)
                intercept_eta_at_arm = _eta(dist_i, max_speed, tag_r)
                blue_score_eta_at_arm = _eta(dist_b, max_speed, CAP_R)
                blue_score_eta_at_arm_int = int(round(blue_score_eta_at_arm))
                expected_blue_score_step = sim + blue_score_eta_at_arm_int

            # First contact: interceptor within tag range of blue carrier.
            if race_on and first_contact_step < 0:
                tgt = int(core.bt_op6_race_target_idx[0].item())
                if tgt >= 0:
                    roles = core.bt_red_role[0]
                    for j in range(int(core.Nr)):
                        if int(roles[j].item()) != ROLE_INTERCEPTOR:
                            continue
                        d = math.hypot(
                            float(core.red_x[0, j].item())
                            - float(core.blue_x[0, tgt].item()),
                            float(core.red_y[0, j].item())
                            - float(core.blue_y[0, tgt].item()),
                        )
                        if d <= tag_r:
                            first_contact_step = sim
                            break

            blue_carry = bool(
                (core.blue_carrying[0] & core.blue_alive[0] & (~core.blue_tagged[0]))
                .any()
                .item()
            )
            # Interrupted: lost blue carry without a score while race was armed.
            if (
                prev_blue_carry
                and (not blue_carry)
                and blue_score_now == prev_blue_score
                and (race_on or race_activate_step >= 0)
            ):
                blue_interrupted = 1
                if blue_first_score_step < 0 and expected_blue_score_step >= 0:
                    # Blue never scored this race window — large delay.
                    blue_delay_steps = max(
                        blue_delay_steps, max(0, sim - race_activate_step)
                    )

            prev_race_act = race_act
            prev_blue_carry = blue_carry
            prev_blue_score = blue_score_now
            steps += 1
            if bool(done.any() if hasattr(done, "any") else done):
                break

        # If blue never scored after arm, count as delayed/interrupted when
        # red scored first or contact happened.
        if (
            race_activate_step >= 0
            and blue_interrupted == 0
            and (
                first_contact_step >= 0
                or (
                    red_first_score_step >= 0
                    and (
                        blue_first_score_step < 0
                        or red_first_score_step <= blue_first_score_step
                    )
                )
            )
        ):
            blue_interrupted = 1

        who = "neither"
        if red_first_score_step >= 0 and (
            blue_first_score_step < 0 or red_first_score_step <= blue_first_score_step
        ):
            who = "red"
        elif blue_first_score_step >= 0:
            who = "blue"

        return {
            "blue_style": blue_style,
            "red_style": RED_STYLE,
            "map": map_name,
            "episode_index": episode_index,
            "episode_seed": episode_seed,
            "race_activate_step": race_activate_step,
            "race_target": race_target,
            "intercept_eta_at_arm": intercept_eta_at_arm,
            "blue_score_eta_at_arm": blue_score_eta_at_arm,
            "first_contact_step": first_contact_step,
            "blue_interrupted": blue_interrupted,
            "blue_delay_steps": blue_delay_steps,
            "who_scores_first": who,
            "red_scored_first": int(who == "red"),
            "red_first_score_step": red_first_score_step,
            "blue_first_score_step": blue_first_score_step,
            "race_fired": int(race_activate_step >= 0),
            "steps": steps,
        }
    finally:
        env.close()


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by[str(r["blue_style"])].append(r)
    out: dict[str, Any] = {"styles": {}}
    for style, style_rows in by.items():
        n = len(style_rows)
        out["styles"][style] = {
            "n": n,
            "race_fired": sum(int(r["race_fired"]) for r in style_rows),
            "blue_interrupted": sum(int(r["blue_interrupted"]) for r in style_rows),
            "red_first_score_seeds": sum(int(r["red_scored_first"]) for r in style_rows),
            "mean_blue_delay_steps": sum(float(r["blue_delay_steps"]) for r in style_rows)
            / max(n, 1),
            "contact_eps": sum(int(r["first_contact_step"] >= 0) for r in style_rows),
        }
    rush = out["styles"].get("BLUE_RUSH", {})
    turtle = out["styles"].get("BLUE_TURTLE", {})
    n = max(int(rush.get("n", 8)), 1)
    out["gates"] = {
        "rush_blue_interrupted_ge_5_8": int(rush.get("blue_interrupted", 0))
        >= max(5, (5 * n) // 8),
        "rush_red_first_ge_5_8": int(rush.get("red_first_score_seeds", 0))
        >= max(5, (5 * n) // 8),
        "turtle_race_le_1_8": int(turtle.get("race_fired", 99))
        <= max(1, (1 * n) // 8),
    }
    out["gates_pass"] = all(out["gates"].values())
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=8)
    p.add_argument("--base-seed", type=int, default=611001)
    p.add_argument("--map", default=DEFAULT_MAP)
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-decision-steps", type=int, default=240)
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args()

    _BTRedMixin._OP6_EXTRACTION_ENABLED = True
    _BTRedMixin._OP6_PREENGAGE_ENABLED = True  # keep instrumentation
    _BTRedMixin._OP6_RACE_DENIAL_ENABLED = True

    if args.out_dir is None:
        args.out_dir = (
            PROJECT_ROOT / "artifacts" / "op6_race_denial_dev36_on_map_a"
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    total = len(BLUE_STYLES) * int(args.episodes)
    done_n = 0
    for style in BLUE_STYLES:
        for i in range(int(args.episodes)):
            seed = _episode_seed(
                int(args.base_seed), red_index=0, map_index=0, episode_index=i
            )
            row = _run_episode(
                blue_style=style,
                episode_index=i,
                episode_seed=seed,
                map_name=args.map,
                max_decision_steps=int(args.max_decision_steps),
                device=args.device,
            )
            rows.append(row)
            done_n += 1
            print(
                f"[{done_n}/{total}] {style} ep{i} "
                f"race={row['race_fired']}@ {row['race_activate_step']} "
                f"contact={row['first_contact_step']} "
                f"interrupt={row['blue_interrupted']} "
                f"who={row['who_scores_first']}",
                flush=True,
            )

    with (args.out_dir / "episode_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summary = _summarize(rows)
    summary["base_seed"] = int(args.base_seed)
    summary["map"] = args.map
    (args.out_dir / "race_denial_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0 if summary["gates_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
