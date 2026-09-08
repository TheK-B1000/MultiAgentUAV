#!/usr/bin/env python3
"""OP6 mutual-carry denial feasibility (map_a) — diagnose before implement.

Replays the same seeds as the rejected dev35 race diagnostic and, at every
simultaneous-carry step (both sides hold a flag), records:

  red non-carrier ETA to blue carrier (tag contact)
  blue carrier ETA to score
  red carrier ETA to score
  distance and closing speed

Viable when red interceptor ETA < blue carrier score ETA often enough that
a 1–3 step delay can flip the RUSH first-score race.

Does not change BT routing. Preengage instrumentation may remain ON (status
quo after rejected-dev35 mechanism) so the race geometry matches the traces
we are explaining.
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
from gpu_env._core._bt_red import _BTRedMixin  # noqa: E402

BLUE_STYLES = ("BLUE_RUSH", "BLUE_TURTLE")
RED_STYLE = "OP6_IMMEDIATE_DUAL_RUSH"
DEFAULT_MAP = "map_a"
CAP_R = 1.2
NEAR_BHOME = 6.0


def _eta_steps(dist: float, speed: float, contact_r: float = 0.0) -> float:
    rem = max(0.0, float(dist) - float(contact_r))
    spd = max(float(speed), 1e-6)
    return rem / spd


def _blue_home_abandoned(core) -> bool:
    mid = float(core.cols) * 0.5
    hx = float(core.blue_flag_home[0, 0].item())
    alive = core.blue_alive[0] & (~core.blue_carrying[0])
    near = (torch.abs(core.blue_x[0] - hx) <= NEAR_BHOME) & (core.blue_x[0] < mid)
    return not bool((alive & near).any().item())


def _run_episode(
    *,
    blue_style: str,
    episode_index: int,
    episode_seed: int,
    map_name: str,
    max_decision_steps: int,
    device: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
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
        tag_r = float(getattr(core.cfg, "tag_range_cells", 2.5))
        rhome_x = float(core.red_flag_home[0, 0].item())
        rhome_y = float(core.red_flag_home[0, 1].item())
        bhome_x = float(core.blue_flag_home[0, 0].item())
        bhome_y = float(core.blue_flag_home[0, 1].item())

        events: list[dict[str, Any]] = []
        first_mutual_step = -1
        mutual_steps = 0
        viable_steps = 0
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

            red_carry = core.red_carrying[0] & core.red_alive[0] & (~core.red_tagged[0])
            blue_carry = core.blue_carrying[0] & core.blue_alive[0] & (~core.blue_tagged[0])
            mutual = bool(red_carry.any().item()) and bool(blue_carry.any().item())
            abandoned = _blue_home_abandoned(core)

            if mutual and abandoned:
                mutual_steps += 1
                if first_mutual_step < 0:
                    first_mutual_step = sim

                rc = int(torch.argmax(red_carry.to(torch.int64)).item())
                bc = int(torch.argmax(blue_carry.to(torch.int64)).item())
                # Non-carrier red: prefer alive/untagged partner; else self.
                candidates = []
                for i in range(int(core.Nr)):
                    if i == rc:
                        continue
                    if bool(core.red_alive[0, i].item()) and (
                        not bool(core.red_tagged[0, i].item())
                    ):
                        candidates.append(i)
                ri = candidates[0] if candidates else rc

                rx = float(core.red_x[0, ri].item())
                ry = float(core.red_y[0, ri].item())
                rcx = float(core.red_x[0, rc].item())
                rcy = float(core.red_y[0, rc].item())
                bcx = float(core.blue_x[0, bc].item())
                bcy = float(core.blue_y[0, bc].item())
                rs = float(core.red_speed[0, ri].item())
                bs = float(core.blue_speed[0, bc].item())
                rcs = float(core.red_speed[0, rc].item())

                dist_int = math.hypot(rx - bcx, ry - bcy)
                dist_blue_home = math.hypot(bcx - bhome_x, bcy - bhome_y)
                dist_red_home = math.hypot(rcx - rhome_x, rcy - rhome_y)

                # Closing speed along interceptor → blue-carrier line.
                # Positive = closing (using current velocities projected).
                ux = (bcx - rx) / max(dist_int, 1e-6)
                uy = (bcy - ry) / max(dist_int, 1e-6)
                # Approximate velocity from heading * speed if available.
                rh = float(core.red_heading[0, ri].item())
                bh = float(core.blue_heading[0, bc].item())
                rvx, rvy = rs * math.cos(rh), rs * math.sin(rh)
                bvx, bvy = bs * math.cos(bh), bs * math.sin(bh)
                closing = (rvx - bvx) * ux + (rvy - bvy) * uy

                # ETA at max speed (optimistic interceptor / blue score race).
                eta_int_max = _eta_steps(dist_int, max_speed, tag_r)
                eta_blue_max = _eta_steps(dist_blue_home, max_speed, CAP_R)
                eta_red_max = _eta_steps(dist_red_home, max_speed, CAP_R)
                # ETA at current speeds (pessimistic if slow).
                eta_int_cur = _eta_steps(dist_int, max(rs, 0.1), tag_r)
                eta_blue_cur = _eta_steps(dist_blue_home, max(bs, 0.1), CAP_R)
                eta_red_cur = _eta_steps(dist_red_home, max(rcs, 0.1), CAP_R)

                viable_max = eta_int_max < eta_blue_max
                if viable_max:
                    viable_steps += 1

                events.append(
                    {
                        "blue_style": blue_style,
                        "episode_index": episode_index,
                        "episode_seed": episode_seed,
                        "step": sim,
                        "interceptor_idx": ri,
                        "blue_carrier_idx": bc,
                        "red_carrier_idx": rc,
                        "dist_interceptor_to_blue_carrier": dist_int,
                        "dist_blue_carrier_to_home": dist_blue_home,
                        "dist_red_carrier_to_home": dist_red_home,
                        "closing_speed": closing,
                        "eta_interceptor_max": eta_int_max,
                        "eta_blue_score_max": eta_blue_max,
                        "eta_red_score_max": eta_red_max,
                        "eta_interceptor_cur": eta_int_cur,
                        "eta_blue_score_cur": eta_blue_cur,
                        "eta_red_score_cur": eta_red_cur,
                        "eta_slack_max": eta_blue_max - eta_int_max,
                        "viable_max": int(viable_max),
                        "red_first_score_step": red_first_score_step,
                        "blue_first_score_step": blue_first_score_step,
                    }
                )

            steps += 1
            if bool(done.any() if hasattr(done, "any") else done):
                break

        who = "neither"
        if red_first_score_step >= 0 and (
            blue_first_score_step < 0 or red_first_score_step <= blue_first_score_step
        ):
            who = "red"
        elif blue_first_score_step >= 0:
            who = "blue"

        summary = {
            "blue_style": blue_style,
            "red_style": RED_STYLE,
            "map": map_name,
            "episode_index": episode_index,
            "episode_seed": episode_seed,
            "first_mutual_step": first_mutual_step,
            "mutual_carry_steps": mutual_steps,
            "viable_mutual_steps": viable_steps,
            "viable_frac": viable_steps / max(mutual_steps, 1),
            "had_mutual_carry": int(mutual_steps > 0),
            "had_viable_window": int(viable_steps > 0),
            "red_scored_first": int(who == "red"),
            "who_scores_first": who,
            "red_first_score_step": red_first_score_step,
            "blue_first_score_step": blue_first_score_step,
            "first_score_margin_steps": (
                abs(red_first_score_step - blue_first_score_step)
                if red_first_score_step >= 0 and blue_first_score_step >= 0
                else -1
            ),
            "steps": steps,
            "max_speed": max_speed,
            "tag_range": tag_r,
        }
        return summary, events
    finally:
        env.close()


def _summarize(
    rows: list[dict[str, Any]], events: list[dict[str, Any]]
) -> dict[str, Any]:
    by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by[str(r["blue_style"])].append(r)
    ev_by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for e in events:
        ev_by[str(e["blue_style"])].append(e)

    out: dict[str, Any] = {"styles": {}, "viability": {}}
    for style, style_rows in by.items():
        n = len(style_rows)
        mutual_eps = sum(int(r["had_mutual_carry"]) for r in style_rows)
        viable_eps = sum(int(r["had_viable_window"]) for r in style_rows)
        red_first = sum(int(r["red_scored_first"]) for r in style_rows)
        evs = ev_by.get(style, [])
        slacks = [float(e["eta_slack_max"]) for e in evs]
        first_slacks = []
        # First mutual step per episode slack.
        seen = set()
        for e in evs:
            key = (e["episode_index"], e["episode_seed"])
            if key in seen:
                continue
            seen.add(key)
            first_slacks.append(float(e["eta_slack_max"]))

        out["styles"][style] = {
            "n": n,
            "episodes_with_mutual_carry": mutual_eps,
            "episodes_with_viable_window": viable_eps,
            "viable_episode_frac": viable_eps / max(n, 1),
            "mutual_episode_frac": mutual_eps / max(n, 1),
            "red_first_score_seeds": red_first,
            "mean_mutual_steps": sum(int(r["mutual_carry_steps"]) for r in style_rows)
            / max(n, 1),
            "n_mutual_events": len(evs),
            "mean_eta_slack_max_all": (
                sum(slacks) / len(slacks) if slacks else float("nan")
            ),
            "mean_eta_slack_max_first_mutual": (
                sum(first_slacks) / len(first_slacks) if first_slacks else float("nan")
            ),
            "frac_mutual_events_viable": (
                sum(int(e["viable_max"]) for e in evs) / max(len(evs), 1)
            ),
        }

    rush = out["styles"].get("BLUE_RUSH", {})
    turtle = out["styles"].get("BLUE_TURTLE", {})
    # Mechanism viable if most RUSH episodes have a window where
    # interceptor ETA < blue score ETA, and TURTLE rarely enters mutual race.
    out["viability"] = {
        "rush_viable_episode_frac_ge_0_5": float(rush.get("viable_episode_frac", 0))
        >= 0.5,
        "rush_first_mutual_mean_slack_positive": float(
            rush.get("mean_eta_slack_max_first_mutual", -1e9)
        )
        > 0.0,
        "turtle_mutual_rare": int(turtle.get("episodes_with_mutual_carry", 99)) <= 2,
    }
    out["viable_to_implement"] = bool(
        out["viability"]["rush_viable_episode_frac_ge_0_5"]
        and out["viability"]["rush_first_mutual_mean_slack_positive"]
    )
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

    # Match rejected-dev35 controller state (instrumentation kept).
    _BTRedMixin._OP6_EXTRACTION_ENABLED = True
    _BTRedMixin._OP6_PREENGAGE_ENABLED = True

    if args.out_dir is None:
        args.out_dir = (
            PROJECT_ROOT / "artifacts" / "op6_mutual_carry_feasibility_dev36_map_a"
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    total = len(BLUE_STYLES) * int(args.episodes)
    done_n = 0
    for style in BLUE_STYLES:
        for i in range(int(args.episodes)):
            seed = _episode_seed(
                int(args.base_seed), red_index=0, map_index=0, episode_index=i
            )
            summary, evs = _run_episode(
                blue_style=style,
                episode_index=i,
                episode_seed=seed,
                map_name=args.map,
                max_decision_steps=int(args.max_decision_steps),
                device=args.device,
            )
            rows.append(summary)
            events.extend(evs)
            done_n += 1
            print(
                f"[{done_n}/{total}] {style} ep{i} seed={seed} "
                f"mutual={summary['mutual_carry_steps']} "
                f"viable={summary['viable_mutual_steps']} "
                f"who={summary['who_scores_first']} "
                f"margin={summary['first_score_margin_steps']}",
                flush=True,
            )

    with (args.out_dir / "episode_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    if events:
        with (args.out_dir / "mutual_carry_events.csv").open(
            "w", newline="", encoding="utf-8"
        ) as f:
            w = csv.DictWriter(f, fieldnames=list(events[0].keys()))
            w.writeheader()
            w.writerows(events)

    summary = _summarize(rows, events)
    summary["base_seed"] = int(args.base_seed)
    summary["map"] = args.map
    summary["note"] = (
        "viable_max = eta_interceptor_max < eta_blue_score_max at abandoned "
        "simultaneous-carry steps; implement denial only if viable_to_implement"
    )
    (args.out_dir / "feasibility_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0 if summary["viable_to_implement"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
