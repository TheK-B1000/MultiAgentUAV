#!/usr/bin/env python3
"""OP6 pre-pickup screener race diagnostic (map_a) — RUSH + TURTLE control.

Records the actual score race against BLUE_RUSH V3 and checks that the
narrow pre-pickup gate is frequent on RUSH / rare on TURTLE (home anchor).

Per RUSH episode:
  red_pickup_step, blue_first_score_step, red_screener_engage_step,
  red_score_step, pickup_to_score, screener_travel_at_pickup,
  who_scores_first, first_score_margin_steps

Does not launch a payoff matrix. Pair with offense-competence gates.
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
from gpu_env._core._bt_red import ROLE_ESCORT, _BTRedMixin  # noqa: E402

BLUE_STYLES = ("BLUE_RUSH", "BLUE_TURTLE")
RED_STYLE = "OP6_IMMEDIATE_DUAL_RUSH"
DEFAULT_MAP = "map_a"


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

        prev_red_carry = bool(core.red_carrying[0].any().item())
        prev_pre_act = int(core.bt_op6_preengage_activations[0].item())

        red_pickup_step = -1
        blue_first_score_step = -1
        red_score_step = -1
        screener_engage_step = -1
        first_contact_step = -1
        predicted_screener_arrival = float("nan")
        predicted_blue_arrival = float("nan")
        screener_travel_at_pickup = float("nan")
        preengage_fired = 0
        preengage_active_steps = 0
        steps = 0
        last_info: dict[str, Any] = {}
        # Nominal step speeds for ETA proxies (scripted aquaticus agents).
        speed = 0.35

        while steps < max_decision_steps:
            action = _zero_action(env)
            env.step_async(action)
            _, _, done, infos = env.step_wait()
            last_info = infos[0] if infos else {}
            sim = steps + 1

            red_carry = bool(core.red_carrying[0].any().item())
            ep_res = last_info.get("episode_result", last_info)
            blue_score_now = int(ep_res.get("blue_score", core.blue_score[0].item()))
            red_score_now = int(ep_res.get("red_score", core.red_score[0].item()))

            pre_ticks = int(core.bt_op6_preengage_ticks[0].item())
            pre_act = int(core.bt_op6_preengage_activations[0].item())
            if pre_act > prev_pre_act:
                preengage_fired = 1
                if screener_engage_step < 0:
                    screener_engage_step = sim
                    # Predicted arrivals at engage time (corridor meet / blue-to-path).
                    scr_i = 0
                    for j in range(int(core.Nr)):
                        if int(core.bt_red_role[0, j].item()) == ROLE_ESCORT:
                            scr_i = j
                            break
                    th = int(core.bt_op6_extract_screener_threat[0].item())
                    if th < 0:
                        th = 0
                    hx = float(core.red_flag_home[0, 0].item())
                    hy = float(core.red_flag_home[0, 1].item())
                    fx = float(core.blue_flag_pos[0, 0].item())
                    fy = float(core.blue_flag_pos[0, 1].item())
                    # Projected intercept on flag→home corridor (same helper).
                    ix, iy = core._bt_op6_projected_intercept_xy(
                        {
                            "idx_env": torch.arange(core.B, device=core.device),
                            "red_flag_home": core.red_flag_home,
                        },
                        torch.tensor([fx], device=core.device),
                        torch.tensor([fy], device=core.device),
                        torch.tensor([th], device=core.device, dtype=torch.int64),
                        float(max(0, core.cols - 1)),
                        float(max(0, core.rows - 1)),
                    )
                    sx = float(core.red_x[0, scr_i].item())
                    sy = float(core.red_y[0, scr_i].item())
                    d_scr = math.sqrt(
                        (sx - float(ix[0].item())) ** 2
                        + (sy - float(iy[0].item())) ** 2
                    )
                    bx = float(core.blue_x[0, th].item())
                    by = float(core.blue_y[0, th].item())
                    # Blue ETA to corridor projection near intercept.
                    vx, vy = hx - fx, hy - fy
                    vv = vx * vx + vy * vy + 1e-8
                    t = ((bx - fx) * vx + (by - fy) * vy) / vv
                    px, py = fx + t * vx, fy + t * vy
                    d_blue = math.sqrt((bx - px) ** 2 + (by - py) ** 2)
                    predicted_screener_arrival = sim + d_scr / speed
                    predicted_blue_arrival = sim + d_blue / speed
            if pre_ticks > 0:
                preengage_active_steps += 1
            if (
                screener_engage_step < 0
                and int(core.bt_op6_extract_ticks[0].item()) > 0
                and ROLE_ESCORT in {
                    int(core.bt_red_role[0, 0].item()),
                    int(core.bt_red_role[0, 1].item()),
                }
            ):
                screener_engage_step = sim

            if red_carry and not prev_red_carry and red_pickup_step < 0:
                red_pickup_step = sim
                carr = int(torch.argmax(core.red_carrying[0].to(torch.int64)).item())
                scr = 1 - carr
                scr_x = float(core.red_x[0, scr].item())
                scr_y = float(core.red_y[0, scr].item())
                th = int(core.bt_op6_extract_screener_threat[0].item())
                if th < 0:
                    dx = core.blue_x[0] - scr_x
                    dy = core.blue_y[0] - scr_y
                    dist = torch.sqrt(dx * dx + dy * dy + 1e-8)
                    dist = torch.where(core.blue_alive[0], dist, dist.new_full((), 1e9))
                    th = int(torch.argmin(dist).item())
                tx = float(core.blue_x[0, th].item())
                ty = float(core.blue_y[0, th].item())
                screener_travel_at_pickup = math.sqrt(
                    (scr_x - tx) ** 2 + (scr_y - ty) ** 2
                )

            # First contact: locked screener threat within tag radius of carrier.
            if first_contact_step < 0 and red_carry:
                carr = int(torch.argmax(core.red_carrying[0].to(torch.int64)).item())
                th = int(core.bt_op6_extract_screener_threat[0].item())
                if th >= 0 and bool(core.blue_alive[0, th].item()):
                    d = math.sqrt(
                        (
                            float(core.blue_x[0, th].item())
                            - float(core.red_x[0, carr].item())
                        )
                        ** 2
                        + (
                            float(core.blue_y[0, th].item())
                            - float(core.red_y[0, carr].item())
                        )
                        ** 2
                    )
                    if d <= 2.5:
                        first_contact_step = sim

            if red_score_step < 0 and red_score_now > 0:
                red_score_step = sim
            if blue_first_score_step < 0 and blue_score_now > 0:
                blue_first_score_step = sim

            prev_red_carry = red_carry
            prev_pre_act = pre_act
            steps += 1
            if bool(done.any() if hasattr(done, "any") else done):
                break

        pickup_to_score = (
            (red_score_step - red_pickup_step)
            if red_pickup_step >= 0 and red_score_step >= 0
            else -1
        )
        engage_lead_time = (
            (red_pickup_step - screener_engage_step)
            if red_pickup_step >= 0 and screener_engage_step >= 0
            else None
        )
        if red_score_step >= 0 and (
            blue_first_score_step < 0 or red_score_step <= blue_first_score_step
        ):
            who = "red"
            margin = (
                blue_first_score_step - red_score_step
                if blue_first_score_step >= 0
                else 999
            )
        elif blue_first_score_step >= 0:
            who = "blue"
            margin = (
                red_score_step - blue_first_score_step
                if red_score_step >= 0
                else 999
            )
        else:
            who = "neither"
            margin = -1

        pre_before_pickup = int(
            screener_engage_step >= 0
            and red_pickup_step >= 0
            and screener_engage_step <= red_pickup_step
            and preengage_fired
        )
        screener_won_race = int(
            predicted_screener_arrival == predicted_screener_arrival
            and predicted_blue_arrival == predicted_blue_arrival
            and predicted_screener_arrival <= predicted_blue_arrival
        )

        return {
            "blue_style": blue_style,
            "red_style": RED_STYLE,
            "map": map_name,
            "episode_index": episode_index,
            "episode_seed": episode_seed,
            "screener_engage_step": screener_engage_step,
            "red_pickup_step": red_pickup_step,
            "engage_lead_time": engage_lead_time,
            "predicted_screener_arrival": predicted_screener_arrival,
            "predicted_blue_arrival": predicted_blue_arrival,
            "screener_won_predicted_race": screener_won_race,
            "actual_first_contact": first_contact_step,
            "red_score_step": red_score_step,
            "blue_score_step": blue_first_score_step,
            "blue_first_score_step": blue_first_score_step,
            "red_screener_engage_step": screener_engage_step,
            "pickup_to_score": pickup_to_score,
            "screener_travel_at_pickup": screener_travel_at_pickup,
            "who_scores_first": who,
            "first_score_margin_steps": margin,
            "preengage_fired": preengage_fired,
            "preengage_before_pickup": pre_before_pickup,
            "preengage_active_steps": preengage_active_steps,
            "red_scored_first": int(who == "red"),
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
        fired = sum(int(r["preengage_fired"]) for r in style_rows)
        before = sum(int(r["preengage_before_pickup"]) for r in style_rows)
        red_first = sum(int(r["red_scored_first"]) for r in style_rows)
        pickups = [r for r in style_rows if int(r["red_pickup_step"]) >= 0]
        travels = [
            float(r["screener_travel_at_pickup"])
            for r in pickups
            if r["screener_travel_at_pickup"] == r["screener_travel_at_pickup"]
        ]
        p2s = [int(r["pickup_to_score"]) for r in pickups if int(r["pickup_to_score"]) >= 0]
        leads = [
            int(r["engage_lead_time"])
            for r in style_rows
            if r.get("engage_lead_time") is not None
        ]
        won = sum(int(r.get("screener_won_predicted_race", 0)) for r in style_rows)
        out["styles"][style] = {
            "n": n,
            "preengage_fired": fired,
            "preengage_before_pickup": before,
            "preengage_frac": fired / max(n, 1),
            "red_first_score_seeds": red_first,
            "mean_engage_lead_time": (sum(leads) / len(leads) if leads else None),
            "screener_won_predicted_race": won,
            "mean_screener_travel_at_pickup": (
                sum(travels) / len(travels) if travels else float("nan")
            ),
            "mean_pickup_to_score": (sum(p2s) / len(p2s) if p2s else float("nan")),
            "n_pickups": len(pickups),
        }
    rush = out["styles"].get("BLUE_RUSH", {})
    turtle = out["styles"].get("BLUE_TURTLE", {})
    out["safety"] = {
        "rush_preengage_frequent": int(rush.get("preengage_fired", 0)) >= 4,
        "turtle_preengage_rare": int(turtle.get("preengage_fired", 0)) <= 2,
    }
    out["safety_pass"] = all(out["safety"].values())
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=8)
    p.add_argument("--base-seed", type=int, default=701001)
    p.add_argument("--map", default=DEFAULT_MAP)
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-decision-steps", type=int, default=240)
    p.add_argument(
        "--preengage",
        choices=("on", "off"),
        default="on",
        help="Toggle OP6 pre-pickup screener (extraction stays on).",
    )
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args()

    _BTRedMixin._OP6_EXTRACTION_ENABLED = True
    _BTRedMixin._OP6_PREENGAGE_ENABLED = args.preengage == "on"

    if args.out_dir is None:
        args.out_dir = (
            PROJECT_ROOT
            / "artifacts"
            / f"op6_preengage_race_dev35_{args.preengage}_map_a"
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
                f"[{done_n}/{total}] {style} ep{i} seed={seed} "
                f"who={row['who_scores_first']} "
                f"pre={row['preengage_fired']} "
                f"pick={row['red_pickup_step']} "
                f"engage={row['screener_engage_step']} "
                f"lead={row['engage_lead_time']} "
                f"red_sc={row['red_score_step']} "
                f"blue_sc={row['blue_score_step']}",
                flush=True,
            )

    csv_path = args.out_dir / "episode_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summary = _summarize(rows)
    summary["base_seed"] = int(args.base_seed)
    summary["preengage"] = args.preengage
    summary["map"] = args.map
    (args.out_dir / "race_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0 if summary["safety_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
