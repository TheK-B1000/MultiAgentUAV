#!/usr/bin/env python3
"""OP6 offensive competence micro-gate (map_a).

Recovery tuning is paused. This audit asks whether OP6's dual assault
actually picks up and scores against empty-home blues, and whether TURTLE
stops those assaults and converts.

Targets (8 paired seeds):
  red first-score: RUSH/SPLIT/ESCORT ≥5/8 each; TURTLE ≤2/8
  TURTLE counter-score after stopped assault: ≥5/8
"""
from __future__ import annotations

import argparse
import csv
import json
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

BLUE_STYLES = (
    "BLUE_RUSH",
    "BLUE_SPLIT",
    "BLUE_ESCORT",
    "BLUE_TURTLE",
)
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

        midline = float(core.cols) * 0.5
        home_x = float(core.blue_flag_home[0, 0].item())

        red_pickups = 0
        red_failed_returns = 0  # lost carry without score
        blue_home_occ_sum = 0.0
        blue_home_occ_n = 0
        red_first_score_step = -1
        blue_first_score_step = -1
        red_first_pickup_step = -1
        assault_stopped = 0
        turtle_anchor_at_stop = 0
        turtle_counter_score_after_stop = 0
        prev_red_carry = False
        prev_red_score = 0
        prev_blue_score = 0
        stop_seen = False
        steps = 0
        last_info: dict[str, Any] = {}

        for _ in range(int(max_decision_steps) + 5):
            sim = int(core.sim_step_count[0].item())
            red_carry = bool(core.red_carrying[0].any().item())
            if red_carry and not prev_red_carry:
                red_pickups += 1
                if red_first_pickup_step < 0:
                    red_first_pickup_step = sim

            # Blue home occupancy: fraction of alive blues near own flag.
            alive_b = core.blue_alive[0]
            if bool(alive_b.any().item()):
                near = alive_b & (torch.abs(core.blue_x[0] - home_x) <= 6.0) & (
                    core.blue_x[0] < midline
                )
                blue_home_occ_sum += float(near.sum().item()) / float(alive_b.sum().item())
                blue_home_occ_n += 1

            # Assault stop: red newly tagged on blue half while dual-committed.
            newly_tagged = (~getattr(core, "_off_prev_tagged", core.red_tagged.clone())) & core.red_tagged
            on_blue = core.red_x[0] < midline
            both = int((core.red_alive[0] & on_blue).sum().item()) >= 2
            if bool((newly_tagged[0] & on_blue).any().item()) and both:
                assault_stopped += 1
                stop_seen = True
                # Non-carrier blue near home = anchor present at stop.
                anchor = (
                    core.blue_alive[0]
                    & (~core.blue_carrying[0])
                    & (core.blue_x[0] < midline)
                    & (torch.abs(core.blue_x[0] - home_x) <= 6.0)
                )
                if bool(anchor.any().item()):
                    turtle_anchor_at_stop += 1

            core._off_prev_tagged = core.red_tagged.detach().clone()

            action = _zero_action(env)
            env.step_async(action)
            _, _, done, infos = env.step_wait()
            last_info = infos[0] if infos else {}
            ep_res = last_info.get("episode_result", last_info)
            blue_score_now = int(ep_res.get("blue_score", core.blue_score[0].item()))
            red_score_now = int(ep_res.get("red_score", core.red_score[0].item()))

            if prev_red_carry and (not red_carry) and red_score_now == prev_red_score:
                red_failed_returns += 1

            if red_first_score_step < 0 and red_score_now > 0:
                red_first_score_step = sim
            if blue_first_score_step < 0 and blue_score_now > 0:
                blue_first_score_step = sim
                if (
                    blue_style == "BLUE_TURTLE"
                    and stop_seen
                    and blue_score_now > prev_blue_score
                ):
                    turtle_counter_score_after_stop = 1

            if (
                blue_style == "BLUE_TURTLE"
                and stop_seen
                and blue_score_now > prev_blue_score
                and turtle_counter_score_after_stop == 0
            ):
                turtle_counter_score_after_stop = 1

            prev_red_carry = red_carry
            prev_red_score = red_score_now
            prev_blue_score = blue_score_now
            steps += 1
            if bool(done.any()):
                break

        ep_res = last_info.get("episode_result", last_info) if last_info else {}
        blue_score = int(ep_res.get("blue_score", core.blue_score[0].item()))
        red_score = int(ep_res.get("red_score", core.red_score[0].item()))
        return {
            "blue_style": blue_style,
            "red_style": RED_STYLE,
            "map": map_name,
            "episode_index": episode_index,
            "episode_seed": episode_seed,
            "red_pickups": red_pickups,
            "red_failed_returns": red_failed_returns,
            "blue_home_occupancy": blue_home_occ_sum / max(blue_home_occ_n, 1),
            "red_first_pickup_step": red_first_pickup_step,
            "red_first_score_step": red_first_score_step,
            "blue_first_score_step": blue_first_score_step,
            "assault_stopped": assault_stopped,
            "turtle_anchor_at_stop": turtle_anchor_at_stop,
            "turtle_counter_score_after_stop": turtle_counter_score_after_stop,
            "red_scored_first": int(
                red_first_score_step >= 0
                and (
                    blue_first_score_step < 0
                    or red_first_score_step <= blue_first_score_step
                )
            ),
            "blue_score": blue_score,
            "red_score": red_score,
            "win_margin": blue_score - red_score,
            "steps": steps,
        }
    finally:
        env.close()


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return sum(float(r[key]) for r in rows) / max(len(rows), 1)


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_style: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_style[str(row["blue_style"])].append(row)

    summary: dict[str, Any] = {"n_episodes_per_style": {}, "styles": {}}
    for style, style_rows in by_style.items():
        n = len(style_rows)
        summary["n_episodes_per_style"][style] = n
        first_score_seeds = sum(int(r["red_scored_first"]) for r in style_rows)
        summary["styles"][style] = {
            "mean_red_pickups": _mean(style_rows, "red_pickups"),
            "mean_red_failed_returns": _mean(style_rows, "red_failed_returns"),
            "mean_blue_home_occupancy": _mean(style_rows, "blue_home_occupancy"),
            "frac_red_scored_first": first_score_seeds / max(n, 1),
            "red_first_score_seeds": first_score_seeds,
            "mean_assault_stopped": _mean(style_rows, "assault_stopped"),
            "mean_turtle_anchor_at_stop": _mean(style_rows, "turtle_anchor_at_stop"),
            "turtle_counter_score_seeds": sum(
                int(r["turtle_counter_score_after_stop"]) for r in style_rows
            ),
            "mean_win_margin": _mean(style_rows, "win_margin"),
        }

    rush = summary["styles"].get("BLUE_RUSH", {})
    split = summary["styles"].get("BLUE_SPLIT", {})
    escort = summary["styles"].get("BLUE_ESCORT", {})
    turtle = summary["styles"].get("BLUE_TURTLE", {})
    n = max(summary["n_episodes_per_style"].get("BLUE_RUSH", 8), 1)

    def _ge5(style_sum: dict[str, Any]) -> bool:
        return int(style_sum.get("red_first_score_seeds", 0)) >= max(5, (5 * n) // 8)

    summary["micro_gates"] = {
        "rush_red_first_score_ge_5_8": _ge5(rush),
        "split_red_first_score_ge_5_8": int(split.get("red_first_score_seeds", 0))
        >= max(6, (6 * n) // 8),
        "escort_red_first_score_ge_5_8": _ge5(escort),
        "turtle_red_first_score_le_2_8": int(turtle.get("red_first_score_seeds", 99))
        <= max(2, (2 * n) // 8),
        "turtle_counter_score_ge_5_8": int(turtle.get("turtle_counter_score_seeds", 0))
        >= max(6, (6 * n) // 8),
    }
    # Optional baseline for failed-return reduction vs a prior OFF/ON run.
    summary["rush_failed_returns"] = float(rush.get("mean_red_failed_returns", 0.0))
    summary["micro_gates_pass"] = all(summary["micro_gates"].values())
    summary["rush_minus_split"] = float(rush.get("mean_win_margin", 0.0)) - float(
        split.get("mean_win_margin", 0.0)
    )
    summary["rush_minus_turtle"] = float(rush.get("mean_win_margin", 0.0)) - float(
        turtle.get("mean_win_margin", 0.0)
    )
    summary["rush_minus_escort"] = float(rush.get("mean_win_margin", 0.0)) - float(
        escort.get("mean_win_margin", 0.0)
    )
    return summary


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=8)
    p.add_argument("--base-seed", type=int, default=701001)
    p.add_argument("--map", default=DEFAULT_MAP)
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-decision-steps", type=int, default=240)
    p.add_argument(
        "--extraction",
        choices=("on", "off"),
        default="on",
        help="OP6 post-pickup extraction support toggle (OFF/ON paired gates).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
    )
    args = p.parse_args()
    _BTRedMixin._OP6_EXTRACTION_ENABLED = args.extraction == "on"
    if args.out_dir is None:
        args.out_dir = (
            PROJECT_ROOT
            / "artifacts"
            / f"op6_offense_competence_dev29_extract_{args.extraction}_map_a"
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    total = len(BLUE_STYLES) * int(args.episodes)
    done_n = 0
    for ep in range(int(args.episodes)):
        seed = _episode_seed(int(args.base_seed), red_index=0, map_index=0, episode_index=ep)
        for style in BLUE_STYLES:
            row = _run_episode(
                blue_style=style,
                episode_index=ep,
                episode_seed=seed,
                map_name=str(args.map),
                max_decision_steps=int(args.max_decision_steps),
                device=str(args.device),
            )
            rows.append(row)
            done_n += 1
            print(
                f"[{done_n}/{total}] {style} ep={ep} "
                f"pickups={row['red_pickups']} "
                f"red_1st={row['red_scored_first']} "
                f"stops={row['assault_stopped']} "
                f"margin={row['win_margin']}",
                flush=True,
            )

    summary = _summarize(rows)
    csv_path = args.out_dir / "episode_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary_path = args.out_dir / "offense_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["styles"], indent=2))
    print(json.dumps(summary["micro_gates"], indent=2))
    print(
        f"rush-split={summary['rush_minus_split']} "
        f"rush-turtle={summary['rush_minus_turtle']} "
        f"rush-escort={summary['rush_minus_escort']}"
    )
    print(f"micro_gates_pass={summary['micro_gates_pass']}")
    print(f"wrote {csv_path}")
    print(f"wrote {summary_path}")
    return 0 if summary["micro_gates_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
