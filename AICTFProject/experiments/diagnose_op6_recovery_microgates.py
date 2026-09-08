#!/usr/bin/env python3
"""Micro-gate diagnostic for OP6 failed-assault recovery (Contract B).

Measures recovery activation and whether TURTLE converts during the window,
without using blue-style ID as a red trigger. Legal state only.
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
    duration = int(_BTRedMixin._OP6_RECOVERY_DURATION)
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
        red_flag_home_x = float(core.red_flag_home[0, 0].item())

        failed_incursions = 0
        recovery_activations = 0
        recovery_active_steps = 0
        recovery_duration_sum = 0
        home_occ_sum = 0.0
        home_occ_n = 0
        blue_pickups_during_recovery = 0
        blue_scores_during_recovery = 0
        red_first_score_step = -1
        blue_first_score_step = -1
        blue_reached_red_flag_during_recovery = 0
        prev_blue_score = 0
        prev_blue_carry = False
        prev_failed = 0
        prev_activations = 0
        steps = 0
        last_info: dict[str, Any] = {}

        for _ in range(int(max_decision_steps) + 5):
            sim = int(core.sim_step_count[0].item())
            recovery_ticks = core.bt_op6_recovery_ticks[0]
            recovery_any = bool((recovery_ticks > 0).any().item())
            cur_failed = int(core.bt_op6_failed_incursions[0].item())
            cur_activations = int(core.bt_op6_recovery_activations[0].item())
            if cur_failed > prev_failed:
                failed_incursions = cur_failed
            if cur_activations > prev_activations:
                recovery_activations = cur_activations
            prev_failed = cur_failed
            prev_activations = cur_activations
            if recovery_any:
                recovery_active_steps += 1
                recovery_duration_sum += int(recovery_ticks.max().item())
                # Blue near red flag during recovery → counter reach.
                blue_near = (
                    (core.blue_alive[0])
                    & (torch.abs(core.blue_x[0] - core.red_flag_pos[0, 0]) < 2.5)
                    & (torch.abs(core.blue_y[0] - core.red_flag_pos[0, 1]) < 2.5)
                )
                if bool(blue_near.any().item()):
                    blue_reached_red_flag_during_recovery = 1

            # Red home occupancy: fraction of alive reds on red half near home.
            alive = core.red_alive[0]
            if bool(alive.any().item()):
                on_home = alive & (core.red_x[0] > midline) & (
                    torch.abs(core.red_x[0] - red_flag_home_x) < 6.0
                )
                home_occ_sum += float(on_home.sum().item()) / float(alive.sum().item())
                home_occ_n += 1

            blue_carry = bool(core.blue_carrying[0].any().item())
            if recovery_any and blue_carry and not prev_blue_carry:
                blue_pickups_during_recovery += 1

            action = _zero_action(env)
            env.step_async(action)
            _, _, done, infos = env.step_wait()
            last_info = infos[0] if infos else {}
            ep_res = last_info.get("episode_result", last_info)
            blue_score_now = int(ep_res.get("blue_score", core.blue_score[0].item()))
            red_score_now = int(ep_res.get("red_score", core.red_score[0].item()))
            if recovery_any and blue_score_now > prev_blue_score:
                blue_scores_during_recovery += 1
            if red_first_score_step < 0 and red_score_now > 0:
                red_first_score_step = sim
            if blue_first_score_step < 0 and blue_score_now > 0:
                blue_first_score_step = sim

            prev_blue_score = blue_score_now
            prev_blue_carry = blue_carry
            steps += 1
            if bool(done.any()):
                break

        # Peak BT counters across the episode (episode-end reset may zero them).
        failed_incursions = max(
            failed_incursions, int(core.bt_op6_failed_incursions[0].item())
        )
        recovery_activations = max(
            recovery_activations,
            int(core.bt_op6_recovery_activations[0].item()),
        )
        recovery_active_steps = max(
            recovery_active_steps,
            int(core.bt_op6_recovery_active_steps[0].item()),
        )

        ep_res = last_info.get("episode_result", last_info) if last_info else {}
        blue_score = int(ep_res.get("blue_score", core.blue_score[0].item()))
        red_score = int(ep_res.get("red_score", core.red_score[0].item()))
        mean_home_occ = home_occ_sum / max(home_occ_n, 1)
        mean_rec_dur = (
            recovery_duration_sum / max(recovery_active_steps, 1)
            if recovery_active_steps > 0
            else 0.0
        )

        return {
            "blue_style": blue_style,
            "red_style": RED_STYLE,
            "map": map_name,
            "episode_index": episode_index,
            "episode_seed": episode_seed,
            "recovery_duration_cfg": duration,
            "failed_incursions": failed_incursions,
            "recovery_activations": recovery_activations,
            "recovery_active_steps": recovery_active_steps,
            "mean_recovery_ticks_when_active": mean_rec_dur,
            "op6_home_occupancy": mean_home_occ,
            "blue_pickups_during_recovery": blue_pickups_during_recovery,
            "blue_scores_during_recovery": blue_scores_during_recovery,
            "blue_reached_red_flag_during_recovery": blue_reached_red_flag_during_recovery,
            "red_first_score_step": red_first_score_step,
            "blue_first_score_step": blue_first_score_step,
            "blue_score": blue_score,
            "red_score": red_score,
            "win_margin": blue_score - red_score,
            "steps": steps,
            "turtle_scored_in_recovery": int(
                blue_style == "BLUE_TURTLE" and blue_scores_during_recovery > 0
            ),
            "red_scored_first": int(
                red_first_score_step >= 0
                and (blue_first_score_step < 0 or red_first_score_step <= blue_first_score_step)
            ),
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
        summary["styles"][style] = {
            "mean_failed_incursions": _mean(style_rows, "failed_incursions"),
            "mean_recovery_activations": _mean(style_rows, "recovery_activations"),
            "mean_recovery_active_steps": _mean(style_rows, "recovery_active_steps"),
            "mean_home_occupancy": _mean(style_rows, "op6_home_occupancy"),
            "mean_blue_pickups_during_recovery": _mean(
                style_rows, "blue_pickups_during_recovery"
            ),
            "mean_blue_scores_during_recovery": _mean(
                style_rows, "blue_scores_during_recovery"
            ),
            "frac_blue_reached_flag_during_recovery": _mean(
                style_rows, "blue_reached_red_flag_during_recovery"
            ),
            "frac_red_scored_first": _mean(style_rows, "red_scored_first"),
            "mean_win_margin": _mean(style_rows, "win_margin"),
            "turtle_recovery_score_seeds": sum(
                int(r["turtle_scored_in_recovery"]) for r in style_rows
            ),
        }

    turtle = summary["styles"].get("BLUE_TURTLE", {})
    split = summary["styles"].get("BLUE_SPLIT", {})
    rush = summary["styles"].get("BLUE_RUSH", {})
    escort = summary["styles"].get("BLUE_ESCORT", {})
    n_turtle = summary["n_episodes_per_style"].get("BLUE_TURTLE", 0)

    turtle_margin = float(turtle.get("mean_win_margin", -99.0))
    others = [
        float(split.get("mean_win_margin", -99.0)),
        float(rush.get("mean_win_margin", -99.0)),
        float(escort.get("mean_win_margin", -99.0)),
    ]
    summary["micro_gates"] = {
        "turtle_recovery_activates": float(turtle.get("mean_recovery_activations", 0.0))
        >= 0.5,
        "turtle_recovery_window_present": float(
            turtle.get("mean_recovery_active_steps", 0.0)
        )
        >= 10.0,
        "turtle_counter_reaches_or_scores": (
            float(turtle.get("frac_blue_reached_flag_during_recovery", 0.0)) >= 0.25
            or float(turtle.get("mean_blue_scores_during_recovery", 0.0)) > 0.0
        ),
        "turtle_recovery_score_seeds_at_least_half": (
            int(turtle.get("turtle_recovery_score_seeds", 0)) >= max(1, n_turtle // 2)
        ),
        "rush_split_recovery_rarer_than_turtle": (
            float(rush.get("mean_recovery_activations", 99.0))
            + float(split.get("mean_recovery_activations", 99.0))
        )
        / 2.0
        < float(turtle.get("mean_recovery_activations", 0.0)) * 0.75,
        "turtle_uniquely_best_margin": turtle_margin > max(others) + 1e-9,
        "turtle_minus_split_ge_0p25": turtle_margin - float(split.get("mean_win_margin", 0.0))
        >= 0.25,
        "split_rush_escort_red_often_scores_first": (
            float(split.get("frac_red_scored_first", 0.0))
            + float(rush.get("frac_red_scored_first", 0.0))
            + float(escort.get("frac_red_scored_first", 0.0))
        )
        / 3.0
        >= 0.4,
    }
    summary["micro_gates_pass"] = all(summary["micro_gates"].values())
    summary["turtle_minus_split"] = turtle_margin - float(
        split.get("mean_win_margin", 0.0)
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
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "op6_recovery_microgates_dev21",
    )
    args = p.parse_args()
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
                f"rec_act={row['recovery_activations']} "
                f"rec_steps={row['recovery_active_steps']} "
                f"b_score_rec={row['blue_scores_during_recovery']} "
                f"margin={row['win_margin']}",
                flush=True,
            )

    summary = _summarize(rows)
    csv_path = args.out_dir / "episode_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary_path = args.out_dir / "microgate_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["styles"], indent=2))
    print(json.dumps(summary["micro_gates"], indent=2))
    print(f"turtle_minus_split={summary['turtle_minus_split']}")
    print(f"micro_gates_pass={summary['micro_gates_pass']}")
    print(f"wrote {csv_path}")
    print(f"wrote {summary_path}")
    return 0 if summary["micro_gates_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
