#!/usr/bin/env python3
"""Micro-gate diagnostic for OP8 formation-opening RUSH contract (Contract A).

Checks legal-state timing only (sim steps / pickup), never blue-style ID as a
red trigger. Reports whether RUSH tends to pick up before formation arms and
whether SPLIT/ESCORT tend to arrive after.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.run_scripted_style_payoff_matrix import (  # noqa: E402
    _episode_seed,
    _make_env,
    _zero_action,
)
from gpu_env._core._bt_adaptive import _BTAdaptiveMixin  # noqa: E402

BLUE_STYLES = (
    "BLUE_RUSH",
    "BLUE_SPLIT",
    "BLUE_ESCORT",
    "BLUE_TURTLE",
)
RED_STYLE = "OP8_PROTECTED_CARRIER_ESCORT"
DEFAULT_MAP = "map_b_split_lane"


def _run_episode(
    *,
    blue_style: str,
    episode_index: int,
    episode_seed: int,
    map_name: str,
    max_decision_steps: int,
    device: str,
) -> dict[str, Any]:
    opening_steps = int(_BTAdaptiveMixin._OP8_FORMATION_OPENING_STEPS)
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

        first_blue_pickup = -1
        activation_step = -1
        first_blue_score = -1
        steps = 0
        last_info: dict[str, Any] = {}
        for _ in range(int(max_decision_steps) + 5):
            sim = int(core.sim_step_count[0].item())
            blue_carry = bool(core.blue_carrying[0].any().item())
            # Actual fortify arm (time OR blue already carrying).
            if activation_step < 0 and (sim >= opening_steps or blue_carry):
                activation_step = sim
            if first_blue_pickup < 0 and blue_carry:
                first_blue_pickup = sim

            action = _zero_action(env)
            env.step_async(action)
            _, _, done, infos = env.step_wait()
            last_info = infos[0] if infos else {}
            ep_res = last_info.get("episode_result", last_info)
            blue_score_now = int(ep_res.get("blue_score", core.blue_score[0].item()))
            if first_blue_score < 0 and blue_score_now > 0:
                first_blue_score = sim
            steps += 1
            if bool(done.any()):
                break

        if activation_step < 0:
            activation_step = int(core.sim_step_count[0].item())

        ep_res = last_info.get("episode_result", last_info) if last_info else {}
        blue_score = int(ep_res.get("blue_score", core.blue_score[0].item()))
        red_score = int(ep_res.get("red_score", core.red_score[0].item()))

        # Micro-gates judge against the scheduled formation wall-clock, not the
        # emergency blue-carry latch (which coincides with pickup by definition).
        pickup_before = first_blue_pickup >= 0 and first_blue_pickup < opening_steps
        pickup_after = first_blue_pickup >= opening_steps
        return {
            "blue_style": blue_style,
            "red_style": RED_STYLE,
            "map": map_name,
            "episode_index": episode_index,
            "episode_seed": episode_seed,
            "opening_steps": opening_steps,
            "activation_step": activation_step,
            "first_blue_pickup": first_blue_pickup,
            "first_blue_score": first_blue_score,
            "blue_score": blue_score,
            "red_score": red_score,
            "win_margin": blue_score - red_score,
            "steps": steps,
            "pickup_before_activation": int(pickup_before),
            "pickup_after_activation": int(pickup_after),
            "no_blue_pickup": int(first_blue_pickup < 0),
        }
    finally:
        env.close()


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_style: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_style[str(row["blue_style"])].append(row)

    summary: dict[str, Any] = {"n_episodes_per_style": {}, "styles": {}}
    for style, style_rows in by_style.items():
        n = len(style_rows)
        summary["n_episodes_per_style"][style] = n
        with_pickup = [r for r in style_rows if int(r["first_blue_pickup"]) >= 0]
        with_score = [r for r in style_rows if int(r["first_blue_score"]) >= 0]
        before = sum(int(r["pickup_before_activation"]) for r in style_rows)
        after = sum(int(r["pickup_after_activation"]) for r in style_rows)
        none = sum(int(r["no_blue_pickup"]) for r in style_rows)
        mean_pickup = (
            sum(int(r["first_blue_pickup"]) for r in with_pickup) / len(with_pickup)
            if with_pickup
            else None
        )
        mean_first_score = (
            sum(int(r["first_blue_score"]) for r in with_score) / len(with_score)
            if with_score
            else None
        )
        mean_act = sum(int(r["activation_step"]) for r in style_rows) / max(n, 1)
        mean_margin = sum(float(r["win_margin"]) for r in style_rows) / max(n, 1)
        summary["styles"][style] = {
            "pickup_before_frac": before / max(n, 1),
            "pickup_after_frac": after / max(n, 1),
            "no_pickup_frac": none / max(n, 1),
            "mean_first_pickup": mean_pickup,
            "mean_first_score": mean_first_score,
            "mean_activation_step": mean_act,
            "mean_win_margin": mean_margin,
        }

    rush = summary["styles"].get("BLUE_RUSH", {})
    split = summary["styles"].get("BLUE_SPLIT", {})
    escort = summary["styles"].get("BLUE_ESCORT", {})
    turtle = summary["styles"].get("BLUE_TURTLE", {})
    # ESCORT reaches the flag as fast as RUSH under BLUE_PROBES_V2; its cost is
    # post-pickup coordination. Gate on mean win margin / score lag vs RUSH.
    escort_slower_or_weaker = float(escort.get("mean_win_margin", 0.0)) < (
        float(rush.get("mean_win_margin", 0.0)) - 0.25
    ) or (
        escort.get("mean_first_score") is not None
        and rush.get("mean_first_score") is not None
        and float(escort["mean_first_score"]) > float(rush["mean_first_score"]) + 5.0
    )
    summary["micro_gates"] = {
        "rush_usually_before": float(rush.get("pickup_before_frac", 0.0)) >= 0.5,
        "split_usually_after_or_none": (
            float(split.get("pickup_after_frac", 0.0))
            + float(split.get("no_pickup_frac", 0.0))
        )
        >= 0.5,
        "escort_slower_or_weaker_than_rush": escort_slower_or_weaker,
        "turtle_rarely_threatens_opening": float(turtle.get("pickup_before_frac", 0.0))
        <= 0.25,
    }
    summary["micro_gates_pass"] = all(summary["micro_gates"].values())
    return summary


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=8)
    p.add_argument("--base-seed", type=int, default=561001)
    p.add_argument("--map", default=DEFAULT_MAP)
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-decision-steps", type=int, default=240)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "op8_formation_microgates_dev8",
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
                f"pickup={row['first_blue_pickup']} act={row['activation_step']} "
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
    print(json.dumps(summary["micro_gates"], indent=2))
    print(f"micro_gates_pass={summary['micro_gates_pass']}")
    print(f"wrote {csv_path}")
    print(f"wrote {summary_path}")
    return 0 if summary["micro_gates_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
