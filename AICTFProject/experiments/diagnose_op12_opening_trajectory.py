"""Audit OP12 opening behavior for BLUE_RUSH vs BLUE_ESCORT.

This diagnostic intentionally does not tune OP12. It measures whether the two
candidate offensive probes are separable during the first opening window.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from statistics import mean

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.run_scripted_style_payoff_matrix import _make_env, _zero_action


BLUE_STYLES = ("BLUE_RUSH", "BLUE_ESCORT")
RED_PRESET = "OP12_LATE_CONVERTER"
MAP_NAME = "map_b_split_lane"


def _as_float(value) -> float:
    try:
        if hasattr(value, "detach"):
            value = value.detach().flatten()[0]
        if hasattr(value, "item"):
            value = value.item()
        return float(value)
    except Exception:
        return float("nan")


def _first_or_none(value):
    return None if value is None else int(value)


def _core_int(core, attr: str, default: int = -1) -> int:
    val = getattr(core, attr, None)
    if val is None:
        return int(default)
    try:
        return int(val[0].item())
    except Exception:
        return int(default)


def _core(env):
    return env.core


def _episode_seed(base_seed: int, episode_index: int) -> int:
    return int(base_seed + episode_index)


def _agent_positions(core):
    xs = [_as_float(core.blue_x[0, i]) for i in range(2)]
    ys = [_as_float(core.blue_y[0, i]) for i in range(2)]
    alive = [bool(core.blue_alive[0, i].item()) and not bool(core.blue_tagged[0, i].item()) for i in range(2)]
    return xs, ys, alive


def _distance(xs, ys) -> float:
    return math.hypot(xs[0] - xs[1], ys[0] - ys[1])


def _lane_sep(ys) -> float:
    return abs(ys[0] - ys[1])


def run_episode(blue_style: str, episode_index: int, base_seed: int, opening_steps: int, max_steps: int, device: str):
    seed = _episode_seed(base_seed, episode_index)
    env = _make_env(
        map_name=MAP_NAME,
        seed=seed,
        max_decision_steps=max_steps,
        device=device,
    )
    env.env_method("set_phase", RED_PRESET)
    env.env_method("set_next_opponent", "SCRIPTED", RED_PRESET)
    core = _core(env)
    core.blue_scripted = True
    core.set_blue_style(blue_style)
    env.reset()
    env.env_method("set_phase", RED_PRESET)
    env.env_method("set_next_opponent", "SCRIPTED", RED_PRESET)
    core = _core(env)
    core.blue_scripted = True
    core.set_blue_style(blue_style)

    core = _core(env)
    midfield = float(core.cols) * 0.5
    red_flag_x = _as_float(core.red_flag_pos[0, 0])
    red_flag_y = _as_float(core.red_flag_pos[0, 1])
    red_flag_touch_radius = 2.0

    first_midfield_any = None
    first_midfield_both = None
    first_flag_touch = None
    first_pickup = None
    first_blue_score = None
    first_red_score = None
    split_detector_first_trigger_step = -1
    split_detector_active_steps = 0
    escort_detector_first_trigger_step = -1
    escort_detector_active_steps = 0
    external_escort_ticks = 0
    external_escort_first_trigger_step = None
    external_escort_active_steps = 0

    prev_xs = None
    blue_score0 = int(_as_float(getattr(core, "blue_score", 0)))
    red_score0 = int(_as_float(getattr(core, "red_score", 0)))
    opening_dist = []
    opening_lane_sep = []
    opening_forward_velocity = []
    opening_clustered = 0
    opening_trailing = 0
    opening_same_leader = {0: 0, 1: 0}
    opening_alive_steps = 0

    pre_pickup_dist = []
    pre_pickup_lane_sep = []
    pre_pickup_clustered = 0
    pre_pickup_steps = 0

    try:
        for step in range(max_steps):
            core = _core(env)
            xs, ys, alive = _agent_positions(core)
            both_alive = alive[0] and alive[1]
            dist = _distance(xs, ys)
            lane_sep = _lane_sep(ys)

            if first_midfield_any is None and any(alive[i] and xs[i] >= midfield for i in range(2)):
                first_midfield_any = step
            if first_midfield_both is None and all(alive[i] and xs[i] >= midfield for i in range(2)):
                first_midfield_both = step
            if first_flag_touch is None:
                if any(alive[i] and math.hypot(xs[i] - red_flag_x, ys[i] - red_flag_y) <= red_flag_touch_radius for i in range(2)):
                    first_flag_touch = step
            if first_pickup is None and any(bool(core.blue_carrying[0, i].item()) for i in range(2)):
                first_pickup = step

            if step < opening_steps and both_alive:
                opening_alive_steps += 1
                opening_dist.append(dist)
                opening_lane_sep.append(lane_sep)
                if prev_xs is not None:
                    opening_forward_velocity.append(mean([xs[i] - prev_xs[i] for i in range(2)]))
                if dist < 6.0:
                    opening_clustered += 1
                dx = xs[0] - xs[1]
                if abs(dx) >= 0.5:
                    leader = 0 if dx > 0.0 else 1
                    opening_same_leader[leader] += 1
                    if dist < 9.0:
                        opening_trailing += 1

            if first_pickup is None and both_alive:
                pre_pickup_steps += 1
                pre_pickup_dist.append(dist)
                pre_pickup_lane_sep.append(lane_sep)
                if dist < 6.0:
                    pre_pickup_clustered += 1

            lead_x = max(xs)
            trail_x = min(xs)
            external_escort_now = (
                step < opening_steps
                and first_pickup is None
                and both_alive
                and lead_x > midfield - 3.0
                and trail_x > midfield - 12.0
                and dist <= 3.0
                and lane_sep <= 2.25
                and abs(xs[0] - xs[1]) >= 0.5
            )
            external_escort_ticks = external_escort_ticks + 1 if external_escort_now else 0
            if external_escort_ticks >= 3:
                external_escort_active_steps += 1
                if external_escort_first_trigger_step is None:
                    external_escort_first_trigger_step = step

            prev_xs = xs
            env.step_async(_zero_action(env))
            _, _, dones, _ = env.step_wait()
            core = _core(env)

            if first_blue_score is None and int(_as_float(getattr(core, "blue_score", 0))) > blue_score0:
                first_blue_score = step + 1
            if first_red_score is None and int(_as_float(getattr(core, "red_score", 0))) > red_score0:
                first_red_score = step + 1
            split_now = _core_int(core, "bt_adapt_split_first_trigger_step", -1)
            split_detector_first_trigger_step = (
                split_now
                if split_detector_first_trigger_step < 0 and split_now >= 0
                else split_detector_first_trigger_step
            )
            split_detector_active_steps = max(
                split_detector_active_steps,
                _core_int(core, "bt_adapt_split_active_steps", 0),
            )
            escort_now = _core_int(core, "bt_adapt_opening_escort_first_trigger_step", -1)
            escort_detector_first_trigger_step = (
                escort_now
                if escort_detector_first_trigger_step < 0 and escort_now >= 0
                else escort_detector_first_trigger_step
            )
            escort_detector_active_steps = max(
                escort_detector_active_steps,
                _core_int(core, "bt_adapt_opening_escort_active_steps", 0),
            )
            if bool(dones.any()):
                break
    finally:
        env.close()

    leader_total = opening_same_leader[0] + opening_same_leader[1]
    return {
        "blue_style": blue_style,
        "red_preset": RED_PRESET,
        "core_opponent_key": str(getattr(core, "_opponent_key", "")),
        "map": MAP_NAME,
        "episode_index": episode_index,
        "seed": seed,
        "steps_observed": step + 1,
        "time_cross_midfield_any": _first_or_none(first_midfield_any),
        "time_cross_midfield_both": _first_or_none(first_midfield_both),
        "time_first_flag_touch": _first_or_none(first_flag_touch),
        "time_first_pickup": _first_or_none(first_pickup),
        "time_first_blue_score": _first_or_none(first_blue_score),
        "time_first_red_score": _first_or_none(first_red_score),
        "split_detector_first_trigger_step": split_detector_first_trigger_step,
        "split_detector_active_steps": split_detector_active_steps,
        "escort_detector_first_trigger_step": escort_detector_first_trigger_step,
        "escort_detector_active_steps": escort_detector_active_steps,
        "external_escort_first_trigger_step": _first_or_none(external_escort_first_trigger_step),
        "external_escort_active_steps": external_escort_active_steps,
        "opening_mean_teammate_dist": mean(opening_dist) if opening_dist else None,
        "opening_mean_lane_sep": mean(opening_lane_sep) if opening_lane_sep else None,
        "opening_mean_forward_velocity": mean(opening_forward_velocity) if opening_forward_velocity else None,
        "opening_clustered_frac": opening_clustered / opening_alive_steps if opening_alive_steps else None,
        "opening_trailing_frac": opening_trailing / opening_alive_steps if opening_alive_steps else None,
        "opening_stable_leader_frac": (max(opening_same_leader.values()) / leader_total) if leader_total else None,
        "pre_pickup_mean_teammate_dist": mean(pre_pickup_dist) if pre_pickup_dist else None,
        "pre_pickup_mean_lane_sep": mean(pre_pickup_lane_sep) if pre_pickup_lane_sep else None,
        "pre_pickup_clustered_frac": pre_pickup_clustered / pre_pickup_steps if pre_pickup_steps else None,
    }


def _mean_present(rows, key):
    vals = [row[key] for row in rows if row[key] is not None]
    return mean(vals) if vals else None


def summarize(rows):
    by_style = {}
    for style in BLUE_STYLES:
        style_rows = [row for row in rows if row["blue_style"] == style]
        by_style[style] = {
            "n": len(style_rows),
            "mean_time_cross_midfield_any": _mean_present(style_rows, "time_cross_midfield_any"),
            "mean_time_cross_midfield_both": _mean_present(style_rows, "time_cross_midfield_both"),
            "mean_time_first_flag_touch": _mean_present(style_rows, "time_first_flag_touch"),
            "mean_time_first_pickup": _mean_present(style_rows, "time_first_pickup"),
            "mean_time_first_blue_score": _mean_present(style_rows, "time_first_blue_score"),
            "mean_split_detector_first_trigger_step": _mean_present(
                [row for row in style_rows if row["split_detector_first_trigger_step"] >= 0],
                "split_detector_first_trigger_step",
            ),
            "split_detector_trigger_episodes": sum(
                1 for row in style_rows if row["split_detector_first_trigger_step"] >= 0
            ),
            "mean_escort_detector_first_trigger_step": _mean_present(
                [row for row in style_rows if row["escort_detector_first_trigger_step"] >= 0],
                "escort_detector_first_trigger_step",
            ),
            "escort_detector_trigger_episodes": sum(
                1 for row in style_rows if row["escort_detector_first_trigger_step"] >= 0
            ),
            "mean_external_escort_first_trigger_step": _mean_present(
                [row for row in style_rows if row["external_escort_first_trigger_step"] is not None],
                "external_escort_first_trigger_step",
            ),
            "external_escort_trigger_episodes": sum(
                1 for row in style_rows if row["external_escort_first_trigger_step"] is not None
            ),
            "mean_opening_teammate_dist": _mean_present(style_rows, "opening_mean_teammate_dist"),
            "mean_opening_lane_sep": _mean_present(style_rows, "opening_mean_lane_sep"),
            "mean_opening_forward_velocity": _mean_present(style_rows, "opening_mean_forward_velocity"),
            "mean_opening_clustered_frac": _mean_present(style_rows, "opening_clustered_frac"),
            "mean_opening_trailing_frac": _mean_present(style_rows, "opening_trailing_frac"),
            "mean_opening_stable_leader_frac": _mean_present(style_rows, "opening_stable_leader_frac"),
            "mean_pre_pickup_teammate_dist": _mean_present(style_rows, "pre_pickup_mean_teammate_dist"),
            "mean_pre_pickup_lane_sep": _mean_present(style_rows, "pre_pickup_mean_lane_sep"),
            "mean_pre_pickup_clustered_frac": _mean_present(style_rows, "pre_pickup_clustered_frac"),
            "pickup_episodes": sum(1 for row in style_rows if row["time_first_pickup"] is not None),
            "blue_score_episodes": sum(1 for row in style_rows if row["time_first_blue_score"] is not None),
        }
    return {
        "red_preset": RED_PRESET,
        "map": MAP_NAME,
        "styles": by_style,
        "interpretation": {
            "purpose": "opening behavior audit only; not a payoff confirmation",
            "opening_steps": "metrics with opening_ prefix use the configured early OP12 window",
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--base-seed", type=int, default=551001)
    parser.add_argument("--opening-steps", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=240)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-dir", default="AICTFProject/artifacts/op12_opening_audit_rush_vs_escort_8seed")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for blue_style in BLUE_STYLES:
        for episode_index in range(args.episodes):
            row = run_episode(
                blue_style,
                episode_index,
                args.base_seed,
                args.opening_steps,
                args.max_steps,
                args.device,
            )
            rows.append(row)
            print(
                f"{blue_style} ep={episode_index} seed={row['seed']} "
                f"mid_any={row['time_cross_midfield_any']} pickup={row['time_first_pickup']} "
                f"score={row['time_first_blue_score']}",
                flush=True,
            )

    csv_path = out_dir / "opening_rows.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows)
    summary_path = out_dir / "opening_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
