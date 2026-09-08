"""Audit OP12 detector behavior for scripted blue styles.

This diagnostic intentionally does not tune OP12. It measures whether detector
signals can distinguish blue formations before any payoff interpretation.
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
ALL_BLUE_STYLES = ("BLUE_RUSH", "BLUE_TURTLE", "BLUE_SPLIT", "BLUE_ESCORT")
BLUE_PROBE_PROTOCOL = "BLUE_PROBES_V2"
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
    escort_detector_score = 0.0
    escort_detector_compact = 0.0
    escort_detector_narrow = 0.0
    escort_detector_leader = 0.0
    escort_detector_heading = 0.0
    escort_detector_speed_penalty = 0.0
    convoy_offensive_seen = 0
    convoy_corridor_seen = 0
    convoy_leader_seen = 0
    convoy_reject_seen = 0
    escort_confirmation_step = -1
    escort_confirmation_active_steps = 0
    escort_confirmation_carrier_id = -1
    escort_confirmation_protector_id = -1
    escort_confirmation_distance = 0.0
    escort_confirmation_same_corridor_steps = 0
    escort_confirmation_to_episode_end_steps = 0
    external_escort_ticks = 0
    external_escort_first_trigger_step = None
    external_escort_active_steps = 0
    external_prev_xs = None
    external_prev_ys = None
    external_leader_sign = 0
    external_leader_streak = 0

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

    post_pickup_steps = 0
    post_pickup_teammate_dist = []
    post_pickup_lane_sep = []
    post_pickup_same_corridor = 0
    post_pickup_shadowing = 0
    post_pickup_independent = 0
    post_pickup_trailing = 0
    post_pickup_stable_carrier_protector = 0
    post_pickup_carrier_id = -1
    post_pickup_protector_id = -1

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

            carrying = [bool(core.blue_carrying[0, i].item()) for i in range(2)]
            if any(carrying) and both_alive:
                carrier_id = 0 if carrying[0] else 1
                protector_id = 1 - carrier_id
                post_pickup_carrier_id = carrier_id
                post_pickup_protector_id = protector_id
                post_pickup_steps += 1
                post_pickup_teammate_dist.append(dist)
                post_pickup_lane_sep.append(lane_sep)
                if lane_sep <= 3.0:
                    post_pickup_same_corridor += 1
                protector_behind_or_beside = xs[protector_id] >= xs[carrier_id] - 0.75
                controlled_distance = 1.0 <= dist <= 4.0
                if protector_behind_or_beside:
                    post_pickup_trailing += 1
                if lane_sep <= 3.0 and controlled_distance:
                    post_pickup_shadowing += 1
                if lane_sep > 4.5 or dist > 7.0:
                    post_pickup_independent += 1
                if protector_behind_or_beside and lane_sep <= 3.0 and controlled_distance:
                    post_pickup_stable_carrier_protector += 1

            lead_x = max(xs)
            trail_x = min(xs)
            dx01 = xs[0] - xs[1]
            dy01 = ys[0] - ys[1]
            leader_sign = 1 if dx01 > 0.5 else (-1 if dx01 < -0.5 else 0)
            external_leader_streak = (
                external_leader_streak + 1
                if leader_sign != 0 and leader_sign == external_leader_sign
                else (1 if leader_sign != 0 else 0)
            )
            external_leader_sign = leader_sign
            if external_prev_xs is None:
                heading_sim = 0.0
                speed_penalty = 0.0
            else:
                step_dx0 = xs[0] - external_prev_xs[0]
                step_dx1 = xs[1] - external_prev_xs[1]
                step_dy0 = ys[0] - external_prev_ys[0]
                step_dy1 = ys[1] - external_prev_ys[1]
                speed0 = math.hypot(step_dx0, step_dy0)
                speed1 = math.hypot(step_dx1, step_dy1)
                dot = step_dx0 * step_dx1 + step_dy0 * step_dy1
                heading_sim = max(0.0, min(1.0, (dot / max(speed0 * speed1, 1e-8) + 1.0) * 0.5))
                avg_forward = (step_dx0 + step_dx1) * 0.5
                speed_penalty = max(0.0, min(1.0, (avg_forward - 0.58) / 0.25))
            compact = max(0.0, min(1.0, (5.5 - dist) / 5.5))
            narrow = max(0.0, min(1.0, (3.5 - lane_sep) / 3.5))
            leader_component = max(0.0, min(1.0, external_leader_streak / 4.0))
            external_score = compact + narrow + leader_component + heading_sim - speed_penalty
            external_escort_now = (
                step < opening_steps
                and first_pickup is None
                and both_alive
                and lead_x > midfield - 3.0
                and trail_x > midfield - 12.0
                and external_prev_xs is not None
                and abs(dx01) >= 0.5
                and external_score >= 3.00
            )
            external_escort_ticks = external_escort_ticks + 1 if external_escort_now else 0
            if external_escort_ticks >= 3:
                external_escort_active_steps += 1
                if external_escort_first_trigger_step is None:
                    external_escort_first_trigger_step = step
            external_prev_xs = list(xs)
            external_prev_ys = list(ys)

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
            escort_detector_score = max(escort_detector_score, _as_float(getattr(core, "bt_adapt_opening_escort_score", 0.0)))
            escort_detector_compact = max(escort_detector_compact, _as_float(getattr(core, "bt_adapt_opening_escort_compact", 0.0)))
            escort_detector_narrow = max(escort_detector_narrow, _as_float(getattr(core, "bt_adapt_opening_escort_narrow", 0.0)))
            escort_detector_leader = max(escort_detector_leader, _as_float(getattr(core, "bt_adapt_opening_escort_leader", 0.0)))
            escort_detector_heading = max(escort_detector_heading, _as_float(getattr(core, "bt_adapt_opening_escort_heading", 0.0)))
            escort_detector_speed_penalty = max(
                escort_detector_speed_penalty,
                _as_float(getattr(core, "bt_adapt_opening_escort_speed_penalty", 0.0)),
            )
            convoy_offensive_seen = max(convoy_offensive_seen, _core_int(core, "bt_adapt_convoy_offensive_active", 0))
            convoy_corridor_seen = max(convoy_corridor_seen, _core_int(core, "bt_adapt_convoy_corridor_active", 0))
            convoy_leader_seen = max(convoy_leader_seen, _core_int(core, "bt_adapt_convoy_leader_active", 0))
            convoy_reject_seen = max(convoy_reject_seen, _core_int(core, "bt_adapt_convoy_reject_rush", 0))
            confirmation_now = _core_int(core, "bt_adapt_escort_confirm_first_step", -1)
            escort_confirmation_step = (
                confirmation_now
                if escort_confirmation_step < 0 and confirmation_now >= 0
                else escort_confirmation_step
            )
            escort_confirmation_active_steps = max(
                escort_confirmation_active_steps,
                _core_int(core, "bt_adapt_escort_confirm_active_steps", 0),
            )
            escort_confirmation_carrier_id = max(
                escort_confirmation_carrier_id,
                _core_int(core, "bt_adapt_escort_confirm_carrier_id", -1),
            )
            escort_confirmation_protector_id = max(
                escort_confirmation_protector_id,
                _core_int(core, "bt_adapt_escort_confirm_protector_id", -1),
            )
            escort_confirmation_distance = max(
                escort_confirmation_distance,
                _as_float(getattr(core, "bt_adapt_escort_confirm_distance", 0.0)),
            )
            escort_confirmation_same_corridor_steps = max(
                escort_confirmation_same_corridor_steps,
                _core_int(core, "bt_adapt_escort_confirm_same_corridor_steps", 0),
            )
            escort_confirmation_to_episode_end_steps = max(
                escort_confirmation_to_episode_end_steps,
                _core_int(core, "bt_adapt_escort_confirm_to_end_steps", 0),
            )
            if bool(dones.any()):
                break
    finally:
        env.close()

    leader_total = opening_same_leader[0] + opening_same_leader[1]
    return {
        "blue_style": blue_style,
        "blue_probe_protocol": BLUE_PROBE_PROTOCOL,
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
        "escort_detector_score": escort_detector_score,
        "escort_detector_compact": escort_detector_compact,
        "escort_detector_narrow": escort_detector_narrow,
        "escort_detector_leader": escort_detector_leader,
        "escort_detector_heading": escort_detector_heading,
        "escort_detector_speed_penalty": escort_detector_speed_penalty,
        "convoy_offensive_seen": convoy_offensive_seen,
        "convoy_corridor_seen": convoy_corridor_seen,
        "convoy_leader_seen": convoy_leader_seen,
        "convoy_reject_seen": convoy_reject_seen,
        "escort_confirmation_step": escort_confirmation_step,
        "escort_confirmation_active_steps": escort_confirmation_active_steps,
        "escort_confirmation_carrier_id": escort_confirmation_carrier_id,
        "escort_confirmation_protector_id": escort_confirmation_protector_id,
        "escort_confirmation_distance": escort_confirmation_distance,
        "escort_confirmation_same_corridor_steps": escort_confirmation_same_corridor_steps,
        "escort_confirmation_to_episode_end_steps": escort_confirmation_to_episode_end_steps,
        "pickup_to_confirmation_steps": (
            escort_confirmation_step - first_pickup
            if first_pickup is not None and escort_confirmation_step >= 0
            else None
        ),
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
        "post_pickup_steps": post_pickup_steps,
        "post_pickup_carrier_id": post_pickup_carrier_id,
        "post_pickup_protector_id": post_pickup_protector_id,
        "post_pickup_mean_teammate_dist": mean(post_pickup_teammate_dist) if post_pickup_teammate_dist else None,
        "post_pickup_mean_lane_sep": mean(post_pickup_lane_sep) if post_pickup_lane_sep else None,
        "post_pickup_same_corridor_frac": (
            post_pickup_same_corridor / post_pickup_steps if post_pickup_steps else None
        ),
        "post_pickup_shadowing_frac": (
            post_pickup_shadowing / post_pickup_steps if post_pickup_steps else None
        ),
        "post_pickup_independent_frac": (
            post_pickup_independent / post_pickup_steps if post_pickup_steps else None
        ),
        "post_pickup_trailing_frac": (
            post_pickup_trailing / post_pickup_steps if post_pickup_steps else None
        ),
        "post_pickup_stable_carrier_protector_frac": (
            post_pickup_stable_carrier_protector / post_pickup_steps if post_pickup_steps else None
        ),
    }


def _mean_present(rows, key):
    vals = [row[key] for row in rows if row[key] is not None]
    return mean(vals) if vals else None


def summarize(rows, blue_styles=BLUE_STYLES):
    by_style = {}
    for style in blue_styles:
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
            "mean_escort_detector_score": _mean_present(style_rows, "escort_detector_score"),
            "mean_escort_detector_compact": _mean_present(style_rows, "escort_detector_compact"),
            "mean_escort_detector_narrow": _mean_present(style_rows, "escort_detector_narrow"),
            "mean_escort_detector_leader": _mean_present(style_rows, "escort_detector_leader"),
            "mean_escort_detector_heading": _mean_present(style_rows, "escort_detector_heading"),
            "mean_escort_detector_speed_penalty": _mean_present(style_rows, "escort_detector_speed_penalty"),
            "convoy_offensive_episodes": sum(1 for row in style_rows if int(row["convoy_offensive_seen"]) > 0),
            "convoy_corridor_episodes": sum(1 for row in style_rows if int(row["convoy_corridor_seen"]) > 0),
            "convoy_leader_episodes": sum(1 for row in style_rows if int(row["convoy_leader_seen"]) > 0),
            "convoy_reject_episodes": sum(1 for row in style_rows if int(row["convoy_reject_seen"]) > 0),
            "mean_escort_confirmation_step": _mean_present(
                [row for row in style_rows if row["escort_confirmation_step"] >= 0],
                "escort_confirmation_step",
            ),
            "escort_confirmation_episodes": sum(
                1 for row in style_rows if row["escort_confirmation_step"] >= 0
            ),
            "mean_pickup_to_confirmation_steps": _mean_present(
                [row for row in style_rows if row["pickup_to_confirmation_steps"] is not None],
                "pickup_to_confirmation_steps",
            ),
            "mean_escort_confirmation_distance": _mean_present(
                style_rows,
                "escort_confirmation_distance",
            ),
            "mean_escort_confirmation_same_corridor_steps": _mean_present(
                style_rows,
                "escort_confirmation_same_corridor_steps",
            ),
            "mean_escort_confirmation_to_episode_end_steps": _mean_present(
                [row for row in style_rows if row["escort_confirmation_step"] >= 0],
                "escort_confirmation_to_episode_end_steps",
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
            "mean_post_pickup_steps": _mean_present(style_rows, "post_pickup_steps"),
            "mean_post_pickup_teammate_dist": _mean_present(style_rows, "post_pickup_mean_teammate_dist"),
            "mean_post_pickup_lane_sep": _mean_present(style_rows, "post_pickup_mean_lane_sep"),
            "mean_post_pickup_same_corridor_frac": _mean_present(style_rows, "post_pickup_same_corridor_frac"),
            "mean_post_pickup_shadowing_frac": _mean_present(style_rows, "post_pickup_shadowing_frac"),
            "mean_post_pickup_independent_frac": _mean_present(style_rows, "post_pickup_independent_frac"),
            "mean_post_pickup_trailing_frac": _mean_present(style_rows, "post_pickup_trailing_frac"),
            "mean_post_pickup_stable_carrier_protector_frac": _mean_present(
                style_rows,
                "post_pickup_stable_carrier_protector_frac",
            ),
            "pickup_episodes": sum(1 for row in style_rows if row["time_first_pickup"] is not None),
            "blue_score_episodes": sum(1 for row in style_rows if row["time_first_blue_score"] is not None),
        }
    return {
        "red_preset": RED_PRESET,
        "map": MAP_NAME,
        "blue_probe_protocol": BLUE_PROBE_PROTOCOL,
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
    parser.add_argument("--blue-styles", nargs="+", default=list(BLUE_STYLES))
    parser.add_argument("--out-dir", default="AICTFProject/artifacts/op12_opening_audit_rush_vs_escort_8seed")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    blue_styles = tuple(args.blue_styles)
    for blue_style in blue_styles:
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

    summary = summarize(rows, blue_styles)
    summary_path = out_dir / "opening_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
