#!/usr/bin/env python3
"""Event-timeline diagnostic for OP6 scripted-blue calibration.

This is a calibration tool, not a pool-admissibility result. It answers why
BLUE_TURTLE loses against OP6/map_b before tuning either side further.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.run_scripted_style_payoff_matrix import (  # noqa: E402
    DEFAULT_REDS,
    _episode_seed,
)
from gpu_env import GPUCTFVecEnv, GPUFieldConfig  # noqa: E402
from gpu_env._core._scripted_blue_styles import BLUE_STYLE_NAMES  # noqa: E402


ROW_FIELDS = [
    "blue_style",
    "red_style",
    "map",
    "episode_index",
    "episode_seed",
    "blue_score",
    "red_score",
    "win_margin",
    "steps",
    "time_first_red_midfield_cross",
    "time_both_red_enter_blue_territory",
    "time_first_red_flag_touch",
    "time_first_red_capture",
    "time_blue_counterattack_start",
    "time_first_blue_flag_touch",
    "time_first_blue_capture",
    "red_death_count",
    "blue_death_count",
    "red_carrier_death_count",
    "blue_carrier_death_count",
    "red_tag_count",
    "blue_tag_count",
    "red_carrier_tag_count",
    "blue_carrier_tag_count",
    "post_tag_carrier_event_count",
    "post_tag_counterattack_launch_count",
    "post_tag_blue_flag_touch_count",
    "post_tag_blue_capture_before_red_reentry_count",
    "post_tag_mean_steps_to_blue_flag_touch",
    "post_tag_mean_steps_to_red_reentry",
    "op6_regroup_active_steps",
    "op6_regroup_blue_flag_touch_count",
    "op6_regroup_blue_capture_count",
    "op7_mean_red_lateral_separation",
    "op7_blue_opposite_lane_penetration_steps",
    "op7_min_blue0_to_red_flag",
    "op7_min_blue1_to_red_flag",
    "op7_red0_target_switches",
    "op7_red1_target_switches",
    "op7_both_red_target_same_blue_steps",
    "op7_uncovered_lane_steps",
    "op7_max_consecutive_uncovered_lane_steps",
    "op7_mean_uncovered_blue_progress",
    "op7_flag_touch_during_uncovered_lane",
    "split_flag_touch_count",
    "split_pickup_count",
    "split_touches_per_pickup",
    "split_capture_count",
    "split_capture_given_pickup",
    "split_mean_carrier_lifetime",
    "split_mean_max_return_progress",
    "split_mean_teammate_dist_at_pickup",
    "split_mean_sep_before_pickup",
    "split_mean_sep_after_pickup",
    "split_converged_after_pickup_count",
    "split_noncarrier_flag_pressure_steps",
    "split_noncarrier_support_steps",
    "split_mean_red_retarget_latency_after_pickup",
    "split_carrier_loss_tag_count",
    "split_carrier_loss_capture_count",
    "split_carrier_loss_other_count",
    "pre_touch_min_red0_to_blue",
    "pre_touch_min_red1_to_blue",
    "pre_touch_min_any_red_to_blue",
    "pre_touch_steps_red0_one_defender_in_tag_range",
    "pre_touch_steps_red1_one_defender_in_tag_range",
    "pre_touch_steps_red0_two_defenders_in_tag_range",
    "pre_touch_steps_red1_two_defenders_in_tag_range",
    "pre_touch_steps_any_red_two_defenders_in_tag_range",
    "pre_touch_max_consecutive_red0_two_defenders",
    "pre_touch_max_consecutive_red1_two_defenders",
    "pre_touch_max_consecutive_any_red_two_defenders",
    "pre_touch_max_red0_tag_accumulator",
    "pre_touch_max_red1_tag_accumulator",
    "tag_threshold_seconds",
    "tag_required_consecutive_steps",
    "pre_touch_red0_accumulator_reset_count",
    "pre_touch_red1_accumulator_reset_count",
    "pre_touch_reset_defender0_left_radius",
    "pre_touch_reset_defender1_left_radius",
    "pre_touch_reset_red_left_blue_side",
    "pre_touch_reset_defender_left_blue_side",
    "pre_touch_reset_target_switched",
    "pre_touch_reset_defender_became_tagged",
    "pre_touch_reset_red_target_became_tagged",
    "pre_touch_reset_carrier_turn_or_lane_change",
    "pre_touch_reset_other",
    "pre_touch_mean_reset_relative_speed",
    "pre_touch_path_cross_count",
    "turtle_def0_target_red0_steps",
    "turtle_def0_target_red1_steps",
    "turtle_def0_target_blue_flag_steps",
    "turtle_def0_target_red_flag_steps",
    "turtle_def1_target_red0_steps",
    "turtle_def1_target_red1_steps",
    "turtle_def1_target_blue_flag_steps",
    "turtle_def1_target_red_flag_steps",
    "min_turtle_defender_dist_to_blue_flag",
    "mean_turtle_defender_dist_to_blue_flag",
    "final_both_red_forward",
    "final_blue_had_counterattack",
]

RESET_EVENT_FIELDS = [
    "blue_style",
    "red_style",
    "map",
    "episode_index",
    "episode_seed",
    "step",
    "red_index",
    "defender0_left_radius",
    "defender1_left_radius",
    "red_left_blue_side",
    "defender_left_blue_side",
    "target_switched",
    "defender_became_tagged",
    "red_target_became_tagged",
    "carrier_turn_or_lane_change",
    "other",
    "relative_speed_def0",
    "relative_speed_def1",
    "red_x",
    "red_y",
    "red_carrying",
    "def0_target",
    "def1_target",
]

SPLIT_PICKUP_EVENT_FIELDS = [
    "blue_style",
    "red_style",
    "map",
    "episode_index",
    "episode_seed",
    "touch_step",
    "pickup_step",
    "carrier_idx",
    "end_step",
    "loss_cause",
    "carrier_lifetime",
    "max_return_progress",
    "crossed_midfield",
    "reached_capture_zone",
    "captured",
    "carrier_loss_x",
    "carrier_loss_y",
    "min_dist_red0",
    "min_dist_red1",
    "teammate_dist_at_pickup",
    "teammate_lane_before",
    "teammate_lane_after",
    "red_retarget_latency",
    "noncarrier_flag_pressure_steps",
    "noncarrier_support_steps",
]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--episodes", type=int, default=16)
    p.add_argument("--base-seed", type=int, default=260726)
    p.add_argument("--red-style", default="OP6_IMMEDIATE_DUAL_RUSH")
    p.add_argument("--map-name", default="map_b_split_lane")
    p.add_argument("--blue-styles", nargs="+", default=list(BLUE_STYLE_NAMES))
    p.add_argument("--max-decision-steps", type=int, default=240)
    p.add_argument("--device", default="cuda")
    p.add_argument("--progress-every", type=int, default=8)
    return p.parse_args()


def _make_env(*, map_name: str, seed: int, max_decision_steps: int, device: str) -> GPUCTFVecEnv:
    cfg = GPUFieldConfig(
        n_envs=1,
        max_blue_agents=2,
        max_red_agents=2,
        map_layout=str(map_name),
        max_decision_steps=int(max_decision_steps),
        aquaticus_profile=True,
        rules_profile="OURS",
        device=str(device),
        seed=int(seed),
    )
    return GPUCTFVecEnv(cfg)


def _zero_action(env: GPUCTFVecEnv) -> Any:
    return np.zeros_like(env.action_space.sample())


def _first_time(value: int | None, condition: bool, step: int) -> int | None:
    return step if value is None and condition else value


def _segments_intersect(a0: np.ndarray, a1: np.ndarray, b0: np.ndarray, b1: np.ndarray) -> bool:
    def orient(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> float:
        return float((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))

    def on_segment(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> bool:
        return (
            min(p[0], r[0]) - 1e-6 <= q[0] <= max(p[0], r[0]) + 1e-6
            and min(p[1], r[1]) - 1e-6 <= q[1] <= max(p[1], r[1]) + 1e-6
        )

    o1 = orient(a0, a1, b0)
    o2 = orient(a0, a1, b1)
    o3 = orient(b0, b1, a0)
    o4 = orient(b0, b1, a1)
    if o1 * o2 < 0.0 and o3 * o4 < 0.0:
        return True
    return (
        abs(o1) <= 1e-6 and on_segment(a0, b0, a1)
        or abs(o2) <= 1e-6 and on_segment(a0, b1, a1)
        or abs(o3) <= 1e-6 and on_segment(b0, a0, b1)
        or abs(o4) <= 1e-6 and on_segment(b0, a1, b1)
    )


def _closest_named_target(
    tx: float,
    ty: float,
    *,
    red_xy: np.ndarray,
    blue_flag_xy: np.ndarray,
    red_flag_xy: np.ndarray,
) -> str:
    candidates = {
        "red0": float(np.linalg.norm(np.array([tx, ty]) - red_xy[0])),
        "red1": float(np.linalg.norm(np.array([tx, ty]) - red_xy[1])),
        "blue_flag": float(np.linalg.norm(np.array([tx, ty]) - blue_flag_xy)),
        "red_flag": float(np.linalg.norm(np.array([tx, ty]) - red_flag_xy)),
    }
    return min(candidates, key=candidates.get)


def _target_index_from_name(name: str) -> int | None:
    if name == "red0":
        return 0
    if name == "red1":
        return 1
    return None


def _run_episode(
    *,
    blue_style: str,
    red_style: str,
    map_name: str,
    episode_index: int,
    episode_seed: int,
    max_decision_steps: int,
    device: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    env = _make_env(
        map_name=map_name,
        seed=episode_seed,
        max_decision_steps=max_decision_steps,
        device=device,
    )
    try:
        core = env.core
        env.env_method("set_phase", red_style)
        env.env_method("set_next_opponent", "SCRIPTED", red_style)
        core.blue_scripted = True
        core.set_blue_style(blue_style)
        env.reset()

        midline = float(core.cols) * 0.5
        blue_flag_x = float(core.blue_flag_home[0, 0].item())
        blue_flag_y = float(core.blue_flag_home[0, 1].item())

        t_red_mid = None
        t_both_red_forward = None
        t_red_touch = None
        t_red_cap = None
        t_blue_counter = None
        t_blue_touch = None
        t_blue_cap = None

        prev_red_score = int(core.red_score[0].item())
        prev_blue_score = int(core.blue_score[0].item())
        prev_red_alive = core.red_alive[0].detach().cpu().numpy().astype(bool)
        prev_blue_alive = core.blue_alive[0].detach().cpu().numpy().astype(bool)
        prev_red_tagged = core.red_tagged[0].detach().cpu().numpy().astype(bool)
        prev_blue_tagged = core.blue_tagged[0].detach().cpu().numpy().astype(bool)
        prev_red_carry = core.red_carrying[0].detach().cpu().numpy().astype(bool)
        prev_blue_carry = core.blue_carrying[0].detach().cpu().numpy().astype(bool)
        prev_rx = core.red_x[0].detach().cpu().numpy().astype(float)
        prev_ry = core.red_y[0].detach().cpu().numpy().astype(float)
        prev_bx = core.blue_x[0].detach().cpu().numpy().astype(float)
        prev_by = core.blue_y[0].detach().cpu().numpy().astype(float)

        red_deaths = 0
        blue_deaths = 0
        red_carrier_deaths = 0
        blue_carrier_deaths = 0
        red_tags = 0
        blue_tags = 0
        red_carrier_tags = 0
        blue_carrier_tags = 0
        post_tag_events: list[dict[str, Any]] = []
        op6_regroup_active_steps = 0
        op6_regroup_blue_touch_count = 0
        op6_regroup_blue_capture_count = 0
        prev_regroup_active = False
        op7_red_lateral_sep_sum = 0.0
        op7_red_lateral_sep_steps = 0
        op7_opposite_lane_penetration_steps = 0
        op7_min_blue_to_red_flag = [float("inf"), float("inf")]
        op7_red_target_switches = [0, 0]
        op7_prev_red_target_idx = [None, None]
        op7_same_blue_target_steps = 0
        op7_uncovered_lane_steps = 0
        op7_consecutive_uncovered = 0
        op7_max_consecutive_uncovered = 0
        op7_uncovered_progress: list[float] = []
        op7_flag_touch_during_uncovered = 0
        split_flag_touch_count = 0
        split_pickup_events: list[dict[str, Any]] = []
        split_active_pickup: dict[str, Any] | None = None
        split_last_touch_step: int | None = None
        split_prev_touch_now = False
        split_sep_before_samples: list[float] = []
        split_sep_after_samples: list[float] = []
        turtle_def_dists: list[float] = []
        pre_touch_min_red_to_blue = [float("inf"), float("inf")]
        one_defender_steps = [0, 0]
        two_defender_steps = [0, 0]
        any_two_defender_steps = 0
        consecutive_two = [0, 0]
        max_consecutive_two = [0, 0]
        max_tag_accumulator = [0.0, 0.0]
        accumulator_resets = [0, 0]
        reset_reason_counts = defaultdict(int)
        reset_relative_speeds: list[float] = []
        reset_events: list[dict[str, Any]] = []
        prev_exact_blue_tags = np.zeros((2, 2), dtype=bool)
        prev_turtle_target_idx = [None, None]
        path_cross_count = 0
        turtle_target_counts = {
            0: defaultdict(int),
            1: defaultdict(int),
        }
        last_info: dict[str, Any] = {}

        for step in range(int(max_decision_steps) + 5):
            env.step_async(_zero_action(env))
            _, _, done, infos = env.step_wait()
            last_info = infos[0] if infos else {}

            rx = core.red_x[0].detach().cpu().numpy()
            ry = core.red_y[0].detach().cpu().numpy()
            bx = core.blue_x[0].detach().cpu().numpy()
            by = core.blue_y[0].detach().cpu().numpy()
            red_alive = core.red_alive[0].detach().cpu().numpy().astype(bool)
            blue_alive = core.blue_alive[0].detach().cpu().numpy().astype(bool)
            red_tagged = core.red_tagged[0].detach().cpu().numpy().astype(bool)
            blue_tagged = core.blue_tagged[0].detach().cpu().numpy().astype(bool)
            red_carry = core.red_carrying[0].detach().cpu().numpy().astype(bool)
            blue_carry = core.blue_carrying[0].detach().cpu().numpy().astype(bool)
            red_flag_xy = core.red_flag_pos[0].detach().cpu().numpy().astype(float)
            blue_flag_xy = core.blue_flag_pos[0].detach().cpu().numpy().astype(float)
            before_red_touch = t_red_touch is None

            op7_red_lateral_sep_sum += abs(float(ry[0] - ry[1]))
            op7_red_lateral_sep_steps += 1
            blue_on_red_for_op7 = blue_alive & (bx > midline)
            if bool(blue_on_red_for_op7.all()):
                lane_sides = np.sign(by - (float(core.rows) * 0.5))
                op7_opposite_lane_penetration_steps += int(lane_sides[0] * lane_sides[1] < 0)
                red_to_blue = np.sqrt((rx[:, None] - bx[None, :]) ** 2 + (ry[:, None] - by[None, :]) ** 2)
                nearest_blue_for_red = np.argmin(red_to_blue, axis=1)
                if nearest_blue_for_red[0] == nearest_blue_for_red[1]:
                    op7_same_blue_target_steps += 1
                red_to_each_blue = red_to_blue.min(axis=0)
                audit_radius = float(core.cfg.tag_range_cells) * 1.5
                uncovered_blue = blue_on_red_for_op7 & (red_to_each_blue > audit_radius)
                if lane_sides[0] * lane_sides[1] < 0 and bool(uncovered_blue.any()):
                    op7_uncovered_lane_steps += 1
                    op7_consecutive_uncovered += 1
                    op7_max_consecutive_uncovered = max(op7_max_consecutive_uncovered, op7_consecutive_uncovered)
                    for blue_i in np.where(uncovered_blue)[0]:
                        op7_uncovered_progress.append(float(bx[blue_i] - midline))
                    op7_flag_touch_during_uncovered += int(bool(blue_carry.any()))
                else:
                    op7_consecutive_uncovered = 0
            else:
                op7_consecutive_uncovered = 0
            for blue_i in range(min(2, len(bx))):
                op7_min_blue_to_red_flag[blue_i] = min(
                    op7_min_blue_to_red_flag[blue_i],
                    float(np.hypot(bx[blue_i] - red_flag_xy[0], by[blue_i] - red_flag_xy[1])),
                )
            if hasattr(core, "_debug_red_target_x"):
                rtx = core._debug_red_target_x[0].detach().cpu().numpy().astype(float)
                rty = core._debug_red_target_y[0].detach().cpu().numpy().astype(float)
                blue_xy = np.stack([bx, by], axis=1)
                for red_i in range(min(2, len(rtx))):
                    d_to_blue_targets = np.sqrt((blue_xy[:, 0] - rtx[red_i]) ** 2 + (blue_xy[:, 1] - rty[red_i]) ** 2)
                    cur_target = int(np.argmin(d_to_blue_targets))
                    if op7_prev_red_target_idx[red_i] is not None and cur_target != op7_prev_red_target_idx[red_i]:
                        op7_red_target_switches[red_i] += 1
                    op7_prev_red_target_idx[red_i] = cur_target

            cur_red_score = int(core.red_score[0].item())
            cur_blue_score = int(core.blue_score[0].item())

            if blue_style == "BLUE_SPLIT":
                split_sep = float(np.hypot(bx[0] - bx[1], by[0] - by[1]))
                if not bool(blue_carry.any()):
                    split_sep_before_samples.append(split_sep)
                else:
                    split_sep_after_samples.append(split_sep)

                blue_to_red_flag = np.sqrt((bx - red_flag_xy[0]) ** 2 + (by - red_flag_xy[1]) ** 2)
                flag_touch_radius = float(getattr(core.cfg, "flag_grab_radius_cells", 1.2))
                touch_now = bool((blue_alive & (blue_to_red_flag <= flag_touch_radius)).any())
                if touch_now and not split_prev_touch_now:
                    split_flag_touch_count += 1
                    split_last_touch_step = int(step)
                split_prev_touch_now = touch_now

                newly_blue_pickup = (~prev_blue_carry) & blue_carry
                if split_active_pickup is None and bool(newly_blue_pickup.any()):
                    carrier_idx = int(np.where(newly_blue_pickup)[0][0])
                    teammate_idx = 1 - carrier_idx
                    teammate_lane_before = float(np.sign(prev_by[teammate_idx] - (float(core.rows) * 0.5)))
                    teammate_lane_after = float(np.sign(by[teammate_idx] - (float(core.rows) * 0.5)))
                    split_active_pickup = {
                        "blue_style": blue_style,
                        "red_style": red_style,
                        "map": map_name,
                        "episode_index": int(episode_index),
                        "episode_seed": int(episode_seed),
                        "touch_step": split_last_touch_step if split_last_touch_step is not None else int(step),
                        "pickup_step": int(step),
                        "carrier_idx": carrier_idx,
                        "end_step": "",
                        "loss_cause": "",
                        "carrier_lifetime": "",
                        "max_return_progress": 0.0,
                        "crossed_midfield": 0,
                        "reached_capture_zone": 0,
                        "captured": 0,
                        "carrier_loss_x": "",
                        "carrier_loss_y": "",
                        "min_dist_red0": float("inf"),
                        "min_dist_red1": float("inf"),
                        "teammate_dist_at_pickup": split_sep,
                        "teammate_lane_before": teammate_lane_before,
                        "teammate_lane_after": teammate_lane_after,
                        "red_retarget_latency": "",
                        "noncarrier_flag_pressure_steps": 0,
                        "noncarrier_support_steps": 0,
                    }

                if split_active_pickup is not None:
                    carrier_idx = int(split_active_pickup["carrier_idx"])
                    teammate_idx = 1 - carrier_idx
                    carrier_xy = np.array([float(bx[carrier_idx]), float(by[carrier_idx])])
                    teammate_xy = np.array([float(bx[teammate_idx]), float(by[teammate_idx])])
                    red_xy = np.stack([rx.astype(float), ry.astype(float)], axis=1)
                    red_dists = np.sqrt(((red_xy - carrier_xy[None, :]) ** 2).sum(axis=1))
                    split_active_pickup["min_dist_red0"] = min(float(split_active_pickup["min_dist_red0"]), float(red_dists[0]))
                    split_active_pickup["min_dist_red1"] = min(float(split_active_pickup["min_dist_red1"]), float(red_dists[1]))
                    split_active_pickup["max_return_progress"] = max(
                        float(split_active_pickup["max_return_progress"]),
                        float(red_flag_xy[0] - bx[carrier_idx]),
                    )
                    split_active_pickup["crossed_midfield"] = int(
                        bool(split_active_pickup["crossed_midfield"]) or bx[carrier_idx] < midline
                    )
                    split_active_pickup["reached_capture_zone"] = int(
                        bool(split_active_pickup["reached_capture_zone"])
                        or np.hypot(bx[carrier_idx] - blue_flag_xy[0], by[carrier_idx] - blue_flag_xy[1])
                        <= flag_touch_radius
                    )
                    if np.hypot(teammate_xy[0] - red_flag_xy[0], teammate_xy[1] - red_flag_xy[1]) <= flag_touch_radius * 2.0:
                        split_active_pickup["noncarrier_flag_pressure_steps"] = int(
                            split_active_pickup["noncarrier_flag_pressure_steps"]
                        ) + 1
                    if np.linalg.norm(teammate_xy - carrier_xy) <= float(core.cfg.tag_range_cells) * 1.5:
                        split_active_pickup["noncarrier_support_steps"] = int(
                            split_active_pickup["noncarrier_support_steps"]
                        ) + 1
                    if hasattr(core, "_debug_red_target_x") and split_active_pickup["red_retarget_latency"] == "":
                        rtx = core._debug_red_target_x[0].detach().cpu().numpy().astype(float)
                        rty = core._debug_red_target_y[0].detach().cpu().numpy().astype(float)
                        red_target_dist = np.sqrt((rtx - carrier_xy[0]) ** 2 + (rty - carrier_xy[1]) ** 2)
                        if bool((red_target_dist <= float(core.cfg.tag_range_cells) * 2.0).any()):
                            split_active_pickup["red_retarget_latency"] = int(step) - int(split_active_pickup["pickup_step"])

                    end_cause = None
                    if cur_blue_score > prev_blue_score:
                        split_active_pickup["captured"] = 1
                        end_cause = "capture"
                    elif bool(prev_blue_carry[carrier_idx]) and not bool(blue_carry[carrier_idx]):
                        end_cause = "tag" if bool(blue_tagged[carrier_idx]) else "other"
                    if end_cause is not None:
                        split_active_pickup["end_step"] = int(step)
                        split_active_pickup["loss_cause"] = end_cause
                        split_active_pickup["carrier_lifetime"] = int(step) - int(split_active_pickup["pickup_step"])
                        split_active_pickup["carrier_loss_x"] = float(bx[carrier_idx])
                        split_active_pickup["carrier_loss_y"] = float(by[carrier_idx])
                        split_pickup_events.append(split_active_pickup)
                        split_active_pickup = None

            red_on_blue_side = red_alive & (rx < midline)
            blue_on_red_side = blue_alive & (bx > midline)
            t_red_mid = _first_time(t_red_mid, bool(red_on_blue_side.any()), step)
            t_both_red_forward = _first_time(
                t_both_red_forward,
                bool((red_on_blue_side | (~red_alive)).all() and red_alive.any()),
                step,
            )
            t_red_touch = _first_time(t_red_touch, bool(red_carry.any()), step)
            t_blue_touch = _first_time(t_blue_touch, bool(blue_carry.any()), step)
            t_blue_counter = _first_time(t_blue_counter, bool(blue_on_red_side.any()), step)

            t_red_cap = _first_time(t_red_cap, cur_red_score > prev_red_score, step)
            t_blue_cap = _first_time(t_blue_cap, cur_blue_score > prev_blue_score, step)
            regroup_active = bool(
                hasattr(core, "bt_op6_regroup_ticks")
                and int(core.bt_op6_regroup_ticks[0].item()) > 0
            )
            op6_regroup_active_steps += int(regroup_active)
            if regroup_active and bool(blue_carry.any()):
                op6_regroup_blue_touch_count += 1
            if regroup_active and cur_blue_score > prev_blue_score:
                op6_regroup_blue_capture_count += 1

            newly_red_dead = prev_red_alive & (~red_alive)
            newly_blue_dead = prev_blue_alive & (~blue_alive)
            newly_red_tagged = (~prev_red_tagged) & red_tagged
            newly_blue_tagged = (~prev_blue_tagged) & blue_tagged
            red_deaths += int(newly_red_dead.sum())
            blue_deaths += int(newly_blue_dead.sum())
            red_carrier_deaths += int((newly_red_dead & prev_red_carry).sum())
            blue_carrier_deaths += int((newly_blue_dead & prev_blue_carry).sum())
            red_tags += int(newly_red_tagged.sum())
            blue_tags += int(newly_blue_tagged.sum())
            red_carrier_tags += int((newly_red_tagged & prev_red_carry).sum())
            blue_carrier_tags += int((newly_blue_tagged & prev_blue_carry).sum())
            for red_i in np.where(newly_red_tagged & prev_red_carry)[0]:
                post_tag_events.append(
                    {
                        "step": int(step),
                        "blue_score": int(prev_blue_score),
                        "blue_counter_step": None,
                        "blue_touch_step": None,
                        "blue_capture_step": None,
                        "red_reentry_step": None,
                    }
                )

            for event in post_tag_events:
                if int(step) <= int(event["step"]):
                    continue
                if event["blue_counter_step"] is None and bool(blue_on_red_side.any()):
                    event["blue_counter_step"] = int(step)
                if event["blue_touch_step"] is None and bool(blue_carry.any()):
                    event["blue_touch_step"] = int(step)
                if event["red_reentry_step"] is None and bool(red_on_blue_side.any()):
                    event["red_reentry_step"] = int(step)
                if event["blue_capture_step"] is None and cur_blue_score > int(event["blue_score"]):
                    event["blue_capture_step"] = int(step)

            if blue_style == "BLUE_TURTLE":
                d = np.sqrt((bx - blue_flag_x) ** 2 + (by - blue_flag_y) ** 2)
                turtle_def_dists.extend([float(x) for x in d])
                if before_red_touch:
                    pair_dist = np.sqrt((rx[:, None] - bx[None, :]) ** 2 + (ry[:, None] - by[None, :]) ** 2)
                    tag_range = float(core.cfg.tag_range_cells)
                    blue_can_tag = (~core.blue_tagged[0].detach().cpu().numpy().astype(bool)) & (bx < midline)
                    red_targetable = (~core.red_tagged[0].detach().cpu().numpy().astype(bool)) & (rx < midline)
                    exact_blue_tags = (pair_dist <= tag_range) & blue_can_tag[None, :] & red_targetable[:, None]
                    for red_i in range(min(2, pair_dist.shape[0])):
                        pre_touch_min_red_to_blue[red_i] = min(
                            pre_touch_min_red_to_blue[red_i],
                            float(pair_dist[red_i].min()),
                        )
                        in_range_count = int(exact_blue_tags[red_i].sum())
                        one_defender_steps[red_i] += int(in_range_count >= 1)
                        two_defender_steps[red_i] += int(in_range_count >= 2)
                        if in_range_count >= 2:
                            consecutive_two[red_i] += 1
                        else:
                            if consecutive_two[red_i] > 0:
                                accumulator_resets[red_i] += 1
                            consecutive_two[red_i] = 0
                        max_consecutive_two[red_i] = max(max_consecutive_two[red_i], consecutive_two[red_i])
                        max_tag_accumulator[red_i] = max(
                            max_tag_accumulator[red_i],
                            float(core.red_tag_pressure_time[0, red_i].item()),
                        )
                    any_two_defender_steps += int(any(v >= 2 for v in [(pair_dist[i] <= tag_range).sum() for i in range(min(2, pair_dist.shape[0]))]))
                    for red_i in range(min(2, pair_dist.shape[0])):
                        for blue_i in range(min(2, pair_dist.shape[1])):
                            if _segments_intersect(
                                np.array([prev_rx[red_i], prev_ry[red_i]]),
                                np.array([rx[red_i], ry[red_i]]),
                                np.array([prev_bx[blue_i], prev_by[blue_i]]),
                                np.array([bx[blue_i], by[blue_i]]),
                            ):
                                path_cross_count += 1
                    if hasattr(core, "_debug_blue_target_x"):
                        tx = core._debug_blue_target_x[0].detach().cpu().numpy().astype(float)
                        ty = core._debug_blue_target_y[0].detach().cpu().numpy().astype(float)
                        red_xy = np.stack([rx, ry], axis=1)
                        cur_turtle_target_idx = [None, None]
                        for blue_i in range(min(2, tx.shape[0])):
                            name = _closest_named_target(
                                float(tx[blue_i]),
                                float(ty[blue_i]),
                                red_xy=red_xy,
                                blue_flag_xy=np.array([blue_flag_x, blue_flag_y]),
                                red_flag_xy=red_flag_xy,
                            )
                            cur_turtle_target_idx[blue_i] = _target_index_from_name(name)
                            turtle_target_counts[blue_i][name] += 1
                    else:
                        cur_turtle_target_idx = [None, None]

                    for red_i in range(min(2, exact_blue_tags.shape[0])):
                        if prev_exact_blue_tags[red_i].sum() >= 2 and exact_blue_tags[red_i].sum() < 2:
                            missing = prev_exact_blue_tags[red_i] & (~exact_blue_tags[red_i])
                            red_v = np.array([rx[red_i] - prev_rx[red_i], ry[red_i] - prev_ry[red_i]])
                            blue_rel_speeds = []
                            for blue_i in range(min(2, exact_blue_tags.shape[1])):
                                blue_v = np.array([bx[blue_i] - prev_bx[blue_i], by[blue_i] - prev_by[blue_i]])
                                blue_rel_speeds.append(float(np.linalg.norm(red_v - blue_v)))
                            defender0_left = bool(len(missing) > 0 and missing[0])
                            defender1_left = bool(len(missing) > 1 and missing[1])
                            red_left_side = bool(not red_targetable[red_i])
                            defender_left_side = bool(not bool(blue_can_tag.all()))
                            blue_tagged_now = core.blue_tagged[0].detach().cpu().numpy().astype(bool)
                            defender_tagged = bool((prev_exact_blue_tags[red_i] & blue_tagged_now).any())
                            red_target_tagged = bool(core.red_tagged[0, red_i].item())
                            prev_targets = [v for v in prev_turtle_target_idx if v is not None]
                            cur_targets = [v for v in cur_turtle_target_idx if v is not None]
                            target_switched = bool(prev_targets and cur_targets and any(v != red_i for v in cur_targets))
                            carrier_turn = bool(
                                red_carry[red_i]
                                and (
                                    abs(float(ry[red_i] - prev_ry[red_i])) > 0.35
                                )
                            )
                            other = bool(
                                not (
                                    defender0_left
                                    or defender1_left
                                    or red_left_side
                                    or defender_left_side
                                    or target_switched
                                    or defender_tagged
                                    or red_target_tagged
                                    or carrier_turn
                                )
                            )
                            if defender0_left:
                                reset_reason_counts["defender0_left_radius"] += 1
                            if defender1_left:
                                reset_reason_counts["defender1_left_radius"] += 1
                            if red_left_side:
                                reset_reason_counts["red_left_blue_side"] += 1
                            if defender_left_side:
                                reset_reason_counts["defender_left_blue_side"] += 1
                            if target_switched:
                                reset_reason_counts["target_switched"] += 1
                            if defender_tagged:
                                reset_reason_counts["defender_became_tagged"] += 1
                            if red_target_tagged:
                                reset_reason_counts["red_target_became_tagged"] += 1
                            if carrier_turn:
                                reset_reason_counts["carrier_turn_or_lane_change"] += 1
                            if other:
                                reset_reason_counts["other"] += 1
                            reset_relative_speeds.extend(blue_rel_speeds)
                            reset_events.append(
                                {
                                    "blue_style": blue_style,
                                    "red_style": red_style,
                                    "map": map_name,
                                    "episode_index": int(episode_index),
                                    "episode_seed": int(episode_seed),
                                    "step": int(step),
                                    "red_index": int(red_i),
                                    "defender0_left_radius": int(defender0_left),
                                    "defender1_left_radius": int(defender1_left),
                                    "red_left_blue_side": int(red_left_side),
                                    "defender_left_blue_side": int(defender_left_side),
                                    "target_switched": int(target_switched),
                                    "defender_became_tagged": int(defender_tagged),
                                    "red_target_became_tagged": int(red_target_tagged),
                                    "carrier_turn_or_lane_change": int(carrier_turn),
                                    "other": int(other),
                                    "relative_speed_def0": blue_rel_speeds[0] if len(blue_rel_speeds) > 0 else "",
                                    "relative_speed_def1": blue_rel_speeds[1] if len(blue_rel_speeds) > 1 else "",
                                    "red_x": float(rx[red_i]),
                                    "red_y": float(ry[red_i]),
                                    "red_carrying": int(red_carry[red_i]),
                                    "def0_target": cur_turtle_target_idx[0] if len(cur_turtle_target_idx) > 0 else "",
                                    "def1_target": cur_turtle_target_idx[1] if len(cur_turtle_target_idx) > 1 else "",
                                }
                            )
                    prev_exact_blue_tags = exact_blue_tags.copy()
                    prev_turtle_target_idx = cur_turtle_target_idx

            prev_red_alive = red_alive
            prev_blue_alive = blue_alive
            prev_red_tagged = red_tagged
            prev_blue_tagged = blue_tagged
            prev_red_carry = red_carry
            prev_blue_carry = blue_carry
            prev_red_score = cur_red_score
            prev_blue_score = cur_blue_score
            prev_regroup_active = regroup_active
            prev_rx = rx.astype(float)
            prev_ry = ry.astype(float)
            prev_bx = bx.astype(float)
            prev_by = by.astype(float)

            if bool(done.any()):
                break

        if split_active_pickup is not None:
            carrier_idx = int(split_active_pickup["carrier_idx"])
            split_active_pickup["end_step"] = int(step)
            split_active_pickup["loss_cause"] = "episode_end"
            split_active_pickup["carrier_lifetime"] = int(step) - int(split_active_pickup["pickup_step"])
            split_active_pickup["carrier_loss_x"] = float(prev_bx[carrier_idx])
            split_active_pickup["carrier_loss_y"] = float(prev_by[carrier_idx])
            split_pickup_events.append(split_active_pickup)
            split_active_pickup = None

        ep = dict(last_info.get("episode_result", last_info))
        blue_score = int(ep.get("blue_score", int(core.blue_score[0].item())))
        red_score = int(ep.get("red_score", int(core.red_score[0].item())))
        split_pickup_count = len(split_pickup_events)
        split_capture_count = sum(int(e.get("captured", 0)) for e in split_pickup_events)
        split_lifetimes = [float(e["carrier_lifetime"]) for e in split_pickup_events if e.get("carrier_lifetime") not in ("", None)]
        split_progress = [float(e["max_return_progress"]) for e in split_pickup_events if e.get("max_return_progress") not in ("", None)]
        split_teammate_dist = [
            float(e["teammate_dist_at_pickup"])
            for e in split_pickup_events
            if e.get("teammate_dist_at_pickup") not in ("", None)
        ]
        split_retarget = [
            float(e["red_retarget_latency"])
            for e in split_pickup_events
            if e.get("red_retarget_latency") not in ("", None)
        ]
        row = {
            "blue_style": blue_style,
            "red_style": red_style,
            "map": map_name,
            "episode_index": int(episode_index),
            "episode_seed": int(episode_seed),
            "blue_score": blue_score,
            "red_score": red_score,
            "win_margin": blue_score - red_score,
            "steps": int(ep.get("decision_steps", step + 1)),
            "time_first_red_midfield_cross": t_red_mid,
            "time_both_red_enter_blue_territory": t_both_red_forward,
            "time_first_red_flag_touch": t_red_touch,
            "time_first_red_capture": t_red_cap,
            "time_blue_counterattack_start": t_blue_counter,
            "time_first_blue_flag_touch": t_blue_touch,
            "time_first_blue_capture": t_blue_cap,
            "red_death_count": red_deaths,
            "blue_death_count": blue_deaths,
            "red_carrier_death_count": red_carrier_deaths,
            "blue_carrier_death_count": blue_carrier_deaths,
            "red_tag_count": red_tags,
            "blue_tag_count": blue_tags,
            "red_carrier_tag_count": red_carrier_tags,
            "blue_carrier_tag_count": blue_carrier_tags,
            "post_tag_carrier_event_count": len(post_tag_events),
            "post_tag_counterattack_launch_count": sum(e["blue_counter_step"] is not None for e in post_tag_events),
            "post_tag_blue_flag_touch_count": sum(e["blue_touch_step"] is not None for e in post_tag_events),
            "post_tag_blue_capture_before_red_reentry_count": sum(
                e["blue_capture_step"] is not None
                and (e["red_reentry_step"] is None or int(e["blue_capture_step"]) < int(e["red_reentry_step"]))
                for e in post_tag_events
            ),
            "post_tag_mean_steps_to_blue_flag_touch": (
                float(np.mean([int(e["blue_touch_step"]) - int(e["step"]) for e in post_tag_events if e["blue_touch_step"] is not None]))
                if any(e["blue_touch_step"] is not None for e in post_tag_events)
                else ""
            ),
            "post_tag_mean_steps_to_red_reentry": (
                float(np.mean([int(e["red_reentry_step"]) - int(e["step"]) for e in post_tag_events if e["red_reentry_step"] is not None]))
                if any(e["red_reentry_step"] is not None for e in post_tag_events)
                else ""
            ),
            "op6_regroup_active_steps": op6_regroup_active_steps,
            "op6_regroup_blue_flag_touch_count": op6_regroup_blue_touch_count,
            "op6_regroup_blue_capture_count": op6_regroup_blue_capture_count,
            "op7_mean_red_lateral_separation": (
                op7_red_lateral_sep_sum / max(1, op7_red_lateral_sep_steps)
            ),
            "op7_blue_opposite_lane_penetration_steps": op7_opposite_lane_penetration_steps,
            "op7_min_blue0_to_red_flag": op7_min_blue_to_red_flag[0] if np.isfinite(op7_min_blue_to_red_flag[0]) else "",
            "op7_min_blue1_to_red_flag": op7_min_blue_to_red_flag[1] if np.isfinite(op7_min_blue_to_red_flag[1]) else "",
            "op7_red0_target_switches": op7_red_target_switches[0],
            "op7_red1_target_switches": op7_red_target_switches[1],
            "op7_both_red_target_same_blue_steps": op7_same_blue_target_steps,
            "op7_uncovered_lane_steps": op7_uncovered_lane_steps,
            "op7_max_consecutive_uncovered_lane_steps": op7_max_consecutive_uncovered,
            "op7_mean_uncovered_blue_progress": float(np.mean(op7_uncovered_progress)) if op7_uncovered_progress else "",
            "op7_flag_touch_during_uncovered_lane": op7_flag_touch_during_uncovered,
            "split_flag_touch_count": split_flag_touch_count,
            "split_pickup_count": split_pickup_count,
            "split_touches_per_pickup": (
                float(split_flag_touch_count) / float(split_pickup_count) if split_pickup_count else ""
            ),
            "split_capture_count": split_capture_count,
            "split_capture_given_pickup": (
                float(split_capture_count) / float(split_pickup_count) if split_pickup_count else ""
            ),
            "split_mean_carrier_lifetime": float(np.mean(split_lifetimes)) if split_lifetimes else "",
            "split_mean_max_return_progress": float(np.mean(split_progress)) if split_progress else "",
            "split_mean_teammate_dist_at_pickup": float(np.mean(split_teammate_dist)) if split_teammate_dist else "",
            "split_mean_sep_before_pickup": float(np.mean(split_sep_before_samples)) if split_sep_before_samples else "",
            "split_mean_sep_after_pickup": float(np.mean(split_sep_after_samples)) if split_sep_after_samples else "",
            "split_converged_after_pickup_count": sum(
                int(float(e.get("noncarrier_support_steps", 0)) > 0) for e in split_pickup_events
            ),
            "split_noncarrier_flag_pressure_steps": sum(
                int(e.get("noncarrier_flag_pressure_steps", 0)) for e in split_pickup_events
            ),
            "split_noncarrier_support_steps": sum(
                int(e.get("noncarrier_support_steps", 0)) for e in split_pickup_events
            ),
            "split_mean_red_retarget_latency_after_pickup": float(np.mean(split_retarget)) if split_retarget else "",
            "split_carrier_loss_tag_count": sum(e.get("loss_cause") == "tag" for e in split_pickup_events),
            "split_carrier_loss_capture_count": sum(e.get("loss_cause") == "capture" for e in split_pickup_events),
            "split_carrier_loss_other_count": sum(e.get("loss_cause") not in ("tag", "capture") for e in split_pickup_events),
            "pre_touch_min_red0_to_blue": pre_touch_min_red_to_blue[0] if np.isfinite(pre_touch_min_red_to_blue[0]) else "",
            "pre_touch_min_red1_to_blue": pre_touch_min_red_to_blue[1] if np.isfinite(pre_touch_min_red_to_blue[1]) else "",
            "pre_touch_min_any_red_to_blue": min(pre_touch_min_red_to_blue) if np.isfinite(min(pre_touch_min_red_to_blue)) else "",
            "pre_touch_steps_red0_one_defender_in_tag_range": one_defender_steps[0],
            "pre_touch_steps_red1_one_defender_in_tag_range": one_defender_steps[1],
            "pre_touch_steps_red0_two_defenders_in_tag_range": two_defender_steps[0],
            "pre_touch_steps_red1_two_defenders_in_tag_range": two_defender_steps[1],
            "pre_touch_steps_any_red_two_defenders_in_tag_range": any_two_defender_steps,
            "pre_touch_max_consecutive_red0_two_defenders": max_consecutive_two[0],
            "pre_touch_max_consecutive_red1_two_defenders": max_consecutive_two[1],
            "pre_touch_max_consecutive_any_red_two_defenders": max(max_consecutive_two),
            "pre_touch_max_red0_tag_accumulator": max_tag_accumulator[0],
            "pre_touch_max_red1_tag_accumulator": max_tag_accumulator[1],
            "tag_threshold_seconds": float(core.cfg.tag_channel_seconds),
            "tag_required_consecutive_steps": int(np.ceil(float(core.cfg.tag_channel_seconds) / float(core.dt))),
            "pre_touch_red0_accumulator_reset_count": accumulator_resets[0],
            "pre_touch_red1_accumulator_reset_count": accumulator_resets[1],
            "pre_touch_reset_defender0_left_radius": reset_reason_counts["defender0_left_radius"],
            "pre_touch_reset_defender1_left_radius": reset_reason_counts["defender1_left_radius"],
            "pre_touch_reset_red_left_blue_side": reset_reason_counts["red_left_blue_side"],
            "pre_touch_reset_defender_left_blue_side": reset_reason_counts["defender_left_blue_side"],
            "pre_touch_reset_target_switched": reset_reason_counts["target_switched"],
            "pre_touch_reset_defender_became_tagged": reset_reason_counts["defender_became_tagged"],
            "pre_touch_reset_red_target_became_tagged": reset_reason_counts["red_target_became_tagged"],
            "pre_touch_reset_carrier_turn_or_lane_change": reset_reason_counts["carrier_turn_or_lane_change"],
            "pre_touch_reset_other": reset_reason_counts["other"],
            "pre_touch_mean_reset_relative_speed": float(np.mean(reset_relative_speeds)) if reset_relative_speeds else "",
            "pre_touch_path_cross_count": path_cross_count,
            "turtle_def0_target_red0_steps": turtle_target_counts[0]["red0"],
            "turtle_def0_target_red1_steps": turtle_target_counts[0]["red1"],
            "turtle_def0_target_blue_flag_steps": turtle_target_counts[0]["blue_flag"],
            "turtle_def0_target_red_flag_steps": turtle_target_counts[0]["red_flag"],
            "turtle_def1_target_red0_steps": turtle_target_counts[1]["red0"],
            "turtle_def1_target_red1_steps": turtle_target_counts[1]["red1"],
            "turtle_def1_target_blue_flag_steps": turtle_target_counts[1]["blue_flag"],
            "turtle_def1_target_red_flag_steps": turtle_target_counts[1]["red_flag"],
            "min_turtle_defender_dist_to_blue_flag": min(turtle_def_dists) if turtle_def_dists else "",
            "mean_turtle_defender_dist_to_blue_flag": float(np.mean(turtle_def_dists)) if turtle_def_dists else "",
            "final_both_red_forward": int(t_both_red_forward is not None),
            "final_blue_had_counterattack": int(t_blue_counter is not None),
        }
        return row, reset_events, split_pickup_events
    finally:
        env.close()


def _mean_defined(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [float(r[key]) for r in rows if r.get(key) not in ("", None)]
    return float(np.mean(vals)) if vals else None


def _classify_turtle(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "NO_TURTLE_ROWS"
    red_caps = sum(r["time_first_red_capture"] not in ("", None) for r in rows)
    red_stop_eps = sum(
        int(r.get("red_death_count", 0)) > 0
        or int(r.get("red_tag_count", 0)) > 0
        or int(r.get("red_carrier_tag_count", 0)) > 0
        for r in rows
    )
    red_carrier_stop_eps = sum(int(r.get("red_carrier_tag_count", 0)) > 0 for r in rows)
    blue_counter_eps = sum(r["time_blue_counterattack_start"] not in ("", None) for r in rows)
    blue_caps = sum(r["time_first_blue_capture"] not in ("", None) for r in rows)
    n = len(rows)
    if red_caps >= max(1, int(0.75 * n)) and red_stop_eps <= int(0.25 * n):
        return "CASE_1_TURTLE_CANNOT_STOP_INITIAL_RUSH"
    if red_caps >= max(1, int(0.50 * n)) and red_carrier_stop_eps >= max(1, int(0.50 * n)):
        return "CASE_2_TURTLE_STOPS_RUSH_BUT_OP6_STILL_CONVERTS"
    if red_caps >= max(1, int(0.50 * n)) and red_stop_eps > int(0.25 * n):
        return "CASE_2_TURTLE_STOPS_PART_OF_RUSH_SECOND_PRESSURE_SCORES"
    if red_caps < int(0.50 * n) and blue_caps == 0:
        return "CASE_3_TURTLE_DEFENDS_BUT_DOES_NOT_SCORE"
    if blue_counter_eps == 0:
        return "CASE_3_TURTLE_TOO_PASSIVE_NO_COUNTERATTACK"
    return "MIXED_FAILURE_REQUIRES_TRAJECTORY_REVIEW"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ROW_FIELDS)
        writer.writeheader()
        writer.writerows([{k: r.get(k, "") for k in ROW_FIELDS} for r in rows])


def _write_reset_events_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESET_EVENT_FIELDS)
        writer.writeheader()
        writer.writerows([{k: r.get(k, "") for k in RESET_EVENT_FIELDS} for r in rows])


def _write_split_pickup_events_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SPLIT_PICKUP_EVENT_FIELDS)
        writer.writeheader()
        writer.writerows([{k: r.get(k, "") for k in SPLIT_PICKUP_EVENT_FIELDS} for r in rows])


def main() -> int:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    red_index = DEFAULT_REDS.index(args.red_style) if args.red_style in DEFAULT_REDS else 0
    rows: list[dict[str, Any]] = []
    reset_events: list[dict[str, Any]] = []
    split_pickup_events: list[dict[str, Any]] = []
    total = len(args.blue_styles) * int(args.episodes)
    count = 0
    for ep_i in range(int(args.episodes)):
        seed = _episode_seed(args.base_seed, red_index=red_index, map_index=0, episode_index=ep_i)
        for style in args.blue_styles:
            row, ep_reset_events, ep_split_pickup_events = _run_episode(
                blue_style=style,
                red_style=args.red_style,
                map_name=args.map_name,
                episode_index=ep_i,
                episode_seed=seed,
                max_decision_steps=args.max_decision_steps,
                device=args.device,
            )
            rows.append(row)
            reset_events.extend(ep_reset_events)
            split_pickup_events.extend(ep_split_pickup_events)
            count += 1
            if count % max(1, int(args.progress_every)) == 0:
                print(f"[op6 timeline] {count}/{total} episodes")

    csv_path = out_dir / "timeline_rows.csv"
    _write_csv(csv_path, rows)
    reset_events_path = out_dir / "reset_events.csv"
    _write_reset_events_csv(reset_events_path, reset_events)
    split_pickup_events_path = out_dir / "split_pickup_events.csv"
    _write_split_pickup_events_csv(split_pickup_events_path, split_pickup_events)

    by_style: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_style[str(row["blue_style"])].append(row)
    summary = {
        "protocol": "op6_failure_timeline",
        "red_style": args.red_style,
        "map": args.map_name,
        "episodes": int(args.episodes),
        "base_seed": int(args.base_seed),
        "rows_csv": str(csv_path),
        "reset_events_csv": str(reset_events_path),
        "split_pickup_events_csv": str(split_pickup_events_path),
        "by_style": {},
    }
    for style, style_rows in by_style.items():
        summary["by_style"][style] = {
            "n": len(style_rows),
            "win_rate": sum(int(r["win_margin"]) > 0 for r in style_rows) / max(1, len(style_rows)),
            "mean_margin": _mean_defined(style_rows, "win_margin"),
            "mean_time_first_red_midfield_cross": _mean_defined(style_rows, "time_first_red_midfield_cross"),
            "mean_time_both_red_enter_blue_territory": _mean_defined(style_rows, "time_both_red_enter_blue_territory"),
            "mean_time_first_red_flag_touch": _mean_defined(style_rows, "time_first_red_flag_touch"),
            "mean_time_first_red_capture": _mean_defined(style_rows, "time_first_red_capture"),
            "mean_time_blue_counterattack_start": _mean_defined(style_rows, "time_blue_counterattack_start"),
            "mean_time_first_blue_flag_touch": _mean_defined(style_rows, "time_first_blue_flag_touch"),
            "mean_time_first_blue_capture": _mean_defined(style_rows, "time_first_blue_capture"),
            "mean_red_death_count": _mean_defined(style_rows, "red_death_count"),
            "mean_blue_death_count": _mean_defined(style_rows, "blue_death_count"),
            "mean_red_carrier_death_count": _mean_defined(style_rows, "red_carrier_death_count"),
            "mean_blue_carrier_death_count": _mean_defined(style_rows, "blue_carrier_death_count"),
            "mean_red_tag_count": _mean_defined(style_rows, "red_tag_count"),
            "mean_blue_tag_count": _mean_defined(style_rows, "blue_tag_count"),
            "mean_red_carrier_tag_count": _mean_defined(style_rows, "red_carrier_tag_count"),
            "mean_blue_carrier_tag_count": _mean_defined(style_rows, "blue_carrier_tag_count"),
            "mean_post_tag_carrier_event_count": _mean_defined(style_rows, "post_tag_carrier_event_count"),
            "mean_post_tag_counterattack_launch_count": _mean_defined(style_rows, "post_tag_counterattack_launch_count"),
            "mean_post_tag_blue_flag_touch_count": _mean_defined(style_rows, "post_tag_blue_flag_touch_count"),
            "mean_post_tag_blue_capture_before_red_reentry_count": _mean_defined(style_rows, "post_tag_blue_capture_before_red_reentry_count"),
            "mean_post_tag_steps_to_blue_flag_touch": _mean_defined(style_rows, "post_tag_mean_steps_to_blue_flag_touch"),
            "mean_post_tag_steps_to_red_reentry": _mean_defined(style_rows, "post_tag_mean_steps_to_red_reentry"),
            "mean_op6_regroup_active_steps": _mean_defined(style_rows, "op6_regroup_active_steps"),
            "mean_op6_regroup_blue_flag_touch_count": _mean_defined(style_rows, "op6_regroup_blue_flag_touch_count"),
            "mean_op6_regroup_blue_capture_count": _mean_defined(style_rows, "op6_regroup_blue_capture_count"),
            "mean_op7_red_lateral_separation": _mean_defined(style_rows, "op7_mean_red_lateral_separation"),
            "mean_op7_blue_opposite_lane_penetration_steps": _mean_defined(style_rows, "op7_blue_opposite_lane_penetration_steps"),
            "mean_op7_min_blue0_to_red_flag": _mean_defined(style_rows, "op7_min_blue0_to_red_flag"),
            "mean_op7_min_blue1_to_red_flag": _mean_defined(style_rows, "op7_min_blue1_to_red_flag"),
            "mean_op7_red0_target_switches": _mean_defined(style_rows, "op7_red0_target_switches"),
            "mean_op7_red1_target_switches": _mean_defined(style_rows, "op7_red1_target_switches"),
            "mean_op7_both_red_target_same_blue_steps": _mean_defined(style_rows, "op7_both_red_target_same_blue_steps"),
            "mean_op7_uncovered_lane_steps": _mean_defined(style_rows, "op7_uncovered_lane_steps"),
            "mean_op7_max_consecutive_uncovered_lane_steps": _mean_defined(style_rows, "op7_max_consecutive_uncovered_lane_steps"),
            "mean_op7_uncovered_blue_progress": _mean_defined(style_rows, "op7_mean_uncovered_blue_progress"),
            "mean_op7_flag_touch_during_uncovered_lane": _mean_defined(style_rows, "op7_flag_touch_during_uncovered_lane"),
            "mean_split_flag_touch_count": _mean_defined(style_rows, "split_flag_touch_count"),
            "mean_split_pickup_count": _mean_defined(style_rows, "split_pickup_count"),
            "mean_split_touches_per_pickup": _mean_defined(style_rows, "split_touches_per_pickup"),
            "mean_split_capture_count": _mean_defined(style_rows, "split_capture_count"),
            "mean_split_capture_given_pickup": _mean_defined(style_rows, "split_capture_given_pickup"),
            "mean_split_carrier_lifetime": _mean_defined(style_rows, "split_mean_carrier_lifetime"),
            "mean_split_max_return_progress": _mean_defined(style_rows, "split_mean_max_return_progress"),
            "mean_split_teammate_dist_at_pickup": _mean_defined(style_rows, "split_mean_teammate_dist_at_pickup"),
            "mean_split_sep_before_pickup": _mean_defined(style_rows, "split_mean_sep_before_pickup"),
            "mean_split_sep_after_pickup": _mean_defined(style_rows, "split_mean_sep_after_pickup"),
            "mean_split_converged_after_pickup_count": _mean_defined(style_rows, "split_converged_after_pickup_count"),
            "mean_split_noncarrier_flag_pressure_steps": _mean_defined(style_rows, "split_noncarrier_flag_pressure_steps"),
            "mean_split_noncarrier_support_steps": _mean_defined(style_rows, "split_noncarrier_support_steps"),
            "mean_split_red_retarget_latency_after_pickup": _mean_defined(style_rows, "split_mean_red_retarget_latency_after_pickup"),
            "mean_split_carrier_loss_tag_count": _mean_defined(style_rows, "split_carrier_loss_tag_count"),
            "mean_split_carrier_loss_capture_count": _mean_defined(style_rows, "split_carrier_loss_capture_count"),
            "mean_split_carrier_loss_other_count": _mean_defined(style_rows, "split_carrier_loss_other_count"),
            "mean_pre_touch_min_any_red_to_blue": _mean_defined(style_rows, "pre_touch_min_any_red_to_blue"),
            "mean_pre_touch_steps_any_red_two_defenders_in_tag_range": _mean_defined(style_rows, "pre_touch_steps_any_red_two_defenders_in_tag_range"),
            "mean_pre_touch_max_consecutive_any_red_two_defenders": _mean_defined(style_rows, "pre_touch_max_consecutive_any_red_two_defenders"),
            "mean_pre_touch_max_red0_tag_accumulator": _mean_defined(style_rows, "pre_touch_max_red0_tag_accumulator"),
            "mean_pre_touch_max_red1_tag_accumulator": _mean_defined(style_rows, "pre_touch_max_red1_tag_accumulator"),
            "mean_pre_touch_red0_accumulator_reset_count": _mean_defined(style_rows, "pre_touch_red0_accumulator_reset_count"),
            "mean_pre_touch_red1_accumulator_reset_count": _mean_defined(style_rows, "pre_touch_red1_accumulator_reset_count"),
            "mean_pre_touch_reset_defender0_left_radius": _mean_defined(style_rows, "pre_touch_reset_defender0_left_radius"),
            "mean_pre_touch_reset_defender1_left_radius": _mean_defined(style_rows, "pre_touch_reset_defender1_left_radius"),
            "mean_pre_touch_reset_red_left_blue_side": _mean_defined(style_rows, "pre_touch_reset_red_left_blue_side"),
            "mean_pre_touch_reset_defender_left_blue_side": _mean_defined(style_rows, "pre_touch_reset_defender_left_blue_side"),
            "mean_pre_touch_reset_target_switched": _mean_defined(style_rows, "pre_touch_reset_target_switched"),
            "mean_pre_touch_reset_defender_became_tagged": _mean_defined(style_rows, "pre_touch_reset_defender_became_tagged"),
            "mean_pre_touch_reset_red_target_became_tagged": _mean_defined(style_rows, "pre_touch_reset_red_target_became_tagged"),
            "mean_pre_touch_reset_carrier_turn_or_lane_change": _mean_defined(style_rows, "pre_touch_reset_carrier_turn_or_lane_change"),
            "mean_pre_touch_reset_other": _mean_defined(style_rows, "pre_touch_reset_other"),
            "mean_pre_touch_reset_relative_speed": _mean_defined(style_rows, "pre_touch_mean_reset_relative_speed"),
            "mean_pre_touch_path_cross_count": _mean_defined(style_rows, "pre_touch_path_cross_count"),
        }
    turtle_rows = by_style.get("BLUE_TURTLE", [])
    summary["turtle_failure_classification"] = _classify_turtle(turtle_rows)

    json_path = out_dir / "timeline_summary.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Artifacts in: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
