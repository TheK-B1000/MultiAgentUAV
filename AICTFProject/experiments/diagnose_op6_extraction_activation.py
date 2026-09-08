#!/usr/bin/env python3
"""OP6 extraction activation chain vs BLUE_RUSH (map_a) — observe only.

Does not change BT routing, recovery, or lane_amplitude_frac.
Asks: when OP6 picks up against RUSH V3, does extraction actually arm,
and which predicate blocks activation?

Failed-return classes:
  A. extraction never armed
  B. armed too late
  C. armed, but selected wrong threat
  D. armed correctly, but route ineffective
  E. carrier tagged before response could matter
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
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

BLUE_STYLE = "BLUE_RUSH"
RED_STYLE = "OP6_IMMEDIATE_DUAL_RUSH"
DEFAULT_MAP = "map_a"
NEAR_BHOME = 6.0
IMMEDIATE_TAG_STEPS = 4
LATE_ARM_STEPS = 8


def _blue_home_anchor(core) -> torch.Tensor:
    """Env-level mask: any non-carrier blue near blue home on blue half."""
    mid = float(core.cols) * 0.5
    bhome_x = core.blue_flag_home[:, 0:1]
    near = (core.blue_x - bhome_x).abs() <= NEAR_BHOME
    return (
        core.blue_alive
        & (~core.blue_carrying)
        & (core.blue_x < mid)
        & near
    ).any(dim=1)


def _between_carrier_and_home(core, carr_i: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-blue on-segment / ahead flags mirroring OP6 screen-break geometry."""
    mid = float(core.cols) * 0.5
    carr_x = float(core.red_x[0, carr_i].item())
    carr_y = float(core.red_y[0, carr_i].item())
    home_x = float(core.red_flag_home[0, 0].item())
    home_y = float(core.red_flag_home[0, 1].item())
    vx = home_x - carr_x
    vy = home_y - carr_y
    vv = vx * vx + vy * vy + 1e-8
    dist_carr_home = vv ** 0.5

    bx = core.blue_x[0]
    by = core.blue_y[0]
    wx = bx - carr_x
    wy = by - carr_y
    t = (wx * vx + wy * vy) / vv
    proj_x = carr_x + t * vx
    proj_y = carr_y + t * vy
    lat = torch.sqrt((bx - proj_x) ** 2 + (by - proj_y) ** 2 + 1e-8)
    on_seg = (t > 0.05) & (t < 0.98) & (lat < 5.0)

    dist_b_home = torch.sqrt((bx - home_x) ** 2 + (by - home_y) ** 2 + 1e-8)
    ahead = (
        (bx > mid)
        & (bx > carr_x)
        & (dist_b_home < (dist_carr_home + 1.0))
        & ((by - home_y).abs() < 6.0)
    )
    eligible = core.blue_alive[0] & (~core.blue_carrying[0]) & (on_seg | ahead)
    return eligible, on_seg | ahead


def _nearest_blue(core, x: float, y: float) -> tuple[int, float]:
    dx = core.blue_x[0] - x
    dy = core.blue_y[0] - y
    dist = torch.sqrt(dx * dx + dy * dy + 1e-8)
    dist = torch.where(core.blue_alive[0], dist, dist.new_full((), 1e9))
    i = int(torch.argmin(dist).item())
    return i, float(dist[i].item())


def _screen_break_snapshot(core, carr_i: int) -> dict[str, Any]:
    """Mirror screen-break / locked dual-threat assignment (read-only)."""
    mid = float(core.cols) * 0.5
    max_y = float(max(0, core.rows - 1))
    home_x = float(core.red_flag_home[0, 0].item())
    home_y = float(core.red_flag_home[0, 1].item())
    carr_x = float(core.red_x[0, carr_i].item())
    carr_y = float(core.red_y[0, carr_i].item())
    extract_on = int(core.bt_op6_extract_ticks[0].item()) > 0
    home_def = bool(_blue_home_anchor(core)[0].item())
    carrier_threat = int(getattr(core, "bt_op6_extract_carrier_threat", torch.tensor([-1]))[0].item())
    screener_threat = int(getattr(core, "bt_op6_extract_screener_threat", torch.tensor([-1]))[0].item())
    eligible, _ = _between_carrier_and_home(core, carr_i)
    d_carr = torch.sqrt(
        (core.blue_x[0] - carr_x) ** 2 + (core.blue_y[0] - carr_y) ** 2 + 1e-8
    )
    d_m = torch.where(eligible, d_carr, d_carr.new_full((), 1e9))
    peel_blocker = int(torch.argmin(d_m).item()) if bool(eligible.any().item()) else -1
    has_peel = bool(eligible.any().item()) and extract_on and (not home_def)

    corridor = "none"
    screening_red = -1
    if has_peel and peel_blocker >= 0:
        by_ = float(core.blue_y[0, peel_blocker].item())
        amp = float(max(3.0, min(7.0, max_y * 0.30)))
        y_hi = min(max(home_y + amp, 0.0), max_y)
        y_lo = min(max(home_y - amp, 0.0), max_y)
        use_hi = abs(y_hi - by_) >= abs(y_lo - by_)
        corridor = "hi" if use_hi else "lo"
    if extract_on:
        roles = core.bt_red_role[0]
        for j in range(int(core.Nr)):
            if j == carr_i:
                continue
            if int(roles[j].item()) == ROLE_ESCORT and bool(core.red_alive[0, j].item()):
                screening_red = j
                break
        if screening_red < 0:
            screening_red = 1 - carr_i

    threat_id, _ = _nearest_blue(core, carr_x, carr_y)
    danger_rank0 = -1
    danger_rank1 = -1
    if hasattr(core, "_bt_op6_projected_threat_danger"):
        bb = {
            "red_flag_home": core.red_flag_home,
            "midline": torch.tensor([mid], device=core.device),
            "idx_env": torch.arange(core.B, device=core.device),
        }
        danger = core._bt_op6_projected_threat_danger(
            bb,
            torch.tensor([carr_x], dtype=torch.float32, device=core.device),
            torch.tensor([carr_y], dtype=torch.float32, device=core.device),
        )[0]
        order = torch.argsort(danger, descending=True)
        ranks = {int(order[i].item()): i + 1 for i in range(min(2, int(order.numel())))}
        danger_rank0 = int(ranks.get(0, -1))
        danger_rank1 = int(ranks.get(1, -1))

    distinct = int(
        carrier_threat >= 0
        and screener_threat >= 0
        and carrier_threat != screener_threat
    )
    return {
        "selected_return_corridor": corridor,
        "screening_red_agent_id": screening_red,
        "nearest_blue_threat_id": threat_id,
        "screen_blocker_id": peel_blocker if has_peel else -1,
        "screen_break_active": int(has_peel),
        "carrier_evasion_threat_id": carrier_threat,
        "screener_threat_id": screener_threat,
        "assignments_distinct": distinct,
        "danger_rank_blue0": danger_rank0,
        "danger_rank_blue1": danger_rank1,
    }


def _condition_vector(core, carr_i: int) -> dict[str, Any]:
    extract_enabled = bool(getattr(_BTRedMixin, "_OP6_EXTRACTION_ENABLED", True))
    carrier_has_flag = bool(core.red_carrying[0, carr_i].item())
    home_def = bool(_blue_home_anchor(core)[0].item())
    blue_home_defense_absent = not home_def
    eligible, between = _between_carrier_and_home(core, carr_i)
    blue_between = bool(eligible.any().item())
    ticks = int(core.bt_op6_extract_ticks[0].item())
    # Arming has no min-distance gate in current BT; report True for arm path.
    # Screen-break uses lateral/ahead thresholds — separately as geometry_ok.
    minimum_distance_condition = True
    role_lock_available = True  # extract forces lock over slots; not a gate
    timeout_not_expired = ticks > 0 or carrier_has_flag
    extraction_eligible = (
        extract_enabled
        and carrier_has_flag
        and blue_home_defense_absent
    )
    extraction_armed = ticks > 0
    return {
        "carrier_has_flag": int(carrier_has_flag),
        "blue_noncarrier_between_carrier_and_home": int(blue_between),
        "blue_home_defense_absent": int(blue_home_defense_absent),
        "minimum_distance_condition": int(minimum_distance_condition),
        "role_lock_available": int(role_lock_available),
        "timeout_not_expired": int(timeout_not_expired),
        "extraction_eligible": int(extraction_eligible),
        "extraction_armed": int(extraction_armed),
        "extract_ticks": ticks,
        "blue_anchor_present": int(home_def),
        "any_blue_on_seg_or_ahead": int(bool(between.any().item())),
    }


def _flag_area_geometry(core, carr_i: int) -> dict[str, Any]:
    cx = float(core.red_x[0, carr_i].item())
    cy = float(core.red_y[0, carr_i].item())
    hx = float(core.red_flag_home[0, 0].item())
    hy = float(core.red_flag_home[0, 1].item())
    out: dict[str, Any] = {
        "dist_carrier_blue0": None,
        "dist_carrier_blue1": None,
        "dist_blue0_blue1": None,
        "dist_blue0_red_home": None,
        "dist_blue1_red_home": None,
        "blue_between_carrier_and_home_mask": "",
    }
    if int(core.Nb) >= 2:
        b0x, b0y = float(core.blue_x[0, 0].item()), float(core.blue_y[0, 0].item())
        b1x, b1y = float(core.blue_x[0, 1].item()), float(core.blue_y[0, 1].item())
        out["dist_carrier_blue0"] = ((cx - b0x) ** 2 + (cy - b0y) ** 2) ** 0.5
        out["dist_carrier_blue1"] = ((cx - b1x) ** 2 + (cy - b1y) ** 2) ** 0.5
        out["dist_blue0_blue1"] = ((b0x - b1x) ** 2 + (b0y - b1y) ** 2) ** 0.5
        out["dist_blue0_red_home"] = ((b0x - hx) ** 2 + (b0y - hy) ** 2) ** 0.5
        out["dist_blue1_red_home"] = ((b1x - hx) ** 2 + (b1y - hy) ** 2) ** 0.5
        _, between = _between_carrier_and_home(core, carr_i)
        mask = []
        for bi in range(2):
            if bool(between[bi].item()) and bool(core.blue_alive[0, bi].item()):
                mask.append(str(bi))
        out["blue_between_carrier_and_home_mask"] = ",".join(mask) if mask else "none"
    return out


def _classify_failed(
    *,
    ever_armed: bool,
    pickup_to_arm_delay: int | None,
    tag_delay: int,
    carrier_threat: int,
    screener_threat: int,
    tagger_id: int,
    first_contact_by_screener_target: bool,
    dual_threat_pickup: bool,
) -> str:
    if tag_delay <= IMMEDIATE_TAG_STEPS:
        return "E"
    if not ever_armed:
        return "A"
    if pickup_to_arm_delay is not None and pickup_to_arm_delay > LATE_ARM_STEPS:
        return "B"
    if dual_threat_pickup and carrier_threat >= 0 and screener_threat >= 0:
        if carrier_threat == screener_threat:
            return "C1"
        if tagger_id == screener_threat:
            return "C3"
        if tagger_id >= 0 and tagger_id != screener_threat:
            # Tagger was not the screener's assigned threat.
            if not first_contact_by_screener_target:
                return "C2"
            return "C3"
    return "D"


def _run_episode(
    *,
    episode_index: int,
    episode_seed: int,
    map_name: str,
    max_decision_steps: int,
    device: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    env = _make_env(
        map_name=map_name,
        seed=episode_seed,
        max_decision_steps=max_decision_steps,
        device=device,
    )
    pickup_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    try:
        core = env.core
        env.env_method("set_phase", RED_STYLE)
        env.env_method("set_next_opponent", "SCRIPTED", RED_STYLE)
        core.blue_scripted = True
        core.set_blue_style(BLUE_STYLE)
        env.reset()
        env.env_method("set_phase", RED_STYLE)
        env.env_method("set_next_opponent", "SCRIPTED", RED_STYLE)
        core.blue_scripted = True
        core.set_blue_style(BLUE_STYLE)

        prev_carry = torch.zeros_like(core.red_carrying[0])
        prev_ticks = 0
        prev_red_score = 0
        active: dict[str, Any] | None = None
        pickups = 0
        scores = 0
        failed = 0
        class_counts: Counter[str] = Counter()
        steps = 0
        last_info: dict[str, Any] = {}
        # Local monotonic decision-step clock. sim_step_count resets on
        # score/tag in-episode, which would corrupt pickup→disarm delays.
        local_step = 0

        for _ in range(int(max_decision_steps) + 5):
            action = _zero_action(env)
            env.step_async(action)
            _, _, done, infos = env.step_wait()
            last_info = infos[0] if infos else {}
            local_step += 1
            sim = local_step
            carry = core.red_carrying[0].clone()
            ticks = int(core.bt_op6_extract_ticks[0].item())
            red_score_now = int(core.red_score[0].item())

            newly = carry & (~prev_carry)
            lost = prev_carry & (~carry)

            if bool(newly.any().item()):
                carr_i = int(torch.argmax(newly.to(torch.int64)).item())
                pickups += 1
                cond = _condition_vector(core, carr_i)
                geom = _flag_area_geometry(core, carr_i)
                screen = _screen_break_snapshot(core, carr_i)
                armed_now = ticks > 0
                eligible_now = int(cond["extraction_eligible"])
                if armed_now:
                    block_reason = "none_armed"
                elif not cond["carrier_has_flag"]:
                    block_reason = "no_carrier_flag"
                elif not cond["blue_home_defense_absent"]:
                    block_reason = "blue_home_anchor_present"
                elif not bool(getattr(_BTRedMixin, "_OP6_EXTRACTION_ENABLED", True)):
                    block_reason = "extraction_disabled"
                else:
                    # Eligible by geometry, but ticks still 0: BT typically
                    # ran earlier this physics step before the pickup landed.
                    block_reason = "bt_before_pickup_lag"
                active = {
                    "episode_index": episode_index,
                    "episode_seed": episode_seed,
                    "pickup_index": pickups,
                    "pickup_step": sim,
                    "sim_step_count": int(core.sim_step_count[0].item()),
                    "carrier_agent_id": carr_i,
                    "extraction_eligible_at_pickup": eligible_now,
                    "extraction_armed_at_pickup": int(armed_now),
                    "extraction_eligible": eligible_now,
                    "extraction_armed": int(armed_now),
                    "arm_step": sim if armed_now else None,
                    "pickup_to_arm_delay": 0 if armed_now else None,
                    "disarm_step": None,
                    "disarm_reason": None,
                    "ever_armed": int(armed_now),
                    "first_screen_blocker_id": screen["screen_blocker_id"],
                    "locked_carrier_threat": screen["carrier_evasion_threat_id"],
                    "locked_screener_threat": screen["screener_threat_id"],
                    "assignments_distinct": screen["assignments_distinct"],
                    "first_contact_blue0_step": None,
                    "first_contact_blue1_step": None,
                    "first_contact_by_screener_target": 0,
                    "dual_threat_pickup": int(
                        ","
                        in str(geom.get("blue_between_carrier_and_home_mask", ""))
                    ),
                    "tagger_id": -1,
                    "outcome": None,
                    "fail_class": None,
                    "tag_delay": None,
                    "carry_duration": None,
                    "block_reason_at_pickup": block_reason,
                    # Freeze pickup-time condition + geometry (do not overwrite).
                    **{f"pu_{k}": v for k, v in cond.items()},
                    **{f"pu_{k}": v for k, v in geom.items()},
                    **cond,
                    **geom,
                    **screen,
                }
                pickup_rows.append(active)
                step_rows.append(
                    {
                        "episode_index": episode_index,
                        "pickup_index": pickups,
                        "step": sim,
                        "event": "pickup",
                        **{k: active[k] for k in cond},
                        **{k: active[k] for k in geom},
                        **{k: active[k] for k in screen},
                    }
                )

            if active is not None and bool(carry.any().item()):
                carr_i = int(active["carrier_agent_id"])
                # Still carrying the same agent?
                if not bool(carry[carr_i].item()):
                    # Carrier id flipped mid-window (rare); retarget.
                    carr_i = int(torch.argmax(carry.to(torch.int64)).item())
                    active["carrier_agent_id"] = carr_i
                cond = _condition_vector(core, carr_i)
                screen = _screen_break_snapshot(core, carr_i)
                if ticks > 0 and not active["ever_armed"]:
                    active["ever_armed"] = 1
                    active["arm_step"] = sim
                    active["pickup_to_arm_delay"] = sim - int(active["pickup_step"])
                    active["extraction_armed"] = 1
                    active["extraction_eligible"] = cond["extraction_eligible"]
                    if active["first_screen_blocker_id"] < 0:
                        active["first_screen_blocker_id"] = screen["screen_blocker_id"]
                    active["locked_carrier_threat"] = screen["carrier_evasion_threat_id"]
                    active["locked_screener_threat"] = screen["screener_threat_id"]
                    active["assignments_distinct"] = screen["assignments_distinct"]
                    active["selected_return_corridor"] = screen["selected_return_corridor"]
                    active["screening_red_agent_id"] = screen["screening_red_agent_id"]
                    active["nearest_blue_threat_id"] = screen["nearest_blue_threat_id"]
                    active["danger_rank_blue0"] = screen["danger_rank_blue0"]
                    active["danger_rank_blue1"] = screen["danger_rank_blue1"]
                    step_rows.append(
                        {
                            "episode_index": episode_index,
                            "pickup_index": active["pickup_index"],
                            "step": sim,
                            "event": "arm",
                            **cond,
                            **_flag_area_geometry(core, carr_i),
                            **screen,
                        }
                    )
                elif ticks > 0 and active["locked_carrier_threat"] < 0:
                    # Capture lock if arm landed same frame as pickup observe.
                    active["locked_carrier_threat"] = screen["carrier_evasion_threat_id"]
                    active["locked_screener_threat"] = screen["screener_threat_id"]
                    active["assignments_distinct"] = screen["assignments_distinct"]
                # First contact: blue enters tag-radius (~2.5) of carrier.
                cx = float(core.red_x[0, carr_i].item())
                cy = float(core.red_y[0, carr_i].item())
                for bi, key in ((0, "first_contact_blue0_step"), (1, "first_contact_blue1_step")):
                    if active[key] is not None:
                        continue
                    if not bool(core.blue_alive[0, bi].item()):
                        continue
                    d = (
                        (float(core.blue_x[0, bi].item()) - cx) ** 2
                        + (float(core.blue_y[0, bi].item()) - cy) ** 2
                    ) ** 0.5
                    if d <= 2.5:
                        active[key] = sim
                        if bi == int(active.get("locked_screener_threat", -1)):
                            active["first_contact_by_screener_target"] = 1
                if (
                    ticks > 0
                    and prev_ticks <= 0
                    and active["ever_armed"]
                    and active.get("arm_step") != sim
                ):
                    # Re-arm after timeout while still carrying.
                    step_rows.append(
                        {
                            "episode_index": episode_index,
                            "pickup_index": active["pickup_index"],
                            "step": sim,
                            "event": "rearm",
                            **cond,
                            **_flag_area_geometry(core, carr_i),
                            **screen,
                        }
                    )
                # Soft timeout while still carrying.
                if prev_ticks > 0 and ticks <= 0 and bool(carry[carr_i].item()):
                    step_rows.append(
                        {
                            "episode_index": episode_index,
                            "pickup_index": active["pickup_index"],
                            "step": sim,
                            "event": "extract_timeout",
                            **cond,
                            **_flag_area_geometry(core, carr_i),
                            **screen,
                        }
                    )

            if active is not None and bool(lost.any().item()):
                carr_i = int(active["carrier_agent_id"])
                scored = red_score_now > prev_red_score
                # Nearest blue to last carrier position as tagger proxy.
                tagger_id, _ = _nearest_blue(
                    core,
                    float(core.red_x[0, carr_i].item()),
                    float(core.red_y[0, carr_i].item()),
                )
                # Prefer blue that is newly having tagged contact — nearest is fine.
                active["tagger_id"] = tagger_id
                active["disarm_step"] = sim
                tag_delay = sim - int(active["pickup_step"])
                active["tag_delay"] = tag_delay
                active["carry_duration"] = tag_delay
                if scored:
                    scores += 1
                    active["disarm_reason"] = "scored"
                    active["outcome"] = "score"
                    active["fail_class"] = None
                else:
                    failed += 1
                    active["disarm_reason"] = "tagged_or_drop"
                    active["outcome"] = "failed_return"
                    cls = _classify_failed(
                        ever_armed=bool(active["ever_armed"]),
                        pickup_to_arm_delay=active["pickup_to_arm_delay"],
                        tag_delay=tag_delay,
                        carrier_threat=int(active.get("locked_carrier_threat", -1)),
                        screener_threat=int(active.get("locked_screener_threat", -1)),
                        tagger_id=tagger_id,
                        first_contact_by_screener_target=bool(
                            active.get("first_contact_by_screener_target")
                        ),
                        dual_threat_pickup=bool(active.get("dual_threat_pickup")),
                    )
                    active["fail_class"] = cls
                    class_counts[cls] += 1
                step_rows.append(
                    {
                        "episode_index": episode_index,
                        "pickup_index": active["pickup_index"],
                        "step": sim,
                        "event": "disarm_" + str(active["disarm_reason"]),
                        "fail_class": active["fail_class"],
                        "tagger_id": tagger_id,
                        "tag_delay": tag_delay,
                        **_condition_vector(core, carr_i),
                        **_flag_area_geometry(core, carr_i),
                        **_screen_break_snapshot(core, carr_i),
                    }
                )
                active = None

            prev_carry = carry
            prev_ticks = ticks
            prev_red_score = red_score_now
            steps += 1
            if bool(done.any()):
                break

        if active is not None:
            active["disarm_step"] = int(core.sim_step_count[0].item())
            active["disarm_reason"] = "episode_end_still_carrying"
            active["outcome"] = "open_carry_at_end"
            active = None

        ep_res = last_info.get("episode_result", last_info) if last_info else {}
        summary = {
            "blue_style": BLUE_STYLE,
            "red_style": RED_STYLE,
            "map": map_name,
            "episode_index": episode_index,
            "episode_seed": episode_seed,
            "pickups": pickups,
            "scores": scores,
            "failed_returns": failed,
            "fail_class_counts": dict(class_counts),
            "armed_at_pickup_n": sum(
                1 for r in pickup_rows if r.get("episode_index") == episode_index and r.get("arm_step") == r.get("pickup_step")
            ),
            "never_armed_n": sum(
                1
                for r in pickup_rows
                if r.get("episode_index") == episode_index and not r.get("ever_armed")
            ),
            "block_blue_anchor_at_pickup_n": sum(
                1
                for r in pickup_rows
                if r.get("episode_index") == episode_index
                and r.get("block_reason_at_pickup") == "blue_home_anchor_present"
            ),
            "red_score": int(ep_res.get("red_score", core.red_score[0].item())),
            "blue_score": int(ep_res.get("blue_score", core.blue_score[0].item())),
            "steps": steps,
        }
        # Fix armed_at_pickup / never_armed to only this episode's rows appended this call.
        ep_pickups = [r for r in pickup_rows]
        summary["armed_at_pickup_n"] = sum(
            1 for r in ep_pickups if r.get("pickup_to_arm_delay") == 0
        )
        summary["never_armed_n"] = sum(1 for r in ep_pickups if not r.get("ever_armed"))
        summary["block_blue_anchor_at_pickup_n"] = sum(
            1
            for r in ep_pickups
            if r.get("block_reason_at_pickup") == "blue_home_anchor_present"
        )
        summary["mean_pickup_to_arm_delay"] = (
            sum(
                float(r["pickup_to_arm_delay"])
                for r in ep_pickups
                if r.get("pickup_to_arm_delay") is not None
            )
            / max(sum(1 for r in ep_pickups if r.get("pickup_to_arm_delay") is not None), 1)
        )
        return summary, pickup_rows, step_rows
    finally:
        env.close()


def _aggregate(
    ep_rows: list[dict[str, Any]],
    pickup_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    fail_classes = Counter()
    for r in pickup_rows:
        if r.get("fail_class"):
            fail_classes[str(r["fail_class"])] += 1
    block_reasons = Counter(
        str(r.get("block_reason_at_pickup", "unknown")) for r in pickup_rows
    )
    n_pick = len(pickup_rows)
    n_fail = sum(1 for r in pickup_rows if r.get("outcome") == "failed_return")
    n_score = sum(1 for r in pickup_rows if r.get("outcome") == "score")
    armed_any = sum(1 for r in pickup_rows if r.get("ever_armed"))
    armed_at_pu = sum(1 for r in pickup_rows if r.get("pickup_to_arm_delay") == 0)
    delays = [
        int(r["pickup_to_arm_delay"])
        for r in pickup_rows
        if r.get("pickup_to_arm_delay") is not None
    ]
    both_between = sum(
        1
        for r in pickup_rows
        if "," in str(r.get("pu_blue_between_carrier_and_home_mask") or r.get("blue_between_carrier_and_home_mask") or "")
    )
    fail_durs = [
        int(r["carry_duration"])
        for r in pickup_rows
        if r.get("outcome") == "failed_return" and r.get("carry_duration") is not None
    ]
    # Flag-area overlap at pickup.
    both_near_flag = 0
    for r in pickup_rows:
        # Blues near blue flag if not abandoned (anchor present) OR colocated.
        if r.get("block_reason_at_pickup") == "blue_home_anchor_present":
            both_near_flag += 1
        elif (
            r.get("pu_dist_blue0_blue1", r.get("dist_blue0_blue1")) is not None
            and float(r.get("pu_dist_blue0_blue1", r.get("dist_blue0_blue1"))) < 4.0
            and r.get("pu_dist_carrier_blue0", r.get("dist_carrier_blue0")) is not None
            and float(r.get("pu_dist_carrier_blue0", r.get("dist_carrier_blue0"))) < 5.0
            and float(r.get("pu_dist_carrier_blue1", r.get("dist_carrier_blue1"))) < 5.0
        ):
            both_near_flag += 1

    dual_pickups = [r for r in pickup_rows if r.get("dual_threat_pickup")]
    # Prefer post-arm locked assignment (pickup-time lock is often still -1/-1).
    def _is_distinct(r: dict[str, Any]) -> bool:
        c = int(r.get("locked_carrier_threat", -1))
        s = int(r.get("locked_screener_threat", -1))
        return c >= 0 and s >= 0 and c != s

    distinct_n = sum(1 for r in dual_pickups if _is_distinct(r))
    distinct_frac = distinct_n / max(len(dual_pickups), 1)
    wrong_threat_fails = sum(
        1 for r in pickup_rows if r.get("fail_class") in ("C", "C1", "C2")
    )
    baseline_wrong_threat = 6  # dev33 class-C count on same 8 RUSH seeds
    wrong_threat_reduction = (
        1.0 - (wrong_threat_fails / max(baseline_wrong_threat, 1))
        if baseline_wrong_threat
        else None
    )

    decision = "unknown"
    if n_fail == 0:
        decision = "no_failed_returns"
    else:
        top = fail_classes.most_common(1)[0][0]
        decision = {
            "A": "Never arms → fix activation predicate or state wiring",
            "B": "Arms late → arm immediately on pickup, then validate conditions afterward",
            "C": "Targets wrong blue → distinguish blocker geometry from generic nearest-threat",
            "C1": "Carrier and screener duplicated the same threat",
            "C2": "Screener selected the wrong distinct threat",
            "C3": "Correct target, but screener arrived too late",
            "D": "Both threats handled, route still failed",
            "E": "Carrier dies immediately → opening must begin before pickup, not after",
        }.get(top, f"dominant_class={top}")

    return {
        "n_episodes": len(ep_rows),
        "n_pickups": n_pick,
        "n_scores": n_score,
        "n_failed_returns": n_fail,
        "mean_pickups": sum(int(e["pickups"]) for e in ep_rows) / max(len(ep_rows), 1),
        "mean_failed_returns": sum(int(e["failed_returns"]) for e in ep_rows)
        / max(len(ep_rows), 1),
        "armed_any_frac": armed_any / max(n_pick, 1),
        "armed_at_pickup_frac": armed_at_pu / max(n_pick, 1),
        "never_armed_frac": (n_pick - armed_any) / max(n_pick, 1),
        "mean_pickup_to_arm_delay_when_armed": (
            sum(delays) / max(len(delays), 1) if delays else None
        ),
        "mean_fail_carry_duration": (
            sum(fail_durs) / max(len(fail_durs), 1) if fail_durs else None
        ),
        "both_blues_between_at_pickup_frac": both_between / max(n_pick, 1),
        "dual_threat_pickups": len(dual_pickups),
        "distinct_assignment_frac_dual": distinct_frac,
        "wrong_threat_fail_n": wrong_threat_fails,
        "wrong_threat_reduction_vs_dev33": wrong_threat_reduction,
        "block_reason_at_pickup": dict(block_reasons),
        "fail_class_counts": dict(fail_classes),
        "fail_class_frac_of_failures": {
            k: v / max(n_fail, 1) for k, v in fail_classes.items()
        },
        "flag_area_overlap_proxy_n": both_near_flag,
        "flag_area_overlap_proxy_frac": both_near_flag / max(n_pick, 1),
        "immediate_tag_threshold": IMMEDIATE_TAG_STEPS,
        "late_arm_threshold": LATE_ARM_STEPS,
        "micro_gates": {
            "distinct_assignment_ge_90pct": distinct_frac >= 0.90,
            "wrong_threat_reduced_ge_75pct": (wrong_threat_reduction or 0.0) >= 0.75,
            "true_failed_returns_le_0_5": (
                sum(int(e["failed_returns"]) for e in ep_rows) / max(len(ep_rows), 1)
            )
            <= 0.5,
        },
        "decision_tree_recommendation": decision,
    }


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
        default=None,
    )
    args = p.parse_args()
    _BTRedMixin._OP6_EXTRACTION_ENABLED = True
    if args.out_dir is None:
        args.out_dir = (
            PROJECT_ROOT
            / "artifacts"
            / "op6_extraction_activation_dev34_dual_threat_rush_map_a"
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    ep_rows: list[dict[str, Any]] = []
    all_pickups: list[dict[str, Any]] = []
    all_steps: list[dict[str, Any]] = []
    for ep in range(int(args.episodes)):
        seed = _episode_seed(
            int(args.base_seed), red_index=0, map_index=0, episode_index=ep
        )
        summary, pickups, steps = _run_episode(
            episode_index=ep,
            episode_seed=seed,
            map_name=str(args.map),
            max_decision_steps=int(args.max_decision_steps),
            device=str(args.device),
        )
        ep_rows.append(summary)
        all_pickups.extend(pickups)
        all_steps.extend(steps)
        print(
            f"[{ep + 1}/{args.episodes}] seed={seed} "
            f"pickups={summary['pickups']} scores={summary['scores']} "
            f"failed={summary['failed_returns']} "
            f"never_armed={summary['never_armed_n']} "
            f"anchor_block={summary['block_blue_anchor_at_pickup_n']} "
            f"classes={summary['fail_class_counts']}",
            flush=True,
        )

    agg = _aggregate(ep_rows, all_pickups)
    (args.out_dir / "episode_summary.csv").write_text("", encoding="utf-8")
    with (args.out_dir / "episode_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(ep_rows[0].keys()))
        w.writeheader()
        # Counter dicts → JSON strings for CSV.
        for row in ep_rows:
            out = dict(row)
            out["fail_class_counts"] = json.dumps(out["fail_class_counts"])
            w.writerow(out)

    # Flatten pickup rows for CSV (drop nested none).
    flat_keys: list[str] = []
    for r in all_pickups:
        for k in r.keys():
            if k not in flat_keys:
                flat_keys.append(k)
    with (args.out_dir / "pickup_traces.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=flat_keys)
        w.writeheader()
        for r in all_pickups:
            w.writerow({k: r.get(k) for k in flat_keys})

    if all_steps:
        step_keys: list[str] = []
        for r in all_steps:
            for k in r.keys():
                if k not in step_keys:
                    step_keys.append(k)
        with (args.out_dir / "activation_steps.csv").open(
            "w", newline="", encoding="utf-8"
        ) as f:
            w = csv.DictWriter(f, fieldnames=step_keys)
            w.writeheader()
            for r in all_steps:
                w.writerow({k: r.get(k) for k in step_keys})

    summary_path = args.out_dir / "activation_summary.json"
    summary_path.write_text(json.dumps(agg, indent=2), encoding="utf-8")
    print(json.dumps(agg, indent=2))
    print(f"wrote {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
