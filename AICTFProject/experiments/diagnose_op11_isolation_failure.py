#!/usr/bin/env python3
"""Phase-1 OP11 ESCORT rescue diagnostic (no BT redesign).

On fresh paired seeds × map_a, trace all four blue styles and quantify why
SPLIT still converts more consistently than ESCORT under current OP11.

Failure-mode buckets (counted per episode; summary ranks dominance):
  A. both reds commit to one blue attacker
  B. second blue lane remains uncovered
  C. red target ownership churns between threats
  D. red pressures ESCORT protector too effectively (protector chase / tag)
  E. supported vs isolated carriers treated almost identically

Does not read blue style IDs inside the env BT path — style is only the
scripted controller selector for this off-policy diagnostic.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.run_scripted_style_payoff_matrix import _make_env, _zero_action
from gpu_env._maps import normalize_map_layout

RED = "OP11_ADAPTIVE_EXPLOITER"
STYLES = ("BLUE_SPLIT", "BLUE_ESCORT", "BLUE_RUSH", "BLUE_TURTLE")
# Retired / prior OP11 seed blocks — refuse reuse.
RETIRED_BASE_SEEDS = frozenset({541001, 551001, 561001})

# Geometry thresholds (diagnostic only; mirror OP11 split gate where noted).
OFFENSIVE_BUFFER = 3.0
SUPPORT_NEAR = 6.0
SAME_TARGET_DIST = 3.0
UNCOVERED_NEAR_RED = 8.0
CHURN_MIN_LATCH_STEPS = 8


def _as_float(value: Any) -> float:
    try:
        if hasattr(value, "detach"):
            value = value.detach().flatten()[0]
        if hasattr(value, "item"):
            value = value.item()
        return float(value)
    except Exception:
        return float("nan")


def _as_bool(value: Any) -> bool:
    try:
        if hasattr(value, "item"):
            return bool(value.item())
        return bool(value)
    except Exception:
        return False


def _nearest_blue_idx(rx: float, ry: float, bx0: float, by0: float, bx1: float, by1: float) -> int:
    d0 = math.hypot(rx - bx0, ry - by0)
    d1 = math.hypot(rx - bx1, ry - by1)
    return 0 if d0 <= d1 else 1


def _run_episode(
    *,
    style: str,
    episode_index: int,
    base_seed: int,
    map_name: str,
    max_steps: int,
    device: str,
) -> dict[str, Any]:
    seed = int(base_seed) + int(episode_index)
    env = _make_env(map_name=map_name, seed=seed, max_decision_steps=max_steps, device=device)
    try:
        env.env_method("set_phase", RED)
        env.env_method("set_next_opponent", "SCRIPTED", RED)
        env.reset()
        core = env.core
        env.env_method("set_phase", RED)
        env.env_method("set_next_opponent", "SCRIPTED", RED)
        core.blue_scripted = True
        core.set_blue_style(style)
        action = _zero_action(env)

        midline = float(core.cols) * 0.5
        center_y = float(core.rows) * 0.5
        opposite_lane_min_sep = float(core.rows) * 0.55

        first_mid = None
        first_pickup = None
        first_blue_score = None
        first_red_score = None
        first_latch = None

        steps_opposite_lanes = 0
        steps_both_offensive = 0
        steps_sep_ge_12 = 0
        sum_teammate = 0.0
        sum_lateral = 0.0
        max_teammate = 0.0
        max_lateral = 0.0
        n_geo = 0

        latch_steps = 0
        latch_both_same_target = 0
        latch_second_uncovered = 0
        latch_target_churn = 0
        latch_protector_chase = 0
        latch_both_on_carrier = 0
        post_pickup_steps = 0
        post_support_near_steps = 0
        post_support_far_steps = 0
        isolated_return_attempts = 0
        isolated_return_success = 0
        supported_return_attempts = 0
        supported_return_success = 0
        protector_tagged_while_carry = 0
        role_change_steps = 0

        prev_mark = (-1, -1)
        prev_roles = (-1, -1)
        prev_carrying = False
        prev_support_near = False
        prev_blue = 0
        prev_red = 0
        blue_score = 0
        red_score = 0
        step = -1

        for step in range(max_steps):
            env.step_async(action)
            _, _reward, done, _infos = env.step_wait()

            bx0 = _as_float(core.blue_x[0, 0])
            bx1 = _as_float(core.blue_x[0, 1])
            by0 = _as_float(core.blue_y[0, 0])
            by1 = _as_float(core.blue_y[0, 1])
            rx0 = _as_float(core.red_x[0, 0])
            rx1 = _as_float(core.red_x[0, 1])
            ry0 = _as_float(core.red_y[0, 0])
            ry1 = _as_float(core.red_y[0, 1])
            teammate = math.hypot(bx0 - bx1, by0 - by1)
            lateral = abs(by0 - by1)
            opposite = ((by0 > center_y) and (by1 <= center_y)) or ((by1 > center_y) and (by0 <= center_y))
            both_off = (bx0 > midline - OFFENSIVE_BUFFER) and (bx1 > midline - OFFENSIVE_BUFFER)
            if opposite and lateral >= opposite_lane_min_sep:
                steps_opposite_lanes += 1
            if both_off:
                steps_both_offensive += 1
            if teammate >= 12.0:
                steps_sep_ge_12 += 1
            sum_teammate += teammate
            sum_lateral += lateral
            max_teammate = max(max_teammate, teammate)
            max_lateral = max(max_lateral, lateral)
            n_geo += 1

            if first_mid is None and (bx0 > midline or bx1 > midline):
                first_mid = step

            carry0 = _as_bool(core.blue_carrying[0, 0])
            carry1 = _as_bool(core.blue_carrying[0, 1])
            carrying = carry0 or carry1
            if first_pickup is None and carrying:
                first_pickup = step

            carrier_idx = 0 if carry0 else (1 if carry1 else -1)
            support_idx = 1 - carrier_idx if carrier_idx >= 0 else -1
            support_dist = float("nan")
            support_near = False
            if carrier_idx >= 0:
                cx = bx0 if carrier_idx == 0 else bx1
                cy = by0 if carrier_idx == 0 else by1
                sx = bx0 if support_idx == 0 else bx1
                sy = by0 if support_idx == 0 else by1
                support_dist = math.hypot(cx - sx, cy - sy)
                support_near = support_dist <= SUPPORT_NEAR

            latch_on = int(core.bt_adapt_split_first_trigger_step[0].item()) >= 0
            if latch_on and first_latch is None:
                first_latch = step

            # Prefer BT debug targets; else recompute OP11 latch assignment;
            # else nearest-blue from red position.
            mark0 = mark1 = -1
            dbg_x = getattr(core, "_debug_red_target_x", None)
            dbg_y = getattr(core, "_debug_red_target_y", None)
            if dbg_x is not None and dbg_y is not None:
                tx0 = _as_float(dbg_x[0, 0])
                ty0 = _as_float(dbg_y[0, 0])
                tx1 = _as_float(dbg_x[0, 1])
                ty1 = _as_float(dbg_y[0, 1])
                if not any(math.isnan(v) for v in (tx0, ty0, tx1, ty1)):
                    mark0 = _nearest_blue_idx(tx0, ty0, bx0, by0, bx1, by1)
                    mark1 = _nearest_blue_idx(tx1, ty1, bx0, by0, bx1, by1)
            if mark0 < 0:
                if latch_on and carrying and carrier_idx >= 0:
                    # Mirror OP11 post-pickup: nearer red → carrier, other → support.
                    r0c = math.hypot(rx0 - (bx0 if carrier_idx == 0 else bx1), ry0 - (by0 if carrier_idx == 0 else by1))
                    r1c = math.hypot(rx1 - (bx0 if carrier_idx == 0 else bx1), ry1 - (by0 if carrier_idx == 0 else by1))
                    if r0c <= r1c:
                        mark0, mark1 = carrier_idx, support_idx
                    else:
                        mark0, mark1 = support_idx, carrier_idx
                elif latch_on:
                    # Mirror OP11 pre-pickup cheapest 2x2 assignment.
                    d00 = math.hypot(rx0 - bx0, ry0 - by0)
                    d01 = math.hypot(rx0 - bx1, ry0 - by1)
                    d10 = math.hypot(rx1 - bx0, ry1 - by0)
                    d11 = math.hypot(rx1 - bx1, ry1 - by1)
                    if (d01 + d10) < (d00 + d11):
                        mark0, mark1 = 1, 0
                    else:
                        mark0, mark1 = 0, 1
                else:
                    mark0 = _nearest_blue_idx(rx0, ry0, bx0, by0, bx1, by1)
                    mark1 = _nearest_blue_idx(rx1, ry1, bx0, by0, bx1, by1)

            roles = tuple(int(x) for x in core.bt_red_role[0].tolist()[:2])
            if prev_roles != (-1, -1) and roles != prev_roles:
                role_change_steps += 1
            prev_roles = roles

            if latch_on:
                latch_steps += 1
                if mark0 == mark1:
                    latch_both_same_target += 1  # A
                # B: one blue farther than UNCOVERED_NEAR_RED from both reds
                d_b0 = min(math.hypot(rx0 - bx0, ry0 - by0), math.hypot(rx1 - bx0, ry1 - by0))
                d_b1 = min(math.hypot(rx0 - bx1, ry0 - by1), math.hypot(rx1 - bx1, ry1 - by1))
                covered0 = d_b0 <= UNCOVERED_NEAR_RED
                covered1 = d_b1 <= UNCOVERED_NEAR_RED
                if covered0 ^ covered1:
                    latch_second_uncovered += 1  # B: exactly one lane covered
                if prev_mark != (-1, -1) and (mark0, mark1) != prev_mark:
                    latch_target_churn += 1  # C

                if carrying and support_idx >= 0:
                    # D: both marks on protector, or nearest red to protector < to carrier
                    if mark0 == support_idx and mark1 == support_idx:
                        latch_protector_chase += 1
                    elif mark0 == support_idx or mark1 == support_idx:
                        # count partial protector chase as soft D signal
                        latch_protector_chase += 1
                    # E: both on carrier while support is near
                    if mark0 == carrier_idx and mark1 == carrier_idx:
                        latch_both_on_carrier += 1

            if carrying:
                post_pickup_steps += 1
                if support_near:
                    post_support_near_steps += 1
                else:
                    post_support_far_steps += 1

            # Return attempt resolution: score while carrying ends an attempt window.
            blue_score = int(core.blue_score[0].item())
            red_score = int(core.red_score[0].item())
            if first_blue_score is None and blue_score > prev_blue:
                first_blue_score = step
                if prev_carrying:
                    if prev_support_near:
                        supported_return_attempts += 1
                        supported_return_success += 1
                    else:
                        isolated_return_attempts += 1
                        isolated_return_success += 1
            if first_red_score is None and red_score > prev_red:
                first_red_score = step

            # Carrier tagged / flag lost without scoring → failed return.
            if prev_carrying and not carrying and blue_score == prev_blue:
                if prev_support_near:
                    supported_return_attempts += 1
                else:
                    isolated_return_attempts += 1

            if carrying and support_idx >= 0:
                tagged = core.blue_tagged[0]
                if _as_bool(tagged[support_idx]):
                    protector_tagged_while_carry += 1

            prev_mark = (mark0, mark1)
            prev_carrying = carrying
            prev_support_near = support_near
            prev_blue = blue_score
            prev_red = red_score
            if bool(done.any()):
                break

        # Episode-level failure flags (dominant-mode vote uses rates under latch).
        fail_a = int(latch_steps > 0 and (latch_both_same_target / max(latch_steps, 1)) >= 0.35)
        fail_b = int(latch_steps > 0 and (latch_second_uncovered / max(latch_steps, 1)) >= 0.35)
        fail_c = int(
            latch_steps >= CHURN_MIN_LATCH_STEPS
            and (latch_target_churn / max(latch_steps, 1)) >= 0.25
        )
        fail_d = int(
            post_pickup_steps > 0
            and (
                (latch_protector_chase / max(latch_steps, 1)) >= 0.25
                or protector_tagged_while_carry >= 3
            )
        )
        # E: when support is near, reds still both on carrier often
        fail_e = int(
            post_support_near_steps >= 5
            and latch_both_on_carrier >= max(3, post_support_near_steps // 4)
        )

        return {
            "blue_style": style,
            "episode_index": episode_index,
            "episode_seed": seed,
            "map": map_name,
            "normalized_map": normalize_map_layout(map_name),
            "steps": step + 1,
            "blue_score": blue_score,
            "red_score": red_score,
            "win_margin": blue_score - red_score,
            "first_midfield_step": first_mid if first_mid is not None else -1,
            "first_pickup_step": first_pickup if first_pickup is not None else -1,
            "first_blue_score_step": first_blue_score if first_blue_score is not None else -1,
            "first_red_score_step": first_red_score if first_red_score is not None else -1,
            "first_latch_step": first_latch if first_latch is not None else -1,
            "latch_fired": int(first_latch is not None),
            "picked_up": int(first_pickup is not None),
            "blue_scored": int(blue_score > 0),
            "mean_teammate_dist": (sum_teammate / n_geo) if n_geo else 0.0,
            "mean_lateral_sep": (sum_lateral / n_geo) if n_geo else 0.0,
            "max_teammate_dist": max_teammate,
            "max_lateral_sep": max_lateral,
            "frac_opposite_lanes": steps_opposite_lanes / max(n_geo, 1),
            "frac_both_offensive": steps_both_offensive / max(n_geo, 1),
            "frac_sep_ge_12": steps_sep_ge_12 / max(n_geo, 1),
            "latch_steps": latch_steps,
            "frac_latch_both_same_target": latch_both_same_target / max(latch_steps, 1),
            "frac_latch_second_uncovered": latch_second_uncovered / max(latch_steps, 1),
            "frac_latch_target_churn": latch_target_churn / max(latch_steps, 1),
            "frac_latch_protector_chase": latch_protector_chase / max(latch_steps, 1),
            "frac_latch_both_on_carrier": latch_both_on_carrier / max(latch_steps, 1),
            "role_change_steps": role_change_steps,
            "post_pickup_steps": post_pickup_steps,
            "frac_post_support_near": post_support_near_steps / max(post_pickup_steps, 1),
            "unsupported_carry_episode": int(
                post_pickup_steps >= 5 and (post_support_near_steps / max(post_pickup_steps, 1)) < 0.5
            ),
            "isolated_return_attempts": isolated_return_attempts,
            "isolated_return_success": isolated_return_success,
            "supported_return_attempts": supported_return_attempts,
            "supported_return_success": supported_return_success,
            "protector_tagged_while_carry_steps": protector_tagged_while_carry,
            "fail_A_both_on_one": fail_a,
            "fail_B_second_uncovered": fail_b,
            "fail_C_role_churn": fail_c,
            "fail_D_protector_pressure": fail_d,
            "fail_E_ignore_support": fail_e,
        }
    finally:
        env.close()


def _rate(rows: list[dict], key: str) -> float:
    if not rows:
        return float("nan")
    return mean(float(r[key]) for r in rows)


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else float("nan")


def _summarize(rows: list[dict]) -> dict[str, Any]:
    by_style: dict[str, Any] = {}
    for style in STYLES:
        srows = [r for r in rows if r["blue_style"] == style]
        iso_att = sum(r["isolated_return_attempts"] for r in srows)
        iso_ok = sum(r["isolated_return_success"] for r in srows)
        sup_att = sum(r["supported_return_attempts"] for r in srows)
        sup_ok = sum(r["supported_return_success"] for r in srows)
        mode_counts = Counter()
        for r in srows:
            for letter, key in (
                ("A", "fail_A_both_on_one"),
                ("B", "fail_B_second_uncovered"),
                ("C", "fail_C_role_churn"),
                ("D", "fail_D_protector_pressure"),
                ("E", "fail_E_ignore_support"),
            ):
                if int(r[key]):
                    mode_counts[letter] += 1
        by_style[style] = {
            "n": len(srows),
            "mean_margin": _rate(srows, "win_margin"),
            "pickup_rate": _rate(srows, "picked_up"),
            "score_rate": _rate(srows, "blue_scored"),
            "latch_rate": _rate(srows, "latch_fired"),
            "mean_first_pickup": mean(
                r["first_pickup_step"] for r in srows if r["first_pickup_step"] >= 0
            )
            if any(r["first_pickup_step"] >= 0 for r in srows)
            else None,
            "mean_first_blue_score": mean(
                r["first_blue_score_step"] for r in srows if r["first_blue_score_step"] >= 0
            )
            if any(r["first_blue_score_step"] >= 0 for r in srows)
            else None,
            "mean_teammate_dist": _rate(srows, "mean_teammate_dist"),
            "mean_lateral_sep": _rate(srows, "mean_lateral_sep"),
            "mean_frac_opposite_lanes": _rate(srows, "frac_opposite_lanes"),
            "mean_frac_both_offensive": _rate(srows, "frac_both_offensive"),
            "mean_frac_latch_both_same_target": _rate(srows, "frac_latch_both_same_target"),
            "mean_frac_latch_second_uncovered": _rate(srows, "frac_latch_second_uncovered"),
            "mean_frac_latch_target_churn": _rate(srows, "frac_latch_target_churn"),
            "mean_frac_latch_protector_chase": _rate(srows, "frac_latch_protector_chase"),
            "mean_frac_latch_both_on_carrier": _rate(srows, "frac_latch_both_on_carrier"),
            "mean_frac_post_support_near": _rate(srows, "frac_post_support_near"),
            "unsupported_carry_rate": _rate(srows, "unsupported_carry_episode"),
            "mean_role_change_steps": _rate(srows, "role_change_steps"),
            "isolated_return_success_rate": _safe_div(iso_ok, iso_att),
            "supported_return_success_rate": _safe_div(sup_ok, sup_att),
            "isolated_return_n": iso_att,
            "supported_return_n": sup_att,
            "failure_mode_episode_counts": dict(mode_counts),
            "dominant_failure_modes": [m for m, _ in mode_counts.most_common(3)],
        }

    split = by_style["BLUE_SPLIT"]
    escort = by_style["BLUE_ESCORT"]
    # Cross-style dominance: which modes fire most on SPLIT episodes that
    # outscore ESCORT on matched seeds.
    matched_gaps = []
    split_mode_when_ahead = Counter()
    for ep in sorted({r["episode_index"] for r in rows}):
        s = next(r for r in rows if r["blue_style"] == "BLUE_SPLIT" and r["episode_index"] == ep)
        e = next(r for r in rows if r["blue_style"] == "BLUE_ESCORT" and r["episode_index"] == ep)
        gap = float(s["win_margin"]) - float(e["win_margin"])
        matched_gaps.append(gap)
        if gap > 0:
            for letter, key in (
                ("A", "fail_A_both_on_one"),
                ("B", "fail_B_second_uncovered"),
                ("C", "fail_C_role_churn"),
                ("D", "fail_D_protector_pressure"),
                ("E", "fail_E_ignore_support"),
            ):
                if int(s[key]):
                    split_mode_when_ahead[letter] += 1

    ranked = split_mode_when_ahead.most_common()
    return {
        "by_style": by_style,
        "escort_minus_split_mean_margin": float(escort["mean_margin"]) - float(split["mean_margin"]),
        "mean_matched_split_minus_escort_margin": mean(matched_gaps) if matched_gaps else float("nan"),
        "split_failure_modes_when_ahead_of_escort": dict(split_mode_when_ahead),
        "dominant_failure_mode_hypothesis": ranked[0][0] if ranked else "none",
        "failure_mode_legend": {
            "A": "both reds commit to one SPLIT attacker",
            "B": "second SPLIT lane remains uncovered",
            "C": "red roles/targets churn between threats",
            "D": "OP11 pressures ESCORT protector too effectively",
            "E": "supported and isolated carriers treated almost identically",
        },
        "selectivity_check": {
            "SPLIT_isolation_latch_rate": split["latch_rate"],
            "RUSH_unsupported_carry_rate": by_style["BLUE_RUSH"]["unsupported_carry_rate"],
            "ESCORT_false_latch_rate": escort["latch_rate"],
            "TURTLE_latch_rate": by_style["BLUE_TURTLE"]["latch_rate"],
            "target_SPLIT_latch": ">=6/8 episodes",
            "target_RUSH_unsupported": ">=6/8 episodes",
            "target_ESCORT_false_latch": "<=2/8 episodes",
            "target_TURTLE": "rare / irrelevant",
        },
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--episodes", type=int, default=8)
    p.add_argument("--base-seed", type=int, default=571001)
    p.add_argument("--maps", nargs="+", default=["map_a"])
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-decision-steps", type=int, default=240)
    p.add_argument(
        "--allow-retired-seed",
        action="store_true",
        help="Override guard against 541001/551001/561001 reuse.",
    )
    args = p.parse_args()

    if int(args.base_seed) in RETIRED_BASE_SEEDS and not args.allow_retired_seed:
        raise SystemExit(
            f"base-seed {args.base_seed} is retired for OP11 rescue. "
            "Use a fresh block (e.g. 571001) or pass --allow-retired-seed."
        )

    maps = [normalize_map_layout(m) if m in ("map_a", "a", "open") else m for m in args.maps]
    # Keep user-facing alias in artifacts when they asked for map_a.
    display_maps = list(args.maps)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    manifest = {
        "protocol": "op11_isolation_failure_phase1",
        "blue_probe_protocol": "BLUE_PROBES_V3",
        "red": RED,
        "maps_requested": display_maps,
        "maps_normalized": [normalize_map_layout(m) for m in display_maps],
        "episodes": int(args.episodes),
        "base_seed": int(args.base_seed),
        "device": str(args.device),
        "max_decision_steps": int(args.max_decision_steps),
        "styles": list(STYLES),
        "retired_seeds_guard": sorted(RETIRED_BASE_SEEDS),
        "note": "Phase-1 diagnosis only — no BT redesign until failure mode ranked.",
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    for map_name in display_maps:
        for ep in range(int(args.episodes)):
            for style in STYLES:
                print(f"[op11 phase1] map={map_name} {style} ep={ep}", flush=True)
                rows.append(
                    _run_episode(
                        style=style,
                        episode_index=ep,
                        base_seed=int(args.base_seed),
                        map_name=map_name,
                        max_steps=int(args.max_decision_steps),
                        device=str(args.device),
                    )
                )

    csv_path = out_dir / "phase1_rows.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = _summarize(rows)
    summary["manifest"] = manifest
    (out_dir / "phase1_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Human-readable verdict block.
    lines = [
        "OP11 Phase-1 isolation failure diagnosis",
        f"base_seed={args.base_seed} maps={display_maps} episodes={args.episodes}",
        f"dominant_hypothesis={summary['dominant_failure_mode_hypothesis']}",
        f"ESCORT-SPLIT mean margin={summary['escort_minus_split_mean_margin']:+.3f}",
        f"matched SPLIT-ESCORT margin={summary['mean_matched_split_minus_escort_margin']:+.3f}",
        "selectivity: "
        + ", ".join(f"{k}={v:.3f}" for k, v in summary["selectivity_check"].items() if k.endswith("_rate")),
        "SPLIT failure modes when ahead of ESCORT: "
        + json.dumps(summary["split_failure_modes_when_ahead_of_escort"]),
    ]
    for style in STYLES:
        s = summary["by_style"][style]
        lines.append(
            f"{style}: margin={s['mean_margin']:+.3f} latch={s['latch_rate']:.3f} "
            f"pickup={s['pickup_rate']:.3f} score={s['score_rate']:.3f} "
            f"modes={s['failure_mode_episode_counts']}"
        )
    text = "\n".join(lines) + "\n"
    (out_dir / "phase1_verdict.txt").write_text(text, encoding="utf-8")
    print(text, flush=True)


if __name__ == "__main__":
    main()
