#!/usr/bin/env python3
"""Latch + trajectory micro-gates for OP7 separated-threat SPLIT lever.

Runs BLUE_PROBES_V3 styles vs OP7 on map_a. Does not tune mid-run.
Judges detector selectivity and whether the second lane is uncovered after latch.
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
    BLUE_PROBE_PROTOCOL,
    _episode_seed,
    _make_env,
    _zero_action,
    artifact_map_label,
)

BLUE_STYLES = (
    "BLUE_RUSH",
    "BLUE_SPLIT",
    "BLUE_ESCORT",
    "BLUE_TURTLE",
)
RED_STYLE = "OP7_DEEP_FORTRESS"
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
        center_y = float(core.rows) * 0.5
        audit_radius = float(core.cfg.tag_range_cells) * 1.5

        first_blue_pickup = -1
        first_blue_score = -1
        latch_step = -1
        compact_step = -1
        response_steps = 0
        compact_response_steps = 0
        same_corridor_steps = 0
        uncovered_lane_steps = 0
        uncovered_after_latch = 0
        max_lateral_sep = 0.0
        steps = 0
        last_info: dict[str, Any] = {}

        for _ in range(int(max_decision_steps) + 5):
            action = _zero_action(env)
            env.step_async(action)
            _, _, done, infos = env.step_wait()
            last_info = infos[0] if infos else {}

            sim = int(core.sim_step_count[0].item())
            blue_carry = bool(core.blue_carrying[0].any().item())
            if first_blue_pickup < 0 and blue_carry:
                first_blue_pickup = sim

            latch_on = int(core.bt_op7_split_first_trigger_step[0].item()) >= 0 and (
                int(core.bt_op7_split_response_expiry_step[0].item()) >= sim
            )
            compact_on = int(core.bt_op7_compact_first_trigger_step[0].item()) >= 0 and (
                int(core.bt_op7_compact_response_expiry_step[0].item()) >= sim
            )
            if latch_on:
                response_steps += 1
                if latch_step < 0:
                    latch_step = int(core.bt_op7_split_first_trigger_step[0].item())
            if compact_on:
                compact_response_steps += 1
                if compact_step < 0:
                    compact_step = int(core.bt_op7_compact_first_trigger_step[0].item())

            rx = core.red_x[0].detach().cpu().numpy()
            ry = core.red_y[0].detach().cpu().numpy()
            bx = core.blue_x[0].detach().cpu().numpy()
            by = core.blue_y[0].detach().cpu().numpy()
            blue_alive = core.blue_alive[0].detach().cpu().numpy().astype(bool)
            red_alive = core.red_alive[0].detach().cpu().numpy().astype(bool)
            red_tagged = core.red_tagged[0].detach().cpu().numpy().astype(bool)

            lateral_sep = abs(float(by[0] - by[1]))
            max_lateral_sep = max(max_lateral_sep, lateral_sep)

            if latch_on and red_alive.any() and hasattr(core, "_debug_red_target_y"):
                primary = int(core.bt_op7_split_primary_blue_idx[0].item())
                if primary < 0:
                    primary = 0
                corridor_y = float(core.bt_op7_split_corridor_y[0].item())
                ty = core._debug_red_target_y[0].detach().cpu().numpy()
                live = red_alive & (~red_tagged)
                if int(live.sum()) >= 2:
                    other = 1 - primary
                    same_corr = (
                        abs(float(ty[0]) - corridor_y) < abs(float(ty[0]) - float(by[other]))
                        and abs(float(ty[1]) - corridor_y) < abs(float(ty[1]) - float(by[other]))
                    )
                    if same_corr:
                        same_corridor_steps += 1

                blue_on_red = blue_alive & (bx > midline)
                if bool(blue_on_red.all()):
                    lane_sides = np.sign(by - center_y)
                    red_to_blue = np.sqrt(
                        (rx[:, None] - bx[None, :]) ** 2 + (ry[:, None] - by[None, :]) ** 2
                    )
                    red_to_each = red_to_blue.min(axis=0)
                    uncovered = blue_on_red & (red_to_each > audit_radius)
                    if lane_sides[0] * lane_sides[1] < 0 and bool(uncovered.any()):
                        uncovered_lane_steps += 1
                        uncovered_after_latch += 1

            ep_res = last_info.get("episode_result", last_info)
            blue_score_now = int(ep_res.get("blue_score", core.blue_score[0].item()))
            if first_blue_score < 0 and blue_score_now > 0:
                first_blue_score = sim
            steps += 1
            if bool(done.any()):
                break

        ep_res = last_info.get("episode_result", last_info) if last_info else {}
        blue_score = int(ep_res.get("blue_score", core.blue_score[0].item()))
        red_score = int(ep_res.get("red_score", core.red_score[0].item()))
        latched = int(core.bt_op7_split_activations[0].item()) > 0 or latch_step >= 0
        compact = int(core.bt_op7_compact_activations[0].item()) > 0 or compact_step >= 0
        return {
            "blue_style": blue_style,
            "blue_probe_protocol": BLUE_PROBE_PROTOCOL,
            "red_style": RED_STYLE,
            "map": artifact_map_label(map_name),
            "episode_index": episode_index,
            "episode_seed": episode_seed,
            "steps": steps,
            "blue_score": blue_score,
            "red_score": red_score,
            "win_margin": blue_score - red_score,
            "first_blue_pickup": first_blue_pickup,
            "first_blue_score": first_blue_score,
            "time_to_first_score": first_blue_score if first_blue_score >= 0 else "",
            "latch_triggered": int(latched),
            "latch_step": latch_step,
            "compact_triggered": int(compact),
            "compact_step": compact_step,
            "response_steps": response_steps,
            "compact_response_steps": compact_response_steps,
            "same_corridor_steps": same_corridor_steps,
            "uncovered_lane_steps": uncovered_lane_steps,
            "uncovered_after_latch_steps": uncovered_after_latch,
            "max_lateral_sep": round(max_lateral_sep, 3),
            "op7_split_activations": int(core.bt_op7_split_activations[0].item()),
            "op7_split_max_lateral_sep": float(core.bt_op7_split_max_lateral_sep[0].item()),
            "op7_split_max_teammate_dist": float(core.bt_op7_split_max_teammate_dist[0].item()),
            "op7_compact_activations": int(core.bt_op7_compact_activations[0].item()),
            "op7_compact_min_teammate_dist": float(core.bt_op7_compact_min_teammate_dist[0].item()),
            "op7_split_lever_enabled": int(bool(getattr(type(core), "_OP7_SPLIT_LEVER_ENABLED", True))),
            "op7_compact_lever_enabled": int(
                bool(getattr(type(core), "_OP7_COMPACT_LEVER_ENABLED", True))
            ),
        }
    finally:
        env.close()


def _mean(vals: list[float]) -> float:
    return float(sum(vals) / len(vals)) if vals else float("nan")


def _judge(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_style: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_style[r["blue_style"]].append(r)

    latch_counts = {
        s.replace("BLUE_", ""): sum(int(r["latch_triggered"]) for r in eps)
        for s, eps in by_style.items()
    }
    compact_counts = {
        s.replace("BLUE_", ""): sum(int(r["compact_triggered"]) for r in eps)
        for s, eps in by_style.items()
    }
    n = max((len(v) for v in by_style.values()), default=0)
    gates = {
        "split_latch_ge_6_of_8": latch_counts.get("SPLIT", 0) >= 6,
        "split_compact_le_1_of_8": compact_counts.get("SPLIT", 0) <= 1,
        "rush_compact_ge_6_of_8": compact_counts.get("RUSH", 0) >= 6,
        "escort_compact_ge_5_of_8": compact_counts.get("ESCORT", 0) >= 5,
        "turtle_compact_le_2_of_8": compact_counts.get("TURTLE", 0) <= 2,
        # Preserve prior SPLIT false-latch floors for concentrated styles.
        "rush_false_split_latch_le_2_of_8": latch_counts.get("RUSH", 0) <= 2,
        "escort_false_split_latch_le_2_of_8": latch_counts.get("ESCORT", 0) <= 2,
        "turtle_false_split_latch_le_2_of_8": latch_counts.get("TURTLE", 0) <= 2,
    }

    style_stats: dict[str, Any] = {}
    for s, eps in sorted(by_style.items()):
        ttfs = [float(r["first_blue_score"]) for r in eps if int(r["first_blue_score"]) >= 0]
        style_stats[s] = {
            "n": len(eps),
            "latch_count": sum(int(r["latch_triggered"]) for r in eps),
            "compact_count": sum(int(r["compact_triggered"]) for r in eps),
            "mean_win_margin": _mean([float(r["win_margin"]) for r in eps]),
            "mean_ttfs": _mean(ttfs),
            "mean_uncovered_after_latch": _mean(
                [float(r["uncovered_after_latch_steps"]) for r in eps]
            ),
            "mean_same_corridor_steps": _mean([float(r["same_corridor_steps"]) for r in eps]),
            "mean_response_steps": _mean([float(r["response_steps"]) for r in eps]),
            "mean_compact_response_steps": _mean(
                [float(r["compact_response_steps"]) for r in eps]
            ),
        }

    split = style_stats.get("BLUE_SPLIT", {})
    rush = style_stats.get("BLUE_RUSH", {})
    tactical = {
        "split_uncovered_after_latch_gt_0": float(split.get("mean_uncovered_after_latch", 0)) > 0,
        "split_same_corridor_gt_0": float(split.get("mean_same_corridor_steps", 0)) > 0,
        "split_ttfs_near_improved_38": (
            float(split.get("mean_ttfs", 999)) < 60.0
            if split.get("mean_ttfs") == split.get("mean_ttfs")
            else False
        ),
        "rush_margin_not_above_prior_tie": float(rush.get("mean_win_margin", 99)) <= 1.75,
    }
    all_pass = all(gates.values()) and all(
        [
            tactical["split_uncovered_after_latch_gt_0"],
            tactical["split_same_corridor_gt_0"],
            tactical["split_ttfs_near_improved_38"],
        ]
    )
    return {
        "episodes_per_style": n,
        "latch_counts": latch_counts,
        "compact_counts": compact_counts,
        "gates": gates,
        "tactical": tactical,
        "style_stats": style_stats,
        "microgates_pass": bool(all_pass),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=8)
    p.add_argument("--base-seed", type=int, default=610001)
    p.add_argument("--map", default=DEFAULT_MAP)
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-decision-steps", type=int, default=240)
    args = p.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for style in BLUE_STYLES:
        for ep_i in range(int(args.episodes)):
            # Paired across styles: same (red, map, episode) seed — never style-keyed.
            seed = _episode_seed(int(args.base_seed), red_index=0, map_index=0, episode_index=ep_i)
            row = _run_episode(
                blue_style=style,
                episode_index=ep_i,
                episode_seed=seed,
                map_name=str(args.map),
                max_decision_steps=int(args.max_decision_steps),
                device=str(args.device),
            )
            rows.append(row)
            print(
                f"[{style} ep{ep_i}] split={row['latch_triggered']} "
                f"compact={row['compact_triggered']} "
                f"ttfs={row['first_blue_score']} margin={row['win_margin']} "
                f"uncovered_after={row['uncovered_after_latch_steps']}",
                flush=True,
            )

    csv_path = out_dir / "episode_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summary = _judge(rows)
    manifest = {
        "protocol": "op7_split_and_compact_microgates",
        "blue_probe_protocol": BLUE_PROBE_PROTOCOL,
        "red_style": RED_STYLE,
        "map": artifact_map_label(str(args.map)),
        "episodes": int(args.episodes),
        "base_seed": int(args.base_seed),
        "op7_split_lever_enabled": True,
        "op7_split_response_enabled": True,
        "op7_compact_lever_enabled": True,
        "op7_compact_response_enabled": True,
        "summary": summary,
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out_dir / "microgate_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\n=== Micro-gate summary ===")
    print(json.dumps(summary, indent=2))
    print(f"MICROGATES_PASS: {summary['microgates_pass']}")
    print(f"Artifacts: {out_dir}")
    return 0 if summary["microgates_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
