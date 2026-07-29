#!/usr/bin/env python3
"""Map C V2 TURTLE micro-gate (frozen geometry; no wall retune).

Default opponent: OP6 dual-assault (strongest empty-home punisher candidate).

Intended causal chain:
  RUSH/SPLIT/ESCORT abandon home → red uses the single top gap → scores
  TURTLE anchors the gap → stops red → counter-scores

Require before any four-style payoff matrix (8 seeds):
  red first-score vs RUSH/SPLIT/ESCORT: frequent (≥5/8 each)
  red first-score vs TURTLE: rare (≤2/8)
  TURTLE stop-at-gap: frequent (≥5/8)
  TURTLE counter-after-stop: frequent (≥5/8)
  geometry: 0 bottom bypasses; returns/assaults use top gap
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
    _zero_action,
)
from gpu_env import GPUCTFVecEnv, GPUFieldConfig  # noqa: E402
from gpu_env._maps import MAP_C_HOME_CORRIDOR, normalize_map_layout  # noqa: E402

BLUE_STYLES = ("BLUE_RUSH", "BLUE_SPLIT", "BLUE_ESCORT", "BLUE_TURTLE")
DEFAULT_RED = "OP6_IMMEDIATE_DUAL_RUSH"
DEFAULT_MAP = "map_c_home_corridor"


def _make_env(*, seed: int, max_decision_steps: int, device: str) -> GPUCTFVecEnv:
    # Force non-mirrored wall so the mandatory gap is always at the TOP
    # (matches the frozen V2 contract under test).
    cfg = GPUFieldConfig(
        n_envs=1,
        max_blue_agents=2,
        max_red_agents=2,
        map_layout=normalize_map_layout(DEFAULT_MAP),
        map_b_vertical_mirror_prob=0.0,
        max_decision_steps=int(max_decision_steps),
        aquaticus_profile=True,
        rules_profile="OURS",
        device=str(device),
        seed=int(seed),
    )
    return GPUCTFVecEnv(cfg)


def _gap_bounds(core) -> tuple[float, float, float, float, float]:
    rect = core.obstacle_rects[0, 0]
    x0, y0, x1, y1 = (float(rect[i].item()) for i in range(4))
    max_y = float(max(0, core.rows - 1))
    return x0, y0, x1, y1, max_y


def _in_top_gap(x: torch.Tensor, y: torch.Tensor, x0: float, y0: float, x1: float) -> torch.Tensor:
    # Crossing the wall's x-band above the wall top edge.
    return (x >= x0) & (x <= x1) & (y < y0)


def _bottom_bypass(x: torch.Tensor, y: torch.Tensor, x0: float, x1: float, y1: float, max_y: float) -> torch.Tensor:
    # Any agent in the wall x-band below a sealed bottom is a geometry leak.
    # With V2, y1 == max_y so this should never fire for centers.
    return (x >= x0) & (x <= x1) & (y > y1 + 1e-3) & (y <= max_y + 1e-3)


def _run_episode(
    *,
    blue_style: str,
    red_style: str,
    episode_index: int,
    episode_seed: int,
    max_decision_steps: int,
    device: str,
) -> dict[str, Any]:
    env = _make_env(seed=episode_seed, max_decision_steps=max_decision_steps, device=device)
    try:
        core = env.core
        env.env_method("set_phase", red_style)
        env.env_method("set_next_opponent", "SCRIPTED", red_style)
        core.blue_scripted = True
        core.set_blue_style(blue_style)
        env.reset()
        env.env_method("set_phase", red_style)
        env.env_method("set_next_opponent", "SCRIPTED", red_style)
        core.blue_scripted = True
        core.set_blue_style(blue_style)

        assert str(core.map_layout) == MAP_C_HOME_CORRIDOR
        x0, y0, x1, y1, max_y = _gap_bounds(core)
        home_x = float(core.blue_flag_home[0, 0].item())
        midline = float(core.cols) * 0.5

        red_first_score_step = -1
        blue_first_score_step = -1
        red_top_gap_cross = 0
        blue_top_gap_cross = 0
        bottom_bypass_events = 0
        router_stall_steps = 0
        assault_stop_at_gap = 0
        turtle_anchor_at_gap_stop = 0
        turtle_counter_after_stop = 0
        stop_seen = False
        prev_red_tagged = core.red_tagged.detach().clone()
        prev_blue_score = 0
        steps = 0
        last_info: dict[str, Any] = {}

        for _ in range(int(max_decision_steps) + 5):
            prev_rx = core.red_x.detach().clone()
            action = _zero_action(env)
            env.step_async(action)
            _, _, done, infos = env.step_wait()
            last_info = infos[0] if infos else {}
            steps += 1
            ep_res = last_info.get("episode_result", last_info)
            blue_score_now = int(ep_res.get("blue_score", core.blue_score[0].item()))
            red_score_now = int(ep_res.get("red_score", core.red_score[0].item()))

            rx, ry = core.red_x[0], core.red_y[0]
            bx, by = core.blue_x[0], core.blue_y[0]
            if bool((_in_top_gap(rx, ry, x0, y0, x1) & core.red_alive[0]).any().item()):
                red_top_gap_cross += 1
            if bool((_in_top_gap(bx, by, x0, y0, x1) & core.blue_alive[0]).any().item()):
                blue_top_gap_cross += 1
            if bool((_bottom_bypass(rx, ry, x0, x1, y1, max_y) & core.red_alive[0]).any().item()):
                bottom_bypass_events += 1
            if bool((_bottom_bypass(bx, by, x0, x1, y1, max_y) & core.blue_alive[0]).any().item()):
                bottom_bypass_events += 1

            # Stall proxy: alive red barely moved while straddling the wall x-band.
            moved = torch.abs(core.red_x[0] - prev_rx[0]) + torch.abs(
                core.red_y[0] - getattr(core, "_prev_ry_mapc", core.red_y).detach()[0]
            )
            core._prev_ry_mapc = core.red_y.detach().clone()
            near_wall_x = (torch.minimum(prev_rx[0], core.red_x[0]) <= x1) & (
                torch.maximum(prev_rx[0], core.red_x[0]) >= x0
            )
            stalled = core.red_alive[0] & near_wall_x & (moved < 0.02)
            if bool(stalled.any().item()):
                router_stall_steps += 1

            newly_tagged = (~prev_red_tagged) & core.red_tagged
            on_blue = core.red_x[0] < midline
            in_gap = _in_top_gap(core.red_x[0], core.red_y[0], x0, y0, x1) | (
                (core.red_x[0] >= x0 - 1.5)
                & (core.red_x[0] <= x1 + 1.5)
                & (core.red_y[0] <= y0 + 1.0)
            )
            if bool((newly_tagged[0] & on_blue & in_gap).any().item()):
                assault_stop_at_gap += 1
                stop_seen = True
                anchor = (
                    core.blue_alive[0]
                    & (~core.blue_carrying[0])
                    & _in_top_gap(core.blue_x[0], core.blue_y[0], x0 - 1.0, y0 + 2.0, x1 + 1.0)
                ) | (
                    core.blue_alive[0]
                    & (~core.blue_carrying[0])
                    & (core.blue_x[0] < midline)
                    & (torch.abs(core.blue_x[0] - home_x) <= 6.0)
                    & (core.blue_y[0] <= y0 + 2.0)
                )
                if bool(anchor.any().item()):
                    turtle_anchor_at_gap_stop += 1
            prev_red_tagged = core.red_tagged.detach().clone()

            if red_first_score_step < 0 and red_score_now > 0:
                red_first_score_step = steps
            if blue_first_score_step < 0 and blue_score_now > 0:
                blue_first_score_step = steps
            if (
                blue_style == "BLUE_TURTLE"
                and stop_seen
                and blue_score_now > prev_blue_score
            ):
                turtle_counter_after_stop = 1
            prev_blue_score = blue_score_now
            if bool(done.any()):
                break

        ep_res = last_info.get("episode_result", last_info) if last_info else {}
        blue_score = int(ep_res.get("blue_score", core.blue_score[0].item()))
        red_score = int(ep_res.get("red_score", core.red_score[0].item()))
        return {
            "blue_style": blue_style,
            "red_style": red_style,
            "map": DEFAULT_MAP,
            "map_version": "map_c_v2",
            "episode_index": episode_index,
            "episode_seed": episode_seed,
            "wall_y0": y0,
            "wall_y1": y1,
            "red_first_score_step": red_first_score_step,
            "blue_first_score_step": blue_first_score_step,
            "red_scored_first": int(
                red_first_score_step >= 0
                and (blue_first_score_step < 0 or red_first_score_step <= blue_first_score_step)
            ),
            "red_top_gap_steps": red_top_gap_cross,
            "blue_top_gap_steps": blue_top_gap_cross,
            "bottom_bypass_events": bottom_bypass_events,
            "router_stall_steps": router_stall_steps,
            "assault_stop_at_gap": assault_stop_at_gap,
            "turtle_anchor_at_gap_stop": turtle_anchor_at_gap_stop,
            "turtle_counter_after_stop": turtle_counter_after_stop,
            "used_top_gap": int(red_top_gap_cross > 0 or blue_top_gap_cross > 0),
            "blue_score": blue_score,
            "red_score": red_score,
            "win_margin": blue_score - red_score,
            "steps": steps,
        }
    finally:
        env.close()


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by[str(r["blue_style"])].append(r)
    styles: dict[str, Any] = {}
    for style, rs in by.items():
        n = len(rs)
        styles[style] = {
            "n": n,
            "red_first_score_seeds": sum(int(r["red_scored_first"]) for r in rs),
            "stop_at_gap_seeds": sum(int(r["assault_stop_at_gap"] > 0) for r in rs),
            "anchor_at_gap_stop_seeds": sum(int(r["turtle_anchor_at_gap_stop"] > 0) for r in rs),
            "counter_after_stop_seeds": sum(int(r["turtle_counter_after_stop"]) for r in rs),
            "top_gap_use_seeds": sum(int(r["used_top_gap"]) for r in rs),
            "bottom_bypass_events": sum(int(r["bottom_bypass_events"]) for r in rs),
            "router_stall_steps_sum": sum(int(r["router_stall_steps"]) for r in rs),
            "mean_win_margin": sum(float(r["win_margin"]) for r in rs) / max(n, 1),
        }
    n = max((styles[s]["n"] for s in BLUE_STYLES if s in styles), default=8)
    rush, split, escort, turtle = (
        styles.get("BLUE_RUSH", {}),
        styles.get("BLUE_SPLIT", {}),
        styles.get("BLUE_ESCORT", {}),
        styles.get("BLUE_TURTLE", {}),
    )
    gates = {
        "rush_red_first_ge_5_8": int(rush.get("red_first_score_seeds", 0)) >= max(5, (5 * n) // 8),
        "split_red_first_ge_5_8": int(split.get("red_first_score_seeds", 0)) >= max(5, (5 * n) // 8),
        "escort_red_first_ge_5_8": int(escort.get("red_first_score_seeds", 0)) >= max(5, (5 * n) // 8),
        "turtle_red_first_le_2_8": int(turtle.get("red_first_score_seeds", 99)) <= max(2, (2 * n) // 8),
        "turtle_stop_at_gap_ge_5_8": int(turtle.get("stop_at_gap_seeds", 0)) >= max(5, (5 * n) // 8),
        "turtle_counter_ge_5_8": int(turtle.get("counter_after_stop_seeds", 0)) >= max(5, (5 * n) // 8),
        "zero_bottom_bypasses": sum(int(r["bottom_bypass_events"]) for r in rows) == 0,
    }
    return {
        "map": DEFAULT_MAP,
        "map_version": "map_c_v2_frozen",
        "styles": styles,
        "gates": gates,
        "gates_pass": all(gates.values()),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=8)
    p.add_argument("--base-seed", type=int, default=801001)
    p.add_argument("--red", default=DEFAULT_RED)
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-decision-steps", type=int, default=240)
    p.add_argument("--out-dir", type=Path, default=Path("artifacts/mapc_v2_turtle_microgates_op6_8seed"))
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    total = len(BLUE_STYLES) * int(args.episodes)
    k = 0
    for ep in range(int(args.episodes)):
        seed = _episode_seed(int(args.base_seed), red_index=0, map_index=0, episode_index=ep)
        for style in BLUE_STYLES:
            k += 1
            row = _run_episode(
                blue_style=style,
                red_style=str(args.red),
                episode_index=ep,
                episode_seed=seed,
                max_decision_steps=int(args.max_decision_steps),
                device=str(args.device),
            )
            rows.append(row)
            print(
                f"[{k}/{total}] {style} ep={ep} red_1st={row['red_scored_first']} "
                f"gap_stop={row['assault_stop_at_gap']} bypass={row['bottom_bypass_events']} "
                f"margin={row['win_margin']}",
                flush=True,
            )

    summary = _summarize(rows)
    summary["base_seed"] = int(args.base_seed)
    summary["red"] = str(args.red)
    csv_path = args.out_dir / "episode_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    json_path = args.out_dir / "turtle_microgate_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["gates"], indent=2))
    print(f"gates_pass={summary['gates_pass']}")
    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")
    return 0 if summary["gates_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
