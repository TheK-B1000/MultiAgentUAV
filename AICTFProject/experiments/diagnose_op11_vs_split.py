"""Diagnose why OP11's latched split-isolation still loses to BLUE_SPLIT.

Measures the conversion funnel under matched seeds:
  approach -> first pickup -> first blue score -> final margin

Compares against BLUE_ESCORT on the same seeds so the ESCORT niche gap is
visible in the same units. Does not change BT code.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import mean

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gpu_env._core._bt_red import ROLE_INTERCEPTOR
from experiments.run_scripted_style_payoff_matrix import _make_env, _zero_action

RED = "OP11_ADAPTIVE_EXPLOITER"
MAP_NAME = "map_b_split_lane"
STYLES = ("BLUE_SPLIT", "BLUE_ESCORT")


def _as_float(value) -> float:
    try:
        if hasattr(value, "detach"):
            value = value.detach().flatten()[0]
        if hasattr(value, "item"):
            value = value.item()
        return float(value)
    except Exception:
        return float("nan")


def _run_episode(style: str, episode_index: int, base_seed: int, max_steps: int, device: str) -> dict:
    seed = int(base_seed + episode_index)
    env = _make_env(map_name=MAP_NAME, seed=seed, max_decision_steps=max_steps, device=device)
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

        first_mid = None
        first_pickup = None
        first_blue_score = None
        first_red_score = None
        first_latch = None
        latch_pre_pickup_steps = 0
        latch_post_pickup_steps = 0
        both_interceptor_steps = 0
        max_teammate = 0.0
        max_lat = 0.0

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
            midline = float(core.cols) * 0.5
            if first_mid is None and (bx0 > midline or bx1 > midline):
                first_mid = step
            teammate = ((bx0 - bx1) ** 2 + (by0 - by1) ** 2) ** 0.5
            lat = abs(by0 - by1)
            max_teammate = max(max_teammate, teammate)
            max_lat = max(max_lat, lat)

            carrying = bool(core.blue_carrying[0].any().item())
            if first_pickup is None and carrying:
                first_pickup = step

            latch_on = int(core.bt_adapt_split_first_trigger_step[0].item()) >= 0
            if latch_on and first_latch is None:
                first_latch = step
            if latch_on:
                if carrying:
                    latch_post_pickup_steps += 1
                else:
                    latch_pre_pickup_steps += 1
                roles = [int(x) for x in core.bt_red_role[0].tolist()]
                if roles.count(int(ROLE_INTERCEPTOR)) >= 2:
                    both_interceptor_steps += 1

            blue_score = int(core.blue_score[0].item())
            red_score = int(core.red_score[0].item())
            if first_blue_score is None and blue_score > prev_blue:
                first_blue_score = step
            if first_red_score is None and red_score > prev_red:
                first_red_score = step
            prev_blue = blue_score
            prev_red = red_score
            if bool(done.any()):
                break

        return {
            "blue_style": style,
            "episode_index": episode_index,
            "episode_seed": seed,
            "steps": step + 1,
            "blue_score": blue_score,
            "red_score": red_score,
            "win_margin": blue_score - red_score,
            "first_midfield_step": first_mid if first_mid is not None else -1,
            "first_pickup_step": first_pickup if first_pickup is not None else -1,
            "first_blue_score_step": first_blue_score if first_blue_score is not None else -1,
            "first_red_score_step": first_red_score if first_red_score is not None else -1,
            "first_latch_step": first_latch if first_latch is not None else -1,
            "latch_pre_pickup_steps": latch_pre_pickup_steps,
            "latch_post_pickup_steps": latch_post_pickup_steps,
            "both_interceptor_steps": both_interceptor_steps,
            "max_teammate_dist": max_teammate,
            "max_lateral_sep": max_lat,
            "picked_up": int(first_pickup is not None),
            "blue_scored": int(blue_score > 0),
            "latch_fired": int(first_latch is not None),
            "latch_before_pickup": int(
                first_latch is not None
                and first_pickup is not None
                and first_latch < first_pickup
            ),
        }
    finally:
        env.close()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--episodes", type=int, default=8)
    p.add_argument("--base-seed", type=int, default=541001)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-decision-steps", type=int, default=240)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for ep in range(args.episodes):
        for style in STYLES:
            print(f"[op11 funnel] {style} ep={ep}", flush=True)
            rows.append(
                _run_episode(style, ep, args.base_seed, args.max_decision_steps, args.device)
            )

    csv_path = out_dir / "funnel_rows.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {}
    for style in STYLES:
        srows = [r for r in rows if r["blue_style"] == style]
        summary[style] = {
            "n": len(srows),
            "mean_margin": mean(r["win_margin"] for r in srows),
            "pickup_rate": mean(r["picked_up"] for r in srows),
            "score_rate": mean(r["blue_scored"] for r in srows),
            "latch_rate": mean(r["latch_fired"] for r in srows),
            "latch_before_pickup_rate": mean(r["latch_before_pickup"] for r in srows),
            "mean_first_latch": mean(
                r["first_latch_step"] for r in srows if r["first_latch_step"] >= 0
            )
            if any(r["first_latch_step"] >= 0 for r in srows)
            else None,
            "mean_first_pickup": mean(
                r["first_pickup_step"] for r in srows if r["first_pickup_step"] >= 0
            )
            if any(r["first_pickup_step"] >= 0 for r in srows)
            else None,
            "mean_latch_pre_pickup_steps": mean(r["latch_pre_pickup_steps"] for r in srows),
            "mean_latch_post_pickup_steps": mean(r["latch_post_pickup_steps"] for r in srows),
            "mean_both_interceptor_steps": mean(r["both_interceptor_steps"] for r in srows),
        }

    (out_dir / "funnel_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
