#!/usr/bin/env python3
"""Map C V2 ESCORT micro-gate (frozen geometry; no wall retune).

Default opponent: OP11 adaptive exploiter (aggressive return-corridor
contester candidate). Compare supported (BLUE_ESCORT) vs brief-screen /
unsupported styles (BLUE_RUSH, BLUE_SPLIT) on the same top-gap return.

Intended causal chain:
  unsupported carrier → trapped/tagged at the top gap
  supported carrier → protector interposes → carrier passes → scores

Measure (do not accept a four-style payoff until these separate):
  unsupported return success (RUSH/SPLIT)
  supported return success (ESCORT)
  mean carrier–protector distance during blue carry
  protector interposition at gap
  tag location (gap vs elsewhere)
  pickup-to-score conversion

Hard distinction: ESCORT support must be persistent and useful; RUSH's
brief screen must not receive the same advantage.
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

# Include RUSH as the critical contrast (brief screen ≠ persistent escort).
BLUE_STYLES = ("BLUE_RUSH", "BLUE_SPLIT", "BLUE_ESCORT")
DEFAULT_RED = "OP11_ADAPTIVE_EXPLOITER"
DEFAULT_MAP = "map_c_home_corridor"


def _make_env(*, seed: int, max_decision_steps: int, device: str) -> GPUCTFVecEnv:
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
    return (x >= x0) & (x <= x1) & (y < y0)


def _near_gap(
    x: torch.Tensor, y: torch.Tensor, x0: float, y0: float, x1: float, *, pad: float = 2.0
) -> torch.Tensor:
    return (x >= x0 - pad) & (x <= x1 + pad) & (y <= y0 + pad)


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

        blue_pickup = False
        blue_converted = False
        blue_carry_steps = 0
        protector_dist_sum = 0.0
        protector_dist_n = 0
        interpose_at_gap_steps = 0
        carrier_gap_steps = 0
        tag_at_gap = 0
        tag_elsewhere = 0
        bottom_bypass_events = 0
        return_success = 0  # pickup then score at least once
        prev_blue_carry = False
        prev_blue_tagged = core.blue_tagged.detach().clone()
        steps = 0
        last_info: dict[str, Any] = {}

        for _ in range(int(max_decision_steps) + 5):
            action = _zero_action(env)
            env.step_async(action)
            _, _, done, infos = env.step_wait()
            last_info = infos[0] if infos else {}
            steps += 1
            ep_res = last_info.get("episode_result", last_info)
            blue_score_now = int(ep_res.get("blue_score", core.blue_score[0].item()))

            bx, by = core.blue_x[0], core.blue_y[0]
            carry = core.blue_carrying[0] & core.blue_alive[0]
            if bool(carry.any().item()):
                if not prev_blue_carry:
                    blue_pickup = True
                blue_carry_steps += 1
                carr_idx = int(torch.argmax(carry.to(torch.int64)).item())
                prot = core.blue_alive[0] & (~core.blue_carrying[0])
                if bool(prot.any().item()):
                    prot_idx = int(torch.argmax(prot.to(torch.int64)).item())
                    dx = float(bx[prot_idx] - bx[carr_idx])
                    dy = float(by[prot_idx] - by[carr_idx])
                    dist = (dx * dx + dy * dy) ** 0.5
                    protector_dist_sum += dist
                    protector_dist_n += 1
                    # Interposition: protector between carrier and nearest red,
                    # while either is at the gap.
                    rx, ry = core.red_x[0], core.red_y[0]
                    red_alive = core.red_alive[0]
                    if bool(red_alive.any().item()):
                        d2 = (rx - bx[carr_idx]) ** 2 + (ry - by[carr_idx]) ** 2
                        d2 = torch.where(red_alive, d2, torch.full_like(d2, 1e9))
                        ti = int(torch.argmin(d2).item())
                        cx, cy = float(bx[carr_idx]), float(by[carr_idx])
                        px, py = float(bx[prot_idx]), float(by[prot_idx])
                        tx, ty = float(rx[ti]), float(ry[ti])
                        # Protector closer to threat than carrier is, and near gap.
                        prot_threat = ((px - tx) ** 2 + (py - ty) ** 2) ** 0.5
                        carr_threat = ((cx - tx) ** 2 + (cy - ty) ** 2) ** 0.5
                        at_gap = bool(
                            _near_gap(
                                torch.tensor([cx, px]),
                                torch.tensor([cy, py]),
                                x0,
                                y0,
                                x1,
                            ).any().item()
                        )
                        if at_gap and prot_threat + 0.25 < carr_threat and dist <= 4.0:
                            interpose_at_gap_steps += 1
                if bool(_in_top_gap(bx[carr_idx : carr_idx + 1], by[carr_idx : carr_idx + 1], x0, y0, x1).item()):
                    carrier_gap_steps += 1
                if blue_score_now > 0:
                    blue_converted = True
                    return_success = 1

            newly_tagged = (~prev_blue_tagged) & core.blue_tagged
            if bool(newly_tagged[0].any().item()):
                tagged_y = by[newly_tagged[0]]
                tagged_x = bx[newly_tagged[0]]
                if bool(_near_gap(tagged_x, tagged_y, x0, y0, x1, pad=2.5).any().item()):
                    tag_at_gap += 1
                else:
                    tag_elsewhere += 1
            prev_blue_tagged = core.blue_tagged.detach().clone()

            bypass = (bx >= x0) & (bx <= x1) & (by > y1 + 1e-3)
            if bool((bypass & core.blue_alive[0]).any().item()):
                bottom_bypass_events += 1
            rbypass = (core.red_x[0] >= x0) & (core.red_x[0] <= x1) & (core.red_y[0] > y1 + 1e-3)
            if bool((rbypass & core.red_alive[0]).any().item()):
                bottom_bypass_events += 1

            prev_blue_carry = bool(carry.any().item())
            if bool(done.any()):
                break

        ep_res = last_info.get("episode_result", last_info) if last_info else {}
        blue_score = int(ep_res.get("blue_score", core.blue_score[0].item()))
        red_score = int(ep_res.get("red_score", core.red_score[0].item()))
        mean_prot = protector_dist_sum / max(protector_dist_n, 1)
        return {
            "blue_style": blue_style,
            "red_style": red_style,
            "map": DEFAULT_MAP,
            "map_version": "map_c_v2",
            "episode_index": episode_index,
            "episode_seed": episode_seed,
            "blue_pickup": int(blue_pickup),
            "blue_converted": int(blue_converted),
            "return_success": int(return_success),
            "blue_carry_steps": blue_carry_steps,
            "mean_carrier_protector_dist": mean_prot if protector_dist_n else float("nan"),
            "protector_near_steps": protector_dist_n,
            "interpose_at_gap_steps": interpose_at_gap_steps,
            "carrier_gap_steps": carrier_gap_steps,
            "tag_at_gap": tag_at_gap,
            "tag_elsewhere": tag_elsewhere,
            "bottom_bypass_events": bottom_bypass_events,
            "used_top_gap": int(carrier_gap_steps > 0),
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
        pickups = sum(int(r["blue_pickup"]) for r in rs)
        conv = sum(int(r["blue_converted"]) for r in rs)
        dists = [
            float(r["mean_carrier_protector_dist"])
            for r in rs
            if r["mean_carrier_protector_dist"] == r["mean_carrier_protector_dist"]
        ]
        styles[style] = {
            "n": n,
            "pickup_seeds": pickups,
            "convert_seeds": conv,
            "return_success_seeds": sum(int(r["return_success"]) for r in rs),
            "pickup_to_score_rate": (conv / pickups) if pickups else None,
            "mean_carrier_protector_dist": (sum(dists) / len(dists)) if dists else None,
            "mean_interpose_at_gap_steps": sum(int(r["interpose_at_gap_steps"]) for r in rs) / max(n, 1),
            "tag_at_gap_sum": sum(int(r["tag_at_gap"]) for r in rs),
            "tag_elsewhere_sum": sum(int(r["tag_elsewhere"]) for r in rs),
            "top_gap_use_seeds": sum(int(r["used_top_gap"]) for r in rs),
            "bottom_bypass_events": sum(int(r["bottom_bypass_events"]) for r in rs),
            "mean_win_margin": sum(float(r["win_margin"]) for r in rs) / max(n, 1),
        }
    n = max((styles[s]["n"] for s in BLUE_STYLES if s in styles), default=8)
    rush = styles.get("BLUE_RUSH", {})
    split = styles.get("BLUE_SPLIT", {})
    escort = styles.get("BLUE_ESCORT", {})
    escort_ret = int(escort.get("return_success_seeds", 0))
    rush_ret = int(rush.get("return_success_seeds", 0))
    split_ret = int(split.get("return_success_seeds", 0))
    unsup_ret = max(rush_ret, split_ret)
    escort_dist = escort.get("mean_carrier_protector_dist")
    rush_dist = rush.get("mean_carrier_protector_dist")
    gates = {
        # Supported returns clearly exceed the best unsupported style.
        "escort_return_gt_unsupported": escort_ret >= max(5, (5 * n) // 8)
        and escort_ret >= unsup_ret + max(2, n // 4),
        # Persistent proximity: ESCORT stays closer than RUSH's brief screen.
        "escort_closer_than_rush": (
            escort_dist is not None
            and rush_dist is not None
            and float(escort_dist) + 0.5 < float(rush_dist)
        ),
        "escort_interpose_gap_active": float(escort.get("mean_interpose_at_gap_steps", 0.0))
        >= 1.0,
        "zero_bottom_bypasses": sum(int(r["bottom_bypass_events"]) for r in rows) == 0,
        # Soft: unsupported styles still attempt the gap (geometry in play).
        "unsupported_uses_top_gap": int(rush.get("top_gap_use_seeds", 0))
        + int(split.get("top_gap_use_seeds", 0))
        >= max(4, n // 2),
    }
    return {
        "map": DEFAULT_MAP,
        "map_version": "map_c_v2_frozen",
        "styles": styles,
        "gates": gates,
        "gates_pass": all(gates.values()),
        "contrast": {
            "escort_return_success": escort_ret,
            "rush_return_success": rush_ret,
            "split_return_success": split_ret,
        },
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=8)
    p.add_argument("--base-seed", type=int, default=811001)
    p.add_argument("--red", default=DEFAULT_RED)
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-decision-steps", type=int, default=240)
    p.add_argument("--out-dir", type=Path, default=Path("artifacts/mapc_v2_escort_microgates_op11_8seed"))
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
                f"[{k}/{total}] {style} ep={ep} pickup={row['blue_pickup']} "
                f"conv={row['blue_converted']} interpose={row['interpose_at_gap_steps']} "
                f"prot_d={row['mean_carrier_protector_dist']} bypass={row['bottom_bypass_events']}",
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
    json_path = args.out_dir / "escort_microgate_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"gates": summary["gates"], "contrast": summary["contrast"]}, indent=2))
    print(f"gates_pass={summary['gates_pass']}")
    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")
    return 0 if summary["gates_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
