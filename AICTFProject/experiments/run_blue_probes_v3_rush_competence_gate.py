#!/usr/bin/env python3
"""RUSH competence gate for BLUE_PROBES_V3.

Requires RUSH to finish its own attack against a soft host before any
niche-host claim. Paired seeds: same episode seed for RUSH vs SPLIT/ESCORT.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.run_scripted_style_payoff_matrix import (  # noqa: E402
    BLUE_PROBE_PROTOCOL,
    _episode_seed,
    _make_env,
    _zero_action,
)

SOFT_HOST = "OP5_RUSHER"
MAP_NAME = "map_b_split_lane"
TARGET_CHANGE_EPS = 0.5
SAME_TARGET_EPS = 1.0


def _run_episode(
    *,
    blue_style: str,
    red_style: str,
    episode_seed: int,
    max_decision_steps: int,
    device: str,
) -> dict[str, Any]:
    env = _make_env(
        map_name=MAP_NAME,
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
        env.env_method("set_phase", red_style)
        env.env_method("set_next_opponent", "SCRIPTED", red_style)
        core.blue_scripted = True
        core.set_blue_style(blue_style)

        first_pickup = -1
        first_score = -1
        cum_dist = [0.0, 0.0]
        path_len = 0.0
        target_changes = 0
        same_target_pre = 0
        lat_sep_sum = 0.0
        pre_steps = 0
        post_noncarrier_dist_sum = 0.0
        post_steps = 0
        prev_tx = prev_ty = None
        prev_bx = prev_by = None
        carrier_idx = None
        last_info: dict[str, Any] = {}

        for _ in range(int(max_decision_steps) + 5):
            bx = core.blue_x[0].detach()
            by = core.blue_y[0].detach()
            carrying = core.blue_carrying[0].detach()
            tx = getattr(core, "_debug_blue_target_x", None)
            ty = getattr(core, "_debug_blue_target_y", None)
            if tx is not None:
                tx = tx[0].detach()
                ty = ty[0].detach()

            if first_pickup < 0:
                if prev_bx is not None:
                    for a in range(2):
                        cum_dist[a] += float(
                            ((bx[a] - prev_bx[a]) ** 2 + (by[a] - prev_by[a]) ** 2).sqrt().item()
                        )
                if tx is not None and prev_tx is not None:
                    if (tx - prev_tx).abs().max() > TARGET_CHANGE_EPS or (
                        ty - prev_ty
                    ).abs().max() > TARGET_CHANGE_EPS:
                        target_changes += 1
                    if float(((tx[0] - tx[1]) ** 2 + (ty[0] - ty[1]) ** 2).sqrt().item()) < SAME_TARGET_EPS:
                        same_target_pre += 1
                if ty is not None:
                    lat_sep_sum += float(abs(ty[0] - ty[1]).item())
                pre_steps += 1
                if bool(carrying.any().item()):
                    first_pickup = int(core.sim_step_count[0].item())
                    carrier_idx = int(torch.argmax(carrying.to(torch.int64)).item())
                    path_len = cum_dist[carrier_idx]
            else:
                if carrier_idx is not None and tx is not None:
                    other = 1 - carrier_idx
                    d = float(
                        (
                            (tx[other] - bx[carrier_idx]) ** 2
                            + (ty[other] - by[carrier_idx]) ** 2
                        ).sqrt().item()
                    )
                    post_noncarrier_dist_sum += d
                    post_steps += 1

            action = _zero_action(env)
            env.step_async(action)
            _, _, done, infos = env.step_wait()
            last_info = infos[0] if infos else {}
            ep_res = last_info.get("episode_result", last_info)
            if first_score < 0 and int(ep_res.get("blue_score", 0)) > 0:
                first_score = int(core.sim_step_count[0].item())
            prev_bx, prev_by = bx.clone(), by.clone()
            if tx is not None:
                prev_tx, prev_ty = tx.clone(), ty.clone()
            if bool(done.any()):
                break

        ep_res = last_info.get("episode_result", last_info) if last_info else {}
        blue_score = int(ep_res.get("blue_score", 0))
        red_score = int(ep_res.get("red_score", 0))
        return {
            "blue_style": blue_style,
            "red_style": red_style,
            "episode_seed": episode_seed,
            "first_pickup": first_pickup,
            "first_score": first_score,
            "path_len_pre_pickup": path_len,
            "target_changes_pre_pickup": target_changes,
            "same_target_frac_pre": same_target_pre / max(pre_steps, 1),
            "mean_lat_target_sep_pre": lat_sep_sum / max(pre_steps, 1),
            "post_noncarrier_mean_target_dist": (
                post_noncarrier_dist_sum / post_steps if post_steps else None
            ),
            "blue_score": blue_score,
            "red_score": red_score,
            "win_margin": blue_score - red_score,
            "scored": int(blue_score > 0),
        }
    finally:
        env.close()


def _judge(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by = {}
    for r in rows:
        by.setdefault(r["blue_style"], []).append(r)
    rush = by["BLUE_RUSH"]
    split = by["BLUE_SPLIT"]
    escort = by["BLUE_ESCORT"]
    n = len(rush)

    rush_earlier = sum(
        1
        for a, b in zip(rush, split)
        if int(a["first_pickup"]) >= 0
        and (int(b["first_pickup"]) < 0 or int(a["first_pickup"]) < int(b["first_pickup"]))
    )
    rush_scores = sum(int(r["scored"]) for r in rush)
    mean_rush_path = float(np.mean([r["path_len_pre_pickup"] for r in rush]))
    mean_split_path = float(np.mean([r["path_len_pre_pickup"] for r in split]))
    mean_churn = float(np.mean([r["target_changes_pre_pickup"] for r in rush]))
    mean_rush_lat = float(np.mean([r["mean_lat_target_sep_pre"] for r in rush]))
    mean_split_lat = float(np.mean([r["mean_lat_target_sep_pre"] for r in split]))
    rush_post = [
        r["post_noncarrier_mean_target_dist"]
        for r in rush
        if r["post_noncarrier_mean_target_dist"] is not None
    ]
    escort_post = [
        r["post_noncarrier_mean_target_dist"]
        for r in escort
        if r["post_noncarrier_mean_target_dist"] is not None
    ]
    mean_rush_post = float(np.mean(rush_post)) if rush_post else None
    mean_escort_post = float(np.mean(escort_post)) if escort_post else None

    gates = {
        "pickup_earlier_than_split_most_seeds": rush_earlier >= max(1, (n + 1) // 2),
        "near_shortest_vs_split": mean_rush_path <= mean_split_path * 1.05,
        "stable_roles_low_churn": mean_churn <= 8.0,
        "return_success_ge_6_of_8": rush_scores >= min(6, n) if n >= 6 else rush_scores >= max(1, (3 * n) // 4),
        "distinct_from_escort_screener": (
            mean_rush_post is not None
            and mean_escort_post is not None
            and mean_rush_post > mean_escort_post + 0.5
        ),
        # Concentrated corridor: lateral target sep must stay far below SPLIT lanes.
        "concentrated_not_split_lanes": mean_rush_lat <= max(3.0, 0.35 * mean_split_lat),
    }
    return {
        "protocol": BLUE_PROBE_PROTOCOL,
        "soft_host": SOFT_HOST,
        "n_seeds": n,
        "rush_earlier_than_split_count": rush_earlier,
        "rush_score_count": rush_scores,
        "mean_rush_path_len": mean_rush_path,
        "mean_split_path_len": mean_split_path,
        "mean_rush_target_changes": mean_churn,
        "mean_rush_lat_target_sep": mean_rush_lat,
        "mean_split_lat_target_sep": mean_split_lat,
        "mean_rush_post_noncarrier_target_dist": mean_rush_post,
        "mean_escort_post_noncarrier_target_dist": mean_escort_post,
        "mean_margins": {
            s: float(np.mean([r["win_margin"] for r in by[s]])) for s in by
        },
        "gates": gates,
        "pass": all(gates.values()),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=8)
    p.add_argument("--base-seed", type=int, default=571001)
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-decision-steps", type=int, default=240)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "blue_probes_v3_rush_competence_gate",
    )
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    styles = ("BLUE_RUSH", "BLUE_SPLIT", "BLUE_ESCORT")
    rows: list[dict[str, Any]] = []
    for ep in range(int(args.episodes)):
        seed = _episode_seed(int(args.base_seed), red_index=0, map_index=0, episode_index=ep)
        for style in styles:
            row = _run_episode(
                blue_style=style,
                red_style=SOFT_HOST,
                episode_seed=seed,
                max_decision_steps=int(args.max_decision_steps),
                device=str(args.device),
            )
            row["episode_index"] = ep
            rows.append(row)
            print(
                f"[{len(rows)}/{int(args.episodes) * len(styles)}] {style} ep={ep} "
                f"pickup={row['first_pickup']} score_at={row['first_score']} "
                f"margin={row['win_margin']}",
                flush=True,
            )

    summary = _judge(rows)
    (args.out_dir / "competence_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    with (args.out_dir / "episode_results.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(json.dumps(summary["gates"], indent=2))
    print(f"competence_pass={summary['pass']}")
    return 0 if summary["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
