#!/usr/bin/env python3
"""Classify OP6 failed-return modes vs RUSH/ESCORT on map_a.

Read-only diagnostic. Does not change BT. Legal state only.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
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

BLUE_STYLES = ("BLUE_RUSH", "BLUE_ESCORT", "BLUE_SPLIT", "BLUE_TURTLE")
RED_STYLE = "OP6_IMMEDIATE_DUAL_RUSH"
DEFAULT_MAP = "map_a"


def _classify_loss(
    *,
    carr_x: float,
    carr_y: float,
    home_x: float,
    home_y: float,
    tagger_x: float,
    tagger_y: float,
    partner_x: float,
    partner_y: float,
    partner_alive: bool,
    role_changed: bool,
    heading_dx: float,
    heading_dy: float,
) -> str:
    to_home_x = home_x - carr_x
    to_home_y = home_y - carr_y
    home_norm = (to_home_x ** 2 + to_home_y ** 2) ** 0.5 + 1e-6
    hx, hy = to_home_x / home_norm, to_home_y / home_norm

    from_tag_x = carr_x - tagger_x
    from_tag_y = carr_y - tagger_y
    tag_norm = (from_tag_x ** 2 + from_tag_y ** 2) ** 0.5 + 1e-6
    # Approach direction of tagger toward carrier.
    app_x, app_y = -from_tag_x / tag_norm, -from_tag_y / tag_norm
    # Dot with home heading: +1 head-on (tagger in front toward home), -1 from behind.
    align = app_x * hx + app_y * hy

    # Carrier path detour: recent heading opposed to home.
    hdg_norm = (heading_dx ** 2 + heading_dy ** 2) ** 0.5 + 1e-6
    hdg_align = (heading_dx / hdg_norm) * hx + (heading_dy / hdg_norm) * hy

    partner_dist = ((partner_x - carr_x) ** 2 + (partner_y - carr_y) ** 2) ** 0.5
    if role_changed:
        return "role_reassignment_during_extraction"
    if partner_alive and partner_dist < 1.5:
        return "red_agents_crossing_blocking"
    if hdg_align < -0.2:
        return "carrier_route_detour"
    if align > 0.35:
        return "head_on_interception"
    if align < -0.35:
        return "tagged_from_behind"
    if partner_alive and partner_dist > 8.0:
        return "support_chasing_wrong_target"
    return "other_or_ambiguous"


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

        home_x = float(core.red_flag_home[0, 0].item())
        home_y = float(core.red_flag_home[0, 1].item())
        prev_carry = torch.zeros_like(core.red_carrying[0])
        prev_roles = core.bt_red_role[0].clone()
        prev_rx = core.red_x[0].clone()
        prev_ry = core.red_y[0].clone()
        loss_classes: list[str] = []
        pickups = 0
        scores = 0
        failed = 0
        steps = 0
        last_info: dict[str, Any] = {}
        prev_red_score = 0

        for _ in range(int(max_decision_steps) + 5):
            carry = core.red_carrying[0].clone()
            newly = carry & (~prev_carry)
            if bool(newly.any().item()):
                pickups += 1

            lost = prev_carry & (~carry)
            if bool(lost.any().item()):
                ep_res_peek = {}
                red_score_now = int(core.red_score[0].item())
                scored = red_score_now > prev_red_score
                if scored:
                    scores += 1
                else:
                    failed += 1
                    # Classify using pre-step geometry (prev positions).
                    ci = int(torch.argmax(prev_carry.to(torch.int64)).item())
                    # Nearest alive blue to previous carrier pos.
                    bx = core.blue_x[0]
                    by = core.blue_y[0]
                    balive = core.blue_alive[0]
                    dx = bx - prev_rx[ci]
                    dy = by - prev_ry[ci]
                    dist = torch.sqrt(dx * dx + dy * dy + 1e-8)
                    dist = torch.where(balive, dist, dist.new_full((), 1e9))
                    ti = int(torch.argmin(dist).item())
                    partner = 1 - ci
                    role_changed = bool(
                        (core.bt_red_role[0] != prev_roles).any().item()
                    )
                    heading_dx = float((prev_rx[ci] - getattr(core, "_trace_rx2", prev_rx)[ci]).item()) if hasattr(core, "_trace_rx2") else float((core.red_x[0, ci] - prev_rx[ci]).item())
                    # Use last two positions if available.
                    if hasattr(core, "_trace_rx2"):
                        heading_dx = float((prev_rx[ci] - core._trace_rx2[ci]).item())
                        heading_dy = float((prev_ry[ci] - core._trace_ry2[ci]).item())
                    else:
                        heading_dx = float((core.red_x[0, ci] - prev_rx[ci]).item())
                        heading_dy = float((core.red_y[0, ci] - prev_ry[ci]).item())
                    cls = _classify_loss(
                        carr_x=float(prev_rx[ci].item()),
                        carr_y=float(prev_ry[ci].item()),
                        home_x=home_x,
                        home_y=home_y,
                        tagger_x=float(bx[ti].item()),
                        tagger_y=float(by[ti].item()),
                        partner_x=float(prev_rx[partner].item()),
                        partner_y=float(prev_ry[partner].item()),
                        partner_alive=bool(core.red_alive[0, partner].item()),
                        role_changed=role_changed,
                        heading_dx=heading_dx,
                        heading_dy=heading_dy,
                    )
                    loss_classes.append(cls)

            action = _zero_action(env)
            env.step_async(action)
            _, _, done, infos = env.step_wait()
            last_info = infos[0] if infos else {}
            prev_red_score = int(core.red_score[0].item())
            if hasattr(core, "_trace_rx2"):
                core._trace_rx2 = prev_rx.clone()
                core._trace_ry2 = prev_ry.clone()
            else:
                core._trace_rx2 = prev_rx.clone()
                core._trace_ry2 = prev_ry.clone()
            prev_carry = carry
            prev_roles = core.bt_red_role[0].clone()
            prev_rx = core.red_x[0].clone()
            prev_ry = core.red_y[0].clone()
            steps += 1
            if bool(done.any()):
                break

        ep_res = last_info.get("episode_result", last_info) if last_info else {}
        counts = Counter(loss_classes)
        return {
            "blue_style": blue_style,
            "episode_index": episode_index,
            "episode_seed": episode_seed,
            "pickups": pickups,
            "scores": scores,
            "failed_returns": failed,
            "blue_score": int(ep_res.get("blue_score", core.blue_score[0].item())),
            "red_score": int(ep_res.get("red_score", core.red_score[0].item())),
            "loss_classes": dict(counts),
            "dominant_loss": counts.most_common(1)[0][0] if counts else "none",
            "steps": steps,
        }
    finally:
        env.close()


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
        default=PROJECT_ROOT / "artifacts" / "op6_failed_return_trace_dev28_map_a",
    )
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for ep in range(int(args.episodes)):
        seed = _episode_seed(int(args.base_seed), 0, 0, ep)
        for style in BLUE_STYLES:
            row = _run_episode(
                blue_style=style,
                episode_index=ep,
                episode_seed=seed,
                map_name=str(args.map),
                max_decision_steps=int(args.max_decision_steps),
                device=str(args.device),
            )
            rows.append(row)
            print(
                f"{style} ep={ep} fail={row['failed_returns']} "
                f"score={row['scores']} dom={row['dominant_loss']} "
                f"classes={row['loss_classes']}",
                flush=True,
            )

    by_style: dict[str, Counter] = defaultdict(Counter)
    fail_tot: dict[str, int] = defaultdict(int)
    for r in rows:
        by_style[r["blue_style"]].update(r["loss_classes"])
        fail_tot[r["blue_style"]] += int(r["failed_returns"])

    summary = {
        "failed_returns_by_style": dict(fail_tot),
        "loss_class_counts": {k: dict(v) for k, v in by_style.items()},
        "dominant_by_style": {
            k: (v.most_common(1)[0] if v else ("none", 0)) for k, v in by_style.items()
        },
    }
    (args.out_dir / "trace_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    # Flatten rows for CSV (loss_classes as json string).
    flat = []
    for r in rows:
        flat.append({**r, "loss_classes": json.dumps(r["loss_classes"])})
    with (args.out_dir / "episode_results.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(flat[0].keys()))
        w.writeheader()
        w.writerows(flat)
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
