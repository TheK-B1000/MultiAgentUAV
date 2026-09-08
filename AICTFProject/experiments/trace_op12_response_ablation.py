#!/usr/bin/env python3
"""Trace paired OP12 confirmed-ESCORT response OFF/ON trajectories.

This is a surgical diagnostic for the dev22 failure: the detector fires, the
response activates, but payoff moves in the wrong direction. It runs identical
matched seeds twice and records the first trajectory/role divergence after
confirmation.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gpu_env import GPUCTFVecEnv, GPUFieldConfig  # noqa: E402


TRACE_CSV = "trace_rows.csv"
SUMMARY_JSON = "summary.json"
BLUE_PROBE_PROTOCOL = "BLUE_PROBES_V2"

TRACE_FIELDS = [
    "episode_index",
    "episode_seed",
    "response_enabled",
    "step",
    "blue_score",
    "red_score",
    "carrier_id",
    "protector_id",
    "confirmed_escort",
    "confirmation_step",
    "carrier_x",
    "carrier_y",
    "protector_x",
    "protector_y",
    "carrier_home_progress",
    "red0_role",
    "red1_role",
    "red0_x",
    "red0_y",
    "red1_x",
    "red1_y",
    "red0_carrier_dist",
    "red1_carrier_dist",
    "red0_protector_dist",
    "red1_protector_dist",
    "red_home_occupancy",
    "red_pair_dist",
    "carrier_intercept_attempts",
    "done",
]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--episode-indices", nargs="+", type=int, default=[0, 1, 5, 7])
    p.add_argument("--base-seed", type=int, default=556001)
    p.add_argument("--max-decision-steps", type=int, default=240)
    p.add_argument("--device", default="cuda")
    p.add_argument("--red", default="OP12_LATE_CONVERTER")
    p.add_argument("--map", default="map_b_split_lane")
    p.add_argument("--blue-style", default="BLUE_ESCORT")
    return p.parse_args()


def _episode_seed(base_seed: int, episode_index: int) -> int:
    return int(base_seed) + int(episode_index)


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
    sample = env.action_space.sample()
    return np.zeros_like(sample)


def _scalar(core: Any, attr: str, default: int = -1) -> int:
    val = getattr(core, attr, None)
    if val is None:
        return int(default)
    try:
        return int(val[0].item())
    except Exception:
        return int(default)


def _trace_row(*, core: Any, episode_index: int, episode_seed: int, response_enabled: bool, done: bool) -> dict[str, Any]:
    carrier_id = _scalar(core, "bt_adapt_escort_confirm_carrier_id", -1)
    protector_id = _scalar(core, "bt_adapt_escort_confirm_protector_id", -1)
    if carrier_id < 0:
        carrying = core.blue_carrying[0].detach().bool()
        carrier_id = int(carrying.nonzero(as_tuple=False)[0, 0].item()) if bool(carrying.any().item()) else -1
    if protector_id < 0 and carrier_id >= 0 and core.Nb >= 2:
        protector_id = 1 - int(carrier_id)

    def bpos(idx: int) -> tuple[float, float]:
        if idx < 0:
            return -1.0, -1.0
        return float(core.blue_x[0, idx].item()), float(core.blue_y[0, idx].item())

    cx, cy = bpos(carrier_id)
    px, py = bpos(protector_id)
    red0_x, red0_y = float(core.red_x[0, 0].item()), float(core.red_y[0, 0].item())
    red1_x, red1_y = float(core.red_x[0, 1].item()), float(core.red_y[0, 1].item())

    def dist(ax: float, ay: float, bx: float, by: float) -> float:
        if bx < 0 or by < 0:
            return -1.0
        return float(((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5)

    home_x = float(core.blue_flag_home[0, 0].item())
    red_half_min_x = float(core.cols) * 0.5
    red_home_occ = int(float(core.red_x[0, 0].item()) >= red_half_min_x) + int(float(core.red_x[0, 1].item()) >= red_half_min_x)
    carrier_home_progress = -1.0 if cx < 0 else float((float(core.cols - 1) - cx) / max(float(core.cols - 1) - home_x, 1.0))

    return {
        "episode_index": int(episode_index),
        "episode_seed": int(episode_seed),
        "response_enabled": int(bool(response_enabled)),
        "step": int(core.sim_step_count[0].item()),
        "blue_score": int(core.blue_score[0].item()),
        "red_score": int(core.red_score[0].item()),
        "carrier_id": int(carrier_id),
        "protector_id": int(protector_id),
        "confirmed_escort": int(_scalar(core, "bt_adapt_escort_confirm_first_step", -1) >= 0),
        "confirmation_step": _scalar(core, "bt_adapt_escort_confirm_first_step", -1),
        "carrier_x": cx,
        "carrier_y": cy,
        "protector_x": px,
        "protector_y": py,
        "carrier_home_progress": carrier_home_progress,
        "red0_role": int(core.bt_red_role[0, 0].item()),
        "red1_role": int(core.bt_red_role[0, 1].item()),
        "red0_x": red0_x,
        "red0_y": red0_y,
        "red1_x": red1_x,
        "red1_y": red1_y,
        "red0_carrier_dist": dist(red0_x, red0_y, cx, cy),
        "red1_carrier_dist": dist(red1_x, red1_y, cx, cy),
        "red0_protector_dist": dist(red0_x, red0_y, px, py),
        "red1_protector_dist": dist(red1_x, red1_y, px, py),
        "red_home_occupancy": red_home_occ,
        "red_pair_dist": dist(red0_x, red0_y, red1_x, red1_y),
        "carrier_intercept_attempts": _scalar(core, "bt_adapt_carrier_intercept_attempts", 0),
        "done": int(bool(done)),
    }


def _run_episode(
    *,
    episode_index: int,
    episode_seed: int,
    response_enabled: bool,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    env = _make_env(
        map_name=args.map,
        seed=episode_seed,
        max_decision_steps=int(args.max_decision_steps),
        device=str(args.device),
    )
    rows: list[dict[str, Any]] = []
    try:
        core = env.core
        core.op12_confirmed_escort_response_enabled = bool(response_enabled)
        env.env_method("set_phase", args.red)
        env.env_method("set_next_opponent", "SCRIPTED", args.red)
        core.blue_scripted = True
        core.set_blue_style(args.blue_style)
        env.reset()
        core.op12_confirmed_escort_response_enabled = bool(response_enabled)
        env.env_method("set_phase", args.red)
        env.env_method("set_next_opponent", "SCRIPTED", args.red)
        core.blue_scripted = True
        core.set_blue_style(args.blue_style)

        last_info: dict[str, Any] = {}
        for _ in range(int(args.max_decision_steps) + 5):
            env.step_async(_zero_action(env))
            _, _, done, infos = env.step_wait()
            last_info = infos[0] if infos else {}
            done_now = bool(done.any())
            rows.append(
                _trace_row(
                    core=core,
                    episode_index=episode_index,
                    episode_seed=episode_seed,
                    response_enabled=response_enabled,
                    done=done_now,
                )
            )
            if done_now:
                break
        ep_res = last_info.get("episode_result", last_info)
        first_confirm = next((int(r["step"]) for r in rows if int(r["confirmed_escort"]) == 1), -1)
        summary = {
            "episode_index": int(episode_index),
            "episode_seed": int(episode_seed),
            "response_enabled": int(bool(response_enabled)),
            "blue_score": int(ep_res.get("blue_score", 0)),
            "red_score": int(ep_res.get("red_score", 0)),
            "win_margin": int(ep_res.get("blue_score", 0)) - int(ep_res.get("red_score", 0)),
            "steps": int(ep_res.get("decision_steps", len(rows))),
            "confirmation_step": first_confirm,
        }
        return rows, summary
    finally:
        env.close()


def main() -> int:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for ep_i in args.episode_indices:
        seed = _episode_seed(int(args.base_seed), int(ep_i))
        for enabled in (False, True):
            rows, summary = _run_episode(
                episode_index=int(ep_i),
                episode_seed=seed,
                response_enabled=bool(enabled),
                args=args,
            )
            all_rows.extend(rows)
            summaries.append(summary)

    with (out_dir / TRACE_CSV).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=TRACE_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)

    paired: list[dict[str, Any]] = []
    for ep_i in args.episode_indices:
        off = next(s for s in summaries if s["episode_index"] == ep_i and s["response_enabled"] == 0)
        on = next(s for s in summaries if s["episode_index"] == ep_i and s["response_enabled"] == 1)
        paired.append(
            {
                "episode_index": int(ep_i),
                "episode_seed": _episode_seed(int(args.base_seed), int(ep_i)),
                "off_margin": off["win_margin"],
                "on_margin": on["win_margin"],
                "delta_margin": int(on["win_margin"]) - int(off["win_margin"]),
                "off_confirmation_step": off["confirmation_step"],
                "on_confirmation_step": on["confirmation_step"],
                "off_steps": off["steps"],
                "on_steps": on["steps"],
            }
        )

    payload = {
        "protocol": "op12_confirmed_escort_response_trace",
        "blue_probe_protocol": BLUE_PROBE_PROTOCOL,
        "blue_style": args.blue_style,
        "red": args.red,
        "map": args.map,
        "base_seed": int(args.base_seed),
        "episode_indices": [int(x) for x in args.episode_indices],
        "response": "OFF vs ON paired within identical episode seeds",
        "paired": paired,
        "summaries": summaries,
        "artifacts": [TRACE_CSV],
    }
    (out_dir / SUMMARY_JSON).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["paired"], indent=2))
    print(f"Artifacts in: {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
