#!/usr/bin/env python3
"""V6I26 map_a 8-channel observation-schema regression (no training).

Checks that continuing V6I23+ checkpoints on map_a_open keeps the same CNN
input contract as map_b (obstacle channel present; zeros on open arena) and
does not shape-skip observation-CNN weights.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.v6i26_lro_core import write_json  # noqa: E402
from gpu_env._constants import OBSTACLE_CHANNEL_INDEX  # noqa: E402
from gpu_env._maps import MAP_A_OPEN, MAP_B_SPLIT_LANE  # noqa: E402


DEFAULT_CKPT = (
    "artifacts/v6i23_population_birth_5u_seed1/"
    "final_v6i23_population_birth_5u_seed1_2v2.zip"
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="map_a 8-channel obs-schema regression")
    p.add_argument("--checkpoint", default=DEFAULT_CKPT)
    p.add_argument(
        "--output",
        default="artifacts/v6i26_map_a_obs_compat_regression_seed1.json",
    )
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def _reset_obs(env, *, seed: int, opponent: str = "OP8"):
    env.env_method("set_next_opponent", "SCRIPTED", opponent)
    try:
        env.env_method("set_phase", "PHASE1")
    except Exception:  # noqa: BLE001
        pass
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    return obs


def _obstacle_plane(obs) -> np.ndarray:
    grid = np.asarray(obs["grid"])
    # (n_agents, C, H, W) or (B, n_agents, C, H, W)
    if grid.ndim == 5:
        plane = grid[0, :, OBSTACLE_CHANNEL_INDEX]
    else:
        plane = grid[:, OBSTACLE_CHANNEL_INDEX]
    return np.asarray(plane, dtype=np.float64)


def main() -> int:
    args = _parse_args()
    ckpt = Path(args.checkpoint)
    if not ckpt.is_file():
        print(f"ERROR: missing checkpoint {ckpt}")
        return 2

    from experiments.run_v6i24_population_eval_gates import _make_env
    from rl.custom_ppo import load_custom_ppo_policy
    from rl.training.obs_schema import obstacle_obs_channel_for_checkpoint

    report: dict = {
        "experiment": "v6i26_map_a_obs_compat_regression",
        "checkpoint": str(ckpt),
        "inferred_obstacle_obs_channel": obstacle_obs_channel_for_checkpoint(ckpt),
        "checks": {},
        "pass": False,
    }

    maps = {
        "map_a_open": MAP_A_OPEN,
        "map_b_split_lane": MAP_B_SPLIT_LANE,
    }
    envs = {}
    load_logs = {}
    policies = {}
    try:
        for key, layout in maps.items():
            env = _make_env(ckpt, layout, int(args.seed), str(args.device), 64)
            envs[key] = env
            obs_ch = int(env.observation_space["grid"].shape[1])
            report["checks"][f"{key}_obs_channels"] = obs_ch
            report["checks"][f"{key}_obstacle_flag"] = bool(env.cfg.obstacle_obs_channel)

            buf = io.StringIO()
            with redirect_stdout(buf):
                pol = load_custom_ppo_policy(
                    str(ckpt),
                    env.observation_space,
                    env.action_space,
                    device=str(args.device),
                )
            log = buf.getvalue()
            load_logs[key] = log
            policies[key] = pol
            skipped = "Shape-mismatched keys skipped" in log or "shape-mismatched" in log.lower()
            report["checks"][f"{key}_cnn_shape_skipped"] = bool(skipped)
            report["checks"][f"{key}_load_log_tail"] = "\n".join(log.strip().splitlines()[-8:])

        # Actor input dims identical across maps.
        ch_a = report["checks"]["map_a_open_obs_channels"]
        ch_b = report["checks"]["map_b_split_lane_obs_channels"]
        report["checks"]["actor_input_dims_identical"] = ch_a == ch_b == 8

        # Obstacle plane content.
        obs_a = _reset_obs(envs["map_a_open"], seed=int(args.seed))
        obs_b = _reset_obs(envs["map_b_split_lane"], seed=int(args.seed))
        plane_a = _obstacle_plane(obs_a)
        plane_b = _obstacle_plane(obs_b)
        report["checks"]["map_a_obstacle_max"] = float(np.max(np.abs(plane_a)))
        report["checks"]["map_a_obstacle_sum"] = float(np.sum(plane_a))
        report["checks"]["map_a_obstacle_exactly_zero"] = bool(np.allclose(plane_a, 0.0))
        report["checks"]["map_b_obstacle_max"] = float(np.max(plane_b))
        report["checks"]["map_b_obstacle_sum"] = float(np.sum(plane_b))
        report["checks"]["map_b_obstacle_nonzero"] = bool(np.max(plane_b) > 0.0)

        # map_b behavior unchanged: both loaders see identical 8-ch spaces, so
        # observation-CNN stems must match exactly (no map-dependent remapping).
        def _cnn_weights(pol):
            model = getattr(pol, "model", None)
            if model is None:
                model = getattr(pol, "policy", pol)
            out = {}
            for name, tensor in model.state_dict().items():
                if "cnn" in name.lower() or "conv" in name.lower():
                    out[name] = tensor.detach().float().cpu()
            return out

        wa = _cnn_weights(policies["map_a_open"])
        wb = _cnn_weights(policies["map_b_split_lane"])
        shared = sorted(set(wa) & set(wb))
        max_diff = 0.0
        for name in shared:
            max_diff = max(max_diff, float((wa[name] - wb[name]).abs().max().item()))
        report["checks"]["map_b_cnn_weight_max_abs_diff_across_loaders"] = max_diff
        report["checks"]["map_b_cnn_shared_keys"] = len(shared)
        report["checks"]["map_b_behavior_unchanged"] = bool(shared) and max_diff < 1e-8
        report["checks"].pop("map_b_behavior_check_error", None)

        gates = {
            "eight_channels_both_maps": report["checks"]["actor_input_dims_identical"],
            "no_cnn_shape_skip_map_a": not report["checks"]["map_a_open_cnn_shape_skipped"],
            "no_cnn_shape_skip_map_b": not report["checks"]["map_b_split_lane_cnn_shape_skipped"],
            "map_a_obstacle_exactly_zero": report["checks"]["map_a_obstacle_exactly_zero"],
            "map_b_obstacle_nonzero": report["checks"]["map_b_obstacle_nonzero"],
            "obstacle_flag_true_both": (
                report["checks"]["map_a_open_obstacle_flag"]
                and report["checks"]["map_b_split_lane_obstacle_flag"]
            ),
        }
        if report["checks"].get("map_b_behavior_unchanged") is not None:
            gates["map_b_behavior_unchanged"] = bool(report["checks"]["map_b_behavior_unchanged"])
        report["gates"] = gates
        report["pass"] = all(bool(v) for v in gates.values())
        report["verdict"] = "PASS" if report["pass"] else "FAIL"
    finally:
        for env in envs.values():
            try:
                env.close()
            except Exception:  # noqa: BLE001
                pass

    out = Path(args.output)
    write_json(out, report)
    print(json.dumps({k: report[k] for k in ("verdict", "pass", "gates", "checks") if k in report}, indent=2))
    print(f"wrote {out}")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
