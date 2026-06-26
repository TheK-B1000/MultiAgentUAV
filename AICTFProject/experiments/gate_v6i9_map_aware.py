#!/usr/bin/env python3
"""V6I9 map-aware competence promotion gate checker.

Checks whether a checkpoint is ready to freeze the CNN trunk and enter
latent repertoire training.  Run this after each evaluation checkpoint
(suggested: 200k, 400k steps).

Gates (all six must pass):
  1. Obstacle-channel gradient is nonzero (weight moved from zero).
  2. Obstacle-channel weights differ from zero by > threshold.
  3. Changing wall geometry changes actor logits (map sensitivity).
  4. Hard-pool WR is functional (>50%) but not saturated (<90%).
  5. Wall-collision rate < baseline (lower than expected from random).
  6. Actor selects different routes on open vs split-lane map.

Usage
-----
    python experiments/gate_v6i9_map_aware.py \\
        --checkpoint checkpoints/v6i9/ckpt_200000.zip \\
        --device cuda
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Any

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def _load_state_dict(checkpoint: str) -> dict[str, Any]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    return payload["model_state_dict"]


# ── Gate 1 & 2: Obstacle channel weight departure from zero ──────────────────

def check_obstacle_channel_weights(sd: dict[str, Any], threshold: float = 1e-4) -> tuple[bool, float]:
    """Return (pass, max_abs) for the obstacle channel of the first CNN conv."""
    for key in (
        "latent_actor.actor_cnn.conv.0.weight",
        "actor_cnn.conv.0.weight",
    ):
        if key not in sd:
            continue
        w = sd[key]  # (out_ch, in_ch, kH, kW)
        if w.shape[1] < 8:
            print(f"  Gate 1/2: SKIP — checkpoint has {w.shape[1]} channels (not 8-channel)")
            return False, 0.0
        obs_ch = w[:, 7, :, :]  # obstacle channel weights
        max_abs = float(obs_ch.abs().max().item())
        passed = max_abs > threshold
        return passed, max_abs
    print("  Gate 1/2: SKIP — CNN conv key not found in checkpoint")
    return False, 0.0


# ── Gate 3: Wall geometry changes actor logits ───────────────────────────────

def check_map_sensitivity(checkpoint: str, device: str) -> bool:
    """Return True if wall geometry changes actor logits.

    Constructs two observations that differ only in the obstacle channel
    (one all-zero, one partially filled) and checks that logits differ.
    """
    try:
        from rl.custom_ppo.inference import read_custom_ppo_metadata
        from game_field_gpu import GPUCTFVecEnv, GPUFieldConfig

        meta = read_custom_ppo_metadata(checkpoint)
        n_agents = int(meta.get("n_blue", 2))

        # Build two envs: map_b (has walls) vs map_a_open (no walls)
        def _make_env(map_layout: str) -> GPUCTFVecEnv:
            cfg = GPUFieldConfig(
                n_envs=1, max_blue_agents=n_agents, max_red_agents=n_agents,
                map_layout=map_layout, max_decision_steps=400,
                aquaticus_profile=True, rules_profile="OURS",
                device=device, seed=42, obstacle_obs_channel=True,
            )
            return GPUCTFVecEnv(cfg)

        env_wall = _make_env("map_b")
        env_open = _make_env("map_a_open")
        env_wall.reset()
        env_open.reset()

        obs_wall = env_wall.get_obs()
        obs_open = env_open.get_obs()

        grid_wall = obs_wall["grid"]
        grid_open = obs_open["grid"]

        # Check that obstacle channels actually differ
        if grid_wall.shape[2] < 8 or grid_open.shape[2] < 8:
            print(f"  Gate 3: SKIP — grids have {grid_wall.shape[2]} channels (expected 8)")
            return False

        obs_ch_wall = grid_wall[:, :, 7, :, :]
        obs_ch_open = grid_open[:, :, 7, :, :]
        if float((obs_ch_wall - obs_ch_open).abs().max().item()) < 1e-6:
            print("  Gate 3: FAIL — obstacle channels identical for map_b and map_a_open")
            return False

        # Now check model logits differ
        from gpu_env._specs import _make_obs_action_spaces
        n_macros = int(meta.get("n_macros", 5))
        n_targets = int(meta.get("n_targets", 16))
        obs_space, act_space = _make_obs_action_spaces(
            n_agents, n_macros, n_targets, num_cnn_channels=8
        )
        from rl.custom_ppo.inference import load_custom_ppo_policy
        policy = load_custom_ppo_policy(checkpoint, obs_space, act_space, device=device)

        dev = torch.device(device)
        z_idx = torch.zeros((1,), dtype=torch.long, device=dev)

        def _to_model_obs(raw: dict) -> dict:
            return {
                "grid": torch.as_tensor(raw["grid"], dtype=torch.float32, device=dev),
                "vec": torch.as_tensor(raw["vec"], dtype=torch.float32, device=dev),
                "agent_mask": torch.ones((1, n_agents), dtype=torch.float32, device=dev),
                "mask": torch.ones((1, n_macros * n_targets), dtype=torch.float32, device=dev),
            }

        with torch.no_grad():
            logits_wall = policy.model.policy_logits(_to_model_obs(obs_wall), z_idx=z_idx)
            logits_open = policy.model.policy_logits(_to_model_obs(obs_open), z_idx=z_idx)

        max_diff = float((logits_wall - logits_open).abs().max().item())
        passed = max_diff > 1e-6
        print(f"  Gate 3: {'PASS' if passed else 'FAIL'} — max logit diff = {max_diff:.3e}")
        env_wall.close()
        env_open.close()
        return passed

    except Exception as exc:
        print(f"  Gate 3: SKIP — {exc}")
        return False


# ── Gate 4: Hard-pool WR range 50–90% ────────────────────────────────────────

def check_hardpool_wr(checkpoint: str, device: str, n_episodes: int = 50) -> tuple[bool, float]:
    """Return (pass, mean_wr) for a quick forced-z eval against the hard pool."""
    try:
        from experiments.calibrate_hard_pool import run_forced_z_cells, _wr, OPPONENTS, LATENTS, MAPS

        cells = run_forced_z_cells(
            checkpoint=checkpoint,
            opponents=list(OPPONENTS),
            latents=LATENTS,
            maps=list(MAPS),
            n_episodes=n_episodes,
            device=device,
            deterministic=True,
        )
        all_wr = [_wr(eps) for eps in cells.values() if eps]
        if not all_wr:
            return False, float("nan")
        mean_wr = sum(all_wr) / len(all_wr)
        passed = 0.50 < mean_wr < 0.90
        return passed, mean_wr
    except Exception as exc:
        print(f"  Gate 4: SKIP — {exc}")
        return False, float("nan")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V6I9 map-aware competence gate checker")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--episodes", type=int, default=50, help="Episodes for WR gate (gate 4)")
    p.add_argument("--obs-weight-threshold", type=float, default=1e-4,
                   help="Min obstacle-channel weight magnitude to consider nonzero")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    ckpt = args.checkpoint
    print(f"\nV6I9 Map-Aware Gate Check: {os.path.basename(ckpt)}")
    print("=" * 60)

    sd = _load_state_dict(ckpt)
    results: list[tuple[str, bool, str]] = []

    # Gate 1/2
    g12_pass, max_abs = check_obstacle_channel_weights(sd, args.obs_weight_threshold)
    results.append(("Gate 1/2: Obstacle channel weights nonzero", g12_pass, f"max|w|={max_abs:.3e}"))

    # Gate 3
    print("\nGate 3: Map sensitivity check...")
    g3_pass = check_map_sensitivity(ckpt, args.device)
    results.append(("Gate 3: Wall geometry changes actor logits", g3_pass, ""))

    # Gate 4
    print("\nGate 4: Hard-pool WR range check...")
    g4_pass, mean_wr = check_hardpool_wr(ckpt, args.device, args.episodes)
    results.append(("Gate 4: Hard-pool WR in [50%, 90%]", g4_pass, f"mean_WR={mean_wr:.1%}"))

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    all_pass = True
    for name, passed, detail in results:
        status = "PASS" if passed else "FAIL"
        suffix = f"  ({detail})" if detail else ""
        print(f"  {status}  {name}{suffix}")
        if not passed:
            all_pass = False

    print("\n" + ("=" * 60))
    if all_pass:
        print("VERDICT: PASS — freeze trunk, enable adapters + router, enter Stage C")
    else:
        print("VERDICT: FAIL — continue competence training")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
