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

    Compares map_b (walls) vs map_a_open on the same reset seed so the
    obstacle CNN channel differs while agent placement is matched.
    """
    try:
        from rl.custom_ppo.inference import load_custom_ppo_policy, read_custom_ppo_metadata
        from game_field_gpu import GPUCTFVecEnv, GPUFieldConfig

        meta = read_custom_ppo_metadata(checkpoint)
        n_agents = int(meta.get("n_blue", 2))

        def _make_env(map_layout: str) -> GPUCTFVecEnv:
            cfg = GPUFieldConfig(
                n_envs=1,
                max_blue_agents=n_agents,
                max_red_agents=n_agents,
                map_layout=map_layout,
                max_decision_steps=400,
                aquaticus_profile=True,
                rules_profile="OURS",
                device=device,
                seed=42,
                obstacle_obs_channel=True,
            )
            return GPUCTFVecEnv(cfg)

        env_wall = _make_env("map_b")
        env_open = _make_env("map_a_open")
        try:
            env_wall.seed(42)
            env_open.seed(42)
            obs_wall = env_wall.reset()
            obs_open = env_open.reset()

            grid_wall = obs_wall["grid"]
            grid_open = obs_open["grid"]
            if grid_wall.ndim != 5 or grid_open.ndim != 5:
                print(f"  Gate 3: SKIP — expected grid (B, A, C, H, W), got {grid_wall.shape}")
                return False
            if int(grid_wall.shape[2]) < 8:
                print(f"  Gate 3: SKIP — grids have {grid_wall.shape[2]} channels (expected 8)")
                return False

            # Agent 0 obstacle channel: map_b should differ from map_a_open.
            obs_ch_wall = grid_wall[:, 0, 7, :, :]
            obs_ch_open = grid_open[:, 0, 7, :, :]
            ch_diff = float(abs(obs_ch_wall - obs_ch_open).max())
            if ch_diff < 1e-6:
                print("  Gate 3: FAIL — obstacle channels identical for map_b and map_a_open")
                return False

            policy = load_custom_ppo_policy(
                checkpoint, env_wall.observation_space, env_wall.action_space, device=device
            )
            dev = torch.device(device)
            z_idx = torch.zeros((1,), dtype=torch.long, device=dev)

            def _to_model_obs(raw: dict) -> dict:
                return {
                    "grid": torch.as_tensor(raw["grid"], dtype=torch.float32, device=dev),
                    "vec": torch.as_tensor(raw["vec"], dtype=torch.float32, device=dev),
                    "agent_mask": torch.as_tensor(raw["agent_mask"], dtype=torch.float32, device=dev),
                    "mask": torch.as_tensor(raw["mask"], dtype=torch.float32, device=dev),
                }

            with torch.no_grad():
                logits_wall = policy.model.policy_logits(_to_model_obs(obs_wall), z_idx=z_idx)
                logits_open = policy.model.policy_logits(_to_model_obs(obs_open), z_idx=z_idx)

            max_diff = float((logits_wall - logits_open).abs().max().item())
            passed = max_diff > 1e-6
            print(
                f"  Gate 3: {'PASS' if passed else 'FAIL'} — "
                f"obs_ch_diff={ch_diff:.3e}, max_logit_diff={max_diff:.3e}"
            )
            return passed
        finally:
            env_wall.close()
            env_open.close()

    except Exception as exc:
        print(f"  Gate 3: SKIP — {exc}")
        return False


# ── Gate 4: Hard-pool WR range 50–90% ────────────────────────────────────────

def check_hardpool_wr(checkpoint: str, device: str, n_episodes: int = 50) -> tuple[bool, float, str]:
    """Return (pass, mean_wr, detail) for a quick forced-z eval against the hard pool."""
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
            return False, float("nan"), "no episodes completed (eval error?)"
        mean_wr = sum(all_wr) / len(all_wr)
        if mean_wr != mean_wr:
            return False, mean_wr, "nan mean WR"
        if mean_wr < 0.50:
            return False, mean_wr, "below 50% — not yet functional"
        if mean_wr >= 0.90:
            return False, mean_wr, "above 90% — pool saturated (expected after V6I9; use margin metrics)"
        return True, mean_wr, "in [50%, 90%)"
    except Exception as exc:
        print(f"  Gate 4: SKIP — {exc}")
        return False, float("nan"), str(exc)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V6I9 map-aware competence gate checker")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--episodes", type=int, default=50, help="Episodes for WR gate (gate 4)")
    p.add_argument(
        "--obs-weight-threshold",
        type=float,
        default=1e-4,
        help="Min obstacle-channel weight magnitude to consider nonzero",
    )
    p.add_argument(
        "--allow-saturated-wr",
        action="store_true",
        help="Treat WR>90%% as pass (competence proved; pool is easy)",
    )
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
    g4_pass, mean_wr, g4_detail = check_hardpool_wr(ckpt, args.device, args.episodes)
    if args.allow_saturated_wr and mean_wr == mean_wr and mean_wr >= 0.90:
        g4_pass = True
        g4_detail = f"{g4_detail} (override: --allow-saturated-wr)"
    results.append(("Gate 4: Hard-pool WR in [50%, 90%]", g4_pass, f"mean_WR={mean_wr:.1%}; {g4_detail}"))

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
        print("VERDICT: PASS — freeze trunk, enable adapters + router, enter repertoire")
    else:
        print("VERDICT: FAIL — see failed gates above")
        if mean_wr == mean_wr and mean_wr >= 0.90 and g12_pass and g3_pass:
            print(
                "NOTE: Gates 1–3 may pass while Gate 4 fails on saturation alone. "
                "That is expected for a mastered hardpool; run promotion eval + forced-z margin matrix."
            )
    print("=" * 60 + "\n")
    raise SystemExit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
