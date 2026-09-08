#!/usr/bin/env python3
"""RUSH_PROBE_ROOT_CAUSE_AUDIT -- Phase 2, diagnostic variant D2.

Baseline audit (diagnose_rush_probe_root_cause.py) showed RUSH decisively
PASSES every pre-pickup competence gate (faster to pickup, shorter path,
zero target churn, no self-duplication) across all 5 tested opponents, but
almost never converts after pickup (0/4 vs SPLIT's 2/4-4/4), consistently
across opponents -- pointing at the CARRIER'S POST-PICKUP ROUTING
specifically, not the approach.

D2 tests this directly: current RUSH's pre-pickup route, unchanged, but
the CARRIER's post-pickup return uses SPLIT's lane-clearance-aware, locked
routing (_blue_split_post_pickup_targets) instead of the generic
_carrier_evasion_target fallback every non-SPLIT style uses. The
non-carrier's "pressure lane" target is left untouched, isolating the one
change (carrier-return algorithm only) per the diagnostic contract.

Diagnostic-only: does NOT modify _scripted_blue_styles.py or
BLUE_PROBES_V2 at all. The D2 controller is monkey-patched onto a single
core instance for this script's own rollouts only.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import types
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gpu_env import GPUCTFVecEnv, GPUFieldConfig  # noqa: E402

OPPONENTS = (
    "OP6_IMMEDIATE_DUAL_RUSH",
    "OP7_DEEP_FORTRESS",
    "OP8_PROTECTED_CARRIER_ESCORT",
    "OP10_AGGRESSIVE_INTERCEPTOR",
    "OP12_LATE_CONVERTER",
)


def _d2_assign_targets(self) -> "tuple[torch.Tensor, torch.Tensor]":
    """RUSH's pre-pickup route, unchanged; SPLIT's lane-clearance-aware
    locked routing for the CARRIER only post-pickup; RUSH's original
    pressure-lane target kept for the non-carrier. Bound onto a core
    instance in place of _assign_blue_style_targets for diagnostic-only
    rollouts -- never touches the real dispatch table."""
    B, N = self.B, self.Nb
    idx_env = torch.arange(B, device=self.device)
    max_x = float(max(0, self.cols - 1))
    max_y = float(max(0, self.rows - 1))
    midline = float(self.cols) * 0.5

    own_x, own_y = self.blue_x, self.blue_y
    own_carrying = self.blue_carrying
    own_alive = self.blue_alive
    own_flag_home = self.blue_flag_home
    enemy_x, enemy_y = self.red_x, self.red_y
    enemy_alive = self.red_alive
    enemy_flag_pos = self.red_flag_pos

    target_x, target_y = self._blue_rush_targets(own_x, own_y, enemy_x, enemy_y, enemy_alive, enemy_flag_pos, idx_env)

    if own_carrying.any():
        # SPLIT's clearance-aware, locked carrier-return computation (reused
        # verbatim); only its CARRIER-slot result is kept.
        split_tx, split_ty = self._blue_split_post_pickup_targets(
            target_x.clone(), target_y.clone(), own_x, own_y, own_alive, own_carrying,
            own_flag_home, enemy_x, enemy_y, enemy_alive, max_y, midline, idx_env,
        )
        carrier_idx = torch.argmax(own_carrying.to(torch.int64), dim=1)
        carrier_slot = torch.arange(N, device=self.device)[None, :] == carrier_idx[:, None]
        target_x = torch.where(own_carrying.any(dim=1)[:, None] & carrier_slot, split_tx, target_x)
        target_y = torch.where(own_carrying.any(dim=1)[:, None] & carrier_slot, split_ty, target_y)

        # RUSH's original non-carrier "pressure lane" target, unchanged.
        other_idx = torch.where(carrier_idx == 0, torch.ones_like(carrier_idx), torch.zeros_like(carrier_idx))
        carrier_y = own_y[idx_env, carrier_idx]
        upper_lane_y = torch.full((B,), max_y * 0.90, dtype=own_y.dtype, device=own_y.device)
        lower_lane_y = torch.full((B,), max_y * 0.10, dtype=own_y.dtype, device=own_y.device)
        pressure_lane_y = torch.where(carrier_y >= max_y * 0.5, lower_lane_y, upper_lane_y)
        noncarrier_slot = torch.arange(N, device=self.device)[None, :] == other_idx[:, None]
        rush_pressure = own_carrying.any(dim=1)[:, None] & noncarrier_slot
        target_x = torch.where(rush_pressure, enemy_flag_pos[:, 0:1], target_x)
        target_y = torch.where(rush_pressure, pressure_lane_y[:, None], target_y)
    else:
        # Keep SPLIT's lock state clean for episodes where no pickup happens.
        self._blue_split_prev_carrying = own_carrying.detach().clone()
        ticks = getattr(self, "_blue_split_escape_ticks", None)
        if ticks is not None and ticks.shape[0] == B:
            self._blue_split_escape_ticks = torch.zeros_like(ticks).detach()

    target_x = torch.clamp(target_x, 0.0, max_x)
    target_y = torch.clamp(target_y, 0.0, max_y)
    self._debug_blue_target_x = target_x.detach()
    self._debug_blue_target_y = target_y.detach()
    return target_x, target_y


def run_episode(controller: str, red: str, seed: int, *, steps: int, device: str) -> dict:
    cfg = GPUFieldConfig(
        n_envs=1, max_blue_agents=2, max_red_agents=2, map_layout="map_b_split_lane",
        max_decision_steps=steps + 10, aquaticus_profile=True, rules_profile="OURS",
        device=device, seed=seed,
    )
    env = GPUCTFVecEnv(cfg)
    core = env.core
    try:
        env.env_method("set_phase", red)
        env.env_method("set_next_opponent", "SCRIPTED", red)
        core.blue_scripted = True
        if controller == "D2":
            core.set_blue_style("BLUE_RUSH")  # seeds the same reset-state hooks
            core._assign_blue_style_targets = types.MethodType(_d2_assign_targets, core)
        else:
            core.set_blue_style(controller)
        env.reset()

        first_pickup_step = None
        first_score_step = None
        return_time = None
        n_steps = 0
        for t in range(steps):
            act = env.action_space.sample() * 0
            env.step_async(act)
            obs, rew, done, infos = env.step_wait()
            n_steps += 1
            carrying = core.blue_carrying[0].detach().cpu()
            blue_score = int(core.blue_score[0].item())
            if carrying.any() and first_pickup_step is None:
                first_pickup_step = t
            if blue_score > 0 and first_score_step is None:
                first_score_step = t
                if first_pickup_step is not None:
                    return_time = t - first_pickup_step
            if done.any():
                break
        return {"seed": seed, "steps": n_steps, "first_pickup_step": first_pickup_step,
                "first_score_step": first_score_step, "return_time": return_time}
    finally:
        env.close()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-seed", type=int, default=591001)
    p.add_argument("--n-episodes", type=int, default=4)
    p.add_argument("--steps", type=int, default=240)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    controllers = ("BLUE_RUSH", "D2", "BLUE_SPLIT")
    for red in OPPONENTS:
        print(f"=== {red} ===")
        for controller in controllers:
            results = [
                run_episode(controller, red, args.base_seed + i, steps=args.steps, device=args.device)
                for i in range(args.n_episodes)
            ]
            n = len(results)
            n_score = sum(1 for r in results if r["first_score_step"] is not None)
            rets = [r["return_time"] for r in results if r["return_time"] is not None]
            ret_s = f"{statistics.mean(rets):.1f}" if rets else "n/a"
            print(f"  {controller:10s} n_score={n_score}/{n} mean_return_time={ret_s} "
                  f"detail={[(r['first_pickup_step'], r['first_score_step'], r['return_time']) for r in results]}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
