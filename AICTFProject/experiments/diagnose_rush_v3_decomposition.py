#!/usr/bin/env python3
"""RUSH_PROBE_ROOT_CAUSE_AUDIT -- Phases 1+2: decompose BLUE_PROBES_V3's
RUSH controller into its two post-pickup components (direct-home carrier
return, screening-blocker teammate) and measure how much each contributes
to the broad dominance found in the map_a held-out confirmations
(OP7/OP9/OP10/OP12 all won by RUSH at n=16-24).

Four diagnostic variants, identical pre-pickup route (V3's current tight
1.5-offset dual approach, held constant across all four so only the
POST-PICKUP mechanism varies):

  R0: old pre-V3 RUSH      -- carrier uses generic _carrier_evasion_target
                              (no direct-home extraction); non-carrier goes
                              to the old "pressure lane near enemy flag"
                              (no screening blocker).
  R1: direct-home only     -- carrier goes straight home, no evasion;
                              non-carrier keeps the OLD pressure-lane
                              target (screening blocker removed).
  R2: screening blocker only -- carrier keeps generic evasion (NOT direct
                              home); non-carrier gets V3's screening-
                              blocker target.
  R3: full V3 (unchanged)  -- direct-home + screening blocker together;
                              calls the real _blue_rush_targets verbatim.

Diagnostic-only: does not modify _scripted_blue_styles.py or any red
opponent/profile. R0/R1/R2 are monkey-patched onto a single core instance
per rollout; R3 uses the real, unmodified method.

Metrics per episode: margin, win/no-win, first pickup step,
pickup->score conversion, carrier return time, tags-while-carrying count,
own-flag-stolen-before-score, teammate-near-carrier fraction (post-pickup,
< 3.0 units), teammate-interposition fraction (post-pickup, geometrically
between carrier and nearest live enemy), mean carrier-teammate distance
post-pickup.
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
    "OP7_DEEP_FORTRESS",
    "OP9_SPLIT_LANE_FEINT",
    "OP10_AGGRESSIVE_INTERCEPTOR",
    "OP12_LATE_CONVERTER",
)
VARIANTS = ("R0", "R1", "R2", "R3")
NEAR_THRESHOLD = 3.0
INTERPOSE_PERP_THRESHOLD = 2.0
RUSH_OFFSET = 1.5


def _pre_pickup(enemy_flag_pos, max_y, device, dtype):
    efx, efy = enemy_flag_pos[:, 0], enemy_flag_pos[:, 1]
    t0x, t0y = efx, efy
    t1x = efx
    t1y = torch.clamp(efy + RUSH_OFFSET, 0.0, max_y)
    return torch.stack([t0x, t1x], dim=1), torch.stack([t0y, t1y], dim=1)


def _old_pressure_lane(carrier_y, max_y, device, dtype):
    B = carrier_y.shape[0]
    upper = torch.full((B,), max_y * 0.90, device=device, dtype=dtype)
    lower = torch.full((B,), max_y * 0.10, device=device, dtype=dtype)
    return torch.where(carrier_y >= max_y * 0.5, lower, upper)


def _screening_blocker(carr_x, carr_y, enemy_x, enemy_y, enemy_alive, hx, hy, efx, efy, max_x, max_y, idx_env):
    dxx = carr_x[:, None] - enemy_x
    dyy = carr_y[:, None] - enemy_y
    dd = torch.sqrt(dxx * dxx + dyy * dyy + 1e-8)
    big = torch.full_like(dd, 1e9)
    dd_live = torch.where(enemy_alive, dd, big)
    near = torch.argmin(dd_live, dim=1)
    nex = enemy_x[idx_env, near]
    ney = enemy_y[idx_env, near]
    inter_x = 0.55 * carr_x + 0.45 * nex
    inter_y = 0.55 * carr_y + 0.45 * ney
    home_dx = hx - carr_x
    home_dy = hy - carr_y
    home_n = torch.sqrt(home_dx * home_dx + home_dy * home_dy + 1e-8)
    ahead = 2.5
    screen_x = torch.clamp(inter_x + (home_dx / home_n) * ahead, 0.0, max_x)
    screen_y = torch.clamp(inter_y + (home_dy / home_n) * ahead, 0.0, max_y)
    any_enemy = enemy_alive.any(dim=1)
    screen_x = torch.where(any_enemy, screen_x, efx)
    screen_y = torch.where(any_enemy, screen_y, torch.clamp(efy + RUSH_OFFSET, 0.0, max_y))
    return screen_x, screen_y


def _make_variant_fn(variant: str):
    def _assign(self) -> "tuple[torch.Tensor, torch.Tensor]":
        B, N = self.B, self.Nb
        idx_env = torch.arange(B, device=self.device)
        max_x = float(max(0, self.cols - 1))
        max_y = float(max(0, self.rows - 1))
        own_x, own_y = self.blue_x, self.blue_y
        own_carrying = self.blue_carrying
        own_alive = self.blue_alive
        own_flag_home = self.blue_flag_home
        enemy_x, enemy_y = self.red_x, self.red_y
        enemy_alive = self.red_alive
        enemy_flag_pos = self.red_flag_pos
        hx, hy = own_flag_home[:, 0], own_flag_home[:, 1]
        efx, efy = enemy_flag_pos[:, 0], enemy_flag_pos[:, 1]

        target_x, target_y = _pre_pickup(enemy_flag_pos, max_y, self.device, own_x.dtype)

        any_carrying = own_carrying.any(dim=1)
        if bool(any_carrying.any().item()):
            carrier_idx = torch.argmax(own_carrying.to(torch.int64), dim=1)
            other_idx = 1 - carrier_idx
            carr_x = own_x[idx_env, carrier_idx]
            carr_y = own_y[idx_env, carrier_idx]

            if variant in ("R1", "R3"):
                c_tx, c_ty = hx, hy
            else:
                c_tx, c_ty = self._carrier_evasion_target(
                    own_x, own_y, hx, hy, enemy_x, enemy_y, enemy_alive, own_carrying, side="blue",
                )
                c_tx, c_ty = c_tx[idx_env, carrier_idx], c_ty[idx_env, carrier_idx]

            if variant in ("R2", "R3"):
                nc_tx, nc_ty = _screening_blocker(carr_x, carr_y, enemy_x, enemy_y, enemy_alive, hx, hy, efx, efy, max_x, max_y, idx_env)
            else:
                nc_tx = efx
                nc_ty = _old_pressure_lane(carr_y, max_y, self.device, own_x.dtype)

            agent_ids = torch.arange(N, device=self.device)[None, :]
            carrier_slot = agent_ids == carrier_idx[:, None]
            other_slot = agent_ids == other_idx[:, None]
            carry_m = any_carrying[:, None]
            target_x = torch.where(carry_m & carrier_slot, c_tx[:, None], target_x)
            target_y = torch.where(carry_m & carrier_slot, c_ty[:, None], target_y)
            other_alive = carry_m & other_slot & own_alive
            target_x = torch.where(other_alive, nc_tx[:, None], target_x)
            target_y = torch.where(other_alive, nc_ty[:, None], target_y)

        target_x = torch.clamp(target_x, 0.0, max_x)
        target_y = torch.clamp(target_y, 0.0, max_y)
        self._debug_blue_target_x = target_x.detach()
        self._debug_blue_target_y = target_y.detach()
        return target_x, target_y
    return _assign


def run_episode(variant: str, red: str, seed: int, *, steps: int, device: str) -> dict:
    cfg = GPUFieldConfig(
        n_envs=1, max_blue_agents=2, max_red_agents=2, map_layout="map_a",
        max_decision_steps=steps + 10, aquaticus_profile=True, rules_profile="OURS",
        device=device, seed=seed,
    )
    env = GPUCTFVecEnv(cfg)
    core = env.core
    try:
        env.env_method("set_phase", red)
        env.env_method("set_next_opponent", "SCRIPTED", red)
        core.blue_scripted = True
        core.set_blue_style("BLUE_RUSH")
        env.reset()
        env.env_method("set_phase", red)
        env.env_method("set_next_opponent", "SCRIPTED", red)
        core.blue_scripted = True
        core.set_blue_style("BLUE_RUSH")
        if variant != "R3":
            core._assign_blue_style_targets = types.MethodType(_make_variant_fn(variant), core)

        first_pickup_step = None
        first_score_step = None
        return_time = None
        tags_while_carrying = 0
        own_flag_lost_step = None
        own_flag_lost_before_score = False
        near_steps = 0
        interpose_steps = 0
        post_pickup_steps = 0
        dist_sum = 0.0
        prev_carrying = None
        n_steps = 0

        act = env.action_space.sample() * 0
        last_info: dict = {}
        for t in range(steps):
            env.step_async(act)
            obs, rew, done, infos = env.step_wait()
            n_steps += 1
            last_info = infos[0] if infos else {}

            bx = core.blue_x[0].detach().cpu()
            by = core.blue_y[0].detach().cpu()
            carrying = core.blue_carrying[0].detach().cpu()
            tagged = core.blue_tagged[0].detach().cpu()
            red_carrying = core.red_carrying[0].detach().cpu()
            red_x = core.red_x[0].detach().cpu()
            red_y = core.red_y[0].detach().cpu()
            red_alive = core.red_alive[0].detach().cpu()
            blue_score = int(core.blue_score[0].item())

            if prev_carrying is not None:
                for j in range(2):
                    if bool(tagged[j]) and bool(prev_carrying[j]):
                        tags_while_carrying += 1

            if carrying.any() and first_pickup_step is None:
                first_pickup_step = t

            if red_carrying.any() and blue_score == 0 and own_flag_lost_step is None:
                own_flag_lost_step = t

            if blue_score > 0 and first_score_step is None:
                first_score_step = t
                if first_pickup_step is not None:
                    return_time = t - first_pickup_step
                if own_flag_lost_step is not None and own_flag_lost_step < t:
                    own_flag_lost_before_score = True

            if carrying.any():
                carrier_idx = int(torch.argmax(carrying.to(torch.int64)).item())
                other_idx = 1 - carrier_idx
                cx, cy = bx[carrier_idx], by[carrier_idx]
                tx_, ty_ = bx[other_idx], by[other_idx]
                d = float(((tx_ - cx) ** 2 + (ty_ - cy) ** 2) ** 0.5)
                dist_sum += d
                post_pickup_steps += 1
                if d < NEAR_THRESHOLD:
                    near_steps += 1
                live = red_alive.nonzero(as_tuple=True)[0].tolist()
                if live:
                    dists = [float(((red_x[k] - cx) ** 2 + (red_y[k] - cy) ** 2) ** 0.5) for k in live]
                    nearest = live[dists.index(min(dists))]
                    ex, ey = red_x[nearest], red_y[nearest]
                    ex_c, ey_c = ex - cx, ey - cy
                    seg_len2 = float(ex_c ** 2 + ey_c ** 2) + 1e-8
                    tpar = ((tx_ - cx) * ex_c + (ty_ - cy) * ey_c) / seg_len2
                    proj_x = cx + tpar * ex_c
                    proj_y = cy + tpar * ey_c
                    perp = float(((tx_ - proj_x) ** 2 + (ty_ - proj_y) ** 2) ** 0.5)
                    if 0.0 <= tpar <= 1.0 and perp < INTERPOSE_PERP_THRESHOLD:
                        interpose_steps += 1

            prev_carrying = carrying
            if done.any():
                break

        ep_res = last_info.get("episode_result", last_info)
        blue_final = int(ep_res.get("blue_score", 0))
        red_final = int(ep_res.get("red_score", 0))
        return {
            "seed": seed, "steps": n_steps, "margin": blue_final - red_final,
            "win": blue_final > red_final,
            "first_pickup_step": first_pickup_step, "first_score_step": first_score_step,
            "return_time": return_time, "tags_while_carrying": tags_while_carrying,
            "own_flag_lost_before_score": own_flag_lost_before_score,
            "teammate_near_frac": (near_steps / post_pickup_steps) if post_pickup_steps else None,
            "teammate_interpose_frac": (interpose_steps / post_pickup_steps) if post_pickup_steps else None,
            "mean_carrier_teammate_dist": (dist_sum / post_pickup_steps) if post_pickup_steps else None,
        }
    finally:
        env.close()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-seed", type=int, default=651001)
    p.add_argument("--n-episodes", type=int, default=8)
    p.add_argument("--steps", type=int, default=240)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    for red in OPPONENTS:
        print(f"=== {red} (map=map_a) ===")
        for variant in VARIANTS:
            results = [
                run_episode(variant, red, args.base_seed + i, steps=args.steps, device=args.device)
                for i in range(args.n_episodes)
            ]
            n = len(results)

            def mean_of(field, results=results):
                vals = [r[field] for r in results if r[field] is not None]
                return statistics.mean(vals) if vals else None

            margin = mean_of("margin")
            wins = sum(1 for r in results if r["win"])
            pickup = mean_of("first_pickup_step")
            n_score = sum(1 for r in results if r["first_score_step"] is not None)
            n_pickup = sum(1 for r in results if r["first_pickup_step"] is not None)
            conv = (n_score / n_pickup) if n_pickup else None
            ret = mean_of("return_time")
            tags = mean_of("tags_while_carrying")
            flag_lost = sum(1 for r in results if r["own_flag_lost_before_score"])
            near = mean_of("teammate_near_frac")
            interpose = mean_of("teammate_interpose_frac")
            dist = mean_of("mean_carrier_teammate_dist")

            def fmt(v, nd=2):
                return f"{v:.{nd}f}" if v is not None else "n/a"

            print(f"  {variant}: margin={fmt(margin)} WR={wins}/{n} pickup={fmt(pickup,1)} "
                  f"conv={fmt(conv)} ({n_score}/{n_pickup}) return={fmt(ret,1)} "
                  f"tags_while_carrying={fmt(tags)} own_flag_lost={flag_lost}/{n} "
                  f"teammate_near={fmt(near)} interpose={fmt(interpose)} "
                  f"carrier_teammate_dist={fmt(dist)}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
