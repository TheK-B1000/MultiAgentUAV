"""Scripted red-team target/action helpers for BatchedCTFCore."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from macro_actions import MacroAction

from .._paths import _resolve_snapshot_path


def macro_commit_ticks(
    macro: torch.Tensor,
    *,
    go_to_ticks: int,
    grab_ticks: int,
    get_flag_ticks: int,
    place_ticks: int,
    go_home_ticks: int,
) -> torch.Tensor:
    ticks = torch.full_like(macro, int(go_to_ticks), dtype=torch.int32)
    ticks = torch.where(macro == int(MacroAction.GRAB_MINE), torch.full_like(ticks, int(grab_ticks)), ticks)
    ticks = torch.where(macro == int(MacroAction.GET_FLAG), torch.full_like(ticks, int(get_flag_ticks)), ticks)
    ticks = torch.where(macro == int(MacroAction.PLACE_MINE), torch.full_like(ticks, int(place_ticks)), ticks)
    ticks = torch.where(macro == int(MacroAction.GO_HOME), torch.full_like(ticks, int(go_home_ticks)), ticks)
    return torch.clamp(ticks, min=1)


class _ScriptedRedMixin:
    def _decode_targets(self, target_idx: torch.Tensor, side: str = "blue") -> torch.Tensor:
        tidx = torch.remainder(target_idx, self.cfg.n_targets).long()
        n_agents = target_idx.shape[1]
        out = self._macro_targets.index_select(0, tidx.reshape(-1)).reshape(self.B, n_agents, 2)
        if side == "red":
            out = out.clone()
            out[..., 0] = self._mirror_x(out[..., 0], side)
        return out

    def _macro_commit_ticks(self, macro: torch.Tensor) -> torch.Tensor:
        return macro_commit_ticks(
            macro,
            go_to_ticks=int(self.cfg.macro_commit_go_to_ticks),
            grab_ticks=int(self.cfg.macro_commit_grab_ticks),
            get_flag_ticks=int(self.cfg.macro_commit_get_flag_ticks),
            place_ticks=int(self.cfg.macro_commit_place_ticks),
            go_home_ticks=int(self.cfg.macro_commit_go_home_ticks),
        )

    # ------------------------------------------------------------------
    # Carrier evasion: multi-threat tangent routing toward home
    # ------------------------------------------------------------------
    def _carrier_evasion_target(
        self,
        own_x: torch.Tensor,
        own_y: torch.Tensor,
        home_x: torch.Tensor,
        home_y: torch.Tensor,
        enemy_x: torch.Tensor,
        enemy_y: torch.Tensor,
        enemy_alive: torch.Tensor,
        carrying: torch.Tensor,
        side: str = "blue",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        For carriers, waypoint that routes *around* defenders using the same
        side-weighted tangent hook: tangent perpendicular to goal (home),
        exponential repulsion by distance. step_side=7 clears the 2.5 tag radius.
        """
        threat_radius = 8.0
        center_y = 10.0
        hx = home_x[:, None].expand_as(own_x)
        hy = home_y[:, None].expand_as(own_y)
        home_dx = hx - own_x
        home_dy = hy - own_y
        home_n = torch.sqrt(home_dx ** 2 + home_dy ** 2 + 1e-8)
        home_ux = home_dx / home_n
        home_uy = home_dy / home_n

        # Tangent perpendicular to GOAL (home) direction
        tan_x = -home_uy
        tan_y = home_ux

        # Pairwise distances [B, N_own, N_enemy]; nearest *alive* enemy per agent
        dxx = own_x[:, :, None] - enemy_x[:, None, :]
        dyy = own_y[:, :, None] - enemy_y[:, None, :]
        dd = torch.sqrt(dxx ** 2 + dyy ** 2 + 1e-8)
        big = torch.full_like(dd, 1e9)
        dd_masked = torch.where(enemy_alive[:, None, :], dd, big)
        nearest_dist = dd_masked.min(dim=2)[0]
        in_range = nearest_dist < threat_radius
        nearest_dist = nearest_dist.clamp(min=1e-6)

        # Same exponential repulsion as striker: ( (8 - d) / 8 )^2
        repulsion = torch.pow(torch.clamp(threat_radius - nearest_dist, min=0.0) / threat_radius, 2.0)
        repulsion = repulsion * in_range.float()

        # Side bias: above center -> go high, below -> go low
        side_bias = torch.where(own_y > center_y, 1.0, -1.0)

        step_fwd = 2.0
        step_side = 7.0
        evade_tx = own_x + home_ux * step_fwd + tan_x * side_bias * repulsion * step_side
        evade_ty = own_y + home_uy * step_fwd + tan_y * side_bias * repulsion * step_side
        evade_tx = torch.clamp(evade_tx, 0.0, float(max(0, self.cols - 1)))
        evade_ty = torch.clamp(evade_ty, 0.0, float(max(0, self.rows - 1)))

        has_threat = repulsion > 0.0
        should_evade = has_threat & carrying
        final_tx = torch.where(should_evade, evade_tx, hx)
        final_ty = torch.where(should_evade, evade_ty, hy)
        return final_tx, final_ty

    # ------------------------------------------------------------------
    # Unified NPC brain for both sides
    # ------------------------------------------------------------------
    def _get_scripted_targets(self, side: str) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._assign_scripted_targets_by_role(side)

    def _assign_scripted_targets_by_role(self, side: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Consolidated NPC brain usable for both "blue" and "red".

        Agent roles (parameterised by *side*); all N agents are assigned a target:
          Agent 0 – Defender / Guardian: patrols own half, intercepts intruders.
          Agent 1 – Striker: attacks enemy flag with lane preference and tangent evasion.
          Agents 2..N-1 – Strikers with lane spread (so 4v4/6v6/8v8 all have roles, no (0,0) targets).

        All carriers (any agent index) get multi-threat tangent evasion toward
        home via ``_carrier_evasion_target`` so they dodge enemies instead of
        running straight back.
        """
        is_blue = (side == "blue")
        B, device = self.B, self.device
        midline = float(self.cols) * 0.5
        idx_env = torch.arange(B, device=device)
        max_x = float(max(0, self.cols - 1))
        max_y = float(max(0, self.rows - 1))
        split_lane_v2 = str(getattr(self, "map_layout", "")) == "map_b_split_lane_v2"
        if split_lane_v2 and (not is_blue):
            op5_mask = torch.as_tensor(
                [str(k).upper() in ("OP5", "OP5_RUSHER") for k in self._opponent_key],
                device=device,
                dtype=torch.bool,
            )
            op6_mask = torch.as_tensor(
                [str(k).upper() in ("OP6", "OP6_TURTLE") for k in self._opponent_key],
                device=device,
                dtype=torch.bool,
            )
            op7_mask = torch.as_tensor(
                [str(k).upper() in ("OP7", "OP7_SWITCHER") for k in self._opponent_key],
                device=device,
                dtype=torch.bool,
            )
        else:
            op5_mask = torch.zeros((B,), device=device, dtype=torch.bool)
            op6_mask = torch.zeros((B,), device=device, dtype=torch.bool)
            op7_mask = torch.zeros((B,), device=device, dtype=torch.bool)

        if is_blue:
            N, Ne = self.Nb, self.Nr
            own_x, own_y = self.blue_x, self.blue_y
            own_carrying = self.blue_carrying
            own_tagged = self.blue_tagged
            own_alive = self.blue_alive
            own_flag_home = self.blue_flag_home
            enemy_x, enemy_y = self.red_x, self.red_y
            enemy_carrying = self.red_carrying
            enemy_alive = self.red_alive
            enemy_flag_pos = self.red_flag_pos
            enemy_flag_home = self.red_flag_home
            atk_medium = torch.ones((B,), dtype=torch.bool, device=device)
            def_medium = torch.ones((B,), dtype=torch.bool, device=device)
            deception_prob = torch.zeros((B,), dtype=torch.float32, device=device)
            role_switch_prob = torch.zeros((B,), dtype=torch.float32, device=device)
        else:
            N, Ne = self.Nr, self.Nb
            own_x, own_y = self.red_x, self.red_y
            own_carrying = self.red_carrying
            own_tagged = self.red_tagged
            own_alive = self.red_alive
            own_flag_home = self.red_flag_home
            enemy_x, enemy_y = self.blue_x, self.blue_y
            enemy_carrying = self.blue_carrying
            enemy_alive = self.blue_alive
            enemy_flag_pos = self.blue_flag_pos
            enemy_flag_home = self.blue_flag_home
            atk_medium = self.red_attacker_style > 0
            def_medium = self.red_defender_style > 0
            deception_prob = self.red_deception_prob
            role_switch_prob = self.red_role_switch_prob

        target = torch.zeros((B, N, 2), dtype=torch.float32, device=device)
        guardian_idx = 0
        striker_idx = 1 if N > 1 else 0
        if (not is_blue) and N > 1:
            flip = self.red_script_role_flip
            guardian_idx_t = torch.where(
                flip,
                torch.ones((B,), dtype=torch.int64, device=device),
                torch.zeros((B,), dtype=torch.int64, device=device),
            )
            striker_idx_t = torch.where(
                flip,
                torch.zeros((B,), dtype=torch.int64, device=device),
                torch.ones((B,), dtype=torch.int64, device=device),
            )
        else:
            guardian_idx_t = torch.full((B,), guardian_idx, dtype=torch.int64, device=device)
            striker_idx_t = torch.full((B,), striker_idx, dtype=torch.int64, device=device)

        # Advanced indexing uses these as column indices; clamp defensively (avoids CUDA assert if corrupted).
        n_max = max(0, N - 1)
        guardian_idx_t = torch.clamp(guardian_idx_t, 0, n_max)
        striker_idx_t = torch.clamp(striker_idx_t, 0, n_max)

        # ---- shared team state ----
        enemy_carrier_exists = enemy_carrying.any(dim=1)
        if is_blue:
            enemy_on_own = enemy_alive & (enemy_x < midline)
        else:
            enemy_on_own = enemy_alive & (enemy_x > midline)
        any_intruder = enemy_on_own.any(dim=1)

        if not is_blue:
            win_u = torch.clamp(self.red_attack_sync_window, 0, 64)
            need_refresh = self.red_coordinated_attack & (self.red_coord_ticks_left <= 0)
            if need_refresh.any():
                ei = torch.where(need_refresh)[0]
                has_c = enemy_carrier_exists[ei]
                ci = torch.argmax(enemy_carrying[ei].to(torch.int64), dim=1)
                qx = torch.where(has_c, enemy_x[ei, ci], enemy_flag_pos[ei, 0])
                qy = torch.where(has_c, enemy_y[ei, ci], enemy_flag_pos[ei, 1])
                self.red_coord_aim_x[ei] = qx
                self.red_coord_aim_y[ei] = qy
                self.red_coord_ticks_left[ei] = torch.maximum(
                    win_u[ei], torch.ones_like(win_u[ei], dtype=torch.int32)
                )

        guardian_out = own_tagged[idx_env, guardian_idx_t] if N > 0 else torch.zeros((B,), dtype=torch.bool, device=device)
        role_coin = torch.rand((B,), generator=self._rng, device=device) < torch.clamp(role_switch_prob, 0.0, 1.0)
        striker_pivot = guardian_out & (enemy_carrier_exists | any_intruder) & role_coin

        # ======== Defender (agent 0) ========
        if guardian_idx < N:
            phase = self.sim_step_count.to(torch.float32) * 0.12
            orbit_r = 2.0
            easy_x = torch.clamp(own_flag_home[:, 0] + orbit_r * torch.cos(phase), 0.0, max_x)
            easy_y = torch.clamp(own_flag_home[:, 1] + orbit_r * torch.sin(phase), 0.0, max_y)
            # Defender loiter: opens midline. Red anchor is randomized per episode.
            if is_blue:
                med_x = torch.full((B,), min(max_x, 3.0), device=device)
                med_y = torch.full((B,), min(max_y, 10.0), device=device)
            else:
                med_x = torch.clamp(self.red_script_guard_x, 0.0, max_x)
                med_y = torch.clamp(self.red_script_guard_y, 0.0, max_y)
                if split_lane_v2:
                    upper_gate_y = torch.full((B,), max(0.0, max_y * 0.24), device=device)
                    lower_gate_y = torch.full((B,), min(max_y, max_y * 0.76), device=device)
                    mid_gate_y = torch.full((B,), max_y * 0.50, device=device)
                    med_y = torch.where(op5_mask, upper_gate_y, med_y)
                    med_y = torch.where(op6_mask, lower_gate_y, med_y)
                    med_y = torch.where(op7_mask, mid_gate_y, med_y)
            gx = torch.where(def_medium, med_x, easy_x)
            gy = torch.where(def_medium, med_y, easy_y)

            if enemy_carrier_exists.any():
                ci = torch.argmax(enemy_carrying.to(torch.int64), dim=1)
                gx = torch.where(enemy_carrier_exists, enemy_x[idx_env, ci], gx)
                gy = torch.where(enemy_carrier_exists, enemy_y[idx_env, ci], gy)
            else:
                chase = def_medium & any_intruder
                if chase.any():
                    guard_x = own_x[idx_env, guardian_idx_t]
                    guard_y = own_y[idx_env, guardian_idx_t]
                    dxx = guard_x[:, None] - enemy_x
                    dyy = guard_y[:, None] - enemy_y
                    dd = torch.sqrt(dxx * dxx + dyy * dyy + 1e-8)
                    big = torch.full_like(dd, 1e9)
                    dd_masked = torch.where(enemy_on_own, dd, big)
                    nearest = torch.argmin(dd_masked, dim=1)
                    gx = torch.where(chase, enemy_x[idx_env, nearest], gx)
                    gy = torch.where(chase, enemy_y[idx_env, nearest], gy)

            target[idx_env, guardian_idx_t, 0] = gx
            target[idx_env, guardian_idx_t, 1] = gy

        # ======== Striker (agent 1): lane preference + side-weighted tangent hook ========
        center_y = 10.0
        lane_y_north = min(max_y, 15.0)
        lane_y_south = max(0.0, 5.0)
        if striker_idx < N:
            efx = enemy_flag_pos[:, 0]
            efy = enemy_flag_pos[:, 1]
            rx = own_x[idx_env, striker_idx_t]
            ry = own_y[idx_env, striker_idx_t]
            # Lane preference: randomized for red each episode instead of fixed south-only routing.
            dist_to_flag = torch.sqrt((rx - efx) ** 2 + (ry - efy) ** 2 + 1e-8)
            if is_blue:
                lane_y = torch.full((B,), lane_y_north, device=device)
            else:
                lane_mid = torch.full((B,), center_y, device=device)
                lane_amp = torch.full((B,), 5.0, device=device)
                lane_y = torch.clamp(lane_mid + self.red_script_lane_sign * lane_amp, 0.0, max_y)
                if split_lane_v2:
                    upper_gate_y = torch.full((B,), max(0.0, max_y * 0.24), device=device)
                    lower_gate_y = torch.full((B,), min(max_y, max_y * 0.76), device=device)
                    mid_gate_y = torch.full((B,), max_y * 0.50, device=device)
                    lane_y = torch.where(op5_mask, upper_gate_y, lane_y)
                    lane_y = torch.where(op6_mask, lower_gate_y, lane_y)
                    lane_y = torch.where(op7_mask, mid_gate_y, lane_y)
            sy_easy = torch.where(dist_to_flag > 4.0, lane_y, efy)
            sx_easy = efx
            sx_med = sx_easy.clone()
            sy_med = sy_easy.clone()

            if atk_medium.any():
                dxx = rx[:, None] - enemy_x
                dyy = ry[:, None] - enemy_y
                dd = torch.sqrt(dxx * dxx + dyy * dyy + 1e-8)
                nearest = torch.argmin(dd, dim=1)
                nbx = enemy_x[idx_env, nearest]
                nby = enemy_y[idx_env, nearest]

                goal_dx = sx_easy - rx
                goal_dy = sy_easy - ry
                goal_n = torch.sqrt(goal_dx * goal_dx + goal_dy * goal_dy + 1e-8)
                goal_dx = goal_dx / goal_n
                goal_dy = goal_dy / goal_n

                # Tangent perpendicular to GOAL direction (not to enemy)
                tan_x = -goal_dy
                tan_y = goal_dx

                away_x = rx - nbx
                away_y = ry - nby
                dist_to_enemy = torch.sqrt(away_x * away_x + away_y * away_y + 1e-8)
                # Exponential repulsion: stronger as enemy gets closer (hook 6 when within 8)
                repulsion = torch.pow(torch.clamp(8.0 - dist_to_enemy, min=0.0) / 8.0, 2.0)
                # Side bias: above center -> go high, below -> go low (breaks symmetry)
                side_bias = torch.where(ry > center_y, 1.0, -1.0)

                tx_med = rx + (goal_dx * 2.0) + (tan_x * side_bias * repulsion * 6.0)
                ty_med = ry + (goal_dy * 2.0) + (tan_y * side_bias * repulsion * 6.0)
                sx_med = torch.clamp(tx_med, 0.0, max_x)
                sy_med = torch.clamp(ty_med, 0.0, max_y)

            sx = torch.where(atk_medium, sx_med, sx_easy)
            sy = torch.where(atk_medium, sy_med, sy_easy)

            # Dynamic pivot when guardian is tagged and threats exist
            if striker_pivot.any():
                threat_mask = enemy_on_own | enemy_carrying
                dxx = rx[:, None] - enemy_x
                dyy = ry[:, None] - enemy_y
                dd = torch.sqrt(dxx * dxx + dyy * dyy + 1e-8)
                big = torch.full_like(dd, 1e9)
                dd_masked = torch.where(threat_mask, dd, big)
                nearest = torch.argmin(dd_masked, dim=1)
                sx = torch.where(striker_pivot, enemy_x[idx_env, nearest], sx)
                sy = torch.where(striker_pivot, enemy_y[idx_env, nearest], sy)

            target[idx_env, striker_idx_t, 0] = sx
            target[idx_env, striker_idx_t, 1] = sy

        # ======== Extra agents (N > 2): assign striker-like targets with lane spread ========
        # So 4v4/6v6/8v8 all have roles; agents 2..N-1 go to enemy flag with y-offset to avoid clustering.
        efx = enemy_flag_pos[:, 0]
        efy = enemy_flag_pos[:, 1]
        for j in range(2, N):
            lane_offset = (j - 1) * (max_y * 0.25 / max(1, N - 1))  # spread across lanes
            if is_blue:
                lane_y_j = torch.clamp(efy + lane_offset, 0.0, max_y)
            else:
                lane_y_j = torch.clamp(efy - lane_offset, 0.0, max_y)
            target[:, j, 0] = efx
            target[:, j, 1] = lane_y_j

        # ======== Carrier evasion (replaces straight-home override) ========
        # Any agent carrying a flag uses multi-threat tangent routing so they
        # dodge nearby enemies instead of heading straight back.
        if own_carrying.any():
            evade_tx, evade_ty = self._carrier_evasion_target(
                own_x, own_y,
                own_flag_home[:, 0], own_flag_home[:, 1],
                enemy_x, enemy_y, enemy_alive,
                own_carrying,
                side=side,
            )
            target[..., 0] = torch.where(own_carrying, evade_tx, target[..., 0])
            target[..., 1] = torch.where(own_carrying, evade_ty, target[..., 1])

        # ======== Safety: non-carriers intercept enemy who stole own flag ========
        if enemy_carrier_exists.any():
            ci = torch.argmax(enemy_carrying.to(torch.int64), dim=1)
            cx = enemy_x[idx_env, ci]
            cy = enemy_y[idx_env, ci]
            not_carrying = ~own_carrying
            intercept = enemy_carrier_exists[:, None] & not_carrying
            target[..., 0] = torch.where(intercept, cx[:, None], target[..., 0])
            target[..., 1] = torch.where(intercept, cy[:, None], target[..., 1])

        # ======== Carrier shielding: escort 4 units perpendicular to carrier path ========
        shield_dist = 4.0
        own_carry_any = own_carrying.any(dim=1)
        if own_carry_any.any() and N > 1:
            carr_idx = torch.argmax(own_carrying.to(torch.int64), dim=1)
            carr_x = own_x[idx_env, carr_idx]
            carr_y = own_y[idx_env, carr_idx]
            home_ux = own_flag_home[:, 0] - carr_x
            home_uy = own_flag_home[:, 1] - carr_y
            home_n = torch.sqrt(home_ux ** 2 + home_uy ** 2 + 1e-8)
            home_ux = home_ux / home_n
            home_uy = home_uy / home_n
            perp1_x = -home_uy
            perp1_y = home_ux
            dxx = carr_x[:, None] - enemy_x
            dyy = carr_y[:, None] - enemy_y
            dd = torch.sqrt(dxx * dxx + dyy * dyy + 1e-8)
            near_enemy = torch.argmin(dd, dim=1)
            nex = enemy_x[idx_env, near_enemy]
            ney = enemy_y[idx_env, near_enemy]
            to_enemy_x = nex - carr_x
            to_enemy_y = ney - carr_y
            dot_perp1 = perp1_x * to_enemy_x + perp1_y * to_enemy_y
            use_perp1 = dot_perp1 >= 0.0
            escort_off_x = torch.where(use_perp1, perp1_x, -perp1_x)
            escort_off_y = torch.where(use_perp1, perp1_y, -perp1_y)
            escort_x = carr_x + escort_off_x * shield_dist
            escort_y = carr_y + escort_off_y * shield_dist
            escort_x = torch.clamp(escort_x, 0.0, max_x)
            escort_y = torch.clamp(escort_y, 0.0, max_y)
            for j in range(N):
                is_not_carrier = own_carry_any & (carr_idx != j) & (~own_carrying[:, j])
                escort_ok = is_not_carrier & (~enemy_carrier_exists)
                target[:, j, 0] = torch.where(escort_ok, escort_x, target[:, j, 0])
                target[:, j, 1] = torch.where(escort_ok, escort_y, target[:, j, 1])

        # ======== Red-only: deception feints (non-carrier agents only) ========
        if (not is_blue) and deception_prob.numel() == B and float(deception_prob.max().item()) > 0.0:
            pulse = (self.sim_step_count % 30 == 0)
            p = torch.clamp(deception_prob, 0.0, 1.0)
            r = torch.rand((B,), generator=self._rng, device=device)
            feint_env = pulse & (r < p)
            if feint_env.any():
                env_idx = torch.where(feint_env)[0]
                hold_x = torch.full((env_idx.numel(),), min(max_x, midline - 0.5), device=device)
                hold_y = enemy_flag_home[env_idx, 1]
                punch_x = enemy_flag_home[env_idx, 0]
                punch_y = enemy_flag_home[env_idx, 1]
                do_hold = ((self.sim_step_count[env_idx] // 30) % 2 == 0)
                tx = torch.where(do_hold, hold_x, punch_x)
                ty = torch.where(do_hold, hold_y, punch_y)
                tx = torch.clamp(tx + self._rand_uniform((env_idx.numel(),), -3.0, 3.0), 0.0, max_x)
                ty = torch.clamp(ty + self._rand_uniform((env_idx.numel(),), -3.0, 3.0), 0.0, max_y)
                for j in range(N):
                    not_carry = ~own_carrying[env_idx, j]
                    target[env_idx, j, 0] = torch.where(not_carry, tx, target[env_idx, j, 0])
                    target[env_idx, j, 1] = torch.where(not_carry, ty, target[env_idx, j, 1])

        if not is_blue:
            own_carry_any = own_carrying.any(dim=1)
            hold = self.red_coordinated_attack & (self.red_coord_ticks_left > 0)
            if hold.any():
                ax = self.red_coord_aim_x
                ay = self.red_coord_aim_y
                for j in range(N):
                    not_g = guardian_idx_t != j
                    m = hold & not_g & (~own_carrying[:, j]) & (~own_carry_any)
                    target[:, j, 0] = torch.where(m, ax, target[:, j, 0])
                    target[:, j, 1] = torch.where(m, ay, target[:, j, 1])
            c = self.red_coordinated_attack
            self.red_coord_ticks_left = torch.where(
                c,
                torch.clamp(self.red_coord_ticks_left - 1, min=0),
                self.red_coord_ticks_left,
            )

        return target[..., 0], target[..., 1]

    def _red_scripted_actions(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Scripted red team -- delegates to the unified NPC brain."""
        return self._get_scripted_targets("red")

    def _get_red_snapshot_actions(self, env_mask: torch.Tensor) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        idx = torch.where(env_mask)[0]
        if idx.numel() == 0:
            return None
        obs_full = self.get_obs_tensors(side="red")
        red_macro = torch.zeros((self.B, self.Nr), device=self.device, dtype=torch.int64)
        red_target = torch.zeros((self.B, self.Nr), device=self.device, dtype=torch.int64)
        any_loaded = False
        grouped: Dict[str, List[int]] = {}
        for env_i in idx.detach().cpu().tolist():
            resolved = _resolve_snapshot_path(self._opponent_key[env_i])
            if resolved is None:
                continue
            grouped.setdefault(self._opponent_key[env_i], []).append(env_i)
        for snapshot_key, env_list in grouped.items():
            model = self._load_snapshot_policy(snapshot_key)
            if model is None:
                continue
            any_loaded = True
            sub_idx = torch.as_tensor(env_list, device=self.device, dtype=torch.int64)
            obs = {
                k: v.index_select(0, sub_idx).detach().cpu().numpy().astype(np.float32)
                for k, v in obs_full.items()
            }
            obs["global_state"] = self.get_global_state()[env_list]
            actions_np, _ = model.predict(obs, deterministic=True)
            actions = torch.as_tensor(actions_np, device=self.device, dtype=torch.int64).reshape(sub_idx.numel(), self.Nr, 2)
            red_macro[sub_idx] = torch.remainder(actions[..., 0], self.cfg.n_macros).long()
            red_target[sub_idx] = torch.remainder(actions[..., 1], self.cfg.n_targets).long()
        if not any_loaded:
            return None
        return red_macro, red_target

    def _apply_red_action_commit(
        self,
        red_action_flat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if red_action_flat.device != self.device:
            red_action_flat = red_action_flat.to(self.device)
        n_red_exp = int(self.B * self.Nr * 2)
        if int(red_action_flat.numel()) != n_red_exp:
            raise ValueError(
                f"BatchedCTFCore.step: expected {n_red_exp} red action ints (B={self.B}, Nr={self.Nr}), "
                f"got numel={int(red_action_flat.numel())} shape={tuple(red_action_flat.shape)}"
            )
        red_a = red_action_flat.reshape(self.B, self.Nr, 2)
        red_requested_macro = torch.remainder(red_a[..., 0].long(), self.cfg.n_macros)
        red_requested_targ = torch.remainder(red_a[..., 1].long(), self.cfg.n_targets)
        red_control_mask = torch.ones((self.B,), device=self.device, dtype=torch.bool)
        external_red_mask = red_control_mask[:, None]
        new_red_commit = external_red_mask & (self.red_commit_ticks_left <= 0)
        self.red_commit_macro = torch.where(new_red_commit, red_requested_macro, self.red_commit_macro)
        self.red_commit_target = torch.where(new_red_commit, red_requested_targ, self.red_commit_target)
        self.red_commit_ticks_left = torch.where(
            new_red_commit,
            self._macro_commit_ticks(red_requested_macro),
            self.red_commit_ticks_left,
        )
        self.red_commit_success = torch.where(
            new_red_commit,
            torch.zeros_like(self.red_commit_success),
            self.red_commit_success,
        )
        red_macro = self.red_commit_macro
        rtx, rty = self._build_targets_from_action(red_macro, self.red_commit_target, side="red")
        return rtx, rty, red_macro, red_control_mask

