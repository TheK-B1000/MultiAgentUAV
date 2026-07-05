"""Intra-episode adaptive memory and counter-play for OP8..OP12 hardpool v2.

OP8-OP12 share the same opponent IDs as the pre-v6i21 scripted profiles but
maintain episode-local counters that track blue attack patterns and shift red
roles/routes to punish repetition.  Memory resets every episode; no cross-episode
state is stored.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch


ROLE_ATTACKER = 0
ROLE_INTERCEPTOR = 3
ROLE_COUNTER = 5
ROLE_2V1_WING = 6


class _BTAdaptiveMixin:
    """Episode-local adaptive blackboard for levels 8..12."""

    _ADAPTIVE_LEVEL_MIN = 8
    _LANE_EMA_ALPHA = 0.08
    _CARRIER_EMA_ALPHA = 0.12
    _NEAR_CAP_DIST = 5.0
    _FAST_CONVERSION_STEPS = 45
    _REPEAT_LANE_STREAK = 8
    _HIGH_ESCORT_DENSITY = 0.45
    _HIGH_OVERCOMMIT = 0.55

    def _alloc_adaptive_memory(self, B: int, dev: torch.device) -> None:
        f32 = torch.float32
        i32 = torch.int32
        self.bt_adapt_attack_lane_top_frac = torch.zeros((B,), dtype=f32, device=dev)
        self.bt_adapt_attack_lane_bot_frac = torch.zeros((B,), dtype=f32, device=dev)
        self.bt_adapt_escort_density = torch.zeros((B,), dtype=f32, device=dev)
        self.bt_adapt_overcommit = torch.zeros((B,), dtype=f32, device=dev)
        self.bt_adapt_fast_conversion_count = torch.zeros((B,), dtype=i32, device=dev)
        self.bt_adapt_blue_first_touch_step = torch.full((B,), -1, dtype=i32, device=dev)
        self.bt_adapt_carrier_x_ema = torch.zeros((B,), dtype=f32, device=dev)
        self.bt_adapt_carrier_y_ema = torch.zeros((B,), dtype=f32, device=dev)
        self.bt_adapt_near_cap_ticks = torch.zeros((B,), dtype=i32, device=dev)
        self.bt_adapt_repeat_lane_streak = torch.zeros((B,), dtype=i32, device=dev)
        self.bt_adapt_prev_lane_sign = torch.zeros((B,), dtype=i32, device=dev)
        self.bt_adapt_blue_score_prev = torch.zeros((B,), dtype=i32, device=dev)

    def _reset_adaptive_memory(self, env_mask: torch.Tensor) -> None:
        idx = torch.where(env_mask)[0]
        if idx.numel() == 0:
            return
        self.bt_adapt_attack_lane_top_frac[idx] = 0.0
        self.bt_adapt_attack_lane_bot_frac[idx] = 0.0
        self.bt_adapt_escort_density[idx] = 0.0
        self.bt_adapt_overcommit[idx] = 0.0
        self.bt_adapt_fast_conversion_count[idx] = 0
        self.bt_adapt_blue_first_touch_step[idx] = -1
        self.bt_adapt_carrier_x_ema[idx] = 0.0
        self.bt_adapt_carrier_y_ema[idx] = 0.0
        self.bt_adapt_near_cap_ticks[idx] = 0
        self.bt_adapt_repeat_lane_streak[idx] = 0
        self.bt_adapt_prev_lane_sign[idx] = 0
        self.bt_adapt_blue_score_prev[idx] = 0

    def _adaptive_active_mask(self, prof: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self._bt_opponent_mask() & prof["adaptive_enabled"]

    def _update_adaptive_memory(self, prof: Dict[str, torch.Tensor]) -> None:
        """Track blue patterns each step for adaptive hardpool opponents."""
        active = self._adaptive_active_mask(prof)
        if not bool(active.any().item()):
            return

        B, device = self.B, self.device
        midline = float(self.cols) * 0.5
        center_y = float(self.rows) * 0.5
        alpha_lane = self._LANE_EMA_ALPHA
        alpha_carrier = self._CARRIER_EMA_ALPHA

        blue_alive = self.blue_alive & (~self.blue_tagged)
        blue_on_enemy = blue_alive & (self.blue_x > midline)
        top = (blue_on_enemy & (self.blue_y > center_y)).sum(dim=1).float()
        bot = (blue_on_enemy & (self.blue_y <= center_y)).sum(dim=1).float()
        total_lane = (top + bot).clamp(min=1.0)
        top_frac = top / total_lane
        bot_frac = bot / total_lane

        lane_sign = torch.zeros((B,), dtype=torch.int32, device=device)
        lane_sign = torch.where(top > bot + 0.5, torch.ones_like(lane_sign), lane_sign)
        lane_sign = torch.where(bot > top + 0.5, -torch.ones_like(lane_sign), lane_sign)

        same_lane = (lane_sign == self.bt_adapt_prev_lane_sign) & (lane_sign != 0)
        streak = torch.where(
            same_lane & active,
            self.bt_adapt_repeat_lane_streak + 1,
            torch.zeros_like(self.bt_adapt_repeat_lane_streak),
        )
        self.bt_adapt_repeat_lane_streak = torch.where(active, streak, self.bt_adapt_repeat_lane_streak)
        self.bt_adapt_prev_lane_sign = torch.where(active, lane_sign, self.bt_adapt_prev_lane_sign)

        self.bt_adapt_attack_lane_top_frac = torch.where(
            active,
            (1.0 - alpha_lane) * self.bt_adapt_attack_lane_top_frac + alpha_lane * top_frac,
            self.bt_adapt_attack_lane_top_frac,
        )
        self.bt_adapt_attack_lane_bot_frac = torch.where(
            active,
            (1.0 - alpha_lane) * self.bt_adapt_attack_lane_bot_frac + alpha_lane * bot_frac,
            self.bt_adapt_attack_lane_bot_frac,
        )

        blue_carry_any = (self.blue_carrying & blue_alive).any(dim=1)
        idx_env = torch.arange(B, device=device)
        blue_ci = torch.where(
            blue_carry_any,
            torch.argmax((self.blue_carrying & (~self.blue_tagged)).to(torch.int64), dim=1),
            torch.zeros((B,), dtype=torch.int64, device=device),
        ).clamp(min=0)
        ec_x = torch.where(blue_carry_any, self.blue_x[idx_env, blue_ci], self.bt_adapt_carrier_x_ema)
        ec_y = torch.where(blue_carry_any, self.blue_y[idx_env, blue_ci], self.bt_adapt_carrier_y_ema)
        self.bt_adapt_carrier_x_ema = torch.where(
            active & blue_carry_any,
            (1.0 - alpha_carrier) * self.bt_adapt_carrier_x_ema + alpha_carrier * ec_x,
            self.bt_adapt_carrier_x_ema,
        )
        self.bt_adapt_carrier_y_ema = torch.where(
            active & blue_carry_any,
            (1.0 - alpha_carrier) * self.bt_adapt_carrier_y_ema + alpha_carrier * ec_y,
            self.bt_adapt_carrier_y_ema,
        )

        if blue_carry_any.any():
            cdx = self.blue_x - ec_x[:, None]
            cdy = self.blue_y - ec_y[:, None]
            cdist = torch.sqrt(cdx ** 2 + cdy ** 2 + 1e-8)
            near = (cdist < 6.0) & blue_alive
            escort = near.sum(dim=1).float() / blue_alive.sum(dim=1).float().clamp(min=1.0)
            self.bt_adapt_escort_density = torch.where(
                active & blue_carry_any,
                (1.0 - alpha_lane) * self.bt_adapt_escort_density + alpha_lane * escort,
                self.bt_adapt_escort_density,
            )

        own_flag_home = (
            torch.abs(self.red_flag_pos[:, 0] - self.red_flag_home[:, 0]) < 1.5
        ) & (
            torch.abs(self.red_flag_pos[:, 1] - self.red_flag_home[:, 1]) < 1.5
        )
        over = blue_on_enemy.sum(dim=1).float() / blue_alive.sum(dim=1).float().clamp(min=1.0)
        self.bt_adapt_overcommit = torch.where(
            active & (~own_flag_home),
            (1.0 - alpha_lane) * self.bt_adapt_overcommit + alpha_lane * over,
            self.bt_adapt_overcommit * 0.98,
        )

        home_dx = self.blue_flag_home[:, 0] - ec_x
        home_dy = self.blue_flag_home[:, 1] - ec_y
        ec_home_dist = torch.sqrt(home_dx ** 2 + home_dy ** 2 + 1e-8)
        near_cap = blue_carry_any & (ec_home_dist < self._NEAR_CAP_DIST)
        cap_ticks = torch.where(
            active & near_cap,
            self.bt_adapt_near_cap_ticks + 1,
            torch.zeros_like(self.bt_adapt_near_cap_ticks),
        )
        self.bt_adapt_near_cap_ticks = torch.where(active, cap_ticks, self.bt_adapt_near_cap_ticks)

        first_touch = self.bt_adapt_blue_first_touch_step < 0
        new_touch = active & blue_carry_any & first_touch
        self.bt_adapt_blue_first_touch_step = torch.where(
            new_touch,
            self.sim_step_count.to(torch.int32),
            self.bt_adapt_blue_first_touch_step,
        )

        scored = active & (self.blue_score > self.bt_adapt_blue_score_prev)
        fast = scored & blue_carry_any & (
            self.sim_step_count.to(torch.int32) - self.bt_adapt_blue_first_touch_step
            < self._FAST_CONVERSION_STEPS
        )
        self.bt_adapt_fast_conversion_count += fast.to(torch.int32)
        self.bt_adapt_blue_score_prev = torch.where(active, self.blue_score, self.bt_adapt_blue_score_prev)

    def _extend_blackboard_adaptive(
        self,
        bb: dict,
        prof: Dict[str, torch.Tensor],
    ) -> dict:
        active = self._adaptive_active_mask(prof)
        preferred_lane_y = float(self.rows) * 0.5
        lane_bias = self.bt_adapt_attack_lane_top_frac - self.bt_adapt_attack_lane_bot_frac
        preferred_lane_y_t = torch.full((self.B,), preferred_lane_y, device=self.device)
        preferred_lane_y_t = preferred_lane_y_t + lane_bias * float(self.rows) * 0.22

        emergency_collapse = (
            active
            & bb["blue_carry_any"]
            & (
                (self.bt_adapt_near_cap_ticks >= 2)
                | (bb["ec_to_home_dist"] < self._NEAR_CAP_DIST)
            )
        )
        repeat_lane = active & (self.bt_adapt_repeat_lane_streak >= self._REPEAT_LANE_STREAK)
        high_escort = active & (self.bt_adapt_escort_density >= self._HIGH_ESCORT_DENSITY)
        high_overcommit = active & (self.bt_adapt_overcommit >= self._HIGH_OVERCOMMIT)
        fast_blue = active & (self.bt_adapt_fast_conversion_count >= 1)

        bb.update(
            adaptive_active=active,
            adapt_preferred_lane_y=preferred_lane_y_t,
            adapt_emergency_collapse=emergency_collapse,
            adapt_repeat_lane=repeat_lane,
            adapt_high_escort=high_escort,
            adapt_high_overcommit=high_overcommit,
            adapt_fast_conversion=fast_blue,
            adapt_intercept_block_boost=torch.where(
                emergency_collapse | fast_blue,
                torch.full((self.B,), 0.18, device=self.device),
                torch.zeros((self.B,), device=self.device),
            ),
        )
        return bb

    def _bt_apply_adaptive_role_overrides(
        self,
        bb: dict,
        roles: torch.Tensor,
        prof: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Second-pass role shifts driven by adaptive memory."""
        if not bool(bb.get("adaptive_active", torch.tensor(False)).any().item()):
            return roles

        B, Nr = self.B, self.Nr
        device = self.device
        idx_env = bb["idx_env"]
        out = roles.clone()
        lock = self.bt_role_lock_ticks.clone()
        eligible = self.red_alive & (~self.red_tagged)

        # Emergency near-cap: pull a second agent into intercept when carrier is close.
        need_extra_int = bb["adapt_emergency_collapse"] & bb["blue_carry_any"] & prof["enable_intercept"]
        if need_extra_int.any():
            mid_x = bb["ec_x"] + (bb["blue_flag_home"][:, 0] - bb["ec_x"]) * 0.35
            mid_y = bb["ec_y"] + (bb["blue_flag_home"][:, 1] - bb["ec_y"]) * 0.35
            mid_dist = torch.sqrt(
                (self.red_x - mid_x[:, None]) ** 2 + (self.red_y - mid_y[:, None]) ** 2 + 1e-8
            )
            already_int = out == ROLE_INTERCEPTOR
            mid_dist = torch.where(eligible & (~already_int), mid_dist, mid_dist.new_full((), 1e9))
            second = torch.argmin(mid_dist, dim=1)
            for j in range(Nr):
                assign = need_extra_int & (second == j) & eligible[:, j]
                out[:, j] = torch.where(
                    assign,
                    torch.full((B,), ROLE_INTERCEPTOR, dtype=torch.int32, device=device),
                    out[:, j],
                )
                lock[:, j] = torch.where(assign, prof["lock_intercept"], lock[:, j])

        # OP12-style overcommit counter when blue leaves home open.
        is_op12ish = prof["is_op12"] | (prof["bt_level"] == 12)
        counter_push = (
            bb["adapt_high_overcommit"]
            & bb["own_flag_at_home"]
            & bb["blue_carry_any"]
            & prof["enable_counter"]
            & is_op12ish
        )
        if counter_push.any():
            efx = bb["blue_flag_pos"][:, 0:1]
            efy = bb["blue_flag_pos"][:, 1:2]
            ef_dist = torch.sqrt((self.red_x - efx) ** 2 + (self.red_y - efy) ** 2 + 1e-8)
            ef_dist = torch.where(eligible, ef_dist, ef_dist.new_full((), 1e9))
            ctr = torch.argmin(ef_dist, dim=1)
            for j in range(Nr):
                assign = counter_push & (ctr == j) & eligible[:, j]
                out[:, j] = torch.where(
                    assign,
                    torch.full((B,), ROLE_COUNTER, dtype=torch.int32, device=device),
                    out[:, j],
                )
                lock[:, j] = torch.where(assign, prof["lock_counter"], lock[:, j])

        # OP10 split pressure when blue stacks escort.
        escort_break = (
            bb["adapt_high_escort"]
            & bb["blue_carry_any"]
            & prof["enable_2v1"]
            & (prof["bt_level"] == 10)
        )
        if escort_break.any():
            dxx = self.red_x - bb["ec_x"][:, None]
            dyy = self.red_y - bb["ec_y"][:, None]
            flank_dist = torch.sqrt(dxx ** 2 + dyy ** 2 + 1e-8)
            flank_dist = torch.where(eligible & (out == ROLE_ATTACKER), flank_dist, flank_dist.new_full((), 1e9))
            wing = torch.argmax(flank_dist, dim=1)
            for j in range(Nr):
                assign = escort_break & (wing == j) & eligible[:, j]
                out[:, j] = torch.where(
                    assign,
                    torch.full((B,), ROLE_2V1_WING, dtype=torch.int32, device=device),
                    out[:, j],
                )
                lock[:, j] = torch.where(assign, prof["lock_2v1"], lock[:, j])

        active = bb["adaptive_active"][:, None]
        out = torch.where(active, out, roles)
        lock = torch.where(active, lock, self.bt_role_lock_ticks)
        self.bt_red_role = out
        self.bt_role_lock_ticks = lock
        return out

    def _bt_apply_adaptive_route_overrides(
        self,
        bb: dict,
        roles: torch.Tensor,
        tx: torch.Tensor,
        ty: torch.Tensor,
        prof: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Bias intercept/counter routes using predicted lanes and collapse boost."""
        if not bool(bb.get("adaptive_active", torch.tensor(False)).any().item()):
            return tx, ty

        B, Nr = self.B, self.Nr
        device = self.device
        max_x, max_y = bb["max_x"], bb["max_y"]

        block_boost = bb["adapt_intercept_block_boost"]
        block_frac = (
            prof["intercept_block_base"]
            + prof["intercept_block_trailing_bonus"] * bb["trailing"].float()
            + block_boost
        ).clamp(max=0.92)

        lane_y = bb["adapt_preferred_lane_y"]
        repeat = bb["adapt_repeat_lane"]
        is_lane_op = (prof["bt_level"] == 8) | (prof["bt_level"] == 11)

        for j in range(Nr):
            role_j = roles[:, j]
            int_mask = (role_j == ROLE_INTERCEPTOR) & bb["blue_carry_any"]
            bx = bb["ec_x"] + (bb["blue_flag_home"][:, 0] - bb["ec_x"]) * block_frac
            by = bb["ec_y"] + (bb["blue_flag_home"][:, 1] - bb["ec_y"]) * block_frac
            by = torch.where(
                repeat & is_lane_op,
                0.65 * by + 0.35 * lane_y,
                by,
            )
            by = torch.where(
                bb["adapt_emergency_collapse"],
                bb["ec_y"] + (bb["blue_flag_home"][:, 1] - bb["ec_y"]) * 0.15,
                by,
            )
            tx[:, j] = torch.where(int_mask, torch.clamp(bx, 0.0, max_x), tx[:, j])
            ty[:, j] = torch.where(int_mask, torch.clamp(by, 0.0, max_y), ty[:, j])

        return tx, ty


__all__ = ["_BTAdaptiveMixin"]
