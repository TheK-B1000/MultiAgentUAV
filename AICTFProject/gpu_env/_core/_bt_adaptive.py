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
ROLE_FLAG_RETR = 4
ROLE_COUNTER = 5
ROLE_2V1_WING = 6


class _BTAdaptiveMixin:
    """Episode-local adaptive blackboard for levels 8..12."""

    _ADAPTIVE_LEVEL_MIN = 8
    _LANE_EMA_ALPHA = 0.14
    _CARRIER_EMA_ALPHA = 0.20
    # v6i21D brutal denial calibration: upper-bound pressure test, not final balance.
    # v6i21E targeted balance: per-level overrides for OP8/OP10/OP11 only (OP9/OP12 unchanged).
    # v6i21F OP8 carrier denial: pure cap-lane / carrier-hunter monster (no counter-scoring).
    # v6i21G targeted easy-cell denial: restore cap-lane body-blocking for OP8/OP11,
    # and make OP10 cut off conversion instead of chasing the carrier.
    # v6i21H saturation fix: disable failed OP8/OP10/OP11 bespoke geometry and
    # let profile-level fortress/counter shapes do the work.
    # v6i21I OP8 extreme: physical upper bound for OP8 only.
    _NEAR_CAP_DIST = 12.0
    _OP8_NEAR_CAP_DIST_MULT = 1.45
    _OP8_PREDICTIVE_LEAD_FRAC = 0.32
    _OP8_INTERCEPT_BLOCK_EXTRA = 0.10
    _OP8_CAP_BODY_X_FRAC = 0.80
    _OP10_HIGH_ESCORT_DENSITY = 0.14
    _OP10_CARRIER_CUTOFF_BLEND = 0.24
    _OP8_CAP_LANE_BODY_FRAC = 0.95
    _OP11_CAP_LANE_BODY_FRAC = 0.85
    _OP8_DUAL_DENIAL_ENABLED = False
    _OP10_ESCORT_BREAK_ENABLED = False
    _OP11_REPEAT_INTERCEPT_ENABLED = False
    _FAST_CONVERSION_STEPS = 70
    _REPEAT_LANE_STREAK = 2
    _HIGH_ESCORT_DENSITY = 0.25
    _HIGH_OVERCOMMIT = 0.25
    _BLUE_CARRIER_SPEED_MULT = 0.75
    _OP8_BLUE_CARRIER_SPEED_MULT = 0.35
    _RED_RESPAWN_MULT = 0.50
    _OP8_RED_SPEED_MULT = 1.60
    _RED_INTERCEPTOR_NEAR_FLAG_BOOST = 1.35
    _OP8_RED_INTERCEPTOR_NEAR_FLAG_BOOST = 1.85
    _RED_INTERCEPTOR_NEAR_FLAG_DIST = 11.0
    _COLLAPSE_ROLE_LOCK_BONUS = 20
    _INTERCEPT_BLOCK_BOOST_COLLAPSE = 0.50
    _INTERCEPT_BLOCK_BOOST_FAST = 0.35
    _PREDICTIVE_LEAD_FRAC = 0.28
    _CAP_LANE_BODY_FRAC = 0.01
    _DUAL_FLAG_RETR_LOCK = 24
    _ADAPTIVE_HARDPOOL_KEYS = frozenset(
        {
            "OP8",
            "OP8_INTERCEPTOR",
            "OP9",
            "OP9_FORTRESS",
            "OP10",
            "OP10_ESCORT",
            "OP11",
            "OP11_BT_BALANCED",
            "OP12",
            "OP12_COUNTER",
        }
    )

    def _adaptive_hardpool_pressure_mask(self) -> torch.Tensor:
        return torch.as_tensor(
            [str(k).strip().upper() in self._ADAPTIVE_HARDPOOL_KEYS for k in self._opponent_key],
            dtype=torch.bool,
            device=self.device,
        )

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
        self.bt_adapt_prev_blue_carry_any = torch.zeros((B,), dtype=torch.bool, device=dev)
        self.bt_adapt_blue_carrier_lost = torch.zeros((B,), dtype=torch.bool, device=dev)

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
        self.bt_adapt_prev_blue_carry_any[idx] = False
        self.bt_adapt_blue_carrier_lost[idx] = False

    def _adaptive_active_mask(self, prof: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self._bt_opponent_mask() & prof["adaptive_enabled"]

    def _repeat_lane_streak_threshold(self, prof: Dict[str, torch.Tensor]) -> torch.Tensor:
        """OP8/OP11 punish lane repetition after one repeat; others keep global streak."""
        bt = prof["bt_level"]
        thresh = torch.full_like(bt, self._REPEAT_LANE_STREAK, dtype=torch.int32)
        return torch.where((bt == 8) | (bt == 11), torch.ones_like(thresh), thresh)

    def _high_escort_threshold(self, prof: Dict[str, torch.Tensor]) -> torch.Tensor:
        """OP10 escort-break fires earlier; other levels keep global density gate."""
        bt = prof["bt_level"].to(dtype=torch.float32, device=self.device)
        thresh = torch.full((self.B,), self._HIGH_ESCORT_DENSITY, dtype=torch.float32, device=self.device)
        return torch.where(bt == 10.0, torch.full_like(thresh, self._OP10_HIGH_ESCORT_DENSITY), thresh)

    def _near_cap_dist_threshold(self, prof: Dict[str, torch.Tensor]) -> torch.Tensor:
        """OP8 widens near-cap collapse zone; other levels keep global distance."""
        bt = prof["bt_level"].to(dtype=torch.float32, device=self.device)
        base = torch.full((self.B,), self._NEAR_CAP_DIST, dtype=torch.float32, device=self.device)
        return torch.where(bt == 8.0, base * self._OP8_NEAR_CAP_DIST_MULT, base)

    def _near_cap_ticks_threshold(self, prof: Dict[str, torch.Tensor]) -> torch.Tensor:
        """OP8/OP11 collapse on first near-cap tick; others require one accumulated tick."""
        bt = prof["bt_level"]
        thresh = torch.ones_like(bt, dtype=torch.int32)
        return torch.where((bt == 8) | (bt == 11), torch.zeros_like(thresh), thresh)

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

        over = blue_on_enemy.sum(dim=1).float() / blue_alive.sum(dim=1).float().clamp(min=1.0)
        self.bt_adapt_overcommit = torch.where(
            active,
            (1.0 - alpha_lane) * self.bt_adapt_overcommit + alpha_lane * over,
            self.bt_adapt_overcommit,
        )

        home_dx = self.blue_flag_home[:, 0] - ec_x
        home_dy = self.blue_flag_home[:, 1] - ec_y
        ec_home_dist = torch.sqrt(home_dx ** 2 + home_dy ** 2 + 1e-8)
        near_cap_dist = self._near_cap_dist_threshold(prof)
        near_cap = blue_carry_any & (ec_home_dist < near_cap_dist)
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

        blue_carrier_lost = active & self.bt_adapt_prev_blue_carry_any & (~blue_carry_any)
        self.bt_adapt_blue_carrier_lost = blue_carrier_lost
        self.bt_adapt_prev_blue_carry_any = torch.where(active, blue_carry_any, self.bt_adapt_prev_blue_carry_any)

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

        near_cap_dist = self._near_cap_dist_threshold(prof)
        near_cap_ticks_needed = self._near_cap_ticks_threshold(prof)
        emergency_collapse = (
            active
            & bb["blue_carry_any"]
            & (
                (self.bt_adapt_near_cap_ticks >= near_cap_ticks_needed)
                | (bb["ec_to_home_dist"] < near_cap_dist)
            )
        )
        repeat_lane = active & (
            self.bt_adapt_repeat_lane_streak >= self._repeat_lane_streak_threshold(prof)
        )
        high_escort = active & (self.bt_adapt_escort_density >= self._high_escort_threshold(prof))
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
            adapt_blue_carrier_lost=getattr(self, "bt_adapt_blue_carrier_lost", torch.zeros((self.B,), dtype=torch.bool, device=self.device)),
            adapt_intercept_block_boost=torch.where(
                emergency_collapse,
                torch.full((self.B,), self._INTERCEPT_BLOCK_BOOST_COLLAPSE, device=self.device),
                torch.where(
                    fast_blue,
                    torch.full((self.B,), self._INTERCEPT_BLOCK_BOOST_FAST, device=self.device),
                    torch.zeros((self.B,), device=self.device),
                ),
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
        active = bb["adaptive_active"]
        emergency_collapse = bb["adapt_emergency_collapse"]
        collapse_lock = prof["lock_intercept"] + self._COLLAPSE_ROLE_LOCK_BONUS

        # Dual flag retrieval when own flag is loose and blue just lost carrier pressure.
        need_dual_retr = (
            active
            & (~bb["own_flag_at_home"])
            & prof["enable_flag_retr"]
            & (bb.get("adapt_blue_carrier_lost", torch.zeros((B,), dtype=torch.bool, device=device)) | emergency_collapse)
        )
        if need_dual_retr.any():
            flag_dx = self.red_x - bb["red_flag_pos"][:, 0:1]
            flag_dy = self.red_y - bb["red_flag_pos"][:, 1:2]
            flag_dist = torch.sqrt(flag_dx ** 2 + flag_dy ** 2 + 1e-8)
            flag_dist_m = torch.where(eligible & need_dual_retr[:, None], flag_dist, flag_dist.new_full((), 1e9))
            order = torch.argsort(flag_dist_m, dim=1)
            for rank in range(min(2, Nr)):
                retr_j = order[:, rank]
                for j in range(Nr):
                    assign = need_dual_retr & (retr_j == j) & eligible[:, j]
                    out[:, j] = torch.where(
                        assign,
                        torch.full((B,), ROLE_FLAG_RETR, dtype=torch.int32, device=device),
                        out[:, j],
                    )
                    lock[:, j] = torch.where(assign, torch.full_like(lock[:, j], self._DUAL_FLAG_RETR_LOCK), lock[:, j])

        # Emergency near-cap: pull a second agent into intercept when carrier is close.
        counter_already_assigned = (out == ROLE_COUNTER).any(dim=1)
        need_extra_int = (
            emergency_collapse
            & bb["blue_carry_any"]
            & prof["enable_intercept"]
            & (~counter_already_assigned)
        )
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
                lock[:, j] = torch.where(assign, collapse_lock, lock[:, j])

        # OP12-style overcommit counter when blue leaves home open.
        is_op12ish = prof["is_op12"] | (prof["bt_level"] == 12)
        counter_push = (
            (bb["adapt_high_overcommit"] | bb.get("adapt_blue_carrier_lost", torch.zeros((B,), dtype=torch.bool, device=device)))
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

        # OP10 split pressure when blue stacks escort; also cut off the carrier.
        is_op10 = prof["bt_level"] == 10
        escort_break = (
            bool(self._OP10_ESCORT_BREAK_ENABLED)
            & bb["adapt_high_escort"]
            & bb["blue_carry_any"]
            & is_op10
        )
        if escort_break.any():
            if bool(prof["enable_2v1"].any().item()):
                dxx = self.red_x - bb["ec_x"][:, None]
                dyy = self.red_y - bb["ec_y"][:, None]
                flank_dist = torch.sqrt(dxx ** 2 + dyy ** 2 + 1e-8)
                flank_dist = torch.where(eligible & (out == ROLE_ATTACKER), flank_dist, flank_dist.new_full((), 1e9))
                wing = torch.argmax(flank_dist, dim=1)
                for j in range(Nr):
                    assign = escort_break & prof["enable_2v1"] & (wing == j) & eligible[:, j]
                    out[:, j] = torch.where(
                        assign,
                        torch.full((B,), ROLE_2V1_WING, dtype=torch.int32, device=device),
                        out[:, j],
                    )
                    lock[:, j] = torch.where(assign, prof["lock_2v1"], lock[:, j])
            if bool(prof["enable_intercept"].any().item()):
                cutoff_dist = torch.sqrt(
                    (self.red_x - bb["ec_x"][:, None]) ** 2 + (self.red_y - bb["ec_y"][:, None]) ** 2 + 1e-8
                )
                already_int = out == ROLE_INTERCEPTOR
                cutoff_dist = torch.where(eligible & (~already_int), cutoff_dist, cutoff_dist.new_full((), 1e9))
                cutter = torch.argmin(cutoff_dist, dim=1)
                cutoff_lock = prof["lock_intercept"] + self._COLLAPSE_ROLE_LOCK_BONUS
                for j in range(Nr):
                    assign = escort_break & prof["enable_intercept"] & (cutter == j) & eligible[:, j]
                    out[:, j] = torch.where(
                        assign,
                        torch.full((B,), ROLE_INTERCEPTOR, dtype=torch.int32, device=device),
                        out[:, j],
                    )
                    lock[:, j] = torch.where(assign, cutoff_lock, lock[:, j])

        # OP11 anti-repeat: second intercept when lane repetition is detected.
        is_op11 = prof["bt_level"] == 11
        repeat_intercept = (
            bool(self._OP11_REPEAT_INTERCEPT_ENABLED)
            & is_op11
            & bb["adapt_repeat_lane"]
            & bb["blue_carry_any"]
            & prof["enable_intercept"]
        )
        if repeat_intercept.any():
            lane_y = bb["adapt_preferred_lane_y"]
            lane_dist = torch.sqrt(
                (self.red_x - bb["ec_x"][:, None]) ** 2 + (self.red_y - lane_y[:, None]) ** 2 + 1e-8
            )
            already_int = out == ROLE_INTERCEPTOR
            lane_dist = torch.where(eligible & (~already_int), lane_dist, lane_dist.new_full((), 1e9))
            lane_int = torch.argmin(lane_dist, dim=1)
            lane_lock = prof["lock_intercept"] + self._COLLAPSE_ROLE_LOCK_BONUS
            for j in range(Nr):
                assign = repeat_intercept & (lane_int == j) & eligible[:, j]
                out[:, j] = torch.where(
                    assign,
                    torch.full((B,), ROLE_INTERCEPTOR, dtype=torch.int32, device=device),
                    out[:, j],
                )
                lock[:, j] = torch.where(assign, lane_lock, lock[:, j])

        # OP8 pure carrier denial: both agents intercept; no counter-scoring while blue carries.
        is_op8 = prof["bt_level"] == 8
        op8_denial = (
            bool(self._OP8_DUAL_DENIAL_ENABLED)
            & is_op8
            & bb["blue_carry_any"]
            & prof["enable_intercept"]
        )
        if op8_denial.any():
            home_x = bb["blue_flag_home"][:, 0]
            home_y = bb["blue_flag_home"][:, 1]
            ec_x = bb["ec_x"]
            ec_y = bb["ec_y"]
            lead = self._OP8_PREDICTIVE_LEAD_FRAC
            pred_x = ec_x + (home_x - ec_x) * lead
            pred_y = ec_y + (home_y - ec_y) * lead
            body_x = ec_x + (home_x - ec_x) * self._OP8_CAP_BODY_X_FRAC
            body_y = home_y
            denial_lock = prof["lock_intercept"] + self._COLLAPSE_ROLE_LOCK_BONUS
            denial_eligible = eligible.clone()
            for target_x, target_y in ((pred_x, pred_y), (body_x, body_y)):
                dist = torch.sqrt(
                    (self.red_x - target_x[:, None]) ** 2 + (self.red_y - target_y[:, None]) ** 2 + 1e-8
                )
                dist = torch.where(denial_eligible, dist, dist.new_full((), 1e9))
                agent = torch.argmin(dist, dim=1)
                for j in range(Nr):
                    assign = op8_denial & (agent == j) & denial_eligible[:, j]
                    out[:, j] = torch.where(
                        assign,
                        torch.full((B,), ROLE_INTERCEPTOR, dtype=torch.int32, device=device),
                        out[:, j],
                    )
                    lock[:, j] = torch.where(assign, denial_lock, lock[:, j])
                    denial_eligible[:, j] = denial_eligible[:, j] & (~assign)

        active_mask = active[:, None]
        out = torch.where(active_mask, out, roles)
        lock = torch.where(active_mask, lock, self.bt_role_lock_ticks)
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
        ).clamp(max=0.98)
        bt_level = prof["bt_level"]
        is_op8 = bt_level == 8
        block_frac = block_frac + torch.where(is_op8, self._OP8_INTERCEPT_BLOCK_EXTRA, 0.0)
        block_frac = block_frac.clamp(max=0.98)
        lane_y = bb["adapt_preferred_lane_y"]
        repeat = bb["adapt_repeat_lane"]
        is_lane_op = (bt_level == 8) | (bt_level == 11)
        is_op10 = bt_level == 10
        is_op11 = bt_level == 11
        cap_lane_body = torch.where(
            is_op8,
            torch.full((B,), self._OP8_CAP_LANE_BODY_FRAC, device=device),
            torch.where(
                is_op11,
                torch.full((B,), self._OP11_CAP_LANE_BODY_FRAC, device=device),
                torch.full((B,), self._CAP_LANE_BODY_FRAC, device=device),
            ),
        )

        for j in range(Nr):
            role_j = roles[:, j]
            int_mask = (role_j == ROLE_INTERCEPTOR) & bb["blue_carry_any"]
            home_x = bb["blue_flag_home"][:, 0]
            home_y = bb["blue_flag_home"][:, 1]
            ec_x = bb["ec_x"]
            ec_y = bb["ec_y"]
            lead_t = torch.where(
                is_op8,
                torch.full((B,), self._OP8_PREDICTIVE_LEAD_FRAC, device=device),
                torch.full((B,), self._PREDICTIVE_LEAD_FRAC, device=device),
            )
            pred_x = ec_x + (home_x - ec_x) * lead_t
            pred_y = ec_y + (home_y - ec_y) * lead_t
            bx = pred_x + (home_x - pred_x) * block_frac
            by = pred_y + (home_y - pred_y) * block_frac
            cutoff_blend = torch.full((B,), self._OP10_CARRIER_CUTOFF_BLEND, device=device)
            bx = torch.where(
                is_op10 & int_mask,
                (1.0 - cutoff_blend) * bx + cutoff_blend * ec_x,
                bx,
            )
            by = torch.where(
                is_op10 & int_mask,
                (1.0 - cutoff_blend) * by + cutoff_blend * ec_y,
                by,
            )
            op8_int = is_op8 & int_mask
            bx = torch.where(op8_int & (j == 0), pred_x + (home_x - pred_x) * block_frac, bx)
            by = torch.where(op8_int & (j == 0), pred_y + (home_y - pred_y) * block_frac, by)
            body_bx = ec_x + (home_x - ec_x) * self._OP8_CAP_BODY_X_FRAC
            bx = torch.where(op8_int & (j == 1), body_bx, bx)
            by = torch.where(op8_int & (j == 1), home_y, by)
            by = torch.where(
                repeat & is_lane_op,
                torch.where(
                    is_op11,
                    0.35 * by + 0.65 * lane_y,
                    0.55 * by + 0.45 * lane_y,
                ),
                by,
            )
            by = torch.where(
                bb["adapt_emergency_collapse"],
                bb["ec_y"] + (home_y - bb["ec_y"]) * cap_lane_body,
                by,
            )
            tx[:, j] = torch.where(int_mask, torch.clamp(bx, 0.0, max_x), tx[:, j])
            ty[:, j] = torch.where(int_mask, torch.clamp(by, 0.0, max_y), ty[:, j])

        return tx, ty


__all__ = ["_BTAdaptiveMixin"]
