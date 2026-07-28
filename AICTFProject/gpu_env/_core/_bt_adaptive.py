"""Intra-episode adaptive memory and counter-play for strategic BT niches.

OP11 and OP12 use episode-local counters that track blue attack patterns and
shift red roles/routes to punish repetition. Memory resets every episode; no
cross-episode state is stored.
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
    # v6i21J balance: keep OP8 hard and add OP10/OP11 physical pressure.
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
    _OP12_SPLIT_RESPONSE_DURATION = 40
    _HIGH_ESCORT_DENSITY = 0.25
    _HIGH_OVERCOMMIT = 0.25
    # Strategic niches use behavior gates, not per-family physical handicaps.
    _BLUE_CARRIER_SPEED_MULT = 1.00
    _OP8_BLUE_CARRIER_SPEED_MULT = 1.00
    _OP10_BLUE_CARRIER_SPEED_MULT = 1.00
    _OP11_BLUE_CARRIER_SPEED_MULT = 1.00
    _RED_RESPAWN_MULT = 1.00
    _OP8_RED_SPEED_MULT = 1.00
    _OP10_RED_SPEED_MULT = 1.00
    _OP11_RED_SPEED_MULT = 1.00
    _RED_INTERCEPTOR_NEAR_FLAG_BOOST = 1.00
    _OP8_RED_INTERCEPTOR_NEAR_FLAG_BOOST = 1.00
    _OP10_RED_INTERCEPTOR_NEAR_FLAG_BOOST = 1.00
    _OP11_RED_INTERCEPTOR_NEAR_FLAG_BOOST = 1.00
    _RED_INTERCEPTOR_NEAR_FLAG_DIST = 11.0
    _COLLAPSE_ROLE_LOCK_BONUS = 20
    _INTERCEPT_BLOCK_BOOST_COLLAPSE = 0.50
    _INTERCEPT_BLOCK_BOOST_FAST = 0.35
    _PREDICTIVE_LEAD_FRAC = 0.28
    _CAP_LANE_BODY_FRAC = 0.01
    _DUAL_FLAG_RETR_LOCK = 24
    _ADAPTIVE_HARDPOOL_KEYS = frozenset(
        {
            # Short aliases
            "OP6",
            "OP7",
            "OP8",
            "OP9",
            "OP10",
            "OP11",
            "OP12",
            # Audited long names + historical synonyms
            "OP6_TURTLE",
            "OP6_IMMEDIATE_DUAL_RUSH",
            "OP7_SWITCHER",
            "OP7_FORTRESS",
            "OP7_DEEP_FORTRESS",
            "OP8_INTERCEPTOR",
            "OP8_ESCORT",
            "OP8_PROTECTED_CARRIER_ESCORT",
            "OP9_FEINT",
            "OP9_FORTRESS",
            "OP9_SPLIT_LANE_FEINT",
            "OP10_ESCORT",
            "OP10_INTERCEPTOR",
            "OP10_AGGRESSIVE_INTERCEPTOR",
            "OP11_EXPLOITER",
            "OP11_BT_BALANCED",
            "OP11_ADAPTIVE_EXPLOITER",
            "OP12_COUNTER",
            "OP12_CONVERTER",
            "OP12_LATE_CONVERTER",
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
        self.bt_adapt_split_pressure_ticks = torch.zeros((B,), dtype=i32, device=dev)
        self.bt_adapt_split_first_trigger_step = torch.full((B,), -1, dtype=i32, device=dev)
        self.bt_adapt_split_response_expiry_step = torch.full((B,), -1, dtype=i32, device=dev)
        self.bt_adapt_split_active_steps = torch.zeros((B,), dtype=i32, device=dev)
        self.bt_adapt_split_max_lateral_sep = torch.zeros((B,), dtype=f32, device=dev)
        self.bt_adapt_split_max_teammate_dist = torch.zeros((B,), dtype=f32, device=dev)
        self.bt_adapt_opening_escort_ticks = torch.zeros((B,), dtype=i32, device=dev)
        self.bt_adapt_opening_escort_first_trigger_step = torch.full((B,), -1, dtype=i32, device=dev)
        self.bt_adapt_opening_escort_active_steps = torch.zeros((B,), dtype=i32, device=dev)
        self.bt_adapt_prev_blue_x = torch.zeros((B, self.Nb), dtype=f32, device=dev)
        self.bt_adapt_prev_blue_y = torch.zeros((B, self.Nb), dtype=f32, device=dev)
        self.bt_adapt_prev_blue_valid = torch.zeros((B,), dtype=torch.bool, device=dev)
        self.bt_adapt_opening_escort_score = torch.zeros((B,), dtype=f32, device=dev)
        self.bt_adapt_opening_escort_compact = torch.zeros((B,), dtype=f32, device=dev)
        self.bt_adapt_opening_escort_narrow = torch.zeros((B,), dtype=f32, device=dev)
        self.bt_adapt_opening_escort_leader = torch.zeros((B,), dtype=f32, device=dev)
        self.bt_adapt_opening_escort_heading = torch.zeros((B,), dtype=f32, device=dev)
        self.bt_adapt_opening_escort_speed_penalty = torch.zeros((B,), dtype=f32, device=dev)
        self.bt_adapt_opening_escort_leader_sign = torch.zeros((B,), dtype=i32, device=dev)
        self.bt_adapt_opening_escort_leader_streak = torch.zeros((B,), dtype=i32, device=dev)
        self.bt_adapt_convoy_offensive_active = torch.zeros((B,), dtype=torch.bool, device=dev)
        self.bt_adapt_convoy_corridor_active = torch.zeros((B,), dtype=torch.bool, device=dev)
        self.bt_adapt_convoy_leader_active = torch.zeros((B,), dtype=torch.bool, device=dev)
        self.bt_adapt_convoy_reject_rush = torch.zeros((B,), dtype=torch.bool, device=dev)
        self.bt_adapt_convoy_leader_id = torch.full((B,), -1, dtype=i32, device=dev)
        self.bt_adapt_convoy_evidence_ticks = torch.zeros((B,), dtype=i32, device=dev)
        self.bt_adapt_convoy_longitudinal_gap = torch.zeros((B,), dtype=f32, device=dev)
        self.bt_adapt_convoy_prev_longitudinal_gap = torch.zeros((B,), dtype=f32, device=dev)
        self.bt_adapt_convoy_gap_stable = torch.zeros((B,), dtype=torch.bool, device=dev)
        self.bt_adapt_convoy_lateral_gap = torch.zeros((B,), dtype=f32, device=dev)
        self.bt_adapt_convoy_heading_similarity = torch.zeros((B,), dtype=f32, device=dev)
        self.bt_adapt_convoy_centroid_forward_speed = torch.zeros((B,), dtype=f32, device=dev)
        self.bt_adapt_prev_centroid_flag_dist = torch.zeros((B,), dtype=f32, device=dev)
        self.bt_adapt_prev_centroid_valid = torch.zeros((B,), dtype=torch.bool, device=dev)
        self.bt_adapt_escort_confirm_ticks = torch.zeros((B,), dtype=i32, device=dev)
        self.bt_adapt_escort_confirm_first_step = torch.full((B,), -1, dtype=i32, device=dev)
        self.bt_adapt_escort_confirm_active_steps = torch.zeros((B,), dtype=i32, device=dev)
        self.bt_adapt_escort_confirm_carrier_id = torch.full((B,), -1, dtype=i32, device=dev)
        self.bt_adapt_escort_confirm_protector_id = torch.full((B,), -1, dtype=i32, device=dev)
        self.bt_adapt_escort_confirm_distance = torch.zeros((B,), dtype=f32, device=dev)
        self.bt_adapt_escort_confirm_same_corridor_steps = torch.zeros((B,), dtype=i32, device=dev)
        self.bt_adapt_escort_confirm_to_end_steps = torch.zeros((B,), dtype=i32, device=dev)
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
        self.bt_adapt_split_pressure_ticks[idx] = 0
        self.bt_adapt_split_first_trigger_step[idx] = -1
        self.bt_adapt_split_response_expiry_step[idx] = -1
        self.bt_adapt_split_active_steps[idx] = 0
        self.bt_adapt_split_max_lateral_sep[idx] = 0.0
        self.bt_adapt_split_max_teammate_dist[idx] = 0.0
        self.bt_adapt_opening_escort_ticks[idx] = 0
        self.bt_adapt_opening_escort_first_trigger_step[idx] = -1
        self.bt_adapt_opening_escort_active_steps[idx] = 0
        self.bt_adapt_prev_blue_x[idx] = 0.0
        self.bt_adapt_prev_blue_y[idx] = 0.0
        self.bt_adapt_prev_blue_valid[idx] = False
        self.bt_adapt_opening_escort_score[idx] = 0.0
        self.bt_adapt_opening_escort_compact[idx] = 0.0
        self.bt_adapt_opening_escort_narrow[idx] = 0.0
        self.bt_adapt_opening_escort_leader[idx] = 0.0
        self.bt_adapt_opening_escort_heading[idx] = 0.0
        self.bt_adapt_opening_escort_speed_penalty[idx] = 0.0
        self.bt_adapt_opening_escort_leader_sign[idx] = 0
        self.bt_adapt_opening_escort_leader_streak[idx] = 0
        self.bt_adapt_convoy_offensive_active[idx] = False
        self.bt_adapt_convoy_corridor_active[idx] = False
        self.bt_adapt_convoy_leader_active[idx] = False
        self.bt_adapt_convoy_reject_rush[idx] = False
        self.bt_adapt_convoy_leader_id[idx] = -1
        self.bt_adapt_convoy_evidence_ticks[idx] = 0
        self.bt_adapt_convoy_longitudinal_gap[idx] = 0.0
        self.bt_adapt_convoy_prev_longitudinal_gap[idx] = 0.0
        self.bt_adapt_convoy_gap_stable[idx] = False
        self.bt_adapt_convoy_lateral_gap[idx] = 0.0
        self.bt_adapt_convoy_heading_similarity[idx] = 0.0
        self.bt_adapt_convoy_centroid_forward_speed[idx] = 0.0
        self.bt_adapt_prev_centroid_flag_dist[idx] = 0.0
        self.bt_adapt_prev_centroid_valid[idx] = False
        self.bt_adapt_escort_confirm_ticks[idx] = 0
        self.bt_adapt_escort_confirm_first_step[idx] = -1
        self.bt_adapt_escort_confirm_active_steps[idx] = 0
        self.bt_adapt_escort_confirm_carrier_id[idx] = -1
        self.bt_adapt_escort_confirm_protector_id[idx] = -1
        self.bt_adapt_escort_confirm_distance[idx] = 0.0
        self.bt_adapt_escort_confirm_same_corridor_steps[idx] = 0
        self.bt_adapt_escort_confirm_to_end_steps[idx] = 0
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

    def _split_pressure_ticks_threshold(self, prof: Dict[str, torch.Tensor]) -> torch.Tensor:
        """OP12 reacts after sustained simultaneous two-lane pressure. OP11's
        split-lane isolation (_bt_adaptive.py's op11_split_isolate branches)
        latches for the rest of the episode once triggered, so it should
        trigger EARLIER than OP12's reactive, non-latching response --
        lower threshold only for bt_level==11, OP12's value is untouched."""
        bt = prof["bt_level"]
        thresh = torch.full_like(bt, 4, dtype=torch.int32)
        thresh = torch.where(bt == 12, torch.full_like(thresh, 4), thresh)
        thresh = torch.where(bt == 11, torch.full_like(thresh, 2), thresh)
        return thresh

    def _opening_escort_ticks_threshold(self, prof: Dict[str, torch.Tensor]) -> torch.Tensor:
        """OP12 recognizes sustained pre-pickup lead/support pressure."""
        bt = prof["bt_level"]
        thresh = torch.full_like(bt, 2, dtype=torch.int32)
        return torch.where(bt == 12, torch.full_like(thresh, 2), thresh)

    def _escort_confirm_ticks_threshold(self, prof: Dict[str, torch.Tensor]) -> torch.Tensor:
        """OP12 confirms ESCORT only after sustained post-pickup support."""
        bt = prof["bt_level"]
        thresh = torch.full_like(bt, 5, dtype=torch.int32)
        return torch.where(bt == 12, torch.full_like(thresh, 5), thresh)

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

            if self.Nb >= 2:
                is_op12 = prof["bt_level"] == 12
                other_idx = torch.where(blue_ci == 0, torch.ones_like(blue_ci), torch.zeros_like(blue_ci))
                protector_alive = blue_alive[idx_env, other_idx]
                carrier_x = self.blue_x[idx_env, blue_ci]
                carrier_y = self.blue_y[idx_env, blue_ci]
                protector_x = self.blue_x[idx_env, other_idx]
                protector_y = self.blue_y[idx_env, other_idx]
                cp_dx = protector_x - carrier_x
                cp_dy = protector_y - carrier_y
                cp_dist = torch.sqrt(cp_dx ** 2 + cp_dy ** 2 + 1e-8)
                carrier_step_x = carrier_x - self.bt_adapt_prev_blue_x[idx_env, blue_ci]
                carrier_step_y = carrier_y - self.bt_adapt_prev_blue_y[idx_env, blue_ci]
                protector_step_x = protector_x - self.bt_adapt_prev_blue_x[idx_env, other_idx]
                protector_step_y = protector_y - self.bt_adapt_prev_blue_y[idx_env, other_idx]
                carrier_speed = torch.sqrt(carrier_step_x ** 2 + carrier_step_y ** 2 + 1e-8)
                protector_speed = torch.sqrt(protector_step_x ** 2 + protector_step_y ** 2 + 1e-8)
                heading_dot = carrier_step_x * protector_step_x + carrier_step_y * protector_step_y
                heading_norm = torch.sqrt((carrier_speed * protector_speed) ** 2 + 1e-8)
                heading_sim = torch.clamp((heading_dot / heading_norm + 1.0) * 0.5, 0.0, 1.0)
                support_distance = (cp_dist >= 0.75) & (cp_dist <= 3.0)
                same_corridor = torch.abs(cp_dy) <= 3.0
                confirmed_now = (
                    active
                    & is_op12
                    & blue_carry_any
                    & protector_alive
                    & self.bt_adapt_prev_blue_valid
                    & support_distance
                    & same_corridor
                )
                confirm_ticks = torch.where(
                    confirmed_now,
                    self.bt_adapt_escort_confirm_ticks + 1,
                    torch.zeros_like(self.bt_adapt_escort_confirm_ticks),
                )
                self.bt_adapt_escort_confirm_ticks = torch.where(is_op12, confirm_ticks, self.bt_adapt_escort_confirm_ticks)
                confirm_active = is_op12 & (
                    self.bt_adapt_escort_confirm_ticks >= self._escort_confirm_ticks_threshold(prof)
                )
                first_confirm = confirm_active & (self.bt_adapt_escort_confirm_first_step < 0)
                self.bt_adapt_escort_confirm_first_step = torch.where(
                    first_confirm,
                    self.sim_step_count.to(torch.int32),
                    self.bt_adapt_escort_confirm_first_step,
                )
                self.bt_adapt_escort_confirm_active_steps += confirm_active.to(torch.int32)
                self.bt_adapt_escort_confirm_carrier_id = torch.where(
                    is_op12 & blue_carry_any,
                    blue_ci.to(torch.int32),
                    self.bt_adapt_escort_confirm_carrier_id,
                )
                self.bt_adapt_escort_confirm_protector_id = torch.where(
                    is_op12 & blue_carry_any,
                    other_idx.to(torch.int32),
                    self.bt_adapt_escort_confirm_protector_id,
                )
                self.bt_adapt_escort_confirm_distance = torch.where(
                    is_op12 & blue_carry_any,
                    cp_dist,
                    self.bt_adapt_escort_confirm_distance,
                )
                same_corridor_ticks = torch.where(
                    is_op12 & same_corridor,
                    self.bt_adapt_escort_confirm_same_corridor_steps + 1,
                    torch.zeros_like(self.bt_adapt_escort_confirm_same_corridor_steps),
                )
                self.bt_adapt_escort_confirm_same_corridor_steps = torch.where(
                    is_op12,
                    same_corridor_ticks,
                    self.bt_adapt_escort_confirm_same_corridor_steps,
                )
                self.bt_adapt_escort_confirm_to_end_steps = torch.where(
                    first_confirm,
                    torch.clamp(
                        torch.full_like(
                            self.sim_step_count.to(torch.int32),
                            int(getattr(self, "max_steps", 0)),
                        )
                        - self.sim_step_count.to(torch.int32),
                        min=0,
                    ),
                    self.bt_adapt_escort_confirm_to_end_steps,
                )

        over = blue_on_enemy.sum(dim=1).float() / blue_alive.sum(dim=1).float().clamp(min=1.0)
        self.bt_adapt_overcommit = torch.where(
            active,
            (1.0 - alpha_lane) * self.bt_adapt_overcommit + alpha_lane * over,
            self.bt_adapt_overcommit,
        )

        if self.Nb >= 2:
            offensive_buffer = 3.0
            blue0_off = blue_alive[:, 0] & (self.blue_x[:, 0] > midline - offensive_buffer)
            blue1_off = blue_alive[:, 1] & (self.blue_x[:, 1] > midline - offensive_buffer)
            lateral_sep = torch.abs(self.blue_y[:, 0] - self.blue_y[:, 1])
            dx01 = self.blue_x[:, 0] - self.blue_x[:, 1]
            dy01 = self.blue_y[:, 0] - self.blue_y[:, 1]
            teammate_dist = torch.sqrt(dx01 ** 2 + dy01 ** 2 + 1e-8)
            self.bt_adapt_split_max_lateral_sep = torch.where(
                active,
                torch.maximum(self.bt_adapt_split_max_lateral_sep, lateral_sep),
                self.bt_adapt_split_max_lateral_sep,
            )
            self.bt_adapt_split_max_teammate_dist = torch.where(
                active,
                torch.maximum(self.bt_adapt_split_max_teammate_dist, teammate_dist),
                self.bt_adapt_split_max_teammate_dist,
            )
            opposite_lanes = (
                ((self.blue_y[:, 0] > center_y) & (self.blue_y[:, 1] <= center_y))
                | ((self.blue_y[:, 1] > center_y) & (self.blue_y[:, 0] <= center_y))
            )
            split_pressure_now = (
                blue0_off
                & blue1_off
                & opposite_lanes
                & (lateral_sep >= float(self.rows) * 0.55)
                & (teammate_dist >= 12.0)
            )
            opening_phase = self.sim_step_count.to(torch.int32) < 20
            no_pickup_yet = ~blue_carry_any
            lead_x = torch.maximum(self.blue_x[:, 0], self.blue_x[:, 1])
            trail_x = torch.minimum(self.blue_x[:, 0], self.blue_x[:, 1])
            lead_forward = lead_x > midline - 3.0
            trailer_committed = trail_x > midline - 12.0
            lead_follow = torch.abs(dx01) >= 0.5
            mean_forward_delta = (self.blue_x[:, :2] - self.bt_adapt_prev_blue_x[:, :2]).mean(dim=1)
            slow_support_push = self.bt_adapt_prev_blue_valid & (mean_forward_delta <= 0.65)
            opening_escort_now = (
                (prof["bt_level"] == 12)
                & opening_phase
                & no_pickup_yet
                & lead_forward
                & trailer_committed
                & (teammate_dist <= 4.0)
                & (lateral_sep <= 2.25)
                & lead_follow
                & slow_support_push
            )
        else:
            split_pressure_now = torch.zeros((B,), dtype=torch.bool, device=device)
            opening_escort_now = torch.zeros((B,), dtype=torch.bool, device=device)
        split_ticks = torch.where(
            active & split_pressure_now,
            self.bt_adapt_split_pressure_ticks + 1,
            torch.zeros_like(self.bt_adapt_split_pressure_ticks),
        )
        self.bt_adapt_split_pressure_ticks = torch.where(
            active,
            split_ticks,
            self.bt_adapt_split_pressure_ticks,
        )
        split_pressure_active = active & (
            self.bt_adapt_split_pressure_ticks >= self._split_pressure_ticks_threshold(prof)
        )
        new_split_trigger = split_pressure_active & (self.bt_adapt_split_first_trigger_step < 0)
        self.bt_adapt_split_first_trigger_step = torch.where(
            new_split_trigger,
            self.sim_step_count.to(torch.int32),
            self.bt_adapt_split_first_trigger_step,
        )
        self.bt_adapt_split_active_steps += split_pressure_active.to(torch.int32)

        # OP12 Stage-2 split response (2026-07-28): a BOUNDED state, not a
        # per-step nudge -- every step that split_pressure_active re-fires
        # (already debounced: split_pressure_ticks must clear the threshold
        # via SEVERAL consecutive qualifying steps, this is the "persistence"
        # requirement), the response window's expiry is pushed out to
        # sim_step_count + duration. If the pattern stops, the window decays
        # on its own duration steps later rather than either latching
        # forever (OP11's choice) or flickering with the raw single-step
        # signal (the bug fixed for OP11 and now avoided here too).
        self.bt_adapt_split_response_expiry_step = torch.where(
            active & split_pressure_active,
            self.sim_step_count.to(torch.int32) + self._OP12_SPLIT_RESPONSE_DURATION,
            self.bt_adapt_split_response_expiry_step,
        )

        opening_escort_ticks = self.bt_adapt_opening_escort_ticks
        opening_escort_active = active & (
            self.bt_adapt_opening_escort_ticks >= self._opening_escort_ticks_threshold(prof)
        )
        new_escort_trigger = opening_escort_active & (self.bt_adapt_opening_escort_first_trigger_step < 0)
        self.bt_adapt_opening_escort_first_trigger_step = torch.where(
            new_escort_trigger,
            self.sim_step_count.to(torch.int32),
            self.bt_adapt_opening_escort_first_trigger_step,
        )
        self.bt_adapt_opening_escort_active_steps += opening_escort_active.to(torch.int32)
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
        split_pressure = active & (
            self.bt_adapt_split_pressure_ticks >= self._split_pressure_ticks_threshold(prof)
        )
        opening_escort = active & (
            self.bt_adapt_opening_escort_ticks >= self._opening_escort_ticks_threshold(prof)
        )
        confirmed_escort = active & (
            self.bt_adapt_escort_confirm_ticks >= self._escort_confirm_ticks_threshold(prof)
        )
        fast_blue = active & (self.bt_adapt_fast_conversion_count >= 1)
        is_op12_bb = prof["bt_level"] == 12
        op12_split_response_active = is_op12_bb & (
            self.bt_adapt_split_response_expiry_step >= self.sim_step_count.to(torch.int32)
        )
        bb.update(
            adaptive_active=active,
            adapt_preferred_lane_y=preferred_lane_y_t,
            adapt_emergency_collapse=emergency_collapse,
            adapt_repeat_lane=repeat_lane,
            adapt_high_escort=high_escort,
            adapt_opening_escort=opening_escort,
            adapt_confirmed_escort=confirmed_escort,
            adapt_high_overcommit=high_overcommit,
            adapt_split_pressure=split_pressure,
            adapt_op12_split_response_active=op12_split_response_active,
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
        op12_opening = (
            (prof["bt_level"] == 12)
            & (self.sim_step_count.to(torch.int32) < 20)
            & (~bb.get("adapt_split_pressure", torch.zeros((B,), dtype=torch.bool, device=device)))
        )
        late_or_not_op12 = ~op12_opening
        emergency_collapse = bb["adapt_emergency_collapse"] & late_or_not_op12
        collapse_lock = prof["lock_intercept"] + self._COLLAPSE_ROLE_LOCK_BONUS

        # Dual flag retrieval when own flag is loose and blue just lost carrier pressure.
        need_dual_retr = (
            active
            & (~bb["own_flag_at_home"])
            & prof["enable_flag_retr"]
            & late_or_not_op12
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
            & late_or_not_op12
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

        live_opening_escort = torch.zeros((B,), dtype=torch.bool, device=device)
        # Role selection sees the current formation used for this target
        # update; keep this path close/narrow so raw RUSH is not punished.
        if self.Nb >= 2:
            midline = float(self.cols) * 0.5
            lead_x = torch.maximum(self.blue_x[:, 0], self.blue_x[:, 1])
            trail_x = torch.minimum(self.blue_x[:, 0], self.blue_x[:, 1])
            dx01 = self.blue_x[:, 0] - self.blue_x[:, 1]
            dy01 = self.blue_y[:, 0] - self.blue_y[:, 1]
            lateral_sep = torch.abs(dy01)
            teammate_dist = torch.sqrt(dx01 ** 2 + dy01 ** 2 + 1e-8)
            blue_alive = self.blue_alive & (~self.blue_tagged)
            step_dx = self.blue_x[:, :2] - self.bt_adapt_prev_blue_x[:, :2]
            step_dy = self.blue_y[:, :2] - self.bt_adapt_prev_blue_y[:, :2]
            speed0 = torch.sqrt(step_dx[:, 0] ** 2 + step_dy[:, 0] ** 2 + 1e-8)
            speed1 = torch.sqrt(step_dx[:, 1] ** 2 + step_dy[:, 1] ** 2 + 1e-8)
            avg_forward = step_dx.mean(dim=1)
            centroid_x = self.blue_x[:, :2].mean(dim=1)
            centroid_y = self.blue_y[:, :2].mean(dim=1)
            red_flag_dx = self.red_flag_pos[:, 0] - centroid_x
            red_flag_dy = self.red_flag_pos[:, 1] - centroid_y
            centroid_flag_dist = torch.sqrt(red_flag_dx ** 2 + red_flag_dy ** 2 + 1e-8)
            centroid_speed = torch.sqrt((step_dx.mean(dim=1)) ** 2 + (step_dy.mean(dim=1)) ** 2 + 1e-8)
            heading_dot = step_dx[:, 0] * step_dx[:, 1] + step_dy[:, 0] * step_dy[:, 1]
            heading_norm = torch.sqrt((speed0 * speed1) ** 2 + 1e-8)
            heading_sim = torch.clamp((heading_dot / heading_norm + 1.0) * 0.5, 0.0, 1.0)
            leader_sign = torch.where(
                dx01 > 0.5,
                torch.ones((B,), dtype=torch.int32, device=device),
                torch.where(dx01 < -0.5, -torch.ones((B,), dtype=torch.int32, device=device), torch.zeros((B,), dtype=torch.int32, device=device)),
            )
            same_leader = (leader_sign == self.bt_adapt_opening_escort_leader_sign) & (leader_sign != 0)
            leader_streak = torch.where(
                same_leader,
                self.bt_adapt_opening_escort_leader_streak + 1,
                torch.where(leader_sign != 0, torch.ones_like(self.bt_adapt_opening_escort_leader_streak), torch.zeros_like(self.bt_adapt_opening_escort_leader_streak)),
            )
            self.bt_adapt_opening_escort_leader_streak = torch.where(is_op12ish, leader_streak, self.bt_adapt_opening_escort_leader_streak)
            self.bt_adapt_opening_escort_leader_sign = torch.where(is_op12ish, leader_sign, self.bt_adapt_opening_escort_leader_sign)

            compact = torch.clamp((5.5 - teammate_dist) / 5.5, 0.0, 1.0)
            narrow = torch.clamp((3.5 - lateral_sep) / 3.5, 0.0, 1.0)
            leader_component = torch.clamp(self.bt_adapt_opening_escort_leader_streak.to(torch.float32) / 4.0, 0.0, 1.0)
            speed_penalty = torch.clamp((avg_forward - 0.58) / 0.25, 0.0, 1.0)
            escort_score = compact + narrow + leader_component + heading_sim - speed_penalty
            self.bt_adapt_opening_escort_score = torch.where(is_op12ish, escort_score, self.bt_adapt_opening_escort_score)
            self.bt_adapt_opening_escort_compact = torch.where(is_op12ish, compact, self.bt_adapt_opening_escort_compact)
            self.bt_adapt_opening_escort_narrow = torch.where(is_op12ish, narrow, self.bt_adapt_opening_escort_narrow)
            self.bt_adapt_opening_escort_leader = torch.where(is_op12ish, leader_component, self.bt_adapt_opening_escort_leader)
            self.bt_adapt_opening_escort_heading = torch.where(is_op12ish, heading_sim, self.bt_adapt_opening_escort_heading)
            self.bt_adapt_opening_escort_speed_penalty = torch.where(is_op12ish, speed_penalty, self.bt_adapt_opening_escort_speed_penalty)
            distance_decreasing = self.bt_adapt_prev_centroid_valid & (
                centroid_flag_dist < self.bt_adapt_prev_centroid_flag_dist - 0.05
            )
            offensive_pair = (
                (lead_x > midline - 3.0)
                & (trail_x > midline - 12.0)
                & (avg_forward > 0.20)
                & distance_decreasing
            )
            same_corridor = (lateral_sep <= 2.75) & (heading_sim >= 0.88)
            longitudinal_gap = torch.abs(dx01)
            stable_leader_follower = (
                (longitudinal_gap >= 1.0)
                & (longitudinal_gap <= 5.5)
                & (leader_streak >= 3)
            )
            gap_stable = (
                self.bt_adapt_prev_blue_valid
                & (self.bt_adapt_convoy_prev_longitudinal_gap > 0.0)
                & (torch.abs(longitudinal_gap - self.bt_adapt_convoy_prev_longitudinal_gap) <= 1.25)
            )
            reject_dual_rush = (
                (longitudinal_gap < 1.0)
                | (leader_streak < 3)
            )
            convoy_evidence = offensive_pair & same_corridor & stable_leader_follower & gap_stable
            evidence_ticks = torch.where(
                convoy_evidence,
                self.bt_adapt_convoy_evidence_ticks + 1,
                torch.zeros_like(self.bt_adapt_convoy_evidence_ticks),
            )
            self.bt_adapt_convoy_evidence_ticks = torch.where(
                is_op12ish,
                evidence_ticks,
                self.bt_adapt_convoy_evidence_ticks,
            )
            self.bt_adapt_convoy_offensive_active = torch.where(is_op12ish, offensive_pair, self.bt_adapt_convoy_offensive_active)
            self.bt_adapt_convoy_corridor_active = torch.where(is_op12ish, same_corridor, self.bt_adapt_convoy_corridor_active)
            self.bt_adapt_convoy_leader_active = torch.where(is_op12ish, stable_leader_follower, self.bt_adapt_convoy_leader_active)
            self.bt_adapt_convoy_reject_rush = torch.where(is_op12ish, reject_dual_rush, self.bt_adapt_convoy_reject_rush)
            self.bt_adapt_convoy_leader_id = torch.where(
                is_op12ish,
                torch.where(leader_sign > 0, torch.zeros_like(leader_sign), torch.where(leader_sign < 0, torch.ones_like(leader_sign), -torch.ones_like(leader_sign))),
                self.bt_adapt_convoy_leader_id,
            )
            self.bt_adapt_convoy_longitudinal_gap = torch.where(is_op12ish, longitudinal_gap, self.bt_adapt_convoy_longitudinal_gap)
            self.bt_adapt_convoy_gap_stable = torch.where(is_op12ish, gap_stable, self.bt_adapt_convoy_gap_stable)
            self.bt_adapt_convoy_lateral_gap = torch.where(is_op12ish, lateral_sep, self.bt_adapt_convoy_lateral_gap)
            self.bt_adapt_convoy_heading_similarity = torch.where(is_op12ish, heading_sim, self.bt_adapt_convoy_heading_similarity)
            self.bt_adapt_convoy_centroid_forward_speed = torch.where(is_op12ish, avg_forward, self.bt_adapt_convoy_centroid_forward_speed)
            live_opening_escort = (
                is_op12ish
                & (self.sim_step_count.to(torch.int32) < 20)
                & (~bb["blue_carry_any"])
                & blue_alive[:, 0]
                & blue_alive[:, 1]
                & self.bt_adapt_prev_blue_valid
                & (self.bt_adapt_convoy_evidence_ticks >= 1)
            )
            live_ticks = torch.where(
                live_opening_escort,
                self.bt_adapt_opening_escort_ticks + 1,
                torch.zeros_like(self.bt_adapt_opening_escort_ticks),
            )
            self.bt_adapt_opening_escort_ticks = torch.where(
                is_op12ish,
                live_ticks,
                self.bt_adapt_opening_escort_ticks,
            )
            live_opening_escort = live_opening_escort & (
                self.bt_adapt_opening_escort_ticks >= self._opening_escort_ticks_threshold(prof)
            )
            new_live_escort = live_opening_escort & (self.bt_adapt_opening_escort_first_trigger_step < 0)
            self.bt_adapt_opening_escort_first_trigger_step = torch.where(
                new_live_escort,
                self.sim_step_count.to(torch.int32),
                self.bt_adapt_opening_escort_first_trigger_step,
            )
            self.bt_adapt_opening_escort_active_steps += live_opening_escort.to(torch.int32)
            self.bt_adapt_prev_blue_x = torch.where(
                is_op12ish[:, None],
                self.blue_x,
                self.bt_adapt_prev_blue_x,
            )
            self.bt_adapt_prev_blue_y = torch.where(
                is_op12ish[:, None],
                self.blue_y,
                self.bt_adapt_prev_blue_y,
            )
            self.bt_adapt_prev_blue_valid = torch.where(
                is_op12ish,
                torch.ones_like(self.bt_adapt_prev_blue_valid),
                self.bt_adapt_prev_blue_valid,
            )
            self.bt_adapt_prev_centroid_flag_dist = torch.where(
                is_op12ish,
                centroid_flag_dist,
                self.bt_adapt_prev_centroid_flag_dist,
            )
            self.bt_adapt_convoy_prev_longitudinal_gap = torch.where(
                is_op12ish,
                longitudinal_gap,
                self.bt_adapt_convoy_prev_longitudinal_gap,
            )
            self.bt_adapt_prev_centroid_valid = torch.where(
                is_op12ish,
                torch.ones_like(self.bt_adapt_prev_centroid_valid),
                self.bt_adapt_prev_centroid_valid,
            )

        # OP12 opening escort detector is suspicion-only. Before pickup, RUSH
        # and ESCORT can look identical often enough that a hard response would
        # punish the intended RUSH niche. Hard anti-escort behavior waits for
        # post-pickup confirmation below.
        opening_escort = (
            is_op12ish
            & (
                bb.get("adapt_opening_escort", torch.zeros((B,), dtype=torch.bool, device=device))
                | live_opening_escort
            )
            & (~bb["blue_carry_any"])
        )
        opening_escort_prepare = torch.zeros((B,), dtype=torch.bool, device=device)
        if opening_escort_prepare.any() and self.Nb >= 2:
            blue_trailer = torch.where(
                self.blue_x[:, 0] <= self.blue_x[:, 1],
                torch.zeros((B,), dtype=torch.int64, device=device),
                torch.ones((B,), dtype=torch.int64, device=device),
            )
            tx = self.blue_x[torch.arange(B, device=device), blue_trailer]
            ty = self.blue_y[torch.arange(B, device=device), blue_trailer]
            trailer_dist = torch.sqrt((self.red_x - tx[:, None]) ** 2 + (self.red_y - ty[:, None]) ** 2 + 1e-8)
            trailer_dist = torch.where(eligible, trailer_dist, trailer_dist.new_full((), 1e9))
            disruptor = torch.argmin(trailer_dist, dim=1)

            home_x = bb["blue_flag_pos"][:, 0:1]
            home_y = bb["blue_flag_pos"][:, 1:2]
            home_dist = torch.sqrt((self.red_x - home_x) ** 2 + (self.red_y - home_y) ** 2 + 1e-8)
            home_dist = torch.where(eligible, home_dist, home_dist.new_full((), 1e9))
            raider = torch.argmin(home_dist, dim=1)
            conflict = raider == disruptor
            if self.Nr >= 2:
                home_dist_alt = home_dist.clone()
                home_dist_alt[torch.arange(B, device=device), disruptor] = 1e9
                raider = torch.where(conflict, torch.argmin(home_dist_alt, dim=1), raider)

            escort_lock = prof["lock_intercept"] + self._COLLAPSE_ROLE_LOCK_BONUS
            for j in range(Nr):
                disrupt = opening_escort & prof["enable_intercept"] & (disruptor == j) & eligible[:, j]
                out[:, j] = torch.where(
                    disrupt,
                    torch.full((B,), ROLE_INTERCEPTOR, dtype=torch.int32, device=device),
                    out[:, j],
                )
                lock[:, j] = torch.where(disrupt, escort_lock, lock[:, j])

                raid = opening_escort & prof["enable_counter"] & (raider == j) & eligible[:, j]
                out[:, j] = torch.where(
                    raid,
                    torch.full((B,), ROLE_COUNTER, dtype=torch.int32, device=device),
                    out[:, j],
                )
                lock[:, j] = torch.where(raid, prof["lock_counter"], lock[:, j])

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

        # OP12 anti-split conversion (Stage 2, 2026-07-28 redesign): if Blue
        # creates SUSTAINED simultaneous two-lane pressure, commit an extra
        # interceptor to deny the split for a bounded duration. Direct
        # clustered rushes do not satisfy this trigger.
        #
        # Two changes from the original version:
        # 1. Trigger switched from the live, single-step-flickery
        #    adapt_split_pressure to adapt_op12_split_response_active (a
        #    bounded, decaying-duration state set in _update_adaptive_memory
        #    -- extends every time the ALREADY-debounced split_pressure_active
        #    signal re-fires, decays _OP12_SPLIT_RESPONSE_DURATION steps
        #    after it stops). This is the "bounded state, not a per-step
        #    utility nudge" the redesign calls for.
        # 2. The blue_carry_any gate is dropped so the role can commit BEFORE
        #    pickup too -- the base routing table already falls back an
        #    INTERCEPTOR with no carrier to the same target logic as
        #    DEFENDER (_bt_red.py's role_j==ROLE_INTERCEPTOR & ~blue_carry_any
        #    branch), so this pre-pickup commitment reads as "prioritize
        #    defending the exposed lane" rather than continuing whatever
        #    role (often ATTACKER) this agent held before the split pattern
        #    was proven -- exactly "changes defended lane/target priorities
        #    for a fixed duration."
        # 3. NEVER reassigns an agent currently holding ROLE_COUNTER --
        #    dev25/dev26 showed that perturbing OP12's own counter-attacking
        #    agent (even indirectly, via route bias) breaks its stable
        #    scoring cadence and manifests as a fake "improvement" that is
        #    really OP12 sabotaging itself. "COUNTER keeps counter pressure"
        #    is enforced structurally here, not just hoped for.
        is_op12 = prof["bt_level"] == 12
        split_denial = (
            is_op12
            & bb.get("adapt_op12_split_response_active", torch.zeros((B,), dtype=torch.bool, device=device))
            & prof["enable_intercept"]
        )
        if split_denial.any():
            split_lock = prof["lock_intercept"] + self._COLLAPSE_ROLE_LOCK_BONUS
            for j in range(Nr):
                assign = split_denial & eligible[:, j] & (out[:, j] != ROLE_COUNTER)
                out[:, j] = torch.where(
                    assign,
                    torch.full((B,), ROLE_INTERCEPTOR, dtype=torch.int32, device=device),
                    out[:, j],
                )
                lock[:, j] = torch.where(assign, split_lock, lock[:, j])

        # OP12 confirmed-escort response deliberately does not replace roles.
        # dev22 showed INTERCEPTOR+COUNTER was useful; the opt-in response is
        # route-only and is applied in _bt_apply_adaptive_route_overrides.

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

        # OP11 split-lane isolation: unlike OP12's split_denial above (gated
        # on blue_carry_any -- reactive, carrier-only), this fires as soon as
        # persistent wide-lateral-separation split pressure is detected,
        # BEFORE any pickup. SPLIT's structural exploit is an uncovered lane
        # at approach time, not just during the carrier's return trip, so
        # denial has to start before the flag is grabbed to matter. Route-
        # level 1:1 marking (red[j] -> blue[j]) is applied in
        # _bt_apply_adaptive_route_overrides; this block only sets the role
        # + lock so that routing loop can see it. Deliberately does not touch
        # high_escort/escort_denial, so a concentrated carrier-and-escort
        # blue formation still faces the unmodified (already
        # escort-vulnerable) OP11 behavior -- teammate_dist>=12.0 in the
        # split_pressure_now gate already excludes a tight escort formation.
        #
        # LATCHED, not live: bt_adapt_split_pressure_ticks flickers back to 0
        # on any single step where the strict simultaneous geometric test
        # fails (blue accelerates/decelerates, obstacle avoidance, etc.), so
        # gating on the live adapt_split_pressure signal only isolates for a
        # few steps at a time and lets SPLIT operate freely in between.
        # bt_adapt_split_first_trigger_step is set once (>=0) the first time
        # the threshold is crossed and never resets within the episode --
        # exactly the PERSISTENT signal the task calls for. Once split play
        # has been proven this episode, stay in isolation mode for the rest
        # of it.
        is_op11 = prof["bt_level"] == 11
        op11_split_isolate = is_op11 & (self.bt_adapt_split_first_trigger_step >= 0)
        if op11_split_isolate.any() and Nr >= 2 and self.Nb >= 2:
            isolate_lock = prof["lock_intercept"] + self._COLLAPSE_ROLE_LOCK_BONUS
            for j in range(min(Nr, self.Nb)):
                assign = op11_split_isolate & eligible[:, j]
                out[:, j] = torch.where(
                    assign,
                    torch.full((B,), ROLE_INTERCEPTOR, dtype=torch.int32, device=device),
                    out[:, j],
                )
                lock[:, j] = torch.where(assign, isolate_lock, lock[:, j])

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
        # Stage-2 redesign (2026-07-28): bounded, decaying-duration trigger
        # instead of the live per-step signal -- see the matching
        # role-override block in _bt_apply_adaptive_role_overrides for the
        # full rationale (persistence-gated, not per-step, and never steals
        # the COUNTER agent).
        split_denial_route = (bt_level == 12) & bb.get(
            "adapt_op12_split_response_active",
            torch.zeros((B,), dtype=torch.bool, device=device),
        )
        escort_denial_route = (
            bool(getattr(self, "op12_confirmed_escort_response_enabled", False))
            & (bt_level == 12)
            & bb.get("adapt_confirmed_escort", torch.zeros((B,), dtype=torch.bool, device=device))
            & bb["blue_carry_any"]
        )
        op12_opening = (
            (bt_level == 12)
            & (self.sim_step_count.to(torch.int32) < 20)
            & (~split_denial_route)
        )
        emergency_collapse = bb["adapt_emergency_collapse"] & (~op12_opening)
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
            split_int = split_denial_route & int_mask
            split_body_bx = ec_x + (home_x - ec_x) * self._OP8_CAP_BODY_X_FRAC
            bx = torch.where(split_int & (j == 0), pred_x + (home_x - pred_x) * block_frac, bx)
            by = torch.where(split_int & (j == 0), pred_y + (home_y - pred_y) * block_frac, by)
            bx = torch.where(split_int & (j == 1), split_body_bx, bx)
            by = torch.where(split_int & (j == 1), home_y, by)

            # Stage-2 pre-pickup branch: same bounded split-response window,
            # but BEFORE blue has possession (int_mask above requires
            # blue_carry_any, so it cannot fire here -- and the final tx/ty
            # write below is widened to include pre_pickup_int for exactly
            # this reason). "Split trigger changes defended lane/target
            # priorities for a fixed duration" -- this agent already
            # committed to ROLE_INTERCEPTOR pre-pickup (role override dropped
            # the blue_carry_any gate for exactly this), and the base routing
            # table's own fallback for an INTERCEPTOR-without-a-carrier is
            # the generic "nearest intruder" DEFENDER-equivalent target,
            # which can pick the WRONG (nearer) attacker while the wide,
            # laterally-separated one is the actual split threat. Explicitly
            # target whichever blue agent is farther from the field's
            # lateral center instead.
            pre_pickup_int = (
                split_denial_route & (role_j == ROLE_INTERCEPTOR) & (~bb["blue_carry_any"])
            )
            if self.Nb >= 2 and pre_pickup_int.any():
                center_y = float(self.rows) * 0.5
                far_is_1 = torch.abs(self.blue_y[:, 1] - center_y) >= torch.abs(self.blue_y[:, 0] - center_y)
                far_x = torch.where(far_is_1, self.blue_x[:, 1], self.blue_x[:, 0])
                far_y = torch.where(far_is_1, self.blue_y[:, 1], self.blue_y[:, 0])
                bx = torch.where(pre_pickup_int, far_x, bx)
                by = torch.where(pre_pickup_int, far_y, by)
            else:
                pre_pickup_int = torch.zeros((B,), dtype=torch.bool, device=device)
            escort_bx = ec_x + (home_x - ec_x) * 0.35
            escort_by = ec_y + (home_y - ec_y) * 0.35
            # ETA-gate (dev26 diagnosis): a single-episode trace showed the
            # unconditional carrier-lane route winning the confirmed-escort
            # duel while leaving THIS agent farther from where its own later
            # COUNTER-role duty (red's second scoring pass) needed it to be --
            # a small early-position perturbation that cascaded into red's
            # own scoring getting stuck (episode ran to the step cap instead
            # of closing out like the unmodified baseline). Only take the
            # modified route when it is not a worse rendezvous than the
            # default intercept point, so the specialized response can never
            # replace an already-superior default path.
            red_jx = self.red_x[:, j]
            red_jy = self.red_y[:, j]
            default_eta = torch.sqrt((red_jx - bx) ** 2 + (red_jy - by) ** 2 + 1e-8)
            modified_eta = torch.sqrt((red_jx - escort_bx) ** 2 + (red_jy - escort_by) ** 2 + 1e-8)
            escort_carrier = escort_denial_route & int_mask & (modified_eta <= default_eta)
            bx = torch.where(escort_carrier, escort_bx, bx)
            by = torch.where(escort_carrier, escort_by, by)
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
                emergency_collapse,
                bb["ec_y"] + (home_y - bb["ec_y"]) * cap_lane_body,
                by,
            )
            write_mask = int_mask | pre_pickup_int
            tx[:, j] = torch.where(write_mask, torch.clamp(bx, 0.0, max_x), tx[:, j])
            ty[:, j] = torch.where(write_mask, torch.clamp(by, 0.0, max_y), ty[:, j])

        # OP12 Stage-3 REDESIGN (round 2, 2026-07-28) -- TRIED AND REVERTED.
        # Root cause diagnosed via experiments/diagnose_op12_vs_turtle.py plus
        # a carrier-tag follow-up trace: of 10 "red tagged while carrying"
        # failures across 8 episodes, the largest single bucket (3/10) was
        # ESCORT correctly assigned to protect the carrier yet still failing
        # to prevent the tag, because the base ESCORT route
        # (_bt_route_target's "interpose" branch) targets the midpoint
        # between the carrier and only the SINGLE nearest blue agent, while
        # TURTLE's defense is an explicit two-agent pincer. Tried: an
        # OP12-only override targeting the midpoint between the carrier and
        # the CENTROID of both blue agents at a tighter 30% blend. Isolated
        # 8-seed screen (base_seed 556001, artifacts/
        # op12_stage3c_escort_geometry_dev1_8seed) showed this made things
        # WORSE, not better -- BLUE_TURTLE margin +0.625 -> +0.750 (WR
        # 4/8 -> 5/8) and BLUE_ESCORT -2.000 -> -2.375 -- while RUSH/SPLIT
        # stayed bit-identical. Averaging toward both blue agents' centroid
        # apparently dilutes the escort's positioning against the CLOSER,
        # more immediate threat without adding real coverage against the
        # farther one, net negative. REVERTED (this block intentionally left
        # as a no-op / removed rather than kept enabled) -- see
        # docs/research-progress-tracker.md for the full three-attempt
        # write-up and the resulting decision to close OP12 as a failed
        # RUSH-niche candidate per the hard two-round budget.

        # OP11 split-lane isolation route (post-dev4 revision, 2026-07-28):
        # Dev4 tried geometric 1:1 + predictive approach cutoff + dual
        # carrier denial with threshold=1. That FAILED the ESCORT niche:
        # SPLIT rose +1.25→+2.75 and ESCORT collapsed +0.50→-1.125 even
        # though ESCORT's latch rate stayed 0/8 (so the damage was not a
        # simple false-latch of ESCORT). Reverted to the proven selective
        # latch (threshold=2) plus index 1:1 chase. The remaining ESCORT-
        # niche work is a SEPARATE, isolated post-pickup dual-deny that
        # must be measured against this restored baseline before any
        # further pre-pickup route changes.
        op11_split_isolate = is_op11 & (self.bt_adapt_split_first_trigger_step >= 0)
        if op11_split_isolate.any() and self.Nb >= 2:
            for j in range(min(Nr, self.Nb)):
                mark = op11_split_isolate & (roles[:, j] == ROLE_INTERCEPTOR)
                tx[:, j] = torch.where(mark, torch.clamp(self.blue_x[:, j], 0.0, max_x), tx[:, j])
                ty[:, j] = torch.where(mark, torch.clamp(self.blue_y[:, j], 0.0, max_y), ty[:, j])

        return tx, ty


__all__ = ["_BTAdaptiveMixin"]
