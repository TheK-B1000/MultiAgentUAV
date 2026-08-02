"""RewardsMixin methods for BatchedCTFCore."""
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from macro_actions import MacroAction
from rl.global_state import build_global_state_batch
from game_manager import (
    get_grab_score_delta,
    get_capture_score_delta,
    SPARSE_TAG_NO_FLAG_POINTS,
    SPARSE_TAG_WITH_FLAG_POINTS,
    SPARSE_FLAG_CAPTURE_POINTS,
    SPARSE_OOB_POINTS,
    SPARSE_MINE_TAG_POINTS,
)

from .._constants import (
    CNN_COLS,
    CNN_ROWS,
    GLOBAL_STATE_CHANNELS,
    METRIC_ZONE_COLS,
    METRIC_ZONE_ROWS,
    NUM_CNN_CHANNELS,
    VEC_OBS_DIM,
)
from .._episode_payload import _build_episode_result_payload


class _RewardsMixin:
    def _compute_potentials(
        self,
        blue_x: torch.Tensor,
        blue_y: torch.Tensor,
        blue_carrying: torch.Tensor,
        red_carrying: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        max_dist = max(1e-6, self.max_dist)

        def closeness(dist: torch.Tensor) -> torch.Tensor:
            return 1.0 - torch.clamp(dist / max_dist, min=0.0, max=1.0)

        attack_dx = self.red_flag_pos[:, None, 0] - blue_x
        attack_dy = self.red_flag_pos[:, None, 1] - blue_y
        attack_dist = torch.sqrt(attack_dx * attack_dx + attack_dy * attack_dy + 1e-8)
        attack_phi = torch.where(
            blue_carrying,
            torch.zeros_like(attack_dist),
            closeness(attack_dist),
        ).mean(dim=1)

        return_dx = self.blue_flag_home[:, None, 0] - blue_x
        return_dy = self.blue_flag_home[:, None, 1] - blue_y
        return_dist = torch.sqrt(return_dx * return_dx + return_dy * return_dy + 1e-8)
        return_phi = torch.where(
            blue_carrying,
            closeness(return_dist),
            torch.zeros_like(return_dist),
        ).mean(dim=1)

        red_has_flag = red_carrying.any(dim=1)
        red_carrier_idx = torch.argmax(red_carrying.to(torch.int64), dim=1)
        red_carrier_x = self.red_x[torch.arange(self.B, device=self.device), red_carrier_idx]
        red_carrier_y = self.red_y[torch.arange(self.B, device=self.device), red_carrier_idx]
        defend_dx = red_carrier_x[:, None] - blue_x
        defend_dy = red_carrier_y[:, None] - blue_y
        defend_dist = torch.sqrt(defend_dx * defend_dx + defend_dy * defend_dy + 1e-8)
        defend_phi = torch.where(
            red_has_flag[:, None],
            closeness(defend_dist),
            torch.zeros_like(defend_dist),
        ).mean(dim=1)
        return attack_phi, return_phi, defend_phi

    def _pbrs_reward(
        self,
        prev_blue_x: torch.Tensor,
        prev_blue_y: torch.Tensor,
        prev_blue_carrying: torch.Tensor,
        prev_red_x: Optional[torch.Tensor] = None,
        prev_red_y: Optional[torch.Tensor] = None,
        prev_red_carrying: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        max_dist = max(1e-6, self.max_dist)

        def closeness(dist: torch.Tensor) -> torch.Tensor:
            return 1.0 - torch.clamp(dist / max_dist, min=0.0, max=1.0)

        def masked_mean(dist: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
            return torch.where(active, closeness(dist), torch.zeros_like(dist)).mean(dim=1)

        # PBRS terms should not punish event transitions. For example, a pickup
        # turns off the attack objective and turns on the return objective; the
        # event reward handles that step, while shaping resumes on stable phases.
        attack_active = (~prev_blue_carrying) & (~self.blue_carrying)
        prev_attack_dist = torch.sqrt(
            (self.red_flag_pos[:, None, 0] - prev_blue_x) ** 2
            + (self.red_flag_pos[:, None, 1] - prev_blue_y) ** 2
            + 1e-8
        )
        cur_attack_dist = torch.sqrt(
            (self.red_flag_pos[:, None, 0] - self.blue_x) ** 2
            + (self.red_flag_pos[:, None, 1] - self.blue_y) ** 2
            + 1e-8
        )
        prev_attack = masked_mean(prev_attack_dist, attack_active)
        cur_attack = masked_mean(cur_attack_dist, attack_active)

        return_active = prev_blue_carrying & self.blue_carrying
        prev_return_dist = torch.sqrt(
            (self.blue_flag_home[:, None, 0] - prev_blue_x) ** 2
            + (self.blue_flag_home[:, None, 1] - prev_blue_y) ** 2
            + 1e-8
        )
        cur_return_dist = torch.sqrt(
            (self.blue_flag_home[:, None, 0] - self.blue_x) ** 2
            + (self.blue_flag_home[:, None, 1] - self.blue_y) ** 2
            + 1e-8
        )
        prev_return = masked_mean(prev_return_dist, return_active)
        cur_return = masked_mean(cur_return_dist, return_active)

        if prev_red_x is None:
            prev_red_x = self.red_x
        if prev_red_y is None:
            prev_red_y = self.red_y
        if prev_red_carrying is None:
            prev_red_carrying = self.red_carrying
        env_idx = torch.arange(self.B, device=self.device)
        prev_red_has_flag = prev_red_carrying.any(dim=1)
        cur_red_has_flag = self.red_carrying.any(dim=1)
        defense_active = (prev_red_has_flag & cur_red_has_flag)[:, None].expand_as(prev_blue_x)
        prev_red_carrier_idx = torch.argmax(prev_red_carrying.to(torch.int64), dim=1)
        cur_red_carrier_idx = torch.argmax(self.red_carrying.to(torch.int64), dim=1)
        prev_red_carrier_x = prev_red_x[env_idx, prev_red_carrier_idx]
        prev_red_carrier_y = prev_red_y[env_idx, prev_red_carrier_idx]
        cur_red_carrier_x = self.red_x[env_idx, cur_red_carrier_idx]
        cur_red_carrier_y = self.red_y[env_idx, cur_red_carrier_idx]
        prev_defend_dist = torch.sqrt(
            (prev_red_carrier_x[:, None] - prev_blue_x) ** 2
            + (prev_red_carrier_y[:, None] - prev_blue_y) ** 2
            + 1e-8
        )
        cur_defend_dist = torch.sqrt(
            (cur_red_carrier_x[:, None] - self.blue_x) ** 2
            + (cur_red_carrier_y[:, None] - self.blue_y) ** 2
            + 1e-8
        )
        prev_defend = masked_mean(prev_defend_dist, defense_active)
        cur_defend = masked_mean(cur_defend_dist, defense_active)

        gamma = float(self.cfg.pbrs_gamma)
        rpbrs = (
            float(self.cfg.pbrs_attack_coef) * (gamma * cur_attack - prev_attack)
            + float(self.cfg.pbrs_return_coef) * (gamma * cur_return - prev_return)
            + float(self.cfg.pbrs_defense_coef) * (gamma * cur_defend - prev_defend)
        )
        self._last_dense_progress = (
            float(self.cfg.pbrs_attack_coef) * (cur_attack - prev_attack)
            + float(self.cfg.pbrs_return_coef) * (cur_return - prev_return)
            + float(self.cfg.pbrs_defense_coef) * (cur_defend - prev_defend)
        )
        return rpbrs

    def _team_coordination_reward(
        self,
        prev_blue_x: torch.Tensor,
        prev_blue_y: torch.Tensor,
        yaw_cmd_blue: torch.Tensor,
    ) -> torch.Tensor:
        red_has_flag = self.red_carrying.any(dim=1)
        home_dx = self.blue_x - self.blue_flag_home[:, None, 0]
        home_dy = self.blue_y - self.blue_flag_home[:, None, 1]
        near_home = (torch.sqrt(home_dx * home_dx + home_dy * home_dy + 1e-8) <= 6.0).to(torch.float32).mean(dim=1)
        defense_presence = torch.where(
            red_has_flag,
            float(self.cfg.team_defense_presence_reward) * near_home,
            torch.zeros_like(near_home),
        )

        blue_has_flag = self.blue_carrying.any(dim=1)
        carrier_idx = torch.argmax(self.blue_carrying.to(torch.int64), dim=1)
        carrier_x = self.blue_x[torch.arange(self.B, device=self.device), carrier_idx]
        carrier_y = self.blue_y[torch.arange(self.B, device=self.device), carrier_idx]
        edx = self.blue_x - carrier_x[:, None]
        edy = self.blue_y - carrier_y[:, None]
        escorting_teammate = (~self.blue_carrying) & (
            torch.sqrt(edx * edx + edy * edy + 1e-8) <= 5.0
        )
        escort_den = (~self.blue_carrying).to(torch.float32).sum(dim=1).clamp_min(1.0)
        escort = escorting_teammate.to(torch.float32).sum(dim=1) / escort_den
        escort_bonus = torch.where(
            blue_has_flag,
            float(self.cfg.team_escort_reward) * escort,
            torch.zeros_like(escort),
        )

        red_carrier_idx = torch.argmax(self.red_carrying.to(torch.int64), dim=1)
        red_carrier_x = self.red_x[torch.arange(self.B, device=self.device), red_carrier_idx]
        red_carrier_y = self.red_y[torch.arange(self.B, device=self.device), red_carrier_idx]
        intercept_dx = self.blue_x - red_carrier_x[:, None]
        intercept_dy = self.blue_y - red_carrier_y[:, None]
        intercept = (torch.sqrt(intercept_dx * intercept_dx + intercept_dy * intercept_dy + 1e-8) <= 5.0).to(torch.float32).mean(dim=1)
        intercept_bonus = torch.where(
            red_has_flag,
            float(self.cfg.team_intercept_reward) * intercept,
            torch.zeros_like(intercept),
        )

        yaw_abs = torch.abs(yaw_cmd_blue) / max(1e-6, float(self.cfg.max_yaw_rate_rps))
        move_dist = torch.sqrt((self.blue_x - prev_blue_x) ** 2 + (self.blue_y - prev_blue_y) ** 2 + 1e-8)
        spin_pen = self.cfg.spin_penalty_coef * (yaw_abs * (move_dist < 0.03).to(torch.float32)).mean(dim=1)
        idle_pen = self.cfg.idle_penalty_coef * (self.blue_speed.mean(dim=1) < 0.15).to(torch.float32)
        return defense_presence + escort_bonus + intercept_bonus - spin_pen - idle_pen

    def _sparse_reward_points(
        self,
        blue_cap_env: torch.Tensor,
        red_cap_env: torch.Tensor,
        blue_tag_noflag: torch.Tensor,
        blue_tag_withflag: torch.Tensor,
        red_tag_total: torch.Tensor,
        blue_oob: torch.Tensor,
        blue_mine_tags: Optional[torch.Tensor] = None,
        red_mine_tags: Optional[torch.Tensor] = None,
        red_oob: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Values from game_manager (sparse event points) so scoring/rewards stay aligned.
        r = torch.zeros((self.B,), dtype=torch.float32, device=self.device)
        r += float(SPARSE_FLAG_CAPTURE_POINTS) * blue_cap_env.to(torch.float32)
        r -= float(SPARSE_FLAG_CAPTURE_POINTS) * red_cap_env.to(torch.float32)
        # Configurable so the tag-farming hypothesis can be ablated. Applied to
        # BOTH directions: zeroing it removes tag income and the being-tagged
        # cost together, leaving the attack/defend trade-off undistorted rather
        # than punishing attack with no compensating income.
        tag_noflag_points = float(
            getattr(self.cfg, "sparse_tag_no_flag_points", SPARSE_TAG_NO_FLAG_POINTS)
        )
        tag_carrier_points = float(
            getattr(self.cfg, "sparse_tag_with_flag_points", SPARSE_TAG_WITH_FLAG_POINTS)
        )
        r += tag_noflag_points * blue_tag_noflag.to(torch.float32)
        r += tag_carrier_points * blue_tag_withflag.to(torch.float32)
        r -= tag_noflag_points * red_tag_total.to(torch.float32)
        if blue_mine_tags is not None:
            r += float(SPARSE_MINE_TAG_POINTS) * blue_mine_tags.to(torch.float32)
        if red_mine_tags is not None:
            r -= float(SPARSE_MINE_TAG_POINTS) * red_mine_tags.to(torch.float32)
        r += float(SPARSE_OOB_POINTS) * blue_oob.sum(dim=1).to(torch.float32)
        if red_oob is not None:
            r -= float(SPARSE_OOB_POINTS) * red_oob.sum(dim=1).to(torch.float32)
        return r

    def _surface_pressure_reward(
        self,
        *,
        blue_cap_env: torch.Tensor,
        red_grab_env: torch.Tensor,
    ) -> torch.Tensor:
        """Default-off margin/tempo pressure terms for diagnostic surfaces."""
        out = torch.zeros((self.B,), dtype=torch.float32, device=self.device)
        step_frac = torch.clamp(
            self.step_count.to(torch.float32) / max(1.0, float(self.max_steps)),
            min=0.0,
            max=1.0,
        )
        out = out + float(self.cfg.surface_blue_capture_tempo_bonus) * blue_cap_env.to(torch.float32) * (1.0 - step_frac)
        out = out - float(self.cfg.surface_red_flag_touch_penalty) * red_grab_env.to(torch.float32)

        red_has_flag = self.red_carrying.any(dim=1)
        red_idx = torch.argmax(self.red_carrying.to(torch.int64), dim=1)
        red_x = self.red_x[torch.arange(self.B, device=self.device), red_idx]
        red_y = self.red_y[torch.arange(self.B, device=self.device), red_idx]
        red_home_dx = self.blue_flag_home[:, 0] - red_x
        red_home_dy = self.blue_flag_home[:, 1] - red_y
        red_home_dist = torch.sqrt(red_home_dx * red_home_dx + red_home_dy * red_home_dy + 1e-8)
        red_progress = 1.0 - torch.clamp(red_home_dist / max(1e-6, self.max_dist), min=0.0, max=1.0)
        out = out - float(self.cfg.surface_red_carrier_progress_penalty) * torch.where(
            red_has_flag,
            red_progress,
            torch.zeros_like(red_progress),
        )

        blue_has_flag = self.blue_carrying.any(dim=1)
        blue_idx = torch.argmax(self.blue_carrying.to(torch.int64), dim=1)
        blue_x = self.blue_x[torch.arange(self.B, device=self.device), blue_idx]
        blue_y = self.blue_y[torch.arange(self.B, device=self.device), blue_idx]
        blue_home_dx = self.blue_flag_home[:, 0] - blue_x
        blue_home_dy = self.blue_flag_home[:, 1] - blue_y
        blue_home_dist = torch.sqrt(blue_home_dx * blue_home_dx + blue_home_dy * blue_home_dy + 1e-8)
        blue_progress = 1.0 - torch.clamp(blue_home_dist / max(1e-6, self.max_dist), min=0.0, max=1.0)
        out = out + float(self.cfg.surface_blue_near_cap_bonus) * torch.where(
            blue_has_flag,
            blue_progress,
            torch.zeros_like(blue_progress),
        )
        return out

    def _reward_total(
        self,
        rterm: torch.Tensor,
        roff: torch.Tensor,
        rpbrs: torch.Tensor,
        rteam: torch.Tensor,
        sparse_points: torch.Tensor,
        rfail: torch.Tensor,
        stalemate_trigger: torch.Tensor,
    ) -> torch.Tensor:
        sparse_norm = sparse_points / 100.0
        dense = rpbrs + rteam
        raw = (
            rterm
            + roff
            + rfail
            + float(self.cfg.dense_weight) * dense
            + float(self.cfg.sparse_weight) * sparse_norm
        )
        raw = raw + torch.where(
            stalemate_trigger,
            torch.tensor(float(self.cfg.stalemate_penalty), device=self.device),
            torch.tensor(0.0, device=self.device),
        )
        scaled = torch.tanh(raw / max(1e-6, float(self.cfg.reward_scale)))
        return torch.clamp(scaled, -float(self.cfg.reward_clip), float(self.cfg.reward_clip))

    def _router_reward_total(
        self,
        rterm: torch.Tensor,
        blue_cap_env: torch.Tensor,
        red_cap_env: torch.Tensor,
        blue_tag_withflag: torch.Tensor,
        red_tag_total: torch.Tensor,
    ) -> torch.Tensor:
        """Sparse team-consequence reward for the V6I7 GRU router.

        Uses exact event tensors (flag captures, carrier tags) rather than the
        aggregated sparse total, so the router sees only strategy-relevant signals.
        Returns zeros when ``router_reward_config`` is absent or disabled.
        """
        rrc = getattr(self.cfg, "router_reward_config", None)
        if rrc is None or not rrc.enabled:
            return torch.zeros((self.B,), dtype=torch.float32, device=self.device)

        win_w = float(rrc.win_weight)
        flag_w = float(rrc.flag_cap_weight)
        sparse_w = float(rrc.sparse_weight)
        scale = float(rrc.scale)
        normalize = bool(rrc.normalize)

        # Flag-capture events (normalized to same scale as sparse_norm = points/100)
        net_cap = (
            float(SPARSE_FLAG_CAPTURE_POINTS) * blue_cap_env.to(torch.float32)
            - float(SPARSE_FLAG_CAPTURE_POINTS) * red_cap_env.to(torch.float32)
        ) / 100.0

        # Carrier-tag events: blue tags enemy carrier (good), red tags blue carrier (bad)
        net_carrier_tags = (
            float(SPARSE_TAG_WITH_FLAG_POINTS) * blue_tag_withflag.to(torch.float32)
            - float(SPARSE_TAG_NO_FLAG_POINTS) * red_tag_total.to(torch.float32)
        ) / 100.0

        raw = win_w * rterm + flag_w * net_cap + sparse_w * net_carrier_tags
        if normalize:
            return torch.tanh(raw / max(1e-6, scale))
        return raw * scale
