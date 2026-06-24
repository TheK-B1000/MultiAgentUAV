"""MetricsMixin methods for BatchedCTFCore."""
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
from .._maps import is_split_lane_layout


class _MetricsMixin:
    def _update_episode_metrics(
        self,
        first_score_mask: torch.Tensor,
        *,
        prev_blue_x: Optional[torch.Tensor] = None,
        prev_blue_y: Optional[torch.Tensor] = None,
        prev_red_x: Optional[torch.Tensor] = None,
        prev_red_y: Optional[torch.Tensor] = None,
    ) -> None:
        """Accumulate outcome-neutral episode telemetry from current positions."""
        step_index = self.step_count.to(torch.float32) + 1.0
        self.metric_time_to_first_score = torch.where(
            first_score_mask & (self.metric_time_to_first_score < 0.0),
            step_index,
            self.metric_time_to_first_score,
        )

        pair_live = self.blue_alive[:, :, None] & self.red_alive[:, None, :]
        pair_dist = torch.sqrt(
            (self.blue_x[:, :, None] - self.red_x[:, None, :]) ** 2
            + (self.blue_y[:, :, None] - self.red_y[:, None, :]) ** 2
            + 1e-8
        )
        pair_count = pair_live.sum(dim=(1, 2))
        pair_dist_sum = torch.where(pair_live, pair_dist, torch.zeros_like(pair_dist)).sum(dim=(1, 2))
        has_pairs = pair_count > 0
        step_mean_dist = torch.zeros((self.B,), dtype=torch.float32, device=self.device)
        step_mean_dist = torch.where(
            has_pairs,
            pair_dist_sum / torch.clamp(pair_count.to(torch.float32), min=1.0),
            step_mean_dist,
        )
        self.metric_inter_robot_dist_sum += torch.where(has_pairs, step_mean_dist, torch.zeros_like(step_mean_dist))
        self.metric_inter_robot_dist_count += has_pairs.to(torch.int32)

        collision_radius = max(0.0, float(self.cfg.avoid_collision_radius_cells))
        near_miss_radius = max(collision_radius, collision_radius * 2.0)
        collision_pairs = pair_live & (pair_dist <= collision_radius)
        near_miss_pairs = pair_live & (pair_dist > collision_radius) & (pair_dist <= near_miss_radius)
        self.metric_collision_events += collision_pairs.sum(dim=(1, 2)).to(torch.int32)
        self.metric_near_misses += near_miss_pairs.sum(dim=(1, 2)).to(torch.int32)

        zx = torch.clamp(
            (self.blue_x / max(1.0, float(self.cols)) * float(METRIC_ZONE_COLS)).to(torch.int64),
            0,
            METRIC_ZONE_COLS - 1,
        )
        zy = torch.clamp(
            (self.blue_y / max(1.0, float(self.rows)) * float(METRIC_ZONE_ROWS)).to(torch.int64),
            0,
            METRIC_ZONE_ROWS - 1,
        )
        zone_idx = zy * METRIC_ZONE_COLS + zx
        env_idx = torch.arange(self.B, device=self.device).view(self.B, 1).expand(self.B, self.Nb)
        live = self.blue_alive
        self.metric_blue_zone_visited[env_idx[live], zone_idx[live]] = True
        if (
            prev_blue_x is not None
            and prev_blue_y is not None
            and prev_red_x is not None
            and prev_red_y is not None
        ):
            self._update_route_crossing_metrics(prev_blue_x, prev_blue_y, prev_red_x, prev_red_y)

    def _route_crossings(
        self,
        prev_x: torch.Tensor,
        prev_y: torch.Tensor,
        cur_x: torch.Tensor,
        cur_y: torch.Tensor,
        alive: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mid_x = float(max(0, self.cols - 1)) * 0.5
        mid_y = float(max(0, self.rows - 1)) * 0.5
        denom = cur_x - prev_x
        safe_denom = torch.where(torch.abs(denom) < 1e-6, torch.full_like(denom, 1e-6), denom)
        t = (mid_x - prev_x) / safe_denom
        crossed = (
            (((prev_x < mid_x) & (cur_x >= mid_x)) | ((prev_x > mid_x) & (cur_x <= mid_x)))
            & (t >= 0.0)
            & (t <= 1.0)
            & alive
        )
        cross_y = prev_y + (cur_y - prev_y) * t
        upper = crossed & (cross_y < mid_y)
        lower = crossed & (~upper)
        return upper.sum(dim=1).to(torch.int32), lower.sum(dim=1).to(torch.int32)

    def _route_crossings_by_context(
        self,
        prev_x: torch.Tensor,
        prev_y: torch.Tensor,
        cur_x: torch.Tensor,
        cur_y: torch.Tensor,
        alive: torch.Tensor,
        carrying: torch.Tensor,
        enemy_carrying: torch.Tensor,
        *,
        side: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mid_x = float(max(0, self.cols - 1)) * 0.5
        mid_y = float(max(0, self.rows - 1)) * 0.5
        denom = cur_x - prev_x
        safe_denom = torch.where(torch.abs(denom) < 1e-6, torch.full_like(denom, 1e-6), denom)
        t = (mid_x - prev_x) / safe_denom
        crossed = (
            (((prev_x < mid_x) & (cur_x >= mid_x)) | ((prev_x > mid_x) & (cur_x <= mid_x)))
            & (t >= 0.0)
            & (t <= 1.0)
            & alive
        )
        cross_y = prev_y + (cur_y - prev_y) * t
        upper = crossed & (cross_y < mid_y)
        lower = crossed & (~upper)
        if side == "blue":
            toward_enemy = prev_x < cur_x
            toward_home = prev_x > cur_x
        else:
            toward_enemy = prev_x > cur_x
            toward_home = prev_x < cur_x
        any_enemy_carrier = enemy_carrying.any(dim=1, keepdim=True)
        attack = (~carrying) & (~any_enemy_carrier) & toward_enemy
        return_ctx = carrying & toward_home
        intercept = (~carrying) & any_enemy_carrier

        def _sum(mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            return (upper & mask).sum(dim=1).to(torch.int32), (lower & mask).sum(dim=1).to(torch.int32)

        attack_u, attack_l = _sum(attack)
        return_u, return_l = _sum(return_ctx)
        intercept_u, intercept_l = _sum(intercept)
        return attack_u, attack_l, return_u, return_l, intercept_u, intercept_l

    def _update_route_crossing_metrics(
        self,
        prev_blue_x: torch.Tensor,
        prev_blue_y: torch.Tensor,
        prev_red_x: torch.Tensor,
        prev_red_y: torch.Tensor,
    ) -> None:
        if not is_split_lane_layout(str(getattr(self, "map_layout", ""))):
            return
        bu, bl = self._route_crossings(prev_blue_x, prev_blue_y, self.blue_x, self.blue_y, self.blue_alive)
        ru, rl = self._route_crossings(prev_red_x, prev_red_y, self.red_x, self.red_y, self.red_alive)
        self.metric_blue_route_upper_crossings += bu
        self.metric_blue_route_lower_crossings += bl
        self.metric_red_route_upper_crossings += ru
        self.metric_red_route_lower_crossings += rl
        bau, bal, bru, brl, biu, bil = self._route_crossings_by_context(
            prev_blue_x,
            prev_blue_y,
            self.blue_x,
            self.blue_y,
            self.blue_alive,
            self.blue_carrying,
            self.red_carrying,
            side="blue",
        )
        rau, ral, rru, rrl, riu, ril = self._route_crossings_by_context(
            prev_red_x,
            prev_red_y,
            self.red_x,
            self.red_y,
            self.red_alive,
            self.red_carrying,
            self.blue_carrying,
            side="red",
        )
        self.metric_blue_attack_upper_crossings += bau
        self.metric_blue_attack_lower_crossings += bal
        self.metric_blue_return_upper_crossings += bru
        self.metric_blue_return_lower_crossings += brl
        self.metric_blue_intercept_upper_crossings += biu
        self.metric_blue_intercept_lower_crossings += bil
        self.metric_red_attack_upper_crossings += rau
        self.metric_red_attack_lower_crossings += ral
        self.metric_red_return_upper_crossings += rru
        self.metric_red_return_lower_crossings += rrl
        self.metric_red_intercept_upper_crossings += riu
        self.metric_red_intercept_lower_crossings += ril

    def _build_info(
        self,
        dense: torch.Tensor,
        sparse_points: torch.Tensor,
        stalemate: torch.Tensor,
        reward_terminal: Optional[torch.Tensor] = None,
        reward_offense: Optional[torch.Tensor] = None,
        reward_pbrs: Optional[torch.Tensor] = None,
        reward_team: Optional[torch.Tensor] = None,
        reward_sparse: Optional[torch.Tensor] = None,
        reward_failure: Optional[torch.Tensor] = None,
        reward_total: Optional[torch.Tensor] = None,
        router_reward: Optional[torch.Tensor] = None,
        terminated: Optional[torch.Tensor] = None,
        truncated: Optional[torch.Tensor] = None,
    ) -> List[dict]:
        out: List[dict] = []
        zero = torch.zeros((self.B,), dtype=torch.float32, device=self.device)
        scalars = torch.stack(
            [
                self.blue_score.to(torch.float32),
                self.red_score.to(torch.float32),
                self.step_count.to(torch.float32),
                self.sim_step_count.to(torch.float32),
                self.metric_time_to_first_score,
                self.metric_inter_robot_dist_sum,
                self.metric_inter_robot_dist_count.to(torch.float32),
                self.metric_collision_events.to(torch.float32),
                self.metric_obstacle_collision_events.to(torch.float32),
                self.metric_near_misses.to(torch.float32),
                self.metric_blue_zone_visited.to(torch.float32).mean(dim=1),
                dense.to(torch.float32),
                sparse_points.to(torch.float32),
                (reward_terminal if reward_terminal is not None else zero).to(torch.float32),
                (reward_offense if reward_offense is not None else zero).to(torch.float32),
                (reward_pbrs if reward_pbrs is not None else zero).to(torch.float32),
                (reward_team if reward_team is not None else zero).to(torch.float32),
                (reward_sparse if reward_sparse is not None else zero).to(torch.float32),
                (reward_failure if reward_failure is not None else zero).to(torch.float32),
                (reward_total if reward_total is not None else zero).to(torch.float32),
                (router_reward if router_reward is not None else zero).to(torch.float32),
            ],
            dim=1,
        ).detach().cpu().numpy()
        term_t = terminated if terminated is not None else torch.zeros((self.B,), dtype=torch.bool, device=self.device)
        trunc_t = truncated if truncated is not None else torch.zeros((self.B,), dtype=torch.bool, device=self.device)
        bools = torch.cat(
            [
                self._league_mode[:, None],
                stalemate[:, None].to(torch.bool),
                term_t[:, None].to(torch.bool),
                trunc_t[:, None].to(torch.bool),
                self.blue_alive,
                self.red_alive,
            ],
            dim=1,
        ).detach().cpu().numpy().astype(np.bool_)
        (
            bs,
            rs,
            steps,
            sim_steps,
            first_score,
            dist_sum,
            dist_count,
            collision_events,
            obstacle_collision_events,
            near_misses,
            zone_coverage,
            d_np,
            s_np,
            rt_np,
            ro_np,
            rp_np,
            rteam_np,
            rsp_np,
            rf_np,
            rtot_np,
            rr_np,
        ) = (scalars[:, col] for col in range(scalars.shape[1]))
        league_np = bools[:, 0]
        st_np = bools[:, 1]
        term_np = bools[:, 2]
        trunc_np = bools[:, 3]
        blue_alive_np = bools[:, 4 : 4 + self.Nb]
        red_alive_np = bools[:, 4 + self.Nb : 4 + self.Nb + self.Nr]
        gs_np = build_global_state_batch(self).detach().cpu().numpy().astype(np.float32)
        action_mask_np = self._build_action_mask(side="blue").detach().cpu().numpy().astype(np.float32)
        max_stale = max(1, int(self.cfg.stalemate_max_steps))
        stalemate_frac_np = (self.stalemate_steps.detach().float() / float(max_stale)).detach().cpu().numpy()
        for i in range(self.B):
            mean_dist = None
            if int(dist_count[i]) > 0:
                mean_dist = float(dist_sum[i]) / float(dist_count[i])
            ttfs = None if float(first_score[i]) < 0.0 else float(first_score[i])
            out.append(
                {
                    "blue_score": int(bs[i]),
                    "red_score": int(rs[i]),
                    "decision_steps": int(steps[i]),
                    "sim_steps": int(sim_steps[i]),
                    "phase": self._phase[i],
                    "league_mode": bool(league_np[i]),
                    "opponent_kind": str(self._opponent_kind[i]).lower(),
                    "opponent_key": self._opponent_key[i],
                    "rules_profile": self.rules_profile,
                    "map_set": self.map_set,
                    "map_layout": str(getattr(self, "map_layout", "map_a_open")),
                    "map_vertical_mirror": bool(self.map_vertical_mirror[i].item()),
                    "dense_reward": float(d_np[i]),
                    "sparse_points": float(s_np[i]),
                    "reward_terminal": float(rt_np[i]),
                    "reward_offense": float(ro_np[i]),
                    "reward_pbrs": float(rp_np[i]),
                    "reward_team": float(rteam_np[i]),
                    "reward_sparse": float(rsp_np[i]),
                    "reward_failure": float(rf_np[i]),
                    "reward_sparse_points": float(s_np[i]),
                    "reward_total": float(rtot_np[i]),
                    "router_reward": float(rr_np[i]),
                    "time_to_first_score": ttfs,
                    "collision_events_per_episode": int(collision_events[i]),
                    "obstacle_collision_events_per_episode": int(obstacle_collision_events[i]),
                    "collision_free_episode": 1 if int(collision_events[i]) == 0 else 0,
                    "near_misses_per_episode": int(near_misses[i]),
                    "blue_route_upper_crossings": int(self.metric_blue_route_upper_crossings[i].item()),
                    "blue_route_lower_crossings": int(self.metric_blue_route_lower_crossings[i].item()),
                    "red_route_upper_crossings": int(self.metric_red_route_upper_crossings[i].item()),
                    "red_route_lower_crossings": int(self.metric_red_route_lower_crossings[i].item()),
                    "blue_attack_upper_crossings": int(self.metric_blue_attack_upper_crossings[i].item()),
                    "blue_attack_lower_crossings": int(self.metric_blue_attack_lower_crossings[i].item()),
                    "blue_return_upper_crossings": int(self.metric_blue_return_upper_crossings[i].item()),
                    "blue_return_lower_crossings": int(self.metric_blue_return_lower_crossings[i].item()),
                    "blue_intercept_upper_crossings": int(self.metric_blue_intercept_upper_crossings[i].item()),
                    "blue_intercept_lower_crossings": int(self.metric_blue_intercept_lower_crossings[i].item()),
                    "red_attack_upper_crossings": int(self.metric_red_attack_upper_crossings[i].item()),
                    "red_attack_lower_crossings": int(self.metric_red_attack_lower_crossings[i].item()),
                    "red_return_upper_crossings": int(self.metric_red_return_upper_crossings[i].item()),
                    "red_return_lower_crossings": int(self.metric_red_return_lower_crossings[i].item()),
                    "red_intercept_upper_crossings": int(self.metric_red_intercept_upper_crossings[i].item()),
                    "red_intercept_lower_crossings": int(self.metric_red_intercept_lower_crossings[i].item()),
                    "mean_inter_robot_dist": mean_dist,
                    "zone_coverage": float(zone_coverage[i]),
                    "terminated": bool(term_np[i]),
                    "truncated": bool(trunc_np[i]),
                    "stalemate_truncated": bool(st_np[i]),
                    "stalemate_frac": float(stalemate_frac_np[i]),
                    "action_mask": action_mask_np[i],
                    "agent_alive": blue_alive_np[i],
                    "blue_alive": blue_alive_np[i],
                    "red_alive": red_alive_np[i],
                    "global_state": gs_np[i],
                }
            )
        return out
