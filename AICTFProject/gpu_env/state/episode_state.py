"""Episode bookkeeping tensors and reset logic.

Allocates step counters, done/truncated flags, stalemate counter, and macro
commitment tensors.  ``reset_all`` and ``reset_indices`` implement the full
per-episode re-initialization sequence.
"""
from __future__ import annotations

import torch


class _EpisodeStateMixin:
    """Manages episode lifecycle: allocation, full reset, and partial env-mask reset."""

    def _alloc_episode_state(
        self,
        B: int,
        Nb: int,
        Nr: int,
        dev: torch.device,
    ) -> None:
        self.step_count = torch.zeros((B,), dtype=torch.int32, device=dev)
        self.sim_step_count = torch.zeros((B,), dtype=torch.int32, device=dev)
        self.done = torch.zeros((B,), dtype=torch.bool, device=dev)
        self.truncated = torch.zeros((B,), dtype=torch.bool, device=dev)
        self.stalemate_steps = torch.zeros((B,), dtype=torch.int32, device=dev)
        self.blue_commit_macro = torch.zeros((B, Nb), dtype=torch.int64, device=dev)
        self.blue_commit_target = torch.zeros((B, Nb), dtype=torch.int64, device=dev)
        self.blue_commit_ticks_left = torch.zeros((B, Nb), dtype=torch.int32, device=dev)
        self.blue_commit_success = torch.zeros((B, Nb), dtype=torch.bool, device=dev)
        self.red_commit_macro = torch.zeros((B, Nr), dtype=torch.int64, device=dev)
        self.red_commit_target = torch.zeros((B, Nr), dtype=torch.int64, device=dev)
        self.red_commit_ticks_left = torch.zeros((B, Nr), dtype=torch.int32, device=dev)
        self.red_commit_success = torch.zeros((B, Nr), dtype=torch.bool, device=dev)

    def reset_all(self) -> None:
        self._phase_tensor_cache.clear()
        self._red_control_mask_dirty = True
        mask = torch.ones((self.B,), dtype=torch.bool, device=self.device)
        self.reset_indices(mask)

    def reset_indices(self, env_mask: torch.Tensor) -> None:
        self._phase_tensor_cache.clear()
        self._red_control_mask_dirty = True
        idx = torch.where(env_mask)[0]
        if idx.numel() == 0:
            return
        self.done[idx] = False
        self.truncated[idx] = False
        self.step_count[idx] = 0
        self.sim_step_count[idx] = 0
        self.stalemate_steps[idx] = 0
        self.blue_score[idx] = 0
        self.red_score[idx] = 0
        self._reset_map_layout(env_mask)
        self.blue_flag_pos[idx] = self.blue_flag_home[idx].clone()
        self.red_flag_pos[idx] = self.red_flag_home[idx].clone()
        self.blue_carrying[idx] = False
        self.red_carrying[idx] = False
        self.blue_tagged[idx] = False
        self.red_tagged[idx] = False
        self._last_dense_progress[idx] = 0.0
        self.red_deception_prob[idx] = 0.0
        self.red_speed_mult[idx] = 1.0
        self.red_attacker_style[idx] = 0
        self.red_defender_style[idx] = 0
        self.red_role_switch_prob[idx] = 0.0
        self.red_coord_ticks_left[idx] = 0
        red_is_op4 = torch.as_tensor(
            [
                str(self._opponent_kind[i]).upper() == "SCRIPTED"
                and str(self._opponent_key[i]).upper() == "OP4"
                for i in idx.detach().cpu().tolist()
            ],
            device=self.device,
            dtype=torch.bool,
        )
        role_flip_p = torch.where(
            red_is_op4,
            torch.full((idx.numel(),), 0.55, dtype=torch.float32, device=self.device),
            torch.full((idx.numel(),), 0.35, dtype=torch.float32, device=self.device),
        )
        self.red_script_role_flip[idx] = (
            torch.rand((idx.numel(),), generator=self._rng, device=self.device) < role_flip_p
        )
        self.red_script_lane_sign[idx] = torch.where(
            torch.rand((idx.numel(),), generator=self._rng, device=self.device) < 0.5,
            torch.tensor(-1.0, dtype=torch.float32, device=self.device),
            torch.tensor(1.0, dtype=torch.float32, device=self.device),
        )
        op4_idx = idx[red_is_op4]
        if op4_idx.numel() > 0:
            guard_x_low = max(0.0, float(self.cols) - 9.0)
            guard_x_high = max(guard_x_low + 0.5, float(self.cols) - 2.5)
            self.red_script_guard_x[op4_idx] = self._rand_uniform(
                (op4_idx.numel(),), guard_x_low, guard_x_high
            )
            self.red_script_guard_y[op4_idx] = self._rand_uniform(
                (op4_idx.numel(),), 3.5, 16.0
            )
        non_op4_idx = idx[~red_is_op4]
        if non_op4_idx.numel() > 0:
            self.red_script_guard_x[non_op4_idx] = self._rand_uniform(
                (non_op4_idx.numel(),), 14.5, 17.5
            )
            self.red_script_guard_y[non_op4_idx] = self._rand_uniform(
                (non_op4_idx.numel(),), 7.0, 13.0
            )
        self.blue_home_contact_frames[idx] = 0
        self.red_home_contact_frames[idx] = 0
        self.blue_commit_macro[idx] = 0
        self.blue_commit_target[idx] = 0
        self.blue_commit_ticks_left[idx] = 0
        self.blue_commit_success[idx] = False
        self.red_commit_macro[idx] = 0
        self.red_commit_target[idx] = 0
        self.red_commit_ticks_left[idx] = 0
        self.red_commit_success[idx] = False
        self.blue_mine_x[idx] = 0.0
        self.blue_mine_y[idx] = 0.0
        self.blue_mine_active[idx] = False
        self.red_mine_x[idx] = 0.0
        self.red_mine_y[idx] = 0.0
        self.red_mine_active[idx] = False
        self.blue_mine_charges[idx] = 0
        self.red_mine_charges[idx] = 0
        self.pickup_active[idx] = True
        self.pickup_respawn[idx] = 0
        self.red_tag_pressure_time[idx] = 0.0
        self.blue_tag_pressure_time[idx] = 0.0
        self.metric_time_to_first_score[idx] = -1.0
        self.metric_inter_robot_dist_sum[idx] = 0.0
        self.metric_inter_robot_dist_count[idx] = 0
        self.metric_collision_events[idx] = 0
        self.metric_obstacle_collision_events[idx] = 0
        self.metric_near_misses[idx] = 0
        self.metric_blue_route_upper_crossings[idx] = 0
        self.metric_blue_route_lower_crossings[idx] = 0
        self.metric_red_route_upper_crossings[idx] = 0
        self.metric_red_route_lower_crossings[idx] = 0
        self.metric_blue_attack_upper_crossings[idx] = 0
        self.metric_blue_attack_lower_crossings[idx] = 0
        self.metric_blue_return_upper_crossings[idx] = 0
        self.metric_blue_return_lower_crossings[idx] = 0
        self.metric_blue_intercept_upper_crossings[idx] = 0
        self.metric_blue_intercept_lower_crossings[idx] = 0
        self.metric_red_attack_upper_crossings[idx] = 0
        self.metric_red_attack_lower_crossings[idx] = 0
        self.metric_red_return_upper_crossings[idx] = 0
        self.metric_red_return_lower_crossings[idx] = 0
        self.metric_red_intercept_upper_crossings[idx] = 0
        self.metric_red_intercept_lower_crossings[idx] = 0
        self.metric_blue_zone_visited[idx] = False
        self._reset_navigation_telemetry(idx)
        self._apply_opponent_params_for_mask(env_mask)
        self._respawn_side(blue=True, env_mask=env_mask)
        self._respawn_side(blue=False, env_mask=env_mask)
        self._apply_train_domain_randomization(env_mask)
