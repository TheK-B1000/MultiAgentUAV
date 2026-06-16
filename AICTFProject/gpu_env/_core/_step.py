"""Step orchestration for BatchedCTFCore."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch

from macro_actions import MacroAction


class _StepMixin:
    def step(
        self,
        blue_action_flat: torch.Tensor,
        *,
        tensor_obs: bool = False,
        red_action_flat: Optional[torch.Tensor] = None,
    ):
        self._apply_profile_runtime()
        macro, targ = self._advance_blue_macros(blue_action_flat)
        snapshot = self._snapshot_step_state()
        targets = self._resolve_step_targets(macro, targ, red_action_flat)
        movement = self._advance_dynamics_phase(targets, snapshot)
        combat = self._advance_combat_phase(movement["blue_oob"], movement["red_oob"])
        mines = self._advance_mines_phase(macro, targets["red_macro"])
        flags = self._advance_flags_phase(snapshot)
        self._record_action_success(macro, targets, mines, flags, snapshot)
        rewards = self._compute_step_reward_components(macro, targets, mines, flags, combat, movement, snapshot)
        terminal = self._advance_episode_end(flags, combat, rewards)
        return self._assemble_step_outputs(tensor_obs, rewards, terminal)

    def _advance_blue_macros(self, blue_action_flat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if blue_action_flat.device != self.device:
            blue_action_flat = blue_action_flat.to(self.device)
        n_act_exp = int(self.B * self.Nb * 2)
        if int(blue_action_flat.numel()) != n_act_exp:
            raise ValueError(
                f"BatchedCTFCore.step: expected {n_act_exp} action ints (B={self.B}, Nb={self.Nb}), "
                f"got numel={int(blue_action_flat.numel())} shape={tuple(blue_action_flat.shape)}"
            )
        a = blue_action_flat.reshape(self.B, self.Nb, 2)
        requested_macro = torch.remainder(a[..., 0].long(), self.cfg.n_macros)
        requested_targ = torch.remainder(a[..., 1].long(), self.cfg.n_targets)
        new_commit = self.blue_commit_ticks_left <= 0
        self.blue_commit_macro = torch.where(new_commit, requested_macro, self.blue_commit_macro)
        self.blue_commit_target = torch.where(new_commit, requested_targ, self.blue_commit_target)
        self.blue_commit_ticks_left = torch.where(new_commit, self._macro_commit_ticks(requested_macro), self.blue_commit_ticks_left)
        self.blue_commit_success = torch.where(new_commit, torch.zeros_like(self.blue_commit_success), self.blue_commit_success)
        return self.blue_commit_macro, self.blue_commit_target

    def _snapshot_step_state(self) -> Dict[str, torch.Tensor]:
        return {
            "prev_blue_x": self.blue_x.clone(),
            "prev_blue_y": self.blue_y.clone(),
            "prev_red_x": self.red_x.clone(),
            "prev_red_y": self.red_y.clone(),
            "prev_blue_alive": self.blue_alive.clone(),
            "prev_red_alive": self.red_alive.clone(),
            "prev_blue_carrying": self.blue_carrying.clone(),
            "prev_red_carrying": self.red_carrying.clone(),
            "prev_red_mine_charges": self.red_mine_charges.clone(),
            "prev_blue_score": self.blue_score.clone(),
            "prev_red_score": self.red_score.clone(),
        }

    def _resolve_step_targets(
        self,
        macro: torch.Tensor,
        targ: torch.Tensor,
        red_action_flat: Optional[torch.Tensor],
    ) -> Dict[str, Optional[torch.Tensor] | torch.Tensor]:
        if self.blue_scripted:
            btx, bty = self._get_scripted_targets("blue")
        else:
            btx, bty = self._build_targets_from_action(macro, targ, side="blue")
        red_macro: Optional[torch.Tensor] = None
        rtx, rty = self._red_scripted_actions()
        if red_action_flat is not None:
            rtx, rty, red_macro, red_control_mask = self._apply_red_action_commit(red_action_flat)
        else:
            red_control_mask = self._get_red_control_mask()
            if red_control_mask.any():
                red_policy_actions = self._get_red_snapshot_actions(red_control_mask)
                if red_policy_actions is not None:
                    red_requested_macro, red_requested_targ = red_policy_actions
                    snapshot_agent_mask = red_control_mask[:, None]
                    new_red_commit = snapshot_agent_mask & (self.red_commit_ticks_left <= 0)
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
                    red_targ = self.red_commit_target
                    red_snapshot_tx, red_snapshot_ty = self._build_targets_from_action(red_macro, red_targ, side="red")
                    rtx = torch.where(snapshot_agent_mask, red_snapshot_tx, rtx)
                    rty = torch.where(snapshot_agent_mask, red_snapshot_ty, rty)
        btx, bty, rtx, rty = self._redirect_tagged_to_home(btx, bty, rtx, rty)
        btx, bty = self._route_targets_around_obstacles(self.blue_x, self.blue_y, btx, bty)
        rtx, rty = self._route_targets_around_obstacles(self.red_x, self.red_y, rtx, rty)
        return {"btx": btx, "bty": bty, "rtx": rtx, "rty": rty, "red_macro": red_macro, "red_control_mask": red_control_mask}

    def _advance_dynamics_phase(self, targets: Dict[str, torch.Tensor], snapshot: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        bscale = self.rt_blue_speed_scale.reshape(self.B, 1).expand_as(self.blue_speed)
        blue_speed_cap = torch.full_like(self.blue_speed, float(self.cfg.max_speed_cps)) * bscale
        B = self.B
        rm = self.red_speed_mult.reshape(-1).to(device=self.red_speed.device, dtype=self.red_speed.dtype)
        if rm.numel() < B:
            rm = torch.cat(
                [rm, torch.ones(B - rm.numel(), device=self.red_speed.device, dtype=self.red_speed.dtype)],
                dim=0,
            )
        elif rm.numel() > B:
            rm = rm[:B]
        red_speed_cap = torch.full_like(self.red_speed, float(self.cfg.max_speed_cps)) * rm[:, None]

        self.blue_x, self.blue_y, self.blue_heading, self.blue_speed, blue_oob, yaw_cmd_blue = self._integrate_side(
            self.blue_x, self.blue_y, self.blue_heading, self.blue_speed, self.blue_alive, targets["btx"], targets["bty"], speed_cap=blue_speed_cap
        )
        self.red_x, self.red_y, self.red_heading, self.red_speed, red_oob, _ = self._integrate_side(
            self.red_x, self.red_y, self.red_heading, self.red_speed, self.red_alive, targets["rtx"], targets["rty"], speed_cap=red_speed_cap
        )
        self.blue_x, self.blue_y, self.blue_speed, blue_wall_hit = self._revert_obstacle_hits(
            snapshot["prev_blue_x"],
            snapshot["prev_blue_y"],
            self.blue_x,
            self.blue_y,
            self.blue_speed,
            self.blue_alive,
        )
        self.red_x, self.red_y, self.red_speed, red_wall_hit = self._revert_obstacle_hits(
            snapshot["prev_red_x"],
            snapshot["prev_red_y"],
            self.red_x,
            self.red_y,
            self.red_speed,
            self.red_alive,
        )
        if blue_wall_hit.any() or red_wall_hit.any():
            self.metric_obstacle_collision_events += (
                blue_wall_hit.sum(dim=1).to(torch.int32)
                + red_wall_hit.sum(dim=1).to(torch.int32)
            )
        pre_guard_blue_x = self.blue_x.clone()
        pre_guard_blue_y = self.blue_y.clone()
        pre_guard_red_x = self.red_x.clone()
        pre_guard_red_y = self.red_y.clone()
        self._apply_avoid_collision_guard(
            snapshot["prev_blue_x"], snapshot["prev_blue_y"], snapshot["prev_red_x"], snapshot["prev_red_y"]
        )
        self.blue_x, self.blue_y, self.blue_speed, blue_guard_hit = self._revert_obstacle_hits(
            pre_guard_blue_x,
            pre_guard_blue_y,
            self.blue_x,
            self.blue_y,
            self.blue_speed,
            self.blue_alive,
        )
        self.red_x, self.red_y, self.red_speed, red_guard_hit = self._revert_obstacle_hits(
            pre_guard_red_x,
            pre_guard_red_y,
            self.red_x,
            self.red_y,
            self.red_speed,
            self.red_alive,
        )
        if blue_guard_hit.any() or red_guard_hit.any():
            self.metric_obstacle_collision_events += (
                blue_guard_hit.sum(dim=1).to(torch.int32)
                + red_guard_hit.sum(dim=1).to(torch.int32)
            )
        return {"blue_oob": blue_oob, "red_oob": red_oob, "yaw_cmd_blue": yaw_cmd_blue}

    def _advance_combat_phase(self, blue_oob: torch.Tensor, red_oob: torch.Tensor) -> Dict[str, torch.Tensor]:
        if bool(self.cfg.aquaticus_profile) or self.rules_profile in ("AQUATICUS_2024", "OURS", "OURS_PLUS"):
            blue_tag_noflag, blue_tag_withflag, red_tag_total = self._apply_aquaticus_tag_rules(blue_oob, red_oob)
            self._untag_if_home()
        else:
            kill_blue, kill_red, blue_had_flag, red_had_flag = self._apply_suppression()
            blue_tag_noflag = (kill_red & (~red_had_flag)).sum(dim=1).to(torch.float32)
            blue_tag_withflag = (kill_red & red_had_flag).sum(dim=1).to(torch.float32)
            red_tag_total = kill_blue.sum(dim=1).to(torch.float32)
            self._respawn_timers()
        return {"blue_tag_noflag": blue_tag_noflag, "blue_tag_withflag": blue_tag_withflag, "red_tag_total": red_tag_total}

    def _advance_mines_phase(self, macro: torch.Tensor, red_macro: Optional[torch.Tensor]) -> Dict[str, torch.Tensor]:
        blue_mine_pickups, blue_mine_pickup_agents = self._apply_mine_pickups(macro, red_macro)
        blue_mine_placements, blue_mine_placement_agents = self._apply_mine_placement(macro, red_macro)
        blue_mine_tags, red_mine_tags = self._apply_mine_triggers()
        self._untag_if_home()
        return {
            "blue_mine_pickups": blue_mine_pickups,
            "blue_mine_pickup_agents": blue_mine_pickup_agents,
            "blue_mine_placements": blue_mine_placements,
            "blue_mine_placement_agents": blue_mine_placement_agents,
            "blue_mine_tags": blue_mine_tags,
            "red_mine_tags": red_mine_tags,
        }

    def _advance_flags_phase(self, snapshot: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        blue_grab_env, red_grab_env, blue_cap_env, red_cap_env = self._apply_flag_rules()
        first_score_mask = (self.blue_score > snapshot["prev_blue_score"]) | (self.red_score > snapshot["prev_red_score"])
        self._update_episode_metrics(
            first_score_mask,
            prev_blue_x=snapshot["prev_blue_x"],
            prev_blue_y=snapshot["prev_blue_y"],
            prev_red_x=snapshot["prev_red_x"],
            prev_red_y=snapshot["prev_red_y"],
        )
        return {
            "blue_grab_env": blue_grab_env,
            "red_grab_env": red_grab_env,
            "blue_cap_env": blue_cap_env,
            "red_cap_env": red_cap_env,
            "blue_grab_agents": self.blue_carrying & (~snapshot["prev_blue_carrying"]),
            "blue_cap_agents": snapshot["prev_blue_carrying"] & (~self.blue_carrying) & blue_cap_env[:, None],
            "red_grab_agents": self.red_carrying & (~snapshot["prev_red_carrying"]),
            "red_cap_agents": snapshot["prev_red_carrying"] & (~self.red_carrying) & red_cap_env[:, None],
            "red_mine_pickup_agents": self.red_mine_charges > snapshot["prev_red_mine_charges"],
            "red_mine_placement_agents": self.red_mine_charges < snapshot["prev_red_mine_charges"],
        }

    def _record_action_success(
        self,
        macro: torch.Tensor,
        targets: Dict[str, Optional[torch.Tensor] | torch.Tensor],
        mines: Dict[str, torch.Tensor],
        flags: Dict[str, torch.Tensor],
        snapshot: Dict[str, torch.Tensor],
    ) -> None:
        commit_target_xy = self._decode_targets(self.blue_commit_target)
        commit_dist = torch.sqrt(
            (self.blue_x - commit_target_xy[..., 0]) ** 2
            + (self.blue_y - commit_target_xy[..., 1]) ** 2
            + 1e-8
        )
        action_success = torch.zeros_like(self.blue_commit_success)
        action_success = action_success | ((self.blue_commit_macro == int(MacroAction.GO_TO)) & (commit_dist <= float(self.cfg.macro_arrival_radius_cells)))
        action_success = action_success | ((self.blue_commit_macro == int(MacroAction.GRAB_MINE)) & mines["blue_mine_pickup_agents"])
        action_success = action_success | ((self.blue_commit_macro == int(MacroAction.GET_FLAG)) & flags["blue_grab_agents"])
        action_success = action_success | ((self.blue_commit_macro == int(MacroAction.PLACE_MINE)) & mines["blue_mine_placement_agents"])
        action_success = action_success | ((self.blue_commit_macro == int(MacroAction.GO_HOME)) & flags["blue_cap_agents"])
        self.blue_commit_success = self.blue_commit_success | action_success
        red_macro = targets["red_macro"]
        if red_macro is not None:
            red_control_mask = targets["red_control_mask"]
            red_commit_target_xy = self._decode_targets(self.red_commit_target, side="red")
            red_commit_dist = torch.sqrt(
                (self.red_x - red_commit_target_xy[..., 0]) ** 2
                + (self.red_y - red_commit_target_xy[..., 1]) ** 2
                + 1e-8
            )
            red_action_success = torch.zeros_like(self.red_commit_success)
            red_action_success = red_action_success | (
                (self.red_commit_macro == int(MacroAction.GO_TO))
                & (red_commit_dist <= float(self.cfg.macro_arrival_radius_cells))
            )
            red_action_success = red_action_success | (
                (self.red_commit_macro == int(MacroAction.GRAB_MINE)) & flags["red_mine_pickup_agents"]
            )
            red_action_success = red_action_success | (
                (self.red_commit_macro == int(MacroAction.GET_FLAG)) & flags["red_grab_agents"]
            )
            red_action_success = red_action_success | (
                (self.red_commit_macro == int(MacroAction.PLACE_MINE)) & flags["red_mine_placement_agents"]
            )
            red_action_success = red_action_success | (
                (self.red_commit_macro == int(MacroAction.GO_HOME)) & flags["red_cap_agents"]
            )
            self.red_commit_success = self.red_commit_success | (red_action_success & red_control_mask[:, None])

    def _compute_step_reward_components(
        self,
        macro: torch.Tensor,
        targets: Dict[str, Optional[torch.Tensor] | torch.Tensor],
        mines: Dict[str, torch.Tensor],
        flags: Dict[str, torch.Tensor],
        combat: Dict[str, torch.Tensor],
        movement: Dict[str, torch.Tensor],
        snapshot: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        sparse_points = self._sparse_reward_points(
            flags["blue_cap_env"], flags["red_cap_env"], combat["blue_tag_noflag"], combat["blue_tag_withflag"], combat["red_tag_total"], movement["blue_oob"],
            blue_mine_tags=mines["blue_mine_tags"], red_mine_tags=mines["red_mine_tags"], red_oob=movement["red_oob"],
        )
        blue_kill_count = combat["blue_tag_noflag"] + combat["blue_tag_withflag"] + mines["blue_mine_tags"]
        roff = torch.zeros((self.B,), dtype=torch.float32, device=self.device)
        roff += float(self.cfg.flag_pickup_reward) * flags["blue_grab_agents"].sum(dim=1).to(torch.float32)
        roff += float(self.cfg.flag_carry_home_reward) * flags["blue_cap_agents"].sum(dim=1).to(torch.float32)
        roff += float(self.cfg.enabled_mine_reward) * mines["blue_mine_placement_agents"].sum(dim=1).to(torch.float32)
        roff += float(self.cfg.enemy_mav_kill_reward) * blue_kill_count
        red_kill_count = combat["red_tag_total"] + mines["red_mine_tags"]
        roff -= float(self.cfg.flag_pickup_reward) * flags["red_grab_agents"].sum(dim=1).to(torch.float32)
        roff -= float(self.cfg.flag_carry_home_reward) * flags["red_cap_agents"].sum(dim=1).to(torch.float32)
        roff -= float(self.cfg.enabled_mine_reward) * flags["red_mine_placement_agents"].sum(dim=1).to(torch.float32)
        roff -= float(self.cfg.enemy_mav_kill_reward) * red_kill_count
        self.blue_commit_ticks_left = torch.clamp(self.blue_commit_ticks_left - 1, min=0)
        ended_commit = self.blue_commit_success | (self.blue_commit_ticks_left <= 0) | (~self.blue_alive) | self.blue_tagged
        failed_commit = ended_commit & (~self.blue_commit_success) & snapshot["prev_blue_alive"]
        rfail = float(self.cfg.action_failed_punishment) * failed_commit.sum(dim=1).to(torch.float32)
        self.blue_commit_ticks_left = torch.where(ended_commit, torch.zeros_like(self.blue_commit_ticks_left), self.blue_commit_ticks_left)
        self.blue_commit_success = torch.where(ended_commit, torch.zeros_like(self.blue_commit_success), self.blue_commit_success)
        if targets["red_macro"] is not None:
            red_control_mask = targets["red_control_mask"]
            self.red_commit_ticks_left = torch.clamp(self.red_commit_ticks_left - 1, min=0)
            ended_red_commit = (
                self.red_commit_success | (self.red_commit_ticks_left <= 0) | (~self.red_alive) | self.red_tagged
            ) & red_control_mask[:, None]
            self.red_commit_ticks_left = torch.where(
                ended_red_commit,
                torch.zeros_like(self.red_commit_ticks_left),
                self.red_commit_ticks_left,
            )
            self.red_commit_success = torch.where(
                ended_red_commit,
                torch.zeros_like(self.red_commit_success),
                self.red_commit_success,
            )
        rpbrs = self._pbrs_reward(
            snapshot["prev_blue_x"],
            snapshot["prev_blue_y"],
            snapshot["prev_blue_carrying"],
            prev_red_x=snapshot["prev_red_x"],
            prev_red_y=snapshot["prev_red_y"],
            prev_red_carrying=snapshot["prev_red_carrying"],
        )
        rteam = self._team_coordination_reward(snapshot["prev_blue_x"], snapshot["prev_blue_y"], movement["yaw_cmd_blue"])
        return {"sparse_points": sparse_points, "roff": roff, "rfail": rfail, "rpbrs": rpbrs, "rteam": rteam}

    def _advance_episode_end(
        self,
        flags: Dict[str, torch.Tensor],
        combat: Dict[str, torch.Tensor],
        rewards: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        self.step_count += 1
        self.sim_step_count += 1
        event_happened = (
            flags["blue_grab_env"]
            | flags["red_grab_env"]
            | flags["blue_cap_env"]
            | flags["red_cap_env"]
            | (combat["blue_tag_noflag"] > 0.0)
            | (combat["blue_tag_withflag"] > 0.0)
            | (combat["red_tag_total"] > 0.0)
            | (torch.abs(rewards["sparse_points"]) > 0.0)
            | (torch.abs(rewards["roff"]) > 0.0)
        )
        low_progress = torch.abs(self._last_dense_progress) < float(self.cfg.stalemate_progress_eps)
        no_event = ~event_happened
        self.stalemate_steps = torch.where(no_event & low_progress, self.stalemate_steps + 1, torch.zeros_like(self.stalemate_steps))
        stalemate_trigger = self.stalemate_steps >= int(self.cfg.stalemate_max_steps)

        terminated = (self.blue_score >= self.score_limit) | (self.red_score >= self.score_limit)
        truncated = (self.step_count >= self.max_steps) | (self.sim_step_count >= self.max_sim_steps) | stalemate_trigger
        self.done = terminated | truncated
        self.truncated = truncated

        rterm = torch.zeros((self.B,), dtype=torch.float32, device=self.device)
        done = terminated | truncated
        rterm = torch.where(
            done & (self.blue_score > self.red_score),
            torch.full_like(rterm, float(self.cfg.win_team_reward)),
            rterm,
        )
        rterm = torch.where(
            done & (self.blue_score < self.red_score),
            torch.full_like(rterm, float(self.cfg.lose_team_punish)),
            rterm,
        )
        rterm = torch.where(
            done & (self.blue_score == self.red_score),
            torch.full_like(rterm, float(self.cfg.draw_team_penalty)),
            rterm,
        )
        reward = self._reward_total(
            rterm,
            rewards["roff"],
            rewards["rpbrs"],
            rewards["rteam"],
            rewards["sparse_points"],
            rewards["rfail"],
            stalemate_trigger,
        )
        reward_sparse = float(self.cfg.sparse_weight) * (rewards["sparse_points"] / 100.0)
        info = self._build_info(
            dense=rewards["rpbrs"] + rewards["rteam"],
            sparse_points=rewards["sparse_points"],
            stalemate=stalemate_trigger,
            reward_terminal=rterm,
            reward_offense=rewards["roff"],
            reward_pbrs=rewards["rpbrs"],
            reward_team=rewards["rteam"],
            reward_sparse=reward_sparse,
            reward_failure=rewards["rfail"],
            reward_total=reward,
            terminated=terminated,
            truncated=truncated,
        )
        return {"reward": reward, "terminated": terminated, "truncated": truncated, "info": info}

    def _assemble_step_outputs(self, tensor_obs: bool, rewards: Dict[str, torch.Tensor], terminal: Dict[str, torch.Tensor]):
        obs_t = self.get_obs_tensors()
        if tensor_obs:
            return obs_t, terminal["reward"], terminal["terminated"], terminal["truncated"], terminal["info"]
        return (
            {k: v.detach().cpu().numpy().astype(np.float32) for k, v in obs_t.items()},
            terminal["reward"].detach().cpu().numpy().astype(np.float32),
            terminal["terminated"].detach().cpu().numpy(),
            terminal["truncated"].detach().cpu().numpy(),
            terminal["info"],
        )
