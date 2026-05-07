"""RulesMixin methods for BatchedCTFCore."""
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


class _RulesMixin:
    def _untag_if_home(self) -> None:
        bdx = self.blue_x - self.blue_flag_home[:, None, 0]
        bdy = self.blue_y - self.blue_flag_home[:, None, 1]
        b_home = torch.sqrt(bdx * bdx + bdy * bdy + 1e-8) <= float(self.cfg.home_untag_radius_cells)
        rdx = self.red_x - self.red_flag_home[:, None, 0]
        rdy = self.red_y - self.red_flag_home[:, None, 1]
        r_home = torch.sqrt(rdx * rdx + rdy * rdy + 1e-8) <= float(self.cfg.home_untag_radius_cells)
        self.blue_tagged = self.blue_tagged & (~b_home)
        self.red_tagged = self.red_tagged & (~r_home)

    def _apply_aquaticus_tag_rules(
        self,
        blue_oob: torch.Tensor,
        red_oob: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Tagging uses a net/tag stack with a short tag channel:

          - Each defender in tag radius contributes +1 pressure on nearby opponents.
          - If pressure >= 2 is sustained for tag_channel_seconds, the target is tagged.
          - If pressure drops below 2, the per-agent channel timer resets.

        OOB does NOT cause tagging; it only drops the flag if the agent is carrying.
        """
        # OOB while carrying -> drop flag (no tag applied)
        blue_oob_carry = blue_oob & self.blue_carrying
        red_oob_carry = red_oob & self.red_carrying
        if blue_oob_carry.any():
            env = blue_oob_carry.any(dim=1)
            self.blue_carrying[blue_oob_carry] = False
            self.red_flag_pos[env] = self.red_flag_home[env]
        if red_oob_carry.any():
            env = red_oob_carry.any(dim=1)
            self.red_carrying[red_oob_carry] = False
            self.blue_flag_pos[env] = self.blue_flag_home[env]

        # Pairwise distances
        dx = self.blue_x[:, :, None] - self.red_x[:, None, :]
        dy = self.blue_y[:, :, None] - self.red_y[:, None, :]
        dist = torch.sqrt(dx * dx + dy * dy + 1e-8)
        in_tag_range = dist <= float(self.cfg.tag_range_cells)

        # Tagger must be untagged and on own side; target must be untagged and on opponent side
        blue_can_tag = (~self.blue_tagged) & self._is_on_home_side("blue", self.blue_x)
        red_can_tag = (~self.red_tagged) & self._is_on_home_side("red", self.red_x)
        red_on_blue_side = self._is_on_home_side("blue", self.red_x)
        blue_on_red_side = self._is_on_home_side("red", self.blue_x)
        red_targetable = (~self.red_tagged) & red_on_blue_side
        blue_targetable = (~self.blue_tagged) & blue_on_red_side

        blue_tags = in_tag_range & blue_can_tag[:, :, None] & red_targetable[:, None, :]
        red_tags = in_tag_range & red_can_tag[:, None, :] & blue_targetable[:, :, None]

        # Pressure counts: how many eligible taggers are in range of each target agent.
        # blue_pressure_on_red: (B, Nr), red_pressure_on_blue: (B, Nb)
        blue_pressure_on_red = blue_tags.sum(dim=1)
        red_pressure_on_blue = red_tags.sum(dim=2)

        dt = self.dt
        channel_T = float(getattr(self.cfg, "tag_channel_seconds", 1.0))

        # Tagging channel: accumulate time when pressure >= 2; reset when below.
        red_under_channel = blue_pressure_on_red >= 2
        blue_under_channel = red_pressure_on_blue >= 2

        self.red_tag_pressure_time = torch.where(
            red_under_channel,
            self.red_tag_pressure_time + dt,
            torch.zeros_like(self.red_tag_pressure_time),
        )
        self.blue_tag_pressure_time = torch.where(
            blue_under_channel,
            self.blue_tag_pressure_time + dt,
            torch.zeros_like(self.blue_tag_pressure_time),
        )

        newly_red_tagged = (
            (self.red_tag_pressure_time >= channel_T)
            & (~self.red_tagged)
            & red_targetable
        )
        newly_blue_tagged = (
            (self.blue_tag_pressure_time >= channel_T)
            & (~self.blue_tagged)
            & blue_targetable
        )

        # Clear timers for agents that just got tagged.
        if newly_red_tagged.any():
            self.red_tag_pressure_time = torch.where(
                newly_red_tagged,
                torch.zeros_like(self.red_tag_pressure_time),
                self.red_tag_pressure_time,
            )
        if newly_blue_tagged.any():
            self.blue_tag_pressure_time = torch.where(
                newly_blue_tagged,
                torch.zeros_like(self.blue_tag_pressure_time),
                self.blue_tag_pressure_time,
            )

        red_had_flag = newly_red_tagged & self.red_carrying
        blue_had_flag = newly_blue_tagged & self.blue_carrying

        self.red_tagged = self.red_tagged | newly_red_tagged
        self.blue_tagged = self.blue_tagged | newly_blue_tagged

        if red_had_flag.any():
            env = red_had_flag.any(dim=1)
            self.red_carrying[red_had_flag] = False
            self.blue_flag_pos[env] = self.blue_flag_home[env]
        if blue_had_flag.any():
            env = blue_had_flag.any(dim=1)
            self.blue_carrying[blue_had_flag] = False
            self.red_flag_pos[env] = self.red_flag_home[env]

        blue_tag_withflag = red_had_flag.sum(dim=1).to(torch.float32)
        blue_tag_total = newly_red_tagged.sum(dim=1).to(torch.float32)
        blue_tag_noflag = torch.clamp(blue_tag_total - blue_tag_withflag, min=0.0)
        red_tag_total = newly_blue_tagged.sum(dim=1).to(torch.float32)
        return blue_tag_noflag, blue_tag_withflag, red_tag_total

    def _respawn_timers(self) -> None:
        dt = self.dt
        for is_blue in (True, False):
            if is_blue:
                t, alive, x, y, h, s = self.blue_respawn, self.blue_alive, self.blue_x, self.blue_y, self.blue_heading, self.blue_speed
            else:
                t, alive, x, y, h, s = self.red_respawn, self.red_alive, self.red_x, self.red_y, self.red_heading, self.red_speed
            dead = ~alive
            t[dead] = torch.clamp(t[dead] - dt, min=0.0)
            revive = dead & (t <= 1e-6)
            if revive.any():
                if is_blue:
                    x_lo, x_hi = 0.0, max(1.0, float(self.cols // 3 - 1))
                else:
                    x_lo = max(0.0, float(self.cols - max(1, self.cols // 3)))
                    x_hi = float(max(0, self.cols - 1))
                x[revive] = self._rand_uniform((int(revive.sum().item()),), x_lo, x_hi)
                y[revive] = self._rand_uniform((int(revive.sum().item()),), 0.0, float(max(0, self.rows - 1)))
                h[revive] = 0.0 if is_blue else math.pi
                s[revive] = 0.0
                alive[revive] = True

    def _kill_agents(self, kill_blue: torch.Tensor, kill_red: torch.Tensor) -> None:
        # If carrier dies, flag returns home (Aquaticus semantics).
        killed_blue_carrier = kill_blue & self.blue_carrying
        killed_red_carrier = kill_red & self.red_carrying
        if killed_blue_carrier.any():
            env = killed_blue_carrier.any(dim=1)
            self.red_flag_pos[env] = self.red_flag_home[env]
        if killed_red_carrier.any():
            env = killed_red_carrier.any(dim=1)
            self.blue_flag_pos[env] = self.blue_flag_home[env]

        if kill_blue.any():
            self.blue_alive[kill_blue] = False
            self.blue_respawn[kill_blue] = 2.0
            self.blue_speed[kill_blue] = 0.0
            self.blue_carrying[kill_blue] = False
        if kill_red.any():
            self.red_alive[kill_red] = False
            self.red_respawn[kill_red] = 2.0
            self.red_speed[kill_red] = 0.0
            self.red_carrying[kill_red] = False

    def _apply_suppression(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dx = self.blue_x[:, :, None] - self.red_x[:, None, :]
        dy = self.blue_y[:, :, None] - self.red_y[:, None, :]
        dist = torch.sqrt(dx * dx + dy * dy + 1e-8)
        close = dist <= float(self.cfg.suppression_range_cells)
        close_blue_count = close.sum(dim=1)
        close_red_count = close.sum(dim=2)
        kill_red = (close_blue_count >= 2) & self.red_alive
        kill_blue = (close_red_count >= 2) & self.blue_alive
        red_had_flag = kill_red & self.red_carrying
        blue_had_flag = kill_blue & self.blue_carrying
        self._kill_agents(kill_blue, kill_red)
        return kill_blue, kill_red, blue_had_flag, red_had_flag

    def _apply_flag_rules(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Attach flags smoothly to their carriers when being carried, so the flag
        # position in observations and scoring is spatially consistent.
        blue_has_carrier = self.blue_carrying.any(dim=1)
        if blue_has_carrier.any():
            idx = torch.argmax(self.blue_carrying.to(torch.int64), dim=1)
            env = torch.where(blue_has_carrier)[0]
            self.red_flag_pos[env] = torch.stack(
                [self.blue_x[env, idx[env]], self.blue_y[env, idx[env]]], dim=1
            )
        red_has_carrier = self.red_carrying.any(dim=1)
        if red_has_carrier.any():
            idx = torch.argmax(self.red_carrying.to(torch.int64), dim=1)
            env = torch.where(red_has_carrier)[0]
            self.blue_flag_pos[env] = torch.stack(
                [self.red_x[env, idx[env]], self.red_y[env, idx[env]]], dim=1
            )

        b_to_red = torch.sqrt(
            (self.blue_x - self.red_flag_pos[:, None, 0]) ** 2
            + (self.blue_y - self.red_flag_pos[:, None, 1]) ** 2
            + 1e-8
        )
        r_to_blue = torch.sqrt(
            (self.red_x - self.blue_flag_pos[:, None, 0]) ** 2
            + (self.red_y - self.blue_flag_pos[:, None, 1]) ** 2
            + 1e-8
        )

        grab_r = 1.2
        # Grace period: no score for grab/capture in the first few steps (avoids spurious
        # points from spawn/initial state or first-frame edge cases in the viewer).
        grace_steps = max(0, int(getattr(self.cfg, "score_grace_steps", 10)))
        grace_ok = (self.step_count >= grace_steps).to(torch.bool)
        red_score_allowed = ~self._phase_tensor_equals(("OP1", "OP2"))

        # Both flags can be taken at once: blue can grab red flag regardless of whether
        # red has blue's flag, and vice versa. Already-carried flags are not new grab
        # events; the flag position is just attached to the current carrier above.
        blue_can_grab_flag = ~self.blue_carrying.any(dim=1)
        red_can_grab_flag = ~self.red_carrying.any(dim=1)
        blue_grab_candidates = (
            (b_to_red <= grab_r)
            & self.blue_alive
            & (~self.blue_tagged)
            & blue_can_grab_flag[:, None]
        )
        red_grab_candidates = (
            (r_to_blue <= grab_r)
            & self.red_alive
            & (~self.red_tagged)
            & red_can_grab_flag[:, None]
        )
        blue_grab_env = blue_grab_candidates.any(dim=1)
        red_grab_env = red_grab_candidates.any(dim=1)

        grab_delta = get_grab_score_delta(self.rules_profile)
        if blue_grab_env.any():
            idx = torch.argmax(blue_grab_candidates.to(torch.int64), dim=1)
            env_idx = torch.where(blue_grab_env)[0]
            self.blue_carrying[env_idx] = False
            self.blue_carrying[env_idx, idx[env_idx]] = True
            if grab_delta > 0:
                score_env = env_idx[grace_ok[env_idx]]
                if score_env.numel() > 0:
                    self.blue_score[score_env] += grab_delta
            self.red_flag_pos[env_idx] = torch.stack(
                [self.blue_x[env_idx, idx[env_idx]], self.blue_y[env_idx, idx[env_idx]]],
                dim=1,
            )

        if red_grab_env.any():
            idx = torch.argmax(red_grab_candidates.to(torch.int64), dim=1)
            env_idx = torch.where(red_grab_env)[0]
            self.red_carrying[env_idx] = False
            self.red_carrying[env_idx, idx[env_idx]] = True
            if grab_delta > 0:
                score_env = env_idx[grace_ok[env_idx] & red_score_allowed[env_idx]]
                if score_env.numel() > 0:
                    self.red_score[score_env] += grab_delta
            self.blue_flag_pos[env_idx] = torch.stack(
                [self.red_x[env_idx, idx[env_idx]], self.red_y[env_idx, idx[env_idx]]],
                dim=1,
            )

        b_home_dist = torch.sqrt(
            (self.blue_x - self.blue_flag_home[:, None, 0]) ** 2
            + (self.blue_y - self.blue_flag_home[:, None, 1]) ** 2
            + 1e-8
        )
        r_home_dist = torch.sqrt(
            (self.red_x - self.red_flag_home[:, None, 0]) ** 2
            + (self.red_y - self.red_flag_home[:, None, 1]) ** 2
            + 1e-8
        )
        cap_r = 1.2
        blue_capture_contact = self.blue_alive & self.blue_carrying & (~self.blue_tagged) & (b_home_dist <= cap_r)
        red_capture_contact = self.red_alive & self.red_carrying & (~self.red_tagged) & (r_home_dist <= cap_r)
        self.blue_home_contact_frames = torch.where(
            blue_capture_contact,
            torch.clamp(self.blue_home_contact_frames + 1, max=1000),
            torch.zeros_like(self.blue_home_contact_frames),
        )
        self.red_home_contact_frames = torch.where(
            red_capture_contact,
            torch.clamp(self.red_home_contact_frames + 1, max=1000),
            torch.zeros_like(self.red_home_contact_frames),
        )
        needed = max(1, int(getattr(self.cfg, "capture_confirm_frames", 2)))
        blue_capture_now = blue_capture_contact & (self.blue_home_contact_frames >= needed)
        red_capture_now = red_capture_contact & (self.red_home_contact_frames >= needed)

        b_cap_env = blue_capture_now.any(dim=1)
        r_cap_env = red_capture_now.any(dim=1)
        cap_delta_b = get_capture_score_delta(self.rules_profile)
        cap_delta_r = get_capture_score_delta(self.rules_profile)
        if b_cap_env.any():
            award_b = b_cap_env & grace_ok
            if award_b.any():
                self.blue_score[award_b] += cap_delta_b
            self.blue_carrying[b_cap_env] = False
            self.red_flag_pos[b_cap_env] = self.red_flag_home[b_cap_env]
            self.blue_home_contact_frames[b_cap_env] = 0
        if r_cap_env.any():
            award_r = r_cap_env & grace_ok & red_score_allowed
            if award_r.any():
                self.red_score[award_r] += cap_delta_r
            self.red_carrying[r_cap_env] = False
            self.blue_flag_pos[r_cap_env] = self.blue_flag_home[r_cap_env]
            self.red_home_contact_frames[r_cap_env] = 0
        return blue_grab_env, red_grab_env, b_cap_env, r_cap_env

    def _build_targets_from_action(self, macro: torch.Tensor, target: torch.Tensor, side: str = "blue") -> Tuple[torch.Tensor, torch.Tensor]:
        side_t = self._side_tensors(side)
        own_flag_home = side_t["own_flag_home"]
        enemy_flag = side_t["enemy_flag"]
        own_carrying = side_t["own_carrying"]
        t_xy = self._decode_targets(target, side=side)
        tx, ty = t_xy[..., 0], t_xy[..., 1]
        get_flag = macro == MacroAction.GET_FLAG
        go_home = macro == MacroAction.GO_HOME
        tx = torch.where(get_flag, enemy_flag[:, None, 0], tx)
        ty = torch.where(get_flag, enemy_flag[:, None, 1], ty)
        tx = torch.where(go_home, own_flag_home[:, None, 0], tx)
        ty = torch.where(go_home, own_flag_home[:, None, 1], ty)
        tx = torch.where(own_carrying, own_flag_home[:, None, 0], tx)
        ty = torch.where(own_carrying, own_flag_home[:, None, 1], ty)
        return tx, ty

    def _redirect_tagged_to_home(
        self,
        btx: torch.Tensor,
        bty: torch.Tensor,
        rtx: torch.Tensor,
        rty: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        btx = torch.where(self.blue_tagged, self.blue_flag_home[:, None, 0], btx)
        bty = torch.where(self.blue_tagged, self.blue_flag_home[:, None, 1], bty)
        rtx = torch.where(self.red_tagged, self.red_flag_home[:, None, 0], rtx)
        rty = torch.where(self.red_tagged, self.red_flag_home[:, None, 1], rty)
        return btx, bty, rtx, rty
