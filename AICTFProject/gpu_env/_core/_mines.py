"""MinesMixin methods for BatchedCTFCore."""
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


class _MinesMixin:
    def _apply_mine_triggers(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Check if any enemy agent stepped on a mine. Triggered mines tag the
        enemy (sets tagged=True) and deactivate the mine. If the tagged agent
        was carrying a flag, the flag is returned home.

        Returns (blue_mine_tags, red_mine_tags): per-env count of mine triggers.
        """
        trigger_r = float(self.cfg.mine_trigger_radius_cells)
        B, device = self.B, self.device
        blue_mine_tags = torch.zeros((B,), dtype=torch.float32, device=device)
        red_mine_tags = torch.zeros((B,), dtype=torch.float32, device=device)

        # Blue mines trigger on red agents
        if self.blue_mine_active.any():
            dx = self.red_x[:, :, None] - self.blue_mine_x[:, None, :]
            dy = self.red_y[:, :, None] - self.blue_mine_y[:, None, :]
            dd = torch.sqrt(dx * dx + dy * dy + 1e-8)
            triggered = (dd <= trigger_r) & self.blue_mine_active[:, None, :] & self.red_alive[:, :, None] & (~self.red_tagged[:, :, None])
            agent_hit = triggered.any(dim=2)
            mine_hit = triggered.any(dim=1)
            if agent_hit.any():
                self.red_tagged = self.red_tagged | agent_hit
                red_carry_hit = agent_hit & self.red_carrying
                if red_carry_hit.any():
                    env = red_carry_hit.any(dim=1)
                    self.red_carrying[red_carry_hit] = False
                    self.blue_flag_pos[env] = self.blue_flag_home[env]
                blue_mine_tags = agent_hit.sum(dim=1).to(torch.float32)
            if mine_hit.any():
                self.blue_mine_active = self.blue_mine_active & (~mine_hit)

        # Red mines trigger on blue agents
        if self.red_mine_active.any():
            dx = self.blue_x[:, :, None] - self.red_mine_x[:, None, :]
            dy = self.blue_y[:, :, None] - self.red_mine_y[:, None, :]
            dd = torch.sqrt(dx * dx + dy * dy + 1e-8)
            triggered = (dd <= trigger_r) & self.red_mine_active[:, None, :] & self.blue_alive[:, :, None] & (~self.blue_tagged[:, :, None])
            agent_hit = triggered.any(dim=2)
            mine_hit = triggered.any(dim=1)
            if agent_hit.any():
                self.blue_tagged = self.blue_tagged | agent_hit
                blue_carry_hit = agent_hit & self.blue_carrying
                if blue_carry_hit.any():
                    env = blue_carry_hit.any(dim=1)
                    self.blue_carrying[blue_carry_hit] = False
                    self.red_flag_pos[env] = self.red_flag_home[env]
                red_mine_tags = agent_hit.sum(dim=1).to(torch.float32)
            if mine_hit.any():
                self.red_mine_active = self.red_mine_active & (~mine_hit)

        return blue_mine_tags, red_mine_tags

    def _apply_mine_pickups_side(
        self,
        side: str,
        macro: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply mine pickup rules for one side and return per-env/agent pickup counts."""
        B, device = self.B, self.device
        Np = self.Np
        radius = float(getattr(self.cfg, "mine_pickup_radius_cells", 1.2))
        respawn_delay = int(getattr(self.cfg, "mine_pickup_respawn_steps", 0))
        max_charge = int(getattr(self.cfg, "max_mine_charges_per_agent", 2))
        side_t = self._side_tensors(side)
        own_x = side_t["own_x"]
        own_y = side_t["own_y"]
        charges = side_t["own_mine_charges"]
        n_agents = self.Nr if side == "red" else self.Nb
        pickups = torch.zeros((B,), dtype=torch.float32, device=device)
        pickup_agents = torch.zeros((B, n_agents), dtype=torch.bool, device=device)

        if macro is None:
            grab = charges < max_charge
        else:
            grab = (macro == int(MacroAction.GRAB_MINE)) & (charges < max_charge)
        if side == "blue" and self.blue_scripted:
            grab = grab | (charges < max_charge)

        for i in range(n_agents):
            dx = own_x[:, i : i + 1] - self.pickup_x[:, :]
            dy = own_y[:, i : i + 1] - self.pickup_y[:, :]
            dist = torch.sqrt(dx * dx + dy * dy + 1e-8)
            near = (dist <= radius) & self.pickup_active & grab[:, i : i + 1]
            took = torch.zeros((B,), dtype=torch.bool, device=device)
            for k in range(Np):
                take = near[:, k] & (~took)
                pickups += take.to(torch.float32)
                pickup_agents[:, i] = pickup_agents[:, i] | take
                charges[:, i] = torch.where(
                    take,
                    torch.clamp(charges[:, i] + 1, max=max_charge),
                    charges[:, i],
                )
                self.pickup_active[:, k] = self.pickup_active[:, k] & (~take)
                self.pickup_respawn[:, k] = torch.where(
                    take,
                    torch.full_like(self.pickup_respawn[:, k], respawn_delay),
                    self.pickup_respawn[:, k],
                )
                took = took | take
        return pickups, pickup_agents

    def _apply_mine_pickups(self, macro_blue: torch.Tensor, macro_red: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Respawn pickups; then blue grabs with GRAB_MINE or auto-grab when scripted; red grabs when near (scripted).
        """
        respawn_delay = int(getattr(self.cfg, "mine_pickup_respawn_steps", 0))
        if respawn_delay > 0:
            self.pickup_respawn = torch.clamp(self.pickup_respawn - 1, min=0)
            self.pickup_active = self.pickup_active | ((self.pickup_respawn <= 0) & (~self.pickup_active))

        blue_pickups, blue_pickup_agents = self._apply_mine_pickups_side("blue", macro_blue)
        self._apply_mine_pickups_side("red", macro_red)
        return blue_pickups, blue_pickup_agents

    def _apply_mine_placement_side(
        self,
        side: str,
        macro: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply mine placement rules for one side and return per-env/agent placement counts."""
        B, device = self.B, self.device
        Nm = self.Nm
        side_t = self._side_tensors(side)
        own_x = side_t["own_x"]
        own_y = side_t["own_y"]
        active = side_t["own_mine_active"]
        mine_x = side_t["own_mine_x"]
        mine_y = side_t["own_mine_y"]
        charges = side_t["own_mine_charges"]
        n_agents = self.Nr if side == "red" else self.Nb
        placements = torch.zeros((B,), dtype=torch.float32, device=device)
        placement_agents = torch.zeros((B, n_agents), dtype=torch.bool, device=device)
        step_50 = (self.sim_step_count % 50) == 0

        if macro is None:
            place = torch.zeros((B, n_agents), dtype=torch.bool, device=device)
            place[:, 0] = (charges[:, 0] > 0) & step_50
        else:
            place = (macro == int(MacroAction.PLACE_MINE)) & (charges > 0)
        if side == "blue" and self.blue_scripted:
            scripted_mask = torch.zeros((B, n_agents), dtype=torch.bool, device=device)
            scripted_mask[:, 0] = step_50
            place = (place | scripted_mask) & (charges > 0)

        for i in range(n_agents):
            placed = torch.zeros((B,), dtype=torch.bool, device=device)
            for slot in range(Nm):
                can = place[:, i] & (~active[:, slot]) & (~placed)
                placements += can.to(torch.float32)
                placement_agents[:, i] = placement_agents[:, i] | can
                mine_x[can, slot] = own_x[can, i]
                mine_y[can, slot] = own_y[can, i]
                active[can, slot] = True
                charges[:, i] = torch.where(
                    can,
                    torch.clamp(charges[:, i] - 1, min=0),
                    charges[:, i],
                )
                placed = placed | can
        return placements, placement_agents

    def _apply_mine_placement(self, macro_blue: torch.Tensor, macro_red: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Blue: PLACE_MINE or scripted (defender every 50 steps) places at current position if charge > 0.
        Red: scripted placement when has charge (e.g. defender places every 50 steps).
        """
        blue_placements, blue_placement_agents = self._apply_mine_placement_side("blue", macro_blue)
        self._apply_mine_placement_side("red", macro_red)
        return blue_placements, blue_placement_agents
