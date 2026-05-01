"""ObservationsMixin methods for BatchedCTFCore."""
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


class _ObservationsMixin:
    def _scatter_points(self, grid: torch.Tensor, ch: int, x: torch.Tensor, y: torch.Tensor, live: torch.Tensor) -> None:
        cx = torch.clamp((x / max(1.0, float(self.cols - 1)) * float(CNN_COLS - 1)).round().long(), 0, CNN_COLS - 1)
        cy = torch.clamp((y / max(1.0, float(self.rows - 1)) * float(CNN_ROWS - 1)).round().long(), 0, CNN_ROWS - 1)
        b_idx = torch.arange(self.B, device=self.device).view(self.B, 1).expand(-1, cx.shape[1])
        if live.any():
            grid[b_idx[live], ch, cy[live], cx[live]] = 1.0

    def _build_grid_obs(self, side: str = "blue") -> torch.Tensor:
        side_t = self._side_tensors(side)
        own_x = side_t["own_x"]
        own_y = side_t["own_y"]
        own_alive = side_t["own_alive"]
        enemy_x = side_t["enemy_x"]
        enemy_y = side_t["enemy_y"]
        enemy_alive = side_t["enemy_alive"]
        own_mine_x = side_t["own_mine_x"]
        own_mine_y = side_t["own_mine_y"]
        own_mine_active = side_t["own_mine_active"]
        own_flag = side_t["own_flag"]
        enemy_flag = side_t["enemy_flag"]
        n_agents = int(own_x.shape[1])
        grid = torch.zeros((self.B, n_agents, NUM_CNN_CHANNELS, CNN_ROWS, CNN_COLS), dtype=torch.float32, device=self.device)
        own_x_obs = self._mirror_x(own_x, side)
        enemy_x_obs = self._mirror_x(enemy_x, side)
        own_mine_x_obs = self._mirror_x(own_mine_x, side)
        pickup_x_obs = self._mirror_x(self.pickup_x, side)
        own_flag_x_obs = self._mirror_x(own_flag[:, 0:1], side)
        enemy_flag_x_obs = self._mirror_x(enemy_flag[:, 0:1], side)
        for i in range(n_agents):
            self_live = own_alive[:, i : i + 1]
            self._scatter_points(grid[:, i], 0, own_x_obs[:, i : i + 1], own_y[:, i : i + 1], self_live)

            friend_live = own_alive.clone()
            friend_live[:, i] = False
            self._scatter_points(grid[:, i], 1, own_x_obs, own_y, friend_live)

            ex = enemy_x_obs
            ey = enemy_y
            elive = enemy_alive
            if self.cfg.sensor_range_cells < 1e8:
                dx = enemy_x - own_x[:, i : i + 1]
                dy = enemy_y - own_y[:, i : i + 1]
                dist = torch.sqrt(dx * dx + dy * dy + 1e-8)
                in_range = dist <= float(self.cfg.sensor_range_cells)
                p_drop = self.rt_sensor_dropout_prob[:, None]
                if float(self.rt_sensor_dropout_prob.max().item()) > 0.0:
                    drop = torch.rand(in_range.shape, generator=self._rng, device=self.device) < p_drop
                    in_range = in_range & (~drop)
                elive = elive & in_range
                sigma = self.rt_sensor_noise_sigma_cells[:, None]
                if float(self.rt_sensor_noise_sigma_cells.max().item()) > 0.0:
                    ex = torch.clamp(ex + self._randn(ex.shape) * sigma, 0.0, float(max(0, self.cols - 1)))
                    ey = torch.clamp(ey + self._randn(ey.shape) * sigma, 0.0, float(max(0, self.rows - 1)))
            else:
                # Unlimited range (typical training): still apply per-episode obs jitter when sampled.
                p_drop = self.rt_sensor_dropout_prob[:, None]
                if float(self.rt_sensor_dropout_prob.max().item()) > 0.0:
                    drop = torch.rand(elive.shape, generator=self._rng, device=self.device) < p_drop
                    elive = elive & (~drop)
                sigma = self.rt_sensor_noise_sigma_cells[:, None]
                if float(self.rt_sensor_noise_sigma_cells.max().item()) > 0.0:
                    ex = torch.clamp(ex + self._randn(ex.shape) * sigma, 0.0, float(max(0, self.cols - 1)))
                    ey = torch.clamp(ey + self._randn(ey.shape) * sigma, 0.0, float(max(0, self.rows - 1)))
            self._scatter_points(grid[:, i], 2, ex, ey, elive)
            self._scatter_points(grid[:, i], 3, own_mine_x_obs, own_mine_y, own_mine_active)
            self._scatter_points(grid[:, i], 4, pickup_x_obs, self.pickup_y, self.pickup_active)

            self._scatter_points(grid[:, i], 5, own_flag_x_obs, own_flag[:, 1:2], torch.ones((self.B, 1), dtype=torch.bool, device=self.device))
            self._scatter_points(grid[:, i], 6, enemy_flag_x_obs, enemy_flag[:, 1:2], torch.ones((self.B, 1), dtype=torch.bool, device=self.device))
        return grid

    def _build_vec_obs(self, side: str = "blue") -> torch.Tensor:
        """
        Normalized observation vector (stable for PPO critic):
          0: x_norm in [0,1]
          1: y_norm in [0,1]
          2: heading/pi in [-1,1]
          3: speed/max_speed in [0,1]
          4-7: relative flag deltas normalized to [-1,1]
          8-10: payload one-hot [none, mine, flag]
          11: nearest enemy distance normalized to [0,1]
          12: time fraction in [0,1]
          13: agent id normalized in [0,1]
          14: decision fraction in [0,1] (step_count/max_steps)
          15: mine charge ratio in [0,1]
          16: nearest active pickup distance in [0,1]
          17: nearest friendly mine distance in [0,1]
          18: own flag-capture score / score_limit in [0,1]
          19: opponent flag-capture score / score_limit in [0,1]
        """
        side_t = self._side_tensors(side)
        own_x = side_t["own_x"]
        own_y = side_t["own_y"]
        own_heading = side_t["own_heading"]
        own_speed = side_t["own_speed"]
        own_alive = side_t["own_alive"]
        own_carrying = side_t["own_carrying"]
        own_flag = side_t["own_flag"]
        own_flag_home = side_t["own_flag_home"]
        own_mine_x = side_t["own_mine_x"]
        own_mine_y = side_t["own_mine_y"]
        own_mine_active = side_t["own_mine_active"]
        own_mine_charges = side_t["own_mine_charges"]
        enemy_x = side_t["enemy_x"]
        enemy_y = side_t["enemy_y"]
        enemy_flag = side_t["enemy_flag"]
        n_agents = int(own_x.shape[1])
        out = torch.zeros((self.B, n_agents, VEC_OBS_DIM), dtype=torch.float32, device=self.device)
        cols = max(1.0, float(self.cols - 1))
        rows = max(1.0, float(self.rows - 1))
        max_speed = max(1e-6, float(self.cfg.max_speed_cps))
        own_x_obs = self._mirror_x(own_x, side)
        enemy_x_obs = self._mirror_x(enemy_x, side)
        own_flag_x_obs = self._mirror_x(own_flag[:, None, 0], side)
        own_flag_home_x_obs = self._mirror_x(own_flag_home[:, None, 0], side)
        enemy_flag_x_obs = self._mirror_x(enemy_flag[:, None, 0], side)
        pickup_x_obs = self._mirror_x(self.pickup_x[:, None, :], side)
        own_mine_x_obs = self._mirror_x(own_mine_x[:, None, :], side)

        out[..., 0] = torch.clamp(own_x_obs / cols, 0.0, 1.0)
        out[..., 1] = torch.clamp(own_y / rows, 0.0, 1.0)
        # Discretized bearing theta_i (formal Aquaticus-style state element).
        heading_norm = (self._mirror_heading(own_heading, side) + math.pi) / (2.0 * math.pi)
        heading_bins = torch.floor(torch.clamp(heading_norm, 0.0, 0.9999) * 16.0) / 15.0
        out[..., 2] = torch.clamp(heading_bins * 2.0 - 1.0, -1.0, 1.0)
        out[..., 3] = torch.clamp(own_speed / max_speed, 0.0, 1.0)
        out[..., 4] = torch.clamp((enemy_flag_x_obs - own_x_obs) / max(1.0, float(self.cols)), -1.0, 1.0)
        out[..., 5] = torch.clamp((enemy_flag[:, None, 1] - own_y) / max(1.0, float(self.rows)), -1.0, 1.0)
        out[..., 6] = torch.clamp((own_flag_x_obs - own_x_obs) / max(1.0, float(self.cols)), -1.0, 1.0)
        out[..., 7] = torch.clamp((own_flag[:, None, 1] - own_y) / max(1.0, float(self.rows)), -1.0, 1.0)
        has_mine = own_mine_charges > 0
        no_payload = (~own_carrying) & (~has_mine)
        out[..., 8] = no_payload.to(torch.float32)
        out[..., 9] = has_mine.to(torch.float32)
        out[..., 10] = own_carrying.to(torch.float32)

        dx = enemy_x_obs[:, None, :] - own_x_obs[:, :, None]
        dy = enemy_y[:, None, :] - own_y[:, :, None]
        d = torch.sqrt(dx * dx + dy * dy + 1e-8)
        nearest_enemy = torch.min(d, dim=2).values
        out[..., 11] = torch.clamp(nearest_enemy / max(1e-6, self.max_dist), 0.0, 1.0)
        time_frac = torch.clamp(self.sim_step_count[:, None].to(torch.float32) / max(1.0, float(self.max_sim_steps)), 0.0, 1.0)
        out[..., 12] = time_frac

        agent_id = torch.arange(n_agents, device=self.device, dtype=torch.float32)
        out[..., 13] = agent_id[None, :] / max(1.0, float(n_agents - 1))

        # Decision budget used (same normalization as time fraction, but kept as a separate feature
        # so the policy can distinguish between absolute time and how many decisions have been spent).
        out[..., 14] = torch.clamp(self.step_count[:, None].to(torch.float32) / max(1.0, float(self.max_steps)), 0.0, 1.0)
        max_charge = max(1.0, float(getattr(self.cfg, "max_mine_charges_per_agent", 2)))
        out[..., 15] = torch.clamp(own_mine_charges.to(torch.float32) / max_charge, 0.0, 1.0)

        pdx = pickup_x_obs - own_x_obs[:, :, None]
        pdy = self.pickup_y[:, None, :] - own_y[:, :, None]
        pickup_dist = torch.sqrt(pdx * pdx + pdy * pdy + 1e-8)
        pickup_mask = self.pickup_active[:, None, :].expand(-1, n_agents, -1)
        pickup_dist = torch.where(
            pickup_mask,
            pickup_dist,
            torch.full_like(pickup_dist, float(self.max_dist)),
        )
        out[..., 16] = torch.clamp(torch.min(pickup_dist, dim=2).values / max(1e-6, self.max_dist), 0.0, 1.0)

        mdx = own_mine_x_obs - own_x_obs[:, :, None]
        mdy = own_mine_y[:, None, :] - own_y[:, :, None]
        mine_dist = torch.sqrt(mdx * mdx + mdy * mdy + 1e-8)
        mine_mask = own_mine_active[:, None, :].expand(-1, n_agents, -1)
        mine_dist = torch.where(
            mine_mask,
            mine_dist,
            torch.full_like(mine_dist, float(self.max_dist)),
        )
        out[..., 17] = torch.clamp(torch.min(mine_dist, dim=2).values / max(1e-6, self.max_dist), 0.0, 1.0)
        if side == "red":
            own_score = self.red_score
            opponent_score = self.blue_score
        else:
            own_score = self.blue_score
            opponent_score = self.red_score
        score_den = max(1.0, float(self.score_limit))
        out[..., 18] = torch.clamp(own_score[:, None].to(torch.float32) / score_den, 0.0, 1.0)
        out[..., 19] = torch.clamp(opponent_score[:, None].to(torch.float32) / score_den, 0.0, 1.0)
        out[..., 0] = out[..., 0] * own_alive.to(torch.float32)
        return out

    def _build_action_mask(self, side: str = "blue") -> torch.Tensor:
        side_t = self._side_tensors(side)
        own_alive = side_t["own_alive"]
        own_carrying = side_t["own_carrying"]
        own_mine_charges = side_t["own_mine_charges"]
        own_x = side_t["own_x"]
        own_y = side_t["own_y"]
        n_agents = int(own_alive.shape[1])
        mask = torch.ones((self.B, n_agents, self.cfg.n_macros + self.cfg.n_targets), dtype=torch.float32, device=self.device)
        dead = ~own_alive
        if dead.any():
            mask[:, :, : self.cfg.n_macros][dead] = 0.0
            mask[:, :, self.cfg.n_macros :][dead] = 0.0
            mask[:, :, 0][dead] = 1.0
            mask[:, :, self.cfg.n_macros + 0][dead] = 1.0
        carrying = own_carrying
        if carrying.any():
            idx_get, idx_grab, idx_place, idx_home = 2, 1, 3, 4
            mask[:, :, idx_get][carrying] = 0.0
            mask[:, :, idx_grab][carrying] = 0.0
            mask[:, :, idx_place][carrying] = 0.0
            mask[:, :, idx_home][carrying] = 1.0
        has_mine = own_mine_charges > 0
        if has_mine.any():
            idx_grab, idx_get, idx_place = 1, 2, 3
            mask[:, :, idx_grab][has_mine] = 0.0
            mask[:, :, idx_get][has_mine] = 0.0
            mask[:, :, idx_place][has_mine] = 1.0
        no_payload = (~own_carrying) & (~has_mine)
        if no_payload.any():
            idx_grab, idx_get, idx_home = 1, 2, 4
            mask[:, :, idx_grab][no_payload] = 1.0
            mask[:, :, idx_get][no_payload] = 1.0
            mask[:, :, idx_home][no_payload] = 0.0
            pickup_radius = float(getattr(self.cfg, "mine_pickup_radius_cells", 1.2))
            pdx = self.pickup_x[:, None, :] - own_x[:, :, None]
            pdy = self.pickup_y[:, None, :] - own_y[:, :, None]
            pickup_dist = torch.sqrt(pdx * pdx + pdy * pdy + 1e-8)
            near_pickup = ((pickup_dist <= pickup_radius) & self.pickup_active[:, None, :]).any(dim=2)
            mask[:, :, idx_grab][no_payload & (~near_pickup)] = 0.0
        no_mine = ~has_mine
        if no_mine.any():
            idx_place = 3
            mask[:, :, idx_place][no_mine] = 0.0
        if side == "blue":
            committed = (self.blue_commit_ticks_left > 0) & self.blue_alive
            commit_macro = self.blue_commit_macro
            commit_target = self.blue_commit_target
        else:
            committed = (self.red_commit_ticks_left > 0) & self.red_alive
            commit_macro = self.red_commit_macro
            commit_target = self.red_commit_target
        if committed is not None and committed.any():
            macro_mask = torch.zeros_like(mask[:, :, : self.cfg.n_macros])
            macro_mask.scatter_(2, commit_macro.unsqueeze(-1), 1.0)
            mask[:, :, : self.cfg.n_macros] = torch.where(committed.unsqueeze(-1), macro_mask, mask[:, :, : self.cfg.n_macros])
            target_mask = torch.zeros_like(mask[:, :, self.cfg.n_macros :])
            target_mask.scatter_(2, commit_target.unsqueeze(-1), 1.0)
            mask[:, :, self.cfg.n_macros :] = torch.where(committed.unsqueeze(-1), target_mask, mask[:, :, self.cfg.n_macros :])
        return mask.reshape(self.B, -1)

    def get_obs_tensors(self, side: str = "blue") -> Dict[str, torch.Tensor]:
        """Observations as GPU tensors -- zero-copy, no CPU round-trip."""
        own_alive = self.red_alive if side == "red" else self.blue_alive
        return {
            "grid": self._build_grid_obs(side=side),
            "vec": self._build_vec_obs(side=side),
            "agent_mask": own_alive.to(torch.float32),
            "mask": self._build_action_mask(side=side),
        }

    def get_obs(self, side: str = "blue") -> Dict[str, np.ndarray]:
        return {k: v.detach().cpu().numpy().astype(np.float32)
                for k, v in self.get_obs_tensors(side=side).items()}

    def get_global_state_tensor(self) -> torch.Tensor:
        """Structured CTDE global state tensor with shape ``(B, GLOBAL_STATE_DIM)``."""
        return build_global_state_batch(self)

    def get_global_state(self) -> np.ndarray:
        return self.get_global_state_tensor().detach().cpu().numpy().astype(np.float32)

    def state(self) -> np.ndarray:
        """PettingZoo-style alias for the structured global state."""
        return self.get_global_state()

    def render_rgb_array(self, env_index: int = 0, cell_size: int = 8) -> np.ndarray:
        """Render one vectorized environment as a uint8 RGB array."""
        env_i = int(max(0, min(env_index, self.B - 1)))
        cell = max(1, int(cell_size))
        h = max(1, self.rows) * cell
        w = max(1, self.cols) * cell
        frame = np.full((h, w, 3), 238, dtype=np.uint8)
        frame[:, : max(1, w // 2), :] = np.array([232, 242, 255], dtype=np.uint8)
        frame[:, max(1, w // 2) :, :] = np.array([255, 235, 235], dtype=np.uint8)

        def draw_point(x: float, y: float, color: tuple[int, int, int], radius: int = 2) -> None:
            cx = int(round(float(x) / max(1.0, float(self.cols - 1)) * float(w - 1)))
            cy = int(round(float(y) / max(1.0, float(self.rows - 1)) * float(h - 1)))
            r = max(1, int(radius))
            x0, x1 = max(0, cx - r), min(w, cx + r + 1)
            y0, y1 = max(0, cy - r), min(h, cy + r + 1)
            frame[y0:y1, x0:x1, :] = np.array(color, dtype=np.uint8)

        bh = self.blue_flag_home[env_i].detach().cpu().numpy()
        rh = self.red_flag_home[env_i].detach().cpu().numpy()
        bf = self.blue_flag_pos[env_i].detach().cpu().numpy()
        rf = self.red_flag_pos[env_i].detach().cpu().numpy()
        draw_point(float(bh[0]), float(bh[1]), (35, 95, 220), radius=3)
        draw_point(float(rh[0]), float(rh[1]), (210, 55, 55), radius=3)
        draw_point(float(bf[0]), float(bf[1]), (50, 130, 255), radius=2)
        draw_point(float(rf[0]), float(rf[1]), (255, 70, 70), radius=2)

        blue_x = self.blue_x[env_i].detach().cpu().numpy()
        blue_y = self.blue_y[env_i].detach().cpu().numpy()
        blue_alive = self.blue_alive[env_i].detach().cpu().numpy()
        red_x = self.red_x[env_i].detach().cpu().numpy()
        red_y = self.red_y[env_i].detach().cpu().numpy()
        red_alive = self.red_alive[env_i].detach().cpu().numpy()
        for x, y, alive in zip(blue_x, blue_y, blue_alive):
            draw_point(float(x), float(y), (20, 80, 210) if alive else (120, 150, 190), radius=2)
        for x, y, alive in zip(red_x, red_y, red_alive):
            draw_point(float(x), float(y), (200, 35, 35) if alive else (190, 130, 130), radius=2)
        return frame


# -------- Adapter for MAPPO/QMIX: GameField-like API over BatchedCTFCore(B=1) --------
