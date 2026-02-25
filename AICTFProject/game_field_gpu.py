from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv


CNN_COLS = 20
CNN_ROWS = 20
NUM_CNN_CHANNELS = 7
GLOBAL_STATE_CHANNELS = 8


@dataclass
class GPUFieldConfig:
    n_envs: int = 64
    # Aquaticus standard setup is 2v2.
    max_blue_agents: int = 2
    max_red_agents: int = 2
    map_rows: int = 20
    map_cols: int = 20
    max_decision_steps: int = 400
    decision_interval_seconds: float = 0.7

    # Dynamics (matching BoatSimConfig defaults in game_field.py)
    max_speed_cps: float = 2.2
    max_accel_cps2: float = 2.0
    max_yaw_rate_rps: float = 4.0
    min_turn_radius_cells: float = 0.75
    current_strength_cps: float = 0.0
    drift_sigma_cells: float = 0.0
    sensor_range_cells: float = 9999.0
    sensor_noise_sigma_cells: float = 0.0
    sensor_dropout_prob: float = 0.0
    suppression_range_cells: float = 2.0
    tag_range_cells: float = 10.0
    home_untag_radius_cells: float = 1.25
    avoid_collision_radius_cells: float = 0.75
    opregion_safe_speed_cps: float = 0.8

    n_macros: int = 5
    n_targets: int = 8
    score_limit: int = 9

    # Profile and reward controls
    aquaticus_profile: bool = True
    rules_profile: str = "AQUATICUS_2024"  # OURS_PLUS | AQUATICUS_2024
    sparse_weight: float = 1.0
    dense_weight: float = 0.35
    reward_scale: float = 2.0
    reward_clip: float = 1.0

    # PPO stability
    stalemate_max_steps: int = 120
    stalemate_progress_eps: float = 0.002
    stalemate_penalty: float = -0.15
    spin_penalty_coef: float = 0.05
    idle_penalty_coef: float = 0.03

    device: str = "cpu"
    seed: int = 42


class BatchedCTFCore:
    """
    GPU-vectorized CTF core with Aquaticus-profile option.

    Hybrid reward math (per env):
        R_sparse_raw = sum(Aquaticus-style event rewards)
        R_sparse = R_sparse_raw / 100
        R_dense = progress + defense_presence + escort - spin_penalty - idle_penalty
        R_total_raw = w_s * R_sparse + w_d * R_dense
        R_total = clip(tanh(R_total_raw / reward_scale), -reward_clip, reward_clip)

    The tanh+clip stage reduces reward variance for PPO and keeps value targets bounded.
    """

    def __init__(self, cfg: GPUFieldConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.B = int(cfg.n_envs)
        self.Nb = int(cfg.max_blue_agents)
        self.Nr = int(cfg.max_red_agents)
        self.rows = int(cfg.map_rows)
        self.cols = int(cfg.map_cols)
        self.max_steps = int(cfg.max_decision_steps)
        self.score_limit = int(cfg.score_limit)
        self.dt = float(cfg.decision_interval_seconds) * 0.99
        self.max_dist = math.sqrt(float(self.cols * self.cols + self.rows * self.rows))

        self._rng = torch.Generator(device=self.device)
        self._rng.manual_seed(int(cfg.seed))

        self._phase = "OP1"
        self._league_mode = False
        self._stress_schedule: Optional[dict] = None
        self._opponent_kind = "SCRIPTED"
        self._opponent_key = "OP1"
        self.rules_profile = str(cfg.rules_profile).upper()

        self._build_macro_targets()
        self._alloc_state()
        self.reset_all()

    def _alloc_state(self) -> None:
        B, Nb, Nr = self.B, self.Nb, self.Nr
        dev = self.device
        f32 = torch.float32

        self.step_count = torch.zeros((B,), dtype=torch.int32, device=dev)
        self.done = torch.zeros((B,), dtype=torch.bool, device=dev)
        self.truncated = torch.zeros((B,), dtype=torch.bool, device=dev)
        self.stalemate_steps = torch.zeros((B,), dtype=torch.int32, device=dev)

        self.blue_x = torch.zeros((B, Nb), dtype=f32, device=dev)
        self.blue_y = torch.zeros((B, Nb), dtype=f32, device=dev)
        self.blue_heading = torch.zeros((B, Nb), dtype=f32, device=dev)
        self.blue_speed = torch.zeros((B, Nb), dtype=f32, device=dev)
        self.blue_alive = torch.ones((B, Nb), dtype=torch.bool, device=dev)
        self.blue_tagged = torch.zeros((B, Nb), dtype=torch.bool, device=dev)
        self.blue_carrying = torch.zeros((B, Nb), dtype=torch.bool, device=dev)
        self.blue_respawn = torch.zeros((B, Nb), dtype=f32, device=dev)

        self.red_x = torch.zeros((B, Nr), dtype=f32, device=dev)
        self.red_y = torch.zeros((B, Nr), dtype=f32, device=dev)
        self.red_heading = torch.zeros((B, Nr), dtype=f32, device=dev)
        self.red_speed = torch.zeros((B, Nr), dtype=f32, device=dev)
        self.red_alive = torch.ones((B, Nr), dtype=torch.bool, device=dev)
        self.red_tagged = torch.zeros((B, Nr), dtype=torch.bool, device=dev)
        self.red_carrying = torch.zeros((B, Nr), dtype=torch.bool, device=dev)
        self.red_respawn = torch.zeros((B, Nr), dtype=f32, device=dev)

        self.blue_score = torch.zeros((B,), dtype=torch.int32, device=dev)
        self.red_score = torch.zeros((B,), dtype=torch.int32, device=dev)

        self.blue_flag_home = torch.stack(
            [
                torch.zeros((B,), dtype=f32, device=dev),
                torch.full((B,), float(self.rows // 2), dtype=f32, device=dev),
            ],
            dim=1,
        )
        self.red_flag_home = torch.stack(
            [
                torch.full((B,), float(self.cols - 1), dtype=f32, device=dev),
                torch.full((B,), float(self.rows // 2), dtype=f32, device=dev),
            ],
            dim=1,
        )
        self.blue_flag_pos = self.blue_flag_home.clone()
        self.red_flag_pos = self.red_flag_home.clone()

        self.rt_current_strength_cps = torch.full((B,), float(self.cfg.current_strength_cps), dtype=f32, device=dev)
        self.rt_drift_sigma_cells = torch.full((B,), float(self.cfg.drift_sigma_cells), dtype=f32, device=dev)
        self._last_dense_progress = torch.zeros((B,), dtype=f32, device=dev)

    def _build_macro_targets(self) -> None:
        c_mid = self.cols // 2
        r_mid = self.rows // 2
        top = max(0, min(self.rows - 1, 5))
        bot = max(0, min(self.rows - 1, self.rows - 5))
        self._macro_targets = torch.tensor(
            [
                [0.0, float(r_mid)],
                [float(self.cols - 1), float(r_mid)],
                [2.0, float(r_mid)],
                [float(self.cols - 3), float(r_mid)],
                [float(c_mid), float(r_mid)],
                [float(c_mid), float(top)],
                [float(c_mid), float(bot)],
                [4.0, float(r_mid)],
            ],
            dtype=torch.float32,
            device=self.device,
        )

    def _rand_uniform(self, shape: Sequence[int], lo: float, hi: float) -> torch.Tensor:
        t = torch.rand(tuple(shape), generator=self._rng, device=self.device)
        return lo + (hi - lo) * t

    def _randn(self, shape: Sequence[int]) -> torch.Tensor:
        return torch.randn(tuple(shape), generator=self._rng, device=self.device)

    def _respawn_side(self, blue: bool, env_mask: Optional[torch.Tensor] = None) -> None:
        if env_mask is None:
            env_mask = torch.ones((self.B,), dtype=torch.bool, device=self.device)
        idx = torch.where(env_mask)[0]
        if idx.numel() == 0:
            return
        E = int(idx.numel())
        if blue:
            x_lo, x_hi = 0.0, max(1.0, float(self.cols // 3 - 1))
            self.blue_x[idx] = self._rand_uniform((E, self.Nb), x_lo, x_hi)
            self.blue_y[idx] = self._rand_uniform((E, self.Nb), 0.0, float(max(0, self.rows - 1)))
            self.blue_heading[idx].zero_()
            self.blue_speed[idx].zero_()
            self.blue_alive[idx].fill_(True)
            self.blue_tagged[idx].fill_(False)
            self.blue_carrying[idx].fill_(False)
            self.blue_respawn[idx].zero_()
        else:
            x_lo = max(0.0, float(self.cols - max(1, self.cols // 3)))
            x_hi = float(max(0, self.cols - 1))
            self.red_x[idx] = self._rand_uniform((E, self.Nr), x_lo, x_hi)
            self.red_y[idx] = self._rand_uniform((E, self.Nr), 0.0, float(max(0, self.rows - 1)))
            self.red_heading[idx].fill_(math.pi)
            self.red_speed[idx].zero_()
            self.red_alive[idx].fill_(True)
            self.red_tagged[idx].fill_(False)
            self.red_carrying[idx].fill_(False)
            self.red_respawn[idx].zero_()

    def _score_grab_delta(self) -> int:
        return 1 if self.rules_profile == "AQUATICUS_2024" else 0

    def _score_capture_delta(self) -> int:
        return 2 if self.rules_profile == "AQUATICUS_2024" else 1

    def reset_all(self) -> None:
        mask = torch.ones((self.B,), dtype=torch.bool, device=self.device)
        self.reset_indices(mask)

    def reset_indices(self, env_mask: torch.Tensor) -> None:
        idx = torch.where(env_mask)[0]
        if idx.numel() == 0:
            return
        self.done[idx] = False
        self.truncated[idx] = False
        self.step_count[idx] = 0
        self.stalemate_steps[idx] = 0
        self.blue_score[idx] = 0
        self.red_score[idx] = 0
        self.blue_flag_pos[idx] = self.blue_flag_home[idx]
        self.red_flag_pos[idx] = self.red_flag_home[idx]
        self.blue_tagged[idx] = False
        self.red_tagged[idx] = False
        self._last_dense_progress[idx] = 0.0
        self._respawn_side(blue=True, env_mask=env_mask)
        self._respawn_side(blue=False, env_mask=env_mask)

    # env_method-compatible setters
    def set_phase(self, phase: str) -> None:
        self._phase = str(phase).upper()

    def set_league_mode(self, league_mode: bool) -> None:
        self._league_mode = bool(league_mode)

    def set_stress_schedule(self, schedule: Optional[dict]) -> None:
        self._stress_schedule = schedule

    def set_next_opponent(self, kind: str, key: str) -> None:
        self._opponent_kind = str(kind).upper()
        self._opponent_key = str(key).upper()

    def set_dynamics_config(self, cfg: Optional[Dict[str, Any]]) -> None:
        if not isinstance(cfg, dict):
            return
        if "rules_profile" in cfg:
            self.rules_profile = str(cfg["rules_profile"]).upper().strip()
        if "aquaticus_profile" in cfg:
            self.cfg.aquaticus_profile = bool(cfg["aquaticus_profile"])
        for key in (
            "max_speed_cps",
            "max_accel_cps2",
            "max_yaw_rate_rps",
            "min_turn_radius_cells",
            "current_strength_cps",
            "drift_sigma_cells",
            "sensor_range_cells",
            "sensor_noise_sigma_cells",
            "sensor_dropout_prob",
        ):
            if key in cfg:
                setattr(self.cfg, key, float(cfg[key]))

    def _apply_profile_runtime(self) -> None:
        # Optional stress schedule by phase (same hook shape used by train_ppo callbacks).
        if isinstance(self._stress_schedule, dict):
            p = self._stress_schedule.get(str(self._phase).upper(), None)
            if isinstance(p, dict):
                if "current_strength_cps" in p:
                    self.rt_current_strength_cps.fill_(float(p["current_strength_cps"]))
                if "drift_sigma_cells" in p:
                    self.rt_drift_sigma_cells.fill_(float(p["drift_sigma_cells"]))

        # Aquaticus profile keeps marine constraints and can add mild stochastic water drift.
        if self.cfg.aquaticus_profile:
            self.cfg.max_speed_cps = min(float(self.cfg.max_speed_cps), 2.2)
            self.cfg.max_accel_cps2 = min(float(self.cfg.max_accel_cps2), 2.0)
            self.cfg.max_yaw_rate_rps = min(float(self.cfg.max_yaw_rate_rps), 4.0)

    def _decode_targets(self, target_idx: torch.Tensor) -> torch.Tensor:
        tidx = torch.remainder(target_idx, self.cfg.n_targets).long()
        return self._macro_targets.index_select(0, tidx.reshape(-1)).reshape(self.B, self.Nb, 2)

    def _red_scripted_actions(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # Lightweight scripted bot: closest blue chase; if carrying, return home.
        dx = self.blue_x[:, :1] - self.red_x
        dy = self.blue_y[:, :1] - self.red_y
        target = torch.stack([self.red_x + dx, self.red_y + dy], dim=-1)
        near_flag = self.red_carrying | ((self.red_x - self.blue_flag_pos[:, 0:1]).abs() < 1.0)
        target[near_flag] = self.red_flag_home[:, None, :].expand(-1, self.Nr, -1)[near_flag]
        return target[..., 0], target[..., 1]

    def _integrate_side(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        heading: torch.Tensor,
        speed: torch.Tensor,
        alive: torch.Tensor,
        target_x: torch.Tensor,
        target_y: torch.Tensor,
        speed_cap: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Marine kinematics with acceleration/yaw constraints and turn-radius limiting.
        """
        dt = self.dt
        dx = target_x - x
        dy = target_y - y
        desired_heading = torch.atan2(dy, dx)
        err = (desired_heading - heading + math.pi) % (2.0 * math.pi) - math.pi

        yaw_rate_cmd = torch.clamp(err / max(1e-6, dt), -self.cfg.max_yaw_rate_rps, self.cfg.max_yaw_rate_rps)
        # Turn-radius bound: |omega| <= v / R_min (with floor for low-speed controllability)
        min_r = max(1e-3, float(self.cfg.min_turn_radius_cells))
        omega_bound = torch.clamp(speed / min_r, min=0.5, max=float(self.cfg.max_yaw_rate_rps))
        yaw_rate_cmd = torch.clamp(yaw_rate_cmd, -omega_bound, omega_bound)

        desired_speed = torch.full_like(speed, float(self.cfg.max_speed_cps))
        if speed_cap is not None:
            desired_speed = torch.minimum(desired_speed, torch.clamp(speed_cap, min=0.0))
        dist = torch.sqrt(dx * dx + dy * dy + 1e-8)
        desired_speed = torch.where(dist < 0.75, desired_speed * (dist / 0.75), desired_speed)
        accel_cmd = torch.clamp(
            (desired_speed - speed) / max(1e-6, dt),
            -float(self.cfg.max_accel_cps2),
            float(self.cfg.max_accel_cps2),
        )

        speed2 = torch.clamp(speed + accel_cmd * dt, 0.0, float(self.cfg.max_speed_cps))
        if speed_cap is not None:
            speed2 = torch.minimum(speed2, torch.clamp(speed_cap, min=0.0))
        heading2 = heading + yaw_rate_cmd * dt
        vx = speed2 * torch.cos(heading2) + self.rt_current_strength_cps[:, None]
        vy = speed2 * torch.sin(heading2)

        nx_raw = x + vx * dt
        ny_raw = y + vy * dt
        oob = (nx_raw < 0.0) | (nx_raw > float(max(0, self.cols - 1))) | (ny_raw < 0.0) | (ny_raw > float(max(0, self.rows - 1)))
        x2 = torch.clamp(nx_raw, 0.0, float(max(0, self.cols - 1)))
        y2 = torch.clamp(ny_raw, 0.0, float(max(0, self.rows - 1)))

        drift_sigma = self.rt_drift_sigma_cells[:, None]
        if float(drift_sigma.max().item()) > 0.0:
            x2 = torch.clamp(x2 + self._randn(x2.shape) * drift_sigma, 0.0, float(max(0, self.cols - 1)))
            y2 = torch.clamp(y2 + self._randn(y2.shape) * drift_sigma, 0.0, float(max(0, self.rows - 1)))

        alive_f = alive.to(x2.dtype)
        x_out = x2 * alive_f + x * (1.0 - alive_f)
        y_out = y2 * alive_f + y * (1.0 - alive_f)
        h_out = heading2 * alive_f + heading * (1.0 - alive_f)
        s_out = speed2 * alive_f + speed * (1.0 - alive_f)
        return x_out, y_out, h_out, s_out, oob & alive, yaw_rate_cmd

    def _is_on_home_side(self, side: str, x: torch.Tensor) -> torch.Tensor:
        mid = float(self.cols - 1) * 0.5
        if side == "blue":
            return x <= mid
        return x >= mid

    def _untag_if_home(self) -> None:
        bdx = self.blue_x - self.blue_flag_home[:, None, 0]
        bdy = self.blue_y - self.blue_flag_home[:, None, 1]
        b_home = torch.sqrt(bdx * bdx + bdy * bdy + 1e-8) <= float(self.cfg.home_untag_radius_cells)
        rdx = self.red_x - self.red_flag_home[:, None, 0]
        rdy = self.red_y - self.red_flag_home[:, None, 1]
        r_home = torch.sqrt(rdx * rdx + rdy * rdy + 1e-8) <= float(self.cfg.home_untag_radius_cells)
        self.blue_tagged = self.blue_tagged & (~b_home)
        self.red_tagged = self.red_tagged & (~r_home)

    def _apply_avoid_collision_guard(
        self,
        prev_blue_x: torch.Tensor,
        prev_blue_y: torch.Tensor,
        prev_red_x: torch.Tensor,
        prev_red_y: torch.Tensor,
    ) -> None:
        """
        Aquaticus-style safety guardrail:
        if any agent pair gets too close, halt by reverting to previous position and zeroing speed.
        """
        rr = float(self.cfg.avoid_collision_radius_cells)
        if rr <= 0.0:
            return

        # Blue-Blue
        ddx = self.blue_x[:, :, None] - self.blue_x[:, None, :]
        ddy = self.blue_y[:, :, None] - self.blue_y[:, None, :]
        d = torch.sqrt(ddx * ddx + ddy * ddy + 1e-8)
        eye = torch.eye(self.Nb, dtype=torch.bool, device=self.device)[None, :, :]
        close_bb = (d < rr) & (~eye)
        halt_b = close_bb.any(dim=2)

        # Red-Red
        ddx_r = self.red_x[:, :, None] - self.red_x[:, None, :]
        ddy_r = self.red_y[:, :, None] - self.red_y[:, None, :]
        d_r = torch.sqrt(ddx_r * ddx_r + ddy_r * ddy_r + 1e-8)
        eye_r = torch.eye(self.Nr, dtype=torch.bool, device=self.device)[None, :, :]
        close_rr = (d_r < rr) & (~eye_r)
        halt_r = close_rr.any(dim=2)

        # Blue-Red
        dx_br = self.blue_x[:, :, None] - self.red_x[:, None, :]
        dy_br = self.blue_y[:, :, None] - self.red_y[:, None, :]
        d_br = torch.sqrt(dx_br * dx_br + dy_br * dy_br + 1e-8)
        close_br = d_br < rr
        halt_b = halt_b | close_br.any(dim=2)
        halt_r = halt_r | close_br.any(dim=1)

        if halt_b.any():
            self.blue_x = torch.where(halt_b, prev_blue_x, self.blue_x)
            self.blue_y = torch.where(halt_b, prev_blue_y, self.blue_y)
            self.blue_speed = torch.where(halt_b, torch.zeros_like(self.blue_speed), self.blue_speed)
        if halt_r.any():
            self.red_x = torch.where(halt_r, prev_red_x, self.red_x)
            self.red_y = torch.where(halt_r, prev_red_y, self.red_y)
            self.red_speed = torch.where(halt_r, torch.zeros_like(self.red_speed), self.red_speed)

    def _apply_aquaticus_tag_rules(
        self,
        blue_oob: torch.Tensor,
        red_oob: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Enforce Aquaticus tagging:
          - tagger must be untagged and on own side
          - tagged carrier drops flag instantly to home
          - OOB causes automatic tag
        Returns per-env counts:
          - blue_tag_noflag_count
          - blue_tag_withflag_count
          - red_tag_total_count
        """
        # OOB -> auto-tag
        self.blue_tagged = self.blue_tagged | blue_oob
        self.red_tagged = self.red_tagged | red_oob

        # OOB while carrying -> instant flag return
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

        blue_can_tag = (~self.blue_tagged) & self._is_on_home_side("blue", self.blue_x)
        red_can_tag = (~self.red_tagged) & self._is_on_home_side("red", self.red_x)
        red_targetable = ~self.red_tagged
        blue_targetable = ~self.blue_tagged

        blue_tags = in_tag_range & blue_can_tag[:, :, None] & red_targetable[:, None, :]
        red_tags = in_tag_range & red_can_tag[:, None, :] & blue_targetable[:, :, None]

        newly_red_tagged = blue_tags.any(dim=1) & (~self.red_tagged)
        newly_blue_tagged = red_tags.any(dim=2) & (~self.blue_tagged)

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
        b_to_red = torch.sqrt((self.blue_x - self.red_flag_pos[:, None, 0]) ** 2 + (self.blue_y - self.red_flag_pos[:, None, 1]) ** 2 + 1e-8)
        r_to_blue = torch.sqrt((self.red_x - self.blue_flag_pos[:, None, 0]) ** 2 + (self.red_y - self.blue_flag_pos[:, None, 1]) ** 2 + 1e-8)

        blue_grab_env = (~self.red_carrying.any(dim=1)) & ((b_to_red <= 0.8) & (~self.blue_tagged)).any(dim=1)
        red_grab_env = (~self.blue_carrying.any(dim=1)) & ((r_to_blue <= 0.8) & (~self.red_tagged)).any(dim=1)

        if blue_grab_env.any():
            idx = torch.argmax(((b_to_red <= 0.8) & (~self.blue_tagged)).to(torch.int64), dim=1)
            env_idx = torch.where(blue_grab_env)[0]
            self.blue_carrying[env_idx] = False
            self.blue_carrying[env_idx, idx[env_idx]] = True
            self.red_flag_pos[env_idx] = torch.stack([self.blue_x[env_idx, idx[env_idx]], self.blue_y[env_idx, idx[env_idx]]], dim=1)
            self.blue_score[env_idx] += int(self._score_grab_delta())

        if red_grab_env.any():
            idx = torch.argmax(((r_to_blue <= 0.8) & (~self.red_tagged)).to(torch.int64), dim=1)
            env_idx = torch.where(red_grab_env)[0]
            self.red_carrying[env_idx] = False
            self.red_carrying[env_idx, idx[env_idx]] = True
            self.blue_flag_pos[env_idx] = torch.stack([self.red_x[env_idx, idx[env_idx]], self.red_y[env_idx, idx[env_idx]]], dim=1)
            self.red_score[env_idx] += int(self._score_grab_delta())

        b_home_dist = torch.sqrt((self.blue_x - self.blue_flag_home[:, None, 0]) ** 2 + (self.blue_y - self.blue_flag_home[:, None, 1]) ** 2 + 1e-8)
        r_home_dist = torch.sqrt((self.red_x - self.red_flag_home[:, None, 0]) ** 2 + (self.red_y - self.red_flag_home[:, None, 1]) ** 2 + 1e-8)
        blue_capture_now = self.blue_alive & self.blue_carrying & (~self.blue_tagged) & (b_home_dist <= 0.8)
        red_capture_now = self.red_alive & self.red_carrying & (~self.red_tagged) & (r_home_dist <= 0.8)

        b_cap_env = blue_capture_now.any(dim=1)
        r_cap_env = red_capture_now.any(dim=1)
        if b_cap_env.any():
            self.blue_score[b_cap_env] += int(self._score_capture_delta())
            self.blue_carrying[b_cap_env] = False
            self.red_flag_pos[b_cap_env] = self.red_flag_home[b_cap_env]
        if r_cap_env.any():
            self.red_score[r_cap_env] += int(self._score_capture_delta())
            self.red_carrying[r_cap_env] = False
            self.blue_flag_pos[r_cap_env] = self.blue_flag_home[r_cap_env]
        return blue_grab_env, red_grab_env, b_cap_env, r_cap_env

    def _build_blue_targets_from_action(self, macro: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        t_xy = self._decode_targets(target)
        tx, ty = t_xy[..., 0], t_xy[..., 1]
        get_flag = macro == 2
        go_home = macro == 4
        tx = torch.where(get_flag, self.red_flag_pos[:, None, 0], tx)
        ty = torch.where(get_flag, self.red_flag_pos[:, None, 1], ty)
        tx = torch.where(go_home, self.blue_flag_home[:, None, 0], tx)
        ty = torch.where(go_home, self.blue_flag_home[:, None, 1], ty)
        tx = torch.where(self.blue_carrying, self.blue_flag_home[:, None, 0], tx)
        ty = torch.where(self.blue_carrying, self.blue_flag_home[:, None, 1], ty)
        return tx, ty

    def _dense_shaping(self, prev_blue_x: torch.Tensor, prev_blue_y: torch.Tensor, yaw_cmd_blue: torch.Tensor) -> torch.Tensor:
        """
        Dense terms designed to avoid PPO stalling/spinning:
          - potential progress to objective
          - defense presence / escort proximity
          - spin penalty and idle penalty
        """
        carrying = self.blue_carrying
        tgt_x = torch.where(carrying, self.blue_flag_home[:, None, 0], self.red_flag_pos[:, None, 0])
        tgt_y = torch.where(carrying, self.blue_flag_home[:, None, 1], self.red_flag_pos[:, None, 1])
        prev_d = torch.sqrt((prev_blue_x - tgt_x) ** 2 + (prev_blue_y - tgt_y) ** 2 + 1e-8)
        cur_d = torch.sqrt((self.blue_x - tgt_x) ** 2 + (self.blue_y - tgt_y) ** 2 + 1e-8)
        progress = torch.clamp((prev_d - cur_d) / max(1e-6, self.max_dist), min=-1.0, max=1.0).mean(dim=1)

        # Defense presence: if red carries blue flag, reward blue near own flag (intercept posture).
        red_has_flag = self.red_carrying.any(dim=1)
        home_dx = self.blue_x - self.blue_flag_home[:, None, 0]
        home_dy = self.blue_y - self.blue_flag_home[:, None, 1]
        near_home = (torch.sqrt(home_dx * home_dx + home_dy * home_dy + 1e-8) <= 6.0).to(torch.float32).mean(dim=1)
        defense_presence = torch.where(red_has_flag, 0.03 * near_home, torch.zeros_like(near_home))

        # Escort: if blue has enemy flag, reward teammates near carrier.
        blue_has_flag = self.blue_carrying.any(dim=1)
        carrier_idx = torch.argmax(self.blue_carrying.to(torch.int64), dim=1)
        carrier_x = self.blue_x[torch.arange(self.B, device=self.device), carrier_idx]
        carrier_y = self.blue_y[torch.arange(self.B, device=self.device), carrier_idx]
        edx = self.blue_x - carrier_x[:, None]
        edy = self.blue_y - carrier_y[:, None]
        escort = (torch.sqrt(edx * edx + edy * edy + 1e-8) <= 5.0).to(torch.float32).mean(dim=1)
        escort_bonus = torch.where(blue_has_flag, 0.02 * escort, torch.zeros_like(escort))

        # Spin penalty: large yaw with tiny translational progress.
        yaw_abs = torch.abs(yaw_cmd_blue) / max(1e-6, float(self.cfg.max_yaw_rate_rps))
        move_dist = torch.sqrt((self.blue_x - prev_blue_x) ** 2 + (self.blue_y - prev_blue_y) ** 2 + 1e-8)
        spin_pen = self.cfg.spin_penalty_coef * (yaw_abs * (move_dist < 0.03).to(torch.float32)).mean(dim=1)

        # Idle penalty: almost no movement by the team.
        idle_pen = self.cfg.idle_penalty_coef * (self.blue_speed.mean(dim=1) < 0.15).to(torch.float32)

        dense = progress + defense_presence + escort_bonus - spin_pen - idle_pen
        self._last_dense_progress = progress
        return dense

    def _sparse_reward_points(
        self,
        blue_grab_env: torch.Tensor,
        red_grab_env: torch.Tensor,
        blue_cap_env: torch.Tensor,
        red_cap_env: torch.Tensor,
        blue_tag_noflag: torch.Tensor,
        blue_tag_withflag: torch.Tensor,
        red_tag_total: torch.Tensor,
        blue_oob: torch.Tensor,
    ) -> torch.Tensor:
        # Aquaticus table (points):
        # tag no-flag +100 / opp -100
        # tag with-flag +50 / opp -100
        # flag grab +50 / opp -50
        # capture +100 / opp -100
        # OOB -100
        r = torch.zeros((self.B,), dtype=torch.float32, device=self.device)

        r += 100.0 * blue_tag_noflag
        r += 50.0 * blue_tag_withflag
        r -= 100.0 * red_tag_total

        r += 50.0 * blue_grab_env.to(torch.float32)
        r -= 50.0 * red_grab_env.to(torch.float32)
        r += 100.0 * blue_cap_env.to(torch.float32)
        r -= 100.0 * red_cap_env.to(torch.float32)
        r += -100.0 * blue_oob.sum(dim=1).to(torch.float32)
        return r

    def _reward_total(
        self,
        dense: torch.Tensor,
        sparse_points: torch.Tensor,
        stalemate_trigger: torch.Tensor,
    ) -> torch.Tensor:
        sparse_norm = sparse_points / 100.0
        raw = self.cfg.sparse_weight * sparse_norm + self.cfg.dense_weight * dense
        raw = raw + torch.where(stalemate_trigger, torch.tensor(float(self.cfg.stalemate_penalty), device=self.device), torch.tensor(0.0, device=self.device))
        scaled = torch.tanh(raw / max(1e-6, float(self.cfg.reward_scale)))
        return torch.clamp(scaled, -float(self.cfg.reward_clip), float(self.cfg.reward_clip))

    def step(self, blue_action_flat: torch.Tensor) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, List[dict]]:
        self._apply_profile_runtime()
        if blue_action_flat.device != self.device:
            blue_action_flat = blue_action_flat.to(self.device)
        a = blue_action_flat.view(self.B, self.Nb, 2)
        macro = torch.remainder(a[..., 0].long(), self.cfg.n_macros)
        targ = torch.remainder(a[..., 1].long(), self.cfg.n_targets)

        prev_blue_x = self.blue_x.clone()
        prev_blue_y = self.blue_y.clone()
        prev_red_x = self.red_x.clone()
        prev_red_y = self.red_y.clone()

        btx, bty = self._build_blue_targets_from_action(macro, targ)
        rtx, rty = self._red_scripted_actions()

        # Tagged agents: OpRegion-like forced safe return to home region.
        if self.blue_tagged.any():
            btx = torch.where(self.blue_tagged, self.blue_flag_home[:, None, 0], btx)
            bty = torch.where(self.blue_tagged, self.blue_flag_home[:, None, 1], bty)
        if self.red_tagged.any():
            rtx = torch.where(self.red_tagged, self.red_flag_home[:, None, 0], rtx)
            rty = torch.where(self.red_tagged, self.red_flag_home[:, None, 1], rty)
        blue_speed_cap = torch.where(
            self.blue_tagged,
            torch.full_like(self.blue_speed, float(self.cfg.opregion_safe_speed_cps)),
            torch.full_like(self.blue_speed, float(self.cfg.max_speed_cps)),
        )
        red_speed_cap = torch.where(
            self.red_tagged,
            torch.full_like(self.red_speed, float(self.cfg.opregion_safe_speed_cps)),
            torch.full_like(self.red_speed, float(self.cfg.max_speed_cps)),
        )

        self.blue_x, self.blue_y, self.blue_heading, self.blue_speed, blue_oob, yaw_cmd_blue = self._integrate_side(
            self.blue_x, self.blue_y, self.blue_heading, self.blue_speed, self.blue_alive, btx, bty, speed_cap=blue_speed_cap
        )
        self.red_x, self.red_y, self.red_heading, self.red_speed, red_oob, _ = self._integrate_side(
            self.red_x, self.red_y, self.red_heading, self.red_speed, self.red_alive, rtx, rty, speed_cap=red_speed_cap
        )

        # AvoidCollision safety guardrail (halt motion if too close)
        self._apply_avoid_collision_guard(prev_blue_x, prev_blue_y, prev_red_x, prev_red_y)

        if bool(self.cfg.aquaticus_profile) or self.rules_profile == "AQUATICUS_2024":
            blue_tag_noflag, blue_tag_withflag, red_tag_total = self._apply_aquaticus_tag_rules(blue_oob, red_oob)
            self._untag_if_home()
            kill_blue = torch.zeros_like(self.blue_tagged)
            kill_red = torch.zeros_like(self.red_tagged)
        else:
            kill_blue, kill_red, blue_had_flag, red_had_flag = self._apply_suppression()
            blue_tag_noflag = (kill_red & (~red_had_flag)).sum(dim=1).to(torch.float32)
            blue_tag_withflag = (kill_red & red_had_flag).sum(dim=1).to(torch.float32)
            red_tag_total = kill_blue.sum(dim=1).to(torch.float32)
            self._respawn_timers()

        blue_grab_env, red_grab_env, blue_cap_env, red_cap_env = self._apply_flag_rules()

        dense = self._dense_shaping(prev_blue_x, prev_blue_y, yaw_cmd_blue)
        sparse_points = self._sparse_reward_points(
            blue_grab_env, red_grab_env, blue_cap_env, red_cap_env, blue_tag_noflag, blue_tag_withflag, red_tag_total, blue_oob
        )

        self.step_count += 1
        event_happened = (
            blue_grab_env
            | red_grab_env
            | blue_cap_env
            | red_cap_env
            | (blue_tag_noflag > 0.0)
            | (blue_tag_withflag > 0.0)
            | (red_tag_total > 0.0)
        )
        low_progress = torch.abs(self._last_dense_progress) < float(self.cfg.stalemate_progress_eps)
        no_event = ~event_happened
        self.stalemate_steps = torch.where(no_event & low_progress, self.stalemate_steps + 1, torch.zeros_like(self.stalemate_steps))
        stalemate_trigger = self.stalemate_steps >= int(self.cfg.stalemate_max_steps)

        terminated = (self.blue_score >= self.score_limit) | (self.red_score >= self.score_limit)
        truncated = (self.step_count >= self.max_steps) | stalemate_trigger
        self.done = terminated | truncated
        self.truncated = truncated

        reward = self._reward_total(dense, sparse_points, stalemate_trigger)
        obs = self.get_obs()
        info = self._build_info(dense=dense, sparse_points=sparse_points, stalemate=stalemate_trigger)
        return (
            obs,
            reward.detach().cpu().numpy().astype(np.float32),
            terminated.detach().cpu().numpy(),
            truncated.detach().cpu().numpy(),
            info,
        )

    def _build_info(self, dense: torch.Tensor, sparse_points: torch.Tensor, stalemate: torch.Tensor) -> List[dict]:
        out: List[dict] = []
        bs = self.blue_score.detach().cpu().numpy()
        rs = self.red_score.detach().cpu().numpy()
        steps = self.step_count.detach().cpu().numpy()
        d_np = dense.detach().cpu().numpy()
        s_np = sparse_points.detach().cpu().numpy()
        st_np = stalemate.detach().cpu().numpy()
        for i in range(self.B):
            out.append(
                {
                    "blue_score": int(bs[i]),
                    "red_score": int(rs[i]),
                    "decision_steps": int(steps[i]),
                    "phase": self._phase,
                    "league_mode": bool(self._league_mode),
                    "opponent_kind": self._opponent_kind.lower(),
                    "opponent_key": self._opponent_key,
                    "rules_profile": self.rules_profile,
                    "dense_reward": float(d_np[i]),
                    "sparse_points": float(s_np[i]),
                    "stalemate_truncated": bool(st_np[i]),
                }
            )
        return out

    def _scatter_points(self, grid: torch.Tensor, ch: int, x: torch.Tensor, y: torch.Tensor, live: torch.Tensor) -> None:
        cx = torch.clamp((x / max(1.0, float(self.cols - 1)) * float(CNN_COLS - 1)).round().long(), 0, CNN_COLS - 1)
        cy = torch.clamp((y / max(1.0, float(self.rows - 1)) * float(CNN_ROWS - 1)).round().long(), 0, CNN_ROWS - 1)
        b_idx = torch.arange(self.B, device=self.device).view(self.B, 1).expand(-1, cx.shape[1])
        if live.any():
            grid[b_idx[live], ch, cy[live], cx[live]] = 1.0

    def _build_grid_obs(self) -> torch.Tensor:
        grid = torch.zeros((self.B, self.Nb, NUM_CNN_CHANNELS, CNN_ROWS, CNN_COLS), dtype=torch.float32, device=self.device)
        for i in range(self.Nb):
            self_live = self.blue_alive[:, i : i + 1]
            self._scatter_points(grid[:, i], 0, self.blue_x[:, i : i + 1], self.blue_y[:, i : i + 1], self_live)

            friend_live = self.blue_alive.clone()
            friend_live[:, i] = False
            self._scatter_points(grid[:, i], 1, self.blue_x, self.blue_y, friend_live)

            ex = self.red_x
            ey = self.red_y
            elive = self.red_alive
            if self.cfg.sensor_range_cells < 1e8:
                dx = ex - self.blue_x[:, i : i + 1]
                dy = ey - self.blue_y[:, i : i + 1]
                dist = torch.sqrt(dx * dx + dy * dy + 1e-8)
                in_range = dist <= float(self.cfg.sensor_range_cells)
                if self.cfg.sensor_dropout_prob > 0.0:
                    drop = torch.rand(in_range.shape, generator=self._rng, device=self.device) < float(self.cfg.sensor_dropout_prob)
                    in_range = in_range & (~drop)
                elive = elive & in_range
                if self.cfg.sensor_noise_sigma_cells > 0.0:
                    ex = torch.clamp(ex + self._randn(ex.shape) * float(self.cfg.sensor_noise_sigma_cells), 0.0, float(max(0, self.cols - 1)))
                    ey = torch.clamp(ey + self._randn(ey.shape) * float(self.cfg.sensor_noise_sigma_cells), 0.0, float(max(0, self.rows - 1)))
            self._scatter_points(grid[:, i], 2, ex, ey, elive)

            self._scatter_points(grid[:, i], 5, self.blue_flag_pos[:, 0:1], self.blue_flag_pos[:, 1:2], torch.ones((self.B, 1), dtype=torch.bool, device=self.device))
            self._scatter_points(grid[:, i], 6, self.red_flag_pos[:, 0:1], self.red_flag_pos[:, 1:2], torch.ones((self.B, 1), dtype=torch.bool, device=self.device))
        return grid

    def _build_vec_obs(self) -> torch.Tensor:
        """
        Normalized observation vector (stable for PPO critic):
          0: x_norm in [0,1]
          1: y_norm in [0,1]
          2: heading/pi in [-1,1]
          3: speed/max_speed in [0,1]
          4-7: relative flag deltas normalized to [-1,1]
          8: carrying flag (0/1)
          9: nearest enemy distance normalized to [0,1]
          10: time fraction in [0,1]
          11: agent id normalized in [0,1]
        """
        out = torch.zeros((self.B, self.Nb, 12), dtype=torch.float32, device=self.device)
        cols = max(1.0, float(self.cols - 1))
        rows = max(1.0, float(self.rows - 1))
        max_speed = max(1e-6, float(self.cfg.max_speed_cps))

        out[..., 0] = torch.clamp(self.blue_x / cols, 0.0, 1.0)
        out[..., 1] = torch.clamp(self.blue_y / rows, 0.0, 1.0)
        # Discretized bearing theta_i (formal Aquaticus-style state element).
        heading_norm = (self.blue_heading + math.pi) / (2.0 * math.pi)
        heading_bins = torch.floor(torch.clamp(heading_norm, 0.0, 0.9999) * 16.0) / 15.0
        out[..., 2] = torch.clamp(heading_bins * 2.0 - 1.0, -1.0, 1.0)
        out[..., 3] = torch.clamp(self.blue_speed / max_speed, 0.0, 1.0)
        out[..., 4] = torch.clamp((self.red_flag_pos[:, None, 0] - self.blue_x) / max(1.0, float(self.cols)), -1.0, 1.0)
        out[..., 5] = torch.clamp((self.red_flag_pos[:, None, 1] - self.blue_y) / max(1.0, float(self.rows)), -1.0, 1.0)
        out[..., 6] = torch.clamp((self.blue_flag_pos[:, None, 0] - self.blue_x) / max(1.0, float(self.cols)), -1.0, 1.0)
        out[..., 7] = torch.clamp((self.blue_flag_pos[:, None, 1] - self.blue_y) / max(1.0, float(self.rows)), -1.0, 1.0)
        out[..., 8] = self.blue_carrying.to(torch.float32)

        dx = self.red_x[:, None, :] - self.blue_x[:, :, None]
        dy = self.red_y[:, None, :] - self.blue_y[:, :, None]
        d = torch.sqrt(dx * dx + dy * dy + 1e-8)
        nearest_enemy = torch.min(d, dim=2).values
        out[..., 9] = torch.clamp(nearest_enemy / max(1e-6, self.max_dist), 0.0, 1.0)
        out[..., 10] = torch.clamp(self.step_count[:, None].to(torch.float32) / max(1.0, float(self.max_steps)), 0.0, 1.0)

        agent_id = torch.arange(self.Nb, device=self.device, dtype=torch.float32)
        out[..., 11] = agent_id[None, :] / max(1.0, float(self.Nb - 1))
        return out

    def _build_action_mask(self) -> torch.Tensor:
        mask = torch.ones((self.B, self.Nb, self.cfg.n_macros + self.cfg.n_targets), dtype=torch.float32, device=self.device)
        dead = ~self.blue_alive
        if dead.any():
            mask[:, :, : self.cfg.n_macros][dead] = 0.0
            mask[:, :, self.cfg.n_macros :][dead] = 0.0
            mask[:, :, 0][dead] = 1.0
            mask[:, :, self.cfg.n_macros + 0][dead] = 1.0
        carrying = self.blue_carrying
        if carrying.any():
            idx_get, idx_grab, idx_place, idx_home = 2, 1, 3, 4
            mask[:, :, idx_get][carrying] = 0.0
            mask[:, :, idx_grab][carrying] = 0.0
            mask[:, :, idx_place][carrying] = 0.0
            mask[:, :, idx_home][carrying] = 1.0
        return mask.reshape(self.B, -1)

    def get_obs(self) -> Dict[str, np.ndarray]:
        return {
            "grid": self._build_grid_obs().detach().cpu().numpy().astype(np.float32),
            "vec": self._build_vec_obs().detach().cpu().numpy().astype(np.float32),
            "agent_mask": self.blue_alive.detach().cpu().numpy().astype(np.float32),
            "mask": self._build_action_mask().detach().cpu().numpy().astype(np.float32),
        }

    def get_global_state(self) -> np.ndarray:
        g = torch.zeros((self.B, GLOBAL_STATE_CHANNELS, CNN_ROWS, CNN_COLS), dtype=torch.float32, device=self.device)
        self._scatter_points(g, 0, self.blue_x, self.blue_y, self.blue_alive)
        self._scatter_points(g, 1, self.red_x, self.red_y, self.red_alive)
        self._scatter_points(g, 6, self.blue_flag_pos[:, 0:1], self.blue_flag_pos[:, 1:2], torch.ones((self.B, 1), dtype=torch.bool, device=self.device))
        self._scatter_points(g, 7, self.red_flag_pos[:, 0:1], self.red_flag_pos[:, 1:2], torch.ones((self.B, 1), dtype=torch.bool, device=self.device))
        return g.reshape(self.B, -1).detach().cpu().numpy().astype(np.float32)


class GPUCTFVecEnv(VecEnv):
    """SB3 VecEnv wrapper around BatchedCTFCore."""

    def __init__(self, cfg: GPUFieldConfig):
        self.core = BatchedCTFCore(cfg)
        self.cfg = cfg
        self._n_macros = int(cfg.n_macros)
        self._n_targets = int(cfg.n_targets)
        self._n_blue = int(cfg.max_blue_agents)
        obs_space = spaces.Dict(
            {
                "grid": spaces.Box(low=0.0, high=1.0, shape=(self._n_blue, NUM_CNN_CHANNELS, CNN_ROWS, CNN_COLS), dtype=np.float32),
                "vec": spaces.Box(low=-1.0, high=1.0, shape=(self._n_blue, 12), dtype=np.float32),
                "agent_mask": spaces.Box(low=0.0, high=1.0, shape=(self._n_blue,), dtype=np.float32),
                "mask": spaces.Box(low=0.0, high=1.0, shape=(self._n_blue * (self._n_macros + self._n_targets),), dtype=np.float32),
            }
        )
        action_space = spaces.MultiDiscrete([self._n_macros, self._n_targets] * self._n_blue)
        super().__init__(cfg.n_envs, obs_space, action_space)
        self._pending_actions: Optional[np.ndarray] = None

    def reset(self) -> Dict[str, np.ndarray]:
        self.core.reset_all()
        return self.core.get_obs()

    def step_async(self, actions: np.ndarray) -> None:
        self._pending_actions = np.asarray(actions, dtype=np.int64)

    def step_wait(self):
        assert self._pending_actions is not None, "step_async() must be called before step_wait()"
        actions = torch.as_tensor(self._pending_actions, dtype=torch.int64, device=self.core.device)
        obs, rew, term, trunc, infos = self.core.step(actions)
        done = np.logical_or(term, trunc)
        if done.any():
            reset_mask = torch.from_numpy(done).to(self.core.device)
            for i in np.where(done)[0]:
                infos[i] = dict(infos[i])
                infos[i]["terminal_observation"] = {k: v[i].copy() for k, v in obs.items()}
            self.core.reset_indices(reset_mask)
            obs = self.core.get_obs()
        self._pending_actions = None
        return obs, rew, done, infos

    def close(self) -> None:
        self._pending_actions = None

    def get_attr(self, attr_name: str, indices=None):
        idx = self._get_indices(indices)
        val = getattr(self.core, attr_name)
        return [val for _ in idx]

    def set_attr(self, attr_name: str, value: Any, indices=None) -> None:
        idx = self._get_indices(indices)
        if len(idx) != self.num_envs:
            return
        setattr(self.core, attr_name, value)

    def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs):
        idx = self._get_indices(indices)
        method = getattr(self.core, method_name)
        out = method(*method_args, **method_kwargs)
        return [out for _ in idx]

    def env_is_wrapped(self, wrapper_class, indices=None):
        idx = self._get_indices(indices)
        return [False for _ in idx]


class GPUCTFSingleEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg: Optional[GPUFieldConfig] = None):
        cfg = cfg or GPUFieldConfig(n_envs=1)
        cfg.n_envs = 1
        self.vec = GPUCTFVecEnv(cfg)
        self.action_space = self.vec.action_space
        self.observation_space = self.vec.observation_space

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            torch.manual_seed(int(seed))
        obs = self.vec.reset()
        return {k: v[0] for k, v in obs.items()}, {}

    def step(self, action):
        self.vec.step_async(np.asarray(action, dtype=np.int64)[None, ...])
        obs, rew, done, infos = self.vec.step_wait()
        terminated = bool(done[0])
        truncated = bool(infos[0].get("decision_steps", 0) >= self.vec.core.max_steps or infos[0].get("stalemate_truncated", False))
        return {k: v[0] for k, v in obs.items()}, float(rew[0]), terminated, truncated, infos[0]

    def close(self):
        self.vec.close()


__all__ = [
    "GPUFieldConfig",
    "BatchedCTFCore",
    "GPUCTFVecEnv",
    "GPUCTFSingleEnv",
    "CNN_COLS",
    "CNN_ROWS",
    "NUM_CNN_CHANNELS",
]
