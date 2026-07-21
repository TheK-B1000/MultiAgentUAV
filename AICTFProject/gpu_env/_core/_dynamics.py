"""Dynamics and runtime-profile helpers for BatchedCTFCore."""
from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import torch

try:
    from opponent_params import sample_batched_opponent_params
except ImportError:
    sample_batched_opponent_params = None


def align_speed_cap_to_speed(speed: torch.Tensor, speed_cap: torch.Tensor) -> torch.Tensor:
    """Ensure speed_cap matches speed's (B, N) layout for torch.minimum / arithmetic."""
    cap = torch.clamp(speed_cap.to(dtype=speed.dtype, device=speed.device), min=0.0)
    if cap.shape == speed.shape:
        return cap
    if cap.dim() == 1 and cap.shape[0] == speed.shape[0]:
        return cap[:, None].expand_as(speed)
    if cap.numel() == 1:
        return cap.view(1, 1).expand_as(speed)
    if cap.numel() == speed.numel():
        return cap.reshape(speed.shape)
    raise RuntimeError(
        f"speed_cap shape {tuple(cap.shape)} / numel={cap.numel()} incompatible with "
        f"speed {tuple(speed.shape)} / numel={speed.numel()}"
    )


class _DynamicsMixin:
    def _apply_dynamics_tensor(
        self,
        cfg: Dict[str, Any],
        key: str,
        attr: str,
        low: float,
        high: float,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Apply a single dynamics config key to a batched tensor. Used by set_dynamics_config."""
        if key not in cfg:
            return
        val = cfg[key]
        tensor = getattr(self, attr)
        if isinstance(val, torch.Tensor):
            t = val.to(device=self.device, dtype=dtype).reshape(-1)
            if t.numel() == self.B:
                setattr(self, attr, torch.clamp(t, low, high))
            else:
                scalar = torch.clamp(t[0], low, high).item()
                tensor.fill_(int(scalar) if dtype == torch.int32 else float(scalar))
        else:
            scalar = max(low, min(high, int(val) if dtype == torch.int32 else float(val)))
            tensor.fill_(scalar)

    def _apply_profile_runtime(self) -> None:
        # Optional stress schedule by phase (same hook shape used by train_ppo callbacks).
        if isinstance(self._stress_schedule, dict):
            for phase_name in set(self._phase):
                env_mask = self._phase_tensor_equals((phase_name,))
                if not env_mask.any():
                    continue
                p = self._stress_schedule.get(str(phase_name).upper(), None)
                if isinstance(p, dict):
                    if "current_strength_cps" in p:
                        self.rt_current_strength_cps[env_mask] = float(p["current_strength_cps"])
                    if "drift_sigma_cells" in p:
                        self.rt_drift_sigma_cells[env_mask] = float(p["drift_sigma_cells"])

        # Aquaticus profile keeps marine constraints and can add mild stochastic water drift.
        if self.cfg.aquaticus_profile:
            self.cfg.max_speed_cps = min(float(self.cfg.max_speed_cps), 2.2)
            self.cfg.max_accel_cps2 = min(float(self.cfg.max_accel_cps2), 2.0)
            self.cfg.max_yaw_rate_rps = min(float(self.cfg.max_yaw_rate_rps), 4.0)
            # Tagging radius should be local, not half-field. Clamp to a few cells.
            self.cfg.tag_range_cells = min(float(self.cfg.tag_range_cells), 2.5)

    def _apply_neutral_red_params(self, env_mask: torch.Tensor) -> None:
        idx = torch.where(env_mask)[0]
        if idx.numel() == 0:
            return
        self.red_deception_prob[idx] = 0.0
        self.red_speed_mult[idx] = 1.0
        self.red_attacker_style[idx] = 0
        self.red_defender_style[idx] = 0
        self.red_role_switch_prob[idx] = 0.0
        self.red_coordinated_attack[idx] = False
        self.red_attack_sync_window[idx] = 0
        self.red_coord_ticks_left[idx] = 0

    def _apply_opponent_params_for_mask(self, env_mask: torch.Tensor) -> None:
        if sample_batched_opponent_params is None:
            return
        # Only SCRIPTED tags with defined ``sample_batched_opponent_params`` branches apply here.
        # SNAPSHOT opponents should keep neutral/default red dynamics so they behave
        # like true self-play rather than inheriting scripted-opponent boosts.
        idx = torch.where(env_mask)[0]
        if idx.numel() == 0:
            return
        snapshot_mask = torch.as_tensor(
            [str(self._opponent_kind[env_i]).upper() == "SNAPSHOT" for env_i in idx.detach().cpu().tolist()],
            device=self.device,
            dtype=torch.bool,
        )
        if snapshot_mask.any():
            neutral_mask = torch.zeros((self.B,), dtype=torch.bool, device=self.device)
            neutral_mask[idx[snapshot_mask]] = True
            self._apply_neutral_red_params(neutral_mask)
        _scripted_param_tags = frozenset(
            {
                "OP1",
                "OP2",
                "OP3",
                "OP4",
                "OP5",
                "OP5_RUSHER",
                "OP6",
                "OP6_TURTLE",
                "OP7",
                "OP7_SWITCHER",
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
        grouped: Dict[Tuple[str, str], List[int]] = {}
        for env_i in idx.detach().cpu().tolist():
            use_kind = str(self._opponent_kind[env_i]).upper()
            use_key = str(self._opponent_key[env_i]).upper()
            if use_kind == "SNAPSHOT":
                continue
            if use_kind not in ("SCRIPTED",) or use_key not in _scripted_param_tags:
                if use_kind == "SPECIES":
                    use_kind = "SCRIPTED"
                    use_key = "OP3"
                else:
                    continue
            grouped.setdefault((use_kind, use_key), []).append(env_i)
        for (use_kind, use_key), env_list in grouped.items():
            sub_idx = torch.as_tensor(env_list, device=self.device, dtype=torch.int64)
            opp_params = sample_batched_opponent_params(
                kind=use_kind,
                key=use_key,
                phase=use_key,
                n_agents=self.Nr,
                batch_size=int(sub_idx.numel()),
                device=self.device,
                generator=self._rng,
            )
            if "deception_prob" in opp_params:
                self.red_deception_prob[sub_idx] = opp_params["deception_prob"].to(device=self.device, dtype=self.red_deception_prob.dtype)
            if "speed_mult" in opp_params:
                sm = opp_params["speed_mult"].to(device=self.device, dtype=self.red_speed_mult.dtype).reshape(-1)
                n_sub = int(sub_idx.numel())
                if sm.numel() == n_sub:
                    self.red_speed_mult[sub_idx] = sm
                elif sm.numel() == 1:
                    self.red_speed_mult[sub_idx] = sm[0]
                else:
                    # e.g. per-agent samples leaked in — collapse to one value per env row
                    self.red_speed_mult[sub_idx] = sm.mean()
            if "attacker_style" in opp_params:
                self.red_attacker_style[sub_idx] = opp_params["attacker_style"].to(device=self.device, dtype=self.red_attacker_style.dtype)
            if "defender_style" in opp_params:
                self.red_defender_style[sub_idx] = opp_params["defender_style"].to(device=self.device, dtype=self.red_defender_style.dtype)
            if "role_switch_prob" in opp_params:
                self.red_role_switch_prob[sub_idx] = opp_params["role_switch_prob"].to(device=self.device, dtype=self.red_role_switch_prob.dtype)
            if "coordinated_attack" in opp_params:
                ca = opp_params["coordinated_attack"].to(device=self.device).reshape(-1).to(torch.bool)
                n_sub = int(sub_idx.numel())
                if ca.numel() == n_sub:
                    self.red_coordinated_attack[sub_idx] = ca
                elif ca.numel() == 1:
                    self.red_coordinated_attack[sub_idx] = bool(ca[0].item())
                else:
                    self.red_coordinated_attack[sub_idx] = ca[:n_sub]
            if "attack_sync_window" in opp_params:
                sw = opp_params["attack_sync_window"].to(device=self.device, dtype=self.red_attack_sync_window.dtype).reshape(-1)
                n_sub = int(sub_idx.numel())
                if sw.numel() == n_sub:
                    self.red_attack_sync_window[sub_idx] = sw
                elif sw.numel() == 1:
                    self.red_attack_sync_window[sub_idx] = int(sw[0].item())
                else:
                    self.red_attack_sync_window[sub_idx] = sw[:n_sub]
            self.red_coord_ticks_left[sub_idx] = 0

    def _align_speed_cap_to_speed(self, speed: torch.Tensor, speed_cap: torch.Tensor) -> torch.Tensor:
        return align_speed_cap_to_speed(speed, speed_cap)

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
        speed_overdrive_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Marine kinematics with acceleration/yaw constraints and turn-radius limiting.
        """
        dt = self.dt
        dx = target_x - x
        dy = target_y - y
        desired_heading = torch.atan2(dy, dx)
        err = (desired_heading - heading + math.pi) % (2.0 * math.pi) - math.pi

        max_yaw = min(4.0, float(self.cfg.max_yaw_rate_rps))
        yaw_rate_cmd = torch.clamp(err / max(1e-6, dt), -max_yaw, max_yaw)
        # Turn-radius bound: |omega| <= v / R_min.
        # Floor raised to 1.5 rad/s so agents near-stopped at a wall face can rotate
        # 0.75 rad/step, enough to clear the wall in ~2 bounces rather than 4.
        min_r = max(1e-3, float(self.cfg.min_turn_radius_cells))
        omega_bound = torch.clamp(speed / min_r, min=1.5, max=max_yaw)
        yaw_rate_cmd = torch.clamp(yaw_rate_cmd, -omega_bound, omega_bound)

        max_speed = min(2.2, float(self.cfg.max_speed_cps))
        max_accel = min(2.0, float(self.cfg.max_accel_cps2))
        max_speed_t = torch.full_like(speed, max_speed)
        cap = None
        if speed_cap is not None:
            cap = torch.clamp(speed_cap, min=0.0)
            if speed_overdrive_mask is not None:
                overdrive = speed_overdrive_mask.to(device=speed.device, dtype=torch.bool)
                max_speed_t = torch.where(overdrive, torch.maximum(max_speed_t, cap), max_speed_t)
        desired_speed = max_speed_t
        if cap is not None:
            desired_speed = torch.minimum(desired_speed, cap)
        dist = torch.sqrt(dx * dx + dy * dy + 1e-8)
        # Deceleration zone near objective to prevent overshoot/spin lock.
        desired_speed = torch.where(dist < 2.0, desired_speed * torch.clamp(dist / 2.0, 0.0, 1.0), desired_speed)
        accel_cmd = torch.clamp(
            (desired_speed - speed) / max(1e-6, dt),
            -max_accel,
            max_accel,
        )

        speed2 = torch.minimum(torch.clamp(speed + accel_cmd * dt, min=0.0), max_speed_t)
        if speed_cap is not None:
            aligned_cap = self._align_speed_cap_to_speed(speed, speed_cap)
            speed2 = torch.minimum(speed2, aligned_cap)
        heading2 = heading + yaw_rate_cmd * dt
        # CUDA/trig can misbehave if heading/speed dtypes differ (e.g. float64 vs float32).
        s2 = speed2.float()
        h2 = heading2.float()
        cur = self.rt_current_strength_cps[:, None].to(dtype=s2.dtype, device=s2.device)
        vx = s2 * torch.cos(h2) + cur
        vy = s2 * torch.sin(h2)

        nx_raw = x + vx * dt
        ny_raw = y + vy * dt
        # Small tolerance prevents numerical edge jitter from causing false OOB tags.
        tol = 1e-4
        oob = (
            (nx_raw < (0.0 - tol))
            | (nx_raw > (float(max(0, self.cols - 1)) + tol))
            | (ny_raw < (0.0 - tol))
            | (ny_raw > (float(max(0, self.rows - 1)) + tol))
        )
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

    def _apply_avoid_collision_guard(
        self,
        prev_blue_x: torch.Tensor,
        prev_blue_y: torch.Tensor,
        prev_red_x: torch.Tensor,
        prev_red_y: torch.Tensor,
    ) -> None:
        """
        Tangential repulsion: if agents get too close, apply a small shove that nudges
        them apart instead of halting them in place. This prevents "nose-to-nose" lockups.

        This shove is a non-physical training convenience (an artificial impulse) and
        differs from the exact Aquaticus boat dynamics, but helps keep agents from
        freezing in unrealistic collision configurations.
        """
        rr = float(self.cfg.avoid_collision_radius_cells)
        if rr <= 0.0:
            return

        shove = 0.5
        # Blue-Blue repulsion
        ddx = self.blue_x[:, :, None] - self.blue_x[:, None, :]
        ddy = self.blue_y[:, :, None] - self.blue_y[:, None, :]
        d = torch.sqrt(ddx * ddx + ddy * ddy + 1e-8)
        eye = torch.eye(self.Nb, dtype=torch.bool, device=self.device)[None, :, :]
        close_bb = (d < rr) & (~eye) & self.blue_alive[:, :, None] & self.blue_alive[:, None, :]
        if close_bb.any():
            dir_x = ddx / d
            dir_y = ddy / d
            # Net repulsion from all close neighbors
            fx = (dir_x * close_bb.to(dir_x.dtype)).sum(dim=2)
            fy = (dir_y * close_bb.to(dir_y.dtype)).sum(dim=2)
            norm = torch.sqrt(fx * fx + fy * fy + 1e-8)
            fx = fx / norm * shove
            fy = fy / norm * shove
            self.blue_x = torch.where(self.blue_alive, torch.clamp(self.blue_x + fx, 0.0, float(max(0, self.cols - 1))), self.blue_x)
            self.blue_y = torch.where(self.blue_alive, torch.clamp(self.blue_y + fy, 0.0, float(max(0, self.rows - 1))), self.blue_y)

        # Red-Red repulsion
        ddx_r = self.red_x[:, :, None] - self.red_x[:, None, :]
        ddy_r = self.red_y[:, :, None] - self.red_y[:, None, :]
        d_r = torch.sqrt(ddx_r * ddx_r + ddy_r * ddy_r + 1e-8)
        eye_r = torch.eye(self.Nr, dtype=torch.bool, device=self.device)[None, :, :]
        close_rr = (d_r < rr) & (~eye_r) & self.red_alive[:, :, None] & self.red_alive[:, None, :]
        if close_rr.any():
            dir_xr = ddx_r / d_r
            dir_yr = ddy_r / d_r
            fx_r = (dir_xr * close_rr.to(dir_xr.dtype)).sum(dim=2)
            fy_r = (dir_yr * close_rr.to(dir_yr.dtype)).sum(dim=2)
            norm_r = torch.sqrt(fx_r * fx_r + fy_r * fy_r + 1e-8)
            fx_r = fx_r / norm_r * shove
            fy_r = fy_r / norm_r * shove
            self.red_x = torch.where(self.red_alive, torch.clamp(self.red_x + fx_r, 0.0, float(max(0, self.cols - 1))), self.red_x)
            self.red_y = torch.where(self.red_alive, torch.clamp(self.red_y + fy_r, 0.0, float(max(0, self.rows - 1))), self.red_y)

        # Blue-Red repulsion
        dx_br = self.blue_x[:, :, None] - self.red_x[:, None, :]
        dy_br = self.blue_y[:, :, None] - self.red_y[:, None, :]
        d_br = torch.sqrt(dx_br * dx_br + dy_br * dy_br + 1e-8)
        close_br = (d_br < rr) & self.blue_alive[:, :, None] & self.red_alive[:, None, :]
        if close_br.any():
            dir_xbr = dx_br / d_br
            dir_ybr = dy_br / d_br
            # For blue, repel away from red
            fx_b = (dir_xbr * close_br.to(dir_xbr.dtype)).sum(dim=2)
            fy_b = (dir_ybr * close_br.to(dir_ybr.dtype)).sum(dim=2)
            norm_b = torch.sqrt(fx_b * fx_b + fy_b * fy_b + 1e-8)
            fx_b = fx_b / norm_b * shove
            fy_b = fy_b / norm_b * shove
            self.blue_x = torch.where(self.blue_alive, torch.clamp(self.blue_x + fx_b, 0.0, float(max(0, self.cols - 1))), self.blue_x)
            self.blue_y = torch.where(self.blue_alive, torch.clamp(self.blue_y + fy_b, 0.0, float(max(0, self.rows - 1))), self.blue_y)
            # For red, repel in the opposite direction
            fx_r2 = -(dir_xbr * close_br.to(dir_xbr.dtype)).sum(dim=1)
            fy_r2 = -(dir_ybr * close_br.to(dir_ybr.dtype)).sum(dim=1)
            norm_r2 = torch.sqrt(fx_r2 * fx_r2 + fy_r2 * fy_r2 + 1e-8)
            fx_r2 = fx_r2 / norm_r2 * shove
            fy_r2 = fy_r2 / norm_r2 * shove
            self.red_x = torch.where(self.red_alive, torch.clamp(self.red_x + fx_r2, 0.0, float(max(0, self.cols - 1))), self.red_x)
            self.red_y = torch.where(self.red_alive, torch.clamp(self.red_y + fy_r2, 0.0, float(max(0, self.rows - 1))), self.red_y)

    # ------------------------------------------------------------------
    # Mine system: pickups spawn; agents GRAB_MINE then PLACE_MINE anywhere.
    # ------------------------------------------------------------------
    def _is_on_home_side(self, side: str, x: torch.Tensor) -> torch.Tensor:
        mid = float(self.cols - 1) * 0.5
        if side == "blue":
            return x <= mid
        return x >= mid

