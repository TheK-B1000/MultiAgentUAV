"""Runtime buffer and mine state — scratch tensors reset each episode.

Two groups:
- **Runtime buffers** (``rt_*``): per-env scalars that parameterise simulation
  noise and scripted-red behaviour; reset each episode via domain-randomisation
  or static config values.
- **Mine state** (``blue_mine_*``, ``red_mine_*``, ``pickup_*``): mine charges,
  active flags, positions, and pickup respawn; fully cleared each episode.
"""
from __future__ import annotations

import torch


class _ScratchStateMixin:
    """Manages runtime buffers, mine state, and per-episode domain randomisation."""

    def _alloc_runtime_buffers(
        self,
        B: int,
        Nb: int,
        Nr: int,
        dev: torch.device,
        f32: torch.dtype,
    ) -> None:
        self.rt_current_strength_cps = torch.full(
            (B,), float(self.cfg.current_strength_cps), dtype=f32, device=dev
        )
        self.rt_drift_sigma_cells = torch.full(
            (B,), float(self.cfg.drift_sigma_cells), dtype=f32, device=dev
        )
        self.rt_sensor_noise_sigma_cells = torch.full(
            (B,), float(self.cfg.sensor_noise_sigma_cells), dtype=f32, device=dev
        )
        self.rt_sensor_dropout_prob = torch.full(
            (B,), float(self.cfg.sensor_dropout_prob), dtype=f32, device=dev
        )
        self.rt_blue_speed_scale = torch.ones((B,), dtype=f32, device=dev)
        self._last_dense_progress = torch.zeros((B,), dtype=f32, device=dev)
        # Scripted-opponent behaviour knobs (batched).
        self.red_deception_prob = torch.zeros((B,), dtype=f32, device=dev)
        self.red_speed_mult = torch.ones((B,), dtype=f32, device=dev)
        self.red_attacker_style = torch.zeros((B,), dtype=torch.int32, device=dev)
        self.red_defender_style = torch.zeros((B,), dtype=torch.int32, device=dev)
        self.red_role_switch_prob = torch.zeros((B,), dtype=f32, device=dev)
        # OP5-style coordinated rush: shared aim + short commitment window.
        self.red_coordinated_attack = torch.zeros((B,), dtype=torch.bool, device=dev)
        self.red_attack_sync_window = torch.zeros((B,), dtype=torch.int32, device=dev)
        self.red_coord_ticks_left = torch.zeros((B,), dtype=torch.int32, device=dev)
        self.red_coord_aim_x = torch.zeros((B,), dtype=f32, device=dev)
        self.red_coord_aim_y = torch.zeros((B,), dtype=f32, device=dev)
        # Per-episode scripted-policy randomisation so agents cannot overfit to
        # one fixed NPC pattern.
        self.red_script_role_flip = torch.zeros((B,), dtype=torch.bool, device=dev)
        self.red_script_lane_sign = torch.ones((B,), dtype=f32, device=dev)
        self.red_script_guard_x = torch.zeros((B,), dtype=f32, device=dev)
        self.red_script_guard_y = torch.zeros((B,), dtype=f32, device=dev)
        # Capture confirmation counters (batched per-agent).
        self.blue_home_contact_frames = torch.zeros((B, Nb), dtype=torch.int32, device=dev)
        self.red_home_contact_frames = torch.zeros((B, Nr), dtype=torch.int32, device=dev)
        # Tagging channel: per-agent timers accumulating time under 2+ defender pressure.
        self.red_tag_pressure_time = torch.zeros((B, Nr), dtype=f32, device=dev)
        self.blue_tag_pressure_time = torch.zeros((B, Nb), dtype=f32, device=dev)

    def _alloc_mine_state(
        self,
        B: int,
        Nb: int,
        Nr: int,
        dev: torch.device,
        f32: torch.dtype,
    ) -> None:
        Nm = int(self.cfg.max_mines_per_team)
        self.Nm = Nm
        self.blue_mine_x = torch.zeros((B, Nm), dtype=f32, device=dev)
        self.blue_mine_y = torch.zeros((B, Nm), dtype=f32, device=dev)
        self.blue_mine_active = torch.zeros((B, Nm), dtype=torch.bool, device=dev)
        self.red_mine_x = torch.zeros((B, Nm), dtype=f32, device=dev)
        self.red_mine_y = torch.zeros((B, Nm), dtype=f32, device=dev)
        self.red_mine_active = torch.zeros((B, Nm), dtype=torch.bool, device=dev)
        self.blue_mine_charges = torch.zeros((B, Nb), dtype=torch.int32, device=dev)
        self.red_mine_charges = torch.zeros((B, Nr), dtype=torch.int32, device=dev)
        Np = int(getattr(self.cfg, "n_mine_pickups", 4))
        self.Np = Np
        self.pickup_x = torch.zeros((B, Np), dtype=f32, device=dev)
        self.pickup_y = torch.zeros((B, Np), dtype=f32, device=dev)
        self.pickup_active = torch.ones((B, Np), dtype=torch.bool, device=dev)
        self.pickup_respawn = torch.zeros((B, Np), dtype=torch.int32, device=dev)
        self._init_pickup_positions()

    def _alloc_bt_state(
        self,
        B: int,
        Nr: int,
        dev: torch.device,
        f32: torch.dtype,
    ) -> None:
        """Allocate behavior-tree telemetry and role-lock buffers for red side."""
        # Max agent slots: Nr (red) or Nb (blue); use max(Nr, Nb) defensively.
        N_max = max(Nr, getattr(self, "Nb", Nr))
        self._alloc_bt_telemetry(B, N_max, dev)

    def _init_pickup_positions(self) -> None:
        """Set fixed spawn positions for mine pickups (2 per side on 20×20)."""
        B, Np = self.B, self.Np
        c, r = self.cols, self.rows
        positions = [
            (min(3.0, float(c - 1)), min(5.0, float(r - 1))),
            (min(3.0, float(c - 1)), min(14.0, float(r - 1))),
            (max(0.0, float(c - 1) - 3.0), min(5.0, float(r - 1))),
            (max(0.0, float(c - 1) - 3.0), min(14.0, float(r - 1))),
        ]
        for k in range(min(Np, len(positions))):
            self.pickup_x[:, k] = positions[k][0]
            self.pickup_y[:, k] = positions[k][1]

    def _apply_train_domain_randomization(self, env_mask: torch.Tensor) -> None:
        """Resample per-episode sim/observation jitter for masked env rows."""
        idx = torch.where(env_mask)[0]
        if idx.numel() == 0:
            return
        if bool(getattr(self.cfg, "train_domain_randomization", False)):
            hi_n = max(0.0, float(getattr(self.cfg, "dr_sensor_noise_sigma_max", 0.0)))
            hi_d = max(0.0, min(1.0, float(getattr(self.cfg, "dr_sensor_dropout_max", 0.0))))
            jit = max(0.0, min(0.75, float(getattr(self.cfg, "dr_blue_speed_jitter", 0.0))))
            self.rt_sensor_noise_sigma_cells[idx] = self._rand_uniform((idx.numel(),), 0.0, hi_n)
            self.rt_sensor_dropout_prob[idx] = self._rand_uniform((idx.numel(),), 0.0, hi_d)
            lo_s = max(0.5, 1.0 - jit)
            hi_s = 1.0
            self.rt_blue_speed_scale[idx] = self._rand_uniform((idx.numel(),), lo_s, hi_s)
            return
        self.rt_sensor_noise_sigma_cells[idx] = float(self.cfg.sensor_noise_sigma_cells)
        self.rt_sensor_dropout_prob[idx] = float(self.cfg.sensor_dropout_prob)
        self.rt_blue_speed_scale[idx] = 1.0
