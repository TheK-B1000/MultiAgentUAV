"""State allocation, reset, and configuration helpers for BatchedCTFCore."""
from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from .._config import GPUFieldConfig
from .._constants import MAP_SET_SEED_OFFSETS, METRIC_ZONE_COLS, METRIC_ZONE_ROWS
from .._paths import _resolve_snapshot_path
from .._specs import _make_obs_action_spaces


class _StateMixin:
    def __init__(self, cfg: GPUFieldConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.B = int(cfg.n_envs)
        self.Nb = int(cfg.max_blue_agents)
        self.Nr = int(cfg.max_red_agents)
        self.rows = int(cfg.map_rows)
        self.cols = int(cfg.map_cols)
        self.max_steps = int(cfg.max_decision_steps)
        self.max_sim_steps = int(cfg.max_decision_steps) * max(
            1,
            int(
                max(
                    cfg.macro_commit_go_to_ticks,
                    cfg.macro_commit_grab_ticks,
                    cfg.macro_commit_get_flag_ticks,
                    cfg.macro_commit_place_ticks,
                    cfg.macro_commit_go_home_ticks,
                )
            ),
        )
        self.score_limit = int(cfg.score_limit)
        self.dt = float(cfg.decision_interval_seconds) * 0.99
        self.max_dist = math.sqrt(float(self.cols * self.cols + self.rows * self.rows))
        self.map_set = str(cfg.map_set).lower()
        self._map_seed_offset = int(MAP_SET_SEED_OFFSETS[self.map_set])

        self._rng = torch.Generator(device=self.device)
        self._rng.manual_seed(int(cfg.seed) + self._map_seed_offset)

        self._phase: List[str] = ["OP3"] * self.B
        self._league_mode = torch.zeros((self.B,), dtype=torch.bool, device=self.device)
        self._stress_schedule: Optional[dict] = None
        self._opponent_kind: List[str] = ["SCRIPTED"] * self.B
        self._opponent_key: List[str] = ["OP3"] * self.B
        self._phase_tensor_cache: Dict[Tuple[str, ...], torch.Tensor] = {}
        self._red_control_mask: Optional[torch.Tensor] = None
        self._red_control_mask_dirty = True
        self._snapshot_policy_cache: Dict[str, Tuple[float, Optional[Any]]] = {}
        self.rules_profile = str(cfg.rules_profile).upper()

        self.blue_scripted = False

        self._build_macro_targets()
        self._alloc_state()
        self.reset_all()

    def reseed(self, seed: int) -> None:
        self.cfg.seed = int(seed)
        self._rng.manual_seed(int(seed) + self._map_seed_offset)

    def _init_pickup_positions(self) -> None:
        """Set fixed spawn positions for mine pickups (2 per side on 20x20)."""
        B, Np = self.B, self.Np
        dev = self.device
        c, r = self.cols, self.rows
        # Fixed positions: blue side (x~3), red side (x~cols-4)
        positions = [
            (min(3.0, float(c - 1)), min(5.0, float(r - 1))),
            (min(3.0, float(c - 1)), min(14.0, float(r - 1))),
            (max(0.0, float(c - 1) - 3.0), min(5.0, float(r - 1))),
            (max(0.0, float(c - 1) - 3.0), min(14.0, float(r - 1))),
        ]
        for k in range(min(Np, len(positions))):
            self.pickup_x[:, k] = positions[k][0]
            self.pickup_y[:, k] = positions[k][1]

    def _alloc_state(self) -> None:
        B, Nb, Nr = self.B, self.Nb, self.Nr
        dev = self.device
        f32 = torch.float32

        self._alloc_episode_state(B, Nb, Nr, dev)
        self._alloc_agent_state(B, Nb, Nr, dev, f32)
        self._alloc_flags_and_scores(B, dev, f32)
        self._alloc_runtime_buffers(B, Nb, Nr, dev, f32)
        self._alloc_mine_state(B, Nb, Nr, dev, f32)
        self._alloc_metric_buffers(B, dev, f32)

    def _alloc_episode_state(self, B: int, Nb: int, Nr: int, dev: torch.device) -> None:
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

    def _alloc_agent_state(self, B: int, Nb: int, Nr: int, dev: torch.device, f32: torch.dtype) -> None:
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

    def _alloc_flags_and_scores(self, B: int, dev: torch.device, f32: torch.dtype) -> None:
        self.blue_score = torch.zeros((B,), dtype=torch.int32, device=dev)
        self.red_score = torch.zeros((B,), dtype=torch.int32, device=dev)

        # Flags at vertical center; a little inward from edges (2 cells) toward middle
        home_y = float(self.rows // 2)
        inward = 2.0
        blue_x = min(inward, float(max(0, self.cols - 1)))
        red_x = max(float(self.cols - 1) - inward, 0.0)
        self.blue_flag_home = torch.stack(
            [
                torch.full((B,), blue_x, dtype=f32, device=dev),
                torch.full((B,), home_y, dtype=f32, device=dev),
            ],
            dim=1,
        )
        self.red_flag_home = torch.stack(
            [
                torch.full((B,), red_x, dtype=f32, device=dev),
                torch.full((B,), home_y, dtype=f32, device=dev),
            ],
            dim=1,
        )
        self.blue_flag_pos = self.blue_flag_home.clone()
        self.red_flag_pos = self.red_flag_home.clone()

    def _alloc_runtime_buffers(self, B: int, Nb: int, Nr: int, dev: torch.device, f32: torch.dtype) -> None:
        self.rt_current_strength_cps = torch.full((B,), float(self.cfg.current_strength_cps), dtype=f32, device=dev)
        self.rt_drift_sigma_cells = torch.full((B,), float(self.cfg.drift_sigma_cells), dtype=f32, device=dev)
        self.rt_sensor_noise_sigma_cells = torch.full((B,), float(self.cfg.sensor_noise_sigma_cells), dtype=f32, device=dev)
        self.rt_sensor_dropout_prob = torch.full((B,), float(self.cfg.sensor_dropout_prob), dtype=f32, device=dev)
        self.rt_blue_speed_scale = torch.ones((B,), dtype=f32, device=dev)
        self._last_dense_progress = torch.zeros((B,), dtype=f32, device=dev)
        # Scripted-opponent behavior knobs (batched).
        self.red_deception_prob = torch.zeros((B,), dtype=f32, device=dev)
        self.red_speed_mult = torch.ones((B,), dtype=f32, device=dev)
        self.red_attacker_style = torch.zeros((B,), dtype=torch.int32, device=dev)  # 0 easy, 1 medium
        self.red_defender_style = torch.zeros((B,), dtype=torch.int32, device=dev)  # 0 easy, 1 medium
        self.red_role_switch_prob = torch.zeros((B,), dtype=f32, device=dev)
        # Per-episode scripted-policy randomization so agents cannot overfit to one fixed NPC pattern.
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

    def _alloc_mine_state(self, B: int, Nb: int, Nr: int, dev: torch.device, f32: torch.dtype) -> None:
        # Mines: each team has max_mines_per_team slots. Agents get charges by GRAB_MINE at pickups, place with PLACE_MINE.
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
        # Mine pickups: spawn points; agents pick up with GRAB_MINE (blue) or by moving near (red scripted).
        Np = int(getattr(self.cfg, "n_mine_pickups", 4))
        self.Np = Np
        self.pickup_x = torch.zeros((B, Np), dtype=f32, device=dev)
        self.pickup_y = torch.zeros((B, Np), dtype=f32, device=dev)
        self.pickup_active = torch.ones((B, Np), dtype=torch.bool, device=dev)
        self.pickup_respawn = torch.zeros((B, Np), dtype=torch.int32, device=dev)
        self._init_pickup_positions()

    def _alloc_metric_buffers(self, B: int, dev: torch.device, f32: torch.dtype) -> None:
        # Per-episode telemetry. These are reset with each env instance and
        # summarized into info["episode_result"] at terminal time.
        self.metric_time_to_first_score = torch.full((B,), -1.0, dtype=f32, device=dev)
        self.metric_inter_robot_dist_sum = torch.zeros((B,), dtype=f32, device=dev)
        self.metric_inter_robot_dist_count = torch.zeros((B,), dtype=torch.int32, device=dev)
        self.metric_collision_events = torch.zeros((B,), dtype=torch.int32, device=dev)
        self.metric_near_misses = torch.zeros((B,), dtype=torch.int32, device=dev)
        self.metric_blue_zone_visited = torch.zeros(
            (B, METRIC_ZONE_ROWS * METRIC_ZONE_COLS),
            dtype=torch.bool,
            device=dev,
        )

    def _build_macro_targets(self) -> None:
        """
        Build a fixed set of 2D macro targets for GoTo/PlaceMine.

        We follow the paper's successful variant: a categorical distribution over
        ~50 predetermined locations spread across the map. Targets are laid out
        on a coarse grid covering both halves of the field, so agents can learn
        richer paths and mine placements while keeping the action space discrete.
        """
        # Use a 5 x 10 grid over the full map (cols x rows) → 50 targets.
        num_x = 5
        num_y = 10
        max_x = float(max(0, self.cols - 1))
        max_y = float(max(0, self.rows - 1))

        xs = []
        ys = []
        for ix in range(num_x):
            # Evenly spaced from 0 to max_x
            x = max_x * (ix / float(max(1, num_x - 1)))
            for iy in range(num_y):
                # Evenly spaced from 0 to max_y
                y = max_y * (iy / float(max(1, num_y - 1)))
                xs.append(x)
                ys.append(y)

        targets = torch.stack(
            [
                torch.tensor(xs, dtype=torch.float32, device=self.device),
                torch.tensor(ys, dtype=torch.float32, device=self.device),
            ],
            dim=1,
        )
        # Reserve the first target slots for mine pickup coordinates so PPO can
        # intentionally route to those action-relevant points.
        pickup_positions = [
            (min(3.0, max_x), min(5.0, max_y)),
            (min(3.0, max_x), min(14.0, max_y)),
            (max(0.0, max_x - 3.0), min(5.0, max_y)),
            (max(0.0, max_x - 3.0), min(14.0, max_y)),
        ]
        n_pickup_targets = min(int(getattr(self.cfg, "n_mine_pickups", 4)), len(pickup_positions), int(targets.shape[0]))
        for k in range(n_pickup_targets):
            targets[k, 0] = pickup_positions[k][0]
            targets[k, 1] = pickup_positions[k][1]
        self._macro_targets = targets

    def _load_snapshot_policy(self, snapshot_key: str) -> Optional[Any]:
        resolved = _resolve_snapshot_path(snapshot_key)
        if resolved is None:
            return None
        try:
            mtime = float(os.path.getmtime(resolved))
        except OSError:
            return None
        cached = self._snapshot_policy_cache.get(resolved)
        if cached is not None:
            cached_mtime, cached_model = cached
            if abs(cached_mtime - mtime) < 1e-9:
                return cached_model
        try:
            from rl.custom_ppo import load_custom_ppo_policy

            obs_space, action_space = _make_obs_action_spaces(self.Nr, self.cfg.n_macros, self.cfg.n_targets)
            model = load_custom_ppo_policy(resolved, obs_space, action_space, device=self.device)
        except Exception:
            model = None
        self._snapshot_policy_cache[resolved] = (mtime, model)
        return model

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
        red_is_op4 = torch.as_tensor(
            [
                str(self._opponent_kind[i]).upper() == "SCRIPTED" and str(self._opponent_key[i]).upper() == "OP4"
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
            self.red_script_guard_x[op4_idx] = self._rand_uniform((op4_idx.numel(),), guard_x_low, guard_x_high)
            self.red_script_guard_y[op4_idx] = self._rand_uniform((op4_idx.numel(),), 3.5, 16.0)
        non_op4_idx = idx[~red_is_op4]
        if non_op4_idx.numel() > 0:
            self.red_script_guard_x[non_op4_idx] = self._rand_uniform((non_op4_idx.numel(),), 14.5, 17.5)
            self.red_script_guard_y[non_op4_idx] = self._rand_uniform((non_op4_idx.numel(),), 7.0, 13.0)
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
        # Mines: fully clear placed mines and charges so a new episode starts with a clean field.
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

        # Reset tagging channel timers.
        self.red_tag_pressure_time[idx] = 0.0
        self.blue_tag_pressure_time[idx] = 0.0
        self.metric_time_to_first_score[idx] = -1.0
        self.metric_inter_robot_dist_sum[idx] = 0.0
        self.metric_inter_robot_dist_count[idx] = 0
        self.metric_collision_events[idx] = 0
        self.metric_near_misses[idx] = 0
        self.metric_blue_zone_visited[idx] = False
        self._apply_opponent_params_for_mask(env_mask)
        self._respawn_side(blue=True, env_mask=env_mask)
        self._respawn_side(blue=False, env_mask=env_mask)
        self._apply_train_domain_randomization(env_mask)

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
        # Eval / default: follow static cfg fields (and disable speed jitter).
        self.rt_sensor_noise_sigma_cells[idx] = float(self.cfg.sensor_noise_sigma_cells)
        self.rt_sensor_dropout_prob[idx] = float(self.cfg.sensor_dropout_prob)
        self.rt_blue_speed_scale[idx] = 1.0

    # env_method-compatible setters
    def _normalize_env_indices(self, env_indices: Optional[Sequence[int]] = None) -> torch.Tensor:
        if env_indices is None:
            return torch.arange(self.B, device=self.device, dtype=torch.int64)
        if isinstance(env_indices, torch.Tensor):
            idx = env_indices.to(device=self.device, dtype=torch.int64).reshape(-1)
        else:
            idx = torch.as_tensor(list(env_indices), device=self.device, dtype=torch.int64).reshape(-1)
        if idx.numel() == 0:
            return idx
        return torch.clamp(idx, 0, max(0, self.B - 1))

    def _phase_tensor_equals(self, phases: Sequence[str]) -> torch.Tensor:
        key = tuple(sorted(str(p).upper() for p in phases))
        cached = self._phase_tensor_cache.get(key)
        if cached is None:
            phase_set = set(key)
            cached = torch.as_tensor(
                [str(p).upper() in phase_set for p in self._phase],
                device=self.device,
                dtype=torch.bool,
            )
            self._phase_tensor_cache[key] = cached
        return cached

    def _get_red_control_mask(self) -> torch.Tensor:
        if self._red_control_mask_dirty or self._red_control_mask is None:
            self._red_control_mask = torch.as_tensor(
                [
                    str(self._opponent_kind[i]).upper() == "SNAPSHOT"
                    and _resolve_snapshot_path(self._opponent_key[i]) is not None
                    for i in range(self.B)
                ],
                device=self.device,
                dtype=torch.bool,
            )
            self._red_control_mask_dirty = False
        return self._red_control_mask

    def set_phase(self, phase: str, env_indices: Optional[Sequence[int]] = None) -> None:
        phase_s = str(phase).upper()
        for env_i in self._normalize_env_indices(env_indices).detach().cpu().tolist():
            self._phase[env_i] = phase_s
        self._phase_tensor_cache.clear()
        self._red_control_mask_dirty = True

    def set_league_mode(self, league_mode: bool, env_indices: Optional[Sequence[int]] = None) -> None:
        idx = self._normalize_env_indices(env_indices)
        if idx.numel() > 0:
            self._league_mode[idx] = bool(league_mode)

    def set_stress_schedule(self, schedule: Optional[dict], env_indices: Optional[Sequence[int]] = None) -> None:
        self._stress_schedule = schedule

    def set_next_opponent(self, kind: str, key: str, env_indices: Optional[Sequence[int]] = None) -> None:
        kind_s = str(kind).upper()
        key_s = str(key) if kind_s == "SNAPSHOT" else str(key).upper()
        idx = self._normalize_env_indices(env_indices)
        for env_i in idx.detach().cpu().tolist():
            self._opponent_kind[env_i] = kind_s
            self._opponent_key[env_i] = key_s
        self._red_control_mask_dirty = True
        try:
            mask = torch.zeros((self.B,), dtype=torch.bool, device=self.device)
            if idx.numel() > 0:
                mask[idx] = True
            self._apply_opponent_params_for_mask(mask)
        except Exception as e:
            import warnings
            warnings.warn(
                f"BatchedCTFCore: set_next_opponent({key_s!r}) failed to apply params: {e}. "
                "Red team may still use previous opponent params; targeted opponent changes may lag."
            )

    def get_opponent_key(self, env_indices: Optional[Sequence[int]] = None) -> str:
        """Return current red opponent key (OP1/OP2/OP3/OP4). For eval verification."""
        idx = self._normalize_env_indices(env_indices)
        if idx.numel() == 0:
            return "OP3"
        return str(self._opponent_key[int(idx[0].item())])

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
        self._apply_dynamics_tensor(cfg, "deception_prob", "red_deception_prob", 0.0, 1.0)
        self._apply_dynamics_tensor(cfg, "speed_mult", "red_speed_mult", 0.25, 2.0)
        self._apply_dynamics_tensor(cfg, "attacker_style", "red_attacker_style", 0, 1, torch.int32)
        self._apply_dynamics_tensor(cfg, "defender_style", "red_defender_style", 0, 1, torch.int32)
        self._apply_dynamics_tensor(cfg, "role_switch_prob", "red_role_switch_prob", 0.0, 1.0)

    def _rand_uniform(self, shape: Sequence[int], lo: float, hi: float) -> torch.Tensor:
        t = torch.rand(tuple(shape), generator=self._rng, device=self.device)
        return lo + (hi - lo) * t

    def _randn(self, shape: Sequence[int]) -> torch.Tensor:
        return torch.randn(tuple(shape), generator=self._rng, device=self.device)

    def _mirror_x(self, x: torch.Tensor, side: str) -> torch.Tensor:
        if side == "red":
            return float(max(0, self.cols - 1)) - x
        return x

    def _mirror_heading(self, heading: torch.Tensor, side: str) -> torch.Tensor:
        if side == "red":
            mirrored = math.pi - heading
            return (mirrored + math.pi) % (2.0 * math.pi) - math.pi
        return heading

    def _side_tensors(self, side: str) -> Dict[str, torch.Tensor]:
        if side == "red":
            return {
                "own_x": self.red_x,
                "own_y": self.red_y,
                "own_heading": self.red_heading,
                "own_speed": self.red_speed,
                "own_alive": self.red_alive,
                "own_carrying": self.red_carrying,
                "own_flag": self.red_flag_pos,
                "own_flag_home": self.red_flag_home,
                "own_mine_x": self.red_mine_x,
                "own_mine_y": self.red_mine_y,
                "own_mine_active": self.red_mine_active,
                "own_mine_charges": self.red_mine_charges,
                "enemy_x": self.blue_x,
                "enemy_y": self.blue_y,
                "enemy_alive": self.blue_alive,
                "enemy_flag": self.blue_flag_pos,
                "n_agents": torch.tensor(self.Nr, device=self.device),
            }
        return {
            "own_x": self.blue_x,
            "own_y": self.blue_y,
            "own_heading": self.blue_heading,
            "own_speed": self.blue_speed,
            "own_alive": self.blue_alive,
            "own_carrying": self.blue_carrying,
            "own_flag": self.blue_flag_pos,
            "own_flag_home": self.blue_flag_home,
            "own_mine_x": self.blue_mine_x,
            "own_mine_y": self.blue_mine_y,
            "own_mine_active": self.blue_mine_active,
            "own_mine_charges": self.blue_mine_charges,
            "enemy_x": self.red_x,
            "enemy_y": self.red_y,
            "enemy_alive": self.red_alive,
            "enemy_flag": self.red_flag_pos,
            "n_agents": torch.tensor(self.Nb, device=self.device),
        }

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

