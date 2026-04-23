"""
GPU-vectorized CTF environment used by PPO/MAPPO/QMIX training and the viewer.

Training (rl/train_ppo.py) uses BatchedCTFCore via
GPUCTFVecEnv. Scoring and sparse reward values are imported from game_manager so
both paths stay aligned (get_grab_score_delta, get_capture_score_delta, sparse event points).
GameManager (game_manager.py) remains the single source of truth for those constants.
"""
from __future__ import annotations

import inspect
import os
import sys
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv

try:
    from opponent_params import sample_batched_opponent_params
except ImportError:
    sample_batched_opponent_params = None

from macro_actions import MacroAction
from agents import AgentHandle
from rl.global_state import build_global_state_batch

# Scoring and sparse reward values: single source of truth from game_manager.
from game_manager import (
    get_grab_score_delta,
    get_capture_score_delta,
    SPARSE_TAG_NO_FLAG_POINTS,
    SPARSE_TAG_WITH_FLAG_POINTS,
    SPARSE_FLAG_GRAB_POINTS,
    SPARSE_FLAG_CAPTURE_POINTS,
    SPARSE_OOB_POINTS,
    SPARSE_MINE_TAG_POINTS,
    DEFAULT_SCORE_LIMIT,
    WIN_TEAM_REWARD,
    LOSE_TEAM_PUNISH,
    DRAW_TEAM_PENALTY,
    FLAG_PICKUP_REWARD,
    FLAG_CARRY_HOME_REWARD,
    ENEMY_MAV_KILL_REWARD,
    ACTION_FAILED_PUNISHMENT,
)

CNN_COLS = 20
CNN_ROWS = 20
NUM_CNN_CHANNELS = 7
GLOBAL_STATE_CHANNELS = 8
VEC_OBS_DIM = 18


def _ensure_numpy_core_compat() -> None:
    """Register numpy._core.* aliases so SB3 models load across NumPy versions."""
    try:
        import types
        if not hasattr(np, "_core"):
            _core = types.ModuleType("numpy._core")
            _core.__path__ = []
            sys.modules["numpy._core"] = _core
            for _name in ("numeric", "multiarray", "umath"):
                try:
                    _sub = __import__(f"numpy.core.{_name}", fromlist=[_name])
                    setattr(_core, _name, _sub)
                    sys.modules[f"numpy._core.{_name}"] = _sub
                except Exception:
                    pass
        if "numpy._core.numeric" not in sys.modules:
            try:
                _sub = __import__("numpy.core.numeric", fromlist=["numeric"])
                sys.modules["numpy._core.numeric"] = _sub
                if hasattr(np, "_core") and not hasattr(np._core, "numeric"):
                    setattr(np._core, "numeric", _sub)
            except Exception:
                pass
    except Exception:
        pass


def _ensure_numpy_random_compat() -> None:
    """Register NumPy random pickle aliases used by older/newer SB3 checkpoints."""
    try:
        try:
            import numpy.random._pcg64 as _pcg64
            sys.modules["numpy.random._pcg64"] = _pcg64
        except Exception:
            import numpy.random
            if hasattr(numpy.random, "_pcg64"):
                sys.modules["numpy.random._pcg64"] = getattr(numpy.random, "_pcg64")

        import numpy.random._pickle as _np_pickle
        from numpy.random.bit_generator import BitGenerator
        _orig_ctor = getattr(_np_pickle, "__bit_generator_ctor", None)
        if _orig_ctor is not None:
            def _bit_generator_ctor_patched(bit_generator):
                if isinstance(bit_generator, type) and issubclass(bit_generator, BitGenerator):
                    return bit_generator()
                return _orig_ctor(bit_generator)
            _np_pickle.__bit_generator_ctor = _bit_generator_ctor_patched
    except Exception:
        pass


def _try_paths(*candidates: str) -> Optional[str]:
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None


def _resolve_snapshot_path(path: str) -> Optional[str]:
    if not path:
        return None
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [path]
    if not path.endswith(".zip"):
        candidates.append(path + ".zip")
    if not os.path.isabs(path):
        local = os.path.join(script_dir, path)
        cwd_local = os.path.join(os.getcwd(), path)
        candidates.extend([local, cwd_local])
        if not local.endswith(".zip"):
            candidates.append(local + ".zip")
        if not cwd_local.endswith(".zip"):
            candidates.append(cwd_local + ".zip")
    return _try_paths(*candidates)


def _make_obs_action_spaces(n_agents: int, n_macros: int, n_targets: int):
    obs_space = spaces.Dict(
        {
            "grid": spaces.Box(
                low=0.0,
                high=1.0,
                shape=(n_agents, NUM_CNN_CHANNELS, CNN_ROWS, CNN_COLS),
                dtype=np.float32,
            ),
            "vec": spaces.Box(low=-1.0, high=1.0, shape=(n_agents, VEC_OBS_DIM), dtype=np.float32),
            "agent_mask": spaces.Box(low=0.0, high=1.0, shape=(n_agents,), dtype=np.float32),
            "mask": spaces.Box(
                low=0.0,
                high=1.0,
                shape=(n_agents * (n_macros + n_targets),),
                dtype=np.float32,
            ),
        }
    )
    action_space = spaces.MultiDiscrete([n_macros, n_targets] * n_agents)
    return obs_space, action_space


@dataclass
class GPUFieldConfig:
    n_envs: int = 64
    # Aquaticus standard setup is 2v2.
    max_blue_agents: int = 2
    max_red_agents: int = 2
    map_rows: int = 20
    map_cols: int = 20
    # Paper-aligned timing (Table 3):
    #   - maxGameTime ≈ 200 s
    #   - Δt_sim ≈ 0.5 s
    #   => 400 decision steps per episode when using dt = decision_interval_seconds.
    max_decision_steps: int = 400
    decision_interval_seconds: float = 0.5

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
    # Local tag radius (in map cells) used by Aquaticus-style tagging.
    tag_range_cells: float = 2.5
    home_untag_radius_cells: float = 2.0
    avoid_collision_radius_cells: float = 0.75
    opregion_safe_speed_cps: float = 0.8

    n_macros: int = 5
    # Number of discrete 2D macro targets for GoTo/PlaceMine.
    # Paper-aligned default: 50 fixed positions sampled over the map.
    n_targets: int = 50
    score_limit: int = DEFAULT_SCORE_LIMIT

    # Mines: pickups spawn; agents must GRAB_MINE to get a charge, then PLACE_MINE to place anywhere.
    # For realism, each team can have at most 2 active mines on the field, and there are 4 pickups total
    # (2 on the blue side, 2 on the red side). Pickups are single-use with no respawn.
    max_mines_per_team: int = 2
    max_mine_charges_per_agent: int = 2
    mine_trigger_radius_cells: float = 1.5
    mine_pickup_radius_cells: float = 1.2
    mine_pickup_respawn_steps: int = 0   # 0 => no respawn; pickups are single-use
    n_mine_pickups: int = 4
    enabled_mine_reward: float = 0.2
    flag_pickup_reward: float = FLAG_PICKUP_REWARD
    flag_carry_home_reward: float = FLAG_CARRY_HOME_REWARD
    enemy_mav_kill_reward: float = ENEMY_MAV_KILL_REWARD
    action_failed_punishment: float = ACTION_FAILED_PUNISHMENT
    win_team_reward: float = WIN_TEAM_REWARD
    lose_team_punish: float = LOSE_TEAM_PUNISH
    draw_team_penalty: float = DRAW_TEAM_PENALTY
    pbrs_gamma: float = 0.995
    pbrs_attack_coef: float = 1.0
    pbrs_return_coef: float = 1.0
    pbrs_defense_coef: float = 0.5
    team_defense_presence_reward: float = 0.03
    team_escort_reward: float = 0.02
    team_intercept_reward: float = 0.02
    macro_commit_go_to_ticks: int = 4
    macro_commit_grab_ticks: int = 3
    macro_commit_get_flag_ticks: int = 4
    macro_commit_place_ticks: int = 2
    macro_commit_go_home_ticks: int = 4
    macro_arrival_radius_cells: float = 1.0

    # Tagging channel controls:
    # - tag_channel_seconds: pressure >= 2 must be sustained for this many seconds before a tag is applied.
    tag_channel_seconds: float = 1.0

    # Profile and reward controls
    aquaticus_profile: bool = False
    rules_profile: str = "OURS"
    sparse_weight: float = 1.0
    dense_weight: float = 0.2
    reward_scale: float = 2.0
    reward_clip: float = 1.0

    # Viewer-only convenience: when True, newly tagged agents are snapped back
    # to their home flag position instead of returning under their own motion.
    # Training configs should generally leave this as False.
    teleport_tagged_home: bool = False
    # Number of consecutive frames an agent must be inside home radius while
    # carrying to count as a capture (filters single-frame tunneling at speed).
    capture_confirm_frames: int = 2
    # No score for grab/capture in the first N steps (avoids spurious points at game start).
    score_grace_steps: int = 10

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

        self._rng = torch.Generator(device=self.device)
        self._rng.manual_seed(int(cfg.seed))

        self._phase: List[str] = ["OP1"] * self.B
        self._league_mode = torch.zeros((self.B,), dtype=torch.bool, device=self.device)
        self._stress_schedule: Optional[dict] = None
        self._opponent_kind: List[str] = ["SCRIPTED"] * self.B
        self._opponent_key: List[str] = ["OP1"] * self.B
        self._snapshot_policy_cache: Dict[str, Tuple[float, Optional[Any]]] = {}
        self.rules_profile = str(cfg.rules_profile).upper()

        self.blue_scripted = False

        self._build_macro_targets()
        self._alloc_state()
        self.reset_all()

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

        self.rt_current_strength_cps = torch.full((B,), float(self.cfg.current_strength_cps), dtype=f32, device=dev)
        self.rt_drift_sigma_cells = torch.full((B,), float(self.cfg.drift_sigma_cells), dtype=f32, device=dev)
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
            _ensure_numpy_core_compat()
            _ensure_numpy_random_compat()
            from stable_baselines3 import PPO as SB3PPO
            obs_space, action_space = _make_obs_action_spaces(self.Nr, int(self.cfg.n_macros), int(self.cfg.n_targets))
            custom_objects = {
                "observation_space": obs_space,
                "action_space": action_space,
                "clip_range": 0.2,
                "lr_schedule": lambda progress_remaining: 3e-4 * progress_remaining,
            }
            try:
                from rl.train_ppo import MaskedMultiInputPolicy
                custom_objects["policy_class"] = MaskedMultiInputPolicy
            except Exception:
                from stable_baselines3.common.policies import MultiInputActorCriticPolicy
                custom_objects["policy_class"] = MultiInputActorCriticPolicy
            model = SB3PPO.load(
                resolved,
                device=str(self.device),
                custom_objects=custom_objects,
            )
            model.policy.set_training_mode(False)
            self._snapshot_policy_cache[resolved] = (mtime, model)
            return model
        except Exception as exc:
            import warnings
            self._snapshot_policy_cache[resolved] = (mtime, None)
            warnings.warn(f"Failed to load snapshot opponent policy from {resolved}: {exc}")
            return None

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

    def reset_all(self) -> None:
        mask = torch.ones((self.B,), dtype=torch.bool, device=self.device)
        self.reset_indices(mask)

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
        return torch.as_tensor(
            [str(p).upper() in phases for p in self._phase],
            device=self.device,
            dtype=torch.bool,
        )

    def _apply_neutral_red_params(self, env_mask: torch.Tensor) -> None:
        idx = torch.where(env_mask)[0]
        if idx.numel() == 0:
            return
        self.red_deception_prob[idx] = 0.0
        self.red_speed_mult[idx] = 1.0
        self.red_attacker_style[idx] = 0
        self.red_defender_style[idx] = 0
        self.red_role_switch_prob[idx] = 0.0

    def _apply_opponent_params_for_mask(self, env_mask: torch.Tensor) -> None:
        if sample_batched_opponent_params is None:
            return
        # Only SCRIPTED with OP1/OP2/OP3/OP4 have defined params.
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
        grouped: Dict[Tuple[str, str], List[int]] = {}
        for env_i in idx.detach().cpu().tolist():
            use_kind = str(self._opponent_kind[env_i]).upper()
            use_key = str(self._opponent_key[env_i]).upper()
            if use_kind == "SNAPSHOT":
                continue
            if use_kind not in ("SCRIPTED",) or use_key not in ("OP1", "OP2", "OP3", "OP4"):
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

    def reset_indices(self, env_mask: torch.Tensor) -> None:
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
        self._apply_opponent_params_for_mask(env_mask)
        self._respawn_side(blue=True, env_mask=env_mask)
        self._respawn_side(blue=False, env_mask=env_mask)

    # env_method-compatible setters
    def set_phase(self, phase: str, env_indices: Optional[Sequence[int]] = None) -> None:
        phase_s = str(phase).upper()
        for env_i in self._normalize_env_indices(env_indices).detach().cpu().tolist():
            self._phase[env_i] = phase_s

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
            return "OP1"
        return str(self._opponent_key[int(idx[0].item())])

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

    def _decode_targets(self, target_idx: torch.Tensor, side: str = "blue") -> torch.Tensor:
        tidx = torch.remainder(target_idx, self.cfg.n_targets).long()
        n_agents = target_idx.shape[1]
        out = self._macro_targets.index_select(0, tidx.reshape(-1)).reshape(self.B, n_agents, 2)
        if side == "red":
            out = out.clone()
            out[..., 0] = self._mirror_x(out[..., 0], side)
        return out

    def _macro_commit_ticks(self, macro: torch.Tensor) -> torch.Tensor:
        ticks = torch.full_like(macro, int(self.cfg.macro_commit_go_to_ticks), dtype=torch.int32)
        ticks = torch.where(macro == int(MacroAction.GRAB_MINE), torch.full_like(ticks, int(self.cfg.macro_commit_grab_ticks)), ticks)
        ticks = torch.where(macro == int(MacroAction.GET_FLAG), torch.full_like(ticks, int(self.cfg.macro_commit_get_flag_ticks)), ticks)
        ticks = torch.where(macro == int(MacroAction.PLACE_MINE), torch.full_like(ticks, int(self.cfg.macro_commit_place_ticks)), ticks)
        ticks = torch.where(macro == int(MacroAction.GO_HOME), torch.full_like(ticks, int(self.cfg.macro_commit_go_home_ticks)), ticks)
        return torch.clamp(ticks, min=1)

    # ------------------------------------------------------------------
    # Carrier evasion: multi-threat tangent routing toward home
    # ------------------------------------------------------------------
    def _carrier_evasion_target(
        self,
        own_x: torch.Tensor,
        own_y: torch.Tensor,
        home_x: torch.Tensor,
        home_y: torch.Tensor,
        enemy_x: torch.Tensor,
        enemy_y: torch.Tensor,
        enemy_alive: torch.Tensor,
        carrying: torch.Tensor,
        side: str = "blue",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        For carriers, waypoint that routes *around* defenders using the same
        side-weighted tangent hook: tangent perpendicular to goal (home),
        exponential repulsion by distance. step_side=7 clears the 2.5 tag radius.
        """
        threat_radius = 8.0
        center_y = 10.0
        hx = home_x[:, None].expand_as(own_x)
        hy = home_y[:, None].expand_as(own_y)
        home_dx = hx - own_x
        home_dy = hy - own_y
        home_n = torch.sqrt(home_dx ** 2 + home_dy ** 2 + 1e-8)
        home_ux = home_dx / home_n
        home_uy = home_dy / home_n

        # Tangent perpendicular to GOAL (home) direction
        tan_x = -home_uy
        tan_y = home_ux

        # Pairwise distances [B, N_own, N_enemy]; nearest *alive* enemy per agent
        dxx = own_x[:, :, None] - enemy_x[:, None, :]
        dyy = own_y[:, :, None] - enemy_y[:, None, :]
        dd = torch.sqrt(dxx ** 2 + dyy ** 2 + 1e-8)
        big = torch.full_like(dd, 1e9)
        dd_masked = torch.where(enemy_alive[:, None, :], dd, big)
        nearest_dist = dd_masked.min(dim=2)[0]
        in_range = nearest_dist < threat_radius
        nearest_dist = nearest_dist.clamp(min=1e-6)

        # Same exponential repulsion as striker: ( (8 - d) / 8 )^2
        repulsion = torch.pow(torch.clamp(threat_radius - nearest_dist, min=0.0) / threat_radius, 2.0)
        repulsion = repulsion * in_range.float()

        # Side bias: above center -> go high, below -> go low
        side_bias = torch.where(own_y > center_y, 1.0, -1.0)

        step_fwd = 2.0
        step_side = 7.0
        evade_tx = own_x + home_ux * step_fwd + tan_x * side_bias * repulsion * step_side
        evade_ty = own_y + home_uy * step_fwd + tan_y * side_bias * repulsion * step_side
        evade_tx = torch.clamp(evade_tx, 0.0, float(max(0, self.cols - 1)))
        evade_ty = torch.clamp(evade_ty, 0.0, float(max(0, self.rows - 1)))

        has_threat = repulsion > 0.0
        should_evade = has_threat & carrying
        final_tx = torch.where(should_evade, evade_tx, hx)
        final_ty = torch.where(should_evade, evade_ty, hy)
        return final_tx, final_ty

    # ------------------------------------------------------------------
    # Unified NPC brain for both sides
    # ------------------------------------------------------------------
    def _get_scripted_targets(self, side: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Consolidated NPC brain usable for both "blue" and "red".

        Agent roles (parameterised by *side*); all N agents are assigned a target:
          Agent 0 – Defender / Guardian: patrols own half, intercepts intruders.
          Agent 1 – Striker: attacks enemy flag with lane preference and tangent evasion.
          Agents 2..N-1 – Strikers with lane spread (so 4v4/6v6/8v8 all have roles, no (0,0) targets).

        All carriers (any agent index) get multi-threat tangent evasion toward
        home via ``_carrier_evasion_target`` so they dodge enemies instead of
        running straight back.
        """
        is_blue = (side == "blue")
        B, device = self.B, self.device
        midline = float(self.cols) * 0.5
        idx_env = torch.arange(B, device=device)
        max_x = float(max(0, self.cols - 1))
        max_y = float(max(0, self.rows - 1))

        if is_blue:
            N, Ne = self.Nb, self.Nr
            own_x, own_y = self.blue_x, self.blue_y
            own_carrying = self.blue_carrying
            own_tagged = self.blue_tagged
            own_alive = self.blue_alive
            own_flag_home = self.blue_flag_home
            enemy_x, enemy_y = self.red_x, self.red_y
            enemy_carrying = self.red_carrying
            enemy_alive = self.red_alive
            enemy_flag_pos = self.red_flag_pos
            enemy_flag_home = self.red_flag_home
            atk_medium = torch.ones((B,), dtype=torch.bool, device=device)
            def_medium = torch.ones((B,), dtype=torch.bool, device=device)
            deception_prob = torch.zeros((B,), dtype=torch.float32, device=device)
            role_switch_prob = torch.zeros((B,), dtype=torch.float32, device=device)
        else:
            N, Ne = self.Nr, self.Nb
            own_x, own_y = self.red_x, self.red_y
            own_carrying = self.red_carrying
            own_tagged = self.red_tagged
            own_alive = self.red_alive
            own_flag_home = self.red_flag_home
            enemy_x, enemy_y = self.blue_x, self.blue_y
            enemy_carrying = self.blue_carrying
            enemy_alive = self.blue_alive
            enemy_flag_pos = self.blue_flag_pos
            enemy_flag_home = self.blue_flag_home
            atk_medium = self.red_attacker_style > 0
            def_medium = self.red_defender_style > 0
            deception_prob = self.red_deception_prob
            role_switch_prob = self.red_role_switch_prob

        target = torch.zeros((B, N, 2), dtype=torch.float32, device=device)
        guardian_idx = 0
        striker_idx = 1 if N > 1 else 0
        if (not is_blue) and N > 1:
            flip = self.red_script_role_flip
            guardian_idx_t = torch.where(
                flip,
                torch.ones((B,), dtype=torch.int64, device=device),
                torch.zeros((B,), dtype=torch.int64, device=device),
            )
            striker_idx_t = torch.where(
                flip,
                torch.zeros((B,), dtype=torch.int64, device=device),
                torch.ones((B,), dtype=torch.int64, device=device),
            )
        else:
            guardian_idx_t = torch.full((B,), guardian_idx, dtype=torch.int64, device=device)
            striker_idx_t = torch.full((B,), striker_idx, dtype=torch.int64, device=device)

        # Advanced indexing uses these as column indices; clamp defensively (avoids CUDA assert if corrupted).
        n_max = max(0, N - 1)
        guardian_idx_t = torch.clamp(guardian_idx_t, 0, n_max)
        striker_idx_t = torch.clamp(striker_idx_t, 0, n_max)

        # ---- shared team state ----
        enemy_carrier_exists = enemy_carrying.any(dim=1)
        if is_blue:
            enemy_on_own = enemy_alive & (enemy_x < midline)
        else:
            enemy_on_own = enemy_alive & (enemy_x > midline)
        any_intruder = enemy_on_own.any(dim=1)

        guardian_out = own_tagged[idx_env, guardian_idx_t] if N > 0 else torch.zeros((B,), dtype=torch.bool, device=device)
        role_coin = torch.rand((B,), generator=self._rng, device=device) < torch.clamp(role_switch_prob, 0.0, 1.0)
        striker_pivot = guardian_out & (enemy_carrier_exists | any_intruder) & role_coin

        # ======== Defender (agent 0) ========
        if guardian_idx < N:
            phase = self.sim_step_count.to(torch.float32) * 0.12
            orbit_r = 2.0
            easy_x = torch.clamp(own_flag_home[:, 0] + orbit_r * torch.cos(phase), 0.0, max_x)
            easy_y = torch.clamp(own_flag_home[:, 1] + orbit_r * torch.sin(phase), 0.0, max_y)
            # Defender loiter: opens midline. Red anchor is randomized per episode.
            if is_blue:
                med_x = torch.full((B,), min(max_x, 3.0), device=device)
                med_y = torch.full((B,), min(max_y, 10.0), device=device)
            else:
                med_x = torch.clamp(self.red_script_guard_x, 0.0, max_x)
                med_y = torch.clamp(self.red_script_guard_y, 0.0, max_y)
            gx = torch.where(def_medium, med_x, easy_x)
            gy = torch.where(def_medium, med_y, easy_y)

            if enemy_carrier_exists.any():
                ci = torch.argmax(enemy_carrying.to(torch.int64), dim=1)
                gx = torch.where(enemy_carrier_exists, enemy_x[idx_env, ci], gx)
                gy = torch.where(enemy_carrier_exists, enemy_y[idx_env, ci], gy)
            else:
                chase = def_medium & any_intruder
                if chase.any():
                    guard_x = own_x[idx_env, guardian_idx_t]
                    guard_y = own_y[idx_env, guardian_idx_t]
                    dxx = guard_x[:, None] - enemy_x
                    dyy = guard_y[:, None] - enemy_y
                    dd = torch.sqrt(dxx * dxx + dyy * dyy + 1e-8)
                    big = torch.full_like(dd, 1e9)
                    dd_masked = torch.where(enemy_on_own, dd, big)
                    nearest = torch.argmin(dd_masked, dim=1)
                    gx = torch.where(chase, enemy_x[idx_env, nearest], gx)
                    gy = torch.where(chase, enemy_y[idx_env, nearest], gy)

            target[idx_env, guardian_idx_t, 0] = gx
            target[idx_env, guardian_idx_t, 1] = gy

        # ======== Striker (agent 1): lane preference + side-weighted tangent hook ========
        center_y = 10.0
        lane_y_north = min(max_y, 15.0)
        lane_y_south = max(0.0, 5.0)
        if striker_idx < N:
            efx = enemy_flag_pos[:, 0]
            efy = enemy_flag_pos[:, 1]
            rx = own_x[idx_env, striker_idx_t]
            ry = own_y[idx_env, striker_idx_t]
            # Lane preference: randomized for red each episode instead of fixed south-only routing.
            dist_to_flag = torch.sqrt((rx - efx) ** 2 + (ry - efy) ** 2 + 1e-8)
            if is_blue:
                lane_y = torch.full((B,), lane_y_north, device=device)
            else:
                lane_mid = torch.full((B,), center_y, device=device)
                lane_amp = torch.full((B,), 5.0, device=device)
                lane_y = torch.clamp(lane_mid + self.red_script_lane_sign * lane_amp, 0.0, max_y)
            sy_easy = torch.where(dist_to_flag > 4.0, lane_y, efy)
            sx_easy = efx
            sx_med = sx_easy.clone()
            sy_med = sy_easy.clone()

            if atk_medium.any():
                dxx = rx[:, None] - enemy_x
                dyy = ry[:, None] - enemy_y
                dd = torch.sqrt(dxx * dxx + dyy * dyy + 1e-8)
                nearest = torch.argmin(dd, dim=1)
                nbx = enemy_x[idx_env, nearest]
                nby = enemy_y[idx_env, nearest]

                goal_dx = sx_easy - rx
                goal_dy = sy_easy - ry
                goal_n = torch.sqrt(goal_dx * goal_dx + goal_dy * goal_dy + 1e-8)
                goal_dx = goal_dx / goal_n
                goal_dy = goal_dy / goal_n

                # Tangent perpendicular to GOAL direction (not to enemy)
                tan_x = -goal_dy
                tan_y = goal_dx

                away_x = rx - nbx
                away_y = ry - nby
                dist_to_enemy = torch.sqrt(away_x * away_x + away_y * away_y + 1e-8)
                # Exponential repulsion: stronger as enemy gets closer (hook 6 when within 8)
                repulsion = torch.pow(torch.clamp(8.0 - dist_to_enemy, min=0.0) / 8.0, 2.0)
                # Side bias: above center -> go high, below -> go low (breaks symmetry)
                side_bias = torch.where(ry > center_y, 1.0, -1.0)

                tx_med = rx + (goal_dx * 2.0) + (tan_x * side_bias * repulsion * 6.0)
                ty_med = ry + (goal_dy * 2.0) + (tan_y * side_bias * repulsion * 6.0)
                sx_med = torch.clamp(tx_med, 0.0, max_x)
                sy_med = torch.clamp(ty_med, 0.0, max_y)

            sx = torch.where(atk_medium, sx_med, sx_easy)
            sy = torch.where(atk_medium, sy_med, sy_easy)

            # Dynamic pivot when guardian is tagged and threats exist
            if striker_pivot.any():
                threat_mask = enemy_on_own | enemy_carrying
                dxx = rx[:, None] - enemy_x
                dyy = ry[:, None] - enemy_y
                dd = torch.sqrt(dxx * dxx + dyy * dyy + 1e-8)
                big = torch.full_like(dd, 1e9)
                dd_masked = torch.where(threat_mask, dd, big)
                nearest = torch.argmin(dd_masked, dim=1)
                sx = torch.where(striker_pivot, enemy_x[idx_env, nearest], sx)
                sy = torch.where(striker_pivot, enemy_y[idx_env, nearest], sy)

            target[idx_env, striker_idx_t, 0] = sx
            target[idx_env, striker_idx_t, 1] = sy

        # ======== Extra agents (N > 2): assign striker-like targets with lane spread ========
        # So 4v4/6v6/8v8 all have roles; agents 2..N-1 go to enemy flag with y-offset to avoid clustering.
        efx = enemy_flag_pos[:, 0]
        efy = enemy_flag_pos[:, 1]
        for j in range(2, N):
            lane_offset = (j - 1) * (max_y * 0.25 / max(1, N - 1))  # spread across lanes
            if is_blue:
                lane_y_j = torch.clamp(efy + lane_offset, 0.0, max_y)
            else:
                lane_y_j = torch.clamp(efy - lane_offset, 0.0, max_y)
            target[:, j, 0] = efx
            target[:, j, 1] = lane_y_j

        # ======== Carrier evasion (replaces straight-home override) ========
        # Any agent carrying a flag uses multi-threat tangent routing so they
        # dodge nearby enemies instead of heading straight back.
        if own_carrying.any():
            evade_tx, evade_ty = self._carrier_evasion_target(
                own_x, own_y,
                own_flag_home[:, 0], own_flag_home[:, 1],
                enemy_x, enemy_y, enemy_alive,
                own_carrying,
                side=side,
            )
            target[..., 0] = torch.where(own_carrying, evade_tx, target[..., 0])
            target[..., 1] = torch.where(own_carrying, evade_ty, target[..., 1])

        # ======== Safety: non-carriers intercept enemy who stole own flag ========
        if enemy_carrier_exists.any():
            ci = torch.argmax(enemy_carrying.to(torch.int64), dim=1)
            cx = enemy_x[idx_env, ci]
            cy = enemy_y[idx_env, ci]
            not_carrying = ~own_carrying
            intercept = enemy_carrier_exists[:, None] & not_carrying
            target[..., 0] = torch.where(intercept, cx[:, None], target[..., 0])
            target[..., 1] = torch.where(intercept, cy[:, None], target[..., 1])

        # ======== Carrier shielding: escort 4 units perpendicular to carrier path ========
        shield_dist = 4.0
        own_carry_any = own_carrying.any(dim=1)
        if own_carry_any.any() and N > 1:
            carr_idx = torch.argmax(own_carrying.to(torch.int64), dim=1)
            carr_x = own_x[idx_env, carr_idx]
            carr_y = own_y[idx_env, carr_idx]
            home_ux = own_flag_home[:, 0] - carr_x
            home_uy = own_flag_home[:, 1] - carr_y
            home_n = torch.sqrt(home_ux ** 2 + home_uy ** 2 + 1e-8)
            home_ux = home_ux / home_n
            home_uy = home_uy / home_n
            perp1_x = -home_uy
            perp1_y = home_ux
            dxx = carr_x[:, None] - enemy_x
            dyy = carr_y[:, None] - enemy_y
            dd = torch.sqrt(dxx * dxx + dyy * dyy + 1e-8)
            near_enemy = torch.argmin(dd, dim=1)
            nex = enemy_x[idx_env, near_enemy]
            ney = enemy_y[idx_env, near_enemy]
            to_enemy_x = nex - carr_x
            to_enemy_y = ney - carr_y
            dot_perp1 = perp1_x * to_enemy_x + perp1_y * to_enemy_y
            use_perp1 = dot_perp1 >= 0.0
            escort_off_x = torch.where(use_perp1, perp1_x, -perp1_x)
            escort_off_y = torch.where(use_perp1, perp1_y, -perp1_y)
            escort_x = carr_x + escort_off_x * shield_dist
            escort_y = carr_y + escort_off_y * shield_dist
            escort_x = torch.clamp(escort_x, 0.0, max_x)
            escort_y = torch.clamp(escort_y, 0.0, max_y)
            for j in range(N):
                is_not_carrier = own_carry_any & (carr_idx != j) & (~own_carrying[:, j])
                escort_ok = is_not_carrier & (~enemy_carrier_exists)
                target[:, j, 0] = torch.where(escort_ok, escort_x, target[:, j, 0])
                target[:, j, 1] = torch.where(escort_ok, escort_y, target[:, j, 1])

        # ======== Red-only: deception feints (non-carrier agents only) ========
        if (not is_blue) and deception_prob.numel() == B and float(deception_prob.max().item()) > 0.0:
            pulse = (self.sim_step_count % 30 == 0)
            p = torch.clamp(deception_prob, 0.0, 1.0)
            r = torch.rand((B,), generator=self._rng, device=device)
            feint_env = pulse & (r < p)
            if feint_env.any():
                env_idx = torch.where(feint_env)[0]
                hold_x = torch.full((env_idx.numel(),), min(max_x, midline - 0.5), device=device)
                hold_y = enemy_flag_home[env_idx, 1]
                punch_x = enemy_flag_home[env_idx, 0]
                punch_y = enemy_flag_home[env_idx, 1]
                do_hold = ((self.sim_step_count[env_idx] // 30) % 2 == 0)
                tx = torch.where(do_hold, hold_x, punch_x)
                ty = torch.where(do_hold, hold_y, punch_y)
                tx = torch.clamp(tx + self._rand_uniform((env_idx.numel(),), -3.0, 3.0), 0.0, max_x)
                ty = torch.clamp(ty + self._rand_uniform((env_idx.numel(),), -3.0, 3.0), 0.0, max_y)
                for j in range(N):
                    not_carry = ~own_carrying[env_idx, j]
                    target[env_idx, j, 0] = torch.where(not_carry, tx, target[env_idx, j, 0])
                    target[env_idx, j, 1] = torch.where(not_carry, ty, target[env_idx, j, 1])

        return target[..., 0], target[..., 1]

    def _red_scripted_actions(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Scripted red team -- delegates to the unified NPC brain."""
        return self._get_scripted_targets("red")

    def _align_speed_cap_to_speed(self, speed: torch.Tensor, speed_cap: torch.Tensor) -> torch.Tensor:
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

        max_yaw = min(4.0, float(self.cfg.max_yaw_rate_rps))
        yaw_rate_cmd = torch.clamp(err / max(1e-6, dt), -max_yaw, max_yaw)
        # Turn-radius bound: |omega| <= v / R_min (with floor for low-speed controllability)
        min_r = max(1e-3, float(self.cfg.min_turn_radius_cells))
        omega_bound = torch.clamp(speed / min_r, min=0.5, max=max_yaw)
        yaw_rate_cmd = torch.clamp(yaw_rate_cmd, -omega_bound, omega_bound)

        max_speed = min(2.2, float(self.cfg.max_speed_cps))
        max_accel = min(2.0, float(self.cfg.max_accel_cps2))
        desired_speed = torch.full_like(speed, max_speed)
        if speed_cap is not None:
            desired_speed = torch.minimum(desired_speed, torch.clamp(speed_cap, min=0.0))
        dist = torch.sqrt(dx * dx + dy * dy + 1e-8)
        # Deceleration zone near objective to prevent overshoot/spin lock.
        desired_speed = torch.where(dist < 2.0, desired_speed * torch.clamp(dist / 2.0, 0.0, 1.0), desired_speed)
        accel_cmd = torch.clamp(
            (desired_speed - speed) / max(1e-6, dt),
            -max_accel,
            max_accel,
        )

        speed2 = torch.clamp(speed + accel_cmd * dt, 0.0, max_speed)
        if speed_cap is not None:
            cap = self._align_speed_cap_to_speed(speed, speed_cap)
            speed2 = torch.minimum(speed2, cap)
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
        close_bb = (d < rr) & (~eye)
        if close_bb.any():
            dir_x = ddx / d
            dir_y = ddy / d
            # Net repulsion from all close neighbors
            fx = (dir_x * close_bb.to(dir_x.dtype)).sum(dim=2)
            fy = (dir_y * close_bb.to(dir_y.dtype)).sum(dim=2)
            norm = torch.sqrt(fx * fx + fy * fy + 1e-8)
            fx = fx / norm * shove
            fy = fy / norm * shove
            self.blue_x = torch.clamp(self.blue_x + fx, 0.0, float(max(0, self.cols - 1)))
            self.blue_y = torch.clamp(self.blue_y + fy, 0.0, float(max(0, self.rows - 1)))

        # Red-Red repulsion
        ddx_r = self.red_x[:, :, None] - self.red_x[:, None, :]
        ddy_r = self.red_y[:, :, None] - self.red_y[:, None, :]
        d_r = torch.sqrt(ddx_r * ddx_r + ddy_r * ddy_r + 1e-8)
        eye_r = torch.eye(self.Nr, dtype=torch.bool, device=self.device)[None, :, :]
        close_rr = (d_r < rr) & (~eye_r)
        if close_rr.any():
            dir_xr = ddx_r / d_r
            dir_yr = ddy_r / d_r
            fx_r = (dir_xr * close_rr.to(dir_xr.dtype)).sum(dim=2)
            fy_r = (dir_yr * close_rr.to(dir_yr.dtype)).sum(dim=2)
            norm_r = torch.sqrt(fx_r * fx_r + fy_r * fy_r + 1e-8)
            fx_r = fx_r / norm_r * shove
            fy_r = fy_r / norm_r * shove
            self.red_x = torch.clamp(self.red_x + fx_r, 0.0, float(max(0, self.cols - 1)))
            self.red_y = torch.clamp(self.red_y + fy_r, 0.0, float(max(0, self.rows - 1)))

        # Blue-Red repulsion
        dx_br = self.blue_x[:, :, None] - self.red_x[:, None, :]
        dy_br = self.blue_y[:, :, None] - self.red_y[:, None, :]
        d_br = torch.sqrt(dx_br * dx_br + dy_br * dy_br + 1e-8)
        close_br = d_br < rr
        if close_br.any():
            dir_xbr = dx_br / d_br
            dir_ybr = dy_br / d_br
            # For blue, repel away from red
            fx_b = (dir_xbr * close_br.to(dir_xbr.dtype)).sum(dim=2)
            fy_b = (dir_ybr * close_br.to(dir_ybr.dtype)).sum(dim=2)
            norm_b = torch.sqrt(fx_b * fx_b + fy_b * fy_b + 1e-8)
            fx_b = fx_b / norm_b * shove
            fy_b = fy_b / norm_b * shove
            self.blue_x = torch.clamp(self.blue_x + fx_b, 0.0, float(max(0, self.cols - 1)))
            self.blue_y = torch.clamp(self.blue_y + fy_b, 0.0, float(max(0, self.rows - 1)))
            # For red, repel in the opposite direction
            fx_r2 = -(dir_xbr * close_br.to(dir_xbr.dtype)).sum(dim=1)
            fy_r2 = -(dir_ybr * close_br.to(dir_ybr.dtype)).sum(dim=1)
            norm_r2 = torch.sqrt(fx_r2 * fx_r2 + fy_r2 * fy_r2 + 1e-8)
            fx_r2 = fx_r2 / norm_r2 * shove
            fy_r2 = fy_r2 / norm_r2 * shove
            self.red_x = torch.clamp(self.red_x + fx_r2, 0.0, float(max(0, self.cols - 1)))
            self.red_y = torch.clamp(self.red_y + fy_r2, 0.0, float(max(0, self.rows - 1)))

    # ------------------------------------------------------------------
    # Mine system: pickups spawn; agents GRAB_MINE then PLACE_MINE anywhere.
    # ------------------------------------------------------------------
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

    def _apply_mine_pickups(self, macro_blue: torch.Tensor, macro_red: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Respawn pickups; then blue grabs with GRAB_MINE or auto-grab when scripted; red grabs when near (scripted).
        """
        B, device = self.B, self.device
        Np = self.Np
        radius = float(getattr(self.cfg, "mine_pickup_radius_cells", 1.2))
        respawn_delay = int(getattr(self.cfg, "mine_pickup_respawn_steps", 0))
        max_charge = int(getattr(self.cfg, "max_mine_charges_per_agent", 2))
        blue_pickups = torch.zeros((B,), dtype=torch.float32, device=device)
        blue_pickup_agents = torch.zeros((B, self.Nb), dtype=torch.bool, device=device)

        # If respawn_delay > 0, pickups will respawn after a cooldown; when <= 0, pickups are single-use.
        if respawn_delay > 0:
            self.pickup_respawn = torch.clamp(self.pickup_respawn - 1, min=0)
            self.pickup_active = self.pickup_active | ((self.pickup_respawn <= 0) & (~self.pickup_active))

        # Blue: (GRAB_MINE or scripted) and near an active pickup and under max charge
        grab_blue = ((macro_blue == MacroAction.GRAB_MINE) | self.blue_scripted) & (self.blue_mine_charges < max_charge)
        for i in range(self.Nb):
            dx = self.blue_x[:, i : i + 1] - self.pickup_x[:, :]
            dy = self.blue_y[:, i : i + 1] - self.pickup_y[:, :]
            dist = torch.sqrt(dx * dx + dy * dy + 1e-8)
            near = (dist <= radius) & self.pickup_active
            for k in range(Np):
                take = grab_blue[:, i] & near[:, k]
                if take.any():
                    blue_pickups += take.to(torch.float32)
                    blue_pickup_agents[:, i] = blue_pickup_agents[:, i] | take
                    self.blue_mine_charges[:, i] = torch.where(
                        take,
                        torch.clamp(self.blue_mine_charges[:, i] + 1, max=max_charge),
                        self.blue_mine_charges[:, i],
                    )
                    self.pickup_active[:, k] = self.pickup_active[:, k] & (~take)
                    self.pickup_respawn[:, k] = torch.where(take, torch.full_like(self.pickup_respawn[:, k], respawn_delay), self.pickup_respawn[:, k])
                    break

        # Red: snapshot opponents use GRAB_MINE; scripted opponents auto-grab when near.
        for i in range(self.Nr):
            under = self.red_mine_charges[:, i] < max_charge
            dx = self.red_x[:, i : i + 1] - self.pickup_x[:, :]
            dy = self.red_y[:, i : i + 1] - self.pickup_y[:, :]
            dist = torch.sqrt(dx * dx + dy * dy + 1e-8)
            near = (dist <= radius) & self.pickup_active & under[:, None]
            if macro_red is not None:
                near = near & (macro_red[:, i : i + 1] == int(MacroAction.GRAB_MINE))
            for k in range(Np):
                take = near[:, k]
                if take.any():
                    self.red_mine_charges[:, i] = torch.where(
                        take,
                        torch.clamp(self.red_mine_charges[:, i] + 1, max=max_charge),
                        self.red_mine_charges[:, i],
                    )
                    self.pickup_active[:, k] = self.pickup_active[:, k] & (~take)
                    self.pickup_respawn[:, k] = torch.where(take, torch.full_like(self.pickup_respawn[:, k], respawn_delay), self.pickup_respawn[:, k])
                    break
        return blue_pickups, blue_pickup_agents

    def _apply_mine_placement(self, macro_blue: torch.Tensor, macro_red: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Blue: PLACE_MINE or scripted (defender every 50 steps) places at current position if charge > 0.
        Red: scripted placement when has charge (e.g. defender places every 50 steps).
        """
        B, device = self.B, self.device
        Nm = self.Nm
        blue_placements = torch.zeros((B,), dtype=torch.float32, device=device)
        blue_placement_agents = torch.zeros((B, self.Nb), dtype=torch.bool, device=device)

        # Blue: (PLACE_MINE or scripted defender) and charge > 0 -> place at (blue_x, blue_y) in first free slot.
        # Use an explicit defender mask with the same (B, Nb) shape as macro_blue to avoid
        # shape/broadcast issues when Nb != B (e.g. 2v2 with n_envs=4 on Colab).
        scripted_mask: torch.Tensor
        if self.blue_scripted and (self.sim_step_count % 50) == 0:
            scripted_mask = torch.zeros_like(macro_blue, dtype=torch.bool, device=device)
            # Defender is agent index 0 for each env
            scripted_mask[:, 0] = True
        else:
            scripted_mask = torch.zeros_like(macro_blue, dtype=torch.bool, device=device)

        place_blue = ((macro_blue == MacroAction.PLACE_MINE) | scripted_mask) & (self.blue_mine_charges > 0)
        for i in range(self.Nb):
            for slot in range(Nm):
                can = place_blue[:, i] & (~self.blue_mine_active[:, slot])
                if can.any():
                    blue_placements += can.to(torch.float32)
                    blue_placement_agents[:, i] = blue_placement_agents[:, i] | can
                    self.blue_mine_x[can, slot] = self.blue_x[can, i]
                    self.blue_mine_y[can, slot] = self.blue_y[can, i]
                    self.blue_mine_active[can, slot] = True
                    self.blue_mine_charges[:, i] = torch.where(can, torch.clamp(self.blue_mine_charges[:, i] - 1, min=0), self.blue_mine_charges[:, i])
                    break

        if macro_red is None:
            # Red: scripted place when defender (agent 0) has charge, every 50 steps
            place_red = (self.red_mine_charges[:, 0] > 0) & ((self.sim_step_count % 50) == 0)
            for slot in range(Nm):
                can = place_red & (~self.red_mine_active[:, slot])
                if can.any():
                    self.red_mine_x[can, slot] = self.red_x[can, 0]
                    self.red_mine_y[can, slot] = self.red_y[can, 0]
                    self.red_mine_active[can, slot] = True
                    self.red_mine_charges[:, 0] = torch.where(can, torch.clamp(self.red_mine_charges[:, 0] - 1, min=0), self.red_mine_charges[:, 0])
                    break
        else:
            place_red = (macro_red == int(MacroAction.PLACE_MINE)) & (self.red_mine_charges > 0)
            for i in range(self.Nr):
                for slot in range(Nm):
                    can = place_red[:, i] & (~self.red_mine_active[:, slot])
                    if can.any():
                        self.red_mine_x[can, slot] = self.red_x[can, i]
                        self.red_mine_y[can, slot] = self.red_y[can, i]
                        self.red_mine_active[can, slot] = True
                        self.red_mine_charges[:, i] = torch.where(can, torch.clamp(self.red_mine_charges[:, i] - 1, min=0), self.red_mine_charges[:, i])
                        break
        return blue_placements, blue_placement_agents

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
        if self.blue_carrying.any():
            idx = torch.argmax(self.blue_carrying.to(torch.int64), dim=1)
            env = torch.arange(self.B, device=self.device)
            self.red_flag_pos[env] = torch.stack(
                [self.blue_x[env, idx], self.blue_y[env, idx]], dim=1
            )
        if self.red_carrying.any():
            idx = torch.argmax(self.red_carrying.to(torch.int64), dim=1)
            env = torch.arange(self.B, device=self.device)
            self.blue_flag_pos[env] = torch.stack(
                [self.red_x[env, idx], self.red_y[env, idx]], dim=1
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
        # red has blue's flag, and vice versa. Each grab only updates that side's carrying state.
        blue_grab_env = ((b_to_red <= grab_r) & (~self.blue_tagged)).any(dim=1)
        red_grab_env = ((r_to_blue <= grab_r) & (~self.red_tagged)).any(dim=1)

        grab_delta = get_grab_score_delta(self.rules_profile)
        if blue_grab_env.any():
            idx = torch.argmax(((b_to_red <= grab_r) & (~self.blue_tagged)).to(torch.int64), dim=1)
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
            idx = torch.argmax(((r_to_blue <= grab_r) & (~self.red_tagged)).to(torch.int64), dim=1)
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

    def _build_blue_targets_from_action(self, macro: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._build_targets_from_action(macro, target, side="blue")

    def _get_red_snapshot_actions(self, env_mask: torch.Tensor) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        idx = torch.where(env_mask)[0]
        if idx.numel() == 0:
            return None
        obs_full = self.get_obs_tensors(side="red")
        red_macro = torch.zeros((self.B, self.Nr), device=self.device, dtype=torch.int64)
        red_target = torch.zeros((self.B, self.Nr), device=self.device, dtype=torch.int64)
        any_loaded = False
        grouped: Dict[str, List[int]] = {}
        for env_i in idx.detach().cpu().tolist():
            resolved = _resolve_snapshot_path(self._opponent_key[env_i])
            if resolved is None:
                continue
            grouped.setdefault(self._opponent_key[env_i], []).append(env_i)
        for snapshot_key, env_list in grouped.items():
            model = self._load_snapshot_policy(snapshot_key)
            if model is None:
                continue
            any_loaded = True
            sub_idx = torch.as_tensor(env_list, device=self.device, dtype=torch.int64)
            obs = {
                k: v.index_select(0, sub_idx).detach().cpu().numpy().astype(np.float32)
                for k, v in obs_full.items()
            }
            actions_np, _ = model.predict(obs, deterministic=True)
            actions = torch.as_tensor(actions_np, device=self.device, dtype=torch.int64).reshape(sub_idx.numel(), self.Nr, 2)
            red_macro[sub_idx] = torch.remainder(actions[..., 0], self.cfg.n_macros).long()
            red_target[sub_idx] = torch.remainder(actions[..., 1], self.cfg.n_targets).long()
        if not any_loaded:
            return None
        return red_macro, red_target

    def _compute_potentials(
        self,
        blue_x: torch.Tensor,
        blue_y: torch.Tensor,
        blue_carrying: torch.Tensor,
        red_carrying: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        attack_dx = self.red_flag_pos[:, None, 0] - blue_x
        attack_dy = self.red_flag_pos[:, None, 1] - blue_y
        attack_dist = torch.sqrt(attack_dx * attack_dx + attack_dy * attack_dy + 1e-8)
        attack_phi = torch.where(
            blue_carrying,
            torch.zeros_like(attack_dist),
            -attack_dist / max(1e-6, self.max_dist),
        ).mean(dim=1)

        return_dx = self.blue_flag_home[:, None, 0] - blue_x
        return_dy = self.blue_flag_home[:, None, 1] - blue_y
        return_dist = torch.sqrt(return_dx * return_dx + return_dy * return_dy + 1e-8)
        return_phi = torch.where(
            blue_carrying,
            -return_dist / max(1e-6, self.max_dist),
            torch.zeros_like(return_dist),
        ).mean(dim=1)

        red_has_flag = red_carrying.any(dim=1)
        red_carrier_idx = torch.argmax(red_carrying.to(torch.int64), dim=1)
        red_carrier_x = self.red_x[torch.arange(self.B, device=self.device), red_carrier_idx]
        red_carrier_y = self.red_y[torch.arange(self.B, device=self.device), red_carrier_idx]
        defend_dx = red_carrier_x[:, None] - blue_x
        defend_dy = red_carrier_y[:, None] - blue_y
        defend_dist = torch.sqrt(defend_dx * defend_dx + defend_dy * defend_dy + 1e-8)
        defend_phi = torch.where(
            red_has_flag[:, None],
            -defend_dist / max(1e-6, self.max_dist),
            torch.zeros_like(defend_dist),
        ).mean(dim=1)
        return attack_phi, return_phi, defend_phi

    def _pbrs_reward(
        self,
        prev_blue_x: torch.Tensor,
        prev_blue_y: torch.Tensor,
        prev_blue_carrying: torch.Tensor,
    ) -> torch.Tensor:
        prev_attack, prev_return, prev_defend = self._compute_potentials(
            prev_blue_x, prev_blue_y, prev_blue_carrying, self.red_carrying
        )
        cur_attack, cur_return, cur_defend = self._compute_potentials(
            self.blue_x, self.blue_y, self.blue_carrying, self.red_carrying
        )
        gamma = float(self.cfg.pbrs_gamma)
        rpbrs = (
            float(self.cfg.pbrs_attack_coef) * (gamma * cur_attack - prev_attack)
            + float(self.cfg.pbrs_return_coef) * (gamma * cur_return - prev_return)
            + float(self.cfg.pbrs_defense_coef) * (gamma * cur_defend - prev_defend)
        )
        self._last_dense_progress = gamma * cur_attack - prev_attack
        return rpbrs

    def _team_coordination_reward(
        self,
        prev_blue_x: torch.Tensor,
        prev_blue_y: torch.Tensor,
        yaw_cmd_blue: torch.Tensor,
    ) -> torch.Tensor:
        red_has_flag = self.red_carrying.any(dim=1)
        home_dx = self.blue_x - self.blue_flag_home[:, None, 0]
        home_dy = self.blue_y - self.blue_flag_home[:, None, 1]
        near_home = (torch.sqrt(home_dx * home_dx + home_dy * home_dy + 1e-8) <= 6.0).to(torch.float32).mean(dim=1)
        defense_presence = torch.where(
            red_has_flag,
            float(self.cfg.team_defense_presence_reward) * near_home,
            torch.zeros_like(near_home),
        )

        blue_has_flag = self.blue_carrying.any(dim=1)
        carrier_idx = torch.argmax(self.blue_carrying.to(torch.int64), dim=1)
        carrier_x = self.blue_x[torch.arange(self.B, device=self.device), carrier_idx]
        carrier_y = self.blue_y[torch.arange(self.B, device=self.device), carrier_idx]
        edx = self.blue_x - carrier_x[:, None]
        edy = self.blue_y - carrier_y[:, None]
        escort = (torch.sqrt(edx * edx + edy * edy + 1e-8) <= 5.0).to(torch.float32).mean(dim=1)
        escort_bonus = torch.where(
            blue_has_flag,
            float(self.cfg.team_escort_reward) * escort,
            torch.zeros_like(escort),
        )

        red_carrier_idx = torch.argmax(self.red_carrying.to(torch.int64), dim=1)
        red_carrier_x = self.red_x[torch.arange(self.B, device=self.device), red_carrier_idx]
        red_carrier_y = self.red_y[torch.arange(self.B, device=self.device), red_carrier_idx]
        intercept_dx = self.blue_x - red_carrier_x[:, None]
        intercept_dy = self.blue_y - red_carrier_y[:, None]
        intercept = (torch.sqrt(intercept_dx * intercept_dx + intercept_dy * intercept_dy + 1e-8) <= 5.0).to(torch.float32).mean(dim=1)
        intercept_bonus = torch.where(
            red_has_flag,
            float(self.cfg.team_intercept_reward) * intercept,
            torch.zeros_like(intercept),
        )

        yaw_abs = torch.abs(yaw_cmd_blue) / max(1e-6, float(self.cfg.max_yaw_rate_rps))
        move_dist = torch.sqrt((self.blue_x - prev_blue_x) ** 2 + (self.blue_y - prev_blue_y) ** 2 + 1e-8)
        spin_pen = self.cfg.spin_penalty_coef * (yaw_abs * (move_dist < 0.03).to(torch.float32)).mean(dim=1)
        idle_pen = self.cfg.idle_penalty_coef * (self.blue_speed.mean(dim=1) < 0.15).to(torch.float32)
        return defense_presence + escort_bonus + intercept_bonus - spin_pen - idle_pen

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
        blue_mine_tags: Optional[torch.Tensor] = None,
        red_mine_tags: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Values from game_manager (sparse event points) so scoring/rewards stay aligned.
        r = torch.zeros((self.B,), dtype=torch.float32, device=self.device)
        r += float(SPARSE_OOB_POINTS) * blue_oob.sum(dim=1).to(torch.float32)
        return r

    def _reward_total(
        self,
        rterm: torch.Tensor,
        roff: torch.Tensor,
        rpbrs: torch.Tensor,
        rteam: torch.Tensor,
        sparse_points: torch.Tensor,
        stalemate_trigger: torch.Tensor,
    ) -> torch.Tensor:
        sparse_norm = sparse_points / 100.0
        raw = rterm + roff + rpbrs + rteam + self.cfg.sparse_weight * sparse_norm
        raw = raw + torch.where(stalemate_trigger, torch.tensor(float(self.cfg.stalemate_penalty), device=self.device), torch.tensor(0.0, device=self.device))
        scaled = torch.tanh(raw / max(1e-6, float(self.cfg.reward_scale)))
        return torch.clamp(scaled, -float(self.cfg.reward_clip), float(self.cfg.reward_clip))

    def step(
        self,
        blue_action_flat: torch.Tensor,
        *,
        tensor_obs: bool = False,
        red_action_flat: Optional[torch.Tensor] = None,
    ):
        self._apply_profile_runtime()
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
        macro = self.blue_commit_macro
        targ = self.blue_commit_target

        prev_blue_x = self.blue_x.clone()
        prev_blue_y = self.blue_y.clone()
        prev_red_x = self.red_x.clone()
        prev_red_y = self.red_y.clone()
        prev_blue_alive = self.blue_alive.clone()
        prev_red_alive = self.red_alive.clone()
        prev_blue_carrying = self.blue_carrying.clone()
        prev_red_carrying = self.red_carrying.clone()
        prev_red_mine_charges = self.red_mine_charges.clone()

        if self.blue_scripted:
            btx, bty = self._get_scripted_targets("blue")
        else:
            btx, bty = self._build_targets_from_action(macro, targ, side="blue")
        red_macro: Optional[torch.Tensor] = None
        rtx, rty = self._red_scripted_actions()
        if red_action_flat is not None:
            if red_action_flat.device != self.device:
                red_action_flat = red_action_flat.to(self.device)
            n_red_exp = int(self.B * self.Nr * 2)
            if int(red_action_flat.numel()) != n_red_exp:
                raise ValueError(
                    f"BatchedCTFCore.step: expected {n_red_exp} red action ints (B={self.B}, Nr={self.Nr}), "
                    f"got numel={int(red_action_flat.numel())} shape={tuple(red_action_flat.shape)}"
                )
            red_a = red_action_flat.reshape(self.B, self.Nr, 2)
            red_requested_macro = torch.remainder(red_a[..., 0].long(), self.cfg.n_macros)
            red_requested_targ = torch.remainder(red_a[..., 1].long(), self.cfg.n_targets)
            red_control_mask = torch.ones((self.B,), device=self.device, dtype=torch.bool)
            external_red_mask = red_control_mask[:, None]
            new_red_commit = external_red_mask & (self.red_commit_ticks_left <= 0)
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
            rtx, rty = self._build_targets_from_action(red_macro, red_targ, side="red")
        else:
            red_control_mask = torch.as_tensor(
                [
                    str(self._opponent_kind[i]).upper() == "SNAPSHOT" and _resolve_snapshot_path(self._opponent_key[i]) is not None
                    for i in range(self.B)
                ],
                device=self.device,
                dtype=torch.bool,
            )
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

        # Tagged agents: OpRegion-like forced safe return to home region.
        if bool(self.blue_tagged.any().item()):
            btx = torch.where(self.blue_tagged, self.blue_flag_home[:, None, 0], btx)
            bty = torch.where(self.blue_tagged, self.blue_flag_home[:, None, 1], bty)
        if bool(self.red_tagged.any().item()):
            rtx = torch.where(self.red_tagged, self.red_flag_home[:, None, 0], rtx)
            rty = torch.where(self.red_tagged, self.red_flag_home[:, None, 1], rty)
        # Use normal speed for all blue agents. Red speed is modulated by the opponent
        # speed multiplier (scripted difficulty) on a per-env basis.
        blue_speed_cap = torch.full_like(self.blue_speed, float(self.cfg.max_speed_cps))
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
            self.blue_x, self.blue_y, self.blue_heading, self.blue_speed, self.blue_alive, btx, bty, speed_cap=blue_speed_cap
        )
        self.red_x, self.red_y, self.red_heading, self.red_speed, red_oob, _ = self._integrate_side(
            self.red_x, self.red_y, self.red_heading, self.red_speed, self.red_alive, rtx, rty, speed_cap=red_speed_cap
        )

        # AvoidCollision safety guardrail (halt motion if too close)
        self._apply_avoid_collision_guard(prev_blue_x, prev_blue_y, prev_red_x, prev_red_y)

        if bool(self.cfg.aquaticus_profile) or self.rules_profile in ("AQUATICUS_2024", "OURS", "OURS_PLUS"):
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

        # Mines: pickups (grab with GRAB_MINE / near for red), then place (PLACE_MINE / scripted red), then trigger
        blue_mine_pickups, blue_mine_pickup_agents = self._apply_mine_pickups(macro, red_macro)
        blue_mine_placements, blue_mine_placement_agents = self._apply_mine_placement(macro, red_macro)
        blue_mine_tags, red_mine_tags = self._apply_mine_triggers()
        self._untag_if_home()

        blue_grab_env, red_grab_env, blue_cap_env, red_cap_env = self._apply_flag_rules()
        blue_grab_agents = self.blue_carrying & (~prev_blue_carrying)
        blue_cap_agents = prev_blue_carrying & (~self.blue_carrying) & blue_cap_env[:, None]
        red_grab_agents = self.red_carrying & (~prev_red_carrying)
        red_cap_agents = prev_red_carrying & (~self.red_carrying) & red_cap_env[:, None]
        red_mine_pickup_agents = self.red_mine_charges > prev_red_mine_charges
        red_mine_placement_agents = self.red_mine_charges < prev_red_mine_charges
        commit_target_xy = self._decode_targets(self.blue_commit_target)
        commit_dist = torch.sqrt(
            (self.blue_x - commit_target_xy[..., 0]) ** 2
            + (self.blue_y - commit_target_xy[..., 1]) ** 2
            + 1e-8
        )
        action_success = torch.zeros_like(self.blue_commit_success)
        action_success = action_success | ((self.blue_commit_macro == int(MacroAction.GO_TO)) & (commit_dist <= float(self.cfg.macro_arrival_radius_cells)))
        action_success = action_success | ((self.blue_commit_macro == int(MacroAction.GRAB_MINE)) & blue_mine_pickup_agents)
        action_success = action_success | ((self.blue_commit_macro == int(MacroAction.GET_FLAG)) & blue_grab_agents)
        action_success = action_success | ((self.blue_commit_macro == int(MacroAction.PLACE_MINE)) & blue_mine_placement_agents)
        action_success = action_success | ((self.blue_commit_macro == int(MacroAction.GO_HOME)) & blue_cap_agents)
        self.blue_commit_success = self.blue_commit_success | action_success
        if bool(red_control_mask.any().item()) and red_macro is not None:
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
                (self.red_commit_macro == int(MacroAction.GRAB_MINE)) & red_mine_pickup_agents
            )
            red_action_success = red_action_success | (
                (self.red_commit_macro == int(MacroAction.GET_FLAG)) & red_grab_agents
            )
            red_action_success = red_action_success | (
                (self.red_commit_macro == int(MacroAction.PLACE_MINE)) & red_mine_placement_agents
            )
            red_action_success = red_action_success | (
                (self.red_commit_macro == int(MacroAction.GO_HOME)) & red_cap_agents
            )
            self.red_commit_success = self.red_commit_success | (red_action_success & red_control_mask[:, None])

        sparse_points = self._sparse_reward_points(
            blue_grab_env, red_grab_env, blue_cap_env, red_cap_env,
            blue_tag_noflag, blue_tag_withflag, red_tag_total, blue_oob,
            blue_mine_tags=blue_mine_tags, red_mine_tags=red_mine_tags,
        )
        blue_kill_count = blue_tag_noflag + blue_tag_withflag + blue_mine_tags
        roff = torch.zeros((self.B,), dtype=torch.float32, device=self.device)
        roff += float(self.cfg.flag_pickup_reward) * blue_grab_agents.sum(dim=1).to(torch.float32)
        roff += float(self.cfg.flag_carry_home_reward) * blue_cap_agents.sum(dim=1).to(torch.float32)
        roff += float(self.cfg.enabled_mine_reward) * blue_mine_placement_agents.sum(dim=1).to(torch.float32)
        roff += float(self.cfg.enemy_mav_kill_reward) * blue_kill_count
        self.blue_commit_ticks_left = torch.clamp(self.blue_commit_ticks_left - 1, min=0)
        ended_commit = self.blue_commit_success | (self.blue_commit_ticks_left <= 0) | (~self.blue_alive) | self.blue_tagged
        failed_commit = ended_commit & (~self.blue_commit_success) & prev_blue_alive
        roff += float(self.cfg.action_failed_punishment) * failed_commit.sum(dim=1).to(torch.float32)
        self.blue_commit_ticks_left = torch.where(ended_commit, torch.zeros_like(self.blue_commit_ticks_left), self.blue_commit_ticks_left)
        self.blue_commit_success = torch.where(ended_commit, torch.zeros_like(self.blue_commit_success), self.blue_commit_success)
        if bool(red_control_mask.any().item()) and red_macro is not None:
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
        rpbrs = self._pbrs_reward(prev_blue_x, prev_blue_y, prev_blue_carrying)
        rteam = self._team_coordination_reward(prev_blue_x, prev_blue_y, yaw_cmd_blue)

        self.step_count += 1
        self.sim_step_count += 1
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
        reward = self._reward_total(rterm, roff, rpbrs, rteam, sparse_points, stalemate_trigger)
        obs_t = self.get_obs_tensors()
        info = self._build_info(dense=rpbrs + rteam, sparse_points=sparse_points, stalemate=stalemate_trigger)
        if tensor_obs:
            return obs_t, reward, terminated, truncated, info
        return (
            {k: v.detach().cpu().numpy().astype(np.float32) for k, v in obs_t.items()},
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
        sim_steps = self.sim_step_count.detach().cpu().numpy()
        d_np = dense.detach().cpu().numpy()
        s_np = sparse_points.detach().cpu().numpy()
        st_np = stalemate.detach().cpu().numpy()
        for i in range(self.B):
            out.append(
                {
                    "blue_score": int(bs[i]),
                    "red_score": int(rs[i]),
                    "decision_steps": int(steps[i]),
                    "sim_steps": int(sim_steps[i]),
                    "phase": self._phase[i],
                    "league_mode": bool(self._league_mode[i].item()),
                    "opponent_kind": str(self._opponent_kind[i]).lower(),
                    "opponent_key": self._opponent_key[i],
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
                if self.cfg.sensor_dropout_prob > 0.0:
                    drop = torch.rand(in_range.shape, generator=self._rng, device=self.device) < float(self.cfg.sensor_dropout_prob)
                    in_range = in_range & (~drop)
                elive = elive & in_range
                if self.cfg.sensor_noise_sigma_cells > 0.0:
                    ex = torch.clamp(ex + self._randn(ex.shape) * float(self.cfg.sensor_noise_sigma_cells), 0.0, float(max(0, self.cols - 1)))
                    ey = torch.clamp(ey + self._randn(ey.shape) * float(self.cfg.sensor_noise_sigma_cells), 0.0, float(max(0, self.rows - 1)))
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
        """Global state grid as a flat GPU tensor [B, GLOBAL_STATE_CHANNELS*H*W]."""
        g = torch.zeros((self.B, GLOBAL_STATE_CHANNELS, CNN_ROWS, CNN_COLS), dtype=torch.float32, device=self.device)
        self._scatter_points(g, 0, self.blue_x, self.blue_y, self.blue_alive)
        self._scatter_points(g, 1, self.red_x, self.red_y, self.red_alive)
        self._scatter_points(g, 2, self.blue_mine_x, self.blue_mine_y, self.blue_mine_active)
        self._scatter_points(g, 3, self.red_mine_x, self.red_mine_y, self.red_mine_active)
        self._scatter_points(g, 4, self.pickup_x, self.pickup_y, self.pickup_active)
        self._scatter_points(g, 6, self.blue_flag_pos[:, 0:1], self.blue_flag_pos[:, 1:2], torch.ones((self.B, 1), dtype=torch.bool, device=self.device))
        self._scatter_points(g, 7, self.red_flag_pos[:, 0:1], self.red_flag_pos[:, 1:2], torch.ones((self.B, 1), dtype=torch.bool, device=self.device))
        return g.reshape(self.B, -1)

    def get_global_state(self) -> np.ndarray:
        return self.get_global_state_tensor().detach().cpu().numpy().astype(np.float32)


# -------- Adapter for MAPPO/QMIX: GameField-like API over BatchedCTFCore(B=1) --------


class _FakeGM:
    """Minimal GameManager stand-in for MAPPO/QMIX (scores, game_over, set_phase, terminal bonus)."""

    def __init__(self, core: BatchedCTFCore):
        assert core.B == 1, "FakeGM only supports single env"
        self._core = core
        self._phase = "OP1"

    @property
    def blue_score(self) -> int:
        return int(self._core.blue_score[0].item())

    @property
    def red_score(self) -> int:
        return int(self._core.red_score[0].item())

    @property
    def game_over(self) -> bool:
        return bool(self._core.done[0].item())

    def set_phase(self, phase: str) -> None:
        self._phase = str(phase).upper()
        self._core.set_phase(phase)

    def pop_reward_events(self):
        """No per-event routing in GPU env; return empty."""
        return iter(())

    def terminal_outcome_bonus(self, blue_score: int, red_score: int) -> float:
        if blue_score > red_score:
            return 1.0
        if blue_score < red_score:
            return -1.0
        return -0.5


class GPUEnvAdapter:
    """
    GameField-like wrapper around BatchedCTFCore with B=1 for MAPPO/QMIX training.
    Provides: reset_default, build_observation(agent), get_macro_target, get_macro_mask,
    get_target_mask, blue_agents, getGameManager(), step(actions_flat), get_global_state, etc.
    """

    def __init__(self, cfg: Optional[GPUFieldConfig] = None):
        cfg = cfg or GPUFieldConfig(n_envs=1)
        cfg.n_envs = 1
        self._cfg = cfg
        self._core = BatchedCTFCore(cfg)
        self.n_macros = int(cfg.n_macros)
        self.num_macro_targets = int(cfg.n_targets)
        self.agents_per_team = int(cfg.max_blue_agents)
        self._gm = _FakeGM(self._core)
        self._blue_agents: List[AgentHandle] = []
        self._refresh_agents()

    def _refresh_agents(self) -> None:
        def alive(i: int) -> bool:
            return bool(self._core.blue_alive[0, i].item())

        self._blue_agents = [
            AgentHandle(i, "blue", alive_getter=alive)
            for i in range(self.agents_per_team)
        ]

    @property
    def blue_agents(self) -> List[AgentHandle]:
        self._refresh_agents()
        return self._blue_agents

    def set_external_control(self, side: str, value: bool) -> None:
        pass

    def set_red_opponent(self, tag: str) -> None:
        self._core.set_phase(tag.upper())

    use_internal_policies: bool = True

    def reset_default(self) -> None:
        self._core.reset_all()
        self._refresh_agents()

    def getGameManager(self) -> _FakeGM:
        return self._gm

    def get_obs(self) -> Dict[str, np.ndarray]:
        return self._core.get_obs()

    def build_observation(self, agent: AgentHandle) -> np.ndarray:
        """Single-agent grid obs (C, H, W) for policy.act()."""
        ot = self._core.get_obs_tensors()
        grid = ot["grid"]
        i = getattr(agent, "agent_id", 0)
        return grid[0, i].detach().cpu().numpy().astype(np.float32)

    def get_macro_target(self, index: int) -> Tuple[float, float]:
        """Return (x, y) for macro target index (for policies that need it)."""
        idx = int(index) % self._core._macro_targets.size(0)
        x = float(self._core._macro_targets[idx, 0].item())
        y = float(self._core._macro_targets[idx, 1].item())
        return (x, y)

    def _mask_for_agent(self, agent: AgentHandle, macro_only: bool) -> np.ndarray:
        m = self._core._build_action_mask()
        i = getattr(agent, "agent_id", 0)
        n_m = self._cfg.n_macros
        n_t = self._cfg.n_targets
        base = i * (n_m + n_t)
        if macro_only:
            return m[0, base : base + n_m].detach().cpu().numpy().astype(np.bool_)
        return m[0, base + n_m : base + n_m + n_t].detach().cpu().numpy().astype(np.bool_)

    def get_macro_mask(self, agent: AgentHandle) -> np.ndarray:
        return self._mask_for_agent(agent, macro_only=True)

    def get_target_mask(self, agent: AgentHandle) -> np.ndarray:
        return self._mask_for_agent(agent, macro_only=False)

    def get_global_state_dim(self) -> int:
        return int(GLOBAL_STATE_CHANNELS * CNN_ROWS * CNN_COLS)

    def get_global_state(self) -> np.ndarray:
        return self._core.get_global_state()[0]

    def step(self, actions_flat: np.ndarray):
        """
        Single env step. actions_flat: (n_agents*2,) or (n_agents, 2) with [macro, target] per agent.
        Returns (obs_dict, reward, terminated, truncated, info).
        """
        a = np.asarray(actions_flat, dtype=np.int64)
        if a.ndim == 2:
            a = a.reshape(-1)
        if a.size < self.agents_per_team * 2:
            pad = np.zeros(self.agents_per_team * 2 - a.size, dtype=np.int64)
            a = np.concatenate([a, pad])
        t = torch.from_numpy(a).to(self._core.device).unsqueeze(0)
        obs, reward, term, trunc, infos = self._core.step(t)
        done = np.logical_or(term, trunc)
        if done.any():
            self._core.reset_indices(torch.from_numpy(done).to(self._core.device))
            obs = self._core.get_obs()
        self._refresh_agents()
        info = infos[0] if isinstance(infos, (list, tuple)) else infos
        return obs, float(reward[0]), bool(term[0]), bool(trunc[0]), info

    @property
    def macro_order(self) -> List[Any]:
        """Placeholder for QMIX; GPU uses fixed n_macros."""
        return list(range(self.n_macros))


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
                "vec": spaces.Box(low=-1.0, high=1.0, shape=(self._n_blue, VEC_OBS_DIM), dtype=np.float32),
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
        flat = np.asarray(self._pending_actions, dtype=np.int64).reshape(-1)
        n_exp = int(self.core.B * self.core.Nb * 2)
        if int(flat.size) != n_exp:
            raise ValueError(
                f"GPUCTFVecEnv: expected {n_exp} action values (B={self.core.B}, Nb={self.core.Nb}), "
                f"got {int(flat.size)} from predict() shape={getattr(self._pending_actions, 'shape', None)}"
            )
        actions = torch.as_tensor(flat, dtype=torch.int64, device=self.core.device)
        obs, rew, term, trunc, infos = self.core.step(actions)
        done = np.logical_or(term, trunc)
        if done.any():
            reset_mask = torch.from_numpy(done).to(self.core.device)
            # MARL / CTDE: fixed-size global features at terminal timestep (before reset) for value bootstrap.
            gs_terminal = build_global_state_batch(self.core).detach().cpu().numpy().astype(np.float32)
            for i in np.where(done)[0]:
                infos[i] = dict(infos[i])
                tobs = {k: v[i].copy() for k, v in obs.items()}
                tobs["global_state"] = gs_terminal[i].copy()
                infos[i]["terminal_observation"] = tobs
                # So training callbacks (parse_episode_result) get a single episode_result dict.
                bs = int(infos[i].get("blue_score", 0))
                rs = int(infos[i].get("red_score", 0))
                okind = str(infos[i].get("opponent_kind", "scripted")).lower()
                okey = str(infos[i].get("opponent_key", "OP3") or "")
                # When red is a snapshot, pass key so callbacks get SNAPSHOT:name instead of SNAPSHOT:unknown
                osnap = okey if okind == "snapshot" else ""
                infos[i]["episode_result"] = {
                    "blue_score": bs,
                    "red_score": rs,
                    "success": 1 if bs > rs else 0,
                    "phase_name": str(infos[i].get("phase", "OP3")),
                    "opponent_kind": okind,
                    "opponent_snapshot": osnap,
                    "scripted_tag": okey if okind == "scripted" else "",
                    "species_tag": "BALANCED",
                    "collisions_per_episode": 0,
                    "collision_events_per_episode": 0,
                    "collision_free_episode": 1,
                    "near_misses_per_episode": 0,
                    "zone_coverage": 0.0,
                    "decision_steps": int(infos[i].get("decision_steps", 0)),
                    "vec_schema_version": 2,
                }
            self.core.reset_indices(reset_mask)
            obs = self.core.get_obs()
        self._pending_actions = None
        return obs, rew, done, infos

    def close(self) -> None:
        self._pending_actions = None

    def get_attr(self, attr_name: str, indices=None):
        idx = self._get_indices(indices)
        if attr_name == "render_mode":
            return [None for _ in idx]
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
        method_kwargs = dict(method_kwargs)
        try:
            sig = inspect.signature(method)
        except Exception:
            sig = None
        if sig is not None and "env_indices" in sig.parameters:
            method_kwargs.setdefault("env_indices", idx)
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
    "GPUEnvAdapter",
    "GPUCTFVecEnv",
    "GPUCTFSingleEnv",
    "CNN_COLS",
    "CNN_ROWS",
    "NUM_CNN_CHANNELS",
]
