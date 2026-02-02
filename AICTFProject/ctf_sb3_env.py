from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from game_field import GameField, CNN_ROWS, CNN_COLS, NUM_CNN_CHANNELS
from red_opponents import make_species_wrapper, make_snapshot_wrapper
from macro_actions import MacroAction


class CTFGameFieldSB3Env(gym.Env):
    """
    SB3-compatible wrapper around GameField for blue-controlled PPO (MultiDiscrete).

    Supports two modes (zero-shot scalability):
    - max_blue_agents=2 (default): legacy 2v2. Action length 4, obs stacked [2*C, H, W].
    - max_blue_agents>2: tokenized/set-based. Agents as sequence of tokens; train 2v2 (mask 2),
      test 4v4 or 8v8 (mask 4 or 8). Action length 2*max_blue_agents; obs grid [max_blue_agents, C, H, W],
      vec [max_blue_agents, vec_per_agent], agent_mask [max_blue_agents].

    Observation (Dict):
      - "grid": [max_blue_agents, NUM_CNN_CHANNELS, CNN_ROWS, CNN_COLS] (tokenized) or [2*C, H, W] (legacy)
      - "vec":  [max_blue_agents, vec_per_agent] (tokenized) or [2*vec_per_agent] (legacy)
      - "agent_mask": [max_blue_agents] 1/0 (tokenized only)
      - optional "mask": action mask (macro+target per agent)

    Notes:
      - Blue is externally controlled (SB3 provides actions)
      - Red runs internal policies, optionally overridden by wrapper (species/snapshot)
      - Rewards are consumed from GameManager reward events, filtered to blue agents
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        make_game_field_fn,  # () -> GameField
        *,
        max_decision_steps: int = 400,
        enforce_masks: bool = True,
        seed: int = 0,
        include_mask_in_obs: bool = False,
        include_high_level_mode: bool = False,
        high_level_mode: int = 0,
        high_level_mode_onehot: bool = True,
        blue_role_macros: Optional[Tuple[list[int], list[int]]] = None,
        default_opponent_kind: str = "SCRIPTED",
        default_opponent_key: str = "OP1",
        # Must match PPO gamma for PBRS invariance (if manager uses shaping)
        ppo_gamma: float = 0.995,
        # Action execution noise (reliability metric)
        action_flip_prob: float = 0.0,
        # Zero-shot scalability: max blue agents; tokenized obs/action when > 2
        max_blue_agents: int = 2,
        # Verification: print obs/action shapes once per reset (Sprint A)
        print_reset_shapes: bool = False,
    ):
        super().__init__()
        self.make_game_field_fn = make_game_field_fn
        self._print_reset_shapes = bool(print_reset_shapes)

        self.max_decision_steps = int(max_decision_steps)
        self.enforce_masks = bool(enforce_masks)

        self.base_seed = int(seed)
        self.include_mask_in_obs = bool(include_mask_in_obs)

        self.include_high_level_mode = bool(include_high_level_mode)
        self._high_level_mode = int(high_level_mode)
        self._high_level_mode_onehot = bool(high_level_mode_onehot)

        self.ppo_gamma = float(ppo_gamma)

        self.default_opponent_kind = str(default_opponent_kind).upper()
        self.default_opponent_key = str(default_opponent_key).upper()

        self.gf: Optional[GameField] = None
        self._decision_step_count = 0
        self._next_opponent: Optional[Tuple[str, str]] = None
        self._phase_name: str = "OP1"

        # Store dynamics config even before reset() (SubprocVecEnv calls env_method early sometimes)
        self._dynamics_config: Optional[dict] = None
        self._disturbance_config: Optional[dict] = None
        self._robotics_config: Optional[dict] = None
        self._sensor_config: Optional[dict] = None
        self._physics_tag: Optional[str] = None

        # max_blue_agents: 2 = legacy 2v2; >2 = tokenized for zero-shot (train 2v2, test 4v4/8v8)
        self._max_blue_agents = max(2, int(max_blue_agents))
        self._n_blue_agents = min(2, self._max_blue_agents)  # actual count set at reset from options

        # Must match GameField.build_continuous_features() default
        self._base_vec_per_agent = 12
        extra = 0
        if self.include_high_level_mode:
            extra = 2 if self._high_level_mode_onehot else 1
        self._vec_per_agent = int(self._base_vec_per_agent + extra)

        # Will be synced from GameField on reset()
        self._n_macros = 5
        self._n_targets = 8

        self._league_mode = False
        self._episode_reward_total = 0.0
        self._blue_role_macros = blue_role_macros
        # Curriculum Axis 2: environment stress by phase (optional)
        self._stress_schedule: Optional[dict] = None  # phase -> {current_strength_cps, action_delay_steps, sensor_noise_sigma_cells, sensor_dropout_prob}
        self._episode_macro_counts: list[list[int]] = []
        self._episode_mine_counts: list[dict[str, int]] = []

        # Anti-stall (blue-only, decision-level)
        self._stall_threshold_cells = 0.15
        self._stall_patience = 3
        self._stall_penalty = -0.05
        self._stall_counters = [0, 0]
        self._last_blue_pos: list[Optional[Tuple[float, float]]] = [None, None]

        # Action space: (macro, target) per blue agent; tokenized uses max_blue_agents
        n_act = self._max_blue_agents if self._max_blue_agents > 2 else 2
        self.action_space = spaces.MultiDiscrete(
            [self._n_macros, self._n_targets] * n_act
        )

        # Observation space: tokenized (max_blue_agents, ...) when > 2, else legacy stacked
        if self._max_blue_agents > 2:
            grid_shape = (self._max_blue_agents, NUM_CNN_CHANNELS, CNN_ROWS, CNN_COLS)
            vec_shape = (self._max_blue_agents, self._vec_per_agent)
            obs_dict = {
                "grid": spaces.Box(low=0.0, high=1.0, shape=grid_shape, dtype=np.float32),
                "vec": spaces.Box(low=-2.0, high=2.0, shape=vec_shape, dtype=np.float32),
                "agent_mask": spaces.Box(low=0.0, high=1.0, shape=(self._max_blue_agents,), dtype=np.float32),
            }
            if self.include_mask_in_obs:
                obs_dict["mask"] = spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self._max_blue_agents * (self._n_macros + self._n_targets),),
                    dtype=np.float32,
                )
        else:
            grid_shape = (self._n_blue_agents * NUM_CNN_CHANNELS, CNN_ROWS, CNN_COLS)
            vec_shape = (self._n_blue_agents * self._vec_per_agent,)
            obs_dict = {
                "grid": spaces.Box(low=0.0, high=1.0, shape=grid_shape, dtype=np.float32),
                "vec": spaces.Box(low=-2.0, high=2.0, shape=vec_shape, dtype=np.float32),
            }
            if self.include_mask_in_obs:
                obs_dict["mask"] = spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self._n_blue_agents * (self._n_macros + self._n_targets),),
                    dtype=np.float32,
                )
        self.observation_space = spaces.Dict(obs_dict)

        # Opponent identity (for Elo + logging)
        self._opponent_kind = "scripted"  # "scripted" | "species" | "snapshot"
        self._opponent_snapshot_path: Optional[str] = None
        self._opponent_species_tag: str = "BALANCED"
        self._opponent_scripted_tag: str = "OP1"

        # Action execution noise (reliability tracking)
        self.action_flip_prob = float(max(0.0, min(1.0, action_flip_prob)))
        self._action_rng = np.random.RandomState(int(seed) + 1000)  # Separate RNG for action noise

    # -----------------------------
    # Dynamics config hook (SB3 SubprocVecEnv expects this if you env_method it)
    # -----------------------------
    def set_dynamics_config(self, cfg: Optional[dict]) -> None:
        """
        Called from training via VecEnv.env_method("set_dynamics_config", cfg).

        We store cfg and forward it to GameField/Manager hooks if present.
        This prevents SubprocVecEnv workers from crashing when training code calls this.
        """
        self._dynamics_config = None if cfg is None else dict(cfg)

        if self.gf is None:
            return

        # Prefer GF-level hook
        if hasattr(self.gf, "set_dynamics_config"):
            try:
                self.gf.set_dynamics_config(self._dynamics_config)
                return
            except Exception:
                pass

        # Fall back to Manager hook
        gm = getattr(self.gf, "manager", None)
        if gm is not None and hasattr(gm, "set_dynamics_config"):
            try:
                gm.set_dynamics_config(self._dynamics_config)
            except Exception:
                pass

    def get_dynamics_config(self) -> Optional[dict]:
        """Convenience accessor for logging/debug."""
        return None if self._dynamics_config is None else dict(self._dynamics_config)

    # -----------------------------
    # Disturbances / robotics / sensing hooks
    # -----------------------------
    def set_disturbance_config(self, *args, **kwargs) -> None:
        """
        Called from training via VecEnv.env_method("set_disturbance_config", cfg)
        or via kwargs: current_strength / drift_sigma.
        """
        cfg = None
        if args and isinstance(args[0], dict):
            cfg = dict(args[0])
        elif "cfg" in kwargs and isinstance(kwargs["cfg"], dict):
            cfg = dict(kwargs["cfg"])

        if cfg is None:
            cfg = dict(kwargs)

        # Support both training names and GameField names
        current_strength = cfg.pop("current_strength", None)
        drift_sigma = cfg.pop("drift_sigma", None)
        current_strength_cps = cfg.pop("current_strength_cps", None)
        drift_sigma_cells = cfg.pop("drift_sigma_cells", None)

        if current_strength_cps is None:
            current_strength_cps = current_strength
        if drift_sigma_cells is None:
            drift_sigma_cells = drift_sigma

        self._disturbance_config = {
            "current_strength_cps": 0.0 if current_strength_cps is None else float(current_strength_cps),
            "drift_sigma_cells": 0.0 if drift_sigma_cells is None else float(drift_sigma_cells),
        }

        if self.gf is None:
            return

        if hasattr(self.gf, "set_disturbance_config"):
            try:
                self.gf.set_disturbance_config(
                    self._disturbance_config["current_strength_cps"],
                    self._disturbance_config["drift_sigma_cells"],
                )
                return
            except Exception:
                pass

        if hasattr(self.gf, "set_disturbance_config_dict"):
            try:
                self.gf.set_disturbance_config_dict(self._disturbance_config)
            except Exception:
                pass

    def get_disturbance_config(self) -> Optional[dict]:
        return None if self._disturbance_config is None else dict(self._disturbance_config)

    def set_robotics_constraints(self, *args, **kwargs) -> None:
        """
        Called from training via VecEnv.env_method("set_robotics_constraints", cfg)
        or via kwargs: action_delay_steps / actuation_noise_sigma.
        """
        cfg = None
        if args and isinstance(args[0], dict):
            cfg = dict(args[0])
        elif "cfg" in kwargs and isinstance(kwargs["cfg"], dict):
            cfg = dict(kwargs["cfg"])

        if cfg is None:
            cfg = dict(kwargs)

        action_delay_steps = cfg.get("action_delay_steps", 0)
        actuation_noise_sigma = cfg.get("actuation_noise_sigma", 0.0)

        self._robotics_config = {
            "action_delay_steps": int(action_delay_steps),
            "actuation_noise_sigma": float(actuation_noise_sigma),
        }

        if self.gf is None:
            return

        if hasattr(self.gf, "set_robotics_constraints"):
            try:
                self.gf.set_robotics_constraints(
                    self._robotics_config["action_delay_steps"],
                    self._robotics_config["actuation_noise_sigma"],
                )
                return
            except Exception:
                pass

        if hasattr(self.gf, "set_robotics_constraints_dict"):
            try:
                self.gf.set_robotics_constraints_dict(self._robotics_config)
            except Exception:
                pass

    def get_robotics_constraints(self) -> Optional[dict]:
        return None if self._robotics_config is None else dict(self._robotics_config)

    def set_sensor_config(self, *args, **kwargs) -> None:
        """
        Called from training via VecEnv.env_method("set_sensor_config", cfg)
        or via kwargs: sensor_range / sensor_noise_sigma / sensor_dropout_prob.
        """
        cfg = None
        if args and isinstance(args[0], dict):
            cfg = dict(args[0])
        elif "cfg" in kwargs and isinstance(kwargs["cfg"], dict):
            cfg = dict(kwargs["cfg"])

        if cfg is None:
            cfg = dict(kwargs)

        sensor_range = cfg.get("sensor_range", cfg.get("sensor_range_cells", 9999.0))
        sensor_noise_sigma = cfg.get("sensor_noise_sigma", cfg.get("sensor_noise_sigma_cells", 0.0))
        sensor_dropout_prob = cfg.get("sensor_dropout_prob", 0.0)

        self._sensor_config = {
            "sensor_range_cells": float(sensor_range),
            "sensor_noise_sigma_cells": float(sensor_noise_sigma),
            "sensor_dropout_prob": float(sensor_dropout_prob),
        }

        if self.gf is None:
            return

        if hasattr(self.gf, "set_sensor_config"):
            try:
                self.gf.set_sensor_config(
                    self._sensor_config["sensor_range_cells"],
                    self._sensor_config["sensor_noise_sigma_cells"],
                    self._sensor_config["sensor_dropout_prob"],
                )
                return
            except Exception:
                pass

        if hasattr(self.gf, "set_sensor_config_dict"):
            try:
                self.gf.set_sensor_config_dict(self._sensor_config)
            except Exception:
                pass

    def get_sensor_config(self) -> Optional[dict]:
        return None if self._sensor_config is None else dict(self._sensor_config)

    def set_physics_tag(self, tag: str) -> None:
        self._physics_tag = str(tag)
        if self.gf is None:
            return
        if hasattr(self.gf, "set_physics_tag"):
            try:
                self.gf.set_physics_tag(self._physics_tag)
            except Exception:
                pass

    def set_physics_enabled(self, enabled: bool) -> None:
        """Turn ASV kinematics + maritime sensors on/off. Used by realism-by-phase curriculum."""
        if self.gf is None:
            return
        if hasattr(self.gf, "set_physics_enabled"):
            try:
                self.gf.set_physics_enabled(bool(enabled))
            except Exception:
                pass

    # -----------------------------
    # Phase / curriculum helpers
    # -----------------------------
    def set_phase(self, phase: str) -> None:
        # Contract: stress schedule keys are OP1|OP2|OP3 only; normalize tags (e.g. OP3_HARD -> OP3)
        from rl.curriculum import phase_from_tag, VALID_PHASES
        raw = str(phase).upper().strip()
        canonical = phase_from_tag(raw)
        assert canonical in VALID_PHASES, f"phase_from_tag({phase!r}) returned {canonical!r}"
        self._phase_name = canonical
        if self.gf is not None and hasattr(self.gf, "manager"):
            try:
                self.gf.manager.set_phase(self._phase_name)
            except Exception:
                pass
        self._apply_stress_for_phase(self._phase_name)

    def set_stress_schedule(self, schedule: Optional[dict]) -> None:
        """Curriculum Axis 2: phase -> {current_strength_cps, drift_sigma_cells, action_delay_steps, sensor_noise_sigma_cells, sensor_dropout_prob}. None = disable."""
        self._stress_schedule = None if schedule is None else dict(schedule)

    def _apply_stress_for_phase(self, phase: str) -> None:
        """Apply environment stress and naval realism for this phase if stress schedule is set."""
        if self.gf is None or not self._stress_schedule:
            return
        cfg = self._stress_schedule.get(str(phase).upper())
        if not cfg or not isinstance(cfg, dict):
            return
        try:
            if "physics_enabled" in cfg:
                self.set_physics_enabled(bool(cfg.get("physics_enabled", False)))
            if cfg.get("relaxed_dynamics"):
                # Gentler ASV params when first enabling physics (OP2)
                self.gf.set_dynamics_config(
                    max_speed_cps=float(cfg.get("max_speed_cps", 2.8)),
                    max_accel_cps2=float(cfg.get("max_accel_cps2", 2.5)),
                    max_yaw_rate_rps=float(cfg.get("max_yaw_rate_rps", 5.0)),
                )
            elif "max_speed_cps" in cfg or "max_accel_cps2" in cfg or "max_yaw_rate_rps" in cfg:
                self.gf.set_dynamics_config(
                    max_speed_cps=float(cfg.get("max_speed_cps", getattr(self.gf.boat_cfg, "max_speed_cps", 2.2))),
                    max_accel_cps2=float(cfg.get("max_accel_cps2", getattr(self.gf.boat_cfg, "max_accel_cps2", 2.0))),
                    max_yaw_rate_rps=float(cfg.get("max_yaw_rate_rps", getattr(self.gf.boat_cfg, "max_yaw_rate_rps", 4.0))),
                )
            if "current_strength_cps" in cfg or "drift_sigma_cells" in cfg:
                self.gf.set_disturbance_config(
                    float(cfg.get("current_strength_cps", 0.0)),
                    float(cfg.get("drift_sigma_cells", 0.0)),
                )
            if "action_delay_steps" in cfg:
                self.gf.set_robotics_constraints(
                    int(cfg.get("action_delay_steps", 0)),
                    float(cfg.get("actuation_noise_sigma", 0.0)),
                )
            if "sensor_noise_sigma_cells" in cfg or "sensor_dropout_prob" in cfg:
                self.gf.set_sensor_config(
                    float(cfg.get("sensor_range_cells", getattr(self.gf.boat_cfg, "sensor_range_cells", 9999.0))),
                    float(cfg.get("sensor_noise_sigma_cells", 0.0)),
                    float(cfg.get("sensor_dropout_prob", 0.0)),
                )
        except Exception:
            pass

    def set_league_mode(self, league_mode: bool) -> None:
        self._league_mode = bool(league_mode)

    def set_high_level_mode(self, mode: int) -> None:
        self._high_level_mode = int(mode)

    # -----------------------------
    # Opponent hot-swap
    # -----------------------------
    def set_next_opponent(self, kind: str, key: str) -> None:
        k = str(kind).upper()
        # Snapshot key is a file path: do not uppercase (breaks paths on case-sensitive / Windows)
        v = str(key).upper() if k != "SNAPSHOT" else str(key)
        self._next_opponent = (k, v)

    def set_opponent_scripted(self, scripted_tag: str) -> None:
        if self.gf is None:
            return
        tag = str(scripted_tag).upper()
        self._opponent_kind = "scripted"
        self._opponent_scripted_tag = tag
        self._opponent_species_tag = "BALANCED"
        self._opponent_snapshot_path = None

        self.gf.set_red_opponent(tag)
        self.gf.set_red_policy_wrapper(None)

    def set_opponent_species(self, species_tag: str) -> None:
        if self.gf is None:
            return
        self._opponent_kind = "species"
        self._opponent_species_tag = str(species_tag).upper()
        self._opponent_snapshot_path = None
        self._opponent_scripted_tag = "OP3"

        wrapper = make_species_wrapper(self._opponent_species_tag)
        self.gf.set_red_policy_wrapper(wrapper)

    def set_opponent_snapshot(self, snapshot_path: str) -> None:
        if self.gf is None:
            return
        path = str(snapshot_path).strip()
        # Resolve path: try as-is, with .zip, lowercase (Windows), normpath
        candidates = [path]
        if not path.lower().endswith(".zip"):
            candidates.append(path + ".zip")
        candidates.append(os.path.normpath(path))
        candidates.append(os.path.normpath(path + ("" if path.lower().endswith(".zip") else ".zip")))
        if path != path.lower():
            candidates.append(path.lower())
            candidates.append(os.path.normpath(path.lower()))
        found = None
        for p in candidates:
            if p and os.path.isfile(p):
                found = p
                break
        if found is None:
            # Snapshot file missing (deleted, wrong cwd, or bad path): fall back to scripted OP3
            try:
                print(f"[WARN] Snapshot not found: {path!r}; falling back to SCRIPTED:OP3")
            except Exception:
                pass
            self.set_opponent_scripted("OP3")
            return
        self._opponent_kind = "snapshot"
        self._opponent_snapshot_path = found
        self._opponent_species_tag = "BALANCED"
        self._opponent_scripted_tag = "OP3"

        wrapper = make_snapshot_wrapper(self._opponent_snapshot_path)
        self.gf.set_red_policy_wrapper(wrapper)

    # -------------
    # Gym API
    # -------------
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        final_seed = self.base_seed if seed is None else int(seed)
        np.random.seed(final_seed)

        self.gf = self.make_game_field_fn()

        # n_agents: 2 = legacy 2v2; 4/8 for zero-shot (train 2v2, test 4v4/8v8)
        n_agents_option = 2
        if options and isinstance(options.get("n_agents"), (int, float)):
            n_agents_option = min(self._max_blue_agents, max(1, int(options["n_agents"])))
        try:
            self.gf.agents_per_team = n_agents_option
        except Exception:
            pass

        self._decision_step_count = 0
        self._episode_reward_total = 0.0

        # Blue externally controlled by SB3; red internal + wrapper override
        self.gf.set_external_control("blue", True)
        self.gf.set_external_control("red", False)
        self.gf.external_missing_action_mode = "idle"
        self.gf.use_internal_policies = True

        # IMPORTANT: Initialize macro targets before reading masks/targets if needed
        if not getattr(self.gf, "macro_targets", None):
            try:
                _ = self.gf.get_all_macro_targets()
            except Exception:
                pass

        # Apply opponent and OpponentParams BEFORE reset_default so red spawn uses speed_mult etc.
        if self._next_opponent is not None:
            kind, key = self._next_opponent
            kind = str(kind).upper()
            key = str(key) if kind == "SNAPSHOT" else str(key).upper()
            self._next_opponent = None
        else:
            kind = str(self.default_opponent_kind).upper()
            key = str(self.default_opponent_key).upper()
        if kind == "SCRIPTED":
            self.set_opponent_scripted(key)
        elif kind == "SPECIES":
            self.set_opponent_species(key)
        elif kind == "SNAPSHOT":
            self.set_opponent_snapshot(key)
        try:
            from opponent_params import sample_opponent_params
            phase = str(getattr(self.gf.manager, "phase_name", self._phase_name)).upper()
            rng = __import__("random").Random(int(final_seed) + 1)
            params = sample_opponent_params(kind=kind, key=key, phase=phase, rng=rng)
            if hasattr(self.gf, "set_opponent_params"):
                self.gf.set_opponent_params(params)
        except Exception:
            pass

        # Reset the underlying sim (spawn uses red_speed_scale etc. from OpponentParams)
        self.gf.reset_default()

        # Re-apply sticky dynamics after GF is created/reset
        if self._dynamics_config is not None:
            self.set_dynamics_config(self._dynamics_config)
        if self._disturbance_config is not None:
            self.set_disturbance_config(self._disturbance_config)
        if self._robotics_config is not None:
            self.set_robotics_constraints(self._robotics_config)
        if self._sensor_config is not None:
            self.set_sensor_config(self._sensor_config)
        if self._physics_tag is not None:
            self.set_physics_tag(self._physics_tag)

        # Apply stress/realism for current phase (OP1 no physics, OP2 relaxed, OP3 full)
        self._apply_stress_for_phase(self._phase_name)

        # Sync macro/target dims from GameField (Sprint C: n_targets fixed for all team sizes)
        self._n_macros = int(getattr(self.gf, "n_macros", 5))
        self._n_targets = int(getattr(self.gf, "num_macro_targets", 8) or 8)

        # Actual blue count this episode (for tokenized: mask padding)
        self._n_blue_agents = min(len(getattr(self.gf, "blue_agents", []) or []), self._max_blue_agents)
        if self._n_blue_agents <= 0:
            self._n_blue_agents = min(2, self._max_blue_agents)

        n_act = self._max_blue_agents if self._max_blue_agents > 2 else 2
        self.action_space = spaces.MultiDiscrete(
            [self._n_macros, self._n_targets] * n_act
        )

        if self._max_blue_agents > 2:
            base_obs = {
                "grid": spaces.Box(
                    low=0.0, high=1.0,
                    shape=(self._max_blue_agents, NUM_CNN_CHANNELS, CNN_ROWS, CNN_COLS),
                    dtype=np.float32,
                ),
                "vec": spaces.Box(
                    low=-2.0, high=2.0,
                    shape=(self._max_blue_agents, self._vec_per_agent),
                    dtype=np.float32,
                ),
                "agent_mask": spaces.Box(low=0.0, high=1.0, shape=(self._max_blue_agents,), dtype=np.float32),
            }
            if self.include_mask_in_obs:
                base_obs["mask"] = spaces.Box(
                    low=0.0, high=1.0,
                    shape=(self._max_blue_agents * (self._n_macros + self._n_targets),),
                    dtype=np.float32,
                )
        else:
            base_obs = {
                "grid": spaces.Box(
                    low=0.0, high=1.0,
                    shape=(self._n_blue_agents * NUM_CNN_CHANNELS, CNN_ROWS, CNN_COLS),
                    dtype=np.float32,
                ),
                "vec": spaces.Box(
                    low=-2.0, high=2.0,
                    shape=(self._n_blue_agents * self._vec_per_agent,),
                    dtype=np.float32,
                ),
            }
            if self.include_mask_in_obs:
                base_obs["mask"] = spaces.Box(
                    low=0.0, high=1.0,
                    shape=(self._n_blue_agents * (self._n_macros + self._n_targets),),
                    dtype=np.float32,
                )
        self.observation_space = spaces.Dict(base_obs)

        n_count = self._max_blue_agents if self._max_blue_agents > 2 else self._n_blue_agents
        self._episode_macro_counts = [[0 for _ in range(self._n_macros)] for _ in range(n_count)]
        self._episode_mine_counts = [{"grab": 0, "place": 0} for _ in range(n_count)]

        self._stall_counters = [0] * n_count
        self._last_blue_pos = [None] * n_count
        for i in range(n_count):
            if i < len(self.gf.blue_agents) and self.gf.blue_agents[i] is not None:
                self._last_blue_pos[i] = tuple(getattr(self.gf.blue_agents[i], "float_pos", (0.0, 0.0)))

        # Bind GF to manager (reward routing hook)
        if hasattr(self.gf, "manager") and hasattr(self.gf.manager, "bind_game_field"):
            try:
                self.gf.manager.bind_game_field(self.gf)
            except Exception:
                pass

        # PBRS gamma must match PPO gamma
        if hasattr(self.gf, "manager") and hasattr(self.gf.manager, "set_shaping_gamma"):
            try:
                self.gf.manager.set_shaping_gamma(self.ppo_gamma)
            except Exception:
                pass

        # Apply curriculum phase
        if hasattr(self.gf, "manager") and hasattr(self.gf.manager, "set_phase"):
            try:
                self.gf.manager.set_phase(self._phase_name)
            except Exception:
                pass

        # Clear any stale reward events
        if hasattr(self.gf, "manager") and hasattr(self.gf.manager, "pop_reward_events"):
            try:
                _ = self.gf.manager.pop_reward_events()
            except Exception:
                pass

        obs = self._get_obs()
        if self._print_reset_shapes:
            try:
                mask_shape = obs["mask"].shape if "mask" in obs else None
                am_shape = obs["agent_mask"].shape if "agent_mask" in obs else None
                nvec = getattr(self.action_space, "nvec", None)
                print(
                    "[CTFEnv] reset shapes: grid=%s vec=%s mask=%s agent_mask=%s action_space.nvec=%s"
                    % (obs["grid"].shape, obs["vec"].shape, mask_shape, am_shape, nvec)
                )
            except Exception:
                pass
        return obs, {}

    def step(self, action: np.ndarray):
        assert self.gf is not None, "Call reset() first"
        self._decision_step_count += 1

        act_flat = np.asarray(action).reshape(-1)
        n_slots = self._max_blue_agents if self._max_blue_agents > 2 else 2
        # Parse (macro, target) per slot; only use first _n_blue_agents
        intended: list[Tuple[int, int]] = []
        for i in range(n_slots):
            off = i * 2
            if off + 1 < len(act_flat):
                intended.append((int(act_flat[off]) % max(1, self._n_macros), int(act_flat[off + 1]) % max(1, self._n_targets)))
            else:
                intended.append((0, 0))
        while len(intended) < n_slots:
            intended.append((0, 0))

        # Apply action execution noise per agent
        executed: list[Tuple[int, int]] = []
        flip_count_step = 0
        macro_flip_count_step = 0
        target_flip_count_step = 0
        for i in range(min(self._n_blue_agents, n_slots)):
            m, t = intended[i]
            if self.action_flip_prob > 0.0:
                if self._action_rng.rand() < self.action_flip_prob:
                    m = int(self._action_rng.randint(0, self._n_macros))
                    flip_count_step += 1
                    macro_flip_count_step += 1
                if self._action_rng.rand() < self.action_flip_prob:
                    t = int(self._action_rng.randint(0, self._n_targets))
                    flip_count_step += 1
                    target_flip_count_step += 1
            m, t = self._sanitize_action_for_agent(i, m, t)
            executed.append((m, t))
            self._record_macro_usage(i, m)
        while len(executed) < n_slots:
            executed.append((0, 0))

        actions_by_agent: Dict[str, Tuple[int, Any]] = {}
        blue_list = getattr(self.gf, "blue_agents", []) or []
        for i in range(min(self._n_blue_agents, len(blue_list))):
            agent = blue_list[i] if i < len(blue_list) else None
            if agent is None:
                continue
            key = str(getattr(agent, "slot_id", getattr(agent, "unique_id", f"blue_{i}")))
            actions_by_agent[key] = executed[i]

        self.gf.submit_external_actions(actions_by_agent)

        # Advance sim ~one decision interval (slightly less to avoid edge jitter)
        interval = float(getattr(self.gf, "decision_interval_seconds", 0.7))
        dt = max(1e-3, 0.99 * interval)
        self.gf.update(dt)

        # Rewards via GameManager events (filtered to blue IDs)
        reward = float(self._consume_blue_reward_events())
        reward += float(self._apply_blue_stall_penalty())

        gm = self.gf.manager
        terminated = bool(getattr(gm, "game_over", False))
        truncated = (self._decision_step_count >= self.max_decision_steps)

        obs = self._get_obs()

        info: Dict[str, Any] = {}
        # Add noise metrics to info every step (for callback tracking)
        info["flip_count_step"] = int(flip_count_step)
        info["macro_flip_count_step"] = int(macro_flip_count_step)
        info["target_flip_count_step"] = int(target_flip_count_step)
        info["num_agents"] = int(self._n_blue_agents)
        info["action_components"] = 2  # macro + target per agent
        info["phase"] = str(self._phase_name)

        if terminated or truncated:
            # Optional terminal shaping/bonus
            if hasattr(gm, "terminal_outcome_bonus"):
                try:
                    reward += float(
                        gm.terminal_outcome_bonus(
                            int(getattr(gm, "blue_score", 0)),
                            int(getattr(gm, "red_score", 0)),
                        )
                    )
                except Exception:
                    pass

            self._episode_reward_total += float(reward)

            blue_score = int(getattr(gm, "blue_score", 0))
            red_score = int(getattr(gm, "red_score", 0))
            # IROS-style Top 5 metrics (publish-friendly)
            success = 1 if blue_score > red_score else 0
            time_to_first = getattr(gm, "time_to_first_score", None)
            time_to_first_score = float(time_to_first) if time_to_first is not None else None
            time_to_game_over = getattr(gm, "time_to_game_over", None)
            time_to_game_over_sec = float(time_to_game_over) if time_to_game_over is not None else None
            if time_to_game_over_sec is None:
                time_to_game_over_sec = float(getattr(gm, "sim_time", 0.0))
            collisions = int(getattr(gm, "collision_count_this_episode", 0))
            near_misses = int(getattr(gm, "near_miss_count_this_episode", 0))
            collision_free = 1 if collisions == 0 else 0
            dists = getattr(gm, "blue_inter_robot_distances", []) or []
            mean_inter_robot_dist = float(np.mean(dists)) if dists else None
            std_inter_robot_dist = float(np.std(dists)) if len(dists) > 1 else (0.0 if dists else None)
            visited = getattr(gm, "blue_zone_visited_cells", set()) or set()
            total_zone = int(getattr(gm, "total_blue_zone_cells", 1)) or 1
            zone_coverage = float(len(visited)) / float(total_zone) if total_zone else 0.0

            info["episode_result"] = {
                "blue_score": blue_score,
                "red_score": red_score,
                "win_by": blue_score - red_score,
                "phase_name": str(getattr(gm, "phase_name", self._phase_name)),
                "league_mode": bool(self._league_mode),
                "blue_rewards_total": float(self._episode_reward_total),
                "opponent_kind": self._opponent_kind,
                "opponent_snapshot": self._opponent_snapshot_path,
                "species_tag": self._opponent_species_tag if self._opponent_kind == "species" else None,
                "scripted_tag": self._opponent_scripted_tag if self._opponent_kind == "scripted" else None,
                "decision_steps": self._decision_step_count,
                "macro_order": self._macro_order_names(),
                "macro_counts": self._episode_macro_counts,
                "mine_counts": self._episode_mine_counts,
                "blue_mine_kills": int(getattr(gm, "blue_mine_kills_this_episode", 0)),
                "mines_placed_enemy_half": int(getattr(gm, "mines_placed_in_enemy_half_this_episode", 0)),
                "mines_triggered_by_red": int(getattr(gm, "mines_triggered_by_red_this_episode", 0)),
                "dynamics_config": self.get_dynamics_config(),
                # Top 5 IROS-style metrics (CSV / Excel)
                "success": success,
                "time_to_first_score": time_to_first_score,
                "time_to_game_over": time_to_game_over_sec,
                "collisions_per_episode": collisions,
                "near_misses_per_episode": near_misses,
                "collision_free_episode": collision_free,
                "mean_inter_robot_dist": mean_inter_robot_dist,
                "std_inter_robot_dist": std_inter_robot_dist,
                "zone_coverage": zone_coverage,
            }
        else:
            self._episode_reward_total += float(reward)

        return obs, float(reward), terminated, truncated, info

    # -----------------
    # Observation helpers
    # -----------------
    def _get_obs(self) -> Dict[str, np.ndarray]:
        assert self.gf is not None

        def _empty_cnn() -> np.ndarray:
            return np.zeros((NUM_CNN_CHANNELS, CNN_ROWS, CNN_COLS), dtype=np.float32)

        def _empty_vec_base() -> np.ndarray:
            return np.zeros((self._base_vec_per_agent,), dtype=np.float32)

        def _coerce_vec_base(vec: np.ndarray) -> np.ndarray:
            v = np.asarray(vec, dtype=np.float32).reshape(-1)
            if v.size == self._base_vec_per_agent:
                return v
            if v.size < self._base_vec_per_agent:
                return np.pad(v, (0, self._base_vec_per_agent - v.size), mode="constant")
            return v[: self._base_vec_per_agent]

        blue = list(getattr(self.gf, "blue_agents", []) or [])
        n_slots = self._max_blue_agents if self._max_blue_agents > 2 else self._n_blue_agents
        while len(blue) < n_slots:
            blue.append(None)

        cnn_list: list[np.ndarray] = []
        vec_list: list[np.ndarray] = []
        mask_list: list[np.ndarray] = []

        for idx, a in enumerate(blue[:n_slots]):
            if a is None:
                cnn_list.append(_empty_cnn())
                vec = self._append_high_level_mode(_empty_vec_base())
                vec_list.append(vec)
                if self.include_mask_in_obs:
                    # Mask out all actions for padding slots (invalid agent)
                    mm = np.zeros((self._n_macros,), dtype=np.float32)
                    tm = np.zeros((self._n_targets,), dtype=np.float32)
                    mask_list.append(np.concatenate([mm, tm], axis=0))
                continue

            cnn = np.asarray(self.gf.build_observation(a), dtype=np.float32)
            cnn_list.append(cnn)
            if hasattr(self.gf, "build_continuous_features"):
                vec = _coerce_vec_base(self.gf.build_continuous_features(a))
            else:
                vec = _empty_vec_base()
            vec = self._append_high_level_mode(vec)
            vec_list.append(vec)
            if self.include_mask_in_obs:
                mm = np.asarray(self.gf.get_macro_mask(a), dtype=np.bool_).reshape(-1)
                mm = self._apply_role_macro_mask(idx, mm)
                if mm.shape != (self._n_macros,) or (not mm.any()):
                    mm = np.ones((self._n_macros,), dtype=np.bool_)
                tm = np.asarray(self.gf.get_target_mask(a), dtype=np.bool_).reshape(-1)
                if tm.shape != (self._n_targets,) or (not tm.any()):
                    tm = np.ones((self._n_targets,), dtype=np.bool_)
                mask_list.append(np.concatenate([mm.astype(np.float32), tm.astype(np.float32)], axis=0))

        if self._max_blue_agents > 2:
            # Tokenized: (max_blue_agents, C, H, W), (max_blue_agents, vec_dim), agent_mask
            grid = np.stack(cnn_list, axis=0).astype(np.float32)
            vec = np.stack(vec_list, axis=0).astype(np.float32)
            agent_mask = np.zeros((n_slots,), dtype=np.float32)
            agent_mask[: self._n_blue_agents] = 1.0
            out = {"grid": grid, "vec": vec, "agent_mask": agent_mask}
            if self.include_mask_in_obs:
                out["mask"] = np.concatenate(mask_list, axis=0).astype(np.float32)
        else:
            grid = np.concatenate(cnn_list, axis=0).astype(np.float32)
            vec = np.concatenate(vec_list, axis=0).astype(np.float32)
            out = {"grid": grid, "vec": vec}
            if self.include_mask_in_obs:
                out["mask"] = np.concatenate(mask_list, axis=0).astype(np.float32)
        return out

    def _append_high_level_mode(self, base_vec: np.ndarray) -> np.ndarray:
        v = np.asarray(base_vec, dtype=np.float32).reshape(-1)
        if not self.include_high_level_mode:
            return v

        if self._high_level_mode_onehot:
            mode = max(0, min(1, int(self._high_level_mode)))
            mode_vec = np.zeros((2,), dtype=np.float32)
            mode_vec[mode] = 1.0
        else:
            mode_vec = np.asarray([float(self._high_level_mode)], dtype=np.float32)

        return np.concatenate([v, mode_vec], axis=0)

    # -----------------
    # Action sanitization / masks
    # -----------------
    def _sanitize_action_for_agent(self, blue_index: int, macro: int, tgt: int) -> Tuple[int, int]:
        macro = int(macro) % max(1, self._n_macros)
        tgt = int(tgt) % max(1, self._n_targets)

        if not self.enforce_masks:
            return macro, tgt

        assert self.gf is not None
        if blue_index >= len(self.gf.blue_agents):
            return 0, 0

        agent = self.gf.blue_agents[blue_index]
        if agent is None:
            return 0, 0

        mm = np.asarray(self.gf.get_macro_mask(agent), dtype=np.bool_).reshape(-1)
        mm = self._apply_role_macro_mask(blue_index, mm)
        if mm.shape != (self._n_macros,) or (not mm.any()):
            return macro, tgt

        if not bool(mm[macro]):
            macro = 0  # GO_TO fallback

        if self._macro_uses_target(macro):
            tm = np.asarray(self.gf.get_target_mask(agent), dtype=np.bool_).reshape(-1)
            if tm.shape == (self._n_targets,) and tm.any():
                if not bool(tm[tgt]):
                    tgt = self._nearest_valid_target(agent, tm)

        return macro, tgt

    def _apply_role_macro_mask(self, blue_index: int, mm: np.ndarray) -> np.ndarray:
        if self._blue_role_macros is None:
            return mm
        if not isinstance(self._blue_role_macros, (tuple, list)):
            return mm
        if blue_index >= len(self._blue_role_macros):
            return mm

        allowed = self._blue_role_macros[blue_index]
        if not allowed:
            return mm

        role_mask = np.zeros_like(mm, dtype=np.bool_)
        for idx in allowed:
            try:
                i = int(idx)
            except Exception:
                continue
            if 0 <= i < role_mask.size:
                role_mask[i] = True

        out = mm & role_mask
        return out if out.any() else mm

    def _macro_uses_target(self, macro_idx: int) -> bool:
        if self.gf is None:
            return True
        try:
            action = self.gf.macro_order[int(macro_idx)]
        except Exception:
            return True
        return action in (MacroAction.GO_TO, MacroAction.PLACE_MINE)

    def _macro_order_names(self) -> list[str]:
        if self.gf is None:
            return []
        names: list[str] = []
        try:
            for m in self.gf.macro_order:
                n = getattr(m, "name", None)
                names.append(str(n) if n is not None else str(m))
        except Exception:
            pass
        return names

    def _record_macro_usage(self, blue_index: int, macro_idx: int) -> None:
        if blue_index >= self._n_blue_agents:
            return
        if not self._episode_macro_counts:
            return
        if 0 <= macro_idx < len(self._episode_macro_counts[blue_index]):
            self._episode_macro_counts[blue_index][macro_idx] += 1

        if not self._episode_mine_counts or blue_index >= len(self._episode_mine_counts):
            return

        action = None
        if self.gf is not None:
            try:
                action = self.gf.macro_order[int(macro_idx)]
            except Exception:
                action = None

        if action is None:
            try:
                action = MacroAction(int(macro_idx))
            except Exception:
                action = None

        if action == MacroAction.GRAB_MINE:
            self._episode_mine_counts[blue_index]["grab"] += 1
        elif action == MacroAction.PLACE_MINE:
            self._episode_mine_counts[blue_index]["place"] += 1

    def _nearest_valid_target(self, agent: Any, tgt_mask: np.ndarray) -> int:
        valid = np.flatnonzero(tgt_mask)
        if valid.size == 0:
            return 0
        assert self.gf is not None

        try:
            ax = float(getattr(agent, "x", 0.0))
            ay = float(getattr(agent, "y", 0.0))
            best_idx = int(valid[0])
            best_d2 = None
            for i in valid:
                tx, ty = self.gf.get_macro_target(int(i))
                d2 = (float(tx) - ax) ** 2 + (float(ty) - ay) ** 2
                if best_d2 is None or d2 < best_d2:
                    best_d2 = d2
                    best_idx = int(i)
            return best_idx
        except Exception:
            return int(valid[0])

    # -----------------
    # Reward + anti-stall
    # -----------------
    def _consume_blue_reward_events(self) -> float:
        assert self.gf is not None
        gm = self.gf.manager

        pop = getattr(gm, "pop_reward_events", None)
        if pop is None or (not callable(pop)):
            return 0.0

        try:
            events = pop()
        except Exception:
            return 0.0

        blue_ids = set()
        for i in range(min(self._n_blue_agents, len(self.gf.blue_agents))):
            agent = self.gf.blue_agents[i]
            if agent is None:
                continue
            blue_ids.add(str(getattr(agent, "unique_id", getattr(agent, "slot_id", f"blue_{i}"))))
            blue_ids.add(str(getattr(agent, "slot_id", getattr(agent, "unique_id", f"blue_{i}"))))

        r = 0.0
        for ev in events:
            try:
                _t, aid, val = ev
            except Exception:
                continue
            if str(aid) in blue_ids:
                r += float(val)
        return float(r)

    def _apply_blue_stall_penalty(self) -> float:
        assert self.gf is not None
        penalty = 0.0

        for i in range(self._n_blue_agents):
            if i >= len(self.gf.blue_agents):
                self._stall_counters[i] = 0
                self._last_blue_pos[i] = None
                continue

            agent = self.gf.blue_agents[i]
            if agent is None:
                self._stall_counters[i] = 0
                self._last_blue_pos[i] = None
                continue

            pos = tuple(
                getattr(
                    agent,
                    "float_pos",
                    (float(getattr(agent, "x", 0)), float(getattr(agent, "y", 0))),
                )
            )
            last = self._last_blue_pos[i]

            if last is None:
                moved = True
            else:
                dx = float(pos[0]) - float(last[0])
                dy = float(pos[1]) - float(last[1])
                moved = (dx * dx + dy * dy) >= (float(self._stall_threshold_cells) ** 2)

            if moved:
                self._stall_counters[i] = 0
            else:
                self._stall_counters[i] += 1
                if self._stall_counters[i] >= int(self._stall_patience):
                    penalty += float(self._stall_penalty)

            self._last_blue_pos[i] = pos

        return float(penalty)

    # -------------
    # Optional Gym hooks
    # -------------
    def render(self):
        return None

    def close(self):
        self.gf = None


__all__ = ["CTFGameFieldSB3Env"]
