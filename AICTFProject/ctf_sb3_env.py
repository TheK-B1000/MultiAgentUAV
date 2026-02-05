from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from game_field import GameField, CNN_ROWS, CNN_COLS, NUM_CNN_CHANNELS
from rl.agent_identity import build_blue_identities, build_reward_id_map
from rl.env_modules import (
    EnvConfigManager,
    EnvOpponentManager,
    EnvRewardManager,
    EnvObsBuilder,
    EnvActionManager,
    StepReward,
    StepInfo,
    EpisodeResult,
    StepResult,
    MAX_OPPONENT_CONTEXT_IDS,
    validate_reward_breakdown,
    validate_obs_order,
    validate_action_keys,
    validate_dropped_reward_events_policy,
)


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
        # Step 2.1: if False, assert GameField has build_continuous_features(); if True, fallback to zeros with warning
        # NOTE: In eval mode, this should be False to avoid hidden nondeterminism
        allow_missing_vec_features: bool = False,
        # Reward contract: TEAM_SUM (default), PER_AGENT, SHAPLEY_APPROX (future). Logged in info.
        reward_mode: str = "TEAM_SUM",
        # Step 3.1: use canonical ObsBuilder (legacy path removed - always True)
        use_obs_builder: bool = True,
        # Step 4.1: add obs["context"] = opponent embedding id (stable int, not full path). Default off.
        include_opponent_context: bool = False,
        # Step 5.1: when True, GameField.build_continuous_features asserts no global state in vec (debug only).
        obs_debug_validate_locality: bool = False,
        # Phase 2: Contract validation (MARL safety hardening)
        validate_contracts: bool = False,
    ):
        super().__init__()
        self.make_game_field_fn = make_game_field_fn
        self._print_reset_shapes = bool(print_reset_shapes)
        self._allow_missing_vec_features = bool(allow_missing_vec_features)
        # Log if fallback is enabled (could create hidden nondeterminism)
        if self._allow_missing_vec_features:
            import warnings
            warnings.warn(
                "allow_missing_vec_features=True: fallback to zeros may create hidden nondeterminism. "
                "Disable in evaluation mode.",
                UserWarning,
            )
        self._reward_mode = str(reward_mode).strip().upper() or "TEAM_SUM"
        # Always use obs builder - legacy path removed per module ownership
        if not use_obs_builder:
            import warnings
            warnings.warn(
                "use_obs_builder=False is deprecated. Legacy observation path removed per module ownership. "
                "Always using canonical build_team_obs().",
                DeprecationWarning,
            )
        self._use_obs_builder = True  # Always True
        self._include_opponent_context = bool(include_opponent_context)
        self._obs_debug_validate_locality = bool(obs_debug_validate_locality)
        self._validate_contracts = bool(validate_contracts)

        self.max_decision_steps = int(max_decision_steps)
        self.enforce_masks = bool(enforce_masks)

        self.base_seed = int(seed)
        self.include_mask_in_obs = bool(include_mask_in_obs)

        self.include_high_level_mode = bool(include_high_level_mode)
        self._high_level_mode = int(high_level_mode)
        self._high_level_mode_onehot = bool(high_level_mode_onehot)

        self.ppo_gamma = float(ppo_gamma)

        self.gf: Optional[GameField] = None
        self._decision_step_count = 0

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

        # Canonical agent identity tracking
        self._blue_identities: list = []  # List[AgentIdentity] - built at reset
        self._reward_id_map: Dict[str, int] = {}  # internal_id -> slot_index

        # Initialize modules
        self._config_manager = EnvConfigManager()
        self._opponent_manager = EnvOpponentManager(
            default_kind=default_opponent_kind,
            default_key=default_opponent_key,
        )
        self._reward_manager = EnvRewardManager()
        self._obs_builder = EnvObsBuilder(
            use_obs_builder=self._use_obs_builder,
            include_mask_in_obs=self.include_mask_in_obs,
            include_opponent_context=self._include_opponent_context,
            include_high_level_mode=self.include_high_level_mode,
            high_level_mode=self._high_level_mode,
            high_level_mode_onehot=self._high_level_mode_onehot,
            base_vec_per_agent=self._base_vec_per_agent,
            obs_debug_validate_locality=self._obs_debug_validate_locality,
        )
        self._action_manager = EnvActionManager(
            enforce_masks=self.enforce_masks,
            action_flip_prob=action_flip_prob,
            n_macros=self._n_macros,
            n_targets=self._n_targets,
            blue_role_macros=blue_role_macros,
            seed=seed,
        )

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
        if self._include_opponent_context:
            obs_dict["context"] = spaces.Box(
                low=0.0, high=float(MAX_OPPONENT_CONTEXT_IDS - 1),
                shape=(1,), dtype=np.float32,
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
        """Delegate to EnvConfigManager."""
        self._config_manager.set_dynamics_config(cfg, self.gf)

    def get_dynamics_config(self) -> Optional[dict]:
        """Convenience accessor for logging/debug."""
        return self._config_manager.get_dynamics_config()

    # -----------------------------
    # Disturbances / robotics / sensing hooks
    # -----------------------------
    def set_disturbance_config(self, *args, **kwargs) -> None:
        """Delegate to EnvConfigManager."""
        self._config_manager.set_disturbance_config(*args, game_field=self.gf, **kwargs)

    def get_disturbance_config(self) -> Optional[dict]:
        return self._config_manager.get_disturbance_config()

    def set_robotics_constraints(self, *args, **kwargs) -> None:
        """Delegate to EnvConfigManager."""
        self._config_manager.set_robotics_constraints(*args, **kwargs, game_field=self.gf)

    def get_robotics_constraints(self) -> Optional[dict]:
        return self._config_manager.get_robotics_constraints()

    def set_sensor_config(self, *args, **kwargs) -> None:
        """Delegate to EnvConfigManager."""
        self._config_manager.set_sensor_config(*args, game_field=self.gf, **kwargs)

    def get_sensor_config(self) -> Optional[dict]:
        return self._config_manager.get_sensor_config()

    def set_physics_tag(self, tag: str) -> None:
        """Delegate to EnvConfigManager."""
        self._config_manager.set_physics_tag(tag, self.gf)

    def set_physics_enabled(self, enabled: bool) -> None:
        """Turn ASV kinematics + maritime sensors on/off. Used by realism-by-phase curriculum."""
        self._config_manager.set_physics_enabled(enabled, self.gf)

    # -----------------------------
    # Phase / curriculum helpers
    # -----------------------------
    def set_phase(self, phase: str) -> None:
        """Delegate to EnvConfigManager."""
        self._config_manager.set_phase(phase, self.gf)

    def set_stress_schedule(self, schedule: Optional[dict]) -> None:
        """Curriculum Axis 2: phase -> {current_strength_cps, drift_sigma_cells, action_delay_steps, sensor_noise_sigma_cells, sensor_dropout_prob}. None = disable."""
        self._config_manager.set_stress_schedule(schedule)

    def set_league_mode(self, league_mode: bool) -> None:
        self._league_mode = bool(league_mode)

    def set_high_level_mode(self, mode: int) -> None:
        """Update high-level mode for next observation."""
        self._high_level_mode = int(mode)
        self._obs_builder.set_high_level_mode(mode)

    # -----------------------------
    # Opponent hot-swap
    # -----------------------------
    def set_next_opponent(self, kind: str, key: str) -> None:
        """Delegate to EnvOpponentManager."""
        self._opponent_manager.set_next_opponent(kind, key)

    def set_opponent_scripted(self, scripted_tag: str) -> None:
        """Delegate to EnvOpponentManager."""
        self._opponent_manager.set_opponent_scripted(scripted_tag, self.gf)

    def set_opponent_species(self, species_tag: str) -> None:
        """Delegate to EnvOpponentManager."""
        self._opponent_manager.set_opponent_species(species_tag, self.gf)

    def set_opponent_snapshot(self, snapshot_path: str) -> None:
        """Delegate to EnvOpponentManager."""
        self._opponent_manager.set_opponent_snapshot(snapshot_path, self.gf)

    # -------------
    # Gym API
    # -------------
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        final_seed = self.base_seed if seed is None else int(seed)
        np.random.seed(final_seed)

        self.gf = self.make_game_field_fn()
        self.gf.obs_debug_validate_locality = self._obs_debug_validate_locality

        # n_agents: 2 = legacy 2v2; 4/8 for zero-shot (train 2v2, test 4v4/8v8)
        n_agents_option = 2
        if options and isinstance(options.get("n_agents"), (int, float)):
            n_agents_option = min(self._max_blue_agents, max(1, int(options["n_agents"])))
        try:
            self.gf.agents_per_team = n_agents_option
        except Exception:
            pass

        self._decision_step_count = 0

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
        phase_name = self._config_manager.get_phase()
        self._opponent_manager.apply_opponent_at_reset(self.gf, phase_name, final_seed)

        # Reset the underlying sim (spawn uses red_speed_scale etc. from OpponentParams)
        self.gf.reset_default()

        # Re-apply all configs after GF is created/reset
        self._config_manager.apply_all_configs(self.gf)

        # Sync macro/target dims from GameField (Sprint C: n_targets fixed for all team sizes)
        self._n_macros = int(getattr(self.gf, "n_macros", 5))
        self._n_targets = int(getattr(self.gf, "num_macro_targets", 8) or 8)

        # Actual blue count this episode (for tokenized: mask padding)
        self._n_blue_agents = min(len(getattr(self.gf, "blue_agents", []) or []), self._max_blue_agents)
        if self._n_blue_agents <= 0:
            self._n_blue_agents = min(2, self._max_blue_agents)

        # Build canonical agent identities (blue_0, blue_1, ...)
        blue_agents_list = getattr(self.gf, "blue_agents", []) or []
        max_for_identity = self._max_blue_agents if self._max_blue_agents > 2 else self._n_blue_agents
        self._blue_identities = build_blue_identities(blue_agents_list, max_agents=max_for_identity)
        self._reward_id_map = build_reward_id_map(self._blue_identities)
        self._dropped_reward_events_this_step = 0
        self._dropped_reward_events_this_episode = 0
        
        # Break symmetry with role tokens: assign each agent a role (attacker/defender/escort)
        # Roles are randomized each episode to prevent specialization collapse
        n_roles = 3  # attacker, defender, escort
        self._agent_roles = []
        for i in range(max_for_identity):
            role_idx = np.random.randint(0, n_roles) if i < len(blue_agents_list) else 0
            self._agent_roles.append(role_idx)
        # Store in obs builder for vec_append_fn
        if hasattr(self._obs_builder, "set_agent_roles"):
            self._obs_builder.set_agent_roles(self._agent_roles)

        # Reset episode-level tracking in managers
        self._reward_manager.reset_episode(n_agents=self._n_blue_agents)
        self._action_manager.reset_episode(n_agents=self._n_blue_agents)

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
            if self._include_opponent_context:
                base_obs["context"] = spaces.Box(
                    low=0.0, high=float(MAX_OPPONENT_CONTEXT_IDS - 1),
                    shape=(1,), dtype=np.float32,
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
            if self._include_opponent_context:
                base_obs["context"] = spaces.Box(
                    low=0.0, high=float(MAX_OPPONENT_CONTEXT_IDS - 1),
                    shape=(1,), dtype=np.float32,
                )
        self.observation_space = spaces.Dict(base_obs)

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
        phase_name = self._config_manager.get_phase()
        if hasattr(self.gf, "manager") and hasattr(self.gf.manager, "set_phase"):
            try:
                self.gf.manager.set_phase(phase_name)
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

        # Parse actions
        act_flat = np.asarray(action).reshape(-1)
        n_slots = self._max_blue_agents if self._max_blue_agents > 2 else 2
        intended: list[Tuple[int, int]] = []
        for i in range(n_slots):
            off = i * 2
            if off + 1 < len(act_flat):
                intended.append((int(act_flat[off]) % max(1, self._n_macros), int(act_flat[off + 1]) % max(1, self._n_targets)))
            else:
                intended.append((0, 0))
        while len(intended) < n_slots:
            intended.append((0, 0))

        # Apply noise and sanitize via EnvActionManager
        executed, flip_count_step, macro_flip_count_step, target_flip_count_step = (
            self._action_manager.apply_noise_and_sanitize(
                intended,
                self._n_blue_agents,
                self.gf,
                role_macro_mask_fn=self._action_manager._apply_role_macro_mask,
            )
        )

        # Submit actions using canonical keys
        actions_by_agent: Dict[str, Tuple[int, Any]] = {}
        for i, ident in enumerate(self._blue_identities):
            if i < len(executed):
                actions_by_agent[ident.key] = executed[i]
        self.gf.submit_external_actions(actions_by_agent)

        # Advance sim
        interval = float(getattr(self.gf, "decision_interval_seconds", 0.7))
        dt = max(1e-3, 0.99 * interval)
        self.gf.update(dt)

        # Rewards via EnvRewardManager
        gm = self.gf.manager
        reward_events_total, reward_per_agent_events, dropped_count = (
            self._reward_manager.consume_reward_events(
                gm,
                self._reward_id_map,
                len(self._blue_identities),
            )
        )
        stall_total, stall_per_agent = self._reward_manager.apply_stall_penalty(
            self.gf,
            self._n_blue_agents,
        )
        n_slots = len(self._blue_identities)
        reward_blue_per_agent = [
            float(reward_per_agent_events[i]) + float(stall_per_agent[i] if i < len(stall_per_agent) else 0.0)
            for i in range(n_slots)
        ]

        # Check termination
        gm = self.gf.manager
        terminated = bool(getattr(gm, "game_over", False))
        truncated = (self._decision_step_count >= self.max_decision_steps)

        # Build observation via EnvObsBuilder
        obs = self._get_obs()
        
        # Contract validation: observation order
        if self._validate_contracts:
            validate_obs_order(obs, self._n_blue_agents, self._max_blue_agents, debug=True)

        # Build StepReward
        reward_breakdown = StepReward(
            team_total=float(reward_events_total) + float(stall_total),
            per_agent=reward_blue_per_agent,
            reward_events_total=float(reward_events_total),
            stall_penalty_total=float(stall_total),
        )
        
        # Contract validation: reward breakdown
        if self._validate_contracts:
            validate_reward_breakdown(
                reward_breakdown.team_total,
                reward_breakdown.per_agent,
                debug=True,
            )
        
        # Contract validation: dropped reward events policy
        if self._validate_contracts:
            # Get total events from GameManager (before consumption)
            pop_fn = getattr(gm, "pop_reward_events", None)
            total_events_estimate = 0
            if pop_fn is not None and callable(pop_fn):
                try:
                    # Note: events were already consumed, so we use dropped_count as lower bound
                    # In practice, we'd need to track total before consumption for accurate validation
                    # For now, validate that dropped_count is reasonable (not excessive)
                    if dropped_count > 0:
                        # If we dropped events, assume at least that many total
                        total_events_estimate = max(dropped_count, 1)
                        validate_dropped_reward_events_policy(
                            dropped_count,
                            total_events_estimate,
                            debug=True,
                        )
                except Exception:
                    pass

        # Build StepInfo
        phase_name = self._config_manager.get_phase()
        step_info = StepInfo(
            reward_mode=str(self._reward_mode),
            reward_blue_per_agent=reward_blue_per_agent,
            reward_blue_team=float(sum(reward_blue_per_agent)),
            flip_count_step=int(flip_count_step),
            macro_flip_count_step=int(macro_flip_count_step),
            target_flip_count_step=int(target_flip_count_step),
            num_agents=int(self._n_blue_agents),
            action_components=2,
            phase=str(phase_name),
            dropped_reward_events=int(self._reward_manager.get_dropped_events_step()),
        )

        # Build EpisodeResult if terminal
        episode_result: Optional[EpisodeResult] = None
        if terminated or truncated:
            terminal_bonus = 0.0
            if hasattr(gm, "terminal_outcome_bonus"):
                try:
                    terminal_bonus = float(
                        gm.terminal_outcome_bonus(
                            int(getattr(gm, "blue_score", 0)),
                            int(getattr(gm, "red_score", 0)),
                        )
                    )
                except Exception:
                    pass
            reward_breakdown.terminal_bonus = terminal_bonus
            self._reward_manager.add_episode_reward(reward_breakdown.team_total + terminal_bonus)

            blue_score = int(getattr(gm, "blue_score", 0))
            red_score = int(getattr(gm, "red_score", 0))
            success = 1 if blue_score > red_score else 0
            time_to_first = getattr(gm, "time_to_first_score", None)
            time_to_first_score = float(time_to_first) if time_to_first is not None else None
            time_to_game_over = getattr(gm, "time_to_game_over", None)
            time_to_game_over_sec = float(time_to_game_over) if time_to_game_over is not None else None
            if time_to_game_over_sec is None:
                time_to_game_over_sec = float(getattr(gm, "sim_time", 0.0))
            collisions_per_tick = int(getattr(gm, "collision_count_this_episode", 0))
            collision_events = int(getattr(gm, "collision_events_this_episode", 0))
            near_misses = int(getattr(gm, "near_miss_count_this_episode", 0))
            collision_free = 1 if collision_events == 0 else 0
            dists = getattr(gm, "blue_inter_robot_distances", []) or []
            mean_inter_robot_dist = float(np.mean(dists)) if dists else None
            std_inter_robot_dist = float(np.std(dists)) if len(dists) > 1 else (0.0 if dists else None)
            visited = getattr(gm, "blue_zone_visited_cells", set()) or set()
            total_zone = int(getattr(gm, "total_blue_zone_cells", 1)) or 1
            zone_coverage = float(len(visited)) / float(total_zone) if total_zone else 0.0

            episode_result = EpisodeResult(
                blue_score=blue_score,
                red_score=red_score,
                win_by=blue_score - red_score,
                phase_name=str(getattr(gm, "phase_name", phase_name)),
                league_mode=bool(self._league_mode),
                blue_rewards_total=float(self._reward_manager.get_episode_reward_total()),
                opponent_kind=self._opponent_manager.get_opponent_kind(),
                opponent_snapshot=self._opponent_manager.get_opponent_snapshot_path(),
                species_tag=self._opponent_manager.get_opponent_species_tag() if self._opponent_manager.get_opponent_kind() == "species" else None,
                scripted_tag=self._opponent_manager.get_opponent_scripted_tag() if self._opponent_manager.get_opponent_kind() == "scripted" else None,
                decision_steps=self._decision_step_count,
                macro_order=self._action_manager.get_macro_order_names(self.gf),
                macro_counts=self._action_manager.get_episode_macro_counts(),
                mine_counts=self._action_manager.get_episode_mine_counts(),
                blue_mine_kills=int(getattr(gm, "blue_mine_kills_this_episode", 0)),
                mines_placed_enemy_half=int(getattr(gm, "mines_placed_in_enemy_half_this_episode", 0)),
                mines_triggered_by_red=int(getattr(gm, "mines_triggered_by_red_this_episode", 0)),
                dynamics_config=self.get_dynamics_config(),
                success=success,
                time_to_first_score=time_to_first_score,
                time_to_game_over=time_to_game_over_sec,
                collisions_per_episode=collisions_per_tick,
                collision_events_per_episode=collision_events,
                near_misses_per_episode=near_misses,
                collision_free_episode=collision_free,
                mean_inter_robot_dist=mean_inter_robot_dist,
                std_inter_robot_dist=std_inter_robot_dist,
                zone_coverage=zone_coverage,
                dropped_reward_events=int(self._reward_manager.get_dropped_events_episode()),
                vec_schema_version=int(getattr(self.gf, "VEC_SCHEMA_VERSION", 1)),
            )
            self._reward_manager.reset_dropped_events_episode()
        else:
            self._reward_manager.add_episode_reward(reward_breakdown.team_total)

        # Build StepResult and convert to gym tuple
        step_result = StepResult.from_components(
            observation=obs,
            reward_breakdown=reward_breakdown,
            step_info=step_info,
            terminated=terminated,
            truncated=truncated,
            episode_result=episode_result,
        )
        return step_result.to_gym_tuple()

    # -----------------
    # Observation helpers
    # -----------------
    def _get_obs(self) -> Dict[str, np.ndarray]:
        """Build observation using EnvObsBuilder."""
        assert self.gf is not None
        opponent_context_id = None
        if self._include_opponent_context:
            # Access opponent context ID via a helper method
            opponent_context_id = self._opponent_manager._opponent_context_id()
        return self._obs_builder.build_observation(
            self.gf,
            self._blue_identities,
            max_blue_agents=self._max_blue_agents,
            n_blue_agents=self._n_blue_agents,
            n_macros=self._n_macros,
            n_targets=self._n_targets,
            opponent_context_id=opponent_context_id,
            role_macro_mask_fn=self._action_manager._apply_role_macro_mask,
        )

    # -----------------
    # Helper methods (delegated to modules)
    # -----------------
    # All action sanitization, reward routing, and observation building
    # are now handled by EnvActionManager, EnvRewardManager, and EnvObsBuilder modules.

    # -------------
    # Optional Gym hooks
    # -------------
    def render(self):
        return None

    def close(self):
        self.gf = None


__all__ = ["CTFGameFieldSB3Env"]
