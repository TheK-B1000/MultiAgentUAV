from __future__ import annotations
from typing import Any, Dict, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from game_field import GameField, CNN_ROWS, CNN_COLS, NUM_CNN_CHANNELS
from red_opponents import make_species_wrapper, make_snapshot_wrapper
from macro_actions import MacroAction


class CTFGameFieldSB3Env(gym.Env):
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
        # NEW: must match PPO gamma for PBRS to be invariant
        ppo_gamma: float = 0.995,
    ):
        super().__init__()
        self.make_game_field_fn = make_game_field_fn
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

        self._n_blue_agents = 2
        self._base_vec_per_agent = 12
        if self.include_high_level_mode:
            extra = 2 if self._high_level_mode_onehot else 1
        else:
            extra = 0
        self._vec_per_agent = int(self._base_vec_per_agent + extra)
        self._n_macros = 5
        self._n_targets = 8
        self._league_mode = False
        self._episode_reward_total = 0.0
        self._blue_role_macros = blue_role_macros
        self._episode_macro_counts: list[list[int]] = []
        self._episode_mine_counts: list[dict[str, int]] = []

        # Anti-stall (blue-only, decision-level)
        self._stall_threshold_cells = 0.15
        self._stall_patience = 3
        self._stall_penalty = -0.05
        self._stall_counters = [0, 0]
        self._last_blue_pos: list[Optional[Tuple[float, float]]] = [None, None]

        self.action_space = spaces.MultiDiscrete([self._n_macros, self._n_targets, self._n_macros, self._n_targets])

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
                shape=(2 * (self._n_macros + self._n_targets),),
                dtype=np.float32,
            )

        self.observation_space = spaces.Dict(obs_dict)

        # Opponent identity (for Elo + logging)
        self._opponent_kind = "scripted"
        self._opponent_snapshot_path: Optional[str] = None
        self._opponent_species_tag: str = "BALANCED"
        self._opponent_scripted_tag: str = "OP1"

    # -----------------------------
    # Phase / curriculum helpers
    # -----------------------------
    def set_phase(self, phase: str) -> None:
        self._phase_name = str(phase).upper().strip()
        if self.gf is not None and hasattr(self.gf, "manager"):
            try:
                self.gf.manager.set_phase(self._phase_name)
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
        self._next_opponent = (str(kind).upper(), str(key).upper())

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
        self._opponent_kind = "snapshot"
        self._opponent_snapshot_path = str(snapshot_path)
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
        self._decision_step_count = 0
        self._episode_reward_total = 0.0

        # Blue externally controlled by SB3; red internal + wrapper override
        self.gf.set_external_control("blue", True)
        self.gf.set_external_control("red", False)
        self.gf.external_missing_action_mode = "idle"
        self.gf.use_internal_policies = True

        if not self.gf.macro_targets:
            _ = self.gf.get_all_macro_targets()

        self._n_macros = int(self.gf.n_macros)
        self._n_targets = int(self.gf.num_macro_targets or 8)
        self.action_space = spaces.MultiDiscrete([self._n_macros, self._n_targets, self._n_macros, self._n_targets])
        self._episode_macro_counts = [[0 for _ in range(self._n_macros)] for _ in range(self._n_blue_agents)]
        self._episode_mine_counts = [{"grab": 0, "place": 0} for _ in range(self._n_blue_agents)]

        if self.include_mask_in_obs:
            self.observation_space = spaces.Dict({
                "grid": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self._n_blue_agents * NUM_CNN_CHANNELS, CNN_ROWS, CNN_COLS),
                    dtype=np.float32,
                ),
                "vec": spaces.Box(
                    low=-2.0,
                    high=2.0,
                    shape=(self._n_blue_agents * self._vec_per_agent,),
                    dtype=np.float32,
                ),
                "mask": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(2 * (self._n_macros + self._n_targets),),
                    dtype=np.float32,
                ),
            })

        self.gf.reset_default()

        self._stall_counters = [0, 0]
        self._last_blue_pos = [None, None]
        for i in range(2):
            if i < len(self.gf.blue_agents) and self.gf.blue_agents[i] is not None:
                self._last_blue_pos[i] = tuple(getattr(self.gf.blue_agents[i], "float_pos", (0.0, 0.0)))

        # IMPORTANT FIXES:
        # 1) bind env to manager so team routing is exact
        if hasattr(self.gf, "manager") and hasattr(self.gf.manager, "bind_game_field"):
            self.gf.manager.bind_game_field(self.gf)

        # 2) PBRS gamma must match PPO gamma (policy invariance)
        if hasattr(self.gf, "manager") and hasattr(self.gf.manager, "set_shaping_gamma"):
            self.gf.manager.set_shaping_gamma(self.ppo_gamma)

        # 3) Apply curriculum phase to manager
        if hasattr(self.gf, "manager") and hasattr(self.gf.manager, "set_phase"):
            try:
                self.gf.manager.set_phase(self._phase_name)
            except Exception:
                pass

        # Clear any stale reward events
        if hasattr(self.gf, "manager") and hasattr(self.gf.manager, "pop_reward_events"):
            _ = self.gf.manager.pop_reward_events()

        # Apply next-opponent override if requested; otherwise use defaults.
        if self._next_opponent is not None:
            kind, key = self._next_opponent
            if kind == "SCRIPTED":
                self.set_opponent_scripted(key)
            elif kind == "SPECIES":
                self.set_opponent_species(key)
            elif kind == "SNAPSHOT":
                self.set_opponent_snapshot(key)
            self._next_opponent = None
        else:
            if self.default_opponent_kind == "SCRIPTED":
                self.set_opponent_scripted(self.default_opponent_key)
            elif self.default_opponent_kind == "SPECIES":
                self.set_opponent_species(self.default_opponent_key)
            elif self.default_opponent_kind == "SNAPSHOT":
                self.set_opponent_snapshot(self.default_opponent_key)

        obs = self._get_obs()
        return obs, {}

    def step(self, action: np.ndarray):
        assert self.gf is not None, "Call reset() first"
        self._decision_step_count += 1

        b0_macro, b0_tgt, b1_macro, b1_tgt = map(int, np.asarray(action).reshape(-1).tolist())

        b0_macro, b0_tgt = self._sanitize_action_for_agent(0, b0_macro, b0_tgt)
        b1_macro, b1_tgt = self._sanitize_action_for_agent(1, b1_macro, b1_tgt)
        self._record_macro_usage(0, b0_macro)
        self._record_macro_usage(1, b1_macro)

        blue0 = self.gf.blue_agents[0] if len(self.gf.blue_agents) > 0 else None
        blue1 = self.gf.blue_agents[1] if len(self.gf.blue_agents) > 1 else None

        actions_by_agent: Dict[str, Tuple[int, Any]] = {}
        if blue0 is not None:
            actions_by_agent[str(getattr(blue0, "slot_id", getattr(blue0, "unique_id", "blue_0")))] = (b0_macro, b0_tgt)
        if blue1 is not None:
            actions_by_agent[str(getattr(blue1, "slot_id", getattr(blue1, "unique_id", "blue_1")))] = (b1_macro, b1_tgt)

        self.gf.submit_external_actions(actions_by_agent)

        interval = float(getattr(self.gf, "decision_interval_seconds", 0.7))
        dt = max(1e-3, 0.99 * interval)
        self.gf.update(dt)

        # CRITICAL FIX:
        # Your GameManager emits rewards as events; _read_reward_total() was always 0,
        # so PPO was learning from a near-zero signal.
        reward = float(self._consume_blue_reward_events())
        reward += float(self._apply_blue_stall_penalty())

        gm = self.gf.manager
        terminated = bool(getattr(gm, "game_over", False))
        truncated = (self._decision_step_count >= self.max_decision_steps)

        obs = self._get_obs()

        info: Dict[str, Any] = {}
        if terminated or truncated:
            if hasattr(gm, "terminal_outcome_bonus"):
                reward += gm.terminal_outcome_bonus(
                    int(getattr(gm, "blue_score", 0)),
                    int(getattr(gm, "red_score", 0)),
                )
            self._episode_reward_total += float(reward)
            info["episode_result"] = {
                "blue_score": int(getattr(gm, "blue_score", 0)),
                "red_score": int(getattr(gm, "red_score", 0)),
                "win_by": int(getattr(gm, "blue_score", 0)) - int(getattr(gm, "red_score", 0)),
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
            }
        else:
            self._episode_reward_total += float(reward)

        return obs, reward, terminated, truncated, info

    def _apply_blue_stall_penalty(self) -> float:
        assert self.gf is not None
        penalty = 0.0

        for i in range(2):
            if i >= len(self.gf.blue_agents):
                self._stall_counters[i] = 0
                self._last_blue_pos[i] = None
                continue

            agent = self.gf.blue_agents[i]
            if agent is None:
                self._stall_counters[i] = 0
                self._last_blue_pos[i] = None
                continue

            pos = tuple(getattr(agent, "float_pos", (float(getattr(agent, "x", 0)), float(getattr(agent, "y", 0)))))
            last = self._last_blue_pos[i]

            if last is None:
                moved = True
            else:
                dx = float(pos[0]) - float(last[0])
                dy = float(pos[1]) - float(last[1])
                moved = (dx * dx + dy * dy) >= (self._stall_threshold_cells ** 2)

            if moved:
                self._stall_counters[i] = 0
            else:
                self._stall_counters[i] += 1
                if self._stall_counters[i] >= int(self._stall_patience):
                    penalty += float(self._stall_penalty)

            self._last_blue_pos[i] = pos

        return float(penalty)

    # -----------------
    # Helpers
    # -----------------
    def _get_obs(self) -> Dict[str, np.ndarray]:
        assert self.gf is not None

        def _empty_cnn():
            return np.zeros((NUM_CNN_CHANNELS, CNN_ROWS, CNN_COLS), dtype=np.float32)

        def _empty_vec():
            return np.zeros((self._base_vec_per_agent,), dtype=np.float32)

        def _coerce_vec(vec: np.ndarray) -> np.ndarray:
            v = np.asarray(vec, dtype=np.float32).reshape(-1)
            if v.size == self._base_vec_per_agent:
                return v
            if v.size < self._base_vec_per_agent:
                return np.pad(v, (0, self._base_vec_per_agent - v.size), mode="constant")
            return v[: self._base_vec_per_agent]

        blue = list(self.gf.blue_agents) if getattr(self.gf, "blue_agents", None) else []
        while len(blue) < 2:
            blue.append(None)

        cnn_list = []
        vec_list = []
        mask_list = []

        for idx, a in enumerate(blue[:2]):
            if a is None:
                cnn_list.append(_empty_cnn())
                vec = _empty_vec()
                if self.include_high_level_mode:
                    if self._high_level_mode_onehot:
                        mode = max(0, min(1, int(self._high_level_mode)))
                        mode_vec = np.zeros((2,), dtype=np.float32)
                        mode_vec[mode] = 1.0
                    else:
                        mode_vec = np.asarray([float(self._high_level_mode)], dtype=np.float32)
                    vec = np.concatenate([vec, mode_vec], axis=0)
                vec_list.append(vec)
                if self.include_mask_in_obs:
                    mm = np.ones((self._n_macros,), dtype=np.float32)
                    tm = np.ones((self._n_targets,), dtype=np.float32)
                    mask_list.append(np.concatenate([mm, tm], axis=0))
                continue

            cnn = np.asarray(self.gf.build_observation(a), dtype=np.float32)
            cnn_list.append(cnn)

            if hasattr(self.gf, "build_continuous_features"):
                vec = _coerce_vec(self.gf.build_continuous_features(a))
            else:
                vec = _empty_vec()
            if self.include_high_level_mode:
                if self._high_level_mode_onehot:
                    mode = max(0, min(1, int(self._high_level_mode)))
                    mode_vec = np.zeros((2,), dtype=np.float32)
                    mode_vec[mode] = 1.0
                else:
                    mode_vec = np.asarray([float(self._high_level_mode)], dtype=np.float32)
                vec = np.concatenate([vec, mode_vec], axis=0)
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

        grid = np.concatenate(cnn_list, axis=0).astype(np.float32)
        vec = np.concatenate(vec_list, axis=0).astype(np.float32)

        out = {"grid": grid, "vec": vec}
        if self.include_mask_in_obs:
            out["mask"] = np.concatenate(mask_list, axis=0).astype(np.float32)
        return out

    def _sanitize_action_for_agent(self, blue_index: int, macro: int, tgt: int) -> Tuple[int, int]:
        macro = int(macro) % self._n_macros
        tgt = int(tgt) % max(1, self._n_targets)

        if not self.enforce_masks:
            return macro, tgt

        assert self.gf is not None
        if blue_index >= len(self.gf.blue_agents):
            return 0, 0

        agent = self.gf.blue_agents[blue_index]
        if agent is None:
            return 0, 0

        mask = np.asarray(self.gf.get_macro_mask(agent), dtype=np.bool_).reshape(-1)
        mask = self._apply_role_macro_mask(blue_index, mask)
        if mask.shape != (self._n_macros,) or (not mask.any()):
            return macro, tgt

        if not bool(mask[macro]):
            macro = 0  # fallback to GO_TO

        # Enforce target mask for macros that consume targets.
        if self._macro_uses_target(macro):
            tgt_mask = np.asarray(self.gf.get_target_mask(agent), dtype=np.bool_).reshape(-1)
            if tgt_mask.shape == (self._n_targets,) and tgt_mask.any():
                if not bool(tgt_mask[tgt]):
                    tgt = self._nearest_valid_target(agent, tgt_mask)
        return macro, tgt

    def _apply_role_macro_mask(self, blue_index: int, mm: np.ndarray) -> np.ndarray:
        if self._blue_role_macros is None:
            return mm
        if not isinstance(self._blue_role_macros, (tuple, list)) or blue_index >= len(self._blue_role_macros):
            return mm
        allowed = self._blue_role_macros[blue_index]
        if not allowed:
            return mm
        mask = np.zeros_like(mm, dtype=np.bool_)
        for idx in allowed:
            try:
                i = int(idx)
            except Exception:
                continue
            if 0 <= i < mask.size:
                mask[i] = True
        out = mm & mask
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
        names = []
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

    def _consume_blue_reward_events(self) -> float:
        """
        Sum reward events for the two blue agents this decision step.

        Assumptions:
          - GameManager.add_reward_event uses agent_id == agent.unique_id (or stable fallback)
          - Our action submission uses the same ids
        """
        assert self.gf is not None
        gm = self.gf.manager

        pop = getattr(gm, "pop_reward_events", None)
        if pop is None or (not callable(pop)):
            return 0.0

        events = pop()

        blue0 = self.gf.blue_agents[0] if len(self.gf.blue_agents) > 0 else None
        blue1 = self.gf.blue_agents[1] if len(self.gf.blue_agents) > 1 else None

        blue_ids = set()
        if blue0 is not None:
            blue_ids.add(str(getattr(blue0, "unique_id", getattr(blue0, "slot_id", "blue_0"))))
        if blue1 is not None:
            blue_ids.add(str(getattr(blue1, "unique_id", getattr(blue1, "slot_id", "blue_1"))))

        r = 0.0
        for _t, aid, val in events:
            if str(aid) in blue_ids:
                r += float(val)
        return float(r)
