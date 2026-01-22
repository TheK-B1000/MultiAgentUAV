# =========================
# ctf_sb3_env.py
#   UPDATED WITH RECOMMENDED FIXES:
#     A) Set PBRS gamma to match PPO gamma (or your training gamma)
#     B) Read rewards from event buffer (NOT a nonexistent "reward_total" attr)
#        -> your current _read_reward_total() always returns 0, so reward is always 0
#     C) Expose opponent metadata in info (kept)
# =========================

from __future__ import annotations
from typing import Any, Dict, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from game_field import GameField, CNN_ROWS, CNN_COLS, NUM_CNN_CHANNELS
from red_opponents import make_species_wrapper, make_snapshot_wrapper


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
        self.ppo_gamma = float(ppo_gamma)
        self.default_opponent_kind = str(default_opponent_kind).upper()
        self.default_opponent_key = str(default_opponent_key).upper()

        self.gf: Optional[GameField] = None
        self._decision_step_count = 0
        self._next_opponent: Optional[Tuple[str, str]] = None

        self._n_blue_agents = 2
        self._vec_per_agent = 4
        self._n_macros = 5
        self._n_targets = 8

        self.action_space = spaces.MultiDiscrete([self._n_macros, self._n_targets, self._n_macros, self._n_targets])

        grid_shape = (self._n_blue_agents * NUM_CNN_CHANNELS, CNN_ROWS, CNN_COLS)
        vec_shape = (self._n_blue_agents * self._vec_per_agent,)

        obs_dict = {
            "grid": spaces.Box(low=0.0, high=1.0, shape=grid_shape, dtype=np.float32),
            "vec": spaces.Box(low=-2.0, high=2.0, shape=vec_shape, dtype=np.float32),
        }
        if self.include_mask_in_obs:
            obs_dict["mask"] = spaces.Box(low=0.0, high=1.0, shape=(2 * self._n_macros,), dtype=np.float32)

        self.observation_space = spaces.Dict(obs_dict)

        # Opponent identity (for Elo + logging)
        self._opponent_kind = "scripted"
        self._opponent_snapshot_path: Optional[str] = None
        self._opponent_species_tag: str = "BALANCED"
        self._opponent_scripted_tag: str = "OP1"

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
                "mask": spaces.Box(low=0.0, high=1.0, shape=(2 * self._n_macros,), dtype=np.float32),
            })

        self.gf.reset_default()

        # IMPORTANT FIXES:
        # 1) bind env to manager so team routing is exact
        if hasattr(self.gf, "manager") and hasattr(self.gf.manager, "bind_game_field"):
            self.gf.manager.bind_game_field(self.gf)

        # 2) PBRS gamma must match PPO gamma (policy invariance)
        if hasattr(self.gf, "manager") and hasattr(self.gf.manager, "set_shaping_gamma"):
            self.gf.manager.set_shaping_gamma(self.ppo_gamma)

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

        blue0 = self.gf.blue_agents[0] if len(self.gf.blue_agents) > 0 else None
        blue1 = self.gf.blue_agents[1] if len(self.gf.blue_agents) > 1 else None

        actions_by_agent: Dict[str, Tuple[int, Any]] = {}
        if blue0 is not None:
            actions_by_agent[str(getattr(blue0, "unique_id", getattr(blue0, "slot_id", "blue_0")))] = (b0_macro, b0_tgt)
        if blue1 is not None:
            actions_by_agent[str(getattr(blue1, "unique_id", getattr(blue1, "slot_id", "blue_1")))] = (b1_macro, b1_tgt)

        self.gf.submit_external_actions(actions_by_agent)

        interval = float(getattr(self.gf, "decision_interval_seconds", 0.7))
        dt = max(1e-3, 0.99 * interval)
        self.gf.update(dt)

        # CRITICAL FIX:
        # Your GameManager emits rewards as events; _read_reward_total() was always 0,
        # so PPO was learning from a near-zero signal.
        reward = float(self._consume_blue_reward_events())

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
            info["episode_result"] = {
                "blue_score": int(getattr(gm, "blue_score", 0)),
                "red_score": int(getattr(gm, "red_score", 0)),
                "opponent_kind": self._opponent_kind,
                "opponent_snapshot": self._opponent_snapshot_path,
                "species_tag": self._opponent_species_tag if self._opponent_kind == "species" else None,
                "scripted_tag": self._opponent_scripted_tag if self._opponent_kind == "scripted" else None,
                "decision_steps": self._decision_step_count,
            }

        return obs, reward, terminated, truncated, info

    # -----------------
    # Helpers
    # -----------------
    def _get_obs(self) -> Dict[str, np.ndarray]:
        assert self.gf is not None

        def _empty_cnn():
            return np.zeros((NUM_CNN_CHANNELS, CNN_ROWS, CNN_COLS), dtype=np.float32)

        def _empty_vec():
            return np.zeros((self._vec_per_agent,), dtype=np.float32)

        def _coerce_vec(vec: np.ndarray) -> np.ndarray:
            v = np.asarray(vec, dtype=np.float32).reshape(-1)
            if v.size == self._vec_per_agent:
                return v
            if v.size < self._vec_per_agent:
                return np.pad(v, (0, self._vec_per_agent - v.size), mode="constant")
            return v[: self._vec_per_agent]

        blue = list(self.gf.blue_agents) if getattr(self.gf, "blue_agents", None) else []
        while len(blue) < 2:
            blue.append(None)

        cnn_list = []
        vec_list = []
        mask_list = []

        for a in blue[:2]:
            if a is None:
                cnn_list.append(_empty_cnn())
                vec_list.append(_empty_vec())
                if self.include_mask_in_obs:
                    mask_list.append(np.ones((self._n_macros,), dtype=np.float32))
                continue

            cnn = np.asarray(self.gf.build_observation(a), dtype=np.float32)
            cnn_list.append(cnn)

            if hasattr(self.gf, "build_continuous_features"):
                vec = _coerce_vec(self.gf.build_continuous_features(a))
            else:
                vec = _empty_vec()
            vec_list.append(vec)

            if self.include_mask_in_obs:
                mm = np.asarray(self.gf.get_macro_mask(a), dtype=np.bool_).reshape(-1)
                if mm.shape != (self._n_macros,) or (not mm.any()):
                    mm = np.ones((self._n_macros,), dtype=np.bool_)
                mask_list.append(mm.astype(np.float32))

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
        if mask.shape != (self._n_macros,) or (not mask.any()):
            return macro, tgt

        if not bool(mask[macro]):
            macro = 0  # fallback to GO_TO
        return macro, tgt

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
