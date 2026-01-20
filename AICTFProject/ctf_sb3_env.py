# ctf_sb3_env.py
from __future__ import annotations
from typing import Any, Dict, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from game_field import GameField, CNN_ROWS, CNN_COLS, NUM_CNN_CHANNELS

from red_opponents import make_species_wrapper, make_snapshot_wrapper


class CTFGameFieldSB3Env(gym.Env):
    """
    SB3 wrapper around GameField.

    Each env.step() == one decision tick for both blue agents.
    Observation is team-based Dict:
      - grid: [14, 20, 20]  (blue0 cnn + blue1 cnn)
      - vec:  [8]          (blue0 vec4 + blue1 vec4)
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        make_game_field_fn,  # () -> GameField
        *,
        max_decision_steps: int = 400,
        enforce_masks: bool = True,
        seed: int = 0,
        include_mask_in_obs: bool = False,  # only useful if you explicitly use it in custom extractor
    ):
        super().__init__()
        self.make_game_field_fn = make_game_field_fn
        self.max_decision_steps = int(max_decision_steps)
        self.enforce_masks = bool(enforce_masks)
        self.base_seed = int(seed)
        self.include_mask_in_obs = bool(include_mask_in_obs)

        self.gf: Optional[GameField] = None
        self._decision_step_count = 0

        # Default sizes (updated on reset)
        self._n_macros = 5
        self._n_targets = 8

        # Two blue agents, each chooses (macro_idx, target_idx)
        self.action_space = spaces.MultiDiscrete([self._n_macros, self._n_targets, self._n_macros, self._n_targets])

        # Team obs:
        # grid = [2*C, H, W]
        # vec  = [2*4]  (frac_x, frac_y, vel_x_norm, vel_y_norm) per agent
        grid_shape = (2 * NUM_CNN_CHANNELS, CNN_ROWS, CNN_COLS)
        vec_shape = (8,)

        obs_dict = {
            "grid": spaces.Box(low=0.0, high=1.0, shape=grid_shape, dtype=np.float32),
            "vec": spaces.Box(low=-2.0, high=2.0, shape=vec_shape, dtype=np.float32),
        }

        if self.include_mask_in_obs:
            obs_dict["mask"] = spaces.Box(low=0.0, high=1.0, shape=(2 * self._n_macros,), dtype=np.float32)

        self.observation_space = spaces.Dict(obs_dict)

        self._last_reward_total = 0.0

        # Opponent identity (for Elo + logging)
        self._opponent_kind = "species"
        self._opponent_snapshot_path: Optional[str] = None
        self._opponent_species_tag: str = "BALANCED"

    # -----------------------------
    # Opponent hot-swap
    # -----------------------------
    def set_opponent_species(self, species_tag: str) -> None:
        if self.gf is None:
            return
        self._opponent_kind = "species"
        self._opponent_species_tag = str(species_tag).upper()
        self._opponent_snapshot_path = None
        wrapper = make_species_wrapper(self._opponent_species_tag)
        self.gf.set_red_policy_wrapper(wrapper)

    def set_opponent_snapshot(self, snapshot_path: str) -> None:
        if self.gf is None:
            return
        self._opponent_kind = "snapshot"
        self._opponent_snapshot_path = str(snapshot_path)
        self._opponent_species_tag = "BALANCED"
        wrapper = make_snapshot_wrapper(self._opponent_snapshot_path)
        self.gf.set_red_policy_wrapper(wrapper)

    # -------------
    # Gym API
    # -------------
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        # Seed numpy for any wrapper-side randomness (opponent selection, etc.)
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

        # Update action dims
        self._n_macros = int(self.gf.n_macros)
        self._n_targets = int(self.gf.num_macro_targets or 8)
        self.action_space = spaces.MultiDiscrete([self._n_macros, self._n_targets, self._n_macros, self._n_targets])

        # If you included mask, update obs space too (since n_macros might change)
        if self.include_mask_in_obs:
            self.observation_space = spaces.Dict({
                "grid": spaces.Box(low=0.0, high=1.0, shape=(2 * NUM_CNN_CHANNELS, CNN_ROWS, CNN_COLS), dtype=np.float32),
                "vec": spaces.Box(low=-2.0, high=2.0, shape=(8,), dtype=np.float32),
                "mask": spaces.Box(low=0.0, high=1.0, shape=(2 * self._n_macros,), dtype=np.float32),
            })

        self.gf.reset_default()
        self._last_reward_total = self._read_reward_total()

        if self.gf.policy_wrappers.get("red") is None:
            self.set_opponent_species("BALANCED")

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

        reward_total = self._read_reward_total()
        reward = float(reward_total - self._last_reward_total)
        self._last_reward_total = reward_total

        gm = self.gf.manager
        terminated = bool(getattr(gm, "game_over", False))
        truncated = (self._decision_step_count >= self.max_decision_steps)

        obs = self._get_obs()

        info: Dict[str, Any] = {}
        if terminated or truncated:
            info["episode_result"] = {
                "blue_score": int(getattr(gm, "blue_score", 0)),
                "red_score": int(getattr(gm, "red_score", 0)),
                "opponent_kind": self._opponent_kind,
                "opponent_snapshot": self._opponent_snapshot_path,
                "species_tag": self._opponent_species_tag if self._opponent_kind == "species" else None,
                "decision_steps": self._decision_step_count,
            }

        return obs, reward, terminated, truncated, info

    # -----------------
    # Helpers
    # -----------------
    def _get_obs(self) -> Dict[str, np.ndarray]:
        assert self.gf is not None

        # Ensure we always have 2 "slots"
        def _empty_cnn():
            return np.zeros((NUM_CNN_CHANNELS, CNN_ROWS, CNN_COLS), dtype=np.float32)

        def _empty_vec():
            return np.zeros((4,), dtype=np.float32)

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

            # Your GameField.build_observation should return the CNN tensor [7,20,20]
            cnn = np.asarray(self.gf.build_observation(a), dtype=np.float32)
            cnn_list.append(cnn)

            # Sub-pixel continuous vec4
            if hasattr(self.gf, "build_continuous_features"):
                vec = np.asarray(self.gf.build_continuous_features(a), dtype=np.float32)
            else:
                vec = _empty_vec()
            vec_list.append(vec)

            if self.include_mask_in_obs:
                mm = np.asarray(self.gf.get_macro_mask(a), dtype=np.bool_).reshape(-1)
                if mm.shape != (self._n_macros,) or (not mm.any()):
                    mm = np.ones((self._n_macros,), dtype=np.bool_)
                mask_list.append(mm.astype(np.float32))

        grid = np.concatenate(cnn_list, axis=0).astype(np.float32)  # [14,20,20]
        vec = np.concatenate(vec_list, axis=0).astype(np.float32)   # [8]

        out = {"grid": grid, "vec": vec}
        if self.include_mask_in_obs:
            out["mask"] = np.concatenate(mask_list, axis=0).astype(np.float32)  # [2*n_macros]
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
        mask = np.asarray(self.gf.get_macro_mask(agent), dtype=np.bool_).reshape(-1)
        if mask.shape != (self._n_macros,) or (not mask.any()):
            return macro, tgt

        if not bool(mask[macro]):
            macro = 0  # fallback to GO_TO
        return macro, tgt

    def _read_reward_total(self) -> float:
        assert self.gf is not None
        gm = self.gf.manager
        for attr in ("blue_reward_total", "reward_total_blue", "total_reward_blue", "blue_total_reward"):
            if hasattr(gm, attr):
                try:
                    return float(getattr(gm, attr))
                except Exception:
                    pass
        return 0.0
