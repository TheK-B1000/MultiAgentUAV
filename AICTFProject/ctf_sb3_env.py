# ctf_sb3_env.py
from __future__ import annotations
from typing import Any, Dict, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from game_field import GameField, CNN_ROWS, CNN_COLS, NUM_CNN_CHANNELS
from macro_actions import MacroAction

# Opponent helpers (you'll paste these from section 2)
from red_opponents import make_species_wrapper, make_snapshot_wrapper


class CTFGameFieldSB3Env(gym.Env):
    """
    SB3 wrapper around your GameField.

    Each env.step() == one "decision tick" for both blue agents:
      - We feed external actions for blue via submit_external_actions()
      - We advance GameField.update(dt) with dt = 0.99 * decision_interval_seconds
        so exactly ONE decision happens per agent per step (no double-decision).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        make_game_field_fn,  # () -> GameField
        *,
        max_decision_steps: int = 400,
        enforce_masks: bool = True,
        seed: int = 0,
    ):
        super().__init__()
        self.make_game_field_fn = make_game_field_fn
        self.max_decision_steps = int(max_decision_steps)
        self.enforce_masks = bool(enforce_masks)
        self.base_seed = int(seed)

        self.gf: Optional[GameField] = None
        self._decision_step_count = 0

        # Will be set on first reset when gf exists
        self._n_macros = 5
        self._n_targets = 8

        # Two blue agents, each chooses (macro_idx, target_idx)
        self.action_space = spaces.MultiDiscrete([self._n_macros, self._n_targets, self._n_macros, self._n_targets])

        # Observation: stack blue0 + blue1 => (14, 20, 20)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(2 * NUM_CNN_CHANNELS, CNN_ROWS, CNN_COLS),
            dtype=np.float32,
        )

        # Reward bookkeeping (best-effort, depending on what GameManager exposes)
        self._last_reward_total = 0.0

        # Opponent identity (for Elo + logging)
        self._opponent_kind = "species"
        self._opponent_snapshot_path: Optional[str] = None
        self._opponent_species_tag: str = "BALANCED"

    # -----------------------------
    # SB3 hot-swap opponent methods
    # -----------------------------
    def set_opponent_species(self, species_tag: str) -> None:
        """
        Uses your NEW GameField wrapper API to override internal red decisions.
        """
        if self.gf is None:
            return
        self._opponent_kind = "species"
        self._opponent_species_tag = str(species_tag).upper()
        self._opponent_snapshot_path = None

        wrapper = make_species_wrapper(self._opponent_species_tag)
        self.gf.set_red_policy_wrapper(wrapper)

    def set_opponent_snapshot(self, snapshot_path: str) -> None:
        """
        Loads a snapshot opponent into a wrapper.
        snapshot_path can be:
          - a SB3 PPO zip path (we'll load PPO and call .predict per red agent)
          - OR your torch snapshot (if you implement that loader in red_opponents.py)
        """
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

        self.gf = self.make_game_field_fn()
        self._decision_step_count = 0

        # Blue is controlled externally by SB3; red stays internal, override via wrapper
        self.gf.set_external_control("blue", True)
        self.gf.set_external_control("red", False)
        self.gf.external_missing_action_mode = "idle"
        self.gf.use_internal_policies = True

        # Ensure targets exist
        if not self.gf.macro_targets:
            _ = self.gf.get_all_macro_targets()

        # Update action space dims to match your field
        self._n_macros = int(self.gf.n_macros)
        self._n_targets = int(self.gf.num_macro_targets or 8)
        self.action_space = spaces.MultiDiscrete([self._n_macros, self._n_targets, self._n_macros, self._n_targets])

        # Reset sim
        self.gf.reset_default()
        self._last_reward_total = self._read_reward_total()

        # Default opponent if not set yet
        if self.gf.policy_wrappers.get("red") is None:
            self.set_opponent_species("BALANCED")

        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action: np.ndarray):
        assert self.gf is not None, "Call reset() first"
        self._decision_step_count += 1

        # Unpack action for two blue agents
        b0_macro, b0_tgt, b1_macro, b1_tgt = map(int, action.tolist())

        # Enforce masks in-env (vanilla SB3 PPO has no mask support)
        b0_macro, b0_tgt = self._sanitize_action_for_agent(blue_index=0, macro=b0_macro, tgt=b0_tgt)
        b1_macro, b1_tgt = self._sanitize_action_for_agent(blue_index=1, macro=b1_macro, tgt=b1_tgt)

        # Submit external actions using keys your GameField will recognize.
        # It checks slot_id, unique_id, then "side_agentId".
        blue0 = self.gf.blue_agents[0] if len(self.gf.blue_agents) > 0 else None
        blue1 = self.gf.blue_agents[1] if len(self.gf.blue_agents) > 1 else None

        actions_by_agent: Dict[str, Tuple[int, Any]] = {}
        if blue0 is not None:
            actions_by_agent[str(getattr(blue0, "unique_id", "blue_0"))] = (b0_macro, b0_tgt)
        if blue1 is not None:
            actions_by_agent[str(getattr(blue1, "unique_id", "blue_1"))] = (b1_macro, b1_tgt)

        self.gf.submit_external_actions(actions_by_agent)

        # Advance sim by one "decision tick" without allowing double-decisions:
        # dt = 0.99*interval => cooldown = -0.99I then +I => +0.01I -> stops.
        interval = float(getattr(self.gf, "decision_interval_seconds", 0.7))
        dt = max(1e-3, 0.99 * interval)
        self.gf.update(dt)

        # Reward: best-effort delta of a cumulative reward total
        reward_total = self._read_reward_total()
        reward = float(reward_total - self._last_reward_total)
        self._last_reward_total = reward_total

        # Termination
        gm = self.gf.manager
        terminated = bool(getattr(gm, "game_over", False))

        # Optional hard cap on decision steps
        truncated = (self._decision_step_count >= self.max_decision_steps)

        obs = self._get_obs()

        blue_score = int(getattr(gm, "blue_score", 0))
        red_score = int(getattr(gm, "red_score", 0))

        info: Dict[str, Any] = {}
        if terminated or truncated:
            info["episode_result"] = {
                "blue_score": blue_score,
                "red_score": red_score,
                "opponent_kind": self._opponent_kind,
                "opponent_snapshot": self._opponent_snapshot_path,
                "species_tag": self._opponent_species_tag if self._opponent_kind == "species" else None,
                "decision_steps": self._decision_step_count,
            }

        return obs, reward, terminated, truncated, info

    # -----------------
    # Helpers
    # -----------------
    def _get_obs(self) -> np.ndarray:
        assert self.gf is not None
        # Blue0 + Blue1 observation stacks
        if len(self.gf.blue_agents) == 0:
            return np.zeros((2 * NUM_CNN_CHANNELS, CNN_ROWS, CNN_COLS), dtype=np.float32)

        blue0 = self.gf.blue_agents[0]
        o0 = np.asarray(self.gf.build_observation(blue0), dtype=np.float32)  # [7,20,20]

        if len(self.gf.blue_agents) > 1:
            blue1 = self.gf.blue_agents[1]
            o1 = np.asarray(self.gf.build_observation(blue1), dtype=np.float32)
        else:
            o1 = np.zeros_like(o0)

        return np.concatenate([o0, o1], axis=0)

    def _sanitize_action_for_agent(self, blue_index: int, macro: int, tgt: int) -> Tuple[int, int]:
        if not self.enforce_masks:
            return int(macro) % self._n_macros, int(tgt) % self._n_targets

        assert self.gf is not None
        if blue_index >= len(self.gf.blue_agents):
            return 0, 0

        agent = self.gf.blue_agents[blue_index]
        macro = int(macro) % self._n_macros
        tgt = int(tgt) % self._n_targets

        mask = np.asarray(self.gf.get_macro_mask(agent), dtype=np.bool_).reshape(-1)
        if mask.shape != (self._n_macros,) or (not mask.any()):
            return macro, tgt

        if not bool(mask[macro]):
            # Fallback macro: GO_TO (idx 0 in your macro_order)
            macro = 0
        return macro, tgt

    def _read_reward_total(self) -> float:
        """
        Best-effort: adapt to how your GameManager stores reward.
        If you already have a clean per-step reward API, plug it in here.
        """
        assert self.gf is not None
        gm = self.gf.manager

        # Common patterns you might already have
        for attr in ("blue_reward_total", "reward_total_blue", "total_reward_blue", "blue_total_reward"):
            if hasattr(gm, attr):
                try:
                    return float(getattr(gm, attr))
                except Exception:
                    pass

        # If nothing exists, you still train (reward=0) but learning will be dead.
        return 0.0
