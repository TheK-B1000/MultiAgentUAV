"""
Observation builder bridge module for CTFGameFieldSB3Env.

Handles:
- Building team observations (canonical ordering)
- High-level mode appending
- Legacy vs tokenized observation paths
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np

from game_field import NUM_CNN_CHANNELS, CNN_ROWS, CNN_COLS

try:
    from rl.obs_builder import build_team_obs
except Exception:
    build_team_obs = None


class EnvObsBuilder:
    """Manages observation building with canonical agent ordering."""

    def __init__(
        self,
        *,
        use_obs_builder: bool = True,
        include_mask_in_obs: bool = False,
        include_opponent_context: bool = False,
        include_high_level_mode: bool = False,
        high_level_mode: int = 0,
        high_level_mode_onehot: bool = True,
        base_vec_per_agent: int = 12,
        obs_debug_validate_locality: bool = False,
    ) -> None:
        self._use_obs_builder = bool(use_obs_builder)
        self._include_mask_in_obs = bool(include_mask_in_obs)
        self._include_opponent_context = bool(include_opponent_context)
        self._include_high_level_mode = bool(include_high_level_mode)
        self._high_level_mode = int(high_level_mode)
        self._high_level_mode_onehot = bool(high_level_mode_onehot)
        self._base_vec_per_agent = int(base_vec_per_agent)
        self._obs_debug_validate_locality = bool(obs_debug_validate_locality)

    def build_observation(
        self,
        game_field: Any,
        blue_identities: List[Any],  # List[AgentIdentity]
        *,
        max_blue_agents: int,
        n_blue_agents: int,
        n_macros: int,
        n_targets: int,
        opponent_context_id: Optional[int] = None,
        role_macro_mask_fn: Optional[Callable[[int, np.ndarray], np.ndarray]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Build team observation using canonical identity ordering.
        
        Args:
            blue_identities: List[AgentIdentity] - canonical ordering (blue_0, blue_1, ...)
            opponent_context_id: Optional context ID for obs["context"]
            role_macro_mask_fn: Optional function to apply role-based macro masking
        
        Returns:
            Dict with "grid", "vec", optionally "mask", "agent_mask" (tokenized), "context"
        """
        n_slots = len(blue_identities)
        blue_ordered = [ident.agent for ident in blue_identities[:n_slots]]

        # Always use canonical build_team_obs() - legacy path removed per module ownership
        if build_team_obs is None:
            raise RuntimeError(
                "build_team_obs() not available. Ensure rl/obs_builder.py is importable. "
                "Legacy observation building removed per module ownership."
            )
        
        out = build_team_obs(
            game_field,
            blue_ordered,
            max_agents=n_slots,
            include_mask=self._include_mask_in_obs,
            include_context=self._include_opponent_context,
            context_value=float(opponent_context_id) if (self._include_opponent_context and opponent_context_id is not None) else None,
            debug_locality=self._obs_debug_validate_locality,
            tokenized=(max_blue_agents > 2),
            vec_size_base=self._base_vec_per_agent,
            n_macros=n_macros,
            n_targets=n_targets,
            role_macro_mask_fn=role_macro_mask_fn,
            vec_append_fn=self._append_high_level_mode,
        )
        return out

    def _build_legacy_obs(
        self,
        game_field: Any,
        blue_ordered: List[Any],
        n_slots: int,
        max_blue_agents: int,
        n_blue_agents: int,
        n_macros: int,
        n_targets: int,
        opponent_context_id: Optional[int],
        role_macro_mask_fn: Optional[Callable[[int, np.ndarray], np.ndarray]],
    ) -> Dict[str, np.ndarray]:
        """
        DEPRECATED: Legacy observation building path removed per module ownership.
        
        This method should never be called - always raises RuntimeError.
        Kept for type safety but will never execute.
        """
        raise RuntimeError(
            "Legacy observation building path removed per module ownership. "
            "Always use rl/obs_builder.py::build_team_obs()."
        )
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

        cnn_list: List[np.ndarray] = []
        vec_list: List[np.ndarray] = []
        mask_list: List[np.ndarray] = []

        for idx, a in enumerate(blue_ordered):
            if a is None:
                cnn_list.append(_empty_cnn())
                vec = self._append_high_level_mode(_empty_vec_base())
                vec_list.append(vec)
                if self._include_mask_in_obs:
                    mm = np.zeros((n_macros,), dtype=np.float32)
                    tm = np.zeros((n_targets,), dtype=np.float32)
                    mask_list.append(np.concatenate([mm, tm], axis=0))
                continue

            cnn = np.asarray(game_field.build_observation(a), dtype=np.float32)
            cnn_list.append(cnn)
            if hasattr(game_field, "build_continuous_features"):
                vec = _coerce_vec_base(game_field.build_continuous_features(a))
            else:
                vec = _empty_vec_base()
            vec = self._append_high_level_mode(vec)
            vec_list.append(vec)
            if self._include_mask_in_obs:
                mm = np.asarray(game_field.get_macro_mask(a), dtype=np.bool_).reshape(-1)
                if role_macro_mask_fn is not None:
                    mm = role_macro_mask_fn(idx, mm)
                if mm.shape != (n_macros,) or (not mm.any()):
                    mm = np.ones((n_macros,), dtype=np.bool_)
                tm = np.asarray(game_field.get_target_mask(a), dtype=np.bool_).reshape(-1)
                if tm.shape != (n_targets,) or (not tm.any()):
                    tm = np.ones((n_targets,), dtype=np.bool_)
                mask_list.append(np.concatenate([mm.astype(np.float32), tm.astype(np.float32)], axis=0))

        if max_blue_agents > 2:
            grid = np.stack(cnn_list, axis=0).astype(np.float32)
            vec = np.stack(vec_list, axis=0).astype(np.float32)
            agent_mask = np.zeros((n_slots,), dtype=np.float32)
            agent_mask[:n_blue_agents] = 1.0
            out = {"grid": grid, "vec": vec, "agent_mask": agent_mask}
            if self._include_mask_in_obs:
                out["mask"] = np.concatenate(mask_list, axis=0).astype(np.float32)
        else:
            grid = np.concatenate(cnn_list, axis=0).astype(np.float32)
            vec = np.concatenate(vec_list, axis=0).astype(np.float32)
            out = {"grid": grid, "vec": vec}
            if self._include_mask_in_obs:
                out["mask"] = np.concatenate(mask_list, axis=0).astype(np.float32)
        if self._include_opponent_context and opponent_context_id is not None:
            out["context"] = np.array([float(opponent_context_id)], dtype=np.float32)
        return out

    def _append_high_level_mode(self, base_vec: np.ndarray) -> np.ndarray:
        """Append high-level mode to vec observation."""
        v = np.asarray(base_vec, dtype=np.float32).reshape(-1)
        if not self._include_high_level_mode:
            return v

        if self._high_level_mode_onehot:
            mode = max(0, min(1, int(self._high_level_mode)))
            mode_vec = np.zeros((2,), dtype=np.float32)
            mode_vec[mode] = 1.0
        else:
            mode_vec = np.asarray([float(self._high_level_mode)], dtype=np.float32)

        return np.concatenate([v, mode_vec], axis=0)

    def set_high_level_mode(self, mode: int) -> None:
        """Update high-level mode for next observation."""
        self._high_level_mode = int(mode)


__all__ = ["EnvObsBuilder"]
