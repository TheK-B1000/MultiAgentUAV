"""
Observation builder bridge for CTFGameFieldSB3Env.
Builds team observations with canonical ordering; optional vec standardization.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np

from game_field import NUM_CNN_CHANNELS, CNN_ROWS, CNN_COLS

try:
    from rl.obs_builder import build_team_obs
except Exception:
    build_team_obs = None

# Per-feature mean/std for vec standardization (schema: [0,1] and [-1,1] dims).
def _vec_standardization_constants(size: int) -> tuple:
    mean = np.zeros((size,), dtype=np.float32)
    std = np.ones((size,), dtype=np.float32)
    for i in range(min(12, size)):
        if i in (2, 4, 5, 6, 7):
            mean[i], std[i] = 0.0, 0.57735  # [-1,1]
        else:
            mean[i], std[i] = 0.5, 0.5  # [0,1]
    return mean, std


def _standardize_vec(vec: np.ndarray, mean: np.ndarray, std: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32)
    m = mean.reshape(-1)
    s = std.reshape(-1)
    if vec.ndim == 1:
        n = min(vec.shape[0], m.shape[0])
        return ((vec[:n] - m[:n]) / (s[:n] + eps)).astype(np.float32)
    n = min(vec.shape[-1], m.shape[0])
    return ((vec[..., :n] - m[:n]) / (s[:n] + eps)).astype(np.float32)


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
        normalize_vec: bool = False,
    ) -> None:
        self._use_obs_builder = bool(use_obs_builder)
        self._include_mask_in_obs = bool(include_mask_in_obs)
        self._include_opponent_context = bool(include_opponent_context)
        self._include_high_level_mode = bool(include_high_level_mode)
        self._high_level_mode = int(high_level_mode)
        self._high_level_mode_onehot = bool(high_level_mode_onehot)
        self._base_vec_per_agent = int(base_vec_per_agent)
        self._obs_debug_validate_locality = bool(obs_debug_validate_locality)
        self._normalize_vec = bool(normalize_vec)
        self._vec_mean, self._vec_std = _vec_standardization_constants(self._base_vec_per_agent)

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
        if self._normalize_vec and "vec" in out:
            out["vec"] = _standardize_vec(out["vec"], self._vec_mean, self._vec_std)
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
