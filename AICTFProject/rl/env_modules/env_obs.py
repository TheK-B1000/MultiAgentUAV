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
        self._agent_roles: Optional[List[int]] = None  # Role indices per agent (0=attacker, 1=defender, 2=escort)

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
        
        # Create per-agent vec_append_fn that includes role for that specific agent
        def vec_append_with_role(agent_idx: int):
            def fn(base_vec: np.ndarray) -> np.ndarray:
                return self._append_high_level_mode_per_agent(base_vec, agent_idx)
            return fn
        
        # Build vec_append_fn that knows agent index (for role lookup)
        # We need to pass agent index to vec_append_fn, but build_team_obs calls it per-agent
        # So we create a closure that captures the agent index
        vec_append_fns = [vec_append_with_role(i) for i in range(n_slots)]
        
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
            vec_append_fn=self._make_vec_append_with_idx(),  # Tracks agent index for role tokens (resets on each call)
        )
        return out
    
    def _make_vec_append_with_idx(self):
        """Create a vec_append_fn that tracks agent index for role token lookup.
        Returns a function that increments agent_idx on each call.
        """
        agent_idx_ref = [0]  # Use list to allow modification in closure
        
        def vec_append_with_idx(base_vec: np.ndarray) -> np.ndarray:
            idx = agent_idx_ref[0]
            result = self._append_high_level_mode(base_vec, agent_idx=idx)
            agent_idx_ref[0] += 1
            return result
        
        # Reset counter before returning (will be reset again before each build_team_obs call)
        agent_idx_ref[0] = 0
        return vec_append_with_idx

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

    def _append_high_level_mode(self, base_vec: np.ndarray, agent_idx: Optional[int] = None) -> np.ndarray:
        """Append high-level mode and role tokens to vec observation.
        Note: Role tokens are now included in observation space, so we append them here.
        The observation space accounts for role_dims=3 per agent.
        
        Args:
            base_vec: Base vector for one agent (vec_size_base dims)
            agent_idx: Optional agent index for role lookup. If None, tries to infer from context.
        """
        v = np.asarray(base_vec, dtype=np.float32).reshape(-1)
        
        # Append role one-hot (3 roles: attacker, defender, escort) for THIS agent only
        # Role tokens are always included (observation space accounts for them)
        if self._agent_roles is not None and agent_idx is not None and 0 <= agent_idx < len(self._agent_roles):
            role_idx = self._agent_roles[agent_idx]
            role_onehot = np.zeros((3,), dtype=np.float32)
            if 0 <= role_idx < 3:
                role_onehot[role_idx] = 1.0
            v = np.concatenate([v, role_onehot], axis=0)
        elif self._agent_roles is not None:
            # Fallback: if agent_idx not provided, append zeros (shouldn't happen)
            v = np.concatenate([v, np.zeros((3,), dtype=np.float32)], axis=0)
        else:
            # Roles not set yet (shouldn't happen after reset)
            v = np.concatenate([v, np.zeros((3,), dtype=np.float32)], axis=0)
        
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
    
    def set_agent_roles(self, roles: List[int]) -> None:
        """Set agent roles for role token appending (0=attacker, 1=defender, 2=escort)."""
        self._agent_roles = list(roles) if roles else None


__all__ = ["EnvObsBuilder"]
