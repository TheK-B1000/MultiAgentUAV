"""
Contract validators for MARL safety hardening.

Validates invariants for:
- Agent keys (canonical blue_i format)
- Reward breakdowns (team sum == per-agent sum)
- Observation ordering (consistent with agent identities)
- Action keys (canonical format)
- Dropped reward events policy
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


def validate_agent_keys(keys: List[str], n_agents: int, debug: bool = False) -> bool:
    """
    Validate agent keys are canonical blue_i format.
    
    Args:
        keys: List of agent keys to validate
        n_agents: Expected number of agents
        debug: If True, raise AssertionError on failure; else return False
    
    Returns:
        True if valid, False if invalid (unless debug=True, then raises)
    """
    if len(keys) != n_agents:
        if debug:
            raise AssertionError(f"Agent keys length mismatch: got {len(keys)}, expected {n_agents}")
        return False
    
    for i, key in enumerate(keys):
        expected = f"blue_{i}"
        if key != expected:
            if debug:
                raise AssertionError(f"Invalid agent key at index {i}: got {key!r}, expected {expected!r}")
            return False
    
    return True


def validate_reward_breakdown(
    team_total: float,
    per_agent: List[float],
    tolerance: float = 1e-5,
    debug: bool = False,
) -> bool:
    """
    Validate reward breakdown: team_total should equal sum(per_agent) within tolerance.
    
    Args:
        team_total: Team reward total
        per_agent: Per-agent reward list
        tolerance: Numerical tolerance for floating-point comparison
        debug: If True, raise AssertionError on failure; else return False
    
    Returns:
        True if valid, False if invalid (unless debug=True, then raises)
    """
    per_agent_sum = sum(per_agent)
    diff = abs(team_total - per_agent_sum)
    
    if diff > tolerance:
        if debug:
            raise AssertionError(
                f"Reward breakdown mismatch: team_total={team_total}, sum(per_agent)={per_agent_sum}, diff={diff}"
            )
        return False
    
    return True


def validate_obs_order(
    obs: Dict[str, Any],
    n_agents: int,
    max_agents: Optional[int] = None,
    debug: bool = False,
) -> bool:
    """
    Validate observation ordering matches canonical agent order.
    
    Checks:
    - obs["grid"] shape matches expected (tokenized or legacy)
    - obs["vec"] shape matches expected
    - obs["agent_mask"] (if tokenized) has correct length
    
    Args:
        obs: Observation dictionary
        n_agents: Actual number of agents
        max_agents: Max agents (for tokenized mode); if None, assumes legacy (n_agents*channels)
        debug: If True, raise AssertionError on failure; else return False
    
    Returns:
        True if valid, False if invalid (unless debug=True, then raises)
    """
    import numpy as np
    
    grid = obs.get("grid")
    vec = obs.get("vec")
    
    if grid is None or vec is None:
        if debug:
            raise AssertionError("Observation missing 'grid' or 'vec' keys")
        return False
    
    grid_shape = np.asarray(grid).shape
    vec_shape = np.asarray(vec).shape
    
    if max_agents is not None and max_agents > 2:
        # Tokenized mode: (max_agents, C, H, W) and (max_agents, vec_dim)
        expected_grid_shape = (max_agents, grid_shape[1] if len(grid_shape) > 1 else 7, grid_shape[2] if len(grid_shape) > 2 else 20, grid_shape[3] if len(grid_shape) > 3 else 20)
        expected_vec_shape = (max_agents, vec_shape[1] if len(vec_shape) > 1 else 12)
        
        if grid_shape[0] != max_agents:
            if debug:
                raise AssertionError(f"Tokenized obs grid shape mismatch: got {grid_shape[0]}, expected {max_agents}")
            return False
        
        if vec_shape[0] != max_agents:
            if debug:
                raise AssertionError(f"Tokenized obs vec shape mismatch: got {vec_shape[0]}, expected {max_agents}")
            return False
        
        agent_mask = obs.get("agent_mask")
        if agent_mask is not None:
            mask_shape = np.asarray(agent_mask).shape
            if mask_shape[0] != max_agents:
                if debug:
                    raise AssertionError(f"Tokenized obs agent_mask shape mismatch: got {mask_shape[0]}, expected {max_agents}")
                return False
    else:
        # Legacy mode: (n_agents*C, H, W) and (n_agents*vec_dim,)
        # We can't validate exact shape without knowing C and vec_dim, but we can check it's 1D/2D
        if len(vec_shape) != 1:
            if debug:
                raise AssertionError(f"Legacy obs vec should be 1D, got shape {vec_shape}")
            return False
    
    return True


def validate_action_keys(
    actions: Dict[str, Any],
    n_agents: int,
    debug: bool = False,
) -> bool:
    """
    Validate action keys are canonical blue_i format.
    
    Args:
        actions: Dictionary mapping agent keys to actions
        n_agents: Expected number of agents
        debug: If True, raise AssertionError on failure; else return False
    
    Returns:
        True if valid, False if invalid (unless debug=True, then raises)
    """
    keys = list(actions.keys())
    
    # Allow fewer keys than n_agents (some agents might be disabled)
    if len(keys) > n_agents:
        if debug:
            raise AssertionError(f"Too many action keys: got {len(keys)}, expected at most {n_agents}")
        return False
    
    for key in keys:
        if not key.startswith("blue_"):
            if debug:
                raise AssertionError(f"Invalid action key format: {key!r} (should start with 'blue_')")
            return False
        
        try:
            idx = int(key.split("_")[1])
            if idx < 0 or idx >= n_agents:
                if debug:
                    raise AssertionError(f"Action key index out of range: {key!r} (index {idx} not in [0, {n_agents}))")
                return False
        except (ValueError, IndexError):
            if debug:
                raise AssertionError(f"Invalid action key format: {key!r} (should be 'blue_N' where N is integer)")
            return False
    
    return True


def validate_dropped_reward_events_policy(
    dropped_count: int,
    total_events: int,
    max_drop_rate: float = 0.1,
    debug: bool = False,
) -> bool:
    """
    Validate dropped reward events policy: drop rate should be below threshold.
    
    Args:
        dropped_count: Number of dropped events
        total_events: Total number of reward events
        max_drop_rate: Maximum allowed drop rate (default 10%)
        debug: If True, raise AssertionError on failure; else return False
    
    Returns:
        True if valid, False if invalid (unless debug=True, then raises)
    """
    if total_events == 0:
        return True  # No events, nothing to drop
    
    drop_rate = float(dropped_count) / float(total_events)
    
    if drop_rate > max_drop_rate:
        if debug:
            raise AssertionError(
                f"Dropped reward events policy violation: drop_rate={drop_rate:.2%} > max_drop_rate={max_drop_rate:.2%} "
                f"(dropped={dropped_count}, total={total_events})"
            )
        return False
    
    return True


__all__ = [
    "validate_agent_keys",
    "validate_reward_breakdown",
    "validate_obs_order",
    "validate_action_keys",
    "validate_dropped_reward_events_policy",
]
