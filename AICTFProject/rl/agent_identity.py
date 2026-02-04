"""
Canonical agent identity contract for the entire CTF stack.

Agent Keys:
  - Blue agents: "blue_0", "blue_1", "blue_2", ... (zero-indexed, slot-based)
  - Red agents: internal only (not exposed to external control)
  - Keys are deterministic: blue_0 = first agent in blue_agents list, blue_1 = second, etc.
  - Keys are stable within an episode (agents don't swap slots)
  - Keys are used for:
    * Action submission (submit_external_actions)
    * Reward routing (mapping reward events to agents)
    * Observation ordering (obs["grid"][i] corresponds to blue_i)
    * MARL wrapper agent_keys

Identity Mapping:
  - Maps canonical keys (blue_i) to actual agent objects
  - Built from current blue_agents list at reset/step
  - Used to route actions and rewards deterministically
  - Handles missing/disabled agents gracefully (dropped events)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class AgentIdentity:
    """Canonical agent identity: slot-based key + agent reference."""
    key: str  # canonical key: "blue_0", "blue_1", ...
    slot_index: int  # zero-indexed slot (0, 1, 2, ...)
    agent: Optional[Any] = None  # actual agent object (may be None if disabled/missing)
    is_enabled: bool = False  # whether agent exists and is enabled

    def __post_init__(self) -> None:
        if self.agent is not None:
            self.is_enabled = bool(getattr(self.agent, "isEnabled", lambda: True)())


def build_blue_identities(blue_agents: List[Any], max_agents: Optional[int] = None) -> List[AgentIdentity]:
    """
    Build canonical identity list from blue_agents.
    
    Args:
        blue_agents: List of agent objects (may include None/disabled)
        max_agents: Optional max count (for tokenized mode); if None, uses len(blue_agents)
    
    Returns:
        List of AgentIdentity, one per slot (blue_0, blue_1, ...)
        Missing/disabled agents have agent=None, is_enabled=False
    """
    if max_agents is None:
        max_agents = len(blue_agents) if blue_agents else 0
    
    identities: List[AgentIdentity] = []
    for i in range(max_agents):
        key = f"blue_{i}"
        agent = blue_agents[i] if i < len(blue_agents) else None
        ident = AgentIdentity(key=key, slot_index=i, agent=agent)
        identities.append(ident)
    
    return identities


def build_reward_id_map(identities: List[AgentIdentity]) -> Dict[str, int]:
    """
    Build mapping from agent internal IDs to canonical slot indices.
    
    Maps various internal identifiers (unique_id, slot_id, side_agent_id) to blue_i slot.
    Used to route reward events from GameManager (which use internal IDs) to canonical slots.
    
    Args:
        identities: List of AgentIdentity from build_blue_identities()
    
    Returns:
        Dict mapping internal_id -> slot_index
        Example: {"blue_0_unique": 0, "slot_1": 1, "blue_1": 1, ...}
    """
    id_map: Dict[str, int] = {}
    
    for ident in identities:
        if ident.agent is None or not ident.is_enabled:
            continue
        
        slot = ident.slot_index
        
        # Try multiple internal ID attributes
        candidates = []
        for attr in ("unique_id", "slot_id"):
            if hasattr(ident.agent, attr):
                try:
                    val = getattr(ident.agent, attr)
                    if val is not None:
                        candidates.append(str(val))
                except Exception:
                    pass
        
        # Fallback: side_agent_id
        side = str(getattr(ident.agent, "side", "blue")).lower()
        agent_id = int(getattr(ident.agent, "agent_id", slot))
        candidates.append(f"{side}_{agent_id}")
        
        # Also add canonical key itself (in case events use it)
        candidates.append(ident.key)
        
        # Map all candidates to this slot
        for candidate in candidates:
            if candidate and candidate.strip():
                id_map[candidate.strip()] = slot
    
    return id_map


def route_reward_events(
    reward_events: List[Tuple[float, str, float]],  # (timestamp, agent_id, value)
    reward_id_map: Dict[str, int],
    n_slots: int,
) -> Tuple[float, List[float], int]:
    """
    Route reward events to canonical slots using identity map.
    
    Args:
        reward_events: List of (timestamp, agent_id, value) from GameManager
        reward_id_map: Mapping from internal agent_id -> slot_index (from build_reward_id_map)
        n_slots: Number of canonical slots (len(identities))
    
    Returns:
        (team_total, per_agent_list, dropped_event_count)
        - team_total: sum of all routed rewards
        - per_agent_list: [reward_0, reward_1, ...] for each slot (len = n_slots)
        - dropped_event_count: number of events that couldn't be routed
    """
    per_agent = [0.0] * n_slots
    dropped = 0
    
    for _timestamp, agent_id, value in reward_events:
        slot = reward_id_map.get(str(agent_id).strip())
        if slot is not None and 0 <= slot < n_slots:
            per_agent[slot] += float(value)
        else:
            dropped += 1
    
    team_total = sum(per_agent)
    return (team_total, per_agent, dropped)


__all__ = [
    "AgentIdentity",
    "build_blue_identities",
    "build_reward_id_map",
    "route_reward_events",
]
