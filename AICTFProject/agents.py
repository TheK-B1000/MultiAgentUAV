"""
Agent types used by the GPU environment and adapters.

AgentHandle: minimal handle for policy.act(obs, agent=..., game_field=...)
when using BatchedCTFCore via GPUEnvAdapter (single-env, GameField-like API).
"""
from __future__ import annotations

from typing import Callable, Optional


class AgentHandle:
    """Minimal agent handle for indexing into batched state (agent_id, side, isEnabled)."""

    __slots__ = ("agent_id", "side", "unique_id", "_alive_getter")

    def __init__(
        self,
        agent_id: int,
        side: str = "blue",
        alive_getter: Optional[Callable[[int], bool]] = None,
    ) -> None:
        self.agent_id = int(agent_id)
        self.side = str(side)
        self.unique_id = f"{self.side}_{self.agent_id}"
        self._alive_getter = alive_getter

    def isEnabled(self) -> bool:
        if self._alive_getter is None:
            return True
        return bool(self._alive_getter(self.agent_id))


__all__ = ["AgentHandle"]
