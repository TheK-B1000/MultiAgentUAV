"""
Reward routing and metrics module for CTFGameFieldSB3Env.

Handles:
- Reward event consumption and routing via identity map
- Stall penalty tracking
- Episode metrics collection
- Dropped reward events tracking
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from rl.agent_identity import route_reward_events


class EnvRewardManager:
    """Manages reward routing, stall penalties, and episode metrics."""

    def __init__(self) -> None:
        self._stall_threshold_cells = 0.15
        self._stall_patience = 3
        self._stall_penalty = -0.05
        self._stall_counters: List[int] = []
        self._last_blue_pos: List[Optional[Tuple[float, float]]] = []
        self._dropped_reward_events_this_step = 0
        self._dropped_reward_events_this_episode = 0
        self._episode_reward_total = 0.0

    def reset_episode(self, n_agents: int) -> None:
        """Reset episode-level tracking."""
        self._stall_counters = [0] * n_agents
        self._last_blue_pos = [None] * n_agents
        self._dropped_reward_events_this_step = 0
        self._dropped_reward_events_this_episode = 0
        self._episode_reward_total = 0.0

    def consume_reward_events(
        self,
        game_manager: Any,
        reward_id_map: Dict[str, int],
        n_slots: int,
    ) -> Tuple[float, List[float], int]:
        """
        Consume blue reward events from GameManager using canonical identity map.
        Returns (team_total, per_agent_list, dropped_event_count).
        """
        pop = getattr(game_manager, "pop_reward_events", None)
        if pop is None or (not callable(pop)):
            return 0.0, [0.0] * n_slots, 0

        try:
            events = pop()
        except Exception:
            return 0.0, [0.0] * n_slots, 0

        # Route events using canonical identity map
        team_total, per_agent_list, dropped_count = route_reward_events(
            reward_events=events,
            reward_id_map=reward_id_map,
            n_slots=n_slots,
        )

        self._dropped_reward_events_this_step = dropped_count
        self._dropped_reward_events_this_episode += dropped_count

        return float(team_total), per_agent_list, dropped_count

    def apply_stall_penalty(
        self,
        game_field: Any,
        n_blue_agents: int,
    ) -> Tuple[float, List[float]]:
        """
        Apply per-agent stall penalty. Returns (team_total, per_agent_penalties).
        """
        per_agent = [0.0] * n_blue_agents
        penalty = 0.0

        blue_agents = getattr(game_field, "blue_agents", []) or []

        # Ensure lists are large enough
        while len(self._stall_counters) < n_blue_agents:
            self._stall_counters.append(0)
        while len(self._last_blue_pos) < n_blue_agents:
            self._last_blue_pos.append(None)

        for i in range(n_blue_agents):
            if i >= len(blue_agents):
                self._stall_counters[i] = 0
                if i < len(self._last_blue_pos):
                    self._last_blue_pos[i] = None
                continue

            agent = blue_agents[i]
            if agent is None:
                self._stall_counters[i] = 0
                if i < len(self._last_blue_pos):
                    self._last_blue_pos[i] = None
                continue

            pos = tuple(
                getattr(
                    agent,
                    "float_pos",
                    (float(getattr(agent, "x", 0)), float(getattr(agent, "y", 0))),
                )
            )
            last = self._last_blue_pos[i] if i < len(self._last_blue_pos) else None

            if last is None:
                moved = True
            else:
                dx = float(pos[0]) - float(last[0])
                dy = float(pos[1]) - float(last[1])
                moved = (dx * dx + dy * dy) >= (float(self._stall_threshold_cells) ** 2)

            if moved:
                self._stall_counters[i] = 0
            else:
                self._stall_counters[i] += 1
                if self._stall_counters[i] >= int(self._stall_patience):
                    p = float(self._stall_penalty)
                    penalty += p
                    per_agent[i] += p

            if i < len(self._last_blue_pos):
                self._last_blue_pos[i] = pos

        return float(penalty), per_agent

    def get_dropped_events_step(self) -> int:
        return self._dropped_reward_events_this_step

    def get_dropped_events_episode(self) -> int:
        return self._dropped_reward_events_this_episode

    def reset_dropped_events_episode(self) -> None:
        self._dropped_reward_events_this_episode = 0

    def add_episode_reward(self, reward: float) -> None:
        self._episode_reward_total += float(reward)

    def get_episode_reward_total(self) -> float:
        return self._episode_reward_total


__all__ = ["EnvRewardManager"]
