"""
behavior_logging.py

Lightweight behavior logging utilities for the 2-vs-2 UAV CTF environment.

These helpers are designed for research:
    - Track macro-action usage per agent and per team.
    - Log basic per-episode stats (scores, mines, kills, etc.).
    - Attach metadata (phase, opponent type, policy labels).
    - Export per-episode records for plotting (e.g., pandas → matplotlib).

The logger is agnostic to whether a policy is scripted or learned: you simply
call `log_decision(agent, macro_id)` whenever an agent chooses a macro-action,
then `finalize_episode(...)` at the end.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

import csv
import os

from macro_actions import MacroAction
from game_manager import GameManager
from agents import Agent


# ---------------------------- Data structures ----------------------------

@dataclass
class AgentBehaviorStats:
    """
    Per-agent behavior statistics for a single episode.

    macro_counts: how often each macro-action ID was chosen.
                  Keys are int(MacroAction.*), values are counts.
    mine_attempts: how many times this agent attempted to PLACE_MINE.
    flag_carries: how many macro-decisions the agent was carrying a flag.
    """
    macro_counts: Dict[int, int] = field(default_factory=dict)
    mine_attempts: int = 0
    flag_carries: int = 0

    def to_dict(self, side: str, agent_id: int) -> Dict[str, Any]:
        """
        Flatten stats into a dict with explicit side/agent_id labels.
        Useful for writing to CSV / DataFrame.
        """
        base = {
            "side": side,
            "agent_id": agent_id,
            "mine_attempts": self.mine_attempts,
            "flag_carries": self.flag_carries,
        }
        # Expand macro counts as macro_X fields for plotting.
        for mid, count in self.macro_counts.items():
            base[f"macro_{mid}"] = count
        return base


@dataclass
class EpisodeBehaviorRecord:
    """
    Aggregated behavior for one episode across all agents and teams.

    - per_agent: nested dict [side]["agent_id"] -> AgentBehaviorStats
    - gm_stats: high-level game stats pulled from GameManager
    - meta: any extra labels you want (phase, opponent tag, etc.)
    """
    per_agent: Dict[str, Dict[int, AgentBehaviorStats]] = field(
        default_factory=lambda: {"blue": {}, "red": {}}
    )
    gm_stats: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_flat_records(self) -> List[Dict[str, Any]]:
        """
        Flatten into a list of dicts, one per agent, with episode-level stats
        attached. Ready for CSV/DF export.
        """
        records: List[Dict[str, Any]] = []
        for side, agents in self.per_agent.items():
            for agent_id, stats in agents.items():
                row = {}
                row.update(self.meta)
                row.update(self.gm_stats)
                row.update(stats.to_dict(side, agent_id))
                records.append(row)
        return records


# ------------------------------ Logger class -----------------------------

class BehaviorLogger:
    """
    BehaviorLogger collects per-episode macro-action stats and basic game stats.

    """

    def __init__(self):
        self.episodes: List[EpisodeBehaviorRecord] = []
        self.current: Optional[EpisodeBehaviorRecord] = None

    # --------- Episode lifecycle ---------

    def start_episode(self) -> None:
        """Begin a new episode record."""
        self.current = EpisodeBehaviorRecord()

    def ensure_agent_entry(self, agent: Agent) -> AgentBehaviorStats:
        """
        Ensure that the current record has an entry for (side, agent_id),
        and return the AgentBehaviorStats object.
        """
        if self.current is None:
            raise RuntimeError("BehaviorLogger.start_episode() must be called first.")

        side = agent.getSide()
        agent_id = getattr(agent, "agent_id", 0)

        if side not in self.current.per_agent:
            self.current.per_agent[side] = {}

        side_dict = self.current.per_agent[side]
        if agent_id not in side_dict:
            side_dict[agent_id] = AgentBehaviorStats()

        return side_dict[agent_id]

    # --------- Per-decision logging ---------

    def log_decision(self, agent: Agent, macro_id: int) -> None:
        """
        Log that `agent` chose macro-action with integer ID `macro_id`.

        This is intended to be called at the same place you call env.apply_macro_action(...)
        or where a policy's select_action(...) is resolved.
        """
        stats = self.ensure_agent_entry(agent)

        # Macro counts (store counts for this exact ID).
        stats.macro_counts[macro_id] = stats.macro_counts.get(macro_id, 0) + 1

        # Mine attempts (when macro == PLACE_MINE).
        try:
            if int(macro_id) == int(MacroAction.PLACE_MINE):
                stats.mine_attempts += 1
        except ValueError:
            # In case macro_id doesn't map cleanly; better to be robust.
            pass

        # Track how many decisions this agent spent carrying a flag.
        if hasattr(agent, "isCarryingFlag") and agent.isCarryingFlag():
            stats.flag_carries += 1

    # --------- Episode finalization ---------

    def finalize_episode(
        self,
        gm: GameManager,
        meta: Optional[Dict[str, Any]] = None,
    ) -> EpisodeBehaviorRecord:
        """
        Finalize current episode record by attaching GameManager stats and metadata.

        Args:
            gm: GameManager instance after the episode.
            meta: Arbitrary metadata dict (phase, policy tags, etc.).

        Returns:
            The EpisodeBehaviorRecord for this episode.
        """
        if self.current is None:
            raise RuntimeError("No active episode; call start_episode() first.")

        # High-level game stats (we can expand this over time).
        self.current.gm_stats = {
            "blue_score": gm.blue_score,
            "red_score": gm.red_score,
            "phase_name": gm.phase_name,
            "max_time": gm.max_time,
            "score_limit": gm.score_limit,
            "blue_mine_kills": gm.blue_mine_kills_this_episode,
            "red_mine_kills": gm.red_mine_kills_this_episode,
            "mines_placed_enemy_half": gm.mines_placed_in_enemy_half_this_episode,
            "mines_triggered_by_red": gm.mines_triggered_by_red_this_episode,
        }

        # Episode metadata from caller.
        self.current.meta = dict(meta or {})

        ep_record = self.current
        self.episodes.append(ep_record)
        self.current = None
        return ep_record

    # --------- Export helpers ---------

    def all_flat_records(self) -> List[Dict[str, Any]]:
        """
        Flatten all stored episodes into a list of dicts.

        Each dict is a row for a single agent in a single episode.
        Perfect for writing to CSV or using with pandas.DataFrame.
        """
        rows: List[Dict[str, Any]] = []
        for ep in self.episodes:
            rows.extend(ep.to_flat_records())
        return rows


def append_records_csv(path: str, records: List[Dict[str, Any]]) -> None:
    if not records:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    exists = os.path.exists(path)
    new_keys = set()
    for r in records:
        new_keys.update(r.keys())
    new_fieldnames = sorted(new_keys)

    if exists:
        with open(path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_fieldnames = reader.fieldnames or []
            existing_rows = list(reader)

        merged_fieldnames = sorted(set(existing_fieldnames) | set(new_fieldnames))
        if merged_fieldnames != existing_fieldnames:
            # Rewrite file with expanded header to handle new fields safely.
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=merged_fieldnames)
                writer.writeheader()
                for row in existing_rows:
                    writer.writerow({k: row.get(k, "") for k in merged_fieldnames})
                for row in records:
                    writer.writerow({k: row.get(k, "") for k in merged_fieldnames})
            return

    fieldnames = new_fieldnames
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerows(records)
