# rl/episode_result.py
"""
Single place to parse env info["episode_result"] into a typed summary.
Use parse_episode_result(info) in all callbacks to avoid duplicate parsing.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional


def path_to_snapshot_key(path: str) -> str:
    """
    Convert a full snapshot path to a stable, machine-independent ID for Elo/logs.
    Example: "checkpoints_sb3/snap_ep100.zip" -> "SNAPSHOT:snap_ep100"
    """
    if not (path or path.strip()):
        return "SNAPSHOT:unknown"
    base = os.path.basename(path.strip())
    name, _ = os.path.splitext(base)
    return f"SNAPSHOT:{name}" if name else "SNAPSHOT:unknown"


@dataclass
class EpisodeSummary:
    """Parsed episode result from info["episode_result"] (and info for phase)."""

    blue_score: int
    red_score: int
    success: int  # 1 if blue won, 0 otherwise
    win_by: int  # blue_score - red_score
    phase_name: str
    opponent_kind: str
    opponent_snapshot: str
    species_tag: str
    scripted_tag: str
    # IROS-style metrics
    time_to_first_score: Optional[float]
    time_to_game_over: Optional[float]
    collisions_per_episode: int
    near_misses_per_episode: int
    collision_free_episode: int
    mean_inter_robot_dist: Optional[float]
    std_inter_robot_dist: Optional[float]
    zone_coverage: float
    vec_schema_version: int = 1  # GameField.VEC_SCHEMA_VERSION; bump when vec schema changes

    def opponent_key(self) -> str:
        """Stable opponent key for league/Elo (SCRIPTED:OP3, SPECIES:BALANCED, SNAPSHOT:snap_ep100)."""
        kind = str(self.opponent_kind or "scripted").upper()
        if kind == "SNAPSHOT":
            return path_to_snapshot_key(self.opponent_snapshot or "")
        if kind == "SPECIES":
            return f"SPECIES:{str(self.species_tag or 'BALANCED').upper()}"
        return f"SCRIPTED:{str(self.scripted_tag or 'OP3').upper()}"


def parse_episode_result(info: Dict[str, Any]) -> Optional[EpisodeSummary]:
    """
    Parse info dict (single-env step info) into EpisodeSummary.
    Returns None if info has no valid episode_result.
    """
    ep = info.get("episode_result")
    if not isinstance(ep, dict):
        return None

    blue_score = int(ep.get("blue_score", 0))
    red_score = int(ep.get("red_score", 0))
    success = int(ep.get("success", 1 if blue_score > red_score else 0))
    if success != 0 and success != 1:
        success = 1 if blue_score > red_score else 0

    # Phase: prefer top-level info["phase"] (canonical), else episode_result
    phase_name = str(info.get("phase", ep.get("phase_name", "")) or "").strip() or str(
        ep.get("phase_name", "")
    ).strip()

    return EpisodeSummary(
        blue_score=blue_score,
        red_score=red_score,
        success=success,
        win_by=blue_score - red_score,
        phase_name=phase_name or "OP3",
        opponent_kind=str(ep.get("opponent_kind", "scripted") or "scripted"),
        opponent_snapshot=str(ep.get("opponent_snapshot", "") or ""),
        species_tag=str(ep.get("species_tag", "BALANCED") or "BALANCED"),
        scripted_tag=str(ep.get("scripted_tag", "") or ""),
        time_to_first_score=ep.get("time_to_first_score"),
        time_to_game_over=ep.get("time_to_game_over"),
        collisions_per_episode=int(ep.get("collisions_per_episode", 0)),
        near_misses_per_episode=int(ep.get("near_misses_per_episode", 0)),
        collision_free_episode=int(
            ep.get("collision_free_episode", 1 if int(ep.get("collision_events_per_episode", ep.get("collisions_per_episode", 0))) == 0 else 0)
        ),
        mean_inter_robot_dist=ep.get("mean_inter_robot_dist"),
        std_inter_robot_dist=ep.get("std_inter_robot_dist"),
        zone_coverage=float(ep.get("zone_coverage", 0.0)) if ep.get("zone_coverage") is not None else 0.0,
        vec_schema_version=int(ep.get("vec_schema_version", 1)),
    )
