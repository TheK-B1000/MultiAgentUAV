"""Episode-boundary helpers for rollout collection."""

from __future__ import annotations

from typing import Any, Dict, Tuple


def episode_scores_from_info(info: Dict[str, Any]) -> Tuple[int, int]:
    er = info.get("episode_result")
    if isinstance(er, dict):
        return int(er.get("blue_score", 0)), int(er.get("red_score", 0))
    return int(info.get("blue_score", 0)), int(info.get("red_score", 0))


__all__ = ["episode_scores_from_info"]
