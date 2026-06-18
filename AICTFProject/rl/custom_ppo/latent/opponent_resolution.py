"""Structured opponent-id resolution from environment info."""

from __future__ import annotations

from typing import Any

from rl.custom_ppo.csv_writers import _opponent_id_int_from_info
from rl.custom_ppo.latent.types import OpponentResolution


def resolve_opponent_id(cfg: Any, info: dict[str, Any]) -> OpponentResolution:
    """Map rollout info to scripted opponent index, or structured failure."""
    er = info.get("episode_result") if isinstance(info.get("episode_result"), dict) else {}
    kind = str(er.get("opponent_kind", info.get("opponent_kind", "scripted")) or "scripted").lower()
    if kind != "scripted":
        return OpponentResolution(value=-1, valid=False, reason="non_scripted_opponent")
    try:
        value = int(_opponent_id_int_from_info(cfg, info))
    except (TypeError, ValueError, KeyError) as exc:
        return OpponentResolution(value=-1, valid=False, reason=f"parse_error:{exc.__class__.__name__}")
    if value < 0:
        return OpponentResolution(value=-1, valid=False, reason="unknown_opponent_tag")
    return OpponentResolution(value=value, valid=True, reason=None)
