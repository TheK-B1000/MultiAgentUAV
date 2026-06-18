"""Opponent labels for preference telemetry keyed by scripted opponent id."""

from __future__ import annotations

from typing import Any

from rl.custom_ppo.csv_writers import _OPPONENT_ID_TO_TAG


def resolve_logged_opponents(cfg: Any) -> list[tuple[str, int]]:
    """Return (telemetry_label, opponent_id) pairs for preference logging."""
    del cfg
    labels: list[tuple[str, int]] = []
    for opp_id, tag in sorted(_OPPONENT_ID_TO_TAG.items()):
        label = tag.lower().replace("_rusher", "").replace("_switcher", "")
        if label.startswith("op"):
            labels.append((label, int(opp_id)))
    return labels
