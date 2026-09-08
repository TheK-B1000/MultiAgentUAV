"""Observation-schema helpers for map/channel continuity across checkpoints.

``map_a_open`` defaults to 7 CNN channels (no obstacle plane). The active LRO
lineage (V6I23+) was trained with the 8-channel obstacle schema used on
``map_b_*``. Loading those checkpoints into a 7-channel env shape-skips the
CNN stem. Callers that continue an 8-channel lineage on open maps must keep
``obstacle_obs_channel=True`` (the wall plane is zeros on ``map_a_open``).
"""
from __future__ import annotations

from pathlib import Path


def obstacle_obs_channel_for_checkpoint(checkpoint: str | Path) -> bool | None:
    """Infer whether an env must keep the obstacle CNN channel for ``checkpoint``.

    Returns
    -------
    True
        Checkpoint was trained with ≥8 grid channels, explicit obstacle ON, or
        a non-open ``map_layout`` in metadata (V6I23+ map_b lineage).
    False
        Checkpoint was trained with the 7-channel open schema.
    None
        Metadata insufficient; let ``GPUFieldConfig`` derive from map layout.
    """
    try:
        from rl.custom_ppo.inference import read_custom_ppo_metadata
    except Exception:  # noqa: BLE001
        return None

    try:
        meta = read_custom_ppo_metadata(str(checkpoint))
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(meta, dict):
        return None

    ch = meta.get("observation_channels")
    if ch is not None:
        try:
            ch_i = int(ch)
            # Some archives store 0 as a sentinel; treat non-positive as missing.
            if ch_i > 0:
                return ch_i >= 8
        except (TypeError, ValueError):
            pass

    cfg = meta.get("cfg") if isinstance(meta.get("cfg"), dict) else {}
    if isinstance(cfg, dict) and cfg.get("obstacle_obs_channel") is not None:
        return bool(cfg["obstacle_obs_channel"])

    layout = str(meta.get("map_layout") or cfg.get("map_layout") or "").strip()
    if layout:
        try:
            from gpu_env._maps import MAP_A_OPEN, normalize_map_layout

            return normalize_map_layout(layout) != MAP_A_OPEN
        except Exception:  # noqa: BLE001
            pass
    return None


def resolve_obstacle_obs_channel(
    *,
    override: bool | None = None,
    checkpoint: str | Path | None = None,
) -> bool | None:
    """Resolve an obstacle-channel override for env construction.

    Precedence: explicit ``override`` → checkpoint inference → ``None``.
    """
    if override is not None:
        return bool(override)
    if checkpoint is not None:
        return obstacle_obs_channel_for_checkpoint(checkpoint)
    return None


__all__ = [
    "obstacle_obs_channel_for_checkpoint",
    "resolve_obstacle_obs_channel",
]
