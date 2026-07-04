"""Map training run configs onto forced-z GPU env kwargs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# PPOConfig field -> GPUFieldConfig / RewardConfig field.
_REWARD_FIELD_MAP: tuple[tuple[str, str], ...] = (
    ("env_stalemate_max_steps", "stalemate_max_steps"),
    ("env_surface_score_margin_coef", "surface_score_margin_coef"),
    ("env_surface_blue_capture_tempo_bonus", "surface_blue_capture_tempo_bonus"),
    ("env_surface_red_flag_touch_penalty", "surface_red_flag_touch_penalty"),
    ("env_surface_red_carrier_progress_penalty", "surface_red_carrier_progress_penalty"),
    ("env_surface_blue_near_cap_bonus", "surface_blue_near_cap_bonus"),
)


def find_run_config_for_checkpoint(checkpoint: str | Path) -> Path | None:
    """Best-effort sibling ``*_run_config.json`` next to a checkpoint zip."""
    ckpt = Path(checkpoint)
    parent = ckpt.parent
    if not parent.is_dir():
        return None
    candidates = sorted(parent.glob("*_run_config.json"))
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    stem = ckpt.stem.removeprefix("final_")
    for path in candidates:
        if stem in path.stem:
            return path
    return candidates[0]


def env_reward_kwargs_from_resolved_config(resolved: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for src, dst in _REWARD_FIELD_MAP:
        raw = resolved.get(src)
        if raw is None:
            continue
        if dst == "stalemate_max_steps":
            out[dst] = max(1, int(raw))
        else:
            out[dst] = float(raw)
    return out


def load_run_config_resolved(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("resolved_ppo_config", "resolved_config"):
        resolved = payload.get(key)
        if isinstance(resolved, dict):
            return resolved
    return payload


def resolve_forced_z_env_overrides(
    *,
    checkpoint: str,
    run_config_path: str | Path | None = None,
    inherit_training_config: bool = False,
    max_decision_steps: int | None = None,
) -> tuple[int, dict[str, Any], str | None]:
    """Return ``(max_decision_steps, env_reward_kwargs, run_config_source)``."""
    source: str | None = None
    resolved: dict[str, Any] | None = None

    if run_config_path is not None:
        source = str(run_config_path)
        resolved = load_run_config_resolved(run_config_path)
    elif inherit_training_config:
        discovered = find_run_config_for_checkpoint(checkpoint)
        if discovered is not None:
            source = str(discovered)
            resolved = load_run_config_resolved(discovered)

    steps = int(max_decision_steps) if max_decision_steps is not None else 400
    env_kwargs: dict[str, Any] = {}
    if resolved is not None:
        if max_decision_steps is None and resolved.get("max_decision_steps") is not None:
            steps = max(1, int(resolved["max_decision_steps"]))
        env_kwargs = env_reward_kwargs_from_resolved_config(resolved)
    return steps, env_kwargs, source


__all__ = [
    "env_reward_kwargs_from_resolved_config",
    "find_run_config_for_checkpoint",
    "load_run_config_resolved",
    "resolve_forced_z_env_overrides",
]
