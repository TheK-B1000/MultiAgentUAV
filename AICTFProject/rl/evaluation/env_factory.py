"""Environment creation and instrumentation for the map-awareness evaluation.

Centralises all GPUCTFVecEnv construction so that every episode sees an
identical configuration.  ``InstrumentedEnv`` wraps the base env to record
navigation proxies when exact telemetry is absent.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from game_field_gpu import GPUCTFVecEnv, GPUFieldConfig


# ---------------------------------------------------------------------------
# Environment descriptor
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EnvironmentDescriptor:
    """Static properties of a constructed environment."""

    requested_map: str
    resolved_map: str
    requested_opponent: str
    resolved_opponent: str
    seed: int
    n_agents: int
    max_decision_steps: int
    device: str


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def _make_config(
    *,
    n_agents: int,
    map_name: str,
    device: str,
    seed: int,
    max_steps: int,
) -> GPUFieldConfig:
    common = {
        "n_envs": 1,
        "map_layout": map_name,
        "max_decision_steps": max_steps,
        "aquaticus_profile": True,
        "rules_profile": "OURS",
        "device": device,
        "seed": seed,
        "obstacle_obs_channel": True,
    }
    errors: list[str] = []
    for kwargs in (
        dict(common, n_agents_per_team=n_agents),
        dict(common, max_blue_agents=n_agents, max_red_agents=n_agents),
    ):
        try:
            return GPUFieldConfig(**kwargs)
        except TypeError as exc:
            errors.append(str(exc))

    raise TypeError(
        "Could not construct GPUFieldConfig with either supported "
        f"agent-count signature. Errors: {errors}"
    )


def make_env(
    *,
    n_agents: int,
    map_name: str,
    device: str,
    seed: int,
    max_steps: int,
    instrumented: bool = True,
) -> GPUCTFVecEnv:
    """Create and return an (optionally instrumented) environment.

    Fails immediately if the configuration is invalid — no silent fallback.
    """
    cfg = _make_config(
        n_agents=n_agents,
        map_name=map_name,
        device=device,
        seed=seed,
        max_steps=max_steps,
    )
    env_type = InstrumentedEnv if instrumented else GPUCTFVecEnv
    return env_type(cfg)


# ---------------------------------------------------------------------------
# Navigation telemetry helpers (internal to env_factory and InstrumentedEnv)
# ---------------------------------------------------------------------------

def _as_numpy(value: Any) -> np.ndarray:
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _positions(core: Any) -> np.ndarray:
    x = _as_numpy(core.blue_x)[0]
    y = _as_numpy(core.blue_y)[0]
    return np.stack((x, y), axis=-1).astype(np.float64, copy=False)


def _field_center(core: Any) -> tuple[float, float]:
    try:
        blue = _as_numpy(core.blue_flag_home)[0]
        red = _as_numpy(core.red_flag_home)[0]
        return (
            float((blue[0] + red[0]) / 2.0),
            float((blue[1] + red[1]) / 2.0),
        )
    except (AttributeError, IndexError, TypeError):
        cfg = getattr(core, "cfg", None)
        cols = float(getattr(cfg, "cols", 0.0) or 0.0)
        rows = float(getattr(cfg, "rows", 0.0) or 0.0)
        return cols / 2.0, rows / 2.0


def _flatten_info(info: Any) -> dict[str, Any]:
    flattened: dict[str, Any] = {}

    def walk(value: Any, prefix: str = "") -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                child_prefix = f"{prefix}.{key}" if prefix else str(key)
                walk(child, child_prefix)
            return
        if isinstance(value, (list, tuple)) and len(value) == 1:
            walk(value[0], prefix)
            return
        import torch
        if isinstance(value, np.ndarray) and value.size == 1:
            value = value.reshape(-1)[0].item()
        if isinstance(value, torch.Tensor) and value.numel() == 1:
            value = value.detach().cpu().item()
        flattened[prefix] = value
        if prefix:
            flattened.setdefault(prefix.rsplit(".", 1)[-1], value)

    walk(info)
    return flattened


def _first_info(infos: Any) -> Any:
    if isinstance(infos, dict):
        return infos
    if isinstance(infos, (list, tuple)) and infos:
        return infos[0]
    if isinstance(infos, np.ndarray) and infos.size:
        return infos.reshape(-1)[0]
    return {}


def _number(info: Any, aliases: tuple[str, ...]) -> float | None:
    """Look up the first matching alias in flattened episode info."""
    flattened = {
        str(key).lower(): value
        for key, value in _flatten_info(info).items()
    }
    for alias in aliases:
        normalized = alias.lower()
        for key, value in flattened.items():
            if key == normalized or key.endswith("." + normalized):
                try:
                    return float(value)
                except (TypeError, ValueError):
                    continue
    return None


# ---------------------------------------------------------------------------
# InstrumentedEnv
# ---------------------------------------------------------------------------

class InstrumentedEnv(GPUCTFVecEnv):
    """GPUCTFVecEnv with conservative position-proxy navigation metrics.

    When exact telemetry (wall collisions, lane usage) is absent from episode
    info, proxy counters derived from agent position history fill in.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._reset_metrics()

    def _reset_metrics(self) -> None:
        self.stuck_proxy = 0
        self.blocked_proxy = 0
        self.upper_proxy = 0
        self.lower_proxy = 0
        self.switch_proxy = 0
        self.steps = 0
        self._last_pos: np.ndarray | None = None
        self._stationary_streak: np.ndarray | None = None
        self._last_field_side: np.ndarray | None = None
        self._last_route: np.ndarray | None = None

    def reset(self, *args: Any, **kwargs: Any) -> Any:
        result = super().reset(*args, **kwargs)
        self._reset_metrics()
        try:
            positions = _positions(self.core)
            center_x, _ = _field_center(self.core)
            self._last_pos = positions.copy()
            self._stationary_streak = np.zeros(len(positions), dtype=np.int64)
            self._last_field_side = np.sign(positions[:, 0] - center_x).astype(np.int8)
            self._last_route = np.zeros(len(positions), dtype=np.int8)
        except (AttributeError, IndexError, TypeError):
            pass
        return result

    def step(self, actions: Any) -> Any:
        result = super().step(actions)
        self.steps += 1
        try:
            positions = _positions(self.core)
            center_x, center_y = _field_center(self.core)

            if self._last_pos is not None:
                movement = np.linalg.norm(positions - self._last_pos, axis=1)
                stationary = movement <= 1e-4
                assert self._stationary_streak is not None
                self._stationary_streak = np.where(
                    stationary,
                    self._stationary_streak + 1,
                    0,
                )
                self.stuck_proxy += int((self._stationary_streak >= 3).sum())
                self.blocked_proxy += int((self._stationary_streak == 3).sum())

            field_side = np.sign(positions[:, 0] - center_x).astype(np.int8)
            if self._last_field_side is not None:
                crossed_midline = (
                    (field_side != 0)
                    & (self._last_field_side != 0)
                    & (field_side != self._last_field_side)
                )
                for agent_index in np.flatnonzero(crossed_midline):
                    route = 1 if positions[agent_index, 1] >= center_y else -1
                    if route > 0:
                        self.upper_proxy += 1
                    else:
                        self.lower_proxy += 1
                    assert self._last_route is not None
                    previous_route = self._last_route[agent_index]
                    if previous_route and previous_route != route:
                        self.switch_proxy += 1
                    self._last_route[agent_index] = route

            self._last_pos = positions.copy()
            self._last_field_side = field_side
        except (AttributeError, IndexError, TypeError):
            pass
        return result

    def episode_metrics_dict(self, info: Any) -> dict[str, Any]:
        """Return a flat metrics dict with source provenance metadata."""
        collisions = _number(
            info,
            (
                "blue_obstacle_collision_events",
                "obstacle_collision_events_blue",
                "obstacle_collision_events",
                "blue_wall_collisions",
                "wall_collisions",
                "blue_obstacle_collisions",
                "obstacle_collisions",
            ),
        )
        upper = _number(
            info,
            (
                "blue_route_upper_crossings",
                "route_upper_crossings_blue",
                "blue_upper_lane_crossings",
                "upper_lane_use",
            ),
        )
        lower = _number(
            info,
            (
                "blue_route_lower_crossings",
                "route_lower_crossings_blue",
                "blue_lower_lane_crossings",
                "lower_lane_use",
            ),
        )
        route_switches = _number(
            info,
            ("blue_route_switches", "route_switches_blue", "route_switches"),
        )
        stuck = _number(
            info,
            ("blue_stuck_steps", "stuck_steps_blue", "stuck_steps"),
        )
        return {
            "wall_collisions": collisions,
            "stuck_steps": stuck if stuck is not None else float(self.stuck_proxy),
            "repeated_blocked_movement": float(self.blocked_proxy),
            "upper_lane_use": upper if upper is not None else float(self.upper_proxy),
            "lower_lane_use": lower if lower is not None else float(self.lower_proxy),
            "route_switches": (
                route_switches if route_switches is not None else float(self.switch_proxy)
            ),
            "episode_steps": self.steps,
        }

    # Backward compat alias used in old experiment script
    def metrics(self, info: Any) -> dict[str, Any]:
        return self.episode_metrics_dict(info)
