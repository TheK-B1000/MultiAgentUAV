#!/usr/bin/env python3
"""V6I9 map-aware competence promotion gate.

This evaluator compares:

* a native 7-channel V6I8 baseline, which receives the environment observation
  with the obstacle channel removed; and
* a native 8-channel V6I9 candidate, which receives the complete observation.

It also selects and verifies each scripted opponent before reset so OP8, OP9,
and OP10 are real evaluation conditions rather than CSV labels.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from game_field_gpu import GPUCTFVecEnv, GPUFieldConfig
from gpu_env._navigation_telemetry import (
    BLOCKED_DISPLACEMENT_THRESHOLD_CELLS,
    MAP_ROUTE_METADATA_VERSION,
    NAVIGATION_TELEMETRY_VERSION,
    ROUTE_CLASSIFIER_VERSION,
    STUCK_CONSECUTIVE_STEP_WINDOW,
    STUCK_DISPLACEMENT_EPSILON_CELLS,
)
from rl.custom_ppo.probe_result import (
    PROBE_ERROR,
    PROBE_SUCCESS,
    CounterfactualProbeResult,
    GradientProbeResult,
    WeightProbeResult,
)
from rl.evaluation.config import config_from_namespace
from rl.evaluation.policy_loader import (
    get_conv0_weight as evaluation_get_conv0_weight,
    get_model as evaluation_get_model,
    load_evaluation_policy,
    policy_device as evaluation_policy_device,
    read_checkpoint_dimensions as evaluation_checkpoint_dimensions,
)
from rl.evaluation.preflight import (
    preflight_distribution_contract as evaluation_preflight_distribution_contract,
    validate_distribution_contract as evaluation_validate_distribution_contract,
)
from rl.evaluation.probes import (
    ObstacleProbeRuntime,
    gradient_probe as evaluation_gradient_probe,
    inspect_obstacle_weights as evaluation_inspect_obstacle_weights,
    obstacle_counterfactual as evaluation_obstacle_counterfactual,
)
from rl.evaluation.episode_runner import (
    EpisodeRunnerRuntime,
    run_episode as evaluation_run_episode,
)
from rl.evaluation.matched_seed import (
    matched_seed_evaluation as evaluation_matched_seed_evaluation,
)
from rl.evaluation.orchestrator import (
    EvaluationRuntime,
    run_evaluation,
)
from rl.evaluation.aggregation import (
    NUMERIC_FIELDS,
    aggregate_conditions as evaluation_aggregate_conditions,
)
from rl.evaluation.gates import build_summary as evaluation_build_summary
from rl.evaluation.artifact_writer import (
    report_text as evaluation_report_text,
    write_csv as evaluation_write_csv,
)


SUPPORTED_OPPONENTS = frozenset({"OP8", "OP9", "OP10"})


# ---------------------------------------------------------------------------
# Manifest helpers (run identity, git, checksums)
# ---------------------------------------------------------------------------

def _git_metadata() -> dict[str, Any]:
    """Return git commit SHA and dirty-tree flag; never raises."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty_out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        dirty = len(dirty_out.strip()) > 0
        return {"git_commit": commit, "git_dirty": dirty}
    except Exception:
        return {"git_commit": None, "git_dirty": None}


def _sha256(path: Path) -> str:
    """Return hex SHA-256 of a file; reads in chunks for large checkpoints."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _runtime_metadata() -> dict[str, Any]:
    import platform
    cuda_version: str | None = None
    if torch.cuda.is_available():
        try:
            cuda_version = torch.version.cuda
        except AttributeError:
            pass
    return {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": cuda_version,
    }


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _done(value: Any) -> bool:
    return bool(_as_numpy(value).reshape(-1).all())


def _reset_obs(value: Any) -> Any:
    if isinstance(value, tuple) and len(value) == 2:
        return value[0]
    return value


def _unpack_step(value: Any) -> tuple[Any, Any, Any, Any]:
    if not isinstance(value, tuple):
        raise TypeError(
            f"env.step() returned {type(value)!r}; expected a tuple."
        )

    if len(value) == 4:
        return value

    if len(value) == 5:
        obs, reward, terminated, truncated, info = value
        done = np.logical_or(_as_numpy(terminated), _as_numpy(truncated))
        return obs, reward, done, info

    raise TypeError(
        f"env.step() returned {len(value)} values; expected 4 or 5."
    )


def _meta_int(
    metadata: Mapping[str, Any],
    names: Sequence[str],
    default: int,
    *,
    positive: bool = False,
) -> int:
    for name in names:
        value = metadata.get(name)
        if value is None:
            continue
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        if positive and parsed <= 0:
            continue
        return parsed
    return int(default)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _normalize_opponent(value: Any) -> str:
    opponent = str(value).strip().upper()
    if opponent.startswith("SCRIPTED:"):
        opponent = opponent.split(":", 1)[1]
    return opponent


def _validate_opponent_name(opponent: str) -> str:
    canonical = _normalize_opponent(opponent)
    if canonical not in SUPPORTED_OPPONENTS:
        raise ValueError(
            f"Unsupported opponent {opponent!r}. "
            f"Expected one of {sorted(SUPPORTED_OPPONENTS)}."
        )
    return canonical


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
    variants = (
        dict(common, n_agents_per_team=n_agents),
        dict(
            common,
            max_blue_agents=n_agents,
            max_red_agents=n_agents,
        ),
    )

    for kwargs in variants:
        try:
            return GPUFieldConfig(**kwargs)
        except TypeError as exc:
            errors.append(str(exc))

    raise TypeError(
        "Could not construct GPUFieldConfig with either supported "
        f"agent-count signature. Errors: {errors}"
    )


def _make_env(
    *,
    n_agents: int,
    map_name: str,
    device: str,
    seed: int,
    max_steps: int,
    instrumented: bool,
) -> GPUCTFVecEnv:
    cfg = _make_config(
        n_agents=n_agents,
        map_name=map_name,
        device=device,
        seed=seed,
        max_steps=max_steps,
    )
    env_type = InstrumentedEnv if instrumented else GPUCTFVecEnv
    return env_type(cfg)


def _checkpoint_dimensions(
    checkpoint_path: str,
) -> tuple[Mapping[str, Any], int, int, int]:
    return evaluation_checkpoint_dimensions(checkpoint_path)


def _load_native_policy(
    checkpoint_path: str,
    *,
    device: str,
    num_cnn_channels: int,
) -> Any:
    """Load a checkpoint using the CNN channel count it was trained with."""
    return load_evaluation_policy(
        "policy",
        checkpoint_path,
        device=device,
        cnn_channels=num_cnn_channels,
    ).policy


def _model(policy: Any) -> torch.nn.Module:
    return evaluation_get_model(policy)


def _validate_distribution_contract(policy: Any, *, label: str) -> None:
    """Fail early when a loaded policy cannot serve public probe distributions."""
    evaluation_validate_distribution_contract(policy, label=label)


def _preflight_distribution_contract(policy: Any, *, label: str) -> None:
    evaluation_preflight_distribution_contract(policy, label=label)


def _conv0_weight(policy: Any) -> torch.nn.Parameter:
    return evaluation_get_conv0_weight(policy)


def _policy_device(policy: Any, fallback: str) -> torch.device:
    return evaluation_policy_device(policy, fallback)


def _spatial_channel_axis(value: Any) -> int | None:
    shape = tuple(getattr(value, "shape", ()))

    if len(shape) == 5:
        return 2
    if len(shape) == 4:
        return 1
    if len(shape) == 3:
        return 0

    return None


def _slice_channels(value: Any, channels: int) -> Any:
    axis = _spatial_channel_axis(value)
    if axis is None:
        raise ValueError(
            f"Unsupported spatial observation shape: "
            f"{getattr(value, 'shape', None)}"
        )

    slices = [slice(None)] * len(value.shape)
    slices[axis] = slice(0, channels)
    return value[tuple(slices)]


def _find_spatial_key(obs: Mapping[str, Any]) -> str:
    preferred = ("grid", "cnn", "image", "spatial", "obs_grid")

    for key in preferred:
        value = obs.get(key)
        axis = _spatial_channel_axis(value)
        if axis is not None and int(value.shape[axis]) >= 7:
            return key

    for key, value in obs.items():
        axis = _spatial_channel_axis(value)
        if axis is not None and int(value.shape[axis]) >= 7:
            return str(key)

    shapes = {
        str(key): getattr(value, "shape", None)
        for key, value in obs.items()
    }
    raise KeyError(
        "Could not locate the spatial observation tensor. "
        f"Observed shapes: {shapes}"
    )


def _adapt_obs_for_policy(obs: Any, policy: Any) -> Any:
    """Slice an 8-channel observation for the native 7-channel baseline."""
    expected_channels = int(_conv0_weight(policy).shape[1])

    if not isinstance(obs, Mapping):
        axis = _spatial_channel_axis(obs)
        if axis is None:
            return obs
        actual_channels = int(obs.shape[axis])
        if actual_channels < expected_channels:
            raise ValueError(
                f"Observation has {actual_channels} CNN channels, "
                f"but policy expects {expected_channels}."
            )
        if actual_channels == expected_channels:
            return obs
        return _slice_channels(obs, expected_channels)

    key = _find_spatial_key(obs)
    value = obs[key]
    axis = _spatial_channel_axis(value)
    assert axis is not None
    actual_channels = int(value.shape[axis])

    if actual_channels < expected_channels:
        raise ValueError(
            f"Observation tensor {key!r} has {actual_channels} CNN "
            f"channels, but policy expects {expected_channels}."
        )

    if actual_channels == expected_channels:
        return obs

    adapted = dict(obs)
    adapted[key] = _slice_channels(value, expected_channels)
    return adapted


def _to_torch(obs: Any, device: torch.device) -> Any:
    if isinstance(obs, Mapping):
        return {
            key: _to_torch(value, device)
            for key, value in obs.items()
        }
    if isinstance(obs, tuple):
        return tuple(_to_torch(value, device) for value in obs)
    if isinstance(obs, list):
        return [_to_torch(value, device) for value in obs]
    if isinstance(obs, torch.Tensor):
        return obs.to(device)
    return torch.as_tensor(obs, device=device)


def _clone_obs(obs: Any) -> Any:
    if isinstance(obs, Mapping):
        return {key: _clone_obs(value) for key, value in obs.items()}
    if isinstance(obs, tuple):
        return tuple(_clone_obs(value) for value in obs)
    if isinstance(obs, list):
        return [_clone_obs(value) for value in obs]
    if isinstance(obs, torch.Tensor):
        return obs.clone()
    return np.array(obs, copy=True)


def _zero_obstacle_channel(obs: Any) -> tuple[Any, str]:
    cloned = _clone_obs(obs)

    if isinstance(cloned, Mapping):
        key = _find_spatial_key(cloned)
        value = cloned[key]
    else:
        key = "<root>"
        value = cloned

    axis = _spatial_channel_axis(value)
    if axis is None:
        raise ValueError(
            f"Unsupported spatial observation shape: "
            f"{getattr(value, 'shape', None)}"
        )

    channels = int(value.shape[axis])
    if channels < 8:
        raise ValueError(
            f"Obstacle counterfactual requires at least 8 channels; "
            f"found {channels}."
        )

    slices = [slice(None)] * len(value.shape)
    slices[axis] = 7
    value[tuple(slices)] = 0

    return cloned, key


def _distribution(policy: Any, obs: Any) -> Any:
    getter = getattr(policy, "get_distribution", None)
    if callable(getter):
        return getter(obs)

    getter = getattr(_model(policy), "get_distribution", None)
    if callable(getter):
        return getter(obs)

    raise AttributeError(
        "Neither the policy nor its model exposes get_distribution()."
    )


def _extract_logits(distribution: Any) -> list[torch.Tensor]:
    found: list[torch.Tensor] = []
    visited: set[int] = set()

    def visit(value: Any) -> None:
        if value is None or id(value) in visited:
            return
        visited.add(id(value))

        logits = getattr(value, "logits", None)
        if isinstance(logits, torch.Tensor):
            found.append(logits)
            return

        for attribute in (
            "distribution",
            "distributions",
            "dists",
            "components",
        ):
            child = getattr(value, attribute, None)
            if isinstance(child, (list, tuple)):
                for item in child:
                    visit(item)
            elif child is not None:
                visit(child)

        if isinstance(value, (list, tuple)):
            for item in value:
                visit(item)

    visit(distribution)

    if not found:
        raise TypeError(
            f"Could not extract logits from distribution type "
            f"{type(distribution)!r}."
        )

    return found


def _predict(policy: Any, obs: Any) -> Any:
    action, _ = policy.predict(obs, deterministic=True)
    return action


def _head_argmax_change_rate(
    real_logits: Sequence[torch.Tensor],
    zero_logits: Sequence[torch.Tensor],
) -> float:
    if len(real_logits) != len(zero_logits):
        raise ValueError(
            "Real and counterfactual distributions have different "
            "numbers of action heads."
        )

    rates = []
    for real, zero in zip(real_logits, zero_logits):
        rate = (
            real.argmax(dim=-1) != zero.argmax(dim=-1)
        ).float().mean()
        rates.append(rate)

    if not rates:
        return 0.0

    return float(torch.stack(rates).mean().detach().cpu().item())


def _positions(core: Any) -> np.ndarray:
    x = _as_numpy(core.blue_x)[0]
    y = _as_numpy(core.blue_y)[0]
    return np.stack((x, y), axis=-1).astype(
        np.float64,
        copy=False,
    )


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
        if isinstance(value, Mapping):
            for key, child in value.items():
                child_prefix = (
                    f"{prefix}.{key}" if prefix else str(key)
                )
                walk(child, child_prefix)
            return

        if isinstance(value, (list, tuple)) and len(value) == 1:
            walk(value[0], prefix)
            return

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
    if isinstance(infos, Mapping):
        return infos
    if isinstance(infos, (list, tuple)) and infos:
        return infos[0]
    if isinstance(infos, np.ndarray) and infos.size:
        return infos.reshape(-1)[0]
    return {}


def _number(
    info: Any,
    aliases: Sequence[str],
) -> float | None:
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


def _bool_value(info: Any, aliases: Sequence[str]) -> bool | None:
    flattened = {
        str(key).lower(): value
        for key, value in _flatten_info(info).items()
    }
    for alias in aliases:
        normalized = alias.lower()
        for key, value in flattened.items():
            if key == normalized or key.endswith("." + normalized):
                if isinstance(value, str):
                    return value.strip().lower() in {"1", "true", "yes", "on"}
                try:
                    return bool(value)
                except Exception:
                    return None
    return None


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None:
        return None
    if denominator <= 0:
        return None
    value = float(numerator) / float(denominator)
    return value if math.isfinite(value) else None


class InstrumentedEnv(GPUCTFVecEnv):
    """Adds conservative navigation proxies when exact telemetry is absent."""

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
            self._stationary_streak = np.zeros(
                len(positions),
                dtype=np.int64,
            )
            self._last_field_side = np.sign(
                positions[:, 0] - center_x
            ).astype(np.int8)
            self._last_route = np.zeros(
                len(positions),
                dtype=np.int8,
            )
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
                movement = np.linalg.norm(
                    positions - self._last_pos,
                    axis=1,
                )
                stationary = movement <= 1e-4

                assert self._stationary_streak is not None
                self._stationary_streak = np.where(
                    stationary,
                    self._stationary_streak + 1,
                    0,
                )

                self.stuck_proxy += int(
                    (self._stationary_streak >= 3).sum()
                )
                self.blocked_proxy += int(
                    (self._stationary_streak == 3).sum()
                )

            field_side = np.sign(
                positions[:, 0] - center_x
            ).astype(np.int8)

            if self._last_field_side is not None:
                crossed_midline = (
                    (field_side != 0)
                    & (self._last_field_side != 0)
                    & (field_side != self._last_field_side)
                )

                for agent_index in np.flatnonzero(crossed_midline):
                    route = (
                        1
                        if positions[agent_index, 1] >= center_y
                        else -1
                    )

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

    def metrics(self, info: Any) -> dict[str, Any]:
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
        blocked = _number(
            info,
            (
                "blue_blocked_movement_events",
                "blocked_movement_events_blue",
                "blocked_movement_events",
            ),
        )
        upper = _number(
            info,
            (
                "blue_upper_lane_steps",
                "blue_route_upper_crossings",
                "route_upper_crossings_blue",
                "blue_upper_lane_crossings",
                "upper_lane_use",
            ),
        )
        lower = _number(
            info,
            (
                "blue_lower_lane_steps",
                "blue_route_lower_crossings",
                "route_lower_crossings_blue",
                "blue_lower_lane_crossings",
                "lower_lane_use",
            ),
        )
        neutral = _number(info, ("blue_neutral_lane_steps",))
        route_switches = _number(
            info,
            (
                "blue_route_switches",
                "route_switches_blue",
                "route_switches",
            ),
        )
        stuck = _number(
            info,
            (
                "blue_stuck_steps",
                "stuck_steps_blue",
                "stuck_steps",
            ),
        )
        repeated_blocked = _number(
            info,
            (
                "blue_repeated_blocked_movement_events",
                "repeated_blocked_movement_events_blue",
                "repeated_blocked_movement",
            ),
        )
        movement_attempts = _number(info, ("blue_movement_attempts",))
        successful_movement = _number(info, ("blue_successful_movement_steps",))
        route_available = _bool_value(info, ("route_telemetry_available",))

        collision_source = "environment_exact" if collisions is not None else "unavailable"
        stuck_source = "environment_exact" if stuck is not None else "evaluator_proxy"
        route_source = (
            "environment_exact"
            if route_available and upper is not None and lower is not None
            else ("evaluator_proxy" if route_available is None else "unavailable")
        )

        stuck_value = stuck if stuck is not None else float(self.stuck_proxy)
        repeated_value = repeated_blocked if repeated_blocked is not None else float(self.blocked_proxy)
        upper_value = upper if route_source == "environment_exact" else (float(self.upper_proxy) if route_source == "evaluator_proxy" else None)
        lower_value = lower if route_source == "environment_exact" else (float(self.lower_proxy) if route_source == "evaluator_proxy" else None)
        neutral_value = neutral if route_source == "environment_exact" else None
        switch_value = route_switches if route_source == "environment_exact" else (float(self.switch_proxy) if route_source == "evaluator_proxy" else None)

        return {
            "wall_collisions": collisions,
            "blocked_movement_events": blocked,
            "stuck_steps": stuck_value,
            "repeated_blocked_movement": repeated_value,
            "upper_lane_use": upper_value,
            "lower_lane_use": lower_value,
            "neutral_lane_use": neutral_value,
            "route_switches": switch_value,
            "movement_attempts": movement_attempts,
            "successful_movement_steps": successful_movement,
            "collision_metric_source": collision_source,
            "stuck_metric_source": stuck_source,
            "route_metric_source": route_source,
            "wall_collision_source": collision_source,
            "stuck_source": stuck_source,
            "route_source": route_source,
            "obstacle_collisions_per_1000_steps": _safe_ratio(collisions * 1000.0 if collisions is not None else None, float(self.steps)),
            "blocked_movements_per_1000_movement_attempts": _safe_ratio(blocked * 1000.0 if blocked is not None else None, movement_attempts),
            "stuck_steps_per_1000_steps": _safe_ratio(stuck_value * 1000.0 if stuck_value is not None else None, float(self.steps)),
            "successful_movement_rate": _safe_ratio(successful_movement, movement_attempts),
            "upper_lane_fraction": _safe_ratio(upper_value, (upper_value or 0.0) + (lower_value or 0.0) + (neutral_value or 0.0) if upper_value is not None and lower_value is not None else None),
            "lower_lane_fraction": _safe_ratio(lower_value, (upper_value or 0.0) + (lower_value or 0.0) + (neutral_value or 0.0) if upper_value is not None and lower_value is not None else None),
            "route_switches_per_episode": switch_value,
            "episode_steps": self.steps,
        }


def _unwrap_env_method_result(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        return value.reshape(-1)[0]

    if isinstance(value, (list, tuple)):
        if not value:
            return None
        return value[0]

    return value


def _get_opponent_key(env: GPUCTFVecEnv) -> str:
    errors: list[str] = []

    env_method = getattr(env, "env_method", None)
    if callable(env_method):
        try:
            result = env_method("get_opponent_key")
            result = _unwrap_env_method_result(result)
            if result is not None:
                return _normalize_opponent(result)
        except Exception as exc:
            errors.append(f"env_method: {exc}")

    core_method = getattr(
        getattr(env, "core", None),
        "get_opponent_key",
        None,
    )
    if callable(core_method):
        try:
            return _normalize_opponent(core_method())
        except Exception as exc:
            errors.append(f"core: {exc}")

    raise RuntimeError(
        "Could not read the environment opponent key. "
        f"Errors: {errors}"
    )


def _set_opponent(
    env: GPUCTFVecEnv,
    opponent: str,
) -> str:
    requested = _validate_opponent_name(opponent)
    errors: list[str] = []

    env_method = getattr(env, "env_method", None)
    if callable(env_method):
        try:
            env_method(
                "set_next_opponent",
                "SCRIPTED",
                requested,
            )
        except Exception as exc:
            errors.append(f"env_method: {exc}")
        else:
            resolved = _get_opponent_key(env)
            if resolved != requested:
                raise RuntimeError(
                    f"Requested opponent {requested}, but environment "
                    f"reported {resolved}."
                )
            return resolved

    core_method = getattr(
        getattr(env, "core", None),
        "set_next_opponent",
        None,
    )
    if callable(core_method):
        try:
            core_method("SCRIPTED", requested)
        except Exception as exc:
            errors.append(f"core: {exc}")
        else:
            resolved = _get_opponent_key(env)
            if resolved != requested:
                raise RuntimeError(
                    f"Requested opponent {requested}, but environment "
                    f"reported {resolved}."
                )
            return resolved

    raise RuntimeError(
        f"Could not select opponent {requested}. Errors: {errors}"
    )


def _preflight_opponents(
    *,
    opponents: Sequence[str],
    n_agents: int,
    map_name: str,
    device: str,
    max_steps: int,
) -> None:
    print("[preflight] Validating scripted opponents...")

    for index, opponent in enumerate(opponents):
        requested = _validate_opponent_name(opponent)
        env = _make_env(
            n_agents=n_agents,
            map_name=map_name,
            device=device,
            seed=9100 + index,
            max_steps=max(8, min(max_steps, 32)),
            instrumented=False,
        )

        try:
            before_reset = _set_opponent(env, requested)
            _reset_obs(env.reset())
            after_reset = _get_opponent_key(env)

            if before_reset != requested or after_reset != requested:
                raise RuntimeError(
                    f"Opponent verification failed for {requested}: "
                    f"before_reset={before_reset}, "
                    f"after_reset={after_reset}."
                )

            print(
                f"[preflight] requested={requested} "
                f"resolved={after_reset} PASS"
            )
        finally:
            env.close()


def _obstacle_probe_runtime() -> ObstacleProbeRuntime:
    return ObstacleProbeRuntime(
        make_env=_make_env,
        model=_model,
        policy_device=_policy_device,
        reset_obs=_reset_obs,
        set_opponent=_set_opponent,
        to_torch=_to_torch,
        zero_obstacle_channel=_zero_obstacle_channel,
        head_argmax_change_rate=_head_argmax_change_rate,
        predict=_predict,
        unpack_step=_unpack_step,
        done=_done,
    )


def inspect_obstacle_weights(policy: Any) -> WeightProbeResult:
    """Return typed weight inspection result via the public diagnostics contract."""
    return evaluation_inspect_obstacle_weights(
        policy,
        runtime=_obstacle_probe_runtime(),
    )


def gradient_probe(
    policy: Any,
    *,
    device: str,
    map_name: str,
    opponent: str,
    n_agents: int,
) -> GradientProbeResult:
    """Measure gradient flow through CNN channel 7 via the public contract."""
    return evaluation_gradient_probe(
        policy,
        runtime=_obstacle_probe_runtime(),
        device=device,
        map_name=map_name,
        opponent=opponent,
        n_agents=n_agents,
    )


def obstacle_counterfactual(
    policy: Any,
    *,
    device: str,
    map_name: str,
    opponent: str,
    n_agents: int,
    steps: int,
) -> CounterfactualProbeResult:
    """Compare real vs. zeroed-obstacle-channel distributions via the public contract."""
    return evaluation_obstacle_counterfactual(
        policy,
        runtime=_obstacle_probe_runtime(),
        device=device,
        map_name=map_name,
        opponent=opponent,
        n_agents=n_agents,
        steps=steps,
    )


def _scores(
    env: GPUCTFVecEnv,
    info: Any,
) -> tuple[float, float]:
    blue_score = _number(
        info,
        ("blue_score", "score_blue"),
    )
    red_score = _number(
        info,
        ("red_score", "score_red"),
    )

    if blue_score is None:
        blue_score = float(
            _as_numpy(env.core.blue_score)
            .reshape(-1)[0]
        )

    if red_score is None:
        red_score = float(
            _as_numpy(env.core.red_score)
            .reshape(-1)[0]
        )

    return blue_score, red_score


def _episode_runner_runtime() -> EpisodeRunnerRuntime:
    return EpisodeRunnerRuntime(
        adapt_obs_for_policy=_adapt_obs_for_policy,
        done=_done,
        first_info=_first_info,
        get_opponent_key=_get_opponent_key,
        make_env=_make_env,
        model=_model,
        predict=_predict,
        reset_obs=_reset_obs,
        scores=_scores,
        set_opponent=_set_opponent,
        unpack_step=_unpack_step,
        validate_opponent_name=_validate_opponent_name,
    )


def run_episode(
    *,
    policy: Any,
    policy_name: str,
    map_name: str,
    opponent: str,
    seed: int,
    device: str,
    n_agents: int,
    max_steps: int,
) -> dict[str, Any]:
    return evaluation_run_episode(
        runtime=_episode_runner_runtime(),
        policy=policy,
        policy_name=policy_name,
        map_name=map_name,
        opponent=opponent,
        seed=seed,
        device=device,
        n_agents=n_agents,
        max_steps=max_steps,
    )


def matched_seed_evaluation(
    args: argparse.Namespace,
    baseline_policy: Any,
    candidate_policy: Any,
    n_agents: int,
) -> list[dict[str, Any]]:
    return evaluation_matched_seed_evaluation(
        args,
        baseline_policy,
        candidate_policy,
        n_agents,
        run_episode_fn=run_episode,
        validate_opponent_name=_validate_opponent_name,
    )


def aggregate_conditions(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return evaluation_aggregate_conditions(rows)


def build_summary(
    args: argparse.Namespace,
    probe: Mapping[str, Any],
    episodes: Sequence[Mapping[str, Any]],
    conditions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return evaluation_build_summary(args, probe, episodes, conditions)


def write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    evaluation_write_csv(path, rows)


def report_text(summary: Mapping[str, Any]) -> str:
    return evaluation_report_text(summary)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="V6I9 map-awareness promotion gate"
    )
    parser.add_argument(
        "--baseline",
        required=True,
        help="Path to the native 7-channel V6I8 checkpoint.",
    )
    parser.add_argument(
        "--candidate",
        required=True,
        help="Path to the native 8-channel V6I9 checkpoint.",
    )
    parser.add_argument(
        "--maps",
        nargs="+",
        default=[
            "map_a_open",
            "map_b_split_lane",
        ],
    )
    parser.add_argument(
        "--opponents",
        nargs="+",
        default=[
            "OP8",
            "OP9",
            "OP10",
        ],
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=7000,
    )
    parser.add_argument(
        "--device",
        default="cpu",
    )
    parser.add_argument(
        "--max-decision-steps",
        type=int,
        default=400,
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/v6i9_map_awareness",
    )
    parser.add_argument(
        "--counterfactual-steps",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--obs-weight-threshold",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--gradient-threshold",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--counterfactual-action-threshold",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--counterfactual-kl-threshold",
        type=float,
        default=1e-5,
    )
    parser.add_argument(
        "--navigation-improvement-threshold",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--route-difference-threshold",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--minimum-win-rate",
        type=float,
        default=0.60,
    )
    parser.add_argument(
        "--competence-retention-tolerance",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--saturation-win-rate",
        type=float,
        default=0.95,
    )
    return parser.parse_args(argv)


def _write_json_text(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(_json_safe(payload), indent=2),
        encoding="utf-8",
    )


def _evaluation_runtime(command: Sequence[str]) -> EvaluationRuntime:
    return EvaluationRuntime(
        project_root=PROJECT_ROOT,
        command=command,
        validate_opponent_name=_validate_opponent_name,
        preflight_opponents=_preflight_opponents,
        preflight_distribution_contract=_preflight_distribution_contract,
        inspect_obstacle_weights=inspect_obstacle_weights,
        gradient_probe=gradient_probe,
        obstacle_counterfactual=obstacle_counterfactual,
        run_episode=run_episode,
        write_json_text=_write_json_text,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = config_from_namespace(args)
    command = sys.argv if argv is None else [str(Path(__file__)), *argv]
    result = run_evaluation(
        config,
        _evaluation_runtime(command),
    )
    return result.exit_code

if __name__ == "__main__":
    raise SystemExit(main())
