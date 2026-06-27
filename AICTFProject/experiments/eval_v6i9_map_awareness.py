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
import inspect
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
from gpu_env._specs import _make_obs_action_spaces
from rl.custom_ppo.inference import (
    load_custom_ppo_policy,
    read_custom_ppo_metadata,
)
from rl.custom_ppo.distributions import MultiHeadActionDistribution
from rl.custom_ppo.policy_contract import PolicyInferenceContract
from rl.custom_ppo.probe_result import (
    PROBE_ERROR,
    PROBE_SUCCESS,
    CounterfactualProbeResult,
    GradientProbeResult,
    WeightProbeResult,
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
    metadata = read_custom_ppo_metadata(checkpoint_path)

    n_agents = _meta_int(
        metadata,
        ("n_blue", "n_agents_per_team", "max_agents", "agents"),
        2,
    )
    n_macros = _meta_int(
        metadata,
        ("n_macros", "num_macros", "macro_actions"),
        5,
        positive=True,
    )
    n_targets = _meta_int(
        metadata,
        ("n_targets", "num_targets", "macro_targets"),
        50,
        positive=True,
    )

    return metadata, n_agents, n_macros, n_targets


def _load_native_policy(
    checkpoint_path: str,
    *,
    device: str,
    num_cnn_channels: int,
) -> Any:
    """Load a checkpoint using the CNN channel count it was trained with."""
    _, n_agents, n_macros, n_targets = _checkpoint_dimensions(
        checkpoint_path
    )

    observation_space, action_space = _make_obs_action_spaces(
        n_agents,
        n_macros,
        n_targets,
        num_cnn_channels=num_cnn_channels,
    )

    print(
        f"[load] checkpoint={checkpoint_path} "
        f"channels={num_cnn_channels} agents={n_agents} "
        f"macros={n_macros} targets={n_targets} "
        f"action_logits={n_macros + n_targets}"
    )

    signature = inspect.signature(load_custom_ppo_policy)
    kwargs: dict[str, Any] = {}
    if "device" in signature.parameters:
        kwargs["device"] = device

    policy = load_custom_ppo_policy(
        checkpoint_path,
        observation_space,
        action_space,
        **kwargs,
    )

    policy.model_path = checkpoint_path

    actual_channels = int(_conv0_weight(policy).shape[1])
    if actual_channels != num_cnn_channels:
        raise ValueError(
            f"Loaded policy has {actual_channels} CNN channels, "
            f"but {num_cnn_channels} were requested."
        )

    return policy


def _model(policy: Any) -> torch.nn.Module:
    model = getattr(policy, "model", None)
    if model is None:
        raise AttributeError("Loaded policy has no .model attribute.")
    return model


def _validate_distribution_contract(policy: Any, *, label: str) -> None:
    """Fail early when a loaded policy cannot serve public probe distributions."""
    if not isinstance(policy, PolicyInferenceContract):
        raise TypeError(
            f"{label} policy does not implement PolicyInferenceContract; "
            "loaded checkpoint probes require CustomPPOInferencePolicy.get_distribution()."
        )
    model = _model(policy)
    if not isinstance(model, PolicyInferenceContract):
        raise TypeError(
            f"{label} policy.model does not implement PolicyInferenceContract; "
            "obstacle probes require SharedActorCentralizedCritic.get_distribution()."
        )
    getter = getattr(policy, "get_distribution", None)
    model_getter = getattr(model, "get_distribution", None)
    if not callable(getter) or not callable(model_getter):
        raise TypeError(
            f"{label} policy distribution contract is incomplete; "
            "both wrapper and model must expose get_distribution()."
        )


def _preflight_distribution_contract(policy: Any, *, label: str) -> None:
    _validate_distribution_contract(policy, label=label)
    model = _model(policy)
    device = _policy_device(policy, "cpu")
    grid_shape = tuple(int(v) for v in getattr(model, "grid_shape", ()))
    if len(grid_shape) != 3:
        raise TypeError(f"{label} model has invalid grid_shape={grid_shape!r}.")
    channels, height, width = grid_shape
    n_agents = int(getattr(model, "n_agents", 0) or 1)
    vec_dim = int(getattr(model, "vec_dim", 20) or 20)
    action_dims = tuple(int(v) for v in getattr(model, "action_dims", ()))
    if not action_dims:
        raise TypeError(f"{label} model has no action_dims for distribution preflight.")
    obs = {
        "grid": torch.zeros((1, n_agents, channels, height, width), dtype=torch.float32, device=device),
        "vec": torch.zeros((1, n_agents, vec_dim), dtype=torch.float32, device=device),
        "agent_mask": torch.ones((1, n_agents), dtype=torch.float32, device=device),
        "mask": torch.ones((1, int(sum(action_dims))), dtype=torch.float32, device=device),
    }
    z_idx = None
    if bool(getattr(model, "uses_latent_strategy", False)):
        z_idx = torch.zeros((1,), dtype=torch.long, device=device)
    dist = policy.get_distribution(obs, z_idx=z_idx)
    if not isinstance(dist, MultiHeadActionDistribution):
        raise TypeError(
            f"{label} policy.get_distribution() returned {type(dist).__name__}, "
            "expected MultiHeadActionDistribution."
        )
    if dist.head_dims() != list(action_dims):
        raise TypeError(
            f"{label} distribution head dims {dist.head_dims()} do not match "
            f"model action_dims {list(action_dims)}."
        )


def _conv0_weight(policy: Any) -> torch.nn.Parameter:
    try:
        return _model(policy).actor_cnn.conv[0].weight
    except (AttributeError, IndexError, TypeError):
        pass

    for name, parameter in _model(policy).named_parameters():
        if name.endswith("actor_cnn.conv.0.weight"):
            return parameter

    raise AttributeError(
        "Could not locate actor_cnn.conv.0.weight in the loaded policy."
    )


def _policy_device(policy: Any, fallback: str) -> torch.device:
    try:
        return next(_model(policy).parameters()).device
    except StopIteration:
        return torch.device(fallback)


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


def inspect_obstacle_weights(policy: Any) -> WeightProbeResult:
    """Return typed weight inspection result via the public diagnostics contract."""
    weight = _model(policy).get_observation_encoder_input_weights()
    channels = int(weight.shape[1])

    if channels < 8:
        return WeightProbeResult(
            status=PROBE_SUCCESS,
            has_obstacle_channel=False,
            cnn_channels=channels,
        )

    obstacle_weights = weight[:, 7].detach()
    return WeightProbeResult(
        status=PROBE_SUCCESS,
        has_obstacle_channel=True,
        cnn_channels=channels,
        obstacle_weight_l2=float(
            torch.linalg.vector_norm(obstacle_weights).item()
        ),
        obstacle_weight_abs_mean=float(
            obstacle_weights.abs().mean().item()
        ),
        obstacle_weight_abs_max=float(
            obstacle_weights.abs().max().item()
        ),
        obstacle_weight_nonzero_fraction=float(
            (obstacle_weights.abs() > 0).float().mean().item()
        ),
    )


def gradient_probe(
    policy: Any,
    *,
    device: str,
    map_name: str,
    opponent: str,
    n_agents: int,
) -> GradientProbeResult:
    """Measure gradient flow through CNN channel 7 via the public contract.

    Uses ``model.get_distribution(obs, z_idx=zeros)`` with an explicit z=0
    rather than duck-typed method discovery.  Returns a typed result — metric
    fields are ``None`` (not zero) when the probe fails.
    """
    env = _make_env(
        n_agents=n_agents,
        map_name=map_name,
        device=device,
        seed=4242,
        max_steps=64,
        instrumented=False,
    )
    model = _model(policy)
    was_training = model.training
    model.train()
    model.zero_grad(set_to_none=True)

    try:
        _set_opponent(env, opponent)
        obs = _reset_obs(env.reset())
        obs_t = _to_torch(obs, _policy_device(policy, device))

        batch = int(obs_t["grid"].shape[0])
        # Explicit z=0 — probe evaluates obstacle sensitivity at a fixed latent.
        z_probe = torch.zeros(batch, dtype=torch.long, device=obs_t["grid"].device)
        dist = model.get_distribution(obs_t, z_idx=z_probe)

        diagnostic_loss = sum(
            head.logits.softmax(dim=-1).square().mean()
            for head in dist.heads
        )
        diagnostic_loss.backward()

        weight = model.get_observation_encoder_input_weights()
        if int(weight.shape[1]) < 8:
            return GradientProbeResult(
                status=PROBE_ERROR,
                error="Candidate policy has fewer than 8 CNN input channels.",
            )

        if weight.grad is None:
            return GradientProbeResult(
                status=PROBE_ERROR,
                error="First CNN convolution gradient is None after backward().",
            )

        obstacle_gradient = weight.grad[:, 7]
        return GradientProbeResult(
            status=PROBE_SUCCESS,
            obstacle_gradient_l2=float(
                torch.linalg.vector_norm(obstacle_gradient).item()
            ),
            obstacle_gradient_abs_mean=float(
                obstacle_gradient.abs().mean().item()
            ),
            diagnostic_loss=float(diagnostic_loss.detach().cpu().item()),
        )
    except Exception as exc:
        return GradientProbeResult(
            status=PROBE_ERROR,
            error=f"{type(exc).__name__}: {exc}",
        )
    finally:
        model.zero_grad(set_to_none=True)
        model.train(was_training)
        env.close()


def obstacle_counterfactual(
    policy: Any,
    *,
    device: str,
    map_name: str,
    opponent: str,
    n_agents: int,
    steps: int,
) -> CounterfactualProbeResult:
    """Compare real vs. zeroed-obstacle-channel distributions via the public contract.

    Uses ``model.get_distribution(obs, z_idx=zeros)`` with an explicit z=0.
    Returns a typed result — metric fields are ``None`` (not zero) when the
    probe fails, preventing silent conversion of exceptions into measurements.
    """
    env = _make_env(
        n_agents=n_agents,
        map_name=map_name,
        device=device,
        seed=4343,
        max_steps=max(steps + 8, 64),
        instrumented=False,
    )
    model = _model(policy)
    was_training = model.training
    model.eval()

    kls: list[float] = []
    l2_values: list[float] = []
    change_rates: list[float] = []
    tensor_key: str | None = None

    try:
        _set_opponent(env, opponent)
        obs = _reset_obs(env.reset())

        for _ in range(steps):
            obs_t = _to_torch(obs, _policy_device(policy, device))
            zero_t, tensor_key = _zero_obstacle_channel(obs_t)

            batch = int(obs_t["grid"].shape[0])
            # Explicit z=0 — probe evaluates at a fixed latent for both sides.
            z_probe = torch.zeros(
                batch, dtype=torch.long, device=obs_t["grid"].device
            )

            with torch.no_grad():
                real_dist = model.get_distribution(obs_t, z_idx=z_probe)
                zero_dist = model.get_distribution(zero_t, z_idx=z_probe)

            if len(real_dist.heads) != len(zero_dist.heads):
                raise RuntimeError(
                    "Distribution head count changed during the obstacle counterfactual."
                )

            per_head_kl = []
            per_head_l2 = []

            for real_head, zero_head in zip(real_dist.heads, zero_dist.heads):
                real_lp = real_head.logits.log_softmax(dim=-1)
                zero_lp = zero_head.logits.log_softmax(dim=-1)
                per_head_kl.append(
                    (real_lp.exp() * (real_lp - zero_lp)).sum(dim=-1).mean()
                )
                per_head_l2.append(
                    torch.linalg.vector_norm(
                        real_head.logits - zero_head.logits, dim=-1
                    ).mean()
                )

            kls.append(
                float(torch.stack(per_head_kl).mean().detach().cpu().item())
            )
            l2_values.append(
                float(torch.stack(per_head_l2).mean().detach().cpu().item())
            )
            change_rates.append(
                _head_argmax_change_rate(
                    [h.logits for h in real_dist.heads],
                    [h.logits for h in zero_dist.heads],
                )
            )

            action = _predict(policy, obs)
            obs, _, done, _ = _unpack_step(env.step(action))
            if _done(done):
                break

        if not kls:
            raise RuntimeError("No counterfactual states were evaluated.")

        return CounterfactualProbeResult(
            status=PROBE_SUCCESS,
            states_evaluated=len(kls),
            observation_tensor=tensor_key,
            mean_action_kl=float(np.mean(kls)),
            max_action_kl=float(np.max(kls)),
            mean_logit_l2=float(np.mean(l2_values)),
            max_logit_l2=float(np.max(l2_values)),
            argmax_action_change_rate=float(np.mean(change_rates)),
        )
    except Exception as exc:
        return CounterfactualProbeResult(
            status=PROBE_ERROR,
            states_evaluated=len(kls),
            error=f"{type(exc).__name__}: {exc}",
        )
    finally:
        model.train(was_training)
        env.close()


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
    requested_opponent = _validate_opponent_name(opponent)

    env = _make_env(
        n_agents=n_agents,
        map_name=map_name,
        device=device,
        seed=seed,
        max_steps=max_steps,
        instrumented=True,
    )
    assert isinstance(env, InstrumentedEnv)

    model = _model(policy)
    was_training = model.training
    model.eval()

    try:
        resolved_before_reset = _set_opponent(
            env,
            requested_opponent,
        )
        obs = _reset_obs(env.reset())
        resolved_after_reset = _get_opponent_key(env)

        if resolved_after_reset != requested_opponent:
            raise RuntimeError(
                f"Opponent changed during reset: requested="
                f"{requested_opponent}, before_reset="
                f"{resolved_before_reset}, after_reset="
                f"{resolved_after_reset}."
            )

        last_info: Any = {}
        terminated = False

        for _ in range(max_steps + 8):
            policy_obs = _adapt_obs_for_policy(
                obs,
                policy,
            )
            action = _predict(policy, policy_obs)

            obs, _, done, infos = _unpack_step(
                env.step(action)
            )
            last_info = _first_info(infos)

            if _done(done):
                terminated = True
                break

        if not terminated:
            raise RuntimeError(
                f"Episode did not terminate within "
                f"{max_steps + 8} evaluator steps."
            )

        blue_score, red_score = _scores(
            env,
            last_info,
        )

        return {
            "policy": policy_name,
            "map": map_name,
            "requested_opponent": requested_opponent,
            "resolved_opponent": resolved_after_reset,
            "opponent": resolved_after_reset,
            "seed": seed,
            "blue_score": blue_score,
            "red_score": red_score,
            "win": int(blue_score > red_score),
            "loss": int(blue_score < red_score),
            "draw": int(blue_score == red_score),
            "score_margin": blue_score - red_score,
            **env.metrics(last_info),
        }
    finally:
        model.train(was_training)
        env.close()


def matched_seed_evaluation(
    args: argparse.Namespace,
    baseline_policy: Any,
    candidate_policy: Any,
    n_agents: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    policies = (
        (baseline_policy, "baseline"),
        (candidate_policy, "candidate"),
    )

    total = (
        len(args.maps)
        * len(args.opponents)
        * args.episodes
        * len(policies)
    )
    completed = 0

    for map_name in args.maps:
        for opponent in args.opponents:
            requested = _validate_opponent_name(opponent)

            for episode_index in range(args.episodes):
                seed = args.seed_start + episode_index

                for policy, policy_name in policies:
                    row = run_episode(
                        policy=policy,
                        policy_name=policy_name,
                        map_name=map_name,
                        opponent=requested,
                        seed=seed,
                        device=args.device,
                        n_agents=n_agents,
                        max_steps=args.max_decision_steps,
                    )
                    rows.append(row)
                    completed += 1

                    print(
                        f"[eval] {completed:>4}/{total} "
                        f"policy={policy_name:9s} "
                        f"map={map_name:24s} "
                        f"requested={requested} "
                        f"resolved={row['resolved_opponent']} "
                        f"seed={seed} "
                        f"score={row['blue_score']:.0f}:"
                        f"{row['red_score']:.0f}"
                    )

    return rows


NUMERIC_FIELDS = (
    "blue_score",
    "red_score",
    "win",
    "loss",
    "draw",
    "score_margin",
    "wall_collisions",
    "blocked_movement_events",
    "stuck_steps",
    "repeated_blocked_movement",
    "upper_lane_use",
    "lower_lane_use",
    "neutral_lane_use",
    "route_switches",
    "movement_attempts",
    "successful_movement_steps",
    "obstacle_collisions_per_1000_steps",
    "blocked_movements_per_1000_movement_attempts",
    "stuck_steps_per_1000_steps",
    "successful_movement_rate",
    "upper_lane_fraction",
    "lower_lane_fraction",
    "route_switches_per_episode",
    "episode_steps",
)


def _mean(values: Iterable[Any]) -> float | None:
    numbers: list[float] = []

    for value in values:
        if value is None or value == "":
            continue

        try:
            number = float(value)
        except (TypeError, ValueError):
            continue

        if math.isfinite(number):
            numbers.append(number)

    if not numbers:
        return None

    return float(np.mean(numbers))


def aggregate_conditions(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[
        tuple[str, str, str],
        list[Mapping[str, Any]],
    ] = defaultdict(list)

    for row in rows:
        grouped[
            (
                str(row["policy"]),
                str(row["map"]),
                str(row["resolved_opponent"]),
            )
        ].append(row)

    output: list[dict[str, Any]] = []

    for (
        policy,
        map_name,
        opponent,
    ), group in sorted(grouped.items()):
        aggregate: dict[str, Any] = {
            "policy": policy,
            "map": map_name,
            "requested_opponent": opponent,
            "resolved_opponent": opponent,
            "opponent": opponent,
            "episodes": len(group),
        }

        for field in NUMERIC_FIELDS:
            aggregate[field] = _mean(
                item.get(field) for item in group
            )

        for source_field in (
            "collision_metric_source",
            "stuck_metric_source",
            "route_metric_source",
        ):
            values = sorted({str(item.get(source_field, "unavailable")) for item in group})
            aggregate[source_field] = values[0] if len(values) == 1 else "mixed"

        upper = aggregate.get("upper_lane_use") or 0.0
        lower = aggregate.get("lower_lane_use") or 0.0
        neutral = aggregate.get("neutral_lane_use") or 0.0
        crossings = upper + lower

        aggregate["route_crossings"] = crossings
        lane_total = upper + lower + neutral
        aggregate["upper_lane_fraction"] = (
            upper / lane_total
            if lane_total > 0
            else None
        )
        aggregate["lower_lane_fraction"] = lower / lane_total if lane_total > 0 else None

        output.append(aggregate)

    return output


def _policy_rows(
    rows: Sequence[Mapping[str, Any]],
    policy_name: str,
    obstacle_maps_only: bool = False,
) -> list[Mapping[str, Any]]:
    selected = [
        row
        for row in rows
        if row.get("policy") == policy_name
    ]

    if obstacle_maps_only:
        selected = [
            row
            for row in selected
            if "open" not in str(
                row.get("map", "")
            ).lower()
        ]

    return selected


def _field_mean(
    rows: Sequence[Mapping[str, Any]],
    field: str,
) -> float | None:
    return _mean(row.get(field) for row in rows)


def _improvement_gate(
    baseline: float | None,
    candidate: float | None,
    minimum_reduction: float,
) -> tuple[str, dict[str, Any]]:
    details: dict[str, Any] = {
        "baseline_mean": baseline,
        "candidate_mean": candidate,
        "minimum_reduction_fraction": minimum_reduction,
    }

    if baseline is None or candidate is None:
        details["reason"] = "Required telemetry is unavailable."
        return "INCONCLUSIVE", details

    if baseline <= 0:
        details["reason"] = (
            "Baseline is zero, so relative reduction is undefined."
        )
        return (
            "PASS" if candidate <= 0 else "FAIL",
            details,
        )

    reduction = (baseline - candidate) / baseline
    details["reduction_fraction"] = reduction

    return (
        "PASS"
        if reduction >= minimum_reduction
        else "FAIL",
        details,
    )


def _gate_probe_weight(
    result: WeightProbeResult,
    threshold: float,
) -> dict[str, Any]:
    """Build a gate entry from a WeightProbeResult."""
    if not result.is_success:
        return {
            "status": "ERROR",
            "error": result.error,
        }
    value = result.obstacle_weight_l2
    if value is None:
        return {"status": "INCONCLUSIVE", "error": "weight L2 not measured"}
    return {
        "status": "PASS" if value > threshold else "FAIL",
        "value": value,
        "threshold": threshold,
    }


def _gate_probe_gradient(
    result: GradientProbeResult,
    threshold: float,
) -> dict[str, Any]:
    """Build a gate entry from a GradientProbeResult."""
    if not result.is_success:
        return {
            "status": "ERROR",
            "error": result.error,
        }
    value = result.obstacle_gradient_l2
    if value is None:
        return {"status": "INCONCLUSIVE", "error": "gradient L2 not measured"}
    return {
        "status": "PASS" if value > threshold else "FAIL",
        "value": value,
        "threshold": threshold,
    }


def _gate_probe_counterfactual(
    result: CounterfactualProbeResult,
    action_threshold: float,
    kl_threshold: float,
) -> dict[str, Any]:
    """Build a gate entry from a CounterfactualProbeResult.

    Distinguishes four statuses:
      PASS         — meets either threshold (strong sensitivity)
      WARN         — measurably nonzero but below both thresholds
      FAIL         — effectively zero (channel ignored)
      ERROR/INCONCLUSIVE — probe did not execute correctly
    """
    if not result.is_success:
        return {
            "status": "ERROR",
            "error": result.error,
            "states_evaluated": result.states_evaluated,
        }
    action_change = result.argmax_action_change_rate
    mean_kl = result.mean_action_kl
    mean_l2 = result.mean_logit_l2
    if action_change is None or mean_kl is None:
        return {
            "status": "INCONCLUSIVE",
            "error": "counterfactual metrics not measured",
            "states_evaluated": result.states_evaluated,
        }
    if action_change >= action_threshold or mean_kl >= kl_threshold:
        status = "PASS"
    elif mean_l2 is not None and mean_l2 > 1e-3:
        # Logits changed but not enough to shift the action distribution:
        # model reads the channel weakly — not a dead channel.
        status = "WARN"
    else:
        status = "FAIL"
    return {
        "status": status,
        "argmax_change_rate": action_change,
        "argmax_change_threshold": action_threshold,
        "mean_action_kl": mean_kl,
        "kl_threshold": kl_threshold,
        "mean_logit_l2": mean_l2,
        "states_evaluated": result.states_evaluated,
    }


def build_summary(
    args: argparse.Namespace,
    probe: Mapping[str, Any],
    episodes: Sequence[Mapping[str, Any]],
    conditions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    candidate_weights: WeightProbeResult = probe["candidate_weights"]
    candidate_gradient: GradientProbeResult = probe["candidate_gradient"]
    candidate_counterfactual: CounterfactualProbeResult = probe[
        "candidate_counterfactual"
    ]

    gates: dict[str, Any] = {
        "obstacle_weights_moved": _gate_probe_weight(
            candidate_weights, args.obs_weight_threshold
        ),
        "obstacle_gradient_connected": _gate_probe_gradient(
            candidate_gradient, args.gradient_threshold
        ),
        "obstacle_counterfactual_effect": _gate_probe_counterfactual(
            candidate_counterfactual,
            args.counterfactual_action_threshold,
            args.counterfactual_kl_threshold,
        ),
    }

    baseline_obstacle_rows = _policy_rows(
        episodes,
        "baseline",
        obstacle_maps_only=True,
    )
    candidate_obstacle_rows = _policy_rows(
        episodes,
        "candidate",
        obstacle_maps_only=True,
    )

    obstacle_rows = baseline_obstacle_rows + candidate_obstacle_rows
    exact_wall_telemetry = bool(obstacle_rows) and all(
        row.get("collision_metric_source") == "environment_exact"
        and row.get("wall_collisions") is not None
        for row in obstacle_rows
    )

    if exact_wall_telemetry:
        status, details = _improvement_gate(
            _field_mean(
                baseline_obstacle_rows,
                "wall_collisions",
            ),
            _field_mean(
                candidate_obstacle_rows,
                "wall_collisions",
            ),
            args.navigation_improvement_threshold,
        )
    else:
        status = "INCONCLUSIVE"
        details = {
            "reason": (
                "No exact obstacle collision counter was found "
                "in terminal episode info."
            )
        }

    gates["wall_collisions_improved"] = {
        "status": status,
        "collision_metric_source": "environment_exact" if exact_wall_telemetry else "unavailable",
        **details,
    }

    exact_blocked_telemetry = bool(obstacle_rows) and all(
        row.get("blocked_movement_events") is not None
        for row in obstacle_rows
    )
    if exact_blocked_telemetry:
        status, details = _improvement_gate(
            _field_mean(baseline_obstacle_rows, "blocked_movement_events"),
            _field_mean(candidate_obstacle_rows, "blocked_movement_events"),
            args.navigation_improvement_threshold,
        )
    else:
        status = "INCONCLUSIVE"
        details = {"reason": "Environment blocked-movement telemetry is unavailable."}
    gates["blocked_movement_improved"] = {
        "status": status,
        "stuck_metric_source": "environment_exact" if exact_blocked_telemetry else "unavailable",
        **details,
    }

    exact_stuck_telemetry = bool(obstacle_rows) and all(
        row.get("stuck_metric_source") == "environment_exact"
        and row.get("stuck_steps") is not None
        for row in obstacle_rows
    )
    status, details = _improvement_gate(
        _field_mean(
            baseline_obstacle_rows,
            "stuck_steps",
        ),
        _field_mean(
            candidate_obstacle_rows,
            "stuck_steps",
        ),
        args.navigation_improvement_threshold,
    )
    gates["stuck_behavior_improved"] = {
        "status": status if exact_stuck_telemetry else "INCONCLUSIVE",
        "stuck_metric_source": "environment_exact" if exact_stuck_telemetry else "evaluator_proxy",
        **details,
    }

    candidate_conditions = [
        row
        for row in conditions
        if row.get("policy") == "candidate"
    ]

    route_by_map: dict[str, Any] = {}
    for map_name in args.maps:
        selected = [
            row
            for row in candidate_conditions
            if row.get("map") == map_name
        ]
        route_exact = bool(selected) and all(row.get("route_metric_source") == "environment_exact" for row in selected)
        upper = _field_mean(selected, "upper_lane_use") if route_exact else None
        lower = _field_mean(selected, "lower_lane_use") if route_exact else None
        total = (upper or 0.0) + (lower or 0.0)

        route_by_map[map_name] = {
            "upper_mean": upper,
            "lower_mean": lower,
            "upper_fraction": (
                upper / total
                if total > 0
                else None
            ),
            "route_metric_source": "environment_exact" if route_exact else "unavailable",
        }

    route_fractions = [
        values["upper_fraction"]
        for values in route_by_map.values()
        if values["upper_fraction"] is not None
    ]
    route_difference = (
        max(route_fractions) - min(route_fractions)
        if len(route_fractions) >= 2
        else None
    )

    if route_difference is None:
        route_status = "INCONCLUSIVE"
    elif route_difference >= args.route_difference_threshold:
        route_status = "PASS"
    else:
        route_status = "FAIL"

    gates["map_dependent_routes"] = {
        "status": route_status,
        "route_metric_source": "environment_exact" if route_fractions else "unavailable",
        "max_upper_fraction_difference": route_difference,
        "threshold": args.route_difference_threshold,
        "per_map": route_by_map,
    }

    baseline_win_rate = _field_mean(
        _policy_rows(episodes, "baseline"),
        "win",
    )
    candidate_win_rate = _field_mean(
        _policy_rows(episodes, "candidate"),
        "win",
    )

    competence_pass = (
        baseline_win_rate is not None
        and candidate_win_rate is not None
        and candidate_win_rate >= args.minimum_win_rate
        and candidate_win_rate
        >= baseline_win_rate
        - args.competence_retention_tolerance
    )

    gates["hard_pool_competence_retained"] = {
        "status": (
            "PASS"
            if competence_pass
            else "FAIL"
        ),
        "baseline_win_rate": baseline_win_rate,
        "candidate_win_rate": candidate_win_rate,
        "minimum_candidate_win_rate": args.minimum_win_rate,
        "maximum_allowed_drop": (
            args.competence_retention_tolerance
        ),
    }

    condition_win_rates = [
        float(row["win"])
        for row in candidate_conditions
        if row.get("win") is not None
    ]
    all_saturated = (
        bool(condition_win_rates)
        and all(
            win_rate >= args.saturation_win_rate
            for win_rate in condition_win_rates
        )
    )

    gates["universal_saturation_avoided"] = {
        "status": (
            "FAIL"
            if all_saturated
            else "PASS"
        ),
        "condition_win_rates": condition_win_rates,
        "saturation_threshold": args.saturation_win_rate,
    }

    statuses = [gate["status"] for gate in gates.values()]
    required_statuses = (
        gates["obstacle_weights_moved"]["status"],
        gates["obstacle_gradient_connected"]["status"],
        gates["obstacle_counterfactual_effect"]["status"],
        gates["hard_pool_competence_retained"]["status"],
    )
    navigation_statuses = (
        gates["wall_collisions_improved"]["status"],
        gates["blocked_movement_improved"]["status"],
        gates["stuck_behavior_improved"]["status"],
    )

    if any(s == "ERROR" for s in statuses):
        verdict = "NOT READY FOR STAGE B — PROBE ERROR (see gate details)"
    elif any(s != "PASS" for s in required_statuses):
        verdict = "NOT READY FOR STAGE B"
    elif not any(s == "PASS" for s in navigation_statuses):
        verdict = "INCONCLUSIVE: ADD MISSING TELEMETRY OR MORE EPISODES"
    elif gates["map_dependent_routes"]["status"] == "FAIL":
        verdict = "NOT READY FOR STAGE B"
    elif gates["universal_saturation_avoided"]["status"] == "FAIL":
        verdict = "NOT READY FOR STAGE B - UNIVERSAL SATURATION"
    elif all(s == "PASS" for s in required_statuses):
        verdict = "READY FOR STAGE B"
    elif any(s == "WARN" for s in statuses):
        verdict = "BEHAVIORAL SENSITIVITY WEAK — REVIEW BEFORE PROCEEDING"
    else:
        verdict = "INCONCLUSIVE: ADD MISSING TELEMETRY OR MORE EPISODES"

    return {
        "verdict": verdict,
        "gates": gates,
        "episodes_per_condition": args.episodes,
        "warning": (
            "Use at least 20 episodes per map/opponent "
            "cell for a promotion decision."
            if args.episodes < 20
            else None
        ),
    }


def write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    if not rows:
        return

    fieldnames: list[str] = []
    seen: set[str] = set()

    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def report_text(summary: Mapping[str, Any]) -> str:
    labels = (
        (
            "obstacle_weights_moved",
            "Obstacle weights moved",
        ),
        (
            "obstacle_gradient_connected",
            "Obstacle gradient connected",
        ),
        (
            "obstacle_counterfactual_effect",
            "Obstacle counterfactual effect",
        ),
        (
            "wall_collisions_improved",
            "Wall collisions improved",
        ),
        (
            "blocked_movement_improved",
            "Blocked movement improved",
        ),
        (
            "stuck_behavior_improved",
            "Stuck behavior improved",
        ),
        (
            "map_dependent_routes",
            "Map-dependent routes observed",
        ),
        (
            "hard_pool_competence_retained",
            "Hard-pool competence retained",
        ),
        (
            "universal_saturation_avoided",
            "Universal saturation avoided",
        ),
    )

    lines = [
        "V6I9 MAP-AWARENESS PROMOTION GATE",
        "",
    ]

    for key, label in labels:
        status = summary["gates"][key]["status"]
        lines.append(
            f"{label + ':':36s} {status}"
        )

    lines.extend(
        (
            "",
            f"VERDICT: {summary['verdict']}",
        )
    )

    if summary.get("warning"):
        lines.extend(
            (
                "",
                f"WARNING: {summary['warning']}",
            )
        )

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_directory = Path(args.output_dir)
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    baseline_path = Path(args.baseline)
    candidate_path = Path(args.candidate)

    if not baseline_path.is_file():
        raise FileNotFoundError(
            f"Baseline checkpoint not found: "
            f"{baseline_path}"
        )
    if not candidate_path.is_file():
        raise FileNotFoundError(
            f"Candidate checkpoint not found: "
            f"{candidate_path}"
        )
    if args.episodes < 1:
        raise ValueError("--episodes must be at least 1.")
    if args.max_decision_steps < 1:
        raise ValueError(
            "--max-decision-steps must be at least 1."
        )
    if not args.maps:
        raise ValueError("At least one map is required.")
    if not args.opponents:
        raise ValueError(
            "At least one opponent is required."
        )

    args.opponents = [
        _validate_opponent_name(opponent)
        for opponent in args.opponents
    ]

    baseline_metadata, baseline_agents, _, _ = (
        _checkpoint_dimensions(args.baseline)
    )
    candidate_metadata, candidate_agents, _, _ = (
        _checkpoint_dimensions(args.candidate)
    )

    if baseline_agents != candidate_agents:
        raise ValueError(
            f"Baseline uses {baseline_agents} agents per team, "
            f"but candidate uses {candidate_agents}."
        )

    n_agents = candidate_agents
    reference_map = args.maps[-1]
    reference_opponent = args.opponents[0]

    run_id = str(uuid.uuid4())
    started_at = datetime.now(timezone.utc).isoformat()

    manifest: dict[str, Any] = {
        "schema_version": 3,
        "telemetry_implementation_version": NAVIGATION_TELEMETRY_VERSION,
        "collision_metric_source": "environment_exact_required",
        "stuck_metric_source": "environment_exact_preferred",
        "route_metric_source": "environment_exact_preferred",
        "stuck_epsilon": STUCK_DISPLACEMENT_EPSILON_CELLS,
        "stuck_consecutive_step_window": STUCK_CONSECUTIVE_STEP_WINDOW,
        "blocked_displacement_threshold": BLOCKED_DISPLACEMENT_THRESHOLD_CELLS,
        "route_classifier_version": ROUTE_CLASSIFIER_VERSION,
        "map_route_metadata_version": MAP_ROUTE_METADATA_VERSION,
        "run_id": run_id,
        "started_at": started_at,
        "completed_at": None,  # written at the end
        "status": "in_progress",
        "command": sys.argv,
        "baseline": str(baseline_path),
        "candidate": str(candidate_path),
        "baseline_sha256": _sha256(baseline_path),
        "candidate_sha256": _sha256(candidate_path),
        "baseline_cnn_channels": 7,
        "candidate_cnn_channels": 8,
        "n_agents": n_agents,
        "maps": list(args.maps),
        "opponents": list(args.opponents),
        "episodes": args.episodes,
        "seed_start": args.seed_start,
        "max_decision_steps": args.max_decision_steps,
        "device": args.device,
        "baseline_metadata": baseline_metadata,
        "candidate_metadata": candidate_metadata,
        **_git_metadata(),
        **_runtime_metadata(),
    }
    # Write in-progress manifest immediately so an interrupted run is
    # distinguishable from a completed one.
    manifest_path = output_directory / "evaluation_manifest.json"
    manifest_path.write_text(
        json.dumps(_json_safe(manifest), indent=2),
        encoding="utf-8",
    )

    _preflight_opponents(
        opponents=args.opponents,
        n_agents=n_agents,
        map_name=reference_map,
        device=args.device,
        max_steps=args.max_decision_steps,
    )

    print("Loading native 7-channel baseline checkpoint...")
    baseline_policy = _load_native_policy(
        args.baseline,
        device=args.device,
        num_cnn_channels=7,
    )
    print("... baseline loaded.")

    print("Loading native 8-channel candidate checkpoint...")
    candidate_policy = _load_native_policy(
        args.candidate,
        device=args.device,
        num_cnn_channels=8,
    )
    print("... candidate loaded.")

    print("Preflighting public distribution contract...")
    _preflight_distribution_contract(baseline_policy, label="baseline")
    _preflight_distribution_contract(candidate_policy, label="candidate")
    print("... distribution contract OK.")

    print("Running obstacle probes...")
    probe: dict[str, Any] = {
        "baseline_weights": inspect_obstacle_weights(baseline_policy),
        "candidate_weights": inspect_obstacle_weights(candidate_policy),
        "candidate_gradient": gradient_probe(
            candidate_policy,
            device=args.device,
            map_name=reference_map,
            opponent=reference_opponent,
            n_agents=n_agents,
        ),
        "candidate_counterfactual": obstacle_counterfactual(
            candidate_policy,
            device=args.device,
            map_name=reference_map,
            opponent=reference_opponent,
            n_agents=n_agents,
            steps=args.counterfactual_steps,
        ),
    }
    # Serialize typed probe results via their to_json_dict() methods.
    probe_json = {
        k: v.to_json_dict() if hasattr(v, "to_json_dict") else v
        for k, v in probe.items()
    }
    (output_directory / "obstacle_probe.json").write_text(
        json.dumps(_json_safe(probe_json), indent=2),
        encoding="utf-8",
    )
    print("... obstacle probes complete.")

    print("Running matched-seed evaluation...")
    episodes = matched_seed_evaluation(
        args,
        baseline_policy,
        candidate_policy,
        n_agents,
    )
    conditions = aggregate_conditions(episodes)

    write_csv(output_directory / "episode_results.csv", episodes)
    write_csv(output_directory / "condition_summary.csv", conditions)
    write_csv(output_directory / "per_episode.csv", episodes)
    write_csv(output_directory / "per_condition.csv", conditions)
    print("... matched-seed evaluation complete.")

    summary = build_summary(
        args,
        probe,
        episodes,
        conditions,
    )
    summary_text = json.dumps(_json_safe(summary), indent=2)
    (output_directory / "final_report.json").write_text(summary_text, encoding="utf-8")
    (output_directory / "summary.json").write_text(summary_text, encoding="utf-8")

    report = report_text(summary)
    print("\n" + report)

    (
        output_directory / "final_report.txt"
    ).write_text(
        report + "\n",
        encoding="utf-8",
    )

    # Update manifest to mark the run as completed.
    manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    manifest["status"] = "completed"
    manifest_path.write_text(
        json.dumps(_json_safe(manifest), indent=2),
        encoding="utf-8",
    )

    print(
        f"\nArtifacts written to: "
        f"{output_directory.resolve()}"
    )


if __name__ == "__main__":
    main()
