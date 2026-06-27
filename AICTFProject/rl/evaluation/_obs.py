"""Observation tensor utilities shared by episode runner and probes.

All functions are pure (no side effects, no I/O) and operate on
numpy arrays or PyTorch tensors.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch


def as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def is_done(value: Any) -> bool:
    return bool(as_numpy(value).reshape(-1).all())


def reset_obs(value: Any) -> Any:
    if isinstance(value, tuple) and len(value) == 2:
        return value[0]
    return value


def unpack_step(value: Any) -> tuple[Any, Any, Any, Any]:
    """Normalize env.step() return to (obs, reward, done, info)."""
    if not isinstance(value, tuple):
        raise TypeError(
            f"env.step() returned {type(value)!r}; expected a tuple."
        )
    if len(value) == 4:
        return value  # type: ignore[return-value]
    if len(value) == 5:
        obs, reward, terminated, truncated, info = value
        done = np.logical_or(as_numpy(terminated), as_numpy(truncated))
        return obs, reward, done, info
    raise TypeError(
        f"env.step() returned {len(value)} values; expected 4 or 5."
    )


def to_torch(obs: Any, device: torch.device) -> Any:
    if isinstance(obs, Mapping):
        return {key: to_torch(value, device) for key, value in obs.items()}
    if isinstance(obs, tuple):
        return tuple(to_torch(v, device) for v in obs)
    if isinstance(obs, list):
        return [to_torch(v, device) for v in obs]
    if isinstance(obs, torch.Tensor):
        return obs.to(device)
    return torch.as_tensor(obs, device=device)


def clone_obs(obs: Any) -> Any:
    if isinstance(obs, Mapping):
        return {key: clone_obs(value) for key, value in obs.items()}
    if isinstance(obs, tuple):
        return tuple(clone_obs(v) for v in obs)
    if isinstance(obs, list):
        return [clone_obs(v) for v in obs]
    if isinstance(obs, torch.Tensor):
        return obs.clone()
    return np.array(obs, copy=True)


def spatial_channel_axis(value: Any) -> int | None:
    shape = tuple(getattr(value, "shape", ()))
    if len(shape) == 5:
        return 2
    if len(shape) == 4:
        return 1
    if len(shape) == 3:
        return 0
    return None


def find_spatial_key(obs: Mapping[str, Any]) -> str:
    preferred = ("grid", "cnn", "image", "spatial", "obs_grid")
    for key in preferred:
        value = obs.get(key)
        axis = spatial_channel_axis(value)
        if axis is not None and int(value.shape[axis]) >= 7:
            return key
    for key, value in obs.items():
        axis = spatial_channel_axis(value)
        if axis is not None and int(value.shape[axis]) >= 7:
            return str(key)
    shapes = {str(k): getattr(v, "shape", None) for k, v in obs.items()}
    raise KeyError(
        f"Could not locate the spatial observation tensor. Observed shapes: {shapes}"
    )


def slice_channels(value: Any, channels: int) -> Any:
    axis = spatial_channel_axis(value)
    if axis is None:
        raise ValueError(
            f"Unsupported spatial observation shape: {getattr(value, 'shape', None)}"
        )
    slices = [slice(None)] * len(value.shape)
    slices[axis] = slice(0, channels)
    return value[tuple(slices)]


def adapt_obs_for_policy(obs: Any, expected_channels: int) -> Any:
    """Slice an observation to the channel count the policy expects."""
    if not isinstance(obs, Mapping):
        axis = spatial_channel_axis(obs)
        if axis is None:
            return obs
        actual = int(obs.shape[axis])
        if actual < expected_channels:
            raise ValueError(
                f"Observation has {actual} channels; policy expects {expected_channels}."
            )
        return obs if actual == expected_channels else slice_channels(obs, expected_channels)

    key = find_spatial_key(obs)
    value = obs[key]
    axis = spatial_channel_axis(value)
    assert axis is not None
    actual = int(value.shape[axis])
    if actual < expected_channels:
        raise ValueError(
            f"Observation tensor {key!r} has {actual} channels; "
            f"policy expects {expected_channels}."
        )
    if actual == expected_channels:
        return obs
    adapted = dict(obs)
    adapted[key] = slice_channels(value, expected_channels)
    return adapted


def zero_obstacle_channel(obs: Any) -> tuple[Any, str]:
    """Return (cloned_obs_with_channel7_zeroed, spatial_key)."""
    cloned = clone_obs(obs)

    if isinstance(cloned, Mapping):
        key = find_spatial_key(cloned)
        value = cloned[key]
    else:
        key = "<root>"
        value = cloned

    axis = spatial_channel_axis(value)
    if axis is None:
        raise ValueError(
            f"Unsupported spatial observation shape: {getattr(value, 'shape', None)}"
        )
    channels = int(value.shape[axis])
    if channels < 8:
        raise ValueError(
            f"Obstacle counterfactual requires at least 8 channels; found {channels}."
        )
    slices = [slice(None)] * len(value.shape)
    slices[axis] = 7
    value[tuple(slices)] = 0
    return cloned, key
