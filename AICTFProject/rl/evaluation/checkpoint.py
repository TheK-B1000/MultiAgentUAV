"""Policy loading utilities for the map-awareness evaluation.

Encapsulates checkpoint introspection so the experiment entry point does not
need to know about CNN channel counts, observation space construction, or
model attribute paths.
"""
from __future__ import annotations

import inspect
from typing import Any, Mapping, Sequence

import torch

from gpu_env._specs import _make_obs_action_spaces
from rl.custom_ppo.inference import load_custom_ppo_policy, read_custom_ppo_metadata


def _meta_int(
    metadata: Mapping[str, Any],
    names: Sequence[str],
    default: int,
) -> int:
    for name in names:
        value = metadata.get(name)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return int(default)


def read_checkpoint_dimensions(
    checkpoint_path: str,
) -> tuple[Mapping[str, Any], int, int, int]:
    """Return (metadata, n_agents, n_macros, n_targets)."""
    metadata = read_custom_ppo_metadata(checkpoint_path)
    n_agents = _meta_int(
        metadata, ("n_blue", "n_agents_per_team", "max_agents", "agents"), 2
    )
    n_macros = _meta_int(
        metadata, ("n_macros", "num_macros", "macro_actions"), 5
    )
    n_targets = _meta_int(
        metadata, ("n_targets", "num_targets", "macro_targets"), 50
    )
    return metadata, n_agents, n_macros, n_targets


def get_model(policy: Any) -> torch.nn.Module:
    """Extract the underlying nn.Module from an inference wrapper."""
    model = getattr(policy, "model", None)
    if model is None:
        raise AttributeError("Loaded policy has no .model attribute.")
    return model


def get_conv0_weight(policy: Any) -> torch.nn.Parameter:
    """Return the first CNN conv layer weight for channel introspection."""
    try:
        return get_model(policy).actor_cnn.conv[0].weight
    except (AttributeError, IndexError, TypeError):
        pass
    for name, parameter in get_model(policy).named_parameters():
        if name.endswith("actor_cnn.conv.0.weight"):
            return parameter
    raise AttributeError(
        "Could not locate actor_cnn.conv.0.weight in the loaded policy."
    )


def load_policy(
    checkpoint_path: str,
    *,
    device: str,
    num_cnn_channels: int,
) -> Any:
    """Load a checkpoint using the CNN channel count it was trained with."""
    _, n_agents, n_macros, n_targets = read_checkpoint_dimensions(checkpoint_path)

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

    sig = inspect.signature(load_custom_ppo_policy)
    kwargs: dict[str, Any] = {}
    if "device" in sig.parameters:
        kwargs["device"] = device

    policy = load_custom_ppo_policy(
        checkpoint_path, observation_space, action_space, **kwargs
    )
    policy.model_path = checkpoint_path

    actual_channels = int(get_conv0_weight(policy).shape[1])
    if actual_channels != num_cnn_channels:
        raise ValueError(
            f"Loaded policy has {actual_channels} CNN channels, "
            f"but {num_cnn_channels} were requested."
        )

    return policy


def policy_device(policy: Any, fallback: str) -> torch.device:
    try:
        return next(get_model(policy).parameters()).device
    except StopIteration:
        return torch.device(fallback)
