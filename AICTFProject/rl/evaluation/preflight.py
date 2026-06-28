"""Preflight checks for evaluation policies."""
from __future__ import annotations

from typing import Any

import torch

from rl.custom_ppo.distributions import MultiHeadActionDistribution
from rl.custom_ppo.policy_contract import PolicyInferenceContract
from rl.evaluation.errors import EvaluationPreflightError
from rl.evaluation.policy_loader import get_model, policy_device


def validate_distribution_contract(policy: Any, *, label: str) -> None:
    """Verify wrapper and underlying model expose the public distribution API."""
    if not isinstance(policy, PolicyInferenceContract):
        raise EvaluationPreflightError(
            f"{label} policy does not implement PolicyInferenceContract."
        )
    model = get_model(policy)
    if not isinstance(model, PolicyInferenceContract):
        raise EvaluationPreflightError(
            f"{label} model does not implement PolicyInferenceContract."
        )
    get_distribution = getattr(policy, "get_distribution", None)
    model_get_distribution = getattr(model, "get_distribution", None)
    if not callable(get_distribution):
        raise EvaluationPreflightError(
            f"{label} policy has no callable get_distribution()."
        )
    if not callable(model_get_distribution):
        raise EvaluationPreflightError(
            f"{label} model has no callable get_distribution()."
        )


def validate_distribution_result(
    distribution: Any,
    *,
    label: str,
    expected_head_dims: tuple[int, ...] | None = None,
) -> None:
    """Validate one concrete distribution result without detaching tensors."""
    if not isinstance(distribution, MultiHeadActionDistribution):
        raise EvaluationPreflightError(
            f"{label} get_distribution() returned {type(distribution)!r}; "
            "expected MultiHeadActionDistribution."
        )
    if len(distribution.heads) < 1:
        raise EvaluationPreflightError(
            f"{label} get_distribution() returned no action heads."
        )
    for index, head in enumerate(distribution.heads):
        logits = getattr(head, "logits", None)
        if logits is None:
            raise EvaluationPreflightError(
                f"{label} distribution head {index} has no logits tensor."
            )
    if expected_head_dims is not None:
        actual = distribution.head_dims()
        expected = list(expected_head_dims)
        if actual != expected:
            raise EvaluationPreflightError(
                f"{label} distribution head dims {actual} do not match "
                f"model action_dims {expected}."
            )


def preflight_distribution_contract(policy: Any, *, label: str) -> None:
    """Run the evaluator's concrete distribution preflight on dummy input."""
    validate_distribution_contract(policy, label=label)
    model = get_model(policy)
    device = policy_device(policy, "cpu")
    grid_shape = tuple(int(v) for v in getattr(model, "grid_shape", ()))
    if len(grid_shape) != 3:
        raise EvaluationPreflightError(
            f"{label} model has invalid grid_shape={grid_shape!r}."
        )
    channels, height, width = grid_shape
    n_agents = int(getattr(model, "n_agents", 0) or 1)
    vec_dim = int(getattr(model, "vec_dim", 20) or 20)
    action_dims = tuple(int(v) for v in getattr(model, "action_dims", ()))
    if not action_dims:
        raise EvaluationPreflightError(
            f"{label} model has no action_dims for distribution preflight."
        )
    obs = {
        "grid": torch.zeros(
            (1, n_agents, channels, height, width),
            dtype=torch.float32,
            device=device,
        ),
        "vec": torch.zeros((1, n_agents, vec_dim), dtype=torch.float32, device=device),
        "agent_mask": torch.ones((1, n_agents), dtype=torch.float32, device=device),
        "mask": torch.ones((1, int(sum(action_dims))), dtype=torch.float32, device=device),
    }
    z_idx = None
    if bool(getattr(model, "uses_latent_strategy", False)):
        z_idx = torch.zeros((1,), dtype=torch.long, device=device)
    distribution = policy.get_distribution(obs, z_idx=z_idx)
    validate_distribution_result(
        distribution,
        label=label,
        expected_head_dims=action_dims,
    )


__all__ = [
    "preflight_distribution_contract",
    "validate_distribution_contract",
    "validate_distribution_result",
]
