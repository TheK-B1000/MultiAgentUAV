"""Preflight checks for evaluation policies."""
from __future__ import annotations

from typing import Any

from rl.custom_ppo.distributions import MultiHeadActionDistribution
from rl.custom_ppo.policy_contract import PolicyInferenceContract
from rl.evaluation.errors import EvaluationPreflightError
from rl.evaluation.policy_loader import get_model


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


def validate_distribution_result(distribution: Any, *, label: str) -> None:
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


__all__ = ["validate_distribution_contract", "validate_distribution_result"]
