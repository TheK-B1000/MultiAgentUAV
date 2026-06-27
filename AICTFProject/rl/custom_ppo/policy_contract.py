"""Public inference contract for CustomPPO policies.

Any object that implements PolicyInferenceContract can be used by
evaluation scripts and probe code without reaching inside the model via
private attribute paths.

Implementors
------------
``SharedActorCentralizedCritic`` (model)
    Implements the contract directly.  ``get_distribution`` requires an
    explicit ``z_idx`` when the model uses latent strategy.

``CustomPPOInferencePolicy`` (inference wrapper)
    Wraps the model and handles z selection internally.  ``get_distribution``
    without ``z_idx`` uses the wrapper's current latent selection state.

Usage in probe code (explicit z=0)
-----------------------------------
    z_idx = torch.zeros(batch, dtype=torch.long, device=obs["grid"].device)
    dist = model.get_distribution(obs, z_idx=z_idx)

Dependency: distributions.py only (no trainer or rollout imports).
"""
from __future__ import annotations

from typing import Dict, Optional

import torch

try:
    from typing import Protocol, runtime_checkable
except ImportError:  # Python < 3.8
    from typing_extensions import Protocol, runtime_checkable  # type: ignore[assignment]

from .distributions import MultiHeadActionDistribution


@runtime_checkable
class PolicyInferenceContract(Protocol):
    """Minimal public surface required by evaluation and probe code.

    All methods accept the observation dict produced by
    ``RolloutCollector.tensor_obs()`` (keys: ``"grid"``, ``"vec"``,
    ``"agent_mask"``, ``"mask"``).

    Latent selection
    ----------------
    ``SharedActorCentralizedCritic.get_distribution`` raises ``ValueError``
    when ``uses_latent_strategy=True`` and ``z_idx`` is ``None``.  Probe code
    that intentionally evaluates under z=0 must construct the tensor
    explicitly — silent defaults are not permitted in evaluation code
    (constraint #9 of the refactoring spec).

    ``CustomPPOInferencePolicy.get_distribution`` selects z internally and
    does not require ``z_idx`` from the caller.
    """

    def get_distribution(
        self,
        obs: Dict[str, torch.Tensor],
        *,
        z_idx: Optional[torch.Tensor] = None,
    ) -> MultiHeadActionDistribution:
        """Return per-head logit distribution for the given observation.

        Parameters
        ----------
        obs:
            Observation dict (keys: ``"grid"``, ``"vec"``, ``"agent_mask"``,
            ``"mask"``).
        z_idx:
            Integer latent index per batch row, shape ``(B,)``.  Required when
            called on the bare model with latent strategy active.

        Raises
        ------
        ValueError
            If the bare model uses latent strategy and ``z_idx`` is ``None``.
        """
        ...

    def get_cnn_input_weights(self) -> torch.Tensor:
        """Return the first CNN conv-layer weight tensor.

        Shape: ``(out_channels, in_channels, kH, kW)``.

        This is the public alternative to the legacy private path
        ``model.actor_cnn.conv[0].weight`` used in pre-contract probe code.
        """
        ...


__all__ = ["PolicyInferenceContract"]
