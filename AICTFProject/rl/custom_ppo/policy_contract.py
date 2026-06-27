"""Public inference contract for CustomPPO policies.

Any object that implements PolicyInferenceContract can be used by evaluation
scripts and probe code without reaching inside the model via private paths.

Architecture-specific diagnostics (e.g., CNN weight access) live in
``diagnostics_contract.py`` as ``PolicyDiagnosticsContract``.  Probe code
that needs weight access must check:

    isinstance(policy, PolicyDiagnosticsContract)

before calling ``get_observation_encoder_input_weights()``.

Implementors
------------
``SharedActorCentralizedCritic`` (model)
    Implements both contracts.  ``get_distribution`` raises when
    ``uses_latent_strategy=True`` and ``z_idx=None``.

``CustomPPOInferencePolicy`` (inference wrapper)
    Implements both contracts.  ``get_distribution`` without ``z_idx``
    uses the wrapper's current latent-selection state.

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
except ImportError:
    from typing_extensions import Protocol, runtime_checkable  # type: ignore[assignment]

from .distributions import MultiHeadActionDistribution


@runtime_checkable
class PolicyInferenceContract(Protocol):
    """Minimal public surface required for action prediction and distribution access.

    Does NOT include weight / gradient inspection (see PolicyDiagnosticsContract).
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


__all__ = ["PolicyInferenceContract"]
