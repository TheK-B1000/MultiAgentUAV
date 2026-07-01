"""Diagnostics-only protocol for architecture-specific weight access.

Separated from ``PolicyInferenceContract`` because diagnostic access to
internal weight tensors is specific to CNN-based policies and is not
required by all inference policies.

The canonical method name is ``get_observation_encoder_input_weights()``,
which is architecture-neutral (works for CNN or MLP obs encoders as long
as "input weights" is meaningful).  The legacy alias
``get_cnn_input_weights()`` is preserved for backward compatibility;
see the docstring there.

Dependency: ``distributions.py`` only — no trainer or rollout imports.
"""
from __future__ import annotations

import torch

try:
    from typing import Protocol, runtime_checkable
except ImportError:
    from typing_extensions import Protocol, runtime_checkable  # type: ignore[assignment]


@runtime_checkable
class PolicyDiagnosticsContract(Protocol):
    """Architecture-specific diagnostic access to obs-encoder parameters.

    Not all inference policies expose this.  Probe code that needs weight
    or gradient access must ``isinstance(policy, PolicyDiagnosticsContract)``
    before calling any method here.

    For the current CNN-based policy (``SharedActorCentralizedCritic``),
    "observation encoder" refers to ``actor_cnn``.  The input weights are
    the first conv layer, shape ``(out_ch, in_ch, kH, kW)``.

    Future Phase D (router boundary): policies that compose multiple
    encoders should override this method to return the appropriate tensor.
    """

    def get_observation_encoder_input_weights(self) -> torch.Tensor:
        """Return first obs-encoder layer weights, shape (out_ch, in_ch, kH, kW).

        Gradient-preserving — callers must not detach.
        """
        ...


__all__ = ["PolicyDiagnosticsContract"]
