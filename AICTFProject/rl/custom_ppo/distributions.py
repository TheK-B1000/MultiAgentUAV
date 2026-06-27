"""Action distribution types for the custom PPO policy.

These dataclasses represent the per-head logit structure of the
multi-discrete action space used by ``SharedActorCentralizedCritic``.  They
replace the earlier private ``_HeadLogits`` / ``_MultiHeadDistribution``
helpers that lived inside policy.py, and form part of the public
``PolicyInferenceContract``.

Dependency: no local rl.* imports (pure dataclasses + torch).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List

import torch


@dataclass(frozen=True)
class ActionHead:
    """Logits for one action head (one agent × one action dimension).

    Shape: ``(batch, action_dim)`` where ``action_dim`` is n_macros or
    n_targets.  The tensor lives in the computation graph — callers may call
    ``.softmax(dim=-1)`` or chain ``.backward()`` without detaching.
    """

    logits: torch.Tensor

    @property
    def probabilities(self) -> torch.Tensor:
        """Softmax probabilities, shape ``(batch, action_dim)``."""
        return self.logits.softmax(dim=-1)

    @property
    def argmax_actions(self) -> torch.Tensor:
        """Greedy action indices, shape ``(batch,)``."""
        return self.logits.argmax(dim=-1)

    @property
    def action_dim(self) -> int:
        """Number of discrete actions for this head."""
        return int(self.logits.shape[-1])


@dataclass
class MultiHeadActionDistribution:
    """Per-head logit container for a multi-discrete action space.

    ``heads`` has length ``n_agents * heads_per_agent``.  Each head carries
    logits for one agent × one action dimension (macro or target).

    Public API
    ----------
    Use ``.logits()``, ``.probabilities()``, ``.argmax_actions()`` for
    access to aggregate tensors.  The per-head API on each ``ActionHead``
    is available for head-level computations.

    The ``distributions`` property is a backward-compatibility alias.
    New evaluation code must not use the alias; use ``.heads`` directly.
    """

    heads: List[ActionHead]

    # ------------------------------------------------------------------
    # Aggregate public methods (new API — use these in new code)
    # ------------------------------------------------------------------

    def logits(self) -> list[torch.Tensor]:
        """Return a list of logit tensors, one per head."""
        return [h.logits for h in self.heads]

    def probabilities(self) -> list[torch.Tensor]:
        """Return a list of softmax probability tensors, one per head."""
        return [h.probabilities for h in self.heads]

    def argmax_actions(self) -> list[torch.Tensor]:
        """Return a list of greedy action index tensors, one per head."""
        return [h.argmax_actions for h in self.heads]

    def head_dims(self) -> list[int]:
        """Return the action dimensionality of each head."""
        return [h.action_dim for h in self.heads]

    def num_heads(self) -> int:
        """Return the number of action heads."""
        return len(self.heads)

    # ------------------------------------------------------------------
    # Backward compatibility alias (deprecated — do not use in new code)
    # ------------------------------------------------------------------

    @property
    def distributions(self) -> List[ActionHead]:
        """Backward-compatibility alias for ``heads``.

        Deprecated: use ``.heads`` directly in new code.
        Will be removed in Phase 3.
        """
        return self.heads

    # ------------------------------------------------------------------
    # Container protocol
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[ActionHead]:
        return iter(self.heads)

    def __len__(self) -> int:
        return len(self.heads)


__all__ = ["ActionHead", "MultiHeadActionDistribution"]
