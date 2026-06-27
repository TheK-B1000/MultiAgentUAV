"""Action distribution types for the custom PPO policy.

These dataclasses represent the per-head logit structure of the
multi-discrete action space used by SharedActorCentralizedCritic.  They
replace the earlier private _HeadLogits / _MultiHeadDistribution helpers
that lived inside policy.py, and form part of the public
PolicyInferenceContract.

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


@dataclass
class MultiHeadActionDistribution:
    """Per-head logit container for a multi-discrete action space.

    ``heads`` has length ``n_agents * heads_per_agent``.  Each head carries
    logits for one agent × one action dimension (macro or target).

    The ``distributions`` property is a compatibility alias for downstream
    traversal code that iterates over ``.distributions`` (e.g.
    ``_extract_logits`` in eval scripts that expect a SB3-style object).
    """

    heads: List[ActionHead]

    @property
    def distributions(self) -> List[ActionHead]:
        return self.heads

    def __iter__(self) -> Iterator[ActionHead]:
        return iter(self.heads)

    def __len__(self) -> int:
        return len(self.heads)


__all__ = ["ActionHead", "MultiHeadActionDistribution"]
