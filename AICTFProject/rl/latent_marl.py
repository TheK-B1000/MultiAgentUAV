"""Pure PyTorch latent-strategy building blocks for the Summer/ICRA implementation.

The authoritative list of how this module relates to the Word spec *Implementation details*
is ``docs/Summer_Implementation_Plan_Implementation_Details_Trace.md`` (and ``docs/rollout_semantics.md`` for the vectorized rollout note).
"""

from __future__ import annotations

import torch
import torch.nn as nn


def expected_strategy_switch_penalty(logits: torch.Tensor, prev_z_idx: torch.Tensor) -> torch.Tensor:
    """Legacy differentiable proxy (tests only; trainer uses :func:`paper_strategy_switch_indicator`)."""
    probs = torch.softmax(logits, dim=-1)
    prev = prev_z_idx.long().clamp(min=0, max=probs.shape[-1] - 1).reshape(-1, 1)
    stay_prob = probs.gather(-1, prev).squeeze(-1)
    return 1.0 - stay_prob


def paper_strategy_switch_indicator(z_idx: torch.Tensor, prev_z_idx: torch.Tensor) -> torch.Tensor:
    """``1[z != z_prev]`` as float, same shape as ``z_idx`` (no grad through discrete compare)."""
    z = z_idx.long()
    p = prev_z_idx.long()
    return (z != p).to(dtype=torch.float32)


class StrategyEncoder(nn.Module):
    """
    ``q_\\phi(z | s)`` as in *Summer Implementation Plan.docx* IMPLEMENTATION §4: only
    ``Linear → ReLU → Linear → ReLU → Linear (logits)`` — no custom init in the spec;
    use PyTorch default ``Linear``/``Module`` initialization.
    """

    def __init__(self, state_dim: int, latent_k: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(state_dim), int(hidden)),
            nn.ReLU(),
            nn.Linear(int(hidden), int(hidden)),
            nn.ReLU(),
            nn.Linear(int(hidden), int(latent_k)),
        )

    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        """Return strategy logits with shape ``(B, K)``."""
        return self.net(global_state.float())


class LatentConditionedActor(nn.Module):
    """
    Word doc IMPLEMENTATION §7: ``concat(local_obs, z_emb)`` then 256–256 ReLU MLP to logits.
    No custom init in the spec (default ``Linear`` weights).
    """

    def __init__(
        self,
        obs_shape: tuple[int, int, int],
        vec_dim: int,
        latent_k: int,
        action_dim: int,
        *,
        z_embed_dim: int = 16,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        c, h, w = (int(x) for x in obs_shape)
        self._flat_dim = c * h * w
        self.strategy_embedding = nn.Embedding(int(latent_k), int(z_embed_dim))
        in_dim = self._flat_dim + int(vec_dim) + int(z_embed_dim)
        self.body = nn.Sequential(
            nn.Linear(in_dim, int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.ReLU(),
        )
        self.action_head = nn.Linear(int(hidden_dim), int(action_dim))

    def forward(self, grid: torch.Tensor, vec: torch.Tensor, z_idx: torch.Tensor) -> torch.Tensor:
        """Return per-agent logits from local observations and shared strategy indices."""
        if grid.dim() != 4:
            raise ValueError(f"grid must be (B, C, H, W), got {tuple(grid.shape)}")
        if vec.dim() != 2:
            raise ValueError(f"vec must be (B, V), got {tuple(vec.shape)}")
        z = z_idx.long().reshape(-1).clamp(min=0, max=self.strategy_embedding.num_embeddings - 1)
        z_emb = self.strategy_embedding(z)
        flat = grid.float().reshape(grid.shape[0], -1)
        return self.action_head(self.body(torch.cat([flat, vec.float(), z_emb], dim=-1)))


__all__ = [
    "StrategyEncoder",
    "LatentConditionedActor",
    "expected_strategy_switch_penalty",
    "paper_strategy_switch_indicator",
]
