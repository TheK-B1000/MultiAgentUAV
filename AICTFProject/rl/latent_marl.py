"""Pure PyTorch latent-strategy building blocks for the Summer/ICRA implementation."""

from __future__ import annotations

import torch
import torch.nn as nn

from rl.networks import CNNEncoder, orthogonal_init


def expected_strategy_switch_penalty(logits: torch.Tensor, prev_z_idx: torch.Tensor) -> torch.Tensor:
    """Return the differentiable proxy ``E[1(z_t != z_{t-1})]``."""
    probs = torch.softmax(logits, dim=-1)
    prev = prev_z_idx.long().clamp(min=0, max=probs.shape[-1] - 1).reshape(-1, 1)
    stay_prob = probs.gather(-1, prev).squeeze(-1)
    return 1.0 - stay_prob


class StrategyEncoder(nn.Module):
    """Global-state encoder ``q_phi(z | s)`` with the locked 128-128 MLP."""

    def __init__(self, state_dim: int, latent_k: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(state_dim), int(hidden)),
            nn.ReLU(),
            nn.Linear(int(hidden), int(hidden)),
            nn.ReLU(),
            nn.Linear(int(hidden), int(latent_k)),
        )
        self.net.apply(orthogonal_init)
        orthogonal_init(self.net[-1], gain=0.01)

    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        """Return strategy logits with shape ``(B, K)``."""
        return self.net(global_state.float())


class LatentConditionedActor(nn.Module):
    """Shared per-agent actor ``pi_i(a_i | o_i, z)`` for future PPO integration."""

    def __init__(
        self,
        obs_shape: tuple[int, int, int],
        vec_dim: int,
        latent_k: int,
        action_dim: int,
        *,
        z_embed_dim: int = 16,
        feature_dim: int = 256,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.cnn = CNNEncoder(obs_shape, feature_dim=int(feature_dim))
        self.strategy_embedding = nn.Embedding(int(latent_k), int(z_embed_dim))
        self.body = nn.Sequential(
            nn.Linear(int(feature_dim) + int(vec_dim) + int(z_embed_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.ReLU(),
        )
        self.action_head = nn.Linear(int(hidden_dim), int(action_dim))
        self.body.apply(orthogonal_init)
        orthogonal_init(self.action_head, gain=0.01)

    def forward(self, grid: torch.Tensor, vec: torch.Tensor, z_idx: torch.Tensor) -> torch.Tensor:
        """Return per-agent logits from local observations and shared strategy indices."""
        if grid.dim() != 4:
            raise ValueError(f"grid must be (B, C, H, W), got {tuple(grid.shape)}")
        if vec.dim() != 2:
            raise ValueError(f"vec must be (B, V), got {tuple(vec.shape)}")
        z = z_idx.long().reshape(-1).clamp(min=0, max=self.strategy_embedding.num_embeddings - 1)
        z_emb = self.strategy_embedding(z)
        features = self.cnn(grid.float())
        return self.action_head(self.body(torch.cat([features, vec.float(), z_emb], dim=-1)))


__all__ = ["StrategyEncoder", "LatentConditionedActor", "expected_strategy_switch_penalty"]
