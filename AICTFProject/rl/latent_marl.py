"""Pure PyTorch latent-strategy building blocks for the Summer/ICRA implementation.

The authoritative list of how this module relates to the Word spec *Implementation details*
is ``docs/Summer_Implementation_Plan_Implementation_Details_Trace.md`` (and ``docs/rollout_semantics.md`` for the vectorized rollout note).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


from rl.global_state import GLOBAL_STATE_DIM

CONTEXT_STATE_DIM: int = GLOBAL_STATE_DIM * 5


class TemporalStateTracker:
    """
    Tracks exponential moving averages (EMAs) and temporal differences of global states
    to supply richer temporal team/opponent context to q_phi and the centralized critic.
    Private to centralized training / q_phi only; decentralized actor execution is z + local obs.
    """

    def __init__(
        self,
        num_envs: int,
        state_dim: int = GLOBAL_STATE_DIM,
        alpha_short: float = 0.2,
        alpha_long: float = 0.05,
        device: str | torch.device = "cpu",
    ) -> None:
        self.num_envs = int(num_envs)
        self.state_dim = int(state_dim)
        self.alpha_short = float(alpha_short)
        self.alpha_long = float(alpha_long)
        self.device = torch.device(device)

        self.ema_short = torch.zeros((self.num_envs, self.state_dim), dtype=torch.float32, device=self.device)
        self.ema_long = torch.zeros((self.num_envs, self.state_dim), dtype=torch.float32, device=self.device)
        self.initialized = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)

    def reset(self, env_indices: torch.Tensor | None = None) -> None:
        """Reset EMAs and initialization flags for specified environments (or all if None)."""
        if env_indices is None:
            self.initialized.fill_(False)
            self.ema_short.zero_()
            self.ema_long.zero_()
        else:
            idx = env_indices.to(self.device).bool() if env_indices.dtype == torch.bool else env_indices.long()
            self.initialized[idx] = False
            self.ema_short[idx] = 0.0
            self.ema_long[idx] = 0.0

    def update(self, raw_state: torch.Tensor, dones: torch.Tensor | None = None) -> torch.Tensor:
        """
        Update EMAs with new raw global state, resetting any completed environments.
        Returns the concatenated (B, CONTEXT_STATE_DIM) context state.
        """
        raw_state = raw_state.to(self.device).float()
        if raw_state.dim() != 2 or int(raw_state.shape[1]) != int(self.state_dim):
            raise AssertionError(
                f"TemporalStateTracker expected raw_state shape (B, {self.state_dim}), "
                f"got {tuple(raw_state.shape)}"
            )

        # 1. Reset any environments that completed (dones is True)
        if dones is not None:
            done_mask = dones.to(self.device).bool()
            if done_mask.any():
                self.initialized[done_mask] = False
                self.ema_short[done_mask] = 0.0
                self.ema_long[done_mask] = 0.0

        # 2. Reset any environments that started a new episode (decision_frac is 0.0, i.e., index 17)
        if raw_state.shape[1] > 17:
            reset_by_frac = raw_state[:, 17] < 1e-5
            if reset_by_frac.any():
                self.initialized[reset_by_frac] = False
                self.ema_short[reset_by_frac] = 0.0
                self.ema_long[reset_by_frac] = 0.0

        # 3. Initialize EMAs for any uninitialized environments
        uninit_mask = ~self.initialized
        if uninit_mask.any():
            self.ema_short[uninit_mask] = raw_state[uninit_mask]
            self.ema_long[uninit_mask] = raw_state[uninit_mask]
            self.initialized[uninit_mask] = True

        # 4. Exponential moving average update
        self.ema_short = self.alpha_short * raw_state + (1.0 - self.alpha_short) * self.ema_short
        self.ema_long = self.alpha_long * raw_state + (1.0 - self.alpha_long) * self.ema_long

        # 5. Temporal difference calculation (derivatives)
        diff_short = raw_state - self.ema_short
        diff_long = raw_state - self.ema_long

        # 6. Concatenate components to yield (B, CONTEXT_STATE_DIM)
        context = torch.cat(
            [raw_state, self.ema_short, self.ema_long, diff_short, diff_long],
            dim=-1,
        )
        expected_dim = int(self.state_dim) * 5
        if int(context.shape[1]) != expected_dim:
            raise AssertionError(f"temporal context dim {int(context.shape[1])} != expected {expected_dim}")
        return context

    def get_current_context(self, raw_state: torch.Tensor) -> torch.Tensor:
        """
        Passive query: construct the context state using the current EMAs without updating them.
        """
        raw_state = raw_state.to(self.device).float()
        if raw_state.dim() != 2 or int(raw_state.shape[1]) != int(self.state_dim):
            raise AssertionError(
                f"TemporalStateTracker expected raw_state shape (B, {self.state_dim}), "
                f"got {tuple(raw_state.shape)}"
            )
        diff_short = raw_state - self.ema_short
        diff_long = raw_state - self.ema_long
        context = torch.cat(
            [raw_state, self.ema_short, self.ema_long, diff_short, diff_long],
            dim=-1,
        )
        expected_dim = int(self.state_dim) * 5
        if int(context.shape[1]) != expected_dim:
            raise AssertionError(f"temporal context dim {int(context.shape[1])} != expected {expected_dim}")
        return context


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
        self.state_dim = int(state_dim)
        self.latent_k = int(latent_k)
        self.hidden_dim = int(hidden)
        self.net = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.latent_k),
        )

    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        """Return strategy logits with shape ``(B, K)``."""
        if global_state.dim() != 2 or int(global_state.shape[1]) != int(self.state_dim):
            raise AssertionError(
                f"q_phi expected context shape (B, {self.state_dim}), got {tuple(global_state.shape)}"
            )
        return self.net(global_state.float())


class LatentConditionedActor(nn.Module):
    """
    Word doc IMPLEMENTATION §7: ``concat(local_features, z_emb)`` then 256–256 ReLU MLP to logits.

    This is the **canonical** decentralized actor head. Callers (e.g.
    :class:`rl.custom_ppo.policy.SharedActorCentralizedCritic`) own their own
    per-token feature extractor (a CNN, a flatten, etc.) and pass the
    pre-encoded ``local_features`` of width ``local_feature_dim``. Keeping the
    feature extractor outside this module lets the same actor body be reused
    across CNN-backed training and any flatten-only tests/utilities.

    When ``latent_k <= 0`` or ``z_embed_dim <= 0`` the module degrades to a
    plain MLP head with no strategy embedding — used by the no-latent
    baseline. ``forward`` then ignores ``z_idx``.

    No custom init in the spec (default ``Linear`` / ``Embedding`` weights).
    """

    def __init__(
        self,
        local_feature_dim: int,
        latent_k: int,
        action_dim: int,
        *,
        z_embed_dim: int = 16,
        hidden_dim: int = 256,
        z_onehot_enabled: bool = False,
        z_onehot_scale: float = 1.0,
        z_embed_scale: float = 1.0,
        z_adapter_enabled: bool = False,
        z_adapter_scale: float = 0.0,
        z_adapter_init_std: float = 0.02,
        z_film_layers: int = 1,
    ) -> None:
        super().__init__()
        self.local_feature_dim = int(local_feature_dim)
        self.latent_k = max(0, int(latent_k))
        self.z_embed_dim = int(z_embed_dim) if (self.latent_k > 0 and int(z_embed_dim) > 0) else 0
        self.hidden_dim = int(hidden_dim)
        self.action_dim = int(action_dim)
        self.z_onehot_enabled = bool(z_onehot_enabled) and self.latent_k > 0
        self.z_onehot_dim = int(self.latent_k) if self.z_onehot_enabled else 0
        self.z_onehot_scale = (
            float(max(0.0, z_onehot_scale)) if self.z_onehot_enabled else 0.0
        )
        self.z_embed_scale = float(max(0.0, z_embed_scale))
        self.z_adapter_enabled = (
            bool(z_adapter_enabled) and self.latent_k > 0
        )
        self.z_adapter_scale = (
            float(max(0.0, z_adapter_scale)) if self.z_adapter_enabled else 0.0
        )
        self.z_film_layers = (
            max(1, min(2, int(z_film_layers))) if self.z_adapter_enabled else 0
        )

        if self.latent_k > 0 and self.z_embed_dim > 0:
            # Doc IMPLEMENTATION §7: nn.Embedding(K, d_z); no special init in the spec.
            self.strategy_embedding = nn.Embedding(self.latent_k, self.z_embed_dim)
        else:
            self.strategy_embedding = None

        in_dim = self.local_feature_dim + self.z_embed_dim + self.z_onehot_dim
        # Doc IMPLEMENTATION §7: 256–256 MLP; no custom init in the spec (default Linear init).
        self.body = nn.Sequential(
            nn.Linear(in_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        )
        if self.z_adapter_enabled and self.z_adapter_scale > 0.0:
            self.z_adapter = nn.Embedding(self.latent_k, self.hidden_dim * 2)
            nn.init.normal_(
                self.z_adapter.weight,
                mean=0.0,
                std=max(0.0, float(z_adapter_init_std)),
            )
        else:
            self.z_adapter = None
        self.action_head = nn.Linear(self.hidden_dim, self.action_dim)

    def _apply_z_film(self, hidden: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if self.z_adapter is None:
            return hidden
        gamma, beta = self.z_adapter(z).chunk(2, dim=-1)
        return hidden + self.z_adapter_scale * (
            hidden * torch.tanh(gamma) + beta
        )

    def forward(
        self, local_features: torch.Tensor, z_idx: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return per-token logits.

        ``local_features`` must have shape ``(N, local_feature_dim)`` where
        ``N`` is the caller's batch-of-tokens (e.g. ``batch_size * n_agents``).
        ``z_idx`` is required iff the strategy embedding is enabled and must
        broadcast to the same leading dim as ``local_features``.
        """
        if local_features.dim() != 2:
            raise ValueError(
                f"local_features must be (N, local_feature_dim), got {tuple(local_features.shape)}"
            )
        if int(local_features.shape[-1]) != int(self.local_feature_dim):
            raise ValueError(
                f"local_features width {int(local_features.shape[-1])} != local_feature_dim "
                f"{int(self.local_feature_dim)}"
            )
        z = None
        has_z = (self.strategy_embedding is not None) or self.z_onehot_enabled or (self.z_adapter is not None)
        if has_z:
            if z_idx is None:
                raise ValueError("z_idx is required when latent actor z conditioning is enabled.")
            z = z_idx.long().reshape(-1).clamp(
                min=0, max=self.latent_k - 1
            )
            if int(z.shape[0]) != int(local_features.shape[0]):
                raise ValueError(
                    f"z_idx leading dim {int(z.shape[0])} must match local_features leading dim "
                    f"{int(local_features.shape[0])}"
                )
            pieces = [local_features.float()]
            if self.strategy_embedding is not None:
                z_emb = self.strategy_embedding(z) * self.z_embed_scale
                pieces.append(z_emb)
            if self.z_onehot_enabled:
                z_onehot = F.one_hot(z, num_classes=self.latent_k).to(
                    dtype=local_features.dtype,
                    device=local_features.device,
                )
                pieces.append(z_onehot * self.z_onehot_scale)
            x = torch.cat(pieces, dim=-1) if len(pieces) > 1 else local_features.float()
        else:
            x = local_features.float()
        if self.z_adapter is not None and self.z_film_layers >= 2:
            if z is None:
                raise ValueError("z_idx is required when z adapter is enabled.")
            hidden = self.body[0](x)
            hidden = self._apply_z_film(hidden, z)
            hidden = self.body[1](hidden)
            hidden = self.body[2](hidden)
            hidden = self._apply_z_film(hidden, z)
            hidden = self.body[3](hidden)
        else:
            hidden = self.body(x)
            if self.z_adapter is not None:
                if z is None:
                    raise ValueError("z_idx is required when z adapter is enabled.")
                hidden = self._apply_z_film(hidden, z)
        return self.action_head(hidden)


__all__ = [
    "StrategyEncoder",
    "LatentConditionedActor",
    "expected_strategy_switch_penalty",
    "paper_strategy_switch_indicator",
    "TemporalStateTracker",
    "CONTEXT_STATE_DIM",
]
