r"""Entity-residual observation repair: g(T, E) added to the existing local
actor input, structured so g(empty, empty) = 0 is a STRUCTURAL fact, not a
trained approximation.

Fusion point (verified by reading rl/custom_ppo/policy.py::_encode_local_obs):

    local_in_base = concat(cnn_features, vec)          shape (B*N, F)
    local_in      = local_in_base + g(teammates, enemies)

``latent_actor`` (the 256-256 MLP + action head) is untouched -- same input
width, so its weights transfer from a pre-repair checkpoint unchanged.
``actor_cnn`` is untouched -- entities never touch grid/vec.

WHY BIAS-FREE, NOT JUST ZERO-INIT (the correction that matters)
    Zero-initializing only the final projection guarantees g=0 at t=0, which is
    valuable (bit-identical warm start) but stops being true the moment training
    moves the weights. The PI's objection was exactly this: "zeros can
    legitimately change the output... if biases [are present]". So every
    ``nn.Linear`` in this module is ``bias=False``, and the empty-entity-set case
    is handled explicitly (mean of zero elements is undefined, not 0 -- guarded
    below). With no bias anywhere and 0 in -> 0 out at every linear/ReLU layer,
    g(empty, empty) = 0 holds for ANY weights, at any point in training, not
    only at initialization. Zero-init of the FINAL layer's weight (kept, on top
    of bias-free) is what gives the SEPARATE, stronger guarantee that g(T, E) = 0
    at t=0 even when T and E are non-empty -- true warm-start equivalence.

THREE ANCHORS (tests/test_entity_residual.py)
    1. EMPTY_IDENTITY   g(empty, empty) == 0, exactly, for random (untrained) weights.
    2. PERMUTATION      shuffling entity order leaves g unchanged.
    3. NO_MUTATION      local_in_base (the CNN/vec path) is bit-identical
                        regardless of entity content -- entities are additive,
                        never a replacement.
"""

from __future__ import annotations

import torch
import torch.nn as nn

ENTITY_FEATURES = 6          # dx, dy, dist, alive, carrying, tagged


class _BiasFreeEntityEncoder(nn.Module):
    """Per-entity MLP, no bias anywhere -> 0 in, 0 out at every layer,
    for any weights. Shared across all entities of one type (teammate or enemy)."""

    def __init__(self, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(ENTITY_FEATURES, hidden, bias=False), nn.ReLU(),
            nn.Linear(hidden, hidden, bias=False), nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _masked_mean_pool(enc: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Mean over the entity axis, restricted to valid entries.

    enc:   (B*N, K, H)   valid: (B*N, K) bool
    Returns (B*N, H). A row with ZERO valid entities returns the EXACT zero
    vector -- guarded explicitly, since a naive mean divides 0/0 into NaN. This
    is half of the empty-identity guarantee (the other half is bias-free layers
    upstream, which is why the encoder output for a masked-out entity is
    already exactly 0 before pooling even begins).
    """
    v = valid.to(enc.dtype).unsqueeze(-1)                       # (B*N, K, 1)
    summed = (enc * v).sum(dim=1)                               # (B*N, H)
    count = v.sum(dim=1).clamp(min=1.0)                         # avoid 0/0
    pooled = summed / count
    has_any = (valid.any(dim=1, keepdim=True)).to(enc.dtype)    # (B*N, 1)
    return pooled * has_any                                     # force exact 0 row


class EntityResidualEncoder(nn.Module):
    """g(teammates, enemies) -> residual vector of width ``out_dim``, added to
    the existing local actor input. Bias-free throughout except the (also
    bias-free) final projection, whose WEIGHT is additionally zero-initialised.

    Shared per-entity encoders + mean pooling make this permutation-invariant
    and size-invariant by construction -- the same module runs unchanged at
    2v2, 4v4, 6v6 (only ``K`` differs).
    """

    def __init__(self, out_dim: int, hidden: int = 32):
        super().__init__()
        self.teammate_enc = _BiasFreeEntityEncoder(hidden)
        self.enemy_enc = _BiasFreeEntityEncoder(hidden)
        self.proj = nn.Linear(2 * hidden, out_dim, bias=False)
        nn.init.zeros_(self.proj.weight)          # g(T,E) = 0 at t=0, ANY T,E

    def forward(self, teammates: torch.Tensor, teammates_valid: torch.Tensor,
                enemies: torch.Tensor, enemies_valid: torch.Tensor) -> torch.Tensor:
        """teammates/enemies: (B*N, K, ENTITY_FEATURES); *_valid: (B*N, K) bool.
        Returns (B*N, out_dim)."""
        t = _masked_mean_pool(self.teammate_enc(teammates), teammates_valid)
        e = _masked_mean_pool(self.enemy_enc(enemies), enemies_valid)
        return self.proj(torch.cat([t, e], dim=-1))


def augmented_local_in(policy, obs: dict, teammates: torch.Tensor,
                       teammates_valid: torch.Tensor, enemies: torch.Tensor,
                       enemies_valid: torch.Tensor, entity_module: EntityResidualEncoder):
    """The actual integration shim: calls the REAL, unmodified
    ``policy._encode_local_obs`` and adds the residual on top. Returns
    ``(local_in_augmented, local_in_base, cnn_features, mask)`` so callers/tests
    can compare the augmented and base paths directly."""
    local_in_base, cnn_features, mask = policy._encode_local_obs(obs)
    g = entity_module(teammates, teammates_valid, enemies, enemies_valid)
    return local_in_base + g, local_in_base, cnn_features, mask
