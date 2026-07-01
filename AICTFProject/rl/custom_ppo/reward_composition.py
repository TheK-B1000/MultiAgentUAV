"""Pure reward composition for PPO training targets.

This is the CPU-side mirror of the env's GPU reward composition: it takes
the per-component reward tensors (terminal, offense, pbrs, team, sparse,
failure) and produces the scalar ``reward_total`` that PPO trains on,
applying optional shaping-coefficient decay and stalemate penalty plus the
tanh + clip squash.

Kept as a pure function (no trainer / no torch.distributions) so the rollout
collector and tests can import it without importing the trainer (which
would otherwise create a circular import via
``RolloutCollector -> trainer._compose_training_reward_components``).
"""

from __future__ import annotations

from typing import Optional

import torch


def _compose_training_reward_components(
    reward_component: dict[str, torch.Tensor],
    *,
    dense_weight: float,
    reward_scale: float,
    reward_clip: float,
    shaping_coef: float,
    stalemate: Optional[torch.Tensor] = None,
    stalemate_penalty: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Mirror GPU reward scaling for PPO targets after optional shaping decay."""
    out = dict(reward_component)
    coef = float(shaping_coef)
    if abs(coef - 1.0) > 1e-9:
        out["reward_offense"] = out["reward_offense"] * coef
        out["reward_pbrs"] = out["reward_pbrs"] * coef
        out["reward_team"] = out["reward_team"] * coef

    dense = out["reward_pbrs"] + out["reward_team"]
    raw = (
        out["reward_terminal"]
        + out["reward_sparse"]
        + out["reward_failure"]
        + out["reward_offense"]
        + float(dense_weight) * dense
    )
    if stalemate is not None:
        raw = raw + torch.where(
            stalemate.bool(),
            torch.full_like(raw, float(stalemate_penalty)),
            torch.zeros_like(raw),
        )
    scaled = torch.tanh(raw / max(1e-6, float(reward_scale)))
    out["reward_total"] = torch.clamp(scaled, -float(reward_clip), float(reward_clip))
    return out


__all__ = ["_compose_training_reward_components"]
