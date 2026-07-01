"""Reward-channel extraction for rollout collection."""

from __future__ import annotations

from typing import Any, Dict, List

import torch

from rl.custom_ppo.reward_composition import _compose_training_reward_components


_REWARD_INFO_KEYS = (
    "reward_terminal",
    "reward_offense",
    "reward_pbrs",
    "reward_team",
    "reward_sparse",
    "reward_sparse_points",
    "reward_failure",
    "reward_total",
)


def compose_step_rewards(
    infos: List[Dict[str, Any]],
    *,
    device: torch.device | str,
    hparams: Any,
    shaping_coef: float,
    router_reward_enabled: bool,
) -> Dict[str, torch.Tensor]:
    reward_component = {
        key: torch.as_tensor(
            [float(info.get(key, 0.0) or 0.0) for info in infos],
            dtype=torch.float32,
            device=device,
        )
        for key in _REWARD_INFO_KEYS
    }
    stalemate = torch.as_tensor(
        [bool(info.get("stalemate_truncated", False)) for info in infos],
        dtype=torch.bool,
        device=device,
    )
    result = _compose_training_reward_components(
        reward_component,
        dense_weight=hparams.reward_dense_weight,
        reward_scale=hparams.reward_scale,
        reward_clip=hparams.reward_clip,
        shaping_coef=shaping_coef,
        stalemate=stalemate,
        stalemate_penalty=hparams.reward_stalemate_penalty,
    )
    if router_reward_enabled and any("router_reward" in info for info in infos):
        result["router_reward"] = torch.as_tensor(
            [float(info.get("router_reward", 0.0) or 0.0) for info in infos],
            dtype=torch.float32,
            device=device,
        )
    return result


__all__ = ["compose_step_rewards"]
