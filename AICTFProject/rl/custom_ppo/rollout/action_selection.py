"""Observation tensor conversion used by rollout action selection."""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch


def tensor_obs_dict(
    obs: Dict[str, np.ndarray],
    *,
    device: torch.device | str,
) -> Dict[str, torch.Tensor]:
    return {
        "grid": torch.as_tensor(obs["grid"], dtype=torch.float32, device=device),
        "vec": torch.as_tensor(obs["vec"], dtype=torch.float32, device=device),
        "agent_mask": torch.as_tensor(obs["agent_mask"], dtype=torch.float32, device=device),
        "mask": torch.as_tensor(obs["mask"], dtype=torch.float32, device=device),
    }


__all__ = ["tensor_obs_dict"]
