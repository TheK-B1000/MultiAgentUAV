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
    out = {
        "grid": torch.as_tensor(obs["grid"], dtype=torch.float32, device=device),
        "vec": torch.as_tensor(obs["vec"], dtype=torch.float32, device=device),
        "agent_mask": torch.as_tensor(obs["agent_mask"], dtype=torch.float32, device=device),
        "mask": torch.as_tensor(obs["mask"], dtype=torch.float32, device=device),
    }
    # 4v4 entity-repair: pass through unchanged when absent (legacy obs dicts
    # never carry these keys, so `out` is identical to before this addition).
    if "teammates" in obs:
        out["teammates"] = torch.as_tensor(obs["teammates"], dtype=torch.float32, device=device)
        out["teammates_valid"] = torch.as_tensor(obs["teammates_valid"], dtype=torch.bool, device=device)
        out["enemies"] = torch.as_tensor(obs["enemies"], dtype=torch.float32, device=device)
        out["enemies_valid"] = torch.as_tensor(obs["enemies_valid"], dtype=torch.bool, device=device)
    return out


__all__ = ["tensor_obs_dict"]
