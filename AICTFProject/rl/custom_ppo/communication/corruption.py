"""Communication corruption modes for V6I3 ablation evaluation."""

from __future__ import annotations

from enum import Enum

import numpy as np
import torch


class CommCorruptionMode(str, Enum):
    NORMAL = "normal"
    SILENCE = "silence"
    SHUFFLE = "shuffle"
    RANDOM = "random"
    EXTRA_DELAY = "extra_delay"
    CONSTANT = "constant"


def parse_corruption_mode(raw: str | None) -> CommCorruptionMode:
    value = str(raw or CommCorruptionMode.NORMAL.value).strip().lower()
    try:
        return CommCorruptionMode(value)
    except ValueError as exc:
        allowed = ", ".join(m.value for m in CommCorruptionMode)
        raise ValueError(f"Unknown comm corruption mode {raw!r}; expected one of {allowed}") from exc


def apply_message_channel_corruption(
    channels: torch.Tensor,
    *,
    mode: CommCorruptionMode,
    symbol_marginal: torch.Tensor | None = None,
    constant_symbol: int = 0,
    rng: torch.Generator | None = None,
) -> torch.Tensor:
    """Return corrupted message CNN channels ``(B, Nb, K, H, W)``."""
    if mode == CommCorruptionMode.NORMAL:
        return channels
    out = channels.clone()
    if mode == CommCorruptionMode.SILENCE:
        return torch.zeros_like(out)
    bsz, nb, num_symbols = int(out.shape[0]), int(out.shape[1]), int(out.shape[2])
    flat = out.reshape(bsz, nb, num_symbols, -1)
    if mode == CommCorruptionMode.SHUFFLE:
        for env in range(bsz):
            for recv in range(nb):
                perm = torch.randperm(nb, generator=rng, device=out.device)
                perm = perm[perm != recv]
                if perm.numel() <= 0:
                    continue
                donor = int(perm[0].item())
                flat[env, recv] = flat[env, donor].clone()
        return flat.reshape_as(out)
    if mode == CommCorruptionMode.RANDOM:
        if symbol_marginal is None:
            probs = torch.full((num_symbols,), 1.0 / float(num_symbols), device=out.device)
        else:
            probs = symbol_marginal.float().reshape(-1)[:num_symbols]
            probs = probs / probs.sum().clamp_min(1e-8)
        for env in range(bsz):
            for recv in range(nb):
                sym = int(torch.multinomial(probs, 1, generator=rng).item())
                flat[env, recv] = 0.0
                if flat[env, recv, sym].numel() > 0:
                    flat[env, recv, sym] = flat[env, recv, sym].amax(dim=-1, keepdim=True).expand_as(
                        flat[env, recv, sym]
                    )
        return flat.reshape_as(out)
    if mode == CommCorruptionMode.CONSTANT:
        sym = int(constant_symbol) % max(1, num_symbols)
        out.zero_()
        out[:, :, sym] = 1.0
        return out
    if mode == CommCorruptionMode.EXTRA_DELAY:
        return torch.zeros_like(out)
    return out


def symbol_marginal_from_channels(channels: torch.Tensor, *, num_symbols: int) -> torch.Tensor:
    """Empirical symbol occupancy from message grid channels."""
    bsz, nb, k, _, _ = channels.shape
    counts = torch.zeros((int(num_symbols),), dtype=torch.float32, device=channels.device)
    for sym in range(int(num_symbols)):
        counts[sym] = float(channels[:, :, sym].sum().item())
    total = counts.sum().clamp_min(1e-8)
    return counts / total


__all__ = [
    "CommCorruptionMode",
    "apply_message_channel_corruption",
    "parse_corruption_mode",
    "symbol_marginal_from_channels",
]
