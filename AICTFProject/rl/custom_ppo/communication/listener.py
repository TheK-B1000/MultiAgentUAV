"""Listener-response diagnostics for V6I3."""

from __future__ import annotations

import math
from typing import Any

import torch


def _jsd(p: torch.Tensor, q: torch.Tensor) -> float:
    p = p.clamp_min(1e-8)
    q = q.clamp_min(1e-8)
    m = 0.5 * (p + q)
    kl_pm = (p * (p.log() - m.log())).sum()
    kl_qm = (q * (q.log() - m.log())).sum()
    return float((0.5 * kl_pm + 0.5 * kl_qm).item())


def inject_message_symbol_into_grid(
    grid: torch.Tensor,
    *,
    receiver_agent: int,
    symbol: int,
    num_symbols: int,
    base_channels: int,
) -> torch.Tensor:
    """Overwrite receiver message channels with a single symbol hotspot."""
    out = grid.clone()
    msg_start = int(base_channels)
    msg_end = msg_start + int(num_symbols)
    if int(out.shape[2]) < msg_end:
        return out
    out[:, receiver_agent, msg_start:msg_end] = 0.0
    sym = int(symbol) % int(num_symbols)
    out[:, receiver_agent, msg_start + sym, 0, 0] = 1.0
    return out


def receiver_macro_jsd_by_message(
    model: Any,
    obs_batch: dict[str, torch.Tensor],
    *,
    z_idx: torch.Tensor,
    receiver_agent: int = 0,
    num_symbols: int = 4,
    base_channels: int | None = None,
) -> dict[str, float]:
    """Intervene on received message symbol; compare macro distributions."""
    grid = obs_batch["grid"]
    if base_channels is None:
        base_channels = int(grid.shape[2]) - int(num_symbols)
    probs: list[torch.Tensor] = []
    device = grid.device
    batch = int(grid.shape[0])
    for sym in range(int(num_symbols)):
        obs_i = dict(obs_batch)
        obs_i["grid"] = inject_message_symbol_into_grid(
            grid,
            receiver_agent=int(receiver_agent),
            symbol=sym,
            num_symbols=int(num_symbols),
            base_channels=int(base_channels),
        )
        logits = model.policy_logits(obs_i, z_idx=z_idx)
        per_agent_start = int(receiver_agent) * int(model.per_agent_logits)
        macro_dim = int(model.per_agent_action_dims[0])
        macro_logits = logits[:, per_agent_start : per_agent_start + macro_dim]
        probs.append(torch.softmax(macro_logits, dim=-1))
    pair_jsds: list[float] = []
    for i in range(int(num_symbols)):
        for j in range(i + 1, int(num_symbols)):
            pair_jsds.append(_jsd(probs[i].mean(dim=0), probs[j].mean(dim=0)))
    argmax_disagree = 0
    for b in range(batch):
        picks = [int(p[b].argmax().item()) for p in probs]
        if len(set(picks)) > 1:
            argmax_disagree += 1
    return {
        "receiver_action_jsd_by_message_pair_mean": float(sum(pair_jsds) / max(1, len(pair_jsds))),
        "receiver_argmax_disagreement_frac": float(argmax_disagree / max(1, batch)),
        "receiver_listener_pairs": float(len(pair_jsds)),
    }


def rollout_listener_telemetry(
    model: Any,
    buffer: Any,
    *,
    cfg: Any,
    max_rows: int = 64,
) -> dict[str, float]:
    if not bool(getattr(model, "communication_enabled", False)):
        return {}
    if "obs_grid" not in buffer.fields or "z" not in buffer.fields:
        return {}
    length = int(buffer.pos)
    if length <= 0:
        return {}
    total = length * int(buffer.n_envs)
    take = min(int(max_rows), int(total))
    idx = torch.arange(take, device=buffer.device)
    obs_batch = {
        "grid": buffer.fields["obs_grid"][:length].reshape(total, *buffer.fields["obs_grid"].shape[2:]).index_select(0, idx),
        "vec": buffer.fields["obs_vec"][:length].reshape(total, *buffer.fields["obs_vec"].shape[2:]).index_select(0, idx),
        "agent_mask": buffer.fields["obs_agent_mask"][:length].reshape(total, *buffer.fields["obs_agent_mask"].shape[2:]).index_select(0, idx),
    }
    if "obs_mask" in buffer.fields:
        obs_batch["mask"] = buffer.fields["obs_mask"][:length].reshape(total, *buffer.fields["obs_mask"].shape[2:]).index_select(0, idx)
    z_idx = buffer.fields["z"][:length].reshape(total).index_select(0, idx)
    num_symbols = int(getattr(cfg, "comm_num_symbols", 4) or 4)
    base_channels = int(model.grid_shape[0]) - num_symbols
    return receiver_macro_jsd_by_message(
        model,
        obs_batch,
        z_idx=z_idx,
        num_symbols=num_symbols,
        base_channels=base_channels,
    )


__all__ = [
    "inject_message_symbol_into_grid",
    "receiver_macro_jsd_by_message",
    "rollout_listener_telemetry",
]
