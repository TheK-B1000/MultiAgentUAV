"""V6I3 communication telemetry from rollout buffers and transport state."""

from __future__ import annotations

import math
from typing import Any

import torch

from rl.ppo_core import TensorDictRolloutBuffer


def _entropy_from_counts(counts: list[int] | torch.Tensor, *, num_symbols: int) -> tuple[float, float, int]:
    vals = [int(c) for c in counts[: int(num_symbols)]]
    while len(vals) < int(num_symbols):
        vals.append(0)
    total = max(1, sum(vals))
    probs = [c / total for c in vals if c > 0]
    entropy = -sum(p * math.log(p + 1e-12) for p in probs if p > 0)
    max_ent = math.log(max(1, int(num_symbols)))
    used = sum(1 for c in vals if c > 0)
    return float(entropy), float(entropy / max_ent) if max_ent > 0 else 0.0, int(used)


def rollout_comm_usage_telemetry(
    buffer: TensorDictRolloutBuffer,
    *,
    num_symbols: int = 4,
    silence_symbol: int = -1,
) -> dict[str, float]:
    """Usage counters from one rollout buffer."""
    if "message_boundary_mask" not in buffer.fields:
        return {}
    length = int(buffer.pos)
    if length <= 0:
        return {}
    boundary = buffer.fields["message_boundary_mask"][:length].reshape(-1).bool()
    symbols = buffer.fields["message_symbols"][:length].reshape(-1, buffer.fields["message_symbols"].shape[-1])
    valid_boundaries = int(boundary.sum().item())
    flat_syms = symbols[boundary.reshape(-1)].reshape(-1).detach().cpu()
    counts = [0] * int(num_symbols)
    for sym in flat_syms.tolist():
        idx = int(sym)
        if 0 <= idx < int(num_symbols):
            counts[idx] += 1
    active_counts = [
        c for i, c in enumerate(counts) if not (int(silence_symbol) >= 0 and i == int(silence_symbol))
    ]
    entropy, entropy_norm, used = _entropy_from_counts(
        active_counts,
        num_symbols=max(1, len(active_counts)),
    )
    active_total = sum(active_counts)
    raw_total = sum(counts)
    dominance = max(active_counts) / max(1, active_total) if active_counts else 1.0
    silence_count = counts[int(silence_symbol)] if 0 <= int(silence_symbol) < len(counts) else 0
    out = {
        "comm_valid_boundaries": float(valid_boundaries),
        "comm_send_count": float(valid_boundaries * int(symbols.shape[1])),
        "comm_silence_count": float(silence_count),
        "comm_active_send_count": float(active_total),
        "comm_silence_share": float(silence_count / max(1, raw_total)),
        "comm_symbol_entropy": float(entropy),
        "comm_symbol_entropy_normalized": float(entropy_norm),
        "comm_symbols_used": float(used),
        "comm_symbol_dominance": float(dominance),
    }
    for i in range(int(num_symbols)):
        out[f"comm_symbol_occupancy_{i}"] = float(counts[i])
    if "message_log_probs" in buffer.fields:
        msg_lp = buffer.fields["message_log_probs"][:length].reshape(-1)
        active_lp = msg_lp[boundary]
        out["comm_message_logprob_mean"] = (
            float(active_lp.mean().item()) if int(active_lp.numel()) > 0 else 0.0
        )
    return out


def merge_transport_telemetry(transport_stats: dict[str, float], usage: dict[str, float]) -> dict[str, float]:
    merged = dict(usage)
    merged.update(transport_stats)
    return merged


def estimate_mi_histogram(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    x_bins: int,
    y_bins: int,
) -> float:
    """Plug-in MI estimate from discrete indices (nats)."""
    x = x.long().reshape(-1)
    y = y.long().reshape(-1)
    if x.numel() <= 0 or x.shape[0] != y.shape[0]:
        return 0.0
    x_max = int(x.max().item()) + 1 if x.numel() > 0 else 1
    y_max = int(y.max().item()) + 1 if y.numel() > 0 else 1
    joint = torch.zeros((max(x_bins, x_max), max(y_bins, y_max)), dtype=torch.float64)
    for xi, yi in zip(x.tolist(), y.tolist()):
        if xi < 0 or yi < 0:
            continue
        joint[xi, yi] += 1.0
    total = joint.sum()
    if total <= 0:
        return 0.0
    joint = joint / total
    px = joint.sum(dim=1)
    py = joint.sum(dim=0)
    mi = 0.0
    for i in range(joint.shape[0]):
        for j in range(joint.shape[1]):
            pxy = float(joint[i, j].item())
            if pxy <= 0.0:
                continue
            mi += pxy * math.log(pxy / (float(px[i].item()) * float(py[j].item()) + 1e-12) + 1e-12)
    return float(max(0.0, mi))


def rollout_comm_information_telemetry(
    buffer: TensorDictRolloutBuffer,
    *,
    num_symbols: int = 4,
) -> dict[str, float]:
    """Conditional MI diagnostics on boundary rows only."""
    if "message_boundary_mask" not in buffer.fields or "message_symbols" not in buffer.fields:
        return {}
    length = int(buffer.pos)
    if length <= 0:
        return {}
    boundary = buffer.fields["message_boundary_mask"][:length].reshape(-1).bool()
    if not bool(boundary.any()):
        return {}
    symbols = buffer.fields["message_symbols"][:length].reshape(-1, buffer.fields["message_symbols"].shape[-1])
    sender_sym = symbols[boundary, 0]
    out: dict[str, float] = {}
    if "z" in buffer.fields:
        z = buffer.fields["z"][:length].reshape(-1)[boundary]
        out["mi_message_z"] = estimate_mi_histogram(sender_sym, z, x_bins=int(num_symbols), y_bins=8)
    if "phase_id" in buffer.fields:
        phase = buffer.fields["phase_id"][:length].reshape(-1)[boundary]
        out["mi_message_phase"] = estimate_mi_histogram(sender_sym, phase, x_bins=int(num_symbols), y_bins=16)
    if "role_bucket_id" in buffer.fields:
        role = buffer.fields["role_bucket_id"][:length].reshape(-1)[boundary]
        out["mi_message_role"] = estimate_mi_histogram(sender_sym, role, x_bins=int(num_symbols), y_bins=32)
    if "actions" in buffer.fields:
        macro = buffer.fields["actions"][:length].reshape(-1, buffer.fields["actions"].shape[-1])[:, 0][boundary]
        out["mi_message_next_macro_action"] = estimate_mi_histogram(
            sender_sym, macro, x_bins=int(num_symbols), y_bins=32
        )
    return out


def collect_rollout_comm_telemetry(
    buffer: TensorDictRolloutBuffer,
    *,
    cfg: Any,
    transport_stats: dict[str, float] | None = None,
) -> dict[str, float]:
    if not bool(getattr(cfg, "communication_enabled", False)):
        return {}
    num_symbols = int(getattr(cfg, "comm_num_symbols", 4) or 4)
    usage = rollout_comm_usage_telemetry(
        buffer,
        num_symbols=num_symbols,
        silence_symbol=int(getattr(cfg, "comm_silence_symbol", -1)),
    )
    info = rollout_comm_information_telemetry(buffer, num_symbols=num_symbols)
    merged = merge_transport_telemetry(dict(transport_stats or {}), usage)
    merged.update(info)
    return merged


__all__ = [
    "collect_rollout_comm_telemetry",
    "estimate_mi_histogram",
    "rollout_comm_information_telemetry",
    "rollout_comm_usage_telemetry",
]
