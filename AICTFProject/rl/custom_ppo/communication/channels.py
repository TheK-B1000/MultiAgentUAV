"""Build per-agent local message CNN channels from transport state."""

from __future__ import annotations

import torch

from gpu_env._constants import CNN_COLS, CNN_ROWS


def scatter_symbol_channels(
    *,
    receiver_x: torch.Tensor,
    receiver_y: torch.Tensor,
    sender_x: torch.Tensor,
    sender_y: torch.Tensor,
    sender_alive: torch.Tensor,
    active_symbol: torch.Tensor,
    in_range: torch.Tensor,
    num_symbols: int,
    cols: float,
    rows: float,
) -> torch.Tensor:
    """Return ``(B, Nb, num_symbols, CNN_ROWS, CNN_COLS)`` message channels.

    ``active_symbol`` is ``(B, Nb_recv, Nb_send)`` with ``-1`` where no signal.
    ``in_range`` is ``(B, Nb_recv, Nb_send)`` bool — ongoing proximity for display.
    """
    device = receiver_x.device
    bsz, nb = receiver_x.shape
    out = torch.zeros((bsz, nb, int(num_symbols), CNN_ROWS, CNN_COLS), device=device)
    cx_scale = float(CNN_COLS - 1) / max(1.0, float(cols) - 1.0)
    cy_scale = float(CNN_ROWS - 1) / max(1.0, float(rows) - 1.0)

    for recv in range(nb):
        for send in range(nb):
            if recv == send:
                continue
            sym = active_symbol[:, recv, send]
            valid = (
                sender_alive[:, send]
                & in_range[:, recv, send]
                & (sym >= 0)
                & (sym < int(num_symbols))
            )
            if not bool(valid.any()):
                continue
            sx = torch.clamp((sender_x[:, send] * cx_scale).round().long(), 0, CNN_COLS - 1)
            sy = torch.clamp((sender_y[:, send] * cy_scale).round().long(), 0, CNN_ROWS - 1)
            for sym_id in range(int(num_symbols)):
                mask = valid & (sym == int(sym_id))
                if not bool(mask.any()):
                    continue
                b_idx = torch.arange(bsz, device=device)[mask]
                out[b_idx, recv, sym_id, sy[mask], sx[mask]] = 1.0
    return out


__all__ = ["scatter_symbol_channels"]
