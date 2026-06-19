"""Resolved local communication configuration from PPOConfig."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CommConfig:
    enabled: bool = False
    protocol_version: str = "v6i3_strategy_local_comm_v1"
    num_symbols: int = 4
    silence_symbol: int = -1
    interval_steps: int = 32
    delivery_delay_steps: int = 1
    radius_cells: float = 6.0
    dropout_probability: float = 0.10
    entropy_coef: float = 0.001
    hold_last_message: bool = True
    local_only: bool = True
    include_sender_position: bool = True
    message_grid_channels: int = 4
    cf_include_message_head: bool = False

    def fingerprint_keys(self) -> tuple[str, ...]:
        return (
            "communication_enabled",
            "comm_protocol_version",
            "comm_num_symbols",
            "comm_silence_symbol",
            "comm_interval_steps",
            "comm_delivery_delay_steps",
            "comm_radius_cells",
            "comm_dropout_probability",
            "comm_entropy_coef",
            "comm_hold_last_message",
            "comm_local_only",
            "comm_include_sender_position",
            "comm_message_grid_channels",
            "comm_cf_include_message_head",
        )


def resolve_comm_config(cfg: Any) -> CommConfig:
    return CommConfig(
        enabled=bool(getattr(cfg, "communication_enabled", False)),
        protocol_version=str(
            getattr(cfg, "comm_protocol_version", "v6i3_strategy_local_comm_v1") or ""
        ),
        num_symbols=int(getattr(cfg, "comm_num_symbols", 4) or 4),
        silence_symbol=int(getattr(cfg, "comm_silence_symbol", -1)),
        interval_steps=int(getattr(cfg, "comm_interval_steps", 32) or 32),
        delivery_delay_steps=int(getattr(cfg, "comm_delivery_delay_steps", 1) or 1),
        radius_cells=float(getattr(cfg, "comm_radius_cells", 6.0) or 6.0),
        dropout_probability=float(getattr(cfg, "comm_dropout_probability", 0.10) or 0.0),
        entropy_coef=float(getattr(cfg, "comm_entropy_coef", 0.001) or 0.0),
        hold_last_message=bool(getattr(cfg, "comm_hold_last_message", True)),
        local_only=bool(getattr(cfg, "comm_local_only", True)),
        include_sender_position=bool(getattr(cfg, "comm_include_sender_position", True)),
        message_grid_channels=int(getattr(cfg, "comm_message_grid_channels", 4) or 4),
        cf_include_message_head=bool(getattr(cfg, "comm_cf_include_message_head", False)),
    )


def extra_cnn_channels(cfg: Any) -> int:
    comm = resolve_comm_config(cfg)
    if not comm.enabled:
        return 0
    return int(comm.message_grid_channels)


def raw_symbol_to_channel(
    symbol: int,
    *,
    num_symbols: int,
    message_grid_channels: int,
    silence_symbol: int = -1,
) -> int:
    """Map a sampled communication symbol to a rendered message channel."""
    sym = int(symbol)
    if sym < 0 or sym >= int(num_symbols):
        return -1
    if int(silence_symbol) >= 0 and sym == int(silence_symbol):
        return -1
    if int(silence_symbol) == 0 and int(num_symbols) == int(message_grid_channels) + 1:
        channel = sym - 1
    else:
        channel = sym
    if channel < 0 or channel >= int(message_grid_channels):
        return -1
    return int(channel)


__all__ = [
    "CommConfig",
    "extra_cnn_channels",
    "raw_symbol_to_channel",
    "resolve_comm_config",
]
