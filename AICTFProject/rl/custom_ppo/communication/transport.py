"""Local range-limited discrete message transport (V6I3 Slice 1)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from rl.custom_ppo.communication.channels import scatter_symbol_channels
from rl.custom_ppo.communication.config import CommConfig, raw_symbol_to_channel


@dataclass
class CommTelemetry:
    send_count: int = 0
    delivery_count: int = 0
    dropout_count: int = 0
    no_receiver_count: int = 0
    symbol_counts: list[int] = field(default_factory=list)

    def to_dict(
        self,
        *,
        num_symbols: int = 4,
        silence_symbol: int = -1,
        message_grid_channels: int | None = None,
    ) -> dict[str, float]:
        counts = list(self.symbol_counts[: int(num_symbols)])
        while len(counts) < int(num_symbols):
            counts.append(0)
        total_raw = max(1, sum(counts))
        grid_channels = int(message_grid_channels or num_symbols)
        active_counts = [0] * grid_channels
        for sym, count in enumerate(counts):
            channel = raw_symbol_to_channel(
                sym,
                num_symbols=int(num_symbols),
                message_grid_channels=grid_channels,
                silence_symbol=int(silence_symbol),
            )
            if channel >= 0:
                active_counts[channel] += int(count)
        total_active = max(1, sum(active_counts))
        import math

        probs = [c / total_active for c in active_counts if c > 0]
        entropy = -sum(p * math.log(p + 1e-12) for p in probs if p > 0)
        max_ent = math.log(max(1, grid_channels))
        silence_count = counts[int(silence_symbol)] if 0 <= int(silence_symbol) < len(counts) else 0
        return {
            "comm_send_count": float(self.send_count),
            "comm_delivery_count": float(self.delivery_count),
            "comm_dropout_count": float(self.dropout_count),
            "comm_no_receiver_count": float(self.no_receiver_count),
            "comm_silence_count": float(silence_count),
            "comm_active_send_count": float(sum(active_counts)),
            "comm_silence_share": float(silence_count / total_raw),
            "comm_symbol_entropy": float(entropy),
            "comm_symbol_entropy_normalized": float(entropy / max_ent) if max_ent > 0 else 0.0,
            "comm_symbols_used": float(sum(1 for c in active_counts if c > 0)),
            **{f"comm_symbol_occupancy_{i}": float(counts[i]) for i in range(int(num_symbols))},
        }


@dataclass
class _PendingDelivery:
    deliver_step: int
    env_idx: int
    sender: int
    symbol: int
    receivers: torch.Tensor
    dropped: torch.Tensor


class LocalCommTransport:
    """Per-batch communication state with one-step delayed delivery."""

    def __init__(self, cfg: CommConfig) -> None:
        self.cfg = cfg
        self.device = torch.device("cpu")
        self.batch_size = 0
        self.num_agents = 0
        self.global_step = 0
        self.held_outbound: torch.Tensor | None = None
        self.active_signal: torch.Tensor | None = None
        self._pending: list[_PendingDelivery] = []
        self.telemetry = CommTelemetry(symbol_counts=[0] * int(cfg.num_symbols))

    def reset(self, *, batch_size: int, num_agents: int, device: torch.device) -> None:
        self.device = device
        self.batch_size = int(batch_size)
        self.num_agents = int(num_agents)
        self.global_step = 0
        self.held_outbound = torch.full(
            (self.batch_size, self.num_agents), -1, dtype=torch.long, device=device
        )
        self.active_signal = torch.full(
            (self.batch_size, self.num_agents, self.num_agents),
            -1,
            dtype=torch.long,
            device=device,
        )
        self._pending.clear()
        self.telemetry = CommTelemetry(symbol_counts=[0] * int(self.cfg.num_symbols))

    def is_comm_boundary(self, step: int | None = None) -> bool:
        s = int(self.global_step if step is None else step)
        interval = max(1, int(self.cfg.interval_steps))
        return s > 0 and (s % interval) == 0

    def clear_dead_agents(self, alive: torch.Tensor) -> None:
        if self.active_signal is None:
            return
        dead = ~alive.bool()
        if not bool(dead.any()):
            return
        dead_recv = dead.unsqueeze(2).expand_as(self.active_signal)
        dead_send = dead.unsqueeze(1).expand_as(self.active_signal)
        self.active_signal = torch.where(
            dead_recv | dead_send,
            torch.full_like(self.active_signal, -1),
            self.active_signal,
        )

    def process_pending_deliveries(self, *, alive: torch.Tensor) -> None:
        if self.active_signal is None:
            return
        remaining: list[_PendingDelivery] = []
        for item in self._pending:
            if int(item.deliver_step) != int(self.global_step):
                remaining.append(item)
                continue
            env = int(item.env_idx)
            if not bool(alive[env, item.sender].item()):
                continue
            for recv in range(self.num_agents):
                if recv == item.sender:
                    continue
                if not bool(item.receivers[recv].item()):
                    continue
                if bool(item.dropped[recv].item()):
                    self.telemetry.dropout_count += 1
                    continue
                if not bool(alive[env, recv].item()):
                    continue
                self.active_signal[env, recv, item.sender] = int(item.symbol)
                self.telemetry.delivery_count += 1
        self._pending = remaining

    def submit_outbound(
        self,
        *,
        symbols: torch.Tensor,
        sender_x: torch.Tensor,
        sender_y: torch.Tensor,
        alive: torch.Tensor,
        rng: torch.Generator | None = None,
        apply_dropout: bool = True,
    ) -> None:
        assert self.held_outbound is not None and self.active_signal is not None
        bsz, nb = symbols.shape
        delay = max(0, int(self.cfg.delivery_delay_steps))
        deliver_step = int(self.global_step) + delay
        radius = float(self.cfg.radius_cells)

        for env in range(bsz):
            for sender in range(nb):
                sym = int(symbols[env, sender].item())
                if not bool(alive[env, sender].item()):
                    continue
                if sym < 0 or sym >= int(self.cfg.num_symbols):
                    continue
                self.telemetry.symbol_counts[sym] += 1
                self.held_outbound[env, sender] = sym
                self.active_signal[env, :, sender] = -1
                channel_sym = raw_symbol_to_channel(
                    sym,
                    num_symbols=int(self.cfg.num_symbols),
                    message_grid_channels=int(self.cfg.message_grid_channels),
                    silence_symbol=int(self.cfg.silence_symbol),
                )
                if channel_sym < 0:
                    continue

                dx = sender_x[env, sender] - sender_x[env]
                dy = sender_y[env, sender] - sender_y[env]
                dist = torch.sqrt(dx * dx + dy * dy + 1e-8)
                receivers = (dist <= radius) & alive[env].bool()
                receivers[sender] = False
                if not bool(receivers.any()):
                    self.telemetry.no_receiver_count += 1
                    continue

                dropped = torch.zeros((nb,), dtype=torch.bool, device=symbols.device)
                if apply_dropout and float(self.cfg.dropout_probability) > 0.0:
                    p = float(self.cfg.dropout_probability)
                    noise = torch.rand((nb,), generator=rng, device=symbols.device)
                    dropped = receivers & (noise < p)

                self._pending.append(
                    _PendingDelivery(
                        deliver_step=deliver_step,
                        env_idx=env,
                        sender=sender,
                        symbol=channel_sym,
                        receivers=receivers.clone(),
                        dropped=dropped.clone(),
                    )
                )
                self.telemetry.send_count += 1

    def advance_step(
        self,
        *,
        alive: torch.Tensor,
        sender_x: torch.Tensor,
        sender_y: torch.Tensor,
        receiver_x: torch.Tensor,
        receiver_y: torch.Tensor,
        cols: float,
        rows: float,
    ) -> torch.Tensor | None:
        self.global_step += 1
        self.process_pending_deliveries(alive=alive)
        self.clear_dead_agents(alive)
        return self.build_message_channels(
            alive=alive,
            sender_x=sender_x,
            sender_y=sender_y,
            receiver_x=receiver_x,
            receiver_y=receiver_y,
            cols=cols,
            rows=rows,
        )

    def build_message_channels(
        self,
        *,
        alive: torch.Tensor,
        sender_x: torch.Tensor,
        sender_y: torch.Tensor,
        receiver_x: torch.Tensor,
        receiver_y: torch.Tensor,
        cols: float,
        rows: float,
    ) -> torch.Tensor | None:
        if self.active_signal is None:
            return None
        dx = receiver_x[:, :, None] - sender_x[:, None, :]
        dy = receiver_y[:, :, None] - sender_y[:, None, :]
        dist = torch.sqrt(dx * dx + dy * dy + 1e-8)
        in_range = (dist <= float(self.cfg.radius_cells)) & alive[:, None, :].bool()
        for i in range(self.num_agents):
            in_range[:, i, i] = False
        return scatter_symbol_channels(
            receiver_x=receiver_x,
            receiver_y=receiver_y,
            sender_x=sender_x,
            sender_y=sender_y,
            sender_alive=alive.bool(),
            active_symbol=self.active_signal,
            in_range=in_range,
            num_symbols=int(self.cfg.message_grid_channels),
            cols=float(cols),
            rows=float(rows),
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "global_step": int(self.global_step),
            "held_outbound": None if self.held_outbound is None else self.held_outbound.cpu(),
            "active_signal": None if self.active_signal is None else self.active_signal.cpu(),
            "telemetry": self.telemetry.to_dict(
                num_symbols=int(self.cfg.num_symbols),
                silence_symbol=int(self.cfg.silence_symbol),
                message_grid_channels=int(self.cfg.message_grid_channels),
            ),
        }

    def reset_env_indices(self, env_mask: torch.Tensor) -> None:
        """Clear held and active messages for finished parallel environments."""
        if self.held_outbound is None or self.active_signal is None:
            return
        mask = env_mask.to(device=self.device, dtype=torch.bool).reshape(-1)
        if not bool(mask.any()):
            return
        self.held_outbound[mask] = -1
        self.active_signal[mask] = -1
        self._pending = [
            item for item in self._pending if not bool(mask[int(item.env_idx)].item())
        ]

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        self.global_step = int(payload.get("global_step", 0))
        ho = payload.get("held_outbound")
        if ho is not None and self.held_outbound is not None:
            self.held_outbound = ho.to(self.device)
        sig = payload.get("active_signal")
        if sig is not None and self.active_signal is not None:
            self.active_signal = sig.to(self.device)


__all__ = ["CommTelemetry", "LocalCommTransport"]
