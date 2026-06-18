"""Rollout-time communication state (V6I3 Slice 2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from rl.custom_ppo.communication.config import CommConfig, resolve_comm_config
from rl.custom_ppo.communication.corruption import (
    CommCorruptionMode,
    apply_message_channel_corruption,
    parse_corruption_mode,
    symbol_marginal_from_channels,
)
from rl.custom_ppo.communication.observation import inject_message_grid_channels
from rl.custom_ppo.communication.transport import LocalCommTransport


@dataclass
class CommStepAux:
    symbols: torch.Tensor
    log_probs: torch.Tensor
    entropy: torch.Tensor
    boundary_mask: torch.Tensor


class CommRolloutRuntime:
    """Owns transport state and observation injection for one trainer."""

    def __init__(self, cfg: Any, *, device: torch.device) -> None:
        self.cfg = cfg
        self.comm = resolve_comm_config(cfg)
        self.device = device
        self.transport: LocalCommTransport | None = None
        self._message_channels: torch.Tensor | None = None
        self._rng: torch.Generator | None = None
        self.corruption_mode: CommCorruptionMode = CommCorruptionMode.NORMAL
        self._extra_delay_pending: int = 0
        self._symbol_marginal: torch.Tensor | None = None

    @property
    def enabled(self) -> bool:
        return bool(self.comm.enabled)

    def reset(self, *, batch_size: int, num_agents: int) -> None:
        if not self.enabled:
            self.transport = None
            self._message_channels = None
            return
        self.transport = LocalCommTransport(self.comm)
        self.transport.reset(
            batch_size=int(batch_size),
            num_agents=int(num_agents),
            device=self.device,
        )
        seed = int(getattr(self.cfg, "seed", 0) or 0)
        self._rng = torch.Generator(device=self.device)
        self._rng.manual_seed(seed + 17_003)
        self._refresh_message_channels()

    def set_corruption_mode(self, mode: str | CommCorruptionMode | None) -> None:
        if mode is None:
            self.corruption_mode = CommCorruptionMode.NORMAL
            return
        self.corruption_mode = (
            mode if isinstance(mode, CommCorruptionMode) else parse_corruption_mode(str(mode))
        )

    def reset_env_indices(self, done_mask: np.ndarray | torch.Tensor) -> None:
        if not self.enabled or self.transport is None:
            return
        mask = torch.as_tensor(done_mask, dtype=torch.bool, device=self.device).reshape(-1)
        self.transport.reset_env_indices(mask)

    def current_boundary_mask(self) -> torch.Tensor:
        if not self.enabled or self.transport is None:
            return torch.zeros((1,), dtype=torch.bool, device=self.device)
        bsz = int(self.transport.batch_size)
        is_boundary = bool(self.transport.is_comm_boundary())
        return torch.full((bsz,), is_boundary, dtype=torch.bool, device=self.device)

    def prepare_obs(
        self,
        obs: dict[str, np.ndarray],
        *,
        expected_grid_channels: int | None = None,
    ) -> dict[str, np.ndarray]:
        if not self.enabled:
            return obs
        return inject_message_grid_channels(
            obs,
            message_channels=self._message_channels,
            cfg=self.cfg,
            expected_grid_channels=expected_grid_channels,
        )

    def submit_sampled_messages(
        self,
        *,
        symbols: torch.Tensor,
        boundary_mask: torch.Tensor,
        env_core: Any,
    ) -> None:
        if not self.enabled or self.transport is None:
            return
        if not bool(boundary_mask.any()):
            return
        alive = env_core.blue_alive.bool()
        x = env_core.blue_x.float()
        y = env_core.blue_y.float()
        submit_symbols = symbols.clone()
        submit_symbols[~boundary_mask] = -1
        apply_dropout = float(self.comm.dropout_probability) > 0.0
        self.transport.submit_outbound(
            symbols=submit_symbols,
            sender_x=x,
            sender_y=y,
            alive=alive,
            rng=self._rng,
            apply_dropout=apply_dropout,
        )

    def advance_after_step(self, env_core: Any) -> None:
        if not self.enabled or self.transport is None:
            return
        alive = env_core.blue_alive.bool()
        x = env_core.blue_x.float()
        y = env_core.blue_y.float()
        cols = float(getattr(env_core, "cols", 20))
        rows = float(getattr(env_core, "rows", 20))
        self.transport.advance_step(
            alive=alive,
            sender_x=x,
            sender_y=y,
            receiver_x=x,
            receiver_y=y,
            cols=cols,
            rows=rows,
        )
        self._refresh_message_channels()

    def _refresh_message_channels(self) -> None:
        if not self.enabled or self.transport is None:
            self._message_channels = None
            return
        core = getattr(self, "_env_core", None)
        if core is None:
            self._message_channels = None
            return
        alive = core.blue_alive.bool()
        x = core.blue_x.float()
        y = core.blue_y.float()
        cols = float(getattr(core, "cols", 20))
        rows = float(getattr(core, "rows", 20))
        self._message_channels = self.transport.build_message_channels(
            alive=alive,
            sender_x=x,
            sender_y=y,
            receiver_x=x,
            receiver_y=y,
            cols=cols,
            rows=rows,
        )
        if self._message_channels is not None:
            self._symbol_marginal = symbol_marginal_from_channels(
                self._message_channels,
                num_symbols=int(self.comm.num_symbols),
            )
            channels = apply_message_channel_corruption(
                self._message_channels,
                mode=self.corruption_mode,
                symbol_marginal=self._symbol_marginal,
                rng=self._rng,
            )
            if self.corruption_mode == CommCorruptionMode.EXTRA_DELAY:
                if self._extra_delay_pending > 0:
                    self._extra_delay_pending -= 1
                    channels = torch.zeros_like(channels)
                else:
                    self._extra_delay_pending = 1
            self._message_channels = channels

    def non_boundary_message_aux(
        self,
        *,
        boundary_mask: torch.Tensor,
        num_agents: int,
    ) -> dict[str, torch.Tensor]:
        """Rollout payload for steps that retain held symbols without a new PPO draw."""
        batch = int(boundary_mask.shape[0])
        device = boundary_mask.device
        if not self.enabled or self.transport is None or self.transport.held_outbound is None:
            return {
                "message_symbols": torch.full(
                    (batch, int(num_agents)),
                    -1,
                    dtype=torch.long,
                    device=device,
                ),
                "message_log_probs": torch.zeros((batch,), dtype=torch.float32, device=device),
                "message_entropy": torch.zeros((batch,), dtype=torch.float32, device=device),
                "message_boundary_mask": boundary_mask.bool(),
            }
        held = self.transport.held_outbound
        return {
            "message_symbols": held.clone(),
            "message_log_probs": torch.zeros(
                (int(held.shape[0]),), dtype=torch.float32, device=held.device
            ),
            "message_entropy": torch.zeros(
                (int(held.shape[0]),), dtype=torch.float32, device=held.device
            ),
            "message_boundary_mask": boundary_mask.bool(),
        }

    def bind_env_core(self, env_core: Any) -> None:
        self._env_core = env_core
        self._refresh_message_channels()

    def state_dict(self) -> dict[str, Any]:
        if not self.enabled or self.transport is None:
            return {"enabled": False}
        return {
            "enabled": True,
            "protocol_version": self.comm.protocol_version,
            "transport": self.transport.state_dict(),
        }

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        if not payload or not bool(payload.get("enabled", False)):
            return
        if self.transport is not None:
            self.transport.load_state_dict(dict(payload.get("transport", {}) or {}))
            self._refresh_message_channels()


__all__ = ["CommRolloutRuntime", "CommStepAux", "CommConfig"]
