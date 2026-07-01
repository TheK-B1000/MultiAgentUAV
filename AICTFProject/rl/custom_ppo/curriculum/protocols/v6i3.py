"""V6I3 gate protocol — v6i2 dual evidence plus communication families."""

from __future__ import annotations

from typing import Any

from rl.config.ppo_config import PPOConfig
from rl.custom_ppo.communication.gates import (
    evaluate_communication_usage,
    evaluate_listener_causal_response,
)
from rl.custom_ppo.curriculum.context import GateContext
from rl.custom_ppo.curriculum.protocols.v6i2 import V6I2GateProtocol
from rl.custom_ppo.curriculum.types import GateResult
from rl.custom_ppo.gate_protocol import GATE_FAMILY_NAMES_V6I3, V6I3_GATE_PROTOCOL


class V6I3GateProtocol(V6I2GateProtocol):
    version = V6I3_GATE_PROTOCOL

    def required_families(self) -> tuple[str, ...]:
        return GATE_FAMILY_NAMES_V6I3

    def evaluate_online(self, context: GateContext) -> dict[str, GateResult]:
        results = super().evaluate_online(context)
        telemetry = self._comm_telemetry(context)
        usage = evaluate_communication_usage(context.cfg, telemetry)
        listener = evaluate_listener_causal_response(context.cfg, telemetry)
        results["communication_usage"] = GateResult(
            status=usage.status,
            reason=usage.reason,
            details=dict(usage.details),
        )
        results["listener_causal_response"] = GateResult(
            status=listener.status,
            reason=listener.reason,
            details=dict(listener.details),
        )
        return results

    @staticmethod
    def _comm_telemetry(context: GateContext) -> dict[str, float]:
        trainer = context.trainer
        stats = dict(getattr(trainer, "last_stats", {}) or {})
        comm_runtime = getattr(trainer, "comm_runtime", None)
        if comm_runtime is not None and comm_runtime.enabled and comm_runtime.transport is not None:
            stats.update(
                comm_runtime.transport.telemetry.to_dict(
                    num_symbols=int(getattr(context.cfg, "comm_num_symbols", 4) or 4),
                    silence_symbol=int(getattr(context.cfg, "comm_silence_symbol", -1)),
                    message_grid_channels=int(
                        getattr(context.cfg, "comm_message_grid_channels", 4) or 4
                    ),
                )
            )
        return {k: float(v) for k, v in stats.items() if k.startswith("comm_") or k.startswith("mi_") or k.startswith("receiver_")}


__all__ = ["V6I3GateProtocol"]
