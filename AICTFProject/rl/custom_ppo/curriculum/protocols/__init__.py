"""Gate protocol factory."""

from __future__ import annotations

from rl.config.ppo_config import PPOConfig
from rl.custom_ppo.curriculum.protocols.base import GateProtocol
from rl.custom_ppo.curriculum.protocols.v6i1 import V6I1GateProtocol
from rl.custom_ppo.curriculum.protocols.v6i2 import V6I2GateProtocol
from rl.custom_ppo.gate_protocol import is_v6i2_gate_protocol


def build_gate_protocol(cfg: PPOConfig) -> GateProtocol:
    if is_v6i2_gate_protocol(cfg):
        return V6I2GateProtocol()
    return V6I1GateProtocol()


__all__ = ["build_gate_protocol"]
