"""Evaluation context for curriculum gate families."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn as nn

from rl.config.ppo_config import PPOConfig
from rl.custom_ppo.inference import CustomPPOInferencePolicy


@contextmanager
def preserve_model_training_mode(model: nn.Module) -> Iterator[None]:
    was_training = bool(model.training)
    try:
        yield
    finally:
        model.train(was_training)


@dataclass
class GateContext:
    """Immutable-ish bundle passed to gate evaluators and protocols."""

    trainer: Any
    cfg: PPOConfig
    step: int
    eval_model: nn.Module
    eval_policy: CustomPPOInferencePolicy | None = None
    latent_k: int = 4
    _policy_cache: CustomPPOInferencePolicy | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.eval_policy is not None:
            self._policy_cache = self.eval_policy
        if self.latent_k <= 0:
            self.latent_k = int(getattr(self.trainer, "latent_k", 4))

    def _policy_wrapper(self) -> CustomPPOInferencePolicy:
        if self._policy_cache is None:
            from dataclasses import asdict

            cfg_payload = (
                asdict(self.cfg)
                if hasattr(self.cfg, "__dataclass_fields__")
                else dict(vars(self.cfg))
            )
            self._policy_cache = CustomPPOInferencePolicy(
                self.eval_model,
                device=getattr(self.trainer, "device", torch.device("cpu")),
                cfg=cfg_payload,
            )
        return self._policy_cache

    def configure_fixed_z(self, z_id: int) -> CustomPPOInferencePolicy:
        policy = self._policy_wrapper()
        policy.fixed_latent_strategy = True
        policy.fixed_latent_strategy_id = int(z_id)
        policy.reset_strategy()
        return policy

    def predict(self, obs: dict[str, Any]) -> np.ndarray:
        policy = self._policy_wrapper()
        act, _ = policy.predict(self.unwrap_vec_obs(obs), deterministic=True)
        return act

    @staticmethod
    def unwrap_vec_obs(obs: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value[0] if hasattr(value, "shape") and getattr(value, "ndim", 0) >= 2 else value
            for key, value in obs.items()
        }


__all__ = ["GateContext", "preserve_model_training_mode"]
