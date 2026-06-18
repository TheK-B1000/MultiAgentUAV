"""Training isolation for gate evaluation — no production state mutation."""

from __future__ import annotations

import copy
import hashlib
import io
import random
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn as nn

from rl.custom_ppo.inference import CustomPPOInferencePolicy

_OPTIMIZER_DIGEST_NAMES: tuple[str, ...] = (
    "primary",
    "actor",
    "critic",
    "router",
    "latent_router",
)


class GateIsolationError(RuntimeError):
    """Raised when gate evaluation mutates production training state."""


def _state_bytes(state: dict[str, Any]) -> bytes:
    buf = io.BytesIO()
    torch.save(state, buf)
    return buf.getvalue()


def digest_module_params(module: nn.Module | None) -> str:
    if module is None:
        return ""
    hasher = hashlib.md5()
    for tensor in module.state_dict().values():
        hasher.update(tensor.detach().cpu().numpy().tobytes())
    return hasher.hexdigest()


def digest_optimizer_state(optimizer: Any) -> str:
    if optimizer is None:
        return ""
    hasher = hashlib.md5()
    hasher.update(_state_bytes(optimizer.state_dict()))
    return hasher.hexdigest()


def _resolve_optimizer(trainer: Any, name: str) -> Any:
    bundle = getattr(trainer, "optimizers", None)
    if bundle is not None:
        return getattr(bundle, name, None)
    legacy_map = {
        "primary": "optimizer",
        "actor": "actor_optimizer",
        "critic": "critic_optimizer",
        "router": "router_optimizer",
        "latent_router": "latent_router_optimizer",
    }
    legacy_attr = legacy_map.get(name)
    if legacy_attr is None:
        return None
    return getattr(trainer, legacy_attr, None)


def digest_all_optimizers(trainer: Any) -> dict[str, str]:
    """Return MD5 digests for every known trainer optimizer slot."""
    return {name: digest_optimizer_state(_resolve_optimizer(trainer, name)) for name in _OPTIMIZER_DIGEST_NAMES}


@dataclass
class TrainingIsolationSnapshot:
    """Captured production state before an isolated gate evaluation."""

    actor_hash: str = ""
    critic_hash: str = ""
    router_hash: str = ""
    optimizer_hashes: dict[str, str] = field(default_factory=dict)
    py_rng_state: Any = None
    np_rng_state: Any = None
    torch_rng_state: Any = None
    torch_cuda_rng_states: list[Any] = field(default_factory=list)
    model_was_training: bool = True
    global_step: int = 0

    @classmethod
    def capture(cls, trainer: Any) -> TrainingIsolationSnapshot:
        snap = cls()
        model = trainer.model
        snap.model_was_training = bool(model.training)
        snap.global_step = int(getattr(trainer, "global_step", 0))
        snap.actor_hash = digest_module_params(getattr(model, "actor", None))
        snap.critic_hash = digest_module_params(getattr(model, "critic", None))
        router_modules = [
            getattr(model, name, None)
            for name in (
                "strategy_encoder",
                "episode_strategy_value_head",
                "phase_predictor",
                "strategy_aux_return_head",
            )
        ]
        router_hasher = hashlib.md5()
        for mod in router_modules:
            router_hasher.update(digest_module_params(mod).encode("utf-8"))
        snap.router_hash = router_hasher.hexdigest()
        snap.optimizer_hashes = digest_all_optimizers(trainer)

        snap.py_rng_state = random.getstate()
        snap.np_rng_state = np.random.get_state()
        snap.torch_rng_state = torch.get_rng_state()
        if torch.cuda.is_available():
            snap.torch_cuda_rng_states = torch.cuda.get_rng_state_all()
        return snap

    def assert_unchanged(self, trainer: Any) -> None:
        after = TrainingIsolationSnapshot.capture(trainer)
        if self.actor_hash != after.actor_hash:
            raise GateIsolationError("actor parameters mutated during gate evaluation")
        if self.critic_hash != after.critic_hash:
            raise GateIsolationError("critic parameters mutated during gate evaluation")
        if self.router_hash != after.router_hash:
            raise GateIsolationError("router parameters mutated during gate evaluation")
        for name in _OPTIMIZER_DIGEST_NAMES:
            before = self.optimizer_hashes.get(name, "")
            current = after.optimizer_hashes.get(name, "")
            if before != current:
                raise GateIsolationError(f"{name} optimizer mutated during gate evaluation")
        if int(getattr(trainer, "global_step", 0)) != self.global_step:
            raise GateIsolationError("global_step mutated during gate evaluation")
        if bool(trainer.model.training) != self.model_was_training:
            raise GateIsolationError("model.training mode mutated during gate evaluation")

    def restore_rng(self) -> None:
        if self.py_rng_state is not None:
            random.setstate(self.py_rng_state)
        if self.np_rng_state is not None:
            np.random.set_state(self.np_rng_state)
        if self.torch_rng_state is not None:
            torch.set_rng_state(self.torch_rng_state)
        if self.torch_cuda_rng_states and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(self.torch_cuda_rng_states)


@contextmanager
def isolated_gate_rng(
    trainer: Any,
    *,
    snapshot: TrainingIsolationSnapshot | None = None,
) -> Iterator[TrainingIsolationSnapshot]:
    """Capture and restore RNG streams (and model training mode) around gate work."""
    snap = snapshot or TrainingIsolationSnapshot.capture(trainer)
    try:
        yield snap
    finally:
        snap.restore_rng()
        trainer.model.train(snap.model_was_training)


class GateIsolationBoundary:
    """Deep-copied eval model + inference policy; production trainer stays untouched."""

    def __init__(self, trainer: Any) -> None:
        self.trainer = trainer
        self.snapshot = TrainingIsolationSnapshot.capture(trainer)
        self.eval_model = copy.deepcopy(trainer.model)
        self._policy: CustomPPOInferencePolicy | None = None

    def policy(self) -> CustomPPOInferencePolicy:
        if self._policy is None:
            cfg = self.trainer.cfg
            cfg_payload = asdict(cfg) if hasattr(cfg, "__dataclass_fields__") else dict(vars(cfg))
            self._policy = CustomPPOInferencePolicy(
                self.eval_model,
                device=getattr(self.trainer, "device", torch.device("cpu")),
                cfg=cfg_payload,
            )
        return self._policy

    def assert_unchanged(self) -> None:
        self.snapshot.assert_unchanged(self.trainer)

    def close(self) -> None:
        self.snapshot.restore_rng()
        self.trainer.model.train(self.snapshot.model_was_training)
        self._policy = None

    def __enter__(self) -> GateIsolationBoundary:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


__all__ = [
    "GateIsolationBoundary",
    "GateIsolationError",
    "TrainingIsolationSnapshot",
    "digest_all_optimizers",
    "digest_module_params",
    "digest_optimizer_state",
    "isolated_gate_rng",
]
