"""Running return-normalization owned by ``CustomPPOTrainer``.

A small Welford-style stats container with normalize / denormalize / update
methods. The trainer holds two instances:

- ``trainer.return_norm`` for value-target normalization (gated by
  ``cfg.normalize_returns``).
- ``trainer.strategy_return_norm`` for the q_phi auxiliary return head's
  target normalization (gating happens at the call site since the head
  may be disabled even when value-target normalization is on).

Background
----------
These six floats used to live on the trainer as ``_return_norm_mean / _var /
_count`` and ``_strategy_return_mean / _var / _count``. They were written by
this module, read by ``ppo_updater`` and ``rollout_collector``, and
serialized by ``trainer.save`` — a textbook cross-module private-attribute
smell. Pulling them into ``ReturnNormalizer`` gives one named owner per
running-stats stream and makes the save / load contract explicit via
``state_dict``.

The module-level ``_return_norm_std`` / ``_normalize_value_targets`` /
``_denormalize_values`` / ``_update_return_norm_stats`` /
``_update_strategy_return_stats`` / ``_normalize_strategy_returns``
functions are kept as thin shims that forward to the sub-component, so
existing imports in ``ppo_updater`` and ``rollout_collector`` keep working.
"""

from __future__ import annotations

from typing import Any

import torch


class ReturnNormalizer:
    """Welford-style running mean / var / count for return normalization.

    ``enabled=False`` makes :meth:`normalize` / :meth:`denormalize` /
    :meth:`update` no-ops. Useful for ``cfg.normalize_returns=False`` where
    the trainer still wants to hold an instance but its math should pass
    values through unchanged.
    """

    _STD_FLOOR = 1e-3
    _VAR_FLOOR = 1e-6
    _COUNT_INIT = 1e-4

    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = bool(enabled)
        self.mean: float = 0.0
        self.var: float = 1.0
        self.count: float = self._COUNT_INIT

    # ------------------------------------------------------------------
    # Serialization helpers (trainer.save / trainer.load).
    # ------------------------------------------------------------------

    def state_dict(self) -> dict[str, float]:
        return {"mean": float(self.mean), "var": float(self.var), "count": float(self.count)}

    def load_state_dict(self, payload: dict[str, float]) -> None:
        self.mean = float(payload.get("mean", 0.0))
        self.var = float(payload.get("var", 1.0))
        self.count = float(payload.get("count", self._COUNT_INIT))

    # ------------------------------------------------------------------
    # Math.
    # ------------------------------------------------------------------

    @property
    def std(self) -> float:
        return max(self._STD_FLOOR, float(self.var) ** 0.5)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x.float()
        return (x.float() - float(self.mean)) / self.std

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x.float()
        return x.float() * self.std + float(self.mean)

    def update(self, values: torch.Tensor) -> None:
        """Fold a batch of observations into the running mean / var / count."""
        if not self.enabled:
            return
        v = values.detach().float().reshape(-1)
        if v.numel() <= 0:
            return
        batch_count = float(v.numel())
        batch_mean = float(v.mean().detach().cpu().item())
        batch_var = float(v.var(unbiased=False).detach().cpu().item()) if v.numel() > 1 else 0.0

        count = float(self.count)
        delta = batch_mean - float(self.mean)
        total = count + batch_count
        new_mean = float(self.mean) + delta * batch_count / max(self._VAR_FLOOR, total)
        m_a = float(self.var) * count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta * delta * count * batch_count / max(self._VAR_FLOOR, total)
        self.mean = new_mean
        self.var = max(self._VAR_FLOOR, m2 / max(self._VAR_FLOOR, total))
        self.count = total


# ---------------------------------------------------------------------------
# Module-level thin shims. These exist purely so callers that import the
# legacy underscore-prefixed names (``from rl.custom_ppo.return_normalization
# import _normalize_value_targets``) keep working without churn. New code
# should prefer ``trainer.return_norm.normalize(...)`` etc. directly.
# ---------------------------------------------------------------------------


def _return_norm_std(trainer: Any) -> float:
    return trainer.return_norm.std


def _normalize_value_targets(trainer: Any, returns: torch.Tensor) -> torch.Tensor:
    return trainer.return_norm.normalize(returns)


def _denormalize_values(trainer: Any, values: torch.Tensor) -> torch.Tensor:
    return trainer.return_norm.denormalize(values)


def _update_return_norm_stats(trainer: Any, returns: torch.Tensor) -> None:
    trainer.return_norm.update(returns)


def _update_strategy_return_stats(trainer: Any, buffer: Any) -> None:
    """Update strategy-return running stats from this rollout's sampled-z steps."""
    if not trainer.latent_strategy_aux_return_head or "z_resampled" not in buffer.fields:
        return
    pos = int(buffer.pos)
    sampled = buffer.fields["z_resampled"][:pos].reshape(-1).bool()
    if not bool(sampled.any().item()):
        return
    returns = buffer.fields["returns"][:pos].reshape(-1).detach().float()
    trainer.strategy_return_norm.update(returns[sampled])


def _normalize_strategy_returns(trainer: Any, returns: torch.Tensor) -> torch.Tensor:
    return trainer.strategy_return_norm.normalize(returns)


__all__ = [
    "ReturnNormalizer",
    "_denormalize_values",
    "_normalize_strategy_returns",
    "_normalize_value_targets",
    "_return_norm_std",
    "_update_return_norm_stats",
    "_update_strategy_return_stats",
]
