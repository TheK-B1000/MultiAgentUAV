"""Factory helpers for training environment and trainer construction.

Re-exports :func:`build_training_env` from :mod:`rl.training.env_factory` as
the canonical factory entry point, and provides
:func:`make_trainer_hyperparams` for deriving the final PPO hyperparameter
dict from a resolved training config.

The resolved-config layer performs the multi-agent learning-rate scaling and
batch-size clamping; this module's role is to package those computed values
into the keyword argument dict that :class:`rl.custom_ppo.CustomPPOTrainer`
accepts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rl.training.env_factory import build_training_env  # noqa: F401 — re-export

if TYPE_CHECKING:
    from rl.training.resolved_config import ResolvedTrainingConfig

__all__ = [
    "build_training_env",
    "make_trainer_hyperparams",
]


def make_trainer_hyperparams(resolved: "ResolvedTrainingConfig") -> dict:
    """Return the keyword arguments for ``CustomPPOTrainer.__init__`` derived from ``resolved``.

    Packaging the kwargs here keeps the orchestrator body free of attribute
    lookups on the resolved config and makes it easy to mock in tests.
    """
    return {
        "learning_rate": resolved.effective_lr,
        "clip_range": resolved.effective_clip_range,
        "ent_coef": resolved.effective_ent_coef,
        "n_epochs": resolved.effective_n_epochs,
        "batch_size": resolved.effective_batch_size,
    }
