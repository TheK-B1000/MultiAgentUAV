"""PPO update loop — thin compatibility facade.

Implementation lives under :mod:`rl.custom_ppo.update`. This module re-exports
the public surface used by the trainer and regression tests.
"""

from __future__ import annotations

from rl.custom_ppo.update.phase_policy import set_model_requires_grad_for_phase
from rl.custom_ppo.update.separation_objectives import (
    extract_rollout_resample_subset as _extract_rollout_resample_subset,
    policy_z_separation_loss as _policy_z_separation_loss,
)
from rl.custom_ppo.update.updater import PPOUpdater

__all__ = [
    "PPOUpdater",
    "set_model_requires_grad_for_phase",
    "_policy_z_separation_loss",
    "_extract_rollout_resample_subset",
]
