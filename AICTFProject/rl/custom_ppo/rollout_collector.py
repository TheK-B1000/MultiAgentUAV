"""Backward-compatible facade for rollout collection.

The Phase 5 rollout refactor moved the implementation into
``rl.custom_ppo.rollout.collector``. Keep this module as the stable import
surface for older trainer code, tests, and external scripts.
"""

from rl.custom_ppo.rollout.collector import RolloutCollector
from rl.custom_ppo.rollout.collector import _denormalize_values

__all__ = ["RolloutCollector", "_denormalize_values"]
