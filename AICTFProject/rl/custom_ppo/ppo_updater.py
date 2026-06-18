"""PPO update loop — thin compatibility facade.

Implementation lives under :mod:`rl.custom_ppo.update`. This module re-exports
the public surface used by the trainer and regression tests.
"""

from __future__ import annotations

from rl.custom_ppo.update.helpers import (
    StrictFaithfulDictWrapper,
    populate_main_loop_qphi_telemetry as _populate_main_loop_qphi_telemetry,
    warmup_ramp_value as _warmup_ramp_value,
)
from rl.custom_ppo.update.phase_policy import set_model_requires_grad_for_phase
from rl.custom_ppo.update.separation_objectives import (
    extract_rollout_resample_subset as _extract_rollout_resample_subset,
    policy_z_separation_loss as _policy_z_separation_loss,
    z_separation_gate_mask as _z_separation_gate_mask,
)
from rl.custom_ppo.update.updater import PPOUpdater

__all__ = [
    "PPOUpdater",
    "StrictFaithfulDictWrapper",
    "set_model_requires_grad_for_phase",
    "_extract_rollout_resample_subset",
    "_policy_z_separation_loss",
    "_populate_main_loop_qphi_telemetry",
    "_warmup_ramp_value",
    "_z_separation_gate_mask",
]
