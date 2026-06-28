"""Backward-compatibility facade.

All implementations live in ``rl.custom_ppo.diagnostics.*``.
This module re-exports every name under the original underscore-prefixed
identifiers so that existing trainer, post-update, and test code continues
to work without changes.

Dependency direction is strictly one-way:
    latent_diagnostics.py (facade)
            ↓
    rl.custom_ppo.diagnostics.*  (implementations)
"""

from rl.custom_ppo.diagnostics.entropy import (
    _flat_long_np,
    _flat_float_np,
    _mi_z_vs,
    _bucket_z_fracs,
    _fill_zero_z_fracs,
    _shannon_entropy_nats,
)
from rl.custom_ppo.diagnostics.specialization import (
    _q_phi_probs_and_entropy,
    _flag_state_per_step,
    _phase_block,
    _behavior_diversity_stats,
)
from rl.custom_ppo.diagnostics.switching import (
    _reward_sum_after_switch_5,
    _flag_return_indices,
    _switch_proximity_fracs,
)
from rl.custom_ppo.diagnostics.counterfactual import (
    _jsd_from_logits,
    _macro_probs_from_logits,
    _batched_policy_trunk_features,
    _batched_policy_logits,
    _forced_z_behavior_profile,
    _policy_z_sensitivity_kl,
    compute_pairwise_actor_jsd,
)
from rl.custom_ppo.diagnostics.competence import (
    compute_adapter_grad_norms,
    compute_critic_value_variance,
    _v6i8_residual_adapter_stats,
    _strategy_resample_advantage_stats,
    _rollout_advantage_diagnostics,
    _latent_option_advantage_stats,
)
from rl.custom_ppo.diagnostics.aggregation import (
    _latent_rollout_stats,
    _latent_opponent_rollout_diag,
    _write_strategy_experience_table,
    _write_refresh_log_table,
)

# V6I8 stable public alias
_v6i8_adapter_stats = _v6i8_residual_adapter_stats

__all__ = [
    "_flat_long_np",
    "_flat_float_np",
    "_mi_z_vs",
    "_bucket_z_fracs",
    "_fill_zero_z_fracs",
    "_shannon_entropy_nats",
    "_q_phi_probs_and_entropy",
    "_flag_state_per_step",
    "_phase_block",
    "_behavior_diversity_stats",
    "_reward_sum_after_switch_5",
    "_flag_return_indices",
    "_switch_proximity_fracs",
    "_jsd_from_logits",
    "_macro_probs_from_logits",
    "_batched_policy_trunk_features",
    "_batched_policy_logits",
    "_forced_z_behavior_profile",
    "_policy_z_sensitivity_kl",
    "compute_pairwise_actor_jsd",
    "compute_adapter_grad_norms",
    "compute_critic_value_variance",
    "_v6i8_residual_adapter_stats",
    "_v6i8_adapter_stats",
    "_strategy_resample_advantage_stats",
    "_rollout_advantage_diagnostics",
    "_latent_option_advantage_stats",
    "_latent_rollout_stats",
    "_latent_opponent_rollout_diag",
    "_write_strategy_experience_table",
    "_write_refresh_log_table",
]
