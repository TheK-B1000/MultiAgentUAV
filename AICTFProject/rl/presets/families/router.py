"""Stage D router preset family.

These presets train the recurrent router (q_φ) to select among trained
adapters.  The actor trunk and adapters are frozen; only the router and
router critic are updated.

Canonical presets
-----------------
* ``plan_faithful_latent_v6i9_mapaware_router_sparse_hardpool`` — recurrent (V6I9)
* ``plan_faithful_latent_v6i9_mapaware_router_feedforward_hardpool`` — feedforward (V6I9)
* ``plan_faithful_latent_v6i7_recurrent_router``                — V6I7 reference
* ``plan_faithful_latent_v6i7_router_critic_warmup``            — V6I7 with critic warm-up
* ``plan_faithful_latent_v6i5_router_z0_z3_frozen_actor``       — V6I5 ablation
"""
from rl.presets.plan_faithful import (
    apply_plan_faithful_latent_v6i5_router_z0_z3_frozen_actor,
    apply_plan_faithful_latent_v6i7_recurrent_router,
    apply_plan_faithful_latent_v6i7_router_critic_warmup,
    apply_plan_faithful_latent_v6i9_mapaware_router_sparse_hardpool,
    apply_plan_faithful_latent_v6i9_mapaware_router_feedforward_hardpool,
    apply_plan_faithful_latent_v6i9_arc_credit_running_mean_feedforward_hardpool,
    apply_plan_faithful_latent_v6i9_arc_credit_running_mean_hardpool,
    apply_plan_faithful_latent_v6i9_arc_credit_specialize_hardpool,
    apply_plan_faithful_latent_v6i11_q_router_hardpool,
)

__all__ = [
    "apply_plan_faithful_latent_v6i9_mapaware_router_sparse_hardpool",
    "apply_plan_faithful_latent_v6i9_mapaware_router_feedforward_hardpool",
    "apply_plan_faithful_latent_v6i9_arc_credit_running_mean_feedforward_hardpool",
    "apply_plan_faithful_latent_v6i9_arc_credit_running_mean_hardpool",
    "apply_plan_faithful_latent_v6i9_arc_credit_specialize_hardpool",
    "apply_plan_faithful_latent_v6i11_q_router_hardpool",
    "apply_plan_faithful_latent_v6i7_recurrent_router",
    "apply_plan_faithful_latent_v6i7_router_critic_warmup",
    "apply_plan_faithful_latent_v6i5_router_z0_z3_frozen_actor",
]
