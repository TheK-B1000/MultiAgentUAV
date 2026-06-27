"""V6I8 preset family — adapter-based specialist training.

These presets extend the V6I7 recurrent-router base with per-strategy
adapter modules trained on hard opponent pools.

Canonical presets
-----------------
* ``plan_faithful_latent_v6i8_adapter_balanced``       — standard pool
* ``plan_faithful_latent_v6i8_adapter_sparse``         — sparse-trigger variant
* ``plan_faithful_latent_v6i8_adapter_balanced_hardpool`` — hard-pool extension
* ``plan_faithful_latent_v6i8_adapter_sparse_hardpool`` — hard-pool + sparse
"""
from rl.presets.plan_faithful import (
    apply_plan_faithful_latent_v6i8_adapter_balanced,
    apply_plan_faithful_latent_v6i8_adapter_balanced_hardpool,
    apply_plan_faithful_latent_v6i8_adapter_sparse,
    apply_plan_faithful_latent_v6i8_adapter_sparse_hardpool,
)

__all__ = [
    "apply_plan_faithful_latent_v6i8_adapter_balanced",
    "apply_plan_faithful_latent_v6i8_adapter_sparse",
    "apply_plan_faithful_latent_v6i8_adapter_balanced_hardpool",
    "apply_plan_faithful_latent_v6i8_adapter_sparse_hardpool",
]
