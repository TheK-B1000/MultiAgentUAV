"""Stage B repertoire-birth preset family.

These presets train isolated per-strategy adapters with JSD-separated rewards.
They are entered after Stage A (map-aware generalist) has produced a competent
shared trunk.

Canonical presets
-----------------
* ``plan_faithful_latent_v6i9_mapaware_repertoire_hardpool`` — primary
* ``plan_faithful_latent_v5i8_repertoire_uniform_z``         — V5 baseline
"""
from rl.presets.plan_faithful import (
    apply_plan_faithful_latent_v5i8_repertoire_uniform_z,
    apply_plan_faithful_latent_v6i9_mapaware_repertoire_hardpool,
)

__all__ = [
    "apply_plan_faithful_latent_v6i9_mapaware_repertoire_hardpool",
    "apply_plan_faithful_latent_v5i8_repertoire_uniform_z",
]
