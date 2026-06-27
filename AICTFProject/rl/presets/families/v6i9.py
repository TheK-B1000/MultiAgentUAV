"""V6I9 preset family — map-aware generalist → repertoire → router.

Stage A: Map-aware generalist (``v6i9_training_stage = "stage1_mapaware_generalist"``)
    The CNN trunk is conditioned on the obstacle map.  A single shared actor
    is trained on the full hard opponent pool to establish a competent generalist
    before repertoire birth.

Stage B: Repertoire birth (``v6i9_training_stage = "stage2_repertoire"``)
    Isolated adapters are added and trained with JSD-separated rewards.

Stage D: Router (``v6i9_training_stage = "stage3_router"``)
    The recurrent router selects adapters.

Canonical presets
-----------------
* ``plan_faithful_latent_v6i9_mapaware_generalist_hardpool``       — Stage A main
* ``plan_faithful_latent_v6i9_mapaware_generalist_hardpool_split`` — Stage A split-lane
* ``plan_faithful_latent_v6i9_mapaware_repertoire_hardpool``       — Stage B
* ``plan_faithful_latent_v6i9_mapaware_router_sparse_hardpool``    — Stage D
"""
from rl.presets.plan_faithful import (
    apply_plan_faithful_latent_v6i9_mapaware_generalist_hardpool,
    apply_plan_faithful_latent_v6i9_mapaware_generalist_hardpool_split,
    apply_plan_faithful_latent_v6i9_mapaware_nav_refinement,
    apply_plan_faithful_latent_v6i9_mapaware_repertoire_hardpool,
    apply_plan_faithful_latent_v6i9_mapaware_router_sparse_hardpool,
)

__all__ = [
    "apply_plan_faithful_latent_v6i9_mapaware_generalist_hardpool",
    "apply_plan_faithful_latent_v6i9_mapaware_generalist_hardpool_split",
    "apply_plan_faithful_latent_v6i9_mapaware_nav_refinement",
    "apply_plan_faithful_latent_v6i9_mapaware_repertoire_hardpool",
    "apply_plan_faithful_latent_v6i9_mapaware_router_sparse_hardpool",
]
