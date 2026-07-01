"""Preset family sub-package.

Each module groups preset functions by training lineage:
* v6i8 — adapter-based specialists (V6I8 lineage)
* v6i9 — map-aware generalist → repertoire → router (V6I9 lineage)
* repertoire — Stage B repertoire birth presets
* router — Stage D router presets

These modules re-export the underlying ``apply_*`` functions from
``rl.presets.plan_faithful`` and are the canonical home for new presets
going forward.
"""
