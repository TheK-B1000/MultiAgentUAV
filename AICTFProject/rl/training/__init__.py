"""Training-pipeline utilities for the local PPO trainer.

Submodules:

* :mod:`rl.training.run_artifacts` -- git metadata, ``run_config.json``
  sidecar, metrics CSV rotation, run-tag lockfile (file/run hygiene only).

Imports are kept narrow to avoid pulling in torch / GPU env when callers only
need filesystem hygiene helpers.
"""

__all__: list[str] = []
