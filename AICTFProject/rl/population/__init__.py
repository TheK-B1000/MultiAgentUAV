"""V6I24 population package — DEFERRED heavy infrastructure.

The lean V6I24 diagnostic uses ordinary independent ``train_ppo`` runs
orchestrated by ``experiments/run_v6i24_full_policy_population.py``.

Do **not** wire ``PopulationTrainer``, pressure rotation, PFSP, Nash, or
distillation into the first diagnostic arm. Build those only after four
independent teachers demonstrate separation under fixed cell pressures.
"""
from __future__ import annotations

# Deferred exports retained for later V6I24B / platform work only.
from rl.population.population_member import PopulationMember, PopulationMemberConfig
from rl.population.population_trainer import PopulationTrainer
from rl.population.pressure_rotation import rotate_pressures

__all__ = [
    "PopulationMember",
    "PopulationMemberConfig",
    "PopulationTrainer",
    "rotate_pressures",
]
