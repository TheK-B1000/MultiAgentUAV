"""Resolved training configuration: computed values derived from a ``PPOConfig``.

:class:`ResolvedTrainingConfig` is a frozen dataclass produced by
:func:`resolve_training_config`.  It captures every value that
:func:`rl.train_ppo.train_ppo` previously computed inline before constructing
the trainer — max agents, team size, curriculum state, opponent seeding, and
the effective PPO hyperparameters after multi-agent learning-rate scaling.

Keeping these computations here removes the coupling between the training
orchestrator and the PPOConfig field layout: the orchestrator consumes a
typed, immutable snapshot rather than picking attributes off the config at
multiple call sites.

Backward-compat note: :func:`_resolve_initial_opponent_and_phase` is extracted
from :mod:`rl.train_ppo` and re-exported from there for existing import paths.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from rl.config.ppo_config import PPOConfig, TrainMode
from rl.curriculum import CurriculumState, jacob_paper_curriculum_state, phase_from_tag
from rl.training.config_validation import _normalize_train_mode


# ---------------------------------------------------------------------------
# Internal helpers (previously in train_ppo.py)
# ---------------------------------------------------------------------------

def _resolve_initial_opponent_and_phase(
    cfg: PPOConfig, max_agents: int
) -> tuple[Optional[CurriculumState], str, str]:
    """Pick the initial scripted opponent tag and phase for the run.

    In ``CURRICULUM`` mode this also constructs the :class:`CurriculumState`
    that the trainer will advance between phases.

    In ``OPPONENT_POOL`` mode (or with ``opponent_randomize=True``) the
    very first env reset uses ``set_next_opponent("SCRIPTED", initial_tag)``
    to seed the env before pool sampling takes over on subsequent resets.
    If ``cfg.fixed_opponent_tag`` is not in ``cfg.opponent_pool`` (the
    common case: the v4i1/v4i3/v5* chain leaves ``fixed_opponent_tag``
    at the ``"OP3"`` default while configuring a pool like
    ``("OP5", "OP6", "OP7")``), that first episode would leak the
    out-of-pool opponent into the very first telemetry slice, producing
    a misleading "OP3 slice" diagnostic even though the audit banner
    correctly reports the configured pool. To make the first episode
    consistent with the audit, fall back to the first pool entry when
    the legacy ``fixed_opponent_tag`` is out-of-pool. An explicit in-pool
    ``fixed_opponent_tag`` still wins (so users who want to seed a
    specific opener can do so).

    Returns ``(curriculum, initial_phase, initial_opponent_tag)``;
    ``curriculum`` is ``None`` outside curriculum mode.
    """
    curriculum: Optional[CurriculumState] = None
    initial_opponent_tag = str(cfg.fixed_opponent_tag).upper()
    initial_phase = phase_from_tag(initial_opponent_tag)
    if cfg.mode == TrainMode.CURRICULUM.value:
        curriculum = jacob_paper_curriculum_state(max_agents)
        initial_opponent_tag = curriculum.phase
        initial_phase = curriculum.phase
        return curriculum, initial_phase, initial_opponent_tag
    pool_mode = cfg.mode == TrainMode.OPPONENT_POOL.value or bool(
        getattr(cfg, "opponent_randomize", False)
    )
    if pool_mode:
        pool = tuple(str(tag).upper() for tag in (getattr(cfg, "opponent_pool", ()) or ()))
        if pool and initial_opponent_tag not in pool:
            initial_opponent_tag = pool[0]
            initial_phase = phase_from_tag(initial_opponent_tag)
    return curriculum, initial_phase, initial_opponent_tag


# ---------------------------------------------------------------------------
# Resolved config type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ResolvedTrainingConfig:
    """Immutable snapshot of computed training parameters.

    Produced by :func:`resolve_training_config` and consumed by
    lifecycle helpers and the orchestrator so they never re-derive the
    same values independently.
    """

    max_agents: int
    team_size: str
    curriculum: Optional[CurriculumState]
    initial_phase: str
    initial_opponent_tag: str
    effective_lr: float
    effective_ent_coef: float
    effective_clip_range: float
    effective_n_epochs: int
    effective_batch_size: int
    rollout_size: int


def resolve_training_config(cfg: PPOConfig) -> ResolvedTrainingConfig:
    """Derive all computed values from ``cfg`` and return a frozen snapshot.

    Mirrors the inline computation in the original ``train_ppo.train_ppo``
    function (lines that computed max_agents, lr scaling, batch size
    clamping, etc.) so the orchestrator body becomes a clean sequence
    of typed calls.
    """
    max_agents = max(1, int(getattr(cfg, "max_blue_agents", 2)))

    def _agents_suffix(n: int) -> str:
        n = max(1, min(int(n), 16))
        return f"{n}v{n}"

    team_size = _agents_suffix(max_agents)
    curriculum, initial_phase, initial_opponent_tag = _resolve_initial_opponent_and_phase(
        cfg, max_agents
    )

    learning_rate = float(cfg.learning_rate)
    ent_coef = float(cfg.ent_coef)
    clip_range = float(cfg.clip_range)
    n_epochs = int(cfg.n_epochs)
    batch_size = int(cfg.batch_size)

    if bool(getattr(cfg, "use_stable_marl_ppo", False)):
        learning_rate = 1.5e-4
        ent_coef = 0.005
        clip_range = 0.10
        n_epochs = 2
        batch_size = 1024

    if max_agents > 2:
        learning_rate *= 0.75

    rollout_size = max(1, int(cfg.n_steps) * max(1, int(cfg.n_envs)))
    if batch_size > rollout_size:
        batch_size = rollout_size

    return ResolvedTrainingConfig(
        max_agents=max_agents,
        team_size=team_size,
        curriculum=curriculum,
        initial_phase=initial_phase,
        initial_opponent_tag=initial_opponent_tag,
        effective_lr=learning_rate,
        effective_ent_coef=ent_coef,
        effective_clip_range=clip_range,
        effective_n_epochs=n_epochs,
        effective_batch_size=batch_size,
        rollout_size=rollout_size,
    )
