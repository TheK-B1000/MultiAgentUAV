"""PPO update loop decomposition (coordinator + explicit owners)."""

from rl.custom_ppo.update.actor_intervention import ActorInterventionEvidenceUpdater
from rl.custom_ppo.update.loss_result import (
    LossComponent,
    MinibatchUpdateResult,
    PairwiseSeparationMeasurement,
)
from rl.custom_ppo.update.optimizer_stepper import clip_optimizer_grad_norm
from rl.custom_ppo.update.pair_utils import latent_pair_count, validate_v6_protocol_latent_k
from rl.custom_ppo.update.phase_policy import (
    PhaseTrainingPolicy,
    apply_phase_requires_grad,
    resolve_training_phase,
)
from rl.custom_ppo.update.update_context import PPOUpdateContext, PPOUpdateContextBuilder

__all__ = [
    "ActorInterventionEvidenceUpdater",
    "LossComponent",
    "MinibatchUpdateResult",
    "PairwiseSeparationMeasurement",
    "PPOUpdateContext",
    "PPOUpdateContextBuilder",
    "PhaseTrainingPolicy",
    "apply_phase_requires_grad",
    "clip_optimizer_grad_norm",
    "latent_pair_count",
    "resolve_training_phase",
    "validate_v6_protocol_latent_k",
]
