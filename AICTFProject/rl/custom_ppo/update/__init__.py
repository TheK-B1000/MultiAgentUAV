"""PPO update loop decomposition (coordinator + explicit owners)."""

from rl.custom_ppo.update.actor_intervention import ActorInterventionEvidenceUpdater
from rl.custom_ppo.update.entropy_objectives import EntropyObjective, RolloutEntropyState
from rl.custom_ppo.update.loss_result import (
    LossComponent,
    MinibatchUpdateResult,
    PairwiseSeparationMeasurement,
)
from rl.custom_ppo.update.minibatch_updater import MinibatchUpdater
from rl.custom_ppo.update.optimizer_stepper import (
    OptimizerStepResult,
    SharedOptimizerStepper,
    ThreeOptimizerStepper,
    build_optimizer_stepper,
    clip_optimizer_grad_norm,
)
from rl.custom_ppo.update.pair_utils import latent_pair_count, validate_v6_protocol_latent_k
from rl.custom_ppo.update.param_registry import OptimizerRegistry, ParameterRegistry
from rl.custom_ppo.update.phase_policy import (
    PhaseTrainingPolicy,
    apply_phase_requires_grad,
    resolve_training_phase,
    set_model_requires_grad_for_phase,
)
from rl.custom_ppo.update.post_update import PostUpdatePipeline, PostUpdateResult
from rl.custom_ppo.update.separation_objectives import SeparationObjective, SeparationResult
from rl.custom_ppo.update.strategy_objectives import StrategyLossBundle, StrategyObjective
from rl.custom_ppo.update.telemetry import AggregationMode, UpdateStatsAccumulator, build_metric_schema
from rl.custom_ppo.update.update_context import PPOUpdateContext, PPOUpdateContextBuilder
from rl.custom_ppo.update.updater import PPOUpdater

__all__ = [
    "ActorInterventionEvidenceUpdater",
    "AggregationMode",
    "EntropyObjective",
    "LossComponent",
    "MinibatchUpdateResult",
    "MinibatchUpdater",
    "OptimizerRegistry",
    "OptimizerStepResult",
    "PPOUpdateContext",
    "PPOUpdateContextBuilder",
    "PPOUpdater",
    "PairwiseSeparationMeasurement",
    "ParameterRegistry",
    "PhaseTrainingPolicy",
    "PostUpdatePipeline",
    "PostUpdateResult",
    "RolloutEntropyState",
    "SeparationObjective",
    "SeparationResult",
    "SharedOptimizerStepper",
    "StrategyLossBundle",
    "StrategyObjective",
    "ThreeOptimizerStepper",
    "UpdateStatsAccumulator",
    "apply_phase_requires_grad",
    "build_metric_schema",
    "build_optimizer_stepper",
    "clip_optimizer_grad_norm",
    "latent_pair_count",
    "resolve_training_phase",
    "set_model_requires_grad_for_phase",
    "validate_v6_protocol_latent_k",
]
