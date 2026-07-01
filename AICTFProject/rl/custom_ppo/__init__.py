from __future__ import annotations

from rl.custom_ppo.policy import SharedActorCentralizedCritic
from rl.custom_ppo.inference import (
    CustomPPOInferencePolicy,
    read_custom_ppo_metadata,
    load_custom_ppo_policy,
    apply_deterministic_sampling_generators,
    _torch_load_checkpoint,
    CUSTOM_PPO_ACTOR_ARCH,
    CUSTOM_PPO_FORMAT,
    CUSTOM_PPO_LATENT_FORMAT,
    CUSTOM_PPO_VEC_SCHEMA_VERSION,
)
from rl.custom_ppo.trainer import CustomPPOTrainer, _compose_training_reward_components
from rl.custom_ppo.csv_writers import E3_STEP_TELEMETRY_FIELDS, _METRICS_CSV_LEGACY_COLUMN_FILL

# Phase 1 / 1.5 public surface: inference contract, diagnostics contract, distribution types.
from rl.custom_ppo.distributions import ActionHead, MultiHeadActionDistribution
from rl.custom_ppo.policy_contract import PolicyInferenceContract
from rl.custom_ppo.diagnostics_contract import PolicyDiagnosticsContract
from rl.custom_ppo.probe_result import (
    ProbeResult,
    WeightProbeResult,
    GradientProbeResult,
    CounterfactualProbeResult,
    PROBE_SUCCESS,
    PROBE_ERROR,
)

__all__ = [
    # Core training objects
    "SharedActorCentralizedCritic",
    "CustomPPOInferencePolicy",
    "read_custom_ppo_metadata",
    "load_custom_ppo_policy",
    "apply_deterministic_sampling_generators",
    "CustomPPOTrainer",
    "_compose_training_reward_components",
    "_torch_load_checkpoint",
    "E3_STEP_TELEMETRY_FIELDS",
    "_METRICS_CSV_LEGACY_COLUMN_FILL",
    "CUSTOM_PPO_ACTOR_ARCH",
    "CUSTOM_PPO_FORMAT",
    "CUSTOM_PPO_LATENT_FORMAT",
    "CUSTOM_PPO_VEC_SCHEMA_VERSION",
    # Phase 1 / 1.5: inference and diagnostics contracts
    "PolicyInferenceContract",
    "PolicyDiagnosticsContract",
    "ActionHead",
    "MultiHeadActionDistribution",
    # Phase 1: probe results
    "ProbeResult",
    "WeightProbeResult",
    "GradientProbeResult",
    "CounterfactualProbeResult",
    "PROBE_SUCCESS",
    "PROBE_ERROR",
]
