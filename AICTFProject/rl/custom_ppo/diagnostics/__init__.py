"""rl.custom_ppo.diagnostics — typed latent strategy diagnostics package.

Public API surface:

    DiagnosticStatus, DiagnosticResult, DiagnosticError
        Typed result envelope for every diagnostic check.

    shannon_entropy_nats, mi_z_vs
        Pure entropy / MI computations.

    jsd_from_logits, compute_pairwise_actor_jsd
        JSD-based actor differentiation metrics.

    compute_adapter_grad_norms, compute_critic_value_variance
        Model-level competence diagnostics.

    validate_occupancy_entropy, validate_jsd_separation,
    validate_mi_proxy, validate_unique_latents
        Typed validators that return DiagnosticResult.

Legacy trainer API (underscore-prefixed) is re-exported from
rl.custom_ppo.latent_diagnostics — the thin facade that bridges old callers
to this package.
"""

from rl.custom_ppo.diagnostics.results import DiagnosticError, DiagnosticResult, DiagnosticStatus
from rl.custom_ppo.diagnostics.schemas import LATENT_DIAGNOSTICS_VERSION, LATENT_METRICS_SCHEMA_VERSION
from rl.custom_ppo.diagnostics.entropy import (
    mi_z_vs,
    shannon_entropy_nats,
    bucket_z_fracs,
    fill_zero_z_fracs,
    flat_long_np,
    flat_float_np,
)
from rl.custom_ppo.diagnostics.occupancy import compute_occupancy_stats
from rl.custom_ppo.diagnostics.counterfactual import (
    jsd_from_logits,
    compute_pairwise_actor_jsd,
)
from rl.custom_ppo.diagnostics.competence import (
    compute_adapter_grad_norms,
    compute_critic_value_variance,
)
from rl.custom_ppo.diagnostics.validation import (
    validate_jsd_separation,
    validate_mi_proxy,
    validate_occupancy_entropy,
    validate_unique_latents,
)

__all__ = [
    "DiagnosticStatus",
    "DiagnosticResult",
    "DiagnosticError",
    "LATENT_DIAGNOSTICS_VERSION",
    "LATENT_METRICS_SCHEMA_VERSION",
    "shannon_entropy_nats",
    "mi_z_vs",
    "bucket_z_fracs",
    "fill_zero_z_fracs",
    "flat_long_np",
    "flat_float_np",
    "compute_occupancy_stats",
    "jsd_from_logits",
    "compute_pairwise_actor_jsd",
    "compute_adapter_grad_norms",
    "compute_critic_value_variance",
    "validate_occupancy_entropy",
    "validate_jsd_separation",
    "validate_mi_proxy",
    "validate_unique_latents",
]
