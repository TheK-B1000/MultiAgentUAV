"""Typed validators that return DiagnosticResult with full observability context.

Every validator preserves observed metric, threshold, sample count, and reason
so that WARN / FAIL verdicts are self-describing without secondary tables.
"""

from __future__ import annotations

from rl.custom_ppo.diagnostics.results import DiagnosticResult, DiagnosticStatus


def validate_occupancy_entropy(
    stats: dict[str, float],
    min_entropy_nats: float = 0.5,
    *,
    metric_version: str = "legacy_v1",
) -> DiagnosticResult[float]:
    """PASS when latent marginal entropy exceeds ``min_entropy_nats``.

    A low entropy indicates occupancy collapse: the router almost always
    selects the same latent regardless of context.
    """
    key = "latent_marginal_entropy_nats"
    if key not in stats:
        return DiagnosticResult.unavailable(f"{key} not in stats")
    value = float(stats[key])
    sample_count = int(stats.get("strategy_unique_count", 0))
    if sample_count == 0:
        return DiagnosticResult.unavailable("zero samples")
    if value >= min_entropy_nats:
        status = DiagnosticStatus.PASS
        reason = f"entropy {value:.4f} >= threshold {min_entropy_nats:.4f}"
    else:
        status = DiagnosticStatus.FAIL
        reason = f"entropy {value:.4f} < threshold {min_entropy_nats:.4f} (collapse)"
    return DiagnosticResult(
        status=status,
        value=value,
        sample_count=sample_count,
        reason=f"[{metric_version}] {reason}",
    )


def validate_jsd_separation(
    stats: dict[str, float],
    min_jsd: float = 0.001,
    *,
    metric_version: str = "legacy_v1",
) -> DiagnosticResult[float]:
    """PASS when mean pairwise actor JSD exceeds ``min_jsd``.

    A JSD near zero means all latent IDs produce nearly identical action
    distributions — the actor ignores the latent conditioning.
    """
    key = "actor_z_jsd_mean"
    if key not in stats:
        return DiagnosticResult.unavailable(f"{key} not in stats")
    value = float(stats[key])
    pair_count = int(stats.get("actor_z_pairs_total", 0))
    if pair_count == 0:
        return DiagnosticResult.unavailable("no latent pairs evaluated")
    if value >= min_jsd:
        status = DiagnosticStatus.PASS
        reason = f"JSD {value:.6f} >= threshold {min_jsd:.6f}"
    else:
        status = DiagnosticStatus.WARN
        reason = f"JSD {value:.6f} < threshold {min_jsd:.6f} (weak separation)"
    return DiagnosticResult(
        status=status,
        value=value,
        sample_count=pair_count,
        reason=f"[{metric_version}] {reason}",
    )


def validate_mi_proxy(
    stats: dict[str, float],
    min_mi_nats: float = 0.01,
    *,
    metric_version: str = "legacy_v1",
) -> DiagnosticResult[float]:
    """PASS when MI(z; phase) proxy exceeds ``min_mi_nats``.

    Near-zero MI means the router selects latents independently of observable
    context (phase, flag state, opponent), indicating the router learned nothing.
    """
    key = "latent_mi_z_phase_nats"
    if key not in stats:
        return DiagnosticResult.unavailable(f"{key} not in stats")
    value = float(stats[key])
    if value >= min_mi_nats:
        status = DiagnosticStatus.PASS
        reason = f"MI(z;phase) {value:.5f} >= threshold {min_mi_nats:.5f}"
    else:
        status = DiagnosticStatus.WARN
        reason = f"MI(z;phase) {value:.5f} < threshold {min_mi_nats:.5f} (context-blind routing)"
    return DiagnosticResult(
        status=status,
        value=value,
        sample_count=int(stats.get("strategy_unique_count", 0)),
        reason=f"[{metric_version}] {reason}",
    )


def validate_unique_latents(
    stats: dict[str, float],
    min_unique: int = 2,
    *,
    metric_version: str = "legacy_v1",
) -> DiagnosticResult[int]:
    """PASS when at least ``min_unique`` distinct latents appeared in the rollout."""
    key = "strategy_unique_count"
    if key not in stats:
        return DiagnosticResult.unavailable(f"{key} not in stats")
    value = int(stats[key])
    if value >= min_unique:
        status = DiagnosticStatus.PASS
        reason = f"{value} unique latents >= required {min_unique}"
    else:
        status = DiagnosticStatus.FAIL
        reason = f"only {value} unique latent(s) observed (required >= {min_unique})"
    return DiagnosticResult(
        status=status,
        value=value,
        sample_count=value,
        reason=f"[{metric_version}] {reason}",
    )


__all__ = [
    "validate_occupancy_entropy",
    "validate_jsd_separation",
    "validate_mi_proxy",
    "validate_unique_latents",
]
