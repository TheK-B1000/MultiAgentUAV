"""Console formatting helpers for telemetry summaries."""

from __future__ import annotations

from rl.custom_ppo.telemetry.events import OptimizationCompleted


def format_optimization_completed(event: OptimizationCompleted) -> str:
    ev = "-" if event.explained_variance is None else f"{event.explained_variance:.3f}"
    return (
        "[telemetry|opt] "
        f"steps={event.global_step} samples={event.samples_processed} "
        f"policy_loss={event.policy_loss:.4f} value_loss={event.value_loss:.4f} "
        f"approx_kl={event.approx_kl:.5f} ev={ev}"
    )


__all__ = ["format_optimization_completed"]
