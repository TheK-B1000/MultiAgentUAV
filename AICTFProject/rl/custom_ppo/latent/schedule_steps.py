"""Resolve absolute schedule steps from config absolute or fractional knobs."""

from __future__ import annotations


def resolve_schedule_step(
    *,
    absolute_step: int | None,
    fraction: float | None,
    nominal_steps: int,
) -> int | None:
    if absolute_step is not None and int(absolute_step) > 0:
        return int(absolute_step)
    if fraction is not None and float(fraction) > 0.0:
        return int(float(fraction) * int(nominal_steps))
    return None
