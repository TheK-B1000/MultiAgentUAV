"""Periodic checkpoint scheduling for fresh runs and resumed stages."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CheckpointSchedule:
    mode: str  # "disabled" | "run_progress" | "global"
    interval: int
    run_start_step: int = 0
    next_progress: int = 0
    next_global_step: int = 0


def resolve_checkpoint_schedule(
    *,
    global_step: int,
    interval: int,
    checkpoint_run_start_step: int = 0,
    additional_timesteps: int = 0,
    load_weights_only: bool = False,
) -> CheckpointSchedule:
    """Build the initial periodic-checkpoint schedule after load."""
    if interval <= 0:
        return CheckpointSchedule(mode="disabled", interval=0)

    gs = int(global_step)
    run_start = int(checkpoint_run_start_step or 0)
    use_run_progress = int(additional_timesteps) > 0 or bool(load_weights_only)

    if use_run_progress:
        if run_start <= 0:
            run_start = gs
        progress = max(0, gs - run_start)
        completed = progress // interval
        next_progress = (completed + 1) * interval
        return CheckpointSchedule(
            mode="run_progress",
            interval=interval,
            run_start_step=run_start,
            next_progress=next_progress,
        )

    next_global = interval if gs <= 0 else ((gs // interval) + 1) * interval
    return CheckpointSchedule(
        mode="global",
        interval=interval,
        next_global_step=next_global,
    )


def checkpoint_due(schedule: CheckpointSchedule, global_step: int) -> int | None:
    """Return the checkpoint filename step label if a save is due, else None."""
    if schedule.mode == "disabled" or schedule.interval <= 0:
        return None

    gs = int(global_step)
    if schedule.mode == "run_progress":
        progress = gs - int(schedule.run_start_step)
        if progress >= int(schedule.next_progress):
            return int(schedule.next_progress)
        return None

    if gs >= int(schedule.next_global_step):
        return int(schedule.next_global_step)
    return None


def advance_checkpoint_schedule(schedule: CheckpointSchedule) -> None:
    """Advance the schedule after a checkpoint has been written."""
    if schedule.mode == "run_progress":
        schedule.next_progress += int(schedule.interval)
    elif schedule.mode == "global":
        schedule.next_global_step += int(schedule.interval)


def format_checkpoint_schedule_banner(schedule: CheckpointSchedule) -> str:
    if schedule.mode == "disabled":
        return "[PPO] Checkpoint schedule: disabled"
    if schedule.mode == "run_progress":
        return (
            f"[PPO] Checkpoint schedule: run-relative every {schedule.interval:,} steps "
            f"(run_start={schedule.run_start_step:,}, "
            f"next_progress={schedule.next_progress:,})"
        )
    return (
        f"[PPO] Checkpoint schedule: global step every {schedule.interval:,} "
        f"(next={schedule.next_global_step:,})"
    )


__all__ = [
    "CheckpointSchedule",
    "advance_checkpoint_schedule",
    "checkpoint_due",
    "format_checkpoint_schedule_banner",
    "resolve_checkpoint_schedule",
]
