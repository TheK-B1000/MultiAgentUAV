"""Phase A gate scheduling and terminal-failure predicates."""

from __future__ import annotations

from dataclasses import dataclass

from rl.config.ppo_config import PPOConfig
from rl.custom_ppo.curriculum.types import CurriculumPhase, GateMode


@dataclass(frozen=True)
class CurriculumSchedule:
    """Resolved phase-boundary and gate-check schedule from PPOConfig."""

    nominal_steps: int
    phase_a_min_end: int
    phase_a_max_end: int
    phase_b_nominal_start: int
    phase_c_nominal_start: int
    phase_b_budget_steps: int
    phase_c_budget_steps: int
    gate_check_interval: int

    def terminal_step_if_promoted_at(self, phase_a_end_step: int) -> int:
        """Full training budget when Phase A ends at ``phase_a_end_step``."""
        return int(phase_a_end_step) + int(self.phase_b_budget_steps) + int(self.phase_c_budget_steps)

    def terminal_step_at_nominal_phase_a_end(self) -> int:
        return self.terminal_step_if_promoted_at(self.phase_a_max_end)


def resolve_schedule(cfg: PPOConfig) -> CurriculumSchedule:
    nominal_steps = int(cfg.curriculum_nominal_timesteps)
    min_frac = float(getattr(cfg, "phase_a_earliest_end_fraction", 0.40) or 0.40)
    max_frac = float(getattr(cfg, "phase_a_max_end_fraction", 0.55) or 0.55)
    b_frac = float(getattr(cfg, "phase_b_fixed_fraction", 0.30) or 0.30)
    c_frac = float(getattr(cfg, "phase_c_fixed_fraction", 0.30) or 0.30)
    c_frac_nominal = float(getattr(cfg, "phase_c_start_fraction", 0.70) or 0.70)
    phase_a_min_end = int(min_frac * nominal_steps)
    return CurriculumSchedule(
        nominal_steps=nominal_steps,
        phase_a_min_end=phase_a_min_end,
        phase_a_max_end=int(max_frac * nominal_steps),
        phase_b_nominal_start=phase_a_min_end,
        phase_c_nominal_start=int(c_frac_nominal * nominal_steps),
        phase_b_budget_steps=int(b_frac * nominal_steps),
        phase_c_budget_steps=int(c_frac * nominal_steps),
        gate_check_interval=max(1, int(cfg.phase_a_gate_check_interval)),
    )


def format_staged_curriculum_budget_contract(
    cfg: PPOConfig,
    *,
    effective_terminal: int | None = None,
) -> list[str]:
    """Human-readable staged A/B/C budget lines for the startup banner."""
    schedule = resolve_schedule(cfg)
    nominal = schedule.nominal_steps
    eff = int(effective_terminal if effective_terminal is not None else cfg.total_timesteps)
    max_terminal = schedule.terminal_step_at_nominal_phase_a_end()
    return [
        f"[PPO] Nominal curriculum budget: {nominal:,}",
        f"[PPO] Current effective terminal: {eff:,}",
        (
            f"[PPO] Phase A window: {schedule.phase_a_min_end:,}"
            f"–{schedule.phase_a_max_end:,}"
        ),
        f"[PPO] Post-promotion Phase B budget: {schedule.phase_b_budget_steps:,}",
        f"[PPO] Post-promotion Phase C budget: {schedule.phase_c_budget_steps:,}",
        (
            f"[PPO] Maximum terminal step if Phase A ends at max window: "
            f"{max_terminal:,}"
        ),
    ]


def format_terminal_extension_banner(
    schedule: CurriculumSchedule,
    *,
    phase_a_end_step: int,
    effective_terminal: int,
) -> list[str]:
    """Stdout lines after late Phase A promotion extends the training terminal."""
    return [
        f"[Curriculum Controller] Phase A promoted at: {int(phase_a_end_step):,}",
        f"[Curriculum Controller] Effective terminal extended to: {int(effective_terminal):,}",
        f"[Curriculum Controller] Phase B budget: {schedule.phase_b_budget_steps:,}",
        f"[Curriculum Controller] Phase C budget: {schedule.phase_c_budget_steps:,}",
    ]


def should_run_phase_a_gate(
    *,
    schedule: CurriculumSchedule,
    phase: str,
    global_step: int,
    last_gate_step_run: int,
    next_gate_step: int,
) -> bool:
    """Return True when a Phase A gate check is due at ``global_step``."""
    step = int(global_step)
    if phase != CurriculumPhase.A.value:
        return False
    if step < schedule.phase_a_min_end:
        return False
    if step > schedule.phase_a_max_end and last_gate_step_run >= schedule.phase_a_max_end:
        return False
    if step == last_gate_step_run:
        return False
    due_by_schedule = step >= next_gate_step
    due_at_final_boundary = (
        step >= schedule.phase_a_max_end and last_gate_step_run < schedule.phase_a_max_end
    )
    if not (due_by_schedule or due_at_final_boundary):
        return False
    if step > schedule.phase_a_max_end and not due_at_final_boundary:
        return False
    return True


def schedule_next_gate_step(
    schedule: CurriculumSchedule,
    *,
    step: int,
) -> int:
    """Compute the next scheduled gate step after a gate run at ``step``."""
    current = int(step)
    if current >= schedule.phase_a_max_end:
        return schedule.phase_a_max_end + 1
    candidate = current + schedule.gate_check_interval
    if candidate > schedule.phase_a_max_end:
        return schedule.phase_a_max_end
    return candidate


def should_trigger_terminal_failure(
    *,
    mode: str,
    phase: str,
    global_step: int,
    phase_a_max_end: int,
    last_gate_step_run: int,
    phase_a_gate_passed: bool,
) -> bool:
    """True when enforce-mode training must halt after the final Phase A gate fails."""
    if GateMode.normalize(mode) == GateMode.OBSERVE_ONLY.value:
        return False
    step = int(global_step)
    final_gate_completed = last_gate_step_run >= int(phase_a_max_end)
    return (
        phase == CurriculumPhase.A.value
        and step >= int(phase_a_max_end)
        and final_gate_completed
        and not bool(phase_a_gate_passed)
    )


__all__ = [
    "CurriculumSchedule",
    "format_staged_curriculum_budget_contract",
    "format_terminal_extension_banner",
    "resolve_schedule",
    "schedule_next_gate_step",
    "should_run_phase_a_gate",
    "should_trigger_terminal_failure",
]
