"""Small scalar schedules used by custom PPO training."""

from __future__ import annotations

from typing import Any


def linear_anneal(
    step: int | float,
    start_value: float,
    end_value: float,
    start_step: int,
    end_step: int,
) -> float:
    """Linearly interpolate between two scalar values inside a step window."""
    step_f = float(step)
    start_step_i = int(start_step)
    end_step_i = int(end_step)
    start_value_f = float(start_value)
    end_value_f = float(end_value)

    if end_step_i <= start_step_i:
        return end_value_f if step_f >= end_step_i else start_value_f
    if step_f <= start_step_i:
        return start_value_f
    if step_f >= end_step_i:
        return end_value_f

    progress = (step_f - start_step_i) / float(end_step_i - start_step_i)
    return start_value_f + progress * (end_value_f - start_value_f)


def resolve_latent_lam_h(cfg: Any, *, global_step: int | float, total_timesteps: int) -> float:
    """Resolve the current latent entropy coefficient without changing old configs."""
    run_tag = str(getattr(cfg, "run_tag", "") or "")
    if "v3h2" in run_tag:
        step_f = float(global_step)
        late_floor = float(getattr(cfg, "late_entropy_floor", 0.0003))
        if step_f < 300_000:
            return 0.003
        elif step_f < 600_000:
            # Linear anneal from 0.003 to 0.001
            progress = (step_f - 300_000) / 300_000.0
            return 0.003 + progress * (0.001 - 0.003)
        else:
            # Linear anneal from 0.001 to late_floor (0.0003) at total_timesteps
            end_step = float(total_timesteps)
            if end_step <= 600_000:
                return late_floor
            if step_f >= end_step:
                return late_floor
            progress = (step_f - 600_000) / (end_step - 600_000)
            return 0.001 + progress * (late_floor - 0.001)

    lam_h_start = getattr(cfg, "latent_lam_h_start", None)
    if lam_h_start is None:
        lam_h_start = getattr(cfg, "latent_lam_h", 0.0) or 0.0
    lam_h_start = max(0.0, float(lam_h_start))

    lam_h_end = getattr(cfg, "latent_lam_h_end", None)
    if lam_h_end is None:
        lam_h_end = lam_h_start
    lam_h_end = max(0.0, float(lam_h_end))

    anneal_start = getattr(cfg, "latent_entropy_anneal_start", None)
    if anneal_start is None:
        anneal_start = 0

    anneal_end = getattr(cfg, "latent_entropy_anneal_end", None)
    if anneal_end is None:
        anneal_end = int(total_timesteps)

    return linear_anneal(
        global_step,
        lam_h_start,
        lam_h_end,
        int(anneal_start),
        int(anneal_end),
    )


def resolve_latent_forced_z_frac(cfg: Any, *, global_step: int | float) -> float:
    """Resolve the current forced-z episode fraction from a (possibly annealed) schedule.

    Reads four optional config fields:

    * ``latent_forced_z_episode_frac_start`` -- value before ``anneal_start``
    * ``latent_forced_z_episode_frac_end`` -- value after ``anneal_end``
    * ``latent_forced_z_anneal_start`` -- step at which the linear ramp begins
    * ``latent_forced_z_anneal_end`` -- step at which the linear ramp finishes

    If any of these is None, the resolver falls back to the legacy constant
    ``latent_forced_z_episode_frac`` field. This keeps every pre-v5i3 preset
    (including v5i2 with ``latent_forced_z_episode_frac = 0.0``) bit-for-bit
    identical: zero start/end fields → constant legacy value at every step.

    Resume-safety: the resolver is a pure function of ``cfg`` plus the
    ``global_step`` passed in. The trainer's ``self.global_step`` is restored
    from the checkpoint before the rollout loop resumes, so a run that
    re-enters mid-anneal picks up the schedule from the restored step
    rather than restarting from zero.
    """
    start = getattr(cfg, "latent_forced_z_episode_frac_start", None)
    end = getattr(cfg, "latent_forced_z_episode_frac_end", None)
    anneal_start = getattr(cfg, "latent_forced_z_anneal_start", None)
    anneal_end = getattr(cfg, "latent_forced_z_anneal_end", None)

    if start is None or end is None or anneal_start is None or anneal_end is None:
        legacy = float(getattr(cfg, "latent_forced_z_episode_frac", 0.0) or 0.0)
        return max(0.0, min(legacy, 1.0))

    value = linear_anneal(
        global_step,
        float(start),
        float(end),
        int(anneal_start),
        int(anneal_end),
    )
    return max(0.0, min(value, 1.0))


def resolve_v6i1_cf_coef(phase: str, step: int | float, t_A: int | float, N: int | float, coef_max: float) -> float:
    """Resolve the counterfactual separation coefficient for v6i1 staged curriculum.

    Phase A locked schedule (fractions of nominal budget ``N``):

    * ``0.00N–0.10N``: CF = 0
    * ``0.10N–0.20N``: linear ramp to ``coef_max``
    * ``0.20N`` onward: CF = ``coef_max``

    Runs started before this schedule used ``0.20N–0.40N`` ramp; retain those
    artifacts under their original run IDs rather than retroactively relabeling.
    """
    step_f = float(step)
    N_f = float(N)
    if phase == "A":
        if step_f < 0.10 * N_f:
            return 0.0
        if step_f < 0.20 * N_f:
            return linear_anneal(
                step_f,
                0.0,
                coef_max,
                int(0.10 * N_f),
                int(0.20 * N_f),
            )
        return float(coef_max)
    elif phase == "B":
        return 0.0
    elif phase == "C":
        return 0.25 * float(coef_max)
    return 0.0


def resolve_v6i1_forced_fraction(phase: str, step: int | float, t_A: int | float, N: int | float) -> float:
    """Resolve the forced uniform episode fraction for v6i1 staged curriculum."""
    step_f = float(step)
    t_A_i = int(t_A)
    N_f = float(N)
    if phase == "A":
        return 1.0
    elif phase == "B":
        # Linearly anneal from 0.50 to 0.25 across Phase B duration (0.30N)
        return linear_anneal(step_f, 0.50, 0.25, t_A_i, t_A_i + int(0.30 * N_f))
    elif phase == "C":
        return 0.25
    return 0.0


def resolve_v6i1_exploration_epsilon(phase: str, step: int | float, t_A: int | float, N: int | float) -> float:
    """Resolve the selector exploration floor epsilon for v6i1 staged curriculum."""
    step_f = float(step)
    t_A_i = int(t_A)
    N_f = float(N)
    if phase == "A":
        return 0.0
    elif phase == "B":
        # Linearly anneal from 0.20 to 0.05 across Phase B duration (0.30N)
        return linear_anneal(step_f, 0.20, 0.05, t_A_i, t_A_i + int(0.30 * N_f))
    elif phase == "C":
        return 0.05
    return 0.05


def resolve_v6i1_usage_coef(phase: str, step: int | float, t_A: int | float, N: int | float) -> float:
    """Resolve the marginal usage objective coefficient lambda_usage for v6i1 staged curriculum."""
    step_f = float(step)
    t_A_i = int(t_A)
    N_f = float(N)
    if phase == "A":
        return 0.0
    elif phase == "B":
        # Linearly anneal from 0.003 to 0.001 across Phase B duration (0.30N)
        return linear_anneal(step_f, 0.003, 0.001, t_A_i, t_A_i + int(0.30 * N_f))
    elif phase == "C":
        return 0.001
    return 0.0

