"""V6I1 staged curriculum Phase B/C runtime helpers."""

from __future__ import annotations

from typing import Any, Optional

import torch

from rl.custom_ppo.curriculum_gates import is_staged_v6i1_curriculum
from rl.custom_ppo.v6i1_cf_loss import v6i1_pair_suffix
from rl.custom_ppo.trainer_optimizers import TrainerOptimizerBundle
from rl.custom_ppo.schedules import (
    resolve_v6i1_cf_coef,
    resolve_v6i1_exploration_epsilon,
    resolve_v6i1_forced_fraction,
    resolve_v6i1_usage_coef,
)


def is_v6i1_staged_trainer(trainer: Any) -> bool:
    return (
        is_staged_v6i1_curriculum(trainer.cfg)
        and getattr(trainer, "v6i1_curriculum", None) is not None
    )


def v6i1_schedule_context(trainer: Any) -> tuple[str, int, int, int]:
    """Return ``(phase, global_step, t_A, nominal_steps)``."""
    curriculum = trainer.v6i1_curriculum
    step = int(getattr(trainer, "global_step", 0) or 0)
    phase = str(curriculum.resolve_phase(step))
    t_a = int(getattr(curriculum, "t_A", -1))
    nominal = int(getattr(curriculum, "nominal_steps", getattr(trainer.cfg, "curriculum_nominal_timesteps", 1_000_000)))
    return phase, step, t_a, nominal


def v6i1_macro_router_active(trainer: Any, *, phase: str | None = None) -> bool:
    if not is_v6i1_staged_trainer(trainer):
        return False
    if phase is None:
        phase, _, _, _ = v6i1_schedule_context(trainer)
    return phase in ("B", "C")


def resolve_v6i1_rollout_usage_coef(trainer: Any) -> float:
    if not is_v6i1_staged_trainer(trainer):
        return 0.0
    phase, step, t_a, nominal = v6i1_schedule_context(trainer)
    if phase == "A":
        return 0.0
    return float(resolve_v6i1_usage_coef(phase, step, t_a, nominal))


def resolve_v6i1_episode_rehearsal_prob(trainer: Any) -> float:
    if not is_v6i1_staged_trainer(trainer):
        return 0.0
    phase, _, _, _ = v6i1_schedule_context(trainer)
    if phase not in ("B", "C"):
        return 0.0
    return float(getattr(trainer.cfg, "v6i1_router_rehearsal_episode_frac", 0.25) or 0.25)


def resolve_v6i1_episode_forced_frac(trainer: Any) -> float:
    """Router-episode forced-z fraction from the staged schedule (not rehearsal)."""
    if not is_v6i1_staged_trainer(trainer):
        return 0.0
    phase, step, t_a, nominal = v6i1_schedule_context(trainer)
    return float(resolve_v6i1_forced_fraction(phase, step, t_a, nominal))


def resolve_v6i1_exploration_epsilon_current(trainer: Any) -> float:
    if not is_v6i1_staged_trainer(trainer):
        return 0.0
    phase, step, t_a, nominal = v6i1_schedule_context(trainer)
    return float(resolve_v6i1_exploration_epsilon(phase, step, t_a, nominal))


def resolve_v6i1_cf_coef_current(trainer: Any) -> float:
    if not is_v6i1_staged_trainer(trainer):
        return 0.0
    phase, step, t_a, nominal = v6i1_schedule_context(trainer)
    coef_max = float(getattr(trainer.cfg, "latent_cf_coef_max", 0.01) or 0.01)
    return float(resolve_v6i1_cf_coef(phase, step, t_a, nominal, coef_max))


def actor_param_names() -> tuple[str, ...]:
    return ("actor_cnn", "latent_actor")


def critic_param_names() -> tuple[str, ...]:
    return ("critic",)


def router_param_names() -> tuple[str, ...]:
    return (
        "strategy_encoder",
        "selector_gru",
        "episode_strategy_value_head",
        "strategy_aux_return_head",
        "phase_predictor",
    )


def _collect_params(model: torch.nn.Module, name_parts: tuple[str, ...]) -> list[torch.nn.Parameter]:
    params: list[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if any(part in name for part in name_parts):
            if "episode_strategy_value_head" in name and "critic" in name:
                continue
            params.append(param)
    return params


def build_v6i1_optimizers(trainer: Any, *, base_lr: float) -> None:
    """Attach actor / critic / router optimizers for staged V6I1 training."""
    bundle = TrainerOptimizerBundle._build_v6i1(
        model=trainer.model,
        cfg=trainer.cfg,
        hparams=trainer.hparams,
        base_lr=float(base_lr),
    )
    trainer.optimizers = bundle


def _optimizer_bundle(trainer: Any) -> TrainerOptimizerBundle:
    return trainer.optimizers


def apply_v6i1_learning_rates(
    trainer: Any,
    *,
    base_lr: float,
    progress_remaining: float,
) -> dict[str, float]:
    """Set per-phase learning rates on the three V6I1 optimizers."""
    bundle = _optimizer_bundle(trainer)
    if not bundle.v6i1_three_optimizer_mode:
        return {"actor_lr": float(base_lr), "critic_lr": float(base_lr), "router_lr": 0.0}

    phase, _, _, _ = v6i1_schedule_context(trainer)
    lr_floor_frac = max(0.0, min(float(getattr(trainer.cfg, "lr_floor_frac", 0.1) or 0.0), 1.0))
    scaled = float(base_lr) * max(float(progress_remaining), lr_floor_frac)
    actor_lr = scaled
    critic_lr = scaled
    router_lr = float(bundle.router.param_groups[0]["lr"]) if bundle.router is not None else 0.0
    if phase == "C":
        actor_frac = float(getattr(trainer.cfg, "v6i1_phase_c_actor_lr_frac", 0.05) or 0.05)
        actor_lr = scaled * actor_frac
    elif phase == "B":
        actor_lr = 0.0

    for group in bundle.actor.param_groups:
        group["lr"] = actor_lr
    for group in bundle.critic.param_groups:
        group["lr"] = critic_lr
    return {"actor_lr": actor_lr, "critic_lr": critic_lr, "router_lr": router_lr}


def step_v6i1_optimizers(
    trainer: Any,
    *,
    phase: str,
    actor_step: bool,
    critic_step: bool,
    router_step: bool,
    max_grad_norm: float,
) -> dict[str, float]:
    """Clip and step the three optimizers independently."""
    bundle = _optimizer_bundle(trainer)
    grad_norms: dict[str, float] = {"actor_grad_norm": 0.0, "critic_grad_norm": 0.0, "router_grad_norm": 0.0}
    if actor_step and phase != "B":
        grad_norms["actor_grad_norm"] = float(
            torch.nn.utils.clip_grad_norm_(
                [p for p in bundle.actor.param_groups[0]["params"] if p.grad is not None],
                max_grad_norm,
            )
        )
        bundle.actor.step()
    if critic_step:
        grad_norms["critic_grad_norm"] = float(
            torch.nn.utils.clip_grad_norm_(
                [p for p in bundle.critic.param_groups[0]["params"] if p.grad is not None],
                max_grad_norm,
            )
        )
        bundle.critic.step()
    if router_step and phase in ("B", "C") and bundle.router is not None:
        grad_norms["router_grad_norm"] = float(
            torch.nn.utils.clip_grad_norm_(
                [p for p in bundle.router.param_groups[0]["params"] if p.grad is not None],
                max_grad_norm,
            )
        )
        bundle.router.step()
    return grad_norms


def v6i1_curriculum_state_dict(curriculum: Any) -> dict[str, Any]:
    return {
        "phase": str(curriculum.phase),
        "t_A": int(curriculum.t_A),
        "phase_a_end_step": int(curriculum.phase_a_end_step),
        "phase_a_gate_passed": bool(curriculum.phase_a_gate_passed),
        "next_gate_step": int(curriculum.next_gate_step),
        "last_gate_step_run": int(curriculum.last_gate_step_run),
        "gate_check_history": list(curriculum.gate_check_history),
        "protected_candidate_checkpoints": list(curriculum.protected_candidate_checkpoints),
        "best_candidate_report": curriculum.best_candidate_report,
    }


def load_v6i1_curriculum_state(curriculum: Any, payload: dict[str, Any]) -> None:
    curriculum.phase = str(payload.get("phase", curriculum.phase))
    curriculum.t_A = int(payload.get("t_A", curriculum.t_A))
    curriculum.phase_a_end_step = int(payload.get("phase_a_end_step", curriculum.phase_a_end_step))
    curriculum.phase_a_gate_passed = bool(payload.get("phase_a_gate_passed", curriculum.phase_a_gate_passed))
    curriculum.next_gate_step = int(payload.get("next_gate_step", curriculum.next_gate_step))
    curriculum.last_gate_step_run = int(payload.get("last_gate_step_run", curriculum.last_gate_step_run))
    curriculum.gate_check_history = list(payload.get("gate_check_history", []))
    curriculum.protected_candidate_checkpoints = list(
        payload.get("protected_candidate_checkpoints", [])
    )
    curriculum.best_candidate_report = payload.get("best_candidate_report", curriculum.best_candidate_report)


def latent_state_v6i1_checkpoint(state: Any) -> dict[str, Any]:
    return {
        "cf_J": state.cf_J.copy(),
        "cf_episode_counts": state.cf_episode_counts.copy(),
        "cf_has_experience": state.cf_has_experience.copy(),
        "cf_return_mean": float(state.cf_return_mean),
        "cf_return_var": float(state.cf_return_var),
        "pair_jsd_ema": state.pair_jsd_ema.copy(),
        "jsd_gate_consecutive_updates": int(state.jsd_gate_consecutive_updates),
        "router_optimizer_step_count": int(state.router_optimizer_step_count),
        "macro_return_running_mean": float(getattr(state, "macro_return_running_mean", 0.0)),
        "macro_return_running_count": int(getattr(state, "macro_return_running_count", 0)),
        "selector_hidden": getattr(state, "selector_hidden", None),
        "v6i1_episode_rehearsal": getattr(state, "v6i1_episode_rehearsal", None),
    }


def restore_latent_state_v6i1_checkpoint(state: Any, payload: dict[str, Any]) -> None:
    if not payload:
        return
    state.cf_J = payload.get("cf_J", state.cf_J)
    state.cf_episode_counts = payload.get("cf_episode_counts", state.cf_episode_counts)
    state.cf_has_experience = payload.get("cf_has_experience", state.cf_has_experience)
    state.cf_return_mean = float(payload.get("cf_return_mean", state.cf_return_mean))
    state.cf_return_var = float(payload.get("cf_return_var", state.cf_return_var))
    state.pair_jsd_ema = payload.get("pair_jsd_ema", state.pair_jsd_ema)
    state.jsd_gate_consecutive_updates = int(
        payload.get("jsd_gate_consecutive_updates", state.jsd_gate_consecutive_updates)
    )
    state.router_optimizer_step_count = int(
        payload.get("router_optimizer_step_count", state.router_optimizer_step_count)
    )
    if hasattr(state, "macro_return_running_mean"):
        state.macro_return_running_mean = float(payload.get("macro_return_running_mean", 0.0))
    if hasattr(state, "macro_return_running_count"):
        state.macro_return_running_count = int(payload.get("macro_return_running_count", 0))
    hidden = payload.get("selector_hidden")
    if hidden is not None and hasattr(state, "selector_hidden"):
        state.selector_hidden = hidden.to(device=state.selector_hidden.device, dtype=state.selector_hidden.dtype)
    rehearsal = payload.get("v6i1_episode_rehearsal")
    if rehearsal is not None and hasattr(state, "v6i1_episode_rehearsal"):
        state.v6i1_episode_rehearsal = rehearsal.to(
            device=state.v6i1_episode_rehearsal.device,
            dtype=state.v6i1_episode_rehearsal.dtype,
        )


def v6i1_intervention_csv_stats(
    latent_state: Any,
    *,
    profile_stats: dict[str, float] | None = None,
    cfg: Any | None = None,
) -> dict[str, float]:
    """V6I1 intervention / competence / gate telemetry for metrics CSV rows."""
    profile_stats = profile_stats or {}
    cfg = cfg or getattr(getattr(latent_state, "trainer", None), "cfg", None)
    margin = float(getattr(cfg, "latent_cf_jsd_margin", 0.01) or 0.01) if cfg is not None else 0.01
    required_consecutive = (
        int(getattr(cfg, "latent_cf_gate_consecutive_updates", 5) or 5) if cfg is not None else 5
    )

    pair_ema = getattr(latent_state, "pair_jsd_ema", None)
    ema_list = (
        [float(pair_ema[i]) for i in range(min(6, int(pair_ema.size)))]
        if pair_ema is not None
        else [0.0] * 6
    )
    while len(ema_list) < 6:
        ema_list.append(0.0)

    num_above = int(sum(1 for v in ema_list if float(v) >= margin))
    min_pair = float(min(ema_list)) if ema_list else 0.0
    update_ok = num_above >= 5 and min_pair >= 0.5 * margin
    streak = int(getattr(latent_state, "jsd_gate_consecutive_updates", 0) or 0)

    comp_scores, competence_ready = latent_state.compute_competence_scores()
    out: dict[str, float] = {
        "jsd_gate_consecutive_updates": float(streak),
        "jsd_pairs_above_margin": float(num_above),
        "jsd_min_pair": min_pair,
        "jsd_gate_update_pass": 1.0 if update_ok else 0.0,
        "jsd_gate_consecutive_required": float(required_consecutive),
        "cf_competence_ready": 1.0 if competence_ready else 0.0,
    }
    latent_k = int(getattr(latent_state.trainer, "latent_k", 4) or 4)
    for z in range(latent_k):
        out[f"cf_competence_z{z}"] = float(comp_scores[z]) if z < len(comp_scores) else 0.0

    for idx in range(6):
        suffix = v6i1_pair_suffix(idx)
        raw = float(profile_stats.get(f"forced_z_pair_jsd_{idx}", 0.0) or 0.0)
        ema = float(ema_list[idx])
        out[f"forced_z_pair_jsd_{idx}"] = raw
        out[f"forced_z_pair_jsd_{suffix}"] = raw
        out[f"pair_jsd_ema_{idx}"] = ema
        out[f"pair_jsd_ema_{suffix}"] = ema
    return out


def format_v6i1_rollout_stdout_line(
    row: dict[str, Any],
    *,
    phase: str,
    required_consecutive: int,
) -> str:
    """Per-update intervention telemetry (distinct from gate-attempt block)."""
    pair_bits = ",".join(
        f"{float(row.get(f'pair_jsd_ema_{idx}', 0.0) or 0.0):.4f}" for idx in range(6)
    )
    return (
        f"      [V6I1] phase={phase} "
        f"cf_coef={float(row.get('v6i1_cf_coef_current', 0.0) or 0.0):.4f} "
        f"sep_train={int(float(row.get('latent_actor_z_separation_train_active', 0.0) or 0.0))} "
        f"jsd_consec={int(float(row.get('jsd_gate_consecutive_updates', 0.0) or 0.0))}"
        f"/{int(required_consecutive)} "
        f"comp_ready={int(float(row.get('cf_competence_ready', 0.0) or 0.0))} "
        f"pair_ema=[{pair_bits}]"
    )


__all__ = [
    "apply_v6i1_learning_rates",
    "build_v6i1_optimizers",
    "is_v6i1_staged_trainer",
    "latent_state_v6i1_checkpoint",
    "load_v6i1_curriculum_state",
    "restore_latent_state_v6i1_checkpoint",
    "resolve_v6i1_cf_coef_current",
    "resolve_v6i1_episode_forced_frac",
    "resolve_v6i1_episode_rehearsal_prob",
    "resolve_v6i1_exploration_epsilon_current",
    "resolve_v6i1_rollout_usage_coef",
    "step_v6i1_optimizers",
    "v6i1_curriculum_state_dict",
    "format_v6i1_rollout_stdout_line",
    "v6i1_intervention_csv_stats",
    "v6i1_macro_router_active",
    "v6i1_schedule_context",
]
