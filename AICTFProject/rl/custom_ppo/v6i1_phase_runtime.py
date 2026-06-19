"""V6I1 staged curriculum Phase B/C runtime helpers."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch

from rl.custom_ppo.curriculum_gates import is_staged_v6i1_curriculum
from rl.custom_ppo.gate_protocol import (
    is_staged_v6_team_intent_curriculum,
    is_v6i2_gate_protocol,
    is_v6i3_gate_protocol,
)
from rl.custom_ppo.v6i1_cf_loss import v6i1_pair_suffix
from rl.custom_ppo.trainer_optimizers import TrainerOptimizerBundle
from rl.custom_ppo.schedules import (
    resolve_v6i1_cf_coef,
    resolve_v6i1_exploration_epsilon,
    resolve_v6i1_forced_fraction,
    resolve_v6i1_usage_coef,
)
from rl.custom_ppo.curriculum.schedule import resolve_schedule


def is_v6i1_staged_trainer(trainer: Any) -> bool:
    cfg = getattr(trainer, "cfg", None)
    if cfg is None:
        return False
    return (
        is_staged_v6_team_intent_curriculum(cfg)
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


def resolve_v6i1_entropy_schedule_total_timesteps(trainer: Any) -> int:
    """Entropy anneal clock: fixed nominal curriculum budget for staged rows."""
    if not is_v6i1_staged_trainer(trainer):
        return int(getattr(trainer.cfg, "total_timesteps", 1) or 1)
    _, _, _, nominal = v6i1_schedule_context(trainer)
    return int(nominal)


def resolve_v6i1_lr_progress_remaining(
    trainer: Any,
    *,
    training_terminal: int | None = None,
) -> float:
    """LR progress clock with phase-local denominators (immune to terminal extension).

    Staged curriculum rows use separate clocks:

    * Phase A actor/critic: fixed nominal ``curriculum_nominal_timesteps``
    * Phase B critic/router: local progress over ``phase_b_budget_steps`` from ``t_A``
    * Phase C actor/critic/router: local progress over ``phase_c_budget_steps`` from
      the end of Phase B

    The dynamic ``training_terminal`` is used only for non-staged trainers.
    """
    if not is_v6i1_staged_trainer(trainer):
        cfg = getattr(trainer, "cfg", None)
        terminal = int(
            training_terminal
            if training_terminal is not None
            else getattr(cfg, "total_timesteps", 1) or 1
        )
        step = int(getattr(trainer, "global_step", 0) or 0)
        return max(0.0, 1.0 - float(step) / max(1.0, float(terminal)))

    phase, step, t_a, nominal = v6i1_schedule_context(trainer)
    schedule = resolve_schedule(trainer.cfg)
    if phase == "A":
        return max(0.0, 1.0 - float(step) / max(1.0, float(nominal)))
    if phase == "B":
        if t_a < 0:
            return max(0.0, 1.0 - float(step) / max(1.0, float(nominal)))
        local_step = max(0, int(step) - int(t_a))
        budget = max(1, int(schedule.phase_b_budget_steps))
        return max(0.0, 1.0 - float(local_step) / float(budget))
    if phase == "C":
        if t_a < 0:
            return max(0.0, 1.0 - float(step) / max(1.0, float(nominal)))
        phase_b_end = int(t_a) + int(schedule.phase_b_budget_steps)
        local_step = max(0, int(step) - phase_b_end)
        budget = max(1, int(schedule.phase_c_budget_steps))
        return max(0.0, 1.0 - float(local_step) / float(budget))
    return max(0.0, 1.0 - float(step) / max(1.0, float(nominal)))


def _resolve_v6i1_router_base_lr(trainer: Any) -> float:
    return float(
        getattr(trainer.cfg, "v6i1_router_lr", None)
        or getattr(trainer.hparams, "latent_episode_strategy_lr", None)
        or 5e-3
    )


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
    training_terminal: int | None = None,
    progress_remaining: float | None = None,
) -> dict[str, float]:
    """Set per-phase learning rates on the three V6I1 optimizers."""
    bundle = _optimizer_bundle(trainer)
    if not bundle.v6i1_three_optimizer_mode:
        return {"actor_lr": float(base_lr), "critic_lr": float(base_lr), "router_lr": 0.0}

    if progress_remaining is None:
        progress_remaining = resolve_v6i1_lr_progress_remaining(
            trainer, training_terminal=training_terminal
        )
    phase, _, _, _ = v6i1_schedule_context(trainer)
    lr_floor_frac = max(0.0, min(float(getattr(trainer.cfg, "lr_floor_frac", 0.1) or 0.0), 1.0))
    scaled = float(base_lr) * max(float(progress_remaining), lr_floor_frac)
    actor_lr = scaled
    critic_lr = scaled
    router_base = _resolve_v6i1_router_base_lr(trainer)
    router_lr = 0.0
    if phase in ("B", "C") and bundle.router is not None:
        router_lr = router_base * max(float(progress_remaining), lr_floor_frac)
    actor_frac = float(getattr(trainer.cfg, "v6i1_phase_c_actor_lr_frac", 0.05) or 0.05)
    if phase == "C":
        actor_lr = scaled * actor_frac
    elif phase == "B":
        actor_lr = 0.0

    for group in bundle.actor.param_groups:
        group["lr"] = actor_lr
    for group in bundle.critic.param_groups:
        group["lr"] = critic_lr
    if bundle.router is not None and phase in ("B", "C"):
        for group in bundle.router.param_groups:
            group["lr"] = router_lr
    return {
        "actor_lr": actor_lr,
        "critic_lr": critic_lr,
        "router_lr": router_lr,
        "v6i1_lr_progress_remaining": float(progress_remaining),
    }


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
    if hasattr(curriculum, "state_dict"):
        return curriculum.state_dict()
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
    if hasattr(curriculum, "load_state_dict"):
        curriculum.load_state_dict(payload)
        return
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
    from rl.custom_ppo.latent.checkpoint import latent_checkpoint_payload

    payload = latent_checkpoint_payload(state)
    flat = {k: v for k, v in payload.items() if k not in ("intervention", "router_runtime")}
    flat.update(payload.get("intervention", {}))
    flat.update(payload.get("router_runtime", {}))
    return flat


def restore_latent_state_v6i1_checkpoint(state: Any, payload: dict[str, Any]) -> None:
    if not payload:
        return
    trainer = getattr(state, "trainer", None)
    cfg = getattr(trainer, "cfg", None) if trainer is not None else None
    from rl.custom_ppo.latent.checkpoint import restore_latent_checkpoint_payload

    restore_latent_checkpoint_payload(state, payload)
    if cfg is not None:
        from rl.custom_ppo.gate_protocol import (
            gate_config_fingerprint,
            is_v6i2_gate_protocol,
            resolve_gate_protocol_version,
        )

        active_protocol = resolve_gate_protocol_version(cfg)
        ckpt_protocol = str(payload.get("gate_protocol_version", active_protocol))
        if ckpt_protocol != active_protocol:
            raise ValueError(
                f"gate_protocol_version mismatch on resume: checkpoint={ckpt_protocol!r} "
                f"active={active_protocol!r}"
            )
        ckpt_fp = str(payload.get("gate_config_fingerprint", "") or "")
        active_fp = gate_config_fingerprint(cfg)
        if ckpt_fp and ckpt_fp != active_fp:
            if not bool(getattr(cfg, "allow_gate_config_mismatch_on_resume", False)):
                raise ValueError(
                    f"gate_config_fingerprint mismatch on resume: checkpoint={ckpt_fp!r} "
                    f"active={active_fp!r}"
                )
            from rl.custom_ppo.gate_protocol import (
                apply_gate_config_mismatch_override,
                format_gate_mismatch_override_warning,
            )

            apply_gate_config_mismatch_override(
                cfg,
                checkpoint_fingerprint=ckpt_fp,
                active_fingerprint=active_fp,
            )
            for line in format_gate_mismatch_override_warning(cfg):
                print(line, flush=True)
        enforce = str(getattr(cfg, "phase_boundary_gate_mode", "enforce")).lower() == "enforce"
        if enforce and (is_v6i2_gate_protocol(cfg) or is_v6i3_gate_protocol(cfg)):
            required = (
                "gate_protocol_version",
                "cf_pair_jsd_ema",
                "macro_pair_jsd_ema",
                "cf_pair_jsd_valid_updates",
                "macro_pair_jsd_valid_updates",
            )
            missing = [key for key in required if key not in payload]
            if missing:
                raise ValueError(
                    f"v6i2/v6i3 enforce resume missing checkpoint gate state: {missing}"
                )


def v6i1_intervention_csv_stats(
    latent_state: Any,
    *,
    profile_stats: dict[str, float] | None = None,
    cfg: Any | None = None,
) -> dict[str, float]:
    """V6I1 intervention / competence / gate telemetry for metrics CSV rows."""
    from rl.custom_ppo.v6i1_cf_loss import forced_z_pairwise_profile_available

    profile_stats = profile_stats or {}
    cfg = cfg or getattr(getattr(latent_state, "trainer", None), "cfg", None)
    from rl.custom_ppo.gate_protocol import (
        is_staged_v6_team_intent_curriculum,
        is_v6i2_dual_evidence_protocol,
        is_v6i3_gate_protocol,
    )

    if is_v6i2_dual_evidence_protocol(cfg) if cfg is not None else False:
        margin = float(cfg.actor_jsd_margin)
        floor_frac = float(cfg.actor_jsd_floor_fraction)
        min_pairs = int(cfg.actor_jsd_min_passing_pairs)
        required_consecutive = int(cfg.actor_jsd_consecutive_updates)
        cf_valid = int(getattr(latent_state, "cf_pair_jsd_valid_updates", 0) or 0)
        macro_valid = int(getattr(latent_state, "macro_pair_jsd_valid_updates", 0) or 0)
        cf_ema = getattr(latent_state, "cf_pair_jsd_ema", None)
        macro_ema = getattr(latent_state, "macro_pair_jsd_ema", None)
        cf_list = (
            [float(cf_ema[i]) for i in range(min(6, int(cf_ema.size)))]
            if cf_ema is not None
            else [0.0] * 6
        )
        macro_list = (
            [float(macro_ema[i]) for i in range(min(6, int(macro_ema.size)))]
            if macro_ema is not None
            else [0.0] * 6
        )
        while len(cf_list) < 6:
            cf_list.append(0.0)
        while len(macro_list) < 6:
            macro_list.append(0.0)
        num_above = int(sum(1 for v in cf_list if float(v) >= margin))
        min_cf = float(min(cf_list)) if cf_list else 0.0
        floor = floor_frac * margin
        update_ok = num_above >= min_pairs and min_cf >= floor
        actor_streak = int(getattr(latent_state, "actor_intervention_consecutive_updates", 0) or 0)
        comp_scores, competence_ready = latent_state.compute_competence_scores()
        trainer = getattr(latent_state, "trainer", None)
        last_stats = dict(getattr(trainer, "last_stats", {}) or {})
        out: dict[str, float] = {
            "pairwise_profile_available": 1.0 if profile_stats else 0.0,
            "cf_pair_jsd_valid_updates": float(cf_valid),
            "cf_pair_jsd_last_update_step": float(
                getattr(latent_state, "cf_pair_jsd_last_update_step", -1) or -1
            ),
            "macro_pair_jsd_valid_updates": float(macro_valid),
            "macro_pair_jsd_last_update_step": float(
                getattr(latent_state, "macro_pair_jsd_last_update_step", -1) or -1
            ),
            "actor_intervention_consecutive_updates": float(actor_streak),
            "actor_intervention_gate_update_pass": 1.0 if update_ok else 0.0,
            "actor_intervention_consecutive_required": float(required_consecutive),
            "cf_pairs_above_actor_margin": float(num_above),
            "cf_min_pair_ema": min_cf,
            "cf_competence_ready": 1.0 if competence_ready else 0.0,
        }
        latent_k = int(getattr(latent_state.trainer, "latent_k", 4) or 4)
        for z in range(latent_k):
            out[f"cf_competence_z{z}"] = float(comp_scores[z]) if z < len(comp_scores) else 0.0
        batch_evidence_valid = float(last_stats.get("cf_batch_evidence_valid", 0.0) or 0.0) > 0.0
        for idx in range(6):
            suffix = v6i1_pair_suffix(idx)
            batch_key = f"cf_batch_pair_jsd_{idx}"
            if batch_evidence_valid and batch_key in last_stats:
                cf_raw = float(last_stats[batch_key])
            else:
                cf_raw = float("nan")
            macro_raw = (
                float(profile_stats.get(f"forced_z_pair_jsd_{idx}", 0.0) or 0.0)
                if profile_stats
                else 0.0
            )
            out[f"cf_pair_jsd_{idx}"] = cf_raw
            out[f"cf_pair_jsd_{suffix}"] = cf_raw
            out[f"cf_pair_jsd_ema_{idx}"] = float(cf_list[idx])
            out[f"cf_pair_jsd_ema_{suffix}"] = float(cf_list[idx])
            out[f"macro_pair_jsd_{idx}"] = macro_raw
            out[f"macro_pair_jsd_{suffix}"] = macro_raw
            out[f"macro_pair_jsd_ema_{idx}"] = float(macro_list[idx])
            out[f"macro_pair_jsd_ema_{suffix}"] = float(macro_list[idx])
        return out

    margin = float(getattr(cfg, "latent_cf_jsd_margin", 0.01) or 0.01) if cfg is not None else 0.01
    required_consecutive = (
        int(getattr(cfg, "latent_cf_gate_consecutive_updates", 5) or 5) if cfg is not None else 5
    )
    profile_available = forced_z_pairwise_profile_available(profile_stats)
    valid_updates = int(getattr(latent_state, "pairwise_ema_valid_updates", 0) or 0)

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
        "pairwise_profile_available": 1.0 if profile_available else 0.0,
        "pairwise_ema_valid_updates": float(valid_updates),
        "pairwise_ema_last_update_step": float(
            getattr(latent_state, "pairwise_ema_last_update_step", -1) or -1
        ),
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
        raw = float(profile_stats.get(f"forced_z_pair_jsd_{idx}", 0.0) or 0.0) if profile_available else 0.0
        ema = float(ema_list[idx])
        out[f"forced_z_pair_jsd_{idx}"] = raw
        out[f"forced_z_pair_jsd_{suffix}"] = raw
        out[f"pair_jsd_ema_{idx}"] = ema
        out[f"pair_jsd_ema_{suffix}"] = ema
    return out


def _format_pair_jsd_values(values: list[float]) -> str:
    """Format pair JSD/EMA values; use extra precision below the 0.01 gate margin."""
    margin = 0.01
    parts: list[str] = []
    for value in values:
        v = float(value)
        if v <= 0.0:
            parts.append("0")
        elif v < margin:
            parts.append(f"{v:.6f}")
        else:
            parts.append(f"{v:.4f}")
    return ",".join(parts)


def format_v6i1_rollout_stdout_line(
    row: dict[str, Any],
    *,
    phase: str,
    required_consecutive: int,
    gate_protocol: str | None = None,
) -> str:
    """Per-update intervention telemetry (distinct from gate-attempt block)."""
    from rl.custom_ppo.gate_protocol import staged_latent_stdout_tag

    tag = staged_latent_stdout_tag(gate_protocol)
    from rl.custom_ppo.gate_protocol import V6I2_GATE_PROTOCOL, V6I3_GATE_PROTOCOL

    if gate_protocol in (V6I2_GATE_PROTOCOL, V6I3_GATE_PROTOCOL):
        ema_vals = [float(row.get(f"cf_pair_jsd_ema_{idx}", 0.0) or 0.0) for idx in range(6)]
        ema_label = "cf_pair_ema"
    else:
        ema_vals = [float(row.get(f"pair_jsd_ema_{idx}", 0.0) or 0.0) for idx in range(6)]
        ema_label = "pair_ema"
    raw_vals = [float(row.get(f"forced_z_pair_jsd_{idx}", 0.0) or 0.0) for idx in range(6)]
    macro_mean = float(row.get("forced_z_macro_jsd_mean", 0.0) or 0.0)
    actor_jsd_mean = float(row.get("actor_z_jsd_mean", 0.0) or 0.0)
    actor_jsd_max = float(row.get("actor_z_jsd_max", 0.0) or 0.0)
    cf_grad = float(row.get("cf_actor_grad_norm", 0.0) or 0.0)
    cf_ratio = float(row.get("cf_to_ppo_grad_ratio", 0.0) or 0.0)
    sep_jsd = float(row.get("latent_actor_z_separation_jsd", 0.0) or 0.0)
    sep_min_batch = float(row.get("latent_actor_z_separation_jsd_min", row.get("jsd_min_pair", 0.0)) or 0.0)
    pairs_below = int(float(row.get("cf_batch_pairs_below_margin", 0.0) or 0.0))
    hinge_active = int(float(row.get("cf_hinge_active", 0.0) or 0.0))
    hinge_effective = int(float(row.get("cf_hinge_effective", 0.0) or 0.0))
    profile_ok = int(float(row.get("pairwise_profile_available", 0.0) or 0.0))
    cf_requires_grad = int(float(row.get("cf_loss_requires_grad", 0.0) or 0.0))
    cf_weight_sum = float(row.get("cf_weight_sum", 0.0) or 0.0)
    cf_effective_pairs = int(float(row.get("cf_effective_pairs", 0.0) or 0.0))
    cf_valid_groups = int(float(row.get("cf_valid_team_groups", 0.0) or 0.0))
    sep_jsd_max = float(row.get("latent_actor_z_separation_jsd_max", 0.0) or 0.0)
    cf_batch_pairs = [
        float(row.get(f"cf_batch_pair_jsd_{idx}", 0.0) or 0.0) for idx in range(6)
    ]
    return (
        f"      [{tag}] phase={phase} "
        f"cf_coef={float(row.get('v6i1_cf_coef_current', 0.0) or 0.0):.4f} "
        f"sep_train={int(float(row.get('latent_actor_z_separation_train_active', 0.0) or 0.0))} "
        f"pairwise_ok={profile_ok} "
        f"jsd_consec={int(float(row.get('jsd_gate_consecutive_updates', 0.0) or 0.0))}"
        f"/{int(required_consecutive)} "
        f"comp_ready={int(float(row.get('cf_competence_ready', 0.0) or 0.0))} "
        f"macro_jsd={macro_mean:.6f} pair_raw=[{_format_pair_jsd_values(raw_vals)}] "
        f"{ema_label}=[{_format_pair_jsd_values(ema_vals)}] "
        f"actor_jsd={actor_jsd_mean:.6f}/{actor_jsd_max:.6f} "
        f"cf_batch=[{_format_pair_jsd_values(cf_batch_pairs)}] "
        f"hinge={hinge_active} eff={hinge_effective} pairs_below={pairs_below}/6 "
        f"cf_wsum={cf_weight_sum:.4f} eff_pairs={cf_effective_pairs} groups={cf_valid_groups} "
        f"sep_jsd={sep_jsd:.6f}/{sep_jsd_max:.6f} min={sep_min_batch:.6f} "
        f"cf_req_grad={cf_requires_grad} cf_grad={cf_grad:.2e} cf/ppo={cf_ratio:.3f}"
    )


__all__ = [
    "apply_v6i1_learning_rates",
    "build_v6i1_optimizers",
    "is_v6i1_staged_trainer",
    "latent_state_v6i1_checkpoint",
    "load_v6i1_curriculum_state",
    "restore_latent_state_v6i1_checkpoint",
    "resolve_v6i1_entropy_schedule_total_timesteps",
    "resolve_v6i1_lr_progress_remaining",
    "resolve_v6i1_episode_forced_frac",
    "resolve_v6i1_episode_rehearsal_prob",
    "resolve_v6i1_exploration_epsilon_current",
    "resolve_v6i1_rollout_usage_coef",
    "step_v6i1_optimizers",
    "v6i1_curriculum_state_dict",
    "format_v6i1_rollout_stdout_line",
    "_format_pair_jsd_values",
    "v6i1_intervention_csv_stats",
    "v6i1_macro_router_active",
    "v6i1_schedule_context",
]
