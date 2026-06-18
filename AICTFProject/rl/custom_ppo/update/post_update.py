"""Post-update deferred latent passes, diagnostics, and final stats."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from rl.custom_ppo.latent_diagnostics import (
    _behavior_diversity_stats,
    _forced_z_behavior_profile,
    _latent_opponent_rollout_diag,
    _latent_option_advantage_stats,
    _latent_rollout_stats,
    _rollout_advantage_diagnostics,
    _strategy_resample_advantage_stats,
    _write_refresh_log_table,
    _write_strategy_experience_table,
    _policy_z_sensitivity_kl,
)
from rl.custom_ppo.schedules import resolve_latent_forced_z_frac
from rl.custom_ppo.update.actor_intervention import ActorInterventionEvidenceUpdater
from rl.custom_ppo.update.entropy_objectives import RolloutMarginalPrep
from rl.custom_ppo.update.helpers import populate_main_loop_qphi_telemetry, warmup_ramp_value
from rl.custom_ppo.update.loss_result import measurement_from_pair_tensor
from rl.custom_ppo.update.minibatch_updater import ACTOR_INTERVENTION_REASON_CODES
from rl.custom_ppo.update.telemetry import UpdateStatsAccumulator
from rl.custom_ppo.update.update_context import PPOUpdateContext
from rl.ppo_core import TensorDictRolloutBuffer


@dataclass(frozen=True)
class PostUpdateResult:
    stats: dict[str, float]


class PostUpdatePipeline:
    def __init__(self, *, intervention_evidence: ActorInterventionEvidenceUpdater) -> None:
        self.intervention_evidence = intervention_evidence

    def run(
        self,
        *,
        updater: Any,
        buffer: TensorDictRolloutBuffer,
        context: PPOUpdateContext,
        accumulator: UpdateStatsAccumulator,
        prep: RolloutMarginalPrep,
        latent_lam_h: float,
        curr_sep_coef: float,
        curr_adapter_scale: float,
        lr: float,
        v6i1_lr_stats: dict[str, float],
        v6i1_usage_coef: float,
        pair_count: int,
        valid_cf_pair_measurements: list[torch.Tensor],
        actor_intervention_valid_minibatches: int,
        last_invalid_reason_code: float,
    ) -> PostUpdateResult:
        from rl.custom_ppo.v6i1_phase_runtime import (
            is_v6i1_staged_trainer,
            resolve_v6i1_cf_coef_current,
            resolve_v6i1_episode_forced_frac,
            resolve_v6i1_rollout_usage_coef,
            v6i1_intervention_csv_stats,
            v6i1_macro_router_active,
        )

        runtime = updater.runtime
        hparams = updater.hparams
        cfg = updater.cfg
        latent_state = updater.latent_state

        episode_strategy_stats = latent_state.apply_episode_strategy_ppo(latent_lam_h=latent_lam_h)
        macro_strategy_stats: dict[str, float] = {}
        if is_v6i1_staged_trainer(runtime) and v6i1_macro_router_active(runtime):
            macro_strategy_stats = latent_state.apply_macro_strategy_ppo()
        arc_strategy_stats = latent_state.apply_arc_strategy_ppo()
        latent_state.reset_arc_credit_rollout_state()
        latent_state.reset_macro_rollout_state()
        rollout_specialist_stats = latent_state.apply_rollout_specialist_router(buffer)
        strategy_experience_stats = _write_strategy_experience_table(runtime)
        refresh_log_stats = _write_refresh_log_table(runtime)
        latent_state.clear_rollout_refresh_records()

        stats = accumulator.finalize()
        value_losses = np.asarray(accumulator.raw_rows.get("value_loss", []), dtype=np.float32)
        if value_losses.size > 0:
            stats.update(
                {
                    "value_loss_min": float(np.min(value_losses)),
                    "value_loss_std": float(np.std(value_losses)),
                    "value_loss_p10": float(np.percentile(value_losses, 10)),
                    "value_loss_p50": float(np.percentile(value_losses, 50)),
                    "value_loss_p90": float(np.percentile(value_losses, 90)),
                    "value_loss_max": float(np.max(value_losses)),
                }
            )
        else:
            stats.update(
                {
                    "value_loss_min": 0.0,
                    "value_loss_std": 0.0,
                    "value_loss_p10": 0.0,
                    "value_loss_p50": 0.0,
                    "value_loss_p90": 0.0,
                    "value_loss_max": 0.0,
                }
            )

        stats["learning_rate"] = float(lr)
        stats["latent_lam_h"] = float(latent_lam_h)
        stats["latent_actor_z_adapter_scale"] = float(curr_adapter_scale)
        stats["latent_actor_z_separation_coef"] = float(curr_sep_coef)

        if is_v6i1_staged_trainer(runtime):
            phase = context.phase
            stats["v6i1_phase"] = float({"A": 0.0, "B": 1.0, "C": 2.0}.get(phase, -1.0))
            stats["v6i1_cf_coef_current"] = float(resolve_v6i1_cf_coef_current(runtime))
            stats["v6i1_usage_coef_current"] = float(resolve_v6i1_rollout_usage_coef(runtime))
            stats["latent_forced_z_episode_frac_current"] = float(
                resolve_v6i1_episode_forced_frac(runtime)
            )
            if v6i1_lr_stats:
                stats.update(v6i1_lr_stats)
        else:
            stats["latent_forced_z_episode_frac_current"] = float(
                resolve_latent_forced_z_frac(cfg, global_step=int(runtime.global_step))
            )

        if hparams.normalize_returns:
            rn = runtime.return_norm
            stats["return_norm_mean"] = float(rn.mean)
            stats["return_norm_std"] = float(rn.std)
            stats["return_norm_count"] = float(rn.count)
        else:
            stats["return_norm_mean"] = 0.0
            stats["return_norm_std"] = 0.0
            stats["return_norm_count"] = 0.0

        stats.update(_strategy_resample_advantage_stats(runtime, buffer))
        stats.update(_latent_option_advantage_stats(runtime, buffer))
        stats.update(_rollout_advantage_diagnostics(runtime, buffer))
        stats.update(_latent_rollout_stats(runtime, buffer))
        stats.update(_latent_opponent_rollout_diag(runtime, buffer))
        stats.update(_behavior_diversity_stats(runtime, buffer))
        forced_z_profile = _forced_z_behavior_profile(runtime, buffer)
        stats.update(forced_z_profile)
        if forced_z_profile:
            ema_updated = latent_state.update_intervention_gate_from_profile(forced_z_profile)
            stats["pairwise_profile_available"] = 1.0 if ema_updated else 0.0
        else:
            stats["pairwise_profile_available"] = 0.0

        if valid_cf_pair_measurements:
            mean_pairs = torch.stack(valid_cf_pair_measurements).mean(dim=0)
            roll_measurement = measurement_from_pair_tensor(
                mean_pairs,
                active_fraction=1.0,
                valid_groups=len(valid_cf_pair_measurements),
            )
        else:
            roll_measurement = measurement_from_pair_tensor(
                None,
                active_fraction=0.0,
                valid_groups=0,
                reason="no_valid_minibatch",
            )
        evidence = self.intervention_evidence.update(
            latent_state,
            roll_measurement,
            cfg=cfg,
            global_step=int(runtime.global_step),
        )
        stats["actor_intervention_gate_updated"] = 1.0 if evidence.gate_updated else 0.0
        stats["actor_intervention_valid_minibatches"] = float(actor_intervention_valid_minibatches)
        stats["actor_intervention_invalid_reason_code"] = float(last_invalid_reason_code)
        if evidence.reason and not evidence.measurement_valid:
            stats["actor_intervention_invalid_reason_code"] = ACTOR_INTERVENTION_REASON_CODES.get(
                evidence.reason, 99.0
            )

        if "actor_intervention_measurement_valid" in accumulator.raw_rows:
            rows = accumulator.raw_rows["actor_intervention_measurement_valid"]
            stats["actor_intervention_measurement_valid"] = float(np.mean(rows)) if rows else 0.0

        stats.update(
            v6i1_intervention_csv_stats(
                latent_state,
                profile_stats=forced_z_profile,
                cfg=cfg,
            )
        )
        if is_v6i1_staged_trainer(runtime):
            from rl.custom_ppo.v6i1_cf_loss import v6i1_pair_suffix

            for idx in range(pair_count):
                suffix = v6i1_pair_suffix(idx)
                batch_key = f"cf_batch_pair_jsd_{idx}"
                if batch_key in stats:
                    stats[f"cf_batch_pair_jsd_{suffix}"] = stats[batch_key]

        stats.update(_policy_z_sensitivity_kl(runtime, buffer))
        stats.update(episode_strategy_stats)
        stats.update(arc_strategy_stats)
        stats.update(macro_strategy_stats)
        if v6i1_lr_stats:
            stats.update(v6i1_lr_stats)
        if v6i1_usage_coef > 0.0:
            stats["v6i1_rollout_usage_coef"] = float(v6i1_usage_coef)
        stats["rollout_marginal_active"] = (
            1.0 if prep.apply_rollout_marginal and prep.resample_states is not None else 0.0
        )
        if prep.skip_reason is not None:
            stats["rollout_marginal_skip_reason"] = float(
                {"empty_rollout": 1.0, "no_resample_rows": 2.0}.get(prep.skip_reason, 0.0)
            )
        populate_main_loop_qphi_telemetry(
            stats,
            cfg=cfg,
            hparams=hparams,
            runtime=runtime,
            latent_lam_h=float(latent_lam_h),
        )
        stats.update(rollout_specialist_stats)
        stats.update(strategy_experience_stats)
        stats.update(refresh_log_stats)
        stats.update(latent_state.behavior_contrast_rollout_stats())
        stats.update(latent_state.event_refresh_rollout_stats())
        stats.update(latent_state.sparse_tactical_refresh_rollout_stats())
        if hparams.use_latent_strategy and "z_forced" in buffer.fields:
            forced_steps = buffer.fields["z_forced"][: int(buffer.pos)].detach().float()
            stats["latent_forced_z_step_fraction"] = (
                float(forced_steps.mean().cpu().item()) if forced_steps.numel() > 0 else 0.0
            )
        else:
            stats["latent_forced_z_step_fraction"] = 0.0

        runtime.last_stats = stats
        return PostUpdateResult(stats=stats)


def resolve_adapter_scale(updater: Any, *, step: int) -> float:
    hparams = updater.hparams
    if not (
        getattr(hparams, "use_latent_strategy", False)
        and getattr(hparams, "latent_actor_z_adapter_enabled", False)
    ):
        return 0.0
    scale = warmup_ramp_value(
        global_step=step,
        warmup_steps=int(getattr(hparams, "latent_actor_z_adapter_warmup_steps", 0) or 0),
        ramp_steps=int(getattr(hparams, "latent_actor_z_adapter_ramp_steps", 0) or 0),
        start_value=0.0,
        target_value=float(getattr(hparams, "latent_actor_z_adapter_scale", 0.0) or 0.0),
    )
    model = updater.model
    if hasattr(model, "latent_actor") and model.latent_actor is not None:
        model.latent_actor.z_adapter_scale = scale
    return scale


def resolve_separation_coef(updater: Any, *, step: int) -> float:
    from rl.custom_ppo.v6i1_phase_runtime import is_v6i1_staged_trainer, resolve_v6i1_cf_coef_current

    hparams = updater.hparams
    runtime = updater.runtime
    if is_v6i1_staged_trainer(runtime):
        return float(resolve_v6i1_cf_coef_current(runtime))
    return warmup_ramp_value(
        global_step=step,
        warmup_steps=int(getattr(hparams, "latent_actor_z_separation_warmup_steps", 0) or 0),
        ramp_steps=int(getattr(hparams, "latent_actor_z_separation_ramp_steps", 0) or 0),
        start_value=float(getattr(hparams, "latent_actor_z_separation_start_coef", 0.0) or 0.0),
        target_value=float(getattr(hparams, "latent_actor_z_separation_coef", 0.0) or 0.0),
    )
