"""One minibatch forward/backward/step producing a typed result."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from rl.custom_ppo.return_normalization import _normalize_value_targets
from rl.custom_ppo.trainer_optimizers import (
    collect_actor_optimizer_parameters,
    collect_actor_parameters,
)
from rl.custom_ppo.update.entropy_objectives import EntropyObjective, RolloutEntropyState, RolloutMarginalPrep
from rl.custom_ppo.update.helpers import tensor_stat
from rl.custom_ppo.update.loss_result import LossComponent, MinibatchUpdateResult
from rl.custom_ppo.update.optimizer_stepper import OptimizerStepper
from rl.custom_ppo.update.phase_policy import PhaseTrainingPolicy
from rl.custom_ppo.update.separation_objectives import SeparationObjective
from rl.custom_ppo.update.strategy_objectives import StrategyObjective
from rl.custom_ppo.update.update_context import PPOUpdateContext
from rl.custom_ppo.v6i1_cf_loss import actor_cf_ppo_grad_diagnostics
from rl.ppo_core import ppo_policy_loss, ppo_value_loss


ACTOR_INTERVENTION_REASON_CODES: dict[str, float] = {
    "missing_pair_jsd": 1.0,
    "no_valid_minibatch": 2.0,
    "not_v6i2_protocol": 3.0,
    "invalid_measurement": 4.0,
    "missing_pair_values": 5.0,
}


@dataclass
class MinibatchUpdaterState:
    actor_grad_diag_done: bool = False


class MinibatchUpdater:
    def __init__(
        self,
        *,
        model: Any,
        cfg: Any,
        hparams: Any,
        runtime: Any,
        latent_state: Any,
        device: Any,
        entropy_objective: EntropyObjective,
        strategy_objective: StrategyObjective,
        separation_objective: SeparationObjective,
        optimizer_stepper: OptimizerStepper,
        separation_generator: torch.Generator,
    ) -> None:
        self.model = model
        self.cfg = cfg
        self.hparams = hparams
        self.runtime = runtime
        self.latent_state = latent_state
        self.device = device
        self.entropy_objective = entropy_objective
        self.strategy_objective = strategy_objective
        self.separation_objective = separation_objective
        self.optimizer_stepper = optimizer_stepper
        self.separation_generator = separation_generator

    def update(
        self,
        *,
        batch: dict[str, torch.Tensor],
        context: PPOUpdateContext,
        phase_policy: PhaseTrainingPolicy,
        epoch_state: RolloutEntropyState,
        prep: RolloutMarginalPrep,
        latent_lam_h: float,
        curr_sep_coef: float,
        ent_coef: float,
        v6i1_usage_coef: float,
        epoch_idx: int,
        mb_idx: int,
        pair_count: int,
        updater_state: MinibatchUpdaterState,
    ) -> MinibatchUpdateResult:
        from rl.custom_ppo.v6i1_phase_runtime import v6i1_macro_router_active

        model = self.model
        hparams = self.hparams
        cfg = self.cfg
        runtime = self.runtime
        device = self.device
        zero_scalar = torch.zeros((), dtype=torch.float32, device=device)

        obs_batch = {
            "grid": batch["obs_grid"],
            "vec": batch["obs_vec"],
            "agent_mask": batch["obs_agent_mask"],
            "mask": batch["obs_mask"],
        }
        z_idx = batch["z"] if hparams.use_latent_strategy else None
        selector_hidden = None
        if hparams.use_latent_strategy and bool(getattr(model, "use_recurrent_selector", False)):
            if "selector_hidden" not in batch:
                raise KeyError(
                    "selector_hidden missing from rollout buffer; required for recurrent replay"
                )
            selector_hidden = batch["selector_hidden"]

        values_norm, action_log_prob, entropy, aux = model.evaluate_actions(
            obs_batch,
            batch["global_state"],
            batch["actions"],
            z_idx=z_idx,
            selector_hidden=selector_hidden,
        )
        advantages = batch["advantages"]
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (
                advantages.std(unbiased=False) + 1e-8
            )

        strategy_entropy = torch.zeros_like(entropy)
        persist_loss_value = 0.0
        latent_loss = zero_scalar
        strategy_policy_loss = zero_scalar
        strategy_aux_return_loss_value = 0.0
        strategy_phase_loss_value = 0.0
        strategy_kl_value = 0.0
        strategy_ppo_stats = {
            "approx_kl": zero_scalar,
            "clip_fraction": zero_scalar,
            "ratio": torch.ones((1,), dtype=torch.float32, device=device),
        }
        marginal_telemetry = {
            "strategy_marginal_entropy_loss_value": 0.0,
            "strategy_marginal_entropy_nats_value": 0.0,
            "strategy_marginal_entropy_kl_value": 0.0,
        }
        resample = torch.zeros_like(entropy, dtype=torch.bool)
        bundle_components: tuple[LossComponent, ...] = ()
        separation_result = self.separation_objective.compute(
            obs_batch=obs_batch,
            batch=batch,
            advantages=advantages,
            entropy=entropy,
            z_idx=z_idx,
            separation_coef=0.0,
            counterfactual_active=False,
            device=device,
            zero_scalar=zero_scalar,
        )

        if hparams.use_latent_strategy:
            resample = batch["z_resampled"].bool()
            strategy_entropy = aux["strategy_entropy"]
            h_mode = prep.h_mode
            h_goal = prep.h_goal
            has_dedicated_router_opt = runtime.latent_router_optimizer is not None
            apply_main_loop_qphi_loss = context.apply_main_loop_qphi_loss
            apply_entropy_loss = (
                hparams.use_latent_strategy
                and (
                    (not has_dedicated_router_opt and float(latent_lam_h or 0.0) > 0.0)
                    or float(v6i1_usage_coef) > 0.0
                )
                and h_goal != "none"
            )
            apply_persistence_loss = (
                hparams.use_latent_strategy
                and not has_dedicated_router_opt
                and (
                    float(getattr(cfg, "latent_lam_p", 0.0) or 0.0) > 0.0
                    or hparams.latent_sparse_tactical_refresh_enabled
                )
            )
            apply_kl_loss = (
                hparams.use_latent_strategy
                and not has_dedicated_router_opt
                and float(hparams.latent_kl_consecutive or 0.0) > 0.0
            )
            entropy_component, _ = self.entropy_objective.conditional_component(
                strategy_entropy=strategy_entropy,
                resample=resample,
                latent_lam_h=latent_lam_h,
                h_mode=h_mode,
                h_goal=h_goal,
                apply_entropy_loss=apply_entropy_loss,
                zero_scalar=zero_scalar,
            )
            marginal_component = self.entropy_objective.marginal_minibatch_component(
                epoch_state,
                mb_idx=mb_idx,
                apply_entropy_loss=apply_entropy_loss,
                apply_rollout_marginal=prep.apply_rollout_marginal,
                zero_scalar=zero_scalar,
            )
            if h_mode == "marginal" and apply_entropy_loss and epoch_state.marginal_stats:
                marginal_telemetry = {
                    "strategy_marginal_entropy_loss_value": float(
                        epoch_state.marginal_stats.get("rollout_marginal_entropy_kl", 0.0)
                    )
                    * float(prep.rollout_marginal_coef),
                    "strategy_marginal_entropy_nats_value": float(
                        epoch_state.marginal_stats.get("rollout_marginal_entropy_nats", 0.0)
                    ),
                    "strategy_marginal_entropy_kl_value": float(
                        epoch_state.marginal_stats.get("rollout_marginal_entropy_kl", 0.0)
                    ),
                }
            bundle = self.strategy_objective.compute(
                batch=batch,
                aux=aux,
                advantages=advantages,
                latent_lam_h=latent_lam_h,
                apply_main_loop_qphi_loss=apply_main_loop_qphi_loss,
                apply_entropy_loss=apply_entropy_loss,
                apply_persistence_loss=apply_persistence_loss,
                apply_kl_loss=apply_kl_loss,
                entropy_component=entropy_component,
                marginal_component=marginal_component,
                epoch_marginal_stats=epoch_state.marginal_stats,
                rollout_marginal_coef=prep.rollout_marginal_coef,
                h_mode=h_mode,
                zero_scalar=zero_scalar,
            )
            latent_loss = bundle.latent_loss
            strategy_kl_value = bundle.strategy_kl
            strategy_ppo_stats = bundle.strategy_ppo_stats
            strategy_policy_loss = bundle.strategy_policy_loss
            persist_loss_value = bundle.persist_loss_value
            strategy_aux_return_loss_value = bundle.strategy_aux_return_loss_value
            strategy_phase_loss_value = bundle.strategy_phase_loss_value
            marginal_telemetry = bundle.marginal_telemetry
            bundle_components = bundle.components

            if hparams.fixed_latent_strategy:
                strategy_entropy = torch.zeros_like(entropy)

            separation_result = self.separation_objective.compute(
                obs_batch=obs_batch,
                batch=batch,
                advantages=advantages,
                entropy=entropy,
                z_idx=z_idx,
                separation_coef=curr_sep_coef,
                counterfactual_active=phase_policy.counterfactual_active,
                device=device,
                zero_scalar=zero_scalar,
            )
            if separation_result.loss.active:
                latent_loss = latent_loss + separation_result.loss.scaled_loss

        log_prob = action_log_prob
        policy_loss, ppo_stats = ppo_policy_loss(
            log_prob,
            batch["log_probs"],
            advantages,
            hparams.clip_range,
        )
        value_targets = _normalize_value_targets(runtime, batch["returns"])
        value_loss = ppo_value_loss(
            values_norm, batch["values_norm"], value_targets, hparams.value_clip_range
        )
        entropy_loss = -entropy.mean()
        ppo_actor_loss = policy_loss + ent_coef * entropy_loss
        total_loss = (
            policy_loss
            + hparams.vf_coef * value_loss
            + ent_coef * entropy_loss
            + latent_loss
        )

        cf_telemetry: dict[str, float] = {}
        if (
            not updater_state.actor_grad_diag_done
            and separation_result.train_active > 0.0
            and float(curr_sep_coef) > 0.0
            and separation_result.loss.scaled_loss.requires_grad
        ):
            v6i1_three_opt = bool(getattr(runtime, "v6i1_three_optimizer_mode", False))
            if v6i1_three_opt:
                actor_parameters = collect_actor_optimizer_parameters(runtime.actor_optimizer)
            else:
                actor_parameters = collect_actor_parameters(model)
            cf_norm, ppo_norm, ratio = actor_cf_ppo_grad_diagnostics(
                scaled_cf_loss=separation_result.loss.scaled_loss,
                ppo_actor_loss=ppo_actor_loss,
                actor_parameters=actor_parameters,
            )
            cf_telemetry = {
                "cf_actor_grad_norm": cf_norm,
                "ppo_actor_grad_norm": ppo_norm,
                "cf_to_ppo_grad_ratio": ratio,
            }
            updater_state.actor_grad_diag_done = True

        step_result = self.optimizer_stepper.step(
            total_loss=total_loss,
            ppo_actor_loss=ppo_actor_loss,
            value_loss=value_loss,
            policy_loss=policy_loss,
            entropy_loss=entropy_loss,
            latent_loss=latent_loss,
            ent_coef=ent_coef,
            vf_coef=float(hparams.vf_coef),
            context=context,
            phase_policy=phase_policy,
            model=model,
            latent_state=self.latent_state,
            epoch_idx=epoch_idx,
            mb_idx=mb_idx,
            max_grad_norm=float(cfg.max_grad_norm),
        )

        approx_kl_value = float(ppo_stats["approx_kl"].detach().cpu().item())
        z_sep_stats = separation_result.raw_stats
        z_sep_loss = separation_result.loss.raw_value
        measurement = separation_result.pairwise_measurement

        pair_telemetry: dict[str, float] = {}
        pair_jsd_batch = z_sep_stats.get("pair_jsd")
        for idx in range(pair_count):
            key = f"cf_batch_pair_jsd_{idx}"
            if pair_jsd_batch is not None and idx < int(pair_jsd_batch.numel()):
                pair_telemetry[key] = float(pair_jsd_batch[idx].detach().cpu().item())
            else:
                pair_telemetry[key] = 0.0

        soft = epoch_state.soft_diag
        telemetry: dict[str, float] = {
            "policy_loss": float(policy_loss.detach().cpu().item()),
            "value_loss": float(value_loss.detach().cpu().item()),
            "entropy": float(entropy.mean().detach().cpu().item()),
            "approx_kl": approx_kl_value,
            "clip_fraction": float(ppo_stats["clip_fraction"].detach().cpu().item()),
            "grad_norm": float(step_result.global_grad_norm),
            "strategy_entropy": float(strategy_entropy.mean().detach().cpu().item()),
            "strategy_policy_loss": float(strategy_policy_loss.detach().cpu().item()),
            "strategy_approx_kl": float(strategy_ppo_stats["approx_kl"].detach().cpu().item()),
            "strategy_clip_fraction": float(
                strategy_ppo_stats["clip_fraction"].detach().cpu().item()
            ),
            "strategy_ratio_std": (
                float(strategy_ppo_stats["ratio"].detach().float().std(unbiased=False).cpu().item())
                if strategy_ppo_stats["ratio"].numel() > 1
                else 0.0
            ),
            "strategy_aux_return_loss": float(strategy_aux_return_loss_value),
            "strategy_persist_loss": float(persist_loss_value),
            "strategy_marginal_entropy_loss": marginal_telemetry[
                "strategy_marginal_entropy_loss_value"
            ],
            "strategy_marginal_entropy_nats": marginal_telemetry[
                "strategy_marginal_entropy_nats_value"
            ],
            "strategy_marginal_entropy_kl": marginal_telemetry[
                "strategy_marginal_entropy_kl_value"
            ],
            "router_rollout_soft_marginal_entropy_nats": float(
                soft.get("router_rollout_soft_marginal_entropy_nats", 0.0)
            ),
            "router_rollout_soft_conditional_entropy_nats": float(
                soft.get("router_rollout_soft_conditional_entropy_nats", 0.0)
            ),
            "router_rollout_soft_mi_proxy_nats": float(
                soft.get("router_rollout_soft_mi_proxy_nats", 0.0)
            ),
            "router_rollout_soft_argmax_occupancy_max": float(
                soft.get("router_rollout_soft_argmax_occupancy_max", 0.0)
            ),
            "router_rollout_soft_argmax_occupancy_min": float(
                soft.get("router_rollout_soft_argmax_occupancy_min", 0.0)
            ),
            "router_rollout_soft_argmax_occupancy_ratio": float(
                soft.get("router_rollout_soft_argmax_occupancy_ratio", 0.0)
            ),
            "router_rollout_resample_count": float(
                soft.get("router_rollout_resample_count", 0.0)
            ),
            "strategy_grad_norm": float(step_result.strategy_grad_norm),
            "strategy_resample_fraction": float(resample.float().mean().detach().cpu().item()),
            "latent_actor_z_separation_loss": float(z_sep_loss.detach().cpu().item()),
            "latent_actor_z_separation_jsd": tensor_stat(z_sep_stats.get("jsd", zero_scalar)),
            "latent_actor_z_separation_jsd_min": tensor_stat(
                z_sep_stats.get("min_jsd", z_sep_stats.get("jsd", zero_scalar))
            ),
            "latent_actor_z_separation_jsd_max": tensor_stat(
                z_sep_stats.get("max_jsd", z_sep_stats.get("jsd", zero_scalar))
            ),
            "latent_actor_z_separation_active": tensor_stat(z_sep_stats.get("active", zero_scalar)),
            "latent_actor_z_separation_train_active": float(separation_result.train_active),
            "cf_batch_pairs_below_margin": tensor_stat(z_sep_stats.get("pairs_below_margin")),
            "cf_hinge_active": tensor_stat(z_sep_stats.get("cf_hinge_active")),
            "cf_hinge_effective": tensor_stat(z_sep_stats.get("cf_hinge_effective")),
            "cf_valid_team_groups": tensor_stat(z_sep_stats.get("cf_valid_team_groups")),
            "cf_weight_sum": tensor_stat(z_sep_stats.get("cf_weight_sum")),
            "cf_effective_pairs": tensor_stat(z_sep_stats.get("cf_effective_pairs")),
            "cf_loss_requires_grad": 1.0 if bool(z_sep_loss.requires_grad) else 0.0,
            "strategy_kl": float(strategy_kl_value),
            "strategy_phase_loss": float(strategy_phase_loss_value),
            "actor_intervention_measurement_valid": 1.0 if measurement.valid else 0.0,
            **pair_telemetry,
            **cf_telemetry,
        }
        for k_idx in range(int(hparams.latent_k)):
            telemetry[f"router_rollout_soft_p_bar_z{k_idx}"] = float(
                soft.get(f"router_rollout_soft_p_bar_z{k_idx}", 0.0)
            )

        stop_for_action = (
            context.action_kl_stop_enabled
            and context.target_action_kl is not None
            and approx_kl_value > 1.5 * context.target_action_kl
        )
        strategy_kl_for_stop = float(strategy_ppo_stats["approx_kl"].detach().cpu().item())
        stop_for_strategy = (
            context.strategy_kl_stop_enabled
            and context.target_strategy_kl is not None
            and strategy_kl_for_stop > 1.5 * context.target_strategy_kl
        )
        should_stop = stop_for_action or stop_for_strategy
        stop_reason = None
        if stop_for_action:
            stop_reason = "action_kl"
        elif stop_for_strategy:
            stop_reason = "strategy_kl"

        latent_components = tuple(
            c for c in (*bundle_components, separation_result.loss) if c.active
        )

        return MinibatchUpdateResult(
            policy=LossComponent(
                name="policy",
                scaled_loss=policy_loss,
                raw_value=policy_loss.detach(),
                active=True,
                metrics={"approx_kl": approx_kl_value},
            ),
            value=LossComponent(
                name="value",
                scaled_loss=value_loss,
                raw_value=value_loss.detach(),
                active=True,
            ),
            entropy=LossComponent(
                name="entropy",
                scaled_loss=entropy_loss,
                raw_value=entropy.mean().detach(),
                active=True,
            ),
            latent_components=latent_components,
            action_kl=approx_kl_value,
            strategy_kl=strategy_kl_for_stop,
            should_stop=should_stop,
            stop_reason=stop_reason,
            grad_norms={
                "actor": step_result.actor_grad_norm,
                "critic": step_result.critic_grad_norm,
                "router": step_result.router_grad_norm,
                "global": step_result.global_grad_norm,
            },
            telemetry=telemetry,
            separation_measurement=measurement,
        )
