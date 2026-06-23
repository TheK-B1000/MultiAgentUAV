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
from rl.custom_ppo.latent.router_mask import apply_router_allowed_latent_mask
from rl.custom_ppo.update.entropy_objectives import EntropyObjective, RolloutEntropyState, RolloutMarginalPrep
from rl.custom_ppo.update.helpers import tensor_stat
from rl.custom_ppo.update.loss_result import LossComponent, MinibatchUpdateResult
from rl.custom_ppo.update.optimizer_stepper import OptimizerStepper
from rl.custom_ppo.update.phase_policy import PhaseTrainingPolicy
from rl.custom_ppo.update.separation_objectives import SeparationObjective
from rl.custom_ppo.update.strategy_objectives import StrategyObjective
from rl.custom_ppo.update.update_context import PPOUpdateContext
from rl.custom_ppo.update.actor_pathway_diagnostics import actor_pathway_grad_diagnostics_for_model
from rl.custom_ppo.v6i1_cf_loss import (
    actor_cf_ppo_grad_diagnostics,
    actor_diagnostic_grad_norm,
    v6i1_cf_separation_loss,
    v6i1_cf_separation_loss_for_action_head,
)
from rl.custom_ppo.v6i1_phase_runtime import is_v6i1_staged_trainer
from rl.ppo_core import ppo_policy_loss, ppo_value_loss


ACTOR_INTERVENTION_REASON_CODES: dict[str, float] = {
    "missing_pair_jsd": 1.0,
    "no_valid_minibatch": 2.0,
    "not_v6i2_protocol": 3.0,
    "invalid_measurement": 4.0,
    "missing_pair_values": 5.0,
    "separation_disabled": 6.0,
    "phase_counterfactual_inactive": 7.0,
    "no_active_rows": 8.0,
    "invalid_pair_jsd": 9.0,
}


def combine_action_and_message_log_probs(
    *,
    action_log_prob: torch.Tensor,
    old_action_log_prob: torch.Tensor,
    message_log_prob: torch.Tensor | None,
    old_message_log_prob: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Combine action and boundary-message log-probs exactly once for PPO."""
    if message_log_prob is None:
        return action_log_prob, old_action_log_prob
    if old_message_log_prob is None:
        raise KeyError("message_log_probs missing from rollout batch for communication PPO.")
    return action_log_prob + message_log_prob, old_action_log_prob + old_message_log_prob


def compute_actor_ppo_cf_cosine(
    *,
    scaled_cf_loss: torch.Tensor,
    ppo_actor_loss: torch.Tensor,
    actor_parameters: list[torch.nn.Parameter],
) -> float:
    params = [p for p in actor_parameters if p.requires_grad]
    if not params or not scaled_cf_loss.requires_grad or not ppo_actor_loss.requires_grad:
        return 0.0
    try:
        g_cf = torch.autograd.grad(scaled_cf_loss, params, retain_graph=True, allow_unused=True)
        g_ppo = torch.autograd.grad(ppo_actor_loss, params, retain_graph=True, allow_unused=True)
    except Exception:
        return 0.0
    cf_parts: list[torch.Tensor] = []
    ppo_parts: list[torch.Tensor] = []
    for p, gc, gp in zip(params, g_cf, g_ppo):
        cf_parts.append(torch.zeros(p.numel(), device=p.device) if gc is None else gc.reshape(-1))
        ppo_parts.append(torch.zeros(p.numel(), device=p.device) if gp is None else gp.reshape(-1))
    cf = torch.cat(cf_parts)
    ppo = torch.cat(ppo_parts)
    denom = torch.norm(cf) * torch.norm(ppo)
    if float(denom.detach().cpu().item()) <= 1e-12:
        return 0.0
    return float((torch.dot(cf, ppo) / denom).detach().cpu().item())


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

        message_symbols = batch.get("message_symbols")
        message_boundary_mask = batch.get("message_boundary_mask")
        values_norm, action_log_prob, entropy, aux = model.evaluate_actions(
            obs_batch,
            batch["global_state"],
            batch["actions"],
            z_idx=z_idx,
            selector_hidden=selector_hidden,
            router_context=batch.get("router_context"),
            message_symbols=message_symbols,
            message_boundary_mask=message_boundary_mask,
        )
        if hparams.use_latent_strategy and "strategy_logits" in aux:
            masked_strategy_logits = apply_router_allowed_latent_mask(
                aux["strategy_logits"],
                cfg=cfg,
                latent_k=int(hparams.latent_k),
            )
            strategy_dist = torch.distributions.Categorical(logits=masked_strategy_logits)
            aux["strategy_logits"] = masked_strategy_logits
            aux["strategy_log_prob"] = strategy_dist.log_prob(batch["z"].long())
            aux["strategy_entropy"] = strategy_dist.entropy()
        message_log_prob = aux.get("message_log_probs")
        message_entropy = aux.get("message_entropy")
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

        # Message PPO is boundary-only: held symbols persist in obs transport for
        # comm_interval_steps, but log-probs are stored/evaluated only on send rows.
        log_prob, old_log_probs = combine_action_and_message_log_probs(
            action_log_prob=action_log_prob,
            old_action_log_prob=batch["log_probs"],
            message_log_prob=message_log_prob,
            old_message_log_prob=batch.get("message_log_probs"),
        )
        policy_loss, ppo_stats = ppo_policy_loss(
            log_prob,
            old_log_probs,
            advantages,
            hparams.clip_range,
        )
        value_targets = _normalize_value_targets(runtime, batch["returns"])
        value_loss = ppo_value_loss(
            values_norm, batch["values_norm"], value_targets, hparams.value_clip_range
        )
        entropy_loss = -entropy.mean()
        comm_entropy_coef = float(getattr(cfg, "comm_entropy_coef", 0.0) or 0.0)
        message_entropy_loss = zero_scalar
        if (
            bool(getattr(model, "communication_enabled", False))
            and message_entropy is not None
            and comm_entropy_coef > 0.0
        ):
            message_entropy_loss = -message_entropy.mean()
        ppo_actor_loss = policy_loss + ent_coef * entropy_loss + comm_entropy_coef * message_entropy_loss
        total_loss = (
            policy_loss
            + hparams.vf_coef * value_loss
            + ent_coef * entropy_loss
            + comm_entropy_coef * message_entropy_loss
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
            cf_telemetry.update(
                actor_pathway_grad_diagnostics_for_model(
                    model=model,
                    scaled_cf_loss=separation_result.loss.scaled_loss,
                    ppo_actor_loss=ppo_actor_loss,
                )
            )
            if is_v6i1_staged_trainer(runtime):
                cf_margin = float(
                    getattr(cfg, "latent_cf_jsd_margin", 0.01)
                    or getattr(hparams, "latent_actor_z_separation_margin", 0.02)
                    or 0.01
                )
                competence, competence_ready = self.latent_state.compute_competence_scores()
                for head_idx, head_key in enumerate(("macro", "waypoint")):
                    if head_idx >= len(model.per_agent_action_dims):
                        break
                    head_loss = v6i1_cf_separation_loss_for_action_head(
                        model,
                        obs_batch,
                        action_head_idx=head_idx,
                        latent_k=int(hparams.latent_k),
                        margin=cf_margin,
                        competence=competence,
                        competence_ready=bool(competence_ready),
                        subsample_generator=self.separation_objective.subsample_generator,
                    )
                    scaled_head = float(curr_sep_coef) * head_loss
                    cf_telemetry[f"cf_{head_key}_grad_norm"] = actor_diagnostic_grad_norm(
                        scaled_head,
                        actor_parameters,
                    )
            updater_state.actor_grad_diag_done = True

        actor_cf_update_mode = str(getattr(cfg, "actor_cf_update_mode", "combined") or "combined")
        if bool(getattr(cfg, "latent_cf_sequential_update", False)):
            actor_cf_update_mode = "ppo_then_cf"
        update_order_stats: dict[str, float] = {
            "actor_cf_update_mode_code": {"combined": 0.0, "ppo_then_cf": 1.0, "cf_then_ppo": 2.0}.get(actor_cf_update_mode, -1.0),
            "actor_ppo_optimizer_step_count": 0.0,
            "actor_cf_optimizer_step_count": 0.0,
            "actor_jsd_update_start": float("nan"),
            "actor_jsd_after_ppo": float("nan"),
            "actor_jsd_after_cf": float("nan"),
            "actor_jsd_after_first_substep": float("nan"),
            "actor_jsd_after_second_substep": float("nan"),
            "ppo_jsd_delta": float("nan"),
            "cf_jsd_delta": float("nan"),
            "cf_gain": float("nan"),
            "retained_cf_gain": float("nan"),
            "cf_retention_ratio": float("nan"),
            "cf_retention_reason_code": 0.0,
            "actor_kl_after_ppo": float("nan"),
            "actor_kl_after_cf": float("nan"),
            "actor_kl_after_second_substep": float("nan"),
        }
        is_sequential = (
            actor_cf_update_mode in {"ppo_then_cf", "cf_then_ppo"}
            and bool(getattr(runtime, "v6i1_three_optimizer_mode", False))
            and getattr(runtime, "actor_cf_optimizer", None) is not None
            and float(curr_sep_coef) > 0.0
            and phase_policy.counterfactual_active
        )
        if is_sequential:
            from rl.custom_ppo.update.optimizer_stepper import OptimizerStepResult, clip_optimizer_grad_norm

            competence, competence_ready = self.latent_state.compute_competence_scores()

            def _measure_jsd() -> tuple[float, list[float]]:
                with torch.no_grad():
                    _, stats = v6i1_cf_separation_loss(
                        model,
                        obs_batch,
                        latent_k=int(hparams.latent_k),
                        margin=float(cfg.latent_cf_jsd_margin),
                        competence=competence,
                        competence_ready=bool(competence_ready),
                        weak_pair_ema=getattr(self.latent_state, "cf_pair_jsd_ema", None),
                        weak_pair_boost=float(getattr(cfg, "latent_cf_weak_pair_boost", 0.0) or 0.0),
                        worst_pair_coef=float(getattr(cfg, "latent_cf_worst_pair_coef", 0.0) or 0.0),
                        require_competence=bool(getattr(cfg, "latent_cf_require_competence", False)),
                        subsample_generator=self.separation_objective.subsample_generator,
                    )
                pair = stats.get("pair_jsd")
                pairs = pair.detach().reshape(-1).cpu().tolist() if isinstance(pair, torch.Tensor) else []
                return float(stats["jsd"].detach().cpu().item()), pairs

            def _kl_now() -> float:
                with torch.no_grad():
                    _, lp, _, aux_now = model.evaluate_actions(
                        obs_batch,
                        batch["global_state"],
                        batch["actions"],
                        z_idx=z_idx,
                        selector_hidden=selector_hidden,
                        router_context=batch.get("router_context"),
                        message_symbols=message_symbols,
                        message_boundary_mask=message_boundary_mask,
                    )
                    new_lp, old_lp = combine_action_and_message_log_probs(
                        action_log_prob=lp,
                        old_action_log_prob=batch["log_probs"],
                        message_log_prob=aux_now.get("message_log_probs"),
                        old_message_log_prob=batch.get("message_log_probs"),
                    )
                    logratio = new_lp - old_lp
                    return float(((logratio.exp() - 1.0) - logratio).mean().detach().cpu().item())

            def _ppo_actor_loss_now() -> torch.Tensor:
                _, lp, ent_now, aux_now = model.evaluate_actions(
                    obs_batch,
                    batch["global_state"],
                    batch["actions"],
                    z_idx=z_idx,
                    selector_hidden=selector_hidden,
                    router_context=batch.get("router_context"),
                    message_symbols=message_symbols,
                    message_boundary_mask=message_boundary_mask,
                )
                new_lp, old_lp = combine_action_and_message_log_probs(
                    action_log_prob=lp,
                    old_action_log_prob=batch["log_probs"],
                    message_log_prob=aux_now.get("message_log_probs"),
                    old_message_log_prob=batch.get("message_log_probs"),
                )
                p_loss, _ = ppo_policy_loss(new_lp, old_lp, advantages, hparams.clip_range)
                return p_loss + ent_coef * (-ent_now.mean())

            def _cf_loss_now() -> torch.Tensor:
                raw, _ = v6i1_cf_separation_loss(
                    model,
                    obs_batch,
                    latent_k=int(hparams.latent_k),
                    margin=float(cfg.latent_cf_jsd_margin),
                    competence=competence,
                    competence_ready=bool(competence_ready),
                    weak_pair_ema=getattr(self.latent_state, "cf_pair_jsd_ema", None),
                    weak_pair_boost=float(getattr(cfg, "latent_cf_weak_pair_boost", 0.0) or 0.0),
                    worst_pair_coef=float(getattr(cfg, "latent_cf_worst_pair_coef", 0.0) or 0.0),
                    require_competence=bool(getattr(cfg, "latent_cf_require_competence", False)),
                    subsample_generator=self.separation_objective.subsample_generator,
                )
                return float(curr_sep_coef) * raw

            actor_params = collect_actor_optimizer_parameters(runtime.actor_optimizer)
            z_emb_params = [p for name, p in model.named_parameters() if "strategy_embedding" in name and p.requires_grad]
            ppo_cf_cosine = compute_actor_ppo_cf_cosine(
                scaled_cf_loss=separation_result.loss.scaled_loss,
                ppo_actor_loss=ppo_actor_loss,
                actor_parameters=actor_params,
            )
            ppo_grad_norm = 0.0
            cf_grad_norm = 0.0
            ppo_parameter_delta = 0.0
            cf_parameter_delta = 0.0
            z_embedding_ppo_delta = 0.0
            z_embedding_cf_delta = 0.0
            pair_jsd_before_ppo = [0.0] * pair_count
            pair_jsd_after_ppo = [0.0] * pair_count
            pair_jsd_after_cf = [0.0] * pair_count
            start_jsd, start_pairs = _measure_jsd()
            update_order_stats["actor_jsd_before_substeps"] = start_jsd
            update_order_stats["actor_jsd_update_start"] = start_jsd
            for i, v in enumerate(start_pairs[:pair_count]):
                pair_jsd_before_ppo[i] = float(v)

            latent_loss_no_cf = latent_loss - separation_result.loss.scaled_loss if separation_result.loss.active else latent_loss
            include_latent_loss_no_cf = (
                isinstance(latent_loss_no_cf, torch.Tensor)
                and latent_loss_no_cf.requires_grad
                and (phase_policy.router_step_enabled or str(getattr(cfg, "router_context_mode", "") or "") != "current_plus_delta")
            )

            def _zero_all() -> None:
                runtime.actor_optimizer.zero_grad(set_to_none=True)
                runtime.actor_cf_optimizer.zero_grad(set_to_none=True)
                runtime.critic_optimizer.zero_grad(set_to_none=True)
                if runtime.router_optimizer is not None:
                    runtime.router_optimizer.zero_grad(set_to_none=True)

            def _ppo_step() -> tuple[float, float, list[float]]:
                nonlocal ppo_grad_norm, ppo_parameter_delta, z_embedding_ppo_delta
                before = [p.detach().clone() for p in actor_params]
                z_before = [p.detach().clone() for p in z_emb_params]
                _zero_all()
                assembled = hparams.vf_coef * value_loss
                if phase_policy.actor_step_enabled:
                    assembled = assembled + _ppo_actor_loss_now()
                if include_latent_loss_no_cf:
                    assembled = assembled + latent_loss_no_cf
                assembled.backward()
                ppo_grad_norm = clip_optimizer_grad_norm(runtime.actor_optimizer, float(cfg.max_grad_norm))
                if phase_policy.critic_step_enabled:
                    clip_optimizer_grad_norm(runtime.critic_optimizer, float(cfg.max_grad_norm))
                    runtime.critic_optimizer.step()
                if runtime.router_optimizer is not None and include_latent_loss_no_cf:
                    clip_optimizer_grad_norm(runtime.router_optimizer, float(cfg.max_grad_norm))
                    runtime.router_optimizer.step()
                if phase_policy.actor_step_enabled:
                    runtime.actor_optimizer.step()
                ppo_parameter_delta = float(sum((p - p0).pow(2).sum() for p, p0 in zip(actor_params, before)).sqrt().item())
                z_embedding_ppo_delta = float(sum((p - p0).pow(2).sum() for p, p0 in zip(z_emb_params, z_before)).sqrt().item()) if z_emb_params else 0.0
                jsd, pairs = _measure_jsd()
                return jsd, _kl_now(), pairs

            def _cf_step() -> tuple[float, float, list[float]]:
                nonlocal cf_grad_norm, cf_parameter_delta, z_embedding_cf_delta
                before = [p.detach().clone() for p in actor_params]
                z_before = [p.detach().clone() for p in z_emb_params]
                _zero_all()
                loss = _cf_loss_now()
                loss.backward()
                cf_grad_norm = clip_optimizer_grad_norm(runtime.actor_cf_optimizer, float(cfg.max_grad_norm))
                runtime.actor_cf_optimizer.step()
                cf_parameter_delta = float(sum((p - p0).pow(2).sum() for p, p0 in zip(actor_params, before)).sqrt().item())
                z_embedding_cf_delta = float(sum((p - p0).pow(2).sum() for p, p0 in zip(z_emb_params, z_before)).sqrt().item()) if z_emb_params else 0.0
                jsd, pairs = _measure_jsd()
                return jsd, _kl_now(), pairs

            if actor_cf_update_mode == "ppo_then_cf":
                after_ppo, kl_ppo, ppo_pairs = _ppo_step()
                update_order_stats["actor_ppo_optimizer_step_count"] = 1.0 if phase_policy.actor_step_enabled else 0.0
                after_cf, kl_cf, cf_pairs = _cf_step()
                update_order_stats["actor_cf_optimizer_step_count"] = 1.0
                update_order_stats.update(
                    {
                        "actor_jsd_after_ppo": after_ppo,
                        "actor_jsd_after_cf": after_cf,
                        "actor_jsd_after_first_substep": after_ppo,
                        "actor_jsd_after_second_substep": after_cf,
                        "ppo_jsd_delta": after_ppo - start_jsd,
                        "cf_jsd_delta": after_cf - after_ppo,
                        "cf_gain": max(0.0, after_cf - after_ppo),
                        "retained_cf_gain": after_cf - after_ppo,
                        "actor_kl_after_ppo": kl_ppo,
                        "actor_kl_after_cf": kl_cf,
                        "actor_kl_after_second_substep": kl_cf,
                    }
                )
            else:
                after_cf, kl_cf, cf_pairs = _cf_step()
                update_order_stats["actor_cf_optimizer_step_count"] = 1.0
                after_ppo, kl_ppo, ppo_pairs = _ppo_step()
                update_order_stats["actor_ppo_optimizer_step_count"] = 1.0 if phase_policy.actor_step_enabled else 0.0
                update_order_stats.update(
                    {
                        "actor_jsd_after_ppo": after_ppo,
                        "actor_jsd_after_cf": after_cf,
                        "actor_jsd_after_first_substep": after_cf,
                        "actor_jsd_after_second_substep": after_ppo,
                        "cf_jsd_delta": after_cf - start_jsd,
                        "ppo_jsd_delta": after_ppo - after_cf,
                        "cf_gain": max(0.0, after_cf - start_jsd),
                        "retained_cf_gain": after_ppo - start_jsd,
                        "actor_kl_after_ppo": kl_ppo,
                        "actor_kl_after_cf": kl_cf,
                        "actor_kl_after_second_substep": kl_ppo,
                    }
                )
            gain = float(update_order_stats["cf_gain"])
            if gain <= 1e-12:
                update_order_stats["cf_retention_ratio"] = float("nan")
                update_order_stats["cf_retention_reason_code"] = 1.0
            else:
                update_order_stats["cf_retention_ratio"] = float(update_order_stats["retained_cf_gain"]) / max(gain, 1e-12)
            for i, v in enumerate(ppo_pairs[:pair_count]):
                pair_jsd_after_ppo[i] = float(v)
            for i, v in enumerate(cf_pairs[:pair_count]):
                pair_jsd_after_cf[i] = float(v)
            denom = max(float(ppo_grad_norm), 1e-12)
            sequential_grad_ratio = float(cf_grad_norm) / denom
            cf_telemetry.update(
                {
                    "cf_actor_grad_norm": float(cf_grad_norm),
                    "ppo_actor_grad_norm": float(ppo_grad_norm),
                    "cf_to_ppo_grad_ratio": sequential_grad_ratio,
                    "actor_grad_norm_cf": float(cf_grad_norm),
                    "actor_grad_norm_ppo": float(ppo_grad_norm),
                    "actor_cf_grad_norm_scaled": float(cf_grad_norm),
                    "actor_ppo_grad_norm": float(ppo_grad_norm),
                    "actor_cf_to_ppo_grad_ratio": sequential_grad_ratio,
                    "actor_grad_ratio_cf_to_ppo": sequential_grad_ratio,
                    "actor_grad_ratio_cf_to_ppo_denominator_clamped": (
                        1.0 if float(ppo_grad_norm) < 1e-12 else 0.0
                    ),
                    "actor_grad_ppo_valid": 1.0,
                    "actor_grad_cf_valid": 1.0,
                    "actor_cf_loss_evaluated": 1.0,
                }
            )
            if cf_pairs:
                cf_pair_values = [float(v) for v in cf_pairs[:pair_count]]
                cf_telemetry.update(
                    {
                        "cf_batch_pair_jsd_mean": float(sum(cf_pair_values) / len(cf_pair_values)),
                        "cf_batch_pair_jsd_min": float(min(cf_pair_values)),
                        "cf_batch_pair_jsd_max": float(max(cf_pair_values)),
                        "cf_batch_pairs_total": float(len(cf_pair_values)),
                        "cf_valid_pair_count": float(len(cf_pair_values)),
                        "actor_cf_valid_pair_count": float(len(cf_pair_values)),
                        "cf_batch_pairs_above_margin": float(
                            sum(
                                1
                                for v in cf_pair_values
                                if v >= float(getattr(cfg, "latent_cf_jsd_margin", 0.0) or 0.0)
                            )
                        ),
                    }
                )
                cf_telemetry["cf_batch_pairs_above_margin_fraction"] = (
                    cf_telemetry["cf_batch_pairs_above_margin"]
                    / max(cf_telemetry["cf_batch_pairs_total"], 1.0)
                )
            step_result = OptimizerStepResult(
                actor_grad_norm=float(ppo_grad_norm),
                critic_grad_norm=0.0,
                router_grad_norm=0.0,
                global_grad_norm=max(float(ppo_grad_norm), float(cf_grad_norm)),
                strategy_grad_norm=float(self.latent_state.strategy_encoder_grad_norm()),
            )
        else:
            ppo_grad_norm = 0.0
            cf_grad_norm = 0.0
            ppo_cf_cosine = 0.0
            ppo_parameter_delta = 0.0
            cf_parameter_delta = 0.0
            z_embedding_ppo_delta = 0.0
            z_embedding_cf_delta = 0.0
            pair_jsd_before_ppo = [0.0] * pair_count
            pair_jsd_after_ppo = [0.0] * pair_count
            pair_jsd_after_cf = [0.0] * pair_count
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
        if measurement.valid and measurement.values is not None:
            values = measurement.values.detach().reshape(-1)
            for idx in range(pair_count):
                if idx < int(values.numel()):
                    pair_telemetry[f"cf_batch_pair_jsd_{idx}"] = float(values[idx].cpu().item())
        pair_hinge = z_sep_stats.get("cf_pair_hinge")
        if isinstance(pair_hinge, torch.Tensor):
            values = pair_hinge.detach().reshape(-1)
            for idx in range(pair_count):
                if idx < int(values.numel()):
                    pair_telemetry[f"cf_pair_hinge_{idx}"] = float(values[idx].cpu().item())
        pair_weight = z_sep_stats.get("cf_pair_weight")
        if isinstance(pair_weight, torch.Tensor):
            values = pair_weight.detach().reshape(-1)
            for idx in range(pair_count):
                if idx < int(values.numel()):
                    pair_telemetry[f"cf_pair_weight_{idx}"] = float(values[idx].cpu().item())

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
            "cf_mean_pair_hinge": tensor_stat(z_sep_stats.get("cf_mean_pair_hinge")),
            "cf_worst_pair_hinge": tensor_stat(z_sep_stats.get("cf_worst_pair_hinge")),
            "cf_worst_pair_index": tensor_stat(z_sep_stats.get("cf_worst_pair_index")),
            "cf_worst_pair_coef": tensor_stat(z_sep_stats.get("cf_worst_pair_coef")),
            "cf_weak_pair_boost": tensor_stat(z_sep_stats.get("cf_weak_pair_boost")),
            "cf_competence_required": tensor_stat(z_sep_stats.get("cf_competence_required")),
            "cf_loss_requires_grad": (
                1.0 if bool(separation_result.loss.scaled_loss.requires_grad) else 0.0
            ),
            "cf_batch_macro_jsd": tensor_stat(z_sep_stats.get("cf_batch_macro_jsd", zero_scalar)),
            "cf_batch_waypoint_jsd": tensor_stat(
                z_sep_stats.get("cf_batch_waypoint_jsd", zero_scalar)
            ),
            "strategy_kl": float(strategy_kl_value),
            "strategy_phase_loss": float(strategy_phase_loss_value),
            "actor_intervention_measurement_valid": 1.0 if measurement.valid else 0.0,
            **pair_telemetry,
            **cf_telemetry,
            **update_order_stats,
            "ppo_parameter_delta": ppo_parameter_delta,
            "cf_parameter_delta": cf_parameter_delta,
            "z_embedding_ppo_delta": z_embedding_ppo_delta,
            "z_embedding_cf_delta": z_embedding_cf_delta,
            "ppo_grad_norm": ppo_grad_norm,
            "cf_grad_norm": cf_grad_norm,
            "ppo_cf_cosine": ppo_cf_cosine,
        }
        for idx in range(pair_count):
            telemetry[f"cf_batch_pair_jsd_before_ppo_{idx}"] = pair_jsd_before_ppo[idx]
            telemetry[f"cf_batch_pair_jsd_after_ppo_{idx}"] = pair_jsd_after_ppo[idx]
            telemetry[f"cf_batch_pair_jsd_after_cf_{idx}"] = pair_jsd_after_cf[idx]
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
