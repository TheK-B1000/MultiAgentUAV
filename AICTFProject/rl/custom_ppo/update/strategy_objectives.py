"""Strategy PPO, persistence, KL, phase-aux, and aux-return objectives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from rl.latent_losses import (
    strategy_aux_return_loss,
    strategy_kl_consecutive_loss,
    strategy_persistence_loss,
    strategy_phase_aux_loss,
    strategy_ppo_loss,
)
from rl.custom_ppo.return_normalization import _normalize_strategy_returns
from rl.custom_ppo.update.loss_result import LossComponent


@dataclass
class StrategyLossBundle:
    components: tuple[LossComponent, ...]
    latent_loss: torch.Tensor
    strategy_kl: float
    resample_fraction: float
    strategy_ppo_stats: dict[str, Any]
    strategy_policy_loss: torch.Tensor
    persist_loss_value: float
    strategy_aux_return_loss_value: float
    strategy_phase_loss_value: float
    marginal_telemetry: dict[str, float]


class StrategyObjective:
    def __init__(
        self,
        *,
        model: Any,
        cfg: Any,
        hparams: Any,
        runtime: Any,
        device: Any,
    ) -> None:
        self.model = model
        self.cfg = cfg
        self.hparams = hparams
        self.runtime = runtime
        self.device = device

    def compute(
        self,
        *,
        batch: dict[str, torch.Tensor],
        aux: dict[str, torch.Tensor],
        advantages: torch.Tensor,
        latent_lam_h: float,
        apply_main_loop_qphi_loss: bool,
        apply_entropy_loss: bool,
        apply_persistence_loss: bool,
        apply_kl_loss: bool,
        entropy_component: LossComponent,
        marginal_component: LossComponent | None,
        epoch_marginal_stats: dict[str, float],
        rollout_marginal_coef: float,
        h_mode: str,
        zero_scalar: torch.Tensor,
    ) -> StrategyLossBundle:
        # V6I7: use router_decision_valid (True only at actual opportunity indices, never forced-z
        # or continuation steps).  Fall back to z_resampled for pre-V6I7 buffers.
        if "router_decision_valid" in batch:
            resample = batch["router_decision_valid"].bool()
        else:
            resample = batch["z_resampled"].bool()
        persist_mask = batch["z_persist_mask"].bool()
        components: list[LossComponent] = []

        persist_term_loss, persist_stats = strategy_persistence_loss(
            aux["strategy_logits"],
            batch["prev_z"],
            persist_mask,
            lam_p=float(getattr(self.cfg, "latent_lam_p", 0.0)),
            device=self.device,
        )
        if (
            self.hparams.latent_resample_every_n == 0
            and not self.hparams.latent_resample_on_flag
            and not self.hparams.latent_event_refresh_enabled
            and not self.hparams.latent_sparse_tactical_refresh_enabled
        ):
            assert persist_stats["persist_term"] == 0.0, (
                "L_persist must be exactly 0 when no mid-episode resampling"
            )
        if not apply_persistence_loss:
            persist_term_loss = torch.zeros_like(persist_term_loss)
        persist_loss_value = persist_stats["persist_term"] if apply_persistence_loss else 0.0
        components.append(
            LossComponent(
                name="persistence",
                scaled_loss=persist_term_loss,
                raw_value=persist_term_loss.detach(),
                active=bool(apply_persistence_loss),
                metrics={"persist_term": persist_loss_value},
            )
        )
        components.append(entropy_component)
        if marginal_component is not None:
            components.append(marginal_component)

        strategy_kl_value = 0.0
        if float(self.hparams.latent_kl_consecutive or 0.0) > 0.0:
            kl_loss, kl_stats = strategy_kl_consecutive_loss(
                aux["strategy_logits"],
                batch["z_logits_prev"],
                batch["z_kl_prev_valid"],
                coef=float(self.hparams.latent_kl_consecutive),
            )
            if not apply_kl_loss:
                kl_loss = torch.zeros_like(kl_loss)
            strategy_kl_value = float(kl_stats["kl_mean"])
            components.append(
                LossComponent(
                    name="strategy_kl",
                    scaled_loss=kl_loss,
                    raw_value=kl_loss.detach(),
                    active=bool(apply_kl_loss),
                    metrics={"kl_mean": strategy_kl_value},
                )
            )

        phase_loss_value = 0.0
        if float(self.hparams.latent_strategy_aux_predict_phase_coef or 0.0) > 0.0:
            phase_logits = self.model.phase_logits_from_strategy_logits(aux["strategy_logits"])
            phase_loss_scaled, phase_stats = strategy_phase_aux_loss(
                phase_logits,
                batch["phase_id"],
                coef=float(self.hparams.latent_strategy_aux_predict_phase_coef),
            )
            has_dedicated = getattr(self.runtime, "latent_router_optimizer", None) is not None
            if has_dedicated:
                phase_loss_scaled = torch.zeros_like(phase_loss_scaled)
            phase_loss_value = float(phase_stats["phase_term"])
            components.append(
                LossComponent(
                    name="phase_aux",
                    scaled_loss=phase_loss_scaled,
                    raw_value=phase_loss_scaled.detach(),
                    active=phase_loss_value > 0.0,
                    metrics={"phase_term": phase_loss_value},
                )
            )

        if components:
            active_scaled = [c.scaled_loss for c in components if c.active]
            latent_loss = (
                sum(active_scaled[1:], start=active_scaled[0])
                if active_scaled
                else zero_scalar
            )
        else:
            latent_loss = zero_scalar

        if self.hparams.fixed_latent_strategy:
            latent_loss = zero_scalar
            persist_loss_value = 0.0
            components = []

        marginal_telemetry = {
            "strategy_marginal_entropy_loss_value": 0.0,
            "strategy_marginal_entropy_nats_value": 0.0,
            "strategy_marginal_entropy_kl_value": 0.0,
        }
        if h_mode == "marginal" and apply_entropy_loss and epoch_marginal_stats:
            marginal_telemetry = {
                "strategy_marginal_entropy_loss_value": float(
                    epoch_marginal_stats.get("rollout_marginal_entropy_kl", 0.0)
                )
                * float(rollout_marginal_coef),
                "strategy_marginal_entropy_nats_value": float(
                    epoch_marginal_stats.get("rollout_marginal_entropy_nats", 0.0)
                ),
                "strategy_marginal_entropy_kl_value": float(
                    epoch_marginal_stats.get("rollout_marginal_entropy_kl", 0.0)
                ),
            }

        strat_adv = (
            batch["option_advantages"]
            if getattr(self.cfg, "latent_q_phi_option_advantage", False)
            else advantages
        )
        if self.hparams.fixed_latent_strategy:
            return StrategyLossBundle(
                components=tuple(components),
                latent_loss=zero_scalar,
                strategy_kl=strategy_kl_value,
                resample_fraction=float(resample.float().mean().detach().cpu().item()),
                strategy_ppo_stats={
                    "approx_kl": zero_scalar,
                    "clip_fraction": zero_scalar,
                    "ratio": torch.ones((1,), dtype=torch.float32, device=self.device),
                },
                strategy_policy_loss=zero_scalar,
                persist_loss_value=0.0,
                strategy_aux_return_loss_value=0.0,
                strategy_phase_loss_value=phase_loss_value,
                marginal_telemetry=marginal_telemetry,
            )

        strategy_policy_loss_scaled, strategy_ppo_stats = strategy_ppo_loss(
            aux["strategy_log_prob"],
            batch["z_log_probs"],
            strat_adv,
            resample,
            clip_range=float(self.hparams.clip_range),
            coef=float(self.hparams.latent_strategy_ppo_coef),
            device=self.device,
        )
        strategy_policy_loss = strategy_ppo_stats.pop("policy_loss", zero_scalar)
        if not apply_main_loop_qphi_loss:
            strategy_policy_loss_scaled = torch.zeros_like(strategy_policy_loss_scaled)
            strategy_policy_loss = torch.zeros_like(strategy_policy_loss)

        aux_return_value = 0.0
        if not self.hparams.fixed_latent_strategy and bool(resample.any().item()):
            latent_loss = latent_loss + strategy_policy_loss_scaled
            components.append(
                LossComponent(
                    name="strategy_ppo",
                    scaled_loss=strategy_policy_loss_scaled,
                    raw_value=strategy_policy_loss.detach(),
                    active=bool(apply_main_loop_qphi_loss),
                    metrics={
                        "approx_kl": float(strategy_ppo_stats["approx_kl"].detach().cpu().item()),
                    },
                )
            )
            if (
                self.hparams.latent_strategy_aux_return_head
                and float(self.hparams.latent_strategy_aux_return_coef or 0.0) > 0.0
            ):
                pred_all = self.model.strategy_aux_return_predictions(batch["global_state"])
                ret_target = _normalize_strategy_returns(
                    self.runtime, batch["returns"][resample]
                )
                aux_scaled, aux_stats = strategy_aux_return_loss(
                    pred_all,
                    batch["z"],
                    ret_target,
                    resample,
                    latent_k=int(self.hparams.latent_k),
                    coef=float(
                        getattr(
                            self.runtime,
                            "latent_strategy_aux_coef",
                            self.hparams.latent_strategy_aux_return_coef,
                        )
                    ),
                    device=self.device,
                )
                aux_return_value = float(aux_stats["aux_return_term"])
                has_dedicated = getattr(self.runtime, "latent_router_optimizer", None) is not None
                if has_dedicated:
                    aux_scaled = torch.zeros_like(aux_scaled)
                    aux_return_value = 0.0
                latent_loss = latent_loss + aux_scaled
                components.append(
                    LossComponent(
                        name="aux_return",
                        scaled_loss=aux_scaled,
                        raw_value=aux_scaled.detach(),
                        active=aux_return_value > 0.0,
                        metrics={"aux_return_term": aux_return_value},
                    )
                )
        else:
            components.append(
                LossComponent(
                    name="strategy_ppo",
                    scaled_loss=zero_scalar,
                    raw_value=zero_scalar,
                    active=False,
                )
            )

        return StrategyLossBundle(
            components=tuple(components),
            latent_loss=latent_loss,
            strategy_kl=strategy_kl_value,
            resample_fraction=float(resample.float().mean().detach().cpu().item()),
            strategy_ppo_stats=strategy_ppo_stats,
            strategy_policy_loss=strategy_policy_loss,
            persist_loss_value=float(persist_loss_value),
            strategy_aux_return_loss_value=aux_return_value,
            strategy_phase_loss_value=phase_loss_value,
            marginal_telemetry=marginal_telemetry,
        )
