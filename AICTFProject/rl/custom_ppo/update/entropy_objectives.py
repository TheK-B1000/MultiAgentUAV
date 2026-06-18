"""Strategy entropy objectives (conditional + rollout marginal)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from rl.latent_losses import (
    rollout_marginal_entropy_loss,
    rollout_router_soft_diagnostics,
)
from rl.custom_ppo.update.loss_result import LossComponent
from rl.custom_ppo.update.separation_objectives import extract_rollout_resample_subset
from rl.ppo_core import TensorDictRolloutBuffer


@dataclass
class RolloutEntropyState:
    """Per-epoch rollout marginal entropy (consumed on first minibatch)."""

    loss_for_epoch: torch.Tensor | None = None
    consumed: bool = False
    marginal_stats: dict[str, float] = field(default_factory=dict)
    soft_diag: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class RolloutMarginalPrep:
    apply_rollout_marginal: bool
    rollout_marginal_coef: float
    resample_states: torch.Tensor | None
    resample_hidden: torch.Tensor | None
    skip_reason: str | None
    h_mode: str
    h_goal: str


class EntropyObjective:
    """Conditional per-minibatch and rollout-level marginal entropy."""

    def __init__(self, *, model: Any, cfg: Any, hparams: Any, device: Any) -> None:
        self.model = model
        self.cfg = cfg
        self.hparams = hparams
        self.device = device

    def prepare_rollout(
        self,
        buffer: TensorDictRolloutBuffer,
        *,
        latent_lam_h: float,
        v6i1_usage_coef: float,
        has_dedicated_router_opt: bool,
    ) -> RolloutMarginalPrep:
        h_mode = str(getattr(self.cfg, "latent_entropy_mode", "conditional") or "conditional").lower()
        h_goal = str(getattr(self.cfg, "latent_entropy_objective", "maximize") or "maximize").lower()
        apply = (
            self.hparams.use_latent_strategy
            and h_mode == "marginal"
            and not has_dedicated_router_opt
            and float(latent_lam_h or 0.0) > 0.0
            and h_goal != "none"
            and not bool(self.hparams.fixed_latent_strategy)
        ) or (
            self.hparams.use_latent_strategy
            and float(v6i1_usage_coef) > 0.0
            and has_dedicated_router_opt
            and not bool(self.hparams.fixed_latent_strategy)
        )
        coef = float(latent_lam_h if float(v6i1_usage_coef) <= 0.0 else v6i1_usage_coef)
        states: torch.Tensor | None = None
        hidden: torch.Tensor | None = None
        skip: str | None = None
        if apply:
            require_hidden = bool(getattr(self.model, "use_recurrent_selector", False))
            states, hidden, skip = extract_rollout_resample_subset(
                buffer, require_selector_hidden=require_hidden
            )
        return RolloutMarginalPrep(
            apply_rollout_marginal=bool(apply),
            rollout_marginal_coef=coef,
            resample_states=states,
            resample_hidden=hidden,
            skip_reason=skip,
            h_mode=h_mode,
            h_goal=h_goal,
        )

    def for_epoch(
        self,
        prep: RolloutMarginalPrep,
        *,
        v6i1_usage_coef: float,
    ) -> RolloutEntropyState:
        state = RolloutEntropyState()
        if not prep.apply_rollout_marginal or prep.resample_states is None:
            return state
        if bool(getattr(self.model, "use_recurrent_selector", False)):
            logits = self.model.strategy_logits(
                prep.resample_states, selector_hidden=prep.resample_hidden
            )
        else:
            logits = self.model.strategy_logits(prep.resample_states)
        objective = "maximize" if float(v6i1_usage_coef) > 0.0 else prep.h_goal
        loss, stats = rollout_marginal_entropy_loss(
            logits,
            objective=objective,
            lam_h=float(prep.rollout_marginal_coef),
            latent_k=int(self.hparams.latent_k),
            device=self.device,
        )
        state.loss_for_epoch = loss
        state.marginal_stats = {k: float(v) for k, v in stats.items()}
        state.soft_diag = rollout_router_soft_diagnostics(
            logits.detach(), latent_k=int(self.hparams.latent_k)
        )
        return state

    def conditional_component(
        self,
        *,
        strategy_entropy: torch.Tensor,
        resample: torch.Tensor,
        latent_lam_h: float,
        h_mode: str,
        h_goal: str,
        apply_entropy_loss: bool,
        zero_scalar: torch.Tensor,
    ) -> tuple[LossComponent, dict[str, float]]:
        if h_mode == "marginal":
            return (
                LossComponent(
                    name="conditional_entropy",
                    scaled_loss=zero_scalar,
                    raw_value=zero_scalar,
                    active=False,
                ),
                {"strategy_marginal_entropy_nats": 0.0, "strategy_marginal_entropy_kl": 0.0},
            )
        from rl.latent_losses import strategy_entropy_loss

        loss, stats = strategy_entropy_loss(
            strategy_entropy,
            resample,
            objective=h_goal,
            lam_h=latent_lam_h,
            device=self.device,
        )
        if not apply_entropy_loss:
            loss = torch.zeros_like(loss)
        return (
            LossComponent(
                name="conditional_entropy",
                scaled_loss=loss,
                raw_value=strategy_entropy.mean().detach(),
                active=bool(apply_entropy_loss),
                metrics=stats,
            ),
            stats,
        )

    def marginal_minibatch_component(
        self,
        epoch_state: RolloutEntropyState,
        *,
        mb_idx: int,
        apply_entropy_loss: bool,
        apply_rollout_marginal: bool,
        zero_scalar: torch.Tensor,
    ) -> LossComponent | None:
        if (
            mb_idx != 0
            or not apply_entropy_loss
            or not apply_rollout_marginal
            or epoch_state.loss_for_epoch is None
            or epoch_state.consumed
        ):
            return None
        epoch_state.consumed = True
        return LossComponent(
            name="rollout_marginal_entropy",
            scaled_loss=epoch_state.loss_for_epoch,
            raw_value=epoch_state.loss_for_epoch.detach(),
            active=True,
        )
