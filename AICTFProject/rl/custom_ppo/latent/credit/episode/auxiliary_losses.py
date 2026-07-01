"""Episode-router auxiliary loss assembly."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from rl.custom_ppo.latent.context_buckets import specialist_context_keys_for_mode
from rl.custom_ppo.latent.credit.episode.awrd_targets import AwrdTargets
from rl.custom_ppo.latent.credit.episode.preference_targets import PreferenceTargets
from rl.custom_ppo.latent.credit.episode.refresh_targets import RefreshTargets
from rl.custom_ppo.latent.preferences import router_specialist_coef_scale as _router_specialist_coef_scale
from rl.custom_ppo.latent.preferences import router_specialist_loss as _router_specialist_loss
from rl.custom_ppo.latent.types import EpisodeAuxiliaryLossBundle, EpisodeRouterBatch, LossComponent


@dataclass
class EpisodeAuxiliaryContext:
    trainer: Any
    host: Any
    batch: EpisodeRouterBatch
    latent_lam_h: float
    preference: PreferenceTargets
    awrd: AwrdTargets
    refresh: RefreshTargets
    specialist_enabled: bool
    specialist_scale: float
    specialist_conditional_start: float
    specialist_context_keys: torch.Tensor | None
    usage_coef: float


def _zero(device: torch.device) -> LossComponent:
    z = torch.zeros((), dtype=torch.float32, device=device)
    return LossComponent(raw=z, scaled=z, active_fraction=0.0)


def build_episode_auxiliary_context(
    *,
    trainer: Any,
    host: Any,
    batch: EpisodeRouterBatch,
    latent_lam_h: float,
    preference: PreferenceTargets,
    awrd: AwrdTargets,
    refresh: RefreshTargets,
) -> EpisodeAuxiliaryContext:
    specialist_enabled = bool(getattr(trainer, "latent_specialist_router_enabled", False)) and not bool(
        getattr(trainer, "latent_specialist_use_rollout_states", False)
    )
    specialist_warmup_steps = int(getattr(trainer, "latent_specialist_warmup_steps", 0) or 0)
    specialist_scale = _router_specialist_coef_scale(
        global_step=int(getattr(trainer, "global_step", 0) or 0),
        warmup_steps=specialist_warmup_steps,
        ramp_steps=int(getattr(trainer, "latent_specialist_ramp_steps", 1) or 0),
    )
    specialist_conditional_start = (
        float(getattr(trainer, "latent_conditional_entropy_min_coef_start", 0.0) or 0.0)
        if int(getattr(trainer, "global_step", 0) or 0) >= specialist_warmup_steps
        else 0.0
    )
    specialist_context_keys = specialist_context_keys_for_mode(
        mode=str(getattr(trainer, "latent_specialist_context_key_mode", "opponent_bucket") or "opponent_bucket"),
        states=batch.states,
        opponent_ids=batch.opponent_ids,
        bucket_ids=batch.bucket_ids,
    )
    usage_coef = max(0.0, float(getattr(trainer, "latent_usage_balance_coef", 0.0) or 0.0))
    from rl.custom_ppo.v6i1_phase_runtime import is_v6i1_staged_trainer, resolve_v6i1_rollout_usage_coef

    if is_v6i1_staged_trainer(trainer):
        usage_coef = float(resolve_v6i1_rollout_usage_coef(trainer))
    return EpisodeAuxiliaryContext(
        trainer=trainer,
        host=host,
        batch=batch,
        latent_lam_h=latent_lam_h,
        preference=preference,
        awrd=awrd,
        refresh=refresh,
        specialist_enabled=specialist_enabled,
        specialist_scale=specialist_scale,
        specialist_conditional_start=specialist_conditional_start,
        specialist_context_keys=specialist_context_keys,
        usage_coef=usage_coef,
    )


def compute_episode_auxiliary_losses(
    logits: torch.Tensor,
    *,
    ctx: EpisodeAuxiliaryContext,
    epoch_index: int,
) -> tuple[EpisodeAuxiliaryLossBundle, dict[str, torch.Tensor]]:
    del epoch_index
    trainer = ctx.trainer
    device = trainer.device
    dist = Categorical(logits=logits)
    z_entropy = dist.entropy().mean()
    h_goal = str(getattr(trainer.cfg, "latent_entropy_objective", "maximize") or "maximize").lower()
    if h_goal == "none" or ctx.latent_lam_h <= 0.0:
        entropy_scaled = torch.zeros((), dtype=torch.float32, device=device)
    elif h_goal == "minimize":
        entropy_scaled = float(ctx.latent_lam_h) * z_entropy
    else:
        entropy_scaled = -float(ctx.latent_lam_h) * z_entropy
    entropy = LossComponent(raw=z_entropy, scaled=entropy_scaled, active_fraction=1.0)

    if ctx.usage_coef > 0.0 and logits.shape[0] > 0:
        p_bar = torch.softmax(logits, dim=-1).mean(dim=0).clamp_min(1e-8)
        usage_kl = (p_bar * (torch.log(p_bar) + torch.log(p_bar.new_tensor(float(trainer.latent_k))))).sum()
        usage_scaled = ctx.usage_coef * usage_kl
    else:
        usage_kl = torch.zeros((), dtype=torch.float32, device=device)
        usage_scaled = usage_kl
    usage_balance = LossComponent(raw=usage_kl, scaled=usage_scaled, active_fraction=float(logits.shape[0] > 0))

    specialist_stats: dict[str, torch.Tensor] = {}
    if ctx.specialist_enabled:
        specialist_raw, specialist_stats = _router_specialist_loss(
            logits,
            context_keys=ctx.specialist_context_keys,
            latent_k=int(trainer.latent_k),
            marginal_balance_coef=float(getattr(trainer, "latent_marginal_balance_coef", 0.0) or 0.0),
            conditional_entropy_min_coef=float(
                getattr(trainer, "latent_conditional_entropy_min_coef", 0.0) or 0.0
            ),
            conditional_entropy_min_coef_start=ctx.specialist_conditional_start,
            conditional_entropy_scope=str(
                getattr(trainer, "latent_specialist_conditional_entropy_scope", "state") or "state"
            ),
            context_mi_coef=float(getattr(trainer, "latent_context_mi_coef", 0.0) or 0.0),
            coef_scale=ctx.specialist_scale,
            min_bucket_count=int(getattr(trainer, "latent_specialist_min_bucket_count", 2) or 2),
        )
        specialist = LossComponent(raw=specialist_raw, scaled=specialist_raw, active_fraction=1.0)
    else:
        specialist = _zero(device)

    pref = ctx.preference
    if pref.coef > 0.0 and bool(pref.mask.any().item()):
        valid_logits = logits[pref.mask]
        valid_targets = pref.target_probs[pref.mask]
        log_probs = torch.log_softmax(valid_logits, dim=-1)
        target_probs_clamped = valid_targets.clamp_min(1e-8)
        target_entropy_eps = -(valid_targets * torch.log(target_probs_clamped)).sum(dim=-1)
        target_confidence = (1.0 - target_entropy_eps / math.log(trainer.latent_k)).clamp(0.0, 1.0)
        confidence_scale = float(getattr(trainer, "latent_preference_confidence_scale", 2.0) or 2.0)
        commit_coef = float(getattr(trainer, "latent_preference_commit_coef", 0.0) or 0.0)
        effective_coef_eps = pref.coef * (1.0 + confidence_scale * target_confidence)
        kl_per_episode = F.kl_div(log_probs, valid_targets, reduction="none").sum(dim=-1)
        opponent_balanced = bool(
            getattr(trainer.cfg, "latent_preference_opponent_balanced", False)
        )
        if opponent_balanced:
            valid_opps = ctx.batch.opponent_ids[pref.mask]
            unique_opps = torch.unique(valid_opps).detach().cpu().tolist()
            opponent_losses = []
            opponent_weighted_losses = []
            for opp_id in unique_opps:
                opp_mask = valid_opps == opp_id
                if bool(opp_mask.any().item()):
                    opponent_losses.append(kl_per_episode[opp_mask].mean())
                    opponent_weighted_losses.append((effective_coef_eps[opp_mask] * kl_per_episode[opp_mask]).mean())
            pref_raw = torch.stack(opponent_losses).mean() if opponent_losses else kl_per_episode.mean()
            pref_scaled = (
                torch.stack(opponent_weighted_losses).mean()
                if opponent_weighted_losses
                else (effective_coef_eps * kl_per_episode).mean()
            )
        else:
            pref_raw = kl_per_episode.mean()
            pref_scaled = (effective_coef_eps * kl_per_episode).mean()

        commit_type = str(getattr(trainer.cfg, "commitment_type", "confidence_weighted_entropy") or "confidence_weighted_entropy")
        if commit_type == "confidence_weighted_entropy" and commit_coef > 0.0:
            valid_q_probs = torch.softmax(valid_logits, dim=-1)
            q_entropy_eps = -(valid_q_probs * torch.log(valid_q_probs + 1e-8)).sum(dim=-1)
            commit_loss_eps = target_confidence * q_entropy_eps
            if opponent_balanced:
                valid_opps = ctx.batch.opponent_ids[pref.mask]
                unique_opps = torch.unique(valid_opps).detach().cpu().tolist()
                opponent_commit_losses = []
                for opp_id in unique_opps:
                    opp_mask = valid_opps == opp_id
                    if bool(opp_mask.any().item()):
                        opponent_commit_losses.append(commit_loss_eps[opp_mask].mean())
                commit_raw = (
                    torch.stack(opponent_commit_losses).mean()
                    if opponent_commit_losses
                    else commit_loss_eps.mean()
                )
            else:
                commit_raw = commit_loss_eps.mean()
            commit_scaled = commit_coef * commit_raw
        else:
            commit_raw = torch.zeros((), dtype=torch.float32, device=device)
            commit_scaled = commit_raw
        preference = LossComponent(
            raw=pref_raw,
            scaled=pref_scaled,
            active_fraction=float(pref.mask.float().mean().item()),
        )
        commitment = LossComponent(raw=commit_raw, scaled=commit_scaled, active_fraction=float(pref.mask.float().mean().item()))
    else:
        preference = _zero(device)
        commitment = _zero(device)

    awrd = ctx.awrd
    if awrd.coef > 0.0 and bool(awrd.mask.any().item()):
        awrd_logits = logits[awrd.mask]
        awrd_targets = awrd.target_probs[awrd.mask]
        awrd_log_probs = torch.log_softmax(awrd_logits, dim=-1)
        awrd_kl = F.kl_div(awrd_log_probs, awrd_targets, reduction="none").sum(dim=-1)
        awrd_raw = awrd_kl.mean()
        if awrd.soft_margin:
            valid_coefs = awrd.per_sample_coefs[awrd.mask]
            awrd_scaled = (valid_coefs * awrd_kl).mean()
        else:
            awrd_scale = float(getattr(trainer, "latent_awrd_margin_scale", 2.0) or 2.0)
            active_count = max(1, int(awrd.mask.sum().item()))
            margin_mean = float(awrd.margin_sum / active_count)
            awrd_scaled = awrd.coef * (1.0 + awrd_scale * margin_mean) * awrd_raw
        awrd_comp = LossComponent(
            raw=awrd_raw,
            scaled=awrd_scaled,
            active_fraction=float(awrd.mask.float().mean().item()),
        )
    else:
        awrd_comp = _zero(device)

    refresh = ctx.refresh
    if (
        refresh.active
        and refresh.valid
        and refresh.refresh_states is not None
        and refresh.mask is not None
        and refresh.target_probs is not None
        and bool(refresh.mask.any().item())
    ):
        v3i3_logits = trainer.model.strategy_logits(
            refresh.refresh_states,
            selector_hidden=refresh.refresh_hidden,
        )
        valid_logits_v3i3 = v3i3_logits[refresh.mask]
        valid_targets_v3i3 = refresh.target_probs[refresh.mask]
        v3i3_log_probs = torch.log_softmax(valid_logits_v3i3, dim=-1)
        v3i3_kl = F.kl_div(v3i3_log_probs, valid_targets_v3i3, reduction="none").sum(dim=-1)
        refresh_raw = v3i3_kl.mean()
        refresh_scaled = refresh.coef * refresh_raw
        refresh_comp = LossComponent(
            raw=refresh_raw,
            scaled=refresh_scaled,
            active_fraction=float(refresh.mask.float().mean().item()),
        )
    else:
        refresh_comp = _zero(device)

    bundle = EpisodeAuxiliaryLossBundle(
        entropy=entropy,
        usage_balance=usage_balance,
        specialist=specialist,
        preference=preference,
        commitment=commitment,
        awrd=awrd_comp,
        refresh_preference=refresh_comp,
    )
    return bundle, specialist_stats


def make_auxiliary_loss_fn(
    ctx: EpisodeAuxiliaryContext,
) -> Callable[[torch.Tensor, int], tuple[EpisodeAuxiliaryLossBundle, dict[str, torch.Tensor]]]:
    def _fn(logits: torch.Tensor, epoch_index: int) -> tuple[EpisodeAuxiliaryLossBundle, dict[str, torch.Tensor]]:
        return compute_episode_auxiliary_losses(logits, ctx=ctx, epoch_index=epoch_index)

    return _fn
