"""PPO update loop for :class:`~rl.custom_ppo.trainer.CustomPPOTrainer`.

This module owns the per-rollout PPO optimisation: LR / entropy / lambda_H
schedules, the epoch + minibatch loop, all per-batch losses (PPO clip,
value, entropy, latent strategy entropy / persistence / KL-consecutive /
phase-aux / strategy-PPO / aux-return), the optimizer step with grad
clip, KL early-stop, and the final stats aggregation (including the
post-update diagnostics: episode-strategy PPO, strategy experience CSV,
advantage / behavior / option-advantage diagnostics, return-norm stats).

Refactored out of ``CustomPPOTrainer.update`` as part of PR-7; the trainer
now delegates via ``self.updater.update(buffer, total_timesteps=...)``.
The class holds a ``trainer`` context object so the migration stayed
mechanical — heavy thinning of dependencies can come in a follow-up.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn.functional as F

from rl.latent_losses import (
    rollout_marginal_entropy_loss as _latent_rollout_marginal_entropy_loss,
    rollout_router_soft_diagnostics as _latent_rollout_router_soft_diagnostics,
    strategy_aux_return_loss as _latent_strategy_aux_return_loss,
    strategy_entropy_loss as _latent_strategy_entropy_loss,
    strategy_kl_consecutive_loss as _latent_strategy_kl_consecutive_loss,
    strategy_marginal_entropy_loss as _latent_strategy_marginal_entropy_loss,
    strategy_persistence_loss as _latent_strategy_persistence_loss,
    strategy_phase_aux_loss as _latent_strategy_phase_aux_loss,
    strategy_ppo_loss as _latent_strategy_ppo_loss,
)
from rl.ppo_core import TensorDictRolloutBuffer, ppo_policy_loss, ppo_value_loss

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
from rl.custom_ppo.return_normalization import (
    _normalize_strategy_returns,
    _normalize_value_targets,
    _update_strategy_return_stats,
)
from rl.custom_ppo.schedules import resolve_latent_forced_z_frac, resolve_latent_lam_h
from rl.custom_ppo.trainer_config import TrainerHyperparams

if TYPE_CHECKING:
    from rl.custom_ppo.latent_strategy_state import LatentStrategyState
    from rl.custom_ppo.trainer import CustomPPOTrainer


_VALID_CURRICULUM_PHASES = frozenset({"A", "B", "C"})


class StrictFaithfulDictWrapper(dict):
    """Block privileged obs keys from the forced-z actor separation path only.

    ``phase_id`` is excluded here so separation cannot read deployment-hidden
    labels through the actor observation dict. Phase-supervised router training
    (``latent_strategy_aux_predict_phase_coef``) is a separate, explicitly
    configured ablation that reads ``batch["phase_id"]`` directly.
    """
    disallowed_keys = {
        "opponent_id", "phase_id", "phase", "outcome_id", "role_bucket_id",
        "spread_bucket_id", "pressure_bucket_id", "attack_defense_ratio_bucket_id",
        "role_bucket", "spread_bucket", "pressure_bucket", "attack_defense_ratio_bucket",
        "opponent", "outcome"
    }

    def __getitem__(self, key):
        if key in self.disallowed_keys:
            raise AssertionError(f"Leakage detected! Disallowed key '{key}' accessed inside _policy_z_separation_loss.")
        return super().__getitem__(key)

    def get(self, key, default=None):
        if key in self.disallowed_keys:
            raise AssertionError(f"Leakage detected! Disallowed key '{key}' accessed inside _policy_z_separation_loss.")
        return super().get(key, default)


def _populate_main_loop_qphi_telemetry(
    row: dict[str, float],
    *,
    cfg: Any,
    hparams: Any,
    runtime: Any,
    latent_lam_h: float,
) -> None:
    """Expose main-loop q_phi activity in the generic q_phi smoke fields.

    The original ``q_phi_grad_norm`` / ``latent_q_phi_train_active`` columns
    were populated by episode-credit / arc-credit extension paths. In the
    paper-faithful v5 path, q_phi is trained in the main PPO backward pass, so
    the corresponding gradient norm is ``strategy_grad_norm``.
    """
    active = (
        bool(getattr(hparams, "use_latent_strategy", False))
        and not bool(getattr(hparams, "fixed_latent_strategy", False))
        and getattr(runtime, "latent_router_optimizer", None) is None
        and (
            float(getattr(hparams, "latent_strategy_ppo_coef", 0.0) or 0.0) > 0.0
            or float(getattr(hparams, "latent_lam_p", 0.0) or 0.0) > 0.0
            or float(latent_lam_h or 0.0) > 0.0
            or float(getattr(hparams, "latent_kl_consecutive", 0.0) or 0.0) > 0.0
        )
    )
    grad = float(row.get("strategy_grad_norm", 0.0) or 0.0)
    row["main_loop_q_phi_train_active"] = 1.0 if active else 0.0
    row["main_loop_q_phi_grad_norm"] = grad
    if active and float(row.get("latent_q_phi_train_active", 0.0) or 0.0) <= 0.0:
        row["latent_q_phi_train_active"] = 1.0
    if grad > 0.0 and float(row.get("q_phi_grad_norm", 0.0) or 0.0) <= 0.0:
        row["q_phi_grad_norm"] = grad
        row["q_phi_strategy_encoder_grad_norm"] = grad


def _warmup_ramp_value(
    *,
    global_step: int,
    warmup_steps: int,
    ramp_steps: int,
    start_value: float,
    target_value: float,
) -> float:
    """Return zero before warmup, then linearly ramp start to target."""
    step = int(global_step)
    warmup = max(0, int(warmup_steps))
    ramp = max(0, int(ramp_steps))
    start = max(0.0, float(start_value))
    target = max(0.0, float(target_value))
    if step < warmup:
        return 0.0
    if ramp <= 0:
        return target
    progress = max(0.0, min(1.0, (step - warmup) / float(ramp)))
    return start + progress * (target - start)


def _z_separation_gate_mask(
    *,
    advantages: torch.Tensor,
    action_entropy: torch.Tensor,
    global_state: torch.Tensor,
    max_action_entropy: float,
    min_abs_advantage: float,
    min_decision_frac: float,
    max_entropy_frac: float,
) -> torch.Tensor:
    """Select tactically meaningful rows for forced-z policy separation."""
    adv = advantages.detach().float().reshape(-1)
    entropy = action_entropy.detach().float().reshape(-1)
    if global_state.dim() != 2 or int(global_state.shape[0]) != int(adv.shape[0]):
        raise ValueError(
            "global_state must be (B, D) and align with advantages for z separation"
        )
    if int(entropy.shape[0]) != int(adv.shape[0]):
        raise ValueError("action_entropy must align with advantages for z separation")

    mask = torch.ones_like(adv, dtype=torch.bool)
    min_adv = max(0.0, float(min_abs_advantage))
    if min_adv > 0.0:
        mask &= adv.abs() >= min_adv

    min_progress = max(0.0, min(1.0, float(min_decision_frac)))
    if min_progress > 0.0:
        if int(global_state.shape[1]) <= 17:
            raise ValueError(
                "global_state needs decision_frac at index 17 for z separation gating"
            )
        mask &= global_state[:, 17].detach().float() >= min_progress

    entropy_frac = max(0.0, min(1.0, float(max_entropy_frac)))
    if entropy_frac < 1.0:
        entropy_ceiling = max(0.0, float(max_action_entropy)) * entropy_frac
        mask &= entropy <= entropy_ceiling
    return mask


def _extract_rollout_resample_subset(
    buffer: TensorDictRolloutBuffer,
    *,
    require_selector_hidden: bool,
) -> tuple[torch.Tensor | None, torch.Tensor | None, str | None]:
    """Return rollout resample rows for the marginal-entropy objective."""
    t_full = int(buffer.pos)
    if t_full <= 0:
        return None, None, "empty_rollout"
    if "global_state" not in buffer.fields:
        raise KeyError("rollout marginal entropy requires global_state in buffer.fields")
    if "z_resampled" not in buffer.fields:
        raise KeyError("rollout marginal entropy requires z_resampled in buffer.fields")

    gs_full = buffer.fields["global_state"][:t_full]
    gs_full = gs_full.reshape(t_full * buffer.n_envs, *gs_full.shape[2:])
    rs_full = buffer.fields["z_resampled"][:t_full].reshape(-1).bool()
    if rs_full.numel() != gs_full.shape[0]:
        raise ValueError(
            f"z_resampled length {rs_full.numel()} != global_state rows {gs_full.shape[0]}"
        )
    if not bool(rs_full.any()):
        return None, None, "no_resample_rows"

    states = gs_full[rs_full]
    hidden: torch.Tensor | None = None
    if require_selector_hidden:
        if "selector_hidden" not in buffer.fields:
            raise KeyError(
                "recurrent selector rollout marginal entropy requires "
                "selector_hidden in buffer.fields"
            )
        sh_full = buffer.fields["selector_hidden"][:t_full]
        sh_full = sh_full.reshape(t_full * buffer.n_envs, *sh_full.shape[2:])
        if sh_full.shape[0] != gs_full.shape[0]:
            raise ValueError("selector_hidden rows misaligned with global_state")
        hidden = sh_full[rs_full]
    return states, hidden, None


def _clip_global_grad_norm(model: torch.nn.Module, max_norm: float) -> float:
    """Clip all parameters that currently have gradients (true global cap)."""
    params = [p for p in model.parameters() if p.grad is not None]
    if not params:
        return 0.0
    return float(torch.nn.utils.clip_grad_norm_(params, float(max_norm)))


def _assert_finite_loss(total_loss: torch.Tensor, *, epoch_idx: int, mb_idx: int) -> None:
    if not torch.isfinite(total_loss).all():
        raise FloatingPointError(
            f"Non-finite PPO loss at epoch={epoch_idx}, minibatch={mb_idx}"
        )


def _assert_finite_gradients(model: torch.nn.Module, *, epoch_idx: int, mb_idx: int) -> None:
    for name, param in model.named_parameters():
        if param.grad is not None and not torch.isfinite(param.grad).all():
            raise FloatingPointError(
                f"Non-finite gradient in {name} at epoch={epoch_idx}, minibatch={mb_idx}"
            )


def _policy_z_separation_loss(
    model: Any,
    obs_batch: dict[str, torch.Tensor],
    z_idx: torch.Tensor,
    *,
    latent_k: int,
    margin: float,
    active_mask: torch.Tensor | None = None,
    subsample_generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Penalize collapsed z policies via per-pair JSD hinge (hinge before averaging)."""
    if int(latent_k) <= 1 or z_idx.numel() <= 0:
        zero = torch.zeros((), dtype=torch.float32, device=z_idx.device)
        return zero, {"jsd": zero, "active": zero}

    # Wrap obs_batch with StrictFaithfulDictWrapper to guarantee zero information leakage
    obs_batch = StrictFaithfulDictWrapper(obs_batch)

    batch_size = int(z_idx.reshape(-1).shape[0])
    device = z_idx.device

    active_fraction = torch.ones((), dtype=torch.float32, device=device)
    if active_mask is not None:
        mask = active_mask.to(device=device, dtype=torch.bool).reshape(-1)
        if int(mask.shape[0]) != batch_size:
            raise ValueError(
                f"active_mask length {int(mask.shape[0])} != batch size {batch_size}"
            )
        active_fraction = mask.float().mean()
        active_indices = torch.where(mask)[0]
        if active_indices.numel() <= 0:
            zero = torch.zeros((), dtype=torch.float32, device=device)
            return zero, {"jsd": zero, "active": zero}
        obs_active: dict[str, Any] = {}
        for key, value in obs_batch.items():
            if isinstance(value, torch.Tensor) and int(value.shape[0]) == batch_size:
                obs_active[key] = value.index_select(0, active_indices)
            else:
                obs_active[key] = value
        obs_batch = StrictFaithfulDictWrapper(obs_active)
        batch_size = int(active_indices.numel())

    # Cap batch size for JSD computation to avoid CUDA memory pressure/OOM
    max_jsd_rows = 512
    if batch_size > max_jsd_rows:
        indices = torch.randperm(
            batch_size, device=device, generator=subsample_generator
        )[:max_jsd_rows]
        obs_sub = {}
        for k, v in obs_batch.items():
            if isinstance(v, torch.Tensor) and v.shape[0] == batch_size:
                obs_sub[k] = v[indices]
            else:
                obs_sub[k] = v
        obs_sub = StrictFaithfulDictWrapper(obs_sub)
        curr_batch_size = max_jsd_rows
    else:
        obs_sub = obs_batch
        curr_batch_size = int(batch_size)

    # Compute logits for all latent strategy classes
    logits_list = []
    for k in range(latent_k):
        z_k = torch.full((curr_batch_size,), k, dtype=torch.long, device=device)
        logits_k = model._mask_logits(model.policy_logits(obs_sub, z_idx=z_k), obs_sub.get("mask"))
        logits_list.append(logits_k)

    all_pairwise_js: list[torch.Tensor] = []
    offset = 0
    for _agent_idx in range(int(model.n_agents)):
        for dim in model.per_agent_action_dims:
            width = int(dim)

            # Extract probability distributions for all K latent classes
            p_list = []
            for k in range(latent_k):
                a_k = logits_list[k][:, offset : offset + width]
                p_k = torch.softmax(a_k, dim=-1).clamp_min(1e-8)
                p_list.append(p_k)

            # Compute all pairwise JS divergences via broadcasting over (K, K).
            p_stacked = torch.stack(p_list, dim=0)  # (K, batch, width)
            p_i = p_stacked.unsqueeze(1)
            p_j = p_stacked.unsqueeze(0)
            m = 0.5 * (p_i + p_j)
            kl_i = (p_i * (p_i.log() - m.log())).sum(dim=-1)
            kl_j = (p_j * (p_j.log() - m.log())).sum(dim=-1)
            js_matrix = 0.5 * kl_i + 0.5 * kl_j
            pair_i, pair_j = torch.triu_indices(int(latent_k), int(latent_k), offset=1, device=device)
            pairwise_js = js_matrix[pair_i, pair_j]
            if pairwise_js.numel() > 0:
                all_pairwise_js.append(pairwise_js)
            offset += width

    if not all_pairwise_js:
        zero = torch.zeros((), dtype=torch.float32, device=z_idx.device)
        return zero, {"jsd": zero, "min_jsd": zero, "active": zero}

    stacked = torch.cat(all_pairwise_js, dim=0)
    margin_t = stacked.new_tensor(float(max(0.0, margin)))
    per_pair_penalty = F.relu(margin_t - stacked)
    loss = per_pair_penalty.mean()
    jsd = stacked.mean()
    return loss, {
        "jsd": jsd.detach(),
        "min_jsd": stacked.min().detach(),
        "active": active_fraction.detach(),
    }


def set_model_requires_grad_for_phase(model: Any, phase: str) -> None:
    if phase not in _VALID_CURRICULUM_PHASES:
        raise ValueError(
            f"Unknown training phase {phase!r}; expected one of {sorted(_VALID_CURRICULUM_PHASES)}"
        )
    # Actor parameters: actor_cnn and latent_actor
    # Critic parameters: critic and episode_strategy_value_head
    # Router parameters: strategy_encoder, strategy_aux_return_head, phase_predictor
    for name, p in model.named_parameters():
        is_actor = "actor_cnn" in name or "latent_actor" in name
        is_critic = "critic" in name and "episode_strategy_value_head" not in name
        is_router = (
            "strategy_encoder" in name
            or "strategy_aux_return_head" in name
            or "phase_predictor" in name
            or "selector_gru" in name
            or "episode_strategy_value_head" in name
        )
        
        if phase == "A":
            if is_actor:
                p.requires_grad = True
            elif is_critic:
                p.requires_grad = True
            elif is_router:
                p.requires_grad = False
        elif phase == "B":
            if is_actor:
                p.requires_grad = False
            elif is_critic:
                p.requires_grad = True
            elif is_router:
                p.requires_grad = True
        elif phase == "C":
            if is_actor:
                p.requires_grad = True
            elif is_critic:
                p.requires_grad = True
            elif is_router:
                p.requires_grad = True


class PPOUpdater:
    """PPO epoch/minibatch loop owner.

    Static collaborators (``model`` / ``optimizer`` / ``device`` / ``cfg`` /
    ``hparams`` / ``latent_state``) are injected explicitly. A thin
    ``runtime`` back-reference to the trainer is kept for the shared
    mutable state that hasn't been extracted into its own owner yet
    (``global_step`` read, ``last_stats`` write). Return-normalization
    stats live on ``runtime.return_norm`` and ``runtime.strategy_return_norm``
    sub-components, which the ``return_normalization`` helper shims
    consume via the trainer-shaped ``runtime`` arg.
    """

    def __init__(
        self,
        *,
        model: Any,
        optimizer: Any,
        device: Any,
        cfg: Any,
        hparams: TrainerHyperparams,
        latent_state: "LatentStrategyState",
        runtime: "CustomPPOTrainer",
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.cfg = cfg
        self.hparams = hparams
        self.latent_state = latent_state
        self.runtime = runtime
        self.cf_grad_ratio_violations = 0
        seed = int(getattr(cfg, "seed", 0) or 0) + 31_337
        self._z_separation_generator = torch.Generator(device=device)
        self._z_separation_generator.manual_seed(seed)

    def compute_latent_lam_h(self, global_step: float, total_timesteps: int) -> float:
        """Return the current latent entropy coefficient for this rollout."""
        return resolve_latent_lam_h(self.cfg, global_step=global_step, total_timesteps=total_timesteps)

    def update(
        self,
        buffer: TensorDictRolloutBuffer,
        *,
        total_timesteps: int,
    ) -> dict[str, float]:
        """Run PPO epochs over one rollout and return the aggregated stats."""
        runtime = self.runtime
        hparams = self.hparams
        cfg = self.cfg
        device = self.device
        progress_remaining = max(
            0.0, 1.0 - float(runtime.global_step) / max(1.0, float(total_timesteps))
        )
        lr_floor_frac = max(
            0.0, min(float(getattr(cfg, "lr_floor_frac", 0.1) or 0.0), 1.0)
        )
        lr = hparams.learning_rate * max(progress_remaining, lr_floor_frac)
        from rl.custom_ppo.v6i1_phase_runtime import (
            apply_v6i1_learning_rates,
            is_v6i1_staged_trainer,
            resolve_v6i1_cf_coef_current,
            resolve_v6i1_episode_forced_frac,
            resolve_v6i1_rollout_usage_coef,
            step_v6i1_optimizers,
            v6i1_macro_router_active,
            v6i1_schedule_context,
        )

        v6i1_lr_stats: dict[str, float] = {}
        if getattr(runtime, "v6i1_three_optimizer_mode", False):
            v6i1_lr_stats = apply_v6i1_learning_rates(
                runtime, base_lr=float(hparams.learning_rate), progress_remaining=progress_remaining
            )
            lr = float(v6i1_lr_stats.get("actor_lr", lr))
        else:
            for group in self.optimizer.param_groups:
                group["lr"] = lr
        ent_coef = hparams.ent_coef if progress_remaining > 0.75 else 0.5 * hparams.ent_coef
        latent_lam_h = self.compute_latent_lam_h(runtime.global_step, total_timesteps)

        # Dynamic scheduling for z_separation_coef
        sep_coef = float(getattr(hparams, "latent_actor_z_separation_coef", 0.0) or 0.0)
        sep_start_coef = float(
            getattr(hparams, "latent_actor_z_separation_start_coef", 0.0) or 0.0
        )
        sep_warmup = int(getattr(hparams, "latent_actor_z_separation_warmup_steps", 0) or 0)
        sep_ramp = int(getattr(hparams, "latent_actor_z_separation_ramp_steps", 0) or 0)
        step = int(runtime.global_step)
        curriculum = getattr(runtime, "v6i1_curriculum", None)
        phase_key = "__legacy__"
        if curriculum is not None:
            phase_key = str(curriculum.resolve_phase(step))
            set_model_requires_grad_for_phase(self.model, phase_key)
        if is_v6i1_staged_trainer(runtime):
            curr_sep_coef = float(resolve_v6i1_cf_coef_current(runtime))
        else:
            curr_sep_coef = _warmup_ramp_value(
                global_step=step,
                warmup_steps=sep_warmup,
                ramp_steps=sep_ramp,
                start_value=sep_start_coef,
                target_value=sep_coef,
            )

        # Dynamic scheduling for z_adapter_scale
        if getattr(hparams, "use_latent_strategy", False) and getattr(hparams, "latent_actor_z_adapter_enabled", False):
            adapter_warmup = int(getattr(hparams, "latent_actor_z_adapter_warmup_steps", 0) or 0)
            adapter_ramp = int(getattr(hparams, "latent_actor_z_adapter_ramp_steps", 0) or 0)
            base_scale = float(getattr(hparams, "latent_actor_z_adapter_scale", 0.0) or 0.0)
            curr_adapter_scale = _warmup_ramp_value(
                global_step=step,
                warmup_steps=adapter_warmup,
                ramp_steps=adapter_ramp,
                start_value=0.0,
                target_value=base_scale,
            )

            # Apply dynamically to the model
            if hasattr(self.model, "latent_actor") and self.model.latent_actor is not None:
                self.model.latent_actor.z_adapter_scale = curr_adapter_scale
        else:
            curr_adapter_scale = 0.0

        _update_strategy_return_stats(runtime, buffer)

        zero_scalar = torch.zeros((), dtype=torch.float32, device=device)

        stats: dict[str, list[float]] = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "approx_kl": [],
            "clip_fraction": [],
            "grad_norm": [],
            "strategy_entropy": [],
            "strategy_policy_loss": [],
            "strategy_approx_kl": [],
            "strategy_clip_fraction": [],
            "strategy_ratio_std": [],
            "strategy_aux_return_loss": [],
            "strategy_persist_loss": [],
            "strategy_grad_norm": [],
            "strategy_resample_fraction": [],
            "strategy_kl": [],
            "strategy_phase_loss": [],
            "strategy_marginal_entropy_loss": [],
            "strategy_marginal_entropy_nats": [],
            "strategy_marginal_entropy_kl": [],
            "router_rollout_soft_marginal_entropy_nats": [],
            "router_rollout_soft_conditional_entropy_nats": [],
            "router_rollout_soft_mi_proxy_nats": [],
            "router_rollout_soft_argmax_occupancy_max": [],
            "router_rollout_soft_argmax_occupancy_min": [],
            "router_rollout_soft_argmax_occupancy_ratio": [],
            "router_rollout_resample_count": [],
            "latent_actor_z_separation_loss": [],
            "latent_actor_z_separation_jsd": [],
            "latent_actor_z_separation_active": [],
            "latent_actor_z_separation_train_active": [],
        }
        for _k_idx in range(int(hparams.latent_k)):
            stats[f"router_rollout_soft_p_bar_z{_k_idx}"] = []
        # Pre-resolved per-update entropy mode / coefficient. Constant across
        # the n_epochs * n_minibatches inner loop because it depends only on
        # cfg and the per-update lambda_H schedule. Used to gate the
        # rollout-level marginal entropy pass below.
        h_mode_global = str(
            getattr(cfg, "latent_entropy_mode", "conditional") or "conditional"
        ).lower()
        h_goal_global = str(
            getattr(cfg, "latent_entropy_objective", "maximize") or "maximize"
        ).lower()
        has_dedicated_router_opt_global = (
            getattr(runtime, "latent_router_optimizer", None) is not None
        )
        v6i1_usage_coef = (
            float(resolve_v6i1_rollout_usage_coef(runtime)) if is_v6i1_staged_trainer(runtime) else 0.0
        )
        apply_rollout_marginal = (
            hparams.use_latent_strategy
            and h_mode_global == "marginal"
            and not has_dedicated_router_opt_global
            and float(latent_lam_h or 0.0) > 0.0
            and h_goal_global != "none"
            and not bool(getattr(hparams, "fixed_latent_strategy", False))
        ) or (
            hparams.use_latent_strategy
            and v6i1_usage_coef > 0.0
            and has_dedicated_router_opt_global
            and not bool(getattr(hparams, "fixed_latent_strategy", False))
        )
        rollout_marginal_coef = float(latent_lam_h if v6i1_usage_coef <= 0.0 else v6i1_usage_coef)
        rollout_resample_states_full: torch.Tensor | None = None
        rollout_resample_hidden_full: torch.Tensor | None = None
        rollout_marginal_skip_reason: str | None = None
        if apply_rollout_marginal:
            require_hidden = bool(getattr(self.model, "use_recurrent_selector", False))
            rollout_resample_states_full, rollout_resample_hidden_full, rollout_marginal_skip_reason = (
                _extract_rollout_resample_subset(
                    buffer,
                    require_selector_hidden=require_hidden,
                )
            )
        # Most-recent rollout-level diagnostics from the current PPO update.
        # Computed once per epoch (when apply_rollout_marginal is True);
        # forward-filled into every minibatch row's CSV so analysis keeps a
        # single time-series even though the underlying quantity updates 6
        # times per PPO update.
        last_rollout_marginal_stats: dict[str, float] = {}
        last_rollout_soft_diag: dict[str, float] = {}
        stop_update = False
        target_kl = getattr(cfg, "target_kl", None)
        strategy_target_kl = getattr(cfg, "strategy_target_kl", None)
        if strategy_target_kl is None:
            strategy_target_kl = target_kl
        model = self.model
        for _epoch_idx in range(hparams.n_epochs):
            # Once-per-epoch rollout-level marginal-entropy forward+loss for v5i6.
            #
            # Why per-epoch and not per-minibatch:
            #   The minibatch path (strategy_marginal_entropy_loss) computes
            #   p_bar over ~16 resample samples per minibatch. KL is convex in
            #   p_bar, so by Jensen the per-minibatch loss is a strict upper
            #   bound on the rollout-level objective; the gap is closed by
            #   softening individual q(z|s) toward uniform — exactly the
            #   conditional-entropy regression v5i6 was designed to avoid.
            #   Aggregating over all rollout resample points before applying
            #   KL recovers the intended objective:
            #     L = lam_H * KL( N^{-1} sum_i q_phi(z|s_i) || U )
            #
            # Why per-epoch and not per-update:
            #   q_phi parameters drift across the n_epochs inner loop. Refresh
            #   the rollout-level p_bar at the start of each epoch so the
            #   gradient is computed against current parameters.
            rollout_marginal_loss_for_epoch: torch.Tensor | None = None
            if apply_rollout_marginal and rollout_resample_states_full is not None:
                if bool(getattr(model, "use_recurrent_selector", False)):
                    rollout_logits = model.strategy_logits(
                        rollout_resample_states_full,
                        selector_hidden=rollout_resample_hidden_full,
                    )
                else:
                    rollout_logits = model.strategy_logits(rollout_resample_states_full)
                rml, rml_stats = _latent_rollout_marginal_entropy_loss(
                    rollout_logits,
                    objective="maximize" if v6i1_usage_coef > 0.0 else h_goal_global,
                    lam_h=float(rollout_marginal_coef),
                    latent_k=int(hparams.latent_k),
                    device=device,
                )
                rollout_marginal_loss_for_epoch = rml
                last_rollout_marginal_stats = rml_stats
                last_rollout_soft_diag = _latent_rollout_router_soft_diagnostics(
                    rollout_logits.detach(),
                    latent_k=int(hparams.latent_k),
                )
            for _mb_idx, batch in enumerate(
                buffer.iter_minibatches(hparams.batch_size, shuffle=True)
            ):
                obs_batch = {
                    "grid": batch["obs_grid"],
                    "vec": batch["obs_vec"],
                    "agent_mask": batch["obs_agent_mask"],
                    "mask": batch["obs_mask"],
                }
                z_idx = batch["z"] if hparams.use_latent_strategy else None
                if (
                    hparams.use_latent_strategy
                    and bool(getattr(model, "use_recurrent_selector", False))
                ):
                    if "selector_hidden" not in batch:
                        raise KeyError(
                            "selector_hidden missing from rollout buffer; required for "
                            "recurrent strategy PPO replay"
                        )
                    selector_hidden = batch["selector_hidden"]
                else:
                    selector_hidden = None
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
                z_sep_loss = zero_scalar
                z_sep_stats = {
                    "jsd": zero_scalar,
                    "active": zero_scalar,
                }
                z_sep_train_active = 0.0

                if hparams.use_latent_strategy:
                    resample = batch["z_resampled"].bool()
                    persist_mask = batch["z_persist_mask"].bool()
                    log_prob = action_log_prob
                    strategy_log_prob = aux["strategy_log_prob"]
                    strategy_entropy = aux["strategy_entropy"]
                    h_mode = str(
                        getattr(cfg, "latent_entropy_mode", "conditional")
                        or "conditional"
                    ).lower()
                    h_goal = str(
                        getattr(cfg, "latent_entropy_objective", "maximize") or "maximize"
                    ).lower()
                    if h_mode == "marginal":
                        # v5i6 marginal entropy is computed once per inner
                        # epoch over the FULL rollout resample subset
                        # (rollout_marginal_loss_for_epoch above), not
                        # per-minibatch. Per-minibatch p_bar over ~N/64
                        # samples is upper-biased vs the rollout-level
                        # objective by Jensen and is not the loss we want
                        # to optimize. Suppress here; the true loss is
                        # added to the first minibatch's latent_loss
                        # below.
                        strategy_entropy_loss = zero_scalar
                        entropy_mode_stats = {
                            "strategy_marginal_entropy_nats": 0.0,
                            "strategy_marginal_entropy_kl": 0.0,
                        }
                    else:
                        strategy_entropy_loss, entropy_mode_stats = (
                            _latent_strategy_entropy_loss(
                                strategy_entropy,
                                resample,
                                objective=h_goal,
                                lam_h=latent_lam_h,
                                device=device,
                            )
                        )
                    persist_term_loss, persist_stats = _latent_strategy_persistence_loss(
                        aux["strategy_logits"],
                        batch["prev_z"],
                        persist_mask,
                        lam_p=float(getattr(cfg, "latent_lam_p", 0.0)),
                        device=device,
                    )
                    if (
                        hparams.latent_resample_every_n == 0
                        and not hparams.latent_resample_on_flag
                        and not hparams.latent_event_refresh_enabled
                        and not hparams.latent_sparse_tactical_refresh_enabled
                    ):
                        assert persist_stats["persist_term"] == 0.0, (
                            "L_persist must be exactly 0 when no mid-episode resampling "
                            "(latent_resample_every_n=0, on_flag off, no event refresh, "
                            "no sparse tactical refresh)"
                        )
                    # v5 decoupled q_phi gradient channels (replaces v3c "Fix 5"
                    # coef-zero gate). Each main-loop term fires when its own
                    # coefficient is > 0 AND no dedicated router optimizer exists.
                    # Reproducibility safeguard: when ``latent_router_optimizer``
                    # is set (v3c-style episode-credit chain), the strategy_encoder +
                    # value_head params are stepped by that optimizer in
                    # ``apply_episode_strategy_ppo`` / ``apply_arc_strategy_ppo``.
                    # Suppress the main-loop q_phi gradient routes in that regime
                    # to avoid double-stepping the same params from two optimizers
                    # per update (the "650x duplicate entropy push" Fix 5 prevented).
                    # When no dedicated router optimizer exists, the main-loop
                    # path is the only thing that can train q_phi via entropy +
                    # persistence + KL, so each fires on its own coefficient
                    # (matches docs/algorithm.md: L = L_PPO + lam_p*L_persist - lam_H*H).
                    has_dedicated_router_opt = (
                        getattr(runtime, "latent_router_optimizer", None) is not None
                    )
                    apply_main_loop_qphi_loss = (
                        float(hparams.latent_strategy_ppo_coef or 0.0) > 0.0
                        and not has_dedicated_router_opt
                        and not v6i1_macro_router_active(runtime)
                    )
                    apply_entropy_loss = (
                        hparams.use_latent_strategy
                        and (
                            (
                                not has_dedicated_router_opt
                                and float(latent_lam_h or 0.0) > 0.0
                            )
                            or float(v6i1_usage_coef) > 0.0
                        )
                        and str(
                            getattr(cfg, "latent_entropy_objective", "maximize")
                            or "maximize"
                        ).lower()
                        != "none"
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
                    if not apply_entropy_loss:
                        strategy_entropy_loss = torch.zeros_like(strategy_entropy_loss)
                    # v5i6 reports the rollout-level scalar loss, KL, and
                    # marginal entropy in the existing
                    # ``strategy_marginal_entropy_*`` columns so a single
                    # time series captures "what entropy term is currently
                    # active". Other modes / non-v5i6 presets continue to
                    # report 0 for these rows. The redundant
                    # ``router_rollout_soft_*`` columns (added separately)
                    # keep an unambiguous label for the same rollout-level
                    # quantities for downstream analysis.
                    if (
                        h_mode == "marginal"
                        and apply_entropy_loss
                        and apply_rollout_marginal
                        and last_rollout_marginal_stats
                    ):
                        strategy_marginal_entropy_loss_value = float(
                            last_rollout_marginal_stats.get("rollout_marginal_entropy_kl", 0.0)
                        ) * float(rollout_marginal_coef)
                        strategy_marginal_entropy_nats_value = float(
                            last_rollout_marginal_stats.get("rollout_marginal_entropy_nats", 0.0)
                        )
                        strategy_marginal_entropy_kl_value = float(
                            last_rollout_marginal_stats.get("rollout_marginal_entropy_kl", 0.0)
                        )
                    else:
                        strategy_marginal_entropy_loss_value = 0.0
                        strategy_marginal_entropy_nats_value = 0.0
                        strategy_marginal_entropy_kl_value = 0.0
                    if not apply_persistence_loss:
                        persist_term_loss = torch.zeros_like(persist_term_loss)
                    persist_loss_value = (
                        persist_stats["persist_term"]
                        if apply_persistence_loss
                        else 0.0
                    )
                    latent_loss = persist_term_loss + strategy_entropy_loss
                    # First minibatch of the inner epoch carries the
                    # rollout-level marginal entropy gradient. Subsequent
                    # minibatches in the same epoch contribute zero to the
                    # marginal term (its gradient was already realized).
                    if (
                        _mb_idx == 0
                        and apply_entropy_loss
                        and apply_rollout_marginal
                        and rollout_marginal_loss_for_epoch is not None
                    ):
                        latent_loss = latent_loss + rollout_marginal_loss_for_epoch
                    if hparams.latent_kl_consecutive > 0.0:
                        kl_loss, kl_stats = _latent_strategy_kl_consecutive_loss(
                            aux["strategy_logits"],
                            batch["z_logits_prev"],
                            batch["z_kl_prev_valid"],
                            coef=float(hparams.latent_kl_consecutive),
                        )
                        if not apply_kl_loss:
                            kl_loss = torch.zeros_like(kl_loss)
                        latent_loss = latent_loss + kl_loss
                        stats["strategy_kl"].append(kl_stats["kl_mean"])
                    else:
                        stats["strategy_kl"].append(0.0)
                    if hparams.latent_strategy_aux_predict_phase_coef > 0.0:
                        # Explicit ablation: privileged phase labels supervise q_phi.
                        phase_logits = model.phase_logits_from_strategy_logits(
                            aux["strategy_logits"]
                        )
                        phase_loss_scaled, phase_stats = _latent_strategy_phase_aux_loss(
                            phase_logits,
                            batch["phase_id"],
                            coef=float(hparams.latent_strategy_aux_predict_phase_coef),
                        )
                        if has_dedicated_router_opt:
                            phase_loss_scaled = torch.zeros_like(phase_loss_scaled)
                        latent_loss = latent_loss + phase_loss_scaled
                        stats["strategy_phase_loss"].append(phase_stats["phase_term"])
                    else:
                        stats["strategy_phase_loss"].append(0.0)

                    if hparams.fixed_latent_strategy:
                        strategy_entropy = torch.zeros_like(entropy)
                        persist_loss_value = 0.0
                        latent_loss = zero_scalar
                        strategy_marginal_entropy_loss_value = 0.0
                        strategy_marginal_entropy_nats_value = 0.0
                        strategy_marginal_entropy_kl_value = 0.0
                    if (
                        curr_sep_coef > 0.0
                        and phase_key != "B"
                        and not hparams.fixed_latent_strategy
                        and z_idx is not None
                    ):
                        max_action_entropy = float(model.n_agents) * sum(
                            math.log(max(1, int(dim)))
                            for dim in model.per_agent_action_dims
                        )
                        separation_gate = _z_separation_gate_mask(
                            advantages=advantages,
                            action_entropy=entropy,
                            global_state=batch["global_state"],
                            max_action_entropy=max_action_entropy,
                            min_abs_advantage=float(
                                getattr(
                                    hparams,
                                    "latent_actor_z_separation_min_abs_advantage",
                                    0.0,
                                )
                                or 0.0
                            ),
                            min_decision_frac=float(
                                getattr(
                                    hparams,
                                    "latent_actor_z_separation_min_decision_frac",
                                    0.0,
                                )
                                or 0.0
                            ),
                            max_entropy_frac=float(
                                getattr(
                                    hparams,
                                    "latent_actor_z_separation_max_entropy_frac",
                                    1.0,
                                )
                                if getattr(
                                    hparams,
                                    "latent_actor_z_separation_max_entropy_frac",
                                    1.0,
                                )
                                is not None
                                else 1.0
                            ),
                        )
                        z_sep_loss, z_sep_stats = _policy_z_separation_loss(
                            model,
                            obs_batch,
                            z_idx.long(),
                            latent_k=int(hparams.latent_k),
                            margin=float(
                                getattr(
                                    hparams,
                                    "latent_actor_z_separation_margin",
                                    0.02,
                                )
                                or 0.0
                            ),
                            active_mask=separation_gate,
                            subsample_generator=self._z_separation_generator,
                        )
                        z_sep_train_active = 1.0
                        latent_loss = latent_loss + curr_sep_coef * z_sep_loss
                else:
                    log_prob = action_log_prob
                    strategy_entropy = torch.zeros_like(entropy)
                    persist_loss_value = 0.0
                    latent_loss = zero_scalar
                    resample = torch.zeros_like(entropy, dtype=torch.bool)
                    strategy_marginal_entropy_loss_value = 0.0
                    strategy_marginal_entropy_nats_value = 0.0
                    strategy_marginal_entropy_kl_value = 0.0
                    stats["strategy_kl"].append(0.0)
                    stats["strategy_phase_loss"].append(0.0)

                if hparams.use_latent_strategy and not hparams.fixed_latent_strategy:
                    strat_adv = (
                        batch["option_advantages"]
                        if getattr(cfg, "latent_q_phi_option_advantage", False)
                        else advantages
                    )
                    strategy_policy_loss_scaled, strategy_ppo_stats = _latent_strategy_ppo_loss(
                        strategy_log_prob,
                        batch["z_log_probs"],
                        strat_adv,
                        resample,
                        clip_range=float(hparams.clip_range),
                        coef=float(hparams.latent_strategy_ppo_coef),
                        device=device,
                    )
                    # Default to a zero tensor so unit tests that mock
                    # ``_latent_strategy_ppo_loss`` with a minimal return value
                    # (e.g. an empty stats dict) do not KeyError. The real
                    # production path always populates this key.
                    strategy_policy_loss = strategy_ppo_stats.pop(
                        "policy_loss",
                        zero_scalar,
                    )
                    # ``apply_main_loop_qphi_loss`` is computed above and already
                    # incorporates the dedicated-router-optimizer safeguard.
                    if not apply_main_loop_qphi_loss:
                        strategy_policy_loss_scaled = torch.zeros_like(strategy_policy_loss_scaled)
                        strategy_policy_loss = torch.zeros_like(strategy_policy_loss)
                    strategy_aux_return_loss_value = 0.0
                    if bool(resample.any().item()):
                        latent_loss = latent_loss + strategy_policy_loss_scaled
                        if (
                            hparams.latent_strategy_aux_return_head
                            and hparams.latent_strategy_aux_return_coef > 0.0
                        ):
                            pred_all = model.strategy_aux_return_predictions(
                                batch["global_state"]
                            )
                            ret_target = _normalize_strategy_returns(
                                runtime, batch["returns"][resample]
                            )
                            # ``latent_strategy_aux_coef`` is a back-compat alias
                            # for ``latent_strategy_aux_return_coef`` that some
                            # legacy cfg paths still set; prefer it when present
                            # on the trainer (mirrors pre-refactor behavior).
                            aux_return_loss_scaled, aux_return_stats = _latent_strategy_aux_return_loss(
                                pred_all,
                                batch["z"],
                                ret_target,
                                resample,
                                latent_k=int(hparams.latent_k),
                                coef=float(
                                    getattr(
                                        runtime,
                                        "latent_strategy_aux_coef",
                                        hparams.latent_strategy_aux_return_coef,
                                    )
                                ),
                                device=device,
                            )
                            strategy_aux_return_loss_value = aux_return_stats["aux_return_term"]
                            if has_dedicated_router_opt:
                                aux_return_loss_scaled = torch.zeros_like(aux_return_loss_scaled)
                                strategy_aux_return_loss_value = 0.0
                            latent_loss = latent_loss + aux_return_loss_scaled
                else:
                    strategy_policy_loss = zero_scalar
                    strategy_aux_return_loss_value = 0.0
                    strategy_ppo_stats = {
                        "approx_kl": zero_scalar,
                        "clip_fraction": zero_scalar,
                        "ratio": torch.ones((1,), dtype=torch.float32, device=device),
                    }
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
                loss = policy_loss + hparams.vf_coef * value_loss + ent_coef * entropy_loss + latent_loss

                v6i1_three_opt = bool(getattr(runtime, "v6i1_three_optimizer_mode", False))
                if v6i1_three_opt:
                    phase, _, _, _ = v6i1_schedule_context(runtime)
                    runtime.actor_optimizer.zero_grad(set_to_none=True)
                    runtime.critic_optimizer.zero_grad(set_to_none=True)
                    runtime.router_optimizer.zero_grad(set_to_none=True)
                    total_loss = hparams.vf_coef * value_loss
                    if phase != "B":
                        total_loss = total_loss + policy_loss + ent_coef * entropy_loss
                    if latent_loss.requires_grad:
                        total_loss = total_loss + latent_loss
                    _assert_finite_loss(total_loss, epoch_idx=_epoch_idx, mb_idx=_mb_idx)
                    total_loss.backward()
                    _assert_finite_gradients(model, epoch_idx=_epoch_idx, mb_idx=_mb_idx)
                    strategy_grad_norm = self.latent_state.strategy_encoder_grad_norm()
                    v6i1_grads = step_v6i1_optimizers(
                        runtime,
                        phase=phase,
                        actor_step=phase != "B",
                        critic_step=True,
                        router_step=phase in ("B", "C"),
                        max_grad_norm=float(cfg.max_grad_norm),
                    )
                    grad_norm = max(
                        float(v6i1_grads.get("actor_grad_norm", 0.0)),
                        float(v6i1_grads.get("critic_grad_norm", 0.0)),
                        float(v6i1_grads.get("router_grad_norm", 0.0)),
                    )
                else:
                    self.optimizer.zero_grad(set_to_none=True)
                    _assert_finite_loss(loss, epoch_idx=_epoch_idx, mb_idx=_mb_idx)
                    loss.backward()
                    _assert_finite_gradients(model, epoch_idx=_epoch_idx, mb_idx=_mb_idx)
                    strategy_grad_norm = self.latent_state.strategy_encoder_grad_norm()
                    grad_norm = _clip_global_grad_norm(model, float(cfg.max_grad_norm))
                    self.optimizer.step()

                approx_kl_value = float(ppo_stats["approx_kl"].detach().cpu().item())
                stats["policy_loss"].append(float(policy_loss.detach().cpu().item()))
                stats["value_loss"].append(float(value_loss.detach().cpu().item()))
                stats["entropy"].append(float(entropy.mean().detach().cpu().item()))
                stats["approx_kl"].append(approx_kl_value)
                stats["clip_fraction"].append(float(ppo_stats["clip_fraction"].detach().cpu().item()))
                stats["grad_norm"].append(float(torch.as_tensor(grad_norm).detach().cpu().item()))
                stats["strategy_entropy"].append(float(strategy_entropy.mean().detach().cpu().item()))
                stats["strategy_policy_loss"].append(float(strategy_policy_loss.detach().cpu().item()))
                stats["strategy_approx_kl"].append(
                    float(strategy_ppo_stats["approx_kl"].detach().cpu().item())
                )
                stats["strategy_clip_fraction"].append(
                    float(strategy_ppo_stats["clip_fraction"].detach().cpu().item())
                )
                ratio_z = strategy_ppo_stats["ratio"].detach().float()
                stats["strategy_ratio_std"].append(
                    float(ratio_z.std(unbiased=False).detach().cpu().item()) if ratio_z.numel() > 1 else 0.0
                )
                stats["strategy_aux_return_loss"].append(float(strategy_aux_return_loss_value))
                stats["strategy_persist_loss"].append(float(persist_loss_value))
                stats["strategy_marginal_entropy_loss"].append(
                    float(strategy_marginal_entropy_loss_value)
                )
                stats["strategy_marginal_entropy_nats"].append(
                    float(strategy_marginal_entropy_nats_value)
                )
                stats["strategy_marginal_entropy_kl"].append(
                    float(strategy_marginal_entropy_kl_value)
                )
                # Rollout-level soft router diagnostics. Computed once per
                # inner epoch (forward-filled into every minibatch row);
                # zero on non-marginal modes / non-v5i6 presets so the
                # column schema is stable.
                stats["router_rollout_soft_marginal_entropy_nats"].append(
                    float(last_rollout_soft_diag.get(
                        "router_rollout_soft_marginal_entropy_nats", 0.0
                    ))
                )
                stats["router_rollout_soft_conditional_entropy_nats"].append(
                    float(last_rollout_soft_diag.get(
                        "router_rollout_soft_conditional_entropy_nats", 0.0
                    ))
                )
                stats["router_rollout_soft_mi_proxy_nats"].append(
                    float(last_rollout_soft_diag.get(
                        "router_rollout_soft_mi_proxy_nats", 0.0
                    ))
                )
                stats["router_rollout_soft_argmax_occupancy_max"].append(
                    float(last_rollout_soft_diag.get(
                        "router_rollout_soft_argmax_occupancy_max", 0.0
                    ))
                )
                stats["router_rollout_soft_argmax_occupancy_min"].append(
                    float(last_rollout_soft_diag.get(
                        "router_rollout_soft_argmax_occupancy_min", 0.0
                    ))
                )
                stats["router_rollout_soft_argmax_occupancy_ratio"].append(
                    float(last_rollout_soft_diag.get(
                        "router_rollout_soft_argmax_occupancy_ratio", 0.0
                    ))
                )
                stats["router_rollout_resample_count"].append(
                    float(last_rollout_soft_diag.get(
                        "router_rollout_resample_count", 0.0
                    ))
                )
                for _k_idx in range(int(hparams.latent_k)):
                    stats[f"router_rollout_soft_p_bar_z{_k_idx}"].append(
                        float(last_rollout_soft_diag.get(
                            f"router_rollout_soft_p_bar_z{_k_idx}", 0.0
                        ))
                    )
                stats["strategy_grad_norm"].append(strategy_grad_norm)
                stats["strategy_resample_fraction"].append(
                    float(resample.float().mean().detach().cpu().item())
                )
                stats["latent_actor_z_separation_loss"].append(
                    float(z_sep_loss.detach().cpu().item())
                )
                stats["latent_actor_z_separation_jsd"].append(
                    float(z_sep_stats["jsd"].detach().cpu().item())
                )
                stats["latent_actor_z_separation_active"].append(
                    float(z_sep_stats["active"].detach().cpu().item())
                )
                stats["latent_actor_z_separation_train_active"].append(float(z_sep_train_active))
                strategy_kl_value = float(strategy_ppo_stats["approx_kl"].detach().cpu().item())
                stop_for_action = (
                    target_kl is not None and approx_kl_value > 1.5 * float(target_kl)
                )
                stop_for_strategy = (
                    strategy_target_kl is not None
                    and strategy_kl_value > 1.5 * float(strategy_target_kl)
                )
                if stop_for_action or stop_for_strategy:
                    stop_update = True
                    break
            if stop_update:
                break

        episode_strategy_stats = self.latent_state.apply_episode_strategy_ppo(
            latent_lam_h=latent_lam_h
        )
        macro_strategy_stats: dict[str, float] = {}
        if is_v6i1_staged_trainer(runtime) and v6i1_macro_router_active(runtime):
            macro_strategy_stats = self.latent_state.apply_macro_strategy_ppo()
        # v3i19 arc-credit PPO update. Runs as a separate inner-PPO pass on
        # q_phi over per-arc records (one per z-arc, not one per episode).
        # No-op when ``latent_arc_credit_enabled`` is False so legacy presets
        # remain untouched; mutually compatible with episode-credit (both can
        # be on, though v3i19 explicitly turns episode-credit off).
        arc_strategy_stats = self.latent_state.apply_arc_strategy_ppo()
        self.latent_state.reset_arc_credit_rollout_state()
        self.latent_state.reset_macro_rollout_state()
        rollout_specialist_stats = (
            self.latent_state.apply_rollout_specialist_router(buffer)
        )
        strategy_experience_stats = _write_strategy_experience_table(runtime)
        # v3i3 per-refresh proof-layer CSV log. Always-safe no-op when the
        # feature is disabled. Runs AFTER apply_episode_strategy_ppo so the
        # finalized ``rollout_refresh_records`` (used by both the loss above
        # and the log below) are consistent.
        refresh_log_stats = _write_refresh_log_table(runtime)
        # Drain the per-rollout refresh records once both consumers (loss +
        # CSV) have run. The cumulative ``refresh_preference_buffer`` is
        # NOT cleared (it is the teacher's growing evidence library).
        self.latent_state.clear_rollout_refresh_records()

        runtime.last_stats = {
            name: float(np.mean(values)) if values else 0.0 for name, values in stats.items()
        }
        value_losses = np.asarray(stats["value_loss"], dtype=np.float32)
        if value_losses.size > 0:
            runtime.last_stats.update(
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
            runtime.last_stats.update(
                {
                    "value_loss_min": 0.0,
                    "value_loss_std": 0.0,
                    "value_loss_p10": 0.0,
                    "value_loss_p50": 0.0,
                    "value_loss_p90": 0.0,
                    "value_loss_max": 0.0,
                }
            )
        runtime.last_stats["learning_rate"] = float(lr)
        runtime.last_stats["latent_lam_h"] = float(latent_lam_h)
        runtime.last_stats["latent_actor_z_adapter_scale"] = float(curr_adapter_scale)
        runtime.last_stats["latent_actor_z_separation_coef"] = float(curr_sep_coef)
        if is_v6i1_staged_trainer(runtime):
            phase, step_sched, t_a, nominal = v6i1_schedule_context(runtime)
            runtime.last_stats["v6i1_phase"] = float({"A": 0.0, "B": 1.0, "C": 2.0}.get(phase, -1.0))
            runtime.last_stats["v6i1_cf_coef_current"] = float(resolve_v6i1_cf_coef_current(runtime))
            runtime.last_stats["v6i1_usage_coef_current"] = float(resolve_v6i1_rollout_usage_coef(runtime))
            runtime.last_stats["latent_forced_z_episode_frac_current"] = float(
                resolve_v6i1_episode_forced_frac(runtime)
            )
            if v6i1_lr_stats:
                runtime.last_stats.update(v6i1_lr_stats)
        else:
            # v5i3 forced-z anneal telemetry: resolved fraction at the start of
            # this update window. Reads the schedule from cfg + current global_step;
            # constant under presets that leave the four anneal fields unset.
            runtime.last_stats["latent_forced_z_episode_frac_current"] = float(
                resolve_latent_forced_z_frac(self.cfg, global_step=int(runtime.global_step))
            )
        if hparams.normalize_returns:
            rn = runtime.return_norm
            runtime.last_stats["return_norm_mean"] = float(rn.mean)
            runtime.last_stats["return_norm_std"] = float(rn.std)
            runtime.last_stats["return_norm_count"] = float(rn.count)
        else:
            runtime.last_stats["return_norm_mean"] = 0.0
            runtime.last_stats["return_norm_std"] = 0.0
            runtime.last_stats["return_norm_count"] = 0.0
        runtime.last_stats.update(_strategy_resample_advantage_stats(runtime, buffer))
        runtime.last_stats.update(_latent_option_advantage_stats(runtime, buffer))
        runtime.last_stats.update(_rollout_advantage_diagnostics(runtime, buffer))
        runtime.last_stats.update(_latent_rollout_stats(runtime, buffer))
        runtime.last_stats.update(_latent_opponent_rollout_diag(runtime, buffer))
        runtime.last_stats.update(_behavior_diversity_stats(runtime, buffer))
        forced_z_profile = _forced_z_behavior_profile(runtime, buffer)
        runtime.last_stats.update(forced_z_profile)
        if forced_z_profile:
            runtime.latent_state.update_intervention_gate_from_profile(forced_z_profile)
        runtime.last_stats.update(_policy_z_sensitivity_kl(runtime, buffer))
        runtime.last_stats.update(episode_strategy_stats)
        runtime.last_stats.update(arc_strategy_stats)
        runtime.last_stats.update(macro_strategy_stats)
        if v6i1_lr_stats:
            runtime.last_stats.update(v6i1_lr_stats)
        if v6i1_usage_coef > 0.0:
            runtime.last_stats["v6i1_rollout_usage_coef"] = float(v6i1_usage_coef)
        runtime.last_stats["rollout_marginal_active"] = (
            1.0 if apply_rollout_marginal and rollout_resample_states_full is not None else 0.0
        )
        if rollout_marginal_skip_reason is not None:
            runtime.last_stats["rollout_marginal_skip_reason"] = float(
                {"empty_rollout": 1.0, "no_resample_rows": 2.0}.get(rollout_marginal_skip_reason, 0.0)
            )
        _populate_main_loop_qphi_telemetry(
            runtime.last_stats,
            cfg=cfg,
            hparams=hparams,
            runtime=runtime,
            latent_lam_h=float(latent_lam_h),
        )
        runtime.last_stats.update(rollout_specialist_stats)
        runtime.last_stats.update(strategy_experience_stats)
        runtime.last_stats.update(refresh_log_stats)
        runtime.last_stats.update(self.latent_state.behavior_contrast_rollout_stats())
        runtime.last_stats.update(self.latent_state.event_refresh_rollout_stats())
        runtime.last_stats.update(
            self.latent_state.sparse_tactical_refresh_rollout_stats()
        )
        if hparams.use_latent_strategy and "z_forced" in buffer.fields:
            forced_steps = buffer.fields["z_forced"][: int(buffer.pos)].detach().float()
            runtime.last_stats["latent_forced_z_step_fraction"] = (
                float(forced_steps.mean().cpu().item()) if forced_steps.numel() > 0 else 0.0
            )
        else:
            runtime.last_stats["latent_forced_z_step_fraction"] = 0.0
        return runtime.last_stats


__all__ = ["PPOUpdater"]
