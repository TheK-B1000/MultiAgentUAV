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

from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from rl.latent_losses import (
    strategy_aux_return_loss as _latent_strategy_aux_return_loss,
    strategy_entropy_loss as _latent_strategy_entropy_loss,
    strategy_kl_consecutive_loss as _latent_strategy_kl_consecutive_loss,
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
from rl.custom_ppo.schedules import resolve_latent_lam_h
from rl.custom_ppo.trainer_config import TrainerHyperparams

if TYPE_CHECKING:
    from rl.custom_ppo.latent_strategy_state import LatentStrategyState
    from rl.custom_ppo.trainer import CustomPPOTrainer


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
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        ent_coef = hparams.ent_coef if progress_remaining > 0.75 else 0.5 * hparams.ent_coef
        latent_lam_h = self.compute_latent_lam_h(runtime.global_step, total_timesteps)
        _update_strategy_return_stats(runtime, buffer)

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
        }
        stop_update = False
        target_kl = getattr(cfg, "target_kl", None)
        model = self.model
        for _ in range(hparams.n_epochs):
            for batch in buffer.iter_minibatches(hparams.batch_size, shuffle=True):
                obs_batch = {
                    "grid": batch["obs_grid"],
                    "vec": batch["obs_vec"],
                    "agent_mask": batch["obs_agent_mask"],
                    "mask": batch["obs_mask"],
                }
                z_idx = batch["z"] if hparams.use_latent_strategy else None
                values_norm, action_log_prob, entropy, aux = model.evaluate_actions(
                    obs_batch,
                    batch["global_state"],
                    batch["actions"],
                    z_idx=z_idx,
                )
                if hparams.use_latent_strategy:
                    resample = batch["z_resampled"].bool()
                    persist_mask = batch["z_persist_mask"].bool()
                    log_prob = action_log_prob
                    strategy_log_prob = aux["strategy_log_prob"]
                    strategy_entropy = aux["strategy_entropy"]
                    h_goal = str(
                        getattr(cfg, "latent_entropy_objective", "maximize") or "maximize"
                    ).lower()
                    strategy_entropy_loss, _ = _latent_strategy_entropy_loss(
                        strategy_entropy,
                        resample,
                        objective=h_goal,
                        lam_h=latent_lam_h,
                        device=device,
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
                    ):
                        assert persist_stats["persist_term"] == 0.0, (
                            "L_persist must be exactly 0 when no mid-episode resampling (latent_resample_every_n=0, on_flag off, no event refresh)"
                        )
                    apply_main_loop_qphi_loss = float(getattr(cfg, "latent_strategy_ppo_coef", 0.1) or 0.0) > 0.0
                    if not apply_main_loop_qphi_loss:
                        strategy_entropy_loss = torch.zeros_like(strategy_entropy_loss)
                        persist_term_loss = torch.zeros_like(persist_term_loss)
                    persist_loss_value = 0.0 if not apply_main_loop_qphi_loss else persist_stats["persist_term"]
                    latent_loss = persist_term_loss + strategy_entropy_loss
                    if hparams.latent_kl_consecutive > 0.0:
                        kl_loss, kl_stats = _latent_strategy_kl_consecutive_loss(
                            batch["z_logits"],
                            batch["z_logits_prev"],
                            batch["z_kl_prev_valid"],
                            coef=float(hparams.latent_kl_consecutive),
                        )
                        if not apply_main_loop_qphi_loss:
                            kl_loss = torch.zeros_like(kl_loss)
                        latent_loss = latent_loss + kl_loss
                        stats["strategy_kl"].append(kl_stats["kl_mean"])
                    else:
                        stats["strategy_kl"].append(0.0)
                    if hparams.latent_strategy_aux_predict_phase_coef > 0.0:
                        phase_logits = model.phase_logits_from_strategy_logits(
                            aux["strategy_logits"]
                        )
                        phase_loss_scaled, phase_stats = _latent_strategy_phase_aux_loss(
                            phase_logits,
                            batch["phase_id"],
                            coef=float(hparams.latent_strategy_aux_predict_phase_coef),
                        )
                        if not apply_main_loop_qphi_loss:
                            phase_loss_scaled = torch.zeros_like(phase_loss_scaled)
                        latent_loss = latent_loss + phase_loss_scaled
                        stats["strategy_phase_loss"].append(phase_stats["phase_term"])
                    else:
                        stats["strategy_phase_loss"].append(0.0)

                    if hparams.fixed_latent_strategy:
                        strategy_entropy = torch.zeros_like(entropy)
                        persist_loss_value = 0.0
                        latent_loss = torch.zeros((), dtype=torch.float32, device=device)
                else:
                    log_prob = action_log_prob
                    strategy_entropy = torch.zeros_like(entropy)
                    persist_loss_value = 0.0
                    latent_loss = torch.zeros((), dtype=torch.float32, device=device)
                    resample = torch.zeros_like(entropy, dtype=torch.bool)
                    stats["strategy_kl"].append(0.0)
                    stats["strategy_phase_loss"].append(0.0)

                advantages = batch["advantages"]
                if advantages.numel() > 1:
                    advantages = (advantages - advantages.mean()) / (
                        advantages.std(unbiased=False) + 1e-8
                    )
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
                        torch.zeros((), dtype=torch.float32, device=device),
                    )
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
                            if not apply_main_loop_qphi_loss:
                                aux_return_loss_scaled = torch.zeros_like(aux_return_loss_scaled)
                                strategy_aux_return_loss_value = 0.0
                            latent_loss = latent_loss + aux_return_loss_scaled
                else:
                    strategy_policy_loss = torch.zeros((), dtype=torch.float32, device=device)
                    strategy_aux_return_loss_value = 0.0
                    strategy_ppo_stats = {
                        "approx_kl": torch.zeros((), dtype=torch.float32, device=device),
                        "clip_fraction": torch.zeros((), dtype=torch.float32, device=device),
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

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                strategy_grad_norm = self.latent_state.strategy_encoder_grad_norm()
                # Clip value/critic parameters and policy/router parameters separately
                policy_router_params = [
                    p for name, p in model.named_parameters()
                    if p.requires_grad and not ("critic" in name or "value" in name)
                ]
                value_params = [
                    p for name, p in model.named_parameters()
                    if p.requires_grad and ("critic" in name or "value" in name)
                ]
                grad_norm_policy = torch.nn.utils.clip_grad_norm_(
                    policy_router_params, float(cfg.max_grad_norm)
                )
                grad_norm_value = torch.nn.utils.clip_grad_norm_(
                    value_params, float(cfg.max_grad_norm)
                )
                grad_norm = max(float(grad_norm_policy), float(grad_norm_value))
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
                stats["strategy_grad_norm"].append(strategy_grad_norm)
                stats["strategy_resample_fraction"].append(
                    float(resample.float().mean().detach().cpu().item())
                )
                if target_kl is not None and approx_kl_value > 1.5 * float(target_kl):
                    stop_update = True
                    break
            if stop_update:
                break

        episode_strategy_stats = self.latent_state.apply_episode_strategy_ppo(
            latent_lam_h=latent_lam_h
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
        runtime.last_stats.update(_forced_z_behavior_profile(runtime, buffer))
        runtime.last_stats.update(_policy_z_sensitivity_kl(runtime, buffer))
        runtime.last_stats.update(episode_strategy_stats)
        runtime.last_stats.update(strategy_experience_stats)
        runtime.last_stats.update(refresh_log_stats)
        runtime.last_stats.update(self.latent_state.behavior_contrast_rollout_stats())
        runtime.last_stats.update(self.latent_state.event_refresh_rollout_stats())
        if hparams.use_latent_strategy and "z_forced" in buffer.fields:
            forced_steps = buffer.fields["z_forced"][: int(buffer.pos)].detach().float()
            runtime.last_stats["latent_forced_z_step_fraction"] = (
                float(forced_steps.mean().cpu().item()) if forced_steps.numel() > 0 else 0.0
            )
        else:
            runtime.last_stats["latent_forced_z_step_fraction"] = 0.0
        return runtime.last_stats


__all__ = ["PPOUpdater"]
