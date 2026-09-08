"""Episode credit coordinator."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.nn.functional as F

from rl import launch_audit_hooks
from rl.custom_ppo.csv_writers import SCRIPTED_OPPONENT_MI_COUNT
from rl.custom_ppo.curriculum_gates import is_staged_v6i1_curriculum
from rl.custom_ppo.latent.context_buckets import episode_bucket_baseline_keys, strategy_experience_bucket_ids
from rl.custom_ppo.latent.credit.episode.advantages import compute_fixed_episode_advantages
from rl.custom_ppo.latent.credit.episode.auxiliary_losses import (
    build_episode_auxiliary_context,
    make_auxiliary_loss_fn,
)
from rl.custom_ppo.latent.credit.episode.awrd_targets import build_awrd_targets
from rl.custom_ppo.latent.credit.episode.batch import EpisodeBatchBuilder
from rl.custom_ppo.latent.credit.episode.preference_targets import build_preference_targets
from rl.custom_ppo.latent.credit.episode.refresh_targets import build_refresh_targets
from rl.custom_ppo.latent.credit.episode.telemetry import empty_episode_strategy_stats
from rl.custom_ppo.latent.optimization.ppo_stats import EpisodeStatsAccumulator
from rl.custom_ppo.latent.opponent_resolution import resolve_opponent_id
from rl.custom_ppo.latent.opponent_telemetry import resolve_logged_opponents
from rl.custom_ppo.latent.optimization.router_ppo import RouterPPOEngine
from rl.custom_ppo.latent.optimization.router_registry import LatentOptimizerRegistry
from rl.custom_ppo.latent.types import RouterAction, RouterActionSource, RouterPPOBatch, RouterPPOConfig

if TYPE_CHECKING:
    from rl.custom_ppo.latent.state import LatentStrategyState


class EpisodeCreditManager:
    def __init__(
        self,
        host: LatentStrategyState,
        *,
        optimizer_registry: LatentOptimizerRegistry | None = None,
        router_ppo_engine: RouterPPOEngine | None = None,
    ) -> None:
        self.host = host
        self._registry = optimizer_registry
        self._engine = router_ppo_engine
        self.batch_builder = EpisodeBatchBuilder(host)

    def _registry_for_host(self) -> LatentOptimizerRegistry | None:
        if self._registry is not None:
            return self._registry
        router_opt = getattr(self.host.trainer, "router_optimizer", None) or getattr(
            self.host.trainer, "latent_router_optimizer", None
        )
        if router_opt is None:
            return None
        return LatentOptimizerRegistry.from_trainer(self.host.trainer)

    def _engine_for_host(self) -> RouterPPOEngine:
        if self._engine is not None:
            return self._engine
        trainer = self.host.trainer
        registry = self._registry_for_host()
        fallback = None if registry is not None else trainer.optimizer
        self._engine = RouterPPOEngine(trainer=trainer, registry=registry, fallback_optimizer=fallback)
        return self._engine

    @staticmethod
    def empty_episode_strategy_stats(latent_k: int = 4) -> dict[str, float]:
        return empty_episode_strategy_stats(latent_k)

    def store_episode_strategy_start(
        self,
        *,
        start_mask: torch.Tensor,
        global_state: torch.Tensor,
        router_action: RouterAction,
        selector_hidden: torch.Tensor | None = None,
        action_sources: list[RouterActionSource] | tuple[RouterActionSource, ...] | None = None,
        z_logits: torch.Tensor | None = None,
    ) -> None:
        trainer = self.host.trainer
        if not trainer.latent_episode_strategy_ppo or not bool(start_mask.any().item()):
            return
        idx = torch.where(start_mask)[0]
        if z_logits is None:
            z_logits = trainer.model.strategy_logits(global_state, selector_hidden=selector_hidden)
        buckets = strategy_experience_bucket_ids(global_state.index_select(0, idx)).detach()
        self.host.episode_strategy_state[idx] = global_state.index_select(0, idx).detach()
        self.host.episode_strategy_z[idx] = router_action.executed_z.index_select(0, idx).detach()
        self.host.episode_strategy_log_prob[idx] = router_action.behavior_log_prob.index_select(0, idx).detach()
        if selector_hidden is not None and self.host.episode_strategy_selector_hidden is not None:
            self.host.episode_strategy_selector_hidden[idx] = selector_hidden.index_select(0, idx).detach()
        probs = router_action.router_probs.detach()
        self.host.episode_strategy_probs[idx, : trainer.latent_k] = probs.index_select(0, idx)
        self.host.episode_strategy_bucket[idx] = buckets
        self.host.episode_strategy_has_start[idx] = True
        for row_i, env_i in enumerate(idx.detach().cpu().tolist()):
            hidden_row = None
            if self.host.episode_strategy_selector_hidden is not None:
                hidden_row = self.host.episode_strategy_selector_hidden[int(env_i)]
            if action_sources is not None and int(env_i) < len(action_sources):
                source = action_sources[int(env_i)]
            else:
                source = router_action.source
            if router_action.proposed_z.ndim > 0:
                proposed = int(router_action.proposed_z[int(env_i)].detach().cpu().item())
                executed = int(router_action.executed_z[int(env_i)].detach().cpu().item())
                behavior_lp = float(router_action.behavior_log_prob[int(env_i)].detach().cpu().item())
                router_lp = float(router_action.router_log_prob[int(env_i)].detach().cpu().item())
            else:
                proposed = int(router_action.proposed_z.detach().cpu().item())
                executed = int(router_action.executed_z.detach().cpu().item())
                behavior_lp = float(router_action.behavior_log_prob.detach().cpu().item())
                router_lp = float(router_action.router_log_prob.detach().cpu().item())
            self.host.episode_strategy_recorder.record_start(
                env_index=int(env_i),
                episode_id=int(self.host.next_strategy_episode_id),
                global_state_0=global_state[int(env_i)],
                proposed_z=proposed,
                executed_z=executed,
                behavior_log_prob=behavior_lp,
                router_log_prob=router_lp,
                action_source=source,
                bucket_id=int(buckets[row_i].detach().cpu().item()),
                q_phi_probs=probs[int(env_i), : trainer.latent_k].detach().cpu().tolist(),
                selector_hidden_0=hidden_row,
            )
            self.host.next_strategy_episode_id += 1

    def episode_strategy_training_batch(self) -> Optional[dict[str, torch.Tensor]]:
        return self.batch_builder.build_legacy_dict()

    def apply_episode_strategy_ppo(self, *, latent_lam_h: float) -> dict[str, float]:
        trainer = self.host.trainer
        stats = self.empty_episode_strategy_stats(trainer.latent_k)
        batch = self.batch_builder.build()
        if batch is None:
            return stats
        stats["latent_episode_count"] = float(batch.episode_returns.numel())
        train_after = max(0, int(getattr(trainer, "latent_q_phi_train_after_steps", 0) or 0))
        if train_after > 0 and int(getattr(trainer, "global_step", 0) or 0) < train_after:
            return stats
        stats["latent_q_phi_train_active"] = 1.0

        bucket_baseline_helper = getattr(trainer, "latent_bucket_baseline", None)
        bucket_mode = getattr(trainer, "latent_q_phi_bucket_baseline", None)
        bucket_baseline_vector = None
        if bucket_baseline_helper is not None and bucket_mode is not None:
            keys = episode_bucket_baseline_keys(
                mode=str(bucket_mode),
                states=batch.states,
                opponent_ids=batch.opponent_ids,
                bucket_ids=batch.bucket_ids,
            )
            bucket_baseline_vector = bucket_baseline_helper.update_and_compute(
                batch.episode_returns.detach(), keys.detach()
            ).detach()

        preference = build_preference_targets(
            trainer=trainer,
            host=self.host,
            batch_size=int(batch.states.shape[0]),
            opponent_ids=batch.opponent_ids,
            bucket_ids=batch.bucket_ids,
            device=trainer.device,
            latent_k=int(trainer.latent_k),
        )
        awrd = build_awrd_targets(
            trainer=trainer,
            host=self.host,
            batch_size=int(batch.states.shape[0]),
            executed_z=batch.executed_z,
            opponent_ids=batch.opponent_ids,
            bucket_ids=batch.bucket_ids,
            device=trainer.device,
            latent_k=int(trainer.latent_k),
        )
        refresh = build_refresh_targets(trainer=trainer, host=self.host)

        return_norm = bool(trainer.latent_episode_strategy_return_norm)
        fixed_adv, _bucket_keys = compute_fixed_episode_advantages(
            trainer=trainer,
            model=trainer.model,
            states=batch.states,
            executed_z=batch.executed_z,
            episode_returns=batch.episode_returns,
            selector_hidden=batch.selector_hidden,
            bucket_baseline_vector=bucket_baseline_vector,
            bucket_mode=str(bucket_mode) if bucket_mode is not None else None,
            opponent_ids=batch.opponent_ids,
            bucket_ids=batch.bucket_ids,
            return_norm=return_norm,
        )

        staged_v6 = is_staged_v6i1_curriculum(trainer.cfg)
        registry = self._registry_for_host()
        if staged_v6:
            assert registry is not None
            registry.require_router_optimizer(staged_v6=True)

        aux_ctx = build_episode_auxiliary_context(
            trainer=trainer,
            host=self.host,
            batch=batch,
            latent_lam_h=latent_lam_h,
            preference=preference,
            awrd=awrd,
            refresh=refresh,
        )
        aux_fn = make_auxiliary_loss_fn(aux_ctx)
        accum = EpisodeStatsAccumulator()
        n_epochs = max(1, int(getattr(trainer, "latent_episode_strategy_n_epochs", 1) or 1))
        target_kl = getattr(trainer.cfg, "target_kl", None)
        config = RouterPPOConfig(
            coef=float(trainer.latent_episode_strategy_coef),
            value_coef=float(trainer.latent_episode_strategy_value_coef),
            clip_epsilon=float(trainer.latent_episode_strategy_clip_eps),
            epochs=n_epochs,
            target_kl=float(target_kl) if target_kl is not None else None,
            max_grad_norm=float(trainer.cfg.max_grad_norm),
            objective_name="episode_credit",
        )
        router_batch = RouterPPOBatch(
            states=batch.states,
            executed_z=batch.executed_z,
            old_behavior_log_prob=batch.old_behavior_log_prob,
            fixed_advantages=fixed_adv,
            returns=batch.episode_returns,
            selector_hidden=batch.selector_hidden,
        )
        engine = self._engine_for_host()
        ppo_stats, step_results = engine.apply(
            router_batch,
            config=config,
            fixed_advantages=fixed_adv,
            value_target=batch.episode_returns,
            auxiliary_loss_fn=aux_fn,
            stats_accumulator=accum,
        )
        if registry is not None:
            self.host.router_optimizer_step_count += accum.optimizer_steps
        elif step_results:
            self.host.router_optimizer_step_count += sum(int(s.stepped) for s in step_results)

        stats.update(ppo_stats)
        stats.update(
            finalize_episode_telemetry(
                host=self.host,
                trainer=trainer,
                batch=batch,
                fixed_adv=fixed_adv,
                preference=preference,
                awrd=awrd,
                refresh=refresh,
                aux_ctx=aux_ctx,
                bucket_baseline_vector=bucket_baseline_vector,
                bucket_baseline_helper=bucket_baseline_helper,
                grad_norm=float(ppo_stats.get("latent_episode_grad_norm_max", ppo_stats.get("grad_norm", 0.0))),
                ppo_stats=ppo_stats,
            )
        )
        return stats

    def record_episode_strategy_outcome(
        self,
        env_index: int,
        info: dict[str, Any],
        *,
        episode_return: float,
    ) -> None:
        trainer = self.host.trainer
        env_i = int(env_index)
        if env_i < 0 or env_i >= int(self.host.episode_strategy_has_start.numel()):
            return

        is_forced_z = bool(self.host.episode_forced_z[env_i].detach().cpu().item())
        z_val = (
            int(self.host.episode_forced_z_id[env_i].detach().cpu().item())
            if is_forced_z
            else int(self.host.current_z[env_i].detach().cpu().item())
        )
        self.host.recent_z_history.append(z_val)

        opponent = resolve_opponent_id(trainer.cfg, info)
        opponent_id = opponent.value

        # Runtime audit seam: z and the LIVE opponent are both known here, once per
        # episode boundary. No-op unless auditors are explicitly attached; on a
        # z->pole drift this raises rather than letting the run finish and be
        # discovered invalid afterwards. See rl/launch_audit_hooks.py.
        launch_audit_hooks.observe_episode_close(trainer, env_i, z_val, opponent_id)

        if is_forced_z:
            er = info.get("episode_result") if isinstance(info.get("episode_result"), dict) else {}
            bs = int(er.get("blue_score", info.get("blue_score", 0)) or 0)
            rs = int(er.get("red_score", info.get("red_score", 0)) or 0)
            episode_win = 1 if bs > rs else 0
            count = max(1, int(self.host.episode_behavior_count[env_i].detach().cpu().item()))
            emb = (self.host.episode_behavior_sum[env_i] / float(count)).detach().cpu().numpy().tolist()
            tactical_bucket = self.host.representative_tactical_bucket(env_i)
            if trainer.latent_episode_strategy_ppo:
                self.host.latent_preference_buffer.append(
                    {
                        "context_bucket": tactical_bucket,
                        "opponent": opponent_id,
                        "phase_flag_state": tactical_bucket,
                        "z": z_val,
                        "return": float(episode_return),
                        "behavior_embedding": emb,
                        "win_loss": episode_win,
                    }
                )
                used_tactical_fallback = int(self.host.episode_tactical_bucket_counts[env_i].sum().item()) <= 0
                self.host.rollout_tactical_bucket_sample_count += 1
                self.host.rollout_tactical_bucket_fallback_count += int(used_tactical_fallback)
            self.host.rollout_completed_episode_count += 1
            self.host.rollout_forced_z_episode_count += 1
            if 0 <= z_val < int(self.host.rollout_forced_z_episode_count_by_z.shape[0]):
                self.host.rollout_forced_z_episode_count_by_z[z_val] += 1
            if (
                0 <= opponent_id < int(self.host.rollout_forced_episode_count_by_opp_z.shape[0])
                and 0 <= z_val < int(self.host.rollout_forced_episode_count_by_opp_z.shape[1])
            ):
                self.host.rollout_forced_episode_count_by_opp_z[opponent_id, z_val] += 1
            alpha = float(getattr(trainer.cfg, "latent_cf_competence_ema_alpha", 0.05))
            self.host.cf_J[z_val] = (1.0 - alpha) * self.host.cf_J[z_val] + alpha * float(episode_return)
            self.host.cf_episode_counts[z_val] += 1
            self.host.cf_has_experience[z_val] = True
            old_mean = self.host.cf_return_mean
            self.host.cf_return_mean = (1.0 - alpha) * old_mean + alpha * float(episode_return)
            self.host.cf_return_var = (1.0 - alpha) * self.host.cf_return_var + alpha * (
                float(episode_return) - old_mean
            ) * (float(episode_return) - self.host.cf_return_mean)
            return

        if not trainer.latent_episode_strategy_ppo:
            return
        if not bool(self.host.episode_strategy_has_start[env_i].detach().cpu().item()):
            return
        used_tactical_fallback = int(self.host.episode_tactical_bucket_counts[env_i].sum().item()) <= 0
        self.host.rollout_tactical_bucket_sample_count += 1
        self.host.rollout_tactical_bucket_fallback_count += int(used_tactical_fallback)
        tactical_bucket = self.host.representative_tactical_bucket(env_i)
        er = info.get("episode_result") if isinstance(info.get("episode_result"), dict) else {}
        bs = int(er.get("blue_score", info.get("blue_score", 0)) or 0)
        rs = int(er.get("red_score", info.get("red_score", 0)) or 0)
        episode_win = 1 if bs > rs else 0
        warmup = int(getattr(trainer, "latent_episode_strategy_warmup_decision_steps", 0) or 0)
        adjusted_return = (
            episode_return - float(self.host.episode_return_baseline_at_commit[env_i].detach().cpu().item())
            if warmup > 0
            else episode_return
        )
        record = self.host.episode_strategy_recorder.record_outcome(
            env_index=env_i,
            episode_return=float(adjusted_return),
            episode_win=episode_win,
            opponent_id=opponent_id,
        )
        if record is not None:
            record["bucket_id"] = tactical_bucket
            self.host.rollout_strategy_episode_records.append(record)
            return
        self.host.missing_episode_record_count += 1


def finalize_episode_telemetry(
    *,
    host: Any,
    trainer: Any,
    batch: Any,
    fixed_adv: torch.Tensor,
    preference: Any,
    awrd: Any,
    refresh: Any,
    aux_ctx: Any,
    bucket_baseline_vector: torch.Tensor | None,
    bucket_baseline_helper: Any,
    grad_norm: float,
    ppo_stats: dict[str, float] | None = None,
) -> dict[str, float]:
    from rl.custom_ppo.latent.types import EpisodeRouterBatch

    assert isinstance(batch, EpisodeRouterBatch)
    stats: dict[str, float] = {}
    stats["episode_credit_grad_norm"] = grad_norm
    stats["episode_credit_adv_mean"] = float(fixed_adv.detach().mean().cpu().item())
    stats["episode_credit_adv_std"] = (
        float(fixed_adv.detach().std(unbiased=False).cpu().item()) if fixed_adv.numel() > 1 else 0.0
    )
    stats["latent_episode_adv_mean"] = stats["episode_credit_adv_mean"]
    stats["latent_episode_adv_std"] = stats["episode_credit_adv_std"]
    stats["latent_episode_return_mean"] = float(batch.episode_returns.detach().mean().cpu().item())
    stats["latent_episode_return_std"] = (
        float(batch.episode_returns.detach().std(unbiased=False).cpu().item())
        if batch.episode_returns.numel() > 1
        else 0.0
    )
    stats["latent_usage_balance_loss"] = float(
        (ppo_stats or {}).get("latent_usage_balance_loss", 0.0)
    )
    stats["latent_preference_loss"] = float(
        (ppo_stats or {}).get("latent_preference_loss", 0.0)
    )
    stats["latent_preference_active_fraction"] = float(preference.mask.float().mean().cpu().item())
    stats["latent_preference_buffer_size"] = float(len(host.latent_preference_buffer))
    stats["latent_preference_num_active_buckets"] = float(preference.active_buckets)
    valid_count = int(preference.mask.sum().item())
    stats["latent_preference_target_entropy"] = (
        float(preference.target_entropy_sum / max(1, valid_count)) if valid_count > 0 else 0.0
    )
    awrd_valid = int(awrd.mask.sum().item())
    stats["latent_awrd_loss"] = float((ppo_stats or {}).get("latent_awrd_loss", 0.0))
    stats["latent_awrd_coef_scale"] = float(awrd.coef_scale)
    stats["latent_awrd_active_fraction"] = float(awrd.mask.float().mean().cpu().item()) if awrd.mask.numel() else 0.0
    stats["latent_awrd_active_buckets"] = float(awrd.active_buckets)
    stats["latent_awrd_buffer_size"] = float(len(host.latent_preference_buffer))
    stats["latent_awrd_target_entropy"] = (
        float(awrd.target_entropy_sum / max(1, awrd_valid)) if awrd_valid > 0 else 0.0
    )
    stats["latent_awrd_margin_mean"] = float(awrd.margin_sum / max(1, awrd_valid)) if awrd_valid > 0 else 0.0
    stats["latent_awrd_wr_spread_mean"] = float(awrd.wr_spread_sum / max(1, awrd_valid)) if awrd_valid > 0 else 0.0
    stats["latent_awrd_best_z_mean"] = float(awrd.best_z_sum / max(1, awrd_valid)) if awrd_valid > 0 else -1.0
    stats["latent_awrd_effective_coef_mean"] = (
        float(awrd.effective_coef_sum / max(1, awrd_valid)) if awrd_valid > 0 else 0.0
    )
    stats["latent_awrd_best_z_match_rate"] = (
        float(awrd.best_z_matches / max(1, awrd_valid)) if awrd_valid > 0 else 0.0
    )
    stats["latent_v3i3_event_pref_active_fraction"] = (
        float(refresh.mask.float().mean().cpu().item())
        if refresh.mask is not None and refresh.mask.numel() > 0
        else 0.0
    )
    stats["latent_v3i3_event_pref_active_buckets"] = float(refresh.active_buckets)
    stats["latent_v3i3_event_pref_active_records"] = float(refresh.active_records)
    stats["latent_v3i3_event_pref_buffer_size"] = float(len(host.refresh_preference_buffer))
    stats["latent_v3i3_event_pref_target_entropy"] = (
        float(refresh.target_entropy_sum / max(1, refresh.active_records))
        if refresh.active_records > 0
        else 0.0
    )
    for key, value in refresh.fallback_counts.items():
        stats[f"latent_v3i3_event_pref_fallback_{key}"] = float(value)
    stats["latent_v3i3_event_pref_rollout_records"] = float(refresh.rollout_records)

    if bucket_baseline_vector is not None and bucket_baseline_helper is not None:
        bs = bucket_baseline_helper.last_stats
        stats.update(
            {
                "bucket_baseline_count": float(bs.get("bucket_count", 0)),
                "bucket_baseline_fallback_frac": float(bs.get("fallback_fraction", 0.0)),
                "bucket_baseline_var_reduction": float(bs.get("variance_reduction_ratio", 1.0)),
                "bucket_baseline_global_mean": float(bs.get("global_mean", 0.0)),
                "bucket_baseline_raw_return_std": float(bs.get("raw_return_std", 0.0)),
                "bucket_baseline_adv_std": float(bs.get("adv_std", 0.0)),
            }
        )

    K = max(1, int(trainer.latent_k))
    z_cpu = batch.executed_z.detach().cpu()
    adv_cpu = fixed_adv.detach().cpu()
    ret_cpu = batch.episode_returns.detach().cpu()
    for z_i in range(K):
        mask = z_cpu == z_i
        count_i = int(mask.sum().item())
        forced_i = int(host.rollout_forced_z_episode_count_by_z[z_i])
        stats[f"router_sample_count_by_z_{z_i}"] = float(count_i)
        stats[f"forced_sample_count_by_z_{z_i}"] = float(forced_i)
        stats[f"episode_count_by_z_{z_i}"] = float(count_i + forced_i)
        if count_i == 0:
            stats[f"mean_episode_advantage_by_z_{z_i}"] = 0.0
            stats[f"std_episode_advantage_by_z_{z_i}"] = 0.0
            stats[f"mean_return_by_z_{z_i}"] = 0.0
            continue
        adv_i = adv_cpu[mask]
        ret_i = ret_cpu[mask]
        stats[f"mean_episode_advantage_by_z_{z_i}"] = float(adv_i.mean().item())
        stats[f"std_episode_advantage_by_z_{z_i}"] = float(adv_i.std(unbiased=False).item()) if count_i > 1 else 0.0
        stats[f"mean_return_by_z_{z_i}"] = float(ret_i.mean().item())

    for o_idx in range(int(SCRIPTED_OPPONENT_MI_COUNT)):
        for z_i in range(K):
            stats[f"forced_episode_opp{o_idx}_z{z_i}_count"] = float(
                host.rollout_forced_episode_count_by_opp_z[o_idx, z_i]
            )

    for opp_name, opp_id in resolve_logged_opponents(trainer.cfg):
        stats[f"latent_pref_{opp_name}_buffer_count"] = float(
            sum(1 for r in host.latent_preference_buffer if r["opponent"] == opp_id)
        )
        stats.setdefault(f"latent_pref_{opp_name}_loss", 0.0)
        stats.setdefault(f"latent_pref_{opp_name}_active_fraction", 0.0)
        stats.setdefault(f"latent_pref_{opp_name}_target_entropy", 0.0)
        stats.setdefault(f"latent_pref_{opp_name}_best_z", -1.0)
        stats.setdefault(f"latent_pref_{opp_name}_active_buckets", 0.0)
        for z in range(K):
            stats.setdefault(f"latent_pref_{opp_name}_target_z{z}", 0.0)

    log_opponent_targets = bool(getattr(trainer.cfg, "latent_preference_log_opponent_targets", False))
    if log_opponent_targets and preference.mask.any():
        valid_logits = trainer.model.strategy_logits(batch.states, selector_hidden=batch.selector_hidden)
        valid_logits = valid_logits[preference.mask]
        valid_targets = preference.target_probs[preference.mask]
        valid_log_probs = torch.log_softmax(valid_logits, dim=-1)
        kl_per_episode = F.kl_div(valid_log_probs, valid_targets, reduction="none").sum(dim=-1)
        valid_opps = batch.opponent_ids[preference.mask]
        for opp_name, opp_id in resolve_logged_opponents(trainer.cfg):
            opp_mask = batch.opponent_ids == opp_id
            opp_active_mask = opp_mask & preference.mask
            opp_active_count = int(opp_active_mask.sum().item())
            opp_episodes_count = int(opp_mask.sum().item())
            stats[f"latent_pref_{opp_name}_active_fraction"] = (
                float(opp_active_count) / opp_episodes_count if opp_episodes_count > 0 else 0.0
            )
            opp_keys_in_batch = [k for k in preference.unique_keys if (k // 256) == opp_id]
            stats[f"latent_pref_{opp_name}_active_buckets"] = float(
                sum(1 for k in opp_keys_in_batch if preference.key_to_target_probs.get(k) is not None)
            )
            if opp_active_count > 0:
                opp_valid_mask = valid_opps == opp_id
                stats[f"latent_pref_{opp_name}_loss"] = float(kl_per_episode[opp_valid_mask].mean().item())
                opp_valid_targets = valid_targets[opp_valid_mask]
                entropy_per_episode = -(opp_valid_targets * torch.log(opp_valid_targets + 1e-12)).sum(dim=-1)
                stats[f"latent_pref_{opp_name}_target_entropy"] = float(entropy_per_episode.mean().item())
                opp_mean_targets = opp_valid_targets.mean(dim=0)
                for z_idx in range(K):
                    stats[f"latent_pref_{opp_name}_target_z{z_idx}"] = float(opp_mean_targets[z_idx].item())
                stats[f"latent_pref_{opp_name}_best_z"] = float(opp_mean_targets.argmax().item())
    return stats
