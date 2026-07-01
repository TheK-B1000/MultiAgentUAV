"""Episode PPO telemetry aggregation semantics."""

from __future__ import annotations

from dataclasses import dataclass, field  # noqa: F401 — kept for future telemetry dataclasses

from rl.custom_ppo.latent.opponent_telemetry import resolve_logged_opponents


def empty_episode_strategy_stats(latent_k: int = 4) -> dict[str, float]:
    res = {
        "latent_preference_loss": 0.0,
        "latent_preference_active_fraction": 0.0,
        "latent_preference_buffer_size": 0.0,
        "latent_preference_num_active_buckets": 0.0,
        "latent_preference_target_entropy": 0.0,
        "latent_awrd_loss": 0.0,
        "latent_awrd_coef_scale": 0.0,
        "latent_awrd_active_fraction": 0.0,
        "latent_awrd_active_buckets": 0.0,
        "latent_awrd_buffer_size": 0.0,
        "latent_awrd_target_entropy": 0.0,
        "latent_awrd_margin_mean": 0.0,
        "latent_awrd_wr_spread_mean": 0.0,
        "latent_awrd_best_z_mean": -1.0,
        "latent_awrd_effective_coef_mean": 0.0,
        "latent_awrd_best_z_match_rate": 0.0,
        "latent_specialist_loss": 0.0,
        "latent_specialist_marginal_entropy": 0.0,
        "latent_specialist_conditional_entropy": 0.0,
        "latent_specialist_context_bucket_entropy": 0.0,
        "latent_specialist_mi": 0.0,
        "latent_specialist_context_mi": 0.0,
        "latent_specialist_active_buckets": 0.0,
        "latent_specialist_coef_scale": 0.0,
        "latent_specialist_rollout_samples": 0.0,
        "latent_episode_pg_loss": 0.0,
        "latent_episode_v_loss": 0.0,
        "latent_episode_entropy": 0.0,
        "latent_episode_adv_mean": 0.0,
        "latent_episode_adv_std": 0.0,
        "latent_episode_return_mean": 0.0,
        "latent_episode_return_std": 0.0,
        "latent_episode_ratio_mean": 0.0,
        "latent_episode_ratio_max": 0.0,
        "latent_episode_ratio_min": 0.0,
        "latent_episode_ratio_std": 0.0,
        "latent_episode_approx_kl": 0.0,
        "latent_episode_clip_fraction": 0.0,
        "latent_episode_count": 0.0,
        "latent_episode_loss": 0.0,
        "strategy_entropy_resample_mean": 0.0,
        "qphi_margin_resample_mean": 0.0,
        "episode_credit_grad_norm": 0.0,
        "episode_credit_adv_mean": 0.0,
        "episode_credit_adv_std": 0.0,
        "bucket_baseline_count": 0.0,
        "bucket_baseline_fallback_frac": 0.0,
        "bucket_baseline_var_reduction": 1.0,
        "bucket_baseline_global_mean": 0.0,
        "bucket_baseline_raw_return_std": 0.0,
        "bucket_baseline_adv_std": 0.0,
        "latent_usage_balance_loss": 0.0,
        "latent_usage_balance_kl": 0.0,
        "latent_q_phi_train_active": 0.0,
        "latent_episode_epochs_completed": 0.0,
        "latent_episode_early_stop": 0.0,
        "latent_episode_early_stop_kl": 0.0,
        "latent_episode_optimizer_steps": 0.0,
    }
    for opp_name, _opp_id in resolve_logged_opponents(None):
        res[f"latent_pref_{opp_name}_loss"] = 0.0
        res[f"latent_pref_{opp_name}_active_fraction"] = 0.0
        res[f"latent_pref_{opp_name}_target_entropy"] = 0.0
        res[f"latent_pref_{opp_name}_best_z"] = -1.0
        res[f"latent_pref_{opp_name}_buffer_count"] = 0.0
        res[f"latent_pref_{opp_name}_active_buckets"] = 0.0
        for z in range(latent_k):
            res[f"latent_pref_{opp_name}_target_z{z}"] = 0.0
    res.update(
        {
            "latent_v3i3_event_pref_loss": 0.0,
            "latent_v3i3_event_pref_active_fraction": 0.0,
            "latent_v3i3_event_pref_active_buckets": 0.0,
            "latent_v3i3_event_pref_active_records": 0.0,
            "latent_v3i3_event_pref_buffer_size": 0.0,
            "latent_v3i3_event_pref_target_entropy": 0.0,
            "latent_v3i3_event_pref_fallback_full": 0.0,
            "latent_v3i3_event_pref_fallback_oef": 0.0,
            "latent_v3i3_event_pref_fallback_oe": 0.0,
            "latent_v3i3_event_pref_fallback_o": 0.0,
            "latent_v3i3_event_pref_rollout_records": 0.0,
        }
    )
    return res


from rl.custom_ppo.latent.optimization.ppo_stats import EpisodeStatsAccumulator  # noqa: F402
