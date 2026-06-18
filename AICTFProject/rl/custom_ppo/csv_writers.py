from __future__ import annotations

import csv
import os
from typing import Any, Optional

from rl.behavior_telemetry import BEHAVIOR_TELEMETRY_NAMES
from rl.forced_z_behavior_vectors import FORCED_Z_BEHAVIOR_VECTOR_NAMES, OPPORTUNITY_MAX_CELLS_REPORTED
from rl.latent_phase_labels import TEAM_PHASES
from rl.latent_marl import CONTEXT_STATE_DIM

E3_STEP_TELEMETRY_FIELDS: tuple[str, ...] = (
    "update",
    "rollout_step",
    "env_id",
    "global_step",
    "z_t",
    "q_phi_entropy",
    "q_phi_argmax",
    "switched",
    "game_phase",
    "team_phase",
    "score_outcome",
    "stalemate_frac",
    "opponent_id",
    "phase_id",
    "blue_ahead",
) + BEHAVIOR_TELEMETRY_NAMES + (
    "spread_bucket",
    "role_bucket",
    "pressure_bucket",
    "attack_defense_ratio_bucket",
    "qlogit_0",
    "qlogit_1",
    "qlogit_2",
    "qlogit_3",
    "qprob_0",
    "qprob_1",
    "qprob_2",
    "qprob_3",
    "strategy_entropy",
    "strategy_entropy_frac",
) + tuple(f"q_phi_context_{i}" for i in range(CONTEXT_STATE_DIM))

# When renaming metrics columns, old CSV headers may still use the legacy name; see ``_write_csv_row``.
_METRICS_CSV_LEGACY_COLUMN_FILL: dict[str, str] = {"strategy_aux_return_loss": "strategy_q_loss"}

# v6i1 staged-curriculum intervention telemetry (six unordered z-pairs for K=4).
V6I1_INTERVENTION_PAIR_COUNT: int = 6

# Columns for MI(z; opponent) and episode_opp{idx}_z* (OP1 … OP5_RUSHER, OP6, OP7).
SCRIPTED_OPPONENT_MI_COUNT: int = 7


def _opponent_id_int_from_info(cfg: Any, info: dict[str, Any]) -> int:
    """Scripted opponent index for MI telemetry: OP1→0 … OP7→6; ``-1`` if unknown / non-scripted."""
    er = info.get("episode_result") if isinstance(info.get("episode_result"), dict) else {}
    kind = str(er.get("opponent_kind", info.get("opponent_kind", "scripted")) or "scripted").lower()
    if kind != "scripted":
        return -1
    tag_raw = str(
        er.get("scripted_tag")
        or info.get("opponent_key", getattr(cfg, "fixed_opponent_tag", "OP3"))
        or ""
    ).strip().upper()
    tag = "OP5_RUSHER" if tag_raw == "OP5" else tag_raw
    if tag == "OP6_TURTLE":
        tag = "OP6"
    if tag == "OP7_SWITCHER":
        tag = "OP7"
    return {"OP1": 0, "OP2": 1, "OP3": 2, "OP4": 3, "OP5_RUSHER": 4, "OP6": 5, "OP7": 6}.get(tag, -1)


def _opponent_id_csv_from_info(cfg: Any, info: dict[str, Any]) -> str:
    oid = _opponent_id_int_from_info(cfg, info)
    return str(int(oid)) if oid >= 0 else ""


# Inverse of the ``_opponent_id_int_from_info`` lookup table. Kept in sync
# manually -- if a new opponent tag is added there, add the matching entry
# here as well. Used by diagnostics that need to print human-readable
# opponent labels alongside or in place of the raw integer id.
_OPPONENT_ID_TO_TAG: dict[int, str] = {
    0: "OP1",
    1: "OP2",
    2: "OP3",
    3: "OP4",
    4: "OP5",
    5: "OP6",
    6: "OP7",
}


def _opponent_tag_from_id(opponent_id: int) -> str:
    """Map an opponent-id integer back to its public OP tag.

    Returns ``OP{N}`` for known ids and ``op{N}`` (lowercase fallback)
    for unmapped ids so the caller can still print something stable.
    """
    if opponent_id in _OPPONENT_ID_TO_TAG:
        return _OPPONENT_ID_TO_TAG[opponent_id]
    return f"op{opponent_id}"


def _opponent_legend(cfg: Any, info: dict[str, Any]) -> str:
    """Compact opponent string for logging (scripted:OP3, snapshot:name, ...)."""
    er = info.get("episode_result") or {}
    kind = str(er.get("opponent_kind", info.get("opponent_kind", "scripted")) or "scripted").lower()
    if kind == "scripted":
        tag = str(er.get("scripted_tag") or info.get("opponent_key", getattr(cfg, "fixed_opponent_tag", "OP3")))
        return f"SCRIPTED:{str(tag).upper()}"
    if kind == "snapshot":
        snap = str(er.get("opponent_snapshot", "") or info.get("opponent_key", ""))
        return f"SNAPSHOT:{snap}" if snap else "SNAPSHOT:unknown"
    return f"{kind.upper()}:?"


def _episode_fieldnames() -> list[str]:
    return [
        "episode_id",
        "run_id",
        "run_pid",
        "timesteps",
        "policy_update",
        "rollout_step",
        "latent_z",
        "curriculum_phase",
        "mode",
        "map_set",
        "map_layout",
        "map_vertical_mirror",
        "opponent",
        "opponent_id",
        "success",
        "blue_score",
        "red_score",
        "win_margin",
        "decision_steps",
        "zone_coverage",
        "collision_free_episode",
        "collision_events_per_episode",
        "obstacle_collision_events_per_episode",
        "near_misses_per_episode",
        "blue_route_upper_crossings",
        "blue_route_lower_crossings",
        "red_route_upper_crossings",
        "red_route_lower_crossings",
        "blue_attack_upper_crossings",
        "blue_attack_lower_crossings",
        "blue_return_upper_crossings",
        "blue_return_lower_crossings",
        "blue_intercept_upper_crossings",
        "blue_intercept_lower_crossings",
        "red_attack_upper_crossings",
        "red_attack_lower_crossings",
        "red_return_upper_crossings",
        "red_return_lower_crossings",
        "red_intercept_upper_crossings",
        "red_intercept_lower_crossings",
        "time_to_first_score",
        "mean_inter_robot_dist",
        "reward_terminal",
        "reward_offense",
        "reward_pbrs",
        "reward_team",
        "reward_sparse",
        "reward_sparse_points",
        "reward_failure",
        "reward_total",
    ]


def _update_fieldnames(use_latent_strategy: bool, latent_k: int) -> list[str]:
    fields = [
        "update",
        "run_id",
        "run_pid",
        "timesteps",
        "episodes_completed",
        "wins",
        "losses",
        "draws",
        "win_rate",
        "rolling_win_rate_50ep",
        "rolling_win_rate_200ep",
        "rollout_reward_mean",
        "rollout_reward_std",
        "rollout_return_mean",
        "rollout_return_std",
        "rollout_episodes",
        "rollout_wins",
        "rollout_losses",
        "rollout_draws",
        "rollout_win_rate",
        "rollout_win_margin_mean",
        "rollout_blue_score_mean",
        "rollout_red_score_mean",
        "explained_variance",
        "reward_terminal_mean",
        "reward_offense_mean",
        "reward_pbrs_mean",
        "reward_team_mean",
        "reward_sparse_mean",
        "reward_sparse_points_mean",
        "reward_failure_mean",
        "reward_behavior_contrast_mean",
        "reward_csia_mean",
        "reward_total_mean",
        "reward_outcome_mean",
        "reward_shaping_mean",
        "reward_shaping_to_outcome_abs_ratio",
        "reward_shaping_coef",
        "reward_failure_to_outcome_abs",
        "policy_loss",
        "value_loss",
        "value_loss_min",
        "value_loss_std",
        "value_loss_p10",
        "value_loss_p50",
        "value_loss_p90",
        "value_loss_max",
        "return_norm_mean",
        "return_norm_std",
        "return_norm_count",
        "entropy",
        "approx_kl",
        "clip_fraction",
        "grad_norm",
        "learning_rate",
        "latent_lam_h",
        "strategy_entropy",
        "strategy_entropy_frac",
        "strategy_policy_loss",
        "strategy_approx_kl",
        "strategy_clip_fraction",
        "strategy_ratio_std",
        "strategy_aux_return_loss",
        "strategy_persist_loss",
        "strategy_grad_norm",
        "strategy_resample_count",
        "strategy_resample_fraction",
        "latent_episode_pg_loss",
        "latent_episode_v_loss",
        "latent_episode_entropy",
        "latent_episode_adv_mean",
        "latent_episode_adv_std",
        "latent_episode_return_mean",
        "latent_episode_return_std",
        "latent_episode_ratio_mean",
        "latent_episode_ratio_max",
        "latent_episode_ratio_min",
        "latent_episode_ratio_std",
        "latent_episode_approx_kl",
        "latent_episode_clip_fraction",
        "latent_episode_count",
        "latent_episode_loss",
        "strategy_bucket_best_match_frac",
        "strategy_experience_records",
        "strategy_experience_buckets",
        "strategy_unique_count",
        "strategy_dominant",
        "strategy_switch_count",
        "strategy_switch_fraction",
        "strategy_wr_spread",
        "strategy_resample_fraction_rollout",
        "rollout_adv_std",
        "rollout_adv_std_at_z_switch",
        "rollout_adv_std_not_z_switch",
        "curriculum_phase",
        "curriculum_phase_idx",
        "curriculum_phase_episodes",
        "curriculum_phase_win_rate",
        "csia_interaction_strength",
        "centered_advantage_matrix",
        "oracle_best_z_per_opponent",
        "router_oracle_gap",
        "routing_gain",
        "regret_weighted_routing_score",
        "gate_A_pass",
        "gate_B_pass",
        "gate_C_pass",
        "csia_bonus_active",
        "csia_payoff_cells",
        "csia_total_count",
        "csia_behavior_spread_max",
        "csia_bonus_mean",
        "csia_last_refresh_update",
        "csia_reward_coef",
    ]
    if use_latent_strategy:
        fields.append("strategy_kl")
        fields.append("policy_z_sensitivity_KL")
        fields.append("z_sensitivity_KL")
        fields.append("z_sep_JSD")
        fields.append("actor_z_jsd_mean")
        fields.append("actor_z_jsd_max")
        fields.append("actor_z_jsd_per_head")
        fields.append("actor_z_argmax_disagree")
        fields.append("actor_z_logit_l2")
        fields.append("actor_z_entropy_by_z")
        fields.append("actor_z_jsd_head_0")
        fields.append("actor_z_jsd_head_1")
        fields.extend(f"actor_z_entropy_z{idx}" for idx in range(latent_k))
        fields.append("actor_input_dim")
        fields.append("z_embed_dim")
        fields.extend(f"strategy_occupancy_{idx}" for idx in range(latent_k))
        # v5i5 occupancy-collapse diagnostics (added with the entropy-floor
        # follow-up to v5i4; pure functions of existing per-z counts, no
        # new gradient channel). See ``_latent_rollout_stats`` in
        # ``rl/custom_ppo/latent_diagnostics.py``.
        fields.append("latent_marginal_entropy_nats")
        fields.append("effective_num_latents")
        fields.append("latent_occupancy_min")
        fields.append("latent_occupancy_max")
        fields.append("latent_occupancy_ratio")
        fields.append("mean_strategy_duration")
        for idx in range(latent_k):
            fields.extend(
                [
                    f"episode_z_{idx}_count",
                    f"episode_z_{idx}_win_rate",
                    f"episode_z_{idx}_blue_score_mean",
                    f"episode_z_{idx}_red_score_mean",
                    f"episode_z_{idx}_win_margin_mean",
                ]
            )
        for idx in range(latent_k):
            fields.extend(
                [
                    f"strategy_resample_adv_mean_z{idx}",
                    f"strategy_resample_adv_std_z{idx}",
                    f"strategy_resample_adv_n_z{idx}",
                ]
            )
        # v5i3 per-z router telemetry. Populated by apply_episode_strategy_ppo;
        # absent or zero under v5_strict_summer (router PPO disabled).
        for idx in range(latent_k):
            fields.extend(
                [
                    f"router_sample_count_by_z_{idx}",
                    f"forced_sample_count_by_z_{idx}",
                    f"episode_count_by_z_{idx}",
                    f"mean_episode_advantage_by_z_{idx}",
                    f"std_episode_advantage_by_z_{idx}",
                    f"mean_return_by_z_{idx}",
                    f"mean_logprob_ratio_by_z_{idx}",
                    f"clip_fraction_by_z_{idx}",
                ]
            )
        # v5i3 forced-z anneal coefficient (the resolver output at the start
        # of the rollout). Lets the post-mortem plot show the anneal
        # trajectory next to the per-z sample counts.
        fields.append("latent_forced_z_episode_frac_current")
        fields.append("latent_mi_z_opponent_nats")
        fields.append("latent_mi_z_phase_nats")
        fields.append("latent_mi_z_outcome_nats")
        fields.append("latent_mi_z_flag_state_nats")
        fields.append("MI_executed_z_phase")
        fields.append("MI_executed_z_flag")
        fields.append("MI_executed_z_outcome")
        fields.append("latent_z_marginal_entropy_nats")
        fields.append("latent_normalized_mi_z_opponent")
        fields.append("latent_normalized_mi_z_phase")
        fields.append("latent_normalized_mi_z_outcome")
        fields.append("latent_normalized_mi_z_flag_state")
        # v3i19 arc-credit telemetry (zeroed when arc credit is disabled).
        fields.append("latent_arc_count")
        fields.append("latent_arc_finalized_count")
        fields.append("latent_arc_dropped_short_count")
        fields.append("latent_arc_mean_length")
        fields.append("latent_arc_mean_return")
        fields.append("latent_arc_advantage_mean")
        fields.append("latent_arc_advantage_std")
        fields.append("latent_arc_policy_loss")
        fields.append("latent_arc_value_loss")
        fields.append("latent_arc_clipfrac")
        fields.append("latent_arc_approx_kl")
        # v3i19 smoke alarm: gradient flow + q_phi posterior shape. If
        # ``latent_arc_credit_coef > 0`` but ``q_phi_grad_norm`` stays near
        # zero across rollouts, the consequence channel is decorative.
        fields.append("latent_arc_credit_coef")
        fields.append("latent_arc_grad_norm")
        fields.append("q_phi_grad_norm")
        # Split grad-norm diagnostic: where does the arc-credit gradient
        # actually land? Encoder controls pi(z|s); value head is the
        # baseline only. L2 sanity: sqrt(enc^2 + vh^2 + other^2) ~= total.
        fields.append("q_phi_strategy_encoder_grad_norm")
        fields.append("q_phi_value_head_grad_norm")
        fields.append("q_phi_other_grad_norm")
        fields.append("q_phi_entropy")
        fields.append("q_phi_mean_max_prob")
        fields.append("latent_mi_z_spread_bucket_nats")
        fields.append("latent_mi_z_role_bucket_nats")
        fields.append("latent_mi_z_pressure_bucket_nats")
        fields.append("latent_mi_z_attack_defense_ratio_bucket_nats")
        for r in range(5):  # N_ROLE_BUCKET_MI is 5
            for z_idx in range(latent_k):
                fields.append(f"latent_role{r}_z{z_idx}_frac")
            fields.append(f"latent_role{r}_switch_mean")
        fields.extend(
            [
                "latent_switch_rate_blue_ahead",
                "latent_switch_rate_blue_trail",
                "latent_reward_sum_5_after_z_switch_mean",
            ]
        )
        for p in range(len(TEAM_PHASES)):
            for z_idx in range(latent_k):
                fields.append(f"latent_phase{p}_z{z_idx}_frac")
            fields.extend(
                [
                    f"latent_phase{p}_switch_mean",
                    f"latent_phase{p}_blue_ahead_mean",
                    f"latent_phase{p}_capture_step_mean",
                    f"q_phi_phase{p}_entropy_mean",
                ]
            )
            for z_idx in range(latent_k):
                fields.append(f"q_phi_phase{p}_z{z_idx}_prob_mean")
        fields.append("latent_behavior_diversity_l2_mean")
        for z_idx in range(latent_k):
            for name in BEHAVIOR_TELEMETRY_NAMES:
                fields.append(f"latent_z{z_idx}_behavior_{name}_mean")
        fields.append("forced_z_macro_jsd_mean")
        fields.append("forced_z_macro_jsd")
        from rl.custom_ppo.inference import FORCED_Z_MACRO_ACTIONS
        for z_idx in range(latent_k):
            for _action_id, action_name in FORCED_Z_MACRO_ACTIONS:
                fields.append(f"forced_z{z_idx}_macro_{action_name}_prob")
            fields.append(f"forced_z{z_idx}_macro_entropy")
        fields.extend(_forced_z_behavior_metrics_fieldnames(latent_k))
        fields.extend(_phase_a_diagnostic_fieldnames())
        for o_idx in range(SCRIPTED_OPPONENT_MI_COUNT):
            for z_idx in range(latent_k):
                fields.append(f"strategy_occupancy_op{o_idx}_z{z_idx}")
        for o_idx in range(SCRIPTED_OPPONENT_MI_COUNT):
            for z_idx in range(latent_k):
                fields.extend(
                    [
                        f"episode_opp{o_idx}_z{z_idx}_count",
                        f"episode_opp{o_idx}_z{z_idx}_win_rate",
                        f"forced_episode_opp{o_idx}_z{z_idx}_count",
                    ]
                )
        # Append new diagnostic columns
        fields.append("strategy_phase_loss")
        fields.extend([
            "latent_switch_near_capture_frac",
            "latent_switch_near_kill_frac",
            "latent_switch_near_return_frac",
            # Denominator + numerators so a 0.000 fraction is interpretable.
            # Without these, "cap=0.000" conflates "no effect" (eligible > 0,
            # event > 0, near == 0) with "no qualifying events" (either count
            # is 0) -- the latter is dominant under episode-start-only
            # presets like v5i1/v5i2/v5i3 where eligible_count is structurally 0.
            "latent_switch_near_eligible_count",
            "latent_switch_near_capture_count",
            "latent_switch_near_kill_count",
            "latent_switch_near_return_count",
            "latent_capture_event_count",
            "latent_kill_event_count",
            "latent_return_event_count",
        ])
        for f in range(4):
            for k in range(latent_k):
                fields.append(f"latent_flag_state{f}_z{k}_frac")
        for s in range(3):
            for k in range(latent_k):
                fields.append(f"latent_spread{s}_z{k}_frac")
        for a in range(3):
            for k in range(latent_k):
                fields.append(f"latent_adr{a}_z{k}_frac")
        for p in range(len(TEAM_PHASES)):
            fields.append(f"latent_phase{p}_entropy")
        fields.extend([
            "latent_role_diversity",
            "latent_spread_diversity",
            "latent_pressure_diversity",
            "latent_adr_diversity",
        ])
        fields.extend([
            "latent_q_phi_option_advantage_mean",
            "latent_q_phi_option_advantage_std",
            "latent_q_phi_option_advantage_count",
        ])
        fields.extend([
            "strategy_entropy_resample_mean",
            "strategy_marginal_entropy_loss",
            "strategy_marginal_entropy_nats",
            "strategy_marginal_entropy_kl",
            # v5i6 rollout-level (pre-Jensen-bias-fix) router diagnostics.
            # All seven below are computed from a single forward pass over
            # ALL rollout resample-decision points per inner epoch — i.e.
            # the same population the rollout-level KL-to-uniform loss is
            # taken over. They are explicitly distinct from the sampled-z
            # ``latent_marginal_entropy_nats`` / ``latent_occupancy_*``
            # columns, which are one-sample-per-state empirical
            # histograms. Use these for the v5i6 H_marginal / H_conditional
            # / MI_proxy three-pattern decoder.
            "router_rollout_soft_marginal_entropy_nats",
            "router_rollout_soft_conditional_entropy_nats",
            "router_rollout_soft_mi_proxy_nats",
            "router_rollout_soft_argmax_occupancy_max",
            "router_rollout_soft_argmax_occupancy_min",
            "router_rollout_soft_argmax_occupancy_ratio",
            "router_rollout_resample_count",
        ])
        fields.extend(
            f"router_rollout_soft_p_bar_z{z}" for z in range(latent_k)
        )
        fields.extend([
            "qphi_margin_resample_mean",
            "episode_credit_grad_norm",
            "episode_credit_adv_mean",
            "episode_credit_adv_std",
            "bucket_baseline_count",
            "bucket_baseline_fallback_frac",
            "bucket_baseline_var_reduction",
            "bucket_baseline_global_mean",
            "bucket_baseline_raw_return_std",
            "bucket_baseline_adv_std",
            "latent_tactical_bucket_fallback_fraction",
            "latent_forced_z_episode_fraction",
            "latent_forced_z_step_fraction",
            "latent_behavior_contrast_bonus_mean",
            "latent_behavior_contrast_distance_mean",
            "latent_behavior_contrast_active_frac",
            "latent_behavior_contrast_coef",
            "latent_actor_z_separation_loss",
            "latent_actor_z_separation_jsd",
            "latent_actor_z_separation_active",
            "latent_actor_z_separation_train_active",
            "latent_actor_z_separation_coef",
            "latent_actor_z_adapter_scale",
            "latent_usage_balance_loss",
            "latent_usage_balance_kl",
            "main_loop_q_phi_train_active",
            "main_loop_q_phi_grad_norm",
            "latent_q_phi_train_active",
            "latent_preference_loss",
            "latent_preference_active_fraction",
            "latent_preference_buffer_size",
            "latent_preference_num_active_buckets",
            "latent_preference_target_entropy",
            "latent_awrd_loss",
            "latent_awrd_coef_scale",
            "latent_awrd_active_fraction",
            "latent_awrd_active_buckets",
            "latent_awrd_buffer_size",
            "latent_awrd_target_entropy",
            "latent_awrd_margin_mean",
            "latent_awrd_wr_spread_mean",
            "latent_awrd_best_z_mean",
            "latent_awrd_effective_coef_mean",
            "latent_awrd_best_z_match_rate",
            "latent_specialist_loss",
            "latent_specialist_marginal_entropy",
            "latent_specialist_conditional_entropy",
            "latent_specialist_context_bucket_entropy",
            "latent_specialist_conditional_term",
            "latent_specialist_conditional_coef",
            "latent_specialist_mi",
            "latent_specialist_context_mi",
            "latent_specialist_active_buckets",
            "latent_specialist_coef_scale",
            "latent_specialist_rollout_samples",
        ])
        for opp_name in ["op5", "op6"]:
            fields.extend([
                f"latent_pref_{opp_name}_loss",
                f"latent_pref_{opp_name}_active_fraction",
                f"latent_pref_{opp_name}_target_entropy",
                f"latent_pref_{opp_name}_best_z",
                f"latent_pref_{opp_name}_buffer_count",
                f"latent_pref_{opp_name}_active_buckets",
            ])
            fields.extend(f"latent_pref_{opp_name}_target_z{z}" for z in range(latent_k))
        # v3i event refresh telemetry fields
        fields.extend([
            "latent_refresh_count",
            "latent_refresh_rate",
            "latent_refresh_reason_enemy_flag",
            "latent_refresh_reason_friendly_flag",
            "latent_refresh_reason_score_change",
            "latent_refresh_reason_near_base",
            "latent_refresh_z_changed_rate",
            "latent_refresh_changed_z_rate",
            "latent_refresh_same_z_rate",
            "latent_refresh_transition_entropy",
        ])
        for i in range(latent_k):
            for j in range(latent_k):
                fields.append(f"latent_refresh_z{i}_to_z{j}")
        fields.extend([
            "z_change_count",
            "z_dwell_mean",
            "z_refresh_attempt_count",
            "z_refresh_accept_count",
            "z_refresh_reject_dwell_count",
            "z_refresh_reason_interval",
            "z_refresh_reason_flag",
            "z_refresh_reason_phase",
            "z_refresh_reason_score_pressure",
            "q_phi_argmax_vs_executed_z_agreement",
        ])
        # v3i3 event-conditioned preference telemetry. Always present in
        # the schema so disabled runs emit zeros (matches the v3i2-era
        # pattern for latent_preference_*).
        fields.extend([
            "latent_v3i3_event_pref_loss",
            "latent_v3i3_event_pref_active_fraction",
            "latent_v3i3_event_pref_active_buckets",
            "latent_v3i3_event_pref_active_records",
            "latent_v3i3_event_pref_buffer_size",
            "latent_v3i3_event_pref_target_entropy",
            "latent_v3i3_event_pref_fallback_full",
            "latent_v3i3_event_pref_fallback_oef",
            "latent_v3i3_event_pref_fallback_oe",
            "latent_v3i3_event_pref_fallback_o",
            "latent_v3i3_event_pref_rollout_records",
        ])
        fields.extend(_v6i1_metrics_fieldnames())
    return fields


def _forced_z_behavior_metrics_fieldnames(latent_k: int) -> list[str]:
    fields: list[str] = []
    for z_idx in range(latent_k):
        for name in FORCED_Z_BEHAVIOR_VECTOR_NAMES:
            fields.append(f"forced_z{z_idx}_behavior_{name}")
    pair_count = latent_k * (latent_k - 1) // 2
    for idx in range(pair_count):
        fields.append(f"forced_z_behavior_pair_distance_{idx}")
        for name in FORCED_Z_BEHAVIOR_VECTOR_NAMES:
            fields.append(f"forced_z_pair_{name}_distance_{idx}")
    fields.extend(
        [
            "forced_z_behavior_pair_distance_mean",
            "forced_z_behavior_pair_distance_max",
            "forced_z_behavior_pair_distance_min",
            "forced_z_behavior_pairs_above_threshold",
            "forced_z_behavior_all_z_represented",
            "forced_z_behavior_components_valid",
        ]
    )
    for name in FORCED_Z_BEHAVIOR_VECTOR_NAMES:
        fields.append(f"behavior_component_scale_{name}")
        fields.append(f"behavior_component_valid_{name}")
    return fields


def _phase_a_diagnostic_fieldnames() -> list[str]:
    latent_k = 4
    fields = [
        "phase_a_stats_source_step",
        "phase_a_actor_jsd_mean",
        "phase_a_actor_jsd_slope_20",
        "phase_a_cf_actor_jsd_mean",
        "phase_a_behavior_distance_mean",
        "phase_a_behavior_distance_min",
        "phase_a_behavior_distance_slope_20",
        "phase_a_intervention_quadrant",
        "phase_a_intervention_quadrant_name",
        "phase_a_cf_regime",
        "phase_a_cf_regime_name",
        "phase_a_competence_min",
        "phase_a_cf_ratio",
        "phase_a_competence_floor_pass",
        "phase_a_cf_ratio_in_band",
        "phase_a_actor_intervention_trending_up",
        "phase_a_behavioral_realization_trending_up",
        "phase_a_corridor_viable",
        "phase_a_behavior_measurement_valid",
        "phase_a_actor_jsd_valid_updates",
        "phase_a_behavior_valid_updates",
        "phase_a_actor_pairs_above_margin",
        "phase_a_actor_weakest_pair_jsd",
        "phase_a_actor_pair_gate_pass",
        "phase_a_behavior_pairs_above_threshold",
        "phase_a_behavior_weakest_pair_distance",
        "phase_a_behavior_pair_gate_pass",
        "opportunity_cell_count",
        "opportunity_eligible_cell_count",
        "opportunity_fork_fraction",
        "opportunity_fork_fraction_valid",
        "opportunity_homogeneous_fraction",
        "opportunity_best_z_unique",
        "opportunity_measurement_valid",
        "opportunity_fork_fraction_forced",
        "opportunity_fork_fraction_valid_forced",
        "opportunity_best_z_unique_forced",
    ]
    for c in range(OPPORTUNITY_MAX_CELLS_REPORTED):
        for z in range(latent_k):
            fields.append(f"opportunity_cell_{c}_count_z{z}")
            fields.append(f"opportunity_cell_{c}_return_mean_z{z}")
            fields.append(f"opportunity_cell_{c}_return_se_z{z}")
        fields.append(f"opportunity_cell_{c}_best_margin")
        fields.append(f"opportunity_cell_{c}_eligible")
    return fields


def _v6i1_metrics_fieldnames() -> list[str]:
    """Per-update v6i1 curriculum / intervention-gate columns (zeroed off v6i1)."""
    fields = [
        "v6i1_phase",
        "v6i1_phase_label",
        "v6i1_cf_coef_current",
        "v6i1_usage_coef_current",
        "jsd_pairs_above_margin",
        "jsd_min_pair",
        "jsd_gate_update_pass",
        "jsd_gate_consecutive_updates",
        "jsd_gate_consecutive_required",
        "cf_competence_ready",
        "cf_competence_z0",
        "cf_competence_z1",
        "cf_competence_z2",
        "cf_competence_z3",
        "pairwise_profile_available",
        "pairwise_ema_valid_updates",
        "pairwise_ema_last_update_step",
        "cf_actor_grad_norm",
        "ppo_actor_grad_norm",
        "cf_to_ppo_grad_ratio",
        "cf_batch_pairs_below_margin",
        "cf_hinge_active",
        "cf_hinge_effective",
        "cf_valid_team_groups",
        "cf_weight_sum",
        "cf_effective_pairs",
        "cf_loss_requires_grad",
        "latent_actor_z_separation_jsd_min",
        "latent_actor_z_separation_jsd_max",
    ]
    pair_suffixes = ("01", "02", "03", "12", "13", "23")
    for idx in range(V6I1_INTERVENTION_PAIR_COUNT):
        fields.append(f"forced_z_pair_jsd_{idx}")
        fields.append(f"pair_jsd_ema_{idx}")
        fields.append(f"cf_batch_pair_jsd_{idx}")
    for suffix in pair_suffixes:
        fields.append(f"forced_z_pair_jsd_{suffix}")
        fields.append(f"pair_jsd_ema_{suffix}")
        fields.append(f"cf_batch_pair_jsd_{suffix}")
    fields.extend(_v6i2_gate_metrics_fieldnames())
    fields.extend(_v6i3_comm_metrics_fieldnames())
    return fields


def _v6i3_comm_metrics_fieldnames() -> list[str]:
    fields = [
        "comm_valid_boundaries",
        "comm_send_count",
        "comm_delivery_count",
        "comm_dropout_count",
        "comm_no_receiver_count",
        "comm_symbol_entropy",
        "comm_symbol_entropy_normalized",
        "comm_symbols_used",
        "comm_symbol_dominance",
        "comm_message_logprob_mean",
        "mi_message_z",
        "mi_message_phase",
        "mi_message_role",
        "mi_message_next_macro_action",
        "receiver_action_jsd_by_message_pair_mean",
        "receiver_argmax_disagreement_frac",
        "receiver_listener_pairs",
        "communication_usage_status",
        "listener_causal_response_status",
    ]
    for i in range(4):
        fields.append(f"comm_symbol_occupancy_{i}")
    return fields


def _v6i2_gate_metrics_fieldnames() -> list[str]:
    """v6i2 dual-gate telemetry (zeroed unless gate_protocol_version is v6i2)."""
    fields = [
        "cf_pair_jsd_valid_updates",
        "cf_pair_jsd_last_update_step",
        "macro_pair_jsd_valid_updates",
        "macro_pair_jsd_last_update_step",
        "actor_intervention_consecutive_updates",
        "actor_intervention_gate_update_pass",
        "actor_intervention_consecutive_required",
        "cf_pairs_above_actor_margin",
        "cf_min_pair_ema",
    ]
    pair_suffixes = ("01", "02", "03", "12", "13", "23")
    for idx in range(V6I1_INTERVENTION_PAIR_COUNT):
        fields.append(f"cf_pair_jsd_{idx}")
        fields.append(f"cf_pair_jsd_ema_{idx}")
        fields.append(f"macro_pair_jsd_{idx}")
        fields.append(f"macro_pair_jsd_ema_{idx}")
    for suffix in pair_suffixes:
        fields.append(f"cf_pair_jsd_{suffix}")
        fields.append(f"cf_pair_jsd_ema_{suffix}")
        fields.append(f"macro_pair_jsd_{suffix}")
        fields.append(f"macro_pair_jsd_ema_{suffix}")
    return fields


def _strategy_experience_fieldnames() -> list[str]:
    return [
        "update",
        "run_id",
        "run_pid",
        "timesteps",
        "bucket_id",
        "z",
        "count",
        "bucket_count",
        "mean_return",
        "win_rate",
        "q_phi_prob_mean",
        "chosen_freq",
        "best_z",
        "best_z_match_frac",
    ]


# v3i3 per-refresh proof-layer log. One CSV row per finalized refresh
# event. The "reason" / event_type integer encoding follows the priority
# used by ``LatentStrategyState.strategy_for_step``:
#     0 = enemy_flag, 1 = friendly_flag, 2 = score_change, 3 = near_base
# (priority left-to-right when multiple triggers fire on the same step).
# ``flag_state_bucket`` is ``2 * enemy_has_our_flag + we_have_enemy_flag``
# (0..3) computed from the global state at the refresh moment.
V3I3_REFRESH_REASON_LABELS: tuple[str, ...] = (
    "enemy_flag",
    "friendly_flag",
    "score_change",
    "near_base",
)


def _v3i3_refresh_log_fieldnames() -> list[str]:
    return [
        "update",
        "run_id",
        "run_pid",
        "timesteps",
        "env_id",
        "episode_id",
        "decision_step",
        "reason_id",
        "reason",
        "prev_z",
        "next_z",
        "opponent_id",
        "flag_state_bucket",
        "carrier_progress_bucket",
        "return_at_refresh",
        "return_from_now_to_end",
    ]


def _write_csv_row(
    path: str,
    fieldnames: list[str],
    row: dict[str, Any],
    *,
    legacy_column_fill: Optional[dict[str, str]] = None,
) -> None:
    """Append one row with a stable header; used for long-run audit telemetry."""
    if not path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    legacy_fill = legacy_column_fill or {}
    nonempty = os.path.isfile(path) and os.path.getsize(path) > 0
    if nonempty:
        with open(path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            old_fields = reader.fieldnames
            if old_fields is None:
                raise ValueError(f"CSV schema mismatch for {path!r}: empty or invalid header.")
            old_list = list(old_fields)
            old_rows = list(reader)
        if old_list != fieldnames:
            dropped = [c for c in old_list if c not in fieldnames]
            if dropped:
                allowed_old = set(legacy_fill.values())
                if not (legacy_fill and set(dropped).issubset(allowed_old)):
                    raise ValueError(
                        f"CSV schema mismatch for {path!r}: existing columns dropped or renamed "
                        f"{dropped!r}; existing header {old_list!r} vs expected {fieldnames!r}. "
                        "Use a new output path or migrate manually."
                    )
            print(
                f"[PPO] Migrating CSV (additive columns): {path}\n"
                f"      was {len(old_list)} cols -> now {len(fieldnames)} cols; "
                f"rewriting {len(old_rows)} row(s)."
            )
            with open(path, "w", newline="", encoding="utf-8") as wf:
                writer = csv.DictWriter(wf, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                for r in old_rows:
                    out_row: dict[str, Any] = {}
                    for k in fieldnames:
                        v = r.get(k, "")
                        if v == "" and k in legacy_fill:
                            v = r.get(legacy_fill[k], "")
                        out_row[k] = v
                    writer.writerow(out_row)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not nonempty:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def _ensure_additive_csv_header(path: str, fieldnames: list[str]) -> None:
    """Rewrite CSV when new columns are appended (additive-only; never drop old columns)."""
    if not path or not (os.path.isfile(path) and os.path.getsize(path) > 0):
        return
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return
        old_list = list(reader.fieldnames)
        old_rows = list(reader)
    if old_list == fieldnames:
        return
    dropped = [c for c in old_list if c not in fieldnames]
    if dropped:
        raise ValueError(
            f"E3 telemetry CSV schema mismatch for {path!r}: cannot drop columns {dropped!r}. "
            f"Use a new --e3 path or migrate manually."
        )
    if len(fieldnames) <= len(old_list):
        return
    print(
        f"[PPO] Migrating E3 step CSV (additive columns): {path}\n"
        f"      was {len(old_list)} cols -> now {len(fieldnames)} cols; "
        f"rewriting {len(old_rows)} row(s).",
        flush=True,
    )
    with open(path, "w", newline="", encoding="utf-8") as wf:
        writer = csv.DictWriter(wf, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in old_rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})
