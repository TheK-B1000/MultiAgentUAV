"""Registry and resolution entry points for training presets."""

from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from rl.train_ppo import PPOConfig

from rl.presets.plan_faithful import (
    apply_plan_faithful_latent,
    apply_plan_faithful_latent_no_persistence,
    apply_plan_faithful_latent_strategic,
    apply_plan_faithful_latent_step6,
    apply_plan_faithful_latent_episode_strategic,
    apply_plan_faithful_latent_v3b_marginal,
    apply_plan_faithful_latent_v3c_router_lr,
    apply_plan_faithful_latent_v3d_smart_router,
    apply_plan_faithful_latent_v3d_delayed_anneal,
    apply_plan_faithful_latent_v3e_strong_z_actor,
    apply_plan_faithful_latent_v3f_behavior_contrast,
    apply_plan_faithful_latent_v3g_preference,
    apply_plan_faithful_latent_v3h_balanced_preference,
    apply_plan_faithful_latent_v3h2_balanced_preference,
    apply_plan_faithful_latent_v3i_event_refresh,
    apply_plan_faithful_latent_v3i2_router_signal,
    apply_plan_faithful_latent_v3i3_event_conditioned_preference,
    apply_plan_faithful_latent_v3i4_event_progress_preference,
    apply_plan_faithful_latent_v3i5_crisp_router,
    apply_plan_faithful_latent_v3i6_stronger_actor_contrast,
    apply_plan_faithful_latent_v3i7_advantage_weighted_router_distill,
    apply_plan_faithful_latent_v3i8_commander_lockin,
    apply_plan_faithful_latent_v3i9_specialist_router,
    apply_plan_faithful_latent_v3i10_role_phase_specialist,
    apply_plan_faithful_latent_v3i11_z_reactive_actor_adapters,
    apply_plan_faithful_latent_v3i12_faithful_z_pressure,
    apply_plan_faithful_latent_v3i13_strict_faithful_z,
    apply_plan_faithful_latent_v3i14_specialized_faithful_z,
    apply_plan_faithful_latent_v3i14_tuned,
    apply_plan_faithful_latent_v3i15_strong_separation,
    apply_plan_faithful_latent_v3i15_sparse_tactical_refresh,
    apply_plan_faithful_latent_v3i16_policy_z_embedding,
    apply_plan_faithful_latent_v3i17_episode_arc,
    apply_plan_faithful_latent_v3i17_long_arc,
    apply_plan_faithful_latent_v3i18_v3i16_plus_128,
    apply_plan_faithful_latent_v3i19_summer_consequence,
    apply_plan_faithful_latent_v4i1_strategic_pressure_qprobe,
    apply_plan_faithful_latent_v4i3_summer_proof,
    apply_plan_faithful_no_latent_v4i3_baseline,
    apply_plan_faithful_latent_v5_strict_summer,
    apply_plan_faithful_latent_v5i1_reward_credit_router,
    apply_plan_faithful_latent_v5i2_stronger_z_conditioning,
    apply_plan_faithful_latent_v5i3_balanced_warmup,
    apply_plan_faithful_latent_v5i4_end_to_end,
    apply_plan_faithful_latent_v5i5_paper_faithful_entropy_floor,
    apply_plan_faithful_latent_v5i6_paper_faithful_marginal_entropy,
    apply_plan_faithful_latent_v5i7_entropy_floor_split_lane,
    apply_plan_faithful_latent_v5i8_split_lane_v2_task_pressure,
    apply_plan_faithful_latent_v5i8_repertoire_uniform_z,
    apply_plan_faithful_latent_v5i9_csia_guided_specialization,
    apply_plan_faithful_latent_v6i1_staged_team_intent_curriculum,
    apply_plan_faithful_latent_v6i2_staged_team_intent_curriculum,
    apply_plan_faithful_latent_v6i5_corrected_team_intent_curriculum,
    apply_plan_faithful_latent_v6i6_strategy_expansion,
    apply_plan_faithful_latent_v6i4_router_ablation_protocol,
    apply_plan_faithful_latent_v6i3_strategy_local_comm,
    apply_plan_faithful_latent_v6i1_repertoire_only_ablation,
    apply_plan_faithful_latent_v4i4post_periodic_router_distill,
    apply_plan_faithful_latent_no_entropy,
    apply_plan_faithful_latent_phase1_coupling,
    apply_plan_faithful_latent_phase2_credit,
    apply_plan_faithful_latent_phase3_reward_geometry,
    apply_plan_faithful_latent_phase3b_outcome_clean,
    apply_plan_faithful_latent_phase3b_ablate_k1,
    apply_plan_faithful_latent_phase3b_ablate_no_persistence,
    apply_plan_faithful_latent_phase4a_rescue,
    apply_plan_faithful_latent_phase4a_rescue_hardpool,
    apply_plan_faithful_latent_episode_z_clean,
    apply_plan_faithful_latent_option_a_episode_credit,
    apply_plan_faithful_latent_option_a,
    apply_plan_faithful_latent_k1,
    apply_plan_faithful_no_latent,
    apply_plan_option_a,
    apply_plan_option_b_lamp,
    apply_latent_a1_plan_faithful,
)
from rl.presets.hypothesis import (
    apply_hypothesis_flat_opprand,
    apply_hypothesis_latent_opprand_optiona,
    apply_hypothesis_latent_opprand_optionb_lamp_coef05,
    apply_hypothesis_latent_opprand_optionb_no_lamp,
    apply_hypothesis_latent_opprand_optionb_coef03,
    apply_hypothesis_flat_opprand_op35,
    apply_hypothesis_latent_opprand_optionb_lamp_coef05_op35,
)
from rl.presets.other import (
    apply_latent_op3_push80_1m,
    apply_latent_train80_op3_1m,
    apply_latent_op3_wrmax_1m,
    apply_latent_op3_wrmax_train_2m,
)

PRESET_REGISTRY = {
    # Plan-faithful family
    "plan_faithful_latent": apply_plan_faithful_latent,
    "plan_faithful_latent_persist_entropy": apply_plan_faithful_latent,
    "latent_plan_faithful": apply_plan_faithful_latent,
    "latent_plan_faithful_persist_entropy": apply_plan_faithful_latent,
    "latent_recommended": apply_plan_faithful_latent,
    "plan_faithful_latent_no_persistence": apply_plan_faithful_latent_no_persistence,
    "latent_plan_faithful_no_persistence": apply_plan_faithful_latent_no_persistence,
    "latent_recommended_no_persistence": apply_plan_faithful_latent_no_persistence,
    "plan_faithful_latent_strategic": apply_plan_faithful_latent_strategic,
    "latent_strategic": apply_plan_faithful_latent_strategic,
    "plan_faithful_latent_option_credit": apply_plan_faithful_latent_strategic,
    "plan_faithful_latent_step6": apply_plan_faithful_latent_step6,
    "latent_step6": apply_plan_faithful_latent_step6,
    "plan_faithful_latent_episode_strategic": apply_plan_faithful_latent_episode_strategic,
    "latent_episode_strategic": apply_plan_faithful_latent_episode_strategic,
    "plan_faithful_latent_v3b_marginal": apply_plan_faithful_latent_v3b_marginal,
    "latent_v3b_marginal": apply_plan_faithful_latent_v3b_marginal,
    "plan_faithful_latent_v3c_router_lr": apply_plan_faithful_latent_v3c_router_lr,
    "latent_v3c_router_lr": apply_plan_faithful_latent_v3c_router_lr,
    "plan_faithful_latent_v3c": apply_plan_faithful_latent_v3c_router_lr,
    "latent_v3c": apply_plan_faithful_latent_v3c_router_lr,
    "plan_faithful_latent_v3d_smart_router": apply_plan_faithful_latent_v3d_smart_router,
    "latent_v3d_smart_router": apply_plan_faithful_latent_v3d_smart_router,
    "plan_faithful_latent_v3d": apply_plan_faithful_latent_v3d_smart_router,
    "latent_v3d": apply_plan_faithful_latent_v3d_smart_router,
    "plan_faithful_latent_v3d_delayed_anneal": apply_plan_faithful_latent_v3d_delayed_anneal,
    "latent_v3d_delayed_anneal": apply_plan_faithful_latent_v3d_delayed_anneal,
    "plan_faithful_latent_v3d_delay": apply_plan_faithful_latent_v3d_delayed_anneal,
    "latent_v3d_delay": apply_plan_faithful_latent_v3d_delayed_anneal,
    "plan_faithful_latent_v3e_strong_z_actor": apply_plan_faithful_latent_v3e_strong_z_actor,
    "latent_v3e_strong_z_actor": apply_plan_faithful_latent_v3e_strong_z_actor,
    "plan_faithful_latent_v3e": apply_plan_faithful_latent_v3e_strong_z_actor,
    "latent_v3e": apply_plan_faithful_latent_v3e_strong_z_actor,
    "plan_faithful_latent_v3f_behavior_contrast": apply_plan_faithful_latent_v3f_behavior_contrast,
    "latent_v3f_behavior_contrast": apply_plan_faithful_latent_v3f_behavior_contrast,
    "plan_faithful_latent_v3f": apply_plan_faithful_latent_v3f_behavior_contrast,
    "latent_v3f": apply_plan_faithful_latent_v3f_behavior_contrast,
    "plan_faithful_latent_v3g_preference": apply_plan_faithful_latent_v3g_preference,
    "latent_v3g_preference": apply_plan_faithful_latent_v3g_preference,
    "plan_faithful_latent_v3g": apply_plan_faithful_latent_v3g_preference,
    "latent_v3g": apply_plan_faithful_latent_v3g_preference,
    "plan_faithful_latent_v3h_balanced_preference": apply_plan_faithful_latent_v3h_balanced_preference,
    "latent_v3h_balanced_preference": apply_plan_faithful_latent_v3h_balanced_preference,
    "plan_faithful_latent_v3h": apply_plan_faithful_latent_v3h_balanced_preference,
    "latent_v3h": apply_plan_faithful_latent_v3h_balanced_preference,
    "plan_faithful_latent_v3h2_balanced_preference": apply_plan_faithful_latent_v3h2_balanced_preference,
    "latent_v3h2_balanced_preference": apply_plan_faithful_latent_v3h2_balanced_preference,
    "plan_faithful_latent_v3h2": apply_plan_faithful_latent_v3h2_balanced_preference,
    "latent_v3h2": apply_plan_faithful_latent_v3h2_balanced_preference,
    "plan_faithful_latent_v3i_event_refresh": apply_plan_faithful_latent_v3i_event_refresh,
    "latent_v3i_event_refresh": apply_plan_faithful_latent_v3i_event_refresh,
    "plan_faithful_latent_v3i": apply_plan_faithful_latent_v3i_event_refresh,
    "latent_v3i": apply_plan_faithful_latent_v3i_event_refresh,
    "plan_faithful_latent_v3i2_router_signal": apply_plan_faithful_latent_v3i2_router_signal,
    "latent_v3i2_router_signal": apply_plan_faithful_latent_v3i2_router_signal,
    "plan_faithful_latent_v3i2": apply_plan_faithful_latent_v3i2_router_signal,
    "latent_v3i2": apply_plan_faithful_latent_v3i2_router_signal,
    "plan_faithful_latent_v3i3_event_conditioned_preference": apply_plan_faithful_latent_v3i3_event_conditioned_preference,
    "latent_v3i3_event_conditioned_preference": apply_plan_faithful_latent_v3i3_event_conditioned_preference,
    "plan_faithful_latent_v3i3": apply_plan_faithful_latent_v3i3_event_conditioned_preference,
    "latent_v3i3": apply_plan_faithful_latent_v3i3_event_conditioned_preference,
    "plan_faithful_latent_v3i4_event_progress_preference": apply_plan_faithful_latent_v3i4_event_progress_preference,
    "latent_v3i4_event_progress_preference": apply_plan_faithful_latent_v3i4_event_progress_preference,
    "plan_faithful_latent_v3i4": apply_plan_faithful_latent_v3i4_event_progress_preference,
    "latent_v3i4": apply_plan_faithful_latent_v3i4_event_progress_preference,
    "plan_faithful_latent_v3i5_crisp_router": apply_plan_faithful_latent_v3i5_crisp_router,
    "latent_v3i5_crisp_router": apply_plan_faithful_latent_v3i5_crisp_router,
    "plan_faithful_latent_v3i5": apply_plan_faithful_latent_v3i5_crisp_router,
    "latent_v3i5": apply_plan_faithful_latent_v3i5_crisp_router,
    "plan_faithful_latent_v3i6_stronger_actor_contrast": apply_plan_faithful_latent_v3i6_stronger_actor_contrast,
    "latent_v3i6_stronger_actor_contrast": apply_plan_faithful_latent_v3i6_stronger_actor_contrast,
    "plan_faithful_latent_v3i6": apply_plan_faithful_latent_v3i6_stronger_actor_contrast,
    "latent_v3i6": apply_plan_faithful_latent_v3i6_stronger_actor_contrast,
    "plan_faithful_latent_v3i7_advantage_weighted_router_distill": apply_plan_faithful_latent_v3i7_advantage_weighted_router_distill,
    "latent_v3i7_advantage_weighted_router_distill": apply_plan_faithful_latent_v3i7_advantage_weighted_router_distill,
    "plan_faithful_latent_v3i7": apply_plan_faithful_latent_v3i7_advantage_weighted_router_distill,
    "latent_v3i7": apply_plan_faithful_latent_v3i7_advantage_weighted_router_distill,
    "plan_faithful_latent_v3i8_commander_lockin": apply_plan_faithful_latent_v3i8_commander_lockin,
    "latent_v3i8_commander_lockin": apply_plan_faithful_latent_v3i8_commander_lockin,
    "plan_faithful_latent_v3i8": apply_plan_faithful_latent_v3i8_commander_lockin,
    "latent_v3i8": apply_plan_faithful_latent_v3i8_commander_lockin,
    "plan_faithful_latent_v3i9_specialist_router": apply_plan_faithful_latent_v3i9_specialist_router,
    "latent_v3i9_specialist_router": apply_plan_faithful_latent_v3i9_specialist_router,
    "plan_faithful_latent_v3i9_context_specialist": apply_plan_faithful_latent_v3i9_specialist_router,
    "latent_v3i9_context_specialist": apply_plan_faithful_latent_v3i9_specialist_router,
    "plan_faithful_latent_v3i9": apply_plan_faithful_latent_v3i9_specialist_router,
    "latent_v3i9": apply_plan_faithful_latent_v3i9_specialist_router,
    "plan_faithful_latent_v3i10_role_phase_specialist": apply_plan_faithful_latent_v3i10_role_phase_specialist,
    "latent_v3i10_role_phase_specialist": apply_plan_faithful_latent_v3i10_role_phase_specialist,
    "plan_faithful_latent_v3i10": apply_plan_faithful_latent_v3i10_role_phase_specialist,
    "latent_v3i10": apply_plan_faithful_latent_v3i10_role_phase_specialist,
    "plan_faithful_latent_v3i11_z_reactive_actor_adapters": apply_plan_faithful_latent_v3i11_z_reactive_actor_adapters,
    "latent_v3i11_z_reactive_actor_adapters": apply_plan_faithful_latent_v3i11_z_reactive_actor_adapters,
    "plan_faithful_latent_v3i11": apply_plan_faithful_latent_v3i11_z_reactive_actor_adapters,
    "latent_v3i11": apply_plan_faithful_latent_v3i11_z_reactive_actor_adapters,
    "plan_faithful_latent_v3i12_faithful_z_pressure": apply_plan_faithful_latent_v3i12_faithful_z_pressure,
    "latent_v3i12_faithful_z_pressure": apply_plan_faithful_latent_v3i12_faithful_z_pressure,
    "plan_faithful_latent_v3i12": apply_plan_faithful_latent_v3i12_faithful_z_pressure,
    "latent_v3i12": apply_plan_faithful_latent_v3i12_faithful_z_pressure,
    "plan_faithful_latent_v3i13_strict_faithful_z": apply_plan_faithful_latent_v3i13_strict_faithful_z,
    "latent_v3i13_strict_faithful_z": apply_plan_faithful_latent_v3i13_strict_faithful_z,
    "plan_faithful_latent_v3i13": apply_plan_faithful_latent_v3i13_strict_faithful_z,
    "latent_v3i13": apply_plan_faithful_latent_v3i13_strict_faithful_z,
    "plan_faithful_latent_v3i14_specialized_faithful_z": apply_plan_faithful_latent_v3i14_specialized_faithful_z,
    "latent_v3i14_specialized_faithful_z": apply_plan_faithful_latent_v3i14_specialized_faithful_z,
    "plan_faithful_latent_v3i14": apply_plan_faithful_latent_v3i14_specialized_faithful_z,
    "latent_v3i14": apply_plan_faithful_latent_v3i14_specialized_faithful_z,
    "plan_faithful_latent_v3i14_tuned": apply_plan_faithful_latent_v3i14_tuned,
    "latent_v3i14_tuned": apply_plan_faithful_latent_v3i14_tuned,
    "latent_v3i14b": apply_plan_faithful_latent_v3i14_tuned,
    "latent_v3i14_tactical_specialist_tuned": apply_plan_faithful_latent_v3i14_tuned,
    "plan_faithful_latent_v3i15_strong_separation": apply_plan_faithful_latent_v3i15_strong_separation,
    "latent_v3i15_strong_separation": apply_plan_faithful_latent_v3i15_strong_separation,
    "plan_faithful_latent_v3i15_sparse_tactical_refresh": apply_plan_faithful_latent_v3i15_sparse_tactical_refresh,
    "latent_v3i15_sparse_tactical_refresh": apply_plan_faithful_latent_v3i15_sparse_tactical_refresh,
    "latent_v3i15": apply_plan_faithful_latent_v3i15_sparse_tactical_refresh,
    "latent_v3i15_sparse_refresh": apply_plan_faithful_latent_v3i15_sparse_tactical_refresh,
    "plan_faithful_latent_v3i16_policy_z_embedding": apply_plan_faithful_latent_v3i16_policy_z_embedding,
    "latent_v3i16_policy_z_embedding": apply_plan_faithful_latent_v3i16_policy_z_embedding,
    "latent_v3i16": apply_plan_faithful_latent_v3i16_policy_z_embedding,
    "latent_v3i16_summer_z_embed": apply_plan_faithful_latent_v3i16_policy_z_embedding,
    "plan_faithful_latent_v3i16_z_embed": apply_plan_faithful_latent_v3i16_policy_z_embedding,
    "v3i16_plan_faithful_z_embed": apply_plan_faithful_latent_v3i16_policy_z_embedding,
    "plan_faithful_latent_v3i17_episode_arc": apply_plan_faithful_latent_v3i17_episode_arc,
    "latent_v3i17_episode_arc": apply_plan_faithful_latent_v3i17_episode_arc,
    "latent_v3i17": apply_plan_faithful_latent_v3i17_episode_arc,
    "v3i17_episode_arc": apply_plan_faithful_latent_v3i17_episode_arc,
    "plan_faithful_latent_v3i17_long_arc": apply_plan_faithful_latent_v3i17_long_arc,
    "latent_v3i17_long_arc": apply_plan_faithful_latent_v3i17_long_arc,
    "latent_v3i17b": apply_plan_faithful_latent_v3i17_long_arc,
    "v3i17_long_arc": apply_plan_faithful_latent_v3i17_long_arc,
    "plan_faithful_latent_v3i18_v3i16_plus_128": apply_plan_faithful_latent_v3i18_v3i16_plus_128,
    "latent_v3i18_v3i16_plus_128": apply_plan_faithful_latent_v3i18_v3i16_plus_128,
    "latent_v3i18": apply_plan_faithful_latent_v3i18_v3i16_plus_128,
    "v3i18_v3i16_plus_128": apply_plan_faithful_latent_v3i18_v3i16_plus_128,
    "plan_faithful_latent_v3i19_summer_consequence": apply_plan_faithful_latent_v3i19_summer_consequence,
    "latent_v3i19_summer_consequence": apply_plan_faithful_latent_v3i19_summer_consequence,
    "latent_v3i19": apply_plan_faithful_latent_v3i19_summer_consequence,
    "v3i19_summer_consequence": apply_plan_faithful_latent_v3i19_summer_consequence,
    "plan_faithful_latent_v4i1_strategic_pressure_qprobe": apply_plan_faithful_latent_v4i1_strategic_pressure_qprobe,
    "latent_v4i1_strategic_pressure_qprobe": apply_plan_faithful_latent_v4i1_strategic_pressure_qprobe,
    "latent_v4i1": apply_plan_faithful_latent_v4i1_strategic_pressure_qprobe,
    "v4i1": apply_plan_faithful_latent_v4i1_strategic_pressure_qprobe,
    # v4i3 canonical: Summer-Faithful Proof Suite (no distill, no aux heads).
    "plan_faithful_latent_v4i3_summer_proof": apply_plan_faithful_latent_v4i3_summer_proof,
    "latent_v4i3_summer_proof": apply_plan_faithful_latent_v4i3_summer_proof,
    "latent_v4i3": apply_plan_faithful_latent_v4i3_summer_proof,
    "v4i3": apply_plan_faithful_latent_v4i3_summer_proof,
    # v4i3 no-latent baseline (same-everything-except-z control).
    "plan_faithful_no_latent_v4i3_baseline": apply_plan_faithful_no_latent_v4i3_baseline,
    "no_latent_v4i3_baseline": apply_plan_faithful_no_latent_v4i3_baseline,
    "no_latent_v4i3": apply_plan_faithful_no_latent_v4i3_baseline,
    "v4i3_no_latent": apply_plan_faithful_no_latent_v4i3_baseline,
    "v4i3_no_latent_baseline": apply_plan_faithful_no_latent_v4i3_baseline,
    # v5 strict-Summer: literal docs/algorithm.md loss (no arc-credit, no
    # per-step PG, no aux heads, plain z-embedding actor).
    "plan_faithful_latent_v5_strict_summer": apply_plan_faithful_latent_v5_strict_summer,
    "latent_v5_strict_summer": apply_plan_faithful_latent_v5_strict_summer,
    "v5_strict_summer": apply_plan_faithful_latent_v5_strict_summer,
    "v5_strict": apply_plan_faithful_latent_v5_strict_summer,
    "v5": apply_plan_faithful_latent_v5_strict_summer,
    "strict_summer": apply_plan_faithful_latent_v5_strict_summer,
    # v5i1: reward-derived router credit with the v5 embedding-only actor.
    "plan_faithful_latent_v5i1_reward_credit_router": apply_plan_faithful_latent_v5i1_reward_credit_router,
    "latent_v5i1_reward_credit_router": apply_plan_faithful_latent_v5i1_reward_credit_router,
    "v5i1_reward_credit_router": apply_plan_faithful_latent_v5i1_reward_credit_router,
    "v5i1": apply_plan_faithful_latent_v5i1_reward_credit_router,
    # v5i2: v5i1 router plus actor-only embedding-driven FiLM.
    "plan_faithful_latent_v5i2_stronger_z_conditioning": apply_plan_faithful_latent_v5i2_stronger_z_conditioning,
    "latent_v5i2_stronger_z_conditioning": apply_plan_faithful_latent_v5i2_stronger_z_conditioning,
    "v5i2_stronger_z_conditioning": apply_plan_faithful_latent_v5i2_stronger_z_conditioning,
    "v5i2": apply_plan_faithful_latent_v5i2_stronger_z_conditioning,
    # v5i3: v5i2 plus forced-z anneal (0.30 -> 0.00 across 200k -> 500k) to
    # repair v5i2's router collapse without changing the loss objective.
    "plan_faithful_latent_v5i3_balanced_warmup": apply_plan_faithful_latent_v5i3_balanced_warmup,
    "latent_v5i3_balanced_warmup": apply_plan_faithful_latent_v5i3_balanced_warmup,
    "v5i3_balanced_warmup": apply_plan_faithful_latent_v5i3_balanced_warmup,
    "v5i3": apply_plan_faithful_latent_v5i3_balanced_warmup,
    "balanced_warmup": apply_plan_faithful_latent_v5i3_balanced_warmup,
    # v5i4: paper-faithful end-to-end. Strict-Summer concat actor + on-policy
    # categorical strategy PPO on q_phi (the task-reward gradient channel that
    # the paper's "trained end-to-end from task reward" wording requires).
    # No FiLM, no episode-credit, no forced-z curriculum, no auxiliary heads.
    "plan_faithful_latent_v5i4_end_to_end": apply_plan_faithful_latent_v5i4_end_to_end,
    "latent_v5i4_end_to_end": apply_plan_faithful_latent_v5i4_end_to_end,
    "latent_v5i4_paper_faithful": apply_plan_faithful_latent_v5i4_end_to_end,
    "v5i4_end_to_end": apply_plan_faithful_latent_v5i4_end_to_end,
    "v5i4_paper_faithful": apply_plan_faithful_latent_v5i4_end_to_end,
    "paper_faithful_end_to_end": apply_plan_faithful_latent_v5i4_end_to_end,
    "v5i4": apply_plan_faithful_latent_v5i4_end_to_end,
    # v5i5: paper-faithful entropy-floor follow-up. Inherits v5i4 verbatim
    # and changes a single field, ``latent_lam_h_end`` 0.0002 -> 0.001,
    # to combat the v5i4 router's late-training occupancy collapse without
    # introducing any new gradient channel. Stays PAPER-FAITHFUL.
    "plan_faithful_latent_v5i5_paper_faithful_entropy_floor": apply_plan_faithful_latent_v5i5_paper_faithful_entropy_floor,
    "plan_faithful_latent_v5i5_entropy_floor": apply_plan_faithful_latent_v5i5_paper_faithful_entropy_floor,
    "latent_v5i5_paper_faithful_entropy_floor": apply_plan_faithful_latent_v5i5_paper_faithful_entropy_floor,
    "latent_v5i5_entropy_floor": apply_plan_faithful_latent_v5i5_paper_faithful_entropy_floor,
    "latent_v5i5_paper_faithful": apply_plan_faithful_latent_v5i5_paper_faithful_entropy_floor,
    "v5i5_paper_faithful_entropy_floor": apply_plan_faithful_latent_v5i5_paper_faithful_entropy_floor,
    "v5i5_paper_faithful": apply_plan_faithful_latent_v5i5_paper_faithful_entropy_floor,
    "v5i5_entropy_floor": apply_plan_faithful_latent_v5i5_paper_faithful_entropy_floor,
    "v5i5": apply_plan_faithful_latent_v5i5_paper_faithful_entropy_floor,
    "paper_faithful_entropy_floor": apply_plan_faithful_latent_v5i5_paper_faithful_entropy_floor,
    # v5i6: canonical marginal-entropy interpretation. Inherits v5i4 and
    # replaces conditional entropy with batch-marginal entropy under the
    # same lambda_H schedule as v5i5.
    "plan_faithful_latent_v5i6_paper_faithful_marginal_entropy": apply_plan_faithful_latent_v5i6_paper_faithful_marginal_entropy,
    "plan_faithful_latent_v5i6_marginal_entropy": apply_plan_faithful_latent_v5i6_paper_faithful_marginal_entropy,
    "latent_v5i6_paper_faithful_marginal_entropy": apply_plan_faithful_latent_v5i6_paper_faithful_marginal_entropy,
    "latent_v5i6_marginal_entropy": apply_plan_faithful_latent_v5i6_paper_faithful_marginal_entropy,
    "latent_v5i6_paper_faithful": apply_plan_faithful_latent_v5i6_paper_faithful_marginal_entropy,
    "v5i6_paper_faithful_marginal_entropy": apply_plan_faithful_latent_v5i6_paper_faithful_marginal_entropy,
    "v5i6_paper_faithful": apply_plan_faithful_latent_v5i6_paper_faithful_marginal_entropy,
    "v5i6_marginal_entropy": apply_plan_faithful_latent_v5i6_paper_faithful_marginal_entropy,
    "v5i6": apply_plan_faithful_latent_v5i6_paper_faithful_marginal_entropy,
    "paper_faithful_marginal_entropy": apply_plan_faithful_latent_v5i6_paper_faithful_marginal_entropy,
    # v5i7: v5i5 entropy-floor paper-faithful row on the split-lane map.
    # The resolved diff vs v5i5 is exactly ``map_layout`` and ``run_tag``.
    "plan_faithful_latent_v5i7_entropy_floor_split_lane": apply_plan_faithful_latent_v5i7_entropy_floor_split_lane,
    "plan_faithful_latent_v5i7_summer_faithful_entropy_floor_split_lane": apply_plan_faithful_latent_v5i7_entropy_floor_split_lane,
    "plan_faithful_latent_v5i7_summer_faithful_split_lane": apply_plan_faithful_latent_v5i7_entropy_floor_split_lane,
    "plan_faithful_latent_v5i7_split_lane": apply_plan_faithful_latent_v5i7_entropy_floor_split_lane,
    "latent_v5i7_entropy_floor_split_lane": apply_plan_faithful_latent_v5i7_entropy_floor_split_lane,
    "latent_v5i7_summer_faithful_entropy_floor_split_lane": apply_plan_faithful_latent_v5i7_entropy_floor_split_lane,
    "latent_v5i7_summer_faithful_split_lane": apply_plan_faithful_latent_v5i7_entropy_floor_split_lane,
    "latent_v5i7_split_lane": apply_plan_faithful_latent_v5i7_entropy_floor_split_lane,
    "v5i7_entropy_floor_split_lane": apply_plan_faithful_latent_v5i7_entropy_floor_split_lane,
    "v5i7_summer_faithful_entropy_floor_split_lane": apply_plan_faithful_latent_v5i7_entropy_floor_split_lane,
    "v5i7_summer_faithful_split_lane": apply_plan_faithful_latent_v5i7_entropy_floor_split_lane,
    "v5i7_split_lane": apply_plan_faithful_latent_v5i7_entropy_floor_split_lane,
    "v5i7": apply_plan_faithful_latent_v5i7_entropy_floor_split_lane,
    # v5i8: v5i7 latent contract on lower-friction split-lane v2 task pressure.
    # The resolved diff vs v5i7 is exactly ``map_layout`` and ``run_tag``.
    "plan_faithful_latent_v5i8_split_lane_v2_task_pressure": apply_plan_faithful_latent_v5i8_split_lane_v2_task_pressure,
    "plan_faithful_latent_v5i8_summer_faithful_split_lane_v2": apply_plan_faithful_latent_v5i8_split_lane_v2_task_pressure,
    "plan_faithful_latent_v5i8_split_lane_v2": apply_plan_faithful_latent_v5i8_split_lane_v2_task_pressure,
    "latent_v5i8_split_lane_v2_task_pressure": apply_plan_faithful_latent_v5i8_split_lane_v2_task_pressure,
    "latent_v5i8_summer_faithful_split_lane_v2": apply_plan_faithful_latent_v5i8_split_lane_v2_task_pressure,
    "latent_v5i8_split_lane_v2": apply_plan_faithful_latent_v5i8_split_lane_v2_task_pressure,
    "v5i8_split_lane_v2_task_pressure": apply_plan_faithful_latent_v5i8_split_lane_v2_task_pressure,
    "v5i8_summer_faithful_split_lane_v2": apply_plan_faithful_latent_v5i8_split_lane_v2_task_pressure,
    "v5i8_split_lane_v2": apply_plan_faithful_latent_v5i8_split_lane_v2_task_pressure,
    "v5i8": apply_plan_faithful_latent_v5i8_split_lane_v2_task_pressure,
    # v5i8 Stage-1 repertoire diagnostic: 100% uniform forced-z for full run.
    "plan_faithful_latent_v5i8_repertoire_uniform_z": apply_plan_faithful_latent_v5i8_repertoire_uniform_z,
    "latent_v5i8_repertoire_uniform_z": apply_plan_faithful_latent_v5i8_repertoire_uniform_z,
    "v5i8_repertoire_uniform_z": apply_plan_faithful_latent_v5i8_repertoire_uniform_z,
    "repertoire_uniform_z": apply_plan_faithful_latent_v5i8_repertoire_uniform_z,
    # v5i9: post-Summer extension. Uses forced-z CSIA evidence as a detached
    # gated reward bonus. It is not a paper-/Summer-faithful row.
    "plan_faithful_latent_v5i9_csia_guided_specialization": apply_plan_faithful_latent_v5i9_csia_guided_specialization,
    "plan_faithful_latent_v5i9_csia": apply_plan_faithful_latent_v5i9_csia_guided_specialization,
    "latent_v5i9_csia_guided_specialization": apply_plan_faithful_latent_v5i9_csia_guided_specialization,
    "latent_v5i9_csia": apply_plan_faithful_latent_v5i9_csia_guided_specialization,
    "v5i9_csia_guided_specialization": apply_plan_faithful_latent_v5i9_csia_guided_specialization,
    "v5i9_csia": apply_plan_faithful_latent_v5i9_csia_guided_specialization,
    "v5i9": apply_plan_faithful_latent_v5i9_csia_guided_specialization,
    # v6i1 staged team-intent curriculum (production row).
    "plan_faithful_latent_v6i1_staged_team_intent_curriculum": apply_plan_faithful_latent_v6i1_staged_team_intent_curriculum,
    "latent_v6i1_staged_team_intent_curriculum": apply_plan_faithful_latent_v6i1_staged_team_intent_curriculum,
    "v6i1_staged_team_intent_curriculum": apply_plan_faithful_latent_v6i1_staged_team_intent_curriculum,
    "v6i1_staged": apply_plan_faithful_latent_v6i1_staged_team_intent_curriculum,
    "v6i1": apply_plan_faithful_latent_v6i1_staged_team_intent_curriculum,
    # v6i2 dual-gate staged team-intent curriculum.
    "plan_faithful_latent_v6i2_staged_team_intent_curriculum": apply_plan_faithful_latent_v6i2_staged_team_intent_curriculum,
    "latent_v6i2_staged_team_intent_curriculum": apply_plan_faithful_latent_v6i2_staged_team_intent_curriculum,
    "v6i2_staged_team_intent_curriculum": apply_plan_faithful_latent_v6i2_staged_team_intent_curriculum,
    "v6i2_staged": apply_plan_faithful_latent_v6i2_staged_team_intent_curriculum,
    "v6i2": apply_plan_faithful_latent_v6i2_staged_team_intent_curriculum,
    # v6i5 corrected team-intent curriculum over v6i2. Single public alias only.
    "v6i5": apply_plan_faithful_latent_v6i5_corrected_team_intent_curriculum,
    # v6i6 evidence-gated repertoire expansion over v6i5. Requires a
    # validated manifest before training can launch.
    "plan_faithful_latent_v6i6_strategy_expansion": apply_plan_faithful_latent_v6i6_strategy_expansion,
    "latent_v6i6_strategy_expansion": apply_plan_faithful_latent_v6i6_strategy_expansion,
    "v6i6_strategy_expansion": apply_plan_faithful_latent_v6i6_strategy_expansion,
    "v6i6": apply_plan_faithful_latent_v6i6_strategy_expansion,
    # v6i4 evaluation-only router-ablation protocol over a promoted v6i2 checkpoint.
    "plan_faithful_latent_v6i4_router_ablation_protocol": apply_plan_faithful_latent_v6i4_router_ablation_protocol,
    "latent_v6i4_router_ablation_protocol": apply_plan_faithful_latent_v6i4_router_ablation_protocol,
    "v6i4_router_ablation_protocol": apply_plan_faithful_latent_v6i4_router_ablation_protocol,
    "v6i4_router_ablation": apply_plan_faithful_latent_v6i4_router_ablation_protocol,
    "v6i4": apply_plan_faithful_latent_v6i4_router_ablation_protocol,
    # v6i3 strategy + local communication on v6i2 dual-evidence curriculum.
    "plan_faithful_latent_v6i3_strategy_local_comm": apply_plan_faithful_latent_v6i3_strategy_local_comm,
    "latent_v6i3_strategy_local_comm": apply_plan_faithful_latent_v6i3_strategy_local_comm,
    "v6i3_strategy_local_comm": apply_plan_faithful_latent_v6i3_strategy_local_comm,
    "v6i3_local_comm": apply_plan_faithful_latent_v6i3_strategy_local_comm,
    "v6i3": apply_plan_faithful_latent_v6i3_strategy_local_comm,
    # v6i1 repertoire-only ablation (no staged controller).
    "plan_faithful_latent_v6i1_repertoire_only_ablation": apply_plan_faithful_latent_v6i1_repertoire_only_ablation,
    "latent_v6i1_repertoire_only_ablation": apply_plan_faithful_latent_v6i1_repertoire_only_ablation,
    "v6i1_repertoire_only_ablation": apply_plan_faithful_latent_v6i1_repertoire_only_ablation,
    "v6i1_repertoire_only": apply_plan_faithful_latent_v6i1_repertoire_only_ablation,
    # v4i4 post-Summer extension: periodic router distillation (was the old v4i3).
    "plan_faithful_latent_v4i4post_periodic_router_distill": apply_plan_faithful_latent_v4i4post_periodic_router_distill,
    "latent_v4i4post_periodic_router_distill": apply_plan_faithful_latent_v4i4post_periodic_router_distill,
    "latent_v4i4post": apply_plan_faithful_latent_v4i4post_periodic_router_distill,
    "v4i4post": apply_plan_faithful_latent_v4i4post_periodic_router_distill,
    "v4i4": apply_plan_faithful_latent_v4i4post_periodic_router_distill,
    "plan_faithful_latent_intent_credit": apply_plan_faithful_latent_episode_strategic,
    "plan_faithful_latent_no_entropy": apply_plan_faithful_latent_no_entropy,
    "latent_plan_faithful_no_entropy": apply_plan_faithful_latent_no_entropy,
    "latent_recommended_no_entropy": apply_plan_faithful_latent_no_entropy,
    "plan_faithful_latent_phase1_coupling": apply_plan_faithful_latent_phase1_coupling,
    "latent_phase1_coupling": apply_plan_faithful_latent_phase1_coupling,
    "plan_faithful_latent_phase2_credit": apply_plan_faithful_latent_phase2_credit,
    "latent_phase2_credit": apply_plan_faithful_latent_phase2_credit,
    "plan_faithful_latent_phase3_reward_geometry": apply_plan_faithful_latent_phase3_reward_geometry,
    "latent_phase3_reward_geometry": apply_plan_faithful_latent_phase3_reward_geometry,
    "plan_faithful_latent_phase3b_outcome_clean": apply_plan_faithful_latent_phase3b_outcome_clean,
    "latent_phase3b_outcome_clean": apply_plan_faithful_latent_phase3b_outcome_clean,
    "plan_faithful_latent_phase3b_ablate_k1": apply_plan_faithful_latent_phase3b_ablate_k1,
    "latent_phase3b_ablate_k1": apply_plan_faithful_latent_phase3b_ablate_k1,
    "plan_faithful_latent_phase3b_ablate_no_persistence": apply_plan_faithful_latent_phase3b_ablate_no_persistence,
    "latent_phase3b_ablate_no_persistence": apply_plan_faithful_latent_phase3b_ablate_no_persistence,
    "plan_faithful_latent_phase4a_rescue": apply_plan_faithful_latent_phase4a_rescue,
    "latent_phase4a_rescue": apply_plan_faithful_latent_phase4a_rescue,
    "plan_faithful_latent_phase4a_rescue_hardpool": apply_plan_faithful_latent_phase4a_rescue_hardpool,
    "latent_phase4a_rescue_hardpool": apply_plan_faithful_latent_phase4a_rescue_hardpool,
    "plan_faithful_latent_episode_z_clean": apply_plan_faithful_latent_episode_z_clean,
    "latent_episode_z_clean": apply_plan_faithful_latent_episode_z_clean,
    "plan_faithful_latent_option_a_episode_credit": apply_plan_faithful_latent_option_a_episode_credit,
    "latent_option_a_episode_credit": apply_plan_faithful_latent_option_a_episode_credit,
    "plan_faithful_latent_episode_credit": apply_plan_faithful_latent_option_a_episode_credit,
    "plan_faithful_latent_option_a": apply_plan_faithful_latent_option_a,
    "latent_option_a": apply_plan_faithful_latent_option_a,
    "plan_faithful_latent_fix_d": apply_plan_faithful_latent_option_a,
    "latent_fix_d": apply_plan_faithful_latent_option_a,
    "plan_faithful_latent_k1": apply_plan_faithful_latent_k1,
    "latent_plan_faithful_k1": apply_plan_faithful_latent_k1,
    "plan_faithful_collapsed_latent": apply_plan_faithful_latent_k1,
    "latent_recommended_collapsed_k1": apply_plan_faithful_latent_k1,
    "plan_faithful_no_latent": apply_plan_faithful_no_latent,
    "no_latent_plan_faithful": apply_plan_faithful_no_latent,
    "no_latent_baseline": apply_plan_faithful_no_latent,
    "plan_option_a": apply_plan_option_a,
    "plan_option_b_lamp": apply_plan_option_b_lamp,
    "plan_option_b": apply_plan_option_b_lamp,

    # Hypothesis family
    "hypothesis_flat_opprand": apply_hypothesis_flat_opprand,
    "hypothesis_latent_opprand_optiona": apply_hypothesis_latent_opprand_optiona,
    "hypothesis_latent_opprand_optionb_lamp_coef05": apply_hypothesis_latent_opprand_optionb_lamp_coef05,
    "hypothesis_latent_opprand_optionb_no_lamp": apply_hypothesis_latent_opprand_optionb_no_lamp,
    "hypothesis_latent_opprand_optionb_coef03": apply_hypothesis_latent_opprand_optionb_coef03,
    "hypothesis_flat_opprand_op35": apply_hypothesis_flat_opprand_op35,
    "hypothesis_latent_opprand_optionb_lamp_coef05_op35": apply_hypothesis_latent_opprand_optionb_lamp_coef05_op35,

    # Other family
    "latent_op3_push80_1m": apply_latent_op3_push80_1m,
    "latent_push80_1m": apply_latent_op3_push80_1m,
    "latent_train80_op3_1m": apply_latent_train80_op3_1m,
    "latent_op3_train80_1m": apply_latent_train80_op3_1m,
    "latent_op3_wrmax_1m": apply_latent_op3_wrmax_1m,
    "latent_wrmax_op3_1m": apply_latent_op3_wrmax_1m,
    "latent_op3_wrmax_2m": apply_latent_op3_wrmax_1m,
    "latent_wrmax_op3_2m": apply_latent_op3_wrmax_1m,
    "latent_a1_plan_faithful": apply_latent_a1_plan_faithful,
    "latent_op3_a1_plan_faithful": apply_latent_a1_plan_faithful,
    "latent_op3_wrmax_train_2m": apply_latent_op3_wrmax_train_2m,
    "latent_wrmax_op3_train_2m": apply_latent_op3_wrmax_train_2m,
}


def apply_preset(cfg: PPOConfig, preset: str) -> PPOConfig:
    """Apply named high-level presets for repeatable training recipes."""
    key = str(preset).strip().lower()
    if not key:
        return cfg
    fn = PRESET_REGISTRY.get(key)
    if fn is None:
        raise ValueError(
            f"Unknown preset {preset!r}. Supported presets: "
            "'plan_option_a', 'plan_option_b_lamp', 'plan_option_b', "
            "'hypothesis_flat_opprand', 'hypothesis_latent_opprand_optiona', "
            "'hypothesis_latent_opprand_optionb_lamp_coef05', 'hypothesis_latent_opprand_optionb_no_lamp', "
            "'hypothesis_latent_opprand_optionb_coef03', "
            "'hypothesis_flat_opprand_op35', 'hypothesis_latent_opprand_optionb_lamp_coef05_op35', "
            "'latent_op3_push80_1m', 'latent_push80_1m', "
            "'latent_train80_op3_1m', 'latent_op3_train80_1m', "
            "'latent_op3_wrmax_1m', 'latent_wrmax_op3_1m', "
            "'latent_op3_wrmax_2m', 'latent_wrmax_op3_2m' (aliases for wrmax 1M), "
            "'latent_op3_wrmax_train_2m', 'latent_wrmax_op3_train_2m', "
            "'latent_a1_plan_faithful', 'latent_op3_a1_plan_faithful', "
            "'plan_faithful_latent_persist_entropy', 'plan_faithful_latent_no_persistence', "
            "'plan_faithful_latent_no_entropy', 'plan_faithful_latent_k1', 'plan_faithful_no_latent', "
            "'plan_faithful_latent_phase1_coupling', 'plan_faithful_latent_phase2_credit', "
            "'plan_faithful_latent_phase3_reward_geometry', "
            "'plan_faithful_latent_phase3b_outcome_clean', "
            "'plan_faithful_latent_phase3b_ablate_k1', "
            "'plan_faithful_latent_phase3b_ablate_no_persistence', "
            "'plan_faithful_latent_phase4a_rescue', "
            "'plan_faithful_latent_phase4a_rescue_hardpool', "
            "'plan_faithful_latent_episode_z_clean', "
            "'plan_faithful_latent_v3f_behavior_contrast', "
            "'plan_faithful_latent_option_a' (a.k.a. 'plan_faithful_latent_fix_d'), "
            "'plan_faithful_latent_option_a_episode_credit', "
            "'plan_faithful_latent_v3i4', 'plan_faithful_latent_v3i5', "
            "'plan_faithful_latent_v3i6', 'plan_faithful_latent_v3i7', "
            "'plan_faithful_latent_v3i8', 'plan_faithful_latent_v3i9', "
            "'plan_faithful_latent_v3i10', 'plan_faithful_latent_v3i11', "
            "'plan_faithful_latent_v3i12', 'latent_v3i14_tuned', "
            "'latent_v3i14b', 'latent_v3i14_tactical_specialist_tuned', "
            "'latent_v3i15_strong_separation', "
            "'latent_v3i15_sparse_tactical_refresh', 'latent_v3i15', "
            "'latent_v3i15_sparse_refresh', "
            "'latent_v3i16_policy_z_embedding', 'latent_v3i16', "
            "'latent_v3i16_summer_z_embed', "
            "'plan_faithful_latent_v3i16_z_embed', "
            "'v3i16_plan_faithful_z_embed'."
            " 'v5i9' / 'v5i9_csia_guided_specialization' for the CSIA post-Summer extension."
        )
    return fn(cfg)
