"""v3i consequence presets — v3i16 policy z embedding through v3i19 summer consequence."""
from __future__ import annotations

from rl.config.ppo_config import PPOConfig, TrainMode

from .base import apply_plan_faithful_latent


def apply_plan_faithful_latent_v3i16_policy_z_embedding(
    cfg: PPOConfig,
) -> PPOConfig:
    """v3i16: strict plan-faithful learned-z actor conditioning.

    The actor receives ``concat(CNN(grid), per_agent_vec, z_embedding)``.
    The critic keeps its existing z one-hot path. All later experimental
    routing teachers, behavior rewards, policy-separation losses, adapters,
    and event-driven refresh paths are disabled for this clean test.
    """
    cfg = apply_plan_faithful_latent(cfg)

    cfg.latent_actor_z_onehot_enabled = False
    cfg.latent_z_embed_dim = 16
    cfg.latent_actor_z_embed_scale = 1.0
    cfg.latent_actor_z_adapter_enabled = False
    cfg.latent_actor_z_adapter_scale = 0.0
    cfg.latent_actor_z_film_layers = 1
    cfg.latent_actor_z_adapter_warmup_steps = 0
    cfg.latent_actor_z_adapter_ramp_steps = 0

    cfg.latent_lam_h = 0.001
    cfg.latent_lam_h_start = 0.001
    cfg.latent_lam_h_end = 0.001
    cfg.latent_entropy_anneal_start = 0
    cfg.latent_entropy_anneal_end = 0
    cfg.latent_entropy_objective = "maximize"
    cfg.latent_lam_p = 0.02

    # Fixed-cadence persistence is the simple Option-B path. It keeps lambda_p
    # active without the event/tactical refresh machinery from v3i15.
    cfg.latent_resample_every_n = 64
    cfg.latent_resample_on_flag = False
    cfg.latent_event_refresh_enabled = False
    cfg.latent_sparse_tactical_refresh_enabled = False
    cfg.latent_kl_consecutive = 0.0
    cfg.latent_gae_reset_on_z_change = True

    cfg.latent_strategy_aux_predict_phase_coef = 0.0
    cfg.latent_strategy_aux_return_head = False
    cfg.latent_strategy_aux_return_coef = 0.0
    cfg.latent_forced_z_episode_frac = 0.0
    cfg.latent_behavior_contrast_coef = 0.0
    cfg.latent_actor_z_separation_start_coef = 0.0
    cfg.latent_actor_z_separation_coef = 0.0
    cfg.latent_usage_balance_coef = 0.0
    cfg.latent_preference_coef = 0.0
    cfg.latent_preference_commit_coef = 0.0
    cfg.latent_awrd_enabled = False
    cfg.latent_awrd_coef = 0.0
    cfg.latent_specialist_router_enabled = False
    cfg.latent_marginal_balance_coef = 0.0
    cfg.latent_conditional_entropy_min_coef_start = 0.0
    cfg.latent_conditional_entropy_min_coef = 0.0
    cfg.latent_context_mi_coef = 0.0
    cfg.latent_v3i3_event_preference_enabled = False
    cfg.latent_v3i3_event_preference_coef = 0.0
    cfg.latent_v3i3_refresh_log_enabled = False

    cfg.run_tag = "v3i16_plan_faithful_z_embed_1m_4v4"
    return cfg


def _apply_v3i17_consequence_only(cfg: PPOConfig) -> PPOConfig:
    """Shared v3i17 configuration: 'reward z consequences, not z existence'.

    Branched into two sibling presets, ``v3i17_episode_arc`` (episode-level z)
    and ``v3i17_long_arc`` (256-step persistence). Both keep v3i16's actor
    z-embedding architecture and disable every "existence" knob, leaving only
    the consequence channel (episode-credit PPO on q_phi) alive after the
    entropy anneal completes.

    Existence pressure removed / annealed away:

    * ``latent_lam_h`` anneals **0.003 -> 0.0** over steps **200k -> 700k**.
      Early phase keeps K=4 alive while q_phi explores; after 700k the
      coefficient is exactly zero so the marginal-entropy reward stops
      contributing gradient.
    * ``latent_actor_z_separation_*``, ``latent_usage_balance_*``,
      ``latent_marginal_balance_*``, ``latent_behavior_contrast_*``,
      ``latent_specialist_*``, ``latent_conditional_entropy_min_*``,
      ``latent_context_mi_*`` all forced to 0 / disabled (inherited from v3i16).

    Consequence channel kept on:

    * ``latent_episode_strategy_ppo = True`` with
      ``latent_episode_strategy_coef = 0.30``. q_phi's gradient is the
      per-episode return advantage. One clean datapoint per episode per
      (context, z) cell; the only signal that pushes q_phi to specialise.
    * ``latent_episode_strategy_warmup_decision_steps = 5`` so q_phi commits
      after the ctx170 EMAs have observed opponent dynamics.
    * ``latent_strategy_ppo_coef = 0.0`` -- no per-step strategy coupling.
    * v3i3 / preference / AWRD distillation channels stay OFF; we want a
      single, audit-able consequence source.
    """
    cfg = apply_plan_faithful_latent_v3i16_policy_z_embedding(cfg)

    cfg.latent_lam_h = 0.003
    cfg.latent_lam_h_start = 0.003
    cfg.latent_lam_h_end = 0.0
    cfg.latent_entropy_anneal_start = 200_000
    cfg.latent_entropy_anneal_end = 700_000
    cfg.latent_entropy_objective = "maximize"

    cfg.latent_strategy_ppo_coef = 0.0
    cfg.latent_episode_strategy_ppo = True
    cfg.latent_episode_strategy_coef = 0.30
    cfg.latent_episode_strategy_clip_eps = 0.2
    cfg.latent_episode_strategy_value_coef = 0.5
    cfg.latent_episode_strategy_return_norm = True
    cfg.latent_episode_strategy_warmup_decision_steps = 5
    cfg.latent_q_phi_option_advantage = False
    return cfg


def apply_plan_faithful_latent_v3i18_v3i16_plus_128(cfg: PPOConfig) -> PPOConfig:
    """v3i18: conservative ``v3i16 + 128`` -- only the resample interval changes.

    Hypothesis: v3i16 had the best-behaved actor-z embedding path of the v3iX
    family but its 64-step strategic horizon may be too short. Doubling the
    persistence window to 128 decision steps gives z a longer arc without
    touching any other dial.

    Inherits ``apply_plan_faithful_latent_v3i16_policy_z_embedding`` verbatim
    and changes exactly **one** runtime knob:

    * ``latent_resample_every_n``: ``64`` -> ``128``

    Everything else from v3i16 is preserved bit-for-bit:

    * actor z embedding path: ``latent_z_embed_dim = 16``,
      ``latent_actor_z_onehot_enabled = False``,
      ``latent_actor_z_adapter_enabled = False``,
      ``latent_actor_z_film_layers = 1``
    * ``latent_strategy_ppo_coef = 0.30`` (per-step PPO strategy gradient,
      inherited from ``apply_plan_faithful_base``)
    * ``latent_lam_p = 0.02`` (persistence loss within the 128-step window)
    * ``latent_lam_h = 0.001`` flat (no anneal)
    * ``latent_episode_strategy_ppo = False`` (no episode-credit channel)
    * no AWRD, no v3i3 event preference, no preference distillation
    * no supervised labels, no phase / flag / outcome heads, no opponent heads
    * no behavior-contrast loss, no actor-z separation loss,
      no usage-balance / marginal-balance / specialist pressure
    * ``latent_event_refresh_enabled = False``,
      ``latent_sparse_tactical_refresh_enabled = False``
    """
    cfg = apply_plan_faithful_latent_v3i16_policy_z_embedding(cfg)

    cfg.latent_resample_every_n = 128

    cfg.run_tag = "v3i18_v3i16_plus_128_1m_4v4"
    return cfg


def apply_plan_faithful_latent_v3i17_episode_arc(cfg: PPOConfig) -> PPOConfig:
    """v3i17 episode-arc: one z per episode, consequence-only gradient.

    "Strategy needs a story arc, not a 5-second costume change."

    Differences vs v3i16:

    * ``latent_resample_every_n = 0``  -- z is sampled once at episode start
      and held for the entire episode. No mid-episode refreshes.
    * ``latent_lam_p = 0.0``  -- with z fixed for the episode, persistence
      loss is a no-op; zeroing it out avoids stale telemetry.
    * ``latent_lam_h`` anneals 0.003 -> 0.0 from 200k -> 700k (consequence-
      only past 700k).
    * ``latent_event_refresh_enabled = False`` (already inherited; reaffirmed).
    * Episode-credit PPO is the sole gradient to q_phi.

    Faithful guarantee unchanged: no labels, no aux heads, no opponent ID.
    """
    cfg = _apply_v3i17_consequence_only(cfg)

    cfg.latent_resample_every_n = 0
    cfg.latent_resample_on_flag = False
    cfg.latent_event_refresh_enabled = False
    cfg.latent_sparse_tactical_refresh_enabled = False
    cfg.latent_lam_p = 0.0
    cfg.latent_kl_consecutive = 0.0
    cfg.latent_gae_reset_on_z_change = True

    cfg.run_tag = "v3i17_episode_arc_1m_4v4"
    return cfg


def apply_plan_faithful_latent_v3i17_long_arc(cfg: PPOConfig) -> PPOConfig:
    """v3i17 long-arc: 256-step z persistence, consequence-only gradient.

    Sibling to ``v3i17_episode_arc``. Keeps the option of mid-episode z
    refreshes but extends the dwell to a 256-step "story arc" -- 4x longer
    than v3i16's 64-step costume change.

    Differences vs v3i16:

    * ``latent_resample_every_n = 256`` (was 64).
    * ``latent_lam_p = 0.01`` (was 0.02) -- small switch cost preserved so
      within-arc continuity is encouraged, but lighter since 256 steps is
      already a long arc.
    * ``latent_lam_h`` anneals 0.003 -> 0.0 from 200k -> 700k.
    * Episode-credit PPO is the sole consequence channel; per-step strategy
      coupling stays off.
    """
    cfg = _apply_v3i17_consequence_only(cfg)

    cfg.latent_resample_every_n = 256
    cfg.latent_resample_on_flag = False
    cfg.latent_event_refresh_enabled = False
    cfg.latent_sparse_tactical_refresh_enabled = False
    cfg.latent_lam_p = 0.01
    cfg.latent_kl_consecutive = 0.0
    cfg.latent_gae_reset_on_z_change = True

    cfg.run_tag = "v3i17_long_arc_1m_4v4"
    return cfg


def apply_plan_faithful_latent_v3i19_summer_consequence(cfg: PPOConfig) -> PPOConfig:
    """v3i19: Summer-faithful per-arc consequence credit.

    Diagnosis driving the design (from v3i18 telemetry):

    * v3i18's z had near-max entropy AND near-zero MI with any context.
    * Per-step ``latent_strategy_ppo_coef = 0.30`` was too noisy a credit
      pathway when ~200 z decisions per episode shared a single env return.
    * Behavior fingerprints across z values were statistically identical;
      the actor learned to ignore z.

    v3i19 changes credit assignment, not the conceptual design. Same K=4,
    same global-state q_phi, same shared actor, same critic-z conditioning,
    same persistence + entropy regularisation, same task-reward-only signal.
    The only change is HOW the task-reward gradient reaches q_phi.

    Recipe (per the locked design):

    1. **Sparse refresh with optional flag-event reactivity** --
       ``latent_resample_every_n = 64`` (vs v3i18's 128). Optional flag-
       event refresh on territory changes via ``latent_resample_on_flag``.
       This gives q_phi more chances to react to flag-state transitions
       while keeping persistence in effect within each arc.
    2. **Persistence**: ``latent_lam_p = 0.03`` (range 0.01-0.05 from plan;
       prevents thrashing without freezing z forever).
    3. **Entropy decays from 0.003 -> 0.0002 over 300k steps.** Early
       training keeps z alive; later training stops paying q_phi to spin
       the roulette wheel. Still Summer-faithful: entropy regularisation
       remains present as a collapse guard, not as the primary objective.
    4. **Per-arc credit replaces per-step PPO.**
       ``latent_strategy_ppo_coef = 0.0``,
       ``latent_episode_strategy_ppo = False``,
       ``latent_arc_credit_enabled = True``,
       ``latent_arc_credit_coef = 1.0``.
       At each z-decision boundary the trainer saves
       (ctx_at_arc_start, z, log_prob(z), V_phi(ctx)). When the arc ends
       (next z-resample or episode end), arc_return = sum env reward over
       the arc, arc_advantage = arc_return - V_phi(ctx). Normalized within
       the rollout batch. q_phi loss = clipped PPO ratio * advantage.
    5. **Stronger actor z conditioning via architecture** (FiLM + onehot
       concat). The actor receives both the FiLM scale/shift modulation
       from the z embedding AND an onehot z appended to the per-agent vector.
       Still inside the Summer policy form pi_i(a_i | o_i, z); no separate
       per-z heads. This makes z harder for the actor to ignore without
       any auxiliary supervision.
    6. **Critic z conditioning** remains on (inherited from v3i16): the
       centralized critic sees ``concat(global_state, joint_actions, z_onehot)``.

    Plan-faithful contract maintained:

    * No labels, no opponent IDs, no phase/flag/outcome heads.
    * No reconstruction loss, no auxiliary prediction heads.
    * No handcrafted strategy rewards, no role-labelled bonuses.
    * Critic-z, persistence, sparse refresh, and entropy regularisation
      all explicitly endorsed by the Summer plan.

    Minimum proof thresholds (analysis tool):

    * ``normalized_MI_z_opponent`` > 0.02 (v3i18 ~= 0.0001)
    * ``normalized_MI_z_phase`` > 0.01-0.02 (v3i18 ~= 0.00006)
    * ``normalized_MI_z_flag`` > 0.02 (v3i18 ~= 0.00024)
    * ``behavior_by_z`` clear spread in >= 3 signals (v3i18: tiny)
    * fixed-z behavior visibly different across z (v3i18: identical)

    The success criterion is "z carries nonzero consequence", NOT WR alone.
    """
    cfg = apply_plan_faithful_latent_v3i16_policy_z_embedding(cfg)

    # 1. Sparse refresh ONLY. ``latent_resample_on_flag`` is disabled because
    #    the current ``_apply_flag_resample_trigger`` fires on continuous
    #    distance-feature changes (>1e-4 in slice [8:12]), which means it
    #    triggers a z resample on essentially every decision step in a 4v4
    #    game with moving agents. The first v3i19 launch confirmed this: arcs
    #    averaged 1.3 steps, 100% dropped by ``min_len=32``, q_phi grad
    #    stayed at 0.0. The user's spec called this "optional if easy"; it's
    #    not easy in the current implementation. Future revisit could enable
    #    the more disciplined ``latent_event_refresh_enabled`` path (which
    #    has min_gap_steps + max_per_episode guardrails and uses discrete
    #    capture-bit transitions instead of distance deltas).
    cfg.latent_resample_every_n = 64
    cfg.latent_resample_on_flag = False
    cfg.latent_event_refresh_enabled = False
    cfg.latent_sparse_tactical_refresh_enabled = False
    cfg.latent_gae_reset_on_z_change = True
    cfg.latent_kl_consecutive = 0.0

    # 2. Persistence (range 0.01-0.05; doubled from v3i16/v3i18's 0.02 to
    #    discourage thrashing now that sparse + flag refresh combine to
    #    propose more resamples per episode).
    cfg.latent_lam_p = 0.03

    # 3. Entropy schedule: 0.003 -> 0.0002 over 300k steps. Collapse guard
    #    early, near-zero late. ``latent_entropy_anneal_start = 0`` so the
    #    decay begins immediately; ``_end`` is the user-spec 300_000 mark.
    cfg.latent_lam_h = 0.003
    cfg.latent_lam_h_start = 0.003
    cfg.latent_lam_h_end = 0.0002
    cfg.latent_entropy_anneal_start = 0
    cfg.latent_entropy_anneal_end = 300_000
    cfg.latent_entropy_objective = "maximize"

    # 4. Per-arc consequence credit. Per-step PPO and episode-credit OFF.
    cfg.latent_strategy_ppo_coef = 0.0
    cfg.latent_episode_strategy_ppo = False
    cfg.latent_episode_strategy_coef = 0.0
    cfg.latent_arc_credit_enabled = True
    cfg.latent_arc_credit_coef = 1.0
    cfg.latent_arc_credit_baseline = "context_value"
    cfg.latent_arc_credit_return_norm = True
    cfg.latent_arc_credit_min_len = 32
    cfg.latent_arc_credit_n_epochs = 4
    cfg.latent_arc_credit_clip_eps = 0.2

    # 5. FiLM + onehot concat actor z conditioning. Stronger than v3i16's
    #    FiLM-only path; still shared-actor form pi_i(a_i | o_i, z).
    cfg.latent_actor_z_onehot_enabled = True
    cfg.latent_actor_z_onehot_scale = 1.0
    cfg.latent_z_embed_dim = 16
    cfg.latent_actor_z_film_layers = 1

    # 6. Defensive zeroing of every "z existence" pressure / labelled head /
    #    aux objective (mostly inherited from v3i16 already; reaffirmed
    #    here for audit clarity).
    cfg.latent_strategy_aux_predict_phase_coef = 0.0
    cfg.latent_strategy_aux_return_head = False
    cfg.latent_strategy_aux_return_coef = 0.0
    cfg.latent_forced_z_episode_frac = 0.0
    cfg.latent_behavior_contrast_coef = 0.0
    cfg.latent_actor_z_separation_coef = 0.0
    cfg.latent_actor_z_separation_start_coef = 0.0
    cfg.latent_usage_balance_coef = 0.0
    cfg.latent_preference_coef = 0.0
    cfg.latent_preference_commit_coef = 0.0
    cfg.latent_awrd_enabled = False
    cfg.latent_awrd_coef = 0.0
    cfg.latent_specialist_router_enabled = False
    cfg.latent_marginal_balance_coef = 0.0
    cfg.latent_conditional_entropy_min_coef_start = 0.0
    cfg.latent_conditional_entropy_min_coef = 0.0
    cfg.latent_context_mi_coef = 0.0
    cfg.latent_v3i3_event_preference_enabled = False
    cfg.latent_v3i3_event_preference_coef = 0.0
    cfg.latent_v3i3_refresh_log_enabled = False

    cfg.run_tag = "v3i19_summer_consequence_1m_4v4"
    return cfg
