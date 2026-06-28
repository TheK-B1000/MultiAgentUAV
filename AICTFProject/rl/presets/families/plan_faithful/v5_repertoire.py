"""v5/v6 repertoire presets — strict Summer through v6i1 repertoire ablation."""
from __future__ import annotations

from rl.config.ppo_config import PPOConfig, TrainMode

from .v4_proof import apply_plan_faithful_latent_v4i3_summer_proof


def apply_plan_faithful_latent_v5_strict_summer(cfg: PPOConfig) -> PPOConfig:
    """v5 (strict-Summer): the literal docs/algorithm.md objective.

    The Summer plan's locked loss is::

        L = L_PPO + lam_p * L_persist - lam_H * H(q_phi(z | s))

    with the explicit clause "PPO clipped ratio uses action log-probs only;
    q_phi is trained through strategy entropy and persistence, plus optional
    consecutive KL." That excludes every auxiliary q_phi gradient channel
    the post-Summer chain accumulated -- per-step strategy PPO
    (``latent_strategy_ppo_coef``), per-episode credit
    (``latent_episode_strategy_ppo``), per-arc credit
    (``latent_arc_credit_enabled``), aux return prediction
    (``latent_strategy_aux_return_head``), and aux phase prediction
    (``latent_strategy_aux_predict_phase_coef``).

    v4i3 inherited the v3i19 arc-credit channel (``coef = 1.0``,
    ``baseline = context_value``). Useful for proving "credit can pull
    q_phi off uniform when given a per-arc PG signal", but not literally
    Summer-strict. v5 is the experiment that tests *whether the docs/
    algorithm.md loss alone* (entropy + persistence on q_phi, with the
    actor receiving z via a plain ``nn.Embedding(K, d_z)`` concat) is
    enough to differentiate the four latent strategies.

    Recipe (one-variable changes vs v4i3):

    1. **No auxiliary q_phi PG channels.**
       ``latent_arc_credit_enabled = False`` (coef = 0),
       ``latent_episode_strategy_ppo = False`` (coef = 0),
       ``latent_strategy_ppo_coef = 0.0``,
       ``latent_strategy_aux_return_head = False``,
       ``latent_strategy_aux_predict_phase_coef = 0.0``.
    2. **Strict actor-z conditioning per algorithm.md.** Only
       ``nn.Embedding(K, d_z)`` concatenated to per-agent features
       (``latent_z_embed_dim = 16``). FiLM
       (``latent_actor_z_adapter_enabled = False``) and z-onehot concat
       (``latent_actor_z_onehot_enabled = False``) are disabled because
       neither appears in the Summer plan's actor spec.
    3. **Regularizers preserved.** ``latent_lam_p = 0.03``,
       ``latent_lam_h`` schedule 0.003 -> 0.0002 over 300k steps,
       ``latent_resample_every_n = 64``. Matches v4i3 exactly.

    Required gate fix (already in place at this commit): the v5 gate in
    ``ppo_updater.update`` no longer silences the main-loop q_phi loss
    when ``latent_strategy_ppo_coef == 0``. It silences only when a
    dedicated ``latent_router_optimizer`` is active (the v3c safeguard).
    Without that fix, ``lam_p`` and ``lam_h`` would be silently zeroed
    here and q_phi would receive zero gradient. See the comment block at
    ``ppo_updater._gate_q_phi_main_loop`` / the ``MainLoopGatingTests``
    in ``test_marginal_baseline.py``.

    Plan-faithful contract (re-asserted):

    * No labels, no opponent IDs, no phase/flag/outcome heads.
    * No reconstruction loss, no auxiliary prediction heads.
    * No handcrafted strategy rewards, no role-labelled bonuses.
    * Critic-z, persistence, and entropy regularisation are explicitly
      endorsed by the Summer plan.

    Expected outcome (this preset's role in the proof table):

    * If H(q_phi) collapses to a single z and/or stays at ln(K), and the
      WR matches the no-latent v4i3 baseline, then the literal-strict
      reading of docs/algorithm.md does NOT actually train q_phi from
      reward. The arc-credit / episode-credit / per-step PG channels in
      v4i3 / v3c were each an answer to this problem.
    * If H(q_phi) sharpens and WR exceeds the no-latent baseline by a
      paired-bootstrap-significant margin, the Summer plan is alive
      *exactly as written*.

    The same opponent pool / map / budget as v4i3 must be used to make
    the comparison meaningful.
    """
    cfg = apply_plan_faithful_latent_v4i3_summer_proof(cfg)

    # 1. Disable every auxiliary q_phi PG / supervision channel.
    cfg.latent_arc_credit_enabled = False
    cfg.latent_arc_credit_coef = 0.0
    cfg.latent_episode_strategy_ppo = False
    cfg.latent_episode_strategy_coef = 0.0
    cfg.latent_strategy_ppo_coef = 0.0
    cfg.latent_strategy_aux_return_head = False
    cfg.latent_strategy_aux_return_coef = 0.0
    cfg.latent_strategy_aux_predict_phase_coef = 0.0

    # 2. Strict actor-z conditioning per docs/algorithm.md: nn.Embedding only.
    cfg.latent_actor_z_onehot_enabled = False
    cfg.latent_actor_z_onehot_scale = 0.0
    cfg.latent_actor_z_adapter_enabled = False
    cfg.latent_actor_z_adapter_scale = 0.0
    cfg.latent_actor_z_film_layers = 1  # ignored when adapter disabled
    cfg.latent_z_embed_dim = 16
    cfg.latent_actor_z_embed_scale = 1.0

    # 3. Defensive zero on every post-Summer separation / preference / specialist
    #    loss inherited indirectly through the v3i19 chain. Most are already
    #    zero in v4i3; re-asserted here so config-diff at PR time catches drift.
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
    cfg.latent_conditional_entropy_min_coef = 0.0
    cfg.latent_conditional_entropy_min_coef_start = 0.0
    cfg.latent_context_mi_coef = 0.0
    cfg.latent_v3i3_event_preference_enabled = False
    cfg.latent_v3i3_event_preference_coef = 0.0
    cfg.latent_router_distill_enabled = False

    cfg.run_tag = "v5_strict_summer_OP5_OP6_OP7_2m_4v4"
    return cfg


def apply_plan_faithful_latent_v5i1_reward_credit_router(
    cfg: PPOConfig,
) -> PPOConfig:
    """v5i1: reward-credit repair for the collapsed strict-Summer router.

    ``v5_strict_summer`` proved that persistence plus entropy does not provide
    task-return credit to q_phi. Persistence self-reinforces whichever latent
    wins the early sampling race, while the entropy coefficient anneals too
    quickly to keep all four choices alive.

    This additive preset preserves v5's plain ``nn.Embedding`` actor contract
    and all no-label/no-auxiliary-head guards, but makes q_phi trainable from
    task reward:

    * commit one z per episode after five context-building decision steps;
    * optimize that sampled z from completed-episode return;
    * subtract the detached z-marginal value baseline;
    * use six router PPO epochs and a dedicated 5e-3 router learning rate;
    * retain a 1e-3 entropy floor as collapse insurance.

    No opponent ID, phase label, handcrafted strategy reward, preference
    target, or distillation target enters the router.
    """
    cfg = apply_plan_faithful_latent_v5_strict_summer(cfg)

    cfg.latent_resample_every_n = 0
    cfg.latent_resample_on_flag = False
    cfg.latent_lam_p = 0.0

    cfg.latent_lam_h = 0.003
    cfg.latent_lam_h_start = 0.003
    cfg.latent_lam_h_end = 0.001
    cfg.latent_entropy_anneal_start = 200_000
    cfg.latent_entropy_anneal_end = 700_000

    cfg.latent_episode_strategy_ppo = True
    cfg.latent_episode_strategy_coef = 0.30
    cfg.latent_episode_strategy_warmup_decision_steps = 5
    cfg.latent_episode_strategy_n_epochs = 6
    cfg.latent_episode_strategy_lr = 5e-3
    cfg.latent_q_phi_marginal_baseline = True

    cfg.run_tag = "v5i1_reward_credit_router_OP5_OP6_OP7_2m_4v4"
    return cfg


def apply_plan_faithful_latent_v5i2_stronger_z_conditioning(
    cfg: PPOConfig,
) -> PPOConfig:
    """v5i2: strengthen actor controllability with embedding-driven FiLM.

    This experiment inherits the v5i1 episode-level reward-credit router
    unchanged. The only behavioral change is an actor-only FiLM projection
    from the existing learned z embedding into the second hidden layer:

        h' = gamma(z) * h + beta(z)

    The projection starts near identity so the embedding-concat policy is
    preserved at initialization while giving PPO a direct multiplicative and
    additive path from z to the policy head. No specialization loss, diversity
    reward, forced-z balancing, role assignment, critic change, or router
    objective is added.
    """
    cfg = apply_plan_faithful_latent_v5i1_reward_credit_router(cfg)

    cfg.enable_actor_z_film = True
    cfg.actor_z_film_init_scale = 0.02
    cfg.actor_z_film_layer = 2

    cfg.run_tag = "v5i2_stronger_z_conditioning_OP5_OP6_OP7_2m_4v4"
    return cfg


def apply_plan_faithful_latent_v5i3_balanced_warmup(
    cfg: PPOConfig,
) -> PPOConfig:
    """v5i3: forced-z anneal layered on top of v5i2.

    Diagnosis of v5i2: the router collapsed (z2 dominant, z1 near-extinct)
    even though FiLM was wired in. The actor's per-z sensitivity grew but
    only on z values q_phi actually picked, so under-sampled latents stayed
    blind regardless of conditioning bandwidth. v5i3 fixes the *coverage*
    problem the same way exploration noise fixes argmax collapse in plain
    PPO: force a fraction of episodes onto a uniformly-sampled z early in
    training, then anneal that fraction to zero so late training is pure
    router-vs-task-reward.

    Schedule:

    *  ``0 -- 200k``: forced fraction = 0.30. Every latent gets balanced
       actor exposure across roughly the same opponent/phase mix.
    *  ``200k -- 500k``: linearly anneal 0.30 -> 0.00.
    *  ``500k -- 1M``: router-only sampling.

    Forced episodes always route into ``latent_preference_buffer`` and are
    excluded from ``rollout_strategy_episode_records`` (see the
    ``is_forced_z`` branch in ``record_episode_strategy_outcome``), so
    q_phi's PPO update only sees true on-policy episodes; off-policy bias
    on the router is structurally avoided.

    Summer-compatibility: forcing is unlabeled uniform exploration, not
    role assignment. Latent meanings still emerge from task reward via the
    inherited v5i1 episode-credit PPO. The preference-distillation hook
    (``latent_v3i3_event_preference_*``) and router-distill hook stay
    disabled.
    """
    cfg = apply_plan_faithful_latent_v5i2_stronger_z_conditioning(cfg)

    # Anneal schedule. ``resolve_latent_forced_z_frac`` reads these four
    # fields at every episode start; the legacy constant below is set to
    # the start value as a safety so any code that inspects
    # ``cfg.latent_forced_z_episode_frac`` directly (without the resolver)
    # still observes a sane warmup value.
    cfg.latent_forced_z_episode_frac_start = 0.30
    cfg.latent_forced_z_episode_frac_end = 0.00
    cfg.latent_forced_z_anneal_start = 200_000
    cfg.latent_forced_z_anneal_end = 500_000
    cfg.latent_forced_z_episode_frac = 0.30

    # Defensive re-assertions: keep every supervised / preference / distill
    # channel disabled. v5i3 must remain a pure forced-z coverage layer on
    # top of v5i2's router-credit + FiLM stack.
    cfg.latent_behavior_contrast_coef = 0.0
    cfg.latent_actor_z_separation_coef = 0.0
    cfg.latent_actor_z_separation_start_coef = 0.0
    cfg.latent_usage_balance_coef = 0.0
    cfg.latent_preference_coef = 0.0
    cfg.latent_preference_commit_coef = 0.0
    cfg.latent_awrd_enabled = False
    cfg.latent_awrd_coef = 0.0
    cfg.latent_v3i3_event_preference_enabled = False
    cfg.latent_v3i3_event_preference_coef = 0.0
    cfg.latent_router_distill_enabled = False
    cfg.latent_specialist_router_enabled = False

    cfg.run_tag = "v5i3_balanced_warmup_OP5_OP6_OP7_2m_4v4"
    return cfg


def apply_plan_faithful_latent_v5i4_end_to_end(
    cfg: PPOConfig,
) -> PPOConfig:
    """v5i4: paper-faithful conditional-entropy reference row.

    Built directly on ``v5_strict_summer`` (NOT on v5i1/v5i2/v5i3), with one
    correction: the on-policy categorical PPO term on ``q_phi`` is enabled.

    The Summer-plan claim that ``q_phi`` is "trained end-to-end from task
    reward" requires a score-function gradient on the discrete latent --
    persistence and entropy alone do not transmit task-reward information
    into the router. The categorical strategy PPO term

        L_strategy_PPO = - E[ min( rho(z) * A, clip(rho(z), 1+/-eps) * A ) ]

    where ``A`` is the centralized critic's GAE advantage at each
    resample step and ``rho(z) = pi_phi(z|s) / pi_phi_old(z|s)``, is the
    operational implementation of that claim. It belongs inside

        L_MARL = L_actor_PPO + c_V*L_critic + c_Z*L_strategy_PPO
                 + lam_p*L_persist - lam_H*H(q_phi)

    and is not an auxiliary prediction task, label, preference target,
    role assignment, distillation target, or curriculum. The
    ``latent_strategy_ppo_coef`` coefficient is the ``c_Z`` weight.

    What's ON:
    *  Discrete categorical ``z``, ``K = 4``, ``z_embed_dim = 16``.
    *  Sparse resampling every 64 decisions (``latent_resample_every_n = 64``).
    *  Actor reads ``z`` via a plain ``nn.Embedding(K, d_z)`` concatenated
       to local CNN features + scalar ``vec`` (no FiLM, no adapter,
       no one-hot, no opponent/phase info in the actor).
    *  Centralized critic ``V(s, a, z)`` supplies the baseline.
    *  Strategy persistence (``lam_p = 0.03``) and strategy entropy
       (``lam_h`` 0.003 -> 0.0002 schedule inherited from v4i3).
    *  Main-loop ``q_phi`` PPO with ``c_Z = 0.10`` (the paper's task-reward
       gradient channel for the router).

    What's OFF (and must stay off for the paper-faithful claim):
    *  Episode-credit extension (v5i1's per-episode router PPO + dedicated
       AdamW). Mutually exclusive with the per-step main-loop PG above.
    *  FiLM and any other non-concat actor-z mechanism.
    *  Forced-z exploration curriculum (v5i3) -- no labels, no scheduled
       uniform sampling; the router learns purely from on-policy reward.
    *  Arc-credit (v3i19), preference distillation, AWRD, router distill,
       behavior contrast, specialist router, auxiliary return / phase
       prediction heads, and any other post-Summer channel.
    *  Event-triggered switching (``latent_resample_on_flag = False``).
    *  Sparse-tactical refresh and event refresh disabled (inherited
       from the v4i3 chain).

    Relationship to the rest of the v5 ladder:

    |  Run               | q_phi gradient channels                       |
    |--------------------|-----------------------------------------------|
    | v5_strict_summer   | entropy + persistence (NO task-reward signal) |
    | v5i1               | + per-episode credit (dedicated AdamW)        |
    | v5i2               | v5i1 + FiLM (actor-only, no q_phi change)     |
    | v5i3               | v5i2 + forced-z anneal (actor coverage)       |
    | v5i4 (this preset) | conditional entropy + persistence + per-step main-loop PG |

    v5i4 remains the conditional-entropy comparison row because (a) the
    actor is the embedding-concat one docs/algorithm.md specifies
    literally, (b) the q_phi gradient is the main-loop categorical PPO
    that the paper's "learned end-to-end from task reward" wording
    requires, and (c) no label or auxiliary prediction target enters
    anywhere. v5i6 inherits this contract and becomes the canonical
    Summer interpretation by changing only the entropy reduction to the
    batch marginal.

    The launch-time audit banner is emitted by
    ``rl.training.banner`` when the resolved run_tag
    contains ``v5i4_paper_faithful`` so a reviewer can verify the
    invariants at the top of the log without diffing config snapshots.
    """
    cfg = apply_plan_faithful_latent_v5_strict_summer(cfg)

    # ------------------------------------------------------------------
    # Core paper design.
    # ------------------------------------------------------------------
    cfg.use_latent_strategy = True
    cfg.latent_k = 4
    cfg.fixed_latent_strategy = False
    # Sparse switching every 64 decisions. v5_strict_summer already inherits
    # 64 from v4i3, but re-assert here so a future v4i3 change does not
    # silently shift v5i4's cadence.
    cfg.latent_resample_every_n = 64
    cfg.latent_resample_on_flag = False

    # ------------------------------------------------------------------
    # Literal actor architecture: only nn.Embedding(K, d_z) concat.
    # ------------------------------------------------------------------
    cfg.enable_actor_z_film = False
    cfg.actor_z_film_init_scale = 0.0
    cfg.latent_actor_z_adapter_enabled = False
    cfg.latent_actor_z_adapter_scale = 0.0
    cfg.latent_actor_z_onehot_enabled = False
    cfg.latent_actor_z_onehot_scale = 0.0
    cfg.latent_z_embed_dim = 16
    cfg.latent_actor_z_embed_scale = 1.0

    # ------------------------------------------------------------------
    # Main-loop categorical PPO on q_phi (the paper's task-reward channel
    # for the router). NOT the v5i1 episode-credit channel.
    # ------------------------------------------------------------------
    cfg.latent_strategy_ppo_coef = 0.10
    # Defensive: keep the shared optimizer driving q_phi via the main-loop
    # gate. Setting latent_episode_strategy_lr would create a dedicated
    # AdamW that suppresses the main-loop PG (see
    # ``apply_main_loop_qphi_loss`` in ppo_updater.update and the
    # ``MainLoopGatingTests`` in test_marginal_baseline.py).
    cfg.latent_episode_strategy_lr = None
    cfg.latent_episode_strategy_ppo = False
    cfg.latent_episode_strategy_coef = 0.0
    cfg.latent_episode_strategy_warmup_decision_steps = 0
    cfg.latent_episode_strategy_n_epochs = 1

    # ------------------------------------------------------------------
    # Paper regularizers.
    # ------------------------------------------------------------------
    cfg.latent_lam_p = 0.03
    cfg.latent_lam_h = 0.003
    cfg.latent_kl_consecutive = 0.0
    # Entropy maximization is the default; pin explicitly so a future
    # default flip cannot silently invert the sign in v5i4.
    cfg.latent_entropy_mode = "conditional"
    cfg.latent_entropy_objective = "maximize"

    # ------------------------------------------------------------------
    # Forced-z curriculum OFF (constant zero; resolver short-circuits to
    # the legacy field because all four schedule fields are None).
    # ------------------------------------------------------------------
    cfg.latent_forced_z_episode_frac = 0.0
    cfg.latent_forced_z_episode_frac_start = None
    cfg.latent_forced_z_episode_frac_end = None
    cfg.latent_forced_z_anneal_start = None
    cfg.latent_forced_z_anneal_end = None

    # ------------------------------------------------------------------
    # Explicitly disable every non-paper q_phi channel inherited up the
    # chain. Most are already zero in v5_strict_summer; re-asserted here
    # so config-diff at PR time catches any future drift.
    # ------------------------------------------------------------------
    cfg.latent_arc_credit_enabled = False
    cfg.latent_arc_credit_coef = 0.0
    cfg.latent_strategy_aux_return_head = False
    cfg.latent_strategy_aux_return_coef = 0.0
    cfg.latent_strategy_aux_predict_phase_coef = 0.0
    cfg.latent_router_distill_enabled = False
    cfg.latent_v3i3_event_preference_enabled = False
    cfg.latent_v3i3_event_preference_coef = 0.0
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
    cfg.latent_conditional_entropy_min_coef = 0.0
    cfg.latent_conditional_entropy_min_coef_start = 0.0
    cfg.latent_context_mi_coef = 0.0

    # NOTE: budget tag matches the actual PPOConfig default (1_000_000 timesteps).
    # The v5_strict_summer / v5i1 / v5i2 / v5i3 chain inherited a misleading
    # "_2m_" suffix from v4i1's run_tag even though none of those presets ever
    # overrode total_timesteps from its 1M default. v5i4 corrects the tag so
    # the run-tag and the trainer's reported total_timesteps agree.
    cfg.run_tag = "v5i4_paper_faithful_end_to_end_OP5_OP6_OP7_1m_4v4"
    return cfg


def apply_plan_faithful_latent_v5i5_paper_faithful_entropy_floor(
    cfg: PPOConfig,
) -> PPOConfig:
    """v5i5: paper-faithful entropy floor.

    Single-axis follow-up to ``v5i4_paper_faithful_end_to_end``. The v5i4
    run shows the actor uses ``z`` and ``q_phi`` receives gradients, but the
    rollout occupancy concentrates heavily on a single latent (~64% on z2
    vs ~7% on z3 at the 150k checkpoint). The smallest intervention aimed
    directly at that failure mode -- without changing the loss objective or
    introducing any new gradient channel -- is to raise the entropy floor
    so the entropy regularizer keeps a stronger pull on under-sampled
    latents late in training.

    Recipe (one-variable change vs v5i4):

    1. ``latent_lam_h_end = 0.001`` (was ``0.0002``). Five times the v5i4
       floor while still inside the documented Summer-plan
       ``lambda_H in [0.001, 0.01]`` range.
    2. Everything else identical to v5i4: concat-only actor (no FiLM /
       adapter / one-hot), ``latent_strategy_ppo_coef = 0.10`` main-loop
       categorical PPO term on ``q_phi``, ``latent_lam_p = 0.03``,
       ``latent_resample_every_n = 64``, no curriculum, no preferences,
       no aux heads, no arc-credit, no episode-credit.

    Classification: PAPER-FAITHFUL. The single change is a hyperparameter
    inside the plan-allowed entropy range; no fidelity rule (R1..R42 in
    ``docs/summer-fidelity-rules.md``) flips state. The run still fires
    the v5i4-family paper-faithful audit banner.

    Decisive comparison: v5i4 vs v5i5 with identical seed, learning rate,
    timesteps, opponent pool {OP5, OP6, OP7}, maps, reward function,
    resampling interval, network architecture, n_envs, and PPO epochs.
    Multiple seeds recommended for headline claims.

    Diagnostics (added in this PR, no new losses):

    * ``effective_num_latents`` = ``exp(strategy_entropy_marginal_nats)``
    * ``latent_occupancy_min`` / ``latent_occupancy_max`` /
      ``latent_occupancy_ratio = max / max(min, eps)``
    * ``mean_strategy_duration`` (rollout-level mean dwell length in
      decisions per latent arc)

    These let a reviewer separate "stronger entropy preserves useful
    diversity" from "stronger entropy makes the router randomly
    uncertain" without needing a new objective term.

    What's deliberately NOT included (would be a different experiment):

    * episode-credit extension (``latent_episode_strategy_ppo``)
    * forced-z curriculum (v5i3-style ``latent_forced_z_episode_frac_*``)
    * supervised phase or opponent labels
    * opponent-ID input to the actor
    * auxiliary return prediction head
    * FiLM / adapter / one-hot actor conditioning
    * behavior diversity rewards
    * handcrafted latent targets
    * marginal-occupancy entropy reward (covered by the separate v5i6
      marginal-entropy interpretation)
    """
    cfg = apply_plan_faithful_latent_v5i4_end_to_end(cfg)

    # Single-variable change: raise the lam_H floor 0.0002 -> 0.001.
    # ``latent_lam_h_start`` (= 0.003), ``latent_entropy_anneal_start``
    # (= 0), and ``latent_entropy_anneal_end`` (= 300_000) are all
    # inherited unchanged from v5i4 -> v5_strict_summer -> v4i3.
    cfg.latent_lam_h_end = 0.001

    # Run tag rolled forward. The audit banner fires when the tag
    # contains ``v5i5_paper_faithful``; the suffix mirrors v5i4's
    # ``_OP5_OP6_OP7_1m_4v4`` (same opponent pool, same total_timesteps
    # of 1_000_000 inherited from v5_strict_summer) so the artifact
    # namespace is parallel and the v5i4-vs-v5i5 comparison is clean.
    cfg.run_tag = "v5i5_paper_faithful_entropy_floor_OP5_OP6_OP7_1m_4v4"
    return cfg


def apply_plan_faithful_latent_v5i6_paper_faithful_marginal_entropy(
    cfg: PPOConfig,
) -> PPOConfig:
    """v5i6: paper-faithful marginal-entropy Summer interpretation.

    Direct child of ``v5i4_paper_faithful_end_to_end``. This preset keeps
    the v5i4 actor, critic, router PPO, persistence, sparse resampling,
    opponent pool, and no-label/no-curriculum contract unchanged. The
    scientific delta is only the entropy reduction:

    * v5i4 / v5i5: maximize mean conditional entropy E_s[H(q_phi(z|s))].
    * v5i6: maximize batch-marginal entropy H(E_s[q_phi(z|s)]) by
      minimizing KL(E_s[q_phi(z|s)] || Uniform).

    The marginal term is driven by the same lambda_H schedule used by v5i5
    (0.003 -> 0.001 over 0..300k), so v5i6 tests the interpretation of
    H(z) as aggregate strategy-repertoire entropy rather than stacking a
    conditional-entropy bonus on top of a usage-balancing extension.
    """
    cfg = apply_plan_faithful_latent_v5i4_end_to_end(cfg)

    cfg.latent_entropy_mode = "marginal"
    cfg.latent_entropy_objective = "maximize"
    cfg.latent_lam_h_end = 0.001
    cfg.latent_usage_balance_coef = 0.0

    cfg.run_tag = "v5i6_paper_faithful_marginal_entropy_OP5_OP6_OP7_1m_4v4"
    return cfg


def apply_plan_faithful_latent_v5i7_entropy_floor_split_lane(
    cfg: PPOConfig,
) -> PPOConfig:
    """v5i7: v5i5 entropy floor on the split-lane map geometry.

    ## Proposed Preset Review

    ### Identity
    - Proposed name: v5i7_entropy_floor_split_lane
    - Parent preset: v5i5_paper_faithful_entropy_floor
    - Classification: PAPER-FAITHFUL
    - Research question: Does the v5i5 entropy-floor fix produce deployed
      latent routing when the task geometry contains lane/chokepoint choices?

    ### Intended delta
    - Fields changed: map_layout, run_tag
    - Why this change is necessary: v5i5's entropy floor can keep a latent
      repertoire alive, but the open map may not create enough return contrast
      for different z choices to matter.
    - Why an existing preset cannot answer the question: v5i5 tests the same
      latent method on the default open arena; it does not test whether explicit
      route geometry makes strategy choice useful.

    ### Fidelity impact
    - Actor architecture changed: NO
    - Router objective changed: NO
    - Exploration schedule changed: NO
    - Reward changed: NO
    - Supervision added: NO
    - Auxiliary task added: NO
    - Resampling changed: NO

    ### Exact deviations from the paper-faithful preset
    - map_layout: map_a_open -> map_b_split_lane; reason: add lane/chokepoint
      structure while preserving the v5i5 latent loss and training contract.
    - run_tag: v5i5... -> v5i7_summer_faithful_entropy_floor_split_lane...; reason:
      artifact namespace must advertise the environment geometry deviation.

    This remains Summer-faithful by inheriting v5i5's actor, critic, q_phi
    losses, entropy schedule, persistence, sparse resampling, opponent pool,
    and no-label/no-curriculum contract. Any comparison against v5i5/no-latent
    must disclose and match map geometry rather than attributing deltas to the
    latent alone.
    """
    cfg = apply_plan_faithful_latent_v5i5_paper_faithful_entropy_floor(cfg)

    cfg.map_layout = "map_b_split_lane"
    cfg.run_tag = "v5i7_summer_faithful_entropy_floor_split_lane_OP5_OP6_OP7_1m_4v4"
    return cfg


def apply_plan_faithful_latent_v5i8_split_lane_v2_task_pressure(
    cfg: PPOConfig,
) -> PPOConfig:
    """v5i8: v5i7 latent contract on the split-lane v2 task-pressure map.

    ## Proposed Preset Review

    ### Identity
    - Proposed name: v5i8_split_lane_v2_task_pressure
    - Parent preset: v5i7_summer_faithful_entropy_floor_split_lane
    - Classification: PAPER-FAITHFUL
    - Research question: Does lower-friction, higher-route-contrast split-lane
      geometry create enough task-return structure for the existing v5i5
      Summer-faithful latent PPO objective to learn deployed strategies?

    ### Intended delta
    - Fields changed: map_layout, run_tag
    - Why this change is necessary: v5i7's first split-lane geometry produced
      high obstacle-collision counts, so navigation friction may drown out the
      strategic route signal.
    - Why an existing preset cannot answer the question: v5i7 tests the first
      split-lane geometry; v5i8 isolates a task-side geometry revision with no
      latent coefficient or objective change.

    ### Fidelity impact
    - Actor architecture changed: NO
    - Router objective changed: NO
    - Exploration schedule changed: NO
    - Reward changed: NO
    - Supervision added: NO
    - Auxiliary task added: NO
    - Resampling changed: NO

    ### Exact deviations from the paper-faithful preset
    - map_layout: map_b_split_lane -> map_b_split_lane_v2; reason: reduce wall
      bump noise and expose clearer route-pressure choices while preserving the
      latent loss and training contract.
    - run_tag: v5i7... -> v5i8_summer_faithful_split_lane_v2_task_pressure...; reason:
      artifact namespace must advertise the environment geometry revision.
    """
    cfg = apply_plan_faithful_latent_v5i7_entropy_floor_split_lane(cfg)

    cfg.map_layout = "map_b_split_lane_v2"
    cfg.run_tag = "v5i8_summer_faithful_split_lane_v2_task_pressure_OP5_OP6_OP7_1m_4v4"
    return cfg


def apply_plan_faithful_latent_v5i8_repertoire_uniform_z(
    cfg: PPOConfig,
) -> PPOConfig:
    """v5i8 repertoire Stage-1 diagnostic: sustained uniform forced-z coverage.

    ## Proposed Preset Review

    ### Identity
    - Proposed name: v5i8_repertoire_uniform_z
    - Parent preset: v5i8_split_lane_v2_task_pressure
    - Classification: DIAGNOSTIC
    - Research question: Is repertoire failure on v5i8 caused mainly by router
      collapse and unequal per-z experience?

    ### Intended delta
    - Fields changed: ``latent_forced_z_episode_frac*``, ``latent_forced_z_anneal_*``,
      ``run_tag`` only.
    - Why this change is necessary: joint router+actor training lets one z
      dominate experience; this ablation removes router choice for the full run
      so every latent receives uniform episode exposure.
    - Why an existing preset cannot answer the question: v5i8 keeps router
      sampling; v5i3 anneals forced coverage back to zero and inherits FiLM.

    ### Fidelity impact
    - Router objective changed: NO (router receives no on-policy episodes while
      forced fraction is 1.0; this is intentional coverage isolation).
    - Exploration schedule changed: YES (100% uniform forced-z episodes).
    - Reward / actor / map / opponents unchanged vs v5i8.
    """
    cfg = apply_plan_faithful_latent_v5i8_split_lane_v2_task_pressure(cfg)

    cfg.latent_forced_z_episode_frac_start = 1.0
    cfg.latent_forced_z_episode_frac_end = 1.0
    cfg.latent_forced_z_anneal_start = 0
    cfg.latent_forced_z_anneal_end = int(cfg.total_timesteps)
    cfg.latent_forced_z_episode_frac = 1.0

    cfg.run_tag = "v5i8_repertoire_uniform_z_OP5_OP6_OP7_1m_4v4"
    return cfg


def apply_plan_faithful_latent_v5i9_csia_guided_specialization(
    cfg: PPOConfig,
) -> PPOConfig:
    """v5i9: CSIA-guided latent specialization extension on v5i8.

    ## Proposed Preset Review

    ### Identity
    - Proposed name: v5i9_csia_guided_specialization
    - Parent preset: v5i8_split_lane_v2_task_pressure
    - Classification: SUMMER-COMPATIBLE EXTENSION
    - Research question: Can causal strategic-impact feedback from frozen
      forced-z evaluations improve opponent-adaptive latent specialization?

    ### Intended delta
    - Fields changed: csia_enabled, csia_reward_coef, run_tag. The CSIA
      gate thresholds use the PPOConfig defaults unless overridden by CLI.
    - Why this change is necessary: v5i8 can prove whether forced z causes
      strategy differences, but it does not feed that causal evidence back
      into training when specialization is useful but weak or unstable.
    - Why an existing preset cannot answer the question: v5i8 is a
      task-pressure/evaluation row only. It keeps the original reward path
      unchanged and therefore cannot test causal-impact feedback.

    ### Fidelity impact
    - Actor architecture changed: NO
    - Router objective changed: NO
    - Exploration schedule changed: NO
    - Reward changed: YES, via detached CSIA bonus after gates pass
    - Supervision added: NO
    - Auxiliary task added: NO
    - Resampling changed: NO

    ### Exact deviations from the parent
    - csia_enabled: False -> True; reason: enable the v5i9 extension.
    - csia_reward_coef: 0.0 -> 0.02; reason: add a small detached bonus
      proportional to centered causal strategic-impact advantage S(o,z).
    - run_tag: v5i8... -> v5i9_csia_guided_specialization...; reason:
      artifact namespace must advertise the post-Summer reward extension.

    This preset must not be described as the original Summer plan. It keeps
    v5i8's actor, critic, q_phi loss, entropy floor, persistence, resampling,
    opponent pool, and map geometry, but the trainer-side reward is no longer
    the paper-faithful reward once CSIA gates activate.
    """
    cfg = apply_plan_faithful_latent_v5i8_split_lane_v2_task_pressure(cfg)

    cfg.csia_enabled = True
    cfg.csia_reward_coef = 0.02
    cfg.csia_probe_interval = 1
    cfg.csia_min_behavior_spread = 0.10
    cfg.csia_min_interaction_strength = 0.05
    cfg.csia_quality_floor_delta = 0.10
    cfg.csia_require_gates = True
    cfg.csia_min_count_per_cell = 1
    cfg.run_tag = "v5i9_csia_guided_specialization_OP5_OP6_OP7_1m_4v4"
    return cfg


def apply_plan_faithful_latent_v6i1_staged_team_intent_curriculum(
    cfg: PPOConfig,
) -> PPOConfig:
    """v6i1 production staged team-intent curriculum on split-lane v2.

    Inherits v5i8 map/opponent geometry and latent contract, then enables the
    V6I1 phase controller with enforce-mode boundary evaluation and probe.
    Forced-z fraction, CF coefficient, usage KL, and exploration epsilon are
    resolved at runtime from ``resolve_v6i1_*`` schedules — do not set the v5i3
    ``latent_forced_z_anneal_*`` fields on this preset.

    Phase B/C selector learning uses the macro-router path only
    (``apply_macro_strategy_ppo``). Legacy episode-level strategy PPO stays
    off so q_phi is not trained through two overlapping credit channels.
    """
    cfg = apply_plan_faithful_latent_v5i8_split_lane_v2_task_pressure(cfg)

    cfg.use_v6i1_curriculum = True
    cfg.training_mode = "staged_team_intent_curriculum"
    cfg.experiment_family = "v6"
    cfg.experiment_id = "v6i1"
    cfg.phase_boundary_gate_mode = "enforce"
    cfg.curriculum_gate_run_boundary_eval = True
    cfg.curriculum_gate_run_probe = True
    cfg.curriculum_nominal_timesteps = int(cfg.total_timesteps)
    cfg.latent_cf_coef_max = 0.01
    cfg.latent_episode_strategy_ppo = False
    cfg.latent_episode_strategy_coef = 0.0
    cfg.latent_episode_strategy_warmup_decision_steps = 0
    cfg.latent_episode_strategy_lr = None
    cfg.latent_usage_balance_coef = 0.0
    cfg.latent_actor_z_separation_coef = 0.0
    cfg.latent_actor_z_separation_start_coef = 0.0
    cfg.run_tag = "v6i1_staged_team_intent_curriculum_OP5_OP6_OP7_1m_4v4"
    return cfg


def apply_plan_faithful_latent_v6i2_staged_team_intent_curriculum(
    cfg: PPOConfig,
) -> PPOConfig:
    """v6i2 staged team-intent curriculum: dual actor/behavioral gate protocol.

    Inherits the v6i1 actor, critic, router, CF objective, and A/B/C phase
    schedule. Only the Phase A evidence and promotion protocol differs: Gate A
    measures CF-batch actor intervention; Gate B composites matched-seed
    behavioral realization (macro pair JSD is diagnostic only).

    Confirmatory v6i2 uses ``latent_cf_coef_max = 1.0`` (calibrated strong CF)
    plus competence-gated pairwise hinge pressure with a worst-pair term and
    persistent weak-pair weighting; v6i1 retains the weak ``0.01`` baseline for
    threshold calibration.
    """
    cfg = apply_plan_faithful_latent_v6i1_staged_team_intent_curriculum(cfg)

    cfg.experiment_id = "v6i2"
    # Confirmatory v6i2: calibrated strong CF ceiling (v6i1 weak baseline keeps 0.01).
    cfg.latent_cf_coef_max = 1.0
    cfg.latent_cf_worst_pair_coef = 0.5
    cfg.latent_cf_weak_pair_boost = 1.0
    cfg.latent_cf_require_competence = True
    cfg.gate_protocol_version = "v6i2_dual_evidence"
    cfg.phase_a_max_end_fraction = 0.70
    # Frozen v6i2 confirmatory gate thresholds; mirrored in docs/v6i2-gate-protocol-freeze.md.
    cfg.actor_jsd_margin = 0.001
    cfg.actor_jsd_floor_fraction = 0.5
    cfg.actor_jsd_min_passing_pairs = 5
    cfg.actor_jsd_consecutive_updates = 3
    cfg.actor_jsd_ema_decay = 0.10
    cfg.macro_jsd_margin = 0.0001
    cfg.macro_jsd_floor_fraction = 0.5
    cfg.macro_jsd_min_passing_pairs = 1
    cfg.macro_jsd_ema_decay = 0.10
    cfg.behavioral_realization_min_opponents_pass = 2
    cfg.behavioral_realization_effect_threshold = 0.02
    cfg.behavioral_realization_adverse_threshold = -0.01
    cfg.behavioral_route_distance_scale = 0.03
    cfg.behavioral_task_behavior_distance_scale = 0.02
    cfg.behavioral_performance_spread_scale = 0.03
    cfg.behavioral_route_distance_weight = 0.25
    cfg.behavioral_task_behavior_distance_weight = 0.50
    cfg.behavioral_performance_spread_weight = 0.25
    cfg.behavioral_aggregate_effect_threshold = 0.75
    cfg.behavioral_min_task_behavior_distance = 0.01
    cfg.behavioral_min_performance_spread = 0.01
    cfg.behavioral_matched_seed_min_seeds_per_opponent = 20
    cfg.curriculum_probe_min_examples = 10
    cfg.run_tag = "v6i2_staged_team_intent_curriculum_OP5_OP6_OP7_1m_4v4"
    return cfg


def apply_plan_faithful_latent_v6i5_corrected_team_intent_curriculum(
    cfg: PPOConfig,
) -> PPOConfig:
    """v6i5 corrected team-intent curriculum over v6i2.

    SUMMER-COMPATIBLE EXTENSION: q_phi uses current opportunity features plus
    their previous-boundary delta, with marginal entropy and bounded router PPO.
    Actor and critic remain on the existing decentralized/ctx170 contracts.
    """
    cfg = apply_plan_faithful_latent_v6i2_staged_team_intent_curriculum(cfg)

    cfg.experiment_id = "v6i5"
    cfg.latent_entropy_mode = "marginal"
    cfg.latent_entropy_objective = "maximize"
    cfg.latent_resample_every_n = 32
    cfg.v6i1_router_lr = 0.001
    cfg.latent_strategy_ppo_coef = 0.20
    cfg.strategy_target_kl = 0.015
    cfg.router_context_mode = "current_plus_delta"
    cfg.router_context_dimension = 68
    cfg.router_persistence_mode = "expected_switch_detached_previous"
    cfg.router_marginal_entropy_coefficient = 0.001
    cfg.router_conditional_entropy_coefficient = 0.0
    cfg.run_tag = "v6i5_corrected_team_intent_curriculum_OP5_OP6_OP7_1m_4v4"
    return cfg


def apply_plan_faithful_latent_v6i5_router_z0_z3_frozen_actor(
    cfg: PPOConfig,
) -> PPOConfig:
    """v6i5 two-option frozen-repertoire router over original z0 and z3.

    SUMMER-COMPATIBLE EXTENSION: freezes the validated actor repertoire,
    masks q_phi choices to original latent IDs z0/z3, and reinitializes q_phi
    after loading the repertoire checkpoint. No opponent labels or handcrafted
    state-bucket labels are added to the router context.
    """
    cfg = apply_plan_faithful_latent_v6i5_corrected_team_intent_curriculum(cfg)
    cfg.experiment_id = "v6i5_router_z0_z3"
    cfg.router_allowed_latents = (0, 3)
    cfg.router_freeze_actor = True
    cfg.router_reinitialize_on_load = True
    cfg.router_marginal_entropy_coefficient = 0.001
    cfg.router_conditional_entropy_coefficient = 0.0
    cfg.run_tag = "v6i5_router_z0_z3_frozen_actor_OP5_OP6_OP7_1m_4v4"
    return cfg


def apply_plan_faithful_latent_v6i6_strategy_expansion(
    cfg: PPOConfig,
) -> PPOConfig:
    """v6i6 evidence-gated repertoire Expansion Stage E1 over v6i5.

    SUMMER-COMPATIBLE EXTENSION: v6i6 is conditional strategy expansion,
    not an automatic next run. A hashed anchor-validation manifest must
    select anchor_latents, target_latent, and dormant_latents before E1
    training is allowed. The preset intentionally does not hardcode z0/z3
    anchors or a z1 target.
    """
    cfg = apply_plan_faithful_latent_v6i5_corrected_team_intent_curriculum(cfg)
    cfg.experiment_id = "v6i6"
    cfg.gate_protocol_version = "v6i6_repertoire_expansion_e1_v1"
    cfg.use_v6i6_expansion = True
    cfg.v6i6_expansion_protocol_version = "v6i6_repertoire_expansion_e1_v1"
    cfg.v6i6_expansion_stage = "E1"
    cfg.v6i6_require_validated_anchors = True
    cfg.v6i6_anchor_validation_manifest = None
    cfg.v6i6_anchor_latents = ()
    cfg.v6i6_target_latent = -1
    cfg.v6i6_dormant_latents = ()
    cfg.v6i6_fixed_z_episode_attribution = True
    cfg.v6i6_target_episode_fraction = 0.50
    cfg.v6i6_anchor_episode_fraction = 0.50
    cfg.v6i6_trainable_scope = "target_embedding_gate_adapter_only"
    cfg.v6i6_use_reference_critic_for_opportunity = True
    cfg.v6i6_restore_masked_latent_rows_after_step = True
    cfg.v6i6_assert_anchor_bitwise_invariant = True
    cfg.v6i6_count_draw_as = 0.5
    cfg.latent_actor_z_adapter_enabled = True
    cfg.latent_actor_z_adapter_scale = 0.05
    cfg.latent_actor_z_adapter_init_std = 0.0
    cfg.latent_resample_every_n = 0
    cfg.run_tag = "v6i6_strategy_expansion_OP5_OP6_OP7_1m_4v4"
    return cfg


def apply_plan_faithful_latent_v6i4_router_ablation_protocol(
    cfg: PPOConfig,
) -> PPOConfig:
    """v6i4: Summer-plan-faithful router ablation protocol over v6i2.

    v6i4 is evaluation-only. It resolves v6i2 actor/critic/q_phi/repertoire,
    reward, opponent, map, and gate-compatibility fields so a promoted v6i2
    checkpoint can be audited against simpler latent-selection rules. It must
    not start PPO, enter Phase A/B/C, retrain q_phi, or add labels/losses.
    """
    cfg = apply_plan_faithful_latent_v6i2_staged_team_intent_curriculum(cfg)

    cfg.experiment_id = "v6i4"
    cfg.evaluation_only_preset = True
    cfg.evaluation_only_runner = "rl/eval_router_ablation.py"
    cfg.evaluation_only_requires_checkpoint = True
    cfg.evaluation_only_checkpoint_family = "promoted_v6i2"
    cfg.router_ablation_protocol_version = "v6i4_router_ablation_v1"
    cfg.router_ablation_claim_label = (
        "event-associated switching; no opponent-specialized or causal-performance claim "
        "without matched ablations"
    )
    cfg.router_ablation_classification = (
        "v6i4 is a Summer-plan-faithful, evaluation-only router-ablation protocol "
        "over a frozen, Phase-A-promoted v6i2 checkpoint. It is currently "
        "planned/pending. No parameters are trained or updated."
    )
    cfg.router_ablation_classification = (
        "Summer-plan-faithful evaluation-only router ablation protocol over promoted v6i2 checkpoint"
    )
    cfg.router_ablation_conditions = (
        "learned_qphi_switching",
        "uniform_episode_fixed",
        "uniform_random_at_router_opportunities",
        "preselected_global_fixed_z",
        "preselected_per_opponent_fixed_z",
        "fixed_z0",
        "fixed_z1",
        "fixed_z2",
        "fixed_z3",
        "qphi_initial_only_no_switch",
        "shuffled_qphi_outputs",
    )
    cfg.router_ablation_conditions = tuple(
        name
        for name in cfg.router_ablation_conditions
        if name != "preselected_per_opponent_fixed_z"
    )
    cfg.router_ablation_oracle_conditions = (
        "posthoc_global_fixed_oracle",
        "posthoc_opponent_oracle",
        "posthoc_episode_oracle",
    )
    cfg.router_ablation_primary_metrics = (
        "return",
        "win_rate",
        "delta_vs_uniform_episode_fixed",
        "delta_vs_uniform_random_at_router_opportunities",
        "delta_vs_preselected_global_fixed_z",
        "delta_vs_qphi_initial_only_no_switch",
        "delta_vs_shuffled_qphi_outputs",
    )
    cfg.router_ablation_diagnostic_metrics = (
        "route_distance",
        "task_behavior_distance",
        "latent_occupancy",
        "strategy_entropy",
        "mi_z_opponent",
        "mi_z_phase",
        "argmax_stability",
        "event_associated_switching",
    )
    cfg.router_ablation_opponents = ("OP5", "OP6", "OP7")
    cfg.router_ablation_calibration_seed_set = "locked_calibration_seeds"
    cfg.router_ablation_evaluation_seed_set = "disjoint_matched_evaluation_seeds"
    cfg.router_ablation_matched_seeds = True
    cfg.router_ablation_identical_initial_states = True
    cfg.router_ablation_identical_action_sampling = True
    cfg.router_ablation_identical_episode_horizon = True
    cfg.router_ablation_episode_oracle_is_deployable = False
    cfg.run_tag = "v6i4_router_ablation_protocol_over_v6i2_OP5_OP6_OP7_1m_4v4"
    return cfg


def apply_plan_faithful_latent_v6i3_strategy_local_comm(
    cfg: PPOConfig,
) -> PPOConfig:
    """v6i3: v6i2 staged curriculum plus local emergent communication.

    Inherits v6i2 dual-evidence gates and adds communication transport as
    Phase A evidence. Listener causal response remains a diagnostic until
    final matched-seed communication-value evaluation.
    """
    cfg = apply_plan_faithful_latent_v6i2_staged_team_intent_curriculum(cfg)

    cfg.experiment_id = "v6i3"
    cfg.gate_protocol_version = "v6i3_strategy_local_comm_v1"
    cfg.communication_enabled = True
    cfg.comm_protocol_version = "v6i3_strategy_local_comm_v1"
    cfg.comm_num_symbols = 5
    cfg.comm_silence_symbol = 0
    cfg.comm_interval_steps = 32
    cfg.comm_delivery_delay_steps = 1
    cfg.comm_radius_cells = 6.0
    cfg.comm_dropout_probability = 0.10
    cfg.comm_entropy_coef = 0.001
    cfg.comm_hold_last_message = True
    cfg.comm_local_only = True
    cfg.comm_include_sender_position = True
    cfg.comm_message_grid_channels = 4
    cfg.comm_cf_include_message_head = False
    # Frozen v6i3 Phase A communication evidence gates; listener response is diagnostic.
    cfg.comm_min_valid_boundaries = 1024
    cfg.comm_min_deliveries = 4096
    cfg.comm_min_symbols_used = 2
    cfg.comm_entropy_floor = 0.0
    cfg.comm_symbol_dominance_ceiling = 1.0
    cfg.comm_listener_jsd_margin = 0.001
    cfg.comm_listener_min_passing_pairs = 3
    cfg.comm_listener_min_states = 64
    cfg.comm_listener_consecutive_updates = 1
    cfg.run_tag = "v6i3_strategy_local_comm_OP5_OP6_OP7_1m_4v4"
    return cfg


def apply_plan_faithful_latent_v6i1_repertoire_only_ablation(
    cfg: PPOConfig,
) -> PPOConfig:
    """v6i1 repertoire-only ablation: uniform forced-z, no staged controller.

    Shares the v6i1 experiment id for artifact grouping but must never mount
    the staged curriculum controller because ``use_v6i1_curriculum=False`` and
    ``training_mode=repertoire_only_ablation``.
    """
    cfg = apply_plan_faithful_latent_v5i8_split_lane_v2_task_pressure(cfg)

    cfg.use_v6i1_curriculum = False
    cfg.training_mode = "repertoire_only_ablation"
    cfg.experiment_family = "v6"
    cfg.experiment_id = "v6i1"
    cfg.latent_forced_z_episode_frac_start = 1.0
    cfg.latent_forced_z_episode_frac_end = 1.0
    cfg.latent_forced_z_anneal_start = 0
    cfg.latent_forced_z_anneal_end = int(cfg.total_timesteps)
    cfg.latent_forced_z_episode_frac = 1.0
    cfg.run_tag = "v6i1_repertoire_only_ablation_OP5_OP6_OP7_1m_4v4"
    return cfg
