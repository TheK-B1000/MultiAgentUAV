"""v3 router presets — marginal baseline through v3h2 balanced preference."""
from __future__ import annotations

from rl.config.ppo_config import PPOConfig, TrainMode

from .base import apply_plan_faithful_latent_episode_strategic


def apply_plan_faithful_latent_v3b_marginal(cfg: PPOConfig) -> PPOConfig:
    """v3b: episode-credit with z-marginal baseline (fixes baseline-eating).

    The v3 episode-credit + warmup + entropy-anneal run held PPO stable
    (~65% WR, EV 0.61) but produced effectively zero MI(z; opponent) and
    kept ``zH`` pinned at maximum entropy even after ``lam_h`` annealed
    to the floor. Code audit identified the math root cause:

        adv_z = R - V(s, z_picked)

    The centralized critic ``V(s, z_picked)`` already absorbs ``E[R | s, z]``,
    so the advantage that arrives at q_phi is "this z vs its own expectation"
    -- i.e. mostly within-z policy noise. The cross-z signal q_phi needs to
    learn from is mathematically subtracted before the gradient is computed.

    v3b applies the variance-optimal AAC fix while keeping everything else
    identical to v3. ``adv`` now uses a z-marginal baseline:

        adv_z = R - E_{z' ~ q_phi(s)}[V(s, z')] = R - sum_k pi_phi(k|s) * V(s, k)

    so the advantage encodes "this z vs the average available z in this
    context" -- the exact signal contextual specialization needs.

    Inherits ``apply_plan_faithful_latent_episode_strategic`` (which already
    has warmup=5, lam_p=0, lam_h anneal 0.003 -> 0.0005 from 200k -> 700k,
    K=4, ctx170, episode-credit on, per-step coupling off). Flips one knob:

      - ``latent_q_phi_marginal_baseline = True``

    Plan-faithful: no labels, no aux heads, no opponent IDs. Only the
    baseline math changes.

    Expected first signs of working (per implementation review):
      ~100k: latent_episode_pg_loss meaningfully nonzero, adv_std > 0.5
      ~300k: latent_episode_approx_kl consistent in 0.001-0.01 range
      ~500k: zH_frac drops below 0.95 (q_phi moves off uniform)
      ~700k: MI(z; opponent) above 0.02 if the fix works
      ~1M:   WR matches the 65% baseline, MI > 0.02

    If MI is still pinned near 0 by 700k under v3b, the problem is deeper
    than the baseline -- next experiment becomes V calibration on off-policy
    z slots (train V(s, z) on all K z per state, not just the picked one)
    or expand the actor's z-conditioning capacity (z_emb 16 -> 32).
    """
    cfg = apply_plan_faithful_latent_episode_strategic(cfg)
    cfg.latent_q_phi_marginal_baseline = True
    cfg.run_tag = "latent_v3b_episodecredit_marginalbaseline_warmup5_lamHanneal_1m_4v4"
    return cfg


def apply_plan_faithful_latent_v3c_router_lr(cfg: PPOConfig) -> PPOConfig:
    """v3c: amplify the router update strength on top of v3b's marginal baseline.

    v3b's experimental finding: the marginal baseline successfully unblocked
    q_phi's gradient signal (``episode_credit_grad_norm`` was non-zero from
    update 1, ~0.005-0.027 per update) but cumulative logit change over a
    1M-step run was only ~10^-5 -- five orders of magnitude short of the
    ~ln(2) ≈ 0.7 needed to differentiate K=4 strategies. q_phi stayed at
    max entropy (zH_frac=1.0), MI(z; opponent) stayed at noise floor.

    Diagnosis (per implementation review): two compounding constraints on the
    router's effective step size:

      (1) ``apply_episode_strategy_ppo`` runs ONE backward step per rollout
          (vs the actor's 6-8 PPO inner epochs). That alone is a ~7x signal
          reduction.

      (2) The shared optimizer's LR (1.35e-4 for 4v4) is calibrated for the
          noisy actor gradient. q_phi's per-step gradient is clean but small
          (~0.01), and at this LR moves logits by only ~1.35e-6 per update.

    v3c lifts both constraints with config-only changes (no architecture
    surgery, no labels, no aux heads):

      - ``latent_episode_strategy_n_epochs = 6``  → q_phi gets PPO inner epochs
        like the actor. The first epoch is ratio==1 REINFORCE-style; epochs 2-6
        see the ratio drift away from 1 and the clipped PG actually does work.

      - ``latent_episode_strategy_lr = 5e-3``     → dedicated AdamW for the
        strategy_encoder + episode_strategy_value_head at ~37x the shared LR.
        Combined with (1), the effective per-update step grows ~7 × 37 = ~260x.

      - ``latent_lam_h_end = 0.001`` (vs v3b's 0.0005) → slightly higher entropy
        floor as collapse insurance. If MI growth threatens to winner-take-all
        one z (because V is poorly calibrated for the disused z slots), the
        entropy regularizer holds the distribution open. Tighten back to 0.0005
        in a follow-up if collapse doesn't materialize.

    Plan-faithful: no labels, no opponent IDs, no aux heads, no Gumbel tricks,
    no imitation. Only changes router update strength (epochs + LR) and a tiny
    entropy floor adjustment.

    Hypothesis tested: "q_phi has the correct reward-derived gradient, but the
    current number of router updates and shared LR are too small to move logits."
    Falsifiable: if zH_frac stays at 1.0 and MI stays at noise floor under
    v3c, the bottleneck is no longer training strength -- next experiment
    becomes V calibration on off-policy z slots or z_embed capacity.

    Expected first signs of working:
      ~50k:  episode-credit ratio drifts off [1.000, 1.000]; clip_fraction > 0
      ~100k: zH_frac drops below 0.99 (first measurable router movement)
      ~300k: MI(z; opponent) above 0.02 if hypothesis is right
      ~700k: zH_frac in 0.80-0.95, MI(z; opponent) > 0.05 if it sharpens
      ~1M:   WR matches v3b 67% baseline, MI > 0.05, occupancy biased not collapsed
    """
    cfg = apply_plan_faithful_latent_v3b_marginal(cfg)
    cfg.latent_episode_strategy_n_epochs = 6
    cfg.latent_episode_strategy_lr = 5e-3
    cfg.latent_lam_h_end = 0.001
    cfg.run_tag = "latent_v3c_routerlr_epochs6_lr5e3_lamHfloor1e3_1m_4v4"
    return cfg


def apply_plan_faithful_latent_v3d_smart_router(cfg: PPOConfig) -> PPOConfig:
    """v3d: context-bucketed marginal baseline ("smart coach router").

    The v3c experimental finding (~851k steps): the dedicated router
    optimizer + 6 inner epochs DID move q_phi off uniform (zH dropped to
    1.335(0.96), z_occ went [.25, .20, .38, .17]). Router gradient is alive
    (episode-credit grad_norm ~0.02). But the movement is "global z preference"
    (z2 is just popular) not "context-conditioned z selection" (different z
    for different situations). MI(z; opponent) still hovering near noise.

    Diagnosis (per analyst): the v3c V-marginal baseline ``mean_k V(s, z_k)``
    depends on V being well-calibrated for off-policy z slots, but V only sees
    value-loss updates for episodes where each z was *actually picked* (~25%
    at uniform). So the marginal baseline subtracts noise approximating the
    right thing -- the cross-z signal q_phi needs is fuzzy.

    v3d replaces the baseline with an *empirical* per-bucket mean of episode
    returns::

        v3c:  adv = R - mean_k V(s, z_k)        # V-marginal
        v3d:  adv = R - mean(R | bucket(s))     # bucket-empirical

    where ``bucket`` defaults to the scripted opponent id (3 buckets for
    OP3/OP5/OP6 in the standard tough pool). q_phi now learns "is this z
    better than the average z WITHIN this opponent's episodes?" rather than
    "better than overall average?". This is variance reduction by
    stratification -- standard Monte Carlo technique, no V noise, no
    architecture changes.

    Plan-faithful guarantee: the bucket id is a GRADIENT-SHAPING signal
    (input to the baseline), NEVER a policy input. q_phi still sees only
    ``s`` and learns ``pi(z|s)``. The bucket only affects the variance of
    the estimator. Two episodes vs OP5 where z=2 won and z=2 lost
    contribute oppositely-signed advantages -- q_phi must still discover
    from ``s`` alone which z to pick under each context.

    Inherits everything from v3c:
      - latent_episode_strategy_n_epochs = 6
      - latent_episode_strategy_lr = 5e-3
      - latent_q_phi_marginal_baseline = True (kept on as a fallback path,
        though v3d's bucket baseline takes priority in apply_episode_strategy_ppo)
      - lamH anneal 0.003 -> 0.001 from 200k -> 700k
      - warmup=5, K=4, ctx170, episode-credit on, coef==0 main-loop gate

    Sets:
      - latent_q_phi_bucket_baseline = "opponent"
      - latent_q_phi_bucket_baseline_ema = 0.9
      - latent_q_phi_bucket_baseline_min_count = 8

    Why "opponent" only at the start (not "opponent_x_bucket")?
      "opponent" gives 3 buckets with ~1000 episodes each per rollout --
      extremely robust per-bucket means. "opponent_x_bucket" splits ~3000
      episodes across ~648 buckets (~5 episodes each), well below min_count,
      so the fallback-to-global path would dominate -- defeating the purpose.
      If v3d works but MI plateaus, the next iteration adds the bucket_id
      composite for sharper context conditioning.

    Hypothesis tested: "q_phi's gradient direction is correct (v3c showed
    movement), but the V-marginal baseline is too noisy for off-policy z to
    produce CONTEXT-CONDITIONED specialization. An empirical bucket mean
    bypasses V entirely and exposes the within-opponent cross-z signal."

    Expected first signs of working (per analyst's success criteria):
      ~50k:   [bucket-baseline] var_reduction < 1.0 (R_std visibly larger
              than adv_std; opponent stratification is removing return
              variance the marginal baseline missed)
      ~200k:  MI(z; opponent) above 0.02 (first real cross-opponent
              differentiation -- this is the metric v3c could not move)
      ~500k:  z_wr_spread > 0.15 (z choices materially affect WR per
              opponent), per-opponent z_occ visibly non-uniform per opponent
      ~700k:  MI(z; opponent) > 0.05, z_wr_spread > 0.20, WR matches or
              beats v3c's 67% baseline

    If MI(z; opponent) stays at noise floor under v3d, the bottleneck is
    no longer signal quality -- next experiment becomes either (a) larger
    z_embedding (16 -> 32 dims) for richer per-z conditioning, or (b)
    auxiliary V-calibration on off-policy z (train V(s, z) on ALL K z per
    state via the bucket-mean as a target, not just the picked one).
    """
    cfg = apply_plan_faithful_latent_v3c_router_lr(cfg)
    cfg.latent_q_phi_bucket_baseline = "opponent"
    cfg.latent_q_phi_bucket_baseline_ema = 0.9
    cfg.latent_q_phi_bucket_baseline_min_count = 8
    cfg.run_tag = "latent_v3d_smartrouter_bucketopp_ema09_min8_1m_4v4"
    return cfg


def apply_plan_faithful_latent_v3d_delayed_anneal(cfg: PPOConfig) -> PPOConfig:
    """v3d variant: same smart-router baseline, delayed entropy anneal.

    Motivating diagnosis from the live v3d run: at ~327k the bucket baseline +
    dedicated router LR are already pushing meaningful router updates
    (``kl=0.0150``, ``grad_norm=0.08``, ``z_occ`` non-uniform) WHILE the
    entropy schedule is still in its 200k-700k decay window (lamH already at
    ~0.0025 by 327k). That means two opposing forces are active at the same
    time:

      (1) The smart router pushing q_phi to commit to whichever z scored best
          in each bucket so far -- with strong updates that early movement
          gets locked in fast.

      (2) The entropy regularizer pulling q_phi back toward uniform -- but
          the leash is already loosening.

    Risk: q_phi latches onto z_k that won during the OPENING ROLLOUTS, where
    the actor hadn't yet had enough updates under all K strategies. Result is
    "louder, not wiser" -- a self-fulfilling z preference where unused z slots
    starve for training because the router stopped sampling them.

    This variant decouples the timing: same start/floor entropy (0.003 -> 0.001
    keeps the curiosity high early and the floor low enough for selection
    late), but the decay starts later and ends later, giving the actor an
    extra 100k steps of fully-uniform-sampled rollouts under all K strategies
    before the entropy leash starts to loosen.

    Plan-faithful: identical to v3d in every other respect. Only changes
    ``latent_entropy_anneal_start`` (200k -> 300k) and
    ``latent_entropy_anneal_end`` (700k -> 800k). Bucket baseline still
    "opponent", router LR still 5e-3, n_epochs still 6.

    When to launch:
      - v3d's MI(z; opponent) stalls below 0.02 by 500k.
      - OR v3d's z_occ shows premature one-z dominance (any slot > 0.5
        before 500k).
      - OR z_wr_spread starts declining (router locking in suboptimally).

    When NOT to launch:
      - v3d's MI is climbing healthily. Don't fix what isn't broken.
      - z_occ stays in the [.15, .40] band across all four z slots through
        the run. That's a router doing its job; no need to give it more
        exploration runway.
    """
    cfg = apply_plan_faithful_latent_v3d_smart_router(cfg)
    cfg.latent_entropy_anneal_start = 300_000
    cfg.latent_entropy_anneal_end = 800_000
    cfg.run_tag = "latent_v3d_delayedanneal_300k_800k_bucketopp_1m_4v4"
    return cfg


def apply_plan_faithful_latent_v3e_strong_z_actor(cfg: PPOConfig) -> PPOConfig:
    """v3e: strong z actor preset.

    Inherits from v3d_delayed_anneal and configures:
      - latent_z_embed_dim: 16 -> 32
      - actor_hidden_dim: 256 -> 384
      - run_tag: latent_v3e_strong_z_actor_1m_4v4
    """
    cfg = apply_plan_faithful_latent_v3d_delayed_anneal(cfg)
    cfg.latent_z_embed_dim = 32
    cfg.actor_hidden_dim = 384
    cfg.run_tag = "latent_v3e_strong_z_actor_1m_4v4"
    return cfg


def apply_plan_faithful_latent_v3f_behavior_contrast(cfg: PPOConfig) -> PPOConfig:
    """v3f: self-supervised latent behavior contrast.

    Inherits v3e's stronger z-conditioned actor and v3d's episode-credit
    router, then adds label-free option separation:

      - 30% forced-z episodes, uniformly sampled across K
      - completed-episode behavior contrast bonus on forced-z episodes only
      - q_phi episode-credit delayed until the actor has seen forced-z data
      - weak aggregate q_phi usage balance, applied only inside q_phi credit

    Plan-faithful: no role labels, no scripted z meanings, no opponent-ID
    heads, no supervised router targets. The behavior embedding is built from
    existing observable team telemetry and compared inside coarse game-state
    buckets so the pressure is "different modes under similar contexts."
    """
    cfg = apply_plan_faithful_latent_v3e_strong_z_actor(cfg)
    cfg.latent_forced_z_episode_frac = 0.30
    cfg.latent_behavior_contrast_coef = 0.05
    cfg.latent_behavior_contrast_margin = 0.25
    cfg.latent_behavior_contrast_ema = 0.90
    cfg.latent_behavior_contrast_anneal_after_steps = 800_000
    cfg.latent_behavior_contrast_anneal_to = 0.005
    cfg.latent_usage_balance_coef = 0.01
    cfg.latent_q_phi_train_after_steps = 100_000
    cfg.run_tag = "latent_v3f_behavior_contrast_1m_4v4"
    return cfg


def apply_plan_faithful_latent_v3g_preference(cfg: PPOConfig) -> PPOConfig:
    """v3g: self-supervised latent preference distillation from forced-z.

    Inherits v3f's contrastive separation (forced-z exploration + actor conditioning)
    and smart router, then distills a soft target probability distribution from
    the returns of forced-z episodes into q_phi.
    """
    cfg = apply_plan_faithful_latent_v3f_behavior_contrast(cfg)
    cfg.latent_preference_coef = 0.03
    cfg.latent_preference_temperature = 0.75
    cfg.latent_preference_min_bucket_count = 8
    cfg.latent_preference_min_distinct_z = 2
    cfg.run_tag = "latent_v3g_preference_1m_4v4"
    return cfg


def apply_plan_faithful_latent_v3h_balanced_preference(cfg: PPOConfig) -> PPOConfig:
    """v3h: self-supervised latent preference distillation with opponent-balanced KL loss and target telemetry."""
    cfg = apply_plan_faithful_latent_v3g_preference(cfg)
    cfg.latent_preference_coef = 0.03
    cfg.latent_preference_opponent_balanced = True
    cfg.latent_preference_log_opponent_targets = True
    cfg.run_tag = "latent_v3h_balanced_preference_1m_4v4"
    return cfg


def apply_plan_faithful_latent_v3h2_balanced_preference(cfg: PPOConfig) -> PPOConfig:
    """v3h2: self-supervised latent preference distillation with confidence-weighted KL + entropy commitment."""
    cfg = apply_plan_faithful_latent_v3h_balanced_preference(cfg)
    cfg.latent_preference_confidence_scale = 2.0
    cfg.latent_preference_commit_coef = 0.003
    cfg.late_entropy_floor = 0.0003
    cfg.commitment_type = "confidence_weighted_entropy"
    # Entropy schedule:
    # 0 - 300k steps: lam_h = 0.003
    # 300k - 600k steps: linear anneal from 0.003 to 0.001
    # 600k+ steps: linear anneal from 0.001 to late_floor (0.0003) at total_timesteps
    cfg.latent_lam_h = 0.003
    cfg.latent_lam_h_start = 0.003
    cfg.latent_lam_h_end = 0.0003
    cfg.latent_entropy_anneal_start = 300_000
    cfg.latent_entropy_anneal_end = 600_000
    cfg.run_tag = "latent_v3h2_balanced_preference_1m_4v4"
    return cfg
