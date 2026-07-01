"""v4 proof presets — strategic pressure qprobe through v4i4post periodic router distill."""
from __future__ import annotations

from rl.config.ppo_config import PPOConfig, TrainMode

from .v3i_consequence import apply_plan_faithful_latent_v3i19_summer_consequence


def apply_plan_faithful_latent_v4i1_strategic_pressure_qprobe(cfg: PPOConfig) -> PPOConfig:
    """v4i1: Strategic Pressure Benchmark + Offline Return Contrast Probe.

    Inherits v3i19 verbatim. The only deltas are:

    1. The opponent pool is restricted to ``{OP5, OP6, OP7}`` so different z
       values have a strategic reason to differ. v3i18/v3i19 trained against
       OP0..OP6 mixtures that included free-win opponents; the agent could
       win without specializing, so the latent had no job. v4i1 removes
       those easy opponents to force the environment to reward distinct
       strategies (OP5 = aggressive flag rush, OP6 = defensive turtle,
       OP7 = switcher / coordination-and-timing).
    2. The run_tag is updated.

    The latent machinery is **intentionally unchanged** -- v4i1 stops
    changing the brain and changes the world instead. Same K=4, same q_phi,
    same arc-credit, same FiLM+onehot actor conditioning, same entropy
    schedule.

    Primary metric for this run is computed OUT-OF-BAND by
    ``tools/q_probe.py``:

        return_contrast = max_z(R) - min_z(R)

    where R is the mean undiscounted episode return per forced z across
    matched probe seeds, per opponent. Failure: contrast < 0.05 means the
    environment does not care about strategy (escalate to Environment v2).
    Success: contrast >= 0.10-0.20 means different z choices create
    different outcomes (proceed to v4i2 = latent regret specialization).

    All existing in-trainer latent diagnostics (MI(z;*), policy_z_sensitivity_KL,
    actor_z_jsd, H(z), behavior_by_z) keep emitting unchanged and are
    demoted to secondary signals.
    """
    cfg = apply_plan_faithful_latent_v3i19_summer_consequence(cfg)

    cfg.opponent_randomize = True
    cfg.opponent_pool = ("OP5", "OP6", "OP7")
    cfg.opponent_pool_weights = ()

    cfg.run_tag = "v4i1_strategic_pressure_qprobe_OP5_OP6_OP7_2m_4v4"
    return cfg


def apply_plan_faithful_latent_v4i3_summer_proof(cfg: PPOConfig) -> PPOConfig:
    """v4i3 (canonical): Summer-Faithful Proof Suite training preset.

    Thesis: under the locked Summer design, does a discrete shared latent
    strategy ``z`` become a meaningful team-level coordination signal?

    v4i3 inherits v4i1 verbatim. The point of v4i3 is **not** to add new
    latent machinery -- it is to *prove or falsify* the Summer plan
    cleanly. No distillation, no auxiliary heads, no labels, no router
    tutoring. The Summer plan's strict claim is that q_phi learns to use
    z end-to-end from reward alone; v4i3 is the experiment that tests
    that claim with proper baselines and proper counterfactual probing
    (see ``tools/q_probe_local_counterfactual.py`` and
    ``tools/summer_proof_report.py``).

    All deltas vs v4i1 are defensive re-assertions (audit clarity). The
    actual config is identical to v4i1 except for the run_tag and the
    explicit guards on post-Summer extensions:

    * ``latent_router_distill_enabled = False`` -- router distillation is
      a v4i4 extension; v4i3 must be a faithful Summer run.
    * ``latent_strategy_aux_predict_phase_coef = 0.0`` and
      ``latent_strategy_aux_return_head = False`` -- no auxiliary
      prediction heads. The Summer plan is strict about z being learned
      end-to-end from task reward.

    Proof artifacts produced after training:

    * Fixed-z q_probe (``tools/q_probe.py``)  -- forced-z return contrast
      per (opp, seed) at matched starts; proves latent modes exist.
    * Local counterfactual probe
      (``tools/q_probe_local_counterfactual.py``) -- at each arc boundary,
      snapshot env state, force each z, roll to completion. Proves
      Q(s, z) contrast at the exact decision points where q_phi acts.
    * No-latent baseline (``apply_plan_faithful_no_latent_v4i3_baseline``)
      run at the same budget; proves the gain from z (if any) over a
      same-everything-except-latent control.
    * Natural q_phi rollout vs. fixed-z oracle and random-z baseline
      (``tools/qualitative_rollout.py``) -- proves q_phi is routing.
    * Summary report (``tools/summer_proof_report.py``) gates 1-5 of the
      Summer-Proof spec.

    If v4i3 passes the gates, the Summer plan is alive. If it fails, the
    honest follow-up is v4i4 (counterfactual router refinement) framed
    as a clearly-labelled post-Summer extension, not a "fix" of the
    Summer plan itself.
    """
    cfg = apply_plan_faithful_latent_v4i1_strategic_pressure_qprobe(cfg)

    # Summer-faithful latent machinery (all already set by v3i19 chain;
    # re-asserted here so any drift in upstream presets is caught by
    # config-diff at PR review time, NOT at run start). Same K=4, same
    # 64-step sparse refresh, same lam_p / lam_h_start / lam_h_end, same
    # arc-credit recipe.
    cfg.latent_k = 4
    cfg.latent_resample_every_n = 64
    cfg.latent_resample_on_flag = False
    cfg.latent_lam_p = 0.03
    cfg.latent_lam_h_start = 0.003
    cfg.latent_lam_h_end = 0.0002
    cfg.latent_strategy_ppo_coef = 0.0
    cfg.latent_arc_credit_enabled = True
    cfg.latent_arc_credit_coef = 1.0

    # Strategic-pressure pool (v4i1 already sets this; re-assert so the
    # CLI guard at training/cli.py is not the only place this is enforced).
    cfg.opponent_randomize = True
    cfg.opponent_pool = ("OP5", "OP6", "OP7")
    cfg.opponent_pool_weights = ()

    # Explicitly OFF -- post-Summer extensions that v4i3 must NOT include.
    # ``latent_router_distill_enabled`` is the v4i4post periodic-distill
    # hook. The aux predict-phase / return heads are forbidden by the
    # Summer plan's "no auxiliary objectives" clause.
    cfg.latent_router_distill_enabled = False
    cfg.latent_strategy_aux_predict_phase_coef = 0.0
    cfg.latent_strategy_aux_return_head = False
    cfg.latent_strategy_aux_return_coef = 0.0

    # Run tag is budget-agnostic on purpose: the preset locks the
    # Summer-faithful machinery, NOT a specific ``--total-steps``. Probes
    # at smaller budgets (e.g. 1M) and the locked 2M proof run share the
    # same artifacts namespace; if you need separate trees, pass
    # ``--run-tag`` to override.
    cfg.run_tag = "v4i3_summer_proof_OP5_OP6_OP7_4v4"
    return cfg


def apply_plan_faithful_no_latent_v4i3_baseline(cfg: PPOConfig) -> PPOConfig:
    """v4i3 no-latent baseline: the same-everything-except-z control.

    The Summer plan calls the no-latent ablation decisive: replace
    ``pi(a | o, z)`` with ``pi(a | o)`` and keep everything else identical.
    To honour "everything else identical", this preset inherits v4i1
    verbatim (same reward, same arc-credit math, same entropy schedule,
    same opponent pool ``{OP5, OP6, OP7}``, same PPO knobs, same map,
    same n_envs, same n_epochs, same n_steps, same total budget) and
    flips ONLY ``use_latent_strategy = False``.

    Important note about the ancestry choice: there is no pre-latent
    v3iN ancestor in the file that mirrors v4i1's reward / opponent
    pool / arc-credit machinery. ``apply_plan_faithful_no_latent`` (the
    legacy 1M-step 2v2 OP3 baseline) does NOT mirror v4i1; using it as
    a control would confound the latent ablation with ~8 other deltas
    (timesteps, team size, opponent pool, reward shaping, arc-credit
    on/off, FiLM scaffolding, ...). Inheriting v4i1-and-flipping is the
    only honest way to do the ablation in this codebase.

    Latent-only coefficients (``latent_arc_credit_*``,
    ``latent_episode_strategy_*``, ``latent_lam_*``, ``latent_actor_z_*``,
    ``latent_router_distill_*``) become no-ops when
    ``use_latent_strategy = False``; we still defensively zero the most
    consequential ones for audit clarity. Anything related to z that
    survives must be either (a) a pure config field with no runtime
    consequence under no-latent, or (b) a bug that needs fixing.
    """
    cfg = apply_plan_faithful_latent_v4i1_strategic_pressure_qprobe(cfg)

    cfg.use_latent_strategy = False
    cfg.fixed_latent_strategy = False
    cfg.latent_arc_credit_enabled = False
    cfg.latent_arc_credit_coef = 0.0
    cfg.latent_strategy_ppo_coef = 0.0
    cfg.latent_episode_strategy_ppo = False
    cfg.latent_episode_strategy_coef = 0.0
    cfg.latent_strategy_aux_return_head = False
    cfg.latent_strategy_aux_return_coef = 0.0
    cfg.latent_strategy_aux_predict_phase_coef = 0.0
    cfg.latent_router_distill_enabled = False
    cfg.latent_actor_z_onehot_enabled = False

    # Same budget-agnostic naming convention as the latent preset above.
    cfg.run_tag = "v4i3_no_latent_baseline_OP5_OP6_OP7_4v4"
    return cfg


def apply_plan_faithful_latent_v4i4post_periodic_router_distill(cfg: PPOConfig) -> PPOConfig:
    """v4i4 (post-Summer extension): Online / Periodic Return-Ranked Router Distillation.

    NOTE: This preset was originally named ``v4i3_periodic_router_distill``,
    but the canonical v4i3 was rescoped to the **Summer Proof Suite**
    (see :func:`apply_plan_faithful_latent_v4i3_summer_proof`). The
    periodic-distill recipe is now explicitly framed as a **post-Summer
    extension** because it introduces counterfactual router supervision,
    which the Summer plan's "no labels, no auxiliary objectives" clause
    forbids. v4i4 is meaningful only AFTER v4i3 has either passed or
    failed its gates; if v4i3 passes, v4i4 is icing; if v4i3 fails, v4i4
    is the next honest experiment.

    Inherits the v4i1 strategic-pressure setup verbatim (same actor, critic,
    reward, opponent pool ``{OP5, OP6, OP7}``, arc-credit math, entropy
    schedule, and PPO loop). The only delta versus v4i1 is that the trainer
    enables :class:`PeriodicRouterDistillHook`: every
    ``latent_router_distill_every_n_steps`` global steps, the trainer pauses
    after saving a checkpoint, spawns

      1. ``tools/q_probe.py``  -- matched-start return contrast + saved
         q_phi contexts on the just-saved checkpoint,
      2. ``tools/router_distill_from_qprobe.py`` -- offline KL distillation
         of ``strategy_encoder`` (q_phi) from those returns,

    then hot-swaps the distilled ``strategy_encoder.*`` weights back into
    the running model and clears the Adam moments for those params on both
    the main optimizer and the dedicated router optimizer.

    Pre-v4i4 story (recap):

    * v4i1: matched-start forced-z probes prove latent modes exist
      (large per-seed return contrasts across OP5/OP6/OP7).
    * v4i2 (offline): ``router_distill_from_qprobe.py`` proves a small
      offline distill round can teach q_phi to route into those modes
      from saved contexts.
    * v4i3 (Summer Proof Suite): proves / falsifies whether pure Summer
      end-to-end routing (no distill, no aux heads) is sufficient.

    v4i4 lifts the offline distill loop into training so q_phi keeps
    catching up to the actor as PPO drifts. The hook is **best-effort**:
    any subprocess or hot-swap failure is logged and training continues
    with the pre-distill weights, so v4i4 cannot deadlock or corrupt PPO
    state.

    Defaults aimed at the 2M-step 4v4 budget:

    * cadence:    250k steps  (8 distill rounds per 2M-step run)
    * probe:      8 seeds x 3 opponents x latent_k=4 = 96 episodes per round
    * distill:    100 epochs, lr 1e-4, temperature 1.0
    * device:     cpu  (so the GPU is not contended; the round runs while
                  the PPO process is paused after the periodic save)
    * artifacts:  ``<checkpoint_dir>/v4i4post_router_distill/step_<N>/``
    """
    cfg = apply_plan_faithful_latent_v4i1_strategic_pressure_qprobe(cfg)

    cfg.latent_router_distill_enabled = True
    cfg.latent_router_distill_every_n_steps = 250_000
    cfg.latent_router_distill_n_seeds = 8
    cfg.latent_router_distill_base_seed = 1000
    cfg.latent_router_distill_opponents = ("OP5", "OP6", "OP7")
    cfg.latent_router_distill_epochs = 100
    cfg.latent_router_distill_lr = 1e-4
    cfg.latent_router_distill_temperature = 1.0
    cfg.latent_router_distill_weight_decay = 0.0
    cfg.latent_router_distill_device = "cpu"
    cfg.latent_router_distill_artifacts_subdir = "v4i4post_router_distill"

    cfg.run_tag = "v4i4post_periodic_router_distill_OP5_OP6_OP7_2m_4v4"
    return cfg
