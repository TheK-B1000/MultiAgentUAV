"""Stdout banner printing for the local PPO trainer.

Extracted from :func:`rl.train_ppo.train_ppo` so the launchpad script stops
mixing pure stdout decoration with control flow. These functions:

* Read ``PPOConfig`` and derived training state (curriculum, max agents).
* Print human-readable banner lines starting with ``[PPO]``.
* Never mutate the config or any other state.
* Never touch the filesystem.

If you need to add a new banner line, add it here, not in
:mod:`rl.train_ppo` -- otherwise the launchpad script grows back into a
spaghetti catapult.

Banner ordering (matches the order before extraction byte-for-byte):

* :func:`print_training_banner`
  - Agents / mode / run_tag
  - Algorithm backend / total timesteps
  - Input dims (q_phi / critic / actor)
  - Actor CNN feature dim / map set
  - Domain randomization (optional)
  - Training profile (curriculum / latent / no-latent)
  - Return normalization (optional)
  - Reward shaping decay (optional)
  - Opponent randomization (optional)
  - Curriculum gates (if curriculum)
  - Latent team strategy block
  - Checkpoint dir
  - Metrics / episode / strategy-experience CSV paths
  - E3 step telemetry CSV (optional)
  - CSV-already-exists warning (optional)

* :func:`print_episode_stats_banner`
  - Episode stats logging cadence and label.

Both functions are pure stdout. Side effects (lock acquisition, CSV rotation,
``run_config.json`` write, runtime clamps, CUDA fallback) stay in
:func:`rl.train_ppo.train_ppo` so callers can interleave them with banner
prints in the historical order.
"""

from __future__ import annotations

from typing import Optional

from rl.config.ppo_config import PPOConfig, TrainMode
from rl.curriculum import CurriculumState
from rl.global_state import GLOBAL_STATE_DIM
from rl.latent_marl import CONTEXT_STATE_DIM
from rl.training.run_artifacts import _metrics_csv_nonempty


def print_training_banner(
    cfg: PPOConfig,
    *,
    curriculum: Optional[CurriculumState],
    max_agents: int,
    team_size: str,
) -> None:
    """Print the main ``[PPO] ...`` startup banner for a training run.

    The banner is pure stdout: it never mutates ``cfg`` or touches the
    filesystem. Callers are responsible for resolving CSV paths into ``cfg``
    *before* this call so the printed paths match what the trainer will
    actually write to.
    """
    print(f"[PPO] Agents: {max_agents} per team ({team_size}) | mode={cfg.mode} | run_tag={cfg.run_tag!r}")
    print("[PPO] Algorithm backend: custom local PPO")
    print(f"[PPO] Total timesteps: {int(cfg.total_timesteps):,}")
    base_gs_dim = GLOBAL_STATE_DIM
    use_latent = bool(getattr(cfg, "use_latent_strategy", False))
    temp_ctx_dim = CONTEXT_STATE_DIM if use_latent else 0
    q_phi_dim = CONTEXT_STATE_DIM if use_latent else 0
    crit_dim = CONTEXT_STATE_DIM if use_latent else GLOBAL_STATE_DIM
    actor_cnn_feat = int(getattr(cfg, "actor_cnn_feature_dim", 128))
    z_embed = int(getattr(cfg, "latent_z_embed_dim", 16))
    act_dim = (actor_cnn_feat + 20 + z_embed) if use_latent else (actor_cnn_feat + 20)
    print(
        f"[PPO] Input dims: base_global_state_dim={base_gs_dim} "
        f"temporal_context_dim={temp_ctx_dim} "
        f"q_phi_input_dim={q_phi_dim} "
        f"critic_context_dim={crit_dim} "
        f"actor_input_dim={act_dim}"
    )
    print(f"[PPO] Actor CNN feature dim: {actor_cnn_feat}")
    print(f"[PPO] Map set: {str(getattr(cfg, 'map_set', 'train')).lower()}")
    if bool(getattr(cfg, "train_domain_randomization", False)):
        print(
            "[PPO] Domain randomization: ON "
            f"(sensor_noise_sigma max={float(getattr(cfg, 'dr_sensor_noise_sigma_max', 0.0)):.3f}, "
            f"sensor_dropout max={float(getattr(cfg, 'dr_sensor_dropout_max', 0.0)):.3f}, "
            f"blue_speed_jitter={float(getattr(cfg, 'dr_blue_speed_jitter', 0.0)):.3f}; "
            "blue-policy side only, slowdown-only speed scale)"
        )
    if curriculum is not None:
        print("[PPO] Training profile: curriculum baseline")
    elif use_latent:
        print("[PPO] Training profile: default latent (Summer implementation)")
    else:
        print("[PPO] Training profile: no-latent baseline")
    if bool(getattr(cfg, "normalize_returns", False)):
        print("[PPO] Return normalization: enabled for critic targets/predictions; GAE uses denormalized values.")
    decay_steps = max(0, int(getattr(cfg, "reward_shaping_decay_steps", 0) or 0))
    if decay_steps > 0:
        print(
            "[PPO] Reward shaping decay: "
            f"coef {float(getattr(cfg, 'reward_shaping_coef_start', 1.0)):.3f} -> "
            f"{float(getattr(cfg, 'reward_shaping_coef_end', 1.0)):.3f} "
            f"over {decay_steps:,} steps before RewardConfig weighting/scaling."
        )
    if cfg.mode == TrainMode.OPPONENT_POOL.value or bool(getattr(cfg, "opponent_randomize", False)):
        label = "OPPONENT_POOL mode" if cfg.mode == TrainMode.OPPONENT_POOL.value else "opponent_randomize flag"
        weights = tuple(getattr(cfg, "opponent_pool_weights", ()) or ())
        if weights and len(weights) == len(cfg.opponent_pool):
            weight_str = ", ".join(f"{tag}={w:.3f}" for tag, w in zip(cfg.opponent_pool, weights))
            sampler_desc = f"weighted per completed episode over pool={list(cfg.opponent_pool)} ({weight_str})"
        else:
            sampler_desc = f"uniform per completed episode over pool={list(cfg.opponent_pool)}"
        print(
            "[PPO] Opponent randomization: enabled "
            f"({label}; {sampler_desc}; "
            "pre-reset hook \u2014 opponent logged for each episode is the one played during that episode)."
        )
    if curriculum is not None:
        print(
            "[PPO] Jacob paper curriculum: enabled "
            "(SCRIPTED:OP1 -> SCRIPTED:OP2 -> SCRIPTED:OP3; scripted-only curriculum)."
        )
        print(
            "[PPO] Curriculum gates: "
            f"min_episodes={curriculum.config.min_episodes}, "
            f"min_winrate={curriculum.config.min_winrate}, "
            f"windows={curriculum.config.winrate_window_by_phase}."
        )
    if use_latent:
        _print_latent_strategy_banner(cfg)
        _maybe_print_paper_faithful_audit(cfg)
    else:
        print("[PPO] Latent team strategy: disabled (vanilla local PPO baseline).")
    print(f"[PPO] Checkpoint dir: {cfg.checkpoint_dir}")
    _print_metrics_csv_banner(cfg)


def _print_latent_strategy_banner(cfg: PPOConfig) -> None:
    """Print the ``[PPO] Latent team strategy: enabled (...)`` line (and notes)."""
    interval = int(getattr(cfg, "latent_resample_every_n", 0) or 0)
    fixed = bool(getattr(cfg, "fixed_latent_strategy", False))
    sparse_tactical_refresh = (
        bool(
            getattr(
                cfg,
                "latent_sparse_tactical_refresh_enabled",
                False,
            )
        )
        and not fixed
    )
    if fixed:
        interval_label = "fixed"
    elif sparse_tactical_refresh:
        sparse_interval = int(
            getattr(
                cfg,
                "latent_sparse_tactical_refresh_interval_steps",
                32,
            )
            or 32
        )
        sparse_dwell = int(
            getattr(
                cfg,
                "latent_sparse_tactical_refresh_min_dwell_steps",
                16,
            )
            or 16
        )
        interval_label = (
            "episode start + tactical transitions/"
            f"{sparse_interval}-step interval (min dwell {sparse_dwell})"
        )
    else:
        interval_label = (
            "episode start"
            if interval <= 0
            else f"every {interval} decision steps"
        )
    on_flag = bool(getattr(cfg, "latent_resample_on_flag", False)) and not fixed
    lam_kl = 0.0 if fixed else float(getattr(cfg, "latent_kl_consecutive", 0.0) or 0.0)
    fixed_label = f", fixed_z={int(getattr(cfg, 'fixed_latent_strategy_id', 0) or 0)}" if fixed else ""
    h_obj = getattr(cfg, "latent_entropy_objective", "maximize") or "maximize"
    aux_head = bool(getattr(cfg, "latent_strategy_aux_return_head", False))
    episode_credit = bool(getattr(cfg, "latent_episode_strategy_ppo", False))
    ep_warmup = int(getattr(cfg, "latent_episode_strategy_warmup_decision_steps", 0) or 0)
    warmup_label = (
        f", episode_warmup_steps={ep_warmup}" if episode_credit and ep_warmup > 0 else ""
    )
    print(
        "[PPO] Latent team strategy: enabled "
        f"(K={int(cfg.latent_k)}, sample={interval_label}, on_flag={on_flag}, "
        f"lambda_p={float(cfg.latent_lam_p):.4f}, lambda_H={float(cfg.latent_lam_h):.4f} "
        f"(H:{h_obj}), "
        f"lambda_KL={lam_kl:.4f}, strategy_ppo_coef={float(cfg.latent_strategy_ppo_coef):.3f}, "
        f"episode_credit={episode_credit}, episode_coef={float(getattr(cfg, 'latent_episode_strategy_coef', 0.0)):.3f}"
        f"{warmup_label}, "
        f"aux_return_head={aux_head}, aux_return_coef={float(cfg.latent_strategy_aux_return_coef):.3f}, "
        f"tau={float(cfg.latent_strategy_tau):.3f}, "
        f"GAE_reset_on_z_change={bool(getattr(cfg, 'latent_gae_reset_on_z_change', True))}, "
        f"bootstrap_z_deterministic={bool(getattr(cfg, 'latent_bootstrap_z_deterministic', True))}"
        f"{fixed_label})"
    )
    if episode_credit and ep_warmup > 0:
        print(
            f"[PPO] Episode-credit warmup: provisional z drives steps 0..{ep_warmup - 1}; "
            f"committed z + ctx170 snapshot taken at decision step {ep_warmup} (after EMAs see opponent dynamics)."
        )
    if episode_credit and ep_warmup == 0:
        print(
            "[PPO] WARNING: episode_credit on with warmup_decision_steps=0; q_phi snapshot uses step-0 context "
            "(canonical initial geometry + zeroed EMAs => opponent-blind). MI(z; opponent) structurally bounded near zero."
        )
    if (
        (not fixed)
        and interval <= 0
        and not sparse_tactical_refresh
        and float(getattr(cfg, "latent_lam_p", 0.0) or 0.0) > 0.0
    ):
        print(
            "[PPO] NOTE: latent_lam_p is active only on sparse mid-episode resamples; "
            "with sample=episode start it has near-zero training effect."
        )
    if fixed:
        print("[PPO] Fixed-latent baseline: q_phi sampling/losses are bypassed; actor/critic receive one z ID.")
    if bool(getattr(cfg, "latent_arc_credit_enabled", False)) and not fixed:
        arc_coef = float(getattr(cfg, "latent_arc_credit_coef", 1.0) or 0.0)
        arc_baseline = str(
            getattr(cfg, "latent_arc_credit_baseline", "context_value") or "context_value"
        )
        arc_min_len = int(getattr(cfg, "latent_arc_credit_min_len", 32) or 1)
        arc_epochs = int(getattr(cfg, "latent_arc_credit_n_epochs", 4) or 1)
        arc_clip = float(getattr(cfg, "latent_arc_credit_clip_eps", 0.2) or 0.2)
        arc_norm = bool(getattr(cfg, "latent_arc_credit_return_norm", True))
        print(
            "[PPO] q_phi arc-credit: enabled "
            f"(coef={arc_coef:.3f}, baseline={arc_baseline}, min_len={arc_min_len}, "
            f"n_epochs={arc_epochs}, clip_eps={arc_clip:.3f}, return_norm={arc_norm})"
        )
        h_start = float(getattr(cfg, "latent_lam_h_start", float(cfg.latent_lam_h)) or 0.0)
        h_end = float(
            getattr(cfg, "latent_lam_h_end", getattr(cfg, "latent_lam_h_final", float(cfg.latent_lam_h)))
            or 0.0
        )
        if h_start != h_end:
            anneal_end = int(
                getattr(cfg, "latent_entropy_anneal_end", getattr(cfg, "latent_entropy_decay_steps", 0))
                or 0
            )
            print(
                f"[PPO] Entropy schedule: lambda_H {h_start:.4f} -> {h_end:.4f} over {anneal_end:,} steps "
                "(collapse guard only; not the credit signal)"
            )


def _maybe_print_paper_faithful_audit(cfg: PPOConfig) -> None:
    """Emit the paper-faithful audit banner when the run is configured per
    a paper-faithful contract (currently the v5i4 / v5i5 family).

    The audit is triggered by either:

    * ``cfg.run_tag`` containing one of the recognized paper-faithful
      family tags (``v5i4_paper_faithful``, ``v5i5_paper_faithful``), or
    * an explicit ``cfg.latent_paper_faithful_audit = True`` opt-in flag
      (used by future paper-faithful presets so they inherit the banner
      without relying on a specific run_tag string).

    The banner lists every invariant the paper-faithful design depends on
    so a reviewer can verify them at the top of the log without diffing
    config snapshots. None of these reads mutate ``cfg``.

    Family detection: the header / warning lines are prefixed with the
    matched family (``v5i4`` or ``v5i5``) so existing v5i4 banner tests
    still see ``"v5i4 paper-faithful audit"`` / ``"v5i4 audit WARNING"``
    while v5i5 runs print the equivalent ``v5i5`` strings. The
    invariants themselves are identical -- v5i5 differs from v5i4 only
    in ``latent_lam_h_end`` (0.0002 -> 0.001), which is a hyperparameter
    inside the documented Summer-plan entropy range, not a fidelity
    flip.
    """
    run_tag = str(getattr(cfg, "run_tag", "") or "")
    explicit_opt_in = bool(getattr(cfg, "latent_paper_faithful_audit", False))
    run_tag_low = run_tag.lower()
    family: str | None = None
    if "v5i5_paper_faithful" in run_tag_low:
        family = "v5i5"
    elif "v5i4_paper_faithful" in run_tag_low:
        family = "v5i4"
    elif explicit_opt_in:
        # Opt-in flag without a recognized run_tag: default to the v5i4
        # label so reviewers see a stable header.
        family = "v5i4"
    if family is None:
        return

    strategy_ppo_coef = float(getattr(cfg, "latent_strategy_ppo_coef", 0.0) or 0.0)
    episode_credit_on = bool(getattr(cfg, "latent_episode_strategy_ppo", False))
    film_on = bool(getattr(cfg, "enable_actor_z_film", False))
    adapter_on = bool(getattr(cfg, "latent_actor_z_adapter_enabled", False))
    onehot_on = bool(getattr(cfg, "latent_actor_z_onehot_enabled", False))
    forced_z_legacy = float(getattr(cfg, "latent_forced_z_episode_frac", 0.0) or 0.0)
    forced_z_start = getattr(cfg, "latent_forced_z_episode_frac_start", None)
    forced_z_curriculum_on = forced_z_legacy > 0.0 or forced_z_start is not None
    aux_return_on = bool(getattr(cfg, "latent_strategy_aux_return_head", False))
    aux_phase_coef = float(
        getattr(cfg, "latent_strategy_aux_predict_phase_coef", 0.0) or 0.0
    )
    aux_heads_on = aux_return_on or aux_phase_coef > 0.0
    pref_on = bool(getattr(cfg, "latent_v3i3_event_preference_enabled", False)) or (
        float(getattr(cfg, "latent_preference_coef", 0.0) or 0.0) > 0.0
    )
    distill_on = bool(getattr(cfg, "latent_router_distill_enabled", False))
    arc_credit_on = bool(getattr(cfg, "latent_arc_credit_enabled", False))
    persistence_on = float(getattr(cfg, "latent_lam_p", 0.0) or 0.0) > 0.0
    entropy_obj = str(getattr(cfg, "latent_entropy_objective", "maximize") or "maximize")
    entropy_max_on = (
        entropy_obj == "maximize"
        and float(getattr(cfg, "latent_lam_h", 0.0) or 0.0) > 0.0
    )
    resample_n = int(getattr(cfg, "latent_resample_every_n", 0) or 0)
    resample_on_flag = bool(getattr(cfg, "latent_resample_on_flag", False))
    k = int(getattr(cfg, "latent_k", 0) or 0)

    def _yn(flag: bool) -> str:
        return "ON" if flag else "OFF"

    print(f"[PPO] {family} paper-faithful audit:")
    print(f"  discrete shared z: K={k}")
    actor_label = "embedding-concat"
    if film_on:
        actor_label += " + FiLM"
    if adapter_on:
        actor_label += " + adapter"
    if onehot_on:
        actor_label += " + one-hot"
    print(f"  actor conditioning: {actor_label}")
    print(f"  FiLM: {_yn(film_on)}")
    print(
        "  q_phi task-reward PPO: "
        f"{_yn(strategy_ppo_coef > 0.0)} (latent_strategy_ppo_coef={strategy_ppo_coef:.3f})"
    )
    print(f"  episode-credit extension: {_yn(episode_credit_on)}")
    print(
        "  forced-z curriculum: "
        f"{_yn(forced_z_curriculum_on)} (legacy_frac={forced_z_legacy:.3f})"
    )
    print("  supervised targets: OFF")
    print(f"  auxiliary heads: {_yn(aux_heads_on)}")
    print(f"  arc-credit: {_yn(arc_credit_on)}")
    print(f"  preferences/distillation: {_yn(pref_on or distill_on)}")
    print(f"  persistence: {_yn(persistence_on)}")
    print(
        "  entropy maximization: "
        f"{_yn(entropy_max_on)} (objective={entropy_obj})"
    )
    cadence_label = (
        "episode start" if resample_n <= 0 else f"every {resample_n} decisions"
    )
    if resample_on_flag:
        cadence_label += " + on-flag"
    print(f"  resampling cadence: {cadence_label}")

    # Diagnostics: surface mis-configurations that would silently void the
    # paper-faithful claim, so reviewers see them at the top of the log.
    if strategy_ppo_coef <= 0.0:
        print(
            f"[PPO] {family} audit WARNING: latent_strategy_ppo_coef <= 0; "
            "q_phi receives no task-reward gradient. This contradicts the "
            "'learned end-to-end from task reward' claim."
        )
    if (
        bool(getattr(cfg, "use_latent_strategy", False))
        and not bool(getattr(cfg, "fixed_latent_strategy", False))
        and getattr(cfg, "latent_episode_strategy_lr", None) is not None
    ):
        print(
            f"[PPO] {family} audit WARNING: latent_episode_strategy_lr is set; "
            "the dedicated router optimizer suppresses the main-loop "
            f"categorical PPO term on q_phi. {family} requires this to be None."
        )
    if film_on or adapter_on or onehot_on:
        print(
            f"[PPO] {family} audit WARNING: actor-z pathway is not concat-only; "
            f"{family} specifies plain nn.Embedding(K, d_z) concat. FiLM/adapter/"
            "one-hot must all be OFF."
        )


def _print_metrics_csv_banner(cfg: PPOConfig) -> None:
    """Print the CSV-paths block (or the ``Metrics CSV logging disabled.`` line).

    Assumes ``_resolve_metrics_csv_paths`` has already populated
    ``cfg.metrics_csv_path`` / ``cfg.episode_csv_path`` /
    ``cfg.strategy_experience_csv_path``.
    """
    if not bool(getattr(cfg, "enable_metrics_csv", True)):
        print("[PPO] Metrics CSV logging disabled.")
        return
    strategy_experience_enabled = bool(getattr(cfg, "use_latent_strategy", False)) and bool(
        getattr(cfg, "latent_episode_strategy_ppo", False)
    )
    print(f"[PPO] Update metrics CSV: {cfg.metrics_csv_path}")
    print(f"[PPO] Episode metrics CSV: {cfg.episode_csv_path}")
    if strategy_experience_enabled and cfg.strategy_experience_csv_path:
        print(f"[PPO] Strategy experience CSV: {cfg.strategy_experience_csv_path}")
    _e3p = str(getattr(cfg, "e3_step_telemetry_path", "") or "").strip()
    if _e3p:
        print(
            "[PPO] E3 step telemetry CSV "
            f"(per-step z, team_phase, behavior telemetry, buckets, MI-related fields): {_e3p}"
        )
    if (not cfg.fresh_metrics_csv) and (not cfg.load_path) and (
        _metrics_csv_nonempty(cfg.metrics_csv_path)
        or _metrics_csv_nonempty(cfg.episode_csv_path)
        or _metrics_csv_nonempty(cfg.strategy_experience_csv_path)
    ):
        print(
            "[PPO] WARNING: metrics/episode/strategy-experience CSV already exists; this run will APPEND. "
            "That duplicates `timestep`/update indices if you reused --run-tag. "
            "Use --fresh-metrics-csv (rotates old files aside) or a new --run-tag."
        )


def print_episode_stats_banner(
    cfg: PPOConfig,
    *,
    curriculum: Optional[CurriculumState],
    initial_opponent_tag: str,
) -> None:
    """Print the ``[PPO] Episode stats: ...`` cadence banner.

    Kept separate from :func:`print_training_banner` so the call site can
    interleave the run-lock + ``run_config.json`` writes between them in the
    historical print order.
    """
    elog = int(getattr(cfg, "episode_log_every", 0) or 0)
    if elog <= 0:
        print("[PPO] Episode stats logging disabled (episode_log_every=0).")
        return
    mode_label = "curriculum phase" if curriculum is not None else "scripted opponent tag"
    if curriculum is not None:
        tag_label: str = initial_opponent_tag
    elif cfg.mode == TrainMode.OPPONENT_POOL.value or bool(getattr(cfg, "opponent_randomize", False)):
        tag_label = f"randomized pool {list(cfg.opponent_pool)}"
    else:
        tag_label = str(cfg.fixed_opponent_tag).upper()
    print(
        f"[PPO] Episode stats: every {elog} completed episode(s) print W/L/D and WR "
        f"(mode={cfg.mode}, {mode_label}={tag_label})."
    )


__all__ = [
    "print_training_banner",
    "print_episode_stats_banner",
]
