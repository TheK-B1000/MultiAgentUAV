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
    else:
        print("[PPO] Latent team strategy: disabled (vanilla local PPO baseline).")
    print(f"[PPO] Checkpoint dir: {cfg.checkpoint_dir}")
    _print_metrics_csv_banner(cfg)


def _print_latent_strategy_banner(cfg: PPOConfig) -> None:
    """Print the ``[PPO] Latent team strategy: enabled (...)`` line (and notes)."""
    interval = int(getattr(cfg, "latent_resample_every_n", 0) or 0)
    fixed = bool(getattr(cfg, "fixed_latent_strategy", False))
    interval_label = "fixed" if fixed else ("episode start" if interval <= 0 else f"every {interval} decision steps")
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
    if (not fixed) and interval <= 0 and float(getattr(cfg, "latent_lam_p", 0.0) or 0.0) > 0.0:
        print(
            "[PPO] NOTE: latent_lam_p is active only on sparse mid-episode resamples; "
            "with sample=episode start it has near-zero training effect."
        )
    if fixed:
        print("[PPO] Fixed-latent baseline: q_phi sampling/losses are bypassed; actor/critic receive one z ID.")


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
