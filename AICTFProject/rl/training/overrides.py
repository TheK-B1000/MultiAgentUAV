"""CLI namespace → PPOConfig override application.

:func:`cfg_from_args` is the canonical implementation extracted from
:mod:`rl.training.cli`.  It also owns several helper functions that
previously lived in :mod:`rl.train_ppo` and were imported by ``cli.py``
at function-call time:

* :func:`_agents_suffix` — ``"2v2"`` / ``"4v4"`` / … tag string
* :func:`_default_run_tag_for_mode` — synthesises the default ``run_tag``
  from training mode, opponent, and agent-count
* :func:`_ensure_run_tag_has_agent_suffix` — appends / replaces the ``_NvN``
  suffix on a run tag

These are re-exported from :mod:`rl.train_ppo` for backward compatibility
with existing import paths (presets, tools, archived scripts).

Dependency direction: module-level imports are limited to stdlib,
``rl.config.ppo_config``, and ``rl.training.config_validation``.  The
``_apply_training_preset`` call is a *lazy* import so this module can be
imported without pulling in ``rl.presets`` and its heavy dependencies.
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

from rl.config.ppo_config import PPOConfig, TrainMode
from rl.training.config_validation import _normalize_train_mode


# ---------------------------------------------------------------------------
# Helper functions moved from train_ppo.py
# ---------------------------------------------------------------------------

def _agents_suffix(n_agents: int) -> str:
    n = max(1, min(int(n_agents), 16))
    return f"{n}v{n}"


def _default_run_tag_for_mode(
    mode: str,
    fixed_opponent_tag: str = "OP3",
    n_agents: int = 2,
    *,
    latent: bool = True,
) -> str:
    suffix = _agents_suffix(n_agents)
    family = "ppo_latent" if latent else "ppo_custom"
    if _normalize_train_mode(mode) == TrainMode.CURRICULUM.value:
        return f"{family}_curriculum_{suffix}"
    if _normalize_train_mode(mode) == TrainMode.OPPONENT_POOL.value:
        return f"{family}_opp_pool_{suffix}"
    if _normalize_train_mode(mode) == TrainMode.FIXED_OPPONENT.value:
        return f"{family}_fixed_{fixed_opponent_tag.lower()}_{suffix}"
    return f"{family}_{suffix}"


def _ensure_run_tag_has_agent_suffix(run_tag: str, n_agents: int) -> str:
    suffix = _agents_suffix(n_agents)
    tag_suffix = f"_{suffix}"
    for existing in ("_2v2", "_4v4", "_6v6", "_8v8"):
        if run_tag.endswith(existing):
            run_tag = run_tag[: -len(existing)]
            break
    if not run_tag.endswith(tag_suffix):
        run_tag = run_tag.rstrip("_") + tag_suffix
    return run_tag


def _apply_training_preset(cfg: PPOConfig, preset: str) -> PPOConfig:
    """Apply named high-level presets for repeatable training recipes."""
    from rl.presets import apply_preset

    return apply_preset(cfg, preset)


# ---------------------------------------------------------------------------
# Core override application
# ---------------------------------------------------------------------------

def cfg_from_args(args: argparse.Namespace) -> PPOConfig:
    """Apply a parsed argparse namespace onto a fresh ``PPOConfig``.

    Mirrors the ordering of the original inline block in :mod:`rl.training.cli`:
    fresh config → optional preset → per-flag overrides → run_tag /
    checkpoint_dir synthesis → ``cli_preset`` bookkeeping. No env / trainer
    construction happens here; pure data prep.
    """
    preset_key = str(args.preset or "").strip()
    if preset_key.lower() in {"", "none"}:
        preset_key = ""

    cfg = PPOConfig()
    if preset_key:
        cfg = _apply_training_preset(cfg, preset_key)
    if args.mode is not None:
        cfg.mode = _normalize_train_mode(args.mode)
    if args.seed is not None:
        cfg.seed = int(args.seed)
    if args.max_blue_agents is not None:
        cfg.max_blue_agents = max(1, min(int(args.max_blue_agents), 16))
    elif args.agents is not None:
        cfg.max_blue_agents = int(args.agents)
    cfg.fixed_opponent_tag = str(args.fixed_opponent).upper()
    if args.opponent_randomize:
        cfg.opponent_randomize = True
    if getattr(args, "opponent_pool", None):
        cfg.opponent_pool = tuple(str(x).strip().upper() for x in args.opponent_pool if str(x).strip())
    # Strategic-pressure pool guard. v4i1 (and everything that inherits its
    # opponent-pool contract: v4i3 Summer-proof, v4i3 no-latent baseline,
    # v4i4post periodic distill) requires opponent_pool == {OP5, OP6, OP7}
    # exactly. A stray ``--opponent-pool`` on the CLI would silently
    # override the preset and invalidate the strategic-pressure ablation.
    _preset_key_lower = preset_key.lower()
    _requires_v4i_pressure_pool = (
        "v4i1" in _preset_key_lower
        or "v4i3" in _preset_key_lower
        or "v4i4post" in _preset_key_lower
    )
    if _requires_v4i_pressure_pool:
        required_pool = frozenset({"OP5", "OP6", "OP7"})
        actual_pool = frozenset(str(x).upper() for x in (cfg.opponent_pool or ()))
        if actual_pool != required_pool:
            raise ValueError(
                f"Preset {preset_key!r} requires opponent_pool == {{OP5, OP6, OP7}} "
                f"exactly (got {sorted(actual_pool)!r}). Remove any --opponent-pool "
                "override from the command line and let the preset own the pool, "
                "or pass --opponent-pool OP5 OP6 OP7 explicitly. (The v4i family's "
                "thesis is that the strategic-pressure pool is the experimental "
                "treatment; mutating it breaks the v4i3 Summer-proof ablation and "
                "the v4i1 return-contrast story.)"
            )
        if not cfg.opponent_randomize:
            raise ValueError(
                f"Preset {preset_key!r} requires --opponent-randomize (preset sets "
                "this by default). Do not disable it on the command line."
            )
    if getattr(args, "opponent_pool_weights", None):
        wmap: dict[str, float] = {}
        for entry in args.opponent_pool_weights:
            text = str(entry).strip()
            if not text:
                continue
            if "=" not in text:
                raise ValueError(
                    f"--opponent-pool-weights entries must be 'TAG=PROB'; got {entry!r}."
                )
            tag, _, val = text.partition("=")
            tag = tag.strip().upper()
            try:
                wmap[tag] = float(val.strip())
            except ValueError as exc:
                raise ValueError(
                    f"--opponent-pool-weights value for {tag!r} is not numeric: {val!r}."
                ) from exc
        pool = tuple(getattr(cfg, "opponent_pool", ()) or ())
        if not pool:
            raise ValueError(
                "--opponent-pool-weights requires --opponent-pool (or a preset that sets one)."
            )
        missing = [tag for tag in pool if tag not in wmap]
        if missing:
            raise ValueError(
                f"--opponent-pool-weights missing entries for pool tag(s) {missing!r}. "
                f"Pool: {list(pool)}; weights given: {sorted(wmap.keys())}."
            )
        cfg.opponent_pool_weights = tuple(wmap[tag] for tag in pool)
    if getattr(args, "allow_op4_in_training_pool", False):
        cfg.allow_op4_in_training_pool = True
    if args.map_set is not None:
        cfg.map_set = str(args.map_set).lower()
    if args.map_layout is not None:
        cfg.map_layout = str(args.map_layout).lower()
    if args.latent_strategy:
        cfg.use_latent_strategy = True
    if args.no_latent_strategy:
        cfg.use_latent_strategy = False
    if args.latent_k is not None:
        cfg.latent_k = max(1, int(args.latent_k))
    if args.latent_resample_every is not None:
        cfg.latent_resample_every_n = max(0, int(args.latent_resample_every))
    if args.fixed_latent_strategy:
        cfg.fixed_latent_strategy = True
    if args.fixed_latent_id is not None:
        cfg.fixed_latent_strategy = True
        cfg.fixed_latent_strategy_id = max(0, int(args.fixed_latent_id))
    if args.latent_lam_p is not None:
        cfg.latent_lam_p = max(0.0, float(args.latent_lam_p))
    if args.latent_lam_h is not None:
        cfg.latent_lam_h = max(0.0, float(args.latent_lam_h))
    if args.latent_cf_coef_max is not None:
        cfg.latent_cf_coef_max = max(0.0, float(args.latent_cf_coef_max))
    if args.no_latent_cf_require_competence:
        cfg.latent_cf_require_competence = False
    if args.actor_cf_update_mode is not None:
        cfg.actor_cf_update_mode = str(args.actor_cf_update_mode)
    if args.latent_cf_sequential_update:
        cfg.latent_cf_sequential_update = True
        cfg.actor_cf_update_mode = "ppo_then_cf"
    if args.latent_strategy_ppo_coef is not None:
        cfg.latent_strategy_ppo_coef = max(0.0, float(args.latent_strategy_ppo_coef))
    if args.latent_episode_strategy_ppo:
        cfg.latent_episode_strategy_ppo = True
    if args.latent_episode_strategy_coef is not None:
        cfg.latent_episode_strategy_coef = max(0.0, float(args.latent_episode_strategy_coef))
    if args.latent_episode_strategy_clip_eps is not None:
        cfg.latent_episode_strategy_clip_eps = max(1e-6, float(args.latent_episode_strategy_clip_eps))
    if args.latent_episode_strategy_value_coef is not None:
        cfg.latent_episode_strategy_value_coef = max(0.0, float(args.latent_episode_strategy_value_coef))
    if args.no_latent_episode_strategy_return_norm:
        cfg.latent_episode_strategy_return_norm = False
    legacy_q_head_used = bool(getattr(args, "latent_strategy_q_head", False))
    legacy_q_coef_used = getattr(args, "latent_strategy_q_coef", None) is not None
    if legacy_q_head_used or legacy_q_coef_used:
        legacy_flags = []
        if legacy_q_head_used:
            legacy_flags.append("--latent-strategy-q-head -> --latent-strategy-aux-return-head")
        if legacy_q_coef_used:
            legacy_flags.append("--latent-strategy-q-coef -> --latent-strategy-aux-return-coef")
        print(
            "[PPO] DEPRECATED CLI flag(s): "
            + "; ".join(legacy_flags)
            + ". The canonical name is the only one written to run_config.json; legacy "
            "flags will be removed in a future cleanup."
        )
    if args.latent_strategy_aux_return_head or legacy_q_head_used:
        cfg.latent_strategy_aux_return_head = True
    aux_coef = getattr(args, "latent_strategy_aux_return_coef", None)
    if aux_coef is None and legacy_q_coef_used:
        aux_coef = getattr(args, "latent_strategy_q_coef", None)
    if aux_coef is not None:
        cfg.latent_strategy_aux_return_coef = max(0.0, float(aux_coef))
    if args.latent_strategy_tau is not None:
        cfg.latent_strategy_tau = max(1e-3, float(args.latent_strategy_tau))
    if getattr(args, "latent_strategy_aux_predict_phase_coef", None) is not None:
        cfg.latent_strategy_aux_predict_phase_coef = max(0.0, float(args.latent_strategy_aux_predict_phase_coef))
    if args.latent_entropy_objective is not None:
        cfg.latent_entropy_objective = args.latent_entropy_objective  # type: ignore[assignment]
    if args.latent_resample_on_flag:
        cfg.latent_resample_on_flag = True
    if args.latent_kl_consecutive is not None:
        cfg.latent_kl_consecutive = max(0.0, float(args.latent_kl_consecutive))
    if args.latent_v3i3_event_preference_normalize:
        cfg.latent_v3i3_event_preference_normalize = True
    if args.no_latent_gae_z_reset:
        cfg.latent_gae_reset_on_z_change = False
    if args.latent_bootstrap_z_stochastic:
        cfg.latent_bootstrap_z_deterministic = False
    if args.domain_randomization:
        cfg.train_domain_randomization = True
    if args.dr_sensor_noise_max is not None:
        cfg.dr_sensor_noise_sigma_max = max(0.0, float(args.dr_sensor_noise_max))
    if args.dr_sensor_dropout_max is not None:
        cfg.dr_sensor_dropout_max = max(0.0, min(1.0, float(args.dr_sensor_dropout_max)))
    if args.dr_blue_speed_jitter is not None:
        cfg.dr_blue_speed_jitter = max(0.0, min(0.75, float(args.dr_blue_speed_jitter)))
    if args.latent_z_embed_dim is not None:
        cfg.latent_z_embed_dim = max(1, int(args.latent_z_embed_dim))
    if args.latent_vf_hidden is not None:
        cfg.latent_vf_hidden = max(1, int(args.latent_vf_hidden))

    if getattr(args, "training_telemetry_mode", None) is not None:
        cfg.training_telemetry_mode = args.training_telemetry_mode
    if getattr(args, "training_events_jsonl_path", None) is not None:
        cfg.training_events_jsonl_path = args.training_events_jsonl_path
    if getattr(args, "telemetry_events_jsonl_path", None) is not None:
        cfg.telemetry_events_jsonl_path = args.telemetry_events_jsonl_path
    if getattr(args, "performance_summary_path", None) is not None:
        cfg.performance_summary_path = args.performance_summary_path
    if getattr(args, "performance_samples_path", None) is not None:
        cfg.performance_samples_path = args.performance_samples_path
    if getattr(args, "gpu_monitor_enabled", None) is not None:
        cfg.gpu_monitor_enabled = args.gpu_monitor_enabled
    if getattr(args, "gpu_monitor_interval_seconds", None) is not None:
        cfg.gpu_monitor_interval_seconds = args.gpu_monitor_interval_seconds

    # Presets set ``cfg.run_tag``; only overwrite when user supplies --run-tag or no preset was applied.
    if args.run_tag is not None:
        cfg.run_tag = args.run_tag
    elif not preset_key:
        cfg.run_tag = _default_run_tag_for_mode(
            cfg.mode,
            cfg.fixed_opponent_tag,
            cfg.max_blue_agents,
            latent=bool(cfg.use_latent_strategy),
        )
    cfg.run_tag = _ensure_run_tag_has_agent_suffix(cfg.run_tag, cfg.max_blue_agents)
    cfg.checkpoint_dir = args.checkpoint_dir or os.path.join("checkpoints", _agents_suffix(cfg.max_blue_agents))
    if getattr(args, "e3_step_telemetry", False):
        if not cfg.use_latent_strategy:
            print("[PPO] WARNING: --e3-step-telemetry ignored (requires latent strategy).")
        else:
            os.makedirs(cfg.checkpoint_dir, exist_ok=True)
            cfg.e3_step_telemetry_path = os.path.join(cfg.checkpoint_dir, f"{cfg.run_tag}_e3_steps.csv")
    if args.fresh_metrics_csv:
        cfg.fresh_metrics_csv = True
    if args.no_metrics_csv:
        cfg.enable_metrics_csv = False
    if args.metrics_csv is not None:
        cfg.metrics_csv_path = args.metrics_csv
    if args.episode_csv is not None:
        cfg.episode_csv_path = args.episode_csv
    if args.strategy_experience_csv is not None:
        cfg.strategy_experience_csv_path = args.strategy_experience_csv
    if args.total_steps is not None:
        cfg.total_timesteps = int(args.total_steps)
    if getattr(args, "additional_steps", None) is not None:
        cfg.additional_timesteps = int(args.additional_steps)
    if args.load is not None:
        cfg.load_path = args.load
    elif args.resume is not None:
        cfg.load_path = args.resume
    if getattr(args, "load_weights_only", False):
        cfg.load_weights_only = True
    if getattr(args, "router_reinitialize_on_load", None) is not None:
        cfg.router_reinitialize_on_load = str(args.router_reinitialize_on_load).lower() == "true"
    if getattr(args, "allow_active_actor_module_migration", False):
        cfg.allow_active_actor_module_migration = True
    if args.learning_rate is not None:
        cfg.learning_rate = max(0.0, float(args.learning_rate))
    if args.lr_floor_frac is not None:
        cfg.lr_floor_frac = max(0.0, min(float(args.lr_floor_frac), 1.0))
    if args.target_kl is not None:
        cfg.target_kl = None if float(args.target_kl) < 0.0 else max(0.0, float(args.target_kl))
    if args.n_epochs is not None:
        cfg.n_epochs = max(1, int(args.n_epochs))
    if args.n_envs is not None:
        cfg.n_envs = max(1, int(args.n_envs))
    if args.n_steps is not None:
        cfg.n_steps = max(1, int(args.n_steps))
    if args.clip_range_vf is not None:
        cfg.clip_range_vf = None if float(args.clip_range_vf) < 0.0 else max(0.0, float(args.clip_range_vf))
    if args.vf_coef is not None:
        cfg.vf_coef = max(0.0, float(args.vf_coef))
    if args.return_normalization:
        cfg.normalize_returns = True
    if args.device is not None:
        cfg.device = str(args.device).strip().lower()
    if args.deterministic:
        cfg.use_deterministic = True
    if args.verbose_training:
        cfg.verbose_training = True
    if args.stable_marl:
        cfg.use_stable_marl_ppo = True
    if args.episode_log_every is not None:
        cfg.episode_log_every = max(0, int(args.episode_log_every))
    if args.env_win_reward is not None:
        cfg.env_win_team_reward = float(args.env_win_reward)
    if args.env_draw_penalty is not None:
        cfg.env_draw_team_penalty = float(args.env_draw_penalty)
    if args.env_lose_penalty is not None:
        cfg.env_lose_team_punish = float(args.env_lose_penalty)
    if args.env_action_failed_penalty is not None:
        cfg.env_action_failed_punishment = float(args.env_action_failed_penalty)
    if args.env_dense_weight is not None:
        cfg.env_dense_weight = max(0.0, float(args.env_dense_weight))
    if args.env_sparse_weight is not None:
        cfg.env_sparse_weight = max(0.0, float(args.env_sparse_weight))
    if args.env_reward_scale is not None:
        cfg.env_reward_scale = max(1e-6, float(args.env_reward_scale))
    if args.env_reward_clip is not None:
        cfg.env_reward_clip = max(1e-6, float(args.env_reward_clip))
    if args.env_stalemate_penalty is not None:
        cfg.env_stalemate_penalty = float(args.env_stalemate_penalty)
    if args.env_stalemate_max_steps is not None:
        cfg.env_stalemate_max_steps = max(1, int(args.env_stalemate_max_steps))
    if args.reward_shaping_coef_start is not None:
        cfg.reward_shaping_coef_start = float(args.reward_shaping_coef_start)
    if args.reward_shaping_coef_end is not None:
        cfg.reward_shaping_coef_end = float(args.reward_shaping_coef_end)
    if args.reward_shaping_decay_steps is not None:
        cfg.reward_shaping_decay_steps = max(0, int(args.reward_shaping_decay_steps))
    if args.periodic_checkpoint_steps is not None:
        cfg.periodic_checkpoint_steps = max(0, int(args.periodic_checkpoint_steps))
    if getattr(args, "max_decision_steps", None) is not None:
        cfg.max_decision_steps = max(1, int(args.max_decision_steps))
    if getattr(args, "phase_a_disable_promotion", False):
        cfg.phase_a_disable_promotion = True
    if getattr(args, "csia_enabled", False):
        cfg.csia_enabled = True
    if getattr(args, "csia_reward_coef", None) is not None:
        cfg.csia_reward_coef = max(0.0, float(args.csia_reward_coef))
    if getattr(args, "csia_payoff_csv", None):
        cfg.csia_payoff_csv_path = str(args.csia_payoff_csv)
    if getattr(args, "csia_strategy_evidence_csv", None):
        cfg.csia_strategy_evidence_csv_path = str(args.csia_strategy_evidence_csv)
    if getattr(args, "csia_probe_interval", None) is not None:
        cfg.csia_probe_interval = max(0, int(args.csia_probe_interval))
    if getattr(args, "csia_min_behavior_spread", None) is not None:
        cfg.csia_min_behavior_spread = max(0.0, float(args.csia_min_behavior_spread))
    if getattr(args, "csia_min_interaction_strength", None) is not None:
        cfg.csia_min_interaction_strength = max(0.0, float(args.csia_min_interaction_strength))
    if getattr(args, "csia_quality_floor_delta", None) is not None:
        cfg.csia_quality_floor_delta = max(0.0, float(args.csia_quality_floor_delta))
    if getattr(args, "csia_min_count_per_cell", None) is not None:
        cfg.csia_min_count_per_cell = max(1, int(args.csia_min_count_per_cell))
    if getattr(args, "no_csia_require_gates", False):
        cfg.csia_require_gates = False
    if getattr(args, "v6i6_anchor_validation_manifest", None):
        cfg.v6i6_anchor_validation_manifest = str(args.v6i6_anchor_validation_manifest)
    # --- v4i4post router-distill overrides ------------------------------
    if getattr(args, "latent_router_distill_enabled", None):
        cfg.latent_router_distill_enabled = True
    if getattr(args, "latent_router_distill_every_n_steps", None) is not None:
        cfg.latent_router_distill_every_n_steps = max(
            1, int(args.latent_router_distill_every_n_steps)
        )
    if getattr(args, "latent_router_distill_n_seeds", None) is not None:
        cfg.latent_router_distill_n_seeds = max(
            1, int(args.latent_router_distill_n_seeds)
        )
    if getattr(args, "latent_router_distill_base_seed", None) is not None:
        cfg.latent_router_distill_base_seed = int(args.latent_router_distill_base_seed)
    if getattr(args, "latent_router_distill_opponents", None):
        cfg.latent_router_distill_opponents = tuple(
            str(o).strip().upper()
            for o in args.latent_router_distill_opponents
            if str(o).strip()
        )
    if getattr(args, "latent_router_distill_epochs", None) is not None:
        cfg.latent_router_distill_epochs = max(
            1, int(args.latent_router_distill_epochs)
        )
    if getattr(args, "latent_router_distill_lr", None) is not None:
        cfg.latent_router_distill_lr = float(args.latent_router_distill_lr)
    if getattr(args, "latent_router_distill_temperature", None) is not None:
        cfg.latent_router_distill_temperature = float(
            args.latent_router_distill_temperature
        )
    if getattr(args, "latent_router_distill_weight_decay", None) is not None:
        cfg.latent_router_distill_weight_decay = float(
            args.latent_router_distill_weight_decay
        )
    if getattr(args, "latent_router_distill_device", None):
        cfg.latent_router_distill_device = str(
            args.latent_router_distill_device
        ).strip() or "cpu"
    if getattr(args, "latent_router_distill_artifacts_subdir", None):
        cfg.latent_router_distill_artifacts_subdir = str(
            args.latent_router_distill_artifacts_subdir
        ).strip() or "v4i4post_router_distill"
    if args.no_progress_bar:
        cfg.enable_progress_bar = False
    if preset_key:
        cfg.cli_preset = preset_key
        print(f"[PPO] Training preset: {cfg.cli_preset!r}")
        if cfg.load_path:
            print(f"[PPO] Warm-start checkpoint: {cfg.load_path}")
    return cfg
