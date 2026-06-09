"""CLI parser and entry point for ``python rl/train_ppo.py``.

Lifts the ``argparse`` definitions and ``cfg = PPOConfig() + per-arg overrides``
chain out of :mod:`rl.train_ppo` so the script body stays a thin launchpad.

Public API:

* :func:`parse_train_args` -- argparse parser, returns the raw namespace.
* :func:`cfg_from_args` -- turn the namespace into a populated ``PPOConfig``
  (presets, mode normalization, run-tag synthesis, deprecation warnings).
* :func:`main` -- orchestrate the script entry point (``--verify-4v4`` /
  ``--test-vec-schema`` shortcuts, otherwise parse + cfg + ``train_ppo``).

Reproducibility contract: CLI flag names, defaults, and the deprecation
warning text are part of the user-facing contract. Treat any rename as a
breaking change for existing run scripts (``experiments/*.ps1``).
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

from rl.config.ppo_config import PPOConfig


def parse_train_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Construct the training argparse parser and parse ``argv`` (defaults ``sys.argv[1:]``)."""
    # Imported lazily so this module can be imported by tests/tools without
    # triggering ``rl.train_ppo`` side effects (sys.path mutation, heavy
    # downstream imports). ``DEFAULT_CLI_TRAINING_PRESET`` is needed up front
    # for the default value and help string of ``--preset``.
    from rl.train_ppo import DEFAULT_CLI_TRAINING_PRESET

    parser = argparse.ArgumentParser(description="Train custom PPO/MAPPO for CTF")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--preset",
        type=str,
        default=DEFAULT_CLI_TRAINING_PRESET,
        help=(
            f"Apply a named training preset before other CLI overrides (default: {DEFAULT_CLI_TRAINING_PRESET!r}). "
            "Use 'none' or '' to skip presets and use PPOConfig + CLI fields only. "
            "Examples: plan_faithful_latent_persist_entropy (recommended), "
            "latent_op3_wrmax_1m (drift wrmax), latent_a1_plan_faithful (legacy A1)."
        ),
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        help="Training mode: FIXED_OPPONENT, OPPONENT_POOL (uniform --opponent-pool per episode), or CURRICULUM.",
    )
    parser.add_argument("--run-tag", type=str, default=None)
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--metrics-csv", type=str, default=None, help="Path for per-update training metrics CSV.")
    parser.add_argument("--episode-csv", type=str, default=None, help="Path for per-episode training outcome CSV.")
    parser.add_argument(
        "--strategy-experience-csv",
        type=str,
        default=None,
        help="Path for latent bucket/z return diagnostics CSV.",
    )
    parser.add_argument("--no-metrics-csv", action="store_true", help="Disable training CSV telemetry.")
    parser.add_argument(
        "--fresh-metrics-csv",
        action="store_true",
        help="Rotate aside existing metrics/episode CSVs for this run_tag so telemetry is not appended.",
    )
    parser.add_argument("--load", type=str, default=None)
    parser.add_argument("--learning-rate", type=float, default=None, help="PPO learning rate.")
    parser.add_argument(
        "--lr-floor-frac",
        type=float,
        default=None,
        help="Minimum fraction of the base learning rate used by the linear schedule.",
    )
    parser.add_argument("--target-kl", type=float, default=None, help="PPO target KL; negative disables early stopping.")
    parser.add_argument("--n-epochs", type=int, default=None, help="Number of PPO optimization epochs per rollout.")
    parser.add_argument(
        "--n-envs",
        type=int,
        default=None,
        help="Number of vectorized GPU env instances (default: PPOConfig.n_envs=8). Increase to keep the GPU busy.",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=None,
        help="Rollout length per env between PPO updates (default: PPOConfig.n_steps=2048).",
    )
    parser.add_argument(
        "--clip-range-vf",
        type=float,
        default=None,
        help="Value-function clip range; negative disables value clipping.",
    )
    parser.add_argument("--vf-coef", type=float, default=None, help="Value-function loss coefficient.")
    parser.add_argument(
        "--return-normalization",
        action="store_true",
        help="Train critic on normalized returns while denormalizing values for GAE/advantages.",
    )
    parser.add_argument("--fixed-opponent", type=str, default="OP3")
    parser.add_argument(
        "--opponent-randomize",
        action="store_true",
        help="Each finished episode samples next scripted opponent uniformly from --opponent-pool (FIXED_OPPONENT or OPPONENT_POOL).",
    )
    parser.add_argument(
        "--opponent-pool",
        nargs="+",
        default=None,
        metavar="TAG",
        help=(
            "Scripted tags when --opponent-randomize or mode=OPPONENT_POOL (config default: OP1 OP2 OP3). "
            "OP4 is removed from training pools unless --allow-op4-in-training-pool is set. "
            "OP5_RUSHER (alias OP5) is a fast-rush stress test; OP6 / OP6_TURTLE is a defensive turtle; "
            "OP7 / OP7_SWITCHER is a trainable deceptive multi-profile scripted opponent."
        ),
    )
    parser.add_argument(
        "--opponent-pool-weights",
        nargs="+",
        default=None,
        metavar="TAG=PROB",
        help=(
            "Per-tag sampling weights for --opponent-pool. Format: TAG=PROB (e.g. "
            "'--opponent-pool-weights OP3=0.2 OP5=0.5 OP6=0.3'). Weights are auto-normalized "
            "to sum 1.0. Missing tags from the pool are rejected; extra tags ignored. "
            "Default: uniform 1/N over the pool. Plan-faithful — does not give the model "
            "opponent identity, only changes contested-signal frequency."
        ),
    )
    parser.add_argument(
        "--allow-op4-in-training-pool",
        action="store_true",
        help="Allow OP4 in opponent_pool during training (default: OP4 is eval-only and stripped).",
    )
    parser.add_argument(
        "--e3-step-telemetry",
        action="store_true",
        help=(
            "Latent only: write per-decision-step CSV (z, entropy, argmax, switched, game_phase) next to metrics "
            "for E2 / strategy-switch analysis (path: <checkpoint-dir>/<run-tag>_e3_steps.csv)."
        ),
    )
    parser.add_argument("--map-set", type=str, choices=["train", "eval"], default=None)
    parser.add_argument("--agents", type=int, choices=[2, 4, 6, 8], default=None)
    parser.add_argument("--max-blue-agents", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--verbose-training", action="store_true")
    parser.add_argument(
        "--stable-marl",
        action="store_true",
        help="Enable stable-MARL PPO hyperparameter override (not defined in Summer Implementation Plan.docx).",
    )
    parser.add_argument(
        "--latent-strategy",
        action="store_true",
        help="Enable latent team strategy training (default in PPOConfig).",
    )
    parser.add_argument(
        "--no-latent-strategy",
        action="store_true",
        help="Disable latent team strategy for vanilla local PPO ablations.",
    )
    parser.add_argument("--latent-k", type=int, default=None, help="Number of discrete team strategies.")
    parser.add_argument(
        "--latent-resample-every",
        type=int,
        default=None,
        help="Sparse strategy refresh interval in decision steps; 0 samples at episode start only.",
    )
    parser.add_argument(
        "--fixed-latent-strategy",
        action="store_true",
        help="Baseline: keep latent actor/critic inputs but clamp all rollouts to one fixed z ID.",
    )
    parser.add_argument(
        "--fixed-latent-id",
        type=int,
        default=None,
        help="Strategy ID used by --fixed-latent-strategy (default: 0). Supplying this implies fixed latent.",
    )
    parser.add_argument("--latent-lam-p", type=float, default=None, help="Strategy persistence penalty weight.")
    parser.add_argument("--latent-lam-h", type=float, default=None, help="Strategy entropy weight (see --latent-entropy-objective).")
    parser.add_argument(
        "--latent-strategy-ppo-coef",
        type=float,
        default=None,
        help="Coefficient for the sampled-z clipped PPO strategy loss.",
    )
    parser.add_argument(
        "--latent-episode-strategy-ppo",
        action="store_true",
        help="Enable Option A episode-level PPO credit for q_phi's sampled episode-start z.",
    )
    parser.add_argument(
        "--latent-episode-strategy-coef",
        type=float,
        default=None,
        help="Weight for episode-level q_phi PPO credit (recommended sweep: 0.25, 0.5, 1.0).",
    )
    parser.add_argument(
        "--latent-episode-strategy-clip-eps",
        type=float,
        default=None,
        help="Clip epsilon for episode-level q_phi PPO credit.",
    )
    parser.add_argument(
        "--latent-episode-strategy-value-coef",
        type=float,
        default=None,
        help="Value baseline coefficient for episode-level q_phi PPO credit.",
    )
    parser.add_argument(
        "--no-latent-episode-strategy-return-norm",
        action="store_true",
        help="Disable rollout-normalized episode-level q_phi advantages.",
    )
    parser.add_argument(
        "--latent-strategy-aux-return-head",
        action="store_true",
        help="Enable A2 auxiliary per-z return regression on the shared q_phi trunk (sampled z only; not Q-learning).",
    )
    parser.add_argument(
        "--latent-strategy-q-head",
        action="store_true",
        help=argparse.SUPPRESS,  # DEPRECATED legacy alias for --latent-strategy-aux-return-head; emits a warning.
    )
    parser.add_argument(
        "--latent-strategy-aux-return-coef",
        type=float,
        default=None,
        help="Weight for the A2 auxiliary return MSE on resampled-z minibatches.",
    )
    parser.add_argument(
        "--latent-strategy-q-coef",
        type=float,
        default=None,
        help=argparse.SUPPRESS,  # DEPRECATED legacy alias for --latent-strategy-aux-return-coef; emits a warning.
    )
    parser.add_argument(
        "--latent-strategy-aux-predict-phase-coef",
        type=float,
        default=None,
        help="Weight for the optional phase prediction auxiliary loss.",
    )
    parser.add_argument(
        "--latent-strategy-tau",
        type=float,
        default=None,
        help="Softmax temperature for strategy logits when the auxiliary return head shares the trunk.",
    )
    parser.add_argument(
        "--latent-entropy-objective",
        type=str,
        choices=("maximize", "minimize", "none"),
        default=None,
        help="How lambda_H shapes H(q_phi): maximize=paper bonus on entropy; minimize=penalty (sharper z); none=off.",
    )
    parser.add_argument(
        "--latent-resample-on-flag",
        action="store_true",
        help="Resample z when the global-state flag/territory features change (optional plan §12).",
    )
    parser.add_argument(
        "--latent-kl-consecutive",
        type=float,
        default=None,
        help="Weight for consecutive-step KL on q_phi logits (0=off; optional plan §12).",
    )
    parser.add_argument(
        "--latent-v3i3-event-preference-normalize",
        action="store_true",
        help="Normalize event preference returns by subtracting the baseline for the specific event key.",
    )
    parser.add_argument(
        "--no-latent-gae-z-reset",
        action="store_true",
        help="Keep legacy GAE: carry lambda-returns across z switches (can smear credit when V(s,z) jumps).",
    )
    parser.add_argument(
        "--latent-bootstrap-z-stochastic",
        action="store_true",
        help="Sample z for V(s') bootstrap; default argmax to avoid RNG mismatch with the next step.",
    )
    parser.add_argument(
        "--domain-randomization",
        action="store_true",
        help="Enable episode-level domain randomization (obs noise/dropout, blue speed jitter).",
    )
    parser.add_argument(
        "--dr-sensor-noise-max",
        type=float,
        default=None,
        help="Max enemy obs position noise (map cells) per episode when domain randomization is on.",
    )
    parser.add_argument(
        "--dr-sensor-dropout-max",
        type=float,
        default=None,
        help="Max in-range enemy dropout probability per episode when domain randomization is on.",
    )
    parser.add_argument(
        "--dr-blue-speed-jitter",
        type=float,
        default=None,
        help="Blue max-speed scale draws from U(1-j,1) per episode (slowdown-only; marine speed cap unchanged).",
    )
    parser.add_argument(
        "--latent-z-embed-dim",
        type=int,
        default=None,
        help="Strategy embedding dimension used by the shared actor.",
    )
    parser.add_argument(
        "--latent-vf-hidden",
        type=int,
        default=None,
        help="Hidden width for the centralized latent critic.",
    )
    parser.add_argument(
        "--episode-log-every",
        type=int,
        default=None,
        metavar="N",
        help="Log W/L/D every N completed episodes (0=off; default from PPOConfig).",
    )
    parser.add_argument(
        "--env-win-reward",
        type=float,
        default=None,
        help="Override GPU env terminal win_team_reward (training-only shaping).",
    )
    parser.add_argument(
        "--env-draw-penalty",
        type=float,
        default=None,
        help="Override GPU env terminal draw_team_penalty.",
    )
    parser.add_argument(
        "--env-lose-penalty",
        type=float,
        default=None,
        help="Override GPU env terminal lose_team_punish.",
    )
    parser.add_argument(
        "--env-action-failed-penalty",
        type=float,
        default=None,
        help="Override GPU env action_failed_punishment; useful when macro failure noise dominates outcomes.",
    )
    parser.add_argument(
        "--env-dense-weight",
        type=float,
        default=None,
        help="Scale dense PBRS/team shaping vs sparse/terminal (GPU RewardConfig.dense_weight).",
    )
    parser.add_argument(
        "--env-sparse-weight",
        type=float,
        default=None,
        help="Sparse event weight before /100 normalization (GPU RewardConfig.sparse_weight).",
    )
    parser.add_argument(
        "--env-reward-scale",
        type=float,
        default=None,
        help="Denominator inside tanh(raw/scale) before reward_clip (GPU RewardConfig.reward_scale).",
    )
    parser.add_argument(
        "--env-reward-clip",
        type=float,
        default=None,
        help="Clamp bound on scaled per-step reward (GPU RewardConfig.reward_clip).",
    )
    parser.add_argument(
        "--env-stalemate-penalty",
        type=float,
        default=None,
        help="Extra penalty when stalemate truncation fires.",
    )
    parser.add_argument(
        "--env-stalemate-max-steps",
        type=int,
        default=None,
        help="Consecutive low-progress steps before stalemate truncation (per env).",
    )
    parser.add_argument(
        "--reward-shaping-coef-start",
        type=float,
        default=None,
        help="Initial multiplier applied to (offense+pbrs+team) in PPO training rewards.",
    )
    parser.add_argument(
        "--reward-shaping-coef-end",
        type=float,
        default=None,
        help="Final multiplier after --reward-shaping-decay-steps.",
    )
    parser.add_argument(
        "--reward-shaping-decay-steps",
        type=int,
        default=None,
        help="Linear decay horizon for reward shaping coefficient; 0 disables.",
    )
    parser.add_argument(
        "--periodic-checkpoint-steps",
        type=int,
        default=None,
        help="Save checkpoint every N env steps during training (0 disables).",
    )
    # --- v4i3: Periodic Return-Ranked Router Distillation ---------------
    parser.add_argument(
        "--latent-router-distill-enabled",
        action="store_true",
        default=None,
        help=(
            "v4i3: after each periodic checkpoint save, run tools/q_probe.py + "
            "tools/router_distill_from_qprobe.py as subprocesses against the "
            "just-saved checkpoint and hot-swap the distilled strategy_encoder "
            "weights back into the running model. Off by default."
        ),
    )
    parser.add_argument(
        "--latent-router-distill-every-n-steps",
        type=int,
        default=None,
        help="v4i3 cadence (default 250000). The hook fires on the first periodic-save boundary >= this.",
    )
    parser.add_argument(
        "--latent-router-distill-n-seeds",
        type=int,
        default=None,
        help="v4i3: matched-start seeds per (opponent, z) for the in-trainer q_probe.",
    )
    parser.add_argument(
        "--latent-router-distill-base-seed",
        type=int,
        default=None,
        help="v4i3: q_probe base seed; the hook uses seeds base..base+n_seeds-1.",
    )
    parser.add_argument(
        "--latent-router-distill-opponents",
        nargs="+",
        default=None,
        help="v4i3: opponent labels for the in-trainer q_probe (default: OP5 OP6 OP7).",
    )
    parser.add_argument(
        "--latent-router-distill-epochs",
        type=int,
        default=None,
        help="v4i3 distill epochs per round (default 100).",
    )
    parser.add_argument(
        "--latent-router-distill-lr",
        type=float,
        default=None,
        help="v4i3 distill learning rate (default 1e-4).",
    )
    parser.add_argument(
        "--latent-router-distill-temperature",
        type=float,
        default=None,
        help="v4i3 soft-target temperature (default 1.0).",
    )
    parser.add_argument(
        "--latent-router-distill-weight-decay",
        type=float,
        default=None,
        help="v4i3 distill optimizer weight decay (default 0.0).",
    )
    parser.add_argument(
        "--latent-router-distill-device",
        type=str,
        default=None,
        help="v4i3 subprocess device for q_probe + distill (default: cpu).",
    )
    parser.add_argument(
        "--latent-router-distill-artifacts-subdir",
        type=str,
        default=None,
        help="v4i3 subdir under --checkpoint-dir for round artifacts (default: v4i3_router_distill).",
    )
    parser.add_argument(
        "--no-progress-bar",
        action="store_true",
        help="Disable the SB3-style tqdm rollout bar (default: on; uses tqdm.rich if installed).",
    )
    return parser.parse_args(argv)


def cfg_from_args(args: argparse.Namespace) -> PPOConfig:
    """Apply a parsed argparse namespace onto a fresh ``PPOConfig``.

    Mirrors the ordering of the original inline block in ``rl/train_ppo.py``:
    fresh config -> optional preset -> per-flag overrides -> run_tag /
    checkpoint_dir synthesis -> ``cli_preset`` bookkeeping. No env / trainer
    construction happens here; pure data prep.
    """
    from rl.train_ppo import (
        _agents_suffix,
        _apply_training_preset,
        _default_run_tag_for_mode,
        _ensure_run_tag_has_agent_suffix,
        _normalize_train_mode,
    )

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
    if "v4i1" in preset_key.lower():
        # v4i1's whole thesis is that the strategic-pressure pool {OP5, OP6, OP7}
        # is the experimental treatment. A stray ``--opponent-pool`` on the CLI
        # would silently override the preset and invalidate the run. Refuse
        # anything other than exactly {OP5, OP6, OP7} (any order) for this preset.
        required_v4i1_pool = frozenset({"OP5", "OP6", "OP7"})
        actual_pool = frozenset(str(x).upper() for x in (cfg.opponent_pool or ()))
        if actual_pool != required_v4i1_pool:
            raise ValueError(
                "v4i1 preset requires opponent_pool == {OP5, OP6, OP7} exactly "
                f"(got {sorted(actual_pool)!r}). Remove any --opponent-pool override "
                "from the command line and let the preset own the pool, or pass "
                "--opponent-pool OP5 OP6 OP7 explicitly."
            )
        if not cfg.opponent_randomize:
            raise ValueError(
                "v4i1 preset requires --opponent-randomize (preset sets this by "
                "default). Do not disable it on the command line."
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
    if args.load is not None:
        cfg.load_path = args.load
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
    # --- v4i3 router-distill overrides ----------------------------------
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
        ).strip() or "v4i3_router_distill"
    if args.no_progress_bar:
        cfg.enable_progress_bar = False
    if preset_key:
        cfg.cli_preset = preset_key
        print(f"[PPO] Training preset: {cfg.cli_preset!r}")
        if cfg.load_path:
            print(f"[PPO] Warm-start checkpoint: {cfg.load_path}")
    return cfg


def main(argv: Optional[list[str]] = None) -> None:
    """``python rl/train_ppo.py`` entry point.

    Honors the two read-only diagnostic flags first (``--verify-4v4`` and
    ``--test-vec-schema``) so they keep working without parsing the full
    training argparse surface; otherwise builds ``PPOConfig`` from CLI and
    runs :func:`rl.train_ppo.train_ppo`.

    ``argv`` follows the same convention as :func:`argparse.parse_args`: pass
    ``None`` (default) to inherit ``sys.argv``; pass an explicit list (without
    the program name) to drive the CLI programmatically.
    """
    flag_source = sys.argv[1:] if argv is None else list(argv)
    if "--verify-4v4" in flag_source:
        from rl.train_ppo import run_verify_4v4

        run_verify_4v4(num_episodes=10)
        return
    if "--test-vec-schema" in flag_source:
        from rl.train_ppo import run_test_vec_schema

        run_test_vec_schema()
        return

    from rl.train_ppo import train_ppo

    args = parse_train_args(argv)
    cfg = cfg_from_args(args)
    train_ppo(cfg)


__all__ = ["cfg_from_args", "main", "parse_train_args"]
