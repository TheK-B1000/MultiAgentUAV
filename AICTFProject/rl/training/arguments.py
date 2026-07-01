"""Argparse parser builder for the PPO training CLI.

:func:`build_train_parser` constructs the full :class:`argparse.ArgumentParser`
and is the single source of truth for all training flag names, types, defaults,
and help strings.  :func:`parse_train_args` is the thin public entry point
(builds parser and parses ``argv``).

The module is intentionally dependency-light: the only project import is a
*lazy* one inside :func:`build_train_parser` for ``DEFAULT_CLI_TRAINING_PRESET``
so that ``import rl.training.arguments`` does not trigger the full
``rl.train_ppo`` import chain during tool / test startup.

Backward-compat note: :func:`parse_train_args` is re-exported from
:mod:`rl.training.cli` so existing callers keep working.
"""

from __future__ import annotations

import argparse
from typing import Optional


def build_train_parser() -> argparse.ArgumentParser:
    """Construct and return the training argparse parser.

    The ``DEFAULT_CLI_TRAINING_PRESET`` constant is imported lazily so this
    module can be imported by tests without triggering ``rl.train_ppo``
    side effects.
    """
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
    parser.add_argument(
        "--additional-steps",
        type=int,
        default=None,
        help=(
            "Run this many additional steps beyond the loaded checkpoint's global_step. "
            "Resolved after checkpoint load: total_timesteps = checkpoint_step + N. "
            "Takes precedence over --total-steps when both are given."
        ),
    )
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
        "--training-telemetry-mode",
        type=str,
        default=None,
        help="Telemetry mode: off, basic, or full (Phase 6.1).",
    )
    parser.add_argument(
        "--training-events-jsonl-path",
        type=str,
        default=None,
        help="Path for JSONL telemetry events.",
    )
    parser.add_argument(
        "--telemetry-events-jsonl-path",
        type=str,
        default=None,
        help="Compatibility alias for JSONL telemetry events.",
    )
    parser.add_argument(
        "--performance-summary-path",
        type=str,
        default=None,
        help="Path for performance summary JSON.",
    )
    parser.add_argument(
        "--performance-samples-path",
        type=str,
        default=None,
        help="Path for performance samples CSV.",
    )
    parser.add_argument(
        "--gpu-monitor-enabled",
        action="store_true",
        help="Enable GPU utilization monitoring.",
    )
    parser.add_argument(
        "--gpu-monitor-interval-seconds",
        type=float,
        default=None,
        help="GPU monitor sample interval in seconds.",
    )
    parser.add_argument(
        "--fresh-metrics-csv",
        action="store_true",
        help="Rotate aside existing metrics/episode/E3 CSVs for this run_tag so telemetry is not appended.",
    )
    parser.add_argument("--load", type=str, default=None, help="Path to a checkpoint to resume training from.")
    parser.add_argument("--resume", type=str, default=None, help="Alias for --load.")
    parser.add_argument(
        "--load-weights-only",
        action="store_true",
        help="Load only model weights from a checkpoint, discarding the optimizer state. Useful when resuming with a different model structure.",
    )
    parser.add_argument(
        "--allow-active-actor-module-migration",
        action="store_true",
        help="Allow active actor-affecting modules (e.g. z_adapter, actor_z_film) to be discarded during compatibility load.",
    )
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
        help="Number of vectorized GPU env instances (default: PPOConfig.n_envs=32). Increase to keep the GPU busy.",
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
    parser.add_argument(
        "--map-layout",
        type=str,
        choices=["map_a_open", "map_b_split_lane", "map_b_split_lane_v2", "open", "split_lane", "split_lane_v2"],
        default=None,
        help="Environment geometry layout. map_a_open preserves the historical open arena; map_b_split_lane enables the split-lane chokepoint; map_b_split_lane_v2 uses the lower-friction task-pressure variant.",
    )
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
        "--latent-cf-coef-max",
        type=float,
        default=None,
        help="V6I1 counterfactual separation ceiling (latent_cf_coef_max); schedule still ramps 0→max in Phase A.",
    )
    parser.add_argument(
        "--no-latent-cf-require-competence",
        action="store_true",
        default=False,
        help="Bypass/disable the competence gate on counterfactual separation loss.",
    )
    parser.add_argument(
        "--latent-cf-sequential-update",
        action="store_true",
        default=False,
        help="Perform sequential PPO and CF actor updates in each training step.",
    )
    parser.add_argument(
        "--actor-cf-update-mode",
        choices=("combined", "ppo_then_cf", "cf_then_ppo"),
        default=None,
        help="Actor update geometry for CF diagnostics: combined, ppo_then_cf, or cf_then_ppo.",
    )
    parser.add_argument(
        "--phase-a-disable-promotion",
        action="store_true",
        default=False,
        help="Diagnostic only: run Phase A gate reports but do not transition to Phase B.",
    )
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
    # --- v5i9 CSIA extension: detached causal strategic-impact reward.
    parser.add_argument(
        "--csia-enabled",
        action="store_true",
        help=(
            "v5i9 extension: enable detached CSIA reward from frozen forced-z "
            "evaluation evidence. Requires --csia-payoff-csv for a nonzero bonus."
        ),
    )
    parser.add_argument(
        "--csia-reward-coef",
        type=float,
        default=None,
        help="Scale on centered causal strategic-impact advantage S(opponent,z).",
    )
    parser.add_argument(
        "--csia-payoff-csv",
        type=str,
        default=None,
        help="Path to *_qualitative_rollout_by_z.csv from tools/v5i8_forced_z_eval.py.",
    )
    parser.add_argument(
        "--csia-strategy-evidence-csv",
        type=str,
        default=None,
        help="Path to *_strategy_evidence.csv for natural router baseline and behavior-spread gate.",
    )
    parser.add_argument(
        "--csia-probe-interval",
        type=int,
        default=None,
        help="Reload CSIA evidence every N PPO updates; 0 loads once.",
    )
    parser.add_argument(
        "--csia-min-behavior-spread",
        type=float,
        default=None,
        help="Gate A threshold on forced-z behavioral spread.",
    )
    parser.add_argument(
        "--csia-min-interaction-strength",
        type=float,
        default=None,
        help="Gate B threshold on centered opponent-latent interaction strength.",
    )
    parser.add_argument(
        "--csia-quality-floor-delta",
        type=float,
        default=None,
        help="Gate C tolerance below natural-router baseline for every forced-z cell.",
    )
    parser.add_argument(
        "--csia-min-count-per-cell",
        type=int,
        default=None,
        help="Minimum forced-z episodes required before a payoff matrix cell is used.",
    )
    parser.add_argument(
        "--no-csia-require-gates",
        action="store_true",
        help="Allow CSIA reward whenever evidence exists, even if gates fail. Diagnostic only.",
    )
    parser.add_argument(
        "--v6i6-anchor-validation-manifest",
        type=str,
        default=None,
        help="v6i6 E1: hashed manifest selecting anchors, expansion target, dormant latents, and evidence.",
    )
    # --- v4i4post (post-Summer extension): Periodic Return-Ranked Router Distillation.
    parser.add_argument(
        "--latent-router-distill-enabled",
        action="store_true",
        default=None,
        help=(
            "v4i4post: after each periodic checkpoint save, run tools/q_probe.py + "
            "tools/router_distill_from_qprobe.py as subprocesses against the "
            "just-saved checkpoint and hot-swap the distilled strategy_encoder "
            "weights back into the running model. Off by default (v4i3 Summer "
            "Proof keeps this off; only v4i4post turns it on)."
        ),
    )
    parser.add_argument(
        "--latent-router-distill-every-n-steps",
        type=int,
        default=None,
        help="v4i4post cadence (default 250000). The hook fires on the first periodic-save boundary >= this.",
    )
    parser.add_argument(
        "--latent-router-distill-n-seeds",
        type=int,
        default=None,
        help="v4i4post: matched-start seeds per (opponent, z) for the in-trainer q_probe.",
    )
    parser.add_argument(
        "--latent-router-distill-base-seed",
        type=int,
        default=None,
        help="v4i4post: q_probe base seed; the hook uses seeds base..base+n_seeds-1.",
    )
    parser.add_argument(
        "--latent-router-distill-opponents",
        nargs="+",
        default=None,
        help="v4i4post: opponent labels for the in-trainer q_probe (default: OP5 OP6 OP7).",
    )
    parser.add_argument(
        "--latent-router-distill-epochs",
        type=int,
        default=None,
        help="v4i4post distill epochs per round (default 100).",
    )
    parser.add_argument(
        "--latent-router-distill-lr",
        type=float,
        default=None,
        help="v4i4post distill learning rate (default 1e-4).",
    )
    parser.add_argument(
        "--latent-router-distill-temperature",
        type=float,
        default=None,
        help="v4i4post soft-target temperature (default 1.0).",
    )
    parser.add_argument(
        "--latent-router-distill-weight-decay",
        type=float,
        default=None,
        help="v4i4post distill optimizer weight decay (default 0.0).",
    )
    parser.add_argument(
        "--latent-router-distill-device",
        type=str,
        default=None,
        help="v4i4post subprocess device for q_probe + distill (default: cpu).",
    )
    parser.add_argument(
        "--latent-router-distill-artifacts-subdir",
        type=str,
        default=None,
        help="v4i4post subdir under --checkpoint-dir for round artifacts (default: v4i4post_router_distill).",
    )
    parser.add_argument(
        "--no-progress-bar",
        action="store_true",
        help="Disable the SB3-style tqdm rollout bar (default: on; uses tqdm.rich if installed).",
    )
    return parser


def parse_train_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Construct the training argparse parser and parse ``argv`` (defaults ``sys.argv[1:]``)."""
    return build_train_parser().parse_args(argv)
