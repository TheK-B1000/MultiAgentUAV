"""Training orchestration: coordinates the full PPO training run lifecycle.

:func:`orchestrate_training_run` replaces the body of
:func:`rl.train_ppo.train_ppo` — it owns the sequence of steps from config
validation through trainer teardown.  The original ``train_ppo`` function
delegates to this function for backward compatibility.

Responsibilities (in order):
1. Config validation gates (evaluation-only preset, gate_open ablation check)
2. Global seed + checkpoint directory creation
3. Resolved training config derivation
4. CSV path resolution + telemetry rotation
5. Training banner printing + run-lock acquisition + run-config JSON sidecar
6. Runtime clamping (team-size, CUDA fallback)
7. Environment construction
8. Trainer construction, checkpoint loading, timestep extension
9. ``trainer.learn`` loop (with ``KeyboardInterrupt`` emergency save)
10. Final checkpoint save + stats print
11. Teardown (telemetry, env, lock) in a ``finally`` block

Dependency direction: imports from ``rl.training.*`` sub-modules only,
never from ``rl.train_ppo`` or ``rl.training.cli``.
"""

from __future__ import annotations

import dataclasses
import json
import os
from typing import Optional

from rl.config.ppo_config import PPOConfig
from rl.training.banner import print_episode_stats_banner, print_training_banner
from rl.training.config_validation import normalize_and_validate_training_config
from rl.training.errors import EvaluationOnlyPresetError
from rl.training.factories import build_training_env
from rl.training.initialization import (
    build_trainer,
    maybe_extend_total_timesteps,
    maybe_load_checkpoint,
)
from rl.training.lifecycle import (
    _clamp_runtime_config_for_team_size,
    _ensure_cuda_or_fallback,
    _resolve_metrics_csv_paths,
    _rotate_fresh_run_telemetry,
    set_global_seed,
    teardown_training,
)
from rl.training.resolved_config import resolve_training_config
from rl.training.run_artifacts import _acquire_run_lock, write_run_config_json
from rl.training.run_context import RunContext


# ---------------------------------------------------------------------------
# Config validation gates
# ---------------------------------------------------------------------------

def _validate_config_gates(cfg: PPOConfig) -> None:
    """Raise typed errors for configs that must not start a training run.

    Two gates are checked in order:

    1. **Evaluation-only preset**: some presets (e.g. v6i2 promoted eval configs)
       set ``evaluation_only_preset=True`` to prevent accidental PPO training.
    2. **gate_open ablation**: the v6i5 CF-sweep gate-open ablation run must
       match the prior 8x sweep config exactly (modulo a small allowed-diff set)
       so the ablation comparison is apples-to-apples.
    """
    if bool(getattr(cfg, "evaluation_only_preset", False)):
        runner = str(getattr(cfg, "evaluation_only_runner", "") or "the evaluation runner")
        raise EvaluationOnlyPresetError(
            f"Preset {getattr(cfg, 'cli_preset', getattr(cfg, 'run_tag', 'unknown'))!r} "
            "is evaluation-only and must not start PPO training. "
            f"Use {runner} with a promoted v6i2 checkpoint."
        )

    if cfg.run_tag and "gate_open" in cfg.run_tag:
        if cfg.latent_cf_require_competence:
            raise ValueError(
                f"Ablation run {cfg.run_tag} requires --no-latent-cf-require-competence "
                f"but resolved latent_cf_require_competence is True!"
            )
        prior_config_path = os.path.join("checkpoints", "4v4_diag", "v6i5_cf_sweep_8x_150k_4v4_run_config.json")
        if not os.path.exists(prior_config_path):
            raise FileNotFoundError(f"Prior run config not found: {prior_config_path}")
        with open(prior_config_path, "r", encoding="utf-8") as f:
            prior_data = json.load(f)
        prior_resolved = prior_data.get("resolved_ppo_config", {})
        allowed_diffs = {
            "run_tag", "metrics_csv_path", "episode_csv_path",
            "latent_cf_require_competence", "checkpoint_dir",
            "utc_timestamp"
        }
        current_dict = dataclasses.asdict(cfg)
        mismatches = []
        for key, prior_val in prior_resolved.items():
            if key in allowed_diffs:
                continue
            if key not in current_dict:
                continue
            curr_val = current_dict[key]
            if isinstance(prior_val, list):
                prior_val = tuple(prior_val)
            if isinstance(curr_val, list):
                curr_val = tuple(curr_val)
            if curr_val != prior_val:
                mismatches.append(f"{key}: prior={prior_val}, current={curr_val}")
        if mismatches:
            raise ValueError(
                "Configuration mismatch vs prior 8x sweep config:\n" + "\n".join(mismatches)
            )


# ---------------------------------------------------------------------------
# Main orchestration entry point
# ---------------------------------------------------------------------------

def orchestrate_training_run(cfg: Optional[PPOConfig] = None) -> None:
    """Run the full local PPO/MAPPO training path.

    This is the canonical implementation extracted from
    :func:`rl.train_ppo.train_ppo`.  The original function delegates here so
    existing callers (scripts, presets, tests) keep working without changes.
    """
    cfg = normalize_and_validate_training_config(cfg or PPOConfig())
    _validate_config_gates(cfg)

    set_global_seed(cfg.seed, torch_seed=True, deterministic=cfg.use_deterministic)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    resolved = resolve_training_config(cfg)

    # Resolve CSV paths before banner so printed paths match trainer write targets.
    _resolve_metrics_csv_paths(cfg)

    print_training_banner(
        cfg,
        curriculum=resolved.curriculum,
        max_agents=resolved.max_agents,
        team_size=resolved.team_size,
    )

    run_lock = _acquire_run_lock(cfg)
    _rotate_fresh_run_telemetry(cfg)

    rc_path: Optional[str] = None
    try:
        rc_path = write_run_config_json(cfg)
        print(f"[PPO] Run config written: {rc_path}")
    except Exception as exc:
        print(f"[PPO] WARNING: could not write run config JSON: {exc}")

    run_context = RunContext(run_lock=run_lock, rc_path=rc_path)

    print_episode_stats_banner(
        cfg,
        curriculum=resolved.curriculum,
        initial_opponent_tag=resolved.initial_opponent_tag,
    )

    _clamp_runtime_config_for_team_size(cfg, resolved.max_agents)
    _ensure_cuda_or_fallback(cfg)

    env = build_training_env(
        cfg,
        initial_phase=resolved.initial_phase,
        initial_opponent_tag=resolved.initial_opponent_tag,
    )

    trainer = None
    try:
        trainer = build_trainer(env, cfg, resolved)
        maybe_load_checkpoint(cfg, trainer)
        maybe_extend_total_timesteps(cfg, trainer)

        try:
            stats = trainer.learn(total_timesteps=int(cfg.total_timesteps))
        except KeyboardInterrupt:
            interrupt_path = os.path.join(
                cfg.checkpoint_dir,
                f"interrupt_{cfg.run_tag}_{int(getattr(trainer, 'global_step', 0))}.zip",
            )
            trainer.save(interrupt_path)
            print(f"[PPO] KeyboardInterrupt: emergency checkpoint saved to: {interrupt_path}")
            raise

        final_path = os.path.join(cfg.checkpoint_dir, f"final_{cfg.run_tag}.zip")
        trainer.save(final_path)
        if stats:
            print(
                "[PPO] Final stats: "
                f"policy_loss={stats.get('policy_loss', 0.0):.4f}, "
                f"value_loss={stats.get('value_loss', 0.0):.4f}, "
                f"approx_kl={stats.get('approx_kl', 0.0):.5f}"
            )
        print(f"[PPO] Training complete. Final checkpoint saved to: {final_path}")
    finally:
        teardown_training(cfg, trainer, env, run_context.run_lock)
