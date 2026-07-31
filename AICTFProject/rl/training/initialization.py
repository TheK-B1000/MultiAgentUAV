"""Trainer construction, checkpoint loading, and timestep extension.

Extracted from :mod:`rl.train_ppo` (the ``train_ppo`` function body that
constructs ``CustomPPOTrainer``, loads a checkpoint, and optionally extends
``total_timesteps`` via ``--additional-steps``).

Each function is a focused, independently testable unit:

* :func:`build_trainer` — constructs :class:`CustomPPOTrainer` from env + resolved config
* :func:`maybe_load_checkpoint` — loads a checkpoint when ``cfg.load_path`` is set
* :func:`maybe_extend_total_timesteps` — increases ``cfg.total_timesteps`` by
  ``cfg.additional_timesteps`` after the base step is known from the checkpoint

Dependency direction: imports from ``rl.custom_ppo``, ``rl.training.resolved_config``,
``rl.training.factories`` — never from ``rl.train_ppo`` or ``rl.training.cli``.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional

from rl.config.ppo_config import PPOConfig
from rl.training.resolved_config import ResolvedTrainingConfig

if TYPE_CHECKING:
    from rl.custom_ppo import CustomPPOTrainer


def build_trainer(
    env: object,
    cfg: PPOConfig,
    resolved: ResolvedTrainingConfig,
    *,
    run_identity,
) -> "CustomPPOTrainer":
    """Construct and return a :class:`CustomPPOTrainer` from ``env`` and resolved config.

    ``run_identity`` is mandatory: the trainer stamps episode rows and checkpoints
    from this frozen object. Missing identity must fail before the first rollout.
    """
    from rl.custom_ppo import CustomPPOTrainer
    from rl.custom_ppo.trainer_audit import log_input_dim_contract
    from rl.ruleset_identity import RunIdentity, RunIdentityError

    if run_identity is None or not isinstance(run_identity, RunIdentity):
        raise RunIdentityError(
            "build_trainer requires the run's resolved RunIdentity; "
            "refusing to construct a trainer that cannot stamp artifacts."
        )

    if bool(getattr(cfg, "use_stable_marl_ppo", False)):
        print(
            "[PPO] Optional stable-MARL override (use_stable_marl_ppo=True; not in Word spec): "
            "lr=1.5e-4, n_epochs=2, clip=0.10, ent=0.005, batch_size=1024."
        )
    if resolved.max_agents > 2:
        print(
            f"[PPO] {resolved.team_size}: using lr={resolved.effective_lr:.2e} for stability."
        )

    if resolved.effective_batch_size != int(cfg.batch_size):
        print(
            f"[PPO] Adjusting batch_size to rollout size: {resolved.effective_batch_size}."
        )

    trainer = CustomPPOTrainer(
        env,
        cfg,
        learning_rate=resolved.effective_lr,
        clip_range=resolved.effective_clip_range,
        ent_coef=resolved.effective_ent_coef,
        n_epochs=resolved.effective_n_epochs,
        batch_size=resolved.effective_batch_size,
        value_clip_range=getattr(cfg, "clip_range_vf", resolved.effective_clip_range),
        curriculum=resolved.curriculum,
        run_identity=run_identity,
    )
    log_input_dim_contract(trainer)
    return trainer


def maybe_load_checkpoint(cfg: PPOConfig, trainer: "CustomPPOTrainer") -> None:
    """Load a checkpoint into ``trainer`` when ``cfg.load_path`` points to an existing file."""
    if cfg.load_path and os.path.isfile(cfg.load_path):
        print(f"[PPO] Resuming checkpoint: {cfg.load_path}")
        trainer.load(cfg.load_path)
        if bool(getattr(cfg, "freeze_return_norm_after_load", False)):
            trainer.return_norm.freeze()
            print(
                "[PPO] freeze_return_norm_after_load: return_norm stats frozen "
                f"(mean={trainer.return_norm.mean:.6f}, std={trainer.return_norm.std:.6f}, "
                f"count={trainer.return_norm.count:.0f})"
            )


def maybe_extend_total_timesteps(cfg: PPOConfig, trainer: "CustomPPOTrainer") -> None:
    """Extend ``cfg.total_timesteps`` by ``cfg.additional_timesteps`` post-checkpoint.

    Must be called *after* :func:`maybe_load_checkpoint` so ``trainer.global_step``
    reflects the loaded step count.
    """
    if int(getattr(cfg, "additional_timesteps", 0) or 0) > 0:
        base_step = int(getattr(trainer, "global_step", 0))
        cfg.checkpoint_run_start_step = base_step
        cfg.total_timesteps = base_step + int(cfg.additional_timesteps)
        print(
            f"[PPO] --additional-steps: base_step={base_step:,} + {int(cfg.additional_timesteps):,} "
            f"= total_timesteps={cfg.total_timesteps:,}"
        )


def maybe_configure_periodic_checkpoints(cfg: PPOConfig, trainer: "CustomPPOTrainer") -> None:
    """Resolve run-relative vs global checkpoint naming after load."""
    if int(getattr(cfg, "additional_timesteps", 0) or 0) > 0:
        if int(getattr(cfg, "checkpoint_run_start_step", 0) or 0) <= 0:
            cfg.checkpoint_run_start_step = int(getattr(trainer, "global_step", 0))
    trainer.configure_periodic_checkpoints()
