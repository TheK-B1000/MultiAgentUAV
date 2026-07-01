"""Training lifecycle helpers: seed setup, CSV path resolution, telemetry rotation,
team-size clamping, CUDA fallback, and training-run teardown.

All functions extracted from :mod:`rl.train_ppo` (previously defined inline in
``train_ppo`` and the module body).  The originals are re-exported from
``rl.train_ppo`` for backward compatibility.

Dependency direction: this module imports only from stdlib, torch, and
``rl.training.{banner,run_artifacts,resolved_config,run_context}``; it never
imports from ``rl.train_ppo`` or ``rl.training.cli``.
"""

from __future__ import annotations

import os
import random
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from rl.config.ppo_config import PPOConfig


# ---------------------------------------------------------------------------
# Seed management (moved from train_ppo.py)
# ---------------------------------------------------------------------------

def set_global_seed(seed: int, torch_seed: bool = True, deterministic: bool = False) -> None:
    """Set Python, NumPy, and Torch seeds."""
    import numpy as np
    import torch

    random.seed(int(seed))
    np.random.seed(int(seed))
    if torch_seed:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# CSV path resolution and rotation (moved from train_ppo.py)
# ---------------------------------------------------------------------------

def _resolve_metrics_csv_paths(cfg: PPOConfig) -> None:
    """Resolve metrics / episode / strategy-experience CSV paths into ``cfg``.

    When metrics CSV logging is enabled, fills in any unset path from
    ``cfg.checkpoint_dir`` + ``cfg.run_tag``. When the strategy-experience CSV
    is not applicable (latent strategy or episode-credit disabled), clears it
    to ``None``. When metrics CSV logging is disabled, clears all three.

    Mutates ``cfg`` in place. Must run *before* :func:`print_training_banner`
    so the banner prints the same paths the trainer will later write to.
    """
    if bool(getattr(cfg, "enable_metrics_csv", True)):
        if not cfg.metrics_csv_path:
            cfg.metrics_csv_path = os.path.join(cfg.checkpoint_dir, f"{cfg.run_tag}_metrics.csv")
        if not cfg.episode_csv_path:
            cfg.episode_csv_path = os.path.join(cfg.checkpoint_dir, f"{cfg.run_tag}_episodes.csv")
        strategy_experience_enabled = bool(getattr(cfg, "use_latent_strategy", False)) and bool(
            getattr(cfg, "latent_episode_strategy_ppo", False)
        )
        if strategy_experience_enabled and not cfg.strategy_experience_csv_path:
            cfg.strategy_experience_csv_path = os.path.join(
                cfg.checkpoint_dir, f"{cfg.run_tag}_strategy_experience.csv"
            )
        elif not strategy_experience_enabled:
            cfg.strategy_experience_csv_path = None
        refresh_log_enabled = bool(getattr(cfg, "latent_v3i3_refresh_log_enabled", False))
        if refresh_log_enabled and not getattr(cfg, "latent_v3i3_refresh_log_path", None):
            cfg.latent_v3i3_refresh_log_path = os.path.join(
                cfg.checkpoint_dir, f"{cfg.run_tag}_refresh_log.csv"
            )
        elif not refresh_log_enabled:
            cfg.latent_v3i3_refresh_log_path = None
    else:
        cfg.metrics_csv_path = None
        cfg.episode_csv_path = None
        cfg.strategy_experience_csv_path = None
        cfg.latent_v3i3_refresh_log_path = None


def _rotate_fresh_run_telemetry(cfg: PPOConfig) -> None:
    """Rotate telemetry files that would otherwise be appended on a fresh run."""
    from rl.training.run_artifacts import _rotate_csv_aside

    if not (bool(getattr(cfg, "enable_metrics_csv", True)) and cfg.fresh_metrics_csv):
        return
    _rotate_csv_aside(cfg.metrics_csv_path, label="metrics")
    _rotate_csv_aside(cfg.episode_csv_path, label="episode")
    _rotate_csv_aside(cfg.strategy_experience_csv_path, label="strategy experience")
    _rotate_csv_aside(
        getattr(cfg, "latent_v3i3_refresh_log_path", None), label="v3i3 refresh log"
    )
    _rotate_csv_aside(
        getattr(cfg, "e3_step_telemetry_path", None), label="E3 step telemetry"
    )


# ---------------------------------------------------------------------------
# Runtime config clamping (moved from train_ppo.py)
# ---------------------------------------------------------------------------

def _clamp_runtime_config_for_team_size(cfg: PPOConfig, max_agents: int) -> None:
    """Reduce rollout / episode-length knobs for the 6v6 / 8v8 memory profile."""
    if max_agents == 6:
        cfg.n_envs = min(int(cfg.n_envs), 1)
        cfg.n_steps = min(int(cfg.n_steps), 512)
        cfg.max_decision_steps = min(int(cfg.max_decision_steps), 400)


def _ensure_cuda_or_fallback(cfg: PPOConfig) -> None:
    """If ``cfg.device`` is CUDA but unavailable for this torch build, fall back to CPU."""
    import torch

    if not str(cfg.device).lower().startswith("cuda"):
        return
    try:
        torch.zeros(1, device=cfg.device)
    except RuntimeError as exc:
        print(f"[PPO] CUDA unavailable for this torch build ({exc}). Falling back to CPU.")
        cfg.device = "cpu"


# ---------------------------------------------------------------------------
# Teardown (extracted from train_ppo.py::train_ppo() finally-block)
# ---------------------------------------------------------------------------

def teardown_training(
    cfg: PPOConfig,
    trainer: Optional[object],
    env: object,
    run_lock: object,
) -> None:
    """Release all acquired resources in a training run's finally-block.

    Closes the trainer's E3 telemetry writer (if any), closes the env,
    and releases the per-run-tag lockfile.  All exceptions from each step
    are swallowed so the finally-block completes even under partial
    initialization.
    """
    if trainer is not None:
        try:
            trainer.telemetry.close_e3_step_telemetry()  # type: ignore[union-attr]
        except Exception:
            pass
    try:
        env.close()  # type: ignore[union-attr]
    except Exception:
        pass
    try:
        run_lock.release()  # type: ignore[union-attr]
    except Exception:
        pass
