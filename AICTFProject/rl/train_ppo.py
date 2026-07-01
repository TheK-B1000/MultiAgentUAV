"""Train the CTF policy with the local PPO/MAPPO implementation.

Backward-compatibility facade: orchestration logic has been extracted to
:mod:`rl.training.orchestrator`; CLI helpers to :mod:`rl.training.overrides`;
lifecycle helpers to :mod:`rl.training.lifecycle`.

All previously exported names are re-exported here so that existing import
paths (presets, tools, tests, archived scripts) continue to work without
changes.  Prefer the canonical sub-module paths in new code:

* ``rl.training.orchestrator.orchestrate_training_run``
* ``rl.training.lifecycle.set_global_seed``
* ``rl.training.overrides._agents_suffix``
* ``rl.training.resolved_config._resolve_initial_opponent_and_phase``
"""

from __future__ import annotations

import os
import sys
from typing import Optional

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_SCRIPT_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

from game_field_gpu import GPUCTFVecEnv, GPUFieldConfig, VEC_OBS_DIM
from rl.config.ppo_config import PPOConfig, TrainMode
from rl.curriculum import CurriculumState, phase_from_tag  # noqa: F401 — re-exported for legacy callers
from rl.global_state import GLOBAL_STATE_DIM
from rl.training.config_validation import (
    EVAL_ONLY_TRAINING_OPPONENT_TAGS,
    _normalize_train_mode,
    _strip_eval_only_opponents_from_training_pool,
    normalize_and_validate_training_config,
)
from rl.training.env_factory import (  # noqa: F401  -- re-exported for tools/critic_ceiling.py and other external callers
    _apply_initial_opponent_params,
    _gpu_env_reward_kwargs,
    build_training_env,
)
from rl.training.lifecycle import (  # noqa: F401 -- re-exported for legacy import paths
    _clamp_runtime_config_for_team_size,
    _ensure_cuda_or_fallback,
    _resolve_metrics_csv_paths,
    _rotate_fresh_run_telemetry,
    set_global_seed,
)
from rl.training.overrides import (  # noqa: F401 -- re-exported for legacy import paths
    _agents_suffix,
    _default_run_tag_for_mode,
    _ensure_run_tag_has_agent_suffix,
)
from rl.training.resolved_config import (  # noqa: F401 -- re-exported for legacy import paths
    _resolve_initial_opponent_and_phase,
)
from rl.training.run_artifacts import (  # noqa: F401  -- re-exported for legacy import paths from tools/tests/presets
    _RunLock,
    _acquire_run_lock,
    _find_git_root,
    _git_metadata,
    _json_safe,
    _metrics_csv_nonempty,
    _pid_is_running,
    _read_run_lock,
    _rotate_csv_aside,
    _run_config_json_path,
    write_run_config_json,
)

# Re-export so existing ``from rl.train_ppo import ...`` call sites (presets,
# tools, tests, archived log preflights) keep working without rewrites. The
# canonical homes are now ``rl.config.ppo_config``,
# ``rl.training.run_artifacts``, ``rl.training.env_factory``,
# ``rl.training.config_validation``, ``rl.training.lifecycle``,
# ``rl.training.overrides``, and ``rl.training.resolved_config`` -- prefer
# those paths in new code.
__all__ = [
    "PPOConfig",
    "TrainMode",
    "train_ppo",
    "write_run_config_json",
    "_apply_initial_opponent_params",
    "EVAL_ONLY_TRAINING_OPPONENT_TAGS",
    "_strip_eval_only_opponents_from_training_pool",
    "_normalize_train_mode",
    "normalize_and_validate_training_config",
    "set_global_seed",
    "_agents_suffix",
    "_default_run_tag_for_mode",
    "_ensure_run_tag_has_agent_suffix",
    "_resolve_initial_opponent_and_phase",
    "_resolve_metrics_csv_paths",
    "_rotate_fresh_run_telemetry",
    "_clamp_runtime_config_for_team_size",
    "_ensure_cuda_or_fallback",
]


def _resolve_2v2_checkpoint(filename: str) -> Optional[str]:  # noqa: F811 — kept for legacy callers
    """Find ``checkpoints/2v2/<filename>`` whether cwd is repo root or ``AICTFProject``."""
    cwd = os.getcwd()
    candidates = (
        os.path.join(_PARENT_DIR, "checkpoints", "2v2", filename),
        os.path.join(cwd, "checkpoints", "2v2", filename),
        os.path.join(cwd, "AICTFProject", "checkpoints", "2v2", filename),
        os.path.join(os.path.dirname(_PARENT_DIR), "AICTFProject", "checkpoints", "2v2", filename),
    )
    for raw in candidates:
        path = os.path.normpath(raw)
        if os.path.isfile(path):
            return path
    return None


# Default ``python rl/train_ppo.py`` recipe when ``--preset`` is omitted: plan-faithful
# latent with sparse persistence and entropy. Pass ``--preset none`` to skip.
DEFAULT_CLI_TRAINING_PRESET = "plan_faithful_latent_persist_entropy"

# _apply_training_preset is kept here (not moved) so that overrides.py can import
# it lazily without a circular dependency via the rl.presets chain.
def _apply_training_preset(cfg: PPOConfig, preset: str) -> PPOConfig:
    """Apply named high-level presets for repeatable training recipes."""
    from rl.presets import apply_preset

    return apply_preset(cfg, preset)


def train_ppo(cfg: Optional[PPOConfig] = None) -> None:
    """Run the default local PPO/MAPPO training path.

    Delegates to :func:`rl.training.orchestrator.orchestrate_training_run`.
    The full implementation now lives there; this function is kept for
    backward compatibility with existing ``train_ppo(cfg)`` call sites.
    """
    from rl.training.orchestrator import orchestrate_training_run

    orchestrate_training_run(cfg)


def run_verify_4v4(num_episodes: int = 10) -> None:
    """Run random-action verification episodes at 4v4."""
    import numpy as np
    set_global_seed(42)
    cfg = GPUFieldConfig(n_envs=1, n_agents_per_team=4, max_decision_steps=400, device="cpu", seed=42)
    env = GPUCTFVecEnv(cfg)
    try:
        for ep in range(num_episodes):
            env.reset()
            done = False
            steps = 0
            while not done and steps < 800:
                env.step_async(np.asarray(env.action_space.sample(), dtype=np.int64)[None, :])
                _, _, done_arr, _ = env.step_wait()
                done = bool(done_arr[0])
                steps += 1
            print(f"[Verify-4v4] episode {ep + 1}/{num_episodes} steps={steps} done={done}")
    finally:
        env.close()


def run_test_vec_schema() -> None:
    """Verify GPU core observation and global-state schemas."""
    import numpy as np
    cfg = GPUFieldConfig(n_envs=1, n_agents_per_team=2, device="cpu", seed=42)
    env = GPUCTFVecEnv(cfg)
    try:
        obs = env.reset()
        vec = obs["vec"]
        state = env.state()
        assert vec.dtype == np.float32, f"vec.dtype {vec.dtype}, expected float32"
        assert vec.ndim == 3 and vec.shape[2] == VEC_OBS_DIM, (
            f"vec.shape {vec.shape}, expected (B,N,{VEC_OBS_DIM})"
        )
        assert np.all(np.isfinite(vec)), "vec has non-finite values"
        assert state.shape == (1, GLOBAL_STATE_DIM), f"state.shape {state.shape}"
        print("[test-vec-schema] obs vec and global state schemas OK.")
    finally:
        env.close()


if __name__ == "__main__":
    from rl.training.cli import main

    main()
