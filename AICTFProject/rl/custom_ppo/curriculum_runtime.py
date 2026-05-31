from __future__ import annotations

from typing import Any, Optional
import numpy as np

from rl.curriculum import phase_from_tag


def _set_curriculum_opponent(trainer: Any, phase: str, env_index: Optional[int] = None) -> None:
    phase_s = str(phase).upper()
    indices = None if env_index is None else [int(env_index)]
    try:
        trainer.env.env_method("set_next_opponent", "SCRIPTED", phase_s, indices=indices)
        trainer.env.env_method("set_phase", phase_s, indices=indices)
    except Exception:
        if indices is not None:
            trainer.env.env_method("set_next_opponent", "SCRIPTED", phase_s)
            trainer.env.env_method("set_phase", phase_s)


def _update_curriculum_after_episode(
    trainer: Any,
    *,
    info: dict[str, Any],
    blue_score: int,
    red_score: int,
    env_index: Optional[int],
) -> None:
    if trainer.curriculum is None:
        return
    episode_phase = str(info.get("phase", trainer.curriculum.phase)).upper()
    old_phase = str(trainer.curriculum.phase).upper()
    win_value = 1.0 if int(blue_score) > int(red_score) else 0.0
    if episode_phase != old_phase:
        trainer.curriculum.record_result(episode_phase, win_value)
        _set_curriculum_opponent(trainer, old_phase, env_index)
        return
    trainer.curriculum.phase_episode_count += 1
    trainer.curriculum.record_result(old_phase, win_value)
    advanced = trainer.curriculum.advance_if_ready(win_by=int(blue_score) - int(red_score))
    new_phase = str(trainer.curriculum.phase).upper()
    if advanced:
        wr = 100.0 * float(trainer.curriculum.phase_winrate(old_phase))
        print(
            f"[PPO] Curriculum advanced: {old_phase} -> {new_phase} "
            f"after episode {trainer.episode_stats.episodes_completed} (gate_wr={wr:.1f}%)."
        )
    _set_curriculum_opponent(trainer, new_phase, env_index)


def _hook_sample_training_opponent_before_reset(trainer: Any, done: np.ndarray, infos: list) -> None:
    """Sample the *next* episode's scripted opponent per finished sub-env (GPUCTFVecEnv hook)."""
    if trainer.curriculum is not None or not trainer._opponent_randomize_training:
        return
    for env_i, done_i in enumerate(done):
        if not bool(done_i):
            continue
        tag = str(trainer._rng_opponent.choice(trainer._opponent_pool_tags)).upper()
        phase_s = phase_from_tag(tag)
        try:
            trainer.env.env_method("set_next_opponent", "SCRIPTED", tag, indices=[env_i])
            trainer.env.env_method("set_phase", phase_s, indices=[env_i])
        except Exception:
            trainer.env.env_method("set_next_opponent", "SCRIPTED", tag)
            trainer.env.env_method("set_phase", phase_s)
