from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from types import SimpleNamespace
from typing import Any, Optional

import numpy as np

from rl.curriculum import phase_from_tag


@dataclass
class TrainingOpponentPool:
    """Training-time scripted opponent pool sampler (OPPONENT_POOL mode)."""

    enabled: bool
    tags: list[str]
    weights: Optional[list[float]]
    rng: np.random.Generator

    @classmethod
    def from_hparams(cls, cfg: Any, hparams: Any) -> TrainingOpponentPool:
        tags = list(hparams.opponent_pool_tags)
        weights = list(hparams.opponent_pool_weights) if hparams.opponent_pool_weights else None
        if weights is not None and len(weights) != len(tags):
            raise ValueError(
                f"opponent_pool_weights length {len(weights)} does not match "
                f"opponent_pool_tags length {len(tags)}."
            )
        if bool(hparams.opponent_randomize_training) and not tags:
            raise ValueError(
                "Opponent pool training (mode=OPPONENT_POOL or opponent_randomize) requires a non-empty "
                "opponent_pool (e.g. OP1–OP3, OP5–OP7; OP4 optional with --allow-op4-in-training-pool)."
            )
        return cls(
            enabled=bool(hparams.opponent_randomize_training),
            tags=tags,
            weights=weights,
            rng=np.random.default_rng(int(getattr(cfg, "seed", 0)) + 901),
        )

    def attach_before_reset_hook(self, env: Any, trainer: Any) -> None:
        if not self.enabled:
            return
        env._before_reset_indices_hook = partial(_hook_sample_training_opponent_before_reset, trainer)


def _resolve_training_opponent_pool(trainer: Any) -> Any:
    pool = getattr(trainer, "opponent_pool", None)
    if pool is not None:
        return pool
    return SimpleNamespace(
        enabled=bool(getattr(trainer, "_opponent_randomize_training", False)),
        tags=list(getattr(trainer, "_opponent_pool_tags", []) or []),
        weights=getattr(trainer, "_opponent_pool_weights", None),
        rng=getattr(trainer, "_rng_opponent", None),
    )


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
    if trainer.curriculum is not None:
        return
    pool = _resolve_training_opponent_pool(trainer)
    if not pool.enabled:
        return
    weights = pool.weights
    for env_i, done_i in enumerate(done):
        if not bool(done_i):
            continue
        if weights is not None:
            tag = str(pool.rng.choice(pool.tags, p=weights)).upper()
        else:
            tag = str(pool.rng.choice(pool.tags)).upper()
        phase_s = phase_from_tag(tag)
        try:
            trainer.env.env_method("set_next_opponent", "SCRIPTED", tag, indices=[env_i])
            trainer.env.env_method("set_phase", phase_s, indices=[env_i])
        except Exception:
            trainer.env.env_method("set_next_opponent", "SCRIPTED", tag)
            trainer.env.env_method("set_phase", phase_s)
