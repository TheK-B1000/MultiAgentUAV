from __future__ import annotations

from dataclasses import dataclass, field
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
    # Optional joint (opponent, map, weight) cells. When set, overrides tag-only sampling.
    cells: Optional[list[tuple[str, str, float]]] = None
    snapshots: list = field(default_factory=list)
    snapshot_rng: Any = None

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
        raw_cells = tuple(getattr(cfg, "training_cell_distribution", ()) or ())
        cells: Optional[list[tuple[str, str, float]]] = None
        if raw_cells:
            parsed: list[tuple[str, str, float]] = []
            for entry in raw_cells:
                if len(entry) != 3:
                    raise ValueError(
                        f"training_cell_distribution entries must be (opp, map, weight); got {entry!r}"
                    )
                parsed.append((str(entry[0]).upper(), str(entry[1]), float(entry[2])))
            total = sum(max(0.0, w) for _, _, w in parsed)
            if total <= 0:
                raise ValueError("training_cell_distribution weights must sum to > 0")
            cells = [(o, m, max(0.0, w) / total) for o, m, w in parsed]
        snapshots = [str(x) for x in (getattr(cfg, "snapshot_opponent_pool", ()) or ())]
        return cls(
            enabled=bool(hparams.opponent_randomize_training) or bool(snapshots),
            tags=tags,
            weights=weights,
            rng=np.random.default_rng(int(getattr(cfg, "seed", 0)) + 901),
            cells=cells,
            snapshots=snapshots,
            # SEPARATE stream (+902). Drawing snapshot picks from the scripted rng
            # would shift the scripted opponent sequence for a given seed, which is
            # exactly the compatibility break the additive design must avoid.
            snapshot_rng=np.random.default_rng(int(getattr(cfg, "seed", 0)) + 902),
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
        cells=None,
        snapshots=list(getattr(trainer, "_snapshot_opponent_pool", []) or []),
        snapshot_rng=getattr(trainer, "_rng_snapshot_opponent", None),
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


def _sample_snapshot_opponents(trainer: Any, pool: Any, snapshots: list, done: np.ndarray) -> None:
    """Fictitious Play: pick a historical checkpoint per finished sub-env.

    Uses pool.snapshot_rng, NOT pool.rng, so enabling FP cannot shift the
    scripted opponent sequence for any given seed.

    Fails CLOSED on load. gpu_env/state/snapshots.py swallows load errors and
    returns None, which would leave red unpiloted while training looked healthy
    -- a counterfeit success. A checkpoint that will not load raises here instead.
    """
    rng = getattr(pool, "snapshot_rng", None)
    if rng is None:
        raise RuntimeError("snapshot opponent pool configured without an RNG")
    for env_i, done_i in enumerate(done):
        if not bool(done_i):
            continue
        pick = str(snapshots[int(rng.integers(0, len(snapshots)))])
        try:
            core = trainer.env.get_attr("core", indices=[env_i])[0]
            if core._load_snapshot_policy(pick) is None:
                raise RuntimeError(
                    f"snapshot opponent {pick!r} loaded as None; refusing to train "
                    f"against an unpiloted red team")
        except RuntimeError:
            raise
        except Exception:
            pass  # env without a reachable core attr; the loadability guard ran pre-training
        try:
            trainer.env.env_method("set_next_opponent", "SNAPSHOT", pick, indices=[env_i])
        except Exception:
            trainer.env.env_method("set_next_opponent", "SNAPSHOT", pick)
        sel = getattr(trainer, "_fp_selected_snapshots", None)
        if sel is None:
            sel = trainer._fp_selected_snapshots = []
        sel.append({"env": int(env_i), "checkpoint": pick})


def _hook_sample_training_opponent_before_reset(trainer: Any, done: np.ndarray, infos: list) -> None:
    """Sample the *next* episode's scripted opponent per finished sub-env (GPUCTFVecEnv hook)."""
    if trainer.curriculum is not None:
        return
    pool = _resolve_training_opponent_pool(trainer)
    if not pool.enabled:
        return
    snapshots = list(getattr(pool, "snapshots", []) or [])
    if snapshots:
        _sample_snapshot_opponents(trainer, pool, snapshots, done)
        return
    cells = getattr(pool, "cells", None)
    weights = pool.weights
    for env_i, done_i in enumerate(done):
        if not bool(done_i):
            continue
        map_layout = None
        if cells:
            probs = [c[2] for c in cells]
            pick = int(pool.rng.choice(len(cells), p=probs))
            tag = str(cells[pick][0]).upper()
            map_layout = str(cells[pick][1])
        elif weights is not None:
            tag = str(pool.rng.choice(pool.tags, p=weights)).upper()
        else:
            tag = str(pool.rng.choice(pool.tags)).upper()
        phase_s = phase_from_tag(tag)
        try:
            trainer.env.env_method("set_next_opponent", "SCRIPTED", tag, indices=[env_i])
            trainer.env.env_method("set_phase", phase_s, indices=[env_i])
            if map_layout is not None:
                trainer.env.env_method("set_next_map_layout", map_layout, indices=[env_i])
        except Exception:
            trainer.env.env_method("set_next_opponent", "SCRIPTED", tag)
            trainer.env.env_method("set_phase", phase_s)
            if map_layout is not None:
                try:
                    trainer.env.env_method("set_next_map_layout", map_layout)
                except Exception:
                    pass
