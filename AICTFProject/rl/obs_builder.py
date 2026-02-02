"""
Canonical team observation builder: grid, vec, mask, agent_mask.

Single implementation for ctf_sb3_env, red_opponents (snapshot), and viewer
so training, evaluation, and visualization use the same obs contract.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from game_field import (
    CNN_COLS,
    CNN_ROWS,
    NUM_CNN_CHANNELS,
)


@dataclass
class TeamObs:
    """Structured team observation (grid, vec, optional mask, optional agent_mask)."""
    grid: np.ndarray
    vec: np.ndarray
    mask: Optional[np.ndarray] = None
    agent_mask: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {"grid": self.grid, "vec": self.vec}
        if self.mask is not None:
            out["mask"] = self.mask
        if self.agent_mask is not None:
            out["agent_mask"] = self.agent_mask
        return out


def _empty_cnn() -> np.ndarray:
    return np.zeros((NUM_CNN_CHANNELS, CNN_ROWS, CNN_COLS), dtype=np.float32)


def _coerce_vec(vec: np.ndarray, size: int) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    if v.size == size:
        return v
    if v.size < size:
        return np.pad(v, (0, size - v.size), mode="constant")
    return v[:size]


def build_agent_obs(
    gf: Any,
    agent: Any,
    *,
    include_context: bool = False,
    context_value: Optional[float] = None,
    debug_locality: bool = False,
    include_mask: bool = False,
    vec_size_base: int = 12,
    n_macros: int = 5,
    n_targets: Optional[int] = None,
    vec_append_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> Dict[str, np.ndarray]:
    """
    Single-agent observation dict for IPPO/MAPPO: grid, vec, optional mask, optional context.

    Args:
        gf: GameField (or ViewerGameField).
        agent: One agent instance (not agent_id).
        include_context: If True and context_value is not None, add "context" key.
        context_value: Scalar to put in obs["context"] when include_context True.
        debug_locality: If True, set gf.obs_debug_validate_locality during vec build.
        include_mask: If True, include "mask" (macro+target for this agent).
        vec_size_base, n_macros, n_targets, vec_append_fn: Same as build_team_obs.

    Returns:
        Dict with "grid", "vec", optionally "mask", optionally "context".
    """
    nt = int(n_targets if n_targets is not None else getattr(gf, "num_macro_targets", 8) or 8)
    out: Dict[str, np.ndarray] = {}
    if agent is None:
        out["grid"] = _empty_cnn()
        base_vec = np.zeros((vec_size_base,), dtype=np.float32)
        vec = vec_append_fn(base_vec) if vec_append_fn else base_vec
        out["vec"] = np.asarray(vec, dtype=np.float32).reshape(-1)
        if include_mask:
            out["mask"] = np.concatenate([
                np.zeros((n_macros,), dtype=np.float32),
                np.zeros((nt,), dtype=np.float32),
            ], axis=0)
        if include_context and context_value is not None:
            out["context"] = np.array([float(context_value)], dtype=np.float32)
        return out

    out["grid"] = np.asarray(gf.build_observation(agent), dtype=np.float32)
    old_debug = getattr(gf, "obs_debug_validate_locality", False)
    if debug_locality:
        gf.obs_debug_validate_locality = True
    try:
        if hasattr(gf, "build_continuous_features"):
            base_vec = _coerce_vec(gf.build_continuous_features(agent), vec_size_base)
        else:
            base_vec = np.zeros((vec_size_base,), dtype=np.float32)
    finally:
        if debug_locality:
            gf.obs_debug_validate_locality = old_debug
    vec = vec_append_fn(base_vec) if vec_append_fn else base_vec
    out["vec"] = np.asarray(vec, dtype=np.float32).reshape(-1)
    if include_mask:
        mm = np.asarray(gf.get_macro_mask(agent), dtype=np.bool_).reshape(-1)
        if mm.shape != (n_macros,) or (not np.any(mm)):
            mm = np.ones((n_macros,), dtype=np.bool_)
        tm = np.asarray(gf.get_target_mask(agent), dtype=np.bool_).reshape(-1)
        if tm.shape != (nt,) or (not np.any(tm)):
            tm = np.ones((nt,), dtype=np.bool_)
        out["mask"] = np.concatenate([mm.astype(np.float32), tm.astype(np.float32)], axis=0)
    if include_context and context_value is not None:
        out["context"] = np.array([float(context_value)], dtype=np.float32)
    return out


def build_team_obs(
    gf: Any,
    agents: List[Any],
    *,
    max_agents: int,
    include_mask: bool,
    include_context: bool = False,
    context_value: Optional[float] = None,
    debug_locality: bool = False,
    tokenized: bool = True,
    vec_size_base: int = 12,
    n_macros: int = 5,
    n_targets: Optional[int] = None,
    role_macro_mask_fn: Optional[Callable[[int, np.ndarray], np.ndarray]] = None,
    vec_append_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> Dict[str, np.ndarray]:
    """
    Build team observation dict: grid, vec, optional mask, optional agent_mask, optional context.

    Args:
        gf: GameField (or ViewerGameField) with build_observation(agent),
            build_continuous_features(agent), get_macro_mask(agent), get_target_mask(agent).
        agents: List of agents (length max_agents); use None for padding slots.
        max_agents: Number of agent slots (2 = legacy, 4/8 = tokenized).
        include_mask: If True, include "mask" in output (macro+target per agent).
        include_context: If True and context_value is not None, add "context" key.
        context_value: Scalar (e.g. opponent id) for obs["context"] when include_context True.
        debug_locality: If True, set gf.obs_debug_validate_locality during vec builds.
        tokenized: If True, output (max_agents, C, H, W) grid and agent_mask; else concatenate.
        vec_size_base: Base vec size per agent (12 from GameField.build_continuous_features).
        n_macros, n_targets, role_macro_mask_fn, vec_append_fn: As before.

    Returns:
        Dict with "grid", "vec", optionally "mask", "agent_mask" (when tokenized), "context".
    """
    nt = int(n_targets if n_targets is not None else getattr(gf, "num_macro_targets", 8) or 8)
    n_slots = max(2, int(max_agents))
    while len(agents) < n_slots:
        agents = list(agents) + [None]

    cnn_list: List[np.ndarray] = []
    vec_list: List[np.ndarray] = []
    mask_list: List[np.ndarray] = []

    for idx, a in enumerate(agents[:n_slots]):
        if a is None:
            cnn_list.append(_empty_cnn())
            base_vec = np.zeros((vec_size_base,), dtype=np.float32)
            vec = vec_append_fn(base_vec) if vec_append_fn else base_vec
            vec = np.asarray(vec, dtype=np.float32).reshape(-1)
            vec_list.append(vec)
            if include_mask:
                mm = np.zeros((n_macros,), dtype=np.float32)
                tm = np.zeros((nt,), dtype=np.float32)
                mask_list.append(np.concatenate([mm, tm], axis=0))
            continue

        cnn = np.asarray(gf.build_observation(a), dtype=np.float32)
        cnn_list.append(cnn)

        if hasattr(gf, "build_continuous_features"):
            base_vec = _coerce_vec(gf.build_continuous_features(a), vec_size_base)
        else:
            base_vec = np.zeros((vec_size_base,), dtype=np.float32)
        vec = vec_append_fn(base_vec) if vec_append_fn else base_vec
        vec = np.asarray(vec, dtype=np.float32).reshape(-1)
        vec_list.append(vec)

        if include_mask:
            mm = np.asarray(gf.get_macro_mask(a), dtype=np.bool_).reshape(-1)
            if role_macro_mask_fn is not None:
                mm = role_macro_mask_fn(idx, mm)
            if mm.shape != (n_macros,) or (not np.any(mm)):
                mm = np.ones((n_macros,), dtype=np.bool_)
            tm = np.asarray(gf.get_target_mask(a), dtype=np.bool_).reshape(-1)
            if tm.shape != (nt,) or (not np.any(tm)):
                tm = np.ones((nt,), dtype=np.bool_)
            mask_list.append(np.concatenate([mm.astype(np.float32), tm.astype(np.float32)], axis=0))

    if tokenized:
        grid = np.stack(cnn_list, axis=0).astype(np.float32)
        vec = np.stack(vec_list, axis=0).astype(np.float32)
        agent_mask = np.zeros((n_slots,), dtype=np.float32)
        n_live = sum(1 for a in agents[:n_slots] if a is not None)
        agent_mask[:n_live] = 1.0
        out = {"grid": grid, "vec": vec, "agent_mask": agent_mask}
    else:
        grid = np.concatenate(cnn_list, axis=0).astype(np.float32)
        vec = np.concatenate(vec_list, axis=0).astype(np.float32)
        out = {"grid": grid, "vec": vec}

    if include_mask:
        out["mask"] = np.concatenate(mask_list, axis=0).astype(np.float32)
    if include_context and context_value is not None:
        out["context"] = np.array([float(context_value)], dtype=np.float32)

    return out


__all__ = ["TeamObs", "build_team_obs", "build_agent_obs"]
