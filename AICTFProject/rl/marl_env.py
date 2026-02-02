"""
MARL env wrapper: parallel multi-agent API.

Returns per-agent dicts so IPPO/MAPPO can use per-agent obs, rewards, dones, infos.
Internally steps the same underlying CTF env and splits obs/reward via ObsBuilder contract.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from game_field import NUM_CNN_CHANNELS, CNN_ROWS, CNN_COLS


def _agent_keys(n: int) -> List[str]:
    return [f"blue_{i}" for i in range(n)]


def _split_team_obs_into_per_agent(
    team_obs: Dict[str, np.ndarray],
    n_agents: int,
    vec_per_agent: int,
    n_macros: int,
    n_targets: int,
    add_agent_id_to_vec: bool = True,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Split team observation (from CTF env) into per-agent dicts.
    team_obs: from env._get_obs() — grid, vec, optional mask, optional context.
    add_agent_id_to_vec: if True, append agent_id (normalized) to vec for parameter-shared IPPO.
    """
    grid = team_obs["grid"]
    vec = team_obs["vec"]
    mask = team_obs.get("mask")
    context = team_obs.get("context")
    out: Dict[str, Dict[str, np.ndarray]] = {}
    agent_id_scale = 1.0 / max(1, n_agents - 1) if add_agent_id_to_vec else 0.0

    if grid.ndim == 4:
        # Tokenized: (M, C, H, W), (M, V)
        for i in range(n_agents):
            key = f"blue_{i}"
            v = np.asarray(vec[i], dtype=np.float32)
            if add_agent_id_to_vec:
                v = np.concatenate([v, np.array([float(i) * agent_id_scale], dtype=np.float32)], axis=0)
            out[key] = {"grid": np.asarray(grid[i], dtype=np.float32), "vec": v}
            if mask is not None:
                per = n_macros + n_targets
                out[key]["mask"] = np.asarray(mask[i * per : (i + 1) * per], dtype=np.float32)
            if context is not None:
                out[key]["context"] = np.asarray(context, dtype=np.float32)
    else:
        # Legacy: (2*C, H, W), (2*V)
        C, H, W = NUM_CNN_CHANNELS, CNN_ROWS, CNN_COLS
        for i in range(n_agents):
            key = f"blue_{i}"
            v = np.asarray(vec[i * vec_per_agent : (i + 1) * vec_per_agent], dtype=np.float32)
            if add_agent_id_to_vec:
                v = np.concatenate([v, np.array([float(i) * agent_id_scale], dtype=np.float32)], axis=0)
            out[key] = {
                "grid": np.asarray(grid[i * C : (i + 1) * C], dtype=np.float32),
                "vec": v,
            }
            if mask is not None:
                per = n_macros + n_targets
                out[key]["mask"] = np.asarray(mask[i * per : (i + 1) * per], dtype=np.float32)
            if context is not None:
                out[key]["context"] = np.asarray(context, dtype=np.float32)
    return out


def _actions_dict_to_flat(actions_dict: Dict[str, Tuple[int, int]], agent_keys: List[str]) -> np.ndarray:
    """Convert {blue_0: (macro, tgt), blue_1: (macro, tgt)} -> flat [m0, t0, m1, t1]."""
    flat = []
    for k in agent_keys:
        a = actions_dict.get(k, (0, 0))
        flat.extend([int(a[0]), int(a[1])])
    return np.array(flat, dtype=np.int64)


class MARLEnvWrapper:
    """
    Wraps a CTFGameFieldSB3Env (or similar) and exposes a parallel multi-agent API:

      obs  = { "blue_0": obs0, "blue_1": obs1 }
      rews  = { "blue_0": r0,  "blue_1": r1  }
      dones = { "blue_0": d,   "blue_1": d   }
      infos = { "blue_0": info0, "blue_1": info1 }

    Underlying env is stepped once per step(); obs and rewards are split per blue agent.
    """

    def __init__(self, env: Any, *, add_agent_id_to_vec: bool = True):
        """
        env: CTFGameFieldSB3Env (or any with .reset(), .step(action), .observation_space, .gf, ._n_blue_agents, etc.)
        add_agent_id_to_vec: if True, append normalized agent_id to each agent's vec for parameter-shared IPPO.
        """
        self._env = env
        self._add_agent_id_to_vec = bool(add_agent_id_to_vec)
        self._agent_keys: List[str] = []
        self._n_macros = 5
        self._n_targets = 8
        self._vec_per_agent = 12

    def _sync_dims(self) -> None:
        """Read n_blue_agents, n_macros, n_targets, vec_per_agent from env after reset."""
        self._agent_keys = _agent_keys(int(getattr(self._env, "_n_blue_agents", 2)))
        self._n_macros = int(getattr(self._env, "_n_macros", 5))
        self._n_targets = int(getattr(self._env, "_n_targets", 8))
        self._vec_per_agent = int(getattr(self._env, "_vec_per_agent", 12))

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Any]]:
        obs_team, info = self._env.reset(seed=seed, options=options)
        self._sync_dims()
        n = len(self._agent_keys)
        obs_dict = _split_team_obs_into_per_agent(
            obs_team, n, self._vec_per_agent, self._n_macros, self._n_targets,
            add_agent_id_to_vec=self._add_agent_id_to_vec,
        )
        return obs_dict, info

    def step(
        self, actions_dict: Dict[str, Tuple[int, int]]
    ) -> Tuple[
        Dict[str, Dict[str, np.ndarray]],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, Dict[str, Any]],
        Dict[str, Any],
    ]:
        """
        actions_dict: { "blue_0": (macro_idx, target_idx), "blue_1": (macro_idx, target_idx), ... }
        Returns: obs_dict, rews_dict, dones_dict, infos_dict, shared_info
        """
        flat_action = _actions_dict_to_flat(actions_dict, self._agent_keys)
        obs_team, reward_team, terminated, truncated, info = self._env.step(flat_action)
        done = terminated or truncated

        n = len(self._agent_keys)
        obs_dict = _split_team_obs_into_per_agent(
            obs_team, n, self._vec_per_agent, self._n_macros, self._n_targets,
            add_agent_id_to_vec=self._add_agent_id_to_vec,
        )
        rba = info.get("reward_by_agent", info.get("reward_blue_per_agent"))
        if rba is None:
            rba = [0.0] * n
        rba = np.asarray(rba, dtype=np.float32).flatten()
        if rba.size < n:
            rba = np.pad(rba, (0, n - rba.size), mode="constant", constant_values=0.0)
        else:
            rba = rba[:n]
        rews_dict = {k: float(rba[i]) for i, k in enumerate(self._agent_keys)}
        dones_dict = {k: done for k in self._agent_keys}
        infos_dict = {}
        for i, k in enumerate(self._agent_keys):
            infos_dict[k] = {
                **info,
                "reward": rews_dict[k],
                "reward_team": float(reward_team),
                "agent_id": i,
            }
        return obs_dict, rews_dict, dones_dict, infos_dict, info

    @property
    def agent_keys(self) -> List[str]:
        return list(self._agent_keys)

    @property
    def n_agents(self) -> int:
        return len(self._agent_keys)

    def close(self) -> None:
        if hasattr(self._env, "close"):
            self._env.close()


__all__ = ["MARLEnvWrapper", "_split_team_obs_into_per_agent", "_actions_dict_to_flat", "_agent_keys"]
