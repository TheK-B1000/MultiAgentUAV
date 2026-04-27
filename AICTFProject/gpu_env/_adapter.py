"""GameField-compatible adapter over BatchedCTFCore."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from agents import AgentHandle
from rl.global_state import GLOBAL_STATE_DIM

from ._config import GPUFieldConfig
from ._core_class import BatchedCTFCore


class _FakeGM:
    """Minimal GameManager stand-in for MAPPO/QMIX (scores, game_over, set_phase, terminal bonus)."""

    def __init__(self, core: BatchedCTFCore):
        assert core.B == 1, "FakeGM only supports single env"
        self._core = core
        self._phase = "OP3"

    @property
    def blue_score(self) -> int:
        return int(self._core.blue_score[0].item())

    @property
    def red_score(self) -> int:
        return int(self._core.red_score[0].item())

    @property
    def game_over(self) -> bool:
        return bool(self._core.done[0].item())

    def set_phase(self, phase: str) -> None:
        self._phase = str(phase).upper()
        self._core.set_phase(phase)

    def pop_reward_events(self):
        """No per-event routing in GPU env; return empty."""
        return iter(())

    def terminal_outcome_bonus(self, blue_score: int, red_score: int) -> float:
        if blue_score > red_score:
            return 1.0
        if blue_score < red_score:
            return -1.0
        return -0.5

class GPUEnvAdapter:
    """
    GameField-like wrapper around BatchedCTFCore with B=1 for MAPPO/QMIX training.
    Provides: reset_default, build_observation(agent), get_macro_target, get_macro_mask,
    get_target_mask, blue_agents, getGameManager(), step(actions_flat), get_global_state, etc.
    """

    def __init__(self, cfg: Optional[GPUFieldConfig] = None):
        cfg = cfg or GPUFieldConfig(n_envs=1)
        cfg.n_envs = 1
        self._cfg = cfg
        self._core = BatchedCTFCore(cfg)
        self.n_macros = int(cfg.n_macros)
        self.num_macro_targets = int(cfg.n_targets)
        self.agents_per_team = int(cfg.max_blue_agents)
        self._gm = _FakeGM(self._core)
        self._blue_agents: List[AgentHandle] = []
        self._refresh_agents()

    def _refresh_agents(self) -> None:
        def alive(i: int) -> bool:
            return bool(self._core.blue_alive[0, i].item())

        self._blue_agents = [
            AgentHandle(i, "blue", alive_getter=alive)
            for i in range(self.agents_per_team)
        ]

    @property
    def blue_agents(self) -> List[AgentHandle]:
        self._refresh_agents()
        return self._blue_agents

    def set_external_control(self, side: str, value: bool) -> None:
        pass

    def set_red_opponent(self, tag: str) -> None:
        self._core.set_phase(tag.upper())

    use_internal_policies: bool = True

    def reset_default(self) -> None:
        self._core.reset_all()
        self._refresh_agents()

    def getGameManager(self) -> _FakeGM:
        return self._gm

    def get_obs(self) -> Dict[str, np.ndarray]:
        return self._core.get_obs()

    def build_observation(self, agent: AgentHandle) -> np.ndarray:
        """Single-agent grid obs (C, H, W) for policy.act()."""
        ot = self._core.get_obs_tensors()
        grid = ot["grid"]
        i = getattr(agent, "agent_id", 0)
        return grid[0, i].detach().cpu().numpy().astype(np.float32)

    def get_macro_target(self, index: int) -> Tuple[float, float]:
        """Return (x, y) for macro target index (for policies that need it)."""
        idx = int(index) % self._core._macro_targets.size(0)
        x = float(self._core._macro_targets[idx, 0].item())
        y = float(self._core._macro_targets[idx, 1].item())
        return (x, y)

    def _mask_for_agent(self, agent: AgentHandle, macro_only: bool) -> np.ndarray:
        m = self._core._build_action_mask()
        i = getattr(agent, "agent_id", 0)
        n_m = self._cfg.n_macros
        n_t = self._cfg.n_targets
        base = i * (n_m + n_t)
        if macro_only:
            return m[0, base : base + n_m].detach().cpu().numpy().astype(np.bool_)
        return m[0, base + n_m : base + n_m + n_t].detach().cpu().numpy().astype(np.bool_)

    def get_macro_mask(self, agent: AgentHandle) -> np.ndarray:
        return self._mask_for_agent(agent, macro_only=True)

    def get_target_mask(self, agent: AgentHandle) -> np.ndarray:
        return self._mask_for_agent(agent, macro_only=False)

    def get_global_state_dim(self) -> int:
        return int(GLOBAL_STATE_DIM)

    def get_global_state(self) -> np.ndarray:
        return self._core.get_global_state()[0]

    def state(self) -> np.ndarray:
        return self.get_global_state()

    def step(self, actions_flat: np.ndarray):
        """
        Single env step. actions_flat: (n_agents*2,) or (n_agents, 2) with [macro, target] per agent.
        Returns (obs_dict, reward, terminated, truncated, info).
        """
        a = np.asarray(actions_flat, dtype=np.int64)
        if a.ndim == 2:
            a = a.reshape(-1)
        if a.size < self.agents_per_team * 2:
            pad = np.zeros(self.agents_per_team * 2 - a.size, dtype=np.int64)
            a = np.concatenate([a, pad])
        t = torch.from_numpy(a).to(self._core.device).unsqueeze(0)
        obs, reward, term, trunc, infos = self._core.step(t)
        done = np.logical_or(term, trunc)
        if done.any():
            self._core.reset_indices(torch.from_numpy(done).to(self._core.device))
            obs = self._core.get_obs()
        self._refresh_agents()
        info = infos[0] if isinstance(infos, (list, tuple)) else infos
        return obs, float(reward[0]), bool(term[0]), bool(trunc[0]), info

    @property
    def macro_order(self) -> List[Any]:
        """Placeholder for QMIX; GPU uses fixed n_macros."""
        return list(range(self.n_macros))

__all__ = ["GPUEnvAdapter", "_FakeGM"]
