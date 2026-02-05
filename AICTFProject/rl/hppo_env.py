from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Callable, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np

from ctf_sb3_env import CTFGameFieldSB3Env


class HighLevelMode(IntEnum):
    DEFEND = 0
    ATTACK = 1


@dataclass
class HighLevelObsConfig:
    include_scores: bool = True
    include_flags: bool = True
    include_time: bool = True
    include_phase_onehot: bool = True
    include_opp_tier_onehot: bool = True


class HighLevelModeScheduler(gym.Wrapper):
    """
    Wrapper for low-level PPO training that resamples high-level mode.
    """

    def __init__(
        self,
        env: CTFGameFieldSB3Env,
        *,
        resample_steps: int = 0,
        attack_prob: float = 0.5,
    ) -> None:
        super().__init__(env)
        self.resample_steps = int(resample_steps)
        self.attack_prob = float(attack_prob)
        self._steps_since_sample = 0

    def _sample_mode(self) -> int:
        return int(HighLevelMode.ATTACK if np.random.rand() < self.attack_prob else HighLevelMode.DEFEND)

    def reset(self, **kwargs):
        self._steps_since_sample = 0
        mode = self._sample_mode()
        self.env.set_high_level_mode(mode)
        obs, info = self.env.reset(**kwargs)
        info["high_level_mode"] = int(mode)
        return obs, info

    def step(self, action):
        if self.resample_steps > 0 and self._steps_since_sample >= self.resample_steps:
            mode = self._sample_mode()
            self.env.set_high_level_mode(mode)
            self._steps_since_sample = 0
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._steps_since_sample += 1
        info["high_level_mode"] = int(self.env._high_level_mode)
        return obs, reward, terminated, truncated, info

    def set_phase(self, phase: str) -> None:
        if hasattr(self.env, "set_phase"):
            self.env.set_phase(phase)

    def set_next_opponent(self, kind: str, key: str) -> None:
        if hasattr(self.env, "set_next_opponent"):
            self.env.set_next_opponent(kind, key)


class CTFHighLevelEnv(gym.Env):
    """
    High-level PPO environment that chooses DEFEND vs ATTACK.
    Executes a fixed number of low-level steps per high-level action.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        make_env_fn: Callable[[], CTFGameFieldSB3Env],
        low_level_model,
        horizon_steps: int = 8,
        obs_config: Optional[HighLevelObsConfig] = None,
    ) -> None:
        super().__init__()
        self.make_env_fn = make_env_fn
        self.low_level_model = low_level_model
        self.horizon_steps = int(horizon_steps)
        self.obs_config = obs_config or HighLevelObsConfig()

        self.base_env: Optional[CTFGameFieldSB3Env] = None
        self.action_space = gym.spaces.Discrete(2)

        obs_dim = 12
        if self.obs_config.include_scores:
            obs_dim += 3
        if self.obs_config.include_flags:
            obs_dim += 2
        if self.obs_config.include_time:
            obs_dim += 1
        if self.obs_config.include_phase_onehot:
            obs_dim += 3
        if self.obs_config.include_opp_tier_onehot:
            obs_dim += 5
        self.observation_space = gym.spaces.Box(low=-2.0, high=2.0, shape=(obs_dim,), dtype=np.float32)

        self._pending_phase: Optional[str] = None
        self._pending_opponent: Optional[Tuple[str, str]] = None

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.base_env = self.make_env_fn()
        if self._pending_phase:
            try:
                self.base_env.set_phase(self._pending_phase)
            except Exception:
                pass
        if self._pending_opponent:
            try:
                kind, key = self._pending_opponent
                self.base_env.set_next_opponent(kind, key)
            except Exception:
                pass
        obs, info = self.base_env.reset(seed=seed, options=options)
        return self._build_high_level_obs(), info

    def step(self, action: int):
        assert self.base_env is not None, "Call reset() first"
        mode = int(action)
        self.base_env.set_high_level_mode(mode)

        total_reward = 0.0
        terminated = False
        truncated = False
        info: Dict[str, object] = {}

        for _ in range(max(1, self.horizon_steps)):
            if self.low_level_model is not None:
                obs = self.base_env._get_obs()
                ll_action, _ = self.low_level_model.predict(obs, deterministic=False)
            else:
                ll_action = self.base_env.action_space.sample()
            obs, reward, terminated, truncated, step_info = self.base_env.step(ll_action)
            total_reward += float(reward)
            info.update(step_info)
            if terminated or truncated:
                break

        info["high_level_mode"] = int(mode)
        return self._build_high_level_obs(), float(total_reward), terminated, truncated, info

    def set_phase(self, phase: str) -> None:
        self._pending_phase = str(phase).upper().strip()
        if self.base_env is not None:
            try:
                self.base_env.set_phase(self._pending_phase)
            except Exception:
                pass

    def set_next_opponent(self, kind: str, key: str) -> None:
        self._pending_opponent = (str(kind).upper(), str(key).upper())
        if self.base_env is not None:
            try:
                self.base_env.set_next_opponent(self._pending_opponent[0], self._pending_opponent[1])
            except Exception:
                pass

    def _build_high_level_obs(self) -> np.ndarray:
        assert self.base_env is not None
        gf = self.base_env.gf
        gm = gf.manager if gf is not None else None

        if gf is None or gm is None:
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        vecs = []
        for agent in getattr(gf, "blue_agents", [])[:2]:
            if agent is None:
                continue
            try:
                if hasattr(agent, "isEnabled") and (not agent.isEnabled()):
                    continue
            except Exception:
                pass
            if hasattr(gf, "build_continuous_features"):
                vecs.append(np.asarray(gf.build_continuous_features(agent), dtype=np.float32))

        if vecs:
            avg_vec = np.mean(np.stack(vecs, axis=0), axis=0)
        else:
            avg_vec = np.zeros((12,), dtype=np.float32)

        extras = []
        if self.obs_config.include_scores:
            score_limit = max(1.0, float(getattr(gm, "score_limit", 3)))
            blue_score = float(getattr(gm, "blue_score", 0)) / score_limit
            red_score = float(getattr(gm, "red_score", 0)) / score_limit
            score_diff = (float(getattr(gm, "blue_score", 0)) - float(getattr(gm, "red_score", 0))) / score_limit
            extras.extend([blue_score, red_score, score_diff])
        if self.obs_config.include_flags:
            extras.extend([
                1.0 if bool(getattr(gm, "blue_flag_taken", False)) else 0.0,
                1.0 if bool(getattr(gm, "red_flag_taken", False)) else 0.0,
            ])
        if self.obs_config.include_time:
            max_time = max(1.0, float(getattr(gm, "max_time", 300.0)))
            current_time = float(getattr(gm, "current_time", max_time))
            extras.append(max(0.0, min(1.0, current_time / max_time)))

        if self.obs_config.include_phase_onehot:
            phase = str(getattr(self.base_env, "_phase_name", getattr(gm, "phase_name", "OP1"))).upper()
            phase_vec = np.zeros((3,), dtype=np.float32)
            phase_map = {"OP1": 0, "OP2": 1, "OP3": 2}
            idx = phase_map.get(phase, None)
            if idx is not None:
                phase_vec[int(idx)] = 1.0
            extras.extend(phase_vec.tolist())

        if self.obs_config.include_opp_tier_onehot:
            opp_vec = np.zeros((5,), dtype=np.float32)
            kind = str(getattr(self.base_env, "_opponent_kind", "scripted")).upper()
            tag = str(getattr(self.base_env, "_opponent_scripted_tag", "OP3")).upper()
            if kind == "SCRIPTED":
                opp_map = {"OP1": 0, "OP2": 1, "OP3_EASY": 2, "OP3": 3, "OP3_HARD": 4}
                idx = opp_map.get(tag, None)
                if idx is not None:
                    opp_vec[int(idx)] = 1.0
            extras.extend(opp_vec.tolist())

        if extras:
            out = np.concatenate([avg_vec, np.asarray(extras, dtype=np.float32)], axis=0)
        else:
            out = avg_vec
        return out.astype(np.float32)
