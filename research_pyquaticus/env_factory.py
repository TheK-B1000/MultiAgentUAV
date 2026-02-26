from __future__ import annotations

import copy
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import gymnasium as gym
from pyquaticus.config import config_dict_std
from pyquaticus.envs.pyquaticus import PyQuaticusEnv
from pyquaticus.envs.rllib_pettingzoo_wrapper import ParallelPettingZooWrapper
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.policy.policy import Policy
from ray.tune.registry import register_env


def _flatten_obs(ob: Any) -> np.ndarray:
    """Flatten dict/array observation to 1D float32 with deterministic key order for RLlib batching."""
    if isinstance(ob, dict):
        keys = sorted(ob.keys(), key=lambda k: (str(k), repr(k)))
        return np.concatenate([_flatten_obs(ob[k]) for k in keys]).astype(np.float32)
    if isinstance(ob, np.ndarray):
        if ob.dtype == object and ob.size > 0 and isinstance(ob.flat[0], dict):
            return np.concatenate([_flatten_obs(ob.flat[i]) for i in range(ob.size)]).astype(np.float32)
        if ob.dtype == object or ob.ndim == 0:
            return np.array(ob, dtype=np.float32).flatten()
        return ob.astype(np.float32).flatten()
    if isinstance(ob, (list, tuple)):
        if ob and isinstance(ob[0], dict):
            return np.concatenate([_flatten_obs(x) for x in ob]).astype(np.float32)
        return np.array(ob, dtype=np.float32).flatten()
    return np.array([float(ob)], dtype=np.float32)


class FlattenDictObsWrapper:
    """Wraps a parallel env that returns dict observations; flattens to 1D arrays so RLlib can batch."""

    def __init__(self, env: Any):
        self._env = env
        self.agents = getattr(env, "agents", None) or getattr(env, "possible_agents", [])
        self.possible_agents = getattr(env, "possible_agents", None) or self.agents
        self._flat_dim: Optional[int] = None
        self._obs_space: Optional[gym.spaces.Dict] = None
        self._act_space = None

    def _ensure_obs_space(self, sample_obs: Dict[str, Any]) -> None:
        if self._flat_dim is not None:
            return
        flat = _flatten_obs(sample_obs)
        self._flat_dim = int(flat.size)
        self._obs_space = gym.spaces.Dict({
            aid: gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self._flat_dim,), dtype=np.float32,
            )
            for aid in self.agents
        })
        self._act_space = gym.spaces.Dict({
            aid: self._env.action_space(aid) for aid in self.agents
        })

    def observation_space(self, agent_id: str) -> gym.Space:
        if self._obs_space is None:
            obs, _ = self._env.reset()
            self._ensure_obs_space(obs[self.agents[0]])
        return self._obs_space[agent_id]

    def action_space(self, agent_id: str) -> gym.Space:
        if self._act_space is None:
            self._act_space = gym.spaces.Dict({
                aid: self._env.action_space(aid) for aid in self.agents
            })
        return self._act_space[agent_id]

    @property
    def observation_spaces(self) -> Dict[str, gym.Space]:
        if self._obs_space is None:
            obs, _ = self._env.reset()
            self._ensure_obs_space(obs[self.agents[0]])
        return {aid: self._obs_space[aid] for aid in self.agents}

    @property
    def action_spaces(self) -> Dict[str, gym.Space]:
        if self._act_space is None:
            self._act_space = gym.spaces.Dict({
                aid: self._env.action_space(aid) for aid in self.agents
            })
        return self._act_space

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        obs, infos = self._env.reset(seed=seed, options=options)
        for aid in list(obs.keys()):
            if self._flat_dim is None:
                self._ensure_obs_space(obs[aid])
            obs[aid] = _flatten_obs(obs[aid])
        return obs, infos

    def step(self, action_dict: Dict[str, Any]):
        obs, rewards, terminated, truncated, infos = self._env.step(action_dict)
        for aid in list(obs.keys()):
            obs[aid] = _flatten_obs(obs[aid])
        return obs, rewards, terminated, truncated, infos

    def close(self):
        return getattr(self._env, "close", lambda: None)()

    def render(self):
        return getattr(self._env, "render", lambda: None)()


def get_learning_agent_ids(team_size: int) -> List[str]:
    # Pyquaticus defaults to agent_0..agent_(2*team_size-1); first half is blue team.
    return [f"agent_{i}" for i in range(int(team_size))]


class RandomOpponentPolicy(Policy):
    """Simple fixed opponent policy used for scripted/species placeholders."""

    def compute_actions(
        self,
        obs_batch,
        state_batches,
        prev_action_batch=None,
        prev_reward_batch=None,
        info_batch=None,
        episodes=None,
        **kwargs,
    ):
        # Mirrors upstream examples where -1 lets env resolve a default/random action.
        return [-1 for _ in obs_batch], [], {}

    def get_weights(self):
        return {}

    def set_weights(self, weights):
        return None

    def learn_on_batch(self, samples):
        return {}


def _build_env_config(config: Dict[str, Any]) -> Dict[str, Any]:
    cfg = copy.deepcopy(config_dict_std)
    cfg.update(copy.deepcopy(config.get("pyquaticus", {}).get("env_config", {})))
    return cfg


def make_parallel_env(config: Dict[str, Any], *, seed: int | None = None):
    pyq = config.get("pyquaticus", {})
    team_size = int(pyq.get("team_size", 2))
    render_mode = pyq.get("render_mode", None)
    action_space = str(pyq.get("action_space", "continuous")).lower()
    env_cfg = _build_env_config(config)
    env = PyQuaticusEnv(
        config_dict=env_cfg,
        render_mode=render_mode,
        reward_config=None,
        team_size=team_size,
        action_space=action_space,
    )
    if seed is not None:
        env.reset(seed=int(seed))
    return env


def register_pyquaticus_env(config: Dict[str, Any]) -> str:
    env_name = str(config.get("pyquaticus", {}).get("env_name", "pyquaticus_research"))

    def _creator(_):
        inner = make_parallel_env(config)
        flat = FlattenDictObsWrapper(inner)
        return ParallelPettingZooWrapper(flat)

    register_env(env_name, _creator)
    return env_name


def build_multiagent_specs(
    config: Dict[str, Any],
    *,
    league_controller: Optional[Any] = None,
) -> Tuple[Dict[str, Any], Callable[..., str], List[str], type]:
    """
    Build RLlib multi-agent specs with learner + opponent pools.
    Red opponent policy is selected per episode via episode.user_data['opponent_policy'].
    """
    sample_env = ParallelPettingZooWrapper(FlattenDictObsWrapper(make_parallel_env(config)))
    team_size = int(config.get("pyquaticus", {}).get("team_size", 2))
    learning_agent_ids = set(get_learning_agent_ids(team_size))
    sample_env.reset()  # so FlattenDictObsWrapper sets flat obs shape
    obs_space = sample_env.observation_space["agent_0"]
    act_space = sample_env.action_space["agent_0"]
    sample_env.close()

    policies = {
        "learner_policy": (None, obs_space, act_space, {}),
        # Scripted phase opponents
        "scripted_op1_policy": (RandomOpponentPolicy, obs_space, act_space, {"style": "op1"}),
        "scripted_op2_policy": (RandomOpponentPolicy, obs_space, act_space, {"style": "op2"}),
        "scripted_op3_policy": (RandomOpponentPolicy, obs_space, act_space, {"style": "op3"}),
        # Species opponents
        "species_rusher_policy": (RandomOpponentPolicy, obs_space, act_space, {"style": "rusher"}),
        "species_camper_policy": (RandomOpponentPolicy, obs_space, act_space, {"style": "camper"}),
        "species_balanced_policy": (RandomOpponentPolicy, obs_space, act_space, {"style": "balanced"}),
        # Snapshot/self-play: same architecture as learner so we can copy weights (frozen via policies_to_train).
        "snapshot_policy": (None, obs_space, act_space, {}),
    }

    def policy_mapping_fn(agent_id, episode=None, worker=None, **kwargs):
        if agent_id in learning_agent_ids:
            return "learner_policy"
        # Red side policy chosen per episode by callbacks/controller.
        if episode is not None:
            pol = episode.user_data.get("opponent_policy", None)
            if isinstance(pol, str) and pol in policies:
                return pol
        # Safe fallback: hardest scripted baseline.
        return "scripted_op3_policy"

    class LeagueCallbacks(DefaultCallbacks):
        def on_episode_created(self, *, worker, base_env, policies, env_index, episode, **kwargs):
            if league_controller is None:
                episode.user_data["opponent_policy"] = "scripted_op3_policy"
                episode.user_data["opponent_key"] = "SCRIPTED:OP3"
                return
            opp = league_controller.select_opponent()
            episode.user_data["opponent_policy"] = opp.policy_name
            episode.user_data["opponent_key"] = opp.key

        def on_episode_end(self, *, worker, base_env, policies, episode, **kwargs):
            blue_score = 0.0
            red_score = 0.0
            try:
                if hasattr(episode, "get_agents"):
                    agents = list(episode.get_agents())
                else:
                    agents = []
                if agents:
                    info = episode.last_info_for(agents[0]) or {}
                    gs = info.get("global_state", {}) if isinstance(info, dict) else {}
                    blue_score = float(gs.get("blue_team_score", 0.0))
                    red_score = float(gs.get("red_team_score", 0.0))
            except Exception:
                pass
            # Expose in result for driver to print (phase, mode, score, opp)
            episode.custom_metrics["blue_score"] = blue_score
            episode.custom_metrics["red_score"] = red_score
            episode.custom_metrics["opponent_key"] = str(episode.user_data.get("opponent_key", "?"))
            if league_controller is not None:
                league_controller.record_episode(
                    opponent_key=str(episode.user_data.get("opponent_key", "SCRIPTED:OP3")),
                    blue_score=blue_score,
                    red_score=red_score,
                )

    return policies, policy_mapping_fn, ["learner_policy"], LeagueCallbacks
