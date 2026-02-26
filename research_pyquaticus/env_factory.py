from __future__ import annotations

import copy
from typing import Any, Callable, Dict, List, Tuple

from pyquaticus import pyquaticus_v0
from pyquaticus.config import config_dict_std
from pyquaticus.envs.rllib_pettingzoo_wrapper import ParallelPettingZooWrapper
from ray.rllib.policy.policy import Policy
from ray.tune.registry import register_env


def get_learning_agent_ids(team_size: int) -> List[str]:
    # Pyquaticus defaults to agent_0..agent_(2*team_size-1); first half is blue team.
    return [f"agent_{i}" for i in range(int(team_size))]


class RandomOpponentPolicy(Policy):
    """Simple fixed opponent for comparability baselines."""

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
        # Let env handle random/idle semantics; -1 is used in official examples.
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
    env_cfg = _build_env_config(config)
    env = pyquaticus_v0.PyQuaticusEnv(
        config_dict=env_cfg,
        render_mode=render_mode,
        reward_config=None,
        team_size=team_size,
    )
    if seed is not None:
        env.reset(seed=int(seed))
    return env


def register_pyquaticus_env(config: Dict[str, Any]) -> str:
    env_name = str(config.get("pyquaticus", {}).get("env_name", "pyquaticus_research"))

    def _creator(_):
        return ParallelPettingZooWrapper(make_parallel_env(config))

    register_env(env_name, _creator)
    return env_name


def build_multiagent_specs(config: Dict[str, Any]) -> Tuple[Dict[str, Any], Callable[..., str], List[str]]:
    # Use one learner policy for all blue agents; random fixed policy for red.
    sample_env = ParallelPettingZooWrapper(make_parallel_env(config))
    team_size = int(config.get("pyquaticus", {}).get("team_size", 2))
    learning_agent_ids = set(get_learning_agent_ids(team_size))
    obs_space = sample_env.observation_space["agent_0"]
    act_space = sample_env.action_space["agent_0"]
    sample_env.close()

    policies = {
        "learner_policy": (None, obs_space, act_space, {}),
        "opponent_policy": (RandomOpponentPolicy, obs_space, act_space, {}),
    }

    def policy_mapping_fn(agent_id, *args, **kwargs):
        return "learner_policy" if agent_id in learning_agent_ids else "opponent_policy"

    return policies, policy_mapping_fn, ["learner_policy"]
