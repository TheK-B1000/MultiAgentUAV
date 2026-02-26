from __future__ import annotations

import copy
from typing import Any, Callable, Dict, List, Optional, Tuple

from pyquaticus.config import config_dict_std
from pyquaticus.envs.pyquaticus import PyQuaticusEnv
from pyquaticus.envs.rllib_pettingzoo_wrapper import ParallelPettingZooWrapper
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.policy.policy import Policy
from ray.tune.registry import register_env


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
        return ParallelPettingZooWrapper(make_parallel_env(config))

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
    sample_env = ParallelPettingZooWrapper(make_parallel_env(config))
    team_size = int(config.get("pyquaticus", {}).get("team_size", 2))
    learning_agent_ids = set(get_learning_agent_ids(team_size))
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
            if league_controller is None:
                return
            # Pull score from any agent info (global_state is shared).
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
            league_controller.record_episode(
                opponent_key=str(episode.user_data.get("opponent_key", "SCRIPTED:OP3")),
                blue_score=blue_score,
                red_score=red_score,
            )

    return policies, policy_mapping_fn, ["learner_policy"], LeagueCallbacks
