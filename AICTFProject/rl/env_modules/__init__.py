"""
Environment modules for CTFGameFieldSB3Env.

Modular components:
- env_config: Configuration plumbing (dynamics, disturbance, robotics, sensor, physics, phase)
- env_opponent: Opponent switching (scripted, species, snapshot)
- env_rewards: Reward routing, stall penalties, episode metrics
- env_obs: Observation building with canonical ordering
- env_actions: Action sanitization, masks, noise
- step_result: Unified step-result/reward structure
"""
from rl.env_modules.env_config import EnvConfigManager
from rl.env_modules.env_opponent import EnvOpponentManager, MAX_OPPONENT_CONTEXT_IDS
from rl.env_modules.env_rewards import EnvRewardManager
from rl.env_modules.env_obs import EnvObsBuilder
from rl.env_modules.env_actions import EnvActionManager
from rl.env_modules.step_result import (
    StepReward,
    StepInfo,
    EpisodeResult,
    StepResult,
)
from rl.env_modules.contract_validators import (
    validate_agent_keys,
    validate_reward_breakdown,
    validate_obs_order,
    validate_action_keys,
    validate_dropped_reward_events_policy,
)

__all__ = [
    "EnvConfigManager",
    "EnvOpponentManager",
    "MAX_OPPONENT_CONTEXT_IDS",
    "EnvRewardManager",
    "EnvObsBuilder",
    "EnvActionManager",
    "StepReward",
    "StepInfo",
    "EpisodeResult",
    "StepResult",
    "validate_agent_keys",
    "validate_reward_breakdown",
    "validate_obs_order",
    "validate_action_keys",
    "validate_dropped_reward_events_policy",
]
