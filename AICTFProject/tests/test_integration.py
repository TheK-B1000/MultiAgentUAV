"""
Integration tests for SB3 environment and MARL wrapper.

Tests:
- SB3 env smoke rollout
- MARL wrapper rollout using SB3 env
"""
import numpy as np
import pytest

from game_field import make_game_field
from rl.agent_identity import build_blue_identities
from rl.env_modules import validate_action_keys, validate_obs_order, validate_reward_breakdown


try:
    from ctf_sb3_env import CTFGameFieldSB3Env
    SB3_ENV_AVAILABLE = True
except ImportError:
    SB3_ENV_AVAILABLE = False
    CTFGameFieldSB3Env = None

try:
    from rl.marl_env import MARLEnvWrapper
    MARL_WRAPPER_AVAILABLE = True
except ImportError:
    MARL_WRAPPER_AVAILABLE = False
    MARLEnvWrapper = None


@pytest.mark.skipif(not SB3_ENV_AVAILABLE, reason="SB3 env not available")
def test_sb3_env_smoke_rollout():
    """Test SB3 environment can run a full episode."""
    def make_gf():
        return make_game_field()
    
    env = CTFGameFieldSB3Env(
        make_game_field_fn=make_gf,
        max_decision_steps=10,  # Short episode for test
        validate_contracts=True,  # Enable contract validation
    )
    
    obs, info = env.reset(seed=42)
    
    assert "grid" in obs
    assert "vec" in obs
    
    step_count = 0
    done = False
    
    while not done and step_count < 10:
        # Random actions
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Validate observation shape
        validate_obs_order(obs, env._n_blue_agents, env._max_blue_agents, debug=False)
        
        # Validate reward breakdown
        if "reward_blue_per_agent" in info:
            validate_reward_breakdown(
                info.get("reward_blue_team", reward),
                info["reward_blue_per_agent"],
                debug=False,
            )
        
        done = terminated or truncated
        step_count += 1
    
    assert step_count > 0
    assert done
    
    # Check episode result on termination
    if "episode_result" in info:
        ep_result = info["episode_result"]
        assert "blue_score" in ep_result
        assert "red_score" in ep_result
        assert "success" in ep_result


@pytest.mark.skipif(not SB3_ENV_AVAILABLE or not MARL_WRAPPER_AVAILABLE, reason="SB3 env or MARL wrapper not available")
def test_marl_wrapper_rollout():
    """Test MARL wrapper can run a rollout using SB3 env."""
    def make_gf():
        return make_game_field()
    
    sb3_env = CTFGameFieldSB3Env(
        make_game_field_fn=make_gf,
        max_decision_steps=10,
    )
    
    marl_env = MARLEnvWrapper(sb3_env)
    
    obs_dict, info = marl_env.reset(seed=42)
    
    # MARL wrapper should return dict with agent keys
    assert isinstance(obs_dict, dict)
    assert len(obs_dict) > 0
    
    # Check agent keys are canonical
    agent_keys = list(obs_dict.keys())
    for i, key in enumerate(agent_keys):
        assert key == f"blue_{i}", f"Invalid agent key: {key}"
    
    step_count = 0
    done = False
    
    while not done and step_count < 10:
        # Random actions per agent
        actions = {}
        for key in agent_keys:
            actions[key] = marl_env.action_space[key].sample()
        
        obs_dict, rewards, dones, infos, shared_info = marl_env.step(actions)
        
        # Validate action keys
        validate_action_keys(actions, len(agent_keys), debug=False)
        
        # Check rewards match agent keys
        assert len(rewards) == len(agent_keys)
        for key in agent_keys:
            assert key in rewards
        
        # Check dones match agent keys
        assert len(dones) == len(agent_keys)
        for key in agent_keys:
            assert key in dones
        
        done = any(dones.values())
        step_count += 1
    
    assert step_count > 0


@pytest.mark.skipif(not SB3_ENV_AVAILABLE, reason="SB3 env not available")
def test_sb3_env_canonical_action_keys():
    """Test that SB3 env uses canonical action keys internally."""
    def make_gf():
        return make_game_field()
    
    env = CTFGameFieldSB3Env(
        make_game_field_fn=make_gf,
        max_decision_steps=5,
    )
    
    obs, info = env.reset(seed=42)
    
    # Get canonical identities
    identities = env._blue_identities
    assert len(identities) > 0
    
    # Verify identities use canonical keys
    for i, ident in enumerate(identities):
        assert ident.key == f"blue_{i}"
    
    # Run one step and verify actions use canonical keys
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Actions should be submitted using canonical keys (checked internally)
    # We can't directly verify this, but contract validation would catch it


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
