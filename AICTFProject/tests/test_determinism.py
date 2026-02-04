"""
Determinism tests: run same seed twice, compare episode summaries.

Ensures that with the same seed and deterministic settings, we get identical results.
"""
import numpy as np
import pytest

from game_field import make_game_field
from rl.determinism import SeedAuthority
from rl.eval_mode import EvalModeConfig, EvalModeManager


try:
    from ctf_sb3_env import CTFGameFieldSB3Env
    SB3_ENV_AVAILABLE = True
except ImportError:
    SB3_ENV_AVAILABLE = False
    CTFGameFieldSB3Env = None

try:
    from rl.episode_result import EpisodeSummary, parse_episode_result
    EPISODE_RESULT_AVAILABLE = True
except ImportError:
    EPISODE_RESULT_AVAILABLE = False
    EpisodeSummary = None
    parse_episode_result = None


@pytest.mark.skipif(not SB3_ENV_AVAILABLE, reason="SB3 env not available")
def test_deterministic_episode_same_seed():
    """Test that same seed produces identical episode results."""
    def make_gf():
        return make_game_field()
    
    seed = 42
    max_steps = 20
    
    # Run episode 1
    seed_auth_1 = SeedAuthority(base_seed=seed, deterministic=True)
    seed_auth_1.set_all_seeds()
    
    env_1 = CTFGameFieldSB3Env(
        make_game_field_fn=make_gf,
        max_decision_steps=max_steps,
        seed=seed,
        action_flip_prob=0.0,  # Disable action noise for determinism
    )
    
    obs_1, _ = env_1.reset(seed=seed)
    episode_result_1 = None
    
    # Use seeded random for deterministic actions
    rng_1 = np.random.RandomState(seed)
    actions_1 = []
    for _ in range(max_steps):
        # Generate deterministic actions using seeded random
        if hasattr(env_1.action_space, 'nvec'):
            action = np.array([rng_1.randint(0, n) for n in env_1.action_space.nvec])
        else:
            action = env_1.action_space.sample()
        actions_1.append(action.copy() if hasattr(action, 'copy') else action)
        obs_1, reward_1, terminated_1, truncated_1, info_1 = env_1.step(action)
        if terminated_1 or truncated_1:
            episode_result_1 = info_1.get("episode_result")
            break
    
    # Run episode 2 with same seed
    seed_auth_2 = SeedAuthority(base_seed=seed, deterministic=True)
    seed_auth_2.set_all_seeds()
    
    env_2 = CTFGameFieldSB3Env(
        make_game_field_fn=make_gf,
        max_decision_steps=max_steps,
        seed=seed,
        action_flip_prob=0.0,  # Disable action noise for determinism
    )
    
    obs_2, _ = env_2.reset(seed=seed)
    episode_result_2 = None
    
    # Use same seeded random for deterministic actions (reset to same state)
    rng_2 = np.random.RandomState(seed)
    for i, action_template in enumerate(actions_1):
        # Use same actions as episode 1
        if hasattr(env_2.action_space, 'nvec'):
            action = np.array([rng_2.randint(0, n) for n in env_2.action_space.nvec])
        else:
            action = action_template
        obs_2, reward_2, terminated_2, truncated_2, info_2 = env_2.step(action)
        if terminated_2 or truncated_2:
            episode_result_2 = info_2.get("episode_result")
            break
    
    # Compare episode results
    if episode_result_1 is not None and episode_result_2 is not None:
        # Key metrics should match
        assert episode_result_1["blue_score"] == episode_result_2["blue_score"]
        assert episode_result_1["red_score"] == episode_result_2["red_score"]
        assert episode_result_1["success"] == episode_result_2["success"]
        assert episode_result_1["decision_steps"] == episode_result_2["decision_steps"]
        
        # Collision metrics should match
        assert episode_result_1["collision_events_per_episode"] == episode_result_2["collision_events_per_episode"]
        assert episode_result_1["collision_free_episode"] == episode_result_2["collision_free_episode"]


@pytest.mark.skipif(not SB3_ENV_AVAILABLE, reason="SB3 env not available")
def test_eval_mode_determinism():
    """Test that evaluation mode produces deterministic results."""
    def make_gf():
        return make_game_field()
    
    seed = 123
    max_steps = 15
    
    eval_config = EvalModeConfig(
        enabled=True,
        allow_action_noise=False,  # Disable action noise
        allow_opponent_params_random=False,  # Disable opponent randomness
    )
    eval_manager = EvalModeManager(config=eval_config)
    
    # Run episode 1
    seed_auth_1 = SeedAuthority(base_seed=seed, deterministic=True)
    seed_auth_1.set_all_seeds()
    
    env_1 = CTFGameFieldSB3Env(
        make_game_field_fn=make_gf,
        max_decision_steps=max_steps,
        seed=seed,
        action_flip_prob=0.0 if eval_manager.should_disable_action_noise() else 0.1,
    )
    
    obs_1, _ = env_1.reset(seed=seed)
    rewards_1 = []
    
    # Use seeded random for deterministic actions
    rng_1 = np.random.RandomState(seed)
    actions_1 = []
    for _ in range(max_steps):
        if hasattr(env_1.action_space, 'nvec'):
            action = np.array([rng_1.randint(0, n) for n in env_1.action_space.nvec])
        else:
            action = env_1.action_space.sample()
        actions_1.append(action.copy() if hasattr(action, 'copy') else action)
        obs_1, reward_1, terminated_1, truncated_1, info_1 = env_1.step(action)
        rewards_1.append(reward_1)
        if terminated_1 or truncated_1:
            break
    
    # Run episode 2 with same seed and eval mode
    seed_auth_2 = SeedAuthority(base_seed=seed, deterministic=True)
    seed_auth_2.set_all_seeds()
    
    env_2 = CTFGameFieldSB3Env(
        make_game_field_fn=make_gf,
        max_decision_steps=max_steps,
        seed=seed,
        action_flip_prob=0.0 if eval_manager.should_disable_action_noise() else 0.1,
    )
    
    obs_2, _ = env_2.reset(seed=seed)
    rewards_2 = []
    
    # Use same seeded random for deterministic actions (reset to same state)
    rng_2 = np.random.RandomState(seed)
    for i, action_template in enumerate(actions_1):
        if hasattr(env_2.action_space, 'nvec'):
            action = np.array([rng_2.randint(0, n) for n in env_2.action_space.nvec])
        else:
            action = action_template
        obs_2, reward_2, terminated_2, truncated_2, info_2 = env_2.step(action)
        rewards_2.append(reward_2)
        if terminated_2 or truncated_2:
            break
    
    # Rewards should match (within floating-point tolerance)
    assert len(rewards_1) == len(rewards_2)
    for r1, r2 in zip(rewards_1, rewards_2):
        assert abs(r1 - r2) < 1e-6, f"Reward mismatch: {r1} != {r2}"


@pytest.mark.skipif(not SB3_ENV_AVAILABLE or not EPISODE_RESULT_AVAILABLE, reason="SB3 env or episode_result not available")
def test_episode_summary_consistency():
    """Test that episode summaries are consistent across runs."""
    def make_gf():
        return make_game_field()
    
    seed = 456
    max_steps = 10
    
    env = CTFGameFieldSB3Env(
        make_game_field_fn=make_gf,
        max_decision_steps=max_steps,
        seed=seed,
        action_flip_prob=0.0,
    )
    
    obs, _ = env.reset(seed=seed)
    
    for _ in range(max_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    
    # Parse episode result
    if "episode_result" in info:
        ep_result = info["episode_result"]
        # parse_episode_result expects info dict, not just episode_result
        summary = parse_episode_result(info)
        
        # Verify summary fields are consistent
        assert summary.blue_score == ep_result["blue_score"]
        assert summary.red_score == ep_result["red_score"]
        assert summary.success == ep_result["success"]
        assert summary.collisions_per_episode == ep_result.get("collisions_per_episode", ep_result.get("collision_events_per_episode", 0))
        assert summary.collision_free_episode == ep_result["collision_free_episode"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
