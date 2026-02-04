"""
Unit tests for environment modules.

Tests:
- env_obs: ordering + tokenized/legacy shape checks
- env_actions: sanitize + mask enforcement correctness
- env_rewards: per-agent sum == team, dropped counts behave
- env_opponent: context id stability + snapshot path resolution
"""
import os
import tempfile
from typing import Any, Dict, List

import numpy as np
import pytest

from rl.env_modules import (
    EnvActionManager,
    EnvObsBuilder,
    EnvOpponentManager,
    EnvRewardManager,
)
from rl.agent_identity import AgentIdentity, build_blue_identities


class MockGameField:
    """Mock GameField for testing."""
    
    def __init__(self):
        self.blue_agents = []
        self.n_macros = 5
        self.num_macro_targets = 8
    
    def build_observation(self, agent: Any) -> np.ndarray:
        return np.zeros((7, 20, 20), dtype=np.float32)
    
    def build_continuous_features(self, agent: Any) -> np.ndarray:
        return np.zeros((12,), dtype=np.float32)
    
    def get_macro_mask(self, agent: Any) -> np.ndarray:
        return np.ones((5,), dtype=np.bool_)
    
    def get_target_mask(self, agent: Any) -> np.ndarray:
        return np.ones((8,), dtype=np.bool_)
    
    def get_macro_target(self, idx: int) -> tuple:
        return (float(idx), float(idx))


class MockAgent:
    """Mock agent for testing."""
    
    def __init__(self, agent_id: int = 0):
        self.agent_id = agent_id
        self.unique_id = f"agent_{agent_id}"
        self.slot_id = f"slot_{agent_id}"
        self.side = "blue"
        self.x = 0.0
        self.y = 0.0
        self.isEnabled = lambda: True


class MockGameManager:
    """Mock GameManager for testing."""
    
    def __init__(self, reward_events: List[tuple] = None):
        self.reward_events = reward_events or []
    
    def pop_reward_events(self):
        events = self.reward_events
        self.reward_events = []
        return events


# ============================================================================
# Tests for EnvObsBuilder
# ============================================================================

def test_env_obs_legacy_shape():
    """Test legacy observation shape (2 agents, concatenated)."""
    builder = EnvObsBuilder(
        use_obs_builder=False,  # Use legacy path
        include_mask_in_obs=False,
        include_opponent_context=False,
        include_high_level_mode=False,
        base_vec_per_agent=12,
    )
    
    gf = MockGameField()
    agents = [MockAgent(0), MockAgent(1)]
    identities = build_blue_identities(agents, max_agents=2)
    
    obs = builder.build_observation(
        gf,
        identities,
        max_blue_agents=2,
        n_blue_agents=2,
        n_macros=5,
        n_targets=8,
    )
    
    assert "grid" in obs
    assert "vec" in obs
    assert obs["grid"].shape == (14, 20, 20)  # 2 * 7 channels
    assert len(obs["vec"].shape) == 1  # Concatenated
    assert obs["vec"].shape[0] == 24  # 2 * 12


def test_env_obs_tokenized_shape():
    """Test tokenized observation shape (max_agents > 2)."""
    builder = EnvObsBuilder(
        use_obs_builder=False,  # Use legacy path
        include_mask_in_obs=False,
        include_opponent_context=False,
        include_high_level_mode=False,
        base_vec_per_agent=12,
    )
    
    gf = MockGameField()
    agents = [MockAgent(0), MockAgent(1)]
    identities = build_blue_identities(agents, max_agents=4)
    
    obs = builder.build_observation(
        gf,
        identities,
        max_blue_agents=4,
        n_blue_agents=2,
        n_macros=5,
        n_targets=8,
    )
    
    assert "grid" in obs
    assert "vec" in obs
    assert "agent_mask" in obs
    assert obs["grid"].shape == (4, 7, 20, 20)  # (max_agents, C, H, W)
    assert obs["vec"].shape == (4, 12)  # (max_agents, vec_dim)
    assert obs["agent_mask"].shape == (4,)
    assert obs["agent_mask"][:2].sum() == 2.0  # First 2 agents enabled
    assert obs["agent_mask"][2:].sum() == 0.0  # Last 2 agents disabled


def test_env_obs_canonical_ordering():
    """Test observation ordering matches canonical agent identities."""
    builder = EnvObsBuilder(
        use_obs_builder=False,
        include_mask_in_obs=False,
        include_opponent_context=False,
        include_high_level_mode=False,
        base_vec_per_agent=12,
    )
    
    gf = MockGameField()
    agents = [MockAgent(0), MockAgent(1)]
    identities = build_blue_identities(agents, max_agents=2)
    
    # Verify identities are in canonical order
    assert identities[0].key == "blue_0"
    assert identities[1].key == "blue_1"
    
    obs = builder.build_observation(
        gf,
        identities,
        max_blue_agents=2,
        n_blue_agents=2,
        n_macros=5,
        n_targets=8,
    )
    
    # Observation should match identity order
    assert obs["grid"].shape[0] == 14  # 2 agents * 7 channels
    assert obs["vec"].shape[0] == 24  # 2 agents * 12 vec_dim


# ============================================================================
# Tests for EnvActionManager
# ============================================================================

def test_env_actions_sanitize_mask_enforcement():
    """Test action sanitization enforces masks correctly."""
    manager = EnvActionManager(
        enforce_masks=True,
        action_flip_prob=0.0,
        n_macros=5,
        n_targets=8,
    )
    
    gf = MockGameField()
    
    # Mock agent with restricted macro mask
    class RestrictedAgent(MockAgent):
        def __init__(self):
            super().__init__()
            self.restricted_macros = [0, 1]  # Only GO_TO and one other
    
    agent = RestrictedAgent()
    gf.blue_agents = [agent]
    
    # Override get_macro_mask to return restricted mask
    def restricted_macro_mask(a):
        mask = np.zeros((5,), dtype=np.bool_)
        mask[0] = True  # GO_TO
        mask[1] = True  # One other macro
        return mask
    
    gf.get_macro_mask = restricted_macro_mask
    
    intended = [(2, 0), (3, 0)]  # Invalid macros (not in mask)
    executed, _, _, _ = manager.apply_noise_and_sanitize(
        intended,
        n_blue_agents=1,
        game_field=gf,
    )
    
    # Should fallback to GO_TO (macro 0) for invalid macros
    assert executed[0][0] == 0  # Sanitized to GO_TO


def test_env_actions_noise():
    """Test action execution noise."""
    manager = EnvActionManager(
        enforce_masks=False,  # Disable masks for noise test
        action_flip_prob=1.0,  # Always flip
        n_macros=5,
        n_targets=8,
        seed=42,
    )
    
    gf = MockGameField()
    gf.blue_agents = [MockAgent(0)]
    
    intended = [(0, 0), (0, 0)]  # GO_TO, target 0
    executed, flip_count, macro_flip, target_flip = manager.apply_noise_and_sanitize(
        intended,
        n_blue_agents=1,
        game_field=gf,
    )
    
    # With flip_prob=1.0, actions should be flipped
    assert flip_count > 0
    assert macro_flip > 0 or target_flip > 0


def test_env_actions_macro_usage_tracking():
    """Test macro usage tracking."""
    manager = EnvActionManager(
        enforce_masks=False,
        action_flip_prob=0.0,
        n_macros=5,
        n_targets=8,
    )
    
    manager.reset_episode(n_agents=2)
    
    gf = MockGameField()
    gf.blue_agents = [MockAgent(0), MockAgent(1)]
    gf.macro_order = [None] * 5  # Mock macro order
    
    intended = [(1, 0), (2, 0)]  # Different macros per agent
    executed, _, _, _ = manager.apply_noise_and_sanitize(
        intended,
        n_blue_agents=2,
        game_field=gf,
    )
    
    counts = manager.get_episode_macro_counts()
    assert len(counts) == 2
    assert counts[0][1] == 1  # Agent 0 used macro 1
    assert counts[1][2] == 1  # Agent 1 used macro 2


# ============================================================================
# Tests for EnvRewardManager
# ============================================================================

def test_env_rewards_per_agent_sum_equals_team():
    """Test that per-agent reward sum equals team total."""
    manager = EnvRewardManager()
    manager.reset_episode(n_agents=2)
    
    gm = MockGameManager(reward_events=[
        (0.0, "blue_0", 1.0),
        (0.0, "blue_1", 2.0),
        (0.0, "blue_0", 0.5),
    ])
    
    reward_id_map = {"blue_0": 0, "blue_1": 1}
    
    team_total, per_agent, dropped = manager.consume_reward_events(
        gm,
        reward_id_map,
        n_slots=2,
    )
    
    assert team_total == 3.5
    assert per_agent[0] == 1.5  # blue_0: 1.0 + 0.5
    assert per_agent[1] == 2.0  # blue_1: 2.0
    assert sum(per_agent) == team_total
    assert dropped == 0


def test_env_rewards_dropped_events():
    """Test dropped reward events tracking."""
    manager = EnvRewardManager()
    manager.reset_episode(n_agents=2)
    
    gm = MockGameManager(reward_events=[
        (0.0, "blue_0", 1.0),
        (0.0, "unknown_agent", 2.0),  # Unknown agent ID
        (0.0, "blue_1", 0.5),
    ])
    
    reward_id_map = {"blue_0": 0, "blue_1": 1}
    
    team_total, per_agent, dropped = manager.consume_reward_events(
        gm,
        reward_id_map,
        n_slots=2,
    )
    
    assert dropped == 1  # One event dropped
    assert team_total == 1.5  # Only routed events
    assert manager.get_dropped_events_step() == 1
    assert manager.get_dropped_events_episode() == 1


def test_env_rewards_stall_penalty():
    """Test stall penalty application."""
    manager = EnvRewardManager()
    manager.reset_episode(n_agents=2)
    
    class StationaryAgent(MockAgent):
        def __init__(self, x=0.0, y=0.0):
            super().__init__()
            self.x = x
            self.y = y
            self.float_pos = (x, y)
    
    gf = MockGameField()
    gf.blue_agents = [
        StationaryAgent(0.0, 0.0),  # Stationary (will trigger stall)
        StationaryAgent(1.0, 1.0),  # Moved (no stall)
    ]
    
    # Apply stall penalty multiple times to trigger threshold
    for _ in range(5):
        stall_total, stall_per_agent = manager.apply_stall_penalty(gf, n_blue_agents=2)
    
    # First agent should have stall penalty (didn't move)
    assert stall_total < 0
    assert stall_per_agent[0] < 0  # Stall penalty for agent 0


# ============================================================================
# Tests for EnvOpponentManager
# ============================================================================

def test_env_opponent_context_id_scripted():
    """Test opponent context ID for scripted opponents."""
    manager = EnvOpponentManager(default_kind="SCRIPTED", default_key="OP1")
    
    manager.set_opponent_scripted("OP1", None)
    assert manager._opponent_context_id() == 0
    
    manager.set_opponent_scripted("OP2", None)
    assert manager._opponent_context_id() == 1
    
    manager.set_opponent_scripted("OP3", None)
    assert manager._opponent_context_id() == 2


def test_env_opponent_context_id_species():
    """Test opponent context ID for species opponents."""
    manager = EnvOpponentManager(default_kind="SPECIES", default_key="BALANCED")
    
    manager.set_opponent_species("BALANCED", None)
    assert manager._opponent_context_id() == 3
    
    manager.set_opponent_species("RUSHER", None)
    assert manager._opponent_context_id() == 4
    
    manager.set_opponent_species("CAMPER", None)
    assert manager._opponent_context_id() == 5


def test_env_opponent_context_id_snapshot():
    """Test opponent context ID for snapshot opponents."""
    manager = EnvOpponentManager()
    
    # Create temporary snapshot file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.zip', delete=False) as f:
        snapshot_path = f.name
    
    try:
        manager.set_opponent_snapshot(snapshot_path, None)
        context_id = manager._opponent_context_id()
        
        # Should be in range [6, MAX_OPPONENT_CONTEXT_IDS-1]
        assert 6 <= context_id < 256
        
        # Same snapshot should give same context ID
        manager2 = EnvOpponentManager()
        manager2.set_opponent_snapshot(snapshot_path, None)
        assert manager2._opponent_context_id() == context_id
    finally:
        os.unlink(snapshot_path)


def test_env_opponent_snapshot_path_resolution():
    """Test snapshot path resolution (with/without .zip, case handling)."""
    manager = EnvOpponentManager()
    
    # Create temporary snapshot file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.zip', delete=False) as f:
        snapshot_path = f.name
    
    try:
        # Test with .zip extension
        manager.set_opponent_snapshot(snapshot_path, None)
        assert manager.get_opponent_snapshot_path() == snapshot_path
        
        # Test without .zip extension (should resolve)
        base_path = snapshot_path[:-4]  # Remove .zip
        manager.set_opponent_snapshot(base_path, None)
        assert manager.get_opponent_snapshot_path() == snapshot_path
        
        # Test with non-existent path (should fallback to OP3)
        manager.set_opponent_snapshot("/nonexistent/path.zip", None)
        assert manager.get_opponent_kind() == "scripted"
        assert manager.get_opponent_scripted_tag() == "OP3"
    finally:
        os.unlink(snapshot_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
