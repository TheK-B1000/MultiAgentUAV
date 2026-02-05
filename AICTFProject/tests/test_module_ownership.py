"""
Structural checks: scan for forbidden patterns in non-owner modules.

Tests that enforce module ownership by detecting duplicate logic.
"""
import ast
import os
import re
from pathlib import Path
from typing import List, Set

import pytest


# Forbidden patterns: (pattern, owner_module, description)
FORBIDDEN_PATTERNS = [
    # Agent identity logic
    (r"build_blue_identities\s*\(", "rl/agent_identity.py", "build_blue_identities()"),
    (r"build_reward_id_map\s*\(", "rl/agent_identity.py", "build_reward_id_map()"),
    (r"route_reward_events\s*\(", "rl/agent_identity.py", "route_reward_events()"),
    (r"AgentIdentity\s*\(", "rl/agent_identity.py", "AgentIdentity()"),
    
    # Reward routing (outside env_rewards module)
    (r"pop_reward_events\s*\(\s*\)", "rl/env_modules/env_rewards.py", "pop_reward_events() consumption"),
    
    # Action sanitization (outside env_actions module)
    (r"get_macro_mask\s*\(", "rl/env_modules/env_actions.py", "get_macro_mask() for sanitization"),
    (r"get_target_mask\s*\(", "rl/env_modules/env_actions.py", "get_target_mask() for sanitization"),
    
    # Observation building (outside env_obs/obs_builder)
    (r"build_team_obs\s*\(", "rl/env_modules/env_obs.py or rl/obs_builder.py", "build_team_obs()"),
    
    # Opponent switching (outside env_opponent module)
    (r"make_species_wrapper\s*\(", "rl/env_modules/env_opponent.py", "make_species_wrapper()"),
    (r"make_snapshot_wrapper\s*\(", "rl/env_modules/env_opponent.py", "make_snapshot_wrapper()"),
    
    # Config plumbing (outside env_config module)
    (r"set_dynamics_config\s*\(", "rl/env_modules/env_config.py", "set_dynamics_config() forwarding"),
    (r"set_disturbance_config\s*\(", "rl/env_modules/env_config.py", "set_disturbance_config() forwarding"),
    (r"set_robotics_constraints\s*\(", "rl/env_modules/env_config.py", "set_robotics_constraints() forwarding"),
    (r"set_sensor_config\s*\(", "rl/env_modules/env_config.py", "set_sensor_config() forwarding"),
]


def get_python_files(exclude_dirs: Set[str] = None) -> List[Path]:
    """Get all Python files in AICTFProject, excluding test files and specified directories."""
    if exclude_dirs is None:
        exclude_dirs = {"__pycache__", ".git", "node_modules", ".venv", "venv"}
    
    project_root = Path("AICTFProject")
    python_files = []
    
    for root, dirs, files in os.walk(project_root):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        
        for file in files:
            if file.endswith(".py"):
                python_files.append(Path(root) / file)
    
    return python_files


def is_owner_file(file_path: Path, owner_module: str) -> bool:
    """Check if file is the owner module for a pattern."""
    # Normalize paths
    file_str = str(file_path).replace("\\", "/")
    owner_str = owner_module.replace("\\", "/")
    
    # Check if file path ends with owner module path
    return file_str.endswith(owner_str) or owner_str in file_str


def find_forbidden_patterns(file_path: Path, pattern: str, owner_module: str) -> List[tuple]:
    """Find forbidden patterns in a file."""
    violations = []
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            lines = content.split("\n")
        
        # Skip if this is the owner file
        if is_owner_file(file_path, owner_module):
            return violations
        
        # Search for pattern
        for line_num, line in enumerate(lines, 1):
            if re.search(pattern, line):
                # Check if it's an import (allowed) or actual usage (forbidden)
                if "import" not in line and "from" not in line:
                    violations.append((line_num, line.strip()))
    except Exception as e:
        # Skip files that can't be read
        pass
    
    return violations


def test_no_duplicate_agent_identity_logic():
    """Test that agent identity logic is only in rl/agent_identity.py."""
    violations = []
    
    for file_path in get_python_files():
        for pattern, owner_module, description in FORBIDDEN_PATTERNS[:4]:  # First 4 are agent identity
            found = find_forbidden_patterns(file_path, pattern, owner_module)
            if found:
                violations.extend([
                    (str(file_path), line_num, line, description, owner_module)
                    for line_num, line in found
                ])
    
    if violations:
        msg = "Found forbidden agent identity logic outside owner module:\n"
        for file_path, line_num, line, desc, owner in violations:
            msg += f"  {file_path}:{line_num} - {desc} (owner: {owner})\n"
            msg += f"    {line}\n"
        pytest.fail(msg)


def test_no_duplicate_reward_routing():
    """Test that reward routing logic is only in env_rewards module."""
    violations = []
    
    for file_path in get_python_files():
        pattern, owner_module, description = FORBIDDEN_PATTERNS[4]  # pop_reward_events
        found = find_forbidden_patterns(file_path, pattern, owner_module)
        if found:
            # Allow in env_rewards.py and agent_identity.py (route_reward_events uses it)
            if "env_rewards.py" not in str(file_path) and "agent_identity.py" not in str(file_path):
                violations.extend([
                    (str(file_path), line_num, line, description, owner_module)
                    for line_num, line in found
                ])
    
    if violations:
        msg = "Found forbidden reward routing logic outside owner module:\n"
        for file_path, line_num, line, desc, owner in violations:
            msg += f"  {file_path}:{line_num} - {desc} (owner: {owner})\n"
            msg += f"    {line}\n"
        pytest.fail(msg)


def test_no_duplicate_opponent_switching():
    """Test that opponent switching logic is only in env_opponent module."""
    violations = []
    
    for file_path in get_python_files():
        for pattern, owner_module, description in FORBIDDEN_PATTERNS[10:12]:  # Opponent patterns
            found = find_forbidden_patterns(file_path, pattern, owner_module)
            if found:
                violations.extend([
                    (str(file_path), line_num, line, description, owner_module)
                    for line_num, line in found
                ])
    
    if violations:
        msg = "Found forbidden opponent switching logic outside owner module:\n"
        for file_path, line_num, line, desc, owner in violations:
            msg += f"  {file_path}:{line_num} - {desc} (owner: {owner})\n"
            msg += f"    {line}\n"
        pytest.fail(msg)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
