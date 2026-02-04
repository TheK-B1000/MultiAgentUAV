#!/usr/bin/env python
"""
Quick verification script to check that all test dependencies can be imported.
Run this before running pytest to ensure everything is set up correctly.
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("Verifying test dependencies...")
print()

errors = []

# Check pytest
try:
    import pytest
    print("✓ pytest available")
except ImportError:
    print("[X] pytest NOT available - install with: pip install pytest")
    errors.append("pytest")

# Check numpy
try:
    import numpy as np
    print("✓ numpy available")
except ImportError:
    print("[X] numpy NOT available - install with: pip install numpy")
    errors.append("numpy")

# Check torch (optional for some tests)
try:
    import torch
    print("✓ torch available")
except ImportError:
    print("⚠ torch NOT available (some tests may be skipped)")

# Check project modules
print()
print("Verifying project modules...")
print()

try:
    from rl.env_modules import (
        EnvActionManager,
        EnvObsBuilder,
        EnvOpponentManager,
        EnvRewardManager,
    )
    print("✓ rl.env_modules imports successful")
except ImportError as e:
    print(f"[X] rl.env_modules import failed: {e}")
    errors.append("rl.env_modules")

try:
    from rl.agent_identity import build_blue_identities, AgentIdentity
    print("✓ rl.agent_identity imports successful")
except ImportError as e:
    print(f"✗ rl.agent_identity import failed: {e}")
    errors.append("rl.agent_identity")

try:
    from rl.env_modules.contract_validators import (
        validate_action_keys,
        validate_obs_order,
        validate_reward_breakdown,
    )
    print("✓ rl.env_modules.contract_validators imports successful")
except ImportError as e:
    print(f"[X] rl.env_modules.contract_validators import failed: {e}")
    errors.append("rl.env_modules.contract_validators")

# Summary
print()
if errors:
    print(f"[ERROR] Found {len(errors)} error(s). Please fix before running tests.")
    print(f"   Errors: {', '.join(errors)}")
    sys.exit(1)
else:
    print("✅ All imports successful! You can now run pytest.")
    print()
    print("To run tests:")
    print("  python -m pytest tests/test_env_modules.py -v")
    print("  python -m pytest tests/test_integration.py -v")
    print("  python -m pytest tests/test_determinism.py -v")
    print("  python -m pytest tests/test_module_ownership.py -v")
    print()
    print("Or run all tests:")
    print("  python -m pytest tests/ -v")
    sys.exit(0)
