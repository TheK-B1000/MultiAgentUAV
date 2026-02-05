# Test Suite Guide

This directory contains comprehensive tests for the Multi-Agent RL codebase. These tests ensure correctness, determinism, and architectural integrity.

## Quick Start

### Prerequisites

Install pytest and required dependencies:

```bash
pip install pytest pytest-cov numpy torch
```

### Running Tests

**Run all tests:**
```bash
cd AICTFProject
pytest tests/ -v
```

**Run specific test file:**
```bash
pytest tests/test_env_modules.py -v
pytest tests/test_integration.py -v
pytest tests/test_determinism.py -v
pytest tests/test_module_ownership.py -v
```

**Run specific test:**
```bash
pytest tests/test_env_modules.py::test_env_obs_legacy_shape -v
```

**Run with coverage:**
```bash
pytest tests/ --cov=rl --cov-report=html
```

## Test Files Overview

### 1. `test_env_modules.py` - Module Unit Tests

**Purpose**: Tests individual environment modules in isolation using mocks.

**What it verifies**:
- ✅ Observation building produces correct shapes (legacy vs tokenized)
- ✅ Actions are sanitized correctly (masks enforced, invalid actions rejected)
- ✅ Reward routing is correct (per-agent rewards sum to team total)
- ✅ Opponent context IDs are stable and consistent

**When to run**:
- After modifying `rl/env_modules/*.py`
- Before committing changes to observation/action/reward logic
- When debugging module-specific issues

**Example output**:
```
test_env_modules.py::test_env_obs_legacy_shape PASSED
test_env_modules.py::test_env_obs_tokenized_shape PASSED
test_env_modules.py::test_env_actions_sanitize_mask_enforcement PASSED
test_env_modules.py::test_env_rewards_per_agent_sum_equals_team PASSED
...
```

**Key tests**:
- `test_env_obs_legacy_shape()` - Verifies 2-agent concatenated observation shape
- `test_env_obs_tokenized_shape()` - Verifies tokenized observation shape for N>2 agents
- `test_env_actions_sanitize_mask_enforcement()` - Ensures invalid actions are sanitized
- `test_env_rewards_per_agent_sum_equals_team()` - Validates reward routing correctness
- `test_env_opponent_context_id_snapshot()` - Checks snapshot opponent context ID stability

---

### 2. `test_integration.py` - Integration Tests

**Purpose**: Tests the full environment stack (SB3 env + MARL wrapper) end-to-end.

**What it verifies**:
- ✅ SB3 environment can run complete episodes without crashing
- ✅ MARL wrapper correctly interfaces with SB3 environment
- ✅ Canonical agent keys are used consistently throughout

**When to run**:
- After modifying `ctf_sb3_env.py` or `rl/marl_env.py`
- Before major releases
- When debugging environment initialization issues

**Example output**:
```
test_integration.py::test_sb3_env_smoke_rollout PASSED
test_integration.py::test_marl_wrapper_rollout PASSED
test_integration.py::test_sb3_env_canonical_action_keys PASSED
```

**Key tests**:
- `test_sb3_env_smoke_rollout()` - Runs a full episode in SB3 environment
- `test_marl_wrapper_rollout()` - Tests MARL wrapper with SB3 env
- `test_sb3_env_canonical_action_keys()` - Validates canonical key usage

**Note**: These tests require `ctf_sb3_env.py` and `rl/marl_env.py` to be importable. If they're not available, tests will be skipped.

---

### 3. `test_determinism.py` - Determinism Tests

**Purpose**: Ensures reproducibility - same seed produces identical results.

**What it verifies**:
- ✅ Same seed produces identical episode results
- ✅ Evaluation mode produces deterministic results
- ✅ Episode summaries are consistent across runs

**When to run**:
- After modifying seed handling or RNG usage
- Before running evaluation experiments
- When debugging non-reproducible results

**Example output**:
```
test_determinism.py::test_deterministic_episode_same_seed PASSED
test_determinism.py::test_eval_mode_determinism PASSED
test_determinism.py::test_episode_summary_consistency PASSED
```

**Key tests**:
- `test_deterministic_episode_same_seed()` - Runs same seed twice, compares results
- `test_eval_mode_determinism()` - Verifies eval mode disables stochasticity
- `test_episode_summary_consistency()` - Checks episode summary parsing

**Important**: These tests require deterministic settings (action_noise=0.0, etc.). If determinism is broken, these tests will fail.

---

### 4. `test_module_ownership.py` - Structural Checks

**Purpose**: Enforces architectural boundaries - prevents duplicate logic.

**What it verifies**:
- ✅ No duplicate agent identity logic outside `rl/agent_identity.py`
- ✅ No duplicate reward routing outside `rl/env_modules/env_rewards.py`
- ✅ No duplicate opponent switching outside `rl/env_modules/env_opponent.py`

**When to run**:
- Before committing code changes
- During code reviews
- When refactoring modules

**Example output**:
```
test_module_ownership.py::test_no_duplicate_agent_identity_logic PASSED
test_module_ownership.py::test_no_duplicate_reward_routing PASSED
test_module_ownership.py::test_no_duplicate_opponent_switching PASSED
```

**Key tests**:
- `test_no_duplicate_agent_identity_logic()` - Scans for forbidden patterns
- `test_no_duplicate_reward_routing()` - Ensures reward routing is centralized
- `test_no_duplicate_opponent_switching()` - Checks opponent logic is centralized

**How it works**: Uses regex patterns to scan Python files for forbidden function calls. If duplicate logic is found, test fails with a detailed report.

---

## Common Use Cases

### Before Committing Code

Run all tests to ensure nothing broke:
```bash
pytest tests/ -v
```

### After Modifying Environment Modules

Run module unit tests:
```bash
pytest tests/test_env_modules.py -v
```

### Debugging Non-Reproducible Results

Run determinism tests:
```bash
pytest tests/test_determinism.py -v
```

### Checking for Duplicate Logic

Run ownership tests:
```bash
pytest tests/test_module_ownership.py -v
```

### Full Test Suite Before Release

Run everything with coverage:
```bash
pytest tests/ --cov=rl --cov-report=term-missing -v
```

---

## Understanding Test Failures

### Module Unit Test Failures

**Example**: `test_env_rewards_per_agent_sum_equals_team` fails
- **Meaning**: Reward routing is incorrect - per-agent rewards don't sum to team total
- **Fix**: Check `rl/env_modules/env_rewards.py` and `rl/agent_identity.py`

### Integration Test Failures

**Example**: `test_sb3_env_smoke_rollout` fails
- **Meaning**: Environment crashes or produces invalid outputs
- **Fix**: Check `ctf_sb3_env.py` initialization and step logic

### Determinism Test Failures

**Example**: `test_deterministic_episode_same_seed` fails
- **Meaning**: Same seed produces different results (non-deterministic)
- **Fix**: Check RNG seeding in `rl/determinism.py`, ensure action_noise=0.0 in eval mode

### Ownership Test Failures

**Example**: `test_no_duplicate_agent_identity_logic` fails
- **Meaning**: Found duplicate logic outside owner module
- **Fix**: Remove duplicate code, use canonical module functions instead

---

## Writing New Tests

### Adding a Module Unit Test

```python
def test_my_new_feature():
    """Test description."""
    # Setup
    manager = EnvSomeManager()
    
    # Execute
    result = manager.some_method()
    
    # Assert
    assert result == expected_value
```

### Adding an Integration Test

```python
@pytest.mark.skipif(not SB3_ENV_AVAILABLE, reason="SB3 env not available")
def test_my_integration():
    """Test full stack."""
    env = CTFGameFieldSB3Env(...)
    obs, info = env.reset()
    # ... test logic
```

### Adding a Determinism Test

```python
def test_my_determinism():
    """Test reproducibility."""
    seed = 42
    result1 = run_with_seed(seed)
    result2 = run_with_seed(seed)
    assert result1 == result2
```

---

## Continuous Integration

These tests are designed to run in CI/CD pipelines. Example GitHub Actions workflow:

```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
      - run: pip install pytest pytest-cov
      - run: pytest tests/ -v
```

---

## Troubleshooting

### Import Errors

If tests fail with import errors:
1. Ensure you're running from `AICTFProject/` directory
2. Check that `rl/` modules are importable
3. Verify Python path includes project root

### Skipped Tests

Some tests are skipped if dependencies aren't available:
- `test_integration.py` skips if SB3 env not available
- `test_determinism.py` skips if episode_result module not available

This is normal - skipped tests are marked with `SKIPPED` in output.

### Slow Tests

Integration tests are slower (they run full episodes). Use `-k` to filter:
```bash
pytest tests/ -k "not integration" -v  # Skip integration tests
```

---

## Summary

| Test File | Purpose | When to Run |
|-----------|---------|-------------|
| `test_env_modules.py` | Unit tests for modules | After module changes |
| `test_integration.py` | End-to-end environment tests | Before releases |
| `test_determinism.py` | Reproducibility checks | Before evaluations |
| `test_module_ownership.py` | Architectural checks | Before commits |

**Best Practice**: Run all tests before committing code changes.
