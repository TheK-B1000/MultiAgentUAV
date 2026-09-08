"""The batched legality count must equal the enumerator, state for state.

This test is the scientific bridge across the legality refactor. The 0.7006
precursor audit measured a detector built on the B=1 enumerator, and that audit
is frozen and NOT rerun. What transfers its validity to the batched
implementation the O3 trainer will use is exactly this equivalence -- proof
rather than re-measurement.

If it ever fails, the frozen C_fork predicate no longer means in training what
it meant in the audit, and the audit's coverage number stops describing the
detector actually in use.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

torch = pytest.importorskip("torch")

from rl.analysis.legal_team_responses import (  # noqa: E402
    count_legal_team_responses_batched,
    enumerate_legal_team_responses,
)

CANONICAL_MAP = "map_a"
EPISODE_HORIZON = 240
AGENTS = 2
V2_RULES = dict(
    taggers_required=1,
    tag_min_interval_seconds=10.0,
    tag_nearest_only=True,
    tag_channel_seconds=0.0,
    suppression_attackers_required=2,
)


def _env(n_envs: int, seed: int, device: str = "cpu"):
    from gpu_env import GPUCTFVecEnv, GPUFieldConfig

    cfg = GPUFieldConfig(
        n_envs=n_envs, max_blue_agents=AGENTS, max_red_agents=AGENTS,
        map_set="train", map_layout=CANONICAL_MAP,
        max_decision_steps=EPISODE_HORIZON, aquaticus_profile=True,
        rules_profile="OURS", device=device, seed=seed,
        obstacle_obs_channel=True, tag_telemetry_enabled=True, **V2_RULES,
    )
    env = GPUCTFVecEnv(cfg)
    env.reset()
    return env


def test_exact_count_matches_enumerator_across_many_states():
    """The strong form: integer equality, not just threshold agreement."""
    import numpy as np

    env = _env(1, seed=4242)
    core = env.core
    checked = 0
    observed_counts = set()
    try:
        for step in range(120):
            old = len(enumerate_legal_team_responses(core))
            new = count_legal_team_responses_batched(core)
            assert new.shape == (1,), f"expected shape (1,), got {new.shape}"
            assert int(new[0]) == old, (
                f"legality drift at step {step}: enumerator={old} batched={int(new[0])}"
            )
            observed_counts.add(old)
            checked += 1
            n_actions = AGENTS * 2 * core.B
            env.step_async(np.zeros((n_actions,), dtype=np.int64))
            env.step_wait()
    finally:
        env.close()

    assert checked == 120
    # A test that only ever saw one count would pass with a constant.
    assert len(observed_counts) >= 2, f"degenerate: only saw counts {observed_counts}"


def test_threshold_identity_documents_the_frozen_dependency():
    """The frozen predicate reads `n_legal_team_responses >= 2`."""
    import numpy as np

    env = _env(1, seed=99)
    core = env.core
    try:
        for _ in range(60):
            old = len(enumerate_legal_team_responses(core))
            new = int(count_legal_team_responses_batched(core)[0])
            assert (new >= 2) == (old >= 2)
            n_actions = AGENTS * 2 * core.B
            env.step_async(np.zeros((n_actions,), dtype=np.int64))
            env.step_wait()
    finally:
        env.close()


def test_batched_shape_and_validity_on_a_collector_sized_env():
    """B=16, the training collector width."""
    import numpy as np

    env = _env(16, seed=7)
    core = env.core
    try:
        for _ in range(20):
            counts = count_legal_team_responses_batched(core)
            assert counts.shape == (16,), f"expected (16,), got {counts.shape}"
            assert (counts >= 1).all(), "every env must expose at least one team response"
            n_actions = AGENTS * 2 * core.B
            env.step_async(np.zeros((n_actions,), dtype=np.int64))
            env.step_wait()
    finally:
        env.close()


def test_enumerator_still_rejects_batched_cores():
    """The B=1 contract is preserved, not silently widened."""
    env = _env(4, seed=11)
    try:
        with pytest.raises(ValueError, match="B=1"):
            enumerate_legal_team_responses(env.core)
    finally:
        env.close()


def test_batched_count_equals_product_of_per_agent_legal_macros():
    """Guards the arithmetic itself, independently of the enumerator."""
    import numpy as np

    from rl.analysis.legal_team_responses import _per_agent_legal_macros

    env = _env(8, seed=5150)
    core = env.core
    try:
        per_agent = _per_agent_legal_macros(core)
        counts = count_legal_team_responses_batched(core)
        assert per_agent.shape == (8, AGENTS)
        np.testing.assert_array_equal(counts, per_agent.prod(axis=1))
    finally:
        env.close()
