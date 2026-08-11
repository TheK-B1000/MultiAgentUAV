"""C6 code-safety gate: legacy opponent RESOLUTION is unchanged from 55cf5e7.

The sibling test pins the BTProfile knobs. This one pins the identity plumbing —
canonicalization, level mapping, BT membership, and the alias/synonym tables —
because that seam is where opponent-naming mistakes have actually landed twice in
this project: once reading a synonym's deprecated side as canonical, and once
before that.

A profile can be byte-identical while an alias silently reroutes a key to a
different level, which would repoint every historical experiment that names an
opponent by string. Baseline captured from git history at 55cf5e7, not from
current code.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from gpu_env._core._bt_profiles import (
    OPPONENT_ALIASES,
    OPPONENT_SYNONYMS,
    LRO_AUDITED_OPPONENT_POOL,
    canonicalize_opponent_key,
    is_bt_opponent,
    normalize_bt_level,
)
from rl.evaluation.opponent_resolution import validate_opponent_name

BASELINE = Path(__file__).resolve().parent / "data" / "c6_legacy_resolution_baseline.json"


@pytest.fixture(scope="module")
def baseline() -> dict:
    return json.loads(BASELINE.read_text(encoding="utf-8"))


def _keys(baseline: dict):
    return [k for k in baseline if not k.startswith("_")]


def test_canonicalization_unchanged(baseline):
    """Every legacy key and historical synonym resolves where it always did."""
    for key in _keys(baseline):
        assert canonicalize_opponent_key(key) == baseline[key]["canonical"], (
            f"{key} now canonicalizes differently. Aliases reroute every historical "
            f"experiment that names this opponent by string.")


def test_levels_unchanged(baseline):
    for key in _keys(baseline):
        assert normalize_bt_level(key) == baseline[key]["level"]


def test_bt_membership_unchanged(baseline):
    for key in _keys(baseline):
        assert is_bt_opponent(key) is baseline[key]["is_bt"]


def test_alias_and_synonym_tables_are_supersets(baseline):
    """C6 may ADD entries; it may not change or remove an existing one."""
    for table, recorded in ((OPPONENT_ALIASES, baseline["_ALIASES"]),
                            (OPPONENT_SYNONYMS, baseline["_SYNONYMS"])):
        for k, v in recorded.items():
            assert table.get(k) == v, f"{k} remapped from {v!r} to {table.get(k)!r}"


def test_audited_pool_unchanged(baseline):
    """The permanent board is exactly what it was; C6 is not on it."""
    assert list(LRO_AUDITED_OPPONENT_POOL) == baseline["_LRO_POOL"]


def test_c6_keys_resolve_and_are_distinct(baseline):
    """C6 keys work through the same plumbing without colliding with legacy."""
    legacy_levels = {baseline[k]["level"] for k in _keys(baseline)}
    for key in ("C6A", "C6B"):
        assert validate_opponent_name(key) == key
        assert is_bt_opponent(key)
        assert normalize_bt_level(key) not in legacy_levels


def test_bt_dispatch_levels_unchanged(baseline):
    """Legacy dispatch levels are exactly what they were at 55cf5e7.

    The C6 families needed the dispatch ceiling raised from 12 to 14. Without
    this pin, a future bound change could silently re-route or drop a legacy
    opponent from the BT brain -- the failure mode that made C6A and C6B
    behave identically until Stage 1 caught it.
    """
    from gpu_env._core._scripted_red import bt_dispatch_level_for_opponent_key as D

    for key, expected in baseline["_DISPATCH_LEVELS_AT_55cf5e7"].items():
        assert D(key) == expected, f"{key} dispatch level changed"


def test_c6_families_reach_the_bt_brain():
    """C6A/C6B must DISPATCH, not merely resolve to a profile."""
    from gpu_env._core._scripted_red import bt_dispatch_level_for_opponent_key as D

    assert D("C6A") == 13
    assert D("C6B") == 14


def test_c6_families_actually_behave_differently():
    """The families must produce DIFFERENT red behaviour, not just different configs.

    C6 Stage 1 failed twice with C6A and C6B reporting metrics identical to three
    decimals. Both times every config-level check passed: profiles resolved, knobs
    were correct, identity gates were green. The families were falling through to
    the legacy scripted red because the BT activation mask enumerates levels
    explicitly, so a new level silently no-ops.

    Config correctness and reaching the simulator are different properties. This
    asserts the second one, because a Stage-2 scan between an opponent and itself
    would have produced a perfectly plausible NO_REVERSAL.
    """
    import hashlib

    import numpy as np
    import torch

    from experiments.run_g0_v2_evaluation import (
        AGENTS, CANONICAL_MAP, EPISODE_HORIZON, V2_RULES,
    )
    from gpu_env import GPUCTFVecEnv, GPUFieldConfig
    from rl.evaluation.opponent_resolution import set_opponent

    def red_trace(opponent: str, steps: int = 30) -> str:
        cfg = GPUFieldConfig(
            n_envs=1, max_blue_agents=AGENTS, max_red_agents=AGENTS,
            map_set="train", map_layout=CANONICAL_MAP,
            max_decision_steps=EPISODE_HORIZON, aquaticus_profile=True,
            rules_profile="OURS", device="cpu", seed=9870000,
            obstacle_obs_channel=True, tag_telemetry_enabled=True, **V2_RULES,
        )
        env = GPUCTFVecEnv(cfg)
        try:
            set_opponent(env, opponent)
            env.reset()
            frames = []
            for _ in range(steps):
                frames.append(env.core.red_pos.detach().cpu().numpy().copy())
                env.step(torch.zeros((1, 2 * AGENTS), dtype=torch.long))
        finally:
            env.close()
        return hashlib.sha256(np.stack(frames).tobytes()).hexdigest()

    a, b, op7 = red_trace("C6A"), red_trace("C6B"), red_trace("OP7")
    assert a != b, "C6A and C6B produce identical red trajectories -- neither is reaching the BT brain"
    assert a != op7 and b != op7, "a C6 family is behaving like a legacy opponent"
