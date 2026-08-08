"""Stage-4 analyzer invariants, tested on synthetic fixtures only.

These run against constructed data, never against the live 9810000+ census, so
that implementing the analyzer during the census cannot become analysis design
after seeing data.

Two invariants are easy to get wrong and are the reason this file exists:

1. Zero-anchor episodes. The anchors file contains only anchors, so an episode
   producing none contributes no rows. Bootstrapping over observed episodes
   would condition on episodes that produced at least one anchor and bias
   anchors-per-episode upward.
2. Policy universe. A subset of policies cannot be judged against a
   >=2/3-of-3 replication rule.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

np = pytest.importorskip("numpy")

from experiments.analyze_c3_stage4 import (  # noqa: E402
    anchors_per_episode_counts,
    analyze_leg1,
    episode_universe,
    is_qualified,
)

POLICIES = [3200001, 3200002, 3200003]
OPPONENTS = ["OP6", "OP7", "OP8", "OP9", "OP10", "OP11", "OP12"]
BASE = 9810000
EPISODES = 30


def _frozen(floor: float = 2.5079) -> dict:
    return {
        "policies": POLICIES,
        "opponents": OPPONENTS,
        "seeds": {"base": BASE, "episodes_per_cell": EPISODES},
        "leg_1_fresh_natural": {"floor": floor, "floor_derivation": "0.5 x discovery"},
    }


def test_episode_universe_is_complete_and_sized_from_frozen_cells():
    u = episode_universe(POLICIES, OPPONENTS, BASE, EPISODES)
    assert set(u) == set(POLICIES)
    for p in POLICIES:
        assert len(u[p]) == len(OPPONENTS) * EPISODES == 210
    assert sum(len(v) for v in u.values()) == 630


def test_zero_anchor_episodes_are_counted_as_zero_not_dropped():
    """The core bias guard: one anchor in one episode over a 210-episode universe."""
    u = episode_universe(POLICIES, OPPONENTS, BASE, EPISODES)
    anchors = [{"train_seed": 3200001, "opponent": "OP6", "eval_seed": BASE}]
    counts = anchors_per_episode_counts(anchors, u)
    assert len(counts[3200001]) == 210, "universe must not shrink to observed episodes"
    assert counts[3200001][f"3200001|OP6|{BASE}"] == 1
    assert sum(counts[3200001].values()) == 1
    assert sum(1 for v in counts[3200001].values() if v == 0) == 209


def test_rate_is_diluted_by_empty_episodes_not_conditioned_on_hits():
    """210 anchors in 10 episodes is 1.0/episode overall, not 21.0."""
    u = episode_universe(POLICIES, OPPONENTS, BASE, EPISODES)
    anchors = []
    for i in range(10):
        for _ in range(21):
            anchors.append({"train_seed": 3200001, "opponent": "OP6", "eval_seed": BASE + i})
    res = analyze_leg1(anchors, _frozen())
    per = res["per_policy"]["3200001"]
    assert per["n_anchors"] == 210
    assert per["anchors_per_episode"] == pytest.approx(1.0, abs=1e-9)
    assert per["n_episodes_with_zero_anchors"] == 200
    assert per["LEG1_PASS"] is False  # 1.0 is far below the 2.5079 floor


def test_leg1_passes_only_when_lcb_clears_the_floor():
    u = episode_universe(POLICIES, OPPONENTS, BASE, EPISODES)
    # 8 anchors in every episode of policy 1 -> rate 8.0, comfortably above floor
    rich = [{"train_seed": 3200001, "opponent": o, "eval_seed": BASE + i}
            for o in OPPONENTS for i in range(EPISODES) for _ in range(8)]
    res = analyze_leg1(rich, _frozen())
    p1 = res["per_policy"]["3200001"]
    assert p1["anchors_per_episode"] == pytest.approx(8.0)
    assert p1["LEG1_PASS"] is True
    # policies 2 and 3 contributed nothing -> rate 0, must fail
    assert res["per_policy"]["3200002"]["anchors_per_episode"] == 0.0
    assert res["per_policy"]["3200002"]["LEG1_PASS"] is False


def test_borderline_rate_near_floor_is_decided_by_the_lower_bound():
    """A point estimate just above the floor must not pass on the point alone."""
    n_per = 3  # 3.0/episode vs a 2.5079 floor: point clears, LCB is the test
    anchors = [{"train_seed": 3200001, "opponent": o, "eval_seed": BASE + i}
               for o in OPPONENTS for i in range(EPISODES) for _ in range(n_per)]
    res = analyze_leg1(anchors, _frozen())
    p1 = res["per_policy"]["3200001"]
    assert p1["anchors_per_episode"] == pytest.approx(3.0)
    # constant 3.0 everywhere -> zero variance -> LCB == point > floor
    assert p1["LCB95"] == pytest.approx(3.0)
    assert p1["LEG1_PASS"] is True


def test_anchor_outside_the_frozen_universe_aborts():
    u = episode_universe(POLICIES, OPPONENTS, BASE, EPISODES)
    with pytest.raises(SystemExit, match="outside the frozen episode universe"):
        anchors_per_episode_counts(
            [{"train_seed": 3200001, "opponent": "OP6", "eval_seed": 9999999}], u
        )


def test_unknown_policy_aborts():
    u = episode_universe(POLICIES, OPPONENTS, BASE, EPISODES)
    with pytest.raises(SystemExit, match="outside the frozen episode universe"):
        anchors_per_episode_counts(
            [{"train_seed": 4200001, "opponent": "OP6", "eval_seed": BASE}], u
        )


@pytest.mark.parametrize("value,expected", [
    ("QUALIFIED_COMMITMENT_FORK", True),
    ("NO_COMMITMENT_FORK", False),
])
def test_verdict_reader(value, expected):
    assert is_qualified({"episode_status": value}) is expected


def test_verdict_reader_aborts_on_missing_and_unknown():
    with pytest.raises(SystemExit):
        is_qualified({"train_seed": 1})
    with pytest.raises(SystemExit):
        is_qualified({"episode_status": "MAYBE"})
