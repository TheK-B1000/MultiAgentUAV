"""Injection tests for the Gate 1 event checker.

A checker that never fires is worthless, and the previous Gate 1 fired for the
wrong reasons. These tests feed deliberately illegal event streams to
``check_events`` and require each violation to be caught, plus require a
legal stream to come back clean.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.gate1_opponent_sanity_v2 import check_events  # noqa: E402

TAG_RANGE = 2.5
COOLDOWN = 10.0


def legal_success(**over):
    e = {
        "event_type": "tag_success", "env_index": 0, "simulation_time": 1.0,
        "ruleset_id": "RULESET_V2_AQUATICUS_10S",
        "tagger_team": "blue", "tagger_index": 0,
        "target_team": "red", "target_index": 0,
        "tagger_position_at_decision": (10.0, 10.0),
        "target_position_at_decision": (11.0, 10.0),
        "distance_at_decision": 1.0,
        "tagger_on_own_side": True, "target_on_tagger_side": True,
        "tagger_cooldown_before": 0.0, "tagger_cooldown_after": COOLDOWN,
        "target_was_tagged": False, "target_was_carrying_flag": False,
        "eligible_target_indices": [0], "selected_nearest_target": 0,
    }
    e.update(over)
    return e


def legal_denial(**over):
    e = {
        "event_type": "tag_denied", "reason": "cooldown", "env_index": 0,
        "simulation_time": 2.0, "ruleset_id": "RULESET_V2_AQUATICUS_10S",
        "tagger_team": "blue", "tagger_index": 1,
        "candidate_target_index": 0, "cooldown_remaining": 7.5,
    }
    e.update(over)
    return e


def run(events):
    return check_events(events, tag_range=TAG_RANGE, cooldown_T=COOLDOWN)


def test_legal_stream_is_clean():
    r = run([legal_success(), legal_denial(),
             {"event_type": "episode_reset", "env_index": 0}])
    assert r["violations"] == {}, r["violations"]
    assert r["n_success"] == 1 and r["n_denied"] == 1


@pytest.mark.parametrize("override,expected", [
    ({"tagger_on_own_side": False}, "tagger_not_on_own_side"),
    ({"target_on_tagger_side": False}, "target_not_on_tagger_side"),
    ({"distance_at_decision": TAG_RANGE + 0.5}, "tag_out_of_range"),
    ({"target_was_tagged": True}, "retagged_already_tagged_target"),
    ({"tagger_cooldown_before": 3.0}, "tag_during_cooldown"),
    ({"eligible_target_indices": []}, "tag_with_no_eligible_target"),
    ({"eligible_target_indices": [1], "selected_nearest_target": 0},
     "selected_target_not_eligible"),
    ({"target_team": "blue"}, "friendly_tag"),
])
def test_illegal_success_is_caught(override, expected):
    r = run([legal_success(**override)])
    assert expected in r["violations"], (
        f"checker missed {expected}; got {r['violations']}")


def test_duplicate_event_is_caught():
    e = legal_success()
    r = run([e, dict(e)])
    assert "duplicate_event" in r["violations"]


def test_missing_schema_field_is_caught():
    e = legal_success()
    e.pop("tagger_on_own_side")
    r = run([e])
    assert "schema_missing_field" in r["violations"]


def test_denial_without_cooldown_is_caught():
    r = run([legal_denial(cooldown_remaining=0.0)])
    assert "denial_without_cooldown" in r["violations"]


def test_unexpected_denial_reason_is_caught():
    r = run([legal_denial(reason="out_of_range")])
    assert "unexpected_denial_reason" in r["violations"]


def test_denied_and_succeeded_same_instant_is_caught():
    """One tagger cannot both succeed and be cooldown-denied at one instant."""
    r = run([legal_success(simulation_time=5.0, tagger_index=0),
             legal_denial(simulation_time=5.0, tagger_index=0)])
    assert "denied_and_succeeded_same_instant" in r["violations"]


def test_unknown_event_type_is_caught():
    r = run([{"event_type": "something_else", "env_index": 0}])
    assert "unknown_event_type" in r["violations"]


def test_missing_denial_schema_field_is_caught():
    e = legal_denial()
    e.pop("cooldown_remaining")
    r = run([e])
    assert "schema_missing_field" in r["violations"]


def test_identity_is_scoped_by_episode():
    """simulation_time restarts each episode; identity must not collide.

    Regression: pooling events across episodes made a success in episode 1 and a
    denial in episode 2 by the same tagger look like a same-instant
    contradiction, and made identical (time, tagger, target) tuples look like
    duplicates. Both were checker artifacts, not environment faults.
    """
    stream = [
        legal_success(simulation_time=5.0, tagger_index=0),
        {"event_type": "episode_reset", "env_index": 0},
        legal_denial(simulation_time=5.0, tagger_index=0),
    ]
    r = run(stream)
    assert "denied_and_succeeded_same_instant" not in r["violations"], r["violations"]

    dup_across = [
        legal_success(simulation_time=5.0),
        {"event_type": "episode_reset", "env_index": 0},
        legal_success(simulation_time=5.0),
    ]
    assert "duplicate_event" not in run(dup_across)["violations"]

    # ...but a genuine duplicate WITHIN one episode must still be caught.
    dup_within = [legal_success(simulation_time=5.0), legal_success(simulation_time=5.0)]
    assert "duplicate_event" in run(dup_within)["violations"]
