"""TASK_HEALTH must fail exactly where SYSTEM_HEALTH passed.

Two G0-v2 seeds passed every numerical health check across 33 checkpoints while
having stopped playing CTF entirely. These tests pin the sentinel that would
have caught it.
"""
from __future__ import annotations

import pytest

from rl.training.task_health import (
    VALIDATION_OPPONENTS,
    VALIDATION_SEEDS,
    combined_verdict,
    evaluate_task_health,
)


def _row(**over) -> dict:
    base = {
        "pickups": 2, "captures_blue": 1, "captures_red": 0, "drops": 1,
        "win": 1, "both_forward_frac": 0.2, "none_forward_frac": 0.4,
    }
    base.update(over)
    return base


def _collapsed_row() -> dict:
    """Exactly the signature both failed seeds produced in evaluation."""
    return _row(pickups=0, captures_blue=0, captures_red=3, drops=0, win=0,
                both_forward_frac=0.0, none_forward_frac=1.0)


def test_collapsed_policy_fails_task_health():
    panel = evaluate_task_health([_collapsed_row() for _ in range(9)], global_step=300_000)
    assert panel.verdict == "FAIL"
    assert panel.pickups == 0
    assert panel.offensive_commitment == 0.0
    assert panel.defensive_commitment == 1.0
    assert len(panel.reasons) == 3


def test_collapsed_policy_reports_pass_and_fail_separately():
    """The whole point: numerically alive, behaviourally dead."""
    panel = evaluate_task_health([_collapsed_row() for _ in range(9)], global_step=300_000)
    v = combined_verdict(system_health_ok=True, panel=panel)
    assert v == {"SYSTEM_HEALTH": "PASS", "TASK_HEALTH": "FAIL"}


def test_healthy_policy_passes():
    panel = evaluate_task_health([_row() for _ in range(9)], global_step=300_000)
    assert panel.verdict == "PASS"
    assert panel.reasons == []
    assert panel.capture_conversion == pytest.approx(0.5)
    assert panel.net_captures == 9


def test_zero_pickups_yields_null_conversion_not_zero():
    panel = evaluate_task_health([_collapsed_row() for _ in range(9)], global_step=1)
    assert panel.capture_conversion is None


def test_each_collapse_symptom_fails_independently():
    only_no_pickup = evaluate_task_health(
        [_row(pickups=0, captures_blue=0) for _ in range(9)], global_step=1)
    assert only_no_pickup.verdict == "FAIL"

    only_never_forward = evaluate_task_health(
        [_row(none_forward_frac=1.0) for _ in range(9)], global_step=1)
    assert only_never_forward.verdict == "FAIL"

    only_no_offense = evaluate_task_health(
        [_row(both_forward_frac=0.0) for _ in range(9)], global_step=1)
    assert only_no_offense.verdict == "FAIL"


def test_winning_does_not_excuse_a_dead_policy():
    """A draw-farming policy that never touches the flag is still not playing."""
    rows = [_row(pickups=0, captures_blue=0, captures_red=0, win=0,
                 both_forward_frac=0.0, none_forward_frac=1.0) for _ in range(9)]
    panel = evaluate_task_health(rows, global_step=1)
    assert panel.verdict == "FAIL"
    assert panel.net_captures == 0


def test_validation_seeds_are_disjoint_from_training_and_evaluation():
    """Panel seeds must never contaminate a formal result."""
    training = {2_500_001, 2_500_002, 2_500_003, 2_600_001, 2_600_002, 2_600_003}
    discovery = set(range(9_100_000, 9_100_030))
    diagnostic = set(range(9_200_000, 9_200_010))
    panel = set(VALIDATION_SEEDS)
    assert not (panel & training)
    assert not (panel & discovery)
    assert not (panel & diagnostic)


def test_panel_size_is_large_enough_to_resist_saturation():
    """9 episodes saturated at 9/9 and went blind. 21-30 restores headroom.

    Bounded above as well: the panel runs at every checkpoint inside training,
    so resolution is traded against wall-clock.
    """
    n = len(VALIDATION_OPPONENTS) * len(VALIDATION_SEEDS)
    assert 21 <= n <= 30, f"panel is {n} episodes; contract requires 21-30"


def test_panel_covers_the_full_admitted_opponent_mixture():
    """Subsetting opponents is what let the panel saturate on easy cells."""
    admitted = {"OP6", "OP7", "OP8", "OP9", "OP10", "OP11", "OP12"}
    assert set(VALIDATION_OPPONENTS) == admitted
