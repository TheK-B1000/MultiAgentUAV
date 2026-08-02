"""Integrity of G0-v2 precursor mining: no future leakage, honest controls.

A precursor is only a legitimate routing context if it was observable strictly
BEFORE the failure. These tests pin that property mechanically rather than
trusting the window arithmetic to stay correct.
"""
from __future__ import annotations

import math

import pytest

np = pytest.importorskip("numpy")

from experiments.run_g0_v2_evaluation import (  # noqa: E402
    ACTIONABLE_CONTEXTS,
    FAILURE_INDICATORS,
    OPPORTUNITY_MATCH,
    PRECURSOR_WINDOW,
    _ratio,
    _tot,
    build_windows,
    episode_clustered_ci,
    json_safe,
    window_features,
)


def _step(i: int, **over) -> dict:
    """A synthetic decision-step context with every field legal_context emits."""
    base = {
        "step": i,
        "time_remaining_frac": 1.0 - i / 240.0,
        "blue_score": 0.0,
        "red_score": 0.0,
        "score_diff": 0.0,
        "blue_carrying": 0,
        "red_carrying": 0,
        "blue_tagged": 0,
        "red_tagged": 0,
        "blue_cooldown_active": 0,
        "agents_forward": 1,
        "formation_spread": 0.2,
        "team_separation": 0.2,
        "carrier_present": False,
        "carrier_pressure": float("nan"),
        "carrier_under_pressure": False,
        "escort_available": False,
        "escort_distance": float("nan"),
        "carrier_unescorted": False,
        "red_tag_ready_count": 1,
        "nearest_ready_defender": float("nan"),
        "defender_tag_available": True,
        "home_threatened": False,
        "our_flag_away_from_home": False,
        "nearest_red_to_our_flag": 0.5,
        "nearest_blue_to_our_flag": 0.4,
        "blue_alive_count": 2,
        "red_alive_count": 2,
    }
    base.update(over)
    return base


def _carrier_steps(n: int, *, unescorted: bool = True) -> list[dict]:
    return [
        _step(
            i,
            carrier_present=True,
            blue_carrying=1,
            carrier_pressure=0.30 - 0.005 * i,
            carrier_under_pressure=True,
            escort_distance=0.9 if unescorted else 0.05,
            escort_available=not unescorted,
            carrier_unescorted=unescorted,
            nearest_ready_defender=0.30 - 0.005 * i,
        )
        for i in range(n)
    ]


# --- future leakage ---------------------------------------------------------


def test_window_end_is_exclusive_of_the_failure_step():
    steps = _carrier_steps(80)
    fail_at = 60
    w = window_features(steps, fail_at, kind="failure", label="dropped_the_flag")
    assert w is not None
    assert w["window_end"] == fail_at
    assert w["window_start"] == fail_at - PRECURSOR_WINDOW
    # The window must cover only decisions strictly before the failure.
    assert w["window_start"] + w["window_len"] == fail_at


def test_window_is_invariant_to_everything_after_the_failure():
    """The decisive leakage test: the future cannot change the precursor."""
    steps = _carrier_steps(80)
    fail_at = 60
    before = window_features(steps, fail_at, kind="failure", label="dropped_the_flag")

    # Corrupt every step from the failure onward, including the failing one.
    for i in range(fail_at, len(steps)):
        steps[i].update(
            blue_score=99.0, red_score=99.0, score_diff=-99.0,
            carrier_present=False, carrier_unescorted=False,
            carrier_under_pressure=False, carrier_pressure=0.0,
            escort_distance=0.0, escort_available=True,
            agents_forward=0, home_threatened=True,
            defender_tag_available=False, team_separation=9.9,
        )
    after = window_features(steps, fail_at, kind="failure", label="dropped_the_flag")

    assert before == after, "precursor window changed when only the future changed"


def test_window_does_respond_to_changes_inside_it():
    """Control for the leakage test above: it must not pass vacuously.

    If the window were insensitive to its own contents, the invariance test
    would prove nothing.
    """
    steps = _carrier_steps(80)
    fail_at = 60
    before = window_features(steps, fail_at, kind="failure", label="dropped_the_flag")

    for i in range(fail_at - PRECURSOR_WINDOW, fail_at):
        steps[i].update(carrier_unescorted=False, escort_available=True, escort_distance=0.01)
    after = window_features(steps, fail_at, kind="failure", label="dropped_the_flag")

    assert before != after
    assert before["carrier_unescorted_frac"] == pytest.approx(1.0)
    assert after["carrier_unescorted_frac"] == pytest.approx(0.0)


def test_short_prefix_yields_no_window():
    """A failure too early in the episode has no admissible precursor."""
    steps = _carrier_steps(40)
    assert window_features(steps, 5, kind="failure", label="dropped_the_flag") is None


# --- control construction ---------------------------------------------------


def test_control_windows_avoid_the_failure_neighbourhood():
    steps = _carrier_steps(240)
    failures = [("dropped_the_flag", 100)]
    windows = build_windows(steps, failures, episode_key="OP6:1")
    controls = [w for w in windows if w["kind"] == "control"]
    assert controls, "expected some control windows"
    for w in controls:
        assert abs(w["window_end"] - 100) >= PRECURSOR_WINDOW


def test_every_window_carries_its_episode_cluster_key():
    steps = _carrier_steps(240)
    windows = build_windows(steps, [("dropped_the_flag", 100)], episode_key="OP9:7")
    assert windows
    assert all(w["episode_key"] == "OP9:7" for w in windows)


def test_opportunity_flags_reflect_window_content():
    carrier = build_windows(_carrier_steps(120), [], episode_key="e1")
    assert all(w["opp_has_carrier"] for w in carrier)

    plain = build_windows([_step(i) for i in range(120)], [], episode_key="e2")
    assert not any(w["opp_has_carrier"] for w in plain)
    assert not any(w["opp_leading"] for w in plain)


def test_opportunity_match_covers_every_carrier_failure():
    """Carrier failures must never be compared against carrier-free controls."""
    for label in ("tagged_while_carrying", "dropped_the_flag"):
        assert OPPORTUNITY_MATCH[label] == "opp_has_carrier"
    assert OPPORTUNITY_MATCH["lost_after_leading"] == "opp_leading"
    assert OPPORTUNITY_MATCH["capture_conceded"] == "opp_home_threatened"


# --- clustering -------------------------------------------------------------


def test_bootstrap_clusters_on_episodes_not_windows():
    """Many windows from two episodes must still count as two clusters."""
    fail = [{"episode_key": "A", "f": 1.0} for _ in range(50)]
    fail += [{"episode_key": "B", "f": 1.0} for _ in range(50)]
    ctrl = [{"episode_key": "C", "f": 0.0} for _ in range(50)]
    ctrl += [{"episode_key": "D", "f": 0.0} for _ in range(50)]

    ci = episode_clustered_ci(fail, ctrl, "f", rng=np.random.default_rng(0))
    assert ci["n_failure_episodes"] == 2
    assert ci["n_control_episodes"] == 2
    assert ci["delta"] == pytest.approx(1.0)


def test_bootstrap_reports_insufficient_clusters():
    fail = [{"episode_key": "A", "f": 1.0} for _ in range(100)]
    ctrl = [{"episode_key": "B", "f": 0.0} for _ in range(100)]
    ci = episode_clustered_ci(fail, ctrl, "f", rng=np.random.default_rng(0))
    assert ci["insufficient_clusters"] is True
    assert ci["excludes_zero"] is None


def test_identical_groups_produce_interval_containing_zero():
    fail = [{"episode_key": f"F{i}", "f": 0.5} for i in range(10)]
    ctrl = [{"episode_key": f"C{i}", "f": 0.5} for i in range(10)]
    ci = episode_clustered_ci(fail, ctrl, "f", rng=np.random.default_rng(1))
    assert ci["excludes_zero"] is False


# --- taxonomy + serialization ----------------------------------------------


def test_failure_indicators_are_never_actionable_contexts():
    """A router cannot condition on something that has already happened."""
    assert not (set(FAILURE_INDICATORS) & set(ACTIONABLE_CONTEXTS))


def test_ratio_uses_aggregate_counts_not_mean_of_ratios():
    """A 1-pickup episode must not outweigh a 10-pickup one."""
    # Episode A: 1 pickup, 1 capture (ratio 1.0). Episode B: 10 pickups, 2 captures (0.2).
    rows = [
        {"captures_blue": 1, "captures_red": 0, "pickups": 1, "drops": 0},
        {"captures_blue": 2, "captures_red": 0, "pickups": 10, "drops": 8},
    ]
    aggregate = _ratio(_tot(rows, "captures_blue"), _tot(rows, "pickups"))
    assert aggregate == pytest.approx(3 / 11, abs=1e-4)
    # Averaging the per-episode ratios would give 0.6 -- nearly double.
    mean_of_ratios = (1 / 1 + 2 / 10) / 2
    assert abs(mean_of_ratios - aggregate) > 0.2


def test_ratio_is_null_not_zero_when_no_pickups():
    """'Never had the flag' is not the claim 'converted nothing'."""
    rows = [{"captures_blue": 0, "captures_red": 3, "pickups": 0, "drops": 0}]
    assert _ratio(_tot(rows, "captures_blue"), _tot(rows, "pickups")) is None
    assert _ratio(_tot(rows, "drops"), _tot(rows, "pickups")) is None


def test_net_captures_is_blue_minus_red_on_aggregate_counts():
    rows = [
        {"captures_blue": 5, "captures_red": 2, "pickups": 6, "drops": 1},
        {"captures_blue": 1, "captures_red": 7, "pickups": 3, "drops": 2},
    ]
    assert _tot(rows, "captures_blue") - _tot(rows, "captures_red") == -3


def test_json_safe_replaces_non_finite_with_null():
    payload = {"a": float("nan"), "b": [float("inf"), 1.0], "c": {"d": float("-inf")}}
    assert json_safe(payload) == {"a": None, "b": [None, 1.0], "c": {"d": None}}


def test_json_safe_preserves_ordinary_values():
    payload = {"a": 1, "b": "x", "c": [True, 2.5], "d": None}
    assert json_safe(payload) == payload
