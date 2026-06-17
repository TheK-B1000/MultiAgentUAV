"""Unit tests for ``rl.route_telemetry``."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rl.route_telemetry import (
    ROUTE_CROSSING_KEYS,
    attach_route_telemetry_to_row,
    route_derived_features,
    route_step_crossings,
    route_switch_indicator,
    zero_route_cumulative,
)


def test_route_step_crossings_from_cumulative_counters():
    prev = zero_route_cumulative()
    prev["blue_attack_upper_crossings"] = 2
    cum = dict(prev)
    cum["blue_attack_upper_crossings"] = 5
    cum["blue_attack_lower_crossings"] = 1
    deltas = route_step_crossings(prev, cum)
    assert deltas["blue_attack_upper_crossings"] == pytest.approx(3.0)
    assert deltas["blue_attack_lower_crossings"] == pytest.approx(1.0)
    assert deltas["blue_return_upper_crossings"] == pytest.approx(0.0)


def test_route_derived_fractions_attack_upper_dominant():
    deltas = {k: 0.0 for k in ROUTE_CROSSING_KEYS}
    deltas["blue_attack_upper_crossings"] = 3.0
    deltas["blue_attack_lower_crossings"] = 1.0
    derived = route_derived_features(deltas)
    assert derived["upper_attack_fraction"] == pytest.approx(0.75)
    assert derived["lower_attack_fraction"] == pytest.approx(0.25)
    assert derived["intercept_lane_fraction"] == pytest.approx(0.0)


def test_route_switch_indicator_fires_on_lane_change():
    a = {k: 0.0 for k in ROUTE_CROSSING_KEYS}
    a["blue_attack_upper_crossings"] = 1.0
    b = {k: 0.0 for k in ROUTE_CROSSING_KEYS}
    b["blue_return_lower_crossings"] = 1.0
    prev_lane = attach_route_telemetry_to_row({}, step_crossings=a, prev_dominant_lane=None)
    assert prev_lane == 0
    assert route_switch_indicator(b, prev_lane) == pytest.approx(1.0)
    assert route_switch_indicator(b, 3) == pytest.approx(0.0)
