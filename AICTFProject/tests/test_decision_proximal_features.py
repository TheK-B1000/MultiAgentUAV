from __future__ import annotations

import math
import pytest
import torch
import numpy as np
from dataclasses import asdict
from unittest.mock import MagicMock

from rl.analysis.decision_proximal_features import (
    DecisionProximalExtractor,
    DecisionProximalFeatures,
    ESCORT_RADIUS_FRAC,
    PRESSURE_RADIUS_FRAC,
    DEFAULT_SPEED,
)


def _make_mock_core(*, blue_x, blue_y, red_x, red_y, blue_alive, red_alive,
                     blue_carrying, red_carrying=None, blue_tag_cooldown=None,
                     red_tag_cooldown=None, blue_tagged=None, red_tagged=None,
                     blue_flag_home=None, blue_flag_pos=None, red_flag_pos=None,
                     blue_score=0.0, red_score=0.0, step_count=0,
                     cols=160.0, rows=80.0, max_dist=178.9):
    """Create a mock core with B=1 tensors."""
    core = MagicMock()
    N = len(blue_x)
    Nr = len(red_x)
    core.blue_x = torch.tensor([[*blue_x]], dtype=torch.float32)
    core.blue_y = torch.tensor([[*blue_y]], dtype=torch.float32)
    core.red_x = torch.tensor([[*red_x]], dtype=torch.float32)
    core.red_y = torch.tensor([[*red_y]], dtype=torch.float32)
    core.blue_alive = torch.tensor([[*blue_alive]], dtype=torch.bool)
    core.red_alive = torch.tensor([[*red_alive]], dtype=torch.bool)
    core.blue_carrying = torch.tensor([[*blue_carrying]], dtype=torch.bool)
    core.red_carrying = torch.tensor([[*(red_carrying or [False]*Nr)]], dtype=torch.bool)
    core.blue_tag_cooldown = torch.tensor([[*(blue_tag_cooldown or [0.0]*N)]], dtype=torch.float32)
    core.red_tag_cooldown = torch.tensor([[*(red_tag_cooldown or [0.0]*Nr)]], dtype=torch.float32)
    core.blue_tagged = torch.tensor([[*(blue_tagged or [False]*N)]], dtype=torch.bool)
    core.red_tagged = torch.tensor([[*(red_tagged or [False]*Nr)]], dtype=torch.bool)
    core.blue_flag_home = torch.tensor([blue_flag_home or [10.0, 40.0]], dtype=torch.float32)
    core.blue_flag_pos = torch.tensor([blue_flag_pos or [10.0, 40.0]], dtype=torch.float32)
    core.red_flag_pos = torch.tensor([red_flag_pos or [150.0, 40.0]], dtype=torch.float32)
    core.blue_score = torch.tensor([blue_score], dtype=torch.float32)
    core.red_score = torch.tensor([red_score], dtype=torch.float32)
    core.step_count = torch.tensor([step_count], dtype=torch.int64)
    core.cols = cols
    core.rows = rows
    core.max_dist = max_dist
    core.cfg = MagicMock()
    core.cfg.max_decision_steps = 240
    return core


def test_no_carrier_returns_nan():
    core = _make_mock_core(
        blue_x=[20.0, 30.0], blue_y=[20.0, 30.0],
        red_x=[140.0, 130.0], red_y=[20.0, 30.0],
        blue_alive=[True, True], red_alive=[True, True],
        blue_carrying=[False, False]
    )
    extractor = DecisionProximalExtractor()
    features = extractor.extract(core)
    
    assert math.isnan(features.time_to_intercept)
    assert math.isnan(features.relative_closing_velocity)
    assert math.isnan(features.carrier_dist_home)
    assert math.isnan(features.nearest_ready_defender_dist)
    assert math.isnan(features.escort_dist)
    assert math.isnan(features.cooldown_remaining)
    assert math.isnan(features.carrier_progress_frac)
    assert math.isnan(features.mate_intervention_eta)
    assert math.isnan(features.intercept_margin)
    assert features.is_carrier_pressure_onset is False


def test_carrier_features_non_nan():
    core = _make_mock_core(
        blue_x=[140.0, 30.0], blue_y=[40.0, 30.0],
        red_x=[130.0, 130.0], red_y=[40.0, 30.0],
        blue_alive=[True, True], red_alive=[True, True],
        blue_carrying=[True, False]
    )
    extractor = DecisionProximalExtractor()
    features = extractor.extract(core)
    
    assert not math.isnan(features.time_to_intercept)
    assert not math.isnan(features.relative_closing_velocity)
    assert not math.isnan(features.carrier_dist_home)
    assert not math.isnan(features.nearest_ready_defender_dist)
    assert not math.isnan(features.escort_dist)
    assert not math.isnan(features.cooldown_remaining)
    assert not math.isnan(features.carrier_progress_frac)
    assert not math.isnan(features.mate_intervention_eta)
    assert not math.isnan(features.intercept_margin)


def test_onset_detection_pickup():
    extractor = DecisionProximalExtractor()
    
    core_prev = _make_mock_core(
        blue_x=[20.0], blue_y=[20.0],
        red_x=[140.0], red_y=[20.0],
        blue_alive=[True], red_alive=[True],
        blue_carrying=[False]
    )
    extractor.extract(core_prev)
    
    core_curr = _make_mock_core(
        blue_x=[20.0], blue_y=[20.0],
        red_x=[140.0], red_y=[20.0],
        blue_alive=[True], red_alive=[True],
        blue_carrying=[True]
    )
    features = extractor.extract(core_curr)
    
    assert features.is_carrier_pressure_onset is True


def test_onset_detection_pressure_transition():
    extractor = DecisionProximalExtractor()
    cols = 100.0
    
    # Carrier not under pressure
    core_prev = _make_mock_core(
        blue_x=[20.0], blue_y=[20.0],
        red_x=[90.0], red_y=[20.0],
        blue_alive=[True], red_alive=[True],
        blue_carrying=[True],
        cols=cols
    )
    extractor.extract(core_prev)
    
    # Red moves within PRESSURE_RADIUS_FRAC
    pressure_dist = PRESSURE_RADIUS_FRAC * cols - 1.0 # just inside
    core_curr = _make_mock_core(
        blue_x=[20.0], blue_y=[20.0],
        red_x=[20.0 + pressure_dist], red_y=[20.0],
        blue_alive=[True], red_alive=[True],
        blue_carrying=[True],
        cols=cols
    )
    features = extractor.extract(core_curr)
    
    assert features.is_carrier_pressure_onset is True


def test_deterministic_extraction():
    core = _make_mock_core(
        blue_x=[140.0, 30.0], blue_y=[40.0, 30.0],
        red_x=[130.0, 130.0], red_y=[40.0, 30.0],
        blue_alive=[True, True], red_alive=[True, True],
        blue_carrying=[True, False]
    )
    
    extractor1 = DecisionProximalExtractor()
    features1 = extractor1.extract(core)
    
    extractor2 = DecisionProximalExtractor()
    features2 = extractor2.extract(core)
    
    assert asdict(features1) == asdict(features2)


def test_intercept_margin_sign_convention():
    extractor = DecisionProximalExtractor()
    cols = 100.0
    
    # Red and mate moving - set up prev
    core_prev = _make_mock_core(
        blue_x=[50.0, 20.0], blue_y=[50.0, 50.0],
        red_x=[80.0], red_y=[50.0],
        blue_alive=[True, True], red_alive=[True],
        blue_carrying=[True, False],
        cols=cols
    )
    extractor.extract(core_prev)
    
    # Mate is at 30.0, Red is at 70.0 (dist 20.0 each). 
    # Let's make mate closer: mate at dist 10, red at dist 30
    core_mate_closer = _make_mock_core(
        blue_x=[50.0, 40.0], blue_y=[50.0, 50.0],
        red_x=[80.0], red_y=[50.0],
        blue_alive=[True, True], red_alive=[True],
        blue_carrying=[True, False],
        cols=cols
    )
    # Re-init for clean start, give red some closing velocity
    extractor.reset()
    # Fake prev red pos for positive closing velocity
    extractor._prev_red_pos = np.array([[85.0, 50.0]])
    extractor._prev_blue_pos = np.array([[50.0, 50.0], [40.0, 50.0]])
    extractor._prev_carrier_pressure = 0.0
    extractor._prev_under_pressure = False
    extractor._prev_carrying = np.array([True, False])
    
    features_mate = extractor.extract(core_mate_closer)
    assert features_mate.intercept_margin > 0
    
    # Let's make red closer: mate at dist 30, red at dist 10
    extractor.reset()
    core_red_closer = _make_mock_core(
        blue_x=[50.0, 20.0], blue_y=[50.0, 50.0],
        red_x=[60.0], red_y=[50.0],
        blue_alive=[True, True], red_alive=[True],
        blue_carrying=[True, False],
        cols=cols
    )
    extractor._prev_red_pos = np.array([[65.0, 50.0]])
    extractor._prev_blue_pos = np.array([[50.0, 50.0], [20.0, 50.0]])
    extractor._prev_carrier_pressure = 0.0
    extractor._prev_under_pressure = False
    extractor._prev_carrying = np.array([True, False])
    
    features_red = extractor.extract(core_red_closer)
    assert features_red.intercept_margin < 0


def test_reset_clears_state():
    extractor = DecisionProximalExtractor()
    core = _make_mock_core(
        blue_x=[20.0], blue_y=[20.0],
        red_x=[140.0], red_y=[20.0],
        blue_alive=[True], red_alive=[True],
        blue_carrying=[True]
    )
    
    features1 = extractor.extract(core)
    extractor.reset()
    features2 = extractor.extract(core)
    
    assert asdict(features1) == asdict(features2)


def test_feature_dict_round_trip():
    core = _make_mock_core(
        blue_x=[140.0, 30.0], blue_y=[40.0, 30.0],
        red_x=[130.0, 130.0], red_y=[40.0, 30.0],
        blue_alive=[True, True], red_alive=[True, True],
        blue_carrying=[True, False]
    )
    extractor = DecisionProximalExtractor()
    features = extractor.extract(core)
    
    d = asdict(features)
    assert 'time_to_intercept' in d
    assert 'relative_closing_velocity' in d
    assert 'is_carrier_pressure_onset' in d
    assert 'intercept_margin' in d
    
    # Can recreate
    f2 = DecisionProximalFeatures(**d)
    assert f2.is_carrier_pressure_onset == features.is_carrier_pressure_onset
