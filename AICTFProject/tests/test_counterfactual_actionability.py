from __future__ import annotations

import pytest
import math

from rl.analysis.counterfactual_actionability import (
    BranchResult,
    ActionabilityResult,
    compute_actionability
)


def test_branch_result_dataclass():
    br = BranchResult(
        onset_step=10,
        baseline_action=(0, 1, 0, 1),
        branch_action=(1, 1, 0, 1),
        baseline_carrier_survived=True,
        branch_carrier_survived=False,
        baseline_blue_score_delta=1,
        branch_blue_score_delta=0,
        baseline_return=1.5,
        branch_return=0.5,
        outcome_shift=1.0
    )
    assert br.onset_step == 10
    assert br.baseline_action == (0, 1, 0, 1)
    assert br.baseline_carrier_survived is True
    assert br.branch_carrier_survived is False
    assert br.outcome_shift == 1.0


def test_actionability_result_dataclass():
    ar = ActionabilityResult(
        n_onsets=5,
        n_actionable=2,
        actionability_rate=0.4,
        mean_outcome_shift=0.5,
        branches=[],
        determinism_self_test_passed=True
    )
    assert ar.n_onsets == 5
    assert ar.n_actionable == 2
    assert math.isclose(ar.actionability_rate, 0.4)


def test_compute_actionability_all_zero_shift():
    branches = [
        BranchResult(
            onset_step=i,
            baseline_action=(0, 0), branch_action=(1, 0),
            baseline_carrier_survived=True, branch_carrier_survived=True,
            baseline_blue_score_delta=0, branch_blue_score_delta=0,
            baseline_return=0.0, branch_return=0.0,
            outcome_shift=0.0
        ) for i in range(3)
    ]
    
    result = compute_actionability(branches, min_effect=0.05)
    assert result.n_onsets == 3
    assert result.n_actionable == 0
    assert result.actionability_rate == 0.0
    assert result.mean_outcome_shift == 0.0


def test_compute_actionability_some_actionable():
    branches = [
        # Onset 1: Max shift 0.0 (Not actionable)
        BranchResult(onset_step=1, baseline_action=(0,0), branch_action=(1,0),
                     baseline_carrier_survived=True, branch_carrier_survived=True,
                     baseline_blue_score_delta=0, branch_blue_score_delta=0,
                     baseline_return=0.0, branch_return=0.0, outcome_shift=0.0),
        # Onset 2: Max shift 1.0 (Actionable)
        BranchResult(onset_step=2, baseline_action=(0,0), branch_action=(1,0),
                     baseline_carrier_survived=True, branch_carrier_survived=False,
                     baseline_blue_score_delta=0, branch_blue_score_delta=0,
                     baseline_return=0.0, branch_return=0.0, outcome_shift=1.0),
        # Onset 3: Max shift 0.5 (Actionable if min_effect <= 0.5)
        BranchResult(onset_step=3, baseline_action=(0,0), branch_action=(1,0),
                     baseline_carrier_survived=True, branch_carrier_survived=True,
                     baseline_blue_score_delta=0, branch_blue_score_delta=0,
                     baseline_return=0.0, branch_return=0.0, outcome_shift=0.5),
    ]
    
    result = compute_actionability(branches, min_effect=0.5)
    assert result.n_onsets == 3
    assert result.n_actionable == 2
    assert math.isclose(result.actionability_rate, 2/3)
    assert math.isclose(result.mean_outcome_shift, 1.5 / 3)


def test_compute_actionability_threshold_boundary():
    branches = [
        BranchResult(onset_step=1, baseline_action=(0,0), branch_action=(1,0),
                     baseline_carrier_survived=True, branch_carrier_survived=False,
                     baseline_blue_score_delta=0, branch_blue_score_delta=0,
                     baseline_return=0.0, branch_return=0.0, outcome_shift=0.05)
    ]
    
    # Exactly at threshold
    result = compute_actionability(branches, min_effect=0.05)
    assert result.n_actionable == 1
    
    # Just below threshold
    result = compute_actionability(branches, min_effect=0.06)
    assert result.n_actionable == 0


def test_compute_actionability_empty_branches():
    result = compute_actionability([], min_effect=0.05)
    assert result.n_onsets == 0
    assert result.n_actionable == 0
    assert result.actionability_rate == 0.0
    assert result.mean_outcome_shift == 0.0
