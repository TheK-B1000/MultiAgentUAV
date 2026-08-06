from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

import rl.analysis.counterfactual_actionability as actionability_module

from rl.analysis.counterfactual_actionability import (
    NO_COMMITMENT_FORK,
    QUALIFIED_COMMITMENT_FORK,
    ActionabilityResult,
    BranchResult,
    CounterfactualBranchSet,
    RolloutOutcome,
    backward_trace_steps,
    compute_actionability,
    enumerate_legal_team_responses,
    find_earliest_commitment_fork,
    resolve_utility,
    run_counterfactual_branches,
)


def _outcome(utility: float) -> RolloutOutcome:
    return RolloutOutcome(
        cumulative_return=float(utility),
        carrier_survived=utility > 0,
        blue_score_delta=int(utility > 1),
        steps_executed=15,
    )


def _branch_set(
    *,
    baseline_utility: float,
    branch_utilities: tuple[float, ...],
    legal_responses: tuple[tuple[int, ...], ...] | None = None,
    response_horizon: int = 15,
    candidate_step: int = 4,
) -> CounterfactualBranchSet:
    if legal_responses is None:
        legal_responses = tuple((macro, 0) for macro in range(len(branch_utilities) + 1))
    baseline_response = legal_responses[0]
    baseline_action = (baseline_response[0], 0, baseline_response[1], 0)
    branches = []
    for response, utility in zip(legal_responses[1:], branch_utilities):
        branch_action = (response[0], 0, response[1], 0)
        branches.append(
            BranchResult(
                candidate_step=candidate_step,
                response_horizon=response_horizon,
                baseline_action=baseline_action,
                branch_action=branch_action,
                team_response=response,
                baseline_outcome=_outcome(baseline_utility),
                branch_outcome=_outcome(utility),
                baseline_utility=baseline_utility,
                branch_utility=utility,
                utility_improvement=utility - baseline_utility,
            )
        )
    return CounterfactualBranchSet(
        candidate_step=candidate_step,
        response_horizon=response_horizon,
        baseline_action=baseline_action,
        baseline_team_response=baseline_response,
        legal_team_responses=legal_responses,
        baseline_outcome=_outcome(baseline_utility),
        baseline_utility=baseline_utility,
        branches=tuple(branches),
    )


class _MaskedCore:
    def __init__(self):
        self.cfg = SimpleNamespace(n_macros=5, n_targets=2)
        self.blue_alive = torch.tensor([[True, True]])

    def _build_action_mask(self, side="blue"):
        assert side == "blue"
        # Agent 0: macros {0,4}; agent 1: macros {1,2}. Targets are legal but
        # are not part of the team-response Cartesian product.
        return torch.tensor(
            [[1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1]],
            dtype=torch.float32,
        )


def test_legal_team_responses_come_from_authoritative_mask_and_include_joint_options():
    assert enumerate_legal_team_responses(_MaskedCore()) == (
        (0, 1),
        (0, 2),
        (4, 1),
        (4, 2),
    )


def test_at_least_two_legal_team_responses_are_required():
    branch_set = _branch_set(
        baseline_utility=0.0,
        branch_utilities=(),
        legal_responses=((0, 0),),
    )
    result = compute_actionability(
        branch_set,
        delta=0.1,
        doomed_utility_threshold=-1.0,
    )
    assert result.n_legal_team_responses == 1
    assert result.is_actionable is False


def test_only_improvement_over_g0_counts_not_absolute_change():
    result = compute_actionability(
        _branch_set(baseline_utility=1.0, branch_utilities=(-2.0, -0.5)),
        delta=0.1,
        doomed_utility_threshold=-3.0,
    )
    assert result.max_expected_utility_improvement == -1.5
    assert result.has_persistent_utility_divergence is False
    assert result.is_actionable is False


def test_improvement_at_delta_over_response_horizon_is_actionable():
    result = compute_actionability(
        _branch_set(
            baseline_utility=0.2,
            branch_utilities=(0.3,),
            response_horizon=23,
        ),
        delta=0.1,
        doomed_utility_threshold=0.0,
    )
    assert result.response_horizon == 23
    assert abs(result.max_expected_utility_improvement - 0.1) < 1e-12
    assert result.has_persistent_utility_divergence is True
    assert result.is_actionable is True


def test_doomed_state_is_rejected_even_when_alternative_improves_utility():
    result = compute_actionability(
        _branch_set(baseline_utility=-2.0, branch_utilities=(-1.0, -1.5)),
        delta=0.1,
        doomed_utility_threshold=-0.5,
    )
    assert result.max_expected_utility_improvement == 1.0
    assert result.effectively_doomed is True
    assert result.is_actionable is False


def _evaluation(step: int, actionable: bool) -> ActionabilityResult:
    return ActionabilityResult(
        candidate_step=step,
        response_horizon=15,
        n_legal_team_responses=2,
        baseline_utility=0.0,
        best_expected_utility=1.0 if actionable else 0.0,
        max_expected_utility_improvement=1.0 if actionable else 0.0,
        best_team_response=(1, 0) if actionable else None,
        effectively_doomed=False,
        has_persistent_utility_divergence=actionable,
        is_actionable=actionable,
        branches=(),
    )


def test_backward_trace_excludes_pressure_anchor_and_orders_candidates_chronologically():
    assert backward_trace_steps(pressure_step=8, t_trace=3) == (5, 6, 7)


def test_backward_trace_retains_earliest_qualifying_state():
    evaluated = []

    def evaluate(step: int) -> ActionabilityResult:
        evaluated.append(step)
        return _evaluation(step, actionable=step in (4, 5))

    result = find_earliest_commitment_fork(
        pressure_step=6,
        t_trace=4,
        evaluate_candidate=evaluate,
    )
    assert evaluated == [2, 3, 4]
    assert result.episode_status == QUALIFIED_COMMITMENT_FORK
    assert result.fork_step == 4
    assert result.science_scope == "CONTROLLABILITY_SCREEN_ONLY"


def test_backward_trace_emits_explicit_no_fork_without_promoting_pressure():
    result = find_earliest_commitment_fork(
        pressure_step=3,
        t_trace=3,
        evaluate_candidate=lambda step: _evaluation(step, actionable=False),
    )
    assert result.episode_status == NO_COMMITMENT_FORK
    assert result.fork_step is None
    assert result.pressure_step not in result.candidate_steps


def test_utility_resolution_uses_existing_task_outcomes_only():
    outcome = RolloutOutcome(2.5, True, 1, 15)
    assert resolve_utility("return")(outcome) == 2.5
    assert resolve_utility("carrier_survival")(outcome) == 1.0
    assert resolve_utility("blue_score_delta")(outcome) == 1.0


def test_run_counterfactual_branches_real_api_accepts_team_responses_and_horizon(monkeypatch):
    core = _MaskedCore()
    env = SimpleNamespace(core=core)

    class Model:
        def predict(self, obs, deterministic=True):
            return np.array([0, 0, 1, 0]), None

    observed_horizons = []
    monkeypatch.setattr(actionability_module, "_snapshot_env", lambda env: {})
    monkeypatch.setattr(actionability_module, "_restore_env", lambda env, snap: None)
    monkeypatch.setattr(actionability_module, "_snapshot_policy", lambda model: {})
    monkeypatch.setattr(actionability_module, "_restore_policy", lambda model, snap: None)
    monkeypatch.setattr(actionability_module, "_single_obs", lambda obs, env: obs)
    monkeypatch.setattr(
        actionability_module,
        "_roll_forward",
        lambda env, model, obs, *, horizon: _outcome(0.0),
    )

    def branch_rollout(
        env,
        model,
        obs,
        *,
        horizon,
        baseline_action,
        team_response,
    ):
        observed_horizons.append(horizon)
        return _outcome(float(sum(team_response)))

    monkeypatch.setattr(
        actionability_module,
        "_roll_forward_with_team_override",
        branch_rollout,
    )
    branch_set = run_counterfactual_branches(
        env,
        Model(),
        {"obs": np.zeros(1)},
        candidate_step=7,
        response_horizon=19,
        utility_fn=resolve_utility("return"),
    )
    assert branch_set.candidate_step == 7
    assert branch_set.response_horizon == 19
    assert branch_set.baseline_team_response == (0, 1)
    assert len(branch_set.legal_team_responses) == 4
    assert len(branch_set.branches) == 3
    assert observed_horizons == [19, 19, 19]
