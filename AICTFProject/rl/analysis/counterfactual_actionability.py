"""C3 counterfactual controllability and commitment-fork helpers.

Carrier-pressure events are anchors only. A commitment fork is the earliest
naturally reached state in the bounded backward trace where multiple legal
team responses remain, the state is not already doomed, and a one-decision
team-response intervention improves task utility over the natural G0 response
through the frozen response horizon.

This module implements a controllability screen only. It makes no strategy,
latent-necessity, response-oracle, or routing claim.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from tools.q_probe_local_counterfactual import (
    _restore_env,
    _restore_policy,
    _single_obs,
    _snapshot_env,
    _snapshot_policy,
)


NO_COMMITMENT_FORK = "NO_COMMITMENT_FORK"
QUALIFIED_COMMITMENT_FORK = "QUALIFIED_COMMITMENT_FORK"


@dataclass(frozen=True)
class RolloutOutcome:
    cumulative_return: float
    carrier_survived: bool
    blue_score_delta: int
    steps_executed: int


UtilityFn = Callable[[RolloutOutcome], float]


@dataclass(frozen=True)
class BranchResult:
    candidate_step: int
    response_horizon: int
    baseline_action: tuple[int, ...]
    branch_action: tuple[int, ...]
    team_response: tuple[int, ...]
    baseline_outcome: RolloutOutcome
    branch_outcome: RolloutOutcome
    baseline_utility: float
    branch_utility: float
    utility_improvement: float


@dataclass(frozen=True)
class CounterfactualBranchSet:
    candidate_step: int
    response_horizon: int
    baseline_action: tuple[int, ...]
    baseline_team_response: tuple[int, ...]
    legal_team_responses: tuple[tuple[int, ...], ...]
    baseline_outcome: RolloutOutcome
    baseline_utility: float
    branches: tuple[BranchResult, ...]


@dataclass(frozen=True)
class ActionabilityResult:
    candidate_step: int
    response_horizon: int
    n_legal_team_responses: int
    baseline_utility: float
    best_expected_utility: float
    max_expected_utility_improvement: float
    best_team_response: tuple[int, ...] | None
    effectively_doomed: bool
    has_persistent_utility_divergence: bool
    is_actionable: bool
    branches: tuple[BranchResult, ...]


@dataclass(frozen=True)
class CommitmentForkResult:
    pressure_step: int
    trace_start_step: int
    candidate_steps: tuple[int, ...]
    episode_status: str
    fork_step: int | None
    fork_evaluation: ActionabilityResult | None
    evaluations: tuple[ActionabilityResult, ...]
    science_scope: str = "CONTROLLABILITY_SCREEN_ONLY"


def _np(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def resolve_utility(name: str) -> UtilityFn:
    """Resolve only already-recorded task outcomes; the frozen contract picks one."""
    normalized = str(name).strip().lower()
    if normalized == "return":
        return lambda outcome: float(outcome.cumulative_return)
    if normalized == "carrier_survival":
        return lambda outcome: float(outcome.carrier_survived)
    if normalized == "blue_score_delta":
        return lambda outcome: float(outcome.blue_score_delta)
    raise ValueError(
        f"Unsupported frozen C3 utility {name!r}; expected return, "
        "carrier_survival, or blue_score_delta"
    )


def enumerate_legal_team_responses(core) -> tuple[tuple[int, ...], ...]:
    """Enumerate the Cartesian product of authoritative per-agent macro masks."""
    mask = _np(core._build_action_mask(side="blue"))
    if mask.ndim != 2 or mask.shape[0] != 1:
        raise ValueError(f"C3 requires B=1 action mask, got shape={mask.shape}")

    alive = _np(core.blue_alive)
    if alive.ndim != 2 or alive.shape[0] != 1:
        raise ValueError(f"C3 requires B=1 blue_alive, got shape={alive.shape}")

    n_agents = int(alive.shape[1])
    n_macros = int(core.cfg.n_macros)
    n_targets = int(core.cfg.n_targets)
    per_agent = mask.reshape(1, n_agents, n_macros + n_targets)[0]
    legal_macros = []
    for agent_i in range(n_agents):
        macros = tuple(
            int(macro_i)
            for macro_i in np.flatnonzero(per_agent[agent_i, :n_macros] > 0.0)
        )
        if not macros:
            raise RuntimeError(f"authoritative mask exposes no macro for blue agent {agent_i}")
        legal_macros.append(macros)
    return tuple(tuple(int(m) for m in response) for response in itertools.product(*legal_macros))


def _team_response_from_action(action: tuple[int, ...], n_agents: int) -> tuple[int, ...]:
    if len(action) != n_agents * 2:
        raise ValueError(
            f"expected flattened [macro,target] action for {n_agents} agents, got {len(action)} values"
        )
    return tuple(int(action[agent_i * 2]) for agent_i in range(n_agents))


def _action_with_team_response(
    baseline_action: tuple[int, ...], team_response: tuple[int, ...]
) -> np.ndarray:
    n_agents = len(team_response)
    if len(baseline_action) != n_agents * 2:
        raise ValueError("team-response width does not match flattened action")
    action = np.asarray(baseline_action, dtype=np.int64).copy()
    for agent_i, macro in enumerate(team_response):
        action[agent_i * 2] = int(macro)
    return action


def _predict_action(model, obs, env) -> np.ndarray:
    """Predict with CNN-channel adaptation for G0 checkpoints trained at 7ch."""
    from experiments.eval_v6i9_map_awareness import _adapt_obs_for_policy

    single = _single_obs(obs, env)
    try:
        adapted = _adapt_obs_for_policy(single, model)
    except (AttributeError, KeyError, ValueError, TypeError):
        # Unit-test mocks and non-CNN wrappers have no conv0 to introspect.
        adapted = single
    action, _ = model.predict(adapted, deterministic=True)
    return np.asarray(action)


@torch.no_grad()
def _roll_forward(env, model, obs, *, horizon: int) -> RolloutOutcome:
    core = env.core
    cumulative_return = 0.0
    start_score = float(_np(core.blue_score)[0])
    curr_obs = obs
    steps_executed = 0

    for _ in range(int(horizon)):
        action = _predict_action(model, curr_obs, env)
        curr_obs, reward, done, _info = env.step(action)
        cumulative_return += float(np.asarray(reward).reshape(-1)[0])
        steps_executed += 1
        if bool(np.asarray(done).reshape(-1)[0]):
            break

    return RolloutOutcome(
        cumulative_return=float(cumulative_return),
        carrier_survived=bool(_np(core.blue_carrying)[0].any()),
        blue_score_delta=int(float(_np(core.blue_score)[0]) - start_score),
        steps_executed=steps_executed,
    )


@torch.no_grad()
def _roll_forward_with_team_override(
    env,
    model,
    obs,
    *,
    horizon: int,
    baseline_action: tuple[int, ...],
    team_response: tuple[int, ...],
) -> RolloutOutcome:
    """Force one legal team macro response, then return to natural G0."""
    core = env.core
    cumulative_return = 0.0
    start_score = float(_np(core.blue_score)[0])
    curr_obs = obs
    steps_executed = 0

    for response_step in range(int(horizon)):
        if response_step == 0:
            action = _action_with_team_response(baseline_action, team_response)
        else:
            action = _predict_action(model, curr_obs, env)
        curr_obs, reward, done, _info = env.step(action)
        cumulative_return += float(np.asarray(reward).reshape(-1)[0])
        steps_executed += 1
        if bool(np.asarray(done).reshape(-1)[0]):
            break

    return RolloutOutcome(
        cumulative_return=float(cumulative_return),
        carrier_survived=bool(_np(core.blue_carrying)[0].any()),
        blue_score_delta=int(float(_np(core.blue_score)[0]) - start_score),
        steps_executed=steps_executed,
    )


def run_determinism_self_test(env, model, obs, *, horizon: int) -> tuple[bool, float]:
    env_snap = _snapshot_env(env)
    policy_snap = _snapshot_policy(model)
    first = _roll_forward(env, model, obs, horizon=horizon)
    _restore_env(env, env_snap)
    _restore_policy(model, policy_snap)
    second = _roll_forward(env, model, obs, horizon=horizon)
    _restore_env(env, env_snap)
    _restore_policy(model, policy_snap)
    difference = abs(first.cumulative_return - second.cumulative_return)
    return difference <= 1e-5, float(difference)


def run_counterfactual_branches(
    env,
    model,
    obs,
    *,
    candidate_step: int,
    response_horizon: int,
    utility_fn: UtilityFn,
) -> CounterfactualBranchSet:
    """Evaluate every legal team response from one naturally reached state."""
    if response_horizon <= 0:
        raise ValueError("response_horizon must be positive")

    env_snap = _snapshot_env(env)
    policy_snap = _snapshot_policy(model)
    action = _predict_action(model, obs, env)
    baseline_action = tuple(int(x) for x in np.asarray(action).reshape(-1).tolist())
    legal_team_responses = enumerate_legal_team_responses(env.core)
    n_agents = len(_np(env.core.blue_alive)[0])
    baseline_team_response = _team_response_from_action(baseline_action, n_agents)
    if baseline_team_response not in legal_team_responses:
        raise RuntimeError(
            "natural G0 action is not legal under the authoritative environment mask"
        )

    _restore_env(env, env_snap)
    _restore_policy(model, policy_snap)
    baseline_outcome = _roll_forward(env, model, obs, horizon=response_horizon)
    baseline_utility = float(utility_fn(baseline_outcome))

    branches = []
    for team_response in legal_team_responses:
        if team_response == baseline_team_response:
            continue
        _restore_env(env, env_snap)
        _restore_policy(model, policy_snap)
        branch_outcome = _roll_forward_with_team_override(
            env,
            model,
            obs,
            horizon=response_horizon,
            baseline_action=baseline_action,
            team_response=team_response,
        )
        branch_utility = float(utility_fn(branch_outcome))
        branch_action = tuple(
            int(x)
            for x in _action_with_team_response(baseline_action, team_response).tolist()
        )
        branches.append(
            BranchResult(
                candidate_step=int(candidate_step),
                response_horizon=int(response_horizon),
                baseline_action=baseline_action,
                branch_action=branch_action,
                team_response=team_response,
                baseline_outcome=baseline_outcome,
                branch_outcome=branch_outcome,
                baseline_utility=baseline_utility,
                branch_utility=branch_utility,
                utility_improvement=branch_utility - baseline_utility,
            )
        )

    _restore_env(env, env_snap)
    _restore_policy(model, policy_snap)
    return CounterfactualBranchSet(
        candidate_step=int(candidate_step),
        response_horizon=int(response_horizon),
        baseline_action=baseline_action,
        baseline_team_response=baseline_team_response,
        legal_team_responses=legal_team_responses,
        baseline_outcome=baseline_outcome,
        baseline_utility=baseline_utility,
        branches=tuple(branches),
    )


def compute_actionability(
    branch_set: CounterfactualBranchSet,
    *,
    delta: float,
    doomed_utility_threshold: float,
) -> ActionabilityResult:
    """Apply R2-R4 to one candidate state using improvement over natural G0."""
    best_branch = max(
        branch_set.branches,
        key=lambda branch: branch.branch_utility,
        default=None,
    )
    best_utility = max(
        [branch_set.baseline_utility]
        + [branch.branch_utility for branch in branch_set.branches]
    )
    max_improvement = (
        0.0 if best_branch is None else best_branch.branch_utility - branch_set.baseline_utility
    )
    enough_legal_responses = len(branch_set.legal_team_responses) >= 2
    effectively_doomed = best_utility <= float(doomed_utility_threshold)
    persistent_divergence = (
        max_improvement > float(delta)
        or math.isclose(max_improvement, float(delta), rel_tol=0.0, abs_tol=1e-12)
    )
    is_actionable = (
        enough_legal_responses
        and not effectively_doomed
        and persistent_divergence
    )
    return ActionabilityResult(
        candidate_step=branch_set.candidate_step,
        response_horizon=branch_set.response_horizon,
        n_legal_team_responses=len(branch_set.legal_team_responses),
        baseline_utility=branch_set.baseline_utility,
        best_expected_utility=float(best_utility),
        max_expected_utility_improvement=float(max_improvement),
        best_team_response=None if best_branch is None else best_branch.team_response,
        effectively_doomed=bool(effectively_doomed),
        has_persistent_utility_divergence=bool(persistent_divergence),
        is_actionable=bool(is_actionable),
        branches=branch_set.branches,
    )


def backward_trace_steps(pressure_step: int, t_trace: int) -> tuple[int, ...]:
    """Return strictly upstream candidate decisions in chronological order."""
    if pressure_step <= 0:
        return ()
    if t_trace <= 0:
        raise ValueError("t_trace must be positive")
    start = max(0, int(pressure_step) - int(t_trace))
    return tuple(range(start, int(pressure_step)))


def find_earliest_commitment_fork(
    *,
    pressure_step: int,
    t_trace: int,
    evaluate_candidate: Callable[[int], ActionabilityResult],
) -> CommitmentForkResult:
    """Evaluate upstream states chronologically and retain the first R1-R4 pass."""
    candidates = backward_trace_steps(pressure_step, t_trace)
    evaluations = []
    for candidate_step in candidates:
        evaluation = evaluate_candidate(candidate_step)
        evaluations.append(evaluation)
        if evaluation.is_actionable:
            return CommitmentForkResult(
                pressure_step=int(pressure_step),
                trace_start_step=candidates[0] if candidates else int(pressure_step),
                candidate_steps=candidates,
                episode_status=QUALIFIED_COMMITMENT_FORK,
                fork_step=int(candidate_step),
                fork_evaluation=evaluation,
                evaluations=tuple(evaluations),
            )
    return CommitmentForkResult(
        pressure_step=int(pressure_step),
        trace_start_step=candidates[0] if candidates else int(pressure_step),
        candidate_steps=candidates,
        episode_status=NO_COMMITMENT_FORK,
        fork_step=None,
        fork_evaluation=None,
        evaluations=tuple(evaluations),
    )
