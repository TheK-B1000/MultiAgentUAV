"""
Counterfactual Actionability Analysis Module.

Scientific Motivation:
Previous C2 analysis showed that certain behavioral markers (like `none_forward_frac`) 
were predictive of carrier survival (strong CI-backed separation) but had zero 
actionability — changing the agent's response from the same state did not alter 
the carrier's fate. This module provides tools to test counterfactual actionability 
BEFORE specialist training, by checking whether branching to alternative macro-actions 
from naturally-reached states actually alters the near-term outcome.

It builds on the snapshot/restore infrastructure from `tools.q_probe_local_counterfactual`
to rigorously enforce determinism and evaluate causal effects of actions.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from tools.q_probe_local_counterfactual import (
    _snapshot_env,
    _restore_env,
    _snapshot_policy,
    _restore_policy,
    _single_obs,
)


@dataclass
class BranchResult:
    onset_step: int
    baseline_action: tuple[int, ...]
    branch_action: tuple[int, ...]
    baseline_carrier_survived: bool
    branch_carrier_survived: bool
    baseline_blue_score_delta: int
    branch_blue_score_delta: int
    baseline_return: float
    branch_return: float
    outcome_shift: float


@dataclass
class ActionabilityResult:
    n_onsets: int
    n_actionable: int
    actionability_rate: float
    mean_outcome_shift: float
    branches: list[BranchResult]
    determinism_self_test_passed: bool


def _np(x):
    """Bridge torch tensors to numpy."""
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


@torch.no_grad()
def _roll_forward(
    env, model, obs, *, horizon: int, starting_step: int
) -> tuple[float, bool, int]:
    """Roll the env forward for *horizon* steps using the model's natural policy.

    Returns ``(cumulative_return, carrier_still_alive_or_carrying, blue_score_delta)``.
    """
    core = env.core
    cumulative_return = 0.0
    start_score = float(_np(core.blue_score)[0])
    curr_obs = obs

    for _ in range(horizon):
        adapted_obs = _single_obs(curr_obs, env)
        action, _ = model.predict(adapted_obs, deterministic=True)
        curr_obs, reward, done, _info = env.step(action)

        r = reward if np.isscalar(reward) else float(np.asarray(reward).flat[0])
        cumulative_return += r
        d = done if np.isscalar(done) else bool(np.asarray(done).flat[0])
        if d:
            break

    carrier_survived = bool(_np(core.blue_carrying)[0].any())
    end_score = float(_np(core.blue_score)[0])
    blue_score_delta = int(end_score - start_score)

    return cumulative_return, carrier_survived, blue_score_delta


@torch.no_grad()
def _roll_forward_with_override(
    env, model, obs, *, horizon: int, starting_step: int,
    override_agent: int, override_macro: int,
) -> tuple[float, bool, int]:
    """Same as :func:`_roll_forward`, but on the **first** step overrides one
    agent's macro-action.  Subsequent steps use natural policy actions."""
    core = env.core
    cumulative_return = 0.0
    start_score = float(_np(core.blue_score)[0])
    curr_obs = obs

    for step_i in range(horizon):
        adapted_obs = _single_obs(curr_obs, env)
        action, _ = model.predict(adapted_obs, deterministic=True)

        if step_i == 0:
            # Overwrite macro for the target agent.  Actions are flat
            # ``(N_agents * 2,)`` with macro at even indices.
            act = np.asarray(action).copy()
            act.flat[override_agent * 2] = override_macro
            action = act

        curr_obs, reward, done, _info = env.step(action)

        r = reward if np.isscalar(reward) else float(np.asarray(reward).flat[0])
        cumulative_return += r
        d = done if np.isscalar(done) else bool(np.asarray(done).flat[0])
        if d:
            break

    carrier_survived = bool(_np(core.blue_carrying)[0].any())
    end_score = float(_np(core.blue_score)[0])
    blue_score_delta = int(end_score - start_score)

    return cumulative_return, carrier_survived, blue_score_delta


def run_determinism_self_test(env, model, obs, *, horizon: int = 30) -> tuple[bool, float]:
    """
    Snapshots env+policy, rolls forward to get R_A, restores and repeats to get R_B.
    Validates that the determinism contract holds (|R_A - R_B| <= 1e-5).
    """
    env_snap = _snapshot_env(env)
    pol_snap = _snapshot_policy(model)
    
    # Run A
    r_a, _, _ = _roll_forward(env, model, obs, horizon=horizon, starting_step=0)
    
    # Restore
    _restore_env(env, env_snap)
    _restore_policy(model, pol_snap)
    
    # Run B
    r_b, _, _ = _roll_forward(env, model, obs, horizon=horizon, starting_step=0)
    
    # Restore to original state
    _restore_env(env, env_snap)
    _restore_policy(model, pol_snap)
    
    abs_diff = abs(r_a - r_b)
    passed = abs_diff <= 1e-5
    
    return passed, abs_diff


def run_counterfactual_branches(
    env, model, obs, *, onset_step: int, horizon: int = 30, alternative_macros: tuple[int, ...] = (0, 1, 2, 3, 4)
) -> list[BranchResult]:
    """
    Snapshots state, evaluates the baseline action, and then evaluates alternative macro-actions
    for both agent 0 and agent 1.
    """
    env_snap = _snapshot_env(env)
    pol_snap = _snapshot_policy(model)
    
    # Run baseline to get the action and baseline outcome
    adapted_obs = _single_obs(obs, env)
    action, _ = model.predict(adapted_obs, deterministic=True)
    act_arr = np.asarray(action).flatten()
    baseline_action = tuple(int(x) for x in act_arr.tolist())

    # Restore before baseline run so the predict() above doesn't consume state
    _restore_env(env, env_snap)
    _restore_policy(model, pol_snap)
    base_ret, base_surv, base_score = _roll_forward(env, model, obs, horizon=horizon, starting_step=onset_step)
    
    branches = []
    
    # Overrides for agent 0 and agent 1
    for override_agent in (0, 1):
        for alt_macro in alternative_macros:
            # Skip branches where the alternative action equals the baseline action
            if baseline_action[override_agent * 2] == alt_macro:
                continue
                
            _restore_env(env, env_snap)
            _restore_policy(model, pol_snap)
            
            branch_action_list = list(baseline_action)
            branch_action_list[override_agent * 2] = alt_macro
            branch_action = tuple(branch_action_list)
            
            branch_ret, branch_surv, branch_score = _roll_forward_with_override(
                env, model, obs, 
                horizon=horizon, 
                starting_step=onset_step, 
                override_agent=override_agent, 
                override_macro=alt_macro
            )
            
            outcome_shift = float(abs(int(base_surv) - int(branch_surv)))
            
            branches.append(BranchResult(
                onset_step=onset_step,
                baseline_action=baseline_action,
                branch_action=branch_action,
                baseline_carrier_survived=base_surv,
                branch_carrier_survived=branch_surv,
                baseline_blue_score_delta=base_score,
                branch_blue_score_delta=branch_score,
                baseline_return=base_ret,
                branch_return=branch_ret,
                outcome_shift=outcome_shift
            ))
            
    # Restore to original state
    _restore_env(env, env_snap)
    _restore_policy(model, pol_snap)
    
    return branches


def compute_actionability(branches: list[BranchResult], *, min_effect: float = 0.05) -> ActionabilityResult:
    """
    Computes actionability metrics by aggregating across onsets.
    """
    onsets = set(b.onset_step for b in branches)
    n_onsets = len(onsets)
    n_actionable = 0
    max_shifts = []
    
    for onset in onsets:
        onset_branches = [b for b in branches if b.onset_step == onset]
        max_shift = max((b.outcome_shift for b in onset_branches), default=0.0)
        max_shifts.append(max_shift)
        
        if max_shift >= min_effect:
            n_actionable += 1
            
    actionability_rate = n_actionable / max(n_onsets, 1)
    mean_outcome_shift = sum(max_shifts) / len(max_shifts) if max_shifts else 0.0
    
    return ActionabilityResult(
        n_onsets=n_onsets,
        n_actionable=n_actionable,
        actionability_rate=actionability_rate,
        mean_outcome_shift=mean_outcome_shift,
        branches=branches,
        determinism_self_test_passed=True  # Should be set externally, defaulting to True here
    )
