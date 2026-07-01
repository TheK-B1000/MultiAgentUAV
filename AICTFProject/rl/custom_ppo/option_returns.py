"""Pure option-level return computation for q_phi credit assignment.

This module owns the option-return recursion that previously lived inline in
``CustomPPOTrainer.collect_rollout``. Pulling it out has two benefits:

1. It can be unit-tested without launching the whole trainer (no env, no
   model, no optimizer).
2. The fragile per-env vectorization (every conditional must be dispatched
   via ``torch.where`` so the recursion works for ``n_envs > 1``) is in one
   place where future contributors can see and audit it.

The function is **pure** in the sense that it takes raw tensors and returns
raw tensors. Buffer-field registration, gamma sourcing, and config gating
remain the caller's concern.
"""

from __future__ import annotations

import torch


def compute_option_returns(
    *,
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    z_resampled: torch.Tensor,
    gamma: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute option-level Monte Carlo returns and advantages for q_phi(z|s).

    A latent sample ``z`` taken at time ``t`` is responsible for the
    *option window* — the run of consecutive steps that share that ``z``.
    The option return at step ``t`` is the discounted sum of rewards inside
    the window, with one bootstrap at the window boundary so the value
    target reflects what happens after the option ends.

    Boundary semantics (looking forward from step ``t``):

    - **Termination at t** (``terminated[t]``): the trajectory ended; the
      future contribution past step ``t`` is exactly zero.
    - **Truncation at t** (``truncated[t]``, but not terminated): bootstrap
      from ``next_values[t]`` (= ``V(s')``).
    - **Last buffered step** (``t == T - 1``) with no env-level done:
      bootstrap from ``next_values[t]`` since we cannot see the future.
    - **Option boundary at t+1** (``z_resampled[t + 1]``): a new ``z`` is
      sampled at the next step, so bootstrap from ``values[t + 1]``
      (= ``V(s_{t+1}, z_{t+1})``).
    - **Otherwise**: the option continues — fold in ``option_returns[t + 1]``.

    Every branch is dispatched via ``torch.where`` so the recursion is fully
    vectorized across the ``n_envs`` dimension. Branching on a multi-env
    boolean tensor with plain Python ``if`` would crash, which is precisely
    the bug this extraction protects against.

    Args:
        rewards: ``(T, N)`` per-step reward tensor.
        values: ``(T, N)`` V(s_t, z_t) from the critic.
        next_values: ``(T, N)`` V(s') (or 0 on terminated steps) — typically
            already adjusted by ``align_next_values_to_rollout_actions``.
        terminated: ``(T, N)`` boolean mask of env-level terminations.
        truncated: ``(T, N)`` boolean mask of env-level truncations.
        z_resampled: ``(T, N)`` boolean mask; True at step ``t`` iff a fresh
            ``z`` was sampled at the start of step ``t`` (vs. persisted).
        gamma: scalar discount factor.

    Returns:
        ``(option_returns, option_advantages)`` — both ``(T, N)`` float
        tensors. ``option_advantages = option_returns - values``.
    """
    if rewards.dim() != 2:
        raise ValueError(
            f"compute_option_returns expects 2-D (T, N) tensors; got rewards shape {tuple(rewards.shape)}."
        )
    T = int(rewards.shape[0])
    terminated_b = terminated.bool()
    truncated_b = truncated.bool()
    z_resampled_b = z_resampled.bool()

    option_returns = torch.zeros_like(rewards)
    zero_row = torch.zeros_like(rewards[0])
    gamma_f = float(gamma)

    for t in reversed(range(T)):
        done_t = terminated_b[t] | truncated_b[t]
        done_next = torch.where(terminated_b[t], zero_row, next_values[t])
        if t == T - 1:
            carry = next_values[t]
        else:
            carry = torch.where(
                z_resampled_b[t + 1],
                values[t + 1],
                option_returns[t + 1],
            )
        next_val = torch.where(done_t, done_next, carry)
        option_returns[t] = rewards[t] + gamma_f * next_val

    option_advantages = option_returns - values
    return option_returns, option_advantages


def compute_router_returns(
    *,
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    router_decision_valid: torch.Tensor,
    gamma: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Opportunity-level returns and advantages for the V6I7 recurrent router.

    Identical in structure to :func:`compute_option_returns` but gates
    interval boundaries on ``router_decision_valid`` (True only at actual
    router decision steps, never at forced-z or continuation steps) rather
    than ``z_resampled``.

    The baseline used is the stored scalar ``values[t]`` (= V(s_t, z_t) from
    the critic), NOT the fully-marginalized ``V^z(s)=Σ_z q·Q``.  This is
    correct for V6I7-A plumbing validation; the marginalized baseline can
    replace it once the forward pass is verified.

    Args:
        rewards:              ``(T, N)`` per-step reward tensor.
        values:               ``(T, N)`` V(s_t, z_t) from the critic.
        next_values:          ``(T, N)`` post-step bootstrap values.
        terminated:           ``(T, N)`` boolean termination mask.
        truncated:            ``(T, N)`` boolean truncation mask.
        router_decision_valid: ``(T, N)`` boolean; True at actual router
            opportunity indices (``z_resampled & ~z_forced``).
        gamma:                scalar discount factor.

    Returns:
        ``(router_returns, router_advantages)`` — both ``(T, N)`` float
        tensors. ``router_advantages = router_returns - values``.
    """
    if rewards.dim() != 2:
        raise ValueError(
            f"compute_router_returns expects 2-D (T, N) tensors; got rewards shape {tuple(rewards.shape)}."
        )
    T = int(rewards.shape[0])
    terminated_b = terminated.bool()
    truncated_b = truncated.bool()
    rdv_b = router_decision_valid.bool()

    router_returns = torch.zeros_like(rewards)
    zero_row = torch.zeros_like(rewards[0])
    gamma_f = float(gamma)

    for t in reversed(range(T)):
        done_t = terminated_b[t] | truncated_b[t]
        done_next = torch.where(terminated_b[t], zero_row, next_values[t])
        if t == T - 1:
            carry = next_values[t]
        else:
            carry = torch.where(
                rdv_b[t + 1],
                values[t + 1],          # bootstrap at next decision step
                router_returns[t + 1],  # fold into running return
            )
        next_val = torch.where(done_t, done_next, carry)
        router_returns[t] = rewards[t] + gamma_f * next_val

    router_advantages = router_returns - values
    return router_returns, router_advantages


__all__ = ["compute_option_returns", "compute_router_returns"]
