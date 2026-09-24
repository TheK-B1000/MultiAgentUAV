"""DEFEND_ATTACK_SPLIT_POLICY_A_V1_SPEC: pure splice/gating functions.

Two physically separate networks: a frozen pi_A produces ATTACK-role
actions, a trainable pi_D produces DEFEND-role actions. These functions are
deliberately pure (no state, no I/O, no model access) so the "splice
before env.step, never overwrite after" and "DEFEND-gated log-prob/entropy"
contracts can be checked directly against known inputs, not just observed
indirectly through a full rollout. The SAME functions are called from both
rl.custom_ppo.rollout.collector (collection time) and
rl.custom_ppo.update.minibatch_updater (update time), so the gate a
minibatch trains through is provably the same gate that produced its
stored old log-prob.
"""
from __future__ import annotations

import torch

__all__ = ["role_broadcast_mask", "splice_actions", "defend_gated_sum"]


def role_broadcast_mask(is_defend: torch.Tensor, heads_per_agent: int) -> torch.Tensor:
    """(B, N) bool -> (B, N*heads_per_agent) bool, each agent's role bit
    repeated across its own action heads (macro, waypoint, ...)."""
    if is_defend.dim() != 2:
        raise ValueError(f"is_defend must be (B, N), got {tuple(is_defend.shape)}")
    return is_defend.unsqueeze(-1).expand(-1, -1, heads_per_agent).reshape(is_defend.shape[0], -1)


def splice_actions(
    trainable_actions: torch.Tensor,
    frozen_actions: torch.Tensor,
    is_defend: torch.Tensor,
    heads_per_agent: int,
) -> torch.Tensor:
    """Executed action tensor: DEFEND-role agent slots take the trainable
    model's own proposal verbatim (never overridden); ATTACK-role slots
    take the frozen model's proposal. This is the tensor that must be
    passed to env.step AND stored as the buffer's actions field -- never
    computed from one source and then partially overwritten afterward."""
    mask = role_broadcast_mask(is_defend, heads_per_agent)
    if mask.shape != trainable_actions.shape:
        raise ValueError(
            f"role mask shape {tuple(mask.shape)} != trainable actions shape "
            f"{tuple(trainable_actions.shape)}"
        )
    if mask.shape != frozen_actions.shape:
        raise ValueError(
            f"role mask shape {tuple(mask.shape)} != frozen actions shape "
            f"{tuple(frozen_actions.shape)}"
        )
    return torch.where(mask, trainable_actions, frozen_actions)


def defend_gated_sum(per_agent_values: torch.Tensor, is_defend: torch.Tensor) -> torch.Tensor:
    """(B, N) per-agent scalar (log-prob or entropy) -> (B,), summing only
    the DEFEND-role agents' contribution. ATTACK-role agents' values are
    excluded entirely, not merely down-weighted -- they were never the
    policy that produced the executed action for that slot."""
    if per_agent_values.shape != is_defend.shape:
        raise ValueError(
            f"per_agent_values shape {tuple(per_agent_values.shape)} != "
            f"is_defend shape {tuple(is_defend.shape)}"
        )
    return (per_agent_values * is_defend.float()).sum(dim=-1)
