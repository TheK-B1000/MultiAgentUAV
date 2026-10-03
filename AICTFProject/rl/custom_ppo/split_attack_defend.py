"""Role-split splice/gating functions (defender-only ablation + dual-branch V1).

Two physically separate networks produce role-assigned actions. Historically
ATTACK was frozen (DEFEND_ATTACK_SPLIT_POLICY_A_V1); dual-branch
DUAL_BRANCH_ROLE_COMPOSITE_V1 makes both trainable with sample routing.
These helpers are deliberately pure so the "splice before env.step" and
role-gated log-prob contracts can be checked against known inputs. The SAME
functions are called from collector (collection) and minibatch_updater
(update).
"""
from __future__ import annotations

import math

import torch

__all__ = [
    "role_broadcast_mask",
    "splice_actions",
    "defend_gated_sum",
    "attack_gated_sum",
    "ceil_n_over_3",
]


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


def attack_gated_sum(per_agent_values: torch.Tensor, is_defend: torch.Tensor) -> torch.Tensor:
    """(B, N) -> (B,), summing only ATTACK-role agents (is_defend == False)."""
    if per_agent_values.shape != is_defend.shape:
        raise ValueError(
            f"per_agent_values shape {tuple(per_agent_values.shape)} != "
            f"is_defend shape {tuple(is_defend.shape)}"
        )
    return (per_agent_values * (~is_defend).float()).sum(dim=-1)


def ceil_n_over_3(n: int) -> int:
    """Locked k = ceil(N/3) for DUAL_BRANCH_ROLE_COMPOSITE_V1 (gives 1,2,2)."""
    if int(n) < 1:
        raise ValueError(f"team size N must be >= 1, got {n}")
    return int(math.ceil(int(n) / 3.0))


#: Allocator rules a frozen dual-branch spec may name in ALLOCATOR_RULE_locked.rule. Absent = the V1 rule.
CEIL_RULE = "ceil(N/3)"
NEAREST_THIRD_RULE = "max(1, floor(N/3 + 1/2))"


def nearest_third_k(n: int) -> int:
    """R1 rule k = max(1, floor(N/3 + 1/2)), exact integer form max(1, (2N + 3) // 6): 2v2 1, 4v4 1, 6v6 2."""
    if int(n) < 1:
        raise ValueError(f"team size N must be >= 1, got {n}")
    return max(1, (2 * int(n) + 3) // 6)


def k_for_rule(rule: str, n: int) -> int:
    """k under a named allocator rule; an unknown rule fails closed."""
    if rule == CEIL_RULE:
        return ceil_n_over_3(n)
    if rule == NEAREST_THIRD_RULE:
        return nearest_third_k(n)
    raise ValueError(f"unknown dual-branch allocator rule {rule!r}")


def spec_allocator_rule(spec: dict) -> str:
    """The frozen spec's ALLOCATOR_RULE_locked.rule; absent = the V1 rule ceil(N/3)."""
    return ((spec or {}).get("ALLOCATOR_RULE_locked") or {}).get("rule", CEIL_RULE)


def dual_branch_k(spec: dict, n: int) -> int:
    """k for a dual-branch run under its frozen spec (default ceil(N/3)). An unknown rule fails closed."""
    return k_for_rule(spec_allocator_rule(spec), n)
