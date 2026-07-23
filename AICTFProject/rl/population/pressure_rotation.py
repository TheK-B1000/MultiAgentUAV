"""Pressure rotation for V6I24 population training.

Periodically evaluates the payoff matrix M[k, c] = E[R | π_k, c] across
the population and rotates the coverage member's opponent weights toward
under-covered cells.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from rl.population.population_member import PopulationMember


def compute_payoff_redundancy(
    payoff_matrix: np.ndarray,
) -> np.ndarray:
    """Compute pairwise payoff-row redundancy.
    
    d(i,j) = (1/|C|) * sum_c |M[i,c] - M[j,c]|
    
    Args:
        payoff_matrix: Shape (K, C) matrix of expected returns.
    
    Returns:
        Shape (K, K) symmetric distance matrix.
    """
    k = payoff_matrix.shape[0]
    c = payoff_matrix.shape[1]
    distances = np.zeros((k, k))
    for i in range(k):
        for j in range(i + 1, k):
            d = float(np.mean(np.abs(payoff_matrix[i] - payoff_matrix[j])))
            distances[i, j] = d
            distances[j, i] = d
    return distances


def identify_coverage_gaps(
    payoff_matrix: np.ndarray,
    context_labels: Optional[list[str]] = None,
) -> dict[str, float]:
    """Identify contexts where the population lacks coverage.
    
    A context is 'uncovered' if the best policy's return is significantly
    below the oracle (best-per-context) return, or if all policies perform
    similarly (low redundancy).
    
    Args:
        payoff_matrix: Shape (K, C) matrix of expected returns.
        context_labels: Optional labels for each context column.
    
    Returns:
        Dictionary mapping context labels to coverage gap scores.
        Higher scores indicate more need for coverage attention.
    """
    k, c = payoff_matrix.shape
    if context_labels is None:
        context_labels = [f"c{i}" for i in range(c)]
    
    best_per_context = np.max(payoff_matrix, axis=0)  # (C,)
    mean_per_context = np.mean(payoff_matrix, axis=0)  # (C,)
    
    # Gap = how much the mean underperforms the best
    # Normalized by the range to make it comparable across contexts
    range_per_context = np.ptp(payoff_matrix, axis=0)  # (C,)
    safe_range = np.maximum(range_per_context, 1e-8)
    
    gap_scores = (best_per_context - mean_per_context) / safe_range
    
    # Invert: high gap = well-covered (one specialist is good)
    # Low gap = poorly covered (all similar, or all bad)
    # We want to focus on contexts where NO policy does well
    coverage_need = 1.0 - gap_scores  # Higher = more coverage needed
    
    return {label: float(score) for label, score in zip(context_labels, coverage_need)}


def rotate_pressures(
    members: list[PopulationMember],
    payoff_matrix: np.ndarray,
    context_to_opponent: dict[int, str],
    coverage_member_id: int = 3,
    temperature: float = 1.0,
) -> None:
    """Update coverage member's opponent weights based on payoff gaps.
    
    Computes softmax over coverage-need scores to produce opponent weights
    that focus the coverage member's training on under-performing contexts.
    
    Args:
        members: All population members.
        payoff_matrix: Shape (K, C) payoff matrix.
        context_to_opponent: Maps context column index to opponent tag.
        coverage_member_id: Which member to rotate (default: member 3).
        temperature: Softmax temperature. Lower = more focused.
    """
    coverage_member = None
    for m in members:
        if m.member_id == coverage_member_id:
            coverage_member = m
            break
    if coverage_member is None:
        return
    
    # Compute coverage need per context
    gaps = identify_coverage_gaps(payoff_matrix)
    
    # Map contexts to opponent tags in member's pool
    tag_scores: dict[str, float] = {}
    for c_idx, tag in context_to_opponent.items():
        label = f"c{c_idx}"
        if label in gaps and tag in coverage_member.config.opponent_tags:
            tag_scores.setdefault(tag, 0.0)
            tag_scores[tag] = max(tag_scores[tag], gaps[label])
    
    if not tag_scores:
        return
    
    # Softmax to get weights
    tags = list(coverage_member.config.opponent_tags)
    scores = np.array([tag_scores.get(t, 0.5) for t in tags], dtype=np.float64)
    scores = scores / max(float(temperature), 1e-8)
    exp_scores = np.exp(scores - np.max(scores))  # numerically stable softmax
    weights = exp_scores / exp_scores.sum()
    
    # Update the member's config
    coverage_member.config = PopulationMemberConfig(
        member_id=coverage_member.config.member_id,
        label=coverage_member.config.label,
        opponent_tags=coverage_member.config.opponent_tags,
        opponent_weights=tuple(float(w) for w in weights),
        map_pool=coverage_member.config.map_pool,
        seed_offset=coverage_member.config.seed_offset,
    )
    
    print(f"[PressureRotation] Updated member[{coverage_member_id}] weights:")
    for tag, weight in zip(tags, weights):
        print(f"  {tag}: {weight:.3f}")
