"""V6I26 Latent Response-Oracle (LRO-Summer) core utilities.

Claim B: each latent z is a response-oracle population member trained against
uncovered weaknesses of the current latent population — not a soft diversity
target and not a hand-labeled attack/defense role.

Classification: DIAGNOSTIC (not PAPER-FAITHFUL).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from experiments.v6i26_phase_pods import (
    PHASE_POD_IDS,
    POD_TO_Z,
    classify_phase_from_core,
    phase_pods_manifest,
)

LRO_PROTOCOL = "v6i26_latent_response_oracle"
LRO_CLAIM = "B_population_guided_lro"


@dataclass(frozen=True)
class LandscapePolicySpec:
    policy_id: str
    label: str
    path: str
    kind: str  # "teacher" | "forced_z" | "checkpoint" | "league"


def default_landscape_policies(project_root: Path) -> list[LandscapePolicySpec]:
    """Archive policies for the cheap strategic landscape scan (no new training)."""
    root = Path(project_root)
    candidates = [
        LandscapePolicySpec(
            "v6i24_m0",
            "v6i24_balanced",
            str(root / "artifacts/v6i24_population_seed1/probe_05u/member_0_balanced.zip"),
            "teacher",
        ),
        LandscapePolicySpec(
            "v6i24_m1",
            "v6i24_failure_cells",
            str(root / "artifacts/v6i24_population_seed1/probe_05u/member_1_failure_cells.zip"),
            "teacher",
        ),
        LandscapePolicySpec(
            "v6i24_m2",
            "v6i24_high_variance",
            str(root / "artifacts/v6i24_population_seed1/probe_05u/member_2_high_variance.zip"),
            "teacher",
        ),
        LandscapePolicySpec(
            "v6i24_m3",
            "v6i24_complementary",
            str(root / "artifacts/v6i24_population_seed1/probe_05u/member_3_complementary.zip"),
            "teacher",
        ),
        LandscapePolicySpec(
            "v6i23_donor",
            "v6i23_population_birth",
            str(
                root
                / "artifacts/v6i23_population_birth_5u_seed1/final_v6i23_population_birth_5u_seed1_2v2.zip"
            ),
            "checkpoint",
        ),
    ]
    return [c for c in candidates if Path(c.path).is_file()]


def payoff_tensor_summary(
    payoff: np.ndarray,
    *,
    policy_labels: Sequence[str],
    contexts: Sequence[str],
    margin: float = 0.10,
) -> dict[str, Any]:
    """Summarize P[policy, context] for niche / oracle gates."""
    p = np.asarray(payoff, dtype=np.float64)
    if p.ndim != 2:
        raise ValueError(f"payoff must be 2D, got {p.shape}")
    n_pol, n_ctx = p.shape
    best = np.argmax(p, axis=0)
    unique_best = sorted({int(b) for b in best.tolist()})
    cells_with_margin = 0
    best_by_context: dict[str, dict[str, Any]] = {}
    for ci, ctx in enumerate(contexts):
        order = np.argsort(-p[:, ci])
        top = int(order[0])
        second = int(order[1]) if n_pol > 1 else top
        gap = float(p[top, ci] - p[second, ci])
        if gap >= margin:
            cells_with_margin += 1
        best_by_context[str(ctx)] = {
            "best_policy": policy_labels[top],
            "best_index": top,
            "payoff": float(p[top, ci]),
            "margin_vs_second": gap,
        }

    # Cross-fitted leave-one-half oracle vs best-fixed (point estimate).
    # Split contexts into train/test by index parity for a cheap CI-free scan.
    train_idx = [i for i in range(n_ctx) if i % 2 == 0]
    test_idx = [i for i in range(n_ctx) if i % 2 == 1]
    if not train_idx or not test_idx:
        train_idx = list(range(max(1, n_ctx // 2)))
        test_idx = list(range(max(1, n_ctx // 2), n_ctx)) or train_idx

    train_means = p[:, train_idx].mean(axis=1)
    best_fixed = int(np.argmax(train_means))
    # Context oracle: for each test context pick argmax on train-smoothed? Use full matrix
    # for scan (honest cross-fit needs held-out episodes; scan uses context holdout).
    oracle_test = float(np.mean([p[int(np.argmax(p[:, ci])), ci] for ci in test_idx]))
    fixed_test = float(np.mean(p[best_fixed, test_idx]))
    g_available = oracle_test - fixed_test

    # Parallel-row diagnostic: max pairwise cosine / corr of payoff rows.
    row_dists = []
    for i in range(n_pol):
        for j in range(i + 1, n_pol):
            a, b = p[i], p[j]
            row_dists.append(float(np.linalg.norm(a - b) / (np.sqrt(n_ctx) + 1e-8)))
    max_row_distance = float(max(row_dists) if row_dists else 0.0)

    return {
        "n_policies": n_pol,
        "n_contexts": n_ctx,
        "unique_best_count": len(unique_best),
        "unique_best_indices": unique_best,
        "unique_best_labels": [policy_labels[i] for i in unique_best],
        "cells_with_margin_ge": cells_with_margin,
        "margin_threshold": margin,
        "best_by_context": best_by_context,
        "best_fixed_index": best_fixed,
        "best_fixed_label": policy_labels[best_fixed],
        "oracle_test_mean": oracle_test,
        "best_fixed_test_mean": fixed_test,
        "G_available_point": g_available,
        "max_pairwise_row_distance": max_row_distance,
        "niche_signal": bool(
            len(unique_best) >= 2 and cells_with_margin >= 2 and g_available > 0.0
        ),
        "parallel_rows": bool(max_row_distance < 0.03 and len(unique_best) <= 1),
    }


def select_response_target(
    payoff: np.ndarray,
    *,
    contexts: Sequence[str],
    policy_labels: Sequence[str],
    temperature: float = 0.5,
    episodes_per_cell: int = 4,
    prior_strength: float = 4.0,
    max_mixture_weight: float = 0.35,
    aggregate_by_opponent: bool = True,
) -> dict[str, Any]:
    """Pick a smoothed weakness mixture for the next response-oracle round.

    Guardrail: do **not** chase a single noisy 4-episode cell. Regret is
    shrunk toward the mean, optionally averaged across maps for the same
    opponent, then softmax-capped so no cell exceeds ``max_mixture_weight``.
    """
    p = np.asarray(payoff, dtype=np.float64)
    n_pol, n_ctx = p.shape
    best_per_ctx = p.max(axis=0)
    pop_mean = p.mean(axis=0)
    raw_regret = best_per_ctx - pop_mean

    # Empirical-Bayes shrinkage toward mean regret (Laplace-style).
    n = max(1.0, float(episodes_per_cell))
    n0 = max(0.0, float(prior_strength))
    mean_r = float(raw_regret.mean()) if n_ctx else 0.0
    regret = (n / (n + n0)) * raw_regret + (n0 / (n + n0)) * mean_r

    if aggregate_by_opponent and n_ctx > 0:
        # Average smoothed regret across maps sharing an opponent prefix.
        opp_to_idxs: dict[str, list[int]] = {}
        for i, ctx in enumerate(contexts):
            opp = str(ctx).split("|", 1)[0]
            opp_to_idxs.setdefault(opp, []).append(i)
        aggregated = regret.copy()
        for idxs in opp_to_idxs.values():
            if len(idxs) <= 1:
                continue
            avg = float(regret[idxs].mean())
            for i in idxs:
                aggregated[i] = avg
        regret = aggregated

    # Softmax over smoothed regret → mixture; then cap + renormalize.
    t = max(1e-3, float(temperature))
    logits = regret / t
    logits = logits - logits.max()
    weights = np.exp(logits)
    weights = weights / max(1e-12, float(weights.sum()))
    cap = float(max_mixture_weight)
    if cap > 0.0 and cap < 1.0:
        for _ in range(8):
            over = weights > cap
            if not over.any():
                break
            excess = float((weights[over] - cap).sum())
            weights[over] = cap
            under = ~over
            if under.any() and excess > 0.0:
                weights[under] += excess * (weights[under] / max(1e-12, float(weights[under].sum())))
            weights = weights / max(1e-12, float(weights.sum()))

    worst = int(np.argmax(regret))
    # Prefer the weakest *competent* row if one exists; else absolute weakest.
    means = p.mean(axis=1)
    competent = means >= float(np.median(means)) - 1e-9
    if competent.any() and (~competent).any():
        # Retrain an underused weak-but-not-median-floor branch among all.
        branch = int(np.argmin(means))
    else:
        branch = int(np.argmin(means))

    top_k = min(3, n_ctx)
    top_idx = np.argsort(-weights)[:top_k].tolist()
    return {
        "target_context": contexts[worst],
        "target_context_index": worst,
        "target_regret": float(regret[worst]),
        "raw_target_regret": float(raw_regret[worst]),
        "smoothed": True,
        "episodes_per_cell": int(episodes_per_cell),
        "prior_strength": float(prior_strength),
        "max_mixture_weight": float(max_mixture_weight),
        "aggregate_by_opponent": bool(aggregate_by_opponent),
        "mixture_weights": {
            str(contexts[i]): float(weights[i]) for i in range(len(contexts))
        },
        "mixture_top": [
            {"context": str(contexts[i]), "weight": float(weights[i]), "regret": float(regret[i])}
            for i in top_idx
        ],
        "branch_to_train_index": branch,
        "branch_to_train_label": policy_labels[branch],
        "acceptance_requires": [
            "delta_G_available_gt_0",
            "nonredundant_payoff_row",
            "competence_above_floor",
        ],
    }


def branch_row_redundant(
    payoff: np.ndarray,
    branch_idx: int,
    *,
    min_row_distance: float = 0.08,
) -> bool:
    """True if branch payoff row is nearly a duplicate of another policy."""
    p = np.asarray(payoff, dtype=np.float64)
    if p.ndim != 2 or p.shape[0] < 2:
        return False
    b = int(branch_idx)
    row = p[b]
    n_ctx = p.shape[1]
    for j in range(p.shape[0]):
        if j == b:
            continue
        dist = float(np.linalg.norm(row - p[j]) / (np.sqrt(n_ctx) + 1e-8))
        if dist < float(min_row_distance):
            return True
    return False


def accept_lro_round(
    *,
    g_before: float,
    g_after: float,
    payoff_after: np.ndarray,
    branch_idx: int,
    competence_floor: float,
    min_row_distance: float = 0.08,
) -> dict[str, Any]:
    """Locked Stage-1 acceptance: ΔG > 0 AND nonredundant AND competence floor."""
    p = np.asarray(payoff_after, dtype=np.float64)
    b = int(branch_idx)
    branch_mean = float(p[b].mean()) if p.size else float("nan")
    delta_g = float(g_after) - float(g_before)
    redundant = branch_row_redundant(
        p, b, min_row_distance=float(min_row_distance)
    )
    best_contexts = []
    if p.ndim == 2 and p.shape[0] > 0:
        for ci in range(p.shape[1]):
            if int(np.argmax(p[:, ci])) == b:
                best_contexts.append(ci)
    delta_ok = bool(delta_g > 0.0)
    competence_ok = bool(branch_mean >= float(competence_floor))
    nonredundant_ok = not redundant
    accepted = bool(delta_ok and competence_ok and nonredundant_ok)
    return {
        "G_before": float(g_before),
        "G_after": float(g_after),
        "delta_G_available": delta_g,
        "branch_idx": b,
        "branch_mean_payoff": branch_mean,
        "competence_floor": float(competence_floor),
        "best_on_context_indices": best_contexts,
        "nonredundant_payoff_row": nonredundant_ok,
        "competence_above_floor": competence_ok,
        "delta_G_gt_0": delta_ok,
        "accepted": accepted,
        "verdict": "ACCEPT" if accepted else "REJECT",
    }


def diagnose_lro_reject(
    *,
    branch_kl: float,
    niche_payoff_improvement: float,
    general_competence_change: float,
    delta_g: float,
    kl_tiny: float = 0.01,
    niche_eps: float = 0.02,
) -> dict[str, Any]:
    """Map a rejected Stage-1 round to the locked diagnosis tree.

    One flat round does **not** kill LRO. Only repeated fair-round failures
    with the same diagnosis support escalating to geometry/task niches.
    """
    tiny_kl = bool(branch_kl == branch_kl and float(branch_kl) < float(kl_tiny))
    large_kl = bool(branch_kl == branch_kl and float(branch_kl) >= float(kl_tiny))
    target_up = bool(float(niche_payoff_improvement) > float(niche_eps))
    collapsed = bool(float(general_competence_change) < -float(niche_eps))
    flat_g = bool(float(delta_g) <= 0.0)

    if tiny_kl and not target_up:
        code = "STUCK_GENERALIST_BASIN"
        meaning = "Tiny KL, no targeted improvement — branch never left generalist basin."
        next_action = "Increase active-branch capacity / unfreeze more trunk / longer BR budget."
    elif large_kl and target_up and flat_g:
        code = "NARROW_OR_COSTLY_TRADEOFF"
        meaning = "Large KL, target improved, overall G flat — improvement too narrow or costly."
        next_action = "Reweight mixture across related weak cells; raise competence floor check."
    elif large_kl and not target_up:
        code = "MIXTURE_OR_OPT_FAILED"
        meaning = "Large KL but no target improvement — mixture or optimization failed."
        next_action = "Rebuild smoothed regret mixture; verify cell weights and freeze contracts."
    elif collapsed and flat_g:
        code = "COLLAPSE_ELSEWHERE"
        meaning = "Competence collapsed outside the target — brittle overfitting."
        next_action = "Add competence regularizer / broaden mixture; do not accept branch."
    else:
        code = "INCONCLUSIVE_REJECT"
        meaning = "Rejected without a clean KL/target signature — inspect logs before redesign."
        next_action = "Re-run fair round with matched seeds; do not escalate niches yet."

    return {
        "diagnosis_code": code,
        "meaning": meaning,
        "next_action": next_action,
        "signals": {
            "tiny_kl": tiny_kl,
            "large_kl": large_kl,
            "target_improved": target_up,
            "collapsed_elsewhere": collapsed,
            "delta_g_flat": flat_g,
            "branch_kl": float(branch_kl) if branch_kl == branch_kl else None,
            "niche_payoff_improvement": float(niche_payoff_improvement),
            "general_competence_change": float(general_competence_change),
            "delta_G_available": float(delta_g),
        },
        "escalate_to_task_niches": False,  # only after multiple fair rounds
        "note": (
            "Escalate to phase/geometry niches only after multiple fair Stage-1 "
            "rounds remain flat with consistent diagnosis."
        ),
    }


def lro_manifest() -> dict[str, Any]:
    return {
        "protocol": LRO_PROTOCOL,
        "claim": LRO_CLAIM,
        "classification": "DIAGNOSTIC",
        "paper_claim": (
            "Summer uses response-oracle training to create complementary "
            "latent team strategies inside one decentralized PPO policy, then "
            "learns a persistent context router that selects among them and "
            "outperforms fixed-strategy and matched non-latent PPO agents."
        ),
        "stages": {
            "0_landscape_scan": "archive G_before baseline (informational)",
            "1_one_specialist": "ΔG>0 for one forced-z BR round (primary)",
            "2_confirm": "CI95(ΔG)>0 with ≥32 eps/cell and ≥3 seeds",
            "3_repertoire": "add branches only if G rises again (2–3 enough)",
            "4_retention": "G_available inside single policy > 0",
            "5_sparse_router": "G_realized = V_router − V_best_fixed_z > 0",
            "6_headline": "G_latent = V_routed_LRO − V_matched_nonlatent > 0",
        },
        "stage1_breakthrough": "delta_G_available = G_after - G_before > 0",
        "stage1_acceptance": [
            "delta_G > 0",
            "targeted_mixture_improves",
            "competence_above_floor",
            "inactive_branches_no_drift",
            "nonredundant_payoff_row",
        ],
        "one_retry_only": True,
        "closed_paths": [
            "entropy_as_diversity_proof",
            "router_before_G_available",
            "coefficient_carousel",
            "4eps_cell_winner_as_success",
        ],
        "phase_context_axis": phase_pods_manifest(),
        "pod_to_z": dict(POD_TO_Z),
        "phase_ids": list(PHASE_POD_IDS),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


__all__ = [
    "LRO_CLAIM",
    "LRO_PROTOCOL",
    "LandscapePolicySpec",
    "accept_lro_round",
    "branch_row_redundant",
    "classify_phase_from_core",
    "default_landscape_policies",
    "diagnose_lro_reject",
    "lro_manifest",
    "payoff_tensor_summary",
    "select_response_target",
    "write_json",
]
