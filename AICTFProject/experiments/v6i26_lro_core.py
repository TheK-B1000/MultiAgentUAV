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
LRO_CONFIRMATION_MIN_EPISODES_PER_CELL = 32
LRO_CONFIRMATION_MIN_TRAINING_SEEDS = 3

# Default LRO evaluation / birth surface. Includes map_a_open (default arena).
# Training keeps obstacle_obs_channel=True so V6I23+ 8-channel checkpoints remain
# loadable when cells are open-arena only (wall plane is zeros on map_a).
try:
    from gpu_env._maps import MAP_A_OPEN, MAP_B_SPLIT_LANE, MAP_B_SPLIT_LANE_V2

    LRO_DEFAULT_MAPS = (MAP_A_OPEN, MAP_B_SPLIT_LANE, MAP_B_SPLIT_LANE_V2)
except Exception:  # noqa: BLE001
    LRO_DEFAULT_MAPS = ("map_a_open", "map_b_split_lane", "map_b_split_lane_v2")


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
            "forced_z_behavior_nonredundant",
        ],
    }


def calibrate_margin_headroom_threshold(
    margin_std: np.ndarray,
    *,
    n_episodes: int,
    se_multiplier: float = 2.0,
    absolute_floor: float = 0.15,
) -> dict[str, float]:
    """Calibrate recoverable-margin headroom from matched-seed cell variability.

    Uses median cell SE = median(std / sqrt(n)) across finite (z, context) cells,
    then ``max(absolute_floor, se_multiplier * median_se)``.
    """
    std = np.asarray(margin_std, dtype=np.float64)
    n = max(1, int(n_episodes))
    se = std / float(np.sqrt(n))
    finite = se[np.isfinite(se)]
    if finite.size == 0:
        median_se = float(absolute_floor)
    else:
        median_se = float(np.median(finite))
    threshold = float(max(float(absolute_floor), float(se_multiplier) * median_se))
    return {
        "n_episodes": float(n),
        "se_multiplier": float(se_multiplier),
        "absolute_floor": float(absolute_floor),
        "median_cell_se": median_se,
        "min_margin_headroom": threshold,
    }


def select_margin_response_target(
    winrate: np.ndarray,
    margin: np.ndarray,
    *,
    contexts: Sequence[str],
    policy_labels: Sequence[str],
    min_margin_headroom: float,
    wr_competence_floor: float = 0.75,
    branch_wr_floor: float = 0.50,
    target_fraction: float = 0.75,
    max_target_contexts: int = 3,
) -> dict[str, Any]:
    """Select LRO targets by recoverable win-margin headroom.

    Primary score for branch ``z`` on context ``c``:

        headroom(z, c) = best_z margin(c) - margin(z, c)

    Winrate is only a competence / safety gate (not the selection objective).
    TTC and other temporal metrics are intentionally excluded.
    """
    wr = np.asarray(winrate, dtype=np.float64)
    m = np.asarray(margin, dtype=np.float64)
    if wr.shape != m.shape:
        raise ValueError(f"winrate/margin shape mismatch: {wr.shape} vs {m.shape}")
    if m.ndim != 2:
        raise ValueError(f"margin must be 2D, got {m.shape}")
    if m.shape[1] != len(contexts):
        raise ValueError("contexts length must match matrix columns")
    if m.shape[0] != len(policy_labels):
        raise ValueError("policy_labels length must match matrix rows")
    if m.size == 0:
        raise ValueError("margin matrix is empty")

    finite_m = np.where(np.isfinite(m), m, -np.inf)
    finite_wr = np.where(np.isfinite(wr), wr, -np.inf)
    best_margin = finite_m.max(axis=0)
    best_margin_z = finite_m.argmax(axis=0)
    best_wr = finite_wr.max(axis=0)
    best_wr_z = finite_wr.argmax(axis=0)
    row_wr = np.nanmean(wr, axis=1)
    wr_floor = float(wr_competence_floor)
    branch_floor = float(branch_wr_floor)
    thr = float(min_margin_headroom)

    # Recoverable headroom: how far the candidate trails the best margin.
    headroom = best_margin[None, :] - m
    headroom = np.where(np.isfinite(headroom), headroom, -np.inf)

    candidates: list[tuple[float, int, int, float, float]] = []
    # (headroom, -branch_row_wr, context_idx, branch, best_wr, branch_wr)
    for ci in range(m.shape[1]):
        if not np.isfinite(best_wr[ci]) or float(best_wr[ci]) < wr_floor:
            continue
        for z in range(m.shape[0]):
            if int(z) == int(best_margin_z[ci]):
                continue
            hr = float(headroom[z, ci])
            if not np.isfinite(hr) or hr < thr:
                continue
            branch_wr = float(wr[z, ci]) if np.isfinite(wr[z, ci]) else float("nan")
            if not np.isfinite(branch_wr) or branch_wr < branch_floor:
                # Allow training a weak-on-cell but globally competent branch.
                if float(row_wr[z]) < branch_floor:
                    continue
            candidates.append(
                (
                    hr,
                    float(row_wr[z]),
                    int(ci),
                    int(z),
                    float(best_wr[ci]),
                    float(branch_wr) if np.isfinite(branch_wr) else float(row_wr[z]),
                )
            )

    candidates.sort(key=lambda t: (-t[0], -t[1], str(contexts[t[2]]), t[3]))
    if not candidates:
        # Fallback: largest headroom ignoring thresholds (still report gates fail).
        flat = []
        for ci in range(m.shape[1]):
            for z in range(m.shape[0]):
                if int(z) == int(best_margin_z[ci]):
                    continue
                hr = float(headroom[z, ci])
                if not np.isfinite(hr):
                    continue
                flat.append((hr, float(row_wr[z]), int(ci), int(z), float(best_wr[ci]), float(wr[z, ci])))
        flat.sort(key=lambda t: (-t[0], -t[1], str(contexts[t[2]]), t[3]))
        if not flat:
            raise ValueError("no margin-headroom candidates available")
        candidates = flat
        selection_viable = False
    else:
        selection_viable = True

    primary_hr, _, primary, branch, primary_best_wr, branch_wr_on_target = candidates[0]
    # Additional target contexts: same branch, next-best recoverable headrooms.
    extra = [
        (hr, ci)
        for hr, _row, ci, z, _bwr, _br in candidates[1:]
        if int(z) == int(branch) and ci != primary
    ]
    extra.sort(key=lambda t: -t[0])
    target_idx = [primary] + [ci for _, ci in extra[: max(0, int(max_target_contexts) - 1)]]
    target_idx = list(dict.fromkeys(target_idx))

    # Anchors: high-competence contexts (high best WR + high best margin).
    anchors = [i for i in range(m.shape[1]) if i not in set(target_idx)]
    anchors.sort(
        key=lambda i: (
            -float(best_wr[i]) if np.isfinite(best_wr[i]) else 0.0,
            -float(best_margin[i]) if np.isfinite(best_margin[i]) else 0.0,
            str(contexts[i]),
        )
    )
    anchor_idx = anchors[: min(2, len(anchors))]

    target_mass = float(target_fraction) if anchor_idx else 1.0
    anchor_mass = max(0.0, 1.0 - target_mass)
    target_raw = np.asarray(
        [max(1e-6, float(headroom[branch, i])) for i in target_idx], dtype=np.float64
    )
    target_weights = target_raw / max(1e-12, float(target_raw.sum()))
    mixture: dict[str, float] = {}
    for i, w in zip(target_idx, target_weights.tolist()):
        mixture[str(contexts[i])] = float(target_mass * w)
    if anchor_idx:
        per_anchor = anchor_mass / float(len(anchor_idx))
        for i in anchor_idx:
            mixture[str(contexts[i])] = float(mixture.get(str(contexts[i]), 0.0) + per_anchor)
    total = sum(mixture.values())
    if total > 0.0:
        mixture = {k: float(v / total) for k, v in mixture.items()}

    context_to_idx = {str(ctx): i for i, ctx in enumerate(contexts)}
    return {
        "selection_basis": "recoverable_win_margin_headroom",
        "selection_metric": "win_margin",
        "min_margin_headroom": thr,
        "wr_competence_floor": wr_floor,
        "branch_wr_floor": branch_floor,
        "target_fraction": float(target_fraction),
        "selection_viable": bool(selection_viable),
        "best_wr_by_context": {
            str(contexts[i]): float(best_wr[i]) for i in range(m.shape[1])
        },
        "best_wr_z_by_context": {
            str(contexts[i]): int(best_wr_z[i]) for i in range(m.shape[1])
        },
        "best_margin_by_context": {
            str(contexts[i]): float(best_margin[i]) for i in range(m.shape[1])
        },
        "best_margin_z_by_context": {
            str(contexts[i]): int(best_margin_z[i]) for i in range(m.shape[1])
        },
        "row_mean_winrate": {
            str(policy_labels[z]): float(row_wr[z]) for z in range(m.shape[0])
        },
        "target_context_indices": [int(i) for i in target_idx],
        "target_contexts": [str(contexts[i]) for i in target_idx],
        "anchor_context_indices": [int(i) for i in anchor_idx],
        "anchor_contexts": [str(contexts[i]) for i in anchor_idx],
        "mixture_weights": mixture,
        "mixture_top": [
            {
                "context": str(ctx),
                "weight": float(w),
                "best_wr": float(best_wr[context_to_idx[str(ctx)]]),
                "best_margin": float(best_margin[context_to_idx[str(ctx)]]),
                "branch_margin_headroom": float(
                    headroom[branch, context_to_idx[str(ctx)]]
                ),
            }
            for ctx, w in sorted(mixture.items(), key=lambda item: -item[1])[:3]
        ],
        "target_context": str(contexts[primary]),
        "target_context_index": int(primary),
        "target_best_wr": float(primary_best_wr),
        "target_best_margin": float(best_margin[primary]),
        "target_sensitive_headroom": float(primary_hr),
        "current_best_z_on_target": int(best_margin_z[primary]),
        "current_best_wr_z_on_target": int(best_wr_z[primary]),
        "branch_to_train_index": int(branch),
        "branch_to_train_label": str(policy_labels[branch]),
        "branch_margin_on_target": float(m[branch, primary]),
        "branch_wr_on_target": float(branch_wr_on_target),
        "branch_row_mean_winrate": float(row_wr[branch]),
        "n_viable_candidates": int(len(candidates)) if selection_viable else 0,
        "acceptance_requires": [
            "delta_G_available_gt_0",
            "ci95_low_delta_G_gt_0",
            "nonredundant_payoff_row",
            "competence_above_floor",
            "forced_z_behavior_nonredundant",
            "multi_seed_repetition",
        ],
    }


def select_current_response_target(
    payoff: np.ndarray,
    *,
    contexts: Sequence[str],
    policy_labels: Sequence[str],
    saturation_cutoff: float = 0.90,
    target_fraction: float = 0.75,
    competence_floor: float | None = None,
    max_target_contexts: int = 3,
) -> dict[str, Any]:
    """Select the next LRO target from the current forced-z coverage matrix.

    Archive landscape scans define the evaluation surface; this selector chooses
    the weakness and the branch from the **current** forced-z table.

    ``payoff`` MUST be a coverage metric on roughly ``[0, 1]`` (typically
    winrate / ``success``). Do **not** pass ``win_margin`` — a 0.90 saturation
    cutoff is meaningless on unbounded margins.
    """
    p = np.asarray(payoff, dtype=np.float64)
    if p.ndim != 2:
        raise ValueError(f"payoff must be 2D, got {p.shape}")
    if p.shape[1] != len(contexts):
        raise ValueError("contexts length must match payoff columns")
    if p.shape[0] != len(policy_labels):
        raise ValueError("policy_labels length must match payoff rows")
    if p.size == 0:
        raise ValueError("payoff matrix is empty")

    finite = np.where(np.isfinite(p), p, -np.inf)
    coverage = finite.max(axis=0)
    best_z = finite.argmax(axis=0)
    row_means = np.nanmean(p, axis=1)
    median_row = float(np.nanmedian(row_means))
    floor = median_row if competence_floor is None else float(competence_floor)
    stable = row_means >= floor

    headroom = np.maximum(0.0, float(saturation_cutoff) - coverage)
    eligible = [
        i
        for i, cov in enumerate(coverage.tolist())
        if np.isfinite(cov) and cov < float(saturation_cutoff)
    ]
    if not eligible:
        eligible = [int(np.argmin(coverage))]
    target_scores = [(i, float(headroom[i]), float(coverage[i])) for i in eligible]
    target_scores.sort(key=lambda item: (-item[1], item[2], str(contexts[item[0]])))
    target_idx = [i for i, _, _ in target_scores[: max(1, int(max_target_contexts))]]

    primary = int(target_idx[0])
    dominant_counts = np.bincount(best_z.astype(np.int64), minlength=p.shape[0])
    branch_candidates = [
        z
        for z in range(p.shape[0])
        if z != int(best_z[primary]) and bool(stable[z]) and int(dominant_counts[z]) == 0
    ]
    if not branch_candidates:
        branch_candidates = [
            z
            for z in range(p.shape[0])
            if z != int(best_z[primary]) and bool(stable[z])
        ]
    if not branch_candidates:
        branch_candidates = [z for z in range(p.shape[0]) if z != int(best_z[primary])]
    if not branch_candidates:
        branch_candidates = [int(best_z[primary])]
    branch = max(
        branch_candidates,
        key=lambda z: (float(row_means[z]), -float(finite[z, primary]), -int(z)),
    )

    anchors = [i for i in range(p.shape[1]) if i not in set(target_idx)]
    anchors.sort(key=lambda i: (-float(np.nanmean(p[:, i])), str(contexts[i])))
    anchor_idx = anchors[: min(2, len(anchors))]

    target_mass = float(target_fraction) if anchor_idx else 1.0
    anchor_mass = max(0.0, 1.0 - target_mass)
    target_raw = np.asarray([max(1e-6, float(headroom[i])) for i in target_idx], dtype=np.float64)
    target_weights = target_raw / max(1e-12, float(target_raw.sum()))

    mixture: dict[str, float] = {}
    for i, w in zip(target_idx, target_weights.tolist()):
        mixture[str(contexts[i])] = float(target_mass * w)
    if anchor_idx:
        per_anchor = anchor_mass / float(len(anchor_idx))
        for i in anchor_idx:
            mixture[str(contexts[i])] = float(mixture.get(str(contexts[i]), 0.0) + per_anchor)

    total = sum(mixture.values())
    if total > 0.0:
        mixture = {k: float(v / total) for k, v in mixture.items()}

    context_to_idx = {str(ctx): i for i, ctx in enumerate(contexts)}
    saturated = [
        {
            "context": str(contexts[i]),
            "coverage": float(coverage[i]),
            "best_z": int(best_z[i]),
        }
        for i in range(p.shape[1])
        if i not in eligible
    ]
    return {
        "selection_basis": "current_forced_z_payoff",
        "saturation_cutoff": float(saturation_cutoff),
        "target_fraction": float(target_fraction),
        "competence_floor": floor,
        "coverage_by_context": {
            str(contexts[i]): float(coverage[i]) for i in range(p.shape[1])
        },
        "best_z_by_context": {
            str(contexts[i]): int(best_z[i]) for i in range(p.shape[1])
        },
        "row_mean_payoff": {
            str(policy_labels[z]): float(row_means[z]) for z in range(p.shape[0])
        },
        "dominant_context_count_by_branch": {
            str(policy_labels[z]): int(dominant_counts[z]) for z in range(p.shape[0])
        },
        "stable_branch_indices": [int(z) for z in range(p.shape[0]) if bool(stable[z])],
        "excluded_saturated_contexts": saturated,
        "target_context_indices": [int(i) for i in target_idx],
        "target_contexts": [str(contexts[i]) for i in target_idx],
        "anchor_context_indices": [int(i) for i in anchor_idx],
        "anchor_contexts": [str(contexts[i]) for i in anchor_idx],
        "mixture_weights": mixture,
        "mixture_top": [
            {
                "context": str(ctx),
                "weight": float(w),
                "coverage": float(coverage[context_to_idx[str(ctx)]]),
            }
            for ctx, w in sorted(mixture.items(), key=lambda item: -item[1])[:3]
        ],
        "target_context": str(contexts[primary]),
        "target_context_index": primary,
        "target_coverage": float(coverage[primary]),
        "target_headroom": float(headroom[primary]),
        "current_best_z_on_target": int(best_z[primary]),
        "branch_to_train_index": int(branch),
        "branch_to_train_label": str(policy_labels[branch]),
        "branch_payoff_on_target": float(p[branch, primary]),
        "branch_row_mean_payoff": float(row_means[branch]),
        "acceptance_requires": [
            "delta_G_available_gt_0",
            "ci95_low_delta_G_gt_0",
            "nonredundant_payoff_row",
            "competence_above_floor",
            "forced_z_behavior_nonredundant",
            "multi_seed_repetition",
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


def behavior_distinctness_summary(
    behavior_report: dict[str, Any] | None,
    *,
    branch_idx: int,
    min_branch_distance: float | None = None,
) -> dict[str, Any]:
    """Summarize whether the candidate branch realizes distinct forced-z behavior."""
    from rl.forced_z_behavior_vectors import (
        DEFAULT_BEHAVIOR_PAIR_THRESHOLD,
        FORCED_Z_BEHAVIOR_VECTOR_NAMES,
        normalize_behavior_vectors,
    )

    threshold = (
        float(DEFAULT_BEHAVIOR_PAIR_THRESHOLD)
        if min_branch_distance is None
        else float(min_branch_distance)
    )
    branch = int(branch_idx)
    invalid = {
        "behavior_measurement_valid": False,
        "behavior_distance_threshold": threshold,
        "branch_idx": branch,
        "branch_behavior_nonredundant": False,
        "verdict": "BEHAVIOR_DISTINCT_FAIL",
    }
    if not isinstance(behavior_report, dict):
        return {**invalid, "reason": "missing_behavior_report"}
    per_z = behavior_report.get("per_z_behavior_vectors")
    if not isinstance(per_z, dict) or not per_z:
        return {**invalid, "reason": "missing_per_z_behavior_vectors"}

    indexed: list[tuple[int, np.ndarray]] = []
    for key, values in per_z.items():
        label = str(key)
        if not label.startswith("z") or not isinstance(values, dict):
            continue
        try:
            z_idx = int(label[1:])
        except ValueError:
            continue
        raw = np.asarray(
            [float(values.get(name, np.nan)) for name in FORCED_Z_BEHAVIOR_VECTOR_NAMES],
            dtype=np.float64,
        )
        if raw.shape[0] == len(FORCED_Z_BEHAVIOR_VECTOR_NAMES) and np.isfinite(raw).all():
            indexed.append((z_idx, raw))

    indexed.sort(key=lambda item: item[0])
    if len(indexed) < 2:
        return {**invalid, "reason": "need_at_least_two_valid_z_vectors"}
    z_indices = [int(z) for z, _ in indexed]
    if branch not in z_indices:
        return {
            **invalid,
            "valid_z_indices": z_indices,
            "reason": "branch_vector_missing",
        }

    normalized = normalize_behavior_vectors([raw for _, raw in indexed], source="telemetry")
    branch_pos = z_indices.index(branch)
    pairs: list[dict[str, Any]] = []
    distances: list[float] = []
    branch_pairs: list[dict[str, Any]] = []
    for i in range(len(normalized)):
        for j in range(i + 1, len(normalized)):
            dist = float(np.linalg.norm(normalized[i] - normalized[j]))
            pair = {
                "z_i": z_indices[i],
                "z_j": z_indices[j],
                "distance": dist,
                "above_threshold": bool(dist >= threshold),
            }
            pairs.append(pair)
            distances.append(dist)
            if i == branch_pos or j == branch_pos:
                branch_pairs.append(pair)

    if not branch_pairs:
        return {
            **invalid,
            "valid_z_indices": z_indices,
            "reason": "branch_has_no_behavior_neighbor",
        }
    nearest = min(branch_pairs, key=lambda pair: float(pair["distance"]))
    nearest_neighbor = (
        int(nearest["z_j"]) if int(nearest["z_i"]) == branch else int(nearest["z_i"])
    )
    nearest_distance = float(nearest["distance"])
    branch_nonredundant = bool(nearest_distance >= threshold)
    return {
        "behavior_measurement_valid": True,
        "behavior_distance_threshold": threshold,
        "behavior_vector_names": list(FORCED_Z_BEHAVIOR_VECTOR_NAMES),
        "valid_z_indices": z_indices,
        "branch_idx": branch,
        "branch_nearest_behavior_distance": nearest_distance,
        "branch_nearest_behavior_neighbor": nearest_neighbor,
        "branch_behavior_nonredundant": branch_nonredundant,
        "behavior_pair_count": len(distances),
        "behavior_pair_distance_min": float(min(distances)) if distances else None,
        "behavior_pair_distance_mean": float(np.mean(distances)) if distances else None,
        "behavior_pair_distance_max": float(max(distances)) if distances else None,
        "behavior_pairs_above_threshold": int(sum(d >= threshold for d in distances)),
        "behavior_pairs": pairs,
        "verdict": (
            "BEHAVIOR_DISTINCT_PASS"
            if branch_nonredundant
            else "BEHAVIOR_DISTINCT_FAIL"
        ),
    }


def accept_lro_round(
    *,
    g_before: float,
    g_after: float,
    payoff_after: np.ndarray,
    branch_idx: int,
    competence_floor: float,
    min_row_distance: float = 0.08,
    behavior_distinctness: dict[str, Any] | None = None,
    require_behavior_distinctness: bool = False,
    episodes_per_cell: int | None = None,
    ci95_low_delta_g: float | None = None,
    training_seed_count: int | None = None,
    min_confirmation_episodes_per_cell: int = LRO_CONFIRMATION_MIN_EPISODES_PER_CELL,
    min_training_seeds: int = LRO_CONFIRMATION_MIN_TRAINING_SEEDS,
) -> dict[str, Any]:
    """Locked LRO acceptance.

    Small forced-z sweeps are screening only. They can nominate a branch as
    ``PROMISING_DIRECTION`` but cannot approve a strategy birth.
    """
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
    behavior_required = bool(require_behavior_distinctness)
    branch_behavior_nonredundant = (
        None
        if behavior_distinctness is None
        else bool(behavior_distinctness.get("branch_behavior_nonredundant"))
    )
    behavior_ok = bool(branch_behavior_nonredundant) if behavior_required else True
    screening_ok = bool(delta_ok and competence_ok and nonredundant_ok and behavior_ok)
    eps_cell = None if episodes_per_cell is None else int(episodes_per_cell)
    seed_count = None if training_seed_count is None else int(training_seed_count)
    ci_low = None if ci95_low_delta_g is None else float(ci95_low_delta_g)
    confirmation_episode_ok = bool(
        eps_cell is not None and eps_cell >= int(min_confirmation_episodes_per_cell)
    )
    ci_ok = bool(ci_low is not None and ci_low > 0.0)
    multi_seed_ok = bool(seed_count is not None and seed_count >= int(min_training_seeds))
    accepted = bool(screening_ok and confirmation_episode_ok and ci_ok and multi_seed_ok)
    if accepted:
        verdict = "ACCEPT"
    elif screening_ok:
        verdict = "PROMISING_DIRECTION"
    else:
        verdict = "REJECT"
    return {
        "G_before": float(g_before),
        "G_after": float(g_after),
        "delta_G_available": delta_g,
        "branch_idx": b,
        "branch_mean_payoff": branch_mean,
        "competence_floor": float(competence_floor),
        "best_on_context_indices": best_contexts,
        "nonredundant_payoff_row": nonredundant_ok,
        "branch_behavior_nonredundant": branch_behavior_nonredundant,
        "behavior_distinctness_required": behavior_required,
        "behavior_distinctness_pass": behavior_ok,
        "behavior_distinctness": behavior_distinctness,
        "competence_above_floor": competence_ok,
        "delta_G_gt_0": delta_ok,
        "screening_pass": screening_ok,
        "episodes_per_cell": eps_cell,
        "min_confirmation_episodes_per_cell": int(min_confirmation_episodes_per_cell),
        "confirmation_episode_count_pass": confirmation_episode_ok,
        "ci95_low_delta_G": ci_low,
        "ci95_delta_G_gt_0": ci_ok,
        "training_seed_count": seed_count,
        "min_training_seeds": int(min_training_seeds),
        "multi_seed_repetition_pass": multi_seed_ok,
        "accepted": accepted,
        "verdict": verdict,
        "real_acceptance_requires": [
            "delta_G_gt_0",
            "ci95_low_delta_G_gt_0",
            "nonredundant_payoff_row",
            "competence_above_floor",
            "forced_z_behavior_nonredundant",
            f"episodes_per_cell >= {int(min_confirmation_episodes_per_cell)}",
            f"training_seed_count >= {int(min_training_seeds)}",
        ],
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
        "default_maps": list(LRO_DEFAULT_MAPS),
        "obstacle_obs_channel": True,
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
        "screening_verdict": (
            "4 episodes/cell may produce PROMISING_DIRECTION only; never ACCEPT."
        ),
        "target_selection": [
            "measure current forced-z payoff matrix before each response round",
            "exclude contexts whose current coverage is already saturated",
            "select uncovered target contexts by headroom below saturation cutoff",
            "select a stable branch that is not already dominant on the target",
            "train a fixed target/competence-anchor mixture",
        ],
        "stage1_acceptance": [
            "delta_G > 0",
            "CI95(delta_G) lower bound > 0",
            "targeted_mixture_improves",
            "competence_above_floor",
            "inactive_branches_no_drift",
            "nonredundant_payoff_row",
            "forced_z_behavior_nonredundant",
            ">= 32 forced-z episodes/cell",
            "repetition across >= 3 training seeds",
        ],
        "checkpoint_diagnostics": [
            "target_cell_payoff",
            "competence",
            "behavior_distance_from_nearest_branch",
            "branch_KL_from_initialization",
            "inactive_branch_drift",
        ],
        "heldout_confirmation_cells": [
            "targeted_regret_mixture_cells",
            "nearby_niche_or_map_cells",
            "general_competence_anchor_cells",
        ],
        "controlled_retry_if_multiseed_fails": (
            "First inspect learning-signal diagnostics and retarget from the current "
            "forced-z payoff matrix. Try per-z value heads only if the signal is "
            "healthy but the selected branch still collapses into an existing row."
        ),
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


def cell_means_from_episode_df(
    df: Any,
    *,
    opponents: Sequence[str],
    maps: Sequence[str],
    latent_k: int = 4,
    metric: str = "success",
) -> tuple[np.ndarray, list[str]]:
    """Build a (K x C) cell-mean matrix from forced-z episode_results rows."""
    contexts = [f"{o}|{m}" for o in opponents for m in maps]
    out = np.zeros((int(latent_k), len(contexts)), dtype=np.float64)
    for zi in range(int(latent_k)):
        for ci, ctx in enumerate(contexts):
            opp, mp = ctx.split("|", 1)
            sub = df[(df["latent_z"] == zi) & (df["opponent"] == opp) & (df["map"] == mp)]
            if len(sub) == 0:
                out[zi, ci] = float("nan")
            else:
                out[zi, ci] = float(sub[metric].mean())
    return out, contexts


def summarize_training_learning_signal(
    metrics_csv: Path | str,
    *,
    branch_idx: int | None = None,
) -> dict[str, Any]:
    """Summarize PPO learning-signal columns from an LRO metrics CSV.

    Classifies where the reward→advantage→actor-grad→step→z-params→KL
    chain breaks, so the next intervention targets the dead link instead of
    blindly scaling budget or architecture.
    """
    import pandas as pd

    path = Path(metrics_csv)
    df = pd.read_csv(path)
    if len(df) == 0:
        return {"status": "EMPTY", "path": str(path)}

    def _col(name: str) -> np.ndarray:
        if name not in df.columns:
            return np.asarray([], dtype=np.float64)
        return np.asarray(pd.to_numeric(df[name], errors="coerce").dropna(), dtype=np.float64)

    def _first_col(*names: str) -> np.ndarray:
        for name in names:
            arr = _col(name)
            if arr.size:
                return arr
        return np.asarray([], dtype=np.float64)

    approx_kl = _col("approx_kl")
    clip_frac = _col("clip_fraction")
    entropy = _col("entropy")
    ev = _col("explained_variance")
    value_loss = _col("value_loss")
    policy_loss = _col("policy_loss")
    learning_rate = _col("learning_rate")
    grad = _first_col("grad_norm", "ppo_actor_grad_norm", "actor_grad_norm_total")
    actor_grad = _first_col(
        "z_specific_grad_norm",
        "actor_grad_norm_ppo",
        "ppo_actor_grad_norm",
        "actor_ppo_grad_norm",
    )
    critic_grad = _col("critic_grad_norm")
    shared_grad = _col("shared_actor_grad_norm")
    trunk_grad = _col("z_branch_trunk_grad_norm")
    head_grad = _col("z_action_head_grad_norm")
    adapter_grad = _col("z_adapter_grad_norm")
    # Prefer rollout GAE std; latent_episode_adv_* is often unused (zeros).
    adv_std = _first_col(
        "rollout_adv_std",
        "latent_arc_advantage_std",
        "latent_episode_adv_std",
    )
    adv_mean = _first_col(
        "latent_arc_advantage_mean",
        "latent_episode_adv_mean",
    )
    ppo_param_delta = _col("ppo_parameter_delta")
    shared_delta = _col("shared_actor_max_abs_delta")
    z_specific_delta = _col("z_specific_max_abs_delta")

    branch = 0 if branch_idx is None else int(branch_idx)
    branch_adapter_delta = _col(f"latent_adapter_weight_delta_z{branch}")
    branch_trunk_delta = _col(f"latent_branch_trunk_delta_z{branch}")
    branch_head_delta = _col(f"latent_action_head_delta_z{branch}")
    branch_embed_delta = _col(f"z_embedding_delta_z{branch}")
    peer_adapter_deltas = [
        _col(f"latent_adapter_weight_delta_z{z}")
        for z in range(4)
        if z != branch
    ]

    def _stat(arr: np.ndarray) -> dict[str, float | None]:
        if arr.size == 0:
            return {"n": 0, "mean": None, "last": None, "max": None}
        return {
            "n": int(arr.size),
            "mean": float(np.mean(arr)),
            "last": float(arr[-1]),
            "max": float(np.max(arr)),
        }

    def _mean_or_nan(arr: np.ndarray) -> float:
        return float(np.mean(arr)) if arr.size else float("nan")

    def _max_or_nan(arr: np.ndarray) -> float:
        return float(np.max(arr)) if arr.size else float("nan")

    kl_mean = _mean_or_nan(approx_kl)
    clip_mean = _mean_or_nan(clip_frac)
    grad_mean = _mean_or_nan(grad)
    actor_grad_mean = _mean_or_nan(actor_grad)
    critic_grad_mean = _mean_or_nan(critic_grad)
    adv_std_mean = _mean_or_nan(adv_std)
    branch_delta_max = max(
        [
            v
            for v in (
                _max_or_nan(branch_adapter_delta),
                _max_or_nan(branch_trunk_delta),
                _max_or_nan(branch_head_delta),
                _max_or_nan(branch_embed_delta),
            )
            if v == v
        ],
        default=float("nan"),
    )
    peer_delta_max = max(
        [_max_or_nan(arr) for arr in peer_adapter_deltas if arr.size],
        default=0.0,
    )
    shared_delta_max = _max_or_nan(shared_delta)

    tiny_kl = bool(approx_kl.size and kl_mean < 1e-3)
    tiny_clip = bool(clip_frac.size and clip_mean < 1e-2)
    tiny_grad = bool(grad.size and grad_mean < 1e-4)
    flat_adv = bool(adv_std.size and adv_std_mean < 1e-3)
    # CF-path actor_grad_norm_ppo defaults to 0 when unused; treat as missing
    # unless z_specific / trunk / head grads are present.
    actor_grad_observed = bool(
        _col("z_specific_grad_norm").size
        or trunk_grad.size
        or head_grad.size
        or adapter_grad.size
    )
    actor_grad_near_zero = bool(
        actor_grad_observed and actor_grad.size and actor_grad_mean < 1e-4
    )
    critic_dominates = bool(
        critic_grad.size
        and grad.size
        and critic_grad_mean == critic_grad_mean
        and grad_mean == grad_mean
        and critic_grad_mean > 0.1
        and critic_grad_mean >= 0.85 * max(grad_mean, 1e-12)
    )
    branch_moved = bool(branch_delta_max == branch_delta_max and branch_delta_max > 1e-6)
    branch_step_tiny = bool(
        branch_delta_max == branch_delta_max and branch_delta_max < 1e-3
    )
    freeze_mask_ok = bool(
        (not shared_delta.size or shared_delta_max <= 1e-8)
        and peer_delta_max <= 1e-8
    )

    # Decision tree: localize the dead link.
    if flat_adv:
        broken_link = "FLAT_ADVANTAGES"
        code = "NO_USABLE_LEARNING_PRESSURE"
        meaning = (
            "Advantage std near zero — PPO has little credit signal about which "
            "actions were better. Inspect reward / mixture dilution, not capacity."
        )
        next_action = (
            "Audit per-context advantages and target/anchor mixture weights; "
            "do not scale updates or add per-z critics yet."
        )
    elif actor_grad_near_zero and critic_grad_mean == critic_grad_mean and critic_grad_mean > 1e-3:
        broken_link = "ACTOR_GRAD_NEAR_ZERO"
        code = "NO_USABLE_LEARNING_PRESSURE"
        meaning = (
            "Advantages/critic look alive but z-specific actor gradients are near "
            "zero — inspect actor loss graph, detach/mask, or routing bypass."
        )
        next_action = (
            "Trace policy_loss → active branch parameters; verify ratios and "
            "that the forced-z branch modules are on the autograd path."
        )
    elif (not branch_moved) and (not tiny_grad) and tiny_kl:
        broken_link = "ACTOR_STEP_TINY"
        code = "NO_USABLE_LEARNING_PRESSURE"
        meaning = (
            "Gradients exist but active-branch parameters barely moved — LR, "
            "clipping, optimizer membership, or loss scaling."
        )
        next_action = (
            "Compare effective actor LR and optimizer param groups against the "
            "active z branch; raise LR only after confirming membership."
        )
    elif branch_moved and tiny_kl and tiny_clip:
        broken_link = "PARAMS_MOVE_KL_FLAT"
        code = "NO_USABLE_LEARNING_PRESSURE"
        meaning = (
            "Active-branch parameters changed but policy KL stayed tiny — "
            "updates have little influence on action probabilities "
            "(residual/adapter scale, identity trunk, or insensitive directions)."
        )
        next_action = (
            "Probe logit sensitivity of branch trunk / action head; consider "
            "raising residual alpha or confirming deep trunks leave identity."
        )
    elif tiny_kl and tiny_clip and critic_dominates:
        broken_link = "CRITIC_DOMINATED_JOINT_GRAD"
        code = "NO_USABLE_LEARNING_PRESSURE"
        meaning = (
            "Reported grad_norm is critic-dominated while policy KL/clip stay "
            "near zero — do not read joint grad_norm as actor learning pressure."
        )
        next_action = (
            "Use z_specific_grad_norm / branch deltas; keep training paused until "
            "actor-side pressure is visible."
        )
    elif tiny_kl and tiny_clip:
        broken_link = "POLICY_STUCK_BASIN"
        code = "NO_USABLE_LEARNING_PRESSURE"
        meaning = (
            "Policy KL and clip fraction stayed near zero — the branch had "
            "little economic reason / signal to leave its init basin."
        )
        next_action = "Re-check target headroom and branch selection before more budget."
    elif (not tiny_kl) and tiny_grad:
        broken_link = "KL_WITHOUT_GRAD_LOG"
        code = "UPDATES_WITHOUT_GRAD_MAGNITUDE"
        meaning = "KL moved but reported grad norms stayed tiny — check freezing / logging."
        next_action = "Verify grad telemetry wiring; do not trust empty CF-path defaults."
    else:
        broken_link = "NONE_SIGNAL_PRESENT"
        code = "SIGNAL_PRESENT_CHECK_BEHAVIOR"
        meaning = (
            "Learning diagnostics are non-degenerate; require behavior distance "
            "and target payoff movement before claiming specialization."
        )
        next_action = (
            "Evaluate forced-z OP9 margin / behavior distance before continuing "
            "to 10u."
        )

    return {
        "status": code,
        "broken_link": broken_link,
        "meaning": meaning,
        "next_action": next_action,
        "path": str(path),
        "n_updates": int(len(df)),
        "branch_idx": branch,
        "approx_kl": _stat(approx_kl),
        "clip_fraction": _stat(clip_frac),
        "entropy": _stat(entropy),
        "explained_variance": _stat(ev),
        "value_loss": _stat(value_loss),
        "policy_loss": _stat(policy_loss),
        "learning_rate": _stat(learning_rate),
        "grad_norm": _stat(grad),
        "actor_grad_norm": _stat(actor_grad),
        "critic_grad_norm": _stat(critic_grad),
        "shared_actor_grad_norm": _stat(shared_grad),
        "z_branch_trunk_grad_norm": _stat(trunk_grad),
        "z_action_head_grad_norm": _stat(head_grad),
        "z_adapter_grad_norm": _stat(adapter_grad),
        "advantage_mean": _stat(adv_mean),
        "advantage_std": _stat(adv_std),
        "ppo_parameter_delta": _stat(ppo_param_delta),
        "shared_actor_max_abs_delta": _stat(shared_delta),
        "z_specific_max_abs_delta": _stat(z_specific_delta),
        "branch_adapter_weight_delta": _stat(branch_adapter_delta),
        "branch_trunk_delta": _stat(branch_trunk_delta),
        "branch_action_head_delta": _stat(branch_head_delta),
        "branch_embedding_delta": _stat(branch_embed_delta),
        "chain": {
            "advantages_alive": bool(adv_std.size and not flat_adv),
            "critic_grad_alive": bool(
                critic_grad.size and critic_grad_mean == critic_grad_mean and critic_grad_mean > 1e-3
            ),
            "actor_grad_alive": bool(
                actor_grad_observed
                and actor_grad.size
                and actor_grad_mean == actor_grad_mean
                and actor_grad_mean > 1e-4
            ),
            "branch_params_moved": branch_moved,
            "branch_step_tiny": branch_step_tiny,
            "policy_kl_alive": bool(approx_kl.size and not tiny_kl),
            "freeze_mask_ok": freeze_mask_ok,
            "critic_dominates_joint_grad": critic_dominates,
            "peer_branch_adapter_delta_max": float(peer_delta_max),
            "branch_param_delta_max": (
                float(branch_delta_max) if branch_delta_max == branch_delta_max else None
            ),
        },
        "flags": {
            "tiny_approx_kl": tiny_kl,
            "tiny_clip_fraction": tiny_clip,
            "tiny_grad_norm": tiny_grad,
            "flat_advantages": flat_adv,
            "actor_grad_near_zero": actor_grad_near_zero,
            "branch_step_tiny": branch_step_tiny,
            "freeze_mask_ok": freeze_mask_ok,
        },
    }


# ---------------------------------------------------------------------------
# Actor-step ablation gates (optimizer-control retry)
# ---------------------------------------------------------------------------

ACTOR_STEP_WEAK_APPROX_KL_FLOOR = 1.847735062862436e-05  # margin 5u pilot mean
ACTOR_STEP_WEAK_ENTROPY_MEAN = 2.636798652013143  # metrics CSV column ``entropy``
ACTOR_STEP_FIXED_BATCH_KL_MIN = 1e-3
ACTOR_STEP_FIXED_BATCH_KL_MAX = 1e-2  # safety ceiling; 10x probe KL~0.14 is too large
ACTOR_STEP_ENTROPY_FIELD = "entropy"  # summed action-head entropy in metrics CSV
ACTOR_STEP_ENTROPY_TOL = 0.3
ACTOR_STEP_CLIP_FRAC_MAX = 0.5
MARGIN_PILOT_LOCKED = {
    "branch": 0,
    "target_context": "OP9_SPLIT_LANE_FEINT|map_b_split_lane",
    "opponent": "OP9_SPLIT_LANE_FEINT",
    "map": "map_b_split_lane",
}


def actor_step_ablation_contract(*, z_actor_lr_mult: float = 2.0) -> dict[str, Any]:
    """Predeclared pass/fail contract for an actor-step 5u ablation rung."""
    mult = float(z_actor_lr_mult)
    return {
        "protocol": "v6i26_actor_step_ablation",
        "classification": "DIAGNOSTIC",
        "hypothesis": (
            "Joint critic-dominated clipping + insufficient z-actor LR caused "
            "the failed specialization step; separate clip + "
            f"{mult:g}x z-actor LR should produce measurable policy movement "
            "on the locked OP9/z0 surface without leaping policy space."
        ),
        "locked_surface": dict(MARGIN_PILOT_LOCKED),
        "optimizer": {
            "separate_actor_critic_clip": True,
            "z_actor_lr_mult": mult,
            "critic_lr_unchanged": True,
            "base_learning_rate": 5e-4,
            "max_grad_norm_per_group": 0.5,
            "lr_schedule_preserves_group_mult": True,
        },
        "learning_pressure_gates": {
            "fixed_batch_init_to_final_kl_min": ACTOR_STEP_FIXED_BATCH_KL_MIN,
            "fixed_batch_init_to_final_kl_max": ACTOR_STEP_FIXED_BATCH_KL_MAX,
            "fixed_batch_protocol": "run_v6i26_logit_control_authority_probe",
            "training_approx_kl_above_weak_floor": ACTOR_STEP_WEAK_APPROX_KL_FLOOR,
            "training_approx_kl_finite_non_explosive_max": 1.0,
            "clip_fraction_safety_max_mean": ACTOR_STEP_CLIP_FRAC_MAX,
            "clip_fraction_require_nonzero": False,
            "entropy_field": ACTOR_STEP_ENTROPY_FIELD,
            "entropy_aggregation": "policy entropy from metrics CSV (summed action heads)",
            "entropy_weak_run_mean": ACTOR_STEP_WEAK_ENTROPY_MEAN,
            "entropy_tolerance": ACTOR_STEP_ENTROPY_TOL,
            "inactive_branch_delta_approx_zero": True,
            "actor_grad_norm_gt_0": True,
            "critic_grad_norm_gt_0": True,
        },
        "strategic_gates": {
            "op9_margin_improves_directionally": True,
            "op11_op12_anchors_hold": True,
            "z0_behavior_distance_increases": True,
        },
        "decision_rule": {
            "learning_fail_below_floor": (
                "if mult < 5: escalate one rung (3->5); else stop optimizer climb"
            ),
            "learning_fail_above_ceiling": "stop increasing LR; step too large",
            "learning_pass_op9_fail": "active movement, strategically wrong — stop LR climb",
            "learning_op9_pass_behavior_flat": "response refinement, not strategy birth yet",
            "learning_op9_behavior_pass": "continue this exact checkpoint to 10u",
        },
        "do_not": [
            "continue weak or prior actor-step checkpoint to 10u without gates",
            "relaunch an identical completed multiplier recipe",
            "raise residual alpha as first lever",
            "redesign trunk",
            "add per-z critics",
            "switch to Case B opponents",
            "retune KL thresholds after seeing results",
        ],
    }


def evaluate_actor_step_learning_gates(
    *,
    learning_signal: dict[str, Any],
    fixed_batch_kl: float,
    weak_approx_kl_floor: float = ACTOR_STEP_WEAK_APPROX_KL_FLOOR,
    weak_entropy_mean: float = ACTOR_STEP_WEAK_ENTROPY_MEAN,
) -> dict[str, Any]:
    """Evaluate learning-pressure gates for the actor-step ablation."""

    def _mean(block: str, default: float = 0.0) -> float:
        node = learning_signal.get(block) or {}
        if isinstance(node, dict) and node.get("mean") is not None:
            return float(node["mean"])
        return float(default)

    approx_kl = _mean("approx_kl")
    clip_frac = _mean("clip_fraction")
    entropy = _mean("entropy", weak_entropy_mean)
    actor_gn = _mean("z_specific_grad_norm")
    if actor_gn <= 0.0:
        actor_gn = _mean("z_adapter_grad_norm")
    if actor_gn <= 0.0:
        actor_gn = _mean("z_branch_trunk_grad_norm")
    if actor_gn <= 0.0:
        actor_gn = _mean("z_action_head_grad_norm")
    if actor_gn <= 0.0:
        actor_gn = _mean("actor_grad_norm")
    critic_gn = _mean("critic_grad_norm")
    chain = learning_signal.get("chain") or {}
    peer_delta = float(chain.get("peer_branch_adapter_delta_max") or 0.0)
    z_delta = float(chain.get("branch_param_delta_max") or 0.0)
    if z_delta <= 0.0:
        z_delta = _mean("z_specific_max_abs_delta")
    if z_delta <= 0.0:
        z_delta = max(
            _mean("branch_adapter_weight_delta"),
            _mean("branch_trunk_delta"),
            _mean("branch_action_head_delta"),
            _mean("branch_embedding_delta"),
        )

    fixed_kl = float(fixed_batch_kl)
    gates = {
        "fixed_batch_init_to_final_kl_ge_1e-3": bool(
            fixed_kl >= ACTOR_STEP_FIXED_BATCH_KL_MIN
        ),
        "fixed_batch_init_to_final_kl_le_1e-2": bool(
            fixed_kl <= ACTOR_STEP_FIXED_BATCH_KL_MAX
        ),
        "training_approx_kl_above_weak_floor": bool(approx_kl > float(weak_approx_kl_floor)),
        "training_approx_kl_finite_non_explosive": bool(
            np.isfinite(approx_kl) and approx_kl < 1.0
        ),
        "clip_fraction_not_saturated": bool(
            np.isfinite(clip_frac) and clip_frac < ACTOR_STEP_CLIP_FRAC_MAX
        ),
        "entropy_stable_vs_weak_run": bool(
            abs(entropy - float(weak_entropy_mean)) <= ACTOR_STEP_ENTROPY_TOL
        ),
        "actor_grad_norm_gt_0": bool(actor_gn > 0.0 or z_delta > 0.0),
        "critic_grad_norm_gt_0": bool(critic_gn > 0.0),
        "inactive_branch_delta_approx_zero": bool(peer_delta < 1e-5),
        "z0_param_delta_clearly_above_weak": bool(z_delta > 1e-3),
    }
    # clip_fraction > 0 is intentionally NOT required.
    # z0_param_delta_clearly_above_weak: weak pilot max ~7.7e-4.
    learning_pass = all(
        gates[k]
        for k in (
            "fixed_batch_init_to_final_kl_ge_1e-3",
            "fixed_batch_init_to_final_kl_le_1e-2",
            "training_approx_kl_above_weak_floor",
            "training_approx_kl_finite_non_explosive",
            "clip_fraction_not_saturated",
            "entropy_stable_vs_weak_run",
            "actor_grad_norm_gt_0",
            "critic_grad_norm_gt_0",
            "inactive_branch_delta_approx_zero",
            "z0_param_delta_clearly_above_weak",
        )
    )
    return {
        "gates": gates,
        "learning_pass": bool(learning_pass),
        "metrics": {
            "fixed_batch_init_to_final_kl": fixed_kl,
            "approx_kl_mean": approx_kl,
            "clip_fraction_mean": clip_frac,
            "entropy_mean": entropy,
            "entropy_field": ACTOR_STEP_ENTROPY_FIELD,
            "actor_grad_norm_mean": actor_gn,
            "critic_grad_norm_mean": critic_gn,
            "z_specific_max_abs_delta_mean": z_delta,
            "inactive_branch_delta": peer_delta,
            "weak_approx_kl_floor": float(weak_approx_kl_floor),
            "weak_entropy_mean": float(weak_entropy_mean),
            "fixed_batch_kl_floor": ACTOR_STEP_FIXED_BATCH_KL_MIN,
            "fixed_batch_kl_ceiling": ACTOR_STEP_FIXED_BATCH_KL_MAX,
        },
    }


def target_kl_ladder_contract(
    *,
    z_actor_lr_mult: float = 3.0,
    max_updates: int = 5,
    checkpoint_every_updates: int = 1,
) -> dict[str, Any]:
    """Predeclared contract for a target-KL early-stop ladder (not an LR rung)."""
    mult = float(z_actor_lr_mult)
    return {
        "protocol": "v6i26_actor_step_target_kl_ladder",
        "classification": "DIAGNOSTIC",
        "hypothesis": (
            "2x under-floor and 3x over-ceiling show nonlinear KL vs LR; "
            "holding 3x LR fixed and stopping at the first 1u checkpoint whose "
            "fixed-batch init→ckpt KL lands in [1e-3, 1e-2] tests whether a "
            "safe intermediate movement point exists without changing target, "
            "architecture, reward, alpha, opponents, or router."
        ),
        "locked_surface": dict(MARGIN_PILOT_LOCKED),
        "optimizer": {
            "separate_actor_critic_clip": True,
            "z_actor_lr_mult": mult,
            "critic_lr_unchanged": True,
            "base_learning_rate": 5e-4,
            "lr_schedule_preserves_group_mult": True,
            "no_further_lr_rung": True,
        },
        "ladder": {
            "max_updates": int(max_updates),
            "checkpoint_every_updates": int(checkpoint_every_updates),
            "fixed_batch_kl_min": ACTOR_STEP_FIXED_BATCH_KL_MIN,
            "fixed_batch_kl_max": ACTOR_STEP_FIXED_BATCH_KL_MAX,
            "fixed_batch_protocol": "run_v6i26_logit_control_authority_probe",
            "selection_rule": "first_checkpoint_inside_kl_window",
            "evaluate_only_selected_checkpoint": True,
            "early_stop_when_selected": True,
        },
        "decision_rule": {
            "no_rung_enters_window": "FAIL — no safe intermediate; do not raise LR",
            "first_rung_above_ceiling": "FAIL — overshot before landing; do not raise LR",
            "selected_in_window": (
                "evaluate strategic gates ONLY on that checkpoint; "
                "then apply OP9 / anchors / behavior fork"
            ),
        },
        "do_not": [
            "treat this as another LR multiplier rung",
            "evaluate the final 5u ckpt if an earlier rung already entered the window",
            "retune KL thresholds after seeing ladder results",
            "raise residual alpha / redesign trunk / add per-z critics",
            "switch opponents or unlock router",
            "promote z3 Phase-2 candidate",
        ],
        "scientific_boundary": (
            "Niche surface revealed latent payoff variation, but tested LRO "
            "procedures have not yet manufactured a statistically valuable and "
            "behaviorally distinct strategy. No promotion, no router, no "
            "strategy-birth claim yet."
        ),
    }


def select_target_kl_ladder_rung(
    rungs: list[dict[str, Any]],
    *,
    kl_min: float = ACTOR_STEP_FIXED_BATCH_KL_MIN,
    kl_max: float = ACTOR_STEP_FIXED_BATCH_KL_MAX,
) -> dict[str, Any]:
    """Pick the first rung whose fixed-batch KL is inside ``[kl_min, kl_max]``.

    Rungs must be ordered by update index ascending. If a rung exceeds
    ``kl_max`` before any rung enters the window, selection fails as overshoot.
    """
    ordered = sorted(rungs, key=lambda r: int(r.get("update") or 0))
    for rung in ordered:
        kl = float(rung.get("fixed_batch_kl") or 0.0)
        update = int(rung.get("update") or 0)
        if kl_min <= kl <= kl_max:
            return {
                "status": "SELECTED",
                "selected_update": update,
                "selected_kl": kl,
                "reason": "first_checkpoint_inside_kl_window",
                "rung": rung,
            }
        if kl > kl_max:
            return {
                "status": "OVERSHOOT_BEFORE_WINDOW",
                "selected_update": None,
                "selected_kl": kl,
                "reason": f"update_{update}_exceeded_ceiling_before_any_in_window",
                "rung": rung,
            }
    last = ordered[-1] if ordered else {}
    return {
        "status": "NO_RUNG_IN_WINDOW",
        "selected_update": None,
        "selected_kl": float(last.get("fixed_batch_kl") or 0.0) if last else None,
        "reason": "all_rungs_below_floor_or_empty",
        "rung": last or None,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


__all__ = [
    "LRO_CLAIM",
    "LRO_DEFAULT_MAPS",
    "LRO_PROTOCOL",
    "LandscapePolicySpec",
    "accept_lro_round",
    "actor_step_ablation_contract",
    "behavior_distinctness_summary",
    "branch_row_redundant",
    "cell_means_from_episode_df",
    "classify_phase_from_core",
    "default_landscape_policies",
    "diagnose_lro_reject",
    "evaluate_actor_step_learning_gates",
    "lro_manifest",
    "payoff_tensor_summary",
    "select_current_response_target",
    "select_margin_response_target",
    "calibrate_margin_headroom_threshold",
    "select_response_target",
    "select_target_kl_ladder_rung",
    "summarize_training_learning_signal",
    "target_kl_ladder_contract",
    "write_json",
    "ACTOR_STEP_WEAK_APPROX_KL_FLOOR",
    "ACTOR_STEP_WEAK_ENTROPY_MEAN",
    "ACTOR_STEP_FIXED_BATCH_KL_MIN",
    "ACTOR_STEP_FIXED_BATCH_KL_MAX",
    "ACTOR_STEP_ENTROPY_FIELD",
    "MARGIN_PILOT_LOCKED",
]
