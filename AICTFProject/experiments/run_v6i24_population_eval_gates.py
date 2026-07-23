#!/usr/bin/env python3
"""V6I24 population evaluation gates (lean Path C).

Primary gate (comparative advantage; locked after V6I25 FAIL_SIGNAL)
--------------------------------------------------------------------
* >=2 opponent-map cells have different best policies (margin >= 0.10)
* Cross-fitted context oracle > best fixed policy on held-out episodes,
  with paired bootstrap CI excluding zero
  (π*(c) chosen on train seeds only — not per-episode hindsight max)

Supporting evidence (not sufficient alone)
------------------------------------------
* CF action-JSD mean > 0.05 on >=2 cells, OR leave-one-cell-out trajectory
  classifier accuracy > 50%
* Mean payoff-row distance >= 0.10 for at least one policy pair

Smoke default: 32 episodes/cell. Promotion confirmation: 128.
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.v6i24_population_config import DEFAULT_MAPS, DEFAULT_OPPONENTS  # noqa: E402
from rl.router.counterfactual_router import (  # noqa: E402
    paired_delta_ci,
    train_test_split_indices,
)

CF_JSD_THRESHOLD = 0.05
CF_JSD_MIN_CELLS = 2
TRAJECTORY_ACCURACY_THRESHOLD = 0.50
PAYOFF_ROW_DISTANCE_THRESHOLD = 0.10
BEST_MARGIN_THRESHOLD = 0.10
MIN_CELLS_DIFFERENT_BEST = 2


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V6I24 population evaluation gates")
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--opponents", nargs="+", default=list(DEFAULT_OPPONENTS))
    p.add_argument("--maps", nargs="+", default=list(DEFAULT_MAPS))
    p.add_argument("--episodes-per-cell", type=int, default=32)
    p.add_argument(
        "--confirm",
        action="store_true",
        help="Use 128 episodes/cell for promotion confirmation.",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--steps-per-jsd-cell", type=int, default=120)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--max-decision-steps", type=int, default=240)
    p.add_argument(
        "--skip-jsd",
        action="store_true",
        help="Skip CF action-JSD (slim / micro payoff-only scoring).",
    )
    return p.parse_args()


def find_member_checkpoints(checkpoint_dir: Path) -> list[tuple[int, str, Path]]:
    """Return [(member_id, label, path), ...] from member_*_*.zip files."""
    found: list[tuple[int, str, Path]] = []
    for path in sorted(checkpoint_dir.glob("member_*.zip")):
        parts = path.stem.split("_", 2)
        if len(parts) < 3:
            continue
        try:
            mid = int(parts[1])
        except ValueError:
            continue
        label = parts[2]
        found.append((mid, label, path))
    return found


def _make_env(checkpoint: Path, map_name: str, seed: int, device: str, max_steps: int):
    from game_field_gpu import GPUCTFVecEnv, GPUFieldConfig
    from rl.custom_ppo.inference import read_custom_ppo_metadata

    meta = read_custom_ppo_metadata(str(checkpoint))
    agents = int(meta.get("n_blue", 2))
    cfg = GPUFieldConfig(
        n_envs=1,
        max_blue_agents=agents,
        max_red_agents=agents,
        map_layout=map_name,
        max_decision_steps=int(max_steps),
        aquaticus_profile=True,
        rules_profile="OURS",
        device=device,
        seed=seed,
    )
    return GPUCTFVecEnv(cfg)


def _load_policies(
    members: list[tuple[int, str, Path]],
    observation_space,
    action_space,
    device: str,
):
    from rl.custom_ppo import load_custom_ppo_policy

    policies = []
    for mid, label, path in members:
        pol = load_custom_ppo_policy(str(path), observation_space, action_space, device=device)
        policies.append({"id": mid, "label": label, "path": str(path), "policy": pol})
    return policies


def _set_opponent(env, opponent: str) -> None:
    try:
        env.env_method("set_phase", opponent)
        env.env_method("set_next_opponent", "SCRIPTED", opponent)
    except Exception:
        pass


def _episode_return_and_features(
    policy,
    env,
    *,
    opponent: str,
    seed: int,
    deterministic: bool,
    max_steps: int = 400,
) -> tuple[float, bool, np.ndarray]:
    """Run one episode; return (blue_score_proxy, win, action_hist features)."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if hasattr(policy, "reset_strategy"):
        policy.reset_strategy()
    _set_opponent(env, opponent)
    obs = env.reset()

    action_counts = np.zeros(5, dtype=np.float64)
    steps = 0
    blue_score = 0.0
    red_score = 0.0
    step_cap = max(1, int(max_steps))
    while steps < step_cap:
        # Inject real geometry; missing global_state silently zeros the router path.
        if isinstance(obs, dict):
            try:
                obs = dict(obs)
                obs["global_state"] = env.state()[0]
            except Exception:
                pass
        actions, _ = policy.predict(obs, deterministic=deterministic)
        flat = np.asarray(actions).reshape(-1)
        for a in flat:
            ai = int(a)
            if 0 <= ai < 5:
                action_counts[ai] += 1.0
        step_out = env.step(actions)
        if len(step_out) == 5:
            obs, _rew, dones, _trunc, infos = step_out
        else:
            obs, _rew, dones, infos = step_out
        steps += 1
        info0 = infos[0] if isinstance(infos, (list, tuple)) else infos
        if isinstance(info0, dict):
            blue_score = float(info0.get("blue_score", blue_score) or blue_score)
            red_score = float(info0.get("red_score", red_score) or red_score)
        if bool(np.any(dones)):
            break

    total_a = float(action_counts.sum()) + 1e-8
    feats = action_counts / total_a
    win = blue_score > red_score
    # Payoff proxy: win_rate contribution via score margin scaled lightly + win
    payoff = (1.0 if win else 0.0) + 0.05 * (blue_score - red_score)
    return float(payoff), bool(win), feats.astype(np.float64)


def collect_payoff_and_features(
    policies: list[dict[str, Any]],
    *,
    opponents: list[str],
    maps: list[str],
    episodes_per_cell: int,
    base_seed: int,
    device: str,
    max_decision_steps: int,
    matched_seeds_across_members: bool = True,
) -> dict[str, Any]:
    """Collect matched-seed payoffs. Default: same seed across members per episode.

    Matched seeds are required for the cross-fitted context-oracle vs best-fixed
    paired comparison (primary V6I24 gate after V6I25).
    """
    contexts = [f"{o}|{m}" for o in opponents for m in maps]
    k = len(policies)
    n_ctx = len(contexts)
    n_ep = int(episodes_per_cell)
    payoff = np.zeros((k, n_ctx), dtype=np.float64)
    wins = np.zeros((k, n_ctx), dtype=np.float64)
    returns = np.zeros((k, n_ctx, n_ep), dtype=np.float64)
    samples: list[tuple[int, int, np.ndarray]] = []

    ref_ckpt = Path(policies[0]["path"])
    for ci, (opp, mp) in enumerate((o, m) for o in opponents for m in maps):
        env = _make_env(ref_ckpt, mp, base_seed + ci, device, max_decision_steps)
        try:
            for ki, entry in enumerate(policies):
                cell_wins = []
                for ep in range(n_ep):
                    if matched_seeds_across_members:
                        seed = base_seed + 100 * ci + ep
                    else:
                        seed = base_seed + 10_000 * ki + 100 * ci + ep
                    pay, win, feats = _episode_return_and_features(
                        entry["policy"],
                        env,
                        opponent=opp,
                        seed=seed,
                        deterministic=True,
                        max_steps=int(max_decision_steps),
                    )
                    returns[ki, ci, ep] = float(pay)
                    cell_wins.append(1.0 if win else 0.0)
                    samples.append((ci, ki, feats))
                payoff[ki, ci] = float(returns[ki, ci].mean())
                wins[ki, ci] = float(np.mean(cell_wins))
                print(
                    f"  [{entry['label']}] {opp}|{mp}: "
                    f"payoff={payoff[ki, ci]:.3f} wr={wins[ki, ci]:.3f}",
                    flush=True,
                )
        finally:
            env.close()

    return {
        "contexts": contexts,
        "payoff_matrix": payoff,
        "winrate_matrix": wins,
        "returns_kce": returns,
        "samples": samples,
        "member_labels": [p["label"] for p in policies],
        "matched_seeds_across_members": bool(matched_seeds_across_members),
    }


def evaluate_cross_fitted_teacher_oracle(
    returns_kce: np.ndarray,
    *,
    member_labels: list[str],
    context_labels: list[str],
    test_frac: float = 0.25,
    seed: int = 0,
    n_bootstrap: int = 1000,
) -> dict[str, Any]:
    """Cross-fitted π*(c) on train episodes; score vs best-fixed on held-out.

    ``returns_kce`` shape ``(K, C, E)`` with matched seeds across members.
    """
    r = np.asarray(returns_kce, dtype=np.float64)
    if r.ndim != 3:
        raise ValueError(f"returns_kce must be (K,C,E), got {r.shape}")
    k, c, e = r.shape
    if e < 2:
        paired = paired_delta_ci(np.array([]), np.array([]), n_bootstrap=1, seed=seed)
        return {
            "context_oracle_mean": float("nan"),
            "best_fixed_mean": float("nan"),
            "best_fixed_idx": -1,
            "best_fixed_member": None,
            "delta": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "gate_cross_fitted_oracle": False,
            "n_heldout": 0,
            "pi_star_per_context": [],
            "note": "need >=2 episodes/cell for train/test split",
        }
    train_e, test_e = train_test_split_indices(e, test_frac=test_frac, seed=seed)
    q_train = r[:, :, train_e].mean(axis=2)  # (K, C)
    best_fixed_idx = int(np.argmax(q_train.mean(axis=1)))
    pi_star = np.argmax(q_train, axis=0).astype(np.int64)  # (C,)

    oracle_vals: list[float] = []
    fixed_vals: list[float] = []
    for ci in range(c):
        for ei in test_e:
            oracle_vals.append(float(r[int(pi_star[ci]), ci, int(ei)]))
            fixed_vals.append(float(r[best_fixed_idx, ci, int(ei)]))
    paired = paired_delta_ci(
        np.asarray(oracle_vals),
        np.asarray(fixed_vals),
        n_bootstrap=n_bootstrap,
        seed=seed + 7,
    )
    return {
        "context_oracle_mean": paired.mean_a,
        "best_fixed_mean": paired.mean_b,
        "best_fixed_idx": best_fixed_idx,
        "best_fixed_member": member_labels[best_fixed_idx],
        "delta": paired.delta,
        "ci_low": paired.ci_low,
        "ci_high": paired.ci_high,
        "gate_cross_fitted_oracle": bool(paired.ci_excludes_zero_positive),
        "n_heldout": paired.n,
        "n_train_episodes": int(train_e.size),
        "n_test_episodes": int(test_e.size),
        "pi_star_per_context": [
            {"context": context_labels[ci], "member": member_labels[int(pi_star[ci])], "idx": int(pi_star[ci])}
            for ci in range(c)
        ],
        "unique_train_best_policies": int(len(set(int(x) for x in pi_star.tolist()))),
    }


def evaluate_strategic_separation(
    payoff_matrix: np.ndarray,
    context_labels: list[str],
    member_labels: list[str],
    *,
    returns_kce: np.ndarray | None = None,
    test_frac: float = 0.25,
    seed: int = 0,
    n_bootstrap: int = 1000,
) -> dict[str, Any]:
    k, c = payoff_matrix.shape
    pairwise = np.zeros((k, k), dtype=np.float64)
    for i, j in combinations(range(k), 2):
        d = float(np.mean(np.abs(payoff_matrix[i] - payoff_matrix[j])))
        pairwise[i, j] = pairwise[j, i] = d

    max_pair_d = float(pairwise.max()) if k > 1 else 0.0
    best_per_context = np.argmax(payoff_matrix, axis=0)
    unique_best = len(set(best_per_context.tolist()))

    cells_with_margin = []
    for ci in range(c):
        col = payoff_matrix[:, ci]
        order = np.argsort(-col)
        best_i = int(order[0])
        second = float(col[order[1]]) if k > 1 else float("-inf")
        margin = float(col[best_i] - second)
        if margin >= BEST_MARGIN_THRESHOLD:
            cells_with_margin.append(
                {
                    "context": context_labels[ci],
                    "best": member_labels[best_i],
                    "best_idx": best_i,
                    "margin": margin,
                }
            )

    # Primary niche gate: at least 2 cells with different best AND margin >= 0.10
    margin_winners = {row["best_idx"] for row in cells_with_margin}
    gate_different_best = (
        len(cells_with_margin) >= MIN_CELLS_DIFFERENT_BEST
        and len(margin_winners) >= MIN_CELLS_DIFFERENT_BEST
    )

    # Diagnostic only: in-sample hindsight max (NOT the primary gate).
    hindsight_oracle_return = float(np.mean(np.max(payoff_matrix, axis=0)))
    best_fixed_idx = int(np.argmax(np.mean(payoff_matrix, axis=1)))
    best_fixed_return = float(np.mean(payoff_matrix[best_fixed_idx]))
    hindsight_oracle_gap = hindsight_oracle_return - best_fixed_return

    if returns_kce is None:
        # Backward-compatible unit tests: fall back to cell-mean hindsight
        # proxy, but mark that the primary cross-fitted gate is unavailable.
        cross = {
            "context_oracle_mean": hindsight_oracle_return,
            "best_fixed_mean": best_fixed_return,
            "best_fixed_idx": best_fixed_idx,
            "best_fixed_member": member_labels[best_fixed_idx],
            "delta": hindsight_oracle_gap,
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "gate_cross_fitted_oracle": False,
            "n_heldout": 0,
            "note": "returns_kce missing; cross-fitted gate unavailable",
        }
    else:
        cross = evaluate_cross_fitted_teacher_oracle(
            returns_kce,
            member_labels=member_labels,
            context_labels=context_labels,
            test_frac=test_frac,
            seed=seed,
            n_bootstrap=n_bootstrap,
        )

    return {
        "payoff_matrix": payoff_matrix.tolist(),
        "context_labels": context_labels,
        "member_labels": member_labels,
        "pairwise_row_distance": pairwise.tolist(),
        "max_pairwise_row_distance": max_pair_d,
        "best_per_context": best_per_context.tolist(),
        "unique_best_policies": unique_best,
        "cells_with_margin_ge_0_10": cells_with_margin,
        "hindsight_oracle_return": hindsight_oracle_return,
        "hindsight_oracle_gap": hindsight_oracle_gap,
        "oracle_return": cross["context_oracle_mean"],
        "best_fixed_return": cross["best_fixed_mean"],
        "best_fixed_member": cross["best_fixed_member"],
        "oracle_gap": cross["delta"],
        "cross_fitted_oracle": cross,
        "gate_row_distance": max_pair_d >= PAYOFF_ROW_DISTANCE_THRESHOLD,
        "gate_different_best_with_margin": gate_different_best,
        # Primary: cross-fitted CI (V6I25 lesson). Hindsight alone is insufficient.
        "gate_oracle_above_fixed": bool(cross.get("gate_cross_fitted_oracle")),
        "gate_cross_fitted_oracle": bool(cross.get("gate_cross_fitted_oracle")),
    }

def held_out_trajectory_classifier(
    samples: list[tuple[int, int, np.ndarray]],
    n_members: int,
    n_cells: int,
) -> dict[str, Any]:
    """Leave-one-cell-out nearest-centroid classifier on action-frequency features.

    Holding out entire opponent-map cells prevents 'this was OP11 so policy 2'
    leakage from context identity.
    """
    if n_cells < 2 or not samples:
        return {
            "accuracy": float("nan"),
            "n_test": 0,
            "gate_classifier": False,
            "note": "insufficient cells/samples",
        }

    correct = 0
    total = 0
    for hold in range(n_cells):
        train = [(m, f) for c, m, f in samples if c != hold]
        test = [(m, f) for c, m, f in samples if c == hold]
        if not train or not test:
            continue
        centroids = []
        for m in range(n_members):
            feats = [f for mm, f in train if mm == m]
            if feats:
                centroids.append(np.mean(np.stack(feats, axis=0), axis=0))
            else:
                centroids.append(np.zeros_like(train[0][1]))
        centroids_a = np.stack(centroids, axis=0)
        for m_true, feat in test:
            d = np.linalg.norm(centroids_a - feat[None, :], axis=1)
            pred = int(np.argmin(d))
            correct += int(pred == m_true)
            total += 1
    acc = float(correct / total) if total else float("nan")
    return {
        "accuracy": acc,
        "n_test": total,
        "chance": 1.0 / max(1, n_members),
        "gate_classifier": bool(acc == acc and acc > TRAJECTORY_ACCURACY_THRESHOLD),
        "protocol": "leave_one_cell_out_nearest_centroid_action_freqs",
    }


def _collect_shared_history(
    collector_policy,
    env,
    *,
    opponent: str,
    n_steps: int,
    seed: int,
) -> list[dict[str, np.ndarray]]:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(collector_policy, "reset_strategy"):
        collector_policy.reset_strategy()
    _set_opponent(env, opponent)
    obs = env.reset()
    snaps: list[dict[str, np.ndarray]] = []
    for _ in range(n_steps):
        snaps.append({k: np.array(v, copy=True) for k, v in obs.items() if isinstance(v, np.ndarray)})
        actions, _ = collector_policy.predict(obs, deterministic=True)
        step_out = env.step(actions)
        if len(step_out) == 5:
            obs, _rew, dones, _trunc, _info = step_out
        else:
            obs, _rew, dones, _info = step_out
        if bool(np.any(dones)):
            obs = env.reset()
            if hasattr(collector_policy, "reset_strategy"):
                collector_policy.reset_strategy()
    return snaps


def _obs_batch(snaps: list[dict[str, np.ndarray]], device: torch.device) -> dict[str, torch.Tensor]:
    keys = ["grid", "vec", "agent_mask", "mask"]
    out: dict[str, torch.Tensor] = {}
    for k in keys:
        if all(k in s for s in snaps):
            stacked = np.concatenate([s[k] for s in snaps], axis=0)
            out[k] = torch.as_tensor(stacked, device=device)
    return out


def counterfactual_action_jsd(
    policies: list[dict[str, Any]],
    *,
    opponents: list[str],
    maps: list[str],
    steps_per_cell: int,
    base_seed: int,
    device: str,
    max_decision_steps: int,
) -> dict[str, Any]:
    from rl.custom_ppo.diagnostics.counterfactual import jsd_from_logits

    ref = policies[0]
    device_t = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
    cell_rows = []

    for ci, (opp, mp) in enumerate((o, m) for o in opponents for m in maps):
        env = _make_env(Path(ref["path"]), mp, base_seed + ci, device, max_decision_steps)
        try:
            snaps = _collect_shared_history(
                ref["policy"],
                env,
                opponent=opp,
                n_steps=steps_per_cell,
                seed=base_seed + 1000 + ci,
            )
            # Primary CF test: identical raw observation batch + identical masks.
            # Actor pathway is non-recurrent; router RNN is unused (latent off).
            obs_t = _obs_batch(snaps, device_t)
            logits_by_k: list[torch.Tensor] = []
            for entry in policies:
                model = entry["policy"].model
                with torch.no_grad():
                    # Teachers freeze z=0; always score that channel for CF JSD.
                    if bool(getattr(model, "uses_latent_strategy", False)) or bool(
                        getattr(model, "use_latent_strategy", False)
                    ) or int(getattr(model, "latent_k", 0) or 0) > 0:
                        z_idx = torch.zeros(
                            (obs_t["grid"].shape[0],), dtype=torch.long, device=device_t
                        )
                        logits = model.policy_logits(obs_t, z_idx=z_idx)
                    else:
                        logits = model.policy_logits(obs_t)
                    logits = model._mask_logits(logits, obs_t.get("mask"))
                    logits_by_k.append(logits.float())

            pair_rows = []
            n = int(logits_by_k[0].shape[0])
            model0 = policies[0]["policy"].model
            for i, j in combinations(range(len(policies)), 2):
                dists_i = list(model0._categoricals(logits_by_k[i]))
                dists_j = list(model0._categoricals(logits_by_k[j]))
                jsd_sum = torch.zeros((n,), device=logits_by_k[i].device)
                for di, dj in zip(dists_i, dists_j):
                    jsd_sum = jsd_sum + jsd_from_logits(di.logits, dj.logits)
                pair_rows.append(
                    {
                        "i": policies[i]["id"],
                        "j": policies[j]["id"],
                        "action_jsd_mean": float(jsd_sum.mean().item()),
                    }
                )
            jsds = [r["action_jsd_mean"] for r in pair_rows]
            cell_rows.append(
                {
                    "context": f"{opp}|{mp}",
                    "n_states": n,
                    "pair_jsd_mean": float(np.mean(jsds)) if jsds else float("nan"),
                    "pair_jsd_max": float(np.max(jsds)) if jsds else float("nan"),
                    "pairs_above_0_05": int(sum(1 for v in jsds if v > CF_JSD_THRESHOLD)),
                    "pairs": pair_rows,
                }
            )
            print(
                f"  JSD {opp}|{mp}: mean={cell_rows[-1]['pair_jsd_mean']:.4f} "
                f"pairs>0.05={cell_rows[-1]['pairs_above_0_05']}",
                flush=True,
            )
        finally:
            env.close()

    cells_pass = sum(
        1
        for row in cell_rows
        if row["pair_jsd_mean"] == row["pair_jsd_mean"]
        and row["pair_jsd_mean"] > CF_JSD_THRESHOLD
    )
    return {
        "cells": cell_rows,
        "cells_with_jsd_gt_0_05": cells_pass,
        "gate_jsd": cells_pass >= CF_JSD_MIN_CELLS,
    }


def aggregate_verdict(
    *,
    jsd: dict[str, Any],
    classifier: dict[str, Any],
    strategic: dict[str, Any],
) -> dict[str, Any]:
    """Primary = comparative advantage; JSD/classifier are supporting only."""
    supporting = bool(jsd.get("gate_jsd")) or bool(classifier.get("gate_classifier"))
    primary = all(
        [
            strategic.get("gate_different_best_with_margin"),
            strategic.get("gate_cross_fitted_oracle"),
        ]
    )
    # Row distance remains a useful diagnostic but is not required for primary PASS.
    overall = bool(primary)
    if overall:
        decision = "PASS_BUILD_DISTILLATION"
    elif (
        supporting
        or strategic.get("gate_different_best_with_margin")
        or strategic.get("max_pairwise_row_distance", 0) > 0.03
        or float(strategic.get("oracle_gap") or 0.0) > 0.0
    ):
        decision = "TREND_EXTEND_TO_100K"
    else:
        decision = "FAIL_REDESIGN_PRESSURES"
    return {
        "functional_pass": supporting,  # supporting evidence alias
        "supporting_pass": supporting,
        "primary_pass": primary,
        "strategic_pass": primary,
        "overall_pass": overall,
        "decision": decision,
    }


def run_eval_gates(
    *,
    checkpoint_dir: Path,
    output_dir: Path,
    episodes_per_cell: int,
    seed: int,
    device: str,
    opponents: list[str] | None = None,
    maps: list[str] | None = None,
    steps_per_jsd_cell: int = 120,
    max_decision_steps: int = 240,
    confirm_episodes: bool = False,
    skip_jsd: bool = False,
) -> dict[str, Any]:
    opponents = list(opponents or DEFAULT_OPPONENTS)
    maps = list(maps or DEFAULT_MAPS)
    if confirm_episodes:
        episodes_per_cell = max(episodes_per_cell, 128)

    members = find_member_checkpoints(checkpoint_dir)
    if len(members) < 2:
        raise FileNotFoundError(f"Need >=2 member_*.zip in {checkpoint_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 72)
    print("V6I24 Population Evaluation Gates")
    print("=" * 72)
    for mid, label, path in members:
        print(f"  member_{mid}_{label}: {path.name}")
    print(f"Episodes/cell: {episodes_per_cell}")
    print(f"Cells: {len(opponents)}×{len(maps)} opponents×maps")
    print()

    env0 = _make_env(members[0][2], maps[0], seed, device, max_decision_steps)
    try:
        policies = _load_policies(
            members, env0.observation_space, env0.action_space, device
        )
    finally:
        env0.close()

    print("--- Payoff / trajectory features (matched seeds) ---")
    collected = collect_payoff_and_features(
        policies,
        opponents=opponents,
        maps=maps,
        episodes_per_cell=episodes_per_cell,
        base_seed=seed,
        device=device,
        max_decision_steps=max_decision_steps,
        matched_seeds_across_members=True,
    )
    strategic = evaluate_strategic_separation(
        collected["payoff_matrix"],
        collected["contexts"],
        collected["member_labels"],
        returns_kce=collected["returns_kce"],
        seed=seed,
    )
    classifier = held_out_trajectory_classifier(
        collected["samples"],
        n_members=len(policies),
        n_cells=len(collected["contexts"]),
    )

    if skip_jsd:
        print("--- Counterfactual action-JSD: SKIPPED (micro-probe) ---")
        jsd = {
            "cells": [],
            "cells_with_jsd_gt_0_05": 0,
            "gate_jsd": False,
            "skipped": True,
        }
    else:
        print("--- Counterfactual action-JSD (shared histories; supporting) ---")
        jsd = counterfactual_action_jsd(
            policies,
            opponents=opponents,
            maps=maps,
            steps_per_cell=steps_per_jsd_cell,
            base_seed=seed + 777,
            device=device,
            max_decision_steps=max_decision_steps,
        )

    verdict = aggregate_verdict(jsd=jsd, classifier=classifier, strategic=strategic)
    cross = strategic.get("cross_fitted_oracle") or {}
    result = {
        "protocol": "v6i24_population_eval_gates",
        "classification": "DIAGNOSTIC",
        "path": "C_fallback_independent_teachers",
        "primary_gate": "cross_fitted_context_oracle_gt_best_fixed",
        "checkpoint_dir": str(checkpoint_dir),
        "episodes_per_cell": episodes_per_cell,
        "members": [{"id": m[0], "label": m[1], "path": str(m[2])} for m in members],
        "winrate_matrix": collected["winrate_matrix"].tolist(),
        "strategic": strategic,
        "classifier": classifier,
        "action_jsd": jsd,
        "verdict": verdict,
        "thresholds": {
            "cf_jsd": CF_JSD_THRESHOLD,
            "cf_jsd_min_cells": CF_JSD_MIN_CELLS,
            "trajectory_accuracy": TRAJECTORY_ACCURACY_THRESHOLD,
            "payoff_row_distance": PAYOFF_ROW_DISTANCE_THRESHOLD,
            "best_margin": BEST_MARGIN_THRESHOLD,
            "min_cells_different_best": MIN_CELLS_DIFFERENT_BEST,
        },
    }
    out_path = output_dir / "v6i24_eval_gates.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print()
    print("Gate summary:")
    print(f"  JSD cells >0.05: {jsd.get('cells_with_jsd_gt_0_05')} (supporting; need >={CF_JSD_MIN_CELLS})")
    print(f"  Classifier acc:  {classifier.get('accuracy')} (supporting)")
    print(f"  Max row distance:{strategic.get('max_pairwise_row_distance'):.4f}")
    print(
        f"  Cross-fit oracle: {cross.get('context_oracle_mean')} "
        f"best_fixed={cross.get('best_fixed_mean')} "
        f"delta={cross.get('delta')} CI=[{cross.get('ci_low')},{cross.get('ci_high')}]"
    )
    print(f"  Hindsight gap:   {strategic.get('hindsight_oracle_gap'):.4f} (diagnostic only)")
    print(f"  Primary (niche+CF oracle): {'PASS' if verdict['primary_pass'] else 'FAIL'}")
    print(f"  Supporting (JSD/clf):      {'PASS' if verdict['supporting_pass'] else 'FAIL'}")
    print(f"  Decision:        {verdict['decision']}")
    print(f"Wrote {out_path}")
    return result


def main() -> int:
    args = _parse_args()
    ckpt_dir = Path(args.checkpoint_dir)
    if not ckpt_dir.is_dir():
        print(f"ERROR: checkpoint directory not found: {ckpt_dir}")
        return 2
    output_dir = Path(args.output_dir) if args.output_dir else ckpt_dir / "eval_gates"
    try:
        run_eval_gates(
            checkpoint_dir=ckpt_dir,
            output_dir=output_dir,
            episodes_per_cell=int(args.episodes_per_cell),
            seed=int(args.seed),
            device=str(args.device),
            opponents=list(args.opponents),
            maps=list(args.maps),
            steps_per_jsd_cell=int(args.steps_per_jsd_cell),
            max_decision_steps=int(args.max_decision_steps),
            confirm_episodes=bool(args.confirm),
            skip_jsd=bool(args.skip_jsd),
        )
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
