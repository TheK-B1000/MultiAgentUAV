#!/usr/bin/env python3
"""V6I26 leakage-free usable-repertoire test for a two-branch candidate (z0, z3).

Distinguishes two different "oracles" that must not be conflated:

  V_hindsight_oracle  -- per-unit max(outcome_z0, outcome_z3). A useful upper
                          bound on available complementarity, but NOT evidence
                          the repertoire is deployable (it uses the outcome to
                          pick the branch).
  V_legal_selector    -- a selector f(c0) frozen on a SELECTION split, using
                          only the exact vector the eventual Summer router
                          would receive at episode start (env.state()[0] right
                          after reset -- the same 34-D global_state the actor
                          banner reports as base_global_state_dim). No opponent
                          identity, no episode seed, no outcome information.
                          Evaluated on a disjoint HELD-OUT split.

Both branches may come from different checkpoint files (z1/z2/z3 stay frozen
during a single-branch LRO round, so "z3" for this test should be its own
fully-trained checkpoint, not the stale frozen copy sitting inside a z0-only
checkpoint).

Two subcommands, mirroring the --from-run/--analyze-only convention used
elsewhere in this codebase:

  collect   Run matched-seed episodes for z0 and z3 over every (opponent, map)
            cell in the locked target mixture, capturing c0 = env.state()[0]
            at reset time for each matched seed. Writes one row per matched
            unit to a CSV incrementally (resumable, reusable for repeated
            analysis without re-collecting). Env is built ONCE per context and
            reused across all resets in that context, not rebuilt per unit.

  analyze   Reads a raw units CSV (from `collect`). Splits into a selection
            half and a held-out half (first/second half of episode indices
            per context -- deterministic, no extra randomness). On the
            selection split, runs a bounded selector ladder (always-z0,
            always-z3, logistic regression, class-weighted logistic
            regression, depth-limited decision tree, small gradient-boosted
            tree, and an advantage-regression threshold selector) evaluated by
            k-fold CV *within the selection split only*, then freezes the
            single best-scoring selector and evaluates it -- once -- on the
            held-out split via paired bootstrap. Never touches held-out data
            during model selection.

Read-only with respect to training: loads two checkpoints for inference only.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(line_buffering=True)

from experiments.forced_z_eval.runner import _make_env  # noqa: E402
from experiments.forced_z_eval.protocol import ForcedZProtocol  # noqa: E402
from experiments.v6i26_lro_core import write_json  # noqa: E402
from plot.eval_rollout import run_eval_episodes  # noqa: E402
from rl.custom_ppo import load_custom_ppo_policy  # noqa: E402

_RAW_FIELDNAMES = ["opponent", "map", "episode_index", "episode_seed", "c0_json", "outcome_z0", "outcome_z3"]


# ---------------------------------------------------------------------------
# collect
# ---------------------------------------------------------------------------

def _contexts_from_mixture(target: dict) -> list[tuple[str, str]]:
    mixture = target.get("mixture_weights") or {}
    out = []
    for ctx in mixture:
        if "|" not in ctx:
            continue
        opp, mp = ctx.split("|", 1)
        out.append((opp.upper(), mp))
    if not out:
        raise ValueError("locked target has no mixture_weights contexts")
    return out


def _collect_branch_outcomes(
    *,
    checkpoint: str,
    fixed_z: int,
    opponent: str,
    map_name: str,
    cell_seed: int,
    episodes_per_context: int,
    device: str,
    max_decision_steps: int,
) -> dict[int, float]:
    """One context, one branch: build the env/model ONCE, run all episodes."""
    protocol = ForcedZProtocol(
        checkpoint=checkpoint,
        opponents=(opponent,),
        maps=(map_name,),
        episodes_per_cell=episodes_per_context,
        base_seed=cell_seed,
        max_decision_steps=max_decision_steps,
        device=device,
    )
    env = _make_env(protocol, map_name, cell_seed)
    try:
        model = load_custom_ppo_policy(checkpoint, env.observation_space, env.action_space, device=device)
        if hasattr(model, "fixed_latent_strategy"):
            model.fixed_latent_strategy = True
        if hasattr(model, "fixed_latent_strategy_id"):
            model.fixed_latent_strategy_id = int(fixed_z)
        episodes = run_eval_episodes(
            checkpoint, env, int(episodes_per_context), device, opponent,
            fixed_latent_id=int(fixed_z), deterministic=True, latent_eval_seed=cell_seed,
            preloaded_model=model, collect_behavior_mean=False, progress_every=0,
        )
    finally:
        env.close()
    return {int(ep["episode_index"]): float(ep["win_margin"]) for ep in episodes}


def _capture_c0_for_context(
    *, checkpoint: str, opponent: str, map_name: str, cell_seed: int,
    episodes_per_context: int, device: str, max_decision_steps: int,
) -> dict[int, np.ndarray]:
    """One env built ONCE for this context, reused across all episode resets.

    Must reproduce run_eval_episodes' exact legacy-path per-episode global-RNG
    seeding (random/np/torch/cuda + env.seed) before each reset -- env.seed()
    alone is not sufficient, since env.reset() spawn/pose randomization can
    also draw from the global numpy/random streams. Otherwise the captured c0
    would not correspond to the same episode state that produced outcome_z0 /
    outcome_z3 for that episode index.
    """
    import random

    import torch

    protocol = ForcedZProtocol(
        checkpoint=checkpoint, opponents=(opponent,), maps=(map_name,),
        episodes_per_cell=episodes_per_context, base_seed=cell_seed,
        max_decision_steps=max_decision_steps, device=device,
    )
    env = _make_env(protocol, map_name, cell_seed)
    out: dict[int, np.ndarray] = {}
    try:
        env.env_method("set_phase", opponent)
        env.env_method("set_next_opponent", "SCRIPTED", opponent)
        try:
            from rl.stress_schedule import STRESS_BY_PHASE
            env.env_method("set_stress_schedule", STRESS_BY_PHASE)
        except Exception:
            pass
        for ep_idx in range(int(episodes_per_context)):
            seed = cell_seed + ep_idx
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            if hasattr(env, "seed"):
                env.seed(seed)
            env.reset()
            state = env.state()[0]
            out[ep_idx] = np.asarray(state, dtype=np.float64).reshape(-1)
    finally:
        env.close()
    return out


def cmd_collect(args: argparse.Namespace) -> int:
    target = json.loads(Path(args.locked_target_json).read_text(encoding="utf-8"))
    contexts = _contexts_from_mixture(target)
    print(f"Contexts ({len(contexts)}): {contexts}", flush=True)

    out_path = Path(args.output)
    existing_keys: set[tuple[str, str, int]] = set()
    if out_path.is_file() and not args.overwrite:
        with out_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                existing_keys.add((row["opponent"], row["map"], int(row["episode_index"])))
        print(f"Resuming: {len(existing_keys)} units already in {out_path}", flush=True)

    mode = "a" if (out_path.is_file() and not args.overwrite) else "w"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    f = out_path.open(mode, newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=_RAW_FIELDNAMES)
    if mode == "w":
        writer.writeheader()

    t_start = time.time()
    n_written = 0
    for ctx_idx, (opponent, map_name) in enumerate(contexts):
        cell_seed = int(args.base_seed) + 1000 * ctx_idx
        needed = [
            ep for ep in range(int(args.episodes_per_context))
            if (opponent, map_name, ep) not in existing_keys
        ]
        if not needed:
            print(f"  [{opponent}|{map_name}] all {args.episodes_per_context} units already collected, skipping", flush=True)
            continue

        t0 = time.time()
        outcomes_z0 = _collect_branch_outcomes(
            checkpoint=args.checkpoint_z0, fixed_z=0, opponent=opponent, map_name=map_name,
            cell_seed=cell_seed, episodes_per_context=int(args.episodes_per_context),
            device=args.device, max_decision_steps=int(args.max_decision_steps),
        )
        t1 = time.time()
        print(f"  [{opponent}|{map_name}] z0 collected in {t1 - t0:.1f}s", flush=True)

        outcomes_z3 = _collect_branch_outcomes(
            checkpoint=args.checkpoint_z3, fixed_z=3, opponent=opponent, map_name=map_name,
            cell_seed=cell_seed, episodes_per_context=int(args.episodes_per_context),
            device=args.device, max_decision_steps=int(args.max_decision_steps),
        )
        t2 = time.time()
        print(f"  [{opponent}|{map_name}] z3 collected in {t2 - t1:.1f}s", flush=True)

        c0_map = _capture_c0_for_context(
            checkpoint=args.checkpoint_z0, opponent=opponent, map_name=map_name, cell_seed=cell_seed,
            episodes_per_context=int(args.episodes_per_context), device=args.device,
            max_decision_steps=int(args.max_decision_steps),
        )
        t3 = time.time()
        print(f"  [{opponent}|{map_name}] c0 captured in {t3 - t2:.1f}s (1 env, {args.episodes_per_context} resets)", flush=True)

        for ep_idx in sorted(set(outcomes_z0) & set(outcomes_z3) & set(c0_map)):
            writer.writerow({
                "opponent": opponent, "map": map_name, "episode_index": ep_idx,
                "episode_seed": cell_seed + ep_idx,
                "c0_json": json.dumps(c0_map[ep_idx].tolist()),
                "outcome_z0": outcomes_z0[ep_idx], "outcome_z3": outcomes_z3[ep_idx],
            })
            n_written += 1
        f.flush()
        elapsed = time.time() - t_start
        done_contexts = ctx_idx + 1
        eta = (elapsed / done_contexts) * (len(contexts) - done_contexts) if done_contexts > 0 else 0.0
        print(f"  [{opponent}|{map_name}] context done ({done_contexts}/{len(contexts)} contexts, "
              f"{n_written} units written this run, elapsed={elapsed:.0f}s, ETA={eta:.0f}s)", flush=True)

    f.close()
    print(f"\nWrote {n_written} new rows -> {out_path}")
    return 0


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------

def _load_raw_units(path: Path, feature_column: str = "c0_json") -> tuple[list[tuple[str, str, int]], np.ndarray, np.ndarray, np.ndarray]:
    units: list[tuple[str, str, int]] = []
    c0_list: list[np.ndarray] = []
    z0_list: list[float] = []
    z3_list: list[float] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            units.append((row["opponent"], row["map"], int(row["episode_index"])))
            c0_list.append(np.asarray(json.loads(row[feature_column]), dtype=np.float64))
            z0_list.append(float(row["outcome_z0"]))
            z3_list.append(float(row["outcome_z3"]))
    return units, np.stack(c0_list, axis=0), np.asarray(z0_list), np.asarray(z3_list)


def _deterministic_split(units: list[tuple[str, str, int]]) -> tuple[np.ndarray, np.ndarray]:
    by_context: dict[tuple[str, str], list[int]] = {}
    for i, (opponent, map_name, ep_idx) in enumerate(units):
        by_context.setdefault((opponent, map_name), []).append(i)
    selection_idx: list[int] = []
    heldout_idx: list[int] = []
    for ctx, idxs in by_context.items():
        idxs_sorted = sorted(idxs, key=lambda i: units[i][2])
        half = len(idxs_sorted) // 2
        selection_idx.extend(idxs_sorted[:half])
        heldout_idx.extend(idxs_sorted[half:])
    return np.array(sorted(selection_idx)), np.array(sorted(heldout_idx))


class _ConstSelector:
    def __init__(self, pick: int) -> None:
        self.pick = pick

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.full((X.shape[0],), self.pick, dtype=int)


class _AdvantageThresholdSelector:
    """Regress A = outcome_z0 - outcome_z3 on c0. Output follows the same
    label01 convention as the classifiers (1 = pick z3): predict 1 iff the
    predicted advantage favors z3 by more than margin, i.e. A_hat < -margin."""

    def __init__(self, margin: float = 0.0) -> None:
        self.margin = float(margin)
        self._reg = None

    def fit(self, X, advantage):
        from sklearn.linear_model import Ridge
        self._reg = Ridge(alpha=1.0)
        self._reg.fit(X, advantage)
        return self

    def predict(self, X):
        pred = self._reg.predict(X)
        return (pred < -self.margin).astype(int)


def _build_candidate_selectors() -> dict[str, Any]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import GradientBoostingClassifier

    return {
        "always_z0": _ConstSelector(0),
        "always_z3": _ConstSelector(1),
        "logistic": LogisticRegression(max_iter=2000, C=1.0),
        "logistic_balanced": LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced"),
        "tree_depth2": DecisionTreeClassifier(max_depth=2, min_samples_leaf=5, random_state=0),
        "gbt_small": GradientBoostingClassifier(n_estimators=30, max_depth=2, learning_rate=0.1, random_state=0),
        "advantage_ridge": _AdvantageThresholdSelector(margin=0.0),
    }


def _select_best_via_nested_cv(
    X: np.ndarray, label01: np.ndarray, advantage: np.ndarray,
    outcomes_z0: np.ndarray, outcomes_z3: np.ndarray, *, n_folds: int, seed: int,
) -> tuple[str, Any]:
    from sklearn.model_selection import KFold

    candidates = _build_candidate_selectors()
    n = X.shape[0]
    folds = max(2, min(n_folds, n))
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    scores: dict[str, float] = {}
    for name, selector in candidates.items():
        fold_means = []
        for train_idx, test_idx in kf.split(X):
            sel = _build_candidate_selectors()[name]
            try:
                if name == "advantage_ridge":
                    sel.fit(X[train_idx], advantage[train_idx])
                else:
                    sel.fit(X[train_idx], label01[train_idx])
                pred = sel.predict(X[test_idx])
            except Exception:
                continue
            picks_z3 = np.asarray(pred).astype(bool)
            realized = np.where(picks_z3, outcomes_z3[test_idx], outcomes_z0[test_idx])
            fold_means.append(float(realized.mean()))
        scores[name] = float(np.mean(fold_means)) if fold_means else float("-inf")
    best_name = max(scores, key=scores.get)
    print(f"  Nested-CV scores (mean realized outcome, {folds}-fold, selection split only):")
    for name, score in sorted(scores.items(), key=lambda kv: -kv[1]):
        marker = " <== selected" if name == best_name else ""
        print(f"    {name:20s} {score:.4f}{marker}")
    best_selector = _build_candidate_selectors()[best_name]
    if best_name == "advantage_ridge":
        best_selector.fit(X, advantage)
    else:
        best_selector.fit(X, label01)
    return best_name, best_selector


def _paired_bootstrap(
    *, outcomes_z0: np.ndarray, outcomes_z3: np.ndarray, selector_picks_z3: np.ndarray,
    n_boot: int, seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    n = outcomes_z0.shape[0]
    outcomes_selector = np.where(selector_picks_z3, outcomes_z3, outcomes_z0)
    outcomes_hindsight = np.maximum(outcomes_z0, outcomes_z3)

    def _point(idx: np.ndarray) -> dict[str, float]:
        v_z0 = float(outcomes_z0[idx].mean())
        v_z3 = float(outcomes_z3[idx].mean())
        v_sel = float(outcomes_selector[idx].mean())
        v_hind = float(outcomes_hindsight[idx].mean())
        best_fixed = max(v_z0, v_z3)
        return {
            "V_z0": v_z0, "V_z3": v_z3, "V_legal_selector": v_sel, "V_hindsight_oracle": v_hind,
            "best_fixed": best_fixed, "delta_usable": v_sel - best_fixed, "delta_oracle": v_hind - best_fixed,
        }

    point = _point(np.arange(n))
    boot_usable = np.empty(int(n_boot))
    boot_oracle = np.empty(int(n_boot))
    for b in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        rep = _point(idx)
        boot_usable[b] = rep["delta_usable"]
        boot_oracle[b] = rep["delta_oracle"]
    lo_u, hi_u = np.quantile(boot_usable, [0.025, 0.975])
    lo_o, hi_o = np.quantile(boot_oracle, [0.025, 0.975])
    return {
        **point, "n_held_out_units": int(n), "bootstrap_n": int(n_boot),
        "delta_usable_CI95": [float(lo_u), float(hi_u)], "delta_usable_LCB": float(lo_u),
        "delta_usable_LCB_gt_0": bool(lo_u > 0.0),
        "delta_oracle_CI95": [float(lo_o), float(hi_o)], "delta_oracle_LCB": float(lo_o),
        "delta_oracle_LCB_gt_0": bool(lo_o > 0.0),
    }


def cmd_analyze(args: argparse.Namespace) -> int:
    units, c0_arr, outcomes_z0, outcomes_z3 = _load_raw_units(Path(args.raw_csv), feature_column=args.feature_column)
    print(f"Loaded {len(units)} raw matched units from {args.raw_csv} (feature_column={args.feature_column!r})")
    advantage = outcomes_z0 - outcomes_z3  # A = R(z0) - R(z3)
    print(f"Advantage A=R(z0)-R(z3): mean={advantage.mean():.4f} std={advantage.std():.4f} "
          f"frac_z0_better={(advantage > 0).mean():.3f} frac_tie={(advantage == 0).mean():.3f}")

    # Held-out split is derived from the PRIMARY raw CSV only, and frozen from
    # this point on -- extra selection-only data (below) is never eligible to
    # land in held-out, so the promotion test at the end still evaluates on
    # exactly the same untouched split as every prior run of this script.
    selection_idx, heldout_idx = _deterministic_split(units)
    print(f"Selection split (primary file): {len(selection_idx)} units | "
          f"Held-out split (frozen): {len(heldout_idx)} units")

    c0_sel, z0_sel, z3_sel, adv_sel = c0_arr[selection_idx], outcomes_z0[selection_idx], outcomes_z3[selection_idx], advantage[selection_idx]

    if args.extra_selection_csv:
        extra_units, extra_c0, extra_z0, extra_z3 = _load_raw_units(Path(args.extra_selection_csv), feature_column=args.feature_column)
        extra_adv = extra_z0 - extra_z3
        print(f"Extra selection-only data: {len(extra_units)} units from {args.extra_selection_csv} "
              f"(added to selection pool only; held-out split above is unaffected)")
        c0_sel = np.concatenate([c0_sel, extra_c0], axis=0)
        z0_sel = np.concatenate([z0_sel, extra_z0], axis=0)
        z3_sel = np.concatenate([z3_sel, extra_z3], axis=0)
        adv_sel = np.concatenate([adv_sel, extra_adv], axis=0)

    label01_sel = (z3_sel > z0_sel).astype(int)  # 1 = z3 better
    print(f"Selection pool (after any extra data): n={len(label01_sel)}  "
          f"z3_better={int(label01_sel.sum())} z0_better_or_tie={int(len(label01_sel) - label01_sel.sum())}")

    best_name, best_selector = _select_best_via_nested_cv(
        c0_sel, label01_sel, adv_sel, z0_sel, z3_sel,
        n_folds=int(args.cv_folds), seed=int(args.seed),
    )
    print(f"\nChosen selector (nested CV, selection split only): {best_name}")

    heldout_pred = best_selector.predict(c0_arr[heldout_idx])
    heldout_picks_z3 = np.asarray(heldout_pred).astype(bool)

    result = _paired_bootstrap(
        outcomes_z0=outcomes_z0[heldout_idx], outcomes_z3=outcomes_z3[heldout_idx],
        selector_picks_z3=heldout_picks_z3, n_boot=int(args.bootstrap_samples), seed=int(args.seed) + 99,
    )

    report = {
        "protocol": "v6i26_usable_selector_eval_v2",
        "raw_csv": str(args.raw_csv),
        "extra_selection_csv": str(args.extra_selection_csv) if args.extra_selection_csv else None,
        "n_matched_units": len(units),
        "n_selection_units_primary": int(len(selection_idx)),
        "n_selection_units_total": int(len(label01_sel)),
        "n_heldout_units": int(len(heldout_idx)),
        "advantage_summary": {
            "mean": float(advantage.mean()), "std": float(advantage.std()),
            "frac_z0_better": float((advantage > 0).mean()), "frac_tie": float((advantage == 0).mean()),
        },
        "selection_split_label_balance": {
            "n_z3_better": int(label01_sel.sum()), "n_z0_better_or_tie": int(len(label01_sel) - label01_sel.sum()),
        },
        "chosen_selector": best_name,
        "heldout_selector_picks": {
            "n_picked_z0": int((~heldout_picks_z3).sum()), "n_picked_z3": int(heldout_picks_z3.sum()),
        },
        **result,
        "promotion_rule": "delta_usable_LCB_gt_0 (policy-distinction gate evaluated separately)",
        "verdict": "USABLE_REPERTOIRE_PASS" if result["delta_usable_LCB_gt_0"] else "USABLE_REPERTOIRE_HOLD_OR_FAIL",
    }
    write_json(Path(args.output), report)

    print("\n" + "=" * 72)
    print("V6I26 usable-repertoire test v2 (advantage-based, nested-CV selector ladder)")
    print("=" * 72)
    print(f"V_z0={result['V_z0']:.4f}  V_z3={result['V_z3']:.4f}  best_fixed={result['best_fixed']:.4f}")
    print(f"V_hindsight_oracle={result['V_hindsight_oracle']:.4f}  delta_oracle={result['delta_oracle']:.4f}  "
          f"CI95={result['delta_oracle_CI95']} (upper bound -- not deployable evidence)")
    print(f"chosen_selector={best_name}")
    print(f"V_legal_selector={result['V_legal_selector']:.4f}  delta_usable={result['delta_usable']:.4f}  "
          f"CI95={result['delta_usable_CI95']}")
    print(f"LCB(delta_usable)={result['delta_usable_LCB']:.4f} > 0 ? {result['delta_usable_LCB_gt_0']}")
    print(f"verdict={report['verdict']}")
    print(f"wrote {args.output}")
    return 0 if result["delta_usable_LCB_gt_0"] else 1


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pc = sub.add_parser("collect")
    pc.add_argument("--checkpoint-z0", required=True)
    pc.add_argument("--checkpoint-z3", required=True)
    pc.add_argument("--locked-target-json", required=True)
    pc.add_argument("--episodes-per-context", type=int, default=32)
    pc.add_argument("--base-seed", type=int, default=101)
    pc.add_argument("--device", default="cuda")
    pc.add_argument("--max-decision-steps", type=int, default=240)
    pc.add_argument("--output", required=True, help="Raw per-unit CSV path (resumable)")
    pc.add_argument("--overwrite", action="store_true", help="Ignore any existing CSV and start fresh")
    pc.set_defaults(func=cmd_collect)

    pa = sub.add_parser("analyze")
    pa.add_argument("--raw-csv", required=True, help="Primary raw CSV; defines the frozen held-out split")
    pa.add_argument("--extra-selection-csv", default=None,
                     help="Optional additional raw CSV whose units are added to the SELECTION pool only "
                          "(never eligible for held-out) -- for growing selection data without touching "
                          "the frozen held-out split from the primary file")
    pa.add_argument("--feature-column", default="c0_json",
                     help="CSV column holding the JSON feature vector (c0 for Level 1a, context_json for "
                          "Level 1b) -- must match between --raw-csv and --extra-selection-csv")
    pa.add_argument("--cv-folds", type=int, default=5)
    pa.add_argument("--bootstrap-samples", type=int, default=2000)
    pa.add_argument("--seed", type=int, default=0)
    pa.add_argument("--output", required=True)
    pa.set_defaults(func=cmd_analyze)

    return p.parse_args()


def main() -> int:
    args = _parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
