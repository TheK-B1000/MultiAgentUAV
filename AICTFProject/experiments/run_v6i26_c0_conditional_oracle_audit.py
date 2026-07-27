#!/usr/bin/env python3
"""V6I26 repeated-rollout advantage-stability audit (Step 2 of the selector-
observability plan).

The Level 1a usable-selector test (run_v6i26_usable_selector_eval.py) showed
that no bounded selector -- constant, logistic, class-weighted logistic,
depth-2 tree, small GBT, advantage-regression -- beats the trivial "always
z0" baseline when predicting the winner from c0 alone. That result is
consistent with two very different underlying truths, and nothing in that
test distinguishes them:

  A) c0 genuinely does not determine which branch is better: for a FIXED
     initial state, re-running from that same state with different
     downstream randomness sometimes favors z0, sometimes z3. No selector,
     however expressive, could ever have decoded a stable rule from c0.

  B) c0 does determine a stable preference, but the single-draw-per-unit
     dataset (each c0 seen exactly once) was too noisy/small for any
     selector in the ladder to find it.

This script tells the two apart by holding c0 EXACTLY fixed (same reset
seed -> same post-reset environment state, verified bit-identical every
repeat) and re-running matched (z0, z3) episode pairs from it with several
different "continuation" seeds -- i.e. different draws of the opponent's
step-time stochastic behavior (game_field_gpu's internal self._rng, which
drives both spawn randomization at reset AND opponent decision noise during
stepping; see gpu_env/_core/_scripted_red.py's torch.rand(..., generator=
self._rng) calls). Since env.seed() only reseeds that generator (it does not
touch already-computed world state), the "reconstruct" protocol is exact:

  1. env.seed(reset_seed); obs = env.reset()          # produces c0
  2. capture c0 = env.state()[0]; assert unchanged vs. this state's first draw
  3. env.seed(continuation_seed)                       # future randomness only
  4. run to completion with model (fixed z, deterministic actions)

Repeating (3)-(4) with several continuation_seeds, once per branch, with the
SAME continuation_seed shared between the z0 and z3 runs of a given repeat,
gives a matched (z0, z3) outcome pair per (c0, continuation) unit -- exactly
the "collection contract" the user specified.

`collect` builds these raw units (one row per (context, state, repeat)) into
a resumable CSV. `analyze` computes, per c0 state: mean paired advantage,
SE/CI, P(A>0|c0), sign-stability rate, and a within-c0 vs. between-c0
variance decomposition; then runs a "conditional-oracle" audit -- estimate
each state's preferred branch from a FIT half of its repeats, freeze that
per-state choice, and evaluate it (paired bootstrap, same LCB convention as
the Level 1 usable-selector test) on the disjoint TEST half of repeats.

Read-only with respect to training: loads two checkpoints for inference only.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

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
from experiments.run_v6i26_usable_selector_eval import _contexts_from_mixture  # noqa: E402
from rl.custom_ppo import load_custom_ppo_policy  # noqa: E402

_RAW_FIELDNAMES = [
    "opponent", "map", "state_idx", "reset_seed", "repeat_idx", "continuation_seed",
    "c0_json", "outcome_z0", "outcome_z3",
]


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _reset_to_c0(env, reset_seed: int) -> tuple[Any, np.ndarray]:
    _seed_all(reset_seed)
    if hasattr(env, "seed"):
        env.seed(reset_seed)
    obs = env.reset()
    c0 = np.asarray(env.state()[0], dtype=np.float64).reshape(-1)
    return obs, c0


def _run_episode_to_completion(env, model, obs, *, continuation_seed: int) -> float:
    """Reseed for the continuation (post-reset randomness only), then step
    to completion under a fixed-z, deterministic-action policy. Returns
    win_margin = blue_score - red_score."""
    _seed_all(continuation_seed)
    if hasattr(env, "seed"):
        env.seed(continuation_seed)
    if hasattr(model, "reset_strategy"):
        model.reset_strategy()
    while True:
        single = {
            k: v[0] if hasattr(v, "shape") and len(v.shape) > 1 and v.shape[0] == 1 else v
            for k, v in obs.items()
        }
        try:
            single["global_state"] = env.state()[0]
        except Exception:
            pass
        act, _ = model.predict(single, deterministic=True)
        env.step_async(act)
        obs, rew, done, infos = env.step_wait()
        if done.any():
            info = infos[0] if len(infos) > 0 else {}
            ep_res = info.get("episode_result", info)
            bs = int(ep_res.get("blue_score", 0))
            rs = int(ep_res.get("red_score", 0))
            if hasattr(model, "reset_strategy"):
                model.reset_strategy()
            return float(bs - rs)


def _load_fixed_z_model(checkpoint: str, fixed_z: int, env, device: str):
    model = load_custom_ppo_policy(checkpoint, env.observation_space, env.action_space, device=device)
    if hasattr(model, "fixed_latent_strategy"):
        model.fixed_latent_strategy = True
    if hasattr(model, "fixed_latent_strategy_id"):
        model.fixed_latent_strategy_id = int(fixed_z)
    return model


# ---------------------------------------------------------------------------
# collect
# ---------------------------------------------------------------------------

def cmd_collect(args: argparse.Namespace) -> int:
    target = json.loads(Path(args.locked_target_json).read_text(encoding="utf-8"))
    contexts = _contexts_from_mixture(target)
    print(f"Contexts ({len(contexts)}): {contexts}", flush=True)
    print(f"States/context={args.states_per_context}  Repeats/state={args.repeats_per_state}  "
          f"-> {len(contexts) * args.states_per_context * args.repeats_per_state} matched units planned", flush=True)

    out_path = Path(args.output)
    existing_keys: set[tuple[str, str, int, int]] = set()
    if out_path.is_file() and not args.overwrite:
        with out_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                existing_keys.add((row["opponent"], row["map"], int(row["state_idx"]), int(row["repeat_idx"])))
        print(f"Resuming: {len(existing_keys)} units already in {out_path}", flush=True)

    mode = "a" if (out_path.is_file() and not args.overwrite) else "w"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    f = out_path.open(mode, newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=_RAW_FIELDNAMES)
    if mode == "w":
        writer.writeheader()

    t_start = time.time()
    n_written = 0
    n_mismatches = 0
    for ctx_idx, (opponent, map_name) in enumerate(contexts):
        any_needed = any(
            (opponent, map_name, s, r) not in existing_keys
            for s in range(int(args.states_per_context))
            for r in range(int(args.repeats_per_state))
        )
        if not any_needed:
            print(f"  [{opponent}|{map_name}] all units already collected, skipping", flush=True)
            continue

        protocol = ForcedZProtocol(
            checkpoint=args.checkpoint_z0, opponents=(opponent,), maps=(map_name,),
            episodes_per_cell=1, base_seed=args.base_reset_seed,
            max_decision_steps=int(args.max_decision_steps), device=args.device,
        )
        env = _make_env(protocol, map_name, args.base_reset_seed)
        try:
            env.env_method("set_phase", opponent)
            env.env_method("set_next_opponent", "SCRIPTED", opponent)
            try:
                from rl.stress_schedule import STRESS_BY_PHASE
                env.env_method("set_stress_schedule", STRESS_BY_PHASE)
            except Exception:
                pass

            model_z0 = _load_fixed_z_model(args.checkpoint_z0, 0, env, args.device)
            model_z3 = _load_fixed_z_model(args.checkpoint_z3, 3, env, args.device)

            t_ctx0 = time.time()
            for state_idx in range(int(args.states_per_context)):
                reset_seed = int(args.base_reset_seed) + 1000 * ctx_idx + state_idx
                c0_ref: np.ndarray | None = None
                for repeat_idx in range(int(args.repeats_per_state)):
                    if (opponent, map_name, state_idx, repeat_idx) in existing_keys:
                        continue
                    continuation_seed = int(args.base_continuation_seed) + 1000 * ctx_idx + 100 * state_idx + repeat_idx

                    obs, c0 = _reset_to_c0(env, reset_seed)
                    if c0_ref is None:
                        c0_ref = c0
                    elif not np.array_equal(c0, c0_ref):
                        n_mismatches += 1
                        print(f"  [WARN] c0 NOT bit-identical at {opponent}|{map_name} state={state_idx} "
                              f"repeat={repeat_idx}: max_abs_diff={np.abs(c0 - c0_ref).max():.6g}", flush=True)
                    outcome_z0 = _run_episode_to_completion(env, model_z0, obs, continuation_seed=continuation_seed)

                    obs, c0_check = _reset_to_c0(env, reset_seed)
                    if not np.array_equal(c0_check, c0_ref):
                        n_mismatches += 1
                        print(f"  [WARN] c0 NOT bit-identical (z3 leg) at {opponent}|{map_name} state={state_idx} "
                              f"repeat={repeat_idx}: max_abs_diff={np.abs(c0_check - c0_ref).max():.6g}", flush=True)
                    outcome_z3 = _run_episode_to_completion(env, model_z3, obs, continuation_seed=continuation_seed)

                    writer.writerow({
                        "opponent": opponent, "map": map_name, "state_idx": state_idx,
                        "reset_seed": reset_seed, "repeat_idx": repeat_idx, "continuation_seed": continuation_seed,
                        "c0_json": json.dumps(c0_ref.tolist()),
                        "outcome_z0": outcome_z0, "outcome_z3": outcome_z3,
                    })
                    n_written += 1
                f.flush()
        finally:
            env.close()

        elapsed = time.time() - t_start
        done_contexts = ctx_idx + 1
        eta = (elapsed / done_contexts) * (len(contexts) - done_contexts) if done_contexts > 0 else 0.0
        print(f"  [{opponent}|{map_name}] context done ({done_contexts}/{len(contexts)} contexts, "
              f"{n_written} units written this run, {n_mismatches} c0 mismatches so far, "
              f"elapsed={elapsed:.0f}s, ETA={eta:.0f}s)", flush=True)

    f.close()
    print(f"\nWrote {n_written} new rows ({n_mismatches} c0 mismatches detected) -> {out_path}")
    return 0


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------

def _load_raw(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({
                "opponent": row["opponent"], "map": row["map"],
                "state_idx": int(row["state_idx"]), "repeat_idx": int(row["repeat_idx"]),
                "outcome_z0": float(row["outcome_z0"]), "outcome_z3": float(row["outcome_z3"]),
            })
    return rows


def _group_by_state(rows: list[dict[str, Any]]) -> dict[tuple[str, str, int], list[dict[str, Any]]]:
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["opponent"], row["map"], row["state_idx"])
        groups.setdefault(key, []).append(row)
    return groups


def _per_state_stats(groups: dict[tuple[str, str, int], list[dict[str, Any]]]) -> list[dict[str, Any]]:
    out = []
    for key, unit_rows in sorted(groups.items()):
        adv = np.array([r["outcome_z0"] - r["outcome_z3"] for r in unit_rows], dtype=np.float64)
        n = adv.shape[0]
        mean_a = float(adv.mean())
        se = float(adv.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
        p_z0_better = float((adv > 0).mean())
        p_z3_better = float((adv < 0).mean())
        p_tie = float((adv == 0).mean())
        sign_stability = max(p_z0_better, p_z3_better)
        out.append({
            "opponent": key[0], "map": key[1], "state_idx": key[2], "n_repeats": int(n),
            "mean_advantage": mean_a, "se_advantage": se,
            "ci95": [mean_a - 1.96 * se, mean_a + 1.96 * se] if n > 1 else [float("nan"), float("nan")],
            "p_z0_better": p_z0_better, "p_z3_better": p_z3_better, "p_tie": p_tie,
            "sign_stability_rate": sign_stability,
            "within_state_var": float(adv.var(ddof=1)) if n > 1 else 0.0,
        })
    return out


def _variance_decomposition(per_state: list[dict[str, Any]], groups: dict[tuple[str, str, int], list[dict[str, Any]]]) -> dict[str, float]:
    all_adv = np.concatenate([
        np.array([r["outcome_z0"] - r["outcome_z3"] for r in rows], dtype=np.float64)
        for rows in groups.values()
    ])
    total_var = float(all_adv.var(ddof=1))
    within_mean = float(np.mean([s["within_state_var"] for s in per_state]))
    state_means = np.array([s["mean_advantage"] for s in per_state], dtype=np.float64)
    between_var = float(state_means.var(ddof=1)) if len(state_means) > 1 else 0.0
    return {
        "total_advantage_variance": total_var,
        "mean_within_state_variance": within_mean,
        "between_state_variance": between_var,
        "between_over_total": (between_var / total_var) if total_var > 1e-12 else 0.0,
    }


def _conditional_oracle_audit(groups: dict[tuple[str, str, int], list[dict[str, Any]]], *, n_boot: int, seed: int) -> dict[str, Any]:
    """Fit-half decides each state's preferred branch (from that state's OWN
    repeats only); test-half evaluates the frozen per-state choice. Same
    LCB-based promotion rule as the Level 1 usable-selector test, but the
    'selector' here is an oracle over predeclared state identity (fit-half
    mean advantage sign), not a c0-vector classifier -- this isolates
    whether c0 CAN in principle support a stable per-state rule at all.

    Uncertainty is estimated via a two-stage CLUSTER bootstrap (resample c0
    states with replacement, then resample that state's test-half repeats
    with replacement) rather than a flat resample over individual test rows.
    Repeats sharing a c0 state are correlated (same initial state, same
    fit-half decision governs all of them), so a flat bootstrap would treat
    them as independent and understate the true interval width."""
    state_test: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []  # per state: (z0, z3, sel) test arrays
    for key, unit_rows in sorted(groups.items()):
        unit_rows_sorted = sorted(unit_rows, key=lambda r: r["repeat_idx"])
        half = len(unit_rows_sorted) // 2
        fit_rows, test_rows = unit_rows_sorted[:half], unit_rows_sorted[half:]
        if not fit_rows or not test_rows:
            continue
        fit_adv_mean = float(np.mean([r["outcome_z0"] - r["outcome_z3"] for r in fit_rows]))
        pick_z0 = bool(fit_adv_mean >= 0)  # ties default to z0, matching label01 convention elsewhere
        z0_arr = np.array([r["outcome_z0"] for r in test_rows], dtype=np.float64)
        z3_arr = np.array([r["outcome_z3"] for r in test_rows], dtype=np.float64)
        sel_arr = z0_arr if pick_z0 else z3_arr
        state_test.append((z0_arr, z3_arr, sel_arr))

    def _pooled_point(states: list[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> dict[str, float]:
        z0_all = np.concatenate([s[0] for s in states])
        z3_all = np.concatenate([s[1] for s in states])
        sel_all = np.concatenate([s[2] for s in states])
        v_z0, v_z3, v_sel = float(z0_all.mean()), float(z3_all.mean()), float(sel_all.mean())
        best_fixed = max(v_z0, v_z3)
        return {"V_z0": v_z0, "V_z3": v_z3, "V_conditional_oracle": v_sel, "best_fixed": best_fixed,
                "delta_conditional_oracle": v_sel - best_fixed}

    point = _pooled_point(state_test)
    n_states = len(state_test)
    n_test_units = int(sum(s[0].shape[0] for s in state_test))

    rng = np.random.default_rng(seed)
    boot = np.empty(int(n_boot))
    for b in range(int(n_boot)):
        state_ids = rng.integers(0, n_states, size=n_states)
        resampled = []
        for sid in state_ids:
            z0_arr, z3_arr, sel_arr = state_test[sid]
            k = z0_arr.shape[0]
            rep_idx = rng.integers(0, k, size=k)
            resampled.append((z0_arr[rep_idx], z3_arr[rep_idx], sel_arr[rep_idx]))
        boot[b] = _pooled_point(resampled)["delta_conditional_oracle"]
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return {**point, "n_test_units": n_test_units, "n_states_bootstrapped": int(n_states), "bootstrap_n": int(n_boot),
            "bootstrap_method": "cluster_bootstrap_states_then_repeats",
            "delta_conditional_oracle_CI95": [float(lo), float(hi)],
            "delta_conditional_oracle_LCB": float(lo), "delta_conditional_oracle_LCB_gt_0": bool(lo > 0.0)}


def _icc1(per_state: list[dict[str, Any]]) -> dict[str, float]:
    """One-way random-effects ICC(1): the fraction of total advantage
    variance attributable to between-c0-state differences vs. within-state
    continuation-to-continuation noise, via the classic ANOVA mean-square
    estimator (harmonic-mean k correction handles mild repeat-count
    imbalance from resumed/partial collection)."""
    ns = np.array([s["n_repeats"] for s in per_state], dtype=np.float64)
    n_states = len(per_state)
    if n_states < 2 or np.any(ns < 2):
        return {"icc1": float("nan"), "msb": float("nan"), "msw": float("nan"), "k_harmonic_mean": float("nan"),
                "var_between_hat": float("nan"), "var_within_hat": float("nan")}
    means = np.array([s["mean_advantage"] for s in per_state], dtype=np.float64)
    within_vars = np.array([s["within_state_var"] for s in per_state], dtype=np.float64)
    k_bar = float(n_states / np.sum(1.0 / ns))
    msw = float(np.average(within_vars, weights=ns - 1))
    grand_mean = float(np.average(means, weights=ns))
    msb = float(np.sum(ns * (means - grand_mean) ** 2) / (n_states - 1))
    var_within = msw
    var_between = max(0.0, (msb - msw) / k_bar)
    denom = var_between + var_within
    icc = (var_between / denom) if denom > 1e-12 else 0.0
    return {"icc1": icc, "msb": msb, "msw": msw, "k_harmonic_mean": k_bar,
            "var_between_hat": var_between, "var_within_hat": var_within}


def cmd_analyze(args: argparse.Namespace) -> int:
    rows = _load_raw(Path(args.raw_csv))
    groups = _group_by_state(rows)
    print(f"Loaded {len(rows)} raw units across {len(groups)} c0 states from {args.raw_csv}")

    per_state = _per_state_stats(groups)
    for s in per_state:
        print(f"  [{s['opponent']}|{s['map']} state={s['state_idx']}] n={s['n_repeats']} "
              f"mean_A={s['mean_advantage']:+.3f} CI95={[round(v,3) for v in s['ci95']]} "
              f"P(z0>)={s['p_z0_better']:.2f} P(z3>)={s['p_z3_better']:.2f} P(tie)={s['p_tie']:.2f} "
              f"sign_stability={s['sign_stability_rate']:.2f}")

    var_decomp = _variance_decomposition(per_state, groups)
    print(f"\nVariance decomposition (naive screen): total={var_decomp['total_advantage_variance']:.4f} "
          f"within_state_mean={var_decomp['mean_within_state_variance']:.4f} "
          f"between_state={var_decomp['between_state_variance']:.4f} "
          f"between/total={var_decomp['between_over_total']:.3f}")

    icc = _icc1(per_state)
    print(f"ICC(1) one-way random-effects estimate: icc1={icc['icc1']:.3f} "
          f"(var_between_hat={icc['var_between_hat']:.4f} var_within_hat={icc['var_within_hat']:.4f} "
          f"k_harmonic_mean={icc['k_harmonic_mean']:.2f})  "
          f"-- fraction of advantage variance explained by c0 identity")

    mean_sign_stability = float(np.mean([s["sign_stability_rate"] for s in per_state]))
    frac_states_stable = float(np.mean([s["sign_stability_rate"] >= args.stability_threshold for s in per_state]))
    print(f"Mean sign-stability rate across states: {mean_sign_stability:.3f}  "
          f"(fraction of states >= {args.stability_threshold} stable: {frac_states_stable:.3f})")

    oracle = _conditional_oracle_audit(groups, n_boot=int(args.bootstrap_samples), seed=int(args.seed))
    print(f"\nConditional-oracle audit (fit-half decides, test-half evaluates, "
          f"cluster bootstrap over {oracle['n_states_bootstrapped']} states):")
    print(f"  V_z0={oracle['V_z0']:.4f}  V_z3={oracle['V_z3']:.4f}  best_fixed={oracle['best_fixed']:.4f}")
    print(f"  V_conditional_oracle={oracle['V_conditional_oracle']:.4f}  "
          f"delta={oracle['delta_conditional_oracle']:.4f}  CI95={oracle['delta_conditional_oracle_CI95']}")
    print(f"  LCB={oracle['delta_conditional_oracle_LCB']:.4f} > 0 ? {oracle['delta_conditional_oracle_LCB_gt_0']}")

    icc_val = icc["icc1"]
    if np.isnan(icc_val):
        icc_val = var_decomp["between_over_total"]  # fallback screen if ICC undefined (e.g. unbalanced n<2)
    if icc_val < 0.2 or mean_sign_stability < 0.65:
        outcome = "OUTCOME_A_ADVANTAGE_UNSTABLE_GIVEN_C0"
    elif oracle["delta_conditional_oracle_LCB_gt_0"]:
        outcome = "OUTCOME_B_STABLE_AND_USABLE_SIGNAL"
    else:
        outcome = "OUTCOME_B_STABLE_BUT_UNDERSAMPLED_OR_WEAK_SIGNAL"
    print(f"\nClassification: {outcome}")

    report = {
        "protocol": "v6i26_c0_conditional_oracle_audit",
        "raw_csv": str(args.raw_csv),
        "n_units": len(rows),
        "n_states": len(groups),
        "per_state": per_state,
        "variance_decomposition": var_decomp,
        "icc1": icc,
        "mean_sign_stability_rate": mean_sign_stability,
        "stability_threshold": float(args.stability_threshold),
        "frac_states_stable": frac_states_stable,
        "conditional_oracle_audit": oracle,
        "classification": outcome,
    }
    write_json(Path(args.output), report)
    print(f"\nwrote {args.output}")
    return 0


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pc = sub.add_parser("collect")
    pc.add_argument("--checkpoint-z0", required=True)
    pc.add_argument("--checkpoint-z3", required=True)
    pc.add_argument("--locked-target-json", required=True)
    pc.add_argument("--states-per-context", type=int, default=6)
    pc.add_argument("--repeats-per-state", type=int, default=10)
    pc.add_argument("--base-reset-seed", type=int, default=31415)
    pc.add_argument("--base-continuation-seed", type=int, default=27182)
    pc.add_argument("--device", default="cuda")
    pc.add_argument("--max-decision-steps", type=int, default=240)
    pc.add_argument("--output", required=True)
    pc.add_argument("--overwrite", action="store_true")
    pc.set_defaults(func=cmd_collect)

    pa = sub.add_parser("analyze")
    pa.add_argument("--raw-csv", required=True)
    pa.add_argument("--bootstrap-samples", type=int, default=2000)
    pa.add_argument("--seed", type=int, default=0)
    pa.add_argument("--stability-threshold", type=float, default=0.8)
    pa.add_argument("--output", required=True)
    pa.set_defaults(func=cmd_analyze)

    return p.parse_args()


def main() -> int:
    args = _parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
