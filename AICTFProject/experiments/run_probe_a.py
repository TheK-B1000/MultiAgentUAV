#!/usr/bin/env python3
"""Offline context predictability probe A.

Answers: given the exact 35-dim router context at episode start, can any
simple classifier choose a z that beats fixed z2 in expected return?

Input:
  Existing forced-z episode_results.csv from experiments/forced_z_runs/

Output:
  experiments/probe_a_runs/<stamp>/
    probe_a_dataset.csv       -- (opponent, map, ep_idx, context[0..33], return_z*, best_z)
    probe_a_results.json      -- probe metrics + return-regret comparisons
    probe_a_summary.txt       -- printable table

Usage
-----
    uv run python experiments/run_probe_a.py \\
        --checkpoint checkpoints/2v2/final_v6i9-mapaware-repertoire-hardpool-refactor-r1-seed1_2v2.zip \\
        --forced-z-csv experiments/forced_z_runs/20260630_040244/episode_results.csv \\
        [--out-dir experiments/probe_a_runs/<stamp>]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

# ---------------------------------------------------------------------------
# Context collection
# ---------------------------------------------------------------------------

def collect_initial_contexts(
    *,
    checkpoint_path: str,
    cell_seeds: dict[tuple[str, str], int],
    n_episodes: int,
    device: str = "cpu",
) -> dict[tuple[str, str, int], np.ndarray]:
    """Replay forced-z episode resets to collect initial env.state() vectors.

    Replicates the exact seeding from run_eval_episodes (legacy path):
      env.seed(cell_seed + ep_idx) -> env.reset() -> env.state()

    Parameters
    ----------
    cell_seeds : dict mapping (opponent, map_name) -> cell_seed int
        Read directly from the CSV to avoid opp_order/map_order assumptions.

    Returns
    -------
    dict mapping (opponent, map_name, episode_index) -> float32 array (34,)
    """
    import random
    import torch
    from game_field_gpu import GPUCTFVecEnv, GPUFieldConfig
    from rl.custom_ppo.inference import read_custom_ppo_metadata

    meta = read_custom_ppo_metadata(checkpoint_path)
    n_agents = int(meta.get("n_blue", 2))

    contexts: dict[tuple[str, str, int], np.ndarray] = {}

    for (opponent, map_name), cell_seed in sorted(cell_seeds.items()):
        env = GPUCTFVecEnv(GPUFieldConfig(
            n_envs=1,
            max_blue_agents=n_agents,
            max_red_agents=n_agents,
            device=device,
            seed=cell_seed,
            map_layout=map_name,  # pass directly, same as forced-z runner
            max_decision_steps=400,
            aquaticus_profile=True,
            rules_profile="OURS",
        ))
        try:
            env.env_method("set_phase", opponent)
            env.env_method("set_next_opponent", "SCRIPTED", opponent)
            for ep_idx in range(n_episodes):
                # Replicate exact seeding from run_eval_episodes legacy path:
                # actual_env_seed = cell_seed + ep_idx -> env.seed(seed) -> env.reset()
                ep_seed = cell_seed + ep_idx
                random.seed(ep_seed)
                np.random.seed(ep_seed)
                torch.manual_seed(ep_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(ep_seed)
                if hasattr(env, "seed"):
                    env.seed(ep_seed)
                env.reset()
                gs = np.asarray(env.state(), dtype=np.float32)
                # env.state() shape is (n_envs, state_dim); take env 0
                if gs.ndim == 2:
                    gs = gs[0]
                contexts[(opponent, map_name, ep_idx)] = gs
        finally:
            env.close()

        print(f"  Collected contexts: {opponent} x {map_name} (cell_seed={cell_seed}, {n_episodes} episodes)")

    return contexts


# ---------------------------------------------------------------------------
# Dataset building
# ---------------------------------------------------------------------------

def build_dataset(
    forced_z_csv: Path,
    contexts: dict[tuple[str, str, int], np.ndarray],
) -> list[dict]:
    """Pivot forced-z CSV and join initial contexts.

    Returns one row per (opponent, map, episode_index) with columns:
      context[0..33], return_z0..z3, best_z, oracle_return,
      fixed_z2_return, fixed_z2_regret, cell_seed, episode_seed
    """
    from collections import defaultdict

    # Load CSV
    rows = []
    with forced_z_csv.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            rows.append(r)

    # Index by (opponent, map, episode_index, latent_z)
    by_episode: dict[tuple, dict[int, float]] = defaultdict(dict)
    meta_by_episode: dict[tuple, dict] = {}

    for r in rows:
        opponent = r["opponent"].upper()
        map_name = r["map"]
        ep_idx = int(r["episode_index"])
        z = int(r["latent_z"])
        ret = float(r["return"])
        key = (opponent, map_name, ep_idx)
        by_episode[key][z] = ret
        if key not in meta_by_episode:
            meta_by_episode[key] = {
                "cell_seed": r.get("cell_seed", ""),
                "episode_seed": r.get("episode_seed", ""),
            }

    dataset = []
    missing_context = 0
    missing_returns = 0

    for key, z_returns in sorted(by_episode.items()):
        opponent, map_name, ep_idx = key

        if not all(z in z_returns for z in range(4)):
            missing_returns += 1
            continue

        ctx = contexts.get(key)
        if ctx is None:
            missing_context += 1
            continue

        returns = [z_returns[z] for z in range(4)]
        best_z = int(np.argmax(returns))
        oracle_return = float(np.max(returns))
        fixed_z2_return = z_returns[2]

        row: dict = {
            "opponent": opponent,
            "map": map_name,
            "episode_index": ep_idx,
            "best_z": best_z,
            "oracle_return": oracle_return,
            "fixed_z2_return": fixed_z2_return,
            "fixed_z2_regret": oracle_return - fixed_z2_return,
        }
        for z in range(4):
            row[f"return_z{z}"] = z_returns[z]
        for i, v in enumerate(ctx):
            row[f"ctx_{i}"] = float(v)
        row.update(meta_by_episode[key])
        dataset.append(row)

    if missing_context:
        print(f"  Warning: {missing_context} episodes missing context (skipped)")
    if missing_returns:
        print(f"  Warning: {missing_returns} episodes missing some z-returns (skipped)")

    return dataset


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------

def run_probes(
    dataset: list[dict],
    *,
    context_dim: int = 34,
    test_frac: float = 0.2,
    seed: int = 0,
) -> dict:
    """Train logistic regression and 2-layer MLP; evaluate return regret.

    Split is grouped by episode_seed to prevent data leakage.
    """
    import warnings
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.neural_network import MLPClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import balanced_accuracy_score
    except ImportError:
        print("scikit-learn not available; installing...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "scikit-learn", "-q"], check=True)
        from sklearn.linear_model import LogisticRegression
        from sklearn.neural_network import MLPClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import balanced_accuracy_score

    ctx_keys = [f"ctx_{i}" for i in range(context_dim)]

    # Extract arrays
    X = np.array([[r[k] for k in ctx_keys] for r in dataset], dtype=np.float32)
    y = np.array([r["best_z"] for r in dataset], dtype=np.int64)
    oracle_returns = np.array([r["oracle_return"] for r in dataset])
    fixed_z2_returns = np.array([r["fixed_z2_return"] for r in dataset])
    z_returns_matrix = np.array([[r[f"return_z{z}"] for z in range(4)] for r in dataset])
    ep_seeds = np.array([int(r.get("episode_seed", i)) for i, r in enumerate(dataset)])

    # Grouped split by seed
    unique_seeds = np.unique(ep_seeds)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_seeds)
    n_test = max(1, int(len(unique_seeds) * test_frac))
    test_seeds = set(unique_seeds[:n_test].tolist())
    train_mask = np.array([s not in test_seeds for s in ep_seeds])
    test_mask = ~train_mask

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    oracle_test = oracle_returns[test_mask]
    z2_test = fixed_z2_returns[test_mask]
    z_returns_test = z_returns_matrix[test_mask]

    print(f"\n  Dataset: {len(dataset)} episodes  train={train_mask.sum()}  test={test_mask.sum()}")
    print(f"  Best-z class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    print(f"  Fixed-z2 mean return: {z2_test.mean():.4f}  oracle: {oracle_test.mean():.4f}")
    print(f"  Oracle gap: {(oracle_test - z2_test).mean():.4f}")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    results = {}

    # --- Baseline: always predict z2 ---
    z2_pred_return = z2_test
    z2_pred_regret = oracle_test - z2_pred_return
    results["always_z2"] = {
        "accuracy": float(np.mean(y_test == 2)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, np.full_like(y_test, 2))),
        "mean_predicted_return": float(z2_pred_return.mean()),
        "mean_regret": float(z2_pred_regret.mean()),
        "beats_fixed_z2_frac": float(np.mean(z2_pred_return > z2_test)),
    }

    # --- Baseline: random uniform ---
    rng2 = np.random.default_rng(seed + 1)
    rand_preds = rng2.integers(0, 4, size=len(y_test))
    rand_returns = z_returns_test[np.arange(len(y_test)), rand_preds]
    results["random_uniform"] = {
        "accuracy": float(np.mean(y_test == rand_preds)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, rand_preds)),
        "mean_predicted_return": float(rand_returns.mean()),
        "mean_regret": float((oracle_test - rand_returns).mean()),
        "beats_fixed_z2_frac": float(np.mean(rand_returns > z2_test)),
    }

    # --- Probe A: Logistic Regression ---
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lr = LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
        lr.fit(X_train_s, y_train)
    lr_preds = lr.predict(X_test_s)
    lr_returns = z_returns_test[np.arange(len(y_test)), lr_preds]
    lr_regret = oracle_test - lr_returns
    lr_top2 = float(np.mean(
        np.any(np.argsort(lr.predict_proba(X_test_s))[:, -2:] == y_test[:, None], axis=1)
    ))
    results["probe_a_logreg"] = {
        "accuracy": float(np.mean(y_test == lr_preds)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, lr_preds)),
        "top2_accuracy": lr_top2,
        "mean_predicted_return": float(lr_returns.mean()),
        "mean_regret": float(lr_regret.mean()),
        "beats_fixed_z2_frac": float(np.mean(lr_returns > z2_test)),
        "delta_return_vs_fixed_z2": float(lr_returns.mean() - z2_pred_return.mean()),
    }

    # --- Probe A: Small MLP ---
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mlp = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=500,
            random_state=seed,
            early_stopping=True,
            validation_fraction=0.15,
        )
        mlp.fit(X_train_s, y_train)
    mlp_preds = mlp.predict(X_test_s)
    mlp_returns = z_returns_test[np.arange(len(y_test)), mlp_preds]
    mlp_regret = oracle_test - mlp_returns
    mlp_top2 = float(np.mean(
        np.any(np.argsort(mlp.predict_proba(X_test_s))[:, -2:] == y_test[:, None], axis=1)
    ))
    results["probe_a_mlp"] = {
        "accuracy": float(np.mean(y_test == mlp_preds)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, mlp_preds)),
        "top2_accuracy": mlp_top2,
        "mean_predicted_return": float(mlp_returns.mean()),
        "mean_regret": float(mlp_regret.mean()),
        "beats_fixed_z2_frac": float(np.mean(mlp_returns > z2_test)),
        "delta_return_vs_fixed_z2": float(mlp_returns.mean() - z2_pred_return.mean()),
    }

    # --- Per-opponent/map breakdown ---
    per_cell: list[dict] = []
    for r_idx, row in enumerate(dataset):
        if not test_mask[r_idx]:
            continue
        per_cell.append({
            "opponent": row["opponent"],
            "map": row["map"],
            "best_z": int(y_test[r_idx - train_mask[:r_idx+1].sum()]),
        })

    return {
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "oracle_mean_return": float(oracle_test.mean()),
        "fixed_z2_mean_return": float(z2_test.mean()),
        "oracle_gap": float((oracle_test - z2_test).mean()),
        "probes": results,
    }


# ---------------------------------------------------------------------------
# Pretty-print
# ---------------------------------------------------------------------------

def print_results(results: dict) -> None:
    print(f"\n{'='*70}")
    print("Probe A — context predictability vs fixed z2")
    print(f"{'='*70}")
    print(f"  Oracle mean return:    {results['oracle_mean_return']:.4f}")
    print(f"  Fixed-z2 mean return:  {results['fixed_z2_mean_return']:.4f}")
    print(f"  Oracle gap:            {results['oracle_gap']:.4f}  "
          f"(= headroom above fixed-z2)")

    print(f"\n{'Probe':<25} {'Acc':>6} {'BalAcc':>7} {'Top2':>6} "
          f"{'RetMean':>9} {'Regret':>8} {'d_vs_z2':>9} {'Verdict'}")
    print("-" * 80)

    probes = results["probes"]
    z2_ret = results["fixed_z2_mean_return"]

    for name, p in probes.items():
        acc = p.get("accuracy", float("nan"))
        bal = p.get("balanced_accuracy", float("nan"))
        top2 = p.get("top2_accuracy", float("nan"))
        ret = p.get("mean_predicted_return", float("nan"))
        reg = p.get("mean_regret", float("nan"))
        delta = ret - z2_ret
        verdict = ""
        if "probe_a" in name:
            if delta > 0.01:
                verdict = "BEATS z2"
            elif delta > -0.01:
                verdict = "~= z2"
            else:
                verdict = "LOSES to z2"
        print(f"  {name:<23} {acc:6.3f} {bal:7.3f} {top2:6.3f} "
              f"{ret:9.4f} {reg:8.4f} {delta:+9.4f}  {verdict}")

    print(f"\n  Gate: probe mean return > fixed-z2 mean return ({z2_ret:.4f})")
    lr_delta = probes.get("probe_a_logreg", {}).get("delta_return_vs_fixed_z2", float("nan"))
    mlp_delta = probes.get("probe_a_mlp", {}).get("delta_return_vs_fixed_z2", float("nan"))
    if lr_delta > 0.01 or mlp_delta > 0.01:
        print("  RESULT: Context CONTAINS prospective signal (probe beats fixed z2)")
        print("  -> The online router objective or credit is failing, not context sufficiency.")
    elif lr_delta > -0.05 and mlp_delta > -0.05:
        print("  RESULT: Context is MARGINALLY insufficient (probe ~= fixed z2)")
        print("  -> Consider Probe B: add opponent/map identity to context.")
    else:
        print("  RESULT: Context is INSUFFICIENT (probe loses to fixed z2)")
        print("  -> Run Probe B: add opponent/map identity to context.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Probe A: context predictability for router z-selection")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--forced-z-csv", required=True, help="Path to episode_results.csv from forced-z run")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--base-seed", type=int, default=42, help="Base seed used in the forced-z run")
    p.add_argument("--test-frac", type=float, default=0.2)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    ckpt_path = str(Path(args.checkpoint).resolve())
    fz_csv = Path(args.forced_z_csv)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out_dir or f"experiments/probe_a_runs/{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Probe A  ckpt={Path(ckpt_path).name}")
    print(f"  forced-z csv={fz_csv.name}  ({fz_csv.stat().st_size // 1024} KB)")

    # ── Identify cell structure from CSV ──────────────────────────────────
    import csv as csv_mod
    with fz_csv.open(encoding="utf-8") as f:
        sample_rows = list(csv_mod.DictReader(f))

    # Extract cell_seeds directly from CSV to avoid opp_order/map_order assumptions
    cell_seeds: dict[tuple[str, str], int] = {}
    episode_indices_set: set[int] = set()
    for r in sample_rows:
        opp = r["opponent"].upper()
        map_name = r["map"]
        ep_idx = int(r["episode_index"])
        episode_indices_set.add(ep_idx)
        key = (opp, map_name)
        if key not in cell_seeds and "cell_seed" in r:
            cell_seeds[key] = int(r["cell_seed"])

    if not cell_seeds:
        # Fallback: compute from protocol defaults if cell_seed column absent
        print("  Warning: cell_seed not in CSV, computing from protocol defaults...")
        from experiments.forced_z_eval.protocol import ForcedZProtocol
        proto = ForcedZProtocol(checkpoint=ckpt_path, base_seed=args.base_seed)
        for opp_idx, opp in enumerate(proto.opponents):
            for map_idx, m in enumerate(proto.maps):
                cell_seeds[(opp, m)] = proto.cell_seed(opp_idx, map_idx)

    n_episodes = max(episode_indices_set) + 1
    opponents_in_csv = sorted({k[0] for k in cell_seeds})
    maps_in_csv = sorted({k[1] for k in cell_seeds})

    print(f"  Opponents: {opponents_in_csv}  Maps: {maps_in_csv}  Episodes: {n_episodes}")
    print(f"  Cell seeds: { {f'{k[0]}/{k[1]}': v for k, v in cell_seeds.items()} }")

    # ── Collect initial contexts ───────────────────────────────────────────
    print("\nCollecting initial env.state() contexts...")
    contexts = collect_initial_contexts(
        checkpoint_path=ckpt_path,
        cell_seeds=cell_seeds,
        n_episodes=n_episodes,
        device=args.device,
    )
    print(f"  Collected {len(contexts)} context vectors (34-dim each)")

    # ── Build dataset ──────────────────────────────────────────────────────
    print("\nBuilding Probe A dataset...")
    dataset = build_dataset(fz_csv, contexts)
    print(f"  Dataset: {len(dataset)} matched episodes")

    # Write dataset CSV
    if dataset:
        ctx_keys = [f"ctx_{i}" for i in range(34)]
        fields = (
            ["opponent", "map", "episode_index", "best_z", "oracle_return",
             "fixed_z2_return", "fixed_z2_regret",
             "return_z0", "return_z1", "return_z2", "return_z3",
             "cell_seed", "episode_seed"]
            + ctx_keys
        )
        dataset_csv = out_dir / "probe_a_dataset.csv"
        with dataset_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv_mod.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(dataset)
        print(f"  Dataset written: {dataset_csv}")

    # ── Train probes ───────────────────────────────────────────────────────
    print("\nTraining probes...")
    probe_results = run_probes(dataset, context_dim=34, test_frac=args.test_frac)

    # ── Print and write results ────────────────────────────────────────────
    print_results(probe_results)

    result_path = out_dir / "probe_a_results.json"
    result_path.write_text(json.dumps(probe_results, indent=2), encoding="utf-8")

    # Summary text
    summary_lines = [
        "Probe A — context predictability",
        f"checkpoint: {Path(ckpt_path).name}",
        f"forced-z csv: {fz_csv}",
        f"n_train={probe_results['n_train']}  n_test={probe_results['n_test']}",
        f"oracle_mean_return={probe_results['oracle_mean_return']:.4f}",
        f"fixed_z2_mean_return={probe_results['fixed_z2_mean_return']:.4f}",
        f"oracle_gap={probe_results['oracle_gap']:.4f}",
        "",
    ]
    for name, p in probe_results["probes"].items():
        summary_lines.append(
            f"{name}: acc={p.get('accuracy', float('nan')):.3f}  "
            f"bal_acc={p.get('balanced_accuracy', float('nan')):.3f}  "
            f"ret={p.get('mean_predicted_return', float('nan')):.4f}  "
            f"d_vs_z2={p.get('delta_return_vs_fixed_z2', float('nan')):+.4f}"
        )
    (out_dir / "probe_a_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"\nOutputs: {out_dir}/")
    print(f"  probe_a_dataset.csv  probe_a_results.json  probe_a_summary.txt")


if __name__ == "__main__":
    main()
