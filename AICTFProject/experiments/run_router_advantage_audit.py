#!/usr/bin/env python3
"""Router advantage alignment audit.

Answers: does the online router advantage signal correlate with the oracle
z-selection signal derived from matched forced-z returns?

For each episode (at the initial routing decision):
  oracle_selected_advantage = return(selected_z) - mean(return_z0..z3)
  online_advantage_proxy    = return(selected_z) - V(s_0, selected_z)

Key measurements:
  1. corr(online_proxy, oracle_selected_advantage)
  2. sign_agreement: fraction where sign(online) == sign(oracle)
  3. mean online_proxy by oracle signal quartile
  4. Critic calibration: corr(V(s_0, z), return_z) pooled across z
  5. critic_best_z_agree: fraction where argmax V(s_0, z) == oracle best_z

If corr is HIGH: online signal is directionally correct; optimizer or
  credit-aggregation is the problem.
If corr is LOW and critic_best_z_agree is HIGH: critic is "explaining away"
  z-specific returns -> advantages collapse toward zero for all z.
If corr is LOW and critic_best_z_agree is LOW: critic is miscalibrated but
  still kills the signal (high variance baseline).

Usage
-----
    uv run python experiments/run_router_advantage_audit.py \\
        --checkpoint checkpoints/2v2/final_v6i9-mapaware-repertoire-hardpool-refactor-r1-seed1_2v2.zip \\
        --dataset experiments/probe_a_runs/20260702T123105Z/probe_a_dataset.csv \\
        [--out-dir experiments/router_audit_runs/<stamp>]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn.functional as F

CONTEXT_DIM_ENV = 34    # dims from env.state() / probe_a_dataset ctx_0..33
LATENT_K = 4


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_probe_dataset(csv_path: Path) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Load probe_a_dataset.csv.

    Returns
    -------
    X : (N, 34) float32 -- initial env states
    Z_returns : (N, 4) float32 -- return_z0..z3 per episode
    rows : list of dicts (raw CSV rows with metadata)
    """
    rows = []
    with csv_path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)

    ctx_keys = [f"ctx_{i}" for i in range(CONTEXT_DIM_ENV)]
    X = np.array([[float(r[k]) for k in ctx_keys] for r in rows], dtype=np.float32)
    Z = np.array([[float(r[f"return_z{z}"]) for z in range(LATENT_K)] for r in rows], dtype=np.float32)
    return X, Z, rows


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_critic(ckpt_path: str, *, device: str = "cpu") -> tuple["torch.nn.Module", int, int]:
    """Load checkpoint, return (critic, global_state_dim, latent_k)."""
    from game_field_gpu import GPUCTFVecEnv, GPUFieldConfig
    from rl.custom_ppo import load_custom_ppo_policy
    from rl.custom_ppo.inference import read_custom_ppo_metadata

    meta = read_custom_ppo_metadata(ckpt_path)
    n_agents = int(meta.get("n_blue", 2))

    env = GPUCTFVecEnv(GPUFieldConfig(
        n_envs=1,
        max_blue_agents=n_agents,
        max_red_agents=n_agents,
        device=device,
        seed=42,
        map_layout="map_b",
        max_decision_steps=400,
        aquaticus_profile=True,
        rules_profile="OURS",
    ))
    try:
        policy = load_custom_ppo_policy(ckpt_path, env.observation_space, env.action_space, device=device)
    finally:
        env.close()

    model = policy.model
    critic = model.critic
    gs_dim = int(critic.global_state_dim)
    lk = int(model.latent_k) if model.uses_latent_strategy else 0
    extra_dim = int(critic.extra_dim)

    print(f"  Critic: global_state_dim={gs_dim}  extra_dim={extra_dim}  latent_k={lk}")
    print(f"  Critic input: {gs_dim} + {extra_dim} = {gs_dim + extra_dim} dims")

    if extra_dim != lk:
        raise RuntimeError(
            f"critic extra_dim={extra_dim} but latent_k={lk}; expected them to match"
        )

    return critic, gs_dim, lk


# ---------------------------------------------------------------------------
# Per-episode critic evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_critic_values(
    critic: "torch.nn.Module",
    X: np.ndarray,
    *,
    gs_dim: int,
    latent_k: int,
    device: str = "cpu",
) -> np.ndarray:
    """Return V(s_0, z) for all z.  Shape: (N, latent_k) float32.

    At episode start, strategy_age_norm = 0 (augmented dim is zero).
    If gs_dim > CONTEXT_DIM_ENV we zero-pad; if gs_dim == CONTEXT_DIM_ENV we use as-is.
    """
    dev = torch.device(device)
    critic.to(dev).eval()

    N = X.shape[0]
    pad = gs_dim - CONTEXT_DIM_ENV
    if pad > 0:
        X_aug = np.pad(X, ((0, 0), (0, pad)), mode="constant")
    elif pad < 0:
        X_aug = X[:, :gs_dim]
    else:
        X_aug = X

    states = torch.tensor(X_aug, dtype=torch.float32, device=dev)  # (N, gs_dim)

    V = np.zeros((N, latent_k), dtype=np.float32)
    batch = 256
    for start in range(0, N, batch):
        end = min(start + batch, N)
        s = states[start:end]
        B = end - start
        for z in range(latent_k):
            z_idx = torch.full((B,), z, dtype=torch.long, device=dev)
            z_oh = F.one_hot(z_idx, num_classes=latent_k).float()
            V[start:end, z] = critic(s, z_oh).squeeze(-1).cpu().numpy()

    return V


# ---------------------------------------------------------------------------
# Correlation helpers
# ---------------------------------------------------------------------------

def pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) < 1e-10 or np.std(b) < 1e-10:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------

def run_audit(
    X: np.ndarray,
    Z_returns: np.ndarray,
    V_critic: np.ndarray,
    *,
    selected_z: int = 1,
) -> dict:
    """Compute all alignment metrics.

    Parameters
    ----------
    X            : (N, 34)  -- episode-start env states (unused here, reserved)
    Z_returns    : (N, 4)   -- forced-z episode returns
    V_critic     : (N, 4)   -- V(s_0, z) from critic for all z
    selected_z   : the z the router always selects (empirically 1)
    """
    N = len(Z_returns)

    mean_z_return = Z_returns.mean(axis=1)              # (N,)
    return_selected = Z_returns[:, selected_z]           # (N,)

    oracle_adv = return_selected - mean_z_return         # (N,) oracle signal
    online_proxy = return_selected - V_critic[:, selected_z]  # (N,) online signal

    # Correlation
    r_main = pearson_r(online_proxy, oracle_adv)
    sign_agree = float(np.mean(np.sign(online_proxy) == np.sign(oracle_adv)))

    # Mean online proxy by oracle quartile
    quartile_cuts = np.percentile(oracle_adv, [25, 50, 75])
    q_labels = np.digitize(oracle_adv, quartile_cuts)  # 0=Q1(worst)..3=Q4(best)
    quartile_means = {
        f"Q{i+1}": float(online_proxy[q_labels == i].mean()) if np.any(q_labels == i) else float("nan")
        for i in range(4)
    }

    # Critic calibration: corr(V(s_0, z), return_z) pooled across all (episode, z) pairs
    V_flat = V_critic.flatten()      # (N*4,)
    R_flat = Z_returns.flatten()     # (N*4,)
    r_critic_cal = pearson_r(V_flat, R_flat)

    # Per-episode: does argmax V match argmax return?
    critic_best_z = np.argmax(V_critic, axis=1)
    oracle_best_z = np.argmax(Z_returns, axis=1)
    critic_best_agree = float(np.mean(critic_best_z == oracle_best_z))

    # Critic z-spread vs return spread per episode
    V_spread = V_critic.max(axis=1) - V_critic.min(axis=1)   # (N,) max-min V across z
    R_spread = Z_returns.max(axis=1) - Z_returns.min(axis=1)  # (N,) max-min return across z
    mean_V_spread = float(V_spread.mean())
    mean_R_spread = float(R_spread.mean())

    # Critic V(best_z) - V(selected_z) — how much is critic "penalizing" selected_z?
    V_oracle = V_critic[np.arange(N), oracle_best_z]
    V_selected = V_critic[:, selected_z]
    mean_V_oracle_vs_selected = float((V_oracle - V_selected).mean())

    # Amplitude
    oracle_adv_mean = float(oracle_adv.mean())
    oracle_adv_std = float(oracle_adv.std())
    online_proxy_mean = float(online_proxy.mean())
    online_proxy_std = float(online_proxy.std())

    # Fraction of episodes where online signal has WRONG sign vs oracle
    wrong_sign = float(np.mean(np.sign(online_proxy) != np.sign(oracle_adv)))

    return {
        "N": N,
        "selected_z": selected_z,
        "correlation_online_vs_oracle": r_main,
        "sign_agreement": sign_agree,
        "wrong_sign_fraction": wrong_sign,
        "quartile_mean_online": quartile_means,
        "critic_calibration_corr": r_critic_cal,
        "critic_best_z_agree": critic_best_agree,
        "mean_V_spread_across_z": mean_V_spread,
        "mean_return_spread_across_z": mean_R_spread,
        "mean_V_oracle_vs_selected": mean_V_oracle_vs_selected,
        "oracle_selected_advantage_mean": oracle_adv_mean,
        "oracle_selected_advantage_std": oracle_adv_std,
        "online_proxy_mean": online_proxy_mean,
        "online_proxy_std": online_proxy_std,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_audit(result: dict) -> None:
    N = result["N"]
    sz = result["selected_z"]
    print(f"\n{'='*68}")
    print(f"Router Advantage Alignment Audit")
    print(f"  N={N} episodes  selected_z={sz} (logit-collapse confirmed)")
    print(f"  oracle_selected_advantage = return_z{sz} - mean(return_z0..z3)")
    print(f"  online_advantage_proxy    = return_z{sz} - V(s_0, z={sz})")
    print(f"{'='*68}")

    print(f"\n--- Online vs Oracle Alignment ---")
    r = result["correlation_online_vs_oracle"]
    sa = result["sign_agreement"]
    ws = result["wrong_sign_fraction"]
    print(f"  corr(online_proxy, oracle_adv):   {r:+.4f}")
    print(f"  sign_agreement:                   {sa:.3f}")
    print(f"  wrong_sign_fraction:              {ws:.3f}")

    print(f"\n  Mean online_proxy by oracle quartile:")
    qm = result["quartile_mean_online"]
    for q in ["Q1", "Q2", "Q3", "Q4"]:
        label = "(worst 25%)" if q == "Q1" else ("(best 25%)" if q == "Q4" else "")
        v = qm[q]
        print(f"    {q} {label:12s}: {v:+.4f}")

    print(f"\n--- Critic Calibration ---")
    cc = result["critic_calibration_corr"]
    ba = result["critic_best_z_agree"]
    vs = result["mean_V_spread_across_z"]
    rs = result["mean_return_spread_across_z"]
    vo = result["mean_V_oracle_vs_selected"]
    print(f"  corr(V(s_0,z), return_z) pooled:  {cc:+.4f}")
    print(f"  critic best_z agrees oracle:       {ba:.3f}  ({ba*100:.1f}%)")
    print(f"  mean V spread across z:            {vs:.4f}")
    print(f"  mean return spread across z:       {rs:.4f}  (how well critic tracks)")
    print(f"  mean V(oracle_z) - V(selected_z):  {vo:+.4f}")

    print(f"\n--- Advantage Amplitudes ---")
    print(f"  oracle_selected_advantage: mean={result['oracle_selected_advantage_mean']:+.4f}  "
          f"std={result['oracle_selected_advantage_std']:.4f}")
    print(f"  online_proxy:              mean={result['online_proxy_mean']:+.4f}  "
          f"std={result['online_proxy_std']:.4f}")

    # Decision tree
    r_val = result["correlation_online_vs_oracle"]
    ba_val = result["critic_best_z_agree"]
    vs_val = result["mean_V_spread_across_z"]
    rs_val = result["mean_return_spread_across_z"]

    print(f"\n--- Decision Tree Verdict ---")

    if r_val >= 0.30:
        print(f"  [HIGH CORR r={r_val:.3f}] Online signal direction is roughly correct.")
        print(f"  -> Problem is NOT credit alignment. Check: gradient magnitude,")
        print(f"     normalization, or optimizer step size.")
    elif ba_val >= 0.40 and vs_val / max(rs_val, 1e-6) >= 0.20:
        print(f"  [LOW CORR r={r_val:.3f}, critic_best_agree={ba_val:.2f}]")
        print(f"  Critic has learned z-specific values AND they correlate with returns.")
        print(f"  -> Critic is EXPLAINING AWAY z-signal: advantages collapse near zero.")
        print(f"     Fix: use z-agnostic baseline V(s_0) [extra_dim=0], or switch to")
        print(f"     arc credit (latent_arc_credit_enabled=True) which bypasses critic.")
    elif r_val < 0.10:
        print(f"  [LOW CORR r={r_val:.3f}, critic_best_agree={ba_val:.2f}]")
        print(f"  Critic is NOT calibrated to z-specific returns. Online signal is")
        print(f"  effectively random noise -> router gets no useful gradient.")
        print(f"  -> Enable arc credit (bypasses critic) or fix credit attribution.")
    else:
        print(f"  [MODERATE CORR r={r_val:.3f}] Mixed signal. Online proxy weakly")
        print(f"  correlated with oracle. Check critic spread vs return spread.")
        print(f"  V_spread/R_spread ratio: {vs_val/max(rs_val,1e-6):.3f}")

    print(f"{'='*68}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True,
                        help="Path to probe_a_dataset.csv from run_probe_a.py")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--selected-z", type=int, default=1,
                        help="z the router collapses to (empirically 1)")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out_dir) if args.out_dir else Path(f"experiments/router_audit_runs/{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = Path(args.dataset)
    print(f"Router Advantage Alignment Audit  ckpt={Path(args.checkpoint).name}")
    print(f"  dataset={args.dataset}")

    # Load dataset
    print("\nLoading probe_a_dataset...")
    X, Z_returns, rows = load_probe_dataset(dataset_path)
    print(f"  N={len(rows)} episodes  context_dim={X.shape[1]}  latent_k={Z_returns.shape[1]}")

    # Load critic
    print("\nLoading critic from checkpoint...")
    critic, gs_dim, latent_k = load_critic(args.checkpoint, device=args.device)

    # Compute V(s_0, z) for all z
    print("\nComputing V(s_0, z) for all z across all episodes...")
    V_critic = compute_critic_values(
        critic, X, gs_dim=gs_dim, latent_k=latent_k, device=args.device
    )
    print(f"  V_critic shape: {V_critic.shape}  "
          f"mean={V_critic.mean():.4f}  std={V_critic.std():.4f}")
    print(f"  Per-z mean V: " + "  ".join(f"z{z}={V_critic[:,z].mean():.4f}" for z in range(latent_k)))

    # Run audit
    print("\nRunning audit...")
    result = run_audit(X, Z_returns, V_critic, selected_z=args.selected_z)

    print_audit(result)

    # Save
    out_json = out_dir / "router_audit_results.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"  Results -> {out_dir / 'router_audit_results.json'}")

    # Also save per-episode CSV for further analysis
    out_csv = out_dir / "router_audit_episodes.csv"
    mean_z_return = Z_returns.mean(axis=1)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        fnames = (
            ["opponent", "map", "episode_index", "episode_seed", "cell_seed"]
            + [f"return_z{z}" for z in range(latent_k)]
            + [f"V_z{z}" for z in range(latent_k)]
            + ["mean_z_return", "oracle_selected_adv", "online_proxy",
               "oracle_best_z", "critic_best_z"]
        )
        writer = csv.DictWriter(f, fieldnames=fnames)
        writer.writeheader()
        sz = args.selected_z
        for i, row in enumerate(rows):
            oracle_adv = Z_returns[i, sz] - mean_z_return[i]
            online_pr = Z_returns[i, sz] - V_critic[i, sz]
            writer.writerow({
                "opponent": row.get("opponent", ""),
                "map": row.get("map", ""),
                "episode_index": row.get("episode_index", i),
                "episode_seed": row.get("episode_seed", ""),
                "cell_seed": row.get("cell_seed", ""),
                **{f"return_z{z}": Z_returns[i, z] for z in range(latent_k)},
                **{f"V_z{z}": V_critic[i, z] for z in range(latent_k)},
                "mean_z_return": mean_z_return[i],
                "oracle_selected_adv": oracle_adv,
                "online_proxy": online_pr,
                "oracle_best_z": int(np.argmax(Z_returns[i])),
                "critic_best_z": int(np.argmax(V_critic[i])),
            })
    print(f"  Per-episode -> {out_csv}")


if __name__ == "__main__":
    main()
