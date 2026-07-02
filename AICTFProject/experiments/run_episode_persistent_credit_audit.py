#!/usr/bin/env python3
"""Episode-persistent aligned credit audit.

Compares the router's online advantage to the oracle z-selection signal,
with both covering the same decision period (the full episode).

Previous audit (run_router_advantage_audit.py) was not apples-to-apples:
  online advantage   = value of z over one 32-step arc + future switches
  oracle advantage   = value of holding z for the entire episode

This script fixes the mismatch by using episode-persistent z throughout:
  online_advantage   = episode_return(z_router) - V(s_0, z_router)
  oracle_advantage   = forced_return(z_router) - mean_z(forced_return)

Both cover the full episode.  Since we use deterministic eval under the
same seed and the same z, episode_return == forced_return[z_router].  The
shared-R component means the correlation measures how well V(s_0, z_router)
approximates mean_z(forced_return) -- which is exactly the right diagnostic.

Usage
-----
    uv run python experiments/run_episode_persistent_credit_audit.py \\
        --checkpoint checkpoints/2v2/final_v6i9-mapaware-repertoire-hardpool-refactor-r1-seed1_2v2.zip \\
        --dataset experiments/probe_a_runs/20260702T123105Z/probe_a_dataset.csv \\
        [--out-dir experiments/credit_audit_runs/<stamp>]

Additional critic checks reported:
  - critic input: V(s, z) vs V(s)  [V(s,z) may remove between-z contrast]
  - critic error vs between-z return spread
  - V(s, z_oracle_best) - V(s, z_selected): does critic assign higher value to better z?
  - normalization: mean / std of raw advantages per batch
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

try:
    from scipy.stats import spearmanr as _spearmanr

    def spearman_r(a: np.ndarray, b: np.ndarray) -> float:
        if np.std(a) < 1e-10 or np.std(b) < 1e-10:
            return float("nan")
        return float(_spearmanr(a, b).statistic)
except ImportError:
    def spearman_r(a: np.ndarray, b: np.ndarray) -> float:  # type: ignore[misc]
        return float("nan")

CONTEXT_DIM_ENV = 34   # dims from env.state() stored as ctx_0..33
GLOBAL_STATE_DIM = 34  # raw env global state
LATENT_K = 4


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset(csv_path: Path) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Returns X(N,34), Z_returns(N,4), rows."""
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

def load_policy_and_model(ckpt_path: str, *, device: str = "cpu") -> tuple:
    """Return (inference_policy, model, critic, gs_dim, latent_k)."""
    from game_field_gpu import GPUCTFVecEnv, GPUFieldConfig
    from rl.custom_ppo import load_custom_ppo_policy
    from rl.custom_ppo.inference import read_custom_ppo_metadata

    meta = read_custom_ppo_metadata(ckpt_path)
    n_agents = int(meta.get("n_blue", 2))

    env = GPUCTFVecEnv(GPUFieldConfig(
        n_envs=1, max_blue_agents=n_agents, max_red_agents=n_agents,
        device=device, seed=42, map_layout="map_b", max_decision_steps=400,
        aquaticus_profile=True, rules_profile="OURS",
    ))
    try:
        policy = load_custom_ppo_policy(
            ckpt_path, env.observation_space, env.action_space, device=device
        )
    finally:
        env.close()

    model = policy.model
    critic = model.critic
    gs_dim = int(critic.global_state_dim)
    lk = int(model.latent_k) if model.uses_latent_strategy else 0

    print(f"  Model: uses_latent_strategy={model.uses_latent_strategy}  latent_k={lk}")
    print(f"  Router: recurrent_selector_hidden_dim={getattr(model, 'recurrent_selector_hidden_dim', 0)}")
    print(f"  Critic: global_state_dim={gs_dim}  extra_dim={critic.extra_dim}")
    print(f"  Critic type: V(s, {'z)' if critic.extra_dim > 0 else ')'}")
    if critic.extra_dim > 0:
        print(f"  WARNING: critic is V(s,z) not V(s) -- z-conditioned baseline may")
        print(f"           explain away z-specific returns, reducing advantage contrast.")

    return policy, model, critic, gs_dim, lk


# ---------------------------------------------------------------------------
# Batched router forward pass
# ---------------------------------------------------------------------------

@torch.no_grad()
def batch_router_forward(
    model: "torch.nn.Module",
    X: np.ndarray,
    *,
    gs_dim: int,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Run strategy_encoder on all N episodes in one batch.

    Augments 34-dim env state to gs_dim by appending zeros (strategy_age_norm=0
    at episode start).  GRU hidden starts at zeros (fresh episode).

    Returns
    -------
    z_selected : (N,) int64 -- argmax of logits
    log_probs  : (N,) float32 -- log prob of selected z
    """
    dev = torch.device(device)
    model.to(dev).eval()

    N = X.shape[0]
    pad = gs_dim - GLOBAL_STATE_DIM
    if pad > 0:
        X_aug = np.pad(X, ((0, 0), (0, pad)), mode="constant")
    elif pad < 0:
        X_aug = X[:, :gs_dim]
    else:
        X_aug = X
    states = torch.tensor(X_aug, dtype=torch.float32, device=dev)  # (N, gs_dim)

    # GRU hidden starts at zeros for each episode
    recurrent_dim = int(getattr(model, "recurrent_selector_hidden_dim", 0) or 0)
    if recurrent_dim > 0 and model.selector_gru is not None:
        hidden = torch.zeros((N, recurrent_dim), dtype=torch.float32, device=dev)
        # GRU takes 34-dim raw state; encoder takes [35-dim, h_new]
        gru_input = states[:, :GLOBAL_STATE_DIM].float()
        h_new = model.selector_gru(gru_input, hidden)
        encoder_in = torch.cat([states.float(), h_new], dim=-1)
        logits = model.strategy_encoder(encoder_in)
    else:
        logits = model.strategy_encoder(states)

    # Apply tau temperature if present
    tau = float(getattr(model, "strategy_tau", 1.0) or 1.0)
    if tau != 1.0:
        logits = logits / tau

    z_selected = logits.argmax(dim=-1).cpu().numpy().astype(np.int64)
    log_probs = F.log_softmax(logits, dim=-1)
    lp = log_probs[torch.arange(N, device=dev), torch.from_numpy(z_selected).to(dev)]
    return z_selected, lp.cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Batched critic forward
# ---------------------------------------------------------------------------

@torch.no_grad()
def batch_critic_all_z(
    critic: "torch.nn.Module",
    X: np.ndarray,
    *,
    gs_dim: int,
    latent_k: int,
    device: str = "cpu",
) -> np.ndarray:
    """V(s_0, z) for all z.  Shape (N, latent_k)."""
    dev = torch.device(device)
    critic.to(dev).eval()

    N = X.shape[0]
    pad = gs_dim - GLOBAL_STATE_DIM
    if pad > 0:
        X_aug = np.pad(X, ((0, 0), (0, pad)), mode="constant")
    else:
        X_aug = X[:, :gs_dim]

    states = torch.tensor(X_aug, dtype=torch.float32, device=dev)
    V = np.zeros((N, latent_k), dtype=np.float32)

    batch = 512
    for start in range(0, N, batch):
        end = min(start + batch, N)
        s = states[start:end]
        B = end - start
        for z in range(latent_k):
            z_oh = F.one_hot(torch.full((B,), z, dtype=torch.long, device=dev), latent_k).float()
            V[start:end, z] = critic(s, z_oh).squeeze(-1).cpu().numpy()

    return V


# ---------------------------------------------------------------------------
# Pearson helper
# ---------------------------------------------------------------------------

def pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) < 1e-10 or np.std(b) < 1e-10:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


# ---------------------------------------------------------------------------
# Core audit
# ---------------------------------------------------------------------------

def run_audit(
    Z_returns: np.ndarray,
    z_selected: np.ndarray,
    log_probs: np.ndarray,
    V_critic: np.ndarray,
) -> dict:
    """Compute all aligned credit metrics.

    Parameters
    ----------
    Z_returns   : (N, 4) forced-z episode returns
    z_selected  : (N,)   router's argmax z per episode
    log_probs   : (N,)   log P(z_selected | s_0)
    V_critic    : (N, 4) V(s_0, z) for all z

    online_advantage = forced_return[z_selected] - V(s_0, z_selected)
    oracle_advantage = forced_return[z_selected] - mean_z(forced_return)
    """
    N = len(z_selected)
    idx = np.arange(N)

    ep_return = Z_returns[idx, z_selected]        # return for router-selected z
    V_selected = V_critic[idx, z_selected]         # critic baseline for selected z
    mean_z_return = Z_returns.mean(axis=1)         # per-episode mean across z

    online_adv = ep_return - V_selected            # (N,) online credit signal
    oracle_adv = ep_return - mean_z_return         # (N,) oracle signal

    # Alignment
    r_pearson = pearson_r(online_adv, oracle_adv)
    r_spearman = spearman_r(online_adv, oracle_adv)
    sign_agree = float(np.mean(np.sign(online_adv) == np.sign(oracle_adv)))

    # Mean online by oracle quartile
    cuts = np.percentile(oracle_adv, [25, 50, 75])
    q_labels = np.digitize(oracle_adv, cuts)  # 0=Q1(worst)..3=Q4(best)
    quartile_means_online = {
        f"Q{i+1}": float(online_adv[q_labels == i].mean()) if np.any(q_labels == i) else float("nan")
        for i in range(4)
    }
    quartile_means_oracle = {
        f"Q{i+1}": float(oracle_adv[q_labels == i].mean()) if np.any(q_labels == i) else float("nan")
        for i in range(4)
    }

    # z distribution
    z_counts = {int(z): int(np.sum(z_selected == z)) for z in range(4)}

    # Critic calibration
    V_flat = V_critic.flatten()
    R_flat = Z_returns.flatten()
    critic_cal_r = pearson_r(V_flat, R_flat)

    V_oracle = V_critic[idx, np.argmax(Z_returns, axis=1)]
    critic_best_agree = float(np.mean(np.argmax(V_critic, axis=1) == np.argmax(Z_returns, axis=1)))

    V_spread = V_critic.max(axis=1) - V_critic.min(axis=1)
    R_spread = Z_returns.max(axis=1) - Z_returns.min(axis=1)

    # Critic error analysis
    # V(s_0, z_selected) vs mean_z(forced_return): the bias
    bias = V_selected - mean_z_return  # positive = critic overestimates relative to mean_z
    mean_bias = float(bias.mean())
    std_bias = float(bias.std())

    # Amplitude
    mean_R_spread = float(R_spread.mean())
    mean_V_spread = float(V_spread.mean())

    return {
        "N": N,
        "z_selection_counts": z_counts,
        "correlation_pearson": r_pearson,
        "correlation_spearman": r_spearman,
        "sign_agreement": sign_agree,
        "wrong_sign_fraction": float(1.0 - sign_agree),
        "quartile_means_online": quartile_means_online,
        "quartile_means_oracle": quartile_means_oracle,
        "online_advantage_mean": float(online_adv.mean()),
        "online_advantage_std": float(online_adv.std()),
        "oracle_advantage_mean": float(oracle_adv.mean()),
        "oracle_advantage_std": float(oracle_adv.std()),
        "critic_overestimate_bias_mean": mean_bias,
        "critic_overestimate_bias_std": std_bias,
        "critic_calibration_corr": critic_cal_r,
        "critic_best_z_agree": critic_best_agree,
        "mean_V_spread_across_z": mean_V_spread,
        "mean_return_spread_across_z": mean_R_spread,
        "V_spread_over_R_spread_ratio": mean_V_spread / max(mean_R_spread, 1e-6),
        "per_z_mean_V": {f"z{z}": float(V_critic[:, z].mean()) for z in range(4)},
        "per_z_mean_return": {f"z{z}": float(Z_returns[:, z].mean()) for z in range(4)},
        "mean_log_prob": float(log_probs.mean()),
        "std_log_prob": float(log_probs.std()),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_audit(result: dict) -> None:
    N = result["N"]
    zc = result["z_selection_counts"]
    print(f"\n{'='*72}")
    print(f"Episode-Persistent Aligned Credit Audit  (N={N})")
    print(f"  online_advantage = forced_return(z_router) - V(s_0, z_router)")
    print(f"  oracle_advantage = forced_return(z_router) - mean_z(forced_return)")
    print(f"  Both cover the full episode -- credit period matches oracle period.")
    print(f"{'='*72}")

    print(f"\n--- Router z Selection ---")
    for z, cnt in sorted(zc.items()):
        bar = "*" * (cnt // 6)
        print(f"  z={z}: {cnt:4d}/{N}  ({cnt/N*100:.1f}%)  {bar}")
    print(f"  log_prob: mean={result['mean_log_prob']:.4f}  std={result['std_log_prob']:.4f}")

    print(f"\n--- Aligned Online vs Oracle ---")
    rp = result["correlation_pearson"]
    rs = result["correlation_spearman"]
    sa = result["sign_agreement"]
    ws = result["wrong_sign_fraction"]
    print(f"  Pearson  corr(online, oracle): {rp:+.4f}")
    print(f"  Spearman corr(online, oracle): {rs:+.4f}")
    print(f"  sign_agreement:                {sa:.3f}")
    print(f"  wrong_sign_fraction:           {ws:.3f}")

    print(f"\n  Mean online_advantage by oracle quartile:")
    qo = result["quartile_means_online"]
    qq = result["quartile_means_oracle"]
    print(f"  {'Quartile':>10}  {'Online mean':>12}  {'Oracle mean':>12}")
    for q in ["Q1", "Q2", "Q3", "Q4"]:
        tag = " (worst)" if q == "Q1" else (" (best) " if q == "Q4" else "        ")
        print(f"  {q+tag:>18}  {qo[q]:>+12.4f}  {qq[q]:>+12.4f}")

    print(f"\n--- Advantage Amplitudes ---")
    print(f"  online: mean={result['online_advantage_mean']:+.4f}  std={result['online_advantage_std']:.4f}")
    print(f"  oracle: mean={result['oracle_advantage_mean']:+.4f}  std={result['oracle_advantage_std']:.4f}")

    print(f"\n--- Critic Baseline Bias ---")
    mb = result["critic_overestimate_bias_mean"]
    sb = result["critic_overestimate_bias_std"]
    print(f"  bias = V(s_0, z_router) - mean_z(forced_return)")
    print(f"  mean bias: {mb:+.4f}  (positive = critic too optimistic)")
    print(f"  std  bias: {sb:.4f}")
    print(f"  Per-z mean V:      " + "  ".join(f"z{z}={result['per_z_mean_V'][f'z{z}']:.3f}" for z in range(4)))
    print(f"  Per-z mean return: " + "  ".join(f"z{z}={result['per_z_mean_return'][f'z{z}']:.3f}" for z in range(4)))

    print(f"\n--- Critic Calibration ---")
    cc = result["critic_calibration_corr"]
    ba = result["critic_best_z_agree"]
    vr = result["V_spread_over_R_spread_ratio"]
    print(f"  corr(V(s_0,z), return_z) pooled: {cc:+.4f}")
    print(f"  critic best_z agrees oracle:       {ba:.3f}  ({ba*100:.1f}%)")
    print(f"  V_spread / R_spread:               {vr:.4f}  (0=critic flat, 1=matches return spread)")

    print(f"\n--- Decision Tree Verdict ---")
    rp_val = result["correlation_pearson"]
    sa_val = result["sign_agreement"]
    mb_val = result["critic_overestimate_bias_mean"]
    vr_val = result["V_spread_over_R_spread_ratio"]

    if rp_val >= 0.40 and sa_val >= 0.65:
        print(f"  GOOD ALIGNMENT: Online credit signal direction is correct.")
        print(f"  -> Focus on optimizer, learning rate, or gradient magnitude.")
    elif abs(mb_val) > 1.0 and sa_val < 0.65:
        print(f"  BASELINE BIAS: V(s_0, z_router) is off by {mb_val:+.2f} units on average.")
        print(f"  This creates a chronic offset in online_advantage, flipping signs {ws*100:.1f}% of the time.")
        print(f"  -> Use arc credit with running_mean baseline (bypasses critic),")
        print(f"     OR use z-agnostic critic (extra_dim=0) to remove absolute bias.")
    elif vr_val > 0.50 and rp_val < 0.20:
        print(f"  CRITIC EXPLAINS AWAY z-SIGNAL: V(s,z) tracks z-specific returns.")
        print(f"  V spread is {vr_val*100:.0f}% of return spread -- critic absorbs between-z contrast.")
        print(f"  -> Use z-agnostic critic (extra_dim=0).")
    else:
        print(f"  MIXED: Pearson={rp_val:.3f}  sign_agree={sa_val:.3f}  bias={mb_val:+.3f}")
        print(f"  -> Check both critic bias and credit period alignment.")

    print(f"{'='*72}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True,
                        help="probe_a_dataset.csv from run_probe_a.py (has ctx_0..33 + return_z0..z3)")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out_dir) if args.out_dir else Path(f"experiments/credit_audit_runs/{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_name = Path(args.checkpoint).name
    print(f"Episode-Persistent Credit Audit  ckpt={ckpt_name}")
    print(f"  dataset={args.dataset}")

    # Load dataset
    print("\nLoading probe_a_dataset...")
    X, Z_returns, rows = load_dataset(Path(args.dataset))
    N = len(rows)
    print(f"  N={N}  context_dim={X.shape[1]}  forced_returns_per_episode={Z_returns.shape[1]}")

    # Load model
    print("\nLoading model from checkpoint...")
    policy, model, critic, gs_dim, latent_k = load_policy_and_model(args.checkpoint, device=args.device)

    # Batch router forward -- get z_selected and log_prob per episode
    print("\nRunning router forward on all episode contexts (batch, no env)...")
    z_selected, log_probs = batch_router_forward(model, X, gs_dim=gs_dim, device=args.device)
    z_counts = {int(z): int(np.sum(z_selected == z)) for z in range(latent_k)}
    print(f"  Router z selection: " + "  ".join(f"z{z}={c}" for z, c in sorted(z_counts.items())))

    # Batch critic forward -- V(s_0, z) for all z
    print("Running critic forward on all episode contexts (batch, all z)...")
    V_critic = batch_critic_all_z(critic, X, gs_dim=gs_dim, latent_k=latent_k, device=args.device)
    print(f"  V_critic: mean={V_critic.mean():.4f}  std={V_critic.std():.4f}")

    # Run audit
    print("\nComputing alignment metrics...")
    result = run_audit(Z_returns, z_selected, log_probs, V_critic)

    print_audit(result)

    # Save JSON
    out_json = out_dir / "credit_audit_results.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"  Results -> {out_json}")

    # Save per-episode CSV
    out_csv = out_dir / "credit_audit_episodes.csv"
    mean_z_return = Z_returns.mean(axis=1)
    idx = np.arange(N)
    ep_return = Z_returns[idx, z_selected]
    V_sel = V_critic[idx, z_selected]
    online_adv = ep_return - V_sel
    oracle_adv = ep_return - mean_z_return

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        fields = (
            ["opponent", "map", "episode_index", "episode_seed", "cell_seed",
             "z_router", "log_prob", "ep_return"]
            + [f"forced_return_z{z}" for z in range(latent_k)]
            + [f"V_z{z}" for z in range(latent_k)]
            + ["V_selected", "mean_z_return",
               "online_advantage", "oracle_advantage",
               "oracle_best_z", "critic_best_z"]
        )
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i, row in enumerate(rows):
            writer.writerow({
                "opponent": row.get("opponent", ""),
                "map": row.get("map", ""),
                "episode_index": row.get("episode_index", i),
                "episode_seed": row.get("episode_seed", ""),
                "cell_seed": row.get("cell_seed", ""),
                "z_router": int(z_selected[i]),
                "log_prob": float(log_probs[i]),
                "ep_return": float(ep_return[i]),
                **{f"forced_return_z{z}": float(Z_returns[i, z]) for z in range(latent_k)},
                **{f"V_z{z}": float(V_critic[i, z]) for z in range(latent_k)},
                "V_selected": float(V_sel[i]),
                "mean_z_return": float(mean_z_return[i]),
                "online_advantage": float(online_adv[i]),
                "oracle_advantage": float(oracle_adv[i]),
                "oracle_best_z": int(np.argmax(Z_returns[i])),
                "critic_best_z": int(np.argmax(V_critic[i])),
            })
    print(f"  Per-episode -> {out_csv}\n")


if __name__ == "__main__":
    main()
