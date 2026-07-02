#!/usr/bin/env python3
"""Probe A supplement: feature stats, unscaled comparison, StrategyEncoder supervised training.

Decision tree this answers:
  - If unscaled logreg fails but scaled logreg passes: normalization mismatch in online router.
  - If supervised StrategyEncoder (fresh init) fails but scaled sklearn passes:
      architecture or optimization capacity is insufficient.
  - If supervised StrategyEncoder (from checkpoint) fails but fresh succeeds:
      checkpoint weights are stuck in a bad basin (pre-training hurt, not helped).
  - All StrategyEncoder variants fail: encoder is not the bottleneck -> check credit/objective.

Usage
-----
    uv run python experiments/run_probe_a_supervised.py \\
        --checkpoint checkpoints/2v2/final_v6i9-mapaware-repertoire-hardpool-refactor-r1-seed1_2v2.zip \\
        --dataset experiments/probe_a_runs/20260702T123105Z/probe_a_dataset.csv \\
        [--out-dir experiments/probe_a_runs/supervised_<stamp>]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn.functional as F

CONTEXT_DIM_PROBE = 34  # dims collected by run_probe_a.py (env.state())
CONTEXT_DIM_ENCODER = 35  # actual router input: 34 env + 1 scheduler phase


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset(csv_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    """Returns X(N,34), y(N,), oracle_returns(N,), z_returns_matrix(N,4), raw_rows."""
    rows = []
    with csv_path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)

    ctx_keys = [f"ctx_{i}" for i in range(CONTEXT_DIM_PROBE)]
    X = np.array([[float(r[k]) for k in ctx_keys] for r in rows], dtype=np.float32)
    y = np.array([int(r["best_z"]) for r in rows], dtype=np.int64)
    oracle_returns = np.array([float(r["oracle_return"]) for r in rows], dtype=np.float32)
    z_returns = np.array([[float(r[f"return_z{z}"]) for z in range(4)] for r in rows], dtype=np.float32)
    return X, y, oracle_returns, z_returns, rows


def grouped_split(rows: list[dict], test_frac: float = 0.2, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Split by episode_seed to prevent leakage. Returns (train_mask, test_mask)."""
    ep_seeds = np.array([int(r.get("episode_seed", i)) for i, r in enumerate(rows)])
    unique = np.unique(ep_seeds)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    n_test = max(1, int(len(unique) * test_frac))
    test_seeds = set(unique[:n_test].tolist())
    train_mask = np.array([s not in test_seeds for s in ep_seeds])
    return train_mask, ~train_mask


# ---------------------------------------------------------------------------
# Feature stats
# ---------------------------------------------------------------------------

def feature_stats(X: np.ndarray) -> list[dict]:
    """Per-dimension statistics."""
    stats = []
    for i in range(X.shape[1]):
        col = X[:, i]
        stats.append({
            "dim": i,
            "mean": float(col.mean()),
            "std": float(col.std()),
            "min": float(col.min()),
            "max": float(col.max()),
            "range": float(col.max() - col.min()),
            "frac_constant": float(np.mean(col == col[0])),
            "frac_near_zero": float(np.mean(np.abs(col) < 1e-4)),
        })
    return stats


def print_feature_stats(stats: list[dict], top_n: int = 10) -> None:
    print(f"\n  Context feature stats ({len(stats)} dims):")
    print(f"  {'Dim':>4} {'Mean':>8} {'Std':>7} {'Range':>8} {'FracConst':>10} {'FracNear0':>10}")
    print("  " + "-" * 55)
    by_range = sorted(stats, key=lambda s: s["range"], reverse=True)
    for s in by_range[:top_n]:
        print(f"  {s['dim']:4d} {s['mean']:8.3f} {s['std']:7.3f} {s['range']:8.3f} "
              f"{s['frac_constant']:10.3f} {s['frac_near_zero']:10.3f}")
    n_constant = sum(1 for s in stats if s["frac_constant"] > 0.99)
    n_near_zero = sum(1 for s in stats if s["frac_near_zero"] > 0.99)
    print(f"\n  Constant dims (>99% same value): {n_constant}")
    print(f"  Near-zero dims (>99% |v|<1e-4):  {n_near_zero}")


# ---------------------------------------------------------------------------
# Sklearn probes (scaled / unscaled comparison)
# ---------------------------------------------------------------------------

def run_sklearn_probes(
    X: np.ndarray,
    y: np.ndarray,
    oracle_returns: np.ndarray,
    z_returns: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    *,
    scaled: bool,
    seed: int = 0,
) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import balanced_accuracy_score

    X_tr, X_te = X[train_mask], X[test_mask]
    y_tr, y_te = y[train_mask], y[test_mask]
    oracle_te = oracle_returns[test_mask]
    z2_te = z_returns[test_mask, 2]
    z_ret_te = z_returns[test_mask]

    if scaled:
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)

    results = {}
    z2_mean = float(z2_te.mean())

    for name, clf in [
        ("logreg", LogisticRegression(max_iter=2000, C=1.0, random_state=seed)),
        ("mlp64x32", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500,
                                    random_state=seed, early_stopping=True, validation_fraction=0.15)),
    ]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(X_tr, y_tr)
        preds = clf.predict(X_te)
        pred_returns = z_ret_te[np.arange(len(y_te)), preds]
        results[name] = {
            "accuracy": float(np.mean(y_te == preds)),
            "balanced_accuracy": float(balanced_accuracy_score(y_te, preds)),
            "mean_predicted_return": float(pred_returns.mean()),
            "delta_vs_z2": float(pred_returns.mean() - z2_mean),
            "mean_regret": float((oracle_te - pred_returns).mean()),
        }

    return results


# ---------------------------------------------------------------------------
# StrategyEncoder supervised training
# ---------------------------------------------------------------------------

def load_strategy_encoder_from_checkpoint(
    ckpt_path: str,
    *,
    device: str = "cpu",
) -> tuple["torch.nn.Module", int]:
    """Extract StrategyEncoder from checkpoint. Returns (encoder, q_phi_input_dim)."""
    from rl.custom_ppo import load_custom_ppo_policy

    dev = torch.device(device)
    import zipfile, io
    from pathlib import Path as P

    # Load checkpoint via stable-baselines3 zip format
    ckpt = P(ckpt_path)
    if ckpt.suffix == ".zip":
        import zipfile as zf
        with zf.ZipFile(ckpt, "r") as archive:
            with archive.open("policy.pth") as f:
                state = torch.load(io.BytesIO(f.read()), map_location="cpu")
    else:
        state = torch.load(ckpt, map_location="cpu")

    # Try to read q_phi_input_dim from policy_kwargs or params
    q_phi_input_dim = CONTEXT_DIM_ENCODER  # default
    if isinstance(state, dict):
        # Could be SB3 format: state["policy_kwargs"]["q_phi_input_dim"]
        kwargs = state.get("policy_kwargs") or {}
        if "q_phi_input_dim" in kwargs:
            q_phi_input_dim = int(kwargs["q_phi_input_dim"])

    return None, q_phi_input_dim  # placeholder


def load_strategy_encoder_via_inference_policy(
    ckpt_path: str,
    *,
    map_name: str = "map_b",
    device: str = "cpu",
) -> tuple["torch.nn.Module", int]:
    """Load full policy, extract strategy_encoder module."""
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
        map_layout=map_name,
        max_decision_steps=400,
        aquaticus_profile=True,
        rules_profile="OURS",
    ))
    try:
        policy = load_custom_ppo_policy(ckpt_path, env.observation_space, env.action_space, device=device)
        model = policy.model
        encoder = getattr(model, "strategy_encoder", None)
        if encoder is None:
            raise RuntimeError("checkpoint has no strategy_encoder (latent strategy not enabled)")
        q_phi_dim = int(model.q_phi_input_dim)
        print(f"  StrategyEncoder: state_dim={encoder.state_dim}  hidden={encoder.hidden_dim}  latent_k={encoder.latent_k}")
        print(f"  q_phi_input_dim={q_phi_dim}")
        # Detach encoder so we can fine-tune without affecting the loaded policy
        import copy
        encoder_copy = copy.deepcopy(encoder).to("cpu")
        return encoder_copy, q_phi_dim
    finally:
        env.close()


def train_strategy_encoder_supervised(
    encoder: "torch.nn.Module",
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    oracle_test: np.ndarray,
    z_returns_test: np.ndarray,
    *,
    q_phi_input_dim: int,
    n_epochs: int = 300,
    lr: float = 3e-4,
    seed: int = 0,
    label: str = "",
) -> dict:
    """Train encoder with cross-entropy on best_z target; evaluate return regret."""
    torch.manual_seed(seed)

    # Pad 34-dim probe context to q_phi_input_dim (35 or other)
    # The 35th dim is scheduler_phase, which is ~0 at episode start in inference
    pad = q_phi_input_dim - CONTEXT_DIM_PROBE
    if pad > 0:
        X_train = np.pad(X_train, ((0, 0), (0, pad)), mode="constant")
        X_test = np.pad(X_test, ((0, 0), (0, pad)), mode="constant")
    elif pad < 0:
        X_train = X_train[:, :q_phi_input_dim]
        X_test = X_test[:, :q_phi_input_dim]

    Xtr = torch.tensor(X_train, dtype=torch.float32)
    ytr = torch.tensor(y_train, dtype=torch.long)
    Xte = torch.tensor(X_test, dtype=torch.float32)
    yte = torch.tensor(y_test, dtype=torch.long)

    # Optional: normalize input to match StandardScaler
    mu = Xtr.mean(0, keepdim=True)
    sigma = Xtr.std(0, keepdim=True).clamp(min=1e-8)

    encoder = encoder.train()
    opt = torch.optim.Adam(encoder.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

    best_test_acc = 0.0
    best_state = None

    for epoch in range(n_epochs):
        # Mini-batch SGD
        perm = torch.randperm(len(Xtr))
        total_loss = 0.0
        for start in range(0, len(Xtr), 64):
            idx = perm[start:start + 64]
            xb = (Xtr[idx] - mu) / sigma
            yb = ytr[idx]
            logits = encoder(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
        scheduler.step()

        if (epoch + 1) % 50 == 0:
            with torch.no_grad():
                te_logits = encoder((Xte - mu) / sigma)
                te_acc = float((te_logits.argmax(1) == yte).float().mean())
                if te_acc > best_test_acc:
                    best_test_acc = te_acc
                    best_state = {k: v.clone() for k, v in encoder.state_dict().items()}
            # print(f"    epoch {epoch+1}: loss={total_loss:.4f} test_acc={te_acc:.3f}")

    # Evaluate best checkpoint
    if best_state is not None:
        encoder.load_state_dict(best_state)
    encoder.eval()
    with torch.no_grad():
        te_logits = encoder((Xte - mu) / sigma)
        preds = te_logits.argmax(1).numpy()

    from sklearn.metrics import balanced_accuracy_score
    pred_returns = z_returns_test[np.arange(len(y_test)), preds]
    z2_mean = float(z_returns_test[:, 2].mean())
    return {
        "label": label,
        "accuracy": float(np.mean(y_test == preds)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, preds)),
        "mean_predicted_return": float(pred_returns.mean()),
        "delta_vs_z2": float(pred_returns.mean() - z2_mean),
        "mean_regret": float((oracle_test - pred_returns).mean()),
        "best_test_acc_during_training": best_test_acc,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset", required=True, help="probe_a_dataset.csv from run_probe_a.py")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--n-epochs", type=int, default=300)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    ckpt_path = str(Path(args.checkpoint).resolve())
    dataset_path = Path(args.dataset)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out_dir or f"experiments/probe_a_runs/supervised_{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Probe A Supervised  ckpt={Path(ckpt_path).name}")
    print(f"  dataset={dataset_path}")

    # ── Load dataset ──────────────────────────────────────────────────────
    X, y, oracle_returns, z_returns, rows = load_dataset(dataset_path)
    train_mask, test_mask = grouped_split(rows, test_frac=args.test_frac)
    X_tr, X_te = X[train_mask], X[test_mask]
    y_tr, y_te = y[train_mask], y[test_mask]
    oracle_te = oracle_returns[test_mask]
    z_ret_te = z_returns[test_mask]
    z2_baseline = float(z_returns[:, 2].mean())
    z2_te_mean = float(z_ret_te[:, 2].mean())

    print(f"  N={len(rows)}  train={train_mask.sum()}  test={test_mask.sum()}")
    print(f"  Fixed-z2 mean return (test): {z2_te_mean:.4f}")
    print(f"  Oracle mean return (test):   {oracle_te.mean():.4f}")

    # ── Feature stats ─────────────────────────────────────────────────────
    print("\n--- Feature Statistics ---")
    fstats = feature_stats(X)
    print_feature_stats(fstats)
    (out_dir / "feature_stats.json").write_text(json.dumps(fstats, indent=2), encoding="utf-8")

    # ── Scaled vs unscaled sklearn ─────────────────────────────────────────
    print("\n--- Scaled vs Unscaled Sklearn Probes ---")
    all_results = {}

    for scale_mode in ("scaled", "unscaled"):
        scaled = (scale_mode == "scaled")
        res = run_sklearn_probes(X, y, oracle_returns, z_returns, train_mask, test_mask,
                                  scaled=scaled)
        all_results[scale_mode] = res
        print(f"\n  {scale_mode}:")
        for name, r in res.items():
            verdict = "BEATS z2" if r["delta_vs_z2"] > 0.01 else ("~= z2" if r["delta_vs_z2"] > -0.01 else "LOSES")
            print(f"    {name:<12} acc={r['accuracy']:.3f}  ret={r['mean_predicted_return']:.4f}  "
                  f"d_vs_z2={r['delta_vs_z2']:+.4f}  {verdict}")

    # ── Load actual StrategyEncoder from checkpoint ───────────────────────
    print("\n--- Loading StrategyEncoder from checkpoint ---")
    import copy

    try:
        encoder_ckpt, q_phi_dim = load_strategy_encoder_via_inference_policy(
            ckpt_path, map_name="map_b", device=args.device
        )
        encoder_fresh = copy.deepcopy(encoder_ckpt)
        # Reset fresh copy to random init
        for m in encoder_fresh.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.zeros_(m.bias)

        print(f"\n--- StrategyEncoder Supervised Training (n_epochs={args.n_epochs}) ---")

        # Fine-tune from checkpoint weights
        enc_ft = copy.deepcopy(encoder_ckpt)
        res_ft = train_strategy_encoder_supervised(
            enc_ft, X_tr, y_tr, X_te, y_te, oracle_te, z_ret_te,
            q_phi_input_dim=q_phi_dim, n_epochs=args.n_epochs, label="finetune_from_ckpt"
        )
        all_results["strategy_encoder_finetuned"] = res_ft
        print(f"  Fine-tuned (from ckpt):  acc={res_ft['accuracy']:.3f}  "
              f"ret={res_ft['mean_predicted_return']:.4f}  d_vs_z2={res_ft['delta_vs_z2']:+.4f}  "
              f"{'BEATS z2' if res_ft['delta_vs_z2'] > 0.01 else 'LOSES'}")

        # Train from random init
        enc_rnd = copy.deepcopy(encoder_fresh)
        res_rnd = train_strategy_encoder_supervised(
            enc_rnd, X_tr, y_tr, X_te, y_te, oracle_te, z_ret_te,
            q_phi_input_dim=q_phi_dim, n_epochs=args.n_epochs, label="random_init"
        )
        all_results["strategy_encoder_random_init"] = res_rnd
        print(f"  Random init:             acc={res_rnd['accuracy']:.3f}  "
              f"ret={res_rnd['mean_predicted_return']:.4f}  d_vs_z2={res_rnd['delta_vs_z2']:+.4f}  "
              f"{'BEATS z2' if res_rnd['delta_vs_z2'] > 0.01 else 'LOSES'}")

    except Exception as e:
        print(f"  StrategyEncoder loading failed: {e}")
        all_results["strategy_encoder_finetuned"] = {"error": str(e)}
        all_results["strategy_encoder_random_init"] = {"error": str(e)}

    # ── Print decision tree verdict ────────────────────────────────────────
    print(f"\n{'='*65}")
    print("Probe A Supervised -- Decision Tree Verdict")
    print(f"{'='*65}")
    z2_ref = z2_te_mean

    def _beats(key: str, sub_key: str = "default") -> bool:
        d = all_results.get(key, {})
        if sub_key == "default":
            return d.get("delta_vs_z2", -999) > 0.01
        return d.get(sub_key, {}).get("delta_vs_z2", -999) > 0.01

    scaled_lr_beats = all_results.get("scaled", {}).get("logreg", {}).get("delta_vs_z2", -999) > 0.01
    unscaled_lr_beats = all_results.get("unscaled", {}).get("logreg", {}).get("delta_vs_z2", -999) > 0.01
    enc_ft_beats = all_results.get("strategy_encoder_finetuned", {}).get("delta_vs_z2", -999) > 0.01
    enc_rnd_beats = all_results.get("strategy_encoder_random_init", {}).get("delta_vs_z2", -999) > 0.01

    print(f"\n  Scaled logreg beats z2:         {'YES' if scaled_lr_beats else 'NO'}")
    print(f"  Unscaled logreg beats z2:       {'YES' if unscaled_lr_beats else 'NO'}")
    print(f"  StrategyEncoder (finetuned):    {'YES' if enc_ft_beats else 'NO'}")
    print(f"  StrategyEncoder (random init):  {'YES' if enc_rnd_beats else 'NO'}")

    print()
    if scaled_lr_beats and not unscaled_lr_beats:
        print("  -> NORMALIZATION MISMATCH: Router receives unnormalized context but signal")
        print("     only surfaces with StandardScaler. Add z-score normalization to online router.")
    elif scaled_lr_beats and unscaled_lr_beats and not enc_rnd_beats:
        print("  -> ENCODER CAPACITY / OPTIMIZATION: Sklearn succeeds but StrategyEncoder")
        print("     (same architecture, supervised) fails. Check learning rate, batch size,")
        print("     or try adding BatchNorm before the encoder.")
    elif enc_rnd_beats and not enc_ft_beats:
        print("  -> CHECKPOINT STUCK: Random-init encoder learns; checkpoint weights are in")
        print("     a bad basin from online training. PPO gradient damage is likely.")
    elif enc_ft_beats or enc_rnd_beats:
        print("  -> ENCODER CAN LEARN: StrategyEncoder (supervised) succeeds. The online")
        print("     training loop is failing to produce the right gradient signal.")
        print("     Check: credit attribution, advantage normalization, critic baseline.")
    else:
        print("  -> All probes fail. Context may lack sufficient signal,")
        print("     or the probe needs more data / longer training.")

    # ── Save results ─────────────────────────────────────────────────────
    result_path = out_dir / "probe_a_supervised_results.json"
    result_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\n  Results -> {result_path}")


if __name__ == "__main__":
    main()
