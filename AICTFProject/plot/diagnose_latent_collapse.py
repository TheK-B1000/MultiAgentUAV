#!/usr/bin/env python3
"""Mode-collapse diagnostics for q_phi(z|s) in a latent PPO run.

Reads the trainer's metrics + episodes CSVs and produces:
  - strategy entropy and entropy fraction vs training updates
  - per-z occupancy curves vs updates (4 lines for K=4)
  - MI(z;opponent), MI(z;phase), MI(z;outcome) vs updates
  - episode-level histogram of latent_z over the full run
  - per-opponent z occupancy bar chart (episode level)
  - per-opponent z win rate bar chart (episode level)

Also prints a one-screen summary of how collapsed q_phi is and the
"oracle" upper bound (best fixed z per opponent on training rollouts).
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)


def _read_metrics(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def _read_episodes(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["opponent_tag"] = df["opponent"].astype(str).str.upper().str.replace("SCRIPTED:", "", regex=False)
    df["opponent_tag"] = df["opponent_tag"].replace({"OP5": "OP5_RUSHER"})
    return df


def _line(ax, x, y, label, **kw):
    ax.plot(x, y, label=label, **kw)


def plot_entropy(metrics: pd.DataFrame, out_dir: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    x = metrics["update"]
    if "strategy_entropy" in metrics:
        axes[0].plot(x, metrics["strategy_entropy"], color="C0")
    axes[0].axhline(np.log(4), color="gray", linestyle="--", label="ln(K=4) max entropy")
    axes[0].set_title("strategy_entropy (q_phi(z|s)) vs update")
    axes[0].set_xlabel("policy update")
    axes[0].set_ylabel("entropy (nats)")
    axes[0].legend()

    if "strategy_entropy_frac" in metrics:
        axes[1].plot(x, metrics["strategy_entropy_frac"], color="C2")
    axes[1].axhline(1.0, color="gray", linestyle="--", label="uniform = 1.0")
    axes[1].axhline(0.0, color="gray", linestyle=":", label="degenerate = 0.0")
    axes[1].set_title("strategy_entropy_frac (normalized by ln K)")
    axes[1].set_xlabel("policy update")
    axes[1].set_ylim(-0.05, 1.1)
    axes[1].legend()

    fig.tight_layout()
    path = os.path.join(out_dir, "latent_entropy_vs_update.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"wrote {path}")


def plot_occupancy_curves(metrics: pd.DataFrame, out_dir: str, k: int = 4) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = metrics["update"]
    for z in range(k):
        col = f"strategy_occupancy_{z}"
        if col in metrics:
            ax.plot(x, metrics[col], label=f"z={z}")
    ax.axhline(1.0 / k, color="gray", linestyle="--", label=f"uniform = 1/K = {1.0/k:.2f}")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Per-z occupancy across training rollouts (rollout-level)")
    ax.set_xlabel("policy update")
    ax.set_ylabel("fraction of decision steps")
    ax.legend(loc="best", ncols=3)
    fig.tight_layout()
    path = os.path.join(out_dir, "latent_z_occupancy_vs_update.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"wrote {path}")


def plot_mi(metrics: pd.DataFrame, out_dir: str) -> None:
    cols = [
        ("latent_mi_z_opponent_nats", "MI(z; opponent)"),
        ("latent_mi_z_phase_nats", "MI(z; phase)"),
        ("latent_mi_z_outcome_nats", "MI(z; outcome)"),
    ]
    present = [(c, n) for c, n in cols if c in metrics]
    if not present:
        return
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = metrics["update"]
    for c, n in present:
        ax.plot(x, metrics[c], label=n)
    ax.set_title("Mutual information between z and context (nats)")
    ax.set_xlabel("policy update")
    ax.set_ylabel("nats")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(out_dir, "latent_mi_vs_update.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"wrote {path}")


def plot_episode_histogram(eps: pd.DataFrame, out_dir: str, k: int = 4) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    z = eps["latent_z"].dropna().astype(int)
    counts = z.value_counts().reindex(range(k), fill_value=0).sort_index()
    pct = 100.0 * counts / counts.sum()
    bars = ax.bar(counts.index.astype(str), counts.values, color=[f"C{i}" for i in range(k)])
    for b, p in zip(bars, pct.values):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{p:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_title("Episode-level latent_z usage (full training run)")
    ax.set_xlabel("z")
    ax.set_ylabel("episodes")
    fig.tight_layout()
    path = os.path.join(out_dir, "latent_z_episode_histogram.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"wrote {path}")


def plot_per_opponent_z(eps: pd.DataFrame, out_dir: str, k: int = 4) -> None:
    opps = sorted(eps["opponent_tag"].dropna().unique().tolist())
    eps_op = eps[eps["opponent_tag"].isin(opps)].copy()
    eps_op["latent_z"] = eps_op["latent_z"].astype(int)
    pivot_count = (
        eps_op.groupby(["opponent_tag", "latent_z"]).size().unstack(fill_value=0).reindex(columns=range(k), fill_value=0)
    )
    pivot_frac = pivot_count.div(pivot_count.sum(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    pivot_frac.plot(kind="bar", stacked=True, ax=ax, colormap="viridis", width=0.8)
    ax.set_title("Per-opponent latent_z occupancy (episode level)")
    ax.set_ylabel("fraction of episodes")
    ax.set_xlabel("opponent")
    ax.set_ylim(0.0, 1.0)
    ax.legend(title="z", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    path = os.path.join(out_dir, "latent_z_per_opponent_occupancy.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"wrote {path}")

    wr = eps_op.groupby(["opponent_tag", "latent_z"])["success"].mean().unstack(fill_value=np.nan)
    wr = wr.reindex(columns=range(k))
    fig2, ax2 = plt.subplots(figsize=(8, 4.5))
    wr.plot(kind="bar", ax=ax2, colormap="viridis", width=0.85)
    ax2.set_title("Per-opponent win rate by z (episode level, training rollouts)")
    ax2.set_ylabel("episode WR")
    ax2.set_xlabel("opponent")
    ax2.set_ylim(0.0, 1.05)
    ax2.legend(title="z", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig2.tight_layout()
    path2 = os.path.join(out_dir, "latent_z_per_opponent_winrate.png")
    fig2.savefig(path2, dpi=130)
    plt.close(fig2)
    print(f"wrote {path2}")

    return pivot_count, pivot_frac, wr


def _summarize(eps: pd.DataFrame, metrics: pd.DataFrame, pivot_count, pivot_frac, wr, k: int = 4) -> None:
    print()
    print("=" * 78)
    print("MODE-COLLAPSE SUMMARY")
    print("=" * 78)

    z = eps["latent_z"].dropna().astype(int)
    total = len(z)
    counts = z.value_counts().reindex(range(k), fill_value=0).sort_index()
    pct = (100.0 * counts / max(1, total)).round(1).tolist()
    print(f"Episode-level z usage over {total} training episodes:")
    for zi, p in enumerate(pct):
        print(f"  z={zi}: {counts.iloc[zi]:>5d} eps ({p:5.1f}%)")

    if "strategy_entropy" in metrics and "strategy_entropy_frac" in metrics:
        m = metrics.dropna(subset=["strategy_entropy"]).copy()
        if not m.empty:
            ent_start = float(m["strategy_entropy"].iloc[0])
            ent_end = float(m["strategy_entropy"].iloc[-1])
            ent_min = float(m["strategy_entropy"].min())
            frac_end = float(m["strategy_entropy_frac"].iloc[-1])
            print(
                "\nq_phi entropy: start={:.3f}, end={:.3f} nats (ln K={:.3f}); "
                "frac_end={:.2f} (1.0=uniform, 0.0=degenerate)".format(
                    ent_start, ent_end, float(np.log(k)), frac_end
                )
            )

    mi_cols = ["latent_mi_z_opponent_nats", "latent_mi_z_phase_nats", "latent_mi_z_outcome_nats"]
    if any(c in metrics for c in mi_cols):
        print("\nLast-update MI (nats; higher = z carries more info about context):")
        for c in mi_cols:
            if c in metrics:
                v = metrics[c].dropna()
                if not v.empty:
                    print(f"  {c}: {float(v.iloc[-1]):.4f}")

    print("\nPer-opponent z fractions (episode level):")
    print(pivot_frac.round(3).to_string())

    print("\nPer-opponent WR by z (episode level):")
    print(wr.round(3).to_string())

    print("\nOracle bound (best fixed z per opponent on training rollouts):")
    rows = []
    for opp, row in wr.iterrows():
        z_best = int(row.idxmax())
        wr_best = float(row.max())
        z_pi = int(pivot_frac.loc[opp].idxmax())
        wr_pi = float(row.iloc[z_pi]) if not np.isnan(row.iloc[z_pi]) else float("nan")
        rows.append((opp, z_pi, wr_pi, z_best, wr_best, wr_best - wr_pi))
    df = pd.DataFrame(rows, columns=["opp", "q_phi_picks_z", "q_phi_WR", "best_z", "best_WR", "gap"])
    print(df.round(3).to_string(index=False))
    pool_pi = float(df["q_phi_WR"].mean())
    pool_oracle = float(df["best_WR"].mean())
    print(f"\nPool average WR: q_phi-routed={pool_pi:.3f}, oracle-best-z={pool_oracle:.3f}, headroom={pool_oracle - pool_pi:+.3f}")
    print("=" * 78)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--metrics",
        default=os.path.join(PROJECT_ROOT, "checkpoints", "4v4", "latent_ep_hardpool_4v4_seed1_4v4_metrics.csv"),
    )
    ap.add_argument(
        "--episodes",
        default=os.path.join(PROJECT_ROOT, "checkpoints", "4v4", "latent_ep_hardpool_4v4_seed1_4v4_episodes.csv"),
    )
    ap.add_argument("--out-dir", default=os.path.join(PROJECT_ROOT, "plots", "latent_collapse"))
    ap.add_argument("--k", type=int, default=4)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    metrics = _read_metrics(args.metrics)
    eps = _read_episodes(args.episodes)

    plot_entropy(metrics, args.out_dir)
    plot_occupancy_curves(metrics, args.out_dir, k=args.k)
    plot_mi(metrics, args.out_dir)
    plot_episode_histogram(eps, args.out_dir, k=args.k)
    pivot_count, pivot_frac, wr = plot_per_opponent_z(eps, args.out_dir, k=args.k)

    _summarize(eps, metrics, pivot_count, pivot_frac, wr, k=args.k)
    print(f"\nAll figures saved under: {args.out_dir}")


if __name__ == "__main__":
    main()
