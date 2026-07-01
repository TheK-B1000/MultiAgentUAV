#!/usr/bin/env python3
"""Analyze whether q_phi selects different latent strategies for different opponents.

Reads the per-step ``*_e3_steps.csv`` and per-episode ``*_episodes.csv`` produced
by ``rl/train_ppo.py --e3-step-telemetry``, and computes:

    P(z | opponent)            from step-level e3 CSV (chunked read to stay
                                memory-bounded; e3 CSVs are ~250 MB)
    WR(z, opponent)            from episode-level CSV (each row already has
                                ``latent_z``, ``opponent``, ``success``)
    H(z), H(opponent), MI(z; opponent), NMI = MI / min(H(z), H(opponent))

Outputs (under --out-dir, default = checkpoints/2v2/analysis/):

    latent_occupancy_by_opponent.csv     one row per (run_tag, opponent, z)
                                          with P(z|opp), step_count, episode_count
    wr_by_z_opponent.csv                  one row per (run_tag, opponent, z)
                                          with WR, n_episodes, wins/losses/draws
    latent_mi_summary.csv                 one row per run_tag with H(z), H(opp),
                                          MI(z;opp), NMI, dominant_z_per_opp

Optional plots (--plots) saved to --out-dir:

    {run_tag}_P_z_given_opponent.png
    {run_tag}_WR_z_opponent.png

Usage
-----

Run on the five hard-pool ablations (default --run-tag-glob):

    python experiments/analyze_latent_mi.py

Run on a specific set of run_tags:

    python experiments/analyze_latent_mi.py \\
        --checkpoint-dir checkpoints/2v2 \\
        --run-tags plan_faithful_latent_persist_entropy_hardpool_1m_2v2 \\
                   plan_faithful_latent_no_entropy_hardpool_1m_2v2 \\
        --plots

Robustness
----------

- Tolerates column-name variants: ``z_t``/``z``, ``opponent_id``/``opponent_tag``,
  ``latent_z``/``z``.
- Skips runs without a latent column (e.g. plan_faithful_no_latent).
- Chunks e3 CSV reads (default chunksize 200_000 rows) so memory stays under ~50 MB
  even for the ~250 MB hard-pool files.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Iterable, Optional

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


DEFAULT_HARDPOOL_RUN_TAGS: tuple[str, ...] = (
    "plan_faithful_latent_persist_entropy_hardpool_1m_2v2",
    "plan_faithful_no_latent_hardpool_1m_2v2",
    "plan_faithful_latent_k1_hardpool_1m_2v2",
    "plan_faithful_latent_no_persistence_hardpool_1m_2v2",
    "plan_faithful_latent_no_entropy_hardpool_1m_2v2",
)


def _pick_column(df_cols: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    """Return the first candidate present in ``df_cols`` (case-sensitive), else None."""
    cols = set(df_cols)
    for cand in candidates:
        if cand in cols:
            return cand
    return None


def _build_opponent_map_from_episodes(ep_df: pd.DataFrame) -> dict[int, str]:
    """Build opponent_id -> opponent_tag mapping from the episode CSV.

    The training CSV writes both columns; we use the most common tag per id in
    case the integer encoding ever drifted mid-run.
    """
    id_col = _pick_column(ep_df.columns, ("opponent_id",))
    tag_col = _pick_column(ep_df.columns, ("opponent", "opponent_tag"))
    if id_col is None or tag_col is None:
        return {}
    mapping: dict[int, str] = {}
    grouped = ep_df.groupby(id_col)[tag_col].agg(lambda s: s.value_counts().index[0])
    for oid, tag in grouped.items():
        try:
            mapping[int(oid)] = str(tag)
        except (TypeError, ValueError):
            continue
    return mapping


def _normalize_opponent_label(raw: str) -> str:
    """Strip the SCRIPTED: prefix and uppercase; keep the rest as-is."""
    s = str(raw).strip()
    if s.upper().startswith("SCRIPTED:"):
        s = s.split(":", 1)[1]
    return s.upper()


def _wr_by_z_opponent_from_episodes(
    ep_df: pd.DataFrame,
    *,
    run_tag: str,
) -> pd.DataFrame:
    """Episode-level WR per (z, opponent). One row per (opponent, z) for the run."""
    z_col = _pick_column(ep_df.columns, ("latent_z", "z"))
    tag_col = _pick_column(ep_df.columns, ("opponent", "opponent_tag"))
    if z_col is None or tag_col is None:
        return pd.DataFrame()
    df = ep_df.copy()
    # Episodes with no latent (no_latent baseline) have latent_z == NaN or -1; drop them
    # since the question "which z did the agent pick?" doesn't apply.
    df = df.dropna(subset=[z_col])
    df = df[df[z_col].astype(int) >= 0]
    if df.empty:
        return pd.DataFrame()
    df["__opp__"] = df[tag_col].map(_normalize_opponent_label)
    df["__z__"] = df[z_col].astype(int)
    df["__win__"] = (df.get("success", 0).astype(int) == 1).astype(int)
    if "win_margin" in df.columns:
        df["__draw__"] = (
            (df["__win__"] == 0) & (df["win_margin"].astype(float) == 0.0)
        ).astype(int)
    else:
        df["__draw__"] = 0
    df["__loss__"] = ((df["__win__"] == 0) & (df["__draw__"] == 0)).astype(int)
    g = df.groupby(["__opp__", "__z__"])
    out = g.agg(
        n_episodes=("__win__", "size"),
        wins=("__win__", "sum"),
        losses=("__loss__", "sum"),
        draws=("__draw__", "sum"),
    ).reset_index()
    out["wr"] = out["wins"] / out["n_episodes"].clip(lower=1)
    out.insert(0, "run_tag", run_tag)
    out = out.rename(columns={"__opp__": "opponent", "__z__": "z"})
    return out[["run_tag", "opponent", "z", "n_episodes", "wins", "losses", "draws", "wr"]]


def _z_occupancy_from_steps(
    steps_path: str,
    *,
    run_tag: str,
    opponent_id_to_tag: dict[int, str],
    chunksize: int,
) -> tuple[pd.DataFrame, Optional[np.ndarray]]:
    """Step-level P(z|opp), and joint count matrix used for MI.

    Returns
    -------
    occupancy_df : DataFrame with columns
        run_tag, opponent, z, step_count, p_z_given_opp
    joint_counts : np.ndarray of shape (n_z, n_opp) or None if file unreadable
    """
    if not os.path.isfile(steps_path):
        return pd.DataFrame(), None

    # Probe header once to pick column names (variant-tolerant)
    header_df = pd.read_csv(steps_path, nrows=0)
    z_col = _pick_column(header_df.columns, ("z_t", "z"))
    opp_col = _pick_column(header_df.columns, ("opponent_id", "opponent_tag", "opponent"))
    if z_col is None or opp_col is None:
        print(
            f"[analyze_latent_mi] {run_tag}: e3 CSV missing z/opponent columns "
            f"(found {list(header_df.columns)[:8]}...); skipping step-level occupancy."
        )
        return pd.DataFrame(), None

    usecols = [z_col, opp_col]
    joint: dict[tuple[int, int], int] = {}

    n_rows = 0
    for chunk in pd.read_csv(steps_path, usecols=usecols, chunksize=chunksize):
        n_rows += len(chunk)
        chunk = chunk.dropna()
        # Both columns coerce to int (opponent might be a string tag)
        try:
            z_vals = chunk[z_col].astype(int).to_numpy()
        except (ValueError, TypeError):
            continue
        if pd.api.types.is_numeric_dtype(chunk[opp_col]):
            opp_vals = chunk[opp_col].astype(int).to_numpy()
        else:
            # Map string tags through normalization, then synthesize ints from sorted set
            normalized = chunk[opp_col].map(_normalize_opponent_label)
            # If opponent_id_to_tag is empty, build a reverse mapping on the fly
            if not opponent_id_to_tag:
                uniq = sorted(normalized.unique())
                opponent_id_to_tag = {i: t for i, t in enumerate(uniq)}
            tag_to_id = {t: i for i, t in opponent_id_to_tag.items()}
            opp_vals = normalized.map(tag_to_id).fillna(-1).astype(int).to_numpy()
        for z, o in zip(z_vals, opp_vals):
            if o < 0:
                continue
            key = (int(z), int(o))
            joint[key] = joint.get(key, 0) + 1

    if not joint:
        return pd.DataFrame(), None

    zs = sorted({z for z, _ in joint.keys()})
    opps = sorted({o for _, o in joint.keys()})
    n_z = len(zs)
    n_o = len(opps)
    z_idx = {z: i for i, z in enumerate(zs)}
    o_idx = {o: i for i, o in enumerate(opps)}
    M = np.zeros((n_z, n_o), dtype=np.int64)
    for (z, o), c in joint.items():
        M[z_idx[z], o_idx[o]] = c

    occ_rows = []
    col_sums = M.sum(axis=0)
    for oi, o in enumerate(opps):
        tag = opponent_id_to_tag.get(int(o), f"opp_id_{int(o)}")
        col_sum = int(col_sums[oi]) or 1
        for zi, z in enumerate(zs):
            occ_rows.append(
                {
                    "run_tag": run_tag,
                    "opponent": _normalize_opponent_label(tag),
                    "z": int(z),
                    "step_count": int(M[zi, oi]),
                    "p_z_given_opp": float(M[zi, oi]) / float(col_sum),
                }
            )
    occ_df = pd.DataFrame(occ_rows)
    return occ_df, M


def _entropy_bits(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    p = p[p > 0.0]
    return float(-(p * np.log2(p)).sum())


def _mi_summary_from_joint(
    M: np.ndarray,
    *,
    run_tag: str,
    opponent_id_to_tag: dict[int, str],
    opponent_ids_in_order: list[int],
) -> dict:
    """Compute H(z), H(opp), MI(z;opp), NMI from a joint count matrix.

    M : (n_z, n_opp) int counts
    """
    total = float(M.sum()) or 1.0
    p_zo = M.astype(np.float64) / total
    p_z = p_zo.sum(axis=1)
    p_o = p_zo.sum(axis=0)
    h_z = _entropy_bits(p_z)
    h_o = _entropy_bits(p_o)
    # MI in bits
    mi = 0.0
    for i in range(p_zo.shape[0]):
        for j in range(p_zo.shape[1]):
            if p_zo[i, j] > 0 and p_z[i] > 0 and p_o[j] > 0:
                mi += p_zo[i, j] * math.log2(p_zo[i, j] / (p_z[i] * p_o[j]))
    nmi_denom = min(h_z, h_o) if min(h_z, h_o) > 0 else float("nan")
    nmi = mi / nmi_denom if nmi_denom and nmi_denom == nmi_denom else float("nan")

    n_z, n_o = M.shape
    p_z_given_o = M.astype(np.float64) / np.maximum(M.sum(axis=0, keepdims=True), 1.0)
    dominant_z_per_opp = {}
    for j, oid in enumerate(opponent_ids_in_order):
        zstar = int(np.argmax(p_z_given_o[:, j]))
        tag = opponent_id_to_tag.get(int(oid), f"opp_id_{int(oid)}")
        dominant_z_per_opp[_normalize_opponent_label(tag)] = {
            "z": zstar,
            "p_z_given_opp": float(p_z_given_o[zstar, j]),
        }

    return {
        "run_tag": run_tag,
        "n_z": int(n_z),
        "n_opp": int(n_o),
        "step_count": int(M.sum()),
        "H_z_bits": h_z,
        "H_opp_bits": h_o,
        "MI_z_opp_bits": mi,
        "NMI_z_opp": nmi,
        "dominant_z_per_opp": str(dominant_z_per_opp),  # stringify for CSV
    }


def _maybe_plot_heatmap(
    df: pd.DataFrame,
    *,
    value_col: str,
    title: str,
    out_path: str,
    fmt: str = "{:.2f}",
) -> None:
    """Pivot ``df`` to opponent x z and write a heatmap PNG. Requires matplotlib."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"[analyze_latent_mi] matplotlib unavailable; skipping plot {out_path!r}.")
        return
    pivot = df.pivot(index="opponent", columns="z", values=value_col)
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(1.5 + 0.8 * pivot.shape[1], 1.5 + 0.4 * pivot.shape[0]))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis", vmin=0.0, vmax=max(1.0, float(np.nanmax(pivot.values))))
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels([f"z={c}" for c in pivot.columns])
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(list(pivot.index))
    ax.set_title(title)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, fmt.format(v), ha="center", va="center", color="white", fontsize=9)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def analyze_run(
    *,
    run_tag: str,
    checkpoint_dir: str,
    chunksize: int,
) -> tuple[pd.DataFrame, pd.DataFrame, Optional[dict]]:
    """Return (wr_df, occupancy_df, mi_summary_dict) for one run_tag."""
    ep_path = os.path.join(checkpoint_dir, f"{run_tag}_episodes.csv")
    e3_path = os.path.join(checkpoint_dir, f"{run_tag}_e3_steps.csv")

    if not os.path.isfile(ep_path):
        print(f"[analyze_latent_mi] {run_tag}: missing {ep_path}; skipping.")
        return pd.DataFrame(), pd.DataFrame(), None

    ep_df = pd.read_csv(ep_path)
    opp_map = _build_opponent_map_from_episodes(ep_df)

    wr_df = _wr_by_z_opponent_from_episodes(ep_df, run_tag=run_tag)
    if wr_df.empty:
        print(f"[analyze_latent_mi] {run_tag}: no latent_z column in episodes CSV "
              f"(likely no_latent baseline); skipping z analyses.")
        return wr_df, pd.DataFrame(), None

    occ_df, M = _z_occupancy_from_steps(
        e3_path,
        run_tag=run_tag,
        opponent_id_to_tag=opp_map,
        chunksize=chunksize,
    )
    if occ_df.empty or M is None:
        print(f"[analyze_latent_mi] {run_tag}: no step-level z occupancy "
              f"(e3 CSV missing or unreadable); WR table still produced.")
        return wr_df, occ_df, None

    opp_ids = sorted({int(oid) for oid in opp_map.keys()})
    if not opp_ids:
        # Fall back to ints found in the joint matrix's column order; M columns
        # were ordered by sorted unique opponent_id values from the chunks.
        opp_ids = list(range(M.shape[1]))
    mi_sum = _mi_summary_from_joint(
        M,
        run_tag=run_tag,
        opponent_id_to_tag=opp_map,
        opponent_ids_in_order=opp_ids,
    )
    return wr_df, occ_df, mi_sum


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=os.path.join("checkpoints", "2v2"),
        help="Directory containing *_episodes.csv and *_e3_steps.csv (default: checkpoints/2v2).",
    )
    parser.add_argument(
        "--run-tags",
        nargs="+",
        default=None,
        help=f"Explicit run_tag list to analyze. Default: {len(DEFAULT_HARDPOOL_RUN_TAGS)} hard-pool tags.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory for CSVs and optional plots (default: <checkpoint-dir>/analysis).",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=200_000,
        help="Rows per pandas chunk when streaming the e3 CSV (default: 200000).",
    )
    parser.add_argument("--plots", action="store_true", help="Also write PNG heatmaps per run_tag.")
    args = parser.parse_args()

    checkpoint_dir = os.path.abspath(args.checkpoint_dir)
    out_dir = os.path.abspath(args.out_dir or os.path.join(checkpoint_dir, "analysis"))
    os.makedirs(out_dir, exist_ok=True)

    run_tags = args.run_tags or list(DEFAULT_HARDPOOL_RUN_TAGS)

    all_wr: list[pd.DataFrame] = []
    all_occ: list[pd.DataFrame] = []
    all_mi: list[dict] = []
    for tag in run_tags:
        print(f"[analyze_latent_mi] === {tag} ===")
        wr_df, occ_df, mi_sum = analyze_run(
            run_tag=tag,
            checkpoint_dir=checkpoint_dir,
            chunksize=int(args.chunksize),
        )
        if not wr_df.empty:
            all_wr.append(wr_df)
        if not occ_df.empty:
            all_occ.append(occ_df)
        if mi_sum is not None:
            all_mi.append(mi_sum)
            print(
                f"  H(z)={mi_sum['H_z_bits']:.3f} bits  H(opp)={mi_sum['H_opp_bits']:.3f} bits  "
                f"MI(z;opp)={mi_sum['MI_z_opp_bits']:.4f} bits  NMI={mi_sum['NMI_z_opp']:.4f}  "
                f"step_count={mi_sum['step_count']:,}"
            )
        if args.plots and not occ_df.empty:
            _maybe_plot_heatmap(
                occ_df,
                value_col="p_z_given_opp",
                title=f"P(z | opponent) — {tag}",
                out_path=os.path.join(out_dir, f"{tag}_P_z_given_opponent.png"),
                fmt="{:.2f}",
            )
        if args.plots and not wr_df.empty:
            _maybe_plot_heatmap(
                wr_df,
                value_col="wr",
                title=f"WR(z, opponent) — {tag}",
                out_path=os.path.join(out_dir, f"{tag}_WR_z_opponent.png"),
                fmt="{:.2f}",
            )

    if all_wr:
        wr_path = os.path.join(out_dir, "wr_by_z_opponent.csv")
        pd.concat(all_wr, ignore_index=True).to_csv(wr_path, index=False)
        print(f"\n[analyze_latent_mi] wrote {wr_path}")
    if all_occ:
        occ_path = os.path.join(out_dir, "latent_occupancy_by_opponent.csv")
        pd.concat(all_occ, ignore_index=True).to_csv(occ_path, index=False)
        print(f"[analyze_latent_mi] wrote {occ_path}")
    if all_mi:
        mi_path = os.path.join(out_dir, "latent_mi_summary.csv")
        pd.DataFrame(all_mi).to_csv(mi_path, index=False)
        print(f"[analyze_latent_mi] wrote {mi_path}")

    if all_mi:
        print("\n[analyze_latent_mi] MI summary across runs (sorted by NMI descending):")
        df = pd.DataFrame(all_mi).sort_values("NMI_z_opp", ascending=False)
        for _, r in df.iterrows():
            print(
                f"  {r['run_tag']:<58}  H(z)={r['H_z_bits']:.3f}  "
                f"MI={r['MI_z_opp_bits']:.4f}  NMI={r['NMI_z_opp']:.4f}  "
                f"steps={int(r['step_count']):>9,}"
            )


if __name__ == "__main__":
    main()
