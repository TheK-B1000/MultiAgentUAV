"""Test 3 (clean): P(z|opponent) confusion matrix from an EVAL checkpoint.

Reads an eval per-episode CSV (produced by ``plot/eval_checkpoint.py`` *without* ``--fixed-latent-id``)
and produces:

    * P(z|opp) confusion matrix (per-opponent z occupancy from ``strategy_dominant`` or
      from per-row ``strategy_occupancy_z`` columns when both are present).
    * WR by opp x z when sample sizes allow.
    * KL divergence between each per-opponent z-distribution and the marginal P(z) to quantify
      *how* opponent-conditional the routing is. KL=0 -> q_phi ignores opponent identity.
    * One-line verdict.

Why this is the *clean* routing test (vs Test 1 / diagnose_latent_collapse.py on the training CSV):
    Training episode CSVs mix q_phi's choices across millions of intermediate policy updates and
    against an evolving actor. An eval CSV is the snapshot at a fixed checkpoint -- the routing
    function as it actually deploys.

Usage:
    # 1) run a no-fixed-z eval against the four pool opponents:
    python plot/eval_checkpoint.py --checkpoint checkpoints/4v4/final_latent_sharp3_oracleA_4v4_seed1_4v4.zip \
        --opponents OP3 OP5_RUSHER OP6 OP7 --episodes 200 --map-sets train

    # 2) point this tool at the per-episode CSV (typical path: csv/<label>_per_episode_train.csv):
    python tools/eval_qphi_routing.py csv/final_latent_sharp3_oracleA_4v4_seed1_4v4_per_episode_train.csv
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


def _resolve_opponent_column(df: pd.DataFrame) -> str:
    """Pick the column that holds the scripted opponent tag for this CSV variant."""
    for c in ("opponent_tag", "opponent", "opponent_key"):
        if c in df.columns:
            return c
    raise SystemExit("ERROR: could not find an opponent column (opponent_tag / opponent / opponent_key)")


def _normalize_opponent(value: object) -> str:
    s = str(value or "").strip().upper().replace("SCRIPTED:", "")
    if s == "OP5":
        s = "OP5_RUSHER"
    if s == "OP6_TURTLE":
        s = "OP6"
    if s == "OP7_SWITCHER":
        s = "OP7"
    return s


def _resolve_k(df: pd.DataFrame) -> int:
    """Determine K from strategy_occupancy_* columns (preferred) or strategy_dominant range."""
    occ_cols = [c for c in df.columns if c.startswith("strategy_occupancy_")]
    if occ_cols:
        return len(occ_cols)
    if "strategy_dominant" in df.columns:
        vals = df["strategy_dominant"].dropna().astype(int)
        if vals.size > 0:
            return int(vals.max()) + 1
    raise SystemExit("ERROR: could not infer K (no strategy_occupancy_* or strategy_dominant column)")


def _occupancy_per_opp_from_occupancy_cols(df: pd.DataFrame, opp_col: str, k: int) -> pd.DataFrame:
    """Average per-step occupancy fraction within each opponent group (preferred -- finer signal)."""
    occ_cols = [f"strategy_occupancy_{z}" for z in range(k)]
    occ_cols = [c for c in occ_cols if c in df.columns]
    if not occ_cols:
        return pd.DataFrame()
    grouped = df.groupby(opp_col)[occ_cols].mean()
    grouped.columns = [int(c.split("_")[-1]) for c in grouped.columns]
    grouped = grouped.reindex(columns=range(k), fill_value=0.0)
    grouped = grouped.div(grouped.sum(axis=1).clip(lower=1e-12), axis=0)
    return grouped


def _occupancy_per_opp_from_dominant(df: pd.DataFrame, opp_col: str, k: int) -> pd.DataFrame:
    """Fallback: episode-dominant-z occupancy (cruder than per-step)."""
    if "strategy_dominant" not in df.columns:
        return pd.DataFrame()
    z = df["strategy_dominant"].dropna().astype(int)
    g = df.loc[z.index].groupby(opp_col)
    counts = (
        g["strategy_dominant"]
        .apply(lambda s: s.astype(int).value_counts())
        .unstack(fill_value=0)
        .reindex(columns=range(k), fill_value=0)
    )
    return counts.div(counts.sum(axis=1).clip(lower=1e-12), axis=0)


def _per_opp_z_winrate(df: pd.DataFrame, opp_col: str, k: int) -> pd.DataFrame:
    if "strategy_dominant" not in df.columns or "success" not in df.columns:
        return pd.DataFrame()
    sub = df.dropna(subset=["strategy_dominant"]).copy()
    sub["strategy_dominant"] = sub["strategy_dominant"].astype(int)
    return (
        sub.groupby([opp_col, "strategy_dominant"])["success"].mean()
        .unstack(fill_value=np.nan)
        .reindex(columns=range(k))
    )


def _marginal_z(occ: pd.DataFrame) -> np.ndarray:
    """Average across opponents (unweighted) to approximate P(z) under uniform pool sampling."""
    return np.asarray(occ.mean(axis=0).values, dtype=np.float64)


def _kl(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float(np.sum(p * (np.log(p) - np.log(q))))


def _verdict(opp_kl_max: float, opp_kl_mean: float, marginal_entropy_frac: float) -> str:
    if opp_kl_max < 0.02:
        return (
            "FAIL  -- P(z|opp) is essentially the same distribution for every opponent. "
            "q_phi is NOT routing on opponent identity (even with oracle, if oracle was on)."
        )
    if opp_kl_max < 0.10:
        return (
            "BORDERLINE  -- small per-opponent divergence; some structure exists but a "
            "different opponent barely shifts q_phi."
        )
    return (
        "PASS  -- per-opponent P(z|opp) differs meaningfully from the marginal P(z). "
        "q_phi IS conditioning its routing on opponent identity."
    )


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="per-episode CSV from eval_checkpoint.py (no --fixed-latent-id)")
    parser.add_argument(
        "--opponents",
        nargs="+",
        default=None,
        help="restrict to this set (case-insensitive; OP5/OP5_RUSHER aliased). Default: all in CSV.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if not args.csv.exists():
        print(f"ERROR: csv not found: {args.csv}", file=sys.stderr)
        return 2

    df = pd.read_csv(args.csv)
    opp_col = _resolve_opponent_column(df)
    df[opp_col] = df[opp_col].map(_normalize_opponent)
    if args.opponents:
        wanted = {_normalize_opponent(o) for o in args.opponents}
        df = df[df[opp_col].isin(wanted)]
        if df.empty:
            print(f"ERROR: no rows after restricting to opponents {sorted(wanted)}", file=sys.stderr)
            return 2

    k = _resolve_k(df)
    print(f"\nCSV: {args.csv}")
    print(f"rows: {len(df)}    K={k}    opponents: {sorted(df[opp_col].unique().tolist())}\n")

    # Prefer per-step occupancy; fall back to dominant-z if not present.
    occ = _occupancy_per_opp_from_occupancy_cols(df, opp_col, k)
    occ_src = "strategy_occupancy_* (per-step)"
    if occ.empty:
        occ = _occupancy_per_opp_from_dominant(df, opp_col, k)
        occ_src = "strategy_dominant (episode-dominant)"
    if occ.empty:
        print("ERROR: no z information found in CSV", file=sys.stderr)
        return 2

    print(f"P(z | opp) -- source: {occ_src}")
    print(occ.round(3).to_string())

    marginal = _marginal_z(occ)
    print(f"\nmarginal P(z) (unweighted avg over opponents): "
          f"[{', '.join(f'{p:.3f}' for p in marginal)}]")
    H_marginal = -float(np.sum(np.clip(marginal, 1e-12, 1.0) * np.log(np.clip(marginal, 1e-12, 1.0))))
    H_max = math.log(k)
    print(f"H(marginal P(z)) = {H_marginal:.4f} nats   (max ln K = {H_max:.4f}, frac = {H_marginal/H_max:.3f})")

    kl_rows = []
    for opp in occ.index:
        p = occ.loc[opp].values.astype(np.float64)
        kl_to_marg = _kl(p, marginal)
        kl_rows.append((opp, kl_to_marg))
    kl_df = pd.DataFrame(kl_rows, columns=["opp", "KL(P(z|opp) || P(z))"]).set_index("opp")
    print("\nPer-opponent divergence from marginal:")
    print(kl_df.round(4).to_string())

    kl_max = float(kl_df.iloc[:, 0].max())
    kl_mean = float(kl_df.iloc[:, 0].mean())
    print(f"\nKL summary: max={kl_max:.4f}   mean={kl_mean:.4f}")

    wr = _per_opp_z_winrate(df, opp_col, k)
    if not wr.empty:
        print("\nWR by opponent x z (dominant-z, episode level):")
        print(wr.round(3).to_string())
        # Oracle bound
        if wr.notna().any().any():
            print("\nOracle bound (best fixed z per opponent on this eval):")
            for opp in wr.index:
                row = wr.loc[opp]
                if row.notna().any():
                    z_best = int(row.idxmax())
                    wr_best = float(row.max())
                    # q_phi-routed = the realized WR weighted by P(z|opp)
                    p_z = occ.loc[opp].values.astype(np.float64)
                    realized = float(np.nansum(p_z * row.fillna(0.0).values))
                    print(f"  {opp:>12s}: q_phi WR={realized:.3f}   best z={z_best} -> {wr_best:.3f}   gap={wr_best-realized:+.3f}")

    verdict = _verdict(kl_max, kl_mean, H_marginal / H_max)
    print(f"\nVERDICT: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
