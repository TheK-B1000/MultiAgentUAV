#!/usr/bin/env python3
"""Compute training-time diagnostic dashboard metrics for a latent strategy PPO run."""

from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import pandas as pd

LN2 = math.log(2.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute real-time diagnostic dashboard metrics for a latent training run.")
    parser.add_argument(
        "--run-tag",
        type=str,
        required=True,
        help="Training run tag (e.g. plan_faithful_latent_phaseaux_005_hardpool_1m_2v2)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=os.path.join("checkpoints", "2v2"),
        help="Checkpoint directory containing metrics CSV.",
    )
    args = parser.parse_args()

    metrics_path = os.path.join(args.checkpoint_dir, f"{args.run_tag}_metrics.csv")
    if not os.path.isfile(metrics_path):
        print(f"Error: Metrics file not found at {metrics_path}")
        sys.exit(1)

    df = pd.read_csv(metrics_path)
    if df.empty:
        print("Error: Metrics file is empty.")
        sys.exit(1)

    # Check for required columns
    required = ["timesteps", "latent_mi_z_outcome_nats", "strategy_entropy_frac"]
    for col in required:
        if col not in df.columns:
            print(f"Error: Required column '{col}' is missing.")
            sys.exit(1)

    # Extract parameters and convert to bits
    df = df.copy()
    df["MI_outcome_bits"] = df["latent_mi_z_outcome_nats"] / LN2
    df["entropy_frac"] = df["strategy_entropy_frac"]

    # Calculate behavior MIs in bits if present
    behavior_cols = {
        "role": "latent_mi_z_role_bucket_nats",
        "spread": "latent_mi_z_spread_bucket_nats",
        "adr": "latent_mi_z_attack_defense_ratio_bucket_nats",
        "pressure": "latent_mi_z_pressure_bucket_nats",
    }
    for label, col in behavior_cols.items():
        if col in df.columns:
            df[f"MI_{label}_bits"] = df[col] / LN2
        else:
            df[f"MI_{label}_bits"] = 0.0

    df["coupling_index"] = (
        df["MI_role_bits"] + df["MI_spread_bits"] + df["MI_adr_bits"] + df["MI_pressure_bits"]
    ) / 4.0

    # Smooth metrics to avoid noise (rolling average of 3 updates)
    window = min(3, len(df))
    df["MI_outcome_smooth"] = df["MI_outcome_bits"].rolling(window=window, min_periods=1).mean()
    df["entropy_frac_smooth"] = df["entropy_frac"].rolling(window=window, min_periods=1).mean()

    latest = df.iloc[-1]
    latest_step = int(latest["timesteps"])
    episodes = int(latest.get("episodes_completed", 0))
    rolling_wr = latest.get("rolling_win_rate_200ep", float("nan"))
    if not np.isfinite(rolling_wr):
        rolling_wr = latest.get("rolling_win_rate_50ep", float("nan"))

    # Calculate overall rates (from first step to last)
    first = df.iloc[0]
    total_steps_delta = latest_step - int(first["timesteps"])
    if total_steps_delta > 0:
        overall_outcome_rate = (
            (latest["MI_outcome_smooth"] - first["MI_outcome_smooth"])
            / total_steps_delta
            * 100_000
        )
        overall_entropy_decay = (
            (first["entropy_frac_smooth"] - latest["entropy_frac_smooth"])
            / total_steps_delta
            * 100_000
        )
    else:
        overall_outcome_rate = 0.0
        overall_entropy_decay = 0.0

    # Calculate recent rates (over last ~200k steps)
    recent_target = max(0, latest_step - 200_000)
    sub = df[df["timesteps"] >= recent_target]
    if len(sub) > 1:
        recent_first = sub.iloc[0]
        recent_steps_delta = latest_step - int(recent_first["timesteps"])
        if recent_steps_delta > 0:
            recent_outcome_rate = (
                (latest["MI_outcome_smooth"] - recent_first["MI_outcome_smooth"])
                / recent_steps_delta
                * 100_000
            )
            recent_entropy_decay = (
                (recent_first["entropy_frac_smooth"] - latest["entropy_frac_smooth"])
                / recent_steps_delta
                * 100_000
            )
        else:
            recent_outcome_rate = 0.0
            recent_entropy_decay = 0.0
    else:
        recent_outcome_rate = overall_outcome_rate
        recent_entropy_decay = overall_entropy_decay

    # Entropy status
    ent = latest["entropy_frac"]
    if ent < 0.20:
        status = "COLLAPSED"
    elif ent < 0.40:
        status = "DECAYING_LOW"
    elif ent < 0.95:
        status = "ACTIVE_DIVERSE"
    else:
        status = "UNIFORM / NO_PRESSURE"

    print("=" * 80)
    print(f"DIAGNOSTIC DASHBOARD FOR RUN: {args.run_tag}")
    print("=" * 80)
    print(f"Latest Step: {latest_step:,} | Completed Episodes: {episodes:,}")
    if np.isfinite(rolling_wr):
        print(f"Rolling Win Rate: {rolling_wr * 100.0:.1f}%")
    else:
        print("Rolling Win Rate: N/A")
    print()

    print("1. OUTCOME SIGNAL (MI_outcome)")
    print(f"   Latest Value:  {latest['MI_outcome_bits']:.4f} bits")
    print(f"   Recent Rate:   {recent_outcome_rate:+.4f} bits per 100k steps (recent window)")
    print(f"   Overall Rate:  {overall_outcome_rate:+.4f} bits per 100k steps")
    print()

    print("2. LATENT ENTROPY DECAY")
    print(
        f"   Latest Fraction: {latest['entropy_frac']:.4f} (Raw: {latest.get('strategy_entropy', 0.0):.4f} nats)"
    )
    print(f"   Recent Decay:    {recent_entropy_decay:+.4f} per 100k steps")
    print(f"   Overall Decay:   {overall_entropy_decay:+.4f} per 100k steps")
    print(f"   Status:          {status}")
    print()

    print("3. CAUSAL COUPLING INDEX")
    print(f"   Latest Value: {latest['coupling_index']:.4f} bits")
    print("   Behavior Mutual Information Breakdown:")
    print(f"     - Role MI:    {latest['MI_role_bits']:.4f} bits")
    print(f"     - Spread MI:  {latest['MI_spread_bits']:.4f} bits")
    print(f"     - ADR MI:     {latest['MI_adr_bits']:.4f} bits")
    print(f"     - Pressure:   {latest['MI_pressure_bits']:.4f} bits")
    print("=" * 80)


if __name__ == "__main__":
    main()
