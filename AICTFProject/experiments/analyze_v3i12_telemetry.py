#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np

CSV_PATH = r"k:\MultiAgentUAV\AICTFProject\checkpoints\4v4\latent_v3i12_faithful_z_pressure_pool_1m_4v4_e3_steps.csv"

BEHAVIOR_TELEMETRY_NAMES = [
    "team_spread",
    "num_attackers",
    "num_defenders",
    "num_go_to",
    "carrier_escort_count",
    "nearest_blue_to_carrier",
    "nearest_blue_to_enemy_carrier",
    "n_intercept_near_enemy_carrier",
    "avg_blue_to_enemy_flag",
    "avg_blue_to_own_flag",
    "intercept_pressure",
    "defense_pressure",
    "attack_defense_ratio",
]

BUCKET_COLS = [
    "spread_bucket",
    "role_bucket",
    "pressure_bucket",
    "attack_defense_ratio_bucket"
]

def main():
    if not os.path.isfile(CSV_PATH):
        print(f"Error: CSV file not found at {CSV_PATH}")
        return

    print(f"Starting chunked processing of {CSV_PATH}...")
    chunksize = 500_000

    # Accumulators
    # Grouped by z: sum and count for numeric columns
    sums_by_z = {z: {col: 0.0 for col in BEHAVIOR_TELEMETRY_NAMES} for z in range(4)}
    counts_by_z = {z: 0 for z in range(4)}

    # Grouped by z: count distribution for each bucket column
    # e.g., bucket_counts_by_z[z][col][val] = count
    bucket_counts_by_z = {
        z: {
            col: {} for col in BUCKET_COLS
        } for z in range(4)
    }

    total_rows = 0

    # Read column header first
    header = pd.read_csv(CSV_PATH, nrows=0)
    cols_to_use = ["z_t"] + BEHAVIOR_TELEMETRY_NAMES + BUCKET_COLS
    
    # Check if all columns are in header
    missing_cols = [c for c in cols_to_use if c not in header.columns]
    if missing_cols:
        print(f"Warning: Columns missing from CSV: {missing_cols}")
        cols_to_use = [c for c in cols_to_use if c in header.columns]
        
    for chunk in pd.read_csv(CSV_PATH, usecols=cols_to_use, chunksize=chunksize):
        total_rows += len(chunk)
        chunk = chunk.dropna(subset=["z_t"])
        chunk["z_t"] = chunk["z_t"].astype(float).astype(int)
        
        # Filter z_t in [0, 1, 2, 3]
        chunk = chunk[chunk["z_t"].isin([0, 1, 2, 3])]
        if chunk.empty:
            continue
            
        for z in range(4):
            z_mask = chunk["z_t"] == z
            z_df = chunk[z_mask]
            z_count = len(z_df)
            if z_count == 0:
                continue
                
            counts_by_z[z] += z_count
            
            # Numeric sums
            for col in BEHAVIOR_TELEMETRY_NAMES:
                if col in z_df.columns:
                    sums_by_z[z][col] += float(z_df[col].sum())
                    
            # Bucket counts
            for col in BUCKET_COLS:
                if col in z_df.columns:
                    vc = z_df[col].value_counts()
                    for val, count in vc.items():
                        val_int = int(float(val))
                        bucket_counts_by_z[z][col][val_int] = bucket_counts_by_z[z][col].get(val_int, 0) + int(count)

        print(f"Processed {total_rows:,} rows...")

    # Calculate final averages and print results
    print("\n========================================================")
    print("      E3 Step Telemetry Analysis by Latent z (v3i12)")
    print("========================================================\n")
    
    total_valid = sum(counts_by_z.values())
    print(f"Total rows analyzed: {total_rows:,}")
    print(f"Valid z rows: {total_valid:,}\n")
    
    for z in range(4):
        cnt = counts_by_z[z]
        pct = (cnt / total_valid * 100) if total_valid > 0 else 0
        print(f"z = {z}: {cnt:,} steps ({pct:.2f}%)")
    print()

    # Behavior Averages
    print("--- Behavior Telemetry Averages ---")
    header_str = f"{'Metric':<32} | {'z=0':<8} | {'z=1':<8} | {'z=2':<8} | {'z=3':<8}"
    print(header_str)
    print("-" * len(header_str))
    
    for col in BEHAVIOR_TELEMETRY_NAMES:
        means = []
        for z in range(4):
            cnt = counts_by_z[z]
            if cnt > 0:
                means.append(sums_by_z[z][col] / cnt)
            else:
                means.append(float('nan'))
        print(f"{col:<32} | {means[0]:.4f}   | {means[1]:.4f}   | {means[2]:.4f}   | {means[3]:.4f}")
    print("\n")

    # Discrete Bucket Distributions
    print("--- Discrete Buckets Distributions (%) ---")
    for col in BUCKET_COLS:
        print(f"\n{col.upper()}:")
        # Find all unique values observed for this bucket
        all_vals = sorted(list(set(
            val for z in range(4) for val in bucket_counts_by_z[z][col].keys()
        )))
        
        # Print subheader
        val_headers = " | ".join(f"Val {v}" for v in all_vals)
        print(f"{'z':<4} | {val_headers}")
        print("-" * (6 + len(val_headers)))
        
        for z in range(4):
            z_cnt = counts_by_z[z]
            pcts = []
            for val in all_vals:
                c = bucket_counts_by_z[z][col].get(val, 0)
                p = (c / z_cnt * 100) if z_cnt > 0 else 0.0
                pcts.append(f"{p:.1f}%")
            print(f"{z:<4} | " + " | ".join(f"{p:<5}" for p in pcts))
            
if __name__ == "__main__":
    main()
