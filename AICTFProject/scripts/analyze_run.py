import pandas as pd
import numpy as np

# Load metrics CSV
metrics_path = r"k:\MultiAgentUAV\AICTFProject\checkpoints\4v4\latent_v3i14_tuned_tactical_specialist_pool_1m_4v4_metrics.csv"
df = pd.read_csv(metrics_path)

print("Columns in metrics:")
print(df.columns.tolist()[:30])  # print first 30 columns

print("\nLast 5 rows of selected metrics:")
selected_cols = [
    "timesteps",
    "policy_z_sensitivity_KL",
    "latent_actor_z_separation_jsd",
    "latent_actor_z_separation_loss",
    "latent_specialist_marginal_entropy",
    "latent_specialist_conditional_entropy",
    "latent_specialist_context_mi",
    "latent_specialist_active_buckets",
    "latent_mi_z_opponent_nats",
    "latent_mi_z_phase_nats",
    "latent_mi_z_outcome_nats",
    "latent_mi_z_flag_state_nats",
    "forced_z_macro_jsd_mean",
]

# filter existing cols
selected_cols = [c for c in selected_cols if c in df.columns]
print(df[selected_cols].tail(10).to_string())

print("\nMean of selected metrics in the last 200k steps:")
last_200k = df[df["timesteps"] >= 800000]
if not last_200k.empty:
    print(last_200k[selected_cols].mean().to_string())
else:
    print("No data for last 200k steps.")

print("\nMaximum values reached during the run:")
print(df[selected_cols].max().to_string())
