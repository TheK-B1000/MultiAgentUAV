"""Quick check of the v3i15 test metrics CSV."""
import csv
import os
import sys

csv_path = os.path.join(
    "checkpoints", "4v4",
    "latent_v3i15_strong_separation_50k_test_4v4_metrics.csv",
)

if not os.path.exists(csv_path):
    print("CSV not yet created")
    sys.exit(0)

with open(csv_path, "r") as f:
    reader = list(csv.DictReader(f))

print(f"Updates logged: {len(reader)}")
if not reader:
    print("No data rows yet")
    sys.exit(0)

# Key metrics to watch
keys_of_interest = [
    "global_step",
    "policy_z_sensitivity_KL",
    "latent_actor_z_separation_jsd",
    "latent_actor_z_separation_coef",
    "latent_actor_z_adapter_scale",
    "latent_actor_z_separation_loss",
    "latent_actor_z_separation_active",
    "z_entropy",
    "z_wr_spread",
    "MI_z_outcome",
    "MI_z_phase",
    "MI_z_flag",
    "latent_behavior_contrast_loss",
    "latent_behavior_contrast_coef",
]

available_keys = [k for k in keys_of_interest if k in reader[0]]

# Print compact table
header = "  ".join(f"{k[:25]:>25}" for k in available_keys)
print(header)
print("-" * len(header))

for row in reader:
    vals = []
    for k in available_keys:
        v = row.get(k, "n/a")
        try:
            v = f"{float(v):.6f}"
        except (ValueError, TypeError):
            pass
        vals.append(f"{v:>25}")
    print("  ".join(vals))
