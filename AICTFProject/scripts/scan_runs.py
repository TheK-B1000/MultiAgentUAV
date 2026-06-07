import os
import glob
import pandas as pd
import numpy as np

csv_files = glob.glob(r"checkpoints\4v4\*_metrics.csv")
results = []

for f in csv_files:
    basename = os.path.basename(f)
    try:
        df = pd.read_csv(f)
        if df.empty:
            continue
        last_row = df.iloc[-1]
        steps = last_row.get("timesteps", np.nan)
        kl = last_row.get("policy_z_sensitivity_KL", np.nan)
        jsd = last_row.get("latent_actor_z_separation_jsd", np.nan)
        sep_loss = last_row.get("latent_actor_z_separation_loss", np.nan)
        mi_opp = last_row.get("latent_mi_z_opponent_nats", np.nan)
        mi_phase = last_row.get("latent_mi_z_phase_nats", np.nan)
        wr_spread = last_row.get("strategy_wr_spread", np.nan)
        
        results.append({
            "run": basename.replace("_metrics.csv", ""),
            "steps": steps,
            "policy_KL": kl,
            "separation_JSD": jsd,
            "sep_loss": sep_loss,
            "MI(z;opp)": mi_opp,
            "MI(z;phase)": mi_phase,
            "wr_spread": wr_spread
        })
    except Exception as e:
        print(f"Error reading {basename}: {e}")

res_df = pd.DataFrame(results)
print(res_df.to_string(index=False))
