import pandas as pd, os
d = r'K:\MultiAgentUAV\AICTFProject\checkpoints\4v4'
runs = [
  'plan_faithful_no_latent_hardpool_1m_4v4',
  'plan_faithful_latent_phase1_coupling_hardpool_1m_4v4',
  'plan_faithful_latent_phase2_credit_hardpool_1m_4v4',
  'plan_faithful_latent_phase3b_outcome_clean_hardpool_1m_4v4',
  'plan_faithful_latent_phase3b_ablate_k1_hardpool_1m_4v4',
  'plan_faithful_latent_phase3b_ablate_no_persistence_hardpool_1m_4v4',
  'plan_faithful_latent_phase4a_rescue_hardpool_1m_4v4',
  'plan_faithful_latent_episode_strategic_seed0_toughpool_ctx170_warmup5_lamH_anneal3to05_200k700k_v3a_1m_4v4',
  'latent_v3b_episodecredit_marginalbaseline_warmup5_lamHanneal_1m_4v4',
  'latent_v3c_routerlr_epochs6_lr5e3_lamHfloor1e3_1m_4v4',
  'latent_v3d_smartrouter_bucketopp_ema09_min8_1m_4v4',
  'latent_v3d_delayedanneal_300k_800k_bucketopp_1m_4v4',
]
hdr = ['run','WR_last20','zH/lnK','WR_spread','MI(z;opp)','MI(z;phase)','qphi_grad','z_pi_loss','z_ratio_std','unique_z','resamp%','persist_loss']
print('  '.join(f'{h:<22}' if i==0 else f'{h:>12}' for i,h in enumerate(hdr)))
print('-'*220)
for r in runs:
  p = os.path.join(d, r+'_metrics.csv')
  if not os.path.isfile(p):
    print(r, 'MISSING'); continue
  df = pd.read_csv(p, low_memory=False)
  n = len(df); tail = df.tail(max(1, n//5))
  def m(c):
    if c not in df.columns: return float('nan')
    s = pd.to_numeric(tail[c], errors='coerce')
    return float(s.mean())
  def get_metric(*cols):
    """Return tail mean of the first column with a non-zero finite value.

    Order columns by priority: arc-credit > episode-credit > per-step
    strategy-PPO so v3i19+ / v4i1 / v4i3 runs do not silently report 0.0
    for q_phi just because the per-step strategy-PPO path is off.
    """
    for col in cols:
      val = m(col)
      if not pd.isna(val) and val != 0.0:
        return val
    return m(cols[-1])
  row = [r[:38], m('episode_win_rate'), m('strategy_entropy_frac'), m('strategy_wr_spread'),
         m('latent_mi_z_opponent_nats'), m('latent_mi_z_phase_nats'),
         get_metric('q_phi_strategy_encoder_grad_norm', 'episode_credit_grad_norm', 'strategy_grad_norm'),
         get_metric('latent_arc_policy_loss', 'latent_episode_pg_loss', 'strategy_policy_loss'),
         get_metric('latent_arc_clipfrac', 'latent_episode_ratio_std', 'strategy_ratio_std'),
         m('strategy_unique_count'), m('strategy_resample_fraction'), m('strategy_persist_loss')]
  out = []
  for i,v in enumerate(row):
    if i==0:
      out.append(f'{v:<22}')
    else:
      try:
        out.append(f'{float(v):>12.4f}')
      except Exception:
        out.append(f'{str(v):>12}')
  print('  '.join(out))
