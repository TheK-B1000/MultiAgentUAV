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
  row = [r[:38], m('episode_win_rate'), m('strategy_entropy_frac'), m('strategy_wr_spread'),
         m('latent_mi_z_opponent_nats'), m('latent_mi_z_phase_nats'),
         m('strategy_grad_norm'), m('strategy_policy_loss'), m('strategy_ratio_std'),
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
