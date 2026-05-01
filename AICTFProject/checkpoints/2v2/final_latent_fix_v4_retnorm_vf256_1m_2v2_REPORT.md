# Run summary: `final_latent_fix_v4_retnorm_vf256_1m_2v2.zip`

Latent v4 stack (return normalization, `clip_range_vf=None`, `latent_vf_hidden=256`, entropy objective **minimize**, λ_H=0.001, λ_p=0.05), fixed scripted **OP3**, 2v2, **train** maps during learning. Training resumed from `final_latent_fix_v4_retnorm_vf256_2v2.zip` (~213k env steps) to **1M** total env steps.

**What this result is and isn’t.** Latent-strategy PPO trained on fixed-opponent CTF in sim achieves **~77% WR** on held-out **eval** maps against the **training opponent (OP3)** under the greedy `eval_checkpoint.py` protocol (100 episodes per cell, see table below). That validates the training methodology and latent specialization in this environment; it does **not** characterize transfer to varied opponents, domain-randomized or perturbed dynamics, or real platforms. Robustness curves, sim-to-sim transfer, and hardware-relevant execution models are required before drawing transfer-relevant conclusions.

---

## Final training stats (end of `train_ppo` run)

```text
policy_loss=-0.0416, value_loss=0.8386, approx_kl=0.01067
```

---

## Win-rate trajectory (training telemetry)

Logged **rolling / cumulative** win rate vs OP3 reached about **61.5%** late in the 1M run (episode-log snapshots in training log).

Per-update **cumulative** `win_rate` from `latent_fix_v4_retnorm_vf256_1m_2v2_metrics.csv` (rows are **post-resume** optimizer updates only; timesteps ~0.23M → ~1.02M):

| Timesteps (approx.) | Cumulative WR |
|--------------------:|--------------:|
| 229 376 | 50.6% |
| 622 592 | 57.0% |
| 1 015 808 | 61.5% |

---

## Explained-variance trajectory (critic vs rollout targets)

`explained_variance` from the same metrics CSV (same post-resume window):

| Segment | Value |
|---------|------:|
| First row in CSV | 0.057 |
| Mid | 0.089 |
| Last | 0.132 |
| Max in CSV | **0.140** |

---

## Supervised ceiling (`tools/critic_ceiling.py`)

One frozen-policy rollout on **train** maps, **global_state_dim=14** → `return`, 25% hold-out, seed **123**.

| Metric | Value |
|--------|------:|
| `checkpoint_critic_ev_on_collected_rollout` | **0.128** |
| Best held-out **test R²** (HistGradientBoosting) | **0.253** |
| Ridge test R² | 0.100 |
| Random forest test R² | 0.233 |

Interpretation: still low-to-moderate predictability from 14-d global state alone; critic EV on the same batch is in-family with the structural picture (noisy advantages, not a “broken” trainer).

---

## Holdout evaluation (greedy / deterministic, 100 eps each)

`plot/eval_checkpoint.py` on this checkpoint; per-episode CSVs under `csv/`, aggregate: `csv/eval_latent_v4_retnorm_vf256_1m_2v2_aggregate.csv`.

| Map set | Opponent | Wins | Losses | Draws | WR |
|---------|----------|-----:|-------:|------:|---:|
| train | OP3 | 77 | 22 | 1 | **77%** |
| train | OP4 | 88 | 12 | 0 | **88%** |
| **eval** | **OP3** | 77 | 22 | 1 | **77%** |
| **eval** | **OP4** | 84 | 16 | 0 | **84%** |

**Generalization:** **eval-map WR matches train-map WR for OP3 (77%)** in this run; OP4 eval-map WR is slightly lower than train-map (84% vs 88%), still strong on n=100.

---

## Artifacts

| File | Role |
|------|------|
| `final_latent_fix_v4_retnorm_vf256_1m_2v2.zip` | Checkpoint |
| `latent_fix_v4_retnorm_vf256_1m_2v2_metrics.csv` | Per-update training metrics |
| `latent_fix_v4_retnorm_vf256_1m_2v2_episodes.csv` | Per-episode training log |
| `final_latent_fix_v4_retnorm_vf256_1m_2v2_training_curves.png` | WR + EV vs timesteps (post-resume segment) |
| `csv/eval_latent_v4_retnorm_vf256_1m_2v2_*.csv` | Eval rollouts |
| This file | One-page run summary |

---

*Generated for quick handoff; refine numbers if you change eval seeds or episode counts.*
