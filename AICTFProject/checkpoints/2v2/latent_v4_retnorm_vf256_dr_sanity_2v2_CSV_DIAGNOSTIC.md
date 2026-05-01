# CSV diagnostic: `latent_v4_retnorm_vf256_dr_sanity_2v2` (default DR knobs)

Generated from `*_metrics.csv` rows (one row per PPO update). Compare with `latent_fix_v4_retnorm_vf256_2v2_metrics.csv` over the **same number of updates** (13 updates ≈ 212k env steps).

## Strategy / q_phi

| Metric | DR sanity | non-DR v4 (same 13 updates) |
|--------|-----------|------------------------------|
| `strategy_entropy` mean | **1.286** | 1.321 |
| `strategy_entropy` max | 1.368 | 1.371 |
| ln(4) ≈ uniform | **1.386** | 1.386 |

**Read:** DR run is **not** sitting at uniform `q_phi` (entropy **below** ln(4), similar band to non-DR). Not the “washed out / no specialization incentive” mode where entropy ≈ ln(4) throughout.

**Last-update `strategy_occupancy`:** DR `(0.17, 0.33, 0.39, 0.11)` vs non-DR `(0.35, 0.17, 0.20, 0.27)` — different mix, still spread.

**Last-update `episode_z_*_win_rate`:** DR `[0.51, 0.56, 0.48, 0.50]` (spread **0.075**); non-DR `[0.50, 0.54, 0.54, 0.53]` (spread **0.048**). Per-strategy WRs not equal; DR shows slightly **more** spread across z, not collapse to one strategy.

## Critic / value

| Metric | DR sanity | non-DR v4 (13 upd) |
|--------|-----------|---------------------|
| `explained_variance` last | **0.065** | 0.064 |
| `explained_variance` max | 0.065 | 0.068 |
| `value_loss` last | **0.899** | 0.912 |

**Read:** EV is **in the same ballpark** as non-DR over the **same early horizon** — not collapsed vs v4 1M tail (~0.13). Value loss is **similar**, not wildly higher or oscillating in this CSV.

## Win rate (cumulative in metrics)

| | DR | non-DR v4 (13 upd) |
|--|-----|---------------------|
| First `win_rate` | 51.24% | 48.74% |
| Last `win_rate` | 51.50% | 50.24% |
| Delta | **+0.26 pts** | **+1.50 pts** |

Episode-log WR (training stdout) showed ~50.6% → ~51.6% over ~11k episodes; cumulative columns here are consistent with **very flat** learning vs non-DR on the same update count.

## Rolling win rate (`rolling_win_rate_50ep`, `rolling_win_rate_200ep`)

From the same CSV rows (13 updates). **Caveat:** rolling columns are over the **last N completed episodes globally**; with only ~11k episodes and non-stationarity, **`rolling_win_rate_50ep` can swing sharply** (including downward) even when cumulative WR or stdout episode tallies trend up.

| Run | `rolling_win_rate_50ep` first → last (Δ) | `rolling_win_rate_200ep` first → last (Δ) |
|-----|----------------------------------------|-------------------------------------------|
| Latent + default DR | 0.56 → 0.54 (**−0.02**) | 0.52 → 0.49 (**−0.03**) |
| **No-latent + default DR** (`custom_v4hp_dr_sanity_2v2`) | 0.56 → 0.34 (**−0.22**) | 0.535 → 0.49 (**−0.045**) |
| non-DR v4 (13 upd) | 0.52 → 0.44 (**−0.08**) | 0.52 → 0.52 (**0.00**) |

**Read:** Do **not** over-interpret `rolling_win_rate_50ep` in isolation on this short horizon; pair with **cumulative `win_rate`**, **stdout episode blocks**, and (after reruns) **`strategy_resample_adv_*`**. No-latent stdout showed **~51.4% → ~53.6%** (~+2.2 pts) on the same DR defaults—clearly healthier than latent stdout **~50.6% → ~51.6%** (~+1 pt). That disambiguation is the operative result (methodology **2×2**: latent × DR interaction).

## Resolution: no-latent + default DR (May 2025)

`custom_v4hp_dr_sanity_2v2` (200k, return norm, no VF clip, VF hidden 256, no latent): **completed OK**; episode WR slope materially **better** than `latent_v4_retnorm_vf256_dr_sanity_2v2` under the **same DR knobs**. So **default DR is not globally prohibitive**; the weak latent+DR curve is **not** explained by “knobs too harsh for everyone.”

**Next run (per protocol):** latent + **default** DR **200k** with code that logs **`strategy_resample_adv_mean_z*`** / **`strategy_resample_adv_std_z*`** / **`strategy_resample_adv_n_z*`** in `*_metrics.csv` (rerun; old CSV lacks these columns).

## Conclusion for next steps

1. **CSV checks do not** point to uniform-q_phi washout or a critic disaster vs non-DR on this horizon.
2. **No-latent DR** confirms **latent × DR interaction**; halving knobs is a possible axis but **does not replace** telemetry for *why* default DR hurts latent.
3. Do **not** commit to 1M at default DR for flat latent+DR; use telemetry run + (optional) halved-knob latent on second worker if desired.
4. Prefer **methodology rolling/cumulative guidance** (`docs/METHODOLOGY_DOMAIN_RANDOMIZATION.md`) when judging slope.
