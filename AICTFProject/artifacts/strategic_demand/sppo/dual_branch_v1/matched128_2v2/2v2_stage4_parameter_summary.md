# 2v2 Stage-4 parameter table + Stage-4 Top-50 diagnostic

| Condition | Params (M) | Reduction | Agreement A / B | Δ_A | Δ_B |
|---|---|---|---|---|---|
| Ours | 13.86 | 0.0% | N/A | +0.32 [+0.14, +0.48] | +0.66 [+0.52, +0.78] |
| Share-Encoder | 3.64 | 73.8% | 0.909 / 0.893 | +0.22 [+0.06, +0.38] | +0.40 [+0.22, +0.56] |
| Fully Shared+z+r | 3.47 | 75.0% | 0.897 / 0.876 | +0.28 [+0.08, +0.48] | +0.40 [+0.26, +0.54] |
| Role-only (r) | 3.46 | 75.0% | 0.776 / 0.630 | N/A | N/A |

Role-only (r): wins 0.66 on Pole A, 0.02 on Pole B.
No-Role Specialists: 6.93M (50.0% vs Ours).
Seeds: 27 score-2 seeds + 23 score-1 seeds chosen by the pre-specified ascending-seed-ID tie-break; identical for every row.
