# Parameter–specialization tradeoff (2v2 & 4v4)

Paste-ready for professor. Δ cells use sealed/flagged crossover means [LCB95, UCB95]
(paired percentile bootstrap, n_boot=20000, α=0.05, rng=7).
Imitation = holdout argmax z0↔π_A / z1↔π_B. Reduction vs Share-0 (2v2) or Separated (4v4).

Δ_A = V(z0,A) − V(z1,A); Δ_B = V(z1,B) − V(z0,B).

## 2v2

| Condition | Params (M) | Reduction (%) | Imitation agreement | ΔA | ΔB |
|---|---:|---:|---:|---|---|
| Share-0 | 6.947 | 0 | 1.000 / 1.000 | +0.289 [+0.164, +0.406] | +0.258 [+0.141, +0.375] |
| Share-Encoder | 3.613 | 48.0 | 0.967 / 0.989 | +0.266 [+0.148, +0.383] | +0.164 [+0.047, +0.281] |
| Share-Backbone | 3.509 | 49.5 | 0.963 / 0.985 | +0.148 [+0.023, +0.266] | +0.141 [+0.016, +0.266] |
| Share-Macro | 3.508 | 49.5 | 0.936 / 0.969 | +0.125 [0.000, +0.250] **FAIL** | +0.156 [+0.031, +0.281] |
| Fully Shared+z | 3.457 | 50.2 | 0.965 / 0.988 | — (eval pending) | — (eval pending) |

## 4v4 (CLOSEST_DEFENDS k=2)

| Condition | Params (M) | Reduction (%) | Imitation agreement | ΔA | ΔB |
|---|---:|---:|---:|---|---|
| Separated (≈ Share-0) | 6.929 | 0 | — (not distilled) | +0.156 [+0.039, +0.273] | +0.406 [+0.297, +0.508] |
| Share-Encoder | 3.637 | 47.5 | 0.914 / 0.876 | +0.359 [+0.188, +0.516] | −0.219 [−0.344, −0.109] **FLAG** |
| Share-Backbone | — | — | — | not built | not built |
| Share-Macro | — | — | — | not built | not built |
| Fully Shared+z | 3.469 | 49.9 | 0.895 / 0.856 | +0.281 [+0.125, +0.438] | −0.109 [−0.219, 0.000] **FLAG** |

**Notes**
- Share-Macro 2v2 fails because LCB95(Δ_A)=0 (joint gate).
- 4v4 Backbone/Macro are deliberately out of the cross-scale suite (2v2 depth only).
- 4v4 Separated Δ is confirmatory **n=128**. Share-Encoder / Fully Shared+z Δ are exploratory **n=64** under `SUITE_SHARING_4V4_CROSSOVER_EVAL_SPEC`; both wrote `INTEGRITY_REQUIRED` (Δ_B ≤ 0) — no `FROZEN_RESULT` seal. Confirmatory n=128 not authorized.
- Reading: both distilled arms keep strong Pole-A specialization; both reverse on Pole B (z0 beats z1 on B, so Δ_B = V(z1,B)−V(z0,B) is negative). Separated still passes the joint gate.
