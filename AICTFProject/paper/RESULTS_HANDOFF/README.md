# Results handoff — 2v2 + 6v6

All numbers below are read directly from sealed artifacts by
`paper/figures/harvest_results_data.py`. Nothing is retyped by hand.
Re-run that script to refresh `data/RESULTS_DATA_2v2_6v6.json` after any new
evaluation seals.

**6v6 Share-Encoder crossover and Share-0 teacher diagnostic are both sealed
(FAIL). Robustness remains reserved/unspent.**

---

## 1. The criterion (identical at both scales)

```
Δ_A = V(z0,A) − V(z1,A)
Δ_B = V(z1,B) − V(z0,B)
PASS iff both means > 0 AND both LCB95 > 0
paired percentile bootstrap, n_boot=20000, α=0.05, rng_seed=7, seed = resampling unit
```

## 2. Compression is numerically identical at both scales

The actor is per-agent weight-shared, so team size changes the observation batch
dimension, not any parameter shape. Verified programmatically
(`identical_at_2v2_and_6v6: true`).

| quantity | value |
|---|---|
| one specialist branch | 3,473,592 |
| two independent experts (Share-0) | 6,947,184 |
| shared-encoder student (Rung-1) | 3,612,784 |
| shared module (`actor_cnn`) | 3,334,400 |
| private per branch | 139,192 |
| **reduction** | **48.00%** |
| shared fraction of one branch | 96.0% |

## 3. Opponent-regime demand — certified at both scales

| | 2v2 (n=192) | 6v6 (n=64) |
|---|---|---|
| Δ_guard(A) | +0.29688 [+0.20313, +0.39063] | +0.2656 [+0.0938, +0.4375] |
| Δ_breach(B) | +0.45313 [+0.38542, +0.52604] | +0.6250 [+0.4844, +0.7500] |
| guard WR on A | 0.64063 | 0.75000 |
| breach WR on A | 0.34375 | 0.48438 |
| guard WR on B | 0.00521 | 0.07813 |
| breach WR on B | 0.45833 | 0.70313 |
| concealment p_C(A) | 0.94271 | — |
| concealment p_C(B) | 0.91667 | — |
| verdict | V3_STRATEGIC_DEMAND_VALIDATED | CERTIFIED |

6v6 poles are size-normalized (`min_alive_for_defender = 6`); guard distributes
3 defenders (indices 3,4,5) rather than stacking.

**Key point:** demand does *not* weaken with team size — the Regime-B contrast is
larger at 6v6 (+0.625 vs +0.453).

## 4. 2v2 — experts and generalist

| | Δ_A | Δ_B | pass |
|---|---|---|---|
| independent experts (n=64) | +0.1719 [+0.0156, +0.3281] | +0.2969 [+0.1250, +0.4688] | yes |

Generalist π_G: V(π_G,A) = 0.375 [0.265, 0.500], V(π_G,B) = 0.500 [0.375, 0.625].

## 5. 2v2 — sharing ladder (n=128 matched)

| condition | params (M) | red. | Δ_A | Δ_B | criterion |
|---|---|---|---|---|---|
| Share-0 | 6.947 | 0% | +0.2891 [+0.1641, +0.4062] | +0.2578 [+0.1406, +0.3750] | reference |
| Share-Encoder | 3.613 | **48.0%** | +0.2656 [+0.1484, +0.3828] | +0.1641 [+0.0469, +0.2812] | PASS |
| Share-Backbone | 3.509 | 49.5% | +0.1484 [+0.0234, +0.2656] | +0.1406 [+0.0156, +0.2656] | PASS |
| Share-Macro | 3.508 | 49.5% | +0.1250 [+0.0000, +0.2500] | +0.1562 [+0.0312, +0.2812] | **FAIL** |

Paired within-seed vs Share-0 (detectable loss iff UCB95 < 0):

| condition | D_A | D_B | classification |
|---|---|---|---|
| Share-Encoder | −0.0234 [−0.1562, +0.1016] | −0.0938 [−0.2109, +0.0234] | A: no detectable loss; B: no detectable loss |
| Share-Backbone | −0.1406 [−0.2734, **−0.0078**] | −0.1172 [−0.2344, +0.0000] | **A: sharing damaged specialization**; B: none |
| Share-Macro | −0.1641 [−0.3125, **−0.0156**] | −0.1016 [−0.2500, +0.0469] | **A: sharing damaged specialization**; B: none |

Share-Macro fails the absolute criterion via LCB95(Δ_A) = 0.0000 exactly.

## 6. Distillation fidelity (both scales)

| | 2v2 | 6v6 |
|---|---|---|
| holdout agreement z0↔π_A | 0.967 | 0.962938 |
| holdout agreement z1↔π_B | 0.989 | 0.960858 |
| holdout KL (A / B) | — | 0.054201 / 0.094007 |
| student branch JSD | 0.178 | 0.477597 |
| teacher branch JSD | 0.179 | 0.483848 |
| **separation retained** | **99.4%** | **98.7%** |
| roundtrip max abs logit diff | 0.0 | 0.0 |
| preflight | — | 8/8 PASS |

Teachers are far more behaviorally distinct at 6v6 (0.484 vs 0.179), so the
compressed policy must preserve a much wider separation at the larger scale.

**Counterexample that makes fidelity insufficient:** 2v2 Share-Macro holds
0.936/0.969 agreement and still FAILS the payoff criterion.

## 7. 2v2 — robustness dose–response (n=128)

Nominal reference: Δ_A = +0.1797 [+0.0547, +0.3047], Δ_B = +0.1562 [+0.0312, +0.2812] (PASS).

| family | nominal | low | medium | high | pattern |
|---|---|---|---|---|---|
| localization | PASS | 0.03 PASS | 0.06 PASS | 0.12 PASS | robust at every dose |
| motion error | PASS | 0.03 **FAIL** | 0.06 **FAIL** | 0.12 **FAIL** | fails at *every* nonzero dose, always via Pole B; non-monotonic |
| control delay | PASS | 1 tick **FAIL** | 2 ticks **FAIL** | 4 ticks not authorized | monotone Pole-A erosion (Δ_A 0.1797 → 0.1094 → 0.0312) |

No R (paired-vs-nominal) interval clears zero in any family — degradation is
never *statistically confirmed*, so the honest reading is "fails the joint gate"
not "broken." Full per-condition tables in `data/RESULTS_DATA_2v2_6v6.json`
under `2v2.robustness`.

## 8. 6v6 — sealed

| stage | status |
|---|---|
| demand | SEALED, CERTIFIED |
| specialists (teachers) | frozen (seeds 7610001 / 7620001) |
| Rung-1 distillation | SEALED (sha256 `0df6d067…`), preflight 8/8 |
| **Share-Encoder crossover** | **SEALED FAIL** — Δ_A=+0.0312 [−0.0469,+0.1094]; Δ_B=+0.0391 [+0.0078,+0.0781] |
| **Share-0 teacher diagnostic** | **SEALED FAIL** — Δ_A=+0.0078 [−0.0703,+0.0859]; Δ_B=+0.0078 [−0.0234,+0.0391] |
| robustness | RESERVED UNSPENT (not authorized after FAIL) |

Teacher cells (n=128): π_A@A=0.8438, π_A@B=0.9531, π_B@A=0.8359, π_B@B=0.9609.  
Student cells: z0@A=0.8672, z0@B=0.9531, z1@A=0.8359, z1@B=0.9922.

**Interpretation:** teachers were essentially flat under the sealed gate; Rung-1 A-side failure is not well attributed to compression of a strong source specialization gap. Offline fidelity remained high (~99% JSD retained).

```bash
./.venv/Scripts/python.exe paper/figures/harvest_results_data.py
./.venv/Scripts/python.exe paper/figures/build_cross_scale_2v2_6v6.py
./.venv/Scripts/python.exe paper/figures/build_specialization_scaling_2v2_6v6.py
./.venv/Scripts/python.exe paper/figures/build_6v6_payoff_share0_vs_encoder.py
./.venv/Scripts/python.exe paper/figures/build_sharing_ladder_2v2.py
```

---

## 9. Plots

`plots_pdf/` (vector) and `plots_png/` (600 dpi). Draft paths also under `paper/plots/`.

| figure | shows |
|---|---|
| `fig_cross_scale_2v2_6v6` | (a) demand (b) fidelity (c) compressed-policy Δ — 6v6 now FAIL |
| `fig_specialization_scaling_2v2_6v6` | 2v2 Share-0/Encoder + 6v6 teachers/Encoder Δ |
| `fig_6v6_payoff_share0_vs_encoder` | absolute WR: teachers vs Share-Encoder student |
| `fig_sharing_ladder_2v2` | sealed Δ + paired D vs Share-0 |
| `fig_method_pipeline_2v2` | how latent strategies are obtained |
| `fig_claim_a_2v2` | experts → distilled modes → sharing erodes |
| `fig_absolute_winrate_context_2v2` | absolute WR context (2v2) |
| `fig_2v2_measurement_hierarchy` | high imitation ≠ payoff specialization |
| `fig_robustness_delta_dose` | dose–response |
| `fig_trajectory_strip_2v2` / `fig_qualitative_latent_2v2` | qualitative |
| `fig_role_allocation_2v2` / `fig_latent_behavior_2v2` | exploratory telemetry |

## 10. Excluded

4v4 is excluded from Results per your direction. Its sealed records are intact on
disk and unaffected: C2/B2-1 crossover FAIL, vanilla B3-3 crossover FAIL,
role-preservation crossover wrote an INTEGRITY_REQUIRED flag (Δ_A reversal,
0.6328 − 0.8281 = −0.1953) at 18:33Z.
