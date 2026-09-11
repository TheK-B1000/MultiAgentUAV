# Results handoff — 2v2 + 6v6

All numbers below are read directly from sealed artifacts by
`paper/figures/harvest_results_data.py`. Nothing is retyped by hand.
Re-run that script to refresh `data/RESULTS_DATA_2v2_6v6.json` after any new
evaluation seals.

**One item is not yet sealed: the 6v6 crossover Δ. It is marked PENDING and must
not be estimated.**

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

## 8. 6v6 — status

| stage | status |
|---|---|
| demand | SEALED, CERTIFIED |
| specialists (teachers) | trained, terminal ckpts frozen (seeds 7610001 / 7620001, 1M steps) |
| Rung-1 distillation | SEALED, frozen (sha256 `0df6d067…`), preflight 8/8 |
| **crossover (Δ_A, Δ_B)** | **PENDING** — seeds 13640001–128, 512 ep, launched 18:44Z, ETA ≈ 03:35Z |
| robustness | RESERVED UNSPENT — seeds 13660001–128, gated on crossover PASS |

Expert crossover is deliberately **not** re-certified at 6v6, and the Share-0 /
Backbone / Macro rungs are **not** ported — 2v2 localizes *where* sharing breaks
specialization; 6v6 tests only whether the surviving rung survives scale.

When the crossover seals:
```bash
./.venv/Scripts/python.exe paper/figures/harvest_results_data.py   # refresh data
./.venv/Scripts/python.exe paper/figures/build_cross_scale_2v2_6v6.py  # panel (c) auto-fills
```

---

## 9. Plots

`plots_pdf/` (vector, for the paper) and `plots_png/` (600 dpi preview).

| figure | shows | source data |
|---|---|---|
| `fig_cross_scale_2v2_6v6` | **(a)** demand certified at both scales **(b)** ~99% teacher separation retained under identical 48% cut **(c)** criterion on compressed policy — 2v2 sealed, 6v6 pending | harvested + sealed artifacts |
| `fig_method_pipeline_2v2` | how latent strategies are obtained | — |
| `fig_claim_a_2v2` | 3-panel causal chain: experts → distilled modes → sharing erodes | ladder artifacts |
| `fig_sharing_ladder_2v2` | exact sealed Δ (left) + paired D vs Share-0 (right) | RUNG0–3 |
| `fig_absolute_winrate_context_2v2` | absolute WR: π_G / π_A / π_B / z0,z1 | PI_G + specialists + Rung-1 |
| `fig_2v2_measurement_hierarchy` | high imitation ≠ payoff specialization | Share-Macro |
| `fig_robustness_delta_dose` | dose–response across 3 perturbation families | low/med/high tiers |
| `fig_trajectory_strip_2v2` | matched seed 11960003, z0 vs z1 paths, diverge t=4 | sealed rows |
| `fig_qualitative_latent_2v2` | matched frame grid at t=30 | sealed rows |
| `fig_role_allocation_2v2` | role proxies (exploratory, n=24/cell — not a gate) | telemetry |
| `fig_latent_behavior_2v2` | extended role telemetry (exploratory) | telemetry |

`fig_cross_scale_2v2_6v6` is the only figure carrying both scales; the rest are
2v2. All are also in `paper/icra2027/figures/` for Overleaf.

## 10. Excluded

4v4 is excluded from Results per your direction. Its sealed records are intact on
disk and unaffected: C2/B2-1 crossover FAIL, vanilla B3-3 crossover FAIL,
role-preservation crossover wrote an INTEGRITY_REQUIRED flag (Δ_A reversal,
0.6328 − 0.8281 = −0.1953) at 18:33Z.
