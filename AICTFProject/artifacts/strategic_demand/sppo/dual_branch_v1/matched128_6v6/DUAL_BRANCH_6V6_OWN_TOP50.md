# DUAL_BRANCH_6V6_OWN_TOP50

POST-HOC DESCRIPTIVE SUBSET: best-case strategic capability on the system's own best 50 scenarios. Not an unbiased estimate of general performance; the full-128 readout is.

Rule (historical, verified to reproduce the old 6v6 list): score(seed)=outcome(strategy A on Pole A)+outcome(strategy B on Pole B); rank descending; tie-break ascending seed ID; take top 50. Post-hoc descriptive subset; primary results remain sealed n=128.

Scores over 128 seeds: both won (2) 93, one (1) 30, neither (0) 5. Cutoff score 2: 0 seeds above, 93 tied (tie-break ascending seed ID). Overlap with the old Ours top-50: 39/50.

## Win rate

| Seeds | A@A | B@A | A@B | B@B | Δ_A mean ± std [95% CI] | Δ_B mean ± std [95% CI] |
|---|---|---|---|---|---|---|
| own top-50 | 1.000 | 0.880 | 0.340 | 1.000 | +0.120 ± 0.328 [+0.040, +0.220] | +0.660 ± 0.479 [+0.520, +0.780] |
| all 128 | 0.844 | 0.906 | 0.266 | 0.844 | -0.062 ± 0.498 [-0.148, +0.023] | +0.578 ± 0.527 [+0.484, +0.672] |

## Score margin (blue − red)

| Seeds | A@A | B@A | A@B | B@B | Δ_A mean ± std [95% CI] | Δ_B mean ± std [95% CI] |
|---|---|---|---|---|---|---|
| own top-50 | 2.400 | 2.280 | 0.360 | 1.560 | +0.120 ± 1.350 [-0.240, +0.480] | +1.200 ± 0.833 [+0.980, +1.440] |
| all 128 | 1.969 | 2.391 | 0.273 | 1.398 | -0.422 ± 1.625 [-0.711, -0.141] | +1.125 ± 1.080 [+0.938, +1.312] |

