# DUAL_BRANCH_4V4_OWN_TOP50

POST-HOC DESCRIPTIVE SUBSET: best-case strategic capability on the system's own best 50 scenarios. Not an unbiased estimate of general performance; the full-128 readout is.

Rule (historical, verified to reproduce the old 4v4 list): score(seed)=outcome(strategy A on Pole A)+outcome(strategy B on Pole B); rank descending; tie-break ascending seed ID; take top 50. Post-hoc descriptive subset; primary results remain sealed n=128.

Scores over 128 seeds: both won (2) 2, one (1) 73, neither (0) 53. Cutoff score 1: 2 seeds above, 73 tied (tie-break ascending seed ID). Overlap with the old Ours top-50: 20/50.

## Win rate

| Seeds | A@A | B@A | A@B | B@B | Δ_A mean ± std [95% CI] | Δ_B mean ± std [95% CI] |
|---|---|---|---|---|---|---|
| own top-50 | 0.980 | 0.300 | 0.160 | 0.060 | +0.680 ± 0.471 [+0.540, +0.800] | -0.100 ± 0.416 [-0.220, +0.020] |
| all 128 | 0.578 | 0.383 | 0.109 | 0.023 | +0.195 ± 0.722 [+0.070, +0.320] | -0.086 ± 0.333 [-0.148, -0.031] |

## Score margin (blue − red)

| Seeds | A@A | B@A | A@B | B@B | Δ_A mean ± std [95% CI] | Δ_B mean ± std [95% CI] |
|---|---|---|---|---|---|---|
| own top-50 | 1.720 | 0.080 | -0.720 | -1.340 | +1.640 ± 1.321 [+1.280, +2.000] | -0.620 ± 1.665 [-1.060, -0.160] |
| all 128 | 0.836 | 0.195 | -0.734 | -1.375 | +0.641 ± 1.873 [+0.320, +0.961] | -0.641 ± 1.446 [-0.891, -0.391] |

