# DUAL_BRANCH_2V2_OWN_TOP50

POST-HOC DESCRIPTIVE SUBSET: best-case strategic capability on the system's own best 50 scenarios. Not an unbiased estimate of general performance; the full-128 readout is.

Rule (historical, verified to reproduce the old 2v2 list): score(seed)=outcome(strategy A on Pole A)+outcome(strategy B on Pole B); rank descending; tie-break ascending seed ID; take top 50. Post-hoc descriptive subset; primary results remain sealed n=128.

Scores over 128 seeds: both won (2) 27, one (1) 81, neither (0) 20. Cutoff score 1: 27 seeds above, 81 tied (tie-break ascending seed ID). Overlap with the old Ours top-50: 17/50.

## Win rate

| Seeds | A@A | B@A | A@B | B@B | Δ_A mean ± std [95% CI] | Δ_B mean ± std [95% CI] |
|---|---|---|---|---|---|---|
| own top-50 | 0.860 | 0.540 | 0.020 | 0.680 | +0.320 ± 0.621 [+0.140, +0.480] | +0.660 ± 0.479 [+0.520, +0.780] |
| all 128 | 0.625 | 0.469 | 0.008 | 0.430 | +0.156 ± 0.657 [+0.047, +0.266] | +0.422 ± 0.496 [+0.336, +0.508] |

## Score margin (blue − red)

| Seeds | A@A | B@A | A@B | B@B | Δ_A mean ± std [95% CI] | Δ_B mean ± std [95% CI] |
|---|---|---|---|---|---|---|
| own top-50 | 1.400 | 0.920 | -0.660 | 0.760 | +0.480 ± 1.581 [+0.040, +0.920] | +1.420 ± 0.971 [+1.160, +1.700] |
| all 128 | 0.906 | 0.656 | -0.633 | 0.375 | +0.250 ± 1.655 [-0.039, +0.531] | +1.008 ± 1.147 [+0.805, +1.211] |

