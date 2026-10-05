# POSTHOC_MATCHED128_4V4_DUAL_BRANCH (n = 128 seeds, block 21800001..21800128)

POST-HOC MATCHED EVALUATION ON THE HISTORICAL 128-SEED 4V4 BLOCK. PRIMARY Stage-3 evidence. Not confirmatory. Not PAPER-FAITHFUL. Historical top-50 is provenance only.

## Win rate

| System | A@A | B@A | A@B | B@B | Δ_A mean ± std [95% CI] | Δ_B mean ± std [95% CI] |
|---|---|---|---|---|---|---|
| dual_branch | 0.578 | 0.383 | 0.109 | 0.023 | +0.195 ± 0.722 [+0.070, +0.320] | -0.086 ± 0.333 [-0.148, -0.031] |
| old_asymmetric_ours | 0.648 | 0.492 | 0.102 | 0.508 | +0.156 ± 0.657 [+0.039, +0.273] | +0.406 ± 0.620 [+0.297, +0.508] |

Paired dual_branch_minus_old_asymmetric_ours: Δ_A +0.039 ± 0.999 [-0.133, +0.211]; Δ_B -0.492 ± 0.664 [-0.609, -0.375]

## Score margin (blue − red)

| System | A@A | B@A | A@B | B@B | Δ_A mean ± std [95% CI] | Δ_B mean ± std [95% CI] |
|---|---|---|---|---|---|---|
| dual_branch | 0.836 | 0.195 | -0.734 | -1.375 | +0.641 ± 1.873 [+0.320, +0.961] | -0.641 ± 1.446 [-0.891, -0.391] |
| old_asymmetric_ours | 1.211 | 0.625 | -0.633 | 0.758 | +0.586 ± 1.406 [+0.344, +0.828] | +1.391 ± 1.475 [+1.141, +1.641] |

Paired dual_branch_minus_old_asymmetric_ours: Δ_A +0.055 ± 2.315 [-0.344, +0.453]; Δ_B -2.031 ± 1.899 [-2.359, -1.703]

