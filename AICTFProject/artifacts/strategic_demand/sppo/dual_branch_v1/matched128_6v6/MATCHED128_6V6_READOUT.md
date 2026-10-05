# POSTHOC_MATCHED128_6V6_DUAL_BRANCH (n = 128 seeds, block 25800001..25800128)

POST-HOC MATCHED EVALUATION ON THE HISTORICAL 128-SEED 6V6 BLOCK. Not confirmatory. Not fresh seeds. Not PAPER-FAITHFUL. Does not replace any future untouched evaluation of the dual-branch method.

## Win rate

| System | A@A | B@A | A@B | B@B | Δ_A mean ± std [95% CI] | Δ_B mean ± std [95% CI] |
|---|---|---|---|---|---|---|
| dual_branch | 0.844 | 0.906 | 0.266 | 0.844 | -0.062 ± 0.498 [-0.148, +0.023] | +0.578 ± 0.527 [+0.484, +0.672] |
| old_asymmetric_ours | 0.922 | 0.859 | 0.859 | 0.977 | +0.062 ± 0.411 [-0.008, +0.133] | +0.117 ± 0.346 [+0.062, +0.180] |

Paired dual_branch_minus_old_asymmetric_ours: Δ_A -0.125 ± 0.652 [-0.234, -0.016]; Δ_B +0.461 ± 0.651 [+0.344, +0.570]

## Score margin (blue − red)

| System | A@A | B@A | A@B | B@B | Δ_A mean ± std [95% CI] | Δ_B mean ± std [95% CI] |
|---|---|---|---|---|---|---|
| dual_branch | 1.969 | 2.391 | 0.273 | 1.398 | -0.422 ± 1.625 [-0.711, -0.141] | +1.125 ± 1.080 [+0.938, +1.312] |
| old_asymmetric_ours | 2.164 | 1.352 | 1.688 | 2.422 | +0.812 ± 1.344 [+0.578, +1.047] | +0.734 ± 1.187 [+0.531, +0.938] |

Paired dual_branch_minus_old_asymmetric_ours: Δ_A -1.234 ± 2.109 [-1.602, -0.867]; Δ_B +0.391 ± 1.523 [+0.125, +0.656]

