# POSTHOC_MATCHED128_2V2_DUAL_BRANCH (n = 128 seeds, block 23600001..23600128)

POST-HOC MATCHED EVALUATION ON THE HISTORICAL 128-SEED BLOCK. Not confirmatory. Not fresh seeds. Not PAPER-FAITHFUL. Does not replace any future untouched evaluation of the dual-branch method.

## Win rate

| System | A@A | B@A | A@B | B@B | Δ_A mean ± std [95% CI] | Δ_B mean ± std [95% CI] |
|---|---|---|---|---|---|---|
| dual_branch | 0.625 | 0.469 | 0.008 | 0.430 | +0.156 ± 0.657 [+0.047, +0.266] | +0.422 ± 0.496 [+0.336, +0.508] |
| old_asymmetric_ours | 0.641 | 0.523 | 0.047 | 0.586 | +0.117 ± 0.749 [-0.016, +0.250] | +0.539 ± 0.546 [+0.445, +0.633] |

Paired dual_branch_minus_old_asymmetric_ours: Δ_A +0.039 ± 1.023 [-0.141, +0.219]; Δ_B -0.117 ± 0.728 [-0.242, +0.008]

## Score margin (blue − red)

| System | A@A | B@A | A@B | B@B | Δ_A mean ± std [95% CI] | Δ_B mean ± std [95% CI] |
|---|---|---|---|---|---|---|
| dual_branch | 0.906 | 0.656 | -0.633 | 0.375 | +0.250 ± 1.655 [-0.039, +0.531] | +1.008 ± 1.147 [+0.805, +1.211] |
| old_asymmetric_ours | 0.859 | 0.703 | -0.266 | 0.859 | +0.156 ± 1.580 [-0.117, +0.430] | +1.125 ± 1.065 [+0.938, +1.312] |

Paired dual_branch_minus_old_asymmetric_ours: Δ_A +0.094 ± 2.379 [-0.320, +0.500]; Δ_B -0.117 ± 1.560 [-0.391, +0.148]

