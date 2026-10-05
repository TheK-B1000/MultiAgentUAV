# 4v4 home-occupancy diagnostic (HOME_OCCUPANCY_DIAGNOSTIC_4V4_V1)

Read-only diagnostic: A3 and B2 played natively (k = 0) on Pole B, the same 16 throwaway seeds
(99970001..99970016), with no action changes and no training. Spec frozen and committed (ac60bbfe)
before any episode. home_count = agents that are alive, untagged, not carrying, and within 4.5 cells of
their own flag home (the frozen composition-probe instrument). Cap = 2. Not performance evidence.

| | A3 | B2 |
|---|---|---|
| decision ticks | 2782 | 3561 |
| % ticks over cap (home_count > 2) | 25.7% | 16.5% |
| episodes ever over cap | 16/16 | 16/16 |
| episodes with an over-cap run >= 8 ticks | 4/16 (25%) | 8/16 (50%) |
| over-cap periods: count / mean / median / max (ticks) | 89 / 8.0 / 5 / 45 | 122 / 4.8 / 3 / 25 |
| home_count distribution 0/1/2/3/4 | .29/.28/.16/.24/.02 | .26/.33/.25/.16/.00 |

Frozen rule: GO if, for either policy, % ticks over cap > 5% or >= 25% of episodes have an over-cap run of
>= 8 ticks. **Decision: GO** (both policies, on both clauses). GO licenses designing a dynamic defense-cap
follow-up under its own frozen spec and fresh seeds; it is not evidence that capping helps.

The raw count (any alive agent within 4.5 cells, including tagged agents and carriers) is in the RESULT JSON
for transparency only. Scores are recorded for traceability only.
