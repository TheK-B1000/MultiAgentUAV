# C2 candidate qualification — preregistration

**Status:** FROZEN before any scan.
**Date:** 2026-08-05
**Machine-readable:** `artifacts/c2_qualification/C2_QUALIFICATION_FROZEN.json`
**Motivated by:** `artifacts/o1_gates/O1_POSTMORTEM.json`

This document fixes *what makes a C2 candidate admissible* and *how one is
chosen*. It does not name a candidate. Which feature wins is decided by the
Stage 2 audit under these rules, and no scan result informed anything below.

---

## Why this exists

O1 failed for two separable reasons, and only one of them was about O1.

The scientific reason was Gate 4: a fully independent policy trained for 1M
steps entirely on a confirmed niche was not behaviorally distinguishable from
the incumbent beyond ordinary seed variation.

The methodological reason was worse, because it was avoidable. C1 was a real
statistical finding — sustained home threat genuinely predicted losing a lead —
but the niche had almost no room to improve:

```text
G0 lead preservation on natural C1   0.991
headroom                             0.009
Gate 1 required improvement          0.100
```

The gate could not have passed if O1 had been perfect. **A confirmed predictive
weakness is not automatically a trainable strategic niche.** Criterion 7 exists
solely to make that mistake impossible to repeat, and criterion 9 exists to stop
the second trap — features that predict a failure because they are *caused* by it.

## Data source: replay, not new data

Stage 2 uses **existing G0 discovery trajectories**. Concretely, it replays the
original G0-V5 discovery block (seeds 9400000+) with the same three frozen
policies, re-instrumented to capture per-step context.

This is a replay rather than a new experiment. `precursor_windows.csv` stored
only aggregated features over one fixed 30-decision window ending at the
failure, so lag-band analysis is impossible from it — the per-step data was
never persisted. Re-running deterministic policies on the same seeds reproduces
the same trajectories at finer resolution. **No fresh seed block is consumed.**

## Lag bands

```text
t-30 .. t-20      earliest
t-20 .. t-10
t-10 .. t-1       latest
                  t itself is the outcome and is EXCLUDED
```

## The ten criteria

A candidate qualifies only if **all** hold.

| # | criterion | requirement |
|---|---|---|
| 1 | temporal precedence | every value read strictly before the outcome step |
| 2 | window-level unit | exists at the unit a handoff can consume; never "episode ever had X" |
| 3 | replication | same effect direction across all 3 G0-V5 policies |
| 4 | effect size | `abs(delta) >= 0.15` in ≥2/3 policies |
| 5 | uncertainty | episode-clustered CI excludes zero in ≥2/3 policies |
| 6 | support | ≥30 failure and ≥30 matched-control windows per policy per cell |
| 7 | headroom | `headroom >= 0.20` **and** `>= 2x` the planned Gate-1 effect (0.10) |
| 8 | actionability | free teammate can contest before the nearest ready defender, in ≥30% of failure onsets |
| 9 | non-mechanical | survives the lag-decay test **and** the score-stratum test |
| 10 | natural support | onset prevalence ≥2% of episodes, ≥30 onsets per policy |

### Criterion 9 in detail

Two sub-tests, both required.

**Lag decay.** The effect must appear with the correct direction in the
*earliest* band (`t-30..t-20`), not only in the latest. A feature that separates
at `t-10..t-1` but vanishes at `t-30..t-20` is describing the failure in
progress, not a precursor. That result is classified `OUTCOME_CONTAMINATION`
and rejected.

This is aimed squarely at `defender_tag_available_frac`, which was the most
consistent window-level separator in the G0-V5 discovery (about −0.14, CI
excluding zero in all six cells) — but with a *negative* sign, which is exactly
what you would see if the defender had already spent its tag causing the
failure. If that is what it is, this test will say so.

**Score stratum.** The effect must survive within at least one adequately
supported score stratum, not only pooled. `leading_frac` was the only CI-backed
separator for both carrier failures (about −0.18), so any feature correlated
with score state inherits that separation for free.

## Features audited

All recorded actionable Layer-3 features, not a preselected favourite:
`carrier_unescorted_frac`, `mean_escort_distance`, `defender_tag_available_frac`,
`mean_nearest_ready_defender`, `mean_carrier_pressure`,
`carrier_pressure_increasing`, `carrier_pressure_trend`, `both_forward_frac`,
`none_forward_frac`, `mean_team_separation`, `own_cooldown_bound_frac`,
`intervention_margin`, `mate_can_intervene`.

**`carrier_mostly_unescorted` starts REJECTED.** Being the previous runner-up
confers nothing. Its appeal was a layer-2 *episode* descriptor — a 0.256
win-rate drop with no episode-clustered CI — and that is precisely the
episode-level reasoning that invalidated three O1 gates. At the window level in
the G0-V5 discovery it did not separate:

```text
carrier_unescorted_frac, delta vs matched controls (bar 0.15)

tagged_while_carrying    -0.030  +0.053  +0.015
dropped_the_flag         +0.034  +0.110  +0.064
```

It qualifies only if this audit independently says so.

## Selection

Among candidates passing every criterion, rank mechanically:

1. 3/3 replication over 2/3
2. larger absolute effect
3. stronger CI separation
4. greater natural support
5. greater headroom
6. clearer actionable fork

Select exactly one. Substituting a different candidate after seeing results is
prohibited.

## Stopping

```text
zero candidates qualify  ->  C2_NO_QUALIFIED_CANDIDATE.json, STOP
headroom fails           ->  C2_HEADROOM_FAIL.json, STOP
```

Do not invent a niche. A clean "there is no qualifying C2 in this game under
these criteria" is a legitimate and publishable result.

## Immutability

After the Stage 2 audit produces numbers, no criterion, threshold, lag band or
ranking rule above may change.

---

**Note on a discarded scan.** A pickup-anchored discovery run on seed block
9800000 was launched before this contract existed. It was stopped and its output
deleted without being read. It informed no criterion here and is not used for
selection.
