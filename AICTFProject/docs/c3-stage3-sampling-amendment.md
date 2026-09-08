# C3 Stage-3 sampling amendment

**Status:** FROZEN, before any Stage-3 outcome exists.
**Date:** 2026-08-07
**Amends:** `artifacts/c3_discovery/C3_DISCOVERY_PREREG_FROZEN.json`
**Machine-readable:** `artifacts/c3_discovery/C3_STAGE3_SAMPLING_AMENDMENT.json`
**Scope:** `CONTROLLABILITY_SCREEN_ONLY`. Authorises no O3 and makes no latent
necessity claim.

---

## What changes

The **estimator**. Nothing else.

The first C3 attempt hit `ABORTED_OPERATIONAL_SCALE` at Stage 3: exhaustive
per-anchor counterfactual branching over the full legal team-response product
made wall-clock operationally unbounded.

But the frozen aggregate gate is a **rate**:

```text
fork_rate = n_qualified_commitment_forks / n_pressure_anchors  >=  0.20
```

A rate is estimable from a sample with quantifiable precision. Enumerating
every anchor buys precision far past the decision boundary at combinatorial
cost. Sampling anchors is therefore not a concession — it is the appropriate
estimator for the quantity the contract already chose.

Unchanged, and re-stated so the diff is unambiguous:

```text
T_trace              40
H_response           30
delta                0.10
U                    carrier_survival
minimum_fork_rate    0.20
pressure predicate   unchanged
earliest-fork rule   unchanged
doomed rule          unchanged
legal team responses EXHAUSTIVE
```

## What is deliberately NOT sampled

Within every sampled anchor the **full cartesian product of per-agent legal
macros is still enumerated**.

The contract forbids `top2_policy_actions_forbidden` and
`hand_curated_subset_forbidden` for a reason: they stop us testing only the
alternatives we already believe in. Sampling responses is exactly how a named
strategy like ESCORT would smuggle itself back into a screen that is supposed
to let the game speak. Anchors are sampled. Responses never are.

Cost falls linearly with anchor count; the anti-bias guarantee is untouched.

## Sampling design

```text
n_anchors    210
strata       21   (policy x opponent)
per stratum  10
```

Stratifying guarantees every policy/opponent cell is examined, so no single
adversary or seed dominates the screen. At n=210 the standard error of a
proportion near 0.20 is about 0.028, a 95% interval near ±0.055 — enough to
separate 0.20 from a materially larger rate, which is the only decision this
screen makes. Expected Stage-3 cost is roughly 6 hours against ~70 exhaustive.

### The seed is derived, not chosen

```text
c3_contract_sha256   c40eb5e8bca4f9a1d927b4779a312281a2c2fbee38bf7c4afa7fb94c35769ae0
sampling_seed        3289298408
derivation           int(sha256(contract bytes)[:8], 16)
```

Deriving the seed from the frozen contract's own hash makes anchor selection
impossible to massage later: the seed is a function of a document that predates
it and cannot change without changing the hash. The resolved integer is
recorded above so the draw is reproducible by anyone.

### Low-count fallback

```text
if a stratum has < 10 anchors:
    take all available anchors in that stratum
    redistribute the deficit deterministically among strata
    with unused anchors, round-robin by (policy_seed, opponent)
```

Resolved from the Stage-1 census **before** any Stage-3 outcome is inspected, so
the repair rule cannot be tuned once fork outcomes are visible.

## Estimator: weight back to the natural population

An unweighted `qualified / 210` would silently change the estimand. Equal
allocation over-weights rare strata relative to how often those situations
actually occur.

Using the **complete** Stage-1 census to define stratum weights:

```text
W_h   = N_h / N          N_h = complete Stage-1 anchor count for stratum h
p_hat = sum_h W_h * p_hat_h
```

so the estimate describes the fork rate of the **natural anchor population**
while still guaranteeing every cell is examined.

## Uncertainty

Episode-clustered, stratified bootstrap. Several anchors can come from one
trajectory and are not independent, so resampling anchors directly would
understate the interval. Episodes are resampled within each stratum and the
weighted estimate recomputed with the frozen `W_h`.

```text
resamples   2000
seed        12345
```

Frozen before the sample is evaluated.

## Decision rule

```text
C3_PASS      LCB95( weighted natural fork_rate ) > 0.20
otherwise    C3 does not pass
```

The decision is on the **lower bound**. Under sampling, a point estimate above
0.20 with an interval straddling it is not a pass.

## Operational sequence

```text
1. commit this amendment                      <- now, no Stage-3 outcome exists
2. let pid 38124 finish Stage 1 -> 630/630
   persist C3_STAGE1_ANCHORS.jsonl + MANIFEST
3. STOP pid 38124                             <- it runs pre-amendment code and
                                                 would enter exhaustive Stage 3
4. verify the Stage-1 census is intact
5. build the sampled-anchor manifest deterministically
6. relaunch Stage 3 from the persisted Stage 1, sampled
```

**Do not kill pid 38124 before Stage-1 persistence.** The first attempt already
lost four hours of Stage-1 work exactly that way.

The 15-episode operational benchmark must not influence anchor selection. The
sampling frame is the complete 630-episode Stage-1 census and nothing else.

## Immutability

Once any Stage-3 outcome is observed, none of these may change: the anchor
count, the stratification, the sampling seed or its derivation, the low-count
fallback, the weighting scheme, the bootstrap procedure or seed, or the decision
rule.
