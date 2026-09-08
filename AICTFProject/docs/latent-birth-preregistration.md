# Latent birth verification — preregistration

**Status:** FROZEN before the O1 four-gate verdict exists.
**Date:** 2026-08-04
**Scope:** verification only. This document does not specify the birth
*implementation*; see §9.

This protocol is written while the O1 result is genuinely unknown. Seeds
3300001 has completed training, 3300002 is training, 3300003 has not started,
and `run_o1_gates.py` has never been run. Freezing after the verdict would make
the birth criteria a function of O1's numbers, which is the exact failure mode
the C1 preregistration was built to prevent — and a strange thing to reintroduce
at the step that finally makes the latent claim.

---

## 0. What is being tested

The O1 gates, if they pass, establish that **G0 and O1 are two complementary
policies**. They say nothing about latents. Birth asks a separate question:

> When those two policies are compressed into one latent-conditioned network,
> does the complementarity survive being forced through `z`?

Distinct `z` indices are not distinct strategies. V6I26 produced distinct z
indices with indistinguishable behavior; V6I22 measured
`behavior_pair_distance_mean = 0.033`. This protocol exists to make that
outcome a recorded failure rather than a reinterpretation.

## 1. Eligibility

Birth is attempted **only if** the frozen O1 protocol
(`artifacts/o1_preregistration/O1_PREREGISTRATION.json`) returns
`RETAINED` on all four gates.

If O1 fails, this protocol stays dormant. It is not adapted, reused at a lower
bar, or applied to a different candidate.

## 2. Source pairing

```text
B1  <-  G0 3200001  +  O1 3300001
B2  <-  G0 3200002  +  O1 3300002
B3  <-  G0 3200003  +  O1 3300003
```

Index-matched, exactly as O1 gates 1-3 pair them. No cross-seed rematching and
no selecting the best available combination.

## 3. Latent mapping — fixed now

```text
z0 = G0 source family
z1 = O1 source family
```

Fixed before any birth model exists. **No relabeling after evaluation.** If the
branches turn out swapped relative to expectation, that is a result, not a
labelling error to be corrected.

## 4. Latent execution

```text
K = 2 for this birth test
z forced for the entire episode
no router, no q_phi
no mid-episode latent switching
no role reward, no strategy labels supplied to PPO
no DEFEND / RECOVER / ROTATE_HOME semantics encoded anywhere
```

K=4 exists in the architecture. Nothing here obliges the science to fill it.

## 5. Evaluation populations and seeds

Population definitions and primary metrics are **identical** to the
corresponding frozen O1 gates. Only the seeds are new:

```text
C1 crossover                    9700000 - 9700029
anchor / repertoire             9710000 - 9710029
matched-observation distinction 9720000 - 9720029
```

Verified disjoint from 9100000+ (V6I9 discovery), 9200000+ (collapse
diagnostic), 9300000-2 (TASK_HEALTH), 9400000+ (G0-V5 discovery), 9500000+ (C1
confirmation), 9600000+ (O1 gates), 9900000+ (training panels), and every
training seed. **Fail closed on collision:** the harness must abort rather than
proceed if any evaluation seed appears in a prior block.

## 6. The 75% preservation rule

Compression of two independently trained networks into one latent-conditioned
network is expected to cost something. This rule says how much is tolerable,
prospectively: **at most 25% of the pre-birth effect may be lost.**

For each birth gate the retention ratio is

```text
retention = post_birth_quantity / pre_birth_quantity
require retention >= 0.75
```

where the pre-birth quantity is the corresponding value measured by
`run_o1_gates.py` for the *same* index-matched pair.

Retention is only defined when the pre-birth quantity is positive. If it is
zero or negative, that pair did not pass the corresponding O1 gate, so under §1
birth was not eligible at all.

Retention is a **secondary** requirement. Every birth gate also requires the
post-birth measurement to independently satisfy the original O1 pass criterion.
A large pre-birth effect cannot buy a pass for a post-birth effect that is too
small on its own.

## 7. The four birth gates

### BIRTH GATE 1 — C1 crossover survives

```text
population  natural C1 episodes, seeds 9700000-9700029
compare     forced z1 vs forced z0
metric      lead_preserved = NOT lost_after_leading   (episode-level)
require     the same direction and pass criterion as O1 gate 1
            (delta >= 0.10 AND 95% CI excludes 0)
plus        effect_retention >= 0.75, where
            pre  = O1 gate 1 delta for this pair
            post = P(preserve | z1) - P(preserve | z0)
pass in     >= 2/3 pairs
```

### BIRTH GATE 2 — anchor crossover survives

```text
population  natural episodes where C1 never fires, seeds 9710000-9710029
compare     forced z0 vs forced z1, full-episode
require     the O1 gate 2 relation still holds:
            WR(z1) <= WR(z0) + 0.05
plus        margin retention >= 0.75, where the signed margin to the pass
            boundary is
                margin = ( WR(z0) + 0.05 ) - WR(z1)
            pre  = ( WR(G0) + 0.05 ) - WR(O1)   from O1 gate 2
            post = the same quantity under forced z
pass in     >= 2/3 pairs
```

No post-birth reinterpretation of "equivalent", "better", or "non-dominated".
The boundary is the frozen one.

### BIRTH GATE 3 — repertoire value survives

```text
population  seeds 9710000-9710029
compare     oracle selector(z0,z1) vs the best fixed forced-z branch
selector    within-episode, one-way, switches to z1 at first C1 onset --
            identical in construction to the O1 gate 3 selector
require     the O1 gate 3 criterion still holds:
            gain >= 0.05 AND 95% CI excludes 0, and the gain is positive
plus        gain retention >= 0.75 against the O1 gate 3 gain for this pair
pass in     >= 2/3 pairs
```

### BIRTH GATE 4 — behavioral distinction / no collapse

Family-level, not per-pair:

```text
Z0 family = forced-z0 outputs of B1, B2, B3
Z1 family = forced-z1 outputs of B1, B2, B3
```

Two independent measurements, **both required**:

```text
(a) B_distinct, family-level, via analyze_k2_behavior_gate:
        B_distinct = median(JSD_between) - Q_0.95(JSD_within)
        pass: LCB95(B_distinct) > 0
        observation bank: matched observations, seeds 9720000-9720029

(b) behavior_pair_distance_mean >= 0.35
        the 7-dim forced-z behavior vector in
        rl/forced_z_behavior_vectors.py; at K=2 this is the single
        z0-z1 distance
```

(a) measures divergence of action distributions at matched observations.
(b) measures divergence of aggregate behavioral statistics. They can disagree,
and both must pass.

### Absolute behavioral distinction — the three bands

```text
< 0.060          SEVERE_COLLAPSE
0.060 - 0.349    INSUFFICIENT_DISTINCTION
>= 0.350         ABSOLUTE_DISTINCTION_PASS
```

Birth fails in either of the first two cases. Only the third supports the
latent claim.

**Primary threshold: 0.35.** This is exactly
`DEFAULT_BEHAVIOR_PAIR_THRESHOLD` in `rl/forced_z_behavior_vectors.py` — the
repository's canonical bar for this exact metric. The claim this protocol
supports is a strong one, and using a softer threshold than the project already
ships for the same measurement would leave an unnecessary hole in it.

**Historical diagnostic floor: 0.060.** Retained for classification only. It was
V6I22E's birth gate, which scored 0.033 and failed. That tells us what obvious
collapse looks like; it does not define success, and it is **not sufficient for
`LATENT_BIRTH_PASS`**.

The middle band exists so a result can be described accurately. A measurement of,
say, 0.24 does not mean the branches are indistinguishable — it means they differ
somewhat and did not reach the preregistered threshold for distinct latent
strategies. Collapsing that case into "collapsed" would overstate the failure
just as adopting 0.060 would overstate the success.

Neither threshold moves because a latent network "had to compress two policies".
That is the thing being measured.

## 8. Collapse definition and verdict

Gate 4 fails — and birth fails with it — if either:

```text
1. family-level B_distinct fails, OR
2. behavior_pair_distance_mean < 0.35
```

The failure is *classified* by band, because "collapsed" and "not distinct
enough" are different findings and should not be reported as the same one:

```text
< 0.060          SEVERE_COLLAPSE            (V6I22E-class; branches converged)
0.060 - 0.349    INSUFFICIENT_DISTINCTION   (measurable separation, below bar)
>= 0.350         ABSOLUTE_DISTINCTION_PASS
```

```text
LATENT_BIRTH_PASS   only if gates 1, 2, 3 and 4 all pass
LATENT_BIRTH_FAIL   otherwise
```

Failure interpretation, fixed in advance:

| gate | meaning of failure |
|---|---|
| 1 | the C1 specialist behavior was not preserved through `z` |
| 2 | the anchor / crossover structure was lost |
| 3 | the two branches no longer provide repertoire value |
| 4 | behavior collapsed, even though the `z` indices differ |

Only `LATENT_BIRTH_PASS` supports the claim:

> `z0` and `z1` represent two behaviorally distinct, complementary latent team
> strategies.

## 9. What is deliberately NOT frozen here

The birth **implementation** — distillation objective, architecture, optimizer,
step budget, how G0 and O1 weights enter the latent network. That does not exist
yet and pretending otherwise would be dishonest.

It gets its own manifest, frozen immediately before any birth training begins,
and it must be frozen **without reference to birth-evaluation results**. The
verification protocol above is fixed from today and does not move to
accommodate whatever implementation is chosen.

## 10. Protocol immutability

Once the O1 gate results become visible, none of the following may change:

- any threshold, including the 0.75 retention rule, the 0.35 distinction bar,
  and the 0.060 severe-collapse classification floor
- any seed block
- any metric or statistic
- any population definition
- the `z0 = G0`, `z1 = O1` mapping
- the >= 2/3 pair requirement, or the family-level scope of gate 4

A failed birth may motivate a new birth *method*. That new attempt requires its
own preregistration and fresh evaluation seeds, and inherits nothing from this
one.

---

**Freeze record:** the commit introducing this file together with
`artifacts/latent_birth_preregistration/LATENT_BIRTH_PROTOCOL_FROZEN.json`.
That commit contains no executable changes. At that commit, O1 seed 3300001 had
completed training, 3300002 was training, 3300003 had not started, and
`run_o1_gates.py` had never been run.
