# K=2 Specialist Replication at 300k — Preregistration

**Status:** LOCKED / AMENDED before any replication training data existed.
**Date locked:** 2026-07-30
**Revision:** 3 (amended 2026-07-30; seed-block alignment)
**Supersedes nothing about discovery.** The 1M experiment's verdict stands:
**FAIL**, latent birth blocked, router blocked.

---

## 0. Amendment record

### Revision 3 — 2026-07-30, pre-launch (seed-block alignment)

One change, made **before any replication training run started**.

**Training and evaluation seed IDs aligned to the confirmatory design text:**

```text
πR training:  911001–911006   (was 903001–903006 in Rev 2)
πS training:  912001–912006   (was 904001–904006 in Rev 2)
C_RUSH eval:  1110001–1110256 (was 1050001+ in Rev 2)
C_SPLIT eval: 1120001–1120256 (was 1060001+ in Rev 2)
```

Sample size, checkpoint, gates, and Δ_assigned definition are **unchanged** from
Revision 2 (6 seeds/family, 256 eval/context, formal gates A+B only).

Also records that a short-lived staged launcher draft (`5 × 64`,
`LCB(Δ_pool)>0`) was never launched and is void; it must not be used.

### Revision 2 — 2026-07-30, pre-launch

Two changes, both made **before any replication training run started** and therefore before
any replication data existed.

**(a) Evaluation raised from 128 to 256 paired seeds per context.**
Reason: the corrected hierarchical analysis (§2) showed the C_SPLIT direction does not
survive training-seed resampling, and the power simulation (§7) put 128 evals at ~75.5%
nominal — below a comfortable confirmatory target, and optimistic because the simulation can
only resample the three observed training seeds per family. Evaluation is the cheaper lever
(inference-only, no extra training runs), so the additional compute is spent there. Training
design is unchanged at 6 seeds per family.

**(b) Gate structure simplified from four gates to two primary requirements.**
Reason: Δ_assigned = ½·min(gate1, gate2) algebraically, so the former gates 1–3 were not
independent — they were three readings of one joint condition. They are now one primary
payoff requirement with the two component contrasts reported as explanation. See §4.

**One fixed evaluation block. No interim analysis at 64 or 128 episodes.**
**No further sample-size changes after launch.**

### Revision 1 — 2026-07-30

Initial lock: 6 seeds per family, 64 paired evaluation seeds per context, four gates.

---

## 1. Hypothesis

> Complementary responses can emerge early in training, but continued PPO optimization
> allows the OP11-trained policy to absorb the OP9 niche.

The original experiment showed a two-direction payoff crossover at 300k that was erased by
500k and replaced by πR dominance at 1M. This replication tests whether that early
complementarity is **real and reproducible** at a budget fixed in advance, rather than an
artifact of three training seeds.

## 2. What the discovery run actually showed

Reported honestly, because it sets the effect size this replication must detect.

At 300k, with **evaluation-seed-only** bootstrap (training seeds treated as fixed):

```
             C_RUSH   C_SPLIT
πR           2.4583    1.8854
πS           2.0417    2.1354

gate1 πR>πS on C_RUSH : +0.4167  CI95=[+0.2500, +0.5833]  PASS
gate2 πS>πR on C_SPLIT: +0.2500  CI95=[+0.0104, +0.4896]  PASS
```

At 300k, with **hierarchical** bootstrap (training seeds resampled — the standard this
replication adopts):

```
gate1 πR>πS on C_RUSH : +0.4167  CI95=[+0.1250, +0.7604]  PASS
gate2 πS>πR on C_SPLIT: +0.2500  CI95=[-0.0833, +0.5938]  FAIL
Δ_assigned            : +0.1250  CI95=[-0.0417, +0.2292]  FAIL
Δ_pool (legacy)       : +0.1250  CI95=[+0.0000, +0.2292]  FAIL
```

**The C_SPLIT direction does not survive training-seed resampling.** The discovery signal is
therefore weaker than a marginal-CI reading suggests: it is one direction firmly established
(gate1) plus one direction consistent-but-unconfirmed (gate2). This is a discovery signal,
not a near-miss pass.

## 3. Design (locked)

```
πR: 6 fresh training seeds, OP11_ADAPTIVE_EXPLOITER | map_b_split_lane
πS: 6 fresh training seeds, OP9_SPLIT_LANE_FEINT    | map_b_split_lane

Budget:             exactly 300,000 steps per run (12 runs, 3.6M total)
Formal checkpoint:  300k ONLY
Evaluation:         256 fresh paired seeds per context, ONE fixed block,
                    no interim analysis at 64 or 128 episodes (see §7)
Horizon:            240 decision steps
Agents:             2v2
Determinism:        deterministic action selection, no domain randomization, n_envs=1 at eval
```

Training seeds: `πR = 911001..911006`, `πS = 912001..912006`.
Evaluation seed blocks: `C_RUSH = 1_110_001 .. 1_110_256`, `C_SPLIT = 1_120_001 .. 1_120_256`.
All disjoint from every prior block (901xxx/902xxx training; 1010001/1020001 payoff;
1030001/1040001 audit; 903/904 and 105/106 unused reserved IDs from Rev 2; all
context-confirmation blocks).

Trainer, reward configuration, horizon, map, opponent definitions, preset
(`no_latent_baseline`), `n_envs=16`, and evaluation protocol are **unchanged** from the
discovery run.

Intermediate checkpoints are saved every 100k for diagnostics but **must not be inspected**
until the 300k formal analysis is complete and recorded.

Launcher: `experiments/launch_k2v3_300k_replication.py`
Manifest: `artifacts/k2v3_300k_replication/manifest.json`

## 4. Formal gates — TWO primary requirements (both required)

The former gates 1–3 were not independent: Δ_assigned = ½·min(gate1, gate2) exactly, so
requiring all three was three readings of one joint condition. The confirmatory experiment
has two primary requirements.

### A. Joint complementary payoff

```
LCB95(Δ_assigned) > 0
```

This already requires both crossover directions to hold **jointly** in the bootstrap
distribution — it is strictly stronger than each marginal contrast clearing zero
separately.

The two component contrasts are still reported:

```
πR − πS on C_RUSH
πS − πR on C_SPLIT
```

but they are **explanations of the joint result, not independent gates**. A pass or fail is
decided by Δ_assigned alone for the payoff side.

`Δ_pool` is reported for continuity with the failed 1M gate but is **not** a formal gate
(structural floor at zero).

### B. Learned policy distinction

```
LCB95(D_policy) > 0

D_policy = between_family_divergence − mean(within_πR_divergence, within_πS_divergence)
```

A **signed difference**, not a ratio: it can go negative, and its null value is exactly
zero, so a percentile LCB is not clipped at a boundary. Computed on the full six-seed
families. The healthy-seed sensitivity slice remains **diagnostic only** and never decides
this gate.

The threshold is **zero**, fixed now. It will not be tuned based on whether the discovery
audit looks impressive.

### Bootstrap

Hierarchical, resampling per replicate:
- training seeds with replacement within each family;
- evaluation seeds with replacement within each context (shared across families, preserving
  pairing);
- for D_policy, episodes with replacement within each observation source.

When a training-seed resample draws the same seed twice, the resulting degenerate
self-pair has divergence identically zero and is **excluded** from the within-family mean;
including it would bias within-family divergence downward and inflate D_policy.

## 5. The repertoire statistic

The legacy statistic is non-negative by construction:

```
V_selective = mean_c max_f pay(f,c)   >=   max_f mean_c pay(f,c) = V_fixed
Δ_pool      = V_selective - V_fixed   >=   0    always
```

so its percentile bootstrap piles mass at exactly zero, and it silently selects the
policy-to-context assignment *after* seeing outcomes.

Because the assignment is predeclared — πR handles C_RUSH, πS handles C_SPLIT — the
confirmatory statistic is signed:

```
V_assigned     = (R_R + S_S) / 2
V_fixed        = max( (R_R + R_S)/2 , (S_R + S_S)/2 )
Δ_assigned     = V_assigned - V_fixed
               = ½ · min( R_R - S_R , S_S - R_S )
```

(`R_R` = πR on C_RUSH, `R_S` = πR on C_SPLIT, `S_R` = πS on C_RUSH, `S_S` = πS on C_SPLIT.)

Δ_assigned can go negative as soon as either predeclared assignment fails. Note it is
exactly half the **minimum** of the two crossover margins, so requiring LCB95(Δ_assigned)>0
is the *joint* form of the two directional contrasts: it requires both directions to hold
simultaneously in ≥97.5% of replicates, which is strictly stronger than each marginal CI
clearing zero. Δ_pool is still reported, for continuity only.

Implemented in `experiments/analyze_k2_assigned_gain.py`.

## 6. Prohibited (locked)

- No best-seed selection.
- No checkpoint selection — 300k is the only formal checkpoint.
- No seed exclusion, including collapsed runs.
- No replacement of failed runs.
- No inspection of intermediate checkpoints before the formal analysis is recorded.
- No interim analysis at 64 or 128 evaluation episodes.
- No sample-size, checkpoint, seed-block, or gate changes after this freeze.
- Discovery audit results must not change this specification.

A collapsed seed is **data**, not an error to be corrected.

## 7. Power — RESOLVED AND LOCKED

**Locked configuration: 6 training seeds per family, 256 paired evaluation seeds per
context. Nominal power ~85%, approximate cost ~47 h.**

The ~85% figure is a **planning estimate, not a guarantee** — see the caveat below on why
it is optimistic.

Simulated from the discovery-run 300k rows (300 sims × 800 bootstrap replicates), for the
full three-payoff-gate set:

```
seeds/family   evals/context   power
    6              64          51.0%
    6             128          75.5%
    8              96          74.5%
    8             128          79.5%
    6             192          83.0%
   10             128          85.5%
```

Gate 2 is the sole bottleneck throughout (gate 1 runs 98–100%).

**These estimates OVERSTATE true power.** The simulation resamples training seeds from only
three observed values per family, so it cannot generate a seed worse than the worst
observed. Real power is lower than shown.

At the originally specified **6 seeds / 64 evals the design has ~51% power, and that is an
optimistic bound.** Evaluation seeds turn out to be the cheaper lever: they are
inference-only, whereas each additional training seed costs a full 300k run.

Approximate cost (training at ~1.6 h per 300k run, concurrency 2; evaluation at ~21.6 s per
episode, 24 cells):

```
 6 seeds /  64 evals   ~19 h total   51.0%
 6 seeds / 128 evals   ~28 h total   75.5%
 6 seeds / 192 evals   ~37 h total   83.0%
 8 seeds / 128 evals   ~37 h total   79.5%
```

Measured at fixed 6 seeds per family, varying evaluation only:

```
evals/context   power (gate2 / all)
     32              32.0% / 26.0%
     64              54.5% / 51.0%
    128              76.0% / 75.5%
    192                     / 83.0%
    256              85.0% / 85.0%   <- LOCKED
```

**Decision: 6 seeds per family, 256 paired evaluation seeds per context.** The training
design is exactly as originally locked; only evaluation was raised. Evaluation is
inference-only, so this is the cheapest available place to buy power.

The whole of §3–§7 is now locked. Nothing here may be changed once the first training run
starts. Because true power is below the nominal 85%, a FAIL must be read as "no effect
demonstrated at this budget," not as "no effect exists" — the §8 outcome rule is unchanged
either way, but the interpretation of a null result is bounded accordingly.

## 8. Outcome rules (locked in advance)

```
BOTH primary gates pass
  (LCB95(Δ_assigned) > 0  AND  LCB95(D_policy) > 0)
  -> complementary 300k specialists CONFIRMED
  -> retain the replicated policies
  -> proceed to latent branch birth (freeze initially)

EITHER primary gate fails
  -> replication does NOT confirm K=2
  -> do NOT run another checkpoint hunt
  -> promote the strongest πR family as incumbent generalist G0
  -> search for contexts that defeat the learned incumbent
     (sweep OP6-OP12 across legal frozen maps, rank by where G0 actually fails)
```

No intermediate verdict exists. A partial pass is a fail.

A failed replication means **"not confirmed"** — it is not proof that transient
specialization never exists. But it ends this particular OP11/OP9 specialist attempt rather
than opening another round of checkpoint archaeology.

## 9. Freeze

This document is frozen at **Revision 3**. Its SHA-256 is recorded in
`artifacts/k2v3_300k_replication/preregistration.lock.json` together with the hashes of the
launcher, manifest, and analysis scripts that will evaluate it. Any later edit invalidates
the lock and must be recorded as a new revision with its own timestamp and reason.

**Launch policy:** do not launch until discovery 200k + behavior audit finish and the GPU
is free; then launch all 12 runs immediately. Audit results must not change this freeze.
