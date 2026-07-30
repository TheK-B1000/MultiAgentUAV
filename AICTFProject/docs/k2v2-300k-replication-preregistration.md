# K=2 Specialist Replication at 300k — Preregistration

**Status:** LOCKED / AMENDED before any replication training data existed.
**Date locked:** 2026-07-30
**Revision:** 4 (amended 2026-07-30; behavior gate = B_distinct)
**Supersedes nothing about discovery.** The 1M experiment's verdict stands:
**FAIL**, latent birth blocked, router blocked.

---

## 0. Amendment record

### Revision 4 — 2026-07-30, pre-launch (behavior gate correction)

One change, made **before any replication training run started**.

**Formal behavior gate replaced:**

```text
Rev 3 (VOID as formal gate):  LCB95(D_policy) > 0
                              D_policy = mean(JSD_between) − mean(JSD_within)

Rev 4 (LOCKED):               LCB95(B_distinct) > 0
                              B_distinct = median(JSD_between) − Q_0.95(JSD_within)
```

Reason: independently trained networks are always detectably different given
enough matched observations. At the discovery 1M checkpoint, collapsed
generalists already pass `D_policy > 0` with a trivial separation ratio
≈ 1.09 — so that gate is decorative for strategy-family distinctness.
`B_distinct` asks whether the *typical* cross-family difference exceeds 95%
of ordinary same-family seed variation.

Unchanged from Rev 3: 6 seeds/family, 256 eval/context, 300k checkpoint,
`LCB95(Δ_assigned) > 0` payoff gate, seed IDs, launch-after-audit policy.
Discovery audit results must **not** alter this formula or threshold.

Watcher `--force-launch` is disabled; launch is explicit only after this freeze.

### Revision 3 — 2026-07-30, pre-launch (seed-block alignment)

Training/eval seed IDs aligned to `911001–6` / `912001–6` and
`1110001–256` / `1120001–256`. Sample size and Δ_assigned gate unchanged
from Rev 2. Staged `5 × 64` / `LCB(Δ_pool)` draft voided.

### Revision 2 — 2026-07-30, pre-launch

Evaluation raised to 256 paired seeds/context. Gate structure simplified to
two primary requirements. No interim analysis at 64 or 128.

### Revision 1 — 2026-07-30

Initial lock: 6 seeds/family, 64 eval/context, four gates.

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
audit blocks; unused Rev-2 IDs 903/904 and 105/106; all context-confirmation blocks).

Trainer, reward configuration, horizon, map, opponent definitions, preset
(`no_latent_baseline`), `n_envs=16`, and evaluation protocol are **unchanged** from the
discovery run.

Intermediate checkpoints are saved every 100k for diagnostics but **must not be inspected**
until the 300k formal analysis is complete and recorded.

Launcher: `experiments/launch_k2v3_300k_replication.py`
Manifest: `artifacts/k2v3_300k_replication/manifest.json`
Behavior gate: `experiments/analyze_k2_behavior_gate.py`

## 4. Formal gates — TWO primary requirements (both required)

### A. Joint complementary payoff

```
LCB95(Δ_assigned) > 0
```

```
V_assigned     = (R_R + S_S) / 2
V_fixed        = max( (R_R + R_S)/2 , (S_R + S_S)/2 )
Δ_assigned     = V_assigned - V_fixed
               = ½ · min( R_R - S_R , S_S - R_S )
```

This already requires both crossover directions to hold **jointly** in the bootstrap
distribution — it is strictly stronger than each marginal contrast clearing zero
separately.

### B. Learned strategy-family distinctness

```
LCB95(B_distinct) > 0

B_distinct = median(JSD_between) − Q_0.95(JSD_within)
```

In plain language: the typical πR-vs-πS difference must exceed at least 95% of the
differences seen between independently trained seeds of the **same** family. The bar
is set by observed seed-to-seed variation; no arbitrary JSD or ratio threshold.

Unstable / collapsed seeds are **not** excluded from the formal analysis. If within-family
variation is large enough that this gate fails, the proposed family is not behaviorally
coherent enough to serve as a stable latent.

### Descriptive only (not gates)

```
D_policy = mean(between) − mean(within)
separation_ratio = mean(between) / mean(within)
paired directional crossover CIs
Δ_pool (structural floor at 0)
argmax disagreement
pairwise JSD matrices
```

### Bootstrap

Hierarchical, resampling per replicate:
- training seeds with replacement within each family;
- evaluation seeds with replacement within each context (shared across families);
- for B_distinct, episodes with replacement within each observation source;
- exclude degenerate self-pairs created by duplicate seed draws from within-family pools.

## 5. The repertoire statistic

Legacy `Δ_pool` is non-negative by construction and is **not** a formal gate. Confirmatory
payoff statistic is signed `Δ_assigned` (§4A). Implemented in
`experiments/analyze_k2_assigned_gain.py`.

## 6. Prohibited (locked)

- No best-seed selection.
- No checkpoint selection — 300k is the only formal checkpoint.
- No seed exclusion, including collapsed runs.
- No replacement of failed runs.
- No inspection of intermediate checkpoints before the formal analysis is recorded.
- No interim analysis at 64 or 128 evaluation episodes.
- No sample-size, checkpoint, seed-block, gate-formula, or threshold changes after this freeze.
- Discovery audit results must not change this specification.
- No watcher auto-launch under a superseded gate.

A collapsed seed is **data**, not an error to be corrected.

## 7. Power — RESOLVED AND LOCKED

**Locked configuration: 6 training seeds per family, 256 paired evaluation seeds per
context. Nominal power ~85% (optimistic bound; see caveats in Rev 2 analysis).**

Nothing in §7 may be changed once the first training run starts.

## 8. Outcome rules (locked in advance)

```
BOTH primary gates pass
  (LCB95(Δ_assigned) > 0  AND  LCB95(B_distinct) > 0)
  -> complementary 300k specialists CONFIRMED
  -> retain the replicated policies
  -> proceed to latent branch birth (freeze initially)

EITHER primary gate fails
  -> replication does NOT confirm K=2
  -> do NOT run another checkpoint hunt
  -> promote the strongest πR family as incumbent generalist G0
  -> search for contexts that defeat the learned incumbent
```

No intermediate verdict exists. A partial pass is a fail.

## 9. Freeze

This document is frozen at **Revision 4**. Its SHA-256 is recorded in
`artifacts/k2v3_300k_replication/preregistration.lock.json` together with the hashes of the
launcher, manifest, and analysis scripts (`analyze_k2_assigned_gain.py`,
`analyze_k2_behavior_gate.py`) that will evaluate it. Any later edit invalidates the lock
and must be recorded as a new revision with its own timestamp and reason.

**Launch policy:** watcher auto-launch disabled. After Rev 4 freeze is committed and the
discovery behavior audit releases the GPU, launch all 12 runs explicitly via
`launch_k2v3_300k_replication.py --force-launch`.
