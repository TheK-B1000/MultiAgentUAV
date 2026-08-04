# O1 response-oracle preregistration

**Status:** FROZEN before any O1 training data exists.
**Date:** 2026-08-04
**Authorised by:** C1 confirmation, `artifacts/c1_confirmation/C1_CONFIRMATION.json`
(verdict CONFIRMED, 3/3 policies, deltas 0.650–0.687, all CIs excluding zero,
on evaluation seeds 9500000+ disjoint from every prior block).

C1 confirmation opened this gate and nothing more. It established that **among
opportunity-matched leads, sustained home threat predicts failure to preserve
the lead**. It did *not* establish a mechanism, and this document must not be
read as assuming one. O1 is a search for whether PPO discovers a useful
alternative response, not an implementation of a response we have already named.

---

## 0. What is being tested

> Does an independently trained policy exposed to C1 acquire a response that is
> better than G0 *in C1*, without simply being better than G0 everywhere, such
> that selecting between the two beats either alone, and such that the two
> actually behave differently on the same observations?

Four gates, all of which must pass. Failing any one means O1 is not retained and
`z1` is not born.

---

## 1. Naming discipline

O1 is trained with the ordinary V5 task reward. It receives **no** role bonus,
no `DEFEND` / `RECOVER` / `ROTATE_HOME` shaping, and no z label. C1 exposure is
a *scenario filter* on the reset distribution — geometry, score and possession
only.

Nothing about O1's behaviour may be named until it has been observed. Sections
below therefore say "O1's response", never "the defensive response".

---

## 2. The C1 predicate

Frozen, and implemented once:

```text
C1_active  <=>  score_diff > 0  AND  home_threatened
```

Both terms are read from `run_g0_v2_evaluation.legal_context`, the function
whose `home_threatened` definition C1 was confirmed under. The runtime entry
points are `experiments/c1_context.py`:

| symbol | use |
|---|---|
| `c1_active_from_context(ctx)` | the definition; evaluation path |
| `c1_active_mask(core)` | batched replication; training path |

`tests/test_c1_context.py::test_batched_predicate_matches_the_confirmed_one`
asserts the two agree over 120 randomised states covering every branch of
`home_threatened`. If it ever fails, the batched copy is wrong and the
evaluation-side definition wins.

## 3. C1 exposure during training

O1 trains on episodes **reset into C1** by `experiments/c1_context.apply_c1_scenario`,
which implements the construction frozen in `C1_PROPOSAL.json` under
`6_recreatable_from_valid_states` before any confirmation data existed:

```text
BLUE ahead on score          blue_score = 1, red_score = 0  (score_limit = 3)
RED carrying the blue flag   red agent 0 carrying, flag attached to it
both BLUE agents past mid    blue_x > mid_x, by 1.5–4.0 cells
RED defender off cooldown    red_tag_cooldown = 0
```

Randomised within declared ranges (carrier 22–38% of the way home, lateral
jitter ±2.5 cells) so O1 sees the C1 *region* rather than one memorised state.

**`POD_DEFEND_LEAD` in `experiments/v6i26_phase_pods.py` is not used**, for two
independent reasons:

1. It places both blue agents adjacent to their own flag. C1's frozen
   construction requires both blue agents past the midline. The confirmed
   context is blue leading and *out of position*; that pod is blue leading and
   *already home*. It is a different context.
2. Its clock line guards on `core.decision_step`, which does not exist on
   `BatchedCTFCore` (the counter is `step_count`). The pod never set the late
   clock it documents, and its `max_decision_steps` lookup never read the
   configured horizon either.

Those pods belong to the V6I26 phase-pod birth attempt, which produced distinct
z indices with indistinguishable behaviour. This protocol does not reuse them.

### 3.1 Declared validity limitation

Injected C1 starts are **not** natural C1 states. Two differences are known and
recorded now rather than discovered later:

- **The clock.** `step_count` is left at 0, so an injected C1 hands O1 a full
  240-step horizon that a real mid-episode C1 does not have. The frozen
  construction says nothing about the clock and inventing one would add a free
  parameter to a preregistered scenario.
- **The prefix.** A natural C1 is reached by play; an injected one is not.

This is exactly why **every gate below is scored on natural, uninjected
episodes**. The injector buys training density and is allowed to influence
nothing else. If O1 wins only on injected states, it has not earned retention.

## 4. O1 training family — frozen constants

```text
seeds                 3300001, 3300002, 3300003   (fresh; disjoint from 3200001-3
                                                   and from every evaluation block)
initialisation        fresh; warm start FORBIDDEN  (matches the G0-V5 spec)
steps per seed        1,000,000                    (matched to G0, not tuned)
reward                locked Reward V5, identical constants to G0-V5
map / ruleset         map_a / RULESET_V2_AQUATICUS_10S
opponents             admitted OP6-OP12 mixture
horizon               240
domain randomisation  off
reset distribution    100% apply_c1_scenario
primary checkpoint    1,000,000 ONLY
```

Fresh initialisation is deliberate. Warm-starting from G0 reaches competence
far faster but biases O1 toward G0's behaviour, which is the exact direction in
which gate 4 failed in every prior attempt (V6I22, V6I26, K=2).

The 1,000,000-step checkpoint is the only primary evidence. Intermediate
checkpoints exist for health monitoring and may never be promoted afterwards,
however attractive one looks.

## 5. Evaluation construction

Fresh evaluation seed base **9,600,000**, disjoint from 9100000+ (V6I9
discovery), 9200000+ (collapse diagnostic), 9300000–2 (TASK_HEALTH panel),
9400000+ (G0-V5 discovery), 9500000+ (C1 confirmation) and all training seeds.

7 opponents × 30 seeds = 210 episodes per policy pair.

**Paired-prefix rollouts.** For each (opponent, evaluation seed), two episodes
are run from the identical seed:

```text
A:  G0 for the whole episode
B:  G0 until C1_active first becomes true at t*, then O1 to the horizon
```

Up to `t*` the two are the same trajectory. Everything after `t*` is
attributable to the handoff. Comparisons are paired on (opponent, seed).

The **selector** is defined by this construction: within-episode, one-way,
switching to O1 at the first C1 onset. This departs from "z fixed per episode"
in `docs/g0-c1-confirmation-preregistration.md` §4 step 7, deliberately and on
the record: C1 is a mid-episode state, and a per-episode selector could not see
the context that was actually confirmed. The per-episode variant is reported as
a secondary diagnostic and does not gate.

**Minimum support:** ≥ 30 natural C1 episodes per O1 seed, matching C1's own
support rule. Below that, the result is INSUFFICIENT, not a failure and not a
pass.

## 6. The four gates

All statistics: paired bootstrap, 2000 resamples, seed 12345, 95% interval,
cluster unit = episode.

### Gate 1 — O1 owns C1

```text
population   natural episodes in which C1_active fires
metric       lead_preserved = NOT lost_after_leading
             (the exact failure C1 was confirmed against)
statistic    P(preserve | B) - P(preserve | A), paired on (opponent, seed)
pass         delta >= 0.10 AND 95% CI excludes 0, in >= 2 of 3 O1 seeds
```

### Gate 2 — G0 retains an anchor

```text
population   natural episodes in which C1_active never fires
metric       win rate, full-episode G0 vs full-episode O1, same seeds
pass         WR(O1, anchor) <= WR(G0, anchor) + 0.05, in >= 2 of 3 O1 seeds
```

This gate fails when O1 is **better** on the anchor. That is not a happy
accident: an O1 that dominates everywhere is a better generalist, and the
correct response is to replace G0, not to birth `z1`. Complementarity requires
that neither policy is best everywhere.

### Gate 3 — the selector beats both fixed policies

```text
population   the full evaluation pool
compare      WR(selector) vs max( WR(G0 always), WR(O1 always) )
pass         margin >= 0.05 AND 95% CI on (selector - best fixed) excludes 0,
             in >= 2 of 3 O1 seeds
```

### Gate 4 — behaviour is distinct

Reuses the confirmatory statistic already implemented in
`experiments/analyze_k2_behavior_gate.py`, unchanged:

```text
B_distinct = median( JSD_between_families ) - Q_0.95( JSD_within_families )
pass       LCB95(B_distinct) > 0
```

Family-level, not per-seed: `G0 = {3200001-3}`, `O1 = {3300001-3}`.

Observation bank: natural C1 onset states, byte-identical observations for
every compared policy, identical legal-action masking, balanced across
G0-generated and O1-generated states.

The weaker `D_policy = mean(between) - mean(within)` is **not** used. Two
independently trained networks are always distinguishable given enough matched
observations, so that statistic tests "are these networks different at all"
rather than "are they meaningfully different" — it passed on a checkpoint that
had completely collapsed into a single dominant generalist. `B_distinct` sets
the bar at the observed seed-to-seed variation *within* each family, which is
not an arbitrary threshold.

Unstable seeds are not excluded. If a weak O1 seed inflates within-family
variation enough to fail this gate, O1 is not behaviourally coherent enough to
serve as a latent branch. That is a finding.

## 7. Retention rule

```text
all four gates pass   -> O1 retained; z0 = G0, z1 = O1; latent birth unlocks
any gate fails        -> O1 NOT retained; z1 NOT born
insufficient support  -> INSUFFICIENT; rerun with more seeds, change nothing else
```

Birth is a **separate** step with its own verification: forcing `z0` versus
`z1` must preserve the crossover measured here. Retention does not imply birth,
and the router remains locked until ≥ 2 branches are retained.

## 8. On a failure

If O1 fails, report it as a failure of *this* response oracle on *this* niche.
Do not:

- retune thresholds, the step budget, or the injector after seeing results
- promote an intermediate checkpoint
- select the best of the three O1 seeds
- swap in a different metric because it separates better
- reinterpret C1

The legitimate next moves are a different exposure mechanism (G0-prefix
handoff training, declared in a new preregistration), or the runner-up context
`carrier_mostly_unescorted` — which, per `C1_PROPOSAL.json`, inherits no part
of C1's confirmation and needs a brand-new protocol on fresh data.

## 9. Prohibited

- Training O1 on `POD_DEFEND_LEAD` or any other V6I26 phase pod
- Warm-starting O1 from G0
- Scoring any gate on injected episodes
- Naming O1's behaviour before observing it
- Birthing `z1` before all four gates pass
- Training the router before ≥ 2 branches are retained
- Changing any constant in this document after the freeze commit

---

**Freeze record:** this protocol is frozen by the commit that introduces this
file together with `experiments/c1_context.py`,
`tests/test_c1_context.py` and `artifacts/o1_preregistration/O1_PREREGISTRATION.json`.
The commit SHA is the freeze record. No O1 training data existed at that commit.
