# Environment-Demand Gate — preregistration

**Status:** PREREGISTERED 2026-08-06 (criteria fixed **before** any C3 scan
result exists). **Executed only after** an independent task-reward response
oracle (O3) exists. Numeric cells marked `TBD-freeze` must be frozen at O3
protocol freeze time, before any demand-evaluation episode is run.
**Machine-readable freeze target (not yet written):**
`artifacts/environment_demand_gate/DEMAND_GATE_FROZEN.json`

---

## The question this gate answers

Three claims have historically been blended together:

1. **Tactical affordance exists.** V2 geometry proves defenders, tagging,
   commitment, and escort geometry matter mechanically.
2. **A policy can make locally imperfect decisions.** C2/C3-style screens can
   find these.
3. **Multiple policies are actually necessary.** Never demonstrated.

Only #3 justifies a latent repertoire. This gate tests #3 and nothing else:

> Does this task distribution contain **context-dependent policy preference**
> that makes selection valuable?

The scientific hierarchy is fixed:

```text
C3              = Is there an actionable commitment fork?
O3              = Does an independently learned policy exploit that fork?
REPERTOIRE TEST = Is O3 complementary rather than simply better?
LATENT BIRTH    = Consequence of passing all three
```

C3 cannot prove policy specialization. O3 niche improvement cannot prove
repertoire value. Behavioral difference cannot substitute for comparative
utility. Each claim is made only by its owning stage.

A positive C3 counterfactual result
\(A(s)=\max_{a'}\mathbb{E}[U|s,a']-\mathbb{E}[U|s,G0]\)
means only "G0 could have done something better here." A sufficiently
expressive generalist may simply learn the correction. C3 **cannot** establish
strategy demand; this gate is the authority for latent eligibility.

## Position in the pipeline

```text
natural commitment fork
        ↓
C3 counterfactual screen           (candidate detector only)
        ↓
fresh confirmation
        ↓
train independent task-reward response oracle O3
        ↓
evaluate payoff matrix M[c, π]     (fresh demand-evaluation seeds)
        ↓
ENVIRONMENT-DEMAND GATE (this document)
        ↓
only if PASS → latent birth
```

Preregistered now; executed after O3 exists, because policy-level
`G_available` cannot be computed without genuinely independent policies.

---

## O3 training contract

The exact O3 optimizer, budget, seed family, and response horizon are frozen
in the O3 protocol only after C3 fresh confirmation passes. The invariant
training contract is already fixed:

```text
natural G0 prefix
exact confirmed onset
matched onset distribution
independent response policy
task reward only
post-onset PPO credit only
no role labels
no named-response reward
no latent or router involvement
```

O3 is trained over a distribution of confirmed natural onset states, not one
memorized snapshot. C3 exposure defines where learning begins; it does not
assign a semantic identity to the response.

---

## Core metric

For a repertoire \(\mathcal P\), define context-aware selection value and
best-fixed value:

\[
V_{\mathrm{sel}}(\mathcal P)
= \mathbb E_c\left[\max_{\pi\in\mathcal P}M[c,\pi]\right]
\]

\[
V_{\mathrm{fixed}}(\mathcal P)
= \max_{\pi\in\mathcal P}\mathbb E_c\left[M[c,\pi]\right]
\]

Then:

\[
G_{\mathrm{available}}(\mathcal P)
=V_{\mathrm{sel}}(\mathcal P)-V_{\mathrm{fixed}}(\mathcal P)
\]

For two policies: if I knew the context and could choose the better policy,
would I outperform whichever single policy is best overall?

**Context distribution.** The expectation over \(c\) uses **frozen context
frequencies** taken from the natural deployment-like distribution (measured
on discovery data and frozen before demand evaluation). Equal weighting is
used only if explicitly frozen as the scientific intent.

### Incremental repertoire value for later births

For an existing repertoire \(\mathcal P_K\) and candidate oracle
\(O_{K+1}\):

\[
\Delta V_{\mathrm{repertoire}}
=V_{\mathrm{sel}}(\mathcal P_K\cup\{O_{K+1}\})
-V_{\mathrm{sel}}(\mathcal P_K)
\]

For the first birth, positive `G_available` establishes that a repertoire is
useful. For every later birth, the new branch must additionally satisfy:

```text
LCB95(Delta V_repertoire) > 0
```

This prevents admitting an oracle that reproduces a context already covered
by an existing branch. Beating the best fixed policy is insufficient once a
selective repertoire already exists.

## Simultaneous requirements

D1-D4 must PASS for the first birth. D5 is additionally mandatory for every
later birth. Any required failure means NO LATENT BIRTH and NO ROUTER.

```text
D1. Positive routing value
    LCB95(G_available) > 0
    (episode-clustered bootstrap; resamples/seed frozen at freeze time)

D2. Preference reversal
    ∃ c1, c2:
        πA > πB in c1
        πB > πA in c2
    each direction with preregistered effect size and CI excluding zero
    (cells TBD-freeze)

D3. Competence
    every retained policy clears the minimum task-performance floor
    in every evaluated context (floor TBD-freeze; an oracle that is
    incompetent outside its niche is not a repertoire member)

D4. Nonredundancy
    matched-state behavioral distinction clears the frozen threshold
    (metric and threshold TBD-freeze; occupancy/entropy metrics are
    NOT acceptable evidence — see "latent collapse" note)

D5. Incremental repertoire value (mandatory when |P_K| > 1)
    LCB95(
        V_sel(P_K union {O_new}) - V_sel(P_K)
    ) > 0
    on fresh held-out cells under the same frozen context distribution
```

**Pairwise crossover (headline form of D2)** — retained as the strongest
interpretation:

```text
O3 beats G0 in C3 context
G0 beats O3 in anchor context
```

A positive aggregate selector gain without a demonstrated crossover is not
sufficient for latent birth.

Behavior need not look different throughout an episode. A response may differ
only during a short commitment window and still qualify if that difference is
reliable on matched states, produces preference reversal, and adds held-out
repertoire value. Strategy identity comes from comparative utility, not a
handwritten global style label.

## Data separation (mandatory)

`G_available` must be estimated on **fresh evaluation data**, never on
trajectories used to discover contexts or train O3. Otherwise contexts can be
built around where the two policies happened to separate, manufacturing
apparent routing value.

```text
discovery data           → define/freeze contexts (and context frequencies)
training data            → train O3
fresh demand-eval seeds  → construct M[c, π]
                           preference reversals
                           G_available
```

The demand-evaluation seed block must be disjoint from every prior block and
allocated fail-closed (same collision discipline as C2/C3 confirmation).
Spent blocks (`9800001+` C2 confirmation) are never reused.

## Bounded response-policy horizon (H_response)

One-macro deviation (C3 Stage 3) is evidence of **controllability** only.
Strategy evidence requires comparing **response modes**:

```text
same natural snapshot
      ↓
G0 continuation
vs
O3 continuation for H_response decisions
      ↓
normal downstream evaluation
```

`H_response` is frozen prospectively at O3 protocol freeze time (before any
branch is rolled). This tests a temporally extended response mode, not an
isolated action correction.

## Interpretation notes (frozen so results cannot be re-narrated)

**Latent collapse.** Marginal latent usage does not imply behavioral
specialization. If z branches are exchangeable and task reward admits one
broadly successful policy, entropy can distribute occupancy without creating
distinct action semantics. The optimization is not broken; it may be finding
an economically sensible solution. Absence of useful latents may mean:

> the sampled payoff surface does not yet contain enough conditional
> strategic regret to support a repertoire.

That explanation is preferred over "PPO can't learn the latents" unless
directly contradicted by evidence.

**Aggregate margins do not close the question.** A pooled opponent-level
margin (e.g. G0's +1.96 across OP6–OP12) shows no obvious opponent-level
weakness. It does **not** prove the absence of conditional within-episode
weakness:

```text
G0 beats OP9 overall
does NOT imply
G0 is best in every strategic state within OP9
```

Latent demand has not been disproved; it has not been demonstrated. This gate
is how it gets demonstrated or cleanly rejected.

## Latent eligibility (single authority)

```text
LATENT ELIGIBILITY REQUIRES:
1. response-oracle niche advantage           (O3 > G0 in C3 context)
2. anchor/context preference reversal        (G0 > O3 in anchor context)
3. positive LCB95 of G_available             (D1)
4. competence of every retained policy       (D3)
5. behavioral nonredundancy                  (D4)
6. all of the above on fresh evaluation data
7. for every later birth, positive incremental repertoire value (D5)

Failure of any item:
NO LATENT BIRTH
NO ROUTER
```

## Candidate disposition

The result is interpreted exactly as follows:

```text
O3 loses the niche
    -> DISCARD O3

O3 wins the niche and dominates elsewhere
    -> PROMOTE O3 TO GENERALIST
    -> replace or retire the incumbent as justified
    -> repertoire size remains unchanged

O3 wins the niche and loses elsewhere,
but context-aware selection adds no held-out value
    -> INTERESTING SPECIALIZATION
    -> NO LATENT BIRTH

O3 wins the niche
and an incumbent wins an anchor context
and selection beats every fixed policy
and LCB95(G_available) > 0
and behavior is nonredundant
    -> COMPLEMENTARY POLICY
    -> FIRST LATENT BIRTH ALLOWED

For a later candidate, all applicable criteria above
and LCB95(Delta V_repertoire) > 0
    -> ADDITIONAL LATENT BIRTH ALLOWED
```

Repeated `C3_PASS -> O3 improvement -> O3 dominance` diagnoses optimization
headroom, not latent demand. Repeated failure to produce preference reversal
or repertoire gain redirects investigation toward opponent/scenario pressure
or fidelity-correct commitment mechanics, not entropy tuning or latent
diversity regularization.

## Must freeze before demand evaluation (checklist)

1. Context definitions and frozen context frequencies (from discovery data).
2. Payoff statistic \(M[c,\pi]\) (win rate / margin / utility) and episode
   counts per cell.
3. D1 bootstrap procedure (clustering, resamples, seed).
4. D2 effect sizes and CI requirements per direction.
5. D3 competence floor per context.
6. D4 behavioral-distinction metric and threshold (matched-state; not
   occupancy).
7. `H_response` for response-mode branching.
8. Fresh demand-evaluation seed block (disjoint, fail-closed allocation).
9. For later births, the existing repertoire definition, the joint context
   cells used for both old and expanded pools, and the clustered-bootstrap
   procedure for `Delta V_repertoire`.

## Immutability

Once `DEMAND_GATE_FROZEN.json` is written and any demand-evaluation episode
has run, no requirement, threshold, context definition, frequency weighting,
or interpretation rule in this document may change.

## Execution authorization (future hard guard)

When a demand-gate runner exists, it must refuse evaluation until
`artifacts/environment_demand_gate/DEMAND_GATE_EXECUTION_AUTHORIZATION.json`
exists with `status=FROZEN_AND_AUTHORIZED` and matching contract / protocol /
runner hashes — the same pattern as C3's
`C3_EXECUTION_AUTHORIZATION.json`. Do not write that artifact until O3 exists
and the demand-gate freeze checklist is closed.
