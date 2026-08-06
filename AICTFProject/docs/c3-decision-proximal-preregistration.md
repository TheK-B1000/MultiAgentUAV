# C3 commitment-proximal strategic-fork discovery — draft preregistration

**Status:** DRAFT — **not frozen**. Do not run Stages 1–4 against this text as authoritative yet.
**Date:** 2026-08-06
**Machine-readable freeze target (not written):** `artifacts/c3_discovery/C3_DISCOVERY_PREREG_FROZEN.json`
**Motivated by:** `artifacts/c2_confirmation/C2_CONFIRMATION_FROZEN_RESULT.json` (`C2_REJECTED`)
**Companion gate:** [`environment-demand-gate-preregistration.md`](environment-demand-gate-preregistration.md)

This document incorporates the post-C2 design corrections. It is the working
spec to review before any scan. When frozen, every open item in §"Must freeze
before Stage 1" must be closed; after any scan produces numbers, no criterion
may change.

**Existing code caution.** `experiments/run_c3_decision_proximal_discovery.py`,
`rl/analysis/decision_proximal_features.py`, and
`rl/analysis/counterfactual_actionability.py` were written against a
superseded draft. They must be re-audited against this text before freeze;
their existence does not authorize a scan.

---

## C3 purpose and limits

> **Label:** `AUTHORITATIVE INTENDED WORDING, NOT YET FROZEN`.
> These are the intended scientific limits of C3. They are not frozen until
> this document's status flips to FROZEN and
> `artifacts/c3_discovery/C3_DISCOVERY_PREREG_FROZEN.json` exists. The runner
> additionally refuses to execute until
> `artifacts/c3_discovery/C3_EXECUTION_AUTHORIZATION.json` exists and verifies
> contract/prereg/runner hashes (see §Execution authorization).

```text
C3 PURPOSE:
Discover candidate commitment forks.

C3 DOES NOT ESTABLISH:
- latent necessity
- policy complementarity
- routing value
- distinct strategy families

A candidate can advance only to independent
task-reward response-oracle (O3) training.

LATENT ELIGIBILITY REQUIRES (decided by the Environment-Demand Gate,
never by C3):
1. response-oracle niche advantage
2. anchor/context preference reversal
3. positive LCB95 of G_available
4. competence of every retained policy
5. behavioral nonredundancy
6. fresh evaluation data
7. for later births, positive incremental repertoire value beyond the
   existing selective pool

Failure of any item:
NO LATENT BIRTH
NO ROUTER
```

Rationale: a positive per-state improvement
\(A(s)=\max_{a'}\mathbb{E}[U|s,a']-\mathbb{E}[U|s,G0]\)
means "G0 could have done something better here." It does **not** mean
"another persistent policy should own this state" — a sufficiently expressive
generalist may simply learn the correction. C3's one-macro deviation test is
evidence of **controllability**, not strategy demand. Strategy-demand testing
(bounded `H_response` response-mode branching, payoff matrix \(M[c,\pi]\),
`G_available`) is owned by the demand-gate preregistration.

---

## Why this exists

C1 and C2 both failed as specialist training targets:

| Candidate | Natural support | Headroom | Actionability | Verdict |
|---|---|---|---|---|
| C1 (`home_threatened_while_leading`) | PASS | FAIL (~0.9%) | n/a (C1-era) | not a trainable niche |
| C2 (`none_forward_frac` @ fresh `9800001+`) | PASS (~87% onset prevalence) | FAIL (~12%) | FAIL (0.0) | `C2_REJECTED`; O2 do not train |

Both were **correlates of bad outcomes**, not **genuine strategic decision forks**.
C2's confirmation block `9800001+` is spent: no reinterpretation of
`none_forward_frac`, no lag retune, no threshold relax, no runner-up promotion
on that block.

The aggregate-window / lag-band pipeline is structurally biased toward
outcome-correlated signals. C3 pivots to:

> At an authoritative decision state \(s_t\), does G0 have more than one
> **strategically useful** legal response?

---

## Locked pipeline (intended freeze shape)

```text
C2_REJECTED
      ↓
C3 Stage 1  — natural CARRIER_PRESSURE_ONSET discovery (9400000+ replay)
              instantaneous geometry at onset
              + backward trace toward earliest commitment fork
      ↓
C3 Stage 2  — event-anchored temporal qualification
              (NOT C2 lag bands)
              min lead time; matched opportunities; score controls
      ↓
C3 Stage 3  — snapshot exact natural onset
              exhaustive LEGAL response branches
              A(s) = max improvement over G0 baseline (not absolute shift)
              30-step evaluation horizon; brief fork force only
              (controllability screen, NOT strategy-demand evidence)
      ↓
controllability clears frozen threshold?
      ├── NO  → C3_NO_QUALIFIED_STRATEGIC_FORK ; STOP
      └── YES → freeze selected fork
                  ↓
C3 Stage 4  — FRESH 9810000+ natural confirmation
              AND fresh counterfactual replication
                  ↓
            confirmed?
                  ├── NO  → STOP
                  └── YES → freeze O3 protocol
                            G0 natural prefix → exact C3 onset
                            → O3 post-onset PPO credit only
                            → ONE canonical development seed
                              ↓
                            payoff matrix M[c,π] on fresh demand-eval seeds
                              ↓
                            ENVIRONMENT-DEMAND GATE (separate prereg)
                              ↓
                            only if PASS → latent birth
                            later births additionally require positive
                            incremental repertoire value
```

Still asking the environment:

> Where does G0 encounter a state in which another **legal** team response
> demonstrably produces a **better** outcome?

Not declaring “escort” / “retreat” by hand — and not concluding "latent
needed" from C3 alone.

### Commitment fork definition — item 9 SETTLED

> **Label:** item 9 of the freeze checklist is **SETTLED** below.
> The overall C3 document remains **DRAFT** until items 1–8 and 10 close.
> Do not authorize a scan from this section alone.

Pressure onset is an **anchor / symptom locator**, not the fork.
C3 must identify the **earliest** upstream commitment fork, not celebrate
the later pressured state.

#### Four requirements (all required)

A candidate commitment fork \(s_{t^\star}\) must satisfy:

```text
1. Naturally reached
   s comes from an unmodified G0 rollout on map_a.
   No injected scenario, no forced prefix, no hand-placed agents.

2. Legally meaningful alternatives exist
   At that decision, at least two distinct legal team responses
   remain available (carrier and/or mate macros after legality pruning).

3. Upstream of failure
   t* occurs strictly before the later carrier-pressure / failure event
   that anchors the backward trace. The trajectory must not already be
   effectively doomed at t* (outcome still contingent on subsequent play).

4. Commitment matters over time
   A bounded alternative response from s_{t*} must create a measurable
   difference in expected task utility U over horizon H_response —
   not merely change one immediate action then reconverge.
```

`U`, `H_response`, and the measurable-difference threshold \(\delta\) are
numeric cells closed with freeze items 3 / 4 / 6; their **roles** in
requirement 4 are fixed here.

#### Backward-trace algorithm (authoritative)

```text
later carrier-pressure event at t_pressure
        ↓
walk backward through prior decisions on the same natural trajectory
        ↓
find earliest state t* where ALL of:
  (R1) state is naturally reached (same G0 rollout)
  (R2) multiple legal team responses exist
  (R3) t* < t_pressure  (upstream of the pressure/failure event)
  (R4) bounded alternative response yields measurable E[U] divergence
       that persists over H_response
        ↓
that state s_{t*} = candidate commitment fork
```

The operative word is **earliest**. If pressure is at `t=78` but meaningful
divergence begins at `t=52`, C3 must report `t=52`, not `t=78`.

Trace window: walk back at most `T_trace` decisions before `t_pressure`
(`T_trace` numeric cell closes with item 2 / 9 jointly; proposed default to
freeze with the rest of the contract: `T_trace = 40`).

If no earlier state satisfies R1–R4, the pressure event has **no qualified
upstream commitment fork** for that episode (do not promote the pressure
onset itself as the fork).

#### What this forbids

- Treating carrier-pressure onset as the fork by default
- Selecting a later correlated symptom when an earlier diverging state exists
- Counting one-step action flips that reconverge within `H_response`
- Injected / scripted scenarios as discovery population
- Declaring a fork where only one legal team response remains

#### Path after item 9 (do not skip)

```text
item 9 SETTLED (this section)
        ↓
item 10 — audit old C3 code against this definition
        ↓
close remaining freeze items 1–8
        ↓
freeze final C3 contract (human + machine-readable)
        ↓
write C3_EXECUTION_AUTHORIZATION.json
        ↓
smoke
        ↓
full scan
```

**Not authorized:** run C3, train O3, latent birth, or router work.

---

## Seed blocks

| Stage | Seeds | Role |
|---|---|---|
| 1 discovery | `9400000+` (existing G0-V5 discovery / Stage-2 replay) | natural onset + geometry |
| 2 temporal qual | same `9400000+` replay | no fresh block |
| 3 counterfactual branching | same `9400000+` onset snapshots | no fresh block |
| 4 confirmation | **`9810000+` only** | fresh natural **and** fresh counterfactual |

Do **not** spend a fresh natural-trajectory block before Stage 4.
Block `9800001+` is spent by C2 confirmation and must not be reused for C3.

---

## Authoritative onset (Stage 1 population)

**Do not** mix flag pickup with defender-pressure into one population.

For the first C3 search, the only onset family is:

```text
CARRIER_PRESSURE_ONSET:

  BLUE is carrying
  AND previous decision was NOT under actionable carrier pressure
  AND current decision crosses into actionable carrier pressure
```

`actionable carrier pressure` must be an authoritative predicate from
**geometry + tag readiness** (nearest ready defender, closing geometry,
intercept feasibility) — not a vague radius alone, and not
`global_state[i]` magic indices if named env fields exist.

Flag pickup is a **separate event family** for a later search, not mixed here.

Intended fork narrative:

```text
safe-ish carrier state
        ↓
pressure becomes strategically relevant
        ↓
team has a decision
        ↓
eventual tag / escape / capture
```

---

## Decision-proximal features (at \(t_0\) only)

All primary features are **instantaneous** at onset \(t_0\). Optional
pre-onset dynamics \(t_0-5 \ldots t_0\) may define **trend** features only;
they do not replace the event-anchored unit.

| Feature | Corrected definition notes |
|---|---|
| `time_to_intercept` | If closing velocity \(> \varepsilon\): `dist / closing_vel`. Else sentinel `H+1` (or documented cap). **No** `dist/0.15+20` pseudo-ETA when not closing. |
| `relative_closing_velocity` | \((v_{\mathrm{defender}}-v_{\mathrm{carrier}})\cdot\hat{d}\) with explicit sign convention — not BLUE velocity alone. |
| `carrier_dist_home` | From named carrier / home positions (not opaque global_state indices). |
| `nearest_ready_defender_dist` | Nearest red with tag ready. |
| `escort_dist` | Carrier to nearest blue teammate (named state). |
| `cooldown_remaining` | Carrier tag cooldown. |
| `carrier_progress_frac` | Documented transform of home distance if kept. |
| `pressure_trend` | Optional dynamics over \(t_0-5..t_0\) only. |
| `signed_commitment` | `n_attackers - n_defenders` (**keep sign**). Absolute imbalance may be secondary only. |
| `mate_intervention_eta` | Mate reach time under documented motion model. |
| `intercept_margin` | `time_to_intercept - mate_intervention_eta` when both defined. |
| `agents_forward` / `formation_spread` | Instantaneous; secondary. |

---

## Stage 2 — Event-anchored temporal qualification

**Replace** C2-style `[-30,-20), [-20,-10), [-10,0)` lag-band qualification.

```text
t0 = CARRIER_PRESSURE_ONSET

feature vector evaluated at t0
optional dynamics t0-5 … t0 for trend features only

require:
  outcome occurs >= L decisions after t0
```

Then test whether the state predicts differential outcomes **among matched
carrier-pressure opportunities** (opportunity matching + score strata), with
information guaranteed **before** the outcome — without averaging into
another arbitrary 10-step bucket.

`L` and matching rules must be frozen before Stage 1 (see open items).

Horizon diagnostic: report H=15 as a **preregistered diagnostic only**; it
does **not** decide PASS/FAIL. Primary evaluation horizon is **H=30**.

---

## Stage 3 — Counterfactual controllability screen (critical)

**Interpretation limit (`AUTHORITATIVE INTENDED WORDING, NOT YET FROZEN`).**
A Stage 3 pass means the state is **controllable** — a better legal one-macro
alternative exists. It is NOT evidence that a persistent second policy should
own the state. Strategy-mode evidence (G0 continuation vs O3 continuation for
a prospectively frozen `H_response`, from the same natural snapshot) belongs
to the Environment-Demand Gate after O3 exists, not to C3.

### Branch semantics

```text
snapshot s_t at natural onset
      ↓
baseline: G0 chooses its normal response → H=30 evaluation rollout
      ↓
counterfactual: force alternative response AT THE FORK only
      ↓
return to normal G0 policy
      ↓
H=30 evaluation rollout
```

- **H=30** is the **evaluation horizon**, not the forced-action duration.
- Forcing `GO_HOME` / `GO_TO` / etc. for 30 consecutive decisions installs a
  hand-coded mini-specialist — forbidden as the PASS/FAIL definition.
- If macros inherently persist for \(k\) env ticks after one decision, freeze
  exactly what “one response” means (one macro selection at the fork;
  persistence = environment semantics, not extended override).

### Response set

**Exhaustive legal macro-actions** (option a, corrected):

- Not top-2 policy actions (would hide low-prob complementary responses).
- Not a hand-curated tactical subset (would smuggle escort/retreat in by hand).
- `n_macros = 5`: `GO_TO`, `GRAB_MINE`, `GET_FLAG`, `PLACE_MINE`, `GO_HOME`.
- For two relevant BLUE agents: carrier unilateral, mate unilateral, and
  joint legal carrier×mate combinations (≤25 before legality pruning).

### Improvement definition (not absolute shift)

Do **not** use max absolute outcome change.

\[
A(s)=\max_{a' \in A_{\mathrm{legal}}}
\mathbb{E}[U \mid s,a'] - \mathbb{E}[U \mid s,a_{\mathrm{G0}}]
\]

Only a **better** legal alternative counts.

```text
state actionable  iff  A(s) >= δ
```

Proposed freeze target (new metric; not C2's 0.30 observational actionability):

```text
δ = 0.10   (primary-outcome improvement vs G0 baseline)
aggregate: >= 20% of eligible onset states actionable
+ preregistered uncertainty / replication on the aggregate rate
```

Report H=15 diagnostic under the same \(A(s)\) definition; not decisive.

### Probability vs single rollout

A single continuation is **not** a probability shift. Choose and freeze one:

1. **Deterministic branching** — documented near-term utility; compare branch
   outcomes once per (state, action); or
2. **Preferred: stochastic continuation** — from each (snapshot, action), run
   \(N\) controlled continuation seeds; estimate `P(survive H)`, `P(capture H)`,
   `P(tagged H)`. Freeze \(N\) and continuation-seed assignment **before** the
   scan.

Primary outcome \(U\) (e.g. survive / not-tagged / capture / score utility)
must be named before Stage 3.

---

## Stage 4 — Fresh confirmation (both legs)

If Stage 3 passes on `9400000+`, Stage 4 on `9810000+` must replicate:

1. **Fresh natural** strategic-onset effect (observational), **and**
2. **Fresh counterfactual** useful-response effect (same \(A(s)\) / thresholds).

Observational-only confirmation on fresh seeds is insufficient (would allow
overfitting the counterfactual result to discovery states).

Required replication: ≥2/3 policies (exact cells frozen with Stage 4).

A Stage 4 pass authorizes **O3 training only**. Latent birth additionally
requires the Environment-Demand Gate
([`environment-demand-gate-preregistration.md`](environment-demand-gate-preregistration.md)):
payoff matrix \(M[c,\pi]\) on fresh demand-evaluation seeds, preference
reversal, `LCB95(G_available) > 0`, competence floor, and behavioral
nonredundancy. Later births also require
`LCB95(Delta V_repertoire) > 0` beyond the existing selective pool. C3
results are never cited as latent-necessity evidence.

---

## Stopping rules

```text
zero candidates Stage 1  → C3_NO_DISCOVERY_SIGNAL.json, STOP
zero candidates Stage 2  → C3_NO_TEMPORAL_QUALIFICATION.json, STOP
zero candidates Stage 3  → C3_NO_QUALIFIED_STRATEGIC_FORK.json, STOP
Stage 4 fail             → C3_CONFIRMATION_REJECTED.json, STOP
demand gate fail (post-O3) → NO LATENT BIRTH, NO ROUTER
```

Do not invent a niche. `NO_QUALIFIED_STRATEGIC_FORK` is a legitimate
publishable negative result.

---

## Explicitly rejected (lessons from C1/C2)

- Reusing C2 lag bands as the Stage 2 qualification unit
- Mixing flag-pickup onsets with pressure onsets
- Absolute outcome-shift “actionability”
- Forcing an alternative macro for the full H-step horizon as the gate
- Top-2 or hand-curated action sets
- Promoting runner-ups or relaxing thresholds on spent confirmation blocks
- Training O3/O2 before Stage 4 dual confirmation
- Citing C3 controllability as latent-necessity / complementarity evidence
- Latent birth or router work without a demand-gate PASS on fresh data

---

## Must freeze before Stage 1

Open items — **do not implement Stage 1 until items 1–8 and 10 are closed and
this document's status flips to FROZEN** with a matching
`C3_DISCOVERY_PREREG_FROZEN.json`:

1. Exact `actionable carrier pressure` predicate (geometry + tag readiness;
   radius / ε / closing tests).
2. Minimum lead time \(L\); opportunity matching; score strata; cell minima;
   and numeric `T_trace` (proposed default 40) for the item-9 backward walk.
3. Primary utility \(U\); deterministic vs stochastic design; if stochastic, \(N\)
   and continuation-seed schedule; measurable-divergence threshold \(\delta\)
   used by item-9 requirement 4.
4. Exact “one response” / `H_response` semantics given macro persistence ticks
   (item-9 requirement 4 depends on this).
5. Legality function for joint carrier×mate macros (item-9 requirement 2).
6. Aggregate actionability / fork-rate uncertainty / replication rule (not
   point estimate alone).
7. Stage 4 fresh natural + fresh counterfactual acceptance cells (2/3 etc.).
8. Feature list finalization (sign conventions; named-state accessors only).
9. **SETTLED (2026-08-06)** — commitment-fork definition: four requirements
   (natural / legal alternatives / upstream of failure / commitment over
   `H_response`) and earliest-state backward trace from pressure onset.
   See §"Commitment fork definition — item 9 SETTLED". Numeric cells
   (`T_trace`, `H_response`, \(\delta\), \(U\)) close with items 2–6.
10. Re-audit of the existing C3 code
    (`experiments/run_c3_decision_proximal_discovery.py`,
    `rl/analysis/decision_proximal_features.py`,
    `rl/analysis/counterfactual_actionability.py`) against the settled
    item-9 definition and the rest of this draft; the code predates it.
    **Next authorized work after this edit.**

Order after item 9: audit (10) → close 1–8 → freeze final contract →
authorization artifact → smoke → full scan. **Do not run C3 yet.**

---

## Execution authorization (hard guard)

Documentation alone is not enough. The discovery runner refuses to perform
even one rollout unless this artifact exists:

```text
artifacts/c3_discovery/C3_EXECUTION_AUTHORIZATION.json
```

Required fields (verified by the runner before any env construction):

```text
status              = "FROZEN_AND_AUTHORIZED"
c3_contract_hash    = sha256 of the frozen machine-readable contract
                      (C3_DISCOVERY_PREREG_FROZEN.json)
c3_prereg_commit    = git commit that froze the human-readable prereg
runner_commit       = git commit of the authorized runner
authorized_utc      = ISO-8601 UTC
```

Mismatch of any hash/commit, or missing file → `SystemExit` with an explicit
DRAFT / NOT AUTHORIZED message. **Do not write this artifact until** the
freeze checklist is closed, both human-readable and machine-readable
contracts are frozen and committed, and a deliberate authorization step is
taken. Absence of the file is the correct default state.

The Environment-Demand Gate runner will eventually use the same pattern
(`DEMAND_GATE_EXECUTION_AUTHORIZATION.json`).

---

## Immutability (applies only after freeze)

After any scan produces numbers under a FROZEN copy of this protocol, no
criterion, threshold, onset definition, action set, \(A(s)\) definition,
horizon, or ranking rule may change.
