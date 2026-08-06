# C3 decision-proximal discovery — draft preregistration

**Status:** DRAFT — **not frozen**. Do not run Stages 1–4 against this text as authoritative yet.
**Date:** 2026-08-06
**Machine-readable freeze target (not written):** `artifacts/c3_discovery/C3_DISCOVERY_PREREG_FROZEN.json`
**Motivated by:** `artifacts/c2_confirmation/C2_CONFIRMATION_FROZEN_RESULT.json` (`C2_REJECTED`)

This document incorporates the post-C2 design corrections. It is the working
spec to review before any scan. When frozen, every open item in §"Must freeze
before Stage 1" must be closed; after any scan produces numbers, no criterion
may change.

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
      ↓
C3 Stage 2  — event-anchored temporal qualification
              (NOT C2 lag bands)
              min lead time; matched opportunities; score controls
      ↓
C3 Stage 3  — snapshot exact natural onset
              exhaustive LEGAL response branches
              A(s) = max improvement over G0 baseline (not absolute shift)
              30-step evaluation horizon; brief fork force only
      ↓
actionability clears frozen threshold?
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
```

Still asking the environment:

> Where does G0 encounter a state in which another **legal** team response
> demonstrably produces a **better** outcome?

Not declaring “escort” / “retreat” by hand.

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

## Stage 3 — Counterfactual actionability (critical)

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

---

## Stopping rules

```text
zero candidates Stage 1  → C3_NO_DISCOVERY_SIGNAL.json, STOP
zero candidates Stage 2  → C3_NO_TEMPORAL_QUALIFICATION.json, STOP
zero candidates Stage 3  → C3_NO_QUALIFIED_STRATEGIC_FORK.json, STOP
Stage 4 fail             → C3_CONFIRMATION_REJECTED.json, STOP
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

---

## Must freeze before Stage 1

Open items — **do not implement Stage 1 until these are closed and this
document's status flips to FROZEN** with a matching
`C3_DISCOVERY_PREREG_FROZEN.json`:

1. Exact `actionable carrier pressure` predicate (geometry + tag readiness;
   radius / ε / closing tests).
2. Minimum lead time \(L\); opportunity matching; score strata; cell minima.
3. Primary utility \(U\); deterministic vs stochastic design; if stochastic, \(N\)
   and continuation-seed schedule.
4. Exact “one response” semantics given macro persistence ticks.
5. Legality function for joint carrier×mate macros.
6. Aggregate actionability uncertainty / replication rule (not point estimate
   alone).
7. Stage 4 fresh natural + fresh counterfactual acceptance cells (2/3 etc.).
8. Feature list finalization (sign conventions; named-state accessors only).

---

## Immutability (applies only after freeze)

After any scan produces numbers under a FROZEN copy of this protocol, no
criterion, threshold, onset definition, action set, \(A(s)\) definition,
horizon, or ranking rule may change.
