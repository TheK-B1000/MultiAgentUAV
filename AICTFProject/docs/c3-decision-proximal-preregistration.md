# C3 commitment-proximal strategic-fork discovery — preregistration

**Status:** FROZEN 2026-08-06  
**Machine-readable:** `artifacts/c3_discovery/C3_DISCOVERY_PREREG_FROZEN.json`  
**Motivated by:** `artifacts/c2_confirmation/C2_CONFIRMATION_FROZEN_RESULT.json` (`C2_REJECTED`)  
**Companion gate:** [`environment-demand-gate-preregistration.md`](environment-demand-gate-preregistration.md)  
**Item-10 audit:** [`c3-item10-code-audit.md`](c3-item10-code-audit.md)

After this freeze, no criterion, threshold, onset definition, action set,
\(A(s)\) definition, horizon, or ranking rule may change once any authorized
scan episode has run. Execution additionally requires
`artifacts/c3_discovery/C3_EXECUTION_AUTHORIZATION.json`.

---

## C3 purpose and limits

> **Label:** FROZEN.

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

## Locked pipeline

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

### Commitment fork definition — item 9 CLOSED

> **Label:** FROZEN (item 9).

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
frozen in §"Frozen runtime cells" and
`artifacts/c3_discovery/C3_DISCOVERY_PREREG_FROZEN.json`.

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

Trace window: walk back at most `T_trace = 40` decisions before `t_pressure`
(frozen; see §"Frozen runtime cells").

If no earlier state satisfies R1–R4, the pressure event has **no qualified
upstream commitment fork** for that episode (do not promote the pressure
onset itself as the fork).

#### What this forbids

- Treating carrier-pressure onset as the fork by default
- Selecting a later correlated symptom when an earlier diverging state exists
- Counting one-step action flips that reconverge within `H_response`
- Injected / scripted scenarios as discovery population
- Declaring a fork where only one legal team response remains

---

## Frozen runtime cells

All values below are authoritative. The runner refuses to start unless
`C3_DISCOVERY_PREREG_FROZEN.json` supplies matching `runtime_cells` (no code
defaults for these quantities).

| Cell | Frozen value | Rationale |
|---|---|---|
| `T_trace` | **40** | Prior draft default; covers the t≈40–78 commitment window example without inventing new machinery |
| `H_response` | **30** | Same near-term horizon used in C2/eval rollouts (`EPISODE`-style decision horizon window) |
| `delta` | **0.10** | Prior draft improvement threshold for useful alternative vs G0 |
| `minimum_fork_rate` | **0.20** | Explicit aggregate discovery gate (was previously only a code constant; now contract-owned) |
| `U.name` | **`carrier_survival`** | Primary C3 outcome is whether the carrier remains untagged/carrying after the response horizon; existing recorded branch outcome |
| `U.doomed_at_or_below` | **0.0** | If even the best legal team response has survival utility ≤ 0, the state is effectively doomed (R3) |
| `U.estimation` | **deterministic single continuation** | Discovery controllability screen; stochastic \(N\) reserved for Stage 4 / demand gate |
| `L_min` | **1** | Fork must be strictly before pressure (`t* < t_pressure`); at least one intervening decision |
| Pressure predicate | BLUE carrying AND rising edge into `dist_nearest_red/cols ≤ 0.18` | Implemented `PRESSURE_RADIUS_FRAC`; pickup excluded; tag-readiness remains a feature not the onset gate |
| Legal responses | Cartesian product of authoritative blue macro masks from `core._build_action_mask` | Exhaustive legal team macros; no top-2 / curated set |
| One response | Override team macros for **one** decision, then return to natural G0 for `H_response` | Macro persistence ticks = env semantics, not extended force |
| Stage 4 seeds | **`9810000+`** | Fresh natural + fresh counterfactual; block `9800001+` spent by C2 |
| Stage 4 replication | **≥2/3 policies** | Same cells / thresholds as discovery; dual natural+CF confirmation |

### Aggregate fork-rate gate (explicit)

```text
minimum_fork_rate = 0.20

Discovery Stage 3 PASS (controllability screen) requires:
  (# anchors with QUALIFIED_COMMITMENT_FORK) / (# pressure anchors)
  >= 0.20
  on the evaluated policy set under this frozen contract.

This threshold is owned by the frozen contract, not by runner defaults.
```

---

## Checklist — all items CLOSED

1. **CLOSED** — Pressure predicate: carrying + rising edge into radius 0.18; pickup excluded.
2. **CLOSED** — `T_trace=40`, `L_min=1`; matched score strata retained as descriptors.
3. **CLOSED** — `U=carrier_survival`, deterministic continuation, `delta=0.10`, `doomed_at_or_below=0.0`.
4. **CLOSED** — `H_response=30`; one-decision team-response force then natural G0.
5. **CLOSED** — Legal joint macros via authoritative action masks.
6. **CLOSED** — `minimum_fork_rate=0.20` (point estimate for discovery; CI reserved for Stage 4 / demand gate).
7. **CLOSED** — Stage 4: seeds `9810000+`, ≥2/3 policies, natural + counterfactual legs.
8. **CLOSED** — Features: named-state instantaneous geometry; signed commitment; relative closing velocity; non-closing TTI = `+inf` sentinel.
9. **CLOSED** — Commitment-fork definition (earliest R1–R4 backward trace).
10. **CLOSED** — Implementation patched + focused tests; see item-10 audit doc.

**Authorized path after freeze:** write `C3_EXECUTION_AUTHORIZATION.json` → smoke → full discovery scan.

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

**Interpretation limit (FROZEN).**
A Stage 3 pass means the state is **controllable** — a better legal one-decision
team response exists. It is NOT evidence that a persistent second policy should
own the state. Strategy-mode evidence belongs to the Environment-Demand Gate
after O3 exists, not to C3.

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
δ = 0.10   (frozen)
aggregate discovery gate: fork_rate >= minimum_fork_rate = 0.20  (frozen)
```

`U = carrier_survival` with **deterministic** single continuation (frozen for
discovery). Stochastic \(N\)-seed estimation is reserved for Stage 4 / the
Environment-Demand Gate and is not used to decide C3 discovery PASS/FAIL.

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
c3_prereg_sha256    = sha256 of docs/c3-decision-proximal-preregistration.md
authorized_utc      = ISO-8601 UTC
```

Mismatch of any hash/commit, or missing file → `SystemExit`.

---

## Immutability

After any authorized scan episode has run under this FROZEN protocol, no
criterion, threshold, onset definition, action set, \(A(s)\) definition,
horizon, or ranking rule may change.
