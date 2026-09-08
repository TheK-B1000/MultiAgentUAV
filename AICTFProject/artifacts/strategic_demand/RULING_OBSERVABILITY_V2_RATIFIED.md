# RULING — Observability V2 ratified

**Ratified 2026-08-18.** Frozen before any candidate has been evaluated under
the new objective, so no result can have influenced it.

```text
HUMAN DECISION — OBSERVABILITY V2

RATIFIED:

Observability V2 confirmation uses n=64.

Primary statistic:
    p_C = P(COMMIT_FIRST)

Episode classifications:
    COMMIT_FIRST
    INTENT_FIRST
    UNRESOLVED

Ties:
    INTENT_FIRST

UNRESOLVED:
    retained in denominator;
    does not count as COMMIT_FIRST.

Acceptance:
    LCB95(p_C) > 0.50

No sentinel timing arithmetic.
Timing gaps are telemetry only.

Do not reinterpret the completed SDS_G1_4 confirmation.
It remains:
    TWO_WAY_PAYOFF_REVERSAL = CONFIRMED
    V3_STRATEGIC_DEMAND = NOT_VALIDATED

Descriptive Observability-V2 reanalysis of SDS_G1_4:
    p_C = 0.656
    LCB95 = 0.500
    FAIL under strict >0.50 criterion.

OP7 remains unchanged.

Next search targets A only:
    preserve GUARD payoff pressure,
    increase p_C,
    decrease degeneracy.

Suggested development ladder:
    n=16 -> n=32 -> freeze
Untouched confirmation:
    n=64

No PPO, specialists, selector, FP/DO,
or latent training until V3_STRATEGIC_DEMAND is validated.
```

## Why n=64

| n | p_C=0.60 | p_C=0.70 | p_C=0.75 | p_C=0.80 |
|---|---|---|---|---|
| 32 | 0.23 | 0.60 | 0.87 | 0.95 |
| **64** | 0.28 | **0.89** | **0.98** | 1.00 |
| 96 | 0.46 | 0.98 | 1.00 | 1.00 |

400 simulated confirmations per cell, scored with the frozen gate. n=96 buys
about nine further points at p_C = 0.70 for 50% more episodes; not earned yet.

## No-extension rule

Confirmation is 64 from the start. No peeking at 32 and deciding to extend.
A rescue extension would turn a fixed-n test into an optional-stopping
procedure and destroy the error rate the gate is supposed to control.

## Why the metric change is not motivated reasoning

The obvious objection to replacing an estimator after a failure is that the
replacement was chosen to pass the candidate that failed. It was not, and the
record shows it:

> Applied descriptively to `SDS_G1_4`, **V2 also fails it** — p_C = 0.656 with
> LCB95 = 0.500 against a strictly-greater gate.

The improved metric rejects the same candidate. What it adds is a diagnosis
rather than a verdict: intent is readable in 93.8% of episodes against
`SDS_G1_4` versus 34.4% against OP7.

## What is now implemented

| piece | location | status |
|---|---|---|
| event-order estimator | `experiments/observability_v2.py` | frozen |
| synthetic validation | `experiments/test_observability_v2.py` | 12/12 pass |
| V2 candidate evaluation | `experiments/sds_eval_v2.py` | frozen |
| searcher rewiring | `experiments/strategic_demand_searcher_v2.py` | `STAGES=(16,32)`, V1 path unreachable |
| objective | `SEARCH_OBJECTIVE_V2_FROZEN.json` | frozen |
| ratified n and ladder | `OBSERVABILITY_V2_FROZEN.json` | frozen |

Payoff is enforced as a **constraint**, not a summand:
`J_v2 = p_C − degeneracy_penalty`, admissible only while `delta_G` clears the
stage payoff floor. A candidate that conceals beautifully but has stopped
carrying GUARD pressure is rejected outright and its evaluation is cut short
rather than being allowed to score well on concealment alone.

## Standing prohibitions

- No PPO, specialists, selector, FP/DO, or latent training until
  `V3_STRATEGIC_DEMAND = VALIDATED`
- OP7 unchanged; canonical registry OP6–OP12 never overwritten
- `RULESET_V3_M1` unchanged; BLUE responses unchanged
- 2v2 only — isolate mechanisms through telemetry, never by deleting agents
- Blocks 2500001 and 2600001 permanently disqualified; 5000001 spent
- No re-scoring of the completed `SDS_G1_4` confirmation
