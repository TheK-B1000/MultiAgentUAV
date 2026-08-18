# HUMAN_DECISION_REQUIRED — M1 payoff assay

Generated 2026-08-17T15:36:57Z. Updated 2026-08-17 after the post-block recovery test.

## Semantic recovery: **PASS**

The missing transition is implemented:

```
carrier home, own flag away  →  score blocked
own flag returns to the stand, carrier still legally home  →  capture scores
```

A first draft of the test that only teleported `blue_flag_pos` home while red still carried the flag was **not** a return (the engine snaps a carried flag onto the carrier before the M1 check). With a real return (flag on the stand, nobody carrying it), recovery scores. So the Gate B assay is a **scientific FAIL**, not `INVALID_IMPLEMENTATION`.

## Gate B: **FAIL** (accepted, thresholds frozen)

- OP6 GUARD − BREACH: ΔWR=+0.094, LCB95=-0.094 → FAIL
- OP7 BREACH − GUARD: ΔWR=+0.531, LCB95=+0.375 → PASS

M1 alone does not make OP6/OP7 pass Gate B.

## Frozen

```
RULESET_V3_M1
own_flag_home_required_to_score = True
everything else unchanged
```

Do not retune M1. Do not move the floor. Do not add cooldown/respawn/channel.

## Screen (development, not Gate B): **FINISHED**

Unique positive-ΔG A in the existing legal pool is OP6:

| A | ΔG | n | gap (intent−commit) | 0–0 |
|---|-----|---|---------------------|-----|
| OP6 | **+0.188** | 16 | −7.3 | 72% |
| OP8 | −0.875 | 8 | +202 | 44% |
| OP9 | −0.375 | 8 | +144 | 63% |
| OP10 | −0.625 | 8 | +130 | 69% |
| OP11 | −0.500 | 8 | +190 | 25% |
| OP12 | −0.375 | 8 | +50 | 0% |

OP6 is frozen as **GUARD_PAYOFF_CANDIDATE** only (canonical genome, no overlay). It is a reference/parent, not a confirmation candidate.

## Decision (2026-08-17)

**Do not launch `2500001`.** It stays pristine.

The static board has no complete package (ΔG>0.15 AND t_intent>t_commit AND non-degenerate). Mutation/evolution is authorized under the frozen J. Descendants get new genome IDs. OP6 is not rewritten.

PPO still off. Confirmation only after a descendant is development-eligible on the frozen pieces (promote ΔG, positive gap, J>0).
