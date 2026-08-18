# RESOLVED — observability estimator under censoring

> **Ruled 2026-08-18.** Option 3: the statistic is replaced prospectively with an
> event-order gate. The confirmation was NOT rescued. See
> `OBSERVABILITY_V2_FROZEN.json` and `SEARCH_OBJECTIVE_V2_FROZEN.json`.
> Applied descriptively, V2 would not have passed this candidate either
> (p_C = 0.656, LCB95 = 0.500, gate is strictly greater).

**Status of the run: settled. `V3_STRATEGIC_DEMAND_NOT_VALIDATED`.**
This document does not reopen that verdict. It flags a protocol-design question
that the confirmation exposed, so the answer can be frozen *before* it can be
chosen to suit a result.

## What happened

Gate B required three criteria. Two passed decisively on untouched seeds
`5000001..5000032`:

| criterion | value | floor | LCB95 | verdict |
|---|---|---|---|---|
| A pole: WR(GUARD, SDS_G1_4) − WR(BREACH, SDS_G1_4) | +0.4688 | 0.15 | +0.3125 | **PASS** |
| B pole: WR(BREACH, OP7) − WR(GUARD, OP7) | +0.4688 | 0.15 | +0.3125 | **PASS** |
| observability: mean(t_intent − t_commit) | −17.406 | > 0 | — | **FAIL** |

The third failed, so the package fails. A partial pass is not a pass.

## The design question

`t_intent` and `t_commit` are both censorable — either event may never occur in
an episode. The frozen searcher codes a censored value as `MAX_STEPS + 1 = 241`
and takes the mean. That convention was inherited into the confirmation
deliberately, because it is what produced the development value (+16.625), and
changing it would have made development and confirmation incomparable.

The consequence, visible only now:

- 25 of 32 episodes had both events observed → mean **+5.36**, median +4,
  76% positive
- 5 episodes had `t_commit` censored: BREACH never got both agents onto RED's
  half. `t_intent` was observed early, so the coded gap is ≈ **−230**
- 2 episodes had `t_intent` censored → coded ≈ **+230**
- censored-coded median is **+2.5**; 65.6% of episodes are positive; the mean
  is **−17.4**

So the failure is produced by five episodes in which *the BREACH allocation
never actually committed*. The statistic assigns them a 230-step "intent came
first" reading, when what they actually record is that the treatment did not
instantiate.

This is the same species of problem as `7D_OP6 = INVALID_TREATMENT_INSTANTIATION`,
but partial: 5 of 32 episodes, not the whole arm.

## Why this was not decided autonomously

Choosing an estimator after seeing which one passes is exactly the move the
campaign's discipline forbids. Every alternative here — complete-case mean,
median, censoring-aware survival comparison, or excluding non-instantiated
episodes — flips this criterion from FAIL to PASS. That is precisely why the
choice cannot be made now on this data.

The verdict recorded is FAIL under the frozen convention.

## What is being asked

A **prospective** ruling, to apply to future confirmations only:

1. **Keep the 241-censoring mean.** Simple, already frozen, comparable to all
   prior numbers. Accepts that the statistic has unbounded sensitivity to
   censoring and that a handful of non-instantiated episodes can dominate it.

2. **Treat non-instantiated episodes as invalid rather than extreme.** Score
   observability on episodes where the treatment actually instantiated, and
   report the instantiation rate alongside it as its own diagnostic. Consistent
   with the 7D precedent, but it is a new rule and must be frozen before use.

3. **Replace the statistic entirely** with something censoring-aware, and
   re-derive the development threshold under it.

A separate, independent note: `mean_intent_minus_commit` was used as a *search
gate* (`precommitment_uncertain`) across the whole SDS run. With n=8 and n=16 and
unbounded ±230 outliers, that gate was fragile — the development value of
+16.625 and the confirmation value of −17.406 are the same construction on
different seeds. Whatever is decided above should be applied to the search
criterion too, or the searcher will keep promoting candidates on a statistic
that does not replicate.

## Not blocked on this

The payoff-reversal result stands on its own and does not depend on the ruling:
opposite allocations are strictly better against the two poles, replicated on a
fresh audited block, LCB95 = +0.3125 against a 0.15 floor. That is the first
replicated reversal this campaign has produced.

PPO, specialists, selector, and latent policies remain OFF: `V3_STRATEGIC_DEMAND`
is not validated.
