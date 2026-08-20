# ACCIDENTAL_VANILLA_CONTINUATION_1P5M — retired from the SAPPO lineage

These are **legitimate PPO checkpoints** but they are **not SAPPO results**, and
they must not be scored on `7800001..7800192`.

## What they are

`pi_A` and `pi_B` continued from their R1 1M terminals to 1,501,184 steps with
the SAPPO anchor runner *attached but never executed*. The run is therefore a
plain vanilla PPO continuation.

    STATUS: INVALID_TREATMENT_INSTANTIATION
    (not "SAPPO FAILED" -- the defining treatment was never applied)

## Why the treatment was absent

`PPOUpdater.__init__` cached `self.anchor_runner = getattr(runtime,
"sappo_anchor_runner", None)`. The updater is constructed inside
`build_trainer()`, which runs **before** the orchestrator attaches the runner.
The cached value was therefore always `None` and the anchor branch executed
zero times.

That is the definitive evidence. The diagnostics below are corroborating, not
the basis of invalidation.

## Corroborating diagnostics (descriptive only)

Measured on the anchor train split, n=256, comparing the R1 1M terminal against
this 1.5M checkpoint:

| pole | teacher agreement | anchor loss |
|---|---|---|
| A | 0.214 -> 0.146 | 8.42 -> 14.20 |
| B | 0.089 -> 0.095 | 6.30 -> 12.15 |

Anchor loss ROSE on both poles. Rehearsal minimises exactly that quantity, so
an increase is inconsistent with rehearsal having run.

There is a genuinely interesting observation here: vanilla continuation caused
the policies to drift **further away** from the validated teachers rather than
recovering toward them. That strengthens the descriptive rationale for
rehearsal. It must NOT be used to change the SAPPO gate, and it does NOT
constitute a formal finding that extra budget fails -- R2b was prospectively
cancelled and is not being resurrected.

## Why they are not scored

Scoring these on `7800001..7800192` would quietly resurrect the cancelled R2b
experiment (does more budget help?) using a block reserved for SAPPO. The block
stays untouched for the corrected run.

## Why the SAPPO rerun restarts from the 1M terminals

The intended contrast is:

    R2 vanilla specialist at 1M   vs   the SAME 1M specialist + 500k SAPPO

Continuing from these 1.5M checkpoints would change both the starting policy and
the cumulative budget, so a positive result could not be attributed to
rehearsal rather than to another 500k of vanilla learning. The compute is lost;
attribution is worth more.

## Fixes made before the rerun

* the runner is read at USE time, never cached at construction
* a per-minibatch fail-fast invariant aborts the run if
  `n_anchor >= floor(n_ppo/4)` is ever violated
* anchor counters, ratio and loss are written to the metrics CSV from the first
  reporting interval
* the string-presence wiring test was replaced with a lifecycle test that
  reproduces the exact construct-then-attach ordering that broke production
