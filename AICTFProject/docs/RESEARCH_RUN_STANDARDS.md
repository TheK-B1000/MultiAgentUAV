# Research Run Standards

Adopted 2026-09-12. Applies to every GPU experiment in this project from here on.

Goal: **every GPU hour answers a question, every long run is observable, every
result is recoverable.**

These rules exist because we lost real work to their absence. Each rule below
names the specific failure that motivated it, so the reason survives even when
the incident is forgotten.

---

## The fourteen rules

### 1. Live telemetry on every run
tqdm on every episode loop, with current cell/condition, seed, checkpoint,
elapsed and ETA. Long evaluators must print a result **as soon as a cell closes**.
Redirected / headless launches must still emit a **durable** bar (plain ASCII
tqdm when stderr is not a TTY — never silent `tqdm.rich` into an empty
`*.log.err`). Watching `Get-Content … -Wait` from a terminal counts as
observability; if the bar is invisible there, the rule is failed.

> *Motivating failure:* the 4v4 checkpoint-trajectory evaluator computed every
> checkpoint's Δ only after all 480 episodes. Four hours of finished science sat
> invisible in RAM, and a mid-run crash would have surfaced none of it.
> *Follow-on:* PPO `tqdm.rich` under Windows stderr redirect wrote **zero
> bytes** to `b_rule_role_cond.log.err`, so terminal log tails showed no bar.

### 2. Write incrementally, never at exit
Raw rows append after **each episode** (flush + `fsync`). Cell summaries write as
each cell closes. If the process dies at 83%, we keep 83%.

> *Motivating failure:* all three evaluators originally opened the rows CSV once,
> at program exit.

### 3. Frozen spec before any seed is spent
Record the question, hypothesis, intervention, checkpoints + hashes, seed block,
sample size, metrics, **both/all interpretation branches**, stopping rule, and an
explicit list of what the experiment may **not** claim. The question may not
change after the answer is seen.

### 4. Preflight and self-test before the real run
Verify checkpoint hashes, architecture/observation/action compatibility, device,
seed disjointness, output paths, and **intervention isolation**. A dry run must
prove the code modifies exactly what we believe it modifies.

> *Standard to copy:* the opponent-channel ablation self-test asserts that only
> grid channel 2 and vec feature 11 differ between FULL and ablated observations,
> that every other channel/feature and the action mask are bit-identical, that
> world truth is unmutated, and that uninstall restores truthful observation.
> It has caught two real bugs.

### 5. Resumability
Detect completed `(condition, seed)` rows and skip them. Hold a run lock so two
processes cannot write the same artifact. Restarting must never discard hours.
One-shot applies to the **sealed result**, not to a partial rows file.

### 6. A progress artifact, not log archaeology
`<LABEL>_LIVE_STATUS.json`, rewritten every episode: completed/total, current
condition, latest cell result, elapsed, ETA, sec/ep, checkpoint SHAs, PID,
device, heartbeat.

### 7. Automatic integrity checks before a result counts as sealed
Expected row count, unique seeds, exact seed range, binary outcomes, matched
pairing, no duplicate cells, checkpoint hashes, independent bootstrap
reproduction, split-half diagnostics where relevant, config fingerprint.
**A result is not sealed until its audit passes.**

> *Implemented 2026-09-12:* `experiments/run_state.py`. Sealing is a state
> transition — `RUNNING → COMPLETE → AUDITED → SEALED`, with `AUDIT_FAILED` as a
> terminal branch — and there is **no edge from `COMPLETE` to `SEALED`**. An
> evaluator calls `seal(...)`, which refuses a payload that sets `status` itself,
> re-derives every sealed statistic *from the persisted rows CSV*, and writes
> `AUDIT_FAILED` plus a full audit record if any gating check fails. Terminal
> states are terminal: a failed audit cannot be overwritten by re-running into
> the same label. 14 tests in `tests/test_run_state_sealing.py` prove the gate
> rejects tampered derived fields, truncated rows, duplicate seeds, stale
> statistics, substituted checkpoints, unfrozen specs and non-binary outcomes.
>
> Writing those tests immediately caught three defects in the gate itself,
> including a bare `except` that made the split-half check *silently vanish*
> from the audit record rather than fail — an audit that quietly carries one
> fewer check than the reader believes is worse than one that errors.
>
> **What a `SEALED` record does not mean:** the audit is a second implementation
> written to the same contract, so it catches stale, mis-paired, truncated or
> substituted data — but a methodological error made identically in the
> evaluator and the audit would reproduce and pass. `SEALED` means internally
> consistent, never *correct*. Rules 3 and 10 carry correctness; rule 7 only
> makes them unskippable.

### 8. Training runs export everything
Terminal **and intermediate** checkpoints, `metrics.csv`, `episode_rows.csv`,
manifests, configs, summaries, git SHA, environment info, evaluation artifacts.
Validate the bundle against an expected manifest and **fail loudly** if anything
is missing.

> *Motivating failure:* the 6v6 exporter copied only terminal checkpoints,
> manifests and specs. The training curves and every intermediate checkpoint were
> never bundled and became unrecoverable once the source machine no longer had
> them. That permanently cost the ability to diagnose 6v6 training dynamics.
> Fixed: `export_6v6_results.sh --strict`.
>
> *Generalised 2026-09-12:* `experiments/export_run_bundle.py` does this at any
> scale — `export_run_bundle.py 4v4 --suffix _b3 --strict`. `--plan` inventories
> what would be exported without copying, so completeness is checkable before
> committing gigabytes. The addition that matters most is `verify`, which
> re-checks a bundle against its own `MANIFEST.json` by sha256 **on the receiving
> machine**: the original loss was discovered long after the source was gone, and
> destination-side verification is the only step that catches it while recovery
> is still possible. A bundle recorded incomplete at creation fails `verify` even
> when every file it does contain hashes correctly — hash-matching is not
> completeness.
>
> Run against the real 6v6 tree it reproduces the incident exactly: exit 1,
> naming all ten lost artifacts. 11 tests in `tests/test_export_run_bundle.py`.

### 9. Three seed classes, never overlapping
`smoke/debug` (999xxxxx family), `exploratory`, `sealed confirmatory`.
Confirmatory seeds are never spent on debugging. If code changes materially after
a confirmatory run, the old result stays **frozen** rather than being quietly
replaced.

> *Motivating failure:* a plumbing smoke once ran real episodes on four frozen
> 6v6 collection seeds before `--smoke` mode existed. Caught immediately; the
> seeds were formally retired rather than reused.
>
> *Codified 2026-09-12:* `experiments/seed_registry.py` + `artifacts/SEED_REGISTRY.json`.
> The rule was already practised; what was missing was anything that could
> **refuse**. `check_block` rejects exact re-spends, partial overlaps, a smoke
> class outside the reserved `999xxxxx` family and a real class inside it;
> `allocate` refuses on collision; a `SPENT` confirmatory block can only move to
> `RETIRED`. Legitimate nesting is modelled explicitly (a 320-seed collection
> bank subdivided into shards is not a violation) so the check does not
> false-positive, and `audit` catches an undeclared overlap introduced by hand.
> The registry is wired into rule 7: an evaluator that declares a seed class has
> it checked at seal time, so a confirmatory re-spend is caught even if the
> launcher was bypassed. 16 tests in `tests/test_seed_registry.py`.
>
> Backfilled from artifacts by `bootstrap_seed_registry.py`: **42 historical
> blocks, zero undeclared overlaps** — the practice really was clean. 14 blocks
> could not be classified from artifact evidence and are recorded as
> `UNCLASSIFIED`, which makes `audit` exit 1 until a human resolves them. That is
> deliberate: a guessed class in a registry whose whole job is to be trusted is
> worse than an admitted gap.

### 10. No GPU experiment without a decision tree
Before running, finish this sentence: *"If result A happens we do X; if B happens
we do Y."* **If every possible result leads to the same next action, the
experiment is not worth running.**

### 11. No metric before execution-path proof
Before implementing any new diagnostic or scalar, answer these **in order**:

1. What exact scientific quantity are we trying to know?
2. Where in the actual code does that quantity exist?
3. What transformations happen between the conceptual variable and execution?
4. Under what conditions is the proposed metric actually defined?
5. What trivial / golden case has a known answer?
6. Can we deliberately construct a case where the metric should fail?

Only then write the metric. Prefer this sequence:

```text
trace → prove metric semantics → golden test → negative control → small smoke → real experiment
```

not:

```text
good idea → implement → discover assumption afterward
```

**Code path beats comment. Runtime tensor beats config. Executed action beats
intended action.**

> *Motivating failures (same class, different costumes):*
>
> - **\(e_q\) waypoint distance.** Measured nearest-\(W_{50}\) error as if the
>   pipeline were `target → W50`. Real execution is `intent → macro`, and only
>   `GO_TO` uses \(W_{50}\); `GET_FLAG` / `GO_HOME` do not. The 2.076 invariant
>   looked scientific while measuring a quantity often undefined for the
>   executed action.
> - **SNR normalization.** A useful-looking scalar before checking finite-sample
>   behaviour at \(d \approx 3.47\times 10^6\).
> - **4-tick sampling.** Reasoning from commitment horizons almost invented a
>   cadence instead of tracing how scripted Blue actually steps.
> - **RANDOM-OPP.** Grid and vector builders initially sampled different fake
>   worlds under one conceptual “randomize opponents.”
> - **ZERO-OPP.** Literal zero until schema tracing showed `vec[11]=0` means an
>   enemy at zero distance.
> - **Export / canonical pointers.** “Export the run” and “latest” reasoned from
>   filenames and intent instead of required inventory and promotion smoke tests.
>
> Validation machinery caught each of these, but often **one stage too late**.
> Rule 11 moves that skepticism *before* the metric is coded.
>
> *Required for important subsystems:* tiny deterministic golden traces with a
> known answer (and a known failure case) gate any audit that depends on them.
> Example for the action adapter: enemy-flag → `GET_FLAG` → \(e_q\) N/A;
> own-home → `GO_HOME` → \(e_q\) N/A; arbitrary intercept → `GO_TO` → \(e_q\)
> required. If those do not pass, the audit does not run.

### 12. Every experimental arm needs a known-answer contract before the main loop

Rule 11 protects a *metric*. This protects a *harness* — the code that drives
the system under test, which can be wrong in ways no metric will ever flag
because the metric only ever sees the harness's own (wrong) output.

Before spending any real episode, each arm of an experiment must reproduce a
result whose correct answer is already known, from a path independent of the
new harness code:

- an **oracle/replay arm** must reproduce the reference implementation it
  stands in for, on the same seed, exactly;
- a **behaviour-cloning privileged arm** must reproduce a tiny deterministic
  teacher dataset;
- a **perturbation experiment's identity/no-op condition** must reproduce the
  unperturbed evaluator;
- a **checkpoint evaluator** loading a known terminal checkpoint must
  reproduce a frozen reference seed's result.

If no such contract exists for an arm, that absence is recorded explicitly —
*"no known-answer test for this arm"* — never silently treated as validated.

> *Motivating failure:* the projected-teacher oracle's `ORIGINAL` arm was meant
> to call `core.set_blue_style(style)` and then rely on `blue_scripted` already
> being `True`. A code comment asserted `# also sets blue_scripted=True` — the
> exact opposite of what the function's own docstring says two lines above the
> call site (*"Does NOT itself enable scripted blue — callers must also set
> `self.blue_scripted = True`"*). Blue ran fully action-controlled with
> all-zero actions: it drove to waypoint 0 and scored 0 in every episode. 40
> episodes ran before the fabricated-looking `V=0.0000` was checked against
> the certified reference win rate (`poleA_guard_wr = 0.635`) and shown to be
> impossible. Rule 11 had been applied to the *adapter*; this gap is why it
> now also applies to the *harness*. One line —
> `assert core.blue_scripted is True` executed as a contract, not asserted as
> a comment — would have caught it before episode one.
>
> **The generalized principle:** if you assert what another function does,
> prove it by executing a contract, not by writing a comment. *Code path
> beats comment* (rule 11) is about metrics; this is the harness version of
> the same discipline.

### 13. A lock records identity, not just presence — and is never deleted by hand

A lock file's only job is to answer *"is the owner still running"* without
trusting a human's memory of what they killed. That requires more than a
lock file existing: it requires enough identity to tell a live owner from a
stale record, and a procedure for reclaiming a stale one that never guesses.

```text
lock exists
    |
    is the recorded PID alive AND running a matching command line?
        yes -> REFUSE to launch
        no  -> ARCHIVE the stale record (never delete), then acquire fresh
```

Releasing is symmetric: only the process whose PID is recorded may release
its own lock on normal exit; a crash must leave the lock in place, not
release it, so a human investigates rather than a second process quietly
starting against a run that died mid-write. An **operator-initiated** stop
must terminate the recorded PID, **poll until the OS confirms it is actually
gone**, and only then archive the lock — never remove a lock speculatively
because a process merely *looks* dead.

> *Motivating failure:* the projected-teacher oracle's first launch had the
> `blue_scripted` bug above and looked wrong, so its lock was deleted by hand
> and a second, fixed run was launched. The first process was **not** dead —
> `pkill -f` had matched nothing, silently. Both processes then wrote
> `rows.csv` concurrently: null bytes, a missing header, rows beginning
> mid-cell at seed 17600017 instead of 17600001. The corruption produced no
> error; it was caught only because the row count looked short for the
> elapsed time.
>
> *Implemented 2026-09-13:* `experiments/run_lock.py`. `RunLock` records
> `pid`, `run_id`, `hostname`, `start_utc`, `cmdline`; `acquire()` refuses
> against a live, command-line-matching owner and archives (never deletes) a
> genuinely stale one; a recycled PID running an unrelated command is
> correctly treated as a dead owner. `release()` requires PID ownership.
> `stop_and_release()` terminates, polls for confirmed death, and only then
> archives — raising `StillAlive` rather than proceeding if the process
> outlives the timeout. 11 tests in `tests/test_run_lock.py`, including one
> that caught a real bug in the module itself: the context-manager's
> `__exit__` originally released the lock unconditionally, which would have
> silently defeated the crash-preservation guarantee it exists to provide.

### 14. The live environment is proven against the certification, never against the launch arguments

**No experiment may start unless the live resolved environment is independently
proven identical to the governing certified configuration; launch arguments
alone are never evidence of correctness.**

The chain, in order, with only the last step costing real time:

```
certification → resolved config → hash/field equality → live pole attestation
              → known-answer contracts → GPU training
```

Filename heuristics are banned from the safety decision. The certification
record states which opponent it certified — genome id, overlay, team size — and
the launcher must resolve what it will actually instantiate and compare, field
by field and by deterministic hash. A record that cannot express the pole it
certified is not a basis for spending GPU hours (absence is an error state).
Paired studies must additionally print a **cross-policy parity diff**, because
the failure below came from assuming two launches were symmetric.

> *Motivating failure (2026-09-14):* `pi_B3` was **trained** against canonical
> Pole B (`SDS_PARENT_OP7`) while every evaluation **scored** it against the
> certified B3-3 candidate (`SDS2_B3_LOCKDEF10_2V1`, which adds
> `lock_defender=10`, `enable_2v1=True`). The launch omitted
> `--pole-b-genome-json`. The guard that should have caught it demanded that
> flag only when the certification's *filename* contained
> `"CONFIRMATORY_REDESIGN"` — the B3-3 record certifies a candidate genome the
> same way but does not match that substring, so nothing fired. The existing
> `LIVE POLE CHECK` could not catch it either: it verified the live overlay
> against *what the launcher expected*, and the launcher expected canonical, so
> expectation and reality agreed. **~17.5 GPU-hours and five seed blocks were
> spent answering the wrong experiment**, and the resulting Δ_B = −0.086 drove
> weeks of downstream diagnosis into a Pole-B problem that had never been
> fairly tested.
>
> Two further lessons from the same incident, both now enforced:
> *A preflight that installs the canonical genome regardless of an override
> attests an opponent the run never sees* — the throwaway-env check now installs
> the **resolved** genome. And *a "verified_by" field in a frozen spec must cite
> the object actually opened*: the claim that B3-3 passed no genome flag was
> written after reading **pi_A's** run_config and asserting it for both policies.
>
> *Implemented:* `experiments/pole_attestation.py` (certification parsing,
> deterministic `pole_config_hash`, pre-GPU field equality, zero-step live
> attestation read from the behaviour tree's own resolved tensors, cross-policy
> parity). 11 tests in `tests/test_pole_certification_guard.py` prove the exact
> bad launch is refused, a *wrong* genome is refused, the certified genome
> passes, Pole A is unaffected, a mismatch is caught **before** any environment
> is constructed, and a warm start cannot silently swap the opponent.

---

## Bugs these rules are designed to prevent

All of the following were caught by auditing rather than by the runs failing
loudly. Each would have silently corrupted a result:

| Bug | How it would have poisoned the science |
|---|---|
| Export dropped curves + intermediate checkpoints | Unrecoverable loss of training-dynamics evidence |
| Δ printed only at program exit | Multi-hour runs unmonitorable; crash loses everything |
| RANDOM-OPP drew fresh randoms per builder call | Grid and vector channels would describe **two different fake worlds** |
| "Zero the opponent features" | vec[11]=0 means *enemy at zero distance* — the **opposite** of absent |
| PERMUTE-OPP condition | Provably vacuous: the schema is permutation-invariant in opponent identity |
| Native sensor model for ablation | Draws from env RNG → desyncs ablated episodes from FULL, destroying pairing |
| Blanket `_side_tensors` patch | `_mines.py` / `_rules.py` also call it → corrupts mine mechanics and win determination |
| Self-test reading its own method | Would have silently passed a broken uninstall |
| \(e_q\) applied before macro-path proof | Quantization error reported when executed macro never uses \(W_{50}\) |
| `blue_scripted` asserted true in a comment, not executed | ORIGINAL arm ran fully action-controlled with zero actions; scored 0 for 40 episodes before the number was checked against a known reference |
| Lock deleted by hand after a `pkill` that matched nothing | Two processes wrote the same rows.csv concurrently: null bytes, missing header, rows starting mid-cell |
| `RunLock.__exit__` released unconditionally | Would have silently defeated its own crash-preservation guarantee — caught by its own test suite before ever running live |
| Conceptual “zero opponent” / “random opponent” | Schema semantics inverted or inconsistent across observation builders |
| Designing from names/comments | Scripted Blue step cadence, GUARD-at-\(N{=}4\), opponent identity vs occupancy |

---

## Current compliance (2026-09-13)

| Rule | Status |
|---|---|
| 1 live telemetry | ablation ✅ · trajectory ✅ (fixed, applies next run) · track-hold ✅ · Arm-1/oracle ✅ |
| 2 incremental writes | ablation ✅ · trajectory ❌ · track-hold ❌ · oracle ✅ |
| 3 frozen spec | ✅ all experiments |
| 4 preflight/self-test | ✅ track-hold, ablation, Arm-1 · partial (dry-run only) trajectory |
| 5 resumability | ablation ✅ · oracle ✅ · others ❌ |
| 6 LIVE_STATUS | ablation ✅ · oracle ✅ · others ❌ |
| 7 automatic audit gating | ✅ `run_state.py` — enforced state machine, 14 tests |
| 8 export everything | ✅ `export_run_bundle.py` — any scale, + destination `verify`, 11 tests |
| 9 seed classes | ✅ `seed_registry.py` — refuses overlaps, wired into rule 7, 16 tests |
| 10 decision tree | ✅ all recent specs carry `PRECOMMITTED_INTERPRETATION` |
| 11 no metric before path proof | ✅ `teacher_action_adapter.py` — golden + failure-case anchors, gate the adapter audit and the oracle |
| 12 known-answer contract per arm | ✅ oracle's `harness_anchor()` gates the main loop; **caught a real harness bug before it wasted the run** |
| 13 hardened lock | ✅ `run_lock.py` built + tested (11 tests) · ❌ NOT yet retrofit into the running oracle process, which was launched before this module existed and still holds a bare pid/utc lock — deliberately left alone rather than restarted |

**Remaining open gaps**

- Rules 2/5/6 on the two older evaluators (`eval_checkpoint_trajectory_4v4.py`,
  `eval_track_hold_6v6.py`). Retrofitting either means restarting a run, so the
  live trajectory run is **grandfathered** and finishes as-is. It is the last one.
- 14 `UNCLASSIFIED` blocks in the seed registry, pending human classification.
- Rules 7–9 are infrastructure only so far: **no evaluator has been migrated onto
  `seal()` yet.** The next experiment to run is the first that must use it, and
  until one does, the gate is proven by its tests rather than by use.
- Rule 13 is not yet wired into the older evaluators (`eval_opponent_ablation_6v6.py`
  and earlier all use a bare `.run.lock` file with no PID/identity check). The
  projected-teacher oracle is the first to need it, since it was the one that
  broke; retrofit the rest opportunistically, not urgently.

## The overriding rule

**We do not optimise for PASS.** We optimise for understanding the failure well
enough that the intervention is justified *before* it is run. That is also, in
practice, the fastest route to a defensible PASS.
