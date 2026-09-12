# Research Run Standards

Adopted 2026-09-12. Applies to every GPU experiment in this project from here on.

Goal: **every GPU hour answers a question, every long run is observable, every
result is recoverable.**

These rules exist because we lost real work to their absence. Each rule below
names the specific failure that motivated it, so the reason survives even when
the incident is forgotten.

---

## The ten rules

### 1. Live telemetry on every run
tqdm on every episode loop, with current cell/condition, seed, checkpoint,
elapsed and ETA. Long evaluators must print a result **as soon as a cell closes**.

> *Motivating failure:* the 4v4 checkpoint-trajectory evaluator computed every
> checkpoint's Δ only after all 480 episodes. Four hours of finished science sat
> invisible in RAM, and a mid-run crash would have surfaced none of it.

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

---

## Current compliance (2026-09-12, end of day)

| Rule | Status |
|---|---|
| 1 live telemetry | ablation ✅ · trajectory ✅ (fixed, applies next run) · track-hold ✅ |
| 2 incremental writes | ablation ✅ · trajectory ❌ · track-hold ❌ |
| 3 frozen spec | ✅ all experiments |
| 4 preflight/self-test | ✅ track-hold, ablation · partial (dry-run only) trajectory |
| 5 resumability | ablation ✅ · others ❌ |
| 6 LIVE_STATUS | ablation ✅ · others ❌ |
| 7 automatic audit gating | ✅ `run_state.py` — enforced state machine, 14 tests |
| 8 export everything | ✅ `export_run_bundle.py` — any scale, + destination `verify`, 11 tests |
| 9 seed classes | ✅ `seed_registry.py` — refuses overlaps, wired into rule 7, 16 tests |
| 10 decision tree | ✅ all recent specs carry `PRECOMMITTED_INTERPRETATION` |

**Remaining open gaps**

- Rules 2/5/6 on the two older evaluators (`eval_checkpoint_trajectory_4v4.py`,
  `eval_track_hold_6v6.py`). Retrofitting either means restarting a run, so the
  live trajectory run is **grandfathered** and finishes as-is. It is the last one.
- 14 `UNCLASSIFIED` blocks in the seed registry, pending human classification.
- Rules 7–9 are infrastructure only so far: **no evaluator has been migrated onto
  `seal()` yet.** The next experiment to run is the first that must use it, and
  until one does, the gate is proven by its tests rather than by use.

## The overriding rule

**We do not optimise for PASS.** We optimise for understanding the failure well
enough that the intervention is justified *before* it is run. That is also, in
practice, the fastest route to a defensible PASS.
