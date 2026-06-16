# Experiment and Evaluation Protocol

**Owner:** This file is the single source of truth for *how* experiments
are launched, *how* runs are compared, and *what* statistical and
diagnostic protections are required before a delta is allowed to appear
in a table, claim, or PR description.

It does not redefine the method. It locks the *protocol* so that every
row in the proof table was produced under comparable conditions.

> **Read first:**
> [`AGENTS.md`](../../AGENTS.md) ·
> [`summer-method-spec.md`](summer-method-spec.md) ·
> [`summer-fidelity-rules.md`](summer-fidelity-rules.md) ·
> [`latent-preset-registry.md`](latent-preset-registry.md)
>
> **Read also:**
> [`research-progress-tracker.md`](research-progress-tracker.md) ·
> [`Paper_experiment_alignment.md`](Paper_experiment_alignment.md)

---

## 1. Locked invariants across every comparison

Any comparison that intends to attribute a metric delta to "the latent
mechanism" or "the paper-faithful router" must hold the following
invariants identical across the rows being compared. Deviation requires
an explicit note in
[`research-progress-tracker.md`](research-progress-tracker.md) and a
flag on the resulting claim.

| Invariant                                                        | Operational value (canonical paper-faithful comparison)                       |
|------------------------------------------------------------------|-------------------------------------------------------------------------------|
| Opponent pool                                                    | `("OP5", "OP6", "OP7")`, uniform                                              |
| `opponent_randomize`                                             | `True`                                                                        |
| Team size                                                        | `4v4` (`--agents 4`)                                                          |
| Total training budget                                            | `1_000_000` decision steps (`--total-steps 1000000`; `PPOConfig` default)     |
| `n_envs`                                                         | `32`                                                                          |
| `n_steps`                                                        | `2048` (`PPOConfig` default)                                                  |
| `n_epochs`                                                       | `6`                                                                           |
| Seed                                                             | Fixed integer (`--seed 0` for the v5i6 first launch; v5i4 used seed 0)        |
| Device                                                           | `cuda`                                                                        |
| Map / env config                                                 | Default (`GPUFieldConfig` defaults)                                           |
| Periodic checkpoint cadence                                      | `--periodic-checkpoint-steps 50000`                                           |
| E3 step telemetry                                                | ON (`--e3-step-telemetry`)                                                    |
| Fresh metrics CSV                                                | ON (`--fresh-metrics-csv`)                                                    |
| Checkpoint directory                                             | `checkpoints/4v4`                                                             |

The v5i6 launch command below is the canonical open-map reference. Any
new launch in the v5 / v4 ladder must match it except for `--preset` and
(if intentional) `--run-tag`. v5i7 intentionally changes `map_layout` to
`map_b_split_lane`; v5i8 intentionally changes `map_layout` to
`map_b_split_lane_v2`. Compare map-specific rows only against matched-map
controls.

---

## 2. Launch protocol

### 2.1 Canonical paper-faithful launch (v5i6)

```powershell
.\.venv\Scripts\python.exe rl/train_ppo.py `
    --preset v5i6_paper_faithful `
    --total-steps 1000000 `
    --agents 4 `
    --seed 0 `
    --device cuda `
    --n-envs 32 `
    --n-epochs 6 `
    --e3-step-telemetry `
    --checkpoint-dir checkpoints/4v4 `
    --fresh-metrics-csv `
    --periodic-checkpoint-steps 50000
```

### 2.2 Pre-launch checklist

Before pressing enter on any latent training launch:

1. **Confirm classification.** Look up the preset in
   [`latent-preset-registry.md`](latent-preset-registry.md) §3. If
   `UNKNOWN`, stop and resolve via the *Proposed Preset Review*
   template in [`summer-fidelity-rules.md`](summer-fidelity-rules.md) §8.
2. **Confirm resolved configuration.** Run
   `dataclasses.asdict(apply_preset(PPOConfig(), "<preset>"))` and
   compare against
   [`tests/preset_snapshots.json`](../tests/preset_snapshots.json). The
   resolved dict is the only authoritative source.
3. **Confirm audit banner trigger.** For any preset that should print
   the paper-faithful invariant block at training start, verify either
   `cfg.run_tag` contains a recognized family tag
   (`"v5i4_paper_faithful"`, `"v5i5_paper_faithful"`,
   `"v5i6_paper_faithful"`, `"v5i7_summer_faithful"`) or
   `cfg.latent_paper_faithful_audit = True`. The banner code lives in
   [`rl/training/banner.py::_maybe_print_paper_faithful_audit`](../rl/training/banner.py).
4. **Confirm `run_tag` matches the actual budget.** v5i4 corrected
   `_2m_` → `_1m_`; pinned by
   [`tests/test_v5i4_paper_faithful.py::V5i4RunTagAndInitialOpponentConsistencyTests::test_run_tag_advertises_actual_total_timesteps_budget`](../tests/test_v5i4_paper_faithful.py).
   v5_strict_summer / v5i1 / v5i2 / v5i3 still inherit the misleading
   `_2m_` suffix — see
   [`latent-preset-registry.md`](latent-preset-registry.md) §7. Pass
   `--run-tag` to override when launching a new comparison.
5. **Confirm initial-opponent seeding is in-pool.** v5i4 fixed an
   `OP3` leak into the first telemetry slice when the opponent pool was
   `(OP5, OP6, OP7)`; pinned by
   [`tests/test_v5i4_paper_faithful.py::V5i4RunTagAndInitialOpponentConsistencyTests::test_initial_opponent_falls_back_to_pool_first_entry`](../tests/test_v5i4_paper_faithful.py).
   The resolver
   [`rl/train_ppo.py::_resolve_initial_opponent_and_phase`](../rl/train_ppo.py)
   falls back to the first `cfg.opponent_pool` entry when the legacy
   `cfg.fixed_opponent_tag` is out-of-pool.
6. **Smoke first.** Run
   [`tests/test_train_ppo_smoke.py`](../tests/test_train_ppo_smoke.py)
   to verify the preset survives one update. CI-fast and worth the
   minute it costs before a multi-hour training run.

### 2.3 Per-preset launch variants

| Preset                              | `--preset` value             | Extra notes                                                                                                                          |
|-------------------------------------|------------------------------|--------------------------------------------------------------------------------------------------------------------------------------|
| Canonical paper-faithful (v5i6)     | `v5i6_paper_faithful`        | Default audit banner fires. Use any of the ten aliases (§5 in registry) — all resolve identically.                                  |
| Conditional-entropy interpretation  | `v5i4_paper_faithful`        | Preserved comparison row with mean conditional entropy.                                                                              |
| Conditional entropy-floor ablation  | `v5i5_paper_faithful_entropy_floor` | Same conditional entropy as v5i4, but `latent_lam_h_end = 0.001`.                                                              |
| Summer-faithful split-lane row      | `v5i7`                      | Inherits v5i5 and changes only `map_layout = "map_b_split_lane"` plus `run_tag`; compare only to split-lane matched controls.        |
| Summer-faithful split-lane v2 task-pressure row | `v5i8` | Inherits v5i7 and changes only `map_layout = "map_b_split_lane_v2"` plus `run_tag`; compare only to split-lane-v2 matched controls. |
| Literal-strict ablation             | `v5_strict_summer`           | Same launch flags. Tag will carry `_2m_4v4` (historical); pass `--run-tag` to override.                                              |
| No-latent baseline (matched control)| `no_latent_v4i3_baseline`    | Same launch flags. `use_latent_strategy = False`. Tag is budget-agnostic.                                                            |
| Arc-credit row                      | `v4i3_summer_proof`          | Same launch flags. Tag is budget-agnostic; preset is `SUMMER-COMPATIBLE EXTENSION`, **not** literal paper-faithful (arc-credit ON).  |
| K=1 ablation                        | `plan_faithful_latent_k1`    | Hold all v5i4-equivalent invariants except `latent_k`. Note: this preset inherits from a much older parent — diff resolved configs.   |

---

## 3. Decisive comparisons

The minimum comparison set for the paper's central claim
(["latent shared discrete `z` learned end-to-end from task reward
improves CTF coordination"]) is three rows under §1's locked invariants:

| Comparison                                                                                | Isolates                                                  |
|-------------------------------------------------------------------------------------------|-----------------------------------------------------------|
| **v5i6** vs **no_latent_v4i3_baseline**                                                   | The contribution of the entire latent mechanism under the canonical marginal-entropy interpretation. |
| **v5i6** vs **v5_strict_summer**                                                          | The contribution of the on-policy categorical PPO term on `q_phi` plus the canonical entropy interpretation, relative to persistence + entropy alone. |
| **v5i6 router** vs **v5i6 random-matched** (at eval time, same checkpoint)                | Routing quality from actor quality at fixed actor weights. |

Optional rows for paper depth (already in the registry):

| Comparison                                | Isolates                                                                              |
|-------------------------------------------|---------------------------------------------------------------------------------------|
| **v5i6** vs **v5i4**                       | Marginal entropy vs mean conditional entropy at the same actor / critic / router-PPO / persistence / sampling contract. |
| **v5i6** vs **v5i5**                       | Entropy reduction only: both use the same `0.003 → 0.001` lambda_H schedule. |
| **v5i6** vs **v4i3_summer_proof**          | Arc-credit (v3i19 extension) vs main-loop per-step PG plus marginal entropy. |
| **v5i6** vs **v5i1_reward_credit_router**  | Per-step main-loop PG plus marginal entropy vs per-episode router PG with dedicated AdamW. |
| **v5i6** vs **plan_faithful_latent_k1**    | Whether latent *capacity* itself contributes. |
| **v5i6** vs **v4i4post_periodic_router_distill** | Whether counterfactual distillation adds anything to the canonical paper-faithful run. |

---

## 4. Evaluation protocol (`plot/eval_checkpoint.py`)

### 4.1 Frozen eval matrix

Use identical arguments for every checkpoint in a comparison set:

| Parameter        | Frozen suggestion                                                                                       |
|------------------|---------------------------------------------------------------------------------------------------------|
| Script           | `python plot/eval_checkpoint.py`                                                                        |
| `--agents`       | Match training (canonical: `4`)                                                                         |
| `--opponents`    | `OP5_RUSHER OP6 OP7` for the paper-faithful comparison; add `OP3 OP4` only if a held-out style claim is intended. |
| `--map-sets`     | `train eval` (always include `eval` so train-WR saturation does not hide a generalization gap)          |
| `--episodes`     | `≥ 100` per (map_set, opponent); use `300–500` for tighter CIs in final tables                          |
| `--device`       | `cuda` if available                                                                                     |
| `--seed`         | Same integer across all rows                                                                            |
| `--deterministic`| ON (default)                                                                                            |

OP5 carries a tuning tag (`OP5_RUSHER_TUNING_TAG` in
[`rl/opponent_params.py`](../rl/opponent_params.py)) that must be
recorded in any published table caption whenever OP5 appears. The eval
script auto-appends it to labels; do not strip with
`--no-op5-tuning-suffix` for headline tables.

### 4.2 Latent-selection modes (the routing-quality ablation)

[`plot/eval_checkpoint.py`](../plot/eval_checkpoint.py) exposes four
modes via `--latent-selection`:

| Mode             | What it does                                                                                                                                        | Use it for                                                                                       |
|------------------|------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------|
| `router` (default) | Use the trained `q_phi(z|s)`.                                                                                                                       | Headline row.                                                                                    |
| `random-matched` | Uniform-random `z`, resampled at the **same decision steps** the router would have (inherits checkpoint's `latent_resample_every_n`; override via `--latent-resample-every`). | Decisive routing-quality control: identical actor weights, identical latent timing, only the `z` distribution differs. |
| `random-episode` | Uniform-random `z`, episode-start only (forces `strategy_interval = 0`). Different ablation question (changes persistence, not just routing).        | Persistence ablation, not routing.                                                               |
| `fixed`          | Clamp every episode to `--fixed-latent-id`.                                                                                                          | Forced-z behavioral inspection.                                                                  |

**The decisive routing comparison** is `router` vs `random-matched`
with the same checkpoint and the same `--seed`. Anything that beats
`random-matched` was learned by `q_phi`, not by the actor.

### 4.3 Fixed-z behavioral inspection

For each `z ∈ {0, …, K-1}`, run a sweep with
`--latent-selection fixed --fixed-latent-id z` and inspect per-z
metrics. Use the resulting per-z behavior tables to support (or
falsify) the *behavioral non-triviality* sub-claim: forced-`z`
materially changes behavior inside comparable contexts.

> **Causal claim warning.** Per-`z` differences observed in rollout
> telemetry from a non-forced-`z` run are **not causal** —
> [`AGENTS.md`](../../AGENTS.md) §8.7 forbids interpreting them that
> way. Causal claims require forced-`z` matched-seed evaluations.

---

## 5. Statistical protections (required before any "this beats that" claim)

A win-rate (or return) delta is reportable only if **all** of the
following hold.

### 5.1 Sample-count rule

* **Never** report `0.0` or `1.0` win rates without sample counts.
  Both are uninformative without `n` and bracket-bounded estimation.
  ([`AGENTS.md`](../../AGENTS.md) §8.8 forbids this.)
* Minimum `n = 100` episodes per cell. Headline tables: `n = 300` or
  more.

### 5.2 Confidence intervals

* Compute Wilson 95% intervals on every cell win rate. The Wilson
  interval handles `n ≤ 100` honestly and degenerates correctly at
  `wr = 0` or `wr = 1`.
* Report the interval, not just the mean.

### 5.3 Paired bootstrap

* For paired comparisons (matched seeds + matched (map_set, opponent)
  cells), use a paired-bootstrap test (≥ 10 000 resamples) on the
  per-cell delta. Report the bootstrap 95% CI on the delta. A claim of
  "row A beats row B" requires the bootstrap CI on `Δ = wr_A − wr_B`
  to exclude zero at 95%.

### 5.4 Multi-seed requirement

* Single-seed comparisons may be reported in
  [`research-progress-tracker.md`](research-progress-tracker.md) as
  "preliminary" but never in a paper / claim.
* Headline rows require **≥ 3 seeds** trained under §1's invariants.
* Report the seed-aggregated metric with a per-seed scatter strip in
  the supplementary table.

### 5.5 What does **not** count as evidence

* "WR climbed to 64% by step 1M" with no sample counts and no
  comparison row.
* A delta versus a different opponent pool, team size, or budget.
* A per-`z` WR spread from a non-forced-`z` rollout.
* Telemetry-only diagnostics (e.g. `actor_z_jsd`, `argmax_disagree`)
  without a downstream WR or return effect.

---

## 6. MI / separability protocol

### 6.1 Online MI telemetry (E3 step CSV)

`--e3-step-telemetry` writes one row per decision step into
`<run_tag>_e3_steps.csv` with the columns enumerated in
[`rl/custom_ppo/csv_writers.py::E3_STEP_TELEMETRY_FIELDS`](../rl/custom_ppo/csv_writers.py).
Per-update MI estimates are printed inline by the trainer (search the
log for `MI_z_o`, `MI_z_phase`, `MI_z_flag`, `MI_z_outcome`).

The plug-in MI estimator in
[`rl/discrete_mi.py::discrete_mi_plugin`](../rl/discrete_mi.py) is the
canonical implementation; do not roll a new estimator without an
explicit comparison test against `discrete_mi_plugin` on a fixed
joint matrix.

### 6.2 Offline MI recomputation

[`plot/analyze_e3_latent_mi.py`](../plot/analyze_e3_latent_mi.py)
recomputes MI(`z`; phase), MI(`z`; opponent), MI(`z`; outcome) from any
`*_e3_steps.csv`. Use this for finer windowing (e.g. last 100k decision
steps only) or to slice MI by phase / opponent. Run:

```bash
python plot/analyze_e3_latent_mi.py checkpoints/4v4/<run_tag>_e3_steps.csv
```

### 6.3 MI thresholds

* MI(`z`; outcome) and MI(`z`; opponent) values reported in the
  v5i4 in-flight rollout are `O(10^-3)` — informative as a *floor*
  (the latent is not perfectly uncorrelated with these context
  variables) but **not** by itself evidence of meaningful routing.
* The actionable test is the matched-schedule routing-quality
  comparison (§4.2). MI is a diagnostic, not the headline claim.

### 6.4 Forced-z return contrast (`tools/q_probe.py`)

[`tools/q_probe.py`](../tools/q_probe.py) runs a *fixed-z* probe at
matched starts and reports

```text
return_contrast = max_z(R) - min_z(R)
```

per `(opponent, seed)`, where `R` is the mean undiscounted episode
return for each forced `z`. Thresholds (per the v4i1 design):

* `return_contrast < 0.05`: the environment does not care about
  strategy. Escalate to environment v2 before claiming the latent
  helps.
* `return_contrast >= 0.10`: different `z` choices create different
  outcomes. Proceed to routing-quality comparison (§4.2).

Pass `--watch` to poll a checkpoint directory and append rolling
contrast estimates as new checkpoints arrive.

### 6.4a v5i8 forced-z behavioral evaluation (`tools/v5i8_forced_z_eval.py`)

[`tools/v5i8_forced_z_eval.py`](../tools/v5i8_forced_z_eval.py) is the
post-training protocol for the split-lane-v2 latent row. It freezes one
checkpoint, runs `natural` router rollouts and `fixed_z` rollouts for
every latent ID, and writes a manifest, per-step trajectory CSV,
`rollout_by_z` aggregate CSV, and Markdown readout.

This is evaluation only: no backward pass, no supervised strategy labels,
and no training-time dependency on what each `z` means.

```bash
python tools/v5i8_forced_z_eval.py `
    --checkpoint checkpoints/4v4/<v5i8_final>.zip `
    --metrics-csv checkpoints/4v4/<v5i8_metrics>.csv `
    --map-layout map_b_split_lane_v2 `
    --opponents OP5 OP6 OP7 `
    --episodes-per-mode 100 `
    --device cuda
```

Interpretation: if forced `z` changes behavior telemetry and trajectories
for the same opponent, latent strategies emerged. If it changes only win
rate or return, `z` affects performance but strategy meaning remains
unclear. If it changes neither behavior nor outcome, `z` usage is cosmetic.

### 6.5 Local counterfactual probe (`tools/q_probe_local_counterfactual.py`)

[`tools/q_probe_local_counterfactual.py`](../tools/q_probe_local_counterfactual.py)
snapshots the environment state at each arc boundary, forces each
`z ∈ {0, …, K-1}`, and rolls each forced trajectory to completion. This
isolates `Q(s, z)` contrast at the exact decision points where `q_phi`
acts, which the rollout-time `return_contrast` (§6.4) does not.

### 6.6 Behavioral separability (`tools/sliced_z_diagnostics.py`,
`tools/tactical_regime_analysis.py`)

For inspecting *which* behavioral features change with `z` (under fixed
context), use the sliced-z diagnostics and tactical-regime analysis
tools. These produce per-`z` distributional summaries of the
`BEHAVIOR_TELEMETRY_NAMES` columns
(see [`rl/behavior_telemetry.py`](../rl/behavior_telemetry.py)).
Output is descriptive — a separability claim must be combined with a
forced-`z` causal probe (§6.4 / §6.5).

### 6.7 Summary report (`tools/summer_proof_report.py`)

[`tools/summer_proof_report.py`](../tools/summer_proof_report.py)
assembles the v4i3 / v5i4 / v5i6 Summer-Proof gates from the metrics CSV,
the q_probe outputs, and the local counterfactual outputs into a single
Markdown report. The thresholds default to `0.10` (forced-z return
contrast and local Q-contrast); override with `--gate2-threshold`
and `--gate3-threshold` for tighter or looser comparisons.

```bash
python tools/summer_proof_report.py `
    --latent-run-tag v5i4_paper_faithful_end_to_end_OP5_OP6_OP7_2m_4v4 `
    --baseline-run-tag v4i3_no_latent_baseline_OP5_OP6_OP7_4v4 `
    --checkpoint-dir checkpoints/4v4 `
    --qprobe-dir checkpoints/4v4/qualitative `
    --local-cf-dir checkpoints/4v4/v4i3_local_cf_after_envfix `
    --out reports/v5i4_summer_proof.md
```

(The `_2m_` tag in `--latent-run-tag` is the artifact-history filename
for the completed in-flight v5i4 run; see
[`latent-preset-registry.md`](latent-preset-registry.md) §7.1.)

---

## 7. Audit-banner and trainer-side invariants

Even before evaluation, the training run must satisfy:

1. **Audit banner fires for any paper-faithful preset** and lists
   every channel ON/OFF state per
   [`summer-fidelity-rules.md`](summer-fidelity-rules.md) §5. Pinned by
   the v5i4/v5i5/v5i6/v5i7 banner tests.
2. **The actor input dim line reads** `cnn(128) + per_agent_vec(20) +
   z_emb(16) = 164` for v5i4 / v5i5 / v5i6 / v5i7. Any other resolved width means the actor
   pathway has been altered; stop the run.
3. **`qphi_grad` per-update telemetry is nonzero on the resample
   subset** in any paper-faithful run. If it is zero, either
   `latent_strategy_ppo_coef <= 0` or `latent_episode_strategy_lr` is
   set; both are explicit warning lines in the audit banner.
4. **`actor_z_jsd` / `policy_z_sensitivity_KL` are nonzero and
   non-decreasing on long timescales** in any latent-on run. Zero
   sensitivity means `z` is not influencing the actor, regardless of
   what the loss says.
5. **`z_occ` does not collapse to a single index** within the first
   200 k steps. If it does, the run is in a v5i2-style collapse mode
   and the result is a coverage failure, not a router-quality result —
   see [`Paper_experiment_alignment.md`](Paper_experiment_alignment.md)
   §6.6 for the v5i3 forced-z anneal that was designed for this case.

---

## 8. Resume / reload safety

Any comparison that pauses, kills, or restarts a training run must
preserve:

* **`global_step`-conditioned schedules.** The forced-z resolver
  [`rl/custom_ppo/schedules.py::resolve_latent_forced_z_frac`](../rl/custom_ppo/schedules.py)
  is a pure function of `cfg` and the passed `global_step`. The trainer
  restores `global_step` from checkpoint metadata before resuming
  rollout. Pinned by
  [`tests/test_forced_z_anneal.py::ForcedZScheduleResolverTests::test_resume_uses_passed_global_step_not_internal_state`](../tests/test_forced_z_anneal.py)
  and
  [`tests/test_forced_z_anneal.py::ForcedZRuntimeRoutingTests::test_resume_at_mid_anneal_resolves_correctly`](../tests/test_forced_z_anneal.py).
* **CSV append semantics.** Pass `--fresh-metrics-csv` only on the
  *first* launch; subsequent resumes must omit it or the CSV will be
  truncated. Trainer-side append guards exist in
  [`rl/custom_ppo/csv_writers.py`](../rl/custom_ppo/csv_writers.py).
* **Latent state-dict back-compat.** Older checkpoints can be loaded
  by the v5i4/v5i6 actor as long as the actor input width matches; the
  legacy remap lives in
  [`rl/custom_ppo/policy.py`](../rl/custom_ppo/policy.py). Pinned by
  [`tests/test_custom_ppo_policy_parity.py`](../tests/test_custom_ppo_policy_parity.py).

---

## 9. Comparison invariants checklist (compact)

Before producing any table cell or claim from a comparison:

* [ ] Both rows used the same opponent pool and `opponent_randomize`.
* [ ] Both rows used the same `--agents`, `--total-steps`, `--n-envs`, `--n-epochs`, `--seed`.
* [ ] Both rows match `--map-set` at training and at eval.
* [ ] Eval uses identical `--opponents`, `--map-sets`, `--episodes`, `--seed`, `--deterministic`.
* [ ] Latent-selection mode is consistent (default: `router` for headline; add `random-matched` for the routing-quality control).
* [ ] Sample counts ≥ 100 per cell; ≥ 300 for headline.
* [ ] Wilson 95% CI on every cell win rate.
* [ ] Paired-bootstrap 95% CI on Δ for every comparison.
* [ ] OP5 tuning tag recorded in the table caption.
* [ ] Multi-seed: ≥ 3 seeds for any headline row.
* [ ] Resolved-config diff between the two rows attached to the comparison (so a reviewer can see what *did* change).

---

## 10. Cross-references

| Need                                              | Where to look                                                                       |
|---------------------------------------------------|-------------------------------------------------------------------------------------|
| Scientific definition of the paper method         | [`summer-method-spec.md`](summer-method-spec.md)                                    |
| Fidelity rules / classification / proposal form   | [`summer-fidelity-rules.md`](summer-fidelity-rules.md)                              |
| Per-preset facts, aliases, deltas, run tags       | [`latent-preset-registry.md`](latent-preset-registry.md)                            |
| Current run status / next experiment              | [`research-progress-tracker.md`](research-progress-tracker.md)                      |
| Mandatory agent behavior                          | [`AGENTS.md`](../../AGENTS.md)                                                      |
| Code↔manuscript trace                             | [`Paper_experiment_alignment.md`](Paper_experiment_alignment.md)                    |
| Algorithm sketch                                  | [`../../docs/algorithm.md`](../../docs/algorithm.md)                                |
