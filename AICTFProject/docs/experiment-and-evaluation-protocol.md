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
| CSIA guided specialization extension | `v5i9` | Inherits v5i8 and adds detached CSIA reward from forced-z evidence. Not a Summer-faithful row; compare against v5i8 with matched seed/map/opponents. |
| v6i1 staged curriculum (when launched) | `v6i1` | Repertoire-before-routing; evaluate per §6.8. Headline requires `G_available > 0` and `G_realized > 0`, not occupancy alone. |
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

v6i4 is a Summer-plan-faithful, evaluation-only router-ablation protocol
over a frozen, Phase-A-promoted v6i2 checkpoint. It is currently
planned/pending. No parameters are trained or updated.

[`plot/eval_checkpoint.py`](../plot/eval_checkpoint.py) exposes these
modes via `--latent-selection`:

| Condition | What it does | Use it for |
|-----------|--------------|------------|
| `learned_qphi_switching` | Use the trained `q_phi(z|s)` at the checkpoint's router opportunities. | Full proposed method. |
| `uniform_episode_fixed` | Uniform-random `z` once per episode, then hold it. | Arbitrary fixed latent selection. |
| `uniform_random_at_router_opportunities` | Uniform-random `z` from an isolated selector RNG at the same deterministic eligibility opportunities as `q_phi`. | Routing-quality control with identical actor weights and latent timing. |
| `preselected_global_fixed_z` | Pick one global fixed `z` using calibration seeds, then evaluate it on disjoint test seeds. | Deployable fixed-latent baseline without test leakage. |
| `fixed_z0` ... `fixed_z3` | Clamp every episode to one latent. | Repertoire performance map and oracle inputs. |
| `qphi_initial_only_no_switch` | Run trained `q_phi` once at episode start, then hold that initial `z`. | Isolates the causal value of mid-episode switching. |
| `shuffled_qphi_outputs` | Run `q_phi`, preserve eligible opportunity timing and output distribution, then deterministically permute outputs across matched contexts. | Tests whether context-output alignment matters beyond occupancy. |
| `posthoc_global_fixed_oracle` | Best global fixed `z` selected after the evaluation fixed-z sweep. | Non-deployable upper bound. |
| `posthoc_opponent_oracle` | Best fixed `z` per opponent selected after the evaluation fixed-z sweep. | Non-deployable upper bound. |
| `posthoc_episode_oracle` | Best fixed `z` per matched episode selected after seeing outcomes. | Optimistic non-deployable upper bound. |

For v6i4 claims, build the frozen ledger with
[`rl/eval_router_ablation.py`](../rl/eval_router_ablation.py). The
primary causal comparisons are:

```text
learned_qphi_switching - uniform_episode_fixed
learned_qphi_switching - uniform_random_at_router_opportunities
learned_qphi_switching - preselected_global_fixed_z
learned_qphi_switching - qphi_initial_only_no_switch
learned_qphi_switching - shuffled_qphi_outputs
```

Every online condition receives the same deterministic eligibility
opportunities:

```python
if switch_opportunity(context, step):
    z = rule.select_at_opportunity(...)
```

Controls are not forced to make the same actual switch that learned
`q_phi` made. The primary shuffled condition is a context-alignment
ablation: it preserves eligibility times and the `q_phi` output source
distribution, then deterministically reassigns outputs across matched
contexts.

The evaluator rejects a checkpoint unless metadata verifies
`experiment_id = v6i2`, Phase A promotion `PASS`, a recorded gate
fingerprint, a recorded promotion step, a recorded checkpoint hash, and
valid confirmatory gate lineage. Architecture compatibility alone is
insufficient.

Primary outcomes are return and win rate, reported aggregate and
per-opponent for `OP5`, `OP6`, and `OP7`. MI, entropy, occupancy,
argmax stability, route/task-behavior distance, and event-associated
switching are diagnostics, not success criteria.

The v6i4 runner writes `v6i4_manifest.json`,
`v6i4_episode_results.csv`, `v6i4_condition_summary.csv`,
`v6i4_paired_comparisons.json`, `v6i4_per_opponent_matrix.csv`, and
`v6i4_final_report.json`. Calibration seeds and test seeds must be
disjoint; if they overlap, the run is invalid because fixed-best baselines
would leak test evidence into baseline selection. Episode rows record
`environment_seed`, `initial_state_hash`, `action_sampling_seed`,
`selector_seed`, `shuffle_seed`, `opponent`, `episode_index`, and
`switch_opportunity_schedule_hash`. Paired comparisons join on:

```text
(opponent, test_seed, episode_index, initial_state_hash)
```

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

* `return_contrast < 0.05`: under the current checkpoint and evaluation
  design, forced `z` produces little measurable return spread. This does
  **not** by itself prove the bare MDP lacks latent opportunity — see
  §6.8 for the full decomposition. Escalate environment or repertoire
  probes before claiming the latent helps.
* `return_contrast >= 0.10`: different `z` choices create measurably
  different outcomes in the learned policy family. Proceed to routing-
  quality comparison (§4.2) and the specialization ledger (§6.8).

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

### 6.4b v5i9 CSIA training extension

v5i9 is a post-Summer training extension. It consumes the frozen forced-z
evidence produced by the v5i8 harness and adds a small detached reward
bonus only when causal strategy evidence is strong enough.

The required input files are:

```text
<forced_z_out_dir>/<checkpoint_stem>_qualitative_rollout_by_z.csv
<forced_z_out_dir>/<checkpoint_stem>_strategy_evidence.csv
```

Launch shape:

```bash
python rl/train_ppo.py `
    --preset v5i9 `
    --agents 4 `
    --total-steps 1000000 `
    --opponent-randomize `
    --opponent-pool OP5 OP6 OP7 `
    --map-layout map_b_split_lane_v2 `
    --csia-payoff-csv checkpoints/4v4/qualitative/<stem>_qualitative_rollout_by_z.csv `
    --csia-strategy-evidence-csv checkpoints/4v4/qualitative/<stem>_strategy_evidence.csv `
    --fresh-metrics-csv
```

The payoff matrix is `M[o,z] = E[Return | opponent=o, do(Z=z)]`, not a
natural-router correlation. The centered causal signal is
`S[o,z] = M[o,z] - mean_z M[o,*] - mean_o M[*,z] + mean M`.

The bonus is inactive unless all gates pass:

* Gate A: forced-z behavioral spread exceeds `csia_min_behavior_spread`.
* Gate B: centered opponent-latent interaction strength exceeds
  `csia_min_interaction_strength`.
* Gate C: every forced-z cell stays within `csia_quality_floor_delta` of
  the natural-router baseline for that opponent.

Update metrics include `reward_csia_mean`, `csia_interaction_strength`,
`centered_advantage_matrix`, `oracle_best_z_per_opponent`,
`router_oracle_gap`, `routing_gain`, `gate_A_pass`, `gate_B_pass`,
`gate_C_pass`, and `csia_bonus_active`.

Interpretation rule: v5i9 can support the extension claim "causal
strategic-impact feedback improves opponent-adaptive specialization" only
if it beats matched v5i8 controls and the forced-z evaluation after
training still shows behavior plus opponent-dependent differences. A
v5i9 win-rate gain without forced-z behavioral spread is performance
shaping, not evidence of learned latent strategies.

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

### 6.8 v6i1 latent specialization evaluation hierarchy

This section is the **owning interpretation** for whether a latent router
result demonstrates context-dependent strategic value. It applies to
v6i1 and to any headline claim that the learned router specializes
across contexts.

**Do not** equate occupancy entropy, marginal balance, or "all four latents
were used" with specialization. The headline requires evidence that
routing is strategically valuable, not merely diverse.

#### 6.8.1 What forced-z actually measures

Forced-z evaluation estimates the **learned** latent return surface:

```text
Q_theta(s, z) = E[ R | s, do(Z=z), pi_theta ]
```

It is the joint result of task opportunity, the learned repertoire
`pi_theta(·|o,z)`, evaluation design, and statistical power. It does
**not** isolate the bare MDP.

Observed crossover depends jointly on those factors:

```text
observed crossover = f(task opportunity, learned repertoire,
                     evaluation design, statistical power)
```

This is **conceptual**, not an additive variance decomposition. Perfect
evaluation cannot reveal an opportunity the repertoire never instantiated.

Phase A (v6i1 repertoire-before-routing) primarily targets the
**repertoire** term. Phase B/C ask whether the router can harvest
crossover that the learned family already exhibits.

#### 6.8.2 Evidence levels (evaluate in order)

| Level | Question | Primary evidence |
|-------|----------|------------------|
| **1. Controllability** | Does `do(Z=z_i) ≠ do(Z=z_j)` change behavior? | Matched-seed behavioral distances, route/lane/role telemetry, pairwise JSD or policy-level separation |
| **2. Competence** | Are all `z` usable, not one good policy plus broken curiosities? | Minimum forced-z return/WR gates, no catastrophic latent, within-latent stability |
| **3. Comparative advantage** | Are different `z` preferable in different observable contexts? | Pairwise advantage sign reversals, different `argmax_z M[o,z]` across slices, reliable opponent×latent interaction, `Delta_specialization > 0` |
| **4. Router utilization** | Does `q_phi` exploit available advantage? | Routing-gain ledger (§6.8.3): beat random **and** beat best constant `z` |

**Level 3 nuance.** Non-rank-1 opponent×latent payoff matrices and
nonzero centered advantages are **not** sufficient for crossover. A matrix
can have interaction yet share the same best `z` for every opponent.
Stronger evidence is **preference reversal**, e.g.

```text
argmax_z M[o1, z] ≠ argmax_z M[o2, z]
```

or sign reversal for some pair `(z_i, z_j)`:

```text
M[o1, zi] - M[o1, zj] > 0   while   M[o2, zi] - M[o2, zj] < 0
```

Use `M[o,z] = E[Return | opponent=o, do(Z=z)]` from forced-z evaluation
(§6.4a), not natural-router correlation. Centered advantages
`A[o,z] = M[o,z] - mean_z M[o,*] - mean_o M[*,z] + mean M` remove
global opponent difficulty and globally strong latents but still require
slice-level preference checks.

#### 6.8.3 Specialization gain and routing ledger

**Available specialization gain** (context-conditioned oracle vs best
constant latent), weighted by the evaluation opponent distribution
`p_eval(o)`:

```text
Delta_specialization
  = sum_o p_eval(o) * max_z M[o,z]
    - max_z sum_o p_eval(o) * M[o,z]
```

`Delta_specialization >= 0` always (the oracle can imitate the best
constant `z`). It measures gain available from conditioning on the
chosen context slice **among the evaluated learned policies**.

For the locked uniform `(OP5, OP6, OP7)` protocol, `p_eval(o) = 1/3`
and this reduces to the unweighted mean over opponents.

Map return metrics to the ledger (all on the **same** evaluation
distribution):

| Symbol | Definition | Question answered |
|--------|------------|-------------------|
| `G_available` | `J(q_oracle) - J(z_global-best)` | Is there anything useful to route over? |
| `G_realized` | `J(q_phi) - J(z_global-best)` | Does learned routing beat the best fixed `z`? |
| `G_random` | `J(q_phi) - J(q_random)` | Does learned routing beat uninformed selection? |
| `G_oracle-gap` | `J(q_oracle) - J(q_phi)` | How much exploitable value is left on the table? |

Identity:

```text
G_available = G_realized + G_oracle-gap
```

Repository CSV columns `routing_gain`, `router_oracle_gap`, and related
oracle fields should be reported against this ledger. Beating random
alone (`G_random > 0`) is weak evidence of context dependence; a router
that always picks the globally strongest latent can satisfy that without
specializing.

**Interpretation guard.** `G_oracle-gap ≈ 0` means little measurable
return advantage from switching `z` among **learned** policies under the
eval distribution. It does **not** prove the task lacks specialization
potential (identical per-`z` policies, Phase A failure, coarse slicing,
within-episode crossover, variance, or undiscovered alternatives are all
consistent with a small gap).

#### 6.8.4 Diagnostic table (repertoire × crossover × router)

| Repertoire | Crossover in learned Q_theta | Router outcome | Interpretation |
|------------|------------------------------|----------------|----------------|
| No | Unknown | Collapses | Phase A / representation failure |
| Yes | No | Collapses | Rational collapse under current repertoire |
| Yes | Yes | Collapses | Router, credit, observability, or optimization failure |
| Yes | Yes | Beats random only | May identify global-best `z` without context dependence |
| Yes | Yes | `G_realized > 0` | Genuine context-dependent routing value |
| Yes | Yes | Near oracle (`G_oracle-gap ≈ 0`) | Strong end-to-end latent result |

#### 6.8.5 Canonical decision logic and headline claim

1. **No controllability** — actor did not create meaningful
   latent-conditioned behaviors.
2. **Controllability, poor competence** — `z` affects behavior but some
   options are unusable.
3. **Competent repertoire, no available routing gain** —
   `Delta_specialization ≈ 0`; rational constant-`z` behavior.
4. **Available gain, no realized gain** — `G_available > 0` but
   `G_realized ≈ 0`; router/observability/credit/optimization failure.
5. **Positive realized gain** — `G_realized > 0`; genuine
   context-dependent routing over the best constant `z`.
6. **Router near oracle** — `G_oracle-gap ≈ 0` with positive
   `G_realized`; strong Summer latent result.

**Headline claim** (supported by Levels 1–2 evidence):

```text
G_available > 0   AND   G_realized > 0
```

Entropy and usage regularization do not create comparative advantage;
they only preserve access to latent options while learning searches the
task–policy combination. The testable target is:

```text
The learned repertoire exhibits observable, statistically reliable
context-dependent crossover; the router harvests part of that crossover.
```

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
* [ ] v6i1 / specialization claims: Levels 1–2 pass and `G_available > 0`, `G_realized > 0` per §6.8.
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
