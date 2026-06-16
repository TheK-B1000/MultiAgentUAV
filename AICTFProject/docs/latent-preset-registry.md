# Latent Preset Registry — classifications, aliases, resolved deltas

**Owner:** This file is the single source of truth for per-preset facts:
ancestry, aliases, classification, run-tag(s) shipped on disk, and the
resolved-configuration delta versus the canonical paper-faithful baseline
(`v5i4_paper_faithful_end_to_end`). Other docs link here rather than
restating preset facts.

> **Read first:**
> [`AGENTS.md`](../../AGENTS.md) ·
> [`summer-method-spec.md`](summer-method-spec.md) ·
> [`summer-fidelity-rules.md`](summer-fidelity-rules.md)
>
> **Read also:**
> [`experiment-and-evaluation-protocol.md`](experiment-and-evaluation-protocol.md) ·
> [`research-progress-tracker.md`](research-progress-tracker.md) ·
> [`Paper_experiment_alignment.md`](Paper_experiment_alignment.md)

---

## 1. How to use this file

* **Classifying a preset.** Look up the row in §3. The classification was
  computed against the resolved configuration per the decision tree in
  [`summer-fidelity-rules.md`](summer-fidelity-rules.md) §3.
* **Adding a new preset.** Complete the *Proposed Preset Review* template
  in [`summer-fidelity-rules.md`](summer-fidelity-rules.md) §8, then add
  a row to §3 here, then add a delta row to §6 (deltas vs canonical
  paper-faithful), and update §5 (alias map) and §7 (artifact run-tag
  history) if applicable. Tests in `tests/test_preset_resolution.py`
  pin alias resolution.
* **Computing the resolved delta.** Never infer from the preset name.
  Use `dataclasses.asdict(apply_preset(PPOConfig(), name))` and diff
  scalar fields against the parent's resolved dict; see
  `tools/snapshot_presets.py` and `tests/preset_snapshots.json`.

> "Plan-faithful" in a Python function name (e.g.
> `apply_plan_faithful_latent_phase3b_ablate_k1`) does **not** imply
> the row is `PAPER-FAITHFUL`. The function-name family includes
> ablations, diagnostics, and Summer-compatible extensions. The
> classification column in §3 is authoritative.

---

## 2. Canonical paper-faithful preset

The canonical operational paper-faithful row is **v5i4**:

| Property                | Value                                                                                                  |
|-------------------------|--------------------------------------------------------------------------------------------------------|
| Function                | `apply_plan_faithful_latent_v5i4_end_to_end`                                                           |
| File                    | [`rl/presets/plan_faithful.py`](../rl/presets/plan_faithful.py) (line 1614 at the time of writing)     |
| Parent (inheritance)    | `apply_plan_faithful_latent_v5_strict_summer` (literal-strict ablation)                                |
| Classification          | `PAPER-FAITHFUL`                                                                                        |
| `latent_k`              | `4`                                                                                                     |
| `latent_z_embed_dim`    | `16`                                                                                                    |
| Actor conditioning      | `nn.Embedding(K, d_z)` concat (`latent_actor_conditioning = "concat"`; FiLM / adapter / one-hot OFF)   |
| `latent_strategy_ppo_coef` (`c_Z`) | `0.10`                                                                                      |
| `latent_lam_p`          | `0.03`                                                                                                  |
| `latent_lam_h` schedule | `0.003 → 0.0002` linear over `0..300_000` global steps                                                 |
| `latent_resample_every_n` | `64`                                                                                                  |
| `latent_resample_on_flag` | `False`                                                                                               |
| `latent_episode_strategy_lr` | `None` (no dedicated router optimizer)                                                            |
| `latent_forced_z_episode_frac` (+ four `_start/_end/anneal_*` fields) | `0.0` (all `None`); resolver returns `0.0` at every step |
| `total_timesteps`       | `1_000_000` (PPOConfig default; v5i4 does not override)                                                |
| `opponent_pool`         | `("OP5", "OP6", "OP7")` (inherited from v4i1)                                                          |
| `opponent_randomize`    | `True` (inherited from v4i1)                                                                            |
| Audit banner            | Fires when `cfg.run_tag` contains `"v5i4_paper_faithful"` (or `cfg.latent_paper_faithful_audit = True`) |

**Aliases (all seven resolve to the same `PPOConfig`):**

```text
v5i4
v5i4_paper_faithful
v5i4_end_to_end
paper_faithful_end_to_end
latent_v5i4_paper_faithful
latent_v5i4_end_to_end
plan_faithful_latent_v5i4_end_to_end
```

Pinned by
[`tests/test_v5i4_paper_faithful.py::V5i4AliasSnapshotTests`](../tests/test_v5i4_paper_faithful.py)
and the snapshot in
[`tests/preset_snapshots.json`](../tests/preset_snapshots.json).

---

## 3. Classification table (operational latent-strategy presets)

Six classifications, per
[`summer-fidelity-rules.md`](summer-fidelity-rules.md) §3:
`PAPER-FAITHFUL`, `SUMMER-COMPATIBLE EXTENSION`, `ABLATION`,
`DIAGNOSTIC`, `DEPRECATED`, `UNKNOWN`. Only the v5 ladder and the most
recent v4 / v3i19 presets are tabulated; older `v3iN`, `phaseN`, and
`hypothesis_*` presets are kept reachable for reproducibility under
`DEPRECATED` (§4).

### 3.1 v5 ladder (operational, current)

| Preset (apply fn) | Function name suffix | Parent | Classification | One-line reason |
|---|---|---|---|---|
| `v5_strict_summer` | `apply_plan_faithful_latent_v5_strict_summer` | `v4i3_summer_proof` | `ABLATION` | Literal docs/algorithm.md loss: entropy + persistence only on `q_phi`, **no** task-reward PG channel. Exists to test whether the literal equation alone trains `q_phi`. |
| `v5i1_reward_credit_router` | `apply_plan_faithful_latent_v5i1_reward_credit_router` | `v5_strict_summer` | `SUMMER-COMPATIBLE EXTENSION` | Adds per-episode router PPO on `q_phi` with dedicated AdamW (`latent_episode_strategy_lr = 5e-3`). Compound delta: also flips `latent_resample_every_n = 0`, `latent_lam_p = 0`, entropy schedule, marginal baseline ON. |
| `v5i2_stronger_z_conditioning` | `apply_plan_faithful_latent_v5i2_stronger_z_conditioning` | `v5i1_reward_credit_router` | `SUMMER-COMPATIBLE EXTENSION` | v5i1 + actor-only FiLM (`enable_actor_z_film = True`, `actor_z_film_init_scale = 0.02`, `actor_z_film_layer = 2`). FiLM is a post-Summer actor-conditioning extension. |
| `v5i3_balanced_warmup` | `apply_plan_faithful_latent_v5i3_balanced_warmup` | `v5i2_stronger_z_conditioning` | `SUMMER-COMPATIBLE EXTENSION` | v5i2 + forced-z anneal (`0.30 → 0.00` across `200_000 → 500_000`). Forcing is unlabeled uniform exploration, not role assignment; routed via `latent_preference_buffer` so `q_phi`'s PPO update stays on-policy. |
| **`v5i4_end_to_end`** | **`apply_plan_faithful_latent_v5i4_end_to_end`** | **`v5_strict_summer`** | **`PAPER-FAITHFUL`** | **Canonical paper-faithful operational row.** v5_strict_summer + `latent_strategy_ppo_coef = 0.10` (the on-policy categorical PPO term on `q_phi` that is the paper's "trained end-to-end from task reward" gradient channel). No FiLM, no episode-credit, no forced-z, no aux heads. |
| **`v5i5_paper_faithful_entropy_floor`** | **`apply_plan_faithful_latent_v5i5_paper_faithful_entropy_floor`** | **`v5i4_end_to_end`** | **`PAPER-FAITHFUL`** | **Single-axis follow-up to v5i4** to combat the v5i4 router's late-training occupancy collapse (~64% on z2 / ~7% on z3 at 1 M steps). Resolved diff vs v5i4 is exactly `{latent_lam_h_end: 0.0002 → 0.001, run_tag}`. The new floor stays inside the documented Summer-plan `[0.001, 0.01]` entropy range, so no fidelity rule (R1..R42) flips state. Adds occupancy-collapse diagnostics (`effective_num_latents`, `latent_occupancy_{min,max,ratio}`, `mean_strategy_duration`) but no new gradient channel and no new objective term. |

### 3.2 v4 family (Summer-proof + post-Summer extension)

| Preset (apply fn) | Parent | Classification | One-line reason |
|---|---|---|---|
| `v4i1_strategic_pressure_qprobe` | `v3i19_summer_consequence` | `SUMMER-COMPATIBLE EXTENSION` | v3i19 verbatim + opponent pool restricted to `(OP5, OP6, OP7)` to ensure different `z` values have a strategic reason to differ. Latent machinery (arc-credit, FiLM scaffolding) inherited from the v3iN chain — not literal Summer. |
| `v4i3_summer_proof` | `v4i1_strategic_pressure_qprobe` | `SUMMER-COMPATIBLE EXTENSION` (arc-credit row) | Same as v4i1; explicit guards on `latent_router_distill_enabled = False` and aux heads. Carries `latent_arc_credit_enabled = True, coef = 1.0` from v3i19 → **not** literal paper-faithful. Use as the *arc-credit row* of the proof table, not the headline paper row. |
| `no_latent_v4i3_baseline` | `v4i1_strategic_pressure_qprobe` | `DIAGNOSTIC` (no-latent control) | Same-everything-except-z control: inherits v4i1 verbatim and flips `use_latent_strategy = False`. The honest no-latent baseline against which any latent-on row's headline WR delta is measured. |
| `v4i4post_periodic_router_distill` | (built on v4i1) | `SUMMER-COMPATIBLE EXTENSION` | Periodic Return-Ranked Router Distillation; was the old "v4i3", explicitly renamed and reclassified as **post-Summer**. Introduces counterfactual router supervision (offline q_probe targets), which the Summer plan's "no labels, no auxiliary objectives" clause forbids in the literal row. |

### 3.3 Earlier latent-strategy chain (v3iN)

Treated as `DEPRECATED` for new headline claims unless explicitly invoked
as an ablation row. They remain reachable via the registry for
reproducibility. Notable members:

| Preset (apply fn) | Why DEPRECATED for new headline claims |
|---|---|
| `v3i19_summer_consequence` | Introduced the arc-credit channel `latent_arc_credit_enabled = True` (`coef = 1.0`, `baseline = "context_value"`). Useful as the arc-credit row; **not** literal Summer because the paper's locked loss in `docs/algorithm.md` contains no per-arc PG term. |
| `v3i18_v3i16_plus_128` | v3i16 + 128-d expansions; superseded by v3i19. |
| `v3i16_policy_z_embedding` | Restored the plain `nn.Embedding` policy-z path on top of the v3iN preference / distillation chain. Operationally superseded by `v5_strict_summer`, which is the cleaner "literal strict" row. |
| `v3i15_sparse_tactical_refresh` / `v3i15_strong_separation` | Event-triggered refresh + separation losses. Both forbidden in the paper-faithful row (R26, R40). |
| `v3i7_advantage_weighted_router_distill` | AWRD — distillation with advantage weighting. Forbidden in the paper-faithful row (R35). |
| `v3i3_event_conditioned_preference` / `v3i4_event_progress_preference` | Preference learning. Forbidden in the paper-faithful row (R32–R34). |
| `v3i9_specialist_router` / `v3i10_role_phase_specialist` | Specialist / role-labelled routers. Forbidden in the paper-faithful row (R36) and by the §3 no-role-labels rule in `summer-method-spec.md`. |
| `v3iN` not listed above | Each carried at least one forbidden flag (FiLM scaffolding, adapter, aux head, preference target, distillation, separation, specialist). Audit per-preset via `dataclasses.asdict` if a particular row is needed for a comparison. |

### 3.4 Ablation / diagnostic family

| Preset (apply fn) | Classification | Purpose |
|---|---|---|
| `plan_faithful_latent_k1` | `ABLATION` | `latent_k = 1` (collapsed latent). Controls whether any of v5i4's gains come from latent capacity at all. |
| `plan_faithful_latent_no_persistence` | `ABLATION` | `latent_lam_p = 0` ablation against the paper-faithful row. |
| `plan_faithful_latent_no_entropy` | `ABLATION` | `latent_lam_h = 0` ablation. |
| `plan_faithful_latent_phase3b_ablate_k1` | `ABLATION` | Lineage-matched K=1 ablation for the phase3b family. |
| `plan_faithful_latent_phase3b_ablate_no_persistence` | `ABLATION` | Lineage-matched no-persistence ablation. |
| `plan_faithful_no_latent` | `DIAGNOSTIC` (legacy no-latent) | Legacy 1M-step 2v2 OP3 baseline. Does **not** mirror v4i1 — using it as a headline no-latent control would confound the latent ablation with ~8 other deltas (timesteps, team size, opponent pool, reward shaping, ...). Use `no_latent_v4i3_baseline` instead. |
| `qualitative_smoke`, `qualitative_smoke_baseline` (checkpoint dirs, not presets) | `DIAGNOSTIC` | Smoke artifacts produced by `tools/qualitative_rollout.py`. |

### 3.5 Hypothesis / other families

`hypothesis_*` and `latent_op3_*` presets are historical and serve
ad-hoc experiments. Mark `DEPRECATED` for new headline claims; consult
`rl/presets/hypothesis.py` and `rl/presets/other.py` for invocations
that still depend on them.

---

## 4. Deprecated naming surface (kept reachable, do not extend)

The registry retains aliases for older naming conventions so checkpoints
and CSVs produced before the renames still resolve. Any new preset
**must** be added with both the canonical `plan_faithful_latent_<X>`
key and the short `latent_<X>` / `<X>` alias trio, matching the v4 / v5
convention.

| Legacy alias example                     | Status     | Resolves to (current canonical)              |
|------------------------------------------|------------|----------------------------------------------|
| `latent_step6`                           | DEPRECATED | `apply_plan_faithful_latent_step6`           |
| `plan_faithful_latent_intent_credit`     | DEPRECATED | `apply_plan_faithful_latent_episode_strategic` |
| `latent_recommended` family              | DEPRECATED | `apply_plan_faithful_latent`                 |
| `plan_faithful_collapsed_latent`         | DEPRECATED | `apply_plan_faithful_latent_k1`              |
| `plan_faithful_latent_fix_d`             | DEPRECATED | `apply_plan_faithful_latent_option_a`        |

`PRESET_REGISTRY` lists the full alias map in
[`rl/presets/__init__.py`](../rl/presets/__init__.py).
[`tests/test_preset_resolution.py`](../tests/test_preset_resolution.py)
pins the alias map; any deletion or rename must be reflected there.

---

## 5. Alias map for the operational ladder

| Apply function                                              | Aliases (all resolve identically)                                                                                                                                                              |
|-------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `apply_plan_faithful_latent_v4i1_strategic_pressure_qprobe` | `plan_faithful_latent_v4i1_strategic_pressure_qprobe`, `latent_v4i1_strategic_pressure_qprobe`, `latent_v4i1`, `v4i1`                                                                          |
| `apply_plan_faithful_latent_v4i3_summer_proof`              | `plan_faithful_latent_v4i3_summer_proof`, `latent_v4i3_summer_proof`, `latent_v4i3`, `v4i3`                                                                                                    |
| `apply_plan_faithful_no_latent_v4i3_baseline`               | `plan_faithful_no_latent_v4i3_baseline`, `no_latent_v4i3_baseline`, `no_latent_v4i3`, `v4i3_no_latent`, `v4i3_no_latent_baseline`                                                              |
| `apply_plan_faithful_latent_v5_strict_summer`               | `plan_faithful_latent_v5_strict_summer`, `latent_v5_strict_summer`, `v5_strict_summer`, `v5_strict`, `v5`, `strict_summer`                                                                     |
| `apply_plan_faithful_latent_v5i1_reward_credit_router`      | `plan_faithful_latent_v5i1_reward_credit_router`, `latent_v5i1_reward_credit_router`, `v5i1_reward_credit_router`, `v5i1`                                                                      |
| `apply_plan_faithful_latent_v5i2_stronger_z_conditioning`   | `plan_faithful_latent_v5i2_stronger_z_conditioning`, `latent_v5i2_stronger_z_conditioning`, `v5i2_stronger_z_conditioning`, `v5i2`                                                             |
| `apply_plan_faithful_latent_v5i3_balanced_warmup`           | `plan_faithful_latent_v5i3_balanced_warmup`, `latent_v5i3_balanced_warmup`, `v5i3_balanced_warmup`, `v5i3`, `balanced_warmup`                                                                  |
| **`apply_plan_faithful_latent_v5i4_end_to_end`**            | **`plan_faithful_latent_v5i4_end_to_end`, `latent_v5i4_end_to_end`, `latent_v5i4_paper_faithful`, `v5i4_end_to_end`, `v5i4_paper_faithful`, `paper_faithful_end_to_end`, `v5i4`**              |
| **`apply_plan_faithful_latent_v5i5_paper_faithful_entropy_floor`** | **`plan_faithful_latent_v5i5_paper_faithful_entropy_floor`, `plan_faithful_latent_v5i5_entropy_floor`, `latent_v5i5_paper_faithful_entropy_floor`, `latent_v5i5_entropy_floor`, `latent_v5i5_paper_faithful`, `v5i5_paper_faithful_entropy_floor`, `v5i5_paper_faithful`, `v5i5_entropy_floor`, `v5i5`, `paper_faithful_entropy_floor`** |
| `apply_plan_faithful_latent_v4i4post_periodic_router_distill` | `plan_faithful_latent_v4i4post_periodic_router_distill`, `latent_v4i4post_periodic_router_distill`, `latent_v4i4post`, `v4i4post`, `v4i4`                                                    |

The full alias surface lives in
[`rl/presets/__init__.py::PRESET_REGISTRY`](../rl/presets/__init__.py).

---

## 6. Resolved-configuration delta vs canonical paper-faithful (v5i4)

Each row lists fields whose resolved value differs from v5i4 (i.e. from
`dataclasses.asdict(apply_preset(PPOConfig(), "v5i4"))`). All other
scalar fields match v5i4 (re-asserted defensively inside each preset).

Fields not listed have the v5i4 value. Boolean fields are written as
`True` / `False`; numeric fields preserve type. The "Forbidden flag?"
column flags whether the change enables a mechanism listed in
[`summer-method-spec.md`](summer-method-spec.md) §10 / R-rules in
[`summer-fidelity-rules.md`](summer-fidelity-rules.md).

### 6.1 v5_strict_summer (ABLATION — literal Summer equation)

| Field                                | v5i4 value | This preset | Forbidden flag? | Note                                                                 |
|--------------------------------------|------------|-------------|-----------------|----------------------------------------------------------------------|
| `latent_strategy_ppo_coef`           | `0.10`     | `0.0`       | No              | Removes the only task-reward gradient channel on `q_phi`. Diagnostic. |
| `run_tag`                            | `v5i4_paper_faithful_end_to_end_OP5_OP6_OP7_1m_4v4` | `v5_strict_summer_OP5_OP6_OP7_2m_4v4` | — | See §7 for the `_2m_` / `_1m_` artifact-history discrepancy. |

### 6.2 v5i1_reward_credit_router (SUMMER-COMPATIBLE EXTENSION)

| Field                                          | v5i4 value | This preset | Forbidden flag? | Note                                                                  |
|------------------------------------------------|------------|-------------|-----------------|-----------------------------------------------------------------------|
| `latent_strategy_ppo_coef`                     | `0.10`     | `0.0`       | —               | Replaced by the episode-credit channel below.                          |
| `latent_episode_strategy_ppo`                  | `False`    | `True`      | **Yes (R30)**   | Episode-credit extension.                                              |
| `latent_episode_strategy_coef`                 | `0.0`      | `0.30`      | —               | Episode-credit weight.                                                 |
| `latent_episode_strategy_lr`                   | `None`     | `5e-3`      | **Yes (R10)**   | Dedicated router AdamW; silences main-loop PG via the v5 gate.         |
| `latent_episode_strategy_n_epochs`             | `1`        | `6`         | —               | Router PPO inner epochs.                                               |
| `latent_episode_strategy_warmup_decision_steps`| `0`        | `5`         | —               | Per-episode warmup before z commit.                                    |
| `latent_q_phi_marginal_baseline`               | (default)  | `True`      | —               | Detached z-marginal value baseline.                                    |
| `latent_resample_every_n`                      | `64`       | `0`         | No              | Episode-start only (still allowed by R23, but a different cadence).   |
| `latent_lam_p`                                 | `0.03`     | `0.0`       | —               | Persistence disabled.                                                  |
| `latent_lam_h_start`                           | `0.003`    | `0.003`     | —               | Anneal **window** changed: `200_000 → 700_000` (not `0 → 300_000`).    |
| `latent_lam_h_end`                             | `0.0002`   | `0.001`     | —               | Anneal floor raised (`0.001` collapse insurance).                      |
| `run_tag`                                      | (v5i4 tag) | `v5i1_reward_credit_router_OP5_OP6_OP7_2m_4v4` | — | `_2m_` tag retained (see §7).                                          |

### 6.3 v5i2_stronger_z_conditioning (SUMMER-COMPATIBLE EXTENSION)

Inherits v5i1's deltas, plus:

| Field                          | v5i4 value | This preset | Forbidden flag? | Note                                                  |
|--------------------------------|------------|-------------|-----------------|-------------------------------------------------------|
| `enable_actor_z_film`          | `False`    | `True`      | **Yes (R11)**   | Actor-only FiLM gating from learned z embedding.       |
| `actor_z_film_init_scale`      | `0.0`      | `0.02`      | —               | Near-identity init preserves embedding-concat at t=0. |
| `actor_z_film_layer`           | (default)  | `2`         | —               | Applies in actor's second hidden layer.                |
| `run_tag`                      | (v5i4 tag) | `v5i2_stronger_z_conditioning_OP5_OP6_OP7_2m_4v4` | — | `_2m_` tag retained. |

### 6.4 v5i3_balanced_warmup (SUMMER-COMPATIBLE EXTENSION)

Inherits v5i2's deltas, plus:

| Field                                       | v5i4 value | This preset | Forbidden flag? | Note                                                                  |
|---------------------------------------------|------------|-------------|-----------------|-----------------------------------------------------------------------|
| `latent_forced_z_episode_frac`              | `0.0`      | `0.30`      | **Yes (R27)**   | Constant safety value (resolver reads `_start/_end` first).            |
| `latent_forced_z_episode_frac_start`        | `None`     | `0.30`      | **Yes (R28)**   | Anneal start fraction.                                                 |
| `latent_forced_z_episode_frac_end`          | `None`     | `0.00`      | **Yes (R28)**   | Anneal end fraction.                                                   |
| `latent_forced_z_anneal_start`              | `None`     | `200_000`   | **Yes (R28)**   | Anneal start step.                                                     |
| `latent_forced_z_anneal_end`                | `None`     | `500_000`   | **Yes (R28)**   | Anneal end step.                                                       |
| `run_tag`                                   | (v5i4 tag) | `v5i3_balanced_warmup_OP5_OP6_OP7_2m_4v4` | — | `_2m_` tag retained. |

`tests/test_forced_z_anneal.py::ForcedZScheduleResolverTests` pins the
resolver behavior: for any preset whose four `_start/_end` fields are
`None` (v5i4 and earlier), the resolver returns
`latent_forced_z_episode_frac` (which is `0.0` for v5i4).

### 6.5 v4i3_summer_proof (SUMMER-COMPATIBLE EXTENSION — arc-credit row)

| Field                          | v5i4 value | This preset | Forbidden flag? | Note                                                              |
|--------------------------------|------------|-------------|-----------------|-------------------------------------------------------------------|
| `latent_arc_credit_enabled`    | `False`    | `True`      | **Yes (R29)**   | v3i19 arc-credit PG channel (baseline = `"context_value"`).        |
| `latent_arc_credit_coef`       | `0.0`      | `1.0`       | —               | Arc-credit weight.                                                 |
| `latent_strategy_ppo_coef`     | `0.10`     | `0.0`       | —               | Replaced by arc-credit channel above.                              |
| `enable_actor_z_film`          | `False`    | (v3iN FiLM scaffolding inherited; audit before re-running) | **Yes if True** | Verify resolved value via `dataclasses.asdict` per run.            |
| `run_tag`                      | (v5i4 tag) | `v4i3_summer_proof_OP5_OP6_OP7_4v4` | — | Budget-agnostic tag (no `_2m_`/`_1m_` suffix).                     |

### 6.6 no_latent_v4i3_baseline (DIAGNOSTIC — same-everything-except-z control)

| Field                          | v5i4 value | This preset | Forbidden flag? | Note                                                                  |
|--------------------------------|------------|-------------|-----------------|-----------------------------------------------------------------------|
| `use_latent_strategy`          | `True`     | `False`     | —               | Disables the entire latent stack (`q_phi`, z embedding, persistence). |
| `latent_strategy_ppo_coef`     | `0.10`     | `0.0`       | —               | No-op when latent off; zeroed defensively.                            |
| `latent_arc_credit_enabled`    | `False`    | `False`     | —               | Inherited from v4i1's True; explicitly zeroed for ablation cleanliness.|
| `latent_actor_z_onehot_enabled`| `False`    | `False`     | —               | Explicitly zeroed.                                                    |
| `run_tag`                      | (v5i4 tag) | `v4i3_no_latent_baseline_OP5_OP6_OP7_4v4` | — | Budget-agnostic.                                                       |

All other latent-only fields become no-ops when
`use_latent_strategy = False`. Use this as the headline no-latent
control for v5i4 (and v4i3) headline WR deltas; do **not** use
`apply_plan_faithful_no_latent` (the legacy 2v2 OP3 1M baseline), which
diverges in eight other axes.

### 6.7 v4i4post_periodic_router_distill (SUMMER-COMPATIBLE EXTENSION — post-Summer)

| Field                                          | v5i4 value | This preset | Forbidden flag? | Note                                                                       |
|------------------------------------------------|------------|-------------|-----------------|----------------------------------------------------------------------------|
| `latent_router_distill_enabled`                | `False`    | `True`      | **Yes (R31)**   | Periodic Return-Ranked Router Distillation; counterfactual router targets.|
| `latent_router_distill_every_n_steps`          | `250_000`  | `250_000`   | —               | Distillation cadence.                                                       |
| (multiple ancestry-inherited fields)           | —          | —           | —               | Inherits v4i1 / v3iN machinery; audit per run via `dataclasses.asdict`.    |

Post-Summer extension by classification: distillation against an offline
q_probe target reintroduces an auxiliary supervised objective on `q_phi`,
which the Summer plan forbids in the literal row.

### 6.8 v5i5_paper_faithful_entropy_floor (PAPER-FAITHFUL — single-axis)

The v5i4 → v5i5 resolved-config diff is **exactly two keys**
(`latent_lam_h_end`, `run_tag`). This is enforced by
[`tests/test_v5i5_paper_faithful_entropy_floor.py::V5i5PresetInheritanceTests::test_v5i5_minimal_diff_vs_v5i4`](../tests/test_v5i5_paper_faithful_entropy_floor.py).

| Field                  | v5i4 value | This preset | Forbidden flag? | Note                                                                                  |
|------------------------|------------|-------------|-----------------|---------------------------------------------------------------------------------------|
| `latent_lam_h_end`     | `0.0002`   | `0.001`     | No (R20 ✓)      | Single-axis change: raise the entropy floor 5× to combat the v5i4 occupancy collapse. |
| `run_tag`              | `v5i4_paper_faithful_end_to_end_OP5_OP6_OP7_1m_4v4` | `v5i5_paper_faithful_entropy_floor_OP5_OP6_OP7_1m_4v4` | — | Family-aware audit banner detects this tag automatically. |

`latent_lam_h_start` stays at `0.003`. The anneal window stays
`[0, 300_000]`. The new floor `0.001` is **inside** the documented
Summer-plan `[0.001, 0.01]` entropy range, so R20 (`λ_H > 0`) and R21
(`latent_entropy_objective = "maximize"`) remain satisfied. The actor
pathway is identical to v5i4 (no FiLM, no adapter, no one-hot;
embedding-concat only at input dim 164). The router gradient channel is
identical (main-loop categorical PPO term at `latent_strategy_ppo_coef
= 0.10`). No forbidden mechanism is enabled.

**New diagnostics** (see also [`summer-method-spec.md`](summer-method-spec.md) §"Telemetry"):

| Column                          | Source                                                                  | Range / interpretation                            |
|---------------------------------|-------------------------------------------------------------------------|---------------------------------------------------|
| `latent_marginal_entropy_nats`  | `H` of rollout-marginal `z` distribution                                | `0` (full collapse) to `ln(K)` (uniform).         |
| `effective_num_latents`         | `exp(latent_marginal_entropy_nats)`                                     | `1` (full collapse) to `K` (uniform).             |
| `latent_occupancy_min`          | `min_k strategy_occupancy_k`                                            | `[0, 1]`.                                         |
| `latent_occupancy_max`          | `max_k strategy_occupancy_k`                                            | `[1/K, 1]`.                                       |
| `latent_occupancy_ratio`        | `latent_occupancy_max / max(latent_occupancy_min, 1e-8)`                | `1.0` = uniform; `>>1` = severe imbalance.        |
| `mean_strategy_duration`        | `total_decisions / max(1, strategy_resample_count)`                      | Mean dwell length (decisions) per latent arc.     |

These are pure functions of the per-z counts already computed for v5i4;
they add **no** new gradient channel and **no** new objective term.

### 6.9 plan_faithful_latent_k1 (ABLATION)

| Field        | v5i4 value | This preset | Forbidden flag? | Note                                                       |
|--------------|------------|-------------|-----------------|------------------------------------------------------------|
| `latent_k`   | `4`        | `1`         | —               | Collapsed-latent ablation (R2 says `4` is canonical; `1` is a deliberate single-axis change to measure whether latent capacity matters at all). |

---

## 7. Run-tag artifact history (canonical vs on-disk)

### 7.1 v5i4 `_2m_` → `_1m_` correction

**Background.** v4i1 introduced the run-tag suffix `_2m_4v4` reflecting
its originally intended 2 M-step budget. v5_strict_summer / v5i1 / v5i2 /
v5i3 inherited that suffix verbatim without overriding
`total_timesteps`, which stayed at the `PPOConfig` default of
`1_000_000`. v5i4's first launches inherited the same misleading
`_2m_4v4` tag and produced checkpoints / CSVs on disk under that name.

**Fix.** [`apply_plan_faithful_latent_v5i4_end_to_end`](../rl/presets/plan_faithful.py)
now sets `cfg.run_tag = "v5i4_paper_faithful_end_to_end_OP5_OP6_OP7_1m_4v4"`
so the tag agrees with the trainer's reported total timesteps. Pinned by
[`tests/test_v5i4_paper_faithful.py::V5i4RunTagAndInitialOpponentConsistencyTests`](../tests/test_v5i4_paper_faithful.py)
(`test_run_tag_advertises_actual_total_timesteps_budget`,
`test_initial_opponent_falls_back_to_pool_first_entry`,
`test_initial_opponent_respects_explicit_in_pool_fixed_tag`,
`test_initial_opponent_unchanged_for_fixed_mode_presets`). Snapshot delta
captured in
[`tests/preset_snapshots.json`](../tests/preset_snapshots.json) (all
seven v5i4 aliases updated, no other scalar drift).

**Artifact-history discrepancy.** Checkpoints, metrics CSVs, and lock
files produced by v5i4 launches **before** the tag fix carry the old
`_2m_4v4` suffix. Specifically:

```text
AICTFProject/checkpoints/4v4/
  ckpt_v5i4_paper_faithful_end_to_end_OP5_OP6_OP7_2m_4v4_*.zip
  final_v5i4_paper_faithful_end_to_end_OP5_OP6_OP7_2m_4v4.zip
  v5i4_paper_faithful_end_to_end_OP5_OP6_OP7_2m_4v4_metrics.csv
  v5i4_paper_faithful_end_to_end_OP5_OP6_OP7_2m_4v4_episodes.csv
  v5i4_paper_faithful_end_to_end_OP5_OP6_OP7_2m_4v4_e3_steps.csv
  v5i4_paper_faithful_end_to_end_OP5_OP6_OP7_2m_4v4.run.lock
```

These artifacts are the **completed 1 M-step paper-faithful run**
(progress tracker entry §2.1 in
[`research-progress-tracker.md`](research-progress-tracker.md)). They
are valid evaluation inputs; only the run-tag string in the filename
disagrees with the actual `total_timesteps = 1_000_000` budget reported
by the trainer at run start.

**Reproducibility note.** New v5i4 launches produced after this commit
will write artifacts under
`v5i4_paper_faithful_end_to_end_OP5_OP6_OP7_1m_4v4_*`. Comparison runs
should preserve the artifact namespace they intend to compare; do not
mix `_2m_4v4` (pre-fix) and `_1m_4v4` (post-fix) artifacts under the
same evaluation directory without explicit tag mapping.

**Why not retroactively rename on disk.** Renaming changes the
`run_tag` field embedded in the CSV / checkpoint metadata as well as the
filename, breaking `--metrics-csv` consumers that load by tag. Treat
pre-fix artifacts as immutable and rely on filename-level routing.

### 7.2 v5_strict_summer / v5i1 / v5i2 / v5i3 retain `_2m_`

The other v5 presets keep the misleading `_2m_4v4` suffix to preserve
existing artifact paths from completed runs. Any new comparison run
should override `--run-tag` to advertise the actual budget, or accept
that the suffix is historical.

### 7.3 v4i3 family `_4v4` (budget-agnostic)

`v4i3_summer_proof_OP5_OP6_OP7_4v4` and
`v4i3_no_latent_baseline_OP5_OP6_OP7_4v4` deliberately omit a
budget suffix because the preset locks the Summer-faithful machinery,
not a specific `--total-steps`. Probes at smaller budgets share the
same artifact namespace; pass `--run-tag` to disambiguate when needed.

---

## 8. Per-preset audit banner trigger

| Preset              | Banner fires?                                                            |
|---------------------|--------------------------------------------------------------------------|
| `v5i4_paper_faithful_end_to_end` (any alias) | **Yes.** `cfg.run_tag` contains `"v5i4_paper_faithful"`. |
| Any other preset    | Only if `cfg.latent_paper_faithful_audit = True` is set on the resolved config. |

The banner content and trigger logic live in
[`rl/training/banner.py::_maybe_print_paper_faithful_audit`](../rl/training/banner.py)
and are pinned by
[`tests/test_v5i4_paper_faithful.py::V5i4PaperFaithfulAuditBannerTests`](../tests/test_v5i4_paper_faithful.py).
The required banner lines and the three explicit `WARNING` cases are
listed in [`summer-fidelity-rules.md`](summer-fidelity-rules.md) §5.

---

## 9. Snapshot pinning

[`tests/preset_snapshots.json`](../tests/preset_snapshots.json) records
the resolved `PPOConfig` for every alias in `PRESET_REGISTRY`. Snapshot
deltas must be explicit:

1. Regenerate with `python tools/snapshot_presets.py`.
2. Diff against the prior HEAD snapshot (`json.load` + set-diff;
   `tests/test_preset_resolution.py` is the canonical loader).
3. Document the diff in
   [`Paper_experiment_alignment.md`](Paper_experiment_alignment.md) §7
   "Changelog" with the affected aliases and the affected fields.

Unintended scalar drift in any pre-existing field is a regression; the
PR must either roll it back or explicitly justify it under the *Proposed
Preset Review* template in
[`summer-fidelity-rules.md`](summer-fidelity-rules.md) §8.

---

## 10. Cross-references

| Need                                              | Where to look                                                                       |
|---------------------------------------------------|-------------------------------------------------------------------------------------|
| Scientific definition of the paper method         | [`summer-method-spec.md`](summer-method-spec.md)                                    |
| Fidelity rules / classification / proposal form   | [`summer-fidelity-rules.md`](summer-fidelity-rules.md)                              |
| Evaluation, statistical protections, comparisons  | [`experiment-and-evaluation-protocol.md`](experiment-and-evaluation-protocol.md)    |
| Current run status / next experiment              | [`research-progress-tracker.md`](research-progress-tracker.md)                      |
| Mandatory agent behavior                          | [`AGENTS.md`](../../AGENTS.md)                                                      |
| Code↔manuscript trace                             | [`Paper_experiment_alignment.md`](Paper_experiment_alignment.md)                    |
| Spec-to-code trace                                | [`../../docs/Summer_Implementation_Plan_Implementation_Details_Trace.md`](../../docs/Summer_Implementation_Plan_Implementation_Details_Trace.md) |
| Algorithm sketch                                  | [`../../docs/algorithm.md`](../../docs/algorithm.md)                                |
