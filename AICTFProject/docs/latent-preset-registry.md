# Latent Preset Registry — classifications, aliases, resolved deltas

**Owner:** This file is the single source of truth for per-preset facts:
ancestry, aliases, classification, run-tag(s) shipped on disk, and the
resolved-configuration deltas for the paper-faithful ladder. The current
canonical paper-faithful baseline is
`v5i6_paper_faithful_marginal_entropy`. Other docs link here rather than
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
  a row to §3 here, then add a delta row to §6, and update §5 (alias map)
  and §7 (artifact run-tag
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

The canonical operational paper-faithful row is **v5i6**:

| Property                | Value                                                                                                  |
|-------------------------|--------------------------------------------------------------------------------------------------------|
| Function                | `apply_plan_faithful_latent_v5i6_paper_faithful_marginal_entropy`                                      |
| File                    | [`rl/presets/plan_faithful.py`](../rl/presets/plan_faithful.py)                                        |
| Parent (inheritance)    | `apply_plan_faithful_latent_v5i4_end_to_end`                                                           |
| Classification          | `PAPER-FAITHFUL`                                                                                        |
| `latent_k`              | `4`                                                                                                     |
| `latent_z_embed_dim`    | `16`                                                                                                    |
| Actor conditioning      | `nn.Embedding(K, d_z)` concat (`latent_actor_conditioning = "concat"`; FiLM / adapter / one-hot OFF)   |
| `latent_strategy_ppo_coef` (`c_Z`) | `0.10`                                                                                      |
| `latent_lam_p`          | `0.03`                                                                                                  |
| `latent_lam_h` schedule | `0.003 → 0.001` linear over `0..300_000` global steps                                                  |
| Entropy mode            | Batch-marginal strategy entropy (`latent_entropy_mode = "marginal"`)                                   |
| `latent_resample_every_n` | `64`                                                                                                  |
| `latent_resample_on_flag` | `False`                                                                                               |
| `latent_episode_strategy_lr` | `None` (no dedicated router optimizer)                                                            |
| `latent_forced_z_episode_frac` (+ four `_start/_end/anneal_*` fields) | `0.0` (all `None`); resolver returns `0.0` at every step |
| `total_timesteps`       | `1_000_000` (PPOConfig default; v5i4 does not override)                                                |
| `opponent_pool`         | `("OP5", "OP6", "OP7")` (inherited from v4i1)                                                          |
| `opponent_randomize`    | `True` (inherited from v4i1)                                                                            |
| Audit banner            | Fires when `cfg.run_tag` contains `"v5i6_paper_faithful"` (or `cfg.latent_paper_faithful_audit = True`) |

**Aliases (all ten resolve to the same `PPOConfig`):**

```text
v5i6
v5i6_paper_faithful
v5i6_paper_faithful_marginal_entropy
v5i6_marginal_entropy
paper_faithful_marginal_entropy
latent_v5i6_paper_faithful
latent_v5i6_paper_faithful_marginal_entropy
latent_v5i6_marginal_entropy
plan_faithful_latent_v5i6_paper_faithful_marginal_entropy
plan_faithful_latent_v5i6_marginal_entropy
```

Pinned by
[`tests/test_v5i6_paper_faithful_marginal_entropy.py::V5i6AliasSnapshotTests`](../tests/test_v5i6_paper_faithful_marginal_entropy.py)
and the snapshot in
[`tests/preset_snapshots.json`](../tests/preset_snapshots.json).

---

## 3. Classification table (operational latent-strategy presets)

Six classifications, per
[`summer-fidelity-rules.md`](summer-fidelity-rules.md) §3:
`PAPER-FAITHFUL`, `SUMMER-COMPATIBLE EXTENSION`, `ABLATION`,
`DIAGNOSTIC`, `DEPRECATED`, `UNKNOWN`. The v5 ladder, v6 staged lineage,
and the most recent v4 / v3i19 presets are tabulated; older `v3iN`, `phaseN`, and
`hypothesis_*` presets are kept reachable for reproducibility under
`DEPRECATED` (§4).

### 3.1 v5 ladder (operational, current)

| Preset (apply fn) | Function name suffix | Parent | Classification | One-line reason |
|---|---|---|---|---|
| `v5_strict_summer` | `apply_plan_faithful_latent_v5_strict_summer` | `v4i3_summer_proof` | `ABLATION` | Literal docs/algorithm.md loss: entropy + persistence only on `q_phi`, **no** task-reward PG channel. Exists to test whether the literal equation alone trains `q_phi`. |
| `v5i1_reward_credit_router` | `apply_plan_faithful_latent_v5i1_reward_credit_router` | `v5_strict_summer` | `SUMMER-COMPATIBLE EXTENSION` | Adds per-episode router PPO on `q_phi` with dedicated AdamW (`latent_episode_strategy_lr = 5e-3`). Compound delta: also flips `latent_resample_every_n = 0`, `latent_lam_p = 0`, entropy schedule, marginal baseline ON. |
| `v5i2_stronger_z_conditioning` | `apply_plan_faithful_latent_v5i2_stronger_z_conditioning` | `v5i1_reward_credit_router` | `SUMMER-COMPATIBLE EXTENSION` | v5i1 + actor-only FiLM (`enable_actor_z_film = True`, `actor_z_film_init_scale = 0.02`, `actor_z_film_layer = 2`). FiLM is a post-Summer actor-conditioning extension. |
| `v5i3_balanced_warmup` | `apply_plan_faithful_latent_v5i3_balanced_warmup` | `v5i2_stronger_z_conditioning` | `SUMMER-COMPATIBLE EXTENSION` | v5i2 + forced-z anneal (`0.30 → 0.00` across `200_000 → 500_000`). Forcing is unlabeled uniform exploration, not role assignment; routed via `latent_preference_buffer` so `q_phi`'s PPO update stays on-policy. |
| **`v5i4_end_to_end`** | **`apply_plan_faithful_latent_v5i4_end_to_end`** | **`v5_strict_summer`** | **`PAPER-FAITHFUL`** | Preserved conditional-entropy interpretation: v5_strict_summer + `latent_strategy_ppo_coef = 0.10`, with mean conditional entropy `E_s[H(q_phi(z|s))]`. No FiLM, no episode-credit, no forced-z, no aux heads. |
| **`v5i5_paper_faithful_entropy_floor`** | **`apply_plan_faithful_latent_v5i5_paper_faithful_entropy_floor`** | **`v5i4_end_to_end`** | **`PAPER-FAITHFUL`** | Higher conditional-entropy-floor ablation. Resolved diff vs v5i4 is exactly `{latent_lam_h_end: 0.0002 → 0.001, run_tag}`; entropy mode remains `conditional`. |
| **`v5i6_paper_faithful_marginal_entropy`** | **`apply_plan_faithful_latent_v5i6_paper_faithful_marginal_entropy`** | **`v5i4_end_to_end`** | **`PAPER-FAITHFUL`** | **Canonical paper-faithful marginal-entropy interpretation.** Replaces mean conditional entropy with **rollout-level** marginal entropy `KL(N⁻¹Σ_i q_phi(z\|s_i) \|\| U)` over every resample-decision point in the rollout (`latent_entropy_mode = "marginal"`, aggregation = `rollout`, computed once per PPO inner epoch — see method spec §8.1 for the Jensen rationale), uses the v5i5 `λ_H` floor (`0.001`), and changes no actor, critic, sampling, task-PPO, persistence, curriculum, label, or auxiliary channel. |

| `v5i7_summer_faithful_entropy_floor_split_lane` | `apply_plan_faithful_latent_v5i7_entropy_floor_split_lane` | `v5i5_paper_faithful_entropy_floor` | `PAPER-FAITHFUL` | v5i5 entropy-floor method on `map_b_split_lane`. The only resolved diff vs v5i5 is `{map_layout: map_a_open -> map_b_split_lane, run_tag}`. This tests whether lane/chokepoint geometry creates enough return contrast for deployed latent routing without adding labels, curriculum, auxiliary losses, FiLM, marginal entropy, or new `q_phi` gradient channels. |
| `v5i8_split_lane_v2_task_pressure` | `apply_plan_faithful_latent_v5i8_split_lane_v2_task_pressure` | `v5i7_summer_faithful_entropy_floor_split_lane` | `PAPER-FAITHFUL` | v5i7 latent contract on `map_b_split_lane_v2`. The only resolved diff vs v5i7 is `{map_layout: map_b_split_lane -> map_b_split_lane_v2, run_tag}`. This tests whether lower-friction, higher-route-contrast geometry creates enough task-return structure for deployed latent routing without changing latent coefficients, objectives, sampling, labels, or actor conditioning. |
| `v5i9_csia_guided_specialization` | `apply_plan_faithful_latent_v5i9_csia_guided_specialization` | `v5i8_split_lane_v2_task_pressure` | `SUMMER-COMPATIBLE EXTENSION` | v5i8 plus detached CSIA reward from frozen forced-z evaluation evidence. The resolved diff vs v5i8 is `{csia_enabled: False -> True, csia_reward_coef: 0.0 -> 0.02, run_tag}`. This is not the original Summer plan: once gates pass, PPO reward includes `reward_csia = lambda_csia * S(opponent,z)`. |

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

### 3.6 v6 staged specialization and local communication lineage

These rows are not `PAPER-FAITHFUL` Summer rows. They are registered
v6 extensions with frozen gate contracts for confirmatory experiments.
Pre-freeze artifacts under these run tags are exploratory unless the
checkpoint metadata records the matching gate fingerprint and
`confirmatory_gate_lineage_valid=True`.

| Preset (apply fn) | Parent | Classification | One-line reason |
|---|---|---|---|
| `v6i2_staged_team_intent_curriculum` | `v6i1_staged_team_intent_curriculum` | `SUMMER-COMPATIBLE EXTENSION` | Staged team-intent curriculum with dual evidence gates: actor CF pair-JSD EMA plus bounded online matched-seed behavioral realization. Confirmatory gate fingerprint: `224f1aea9ab36319`. |
| `v6i5` | `v6i2_staged_team_intent_curriculum` | `SUMMER-COMPATIBLE EXTENSION` | Corrected q_phi context row using `current_34 || delta_from_previous_boundary_34`, rollout-level marginal entropy, bounded router PPO, and 32-step router opportunities. |
| `v6i6_strategy_expansion` | `v6i5` | `SUMMER-COMPATIBLE EXTENSION` | Conditional repertoire Expansion Stage E1. Requires a validated anchor manifest before training; the manifest selects anchor latents, target latent, and dormant latents. Uses fixed-z episodes for outcome attribution, a frozen reference critic for opportunity weights, no-op adapter initialization, and target-only trainable scope. |
| `v6i3_strategy_local_comm` | `v6i2_staged_team_intent_curriculum` | `SUMMER-COMPATIBLE EXTENSION` | v6i2 plus local learned communication. Phase A requires communication transport without total active-symbol collapse; listener value remains diagnostic until final matched-seed communication evaluation. Confirmatory gate fingerprint: `9ef168d941f046fb`. |
| `v6i4_router_ablation_protocol` | `v6i2_staged_team_intent_curriculum` | `SUMMER-PLAN-FAITHFUL EVALUATION-ONLY` | v6i4 is a Summer-plan-faithful, evaluation-only router-ablation protocol over a frozen, Phase-A-promoted v6i2 checkpoint. It is currently planned/pending. No parameters are trained or updated. |
| `v6i9_arc_credit_running_mean_feedforward_hardpool` (aliases `v6i9_arc_credit_feedforward`, `plan_faithful_latent_v6i9_arc_credit_running_mean_feedforward_hardpool`) | `v6i9_mapaware_router_feedforward_hardpool` | `SUMMER-COMPATIBLE EXTENSION` (arc-credit row) | Feedforward router A/B treatment. Resolved diff vs the feedforward control is exactly four keys: `latent_arc_credit_enabled` False→True, `latent_arc_credit_baseline` context_value→running_mean, `latent_strategy_ppo_coef` 0.1→0.0 (removes the biased critic-based router advantage), and `run_tag`. Router architecture, 35-dim context, strategy interval, LR, entropy coef, opponent/map pool, frozen actor + z-specific params, seed, and budget are held identical. Pinned by `tests/test_v6i9_arc_credit_feedforward.py`. Not a paper-faithful row (arc credit is the v3i19 post-Summer channel). |

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
| **`apply_plan_faithful_latent_v5i6_paper_faithful_marginal_entropy`** | **`plan_faithful_latent_v5i6_paper_faithful_marginal_entropy`, `plan_faithful_latent_v5i6_marginal_entropy`, `latent_v5i6_paper_faithful_marginal_entropy`, `latent_v5i6_marginal_entropy`, `latent_v5i6_paper_faithful`, `v5i6_paper_faithful_marginal_entropy`, `v5i6_paper_faithful`, `v5i6_marginal_entropy`, `v5i6`, `paper_faithful_marginal_entropy`** |
| **`apply_plan_faithful_latent_v5i7_entropy_floor_split_lane`** | **`plan_faithful_latent_v5i7_entropy_floor_split_lane`, `plan_faithful_latent_v5i7_summer_faithful_entropy_floor_split_lane`, `plan_faithful_latent_v5i7_summer_faithful_split_lane`, `plan_faithful_latent_v5i7_split_lane`, `latent_v5i7_entropy_floor_split_lane`, `latent_v5i7_summer_faithful_entropy_floor_split_lane`, `latent_v5i7_summer_faithful_split_lane`, `latent_v5i7_split_lane`, `v5i7_entropy_floor_split_lane`, `v5i7_summer_faithful_entropy_floor_split_lane`, `v5i7_summer_faithful_split_lane`, `v5i7_split_lane`, `v5i7`** |
| **`apply_plan_faithful_latent_v5i8_split_lane_v2_task_pressure`** | **`plan_faithful_latent_v5i8_split_lane_v2_task_pressure`, `plan_faithful_latent_v5i8_summer_faithful_split_lane_v2`, `plan_faithful_latent_v5i8_split_lane_v2`, `latent_v5i8_split_lane_v2_task_pressure`, `latent_v5i8_summer_faithful_split_lane_v2`, `latent_v5i8_split_lane_v2`, `v5i8_split_lane_v2_task_pressure`, `v5i8_summer_faithful_split_lane_v2`, `v5i8_split_lane_v2`, `v5i8`** |
| `apply_plan_faithful_latent_v5i9_csia_guided_specialization` | `plan_faithful_latent_v5i9_csia_guided_specialization`, `plan_faithful_latent_v5i9_csia`, `latent_v5i9_csia_guided_specialization`, `latent_v5i9_csia`, `v5i9_csia_guided_specialization`, `v5i9_csia`, `v5i9` |
| `apply_plan_faithful_latent_v6i2_staged_team_intent_curriculum` | `plan_faithful_latent_v6i2_staged_team_intent_curriculum`, `latent_v6i2_staged_team_intent_curriculum`, `v6i2_staged_team_intent_curriculum`, `v6i2_staged`, `v6i2` |
| `apply_plan_faithful_latent_v6i5_corrected_team_intent_curriculum` | `v6i5` |
| `apply_plan_faithful_latent_v6i6_strategy_expansion` | `plan_faithful_latent_v6i6_strategy_expansion`, `latent_v6i6_strategy_expansion`, `v6i6_strategy_expansion`, `v6i6` |
| `apply_plan_faithful_latent_v6i3_strategy_local_comm` | `plan_faithful_latent_v6i3_strategy_local_comm`, `latent_v6i3_strategy_local_comm`, `v6i3_strategy_local_comm`, `v6i3_local_comm`, `v6i3` |
| `apply_plan_faithful_latent_v6i4_router_ablation_protocol` | `plan_faithful_latent_v6i4_router_ablation_protocol`, `latent_v6i4_router_ablation_protocol`, `v6i4_router_ablation_protocol`, `v6i4_router_ablation`, `v6i4` |
| `apply_plan_faithful_latent_v4i4post_periodic_router_distill` | `plan_faithful_latent_v4i4post_periodic_router_distill`, `latent_v4i4post_periodic_router_distill`, `latent_v4i4post`, `v4i4post`, `v4i4`                                                    |

The full alias surface lives in
[`rl/presets/__init__.py::PRESET_REGISTRY`](../rl/presets/__init__.py).

---

## 6. Resolved-configuration deltas vs v5i4 reference row

Each row lists fields whose resolved value differs from v5i4 (i.e. from
`dataclasses.asdict(apply_preset(PPOConfig(), "v5i4"))`). All other
scalar fields match v5i4 (re-asserted defensively inside each preset).
v5i6 is the current canonical row; this section keeps v5i4 as the
reference to preserve the existing historical delta table and to make
the entropy-interpretation change easy to audit.

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
| `latent_marginal_entropy_nats`  | `H` of **sampled-z** rollout-marginal histogram (one categorical sample per state). Distinct from `router_rollout_soft_marginal_entropy_nats`, which is the soft `H(E_s[q_phi(z\|s)])` over the same population. | `0` (full collapse) to `ln(K)` (uniform).         |
| `effective_num_latents`         | `exp(latent_marginal_entropy_nats)` (sampled).                           | `1` (full collapse) to `K` (uniform).             |
| `latent_occupancy_min`          | `min_k strategy_occupancy_k` over **sampled** `z`.                       | `[0, 1]`.                                         |
| `latent_occupancy_max`          | `max_k strategy_occupancy_k` over **sampled** `z`.                       | `[1/K, 1]`.                                       |
| `latent_occupancy_ratio`        | `latent_occupancy_max / max(latent_occupancy_min, 1e-8)`                | `1.0` = uniform; `>>1` = severe imbalance.        |
| `mean_strategy_duration`        | `total_decisions / max(1, strategy_resample_count)`                      | Mean dwell length (decisions) per latent arc.     |

These are pure functions of the per-z counts already computed for v5i4;
they add **no** new gradient channel and **no** new objective term.

### 6.9 v5i6_paper_faithful_marginal_entropy (PAPER-FAITHFUL — canonical)

The v5i4 → v5i6 resolved-config diff is exactly three keys
(`latent_entropy_mode`, `latent_lam_h_end`, `run_tag`). The v5i5 → v5i6
diff is exactly two keys (`latent_entropy_mode`, `run_tag`). These are
enforced by
[`tests/test_v5i6_paper_faithful_marginal_entropy.py::V5i6PresetInheritanceTests`](../tests/test_v5i6_paper_faithful_marginal_entropy.py).

| Field                  | v5i4 value | This preset | Forbidden flag? | Note |
|------------------------|------------|-------------|-----------------|------|
| `latent_entropy_mode`  | `conditional` | `marginal` | No (R21 ✓) | Replaces mean conditional entropy with rollout-level marginal entropy `KL(q_bar \|\| U)` where `q_bar = N⁻¹ Σ_i q_phi(z\|s_i)` is averaged across **every** resample-decision point in the rollout, computed once per PPO inner epoch via [`rollout_marginal_entropy_loss`](../rl/latent_losses.py). The deprecated per-minibatch helper is kept only for parity tests. |
| `latent_lam_h_end`     | `0.0002`   | `0.001`     | No (R20 ✓) | Uses the v5i5 floor so v5i6 isolates entropy reduction against v5i5. |
| `run_tag`              | `v5i4_paper_faithful_end_to_end_OP5_OP6_OP7_1m_4v4` | `v5i6_paper_faithful_marginal_entropy_OP5_OP6_OP7_1m_4v4` | — | Family-aware audit banner detects this tag automatically and emits `aggregation=rollout`. |

All other v5i4 fidelity fields match: concat-only actor, `K = 4`,
`latent_strategy_ppo_coef = 0.10`, `latent_lam_p = 0.03`, sparse
64-decision resampling, no episode-credit optimizer, no forced-z
curriculum, no auxiliary heads, no labels, no FiLM, no preferences, and
no distillation. `latent_usage_balance_coef` remains `0.0`; the
marginal entropy term is the lambda_H-driven canonical entropy path, not
the legacy episode-router usage-balance coefficient.

**v5i6 telemetry added on top of v5i5's sampled-z columns** (computed in
the same forward pass that produces the rollout-level marginal-entropy
gradient; soft, not sampled):

| Column                                              | Source                                                                                                          | Range                              |
|-----------------------------------------------------|-----------------------------------------------------------------------------------------------------------------|------------------------------------|
| `router_rollout_soft_marginal_entropy_nats`         | `H(N⁻¹ Σ_i q_phi(z\|s_i))` over rollout resample subset                                                          | `[0, ln(K)]`                       |
| `router_rollout_soft_conditional_entropy_nats`      | `N⁻¹ Σ_i H(q_phi(z\|s_i))` over the same subset                                                                  | `[0, ln(K)]`                       |
| `router_rollout_soft_mi_proxy_nats`                 | Difference of the two above                                                                                     | `[0, ln(K)]`                       |
| `router_rollout_soft_p_bar_z<k>` (one per `z`)      | `[N⁻¹ Σ_i q_phi(z\|s_i)]_k`                                                                                      | `[0, 1]` per entry, sum to `1`     |
| `router_rollout_soft_argmax_occupancy_max`          | `max_z mean_i 1[argmax q_phi(·\|s_i) = z]` (soft-argmax population fraction)                                     | `[1/K, 1]`                         |
| `router_rollout_soft_argmax_occupancy_min`          | corresponding `min_z`                                                                                           | `[0, 1/K]`                         |
| `router_rollout_soft_argmax_occupancy_ratio`        | `max / max(min, ε)` — soft-decision imbalance (cf. sampled `latent_occupancy_ratio`)                            | `1` = uniform, ≫ 1 = imbalance     |
| `router_rollout_resample_count`                     | `N` (rollout-level resample-decision count). 0 indicates no resampling occurred (defensive).                    | `≥ 0`                              |

The desired Summer pattern is **high marginal, low conditional** (broad
strategy repertoire, state-specific routing). Low marginal + low
conditional = global collapse; high marginal + high conditional = broad
but indecisive routing.

### 6.10 v5i7_summer_faithful_entropy_floor_split_lane (PAPER-FAITHFUL)

The v5i5 -> v5i7 resolved-config diff is exactly two keys
(`map_layout`, `run_tag`). This is enforced by
[`tests/test_v5i7_entropy_floor_split_lane.py::V5i7PresetInheritanceTests::test_v5i7_diff_vs_v5i5_is_map_layout_and_tag_only`](../tests/test_v5i7_entropy_floor_split_lane.py).

| Field                  | v5i5 value | This preset | Forbidden flag? | Note |
|------------------------|------------|-------------|-----------------|------|
| `map_layout`           | `map_a_open` | `map_b_split_lane` | No | Adds lane/chokepoint route geometry while preserving v5i5's latent contract. |
| `run_tag`              | `v5i5_paper_faithful_entropy_floor_OP5_OP6_OP7_1m_4v4` | `v5i7_summer_faithful_entropy_floor_split_lane_OP5_OP6_OP7_1m_4v4` | - | Artifact namespace advertises the split-lane Summer-faithful row. |

All other v5i5 fidelity fields match: concat-only actor, conditional
entropy maximization with `latent_lam_h_end = 0.001`, `K = 4`,
`latent_strategy_ppo_coef = 0.10`, `latent_lam_p = 0.03`, sparse
64-decision resampling, no episode-credit optimizer, no forced-z
curriculum, no marginal entropy, no FiLM/adapter/one-hot actor path, no
preferences/distillation, no arc-credit, and no auxiliary heads.

Comparisons against v5i5, v5i6, or no-latent controls must use matched
map geometry. A default-open-map row and a split-lane row answer different
environment questions.

### 6.11 v5i8_split_lane_v2_task_pressure (PAPER-FAITHFUL)

The v5i7 -> v5i8 resolved-config diff is exactly two keys
(`map_layout`, `run_tag`). This is enforced by
[`tests/test_v5i8_split_lane_v2_task_pressure.py::V5i8PresetInheritanceTests::test_v5i8_diff_vs_v5i7_is_map_layout_and_tag_only`](../tests/test_v5i8_split_lane_v2_task_pressure.py).

| Field                  | v5i7 value | This preset | Forbidden flag? | Note |
|------------------------|------------|-------------|-----------------|------|
| `map_layout`           | `map_b_split_lane` | `map_b_split_lane_v2` | No | Uses a narrower/shorter central wall, wider route openings, route-aware clearance, and OP5/OP6/OP7 lane-pressure patterns while preserving the v5i7 latent contract. |
| `run_tag`              | `v5i7_summer_faithful_entropy_floor_split_lane_OP5_OP6_OP7_1m_4v4` | `v5i8_summer_faithful_split_lane_v2_task_pressure_OP5_OP6_OP7_1m_4v4` | - | Artifact namespace advertises the split-lane v2 task-pressure row. |

All other v5i7 fidelity fields match: concat-only actor, conditional
entropy maximization with `latent_lam_h_end = 0.001`, `K = 4`,
`latent_strategy_ppo_coef = 0.10`, `latent_lam_p = 0.03`, sparse
64-decision resampling, no episode-credit optimizer, no forced-z
curriculum, no marginal entropy, no FiLM/adapter/one-hot actor path, no
preferences/distillation, no arc-credit, and no auxiliary heads.

The environment adds route-context episode telemetry
(`*_attack_*_crossings`, `*_return_*_crossings`, and
`*_intercept_*_crossings`) so analyses can group route behavior by
`latent_z` without assigning hard-coded meanings to any `z` index.

Comparisons against v5i7 or no-latent controls must use matched map
geometry. A split-lane v1 row and split-lane v2 row answer an environment
question, not a latent-objective question.

### 6.12 v5i9_csia_guided_specialization (SUMMER-COMPATIBLE EXTENSION)

The v5i8 -> v5i9 resolved-config diff is exactly three keys
(`csia_enabled`, `csia_reward_coef`, `run_tag`). This is enforced by
[`tests/test_v5i9_csia_guided_specialization.py::V5i9PresetInheritanceTests::test_v5i9_diff_vs_v5i8_is_csia_and_tag_only`](../tests/test_v5i9_csia_guided_specialization.py).

| Field              | v5i8 value | This preset | Forbidden flag? | Note |
|--------------------|------------|-------------|-----------------|------|
| `csia_enabled`     | `False` | `True` | Yes for paper-faithful | Enables detached reward feedback from forced-z causal evidence. |
| `csia_reward_coef` | `0.0` | `0.02` | Yes for paper-faithful | Adds `reward_csia = 0.02 * S(opponent,z)` after the CSIA gates pass. |
| `run_tag`          | `v5i8_summer_faithful_split_lane_v2_task_pressure_OP5_OP6_OP7_1m_4v4` | `v5i9_csia_guided_specialization_OP5_OP6_OP7_1m_4v4` | - | Artifact namespace advertises the post-Summer reward extension and deliberately avoids `paper_faithful` / `summer_faithful`. |

All other v5i8 latent and environment fields match: concat-only actor,
conditional entropy floor, `K = 4`, `latent_strategy_ppo_coef = 0.10`,
`latent_lam_p = 0.03`, sparse 64-decision resampling, no episode-credit
optimizer, no forced-z curriculum, no marginal entropy, no
FiLM/adapter/one-hot actor path, no preferences/distillation, no
arc-credit, and no auxiliary heads.

CSIA evidence comes from frozen forced-z evaluation outputs:
`*_qualitative_rollout_by_z.csv` supplies `M[o,z] = E[Return |
opponent=o, do(Z=z)]`, and `*_strategy_evidence.csv` supplies natural
router baseline plus forced-z behavioral spread. The centered interaction
is `S[o,z] = M[o,z] - mean_z M[o,*] - mean_o M[*,z] + mean M`.

The bonus is gated. Gate A requires forced-z behavioral spread; Gate B
requires nonzero opponent-latent interaction strength; Gate C requires
every observed forced-z payoff to stay within `csia_quality_floor_delta`
of the natural-router baseline. Raw diversity, occupancy, entropy, MI,
or a globally superior latent cannot activate v5i9 by themselves.

### 6.13 plan_faithful_latent_k1 (ABLATION)

| Field        | v5i4 value | This preset | Forbidden flag? | Note                                                       |
|--------------|------------|-------------|-----------------|------------------------------------------------------------|
| `latent_k`   | `4`        | `1`         | —               | Collapsed-latent ablation (R2 says `4` is canonical; `1` is a deliberate single-axis change to measure whether latent capacity matters at all). |

### 6.14 v6i2_staged_team_intent_curriculum (SUMMER-COMPATIBLE EXTENSION)

v6i2 inherits v6i1's staged team-intent curriculum and freezes the
dual-evidence gate protocol documented in
[`v6i2-gate-protocol-freeze.md`](v6i2-gate-protocol-freeze.md). The
resolved-config diff vs v6i1 is exactly
`{experiment_id, gate_protocol_version, latent_cf_coef_max,
latent_cf_require_competence, latent_cf_weak_pair_boost,
latent_cf_worst_pair_coef, phase_a_max_end_fraction, run_tag}`. Gate
threshold fields match the frozen `PPOConfig` defaults and are part of
the fingerprint.

| Field | v6i1 value | This preset | Note |
|-------|------------|-------------|------|
| `experiment_id` | `v6i1` | `v6i2` | Artifact and protocol identity. |
| `gate_protocol_version` | `v6i1_single_macro_intervention` | `v6i2_dual_evidence` | Actor-intervention + behavioral-realization gate families. |
| `latent_cf_coef_max` | `0.01` | `1.0` | Strong-CF confirmatory ceiling. |
| `latent_cf_require_competence` | `False` | `True` | Actor pair separation applies only after competence is established. |
| `latent_cf_weak_pair_boost` | `0.0` | `1.0` | Persistent weak pairs get extra hinge weight from actor-CF pair EMA. |
| `latent_cf_worst_pair_coef` | `0.0` | `0.5` | Adds a worst-pair hinge term so one collapsed pair cannot hide under the mean. |
| `phase_a_max_end_fraction` | `0.55` | `0.70` | Allows more Phase-A evidence before forced transition. |
| `run_tag` | `v6i1_staged_team_intent_curriculum_OP5_OP6_OP7_1m_4v4` | `v6i2_staged_team_intent_curriculum_OP5_OP6_OP7_1m_4v4` | Artifact namespace. |
| `gate_config_fingerprint` | n/a | `224f1aea9ab36319` | Confirmatory v6i2 lineage hash. |

Core frozen gates: 5 of 6 actor pairs above `actor_jsd_margin = 0.001`,
all 6 above `0.5 * margin`, 3 consecutive valid actor updates,
matched-seed normalized aggregate effect `>= 0.75`, raw
`task_behavior_distance >= 0.01`, raw `performance_spread >= 0.01`,
adverse threshold `>= -0.01`, and at least 2 opponents passing. The
aggregate reports raw `route_distance`, `task_behavior_distance`,
`performance_spread`, and normalized `aggregate_effect` independently so
route distance cannot compensate for zero task behavior. The online Phase
A gate is bounded to 5 seeds per opponent, 64-step horizons, and 900
seconds; the full 20-seed matched-seed and selector analyses are
confirmatory/offline.

### 6.15 v6i5 (SUMMER-COMPATIBLE EXTENSION)

v6i5 inherits v6i2 directly. The scientific delta is a corrected q_phi
router input and router-loss ownership contract, not an actor, critic, or
gate change. It keeps the v6i2 dual-evidence gate protocol and registers
only the public alias `v6i5`.

Resolved-config diff vs v6i2 is exactly:
`{experiment_id, latent_entropy_mode, latent_resample_every_n,
v6i1_router_lr, latent_strategy_ppo_coef, strategy_target_kl,
router_context_mode, router_context_dimension, router_persistence_mode,
router_marginal_entropy_coefficient, run_tag}`.

`router_context_mode = "current_plus_delta"` makes q_phi consume a
68-wide row: the current 34-feature structured state from
`build_global_state_batch` followed by `current - previous_opportunity`.
The 170-wide temporal context remains the critic context and is not the
v6i5 q_phi input. `router_persistence_mode =
"expected_switch_detached_previous"` uses detached previous-router
probabilities for persistence, so persistence gradients only update the
current q_phi branch. Ordinary router minibatches use clipped router PPO
plus expected-switch persistence. The rollout-wide router step owns the
marginal entropy term through
`router_marginal_entropy_coefficient = 0.001`; conditional entropy is
disabled by `router_conditional_entropy_coefficient = 0.0`.

### 6.15a v6i6_strategy_expansion (SUMMER-COMPATIBLE EXTENSION)

v6i6 inherits v6i5 directly. The scientific delta is conditional
repertoire expansion after forced-z and branch evaluations identify a
useful anchor set, one expansion target, and any dormant latents. The
preset intentionally does not hardcode `z0`, `z1`, `z2`, or `z3`.

Launch is fail-closed: with
`v6i6_require_validated_anchors = True`, training validation rejects the
run unless `v6i6_anchor_validation_manifest` points to a JSON manifest
with `verdict = "VALIDATED"`, disjoint `anchors`,
`expansion_target`, and `dormant` entries inside `[0, latent_k)`.
Validation hydrates `v6i6_anchor_latents`,
`v6i6_target_latent`, and `v6i6_dormant_latents` from that manifest at
runtime.

Resolved-config diff vs v6i5 is exactly:
`{experiment_id, gate_protocol_version, use_v6i6_expansion,
v6i6_expansion_protocol_version, v6i6_expansion_stage,
v6i6_trainable_scope, latent_actor_z_adapter_enabled,
latent_actor_z_adapter_scale, latent_actor_z_adapter_init_std,
latent_resample_every_n, run_tag}`.

| Field | v6i5 value | This preset | Note |
|-------|------------|-------------|------|
| `experiment_id` | `v6i5` | `v6i6` | Artifact and protocol identity. |
| `gate_protocol_version` | `v6i2_dual_evidence` | `v6i6_repertoire_expansion_e1_v1` | Separates Expansion Stage E1 from existing Phase B router training. |
| `use_v6i6_expansion` | `False` | `True` | Enables the extension contract. |
| `v6i6_expansion_stage` | `""` | `E1` | Avoids the existing Phase A/B/C names. |
| `v6i6_trainable_scope` | `""` | `target_embedding_gate_adapter_only` | Shared actor parameters are outside the intended trainable set. |
| `latent_actor_z_adapter_enabled` | `False` | `True` | Adds the target-side expansion adapter path. |
| `latent_actor_z_adapter_scale` | `0.0` | `0.05` | Small nonzero gate so zero-initialized adapter params receive gradient. |
| `latent_actor_z_adapter_init_std` | `0.02` | `0.0` | Adapter starts as a no-op. |
| `latent_resample_every_n` | `32` | `0` | Fixed latent per episode for clean outcome attribution. |
| `run_tag` | `v6i5_corrected_team_intent_curriculum_OP5_OP6_OP7_1m_4v4` | `v6i6_strategy_expansion_OP5_OP6_OP7_1m_4v4` | Artifact namespace. |

The E1 contract also requires `v6i6_fixed_z_episode_attribution = True`,
`v6i6_use_reference_critic_for_opportunity = True`,
`v6i6_restore_masked_latent_rows_after_step = True`,
`v6i6_assert_anchor_bitwise_invariant = True`, and
`v6i6_count_draw_as = 0.5`. These fields are defaults so the v6i6-vs-v6i5
snapshot diff stays focused on the activation surface.

### 6.16 v6i3_strategy_local_comm (SUMMER-COMPATIBLE EXTENSION)

v6i3 inherits v6i2 directly and adds local learned communication. The
final preset contract is frozen in
[`v6i3-local-communication-spec.md`](v6i3-local-communication-spec.md).
It is confirmatory only for fresh runs launched after the frozen
fingerprint below; earlier artifacts under the same run tag are
exploratory.

| Field | v6i2 value | This preset | Note |
|-------|------------|-------------|------|
| `experiment_id` | `v6i2` | `v6i3` | Artifact and protocol identity. |
| `gate_protocol_version` | `v6i2_dual_evidence` | `v6i3_strategy_local_comm_v1` | Adds communication transport evidence to the v6i2 gate set. |
| `communication_enabled` | `False` | `True` | Enables local message head, transport, and message observation channels. |
| `comm_num_symbols` | `4` | `5` | `SILENCE` plus four active two-bit symbols. |
| `comm_silence_symbol` | `-1` | `0` | Symbol 0 sends no rendered message. |
| `comm_interval_steps` | `32` | `32` | Boundary-only message-head PPO credit. |
| `comm_delivery_delay_steps` | `1` | `1` | One decision-step delay. |
| `comm_radius_cells` | `6.0` | `6.0` | Local recipient radius. |
| `comm_dropout_probability` | `0.10` | `0.10` | Training-only sender-receiver dropout. |
| `comm_entropy_coef` | `0.001` | `0.001` | Message entropy regularizer. |
| `comm_min_valid_boundaries` | `0` | `1024` | Usage gate is nontrivial. |
| `comm_min_deliveries` | `0` | `4096` | Requires delivered messages. |
| `comm_min_symbols_used` | `3` | `2` | Prevents total active-symbol collapse without requiring full semantics. |
| `comm_entropy_floor` | `0.0` | `0.0` | Diagnostic during Phase A. |
| `comm_symbol_dominance_ceiling` | `1.0` | `1.0` | Diagnostic during Phase A. |
| `comm_listener_jsd_margin` | `0.0` | `0.001` | Diagnostic listener-response intervention threshold. |
| `comm_listener_min_passing_pairs` | `0` | `3` | Diagnostic: at least 3 of 10 silence/active symbol pairs clear the margin. |
| `comm_listener_min_states` | `0` | `64` | Minimum listener intervention states. |
| `comm_listener_consecutive_updates` | `0` | `1` | Diagnostic listener-intervention batch count. |
| `run_tag` | `v6i2_staged_team_intent_curriculum_OP5_OP6_OP7_1m_4v4` | `v6i3_strategy_local_comm_OP5_OP6_OP7_1m_4v4` | Artifact namespace. |
| `gate_config_fingerprint` | `224f1aea9ab36319` | `9ef168d941f046fb` | Confirmatory v6i3 lineage hash. |

Post-training communication dependence requires a matched-seed corruption
test: natural-minus-corrupted mean episode return must be `>= 0.02` for
at least two corruption modes, with the paired-bootstrap 95 percent CI
excluding zero. A WR-only report may use an absolute WR drop `>= 0.03`
with the same CI requirement.

### 6.17 v6i4_router_ablation_protocol (SUMMER-PLAN-FAITHFUL EVALUATION-ONLY)

v6i4 is a Summer-plan-faithful, evaluation-only router-ablation protocol
over a frozen, Phase-A-promoted v6i2 checkpoint. It is currently
planned/pending. No parameters are trained or updated.

It inherits v6i2 configuration only to lock the checkpoint family and
evaluation context, then marks itself `evaluation_only_preset = True`.

| Field | v6i2 value | This preset | Note |
|-------|------------|-------------|------|
| `experiment_id` | `v6i2` | `v6i4` | Artifact and protocol identity. |
| `evaluation_only_preset` | `False` | `True` | Must not launch PPO or enter Phase A/B/C. |
| `evaluation_only_runner` | `""` | `rl/eval_router_ablation.py` | Single entry point for the frozen router-ablation suite. |
| `evaluation_only_requires_checkpoint` | `False` | `True` | Requires an existing promoted v6i2-style checkpoint. |
| `evaluation_only_checkpoint_family` | `""` | `promoted_v6i2` | The checkpoint is the experimental object. |
| `router_ablation_protocol_version` | `""` | `v6i4_router_ablation_v1` | Must match the manifest protocol version. |
| `run_tag` | `v6i2_staged_team_intent_curriculum_OP5_OP6_OP7_1m_4v4` | `v6i4_router_ablation_protocol_over_v6i2_OP5_OP6_OP7_1m_4v4` | Artifact namespace. |

The evaluator rejects a checkpoint unless promotion evidence verifies
`experiment_id = v6i2`, Phase A promotion `PASS`, a recorded gate
fingerprint, a recorded promotion step, a recorded checkpoint hash, and
valid confirmatory gate lineage. Architecture compatibility alone is
insufficient. The evaluator loads the accepted checkpoint once, hashes
parameters before and after, and fails if weights change. Locked online
rows are
`learned_qphi_switching`, `uniform_episode_fixed`,
`uniform_random_at_router_opportunities`, `preselected_global_fixed_z`,
`fixed_z0`, `fixed_z1`, `fixed_z2`, `fixed_z3`,
`qphi_initial_only_no_switch`, and `shuffled_qphi_outputs`. The global
preselected fixed baseline is selected on calibration seeds and
evaluated on disjoint test seeds.
`posthoc_global_fixed_oracle`, `posthoc_opponent_oracle`, and
`posthoc_episode_oracle` are derived only after the fixed-z sweep and
are non-deployable upper bounds.

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
| `v5i5_paper_faithful_entropy_floor` (any alias) | **Yes.** `cfg.run_tag` contains `"v5i5_paper_faithful"`. |
| `v5i6_paper_faithful_marginal_entropy` (any alias) | **Yes.** `cfg.run_tag` contains `"v5i6_paper_faithful"`. |
| `v5i7_summer_faithful_entropy_floor_split_lane` (any alias) | **Yes.** `cfg.run_tag` contains `"v5i7_summer_faithful"`. |
| `v5i8_split_lane_v2_task_pressure` (any alias) | **Yes.** `cfg.run_tag` contains `"v5i8_summer_faithful"`. |
| `v5i9_csia_guided_specialization` (any alias) | **No.** It is a reward extension, and `cfg.run_tag` deliberately omits `paper_faithful` / `summer_faithful`. |
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
