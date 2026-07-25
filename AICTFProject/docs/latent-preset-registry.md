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

| `v6i10_episode_router_explore_hardpool` (aliases `v6i10`, `v6i10_episode_router_explore`, `latent_v6i10_episode_router_explore_hardpool`, `plan_faithful_latent_v6i10_episode_router_explore_hardpool`) | `v6i9_mapaware_router_feedforward_hardpool` | `SUMMER-COMPATIBLE EXTENSION` | Feedforward episode-router experiment over the frozen v6i9 repertoire. It disables mid-episode router decisions, replaces the critic-based router PPO term with one episode-long running-mean arc-credit record, lowers router LR, adds training-only 20 percent uniform behavior exploration, and uses marginal coverage. It adds no labels, opponent IDs, oracle-z targets, auxiliary heads, forced-z curriculum, or actor training. Pinned by `tests/test_v6i10_episode_router_explore.py`. |
| `v6i13_opening_window_advantage_router` (aliases `v6i13`, `v6i13_opening_window`, `v6i13_advantage_router`, `latent_v6i13_opening_window_advantage_router`, `plan_faithful_latent_v6i13_opening_window_advantage_router`) | `v6i12_advantage_router_hardpool` | `SUMMER-COMPATIBLE EXTENSION` | Delayed-commit external advantage-router diagnostic. It runs steps 0..31 under a uniformly sampled warmup latent, commits one router-selected latent at decision step 32, opens the arc at commit, credits only post-commit return, and attaches `[state_0, state_commit, delta]` to replay records. No labels, opponent-ID supervision head, oracle-z targets, forced-z training, auxiliary task, or actor training are added. Pinned by `tests/test_v6i13_opening_window_advantage_router.py`. |
| `v6i14_contract_specialists` (aliases `v6i14`, `v6i14_contract_specialist_repertoire`, `latent_v6i14_contract_specialists`, `plan_faithful_latent_v6i14_contract_specialists`) | `v6i9_mapaware_repertoire_hardpool` | `DIAGNOSTIC` (non-Summer scaffold) | Contract-specialist repertoire birth. Router is off and z is assigned by balanced episodes; normal env reward is augmented with a small explicit z-indexed contract reward for opening pressure, defense recovery, carrier support, and conversion progress. This deliberately uses handcrafted z-role rewards, so it is not paper-faithful and not a Summer-compatible extension. Pinned by `tests/test_v6i14_contract_specialists.py`. |
| `v6i15_contract_pressure_3x` / `6x` / `10x` (aliases include `v6i15` for 3x) | `v6i14_contract_specialists` | `DIAGNOSTIC` (non-Summer scaffold) | Contract-pressure coefficient sweep over the v6i14 scaffold. Router remains off, balanced-episode z assignment and frozen shared actor are preserved, and only `latent_contract_specialist_coef` is raised to 0.75, 1.50, or 2.50 to test whether behavior fingerprints respond when role contracts become loud. Pinned by `tests/test_v6i15_contract_pressure.py`. |
| `v6i16_capacity_sharp_contracts` (aliases include `v6i16`; arms: `v6i16_sharp_contracts`, `v6i16_capacity`) | `v6i15_contract_pressure_3x` | `DIAGNOSTIC` (non-Summer scaffold) | Capacity + contract-feature ablation. Arm A sharpens the handcrafted contract features, Arm B increases z-pathway leverage, and Arm C combines both. Router remains off, balanced-episode z assignment is preserved, and all arms stay at 3x contract pressure. Pinned by `tests/test_v6i16_capacity_feature_ablation.py`. |
| `v6i17_surface_pressure_diagnostic` (aliases include `v6i17`, `v6i17_harder_asymmetric_opponents`) | `v6i16_capacity_sharp_contracts` | `DIAGNOSTIC` (non-Summer scaffold) | Surface-pressure diagnostic. It keeps the v6i16 combined contract/capacity scaffold but expands the training opponent surface from OP8/OP9/OP10 to OP8/OP9/OP10/OP11/OP12. Router remains off, balanced-episode z assignment is preserved, and promotion requires forced-z behavior, margin, tempo, or role-fingerprint separation rather than oracle gap alone. Pinned by `tests/test_v6i17_surface_pressure_diagnostic.py`. |
| `v6i18_margin_tempo_surface_diagnostic` (aliases include `v6i18`, `v6i18_margin_tempo_surface`) | `v6i17_surface_pressure_diagnostic` | `DIAGNOSTIC` (non-Summer scaffold) | Margin/tempo consequence-surface diagnostic. It keeps the v6i17 contract/capacity/opponent scaffold fixed, leaves router training off, and changes only the reward/horizon surface: shorter episodes plus score-margin, tempo, near-cap, red-touch, and red-carrier-progress pressure. Promotion requires forced-z margin, tempo, role, or behavior-fingerprint separation; win rate alone is secondary. Pinned by `tests/test_v6i18_margin_tempo_surface.py`. |
| `v6i20_asymmetry_handicap_surface_diagnostic` (aliases include `v6i20`, `v6i20_asymmetry_handicap_surface`, `v6i20_handicap_surface`) | `v6i19_map_pool_surface_diagnostic` | `DIAGNOSTIC` (non-Summer scaffold) | Asymmetric consequence-pressure diagnostic. It keeps the v6i19 map-pool scaffold fixed and strengthens only the existing enemy-progress penalties and blue tempo/conversion bonuses. Router remains off, balanced-episode z assignment is preserved, and promotion requires forced-z tradeoff separation by opponent × map, not win-rate saturation or oracle gap alone. Pinned by `tests/test_v6i20_asymmetry_handicap_surface.py`. |
| `v6i21_adaptive_op8_op12_hardpool_calibration` (aliases include `v6i21`, `v6i21_adaptive_op8_op12_hardpool`, `v6i21_adaptive_hardpool_calibration`) | `v6i20_asymmetry_handicap_surface_diagnostic` | `DIAGNOSTIC` (non-Summer scaffold) | Calibration fork after in-place OP8-OP12 upgrade to adaptive hardpool v2 (`gpu_env/_core/_bt_adaptive.py`). Same opponent IDs; engine behavior changed at v6i21 and was pressure-tuned in place at v6i21B. Pre-v6i21 and pre-v6i21B OP8-OP12 WR/fingerprint results are not directly comparable. First gate: blue WR calibration band 35-65% via `experiments/run_v6i21_adaptive_hardpool_calibration.py`. Router blocked. Pinned by `tests/test_v6i21_adaptive_hardpool_calibration.py`. |
| `v6i21d_adaptive_hardpool_brutal_denial_calibration` (aliases include `v6i21d`, `v6i21d_adaptive_hardpool_brutal_denial`) | `v6i21c_adaptive_hardpool_denial_calibration` | `DIAGNOSTIC` (non-Summer scaffold) | Upper-bound denial calibration. Same OP8-OP12 IDs, no router, no PPO training, no checkpoint change. Engine tuning pushes blue carrier penalty, red respawn, red overdrive, near-cap collapse, and cap-lane denial hard enough to test whether the arena can break v6i9 blue saturation at all. Pinned by `tests/test_v6i21d_adaptive_hardpool_brutal_denial_calibration.py`. |
| `v6i21e_targeted_denial_balance_calibration` (aliases include `v6i21e`, `v6i21e_targeted_denial_balance`) | `v6i21d_adaptive_hardpool_brutal_denial_calibration` | `DIAGNOSTIC` (non-Summer scaffold) | Targeted balance after v6i21D smoke: harden OP8/OP10/OP11 only (carrier hunter, escort-break/cutoff, anti-repeat collapse). OP9/OP12 unchanged. No router, no PPO training. Pinned by `tests/test_v6i21e_targeted_denial_balance_calibration.py`. |
| `v6i21f_op8_carrier_denial_calibration` (aliases include `v6i21f`, `v6i21f_op8_carrier_denial`) | `v6i21e_targeted_denial_balance_calibration` | `DIAGNOSTIC` (non-Summer scaffold) | OP8-only carrier denial after v6i21E smoke left OP8 saturated with rising red scores and blue still at 3.0. Pure cap-lane / carrier-hunter monster; OP9–OP12 unchanged. Pinned by `tests/test_v6i21f_op8_carrier_denial_calibration.py`. |
| `v6i21g_easy_cell_conversion_denial_calibration` (aliases include `v6i21g`, `v6i21g_easy_cell_conversion_denial`) | `v6i21f_op8_carrier_denial_calibration` | `DIAGNOSTIC` (non-Summer scaffold) | Targeted correction after v6i21F failed OP8 and left OP10/OP11 saturated. Restores OP8/OP11 cap-lane body-blocking, makes OP10 cut off conversion instead of chasing, and raises OP8/OP10/OP11 2v2 pressure. OP9/OP12 unchanged. Pinned by `tests/test_v6i21g_easy_cell_conversion_denial_calibration.py`. |
| `v6i21h_saturation_surrogate_calibration` (aliases include `v6i21h`, `v6i21h_saturation_surrogate`) | `v6i21g_easy_cell_conversion_denial_calibration` | `DIAGNOSTIC` (non-Summer scaffold) | Saturation fix after bespoke OP8/OP10/OP11 geometry failed. OP8 reuses OP9-like fortress pressure; OP10/OP11 reuse OP12-like counter pressure; failed custom adaptive route overrides are disabled. Pinned by `tests/test_v6i21h_saturation_surrogate_calibration.py`. |
| `v6i21i_op8_extreme_physical_calibration` (aliases include `v6i21i`, `v6i21i_op8_extreme_physical`) | `v6i21h_saturation_surrogate_calibration` | `DIAGNOSTIC` (non-Summer scaffold) | OP8-only physical upper-bound test. Crushes blue carrier speed and gives OP8 red explicit overdrive/interceptor boost to test whether OP8 can break saturation at all. Pinned by `tests/test_v6i21i_op8_extreme_physical_calibration.py`. |
| `v6i21j_hardpool_balance_calibration` (aliases include `v6i21j`, `v6i21j_hardpool_balance`) | `v6i21i_op8_extreme_physical_calibration` | `DIAGNOSTIC` (non-Summer scaffold) | Balance pass after OP8I broke OP8 saturation. Keeps OP8 hard and adds targeted OP10/OP11 carrier slowdown plus red overdrive; OP9/OP12 unchanged. Pinned by `tests/test_v6i21j_hardpool_balance_calibration.py`. |
| `v6i22_adaptive_hardpool_repertoire_birth` (aliases include `v6i22`, `v6i22_repertoire_birth`) | `v6i21j_hardpool_balance_calibration` | `SUMMER-COMPATIBLE EXTENSION` | Label-free repertoire-birth fork over the calibrated adaptive hardpool. Router remains off, one z is held per episode through balanced episode assignment, and the inherited v6 z-capacity scaffold trains without handcrafted per-z contract rewards, opponent-ID supervision, oracle-z targets, or auxiliary labels. Not paper-faithful because it uses the v6 staged/frozen/adapted hardpool machinery. Pinned by `tests/test_v6i22_adaptive_hardpool_repertoire_birth.py`. |
| `v6i22b_context_behavior_diversity` (aliases include `v6i22b`, `v6i22b_behavior_diversity_coef003`; sweep arms `coef001`, `coef005`) | `v6i22_adaptive_hardpool_repertoire_birth` | `SUMMER-COMPATIBLE EXTENSION` | Label-free anti-collapse repertoire-birth fork. It keeps V6I22's router-off, balanced-episode, contract-disabled scaffold and adds a small success-gated behavior-contrast reward keyed by opponent x map. The signal uses trajectory fingerprints only; it adds no handcrafted z roles, no opponent-ID actor shortcut, no oracle best-z targets, and no router training. Pinned by `tests/test_v6i22b_context_behavior_diversity.py`. |
| `v6i22c_contextual_outcome_diversity` (aliases include `v6i22c`, `v6i22c_outcome_diversity_coef003`) | `v6i22_adaptive_hardpool_repertoire_birth` | `SUMMER-COMPATIBLE EXTENSION` | Label-free outcome-diversity repertoire-birth fork. It keeps V6I22's router-off, balanced-episode, contract-disabled scaffold and adds a stronger success-gated terminal outcome-diversity reward keyed by opponent x map. The signal uses generic score-margin outcomes only; it adds no handcrafted z roles, no behavior-metric reward targets, no opponent-ID actor shortcut, no oracle best-z targets, and no router training. Pinned by `tests/test_v6i22c_contextual_outcome_diversity.py`. |
| `v6i22d_strong_behavior_diversity` (aliases include `v6i22d`, `v6i22d_behavior_diversity_coef010`; sweep arm `coef005`) | `v6i22_adaptive_hardpool_repertoire_birth` | `SUMMER-COMPATIBLE EXTENSION` | Stronger label-free behavior-contrast repertoire-birth fork after V6I22B/C failed the birth gate. It keeps V6I22's router-off, balanced-episode, contract-disabled scaffold and applies higher behavior-contrast coefficients (`0.10` primary, `0.05` sweep control). Same trajectory-fingerprint signal as V6I22B; no outcome-diversity channel, no handcrafted z roles, no oracle targets, and no router training. Pinned by `tests/test_v6i22d_strong_behavior_diversity.py`. |
| `v6i22e_fixed_alpha_adapters` (aliases include `v6i22e`) | `v6i22_adaptive_hardpool_repertoire_birth` | `SUMMER-COMPATIBLE EXTENSION` | Fixed-alpha (`α=0.1`) gate-free residual adapters with Kaiming init to escape the zero-init / stuck-gate magnitude trap. Same hardpool birth scaffold; no soft diversity rewards. Pinned by `tests/test_v6i22e_fixed_alpha.py`. |
| `v6i23_population_birth` (aliases include `v6i23`) | `v6i22e_fixed_alpha_adapters` | `SUMMER-COMPATIBLE EXTENSION` | Population-style specialist birth: active-z-only residual forward plus independent per-z action heads that are Stage-2 trainable (shared `action_head` stays frozen). Router off; no opponent-ID; no soft diversity rewards. Success gate is CF action-JSD, not paper-faithful. Pinned by `tests/test_v6i23_population_birth.py`. |
| `v6i24_full_policy_population` (aliases include `v6i24`) | `v6i21j_hardpool_balance_calibration` | `DIAGNOSTIC` | Full-policy population diagnostic (**Path C**; soft-contract 5u teachers exist; LRO Stage-0 landscape scan supersedes as primary next spend). K=4 independent teachers under fixed OP8–OP12×map pressures. **Primary gate:** different best policies across cells **and** cross-fitted context oracle > best fixed (CI excludes 0). JSD/classifier supporting only. Pinned by `tests/test_v6i24_full_policy_population.py`. |
| `v6i25_counterfactual_router` (experiment; no training preset) | V6I23 donor checkpoint | `DIAGNOSTIC` | Counterfactual geometry→`q_phi` diagnostic. Cross-fitted context oracle (not per-episode hindsight); Stage A signal gate; soft `softmax(Q̂/τ)` CE loss; geometry asserts; no opponent-ID in router input. Runner: `experiments/run_v6i25_counterfactual_router_diagnostic.py`. Pinned by `tests/test_v6i25_counterfactual_router.py`. |
| `v6i26_latent_response_oracle` (aliases include `v6i26`, `v6i26_lro`, `v6i26_phase_pod_population`) | `v6i23_population_birth` | `DIAGNOSTIC` | **LRO-Summer finite proof ladder** (Claim B). Primary claim: response-oracle birth of complementary latent strategies + sparse router that beats fixed-z and matched non-latent PPO. Not spontaneous emergence. Stage-1 at 4 eps/cell is screening only and may emit `PROMISING_DIRECTION`, never `ACCEPT`. Each response round selects its branch and target/anchor mixture from the current forced-z payoff matrix, excluding saturated cells instead of repeatedly assigning `z3`. Strategy acceptance requires `ΔG>0`, CI95 lower bound above zero, nonredundant payoff row, competence floor, forced-z behavior nonredundancy, ≥32 eps/cell, and ≥3 training seeds. One retry per failure mode; no coefficient carousel. Contract: `artifacts/v6i26_lro_round1_seed1/proof_ladder_contract.json`. Pinned by `tests/test_v6i26_latent_response_oracle.py`. |
| `v6i26_lro_actor_step_ablation` (aliases `v6i26_actor_step`, `v6i26_actor_step_ablation`) | `v6i26_latent_response_oracle` | `DIAGNOSTIC` | Optimizer-control ablation: separate z-actor/critic clip + `latent_lro_z_actor_lr_mult` (CLI `--z-actor-lr-mult`). **2× FAIL** (KL under floor); **3× FAIL** (KL over ceiling) → **no further LR rungs**. Next control: `run_v6i26_actor_step_kl_ladder.py` (hold 3× LR; 1u checkpoints; first fixed-batch KL in `[1e-3,1e-2]`). Phase-2 `z3` remains LOCKED FAIL (no G, behavior redundant). |

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

### 6.18 v6i10_episode_router_explore_hardpool (SUMMER-COMPATIBLE EXTENSION)

v6i10 inherits `v6i9_mapaware_router_feedforward_hardpool` directly. The
scientific delta is to test the simplest dispatch problem first: observe
the legal initial context, choose one latent, hold it for the episode,
and credit the choice with episode return against a running mean. Actor
and z-specific repertoire parameters remain frozen by the inherited
`v6i9_training_stage = "router"` and `router_freeze_actor = True`.

Resolved-config diff vs `v6i9_mapaware_router_feedforward_hardpool` is
exactly:
`{experiment_id, h_mode, latent_arc_credit_baseline,
latent_arc_credit_enabled, latent_arc_credit_min_len,
latent_entropy_anneal_end, latent_entropy_anneal_start,
latent_entropy_mode, latent_entropy_objective, latent_lam_h,
latent_lam_h_end, latent_lam_p, latent_resample_every_n,
latent_strategy_ppo_coef, learning_rate, router_ent_coef,
router_uniform_exploration_prob, run_tag, strategy_interval}`.

| Field | Feedforward parent value | This preset | Note |
|-------|--------------------------|-------------|------|
| `experiment_id` | `v6i9` | `v6i10` | Artifact and protocol identity. |
| `latent_resample_every_n` / `strategy_interval` | `32` / `32` | `0` / `0` | Episode-start router decision only. |
| `latent_strategy_ppo_coef` | `0.10` | `0.0` | Removes the critic-based router PPO channel. |
| `latent_arc_credit_enabled` | `False` | `True` | Uses episode-long arc credit. |
| `latent_arc_credit_baseline` | `context_value` | `running_mean` | Detached running-mean return baseline. |
| `latent_arc_credit_min_len` | `32` | `1` | Terminal episode arcs are accepted. |
| `learning_rate` | parent value | `1e-4` | Slower router update. |
| `router_uniform_exploration_prob` | `0.0` | `0.20` | Training-only behavior mixture; deterministic eval uses q_phi directly. |
| `router_ent_coef` | `0.005` | `0.002` | Lower conditional router entropy. |
| `latent_entropy_mode` / `h_mode` | parent values | `marginal` / `marginal` | Runtime coverage path is marginal. |
| `latent_entropy_objective` | parent value | `maximize` | Maximizes marginal coverage. |
| `latent_lam_h` / `latent_lam_h_end` | parent values | `0.015` / `0.015` | Strong fixed marginal coverage coefficient. |
| `latent_lam_p` | parent value | `0.0` | Persistence is irrelevant with one decision per episode. |
| `run_tag` | parent tag | `v6i10_episode_router_explore_hardpool_OP8_OP9_OP10` | Artifact namespace. |

The aliases are `v6i10`, `v6i10_episode_router_explore_hardpool`,
`v6i10_episode_router_explore`,
`latent_v6i10_episode_router_explore_hardpool`, and
`plan_faithful_latent_v6i10_episode_router_explore_hardpool`.

### 6.19 v6i13_opening_window_advantage_router (SUMMER-COMPATIBLE EXTENSION)

v6i13 inherits `v6i12_advantage_router_hardpool` directly. The scientific
delta is to change when the router commits: run an unsupervised uniform
warmup latent for the opening window, then choose one latent at step 32
from an opening-summary context and hold it to terminal. The external
V/A advantage model is unchanged in kind, but its replay context is
`[state_0, state_commit, state_commit - state_0]` plus the existing
hard-pool opponent input feature.

Resolved-config diff vs `v6i12_advantage_router_hardpool` is exactly:
`{experiment_id, latent_episode_strategy_warmup_decision_steps,
router_arc_post_commit_only, router_opening_context_mode,
router_warmup_uniform_z, run_tag}`.

| Field | v6i12 value | This preset | Note |
|-------|-------------|-------------|------|
| `experiment_id` | `v6i12` | `v6i13` | Artifact and protocol identity. |
| `latent_episode_strategy_warmup_decision_steps` | `0` | `32` | Router commits after the opening window. |
| `router_warmup_uniform_z` | `False` | `True` | Warmup latent is sampled uniformly, not from q_phi. |
| `router_arc_post_commit_only` | `False` | `True` | No warmup arc is recorded; arc return starts at commit. |
| `router_opening_context_mode` | `""` | `initial_commit_delta` | Arc records carry `[state_0, state_commit, delta]`. |
| `run_tag` | `v6i12_advantage_router_hardpool_OP8_OP9_OP10` | `v6i13_opening_window_advantage_router_OP8_OP9_OP10` | Artifact namespace. |

The aliases are `v6i13`, `v6i13_opening_window_advantage_router`,
`v6i13_opening_window`, `v6i13_advantage_router`,
`latent_v6i13_opening_window_advantage_router`, and
`plan_faithful_latent_v6i13_opening_window_advantage_router`.

### 6.20 v6i14_contract_specialists (DIAGNOSTIC -- non-Summer scaffold)

v6i14 inherits `v6i9_mapaware_repertoire_hardpool` directly. The
scientific delta is to stop trying to route weakly differentiated latent
behaviors and instead create behavioral specialists first. During this
phase the router remains off, z is assigned by balanced episodes, and the
v6i9 repertoire-stage trainable scope remains in force: shared actor trunk
frozen, z-specific modules trainable.

This row deliberately breaks the no-handcrafted-role boundary. It is not a
paper-faithful row and not a Summer-compatible extension. The contract
reward is scaffolding for specialist birth, to be removed or reduced before
router-selection claims.

Resolved-config diff vs `v6i9_mapaware_repertoire_hardpool` is exactly:
`{experiment_id, latent_contract_specialist_coef,
latent_contract_specialist_enabled, run_tag}`.

| Field | v6i9 repertoire value | This preset | Note |
|-------|-----------------------|-------------|------|
| `experiment_id` | `v6i9` | `v6i14` | Artifact and protocol identity. |
| `latent_contract_specialist_enabled` | `False` | `True` | Enables trainer-side z-indexed contract rewards. |
| `latent_contract_specialist_coef` | `0.0` | `0.25` | Small but nonzero scaffold weight on top of env reward. |
| `run_tag` | `v6i9_mapaware_repertoire_hardpool_OP8_OP9_OP10` | `v6i14_contract_specialists_OP8_OP9_OP10` | Artifact namespace. |

Contract map:
`z0 = opening pressure`, `z1 = home defense / recovery`,
`z2 = friendly-carrier support`, `z3 = carrier conversion`.

The aliases are `v6i14`, `v6i14_contract_specialists`,
`v6i14_contract_specialist_repertoire`,
`latent_v6i14_contract_specialists`, and
`plan_faithful_latent_v6i14_contract_specialists`.

### 6.21 v6i15_contract_pressure (DIAGNOSTIC -- non-Summer scaffold)

v6i15 inherits `v6i14_contract_specialists` directly. The scientific
delta is to test whether the current frozen-shared-trunk, z-specific
actor pathway can express distinct forced-z behavior when the handcrafted
contract reward is made loud. This is a pressure test, not a router row.

The sweep arms keep router training blocked: router off, z assigned by
balanced episodes, `v6i9_training_stage = "repertoire"`, shared actor
trunk frozen, z-specific modules trainable, same OP8/OP9/OP10 hard pool,
same map surface, same contract map.

Resolved-config diff vs `v6i14_contract_specialists` is exactly:
`{experiment_id, latent_contract_specialist_coef, run_tag}`.

| Field | v6i14 value | 3x arm | 6x arm | 10x arm |
|-------|-------------|--------|--------|---------|
| `experiment_id` | `v6i14` | `v6i15` | `v6i15` | `v6i15` |
| `latent_contract_specialist_coef` | `0.25` | `0.75` | `1.50` | `2.50` |
| `run_tag` | `v6i14_contract_specialists_OP8_OP9_OP10` | `v6i15_contract_pressure_3x_OP8_OP9_OP10` | `v6i15_contract_pressure_6x_OP8_OP9_OP10` | `v6i15_contract_pressure_10x_OP8_OP9_OP10` |

Aliases:
`v6i15`, `v6i15_contract_pressure`, and `v6i15_contract_pressure_3x`
resolve to the 3x arm. The 6x and 10x arms are available as
`v6i15_contract_pressure_6x` and `v6i15_contract_pressure_10x`, plus the
matching `latent_v6i15_...` and `plan_faithful_latent_v6i15_...` aliases.

Promotion logic: do not train a router from any v6i15 arm unless a
complete forced-z behavior grid shows material separation: all z
represented, mean pair distance rising versus v6i14, max pair distance
clearing the prior 0.0717 ceiling by a clear margin, and at least some
pairs above the behavior threshold. If 10x pressure does not move the
fingerprints, the next diagnostic is z-specific capacity or contract
feature design.

### 6.22 v6i16_capacity_sharp_contracts (DIAGNOSTIC -- non-Summer scaffold)

v6i16 inherits the v6i15 3x arm directly. The scientific delta is to test
whether v6i15 failed because the contract features were too easy for one
generic behavior to satisfy, because the z-specific actor pathway lacked
control authority, or because both were true.

This row deliberately keeps router training blocked. It is not a
paper-faithful row and not a Summer-compatible extension: it uses
handcrafted z-role reward shaping and actor z-pathway capacity changes.

Resolved-config diffs vs `v6i15_contract_pressure_3x`:

| Arm | Changed fields | Run tag |
|-----|----------------|---------|
| `v6i16_sharp_contracts` | `{experiment_id, latent_contract_specialist_variant, run_tag}` | `v6i16_sharp_contracts_3x_OP8_OP9_OP10` |
| `v6i16_capacity` | `{experiment_id, latent_actor_z_adapter_enabled, latent_actor_z_adapter_init_std, latent_actor_z_adapter_scale, latent_z_gate_init, run_tag}` | `v6i16_capacity_3x_OP8_OP9_OP10` |
| `v6i16_capacity_sharp_contracts` | `{experiment_id, latent_actor_z_adapter_enabled, latent_actor_z_adapter_init_std, latent_actor_z_adapter_scale, latent_contract_specialist_variant, latent_z_gate_init, run_tag}` | `v6i16_capacity_sharp_contracts_3x_OP8_OP9_OP10` |

Arm settings: `latent_contract_specialist_coef = 0.75` for all arms. The
sharp-contract arms set `latent_contract_specialist_variant = "sharp"`.
The capacity arms set `latent_z_gate_init = 0.08`,
`latent_actor_z_adapter_enabled = True`,
`latent_actor_z_adapter_scale = 0.10`, and
`latent_actor_z_adapter_init_std = 0.05`. The repertoire freeze allowlist
includes `z_adapter`, so the capacity module is trainable while the shared
actor trunk remains frozen.

Sharp contract map:
`z0 = pressure / interception / enemy-carrier disruption`,
`z1 = escort / carrier support / conversion support`,
`z2 = home-flag defense / returns / denial`,
`z3 = spacing / lane control / split pressure`.

Aliases: `v6i16`, `v6i16_capacity_feature_ablation`,
`v6i16_capacity_sharp_contracts`,
`latent_v6i16_capacity_sharp_contracts`, and
`plan_faithful_latent_v6i16_capacity_sharp_contracts` resolve to the
combined Arm C. Arm A and Arm B are available as `v6i16_sharp_contracts`
and `v6i16_capacity`, plus matching `latent_v6i16_...` and
`plan_faithful_latent_v6i16_...` aliases.

Promotion logic: run 5 updates per arm, then a complete forced-z behavior
fingerprint grid. Promotion requires mean pair distance clearly above
v6i15's 0.0436, at least some pairs above threshold, stable role ownership
metrics, and `unique_best_z_count > 1` on non-binary margin/timing
surfaces. Win-rate saturation alone is not evidence.

### 6.23 v6i17_surface_pressure_diagnostic (DIAGNOSTIC -- non-Summer scaffold)

v6i17 inherits the v6i16 combined arm directly. The scientific delta is to
test whether v6i16 failed because the current OP8/OP9/OP10 surface lets one
dominant generalist behavior satisfy every contract without role tradeoffs.

This row deliberately keeps router training blocked. It is not a
paper-faithful row and not a Summer-compatible extension: it inherits
handcrafted z-role reward shaping and actor z-pathway capacity changes from
v6i16, then changes the training/evaluation arena.

Resolved-config diff vs `v6i16_capacity_sharp_contracts` is exactly:
`{experiment_id, opponent_pool, run_tag}`.

| Field | v6i16 combined value | This preset | Note |
|-------|----------------------|-------------|------|
| `experiment_id` | `v6i16` | `v6i17` | Artifact and protocol identity. |
| `opponent_pool` | `("OP8", "OP9", "OP10")` | `("OP8", "OP9", "OP10", "OP11", "OP12")` | Adds the existing elite hardpool BT opponents to create harder/asymmetric role pressure. |
| `run_tag` | `v6i16_capacity_sharp_contracts_3x_OP8_OP9_OP10` | `v6i17_surface_pressure_diagnostic_OP8_OP9_OP10_OP11_OP12` | Artifact namespace advertises the surface-pressure diagnostic. |

All other v6i16 scaffold fields stay unchanged: router off,
`balanced_episode` z assignment, sharp contract variant, 3x contract
coefficient, stronger z pathway, `v6i9_training_stage = "repertoire"`, and
frozen shared actor trunk.

Aliases: `v6i17`, `v6i17_surface_pressure_diagnostic`,
`v6i17_harder_asymmetric_opponents`,
`latent_v6i17_surface_pressure_diagnostic`, and
`plan_faithful_latent_v6i17_surface_pressure_diagnostic`.

Promotion logic: run a short 5-update diagnostic first, then forced-z
fingerprints over the harder surface. Router training remains blocked unless
forced-z behavior pair distance clears the prior ~0.045 ceiling, at least some
pairs exceed threshold, `unique_best_z_count > 1`, and score-margin, tempo, or
role metrics show consequence differences. The `ORACLE_GAP_PLUS_CONTEXT` line
alone is not promotion evidence.

### 6.24 v6i18_margin_tempo_surface_diagnostic (DIAGNOSTIC -- non-Summer scaffold)

v6i18 inherits `v6i17_surface_pressure_diagnostic` directly. The scientific
delta is to test whether v6i17 failed because the arena still graded every z
mostly by saturated binary win/loss. It keeps the v6i17 specialist-birth
machinery fixed and changes only the consequence surface.

This row deliberately keeps router training blocked. It is not a paper-faithful
row and not a Summer-compatible extension: it inherits handcrafted z-role
contract shaping and z-pathway capacity changes, then adds noncanonical
margin/tempo reward pressure.

Resolved-config diff vs `v6i17_surface_pressure_diagnostic` is exactly:
`{env_stalemate_max_steps, env_surface_blue_capture_tempo_bonus,
env_surface_blue_near_cap_bonus, env_surface_red_carrier_progress_penalty,
env_surface_red_flag_touch_penalty, env_surface_score_margin_coef,
experiment_id, max_decision_steps, run_tag}`.

| Field | v6i17 value | This preset | Note |
|-------|-------------|-------------|------|
| `experiment_id` | `v6i17` | `v6i18` | Artifact and protocol identity. |
| `max_decision_steps` | `320` | `240` | Shorter horizon makes tempo and conversion pressure visible. |
| `env_stalemate_max_steps` | `120` | `80` | Shorter no-score window to reduce slow saturated wins. |
| `env_surface_score_margin_coef` | `0.0` | `0.15` | Terminal score-margin pressure. |
| `env_surface_blue_capture_tempo_bonus` | `0.0` | `0.25` | Earlier blue captures receive more reward. |
| `env_surface_red_flag_touch_penalty` | `0.0` | `0.20` | Penalizes allowing red flag grabs. |
| `env_surface_red_carrier_progress_penalty` | `0.0` | `0.025` | Penalizes red carrier progress toward blue home. |
| `env_surface_blue_near_cap_bonus` | `0.0` | `0.015` | Rewards blue carrier near-cap conversion pressure. |
| `run_tag` | `v6i17_surface_pressure_diagnostic_OP8_OP9_OP10_OP11_OP12` | `v6i18_margin_tempo_surface_OP8_OP9_OP10_OP11_OP12` | Artifact namespace advertises the margin/tempo diagnostic. |

All other v6i17 scaffold fields stay unchanged: OP8/OP9/OP10/OP11/OP12
opponent surface, router off, `balanced_episode` z assignment, sharp contract
variant, 3x contract coefficient, stronger z pathway,
`v6i9_training_stage = "repertoire"`, and frozen shared actor trunk.

Aliases: `v6i18`, `v6i18_margin_tempo_surface_diagnostic`,
`v6i18_margin_tempo_surface`, `latent_v6i18_margin_tempo_surface_diagnostic`,
and `plan_faithful_latent_v6i18_margin_tempo_surface_diagnostic`.

Promotion logic: run a short 5-update diagnostic first, then forced-z
fingerprints over OP8..OP12 with margin and tempo metrics included. Router
training remains blocked unless score margin, capture timing, enemy pressure,
near-cap conversion, or role fingerprints separate by z. Win rate can remain
100% and still be useful only if the non-binary consequence metrics separate;
oracle gap alone is not promotion evidence.

### 6.25 v6i19_map_pool_surface_diagnostic (DIAGNOSTIC -- non-Summer scaffold)

v6i19 inherits `v6i18_margin_tempo_surface_diagnostic` directly. The scientific
delta is to test whether fixed-map margin/tempo pressure failed because every
episode saw the same layout. It keeps the v6i18 specialist-birth machinery and
consequence surface fixed and adds only per-episode `map_pool` sampling.

This row deliberately keeps router training blocked. It is not paper-faithful and
not a Summer-compatible extension.

Resolved-config diff vs `v6i18_margin_tempo_surface_diagnostic` is exactly:
`{experiment_id, map_pool, run_tag}`.

| Field | v6i18 value | This preset | Note |
|-------|-------------|-------------|------|
| `experiment_id` | `v6i18` | `v6i19` | Artifact and protocol identity. |
| `map_pool` | `()` | `("map_b_split_lane", "map_b_split_lane_v2")` | Uniform per-episode layout sample; `map_id` recorded in telemetry. |
| `run_tag` | `v6i18_margin_tempo_surface_OP8_OP9_OP10_OP11_OP12` | `v6i19_map_pool_surface_diagnostic_OP8_OP9_OP10_OP11_OP12` | Artifact namespace advertises map-pool diagnostic. |

All other v6i18 scaffold fields stay unchanged: margin/tempo surface coefs,
shorter horizon/stalemate, OP8..OP12 pool, router off, `balanced_episode`,
sharp 3x contracts, v6i16 z capacity, frozen shared actor.

Aliases: `v6i19`, `v6i19_map_pool_surface_diagnostic`, `v6i19_map_pool_surface`,
`latent_v6i19_map_pool_surface_diagnostic`,
`plan_faithful_latent_v6i19_map_pool_surface_diagnostic`.

Promotion logic: 5-update diagnostic from the v6i9 generalist anchor, then
forced-z fingerprints over all `map_pool` layouts x OP8..OP12 grouped by
opponent x map x z. Router training remains blocked unless
`behavior_pair_distance_mean > 0.06`, `unique_best_z_count > 1`, pairs above
threshold, and margin/tempo/role metrics separate by z.

### 6.26 v6i20_asymmetry_handicap_surface_diagnostic (DIAGNOSTIC -- non-Summer scaffold)

v6i20 inherits `v6i19_map_pool_surface_diagnostic` directly. The scientific
delta is to test whether v6i19 failed because layout variation and mild
margin/tempo pressure still allowed one generalist behavior to pass every
context. It keeps the v6i19 specialist-birth machinery fixed and strengthens
only asymmetric consequence pressure: enemy flag touches and enemy carrier
progress become more expensive, while fast blue capture and near-cap conversion
become more valuable.

This row deliberately keeps router training blocked. It is not paper-faithful
and not a Summer-compatible extension.

Resolved-config diff vs `v6i19_map_pool_surface_diagnostic` is exactly:
`{env_surface_blue_capture_tempo_bonus, env_surface_blue_near_cap_bonus,
env_surface_red_carrier_progress_penalty, env_surface_red_flag_touch_penalty,
experiment_id, run_tag}`.

| Field | v6i19 value | This preset | Note |
|-------|-------------|-------------|------|
| `experiment_id` | `v6i19` | `v6i20` | Artifact and protocol identity. |
| `env_surface_blue_capture_tempo_bonus` | `0.25` | `0.45` | Stronger fast-capture pressure. |
| `env_surface_red_flag_touch_penalty` | `0.20` | `0.50` | Stronger penalty for allowing red flag touches. |
| `env_surface_red_carrier_progress_penalty` | `0.025` | `0.075` | Stronger penalty for red carrier progress. |
| `env_surface_blue_near_cap_bonus` | `0.015` | `0.035` | Stronger near-cap conversion pressure. |
| `run_tag` | `v6i19_map_pool_surface_diagnostic_OP8_OP9_OP10_OP11_OP12` | `v6i20_asymmetry_handicap_surface_OP8_OP9_OP10_OP11_OP12` | Artifact namespace advertises the asymmetry diagnostic. |

All other v6i19 scaffold fields stay unchanged: OP8..OP12 pool, two-layout
`map_pool`, `max_decision_steps = 240`, `env_stalemate_max_steps = 80`,
`env_surface_score_margin_coef = 0.15`, router off, `balanced_episode`, sharp
3x contracts, v6i16 z capacity, and frozen shared actor.

Aliases: `v6i20`, `v6i20_asymmetry_handicap_surface_diagnostic`,
`v6i20_asymmetry_handicap_surface`, `v6i20_handicap_surface`,
`latent_v6i20_asymmetry_handicap_surface_diagnostic`, and
`plan_faithful_latent_v6i20_asymmetry_handicap_surface_diagnostic`.

Promotion logic: 5-update diagnostic from the v6i9 generalist anchor, then
surface-matched forced-z fingerprints over OP8..OP12 x both map-pool layouts.
Router training remains blocked unless the tradeoff table separates by z:
score margin, time-to-first-score, enemy flag touches allowed, enemy carrier
progress, returns/interceptions, escort allocation, or near-cap conversions.
The strict gates remain `unique_best_z_count > 1`,
`behavior_pair_distance_mean > 0.06`, at least one pair above threshold, and
best-z variation across opponent x map cells.

---

### 6.27 v6i21_adaptive_op8_op12_hardpool_calibration (DIAGNOSTIC -- non-Summer scaffold)

v6i21 inherits `v6i20_asymmetry_handicap_surface_diagnostic` directly. The
scientific change is **not** in the preset config: OP8-OP12 were upgraded
in-place in the BT engine to adaptive hardpool v2 with intra-episode memory
(lane repetition, escort density, overcommit, near-cap collapse, fast-conversion
response). Same opponent names; stronger adaptive counter-play.

**Historical comparability:** OP8-OP12 results from runs before v6i21 are **not**
directly comparable to post-v6i21 OP8-OP12 results.

**v6i21B in-place pressure tuning:** after the first v6i21 calibration failed
at 99.2% blue WR, OP8-OP12 were tuned again without adding new opponent IDs or
changing the resolved PPO config. The patch strengthens adaptive trigger timing,
near-cap collapse, intercept block geometry, OP12 overcommit counter-push, 2v2
red dynamics floors, and a hardpool-only blue carrier speed multiplier of 0.95.
Pre-v6i21B calibration artifacts are not directly comparable to v6i21B
calibration artifacts.

**Resolved-config diff vs v6i20:** exactly `{experiment_id, run_tag}`.

| Field | v6i20 | v6i21 |
|-------|-------|-------|
| `experiment_id` | `v6i20` | `v6i21` |
| `run_tag` | `v6i20_asymmetry_handicap_surface_OP8_OP9_OP10_OP11_OP12` | `v6i21_adaptive_op8_op12_hardpool_calibration` |

Engine files: `gpu_env/_core/_bt_adaptive.py`, `gpu_env/_core/_bt_profiles.py`
(levels 8-12), `gpu_env/_core/_step.py` (hardpool carrier speed pressure),
`opponent_params.py` (OP8-OP12 dynamics).

Aliases: `v6i21`, `v6i21_adaptive_op8_op12_hardpool`,
`v6i21_adaptive_op8_op12_hardpool_calibration`, `v6i21_adaptive_hardpool_calibration`,
`latent_v6i21_adaptive_op8_op12_hardpool_calibration`,
`plan_faithful_latent_v6i21_adaptive_op8_op12_hardpool_calibration`.

First gate: calibration eval only (`experiments/run_v6i21_adaptive_hardpool_calibration.py`).
Target mean blue WR 35-65%, no cell at 95%+. Router and specialist-birth training
remain blocked until calibration passes and a follow-on forced-z grid shows tradeoff
separation.

---

### 6.28 v6i21d_adaptive_hardpool_brutal_denial_calibration (DIAGNOSTIC -- non-Summer scaffold)

v6i21D inherits `v6i21c_adaptive_hardpool_denial_calibration` directly. The
scientific change is an upper-bound calibration question: can the same OP8-OP12
IDs be made physically/adaptively hard enough to break the v6i9 generalist's
saturated blue WR at all?

This is not a fair final opponent setting. It is a pressure-ceiling probe before
any PPO, router, or specialist-birth run.

**Resolved-config diff vs v6i21C:** exactly `{experiment_id, run_tag}`.

| Field | v6i21C | v6i21D |
|-------|--------|--------|
| `experiment_id` | `v6i21c` | `v6i21d` |
| `run_tag` | `v6i21c_adaptive_hardpool_denial_calibration` | `v6i21d_adaptive_hardpool_brutal_denial_calibration` |

Engine files: `gpu_env/_core/_bt_adaptive.py`,
`gpu_env/_core/_bt_profiles.py`, `gpu_env/_core/_dynamics.py`,
`gpu_env/_core/_step.py`, `gpu_env/_core/_rules.py`, and
`opponent_params.py`.

Aliases: `v6i21d`, `v6i21d_adaptive_hardpool_brutal_denial_calibration`,
`v6i21d_adaptive_hardpool_brutal_denial`,
`latent_v6i21d_adaptive_hardpool_brutal_denial_calibration`, and
`plan_faithful_latent_v6i21d_adaptive_hardpool_brutal_denial_calibration`.

First gate: 10-episode smoke against the v6i9 generalist after v6i21C finishes.
Break-saturation target: mean blue WR below 85%, no more than 3/10 cells at
95%+, at least two cells below 75%, red scores rising, and blue score below
2.5 in several cells. If it overshoots into very low blue WR, tune back in a
follow-up calibration. Router and specialist-birth training remain blocked.

---

### 6.29 v6i21e_targeted_denial_balance_calibration (DIAGNOSTIC -- non-Summer scaffold)

v6i21E inherits `v6i21d_adaptive_hardpool_brutal_denial_calibration` directly. After
v6i21D smoke showed the denial lever works (80% mean WR, 4/10 in-band) but OP8/OP10/OP11
cells remain saturated, v6i21E applies **targeted** per-opponent hardening without
touching OP9/OP12.

**Resolved-config diff vs v6i21D:** exactly `{experiment_id, run_tag}`.

| Field | v6i21D | v6i21E |
|-------|--------|--------|
| `experiment_id` | `v6i21d` | `v6i21e` |
| `run_tag` | `v6i21d_adaptive_hardpool_brutal_denial_calibration` | `v6i21e_targeted_denial_balance_calibration` |

Engine files: `gpu_env/_core/_bt_adaptive.py` (per-level OP8/OP10/OP11 overrides),
`gpu_env/_core/_bt_profiles.py` (OP8/OP10/OP11 profile scalars),
`opponent_params.py` (OP8/OP10/OP11 2v2 dynamics only).

Aliases: `v6i21e`, `v6i21e_targeted_denial_balance_calibration`,
`v6i21e_targeted_denial_balance`,
`latent_v6i21e_targeted_denial_balance_calibration`, and
`plan_faithful_latent_v6i21e_targeted_denial_balance_calibration`.

Smoke target: mean blue WR 60-75%, 6+/10 cells in 35-65% band, at most 2/10
saturated. Router and specialist-birth training remain blocked until full
25-episode calibration passes.

---

### 6.30 v6i21f_op8_carrier_denial_calibration (DIAGNOSTIC -- non-Summer scaffold)

v6i21F inherits `v6i21e_targeted_denial_balance_calibration` directly. After v6i21E
smoke showed OP8 still saturated (100/100 WR, blue pinned at 3.0 despite rising red
scores), v6i21F applies an OP8-only patch: disable counter-capture and 2v1 scoring,
dual intercept on predictive lead + cap-lane body, wider near-cap collapse, longer
interceptor locks, lower coordinated-attack probability. OP9–OP12 unchanged.

**Resolved-config diff vs v6i21E:** exactly `{experiment_id, run_tag}`.

| Field | v6i21E | v6i21F |
|-------|--------|--------|
| `experiment_id` | `v6i21e` | `v6i21f` |
| `run_tag` | `v6i21e_targeted_denial_balance_calibration` | `v6i21f_op8_carrier_denial_calibration` |

Engine files: `gpu_env/_core/_bt_adaptive.py` (OP8-only overrides),
`gpu_env/_core/_bt_profiles.py` (OP8 profile), `opponent_params.py` (OP8 2v2 only).

Aliases: `v6i21f`, `v6i21f_op8_carrier_denial_calibration`, `v6i21f_op8_carrier_denial`,
`latent_v6i21f_op8_carrier_denial_calibration`, and
`plan_faithful_latent_v6i21f_op8_carrier_denial_calibration`.

---

### 6.31 v6i21g_easy_cell_conversion_denial_calibration (DIAGNOSTIC -- non-Summer scaffold)

v6i21G inherits `v6i21f_op8_carrier_denial_calibration` directly. v6i21F showed
that pure OP8 carrier hunting reduced red scoring without breaking blue caps,
while OP10 split_lane and both OP11 cells stayed saturated. v6i21G corrects the
easy-cell geometry: OP8/OP11 cap-lane body-blocking is restored, OP10 cuts off
conversion instead of chasing carrier position, and OP8/OP10/OP11 2v2 pressure is
raised. OP9/OP12 remain unchanged.

**Resolved-config diff vs v6i21F:** exactly `{experiment_id, run_tag}`.

| Field | v6i21F | v6i21G |
|-------|--------|--------|
| `experiment_id` | `v6i21f` | `v6i21g` |
| `run_tag` | `v6i21f_op8_carrier_denial_calibration` | `v6i21g_easy_cell_conversion_denial_calibration` |

Engine files: `gpu_env/_core/_bt_adaptive.py` (OP8/OP10/OP11 route overrides),
`gpu_env/_core/_bt_profiles.py` (OP8/OP10/OP11 profile scalars), and
`opponent_params.py` (OP8/OP10/OP11 2v2 dynamics only).

Aliases: `v6i21g`, `v6i21g_easy_cell_conversion_denial_calibration`,
`v6i21g_easy_cell_conversion_denial`,
`latent_v6i21g_easy_cell_conversion_denial_calibration`, and
`plan_faithful_latent_v6i21g_easy_cell_conversion_denial_calibration`.

---

### 6.32 v6i21h_saturation_surrogate_calibration (DIAGNOSTIC -- non-Summer scaffold)

v6i21H inherits `v6i21g_easy_cell_conversion_denial_calibration` directly. G
confirmed that bespoke OP8/OP10/OP11 denial geometry still saturated; OP9 and
OP12 remained the only calibrated shapes. H stops adding new geometry and reuses
those working shapes: OP8 becomes OP9-like fortress pressure, while OP10/OP11
become OP12-like counter pressure. The failed OP8 dual-denial, OP10 escort-break,
and OP11 repeat-intercept adaptive route overrides are disabled.

**Resolved-config diff vs v6i21G:** exactly `{experiment_id, run_tag}`.

| Field | v6i21G | v6i21H |
|-------|--------|--------|
| `experiment_id` | `v6i21g` | `v6i21h` |
| `run_tag` | `v6i21g_easy_cell_conversion_denial_calibration` | `v6i21h_saturation_surrogate_calibration` |

Engine files: `gpu_env/_core/_bt_adaptive.py`,
`gpu_env/_core/_bt_profiles.py`, and `opponent_params.py`.

Aliases: `v6i21h`, `v6i21h_saturation_surrogate_calibration`,
`v6i21h_saturation_surrogate`, `latent_v6i21h_saturation_surrogate_calibration`,
and `plan_faithful_latent_v6i21h_saturation_surrogate_calibration`.

---

### 6.33 v6i21i_op8_extreme_physical_calibration (DIAGNOSTIC -- non-Summer scaffold)

v6i21I inherits `v6i21h_saturation_surrogate_calibration` directly. H restored
red pressure for OP8 but did not break blue WR. I adds OP8-only physical
pressure: blue carrier speed `0.35`, OP8 red speed `1.60`, OP8 near-flag
interceptor boost `1.85`, and OP8 2v2 speed range `1.35-1.45`.

**Resolved-config diff vs v6i21H:** exactly `{experiment_id, run_tag}`.

| Field | v6i21H | v6i21I |
|-------|--------|--------|
| `experiment_id` | `v6i21h` | `v6i21i` |
| `run_tag` | `v6i21h_saturation_surrogate_calibration` | `v6i21i_op8_extreme_physical_calibration` |

Engine files: `gpu_env/_core/_bt_adaptive.py`, `gpu_env/_core/_step.py`, and
`opponent_params.py`.

Aliases: `v6i21i`, `v6i21i_op8_extreme_physical_calibration`,
`v6i21i_op8_extreme_physical`,
`latent_v6i21i_op8_extreme_physical_calibration`, and
`plan_faithful_latent_v6i21i_op8_extreme_physical_calibration`.

---

### 6.34 v6i21j_hardpool_balance_calibration (DIAGNOSTIC -- non-Summer scaffold)

v6i21J inherits `v6i21i_op8_extreme_physical_calibration` directly. I broke OP8
saturation but left OP8 map_b above the desired band and OP10/OP11 still need
pressure. J keeps OP8 hard and adds OP10/OP11 physical pressure while leaving
OP9/OP12 unchanged.

**Resolved-config diff vs v6i21I:** exactly `{experiment_id, run_tag}`.

| Field | v6i21I | v6i21J |
|-------|--------|--------|
| `experiment_id` | `v6i21i` | `v6i21j` |
| `run_tag` | `v6i21i_op8_extreme_physical_calibration` | `v6i21j_hardpool_balance_calibration` |

Engine files: `gpu_env/_core/_bt_adaptive.py` and `gpu_env/_core/_step.py`.

Aliases: `v6i21j`, `v6i21j_hardpool_balance_calibration`,
`v6i21j_hardpool_balance`, `latent_v6i21j_hardpool_balance_calibration`, and
`plan_faithful_latent_v6i21j_hardpool_balance_calibration`.

---

### 6.35 v6i22_adaptive_hardpool_repertoire_birth (SUMMER-COMPATIBLE EXTENSION)

v6i22 inherits `v6i21j_hardpool_balance_calibration` directly and turns the
next fork into a label-free repertoire-birth run instead of another calibration
or router row. The router is off, `latent_assignment_mode = "balanced_episode"`
holds one unlabeled z for the whole episode, and `v6i9_training_stage =
"repertoire"` keeps the shared trunk frozen while z-specific modules and the
critic remain trainable.

This is **not** a contract-specialist row: the z-indexed contract reward scaffold
is explicitly disabled. It is also not `PAPER-FAITHFUL`, because it inherits the
v6 hardpool, staged freeze, and adapter/z-residual machinery. It is
Summer-compatible in the narrower sense used for this fork: no handcrafted
strategy labels, no opponent-ID supervision, no oracle-z targets, no router
distillation, and no auxiliary label head.

**Resolved-config diff vs v6i21J:** exactly `{experiment_id,
latent_contract_specialist_coef, latent_contract_specialist_enabled,
latent_contract_specialist_variant, run_tag}`.

| Field | v6i21J | v6i22 |
|-------|--------|-------|
| `experiment_id` | `v6i21j` | `v6i22` |
| `latent_contract_specialist_enabled` | `True` | `False` |
| `latent_contract_specialist_coef` | `0.75` | `0.0` |
| `latent_contract_specialist_variant` | `sharp` | `base` |
| `run_tag` | `v6i21j_hardpool_balance_calibration` | `v6i22_adaptive_hardpool_repertoire_birth_OP8_OP9_OP10_OP11_OP12` |

Aliases: `v6i22`, `v6i22_adaptive_hardpool_repertoire_birth`,
`v6i22_repertoire_birth`, `latent_v6i22_adaptive_hardpool_repertoire_birth`,
and `plan_faithful_latent_v6i22_adaptive_hardpool_repertoire_birth`.

Promotion logic: train a short 5-20 update repertoire-birth diagnostic from the
v6i9 generalist anchor, then run forced-z fingerprints over OP8-OP12 x both map
layouts. Router training remains blocked unless forced z produces real options:
behavior pair distance above the prior 0.04-0.05 ceiling, at least one or two
pairs above threshold, role/tempo/margin fingerprints separating by z, and
`unique_best_z_count > 1` by margin, tempo, or WR.

---

### 6.36 v6i22b_context_behavior_diversity (SUMMER-COMPATIBLE EXTENSION)

v6i22B inherits `v6i22_adaptive_hardpool_repertoire_birth` directly. It keeps
router training off, keeps `latent_assignment_mode = "balanced_episode"`, keeps
one unlabeled z for the whole episode, keeps the contract-specialist reward
disabled, and keeps the v6 repertoire-stage freeze/adapters inherited from
V6I22.

The only scientific delta is label-free anti-collapse pressure: successful
terminal episodes receive a small behavior-contrast bonus when their trajectory
fingerprint separates from other z centroids inside the same opponent x map
context. The signal does not assign roles to z indices. It says only that the
z branches should not collapse into the same trajectory signature while still
succeeding.

This remains **not** `PAPER-FAITHFUL`, because it adds a reward-side behavior
contrast term and inherits the v6 hardpool/staged-freeze machinery. It is
Summer-compatible for this fork because it adds no handcrafted z-role contracts,
no supervised strategy labels, no opponent-ID actor shortcut, no oracle best-z
targets, no router distillation, and no router training.

**Resolved-config diff vs v6i22 primary arm:** exactly `{experiment_id,
latent_behavior_contrast_coef, latent_behavior_contrast_margin, run_tag}`.

| Field | v6i22 | v6i22B primary |
|-------|-------|----------------|
| `experiment_id` | `v6i22` | `v6i22b_coef003` |
| `latent_behavior_contrast_coef` | `0.0` | `0.03` |
| `latent_behavior_contrast_margin` | `0.0` | `0.06` |
| `run_tag` | `v6i22_adaptive_hardpool_repertoire_birth_OP8_OP9_OP10_OP11_OP12` | `v6i22b_context_behavior_diversity_coef003_OP8_OP9_OP10_OP11_OP12` |

Coefficient sweep arms keep the same diff surface and set
`latent_behavior_contrast_coef` to `0.01`, `0.03`, or `0.05`.

Aliases: `v6i22b`, `v6i22b_context_behavior_diversity`,
`v6i22b_behavior_diversity_coef003`,
`latent_v6i22b_context_behavior_diversity`, and
`plan_faithful_latent_v6i22b_context_behavior_diversity`. Sweep aliases:
`v6i22b_coef001`, `v6i22b_behavior_diversity_coef001`,
`latent_v6i22b_context_behavior_diversity_coef001`,
`plan_faithful_latent_v6i22b_context_behavior_diversity_coef001`,
`v6i22b_coef005`, `v6i22b_behavior_diversity_coef005`,
`latent_v6i22b_context_behavior_diversity_coef005`, and
`plan_faithful_latent_v6i22b_context_behavior_diversity_coef005`.

Promotion logic: run 5-update coefficient diagnostics from the same v6i9
generalist anchor, then run forced-z fingerprints over OP8-OP12 x both map
layouts. Continue only an arm that keeps Stage-C passing, preserves positive
WR/margin advantage, raises behavior pair distance above V6I22's 20-update
level, and starts moving toward at least one above-threshold behavior pair.
Router training remains blocked until forced-z behaviors are visibly distinct.

---

### 6.37 v6i22c_contextual_outcome_diversity (SUMMER-COMPATIBLE EXTENSION)

v6i22C inherits `v6i22_adaptive_hardpool_repertoire_birth` directly. It keeps
router training off, keeps `latent_assignment_mode = "balanced_episode"`, keeps
one unlabeled z for the whole episode, keeps the contract-specialist reward
disabled, keeps behavior-contrast reward disabled, and keeps the v6
repertoire-stage freeze/adapters inherited from V6I22.

The only scientific delta is label-free contextual outcome diversity:
successful terminal episodes receive a bounded bonus when their generic outcome
scalar differs from other z outcome centroids inside the same opponent x map
context. The current scalar is score margin (`blue_score - red_score`). It does
not use interception, escort, defense, tempo, lane, or other role/fingerprint
metrics as training targets.

This remains **not** `PAPER-FAITHFUL`, because it adds a reward-side outcome
diversity term and inherits the v6 hardpool/staged-freeze machinery. It is
Summer-compatible for this fork because it adds no handcrafted z-role contracts,
no supervised strategy labels, no opponent-ID actor shortcut, no oracle best-z
targets, no router distillation, no behavior-role reward, and no router
training.

**Resolved-config diff vs v6i22 primary arm:** exactly `{experiment_id,
latent_outcome_diversity_coef, run_tag}`.

| Field | v6i22 | v6i22C primary |
|-------|-------|----------------|
| `experiment_id` | `v6i22` | `v6i22c_coef003` |
| `latent_outcome_diversity_coef` | `0.0` | `0.03` |
| `run_tag` | `v6i22_adaptive_hardpool_repertoire_birth_OP8_OP9_OP10_OP11_OP12` | `v6i22c_contextual_outcome_diversity_coef003_OP8_OP9_OP10_OP11_OP12` |

Aliases: `v6i22c`, `v6i22c_contextual_outcome_diversity`,
`v6i22c_outcome_diversity_coef003`,
`latent_v6i22c_contextual_outcome_diversity`, and
`plan_faithful_latent_v6i22c_contextual_outcome_diversity`.

Promotion logic: run a 5-update diagnostic from the same v6i9 generalist anchor,
then run forced-z fingerprints over OP8-OP12 x both map layouts. Continue only
if Stage-C remains passing, unique best-z stays above one, WR/margin advantage
does not collapse, and behavior pair distance moves materially above the
V6I22B ceiling. Router training remains blocked until forced-z behaviors are
visibly distinct.

---

### 6.38 v6i22d_strong_behavior_diversity (SUMMER-COMPATIBLE EXTENSION)

v6i22D inherits `v6i22_adaptive_hardpool_repertoire_birth` directly. It keeps
router training off, keeps `latent_assignment_mode = "balanced_episode"`, keeps
one unlabeled z for the whole episode, keeps the contract-specialist reward
disabled, and keeps outcome-diversity disabled.

The scientific delta is stronger label-free behavior-contrast pressure after
V6I22B (coef `<= 0.05`) and V6I22C failed the forced-z behavior birth gate.
Successful terminal episodes receive a behavior-contrast bonus when their
trajectory fingerprint separates from other z centroids inside the same
opponent x map context. The signal uses the same fingerprint features as the
birth gate (`n_intercept_near_enemy_carrier`, `carrier_escort_count`,
`num_defenders`, `team_spread`, `objective_entry_timing`,
`nearest_blue_to_enemy_carrier`, and related trajectory telemetry).

This remains **not** `PAPER-FAITHFUL`. It is Summer-compatible because it adds
no handcrafted z-role contracts, no supervised strategy labels, no opponent-ID
actor shortcut, no oracle best-z targets, no router distillation, and no router
training.

**Resolved-config diff vs v6i22 primary arm:** exactly `{experiment_id,
latent_behavior_contrast_coef, latent_behavior_contrast_margin, run_tag}`.

| Field | v6i22 | v6i22D primary |
|-------|-------|----------------|
| `experiment_id` | `v6i22` | `v6i22d_coef010` |
| `latent_behavior_contrast_coef` | `0.0` | `0.10` |
| `latent_behavior_contrast_margin` | `0.0` | `0.06` |
| `run_tag` | `v6i22_adaptive_hardpool_repertoire_birth_OP8_OP9_OP10_OP11_OP12` | `v6i22d_strong_behavior_diversity_coef010_OP8_OP9_OP10_OP11_OP12` |

Sweep arm `v6i22d_coef005` sets `latent_behavior_contrast_coef = 0.05` (same
coefficient as `v6i22b_coef005`, included as a paired control under the v6i22D
line).

Aliases: `v6i22d`, `v6i22d_strong_behavior_diversity`,
`v6i22d_behavior_diversity_coef010`,
`latent_v6i22d_strong_behavior_diversity`, and
`plan_faithful_latent_v6i22d_strong_behavior_diversity`. Sweep aliases:
`v6i22d_coef005`, `v6i22d_behavior_diversity_coef005`,
`latent_v6i22d_strong_behavior_diversity_coef005`, and
`plan_faithful_latent_v6i22d_strong_behavior_diversity_coef005`.

Promotion logic: run 5-update coefficient diagnostics from the same v6i9
generalist anchor, then run forced-z fingerprints over OP8-OP12 x both map
layouts. Continue only if Stage-C remains passing, unique best-z stays above
one, WR/margin advantage does not collapse, and `behavior_pair_distance_mean`
moves toward `> 0.06` with at least one above-threshold pair. Router training
remains blocked until forced-z behaviors are visibly distinct.

### 6.39 v6i22e_fixed_alpha_adapters (SUMMER-COMPATIBLE EXTENSION)

v6i22E inherits `v6i22_adaptive_hardpool_repertoire_birth` and sets
`latent_z_residual_alpha = 0.1` (Kaiming adapters, no learned gate).

**Resolved-config diff vs v6i22:** exactly `{experiment_id,
latent_z_residual_alpha, run_tag}`.

Aliases: `v6i22e`, `v6i22e_fixed_alpha_adapters`,
`latent_v6i22e_fixed_alpha_adapters`,
`plan_faithful_latent_v6i22e_fixed_alpha_adapters`.

### 6.40 v6i23_population_birth (SUMMER-COMPATIBLE EXTENSION)

v6i23 inherits `v6i22e_fixed_alpha_adapters` and enables population birth:
`latent_population_birth_active_z_only = True` and
`latent_population_birth_per_z_action_heads = True`.

**Resolved-config diff vs v6i22e:** exactly `{experiment_id,
latent_population_birth_active_z_only,
latent_population_birth_per_z_action_heads, run_tag}`.

Scientific rationale: Stage-2 freezes the shared `action_head`, so residual
adapters alone struggled to separate `π(a|s,z)` (CF action-JSD stayed near
zero after V6I22E). Independent per-z heads are Stage-2 trainable specialists
under forced `balanced_episode` assignment. Not paper-faithful (concat +
shared MLP remains the paper actor); Summer-compatible extension only.

Aliases: `v6i23`, `v6i23_population_birth`,
`latent_v6i23_population_birth`,
`plan_faithful_latent_v6i23_population_birth`.

Promotion logic: CF action-JSD pair mean `> 0.05` on ≥2 oracle-hot cells
(or head0 disagree `> 0.2` with non-tie). Router remains blocked until that
gate clears.

---

### 6.41 v6i24_full_policy_population (DIAGNOSTIC)

V6I24 is the **Path C fallback** after V6I22–V6I23 demonstrated that
shared-trunk training cannot produce functional separation:

* V6I22E: adapters moved in weight space (L2 ~9.2) but shared frozen
  `action_head` kept CF action-JSD at ~0.0002.
* V6I23: per-z heads pairwise L2 grew to ~0.063 but CF action-JSD
  stayed pinned at ~0.0002. Shared representation and optimization
  history pull every specialist into the same functional basin.

**Scientific delta:** four ordinary independent actor-critic policies
cloned from the same V6I21J-competent checkpoint (documented V6I9
generalist under the v6i21J arena). No shared gradients across members,
no adapters, no router training, no PFSP / Nash / snapshot league /
pressure rotation / distillation in this arm. Latent concat scaffold is
retained with frozen `z=0` only so the competent checkpoint can
warm-start without reshaping the actor body.

**Ancestry:** parent configuration and hardpool surface from `v6i21j`.
Checkpoint source: same V6I21J-competent zip (not V6I22E/V6I23).
Does NOT inherit latent/adapter/population-birth machinery.

**Resolved-config diff vs v6i21j:** `{enable_latent_z_residual,
fixed_latent_strategy, freeze_return_norm_after_load,
latent_assignment_mode, latent_lam_h_end, latent_lam_h_start,
latent_strategy_ppo_coef, opponent_randomize,
population_pressure_rotation_interval,
population_round_robin_updates_per_cycle, v6i9_training_stage,
experiment_id, run_tag}`. Latent concat scaffold stays on with frozen
`z=0` so the V6I9/V6I21J checkpoint can warm-start; adapters/router/
strategy losses are off. `population_training_enabled` stays `False`.

**Fixed cell pressures (from V6I21J calibration WR/variance; both maps):**

| Member | Label | Pressure |
|--------|-------|----------|
| π0 | balanced | Uniform OP8–OP12 × both maps |
| π1 | failure_cells | Weight lowest baseline WR cells |
| π2 | high_variance | High Bernoulli-variance / red-score cells |
| π3 | complementary | Complement of π1+π2 |

Budget probes: 5u / 10u / 25u per policy (max initial 25u).

Aliases: `v6i24`, `v6i24_full_policy_population`,
`latent_v6i24_full_policy_population`,
`plan_faithful_latent_v6i24_full_policy_population`.

**Evaluation gates:**

* *Primary (comparative advantage):* ≥2 cells with different best policies
  and margin `≥0.10`; **cross-fitted** context oracle > best fixed on
  held-out episodes with paired CI excluding zero (matched seeds across
  members). Hindsight `max_π` gap is diagnostic only.
* *Supporting:* CF action-JSD mean `> 0.05` on ≥2 cells, OR
  leave-one-cell-out trajectory classifier `> 50%`; payoff-row distance
  reported. Smoke 32 eps/cell; confirm promotions at 128.

**Decision tree:**

1. Primary PASS → build V6I24-D distillation → re-test distilled
   context oracle > best fixed `z` → then geometry router.
2. Trend → extend teachers to 100K.
3. Fail → redesign external training pressures.

**Status (2026-07-23):** `CLOSED_AS_PRIMARY` — soft 5u Path C retained as
landscape probe only. Method path is V6I26 LRO (§6.43 / tracker §3.37).
See `artifacts/v6i24_population_seed1/pathc_close_verdict.json`.

---

### 6.42 v6i25 counterfactual geometry→z router (DIAGNOSTIC)

**Not a training preset.** Experiment runner + helpers that freeze a V6I23
donor and train only `q_phi` against counterfactual matched-seed returns.

**Scientific delta:** test whether
`geometry → z → return` is predictable from permitted episode-start
`global_state` (no opponent ID), then whether soft-Q training of `q_phi`
recovers that gap. Separates a genuine latent-selection solution from a
per-episode hindsight lookup table.

**Corrected contracts:**

* Oracle = **cross-fitted** `z*(c)=argmax_z E_train[R|c,z]`, evaluated on
  held-out seeds — **not** `max_z R` per episode.
* Stage A must pass (`context-oracle > best_fixed`, paired CI excludes 0)
  before Stage B router training.
* Primary loss = `−Σ softmax(Q̂/τ) log q_φ`; centered-advantage is ablation.
* Loud failure if `global_state` missing / non-finite / all-zero / unique
  contexts ≤ 1. Aggregate conflicting opponents under the same geometry.

**Artifacts:** `rl/router/counterfactual_router.py`,
`experiments/run_v6i25_counterfactual_router_diagnostic.py`,
`tests/test_v6i25_counterfactual_router.py`.

**Verdicts:** `PASS` / `PARTIAL` / `FAIL_SIGNAL` / `FAIL_ROUTER`
(see research-progress-tracker §3.36).

---

### 6.43 v6i26_latent_response_oracle (DIAGNOSTIC) — LRO-Summer

**Parent:** `v6i23_population_birth`.
**Classification:** `DIAGNOSTIC` (Claim B method path; not PAPER-FAITHFUL).

**Scientific delta (plain English):** Stop asking four symmetric latent
branches to invent different strategies under the same PPO mixture. Treat
each `z` as an internal response-oracle policy and train it specifically
against uncovered weaknesses of the current latent population
(PSRO / VGC-Bench / Conflux-PSRO lesson). No human strategy labels; task
return and population regret drive which branch updates.

**Resolved-config defining keys vs v6i23:**

* `latent_lro_deep_branches=True` (last-two-layer trunks per z)
* `latent_lro_active_branch_only=True`
* `fixed_latent_strategy=True`, `latent_assignment_mode=fixed`
* `latent_strategy_ppo_coef=0`, router OFF
* `recurrent_selector_hidden_dim=0`
* `freeze_return_norm_after_load=True`, `opponent_randomize=True`
* `obstacle_obs_channel=True` — keep the 8-channel obstacle plane when
  training/eval cells include `map_a_open` so V6I23+ checkpoints do not
  shape-skip the CNN stem (wall plane is zeros on open arenas)
* `experiment_id` / `run_tag` → v6i26 LRO

Contract rewards remain OFF (already on v6i23). Deep trunks sit on top of
inherited residual adapters + per-z action heads. Default LRO map surface is
`LRO_DEFAULT_MAPS = (map_a_open, map_b_split_lane, map_b_split_lane_v2)`.

**Stages:**

0. Strategic landscape scan (`run_v6i26_strategic_landscape_scan.py`)
1. LRO birth rounds (`run_v6i26_lro_oracle_round.py`) — one selected branch
   BR/round; branch and target/anchor mixture come from the current forced-z
   payoff matrix, saturated cells are excluded, and 4 eps/cell screens can only
   nominate `PROMISING_DIRECTION`
2. Confirmation — ≥32 eps/cell, CI95(`ΔG`) lower bound > 0, behavior
   distance pass, competence pass, nonredundant payoff row, and ≥3 seeds
3. Sparse router only if confirmed `G_available > 0` / niche PASS
4. Headline vs K=1 / matched non-latent / end-to-end Summer

**Artifacts:** `experiments/v6i26_lro_core.py`,
`experiments/run_v6i26_strategic_landscape_scan.py`,
`experiments/run_v6i26_lro_oracle_round.py`,
`experiments/run_v6i26_distill_and_route.py`,
`tests/test_v6i26_latent_response_oracle.py`.

Aliases: `v6i26`, `v6i26_lro`, `v6i26_latent_response_oracle`,
`latent_v6i26_latent_response_oracle`,
`plan_faithful_latent_v6i26_latent_response_oracle`,
`v6i26_phase_pod_population` (legacy phase-pod alias → same function).

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
