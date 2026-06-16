# Summer Fidelity Rules — auditable checklist and decision tree

**Owner:** This file converts the scientific specification in
[`summer-method-spec.md`](summer-method-spec.md) into auditable, machine-
scannable rules. It is the document agents must reference when:

* classifying a preset,
* deciding whether the label `paper-faithful` may be used,
* writing or updating tests for a paper-faithful preset,
* reviewing a proposed preset before implementation.

> **Read first:** [`AGENTS.md`](../../AGENTS.md), then
> [`summer-method-spec.md`](summer-method-spec.md).

---

## 1. Required conditions (PAPER-FAITHFUL gate)

A preset is `PAPER-FAITHFUL` only when **every** check below passes against
its **resolved** configuration. "Resolved" means
`dataclasses.asdict(apply_preset(PPOConfig(), preset_name))` — never inferred
from the preset name.

### 1.1 Latent variable

| # | Field / property                         | Required value                               | Verified by |
|---|------------------------------------------|----------------------------------------------|-------------|
| R1 | `use_latent_strategy`                   | `True`                                       | `tests/test_v5i4_paper_faithful.py::V5i4SparseResamplingTests` |
| R2 | `latent_k`                              | `4` (canonical) — any other value must be locked by the experiment | same |
| R3 | `fixed_latent_strategy`                 | `False`                                      | same |
| R4 | `latent_z_embed_dim`                    | `> 0` (canonical: `16`)                      | `tests/test_v5i4_paper_faithful.py::V5i4ConcatOnlyActorTests` |

### 1.2 Strategy inference network `q_phi`

| # | Field / property                                  | Required value                                                                              |
|---|---------------------------------------------------|----------------------------------------------------------------------------------------------|
| R5 | `q_phi` input width                               | `CONTEXT_STATE_DIM` (`5 · GLOBAL_STATE_DIM`); asserted in `SharedActorCentralizedCritic._assert_input_contracts` |
| R6 | `q_phi` output                                    | categorical logits over `K`                                                                  |
| R7 | `latent_strategy_aux_return_head`                 | `False`                                                                                      |
| R8 | `latent_strategy_aux_predict_phase_coef`          | `0.0`                                                                                        |
| R9 | `latent_strategy_ppo_coef`                        | `> 0` (the operational task-reward channel)                                                  |
| R10 | `latent_episode_strategy_lr`                     | `None` (no dedicated router optimizer; otherwise the main-loop categorical PPO is silenced — see `rl/custom_ppo/ppo_updater.py`) |

### 1.3 Decentralized actor

| # | Field / property                                  | Required value                                                                              |
|---|---------------------------------------------------|----------------------------------------------------------------------------------------------|
| R11 | `enable_actor_z_film`                            | `False`                                                                                      |
| R12 | `latent_actor_z_adapter_enabled`                 | `False`                                                                                      |
| R13 | `latent_actor_z_onehot_enabled`                  | `False`                                                                                      |
| R14 | actor input width                                | `actor_cnn_feature_dim + per_agent_vec_dim + latent_z_embed_dim` — for v5i4: `128 + 20 + 16 = 164` |
| R15 | `latent_actor_conditioning`                      | `"concat"` (the only allowed value — `Literal["concat"]` in `PPOConfig`)                     |
| R16 | actor must not receive global state, opponent ID, phase ID, or future info | enforced structurally by `SharedActorCentralizedCritic`'s observation contract |

### 1.4 Centralized critic

| # | Field / property                                  | Required value                                                                              |
|---|---------------------------------------------------|----------------------------------------------------------------------------------------------|
| R17 | critic input                                     | `concat(temporal_context, joint_action_onehot, z_onehot)` when latent is on                  |
| R18 | critic is scalar `V_φ(s, a, z)`                  | not a counterfactual `Q(s, a', z)` (see `rl/networks.py::CentralizedCritic`)                 |

### 1.5 Regularizers

| # | Field / property                                  | Required value                                                                              |
|---|---------------------------------------------------|----------------------------------------------------------------------------------------------|
| R19 | `latent_lam_p`                                   | `> 0` (canonical: `0.03`; allowed plan range `[0.01, 0.05]`)                                 |
| R20 | `latent_lam_h`                                   | `> 0` (canonical v5i6: `0.003 → 0.001` anneal over 0..300k)                                  |
| R21 | `latent_entropy_mode` / `latent_entropy_objective` | `"marginal"` / `"maximize"` for the canonical row. v5i4/v5i5 preserve the conditional-entropy interpretation as comparison rows. |
| R22 | `latent_kl_consecutive`                          | `0.0` (the consecutive-KL term is "optional §12"; off in the canonical row)                  |

### 1.6 Resampling cadence

| # | Field / property                                  | Required value                                                                              |
|---|---------------------------------------------------|----------------------------------------------------------------------------------------------|
| R23 | `latent_resample_every_n`                        | `0` (episode-start only) **or** `>= 32` for sparse refresh; canonical `64`                   |
| R24 | `latent_resample_on_flag`                        | `False`                                                                                      |
| R25 | `latent_event_refresh_enabled`                   | `False`                                                                                      |
| R26 | `latent_sparse_tactical_refresh_enabled`         | `False`                                                                                      |

### 1.7 Curriculum / labels / extensions

| # | Field / property                                  | Required value                                                                              |
|---|---------------------------------------------------|----------------------------------------------------------------------------------------------|
| R27 | `latent_forced_z_episode_frac`                   | `0.0`                                                                                        |
| R28 | `latent_forced_z_episode_frac_start/_end/anneal_*` | all `None` (resolver returns `0.0` at every step, pinned by `tests/test_forced_z_anneal.py`) |
| R29 | `latent_arc_credit_enabled`                      | `False`                                                                                      |
| R30 | `latent_episode_strategy_ppo`                    | `False`                                                                                      |
| R31 | `latent_router_distill_enabled`                  | `False`                                                                                      |
| R32 | `latent_v3i3_event_preference_enabled`           | `False`                                                                                      |
| R33 | `latent_preference_coef`                         | `0.0`                                                                                        |
| R34 | `latent_preference_commit_coef`                  | `0.0`                                                                                        |
| R35 | `latent_awrd_enabled`                            | `False`                                                                                      |
| R36 | `latent_specialist_router_enabled`               | `False`                                                                                      |
| R37 | `latent_marginal_balance_coef`                   | `0.0`                                                                                        |
| R38 | `latent_conditional_entropy_min_coef` and `_start` | `0.0`                                                                                      |
| R39 | `latent_context_mi_coef`                         | `0.0`                                                                                        |
| R40 | `latent_actor_z_separation_coef` and `_start_coef` | `0.0`                                                                                      |
| R41 | `latent_behavior_contrast_coef`                  | `0.0`                                                                                        |
| R42 | `latent_usage_balance_coef`                      | `0.0` for v5i6's lambda_H-driven marginal-entropy implementation; nonzero only allowed as a documented equivalent marginal-entropy implementation with conditional entropy disabled. |

---

## 2. Forbidden mechanisms (literal paper-faithful row)

If any of the following is present in the resolved configuration of a
preset that claims to be `PAPER-FAITHFUL`, the classification is invalid.

* FiLM / adapter / one-hot actor `z` conditioning (R11–R13).
* Forced-z curriculum (R27–R28).
* Episode-credit extension with dedicated AdamW (R10, R30).
* Arc-credit channel (R29).
* Preference / distillation channels (R31–R34).
* Auxiliary prediction heads (R7–R8).
* Opponent-ID actor input (structurally rejected by R16).
* Phase-ID actor input (structurally rejected; `phase` is forbidden in
  `GLOBAL_STATE_FIELD_NAMES` per `rl/custom_ppo/trainer_audit.py`).
* Global actor input (rejected by `_assert_input_contracts`).
* Hand-designed strategic rewards (`env_*` and `reward_shaping_*` deltas
  must be reviewed; if a literal-strict paper row is desired, these
  should match the no-latent baseline).
* Simultaneous conditional-entropy maximization and marginal-entropy
  balancing in the primary paper-faithful row. v5i6 uses one canonical
  entropy objective: batch-marginal entropy over expected router
  probabilities.
* Hard-coded `z`→role mapping in code or doc.
* Event-triggered hard switching (R24, R25).
* Options / hierarchical sub-policies.

---

## 3. Classification decision tree

```text
Does the preset satisfy R1..R42 (every required condition above)?
  YES:
    Does the preset enable any forbidden extension (§2)?
      NO  -> PAPER-FAITHFUL
      YES -> SUMMER-COMPATIBLE EXTENSION       (must enumerate every deviation)
  NO:
    Was the deviation deliberate, scoped, and a single-axis change
    designed to measure a specific contribution?
      YES, removes/disables a paper component (e.g. K=1, λ_h=0)         -> ABLATION
      YES, exists only to test wiring / gradient flow / collapse        -> DIAGNOSTIC
      NO, the preset is retained for reproducibility but not for new
        headline claims                                                 -> DEPRECATED
      NO, evidence is insufficient to classify the change               -> UNKNOWN
```

`UNKNOWN` presets must not be described as "Summer-faithful" or
"paper-faithful" in any document, code comment, run tag, or PR
description.

---

## 4. Naming rules

| Classification           | Allowed in preset name / aliases / run_tag                                                                                                       |
|---------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------|
| `PAPER-FAITHFUL`         | `paper_faithful`, `summer_faithful`, `plan_faithful`, `end_to_end` — only when verified by tests against the resolved configuration              |
| `SUMMER-COMPATIBLE EXTENSION` | The preset name must explicitly identify the deviation (e.g. `_film`, `_episode_credit`, `_balanced_warmup`, `_arc_credit`, `_router_distill`) |
| `ABLATION`               | Should include `_ablate_*` or `_no_*` (e.g. `phase3b_ablate_k1`, `phase3b_ablate_no_persistence`, `no_latent_v4i3_baseline`)                     |
| `DIAGNOSTIC`             | Should include a clear scope tag (e.g. `_qprobe`, `_smoke`, `_collapsed`)                                                                        |
| `DEPRECATED`             | The original name is retained for reproducibility. Mark `DEPRECATED` in the registry, never the preset name                                      |
| `UNKNOWN`                | Mark `UNKNOWN` in the registry. Do **not** add `paper_faithful` / `summer_faithful` / `plan_faithful` to its name                                |

The fact that a Python function is named
`apply_plan_faithful_latent_<X>` does **not** imply classification
`PAPER-FAITHFUL`. The repository's `plan_faithful` family includes
ablations (e.g. `apply_plan_faithful_latent_phase3b_ablate_k1`),
diagnostics, and Summer-compatible extensions.

---

## 5. Required audit banner

A paper-faithful run must emit the paper-faithful audit banner at training start
([`rl/training/banner.py::_maybe_print_paper_faithful_audit`](../rl/training/banner.py)).
The banner is fired by either of:

* `cfg.run_tag` containing a recognized paper-faithful family tag
  (`"v5i4_paper_faithful"`, `"v5i5_paper_faithful"`,
  `"v5i6_paper_faithful"`), or
* `cfg.latent_paper_faithful_audit = True` (opt-in flag for future
  paper-faithful presets).

The banner must include lines containing:

* `discrete shared z: K=<int>`
* `actor conditioning: embedding-concat`
* `FiLM: OFF`
* `q_phi task-reward PPO: ON (latent_strategy_ppo_coef=<float>)`
* `episode-credit extension: OFF`
* `forced-z curriculum: OFF (legacy_frac=<float>)`
* `auxiliary heads: OFF`
* `arc-credit: OFF`
* `preferences/distillation: OFF`
* `persistence: ON`
* `entropy maximization: ON (mode=<conditional|marginal>, objective=<str>)`
* `resampling cadence: every 64 decisions`

Banner output is pinned by
[`tests/test_v5i4_paper_faithful.py::V5i4PaperFaithfulAuditBannerTests`](../tests/test_v5i4_paper_faithful.py),
including three explicit `WARNING` lines for documented
mis-configurations:

* `latent_strategy_ppo_coef <= 0` (no task-reward gradient)
* `latent_episode_strategy_lr is set` (dedicated router AdamW silences
  the main-loop PG term)
* actor-z pathway is not concat-only (FiLM / adapter / one-hot ON)

---

## 6. Required tests for a paper-faithful preset

A new or modified `PAPER-FAITHFUL` preset must have unit / integration
tests that confirm:

1. **Correct canonical parent.** Inheritance derives from
   `apply_plan_faithful_latent_v5i4_end_to_end` or, if explicitly
   intended as a literal-strict ablation, from
   `apply_plan_faithful_latent_v5_strict_summer`.
2. **Discrete `K`.** `latent_k = 4`, unless the experiment explicitly
   locks another allowed value.
3. **Actor uses embedding concatenation.** `enable_actor_z_film == False`,
   adapter / one-hot off, actor input width matches the field formula
   in R14.
4. **Critic receives global state, joint actions, and `z`.** Critic
   `extra_dim = joint_action_onehot_dim + latent_k`.
5. **Strategy PPO coefficient is positive.** `latent_strategy_ppo_coef > 0`.
6. **Task-derived strategy advantages produce a nonzero router policy
   gradient** (cf.
   `tests/test_v5i4_paper_faithful.py::V5i4RouterTaskGradientTests::test_nonzero_advantage_produces_nonzero_qphi_gradient`).
7. **Zero advantages produce zero strategy policy loss** when only the
   strategy PPO term is exercised
   (`test_zero_advantage_produces_zero_policy_loss`).
8. **Episode-credit extension disabled.** `latent_episode_strategy_ppo
   == False`, `latent_episode_strategy_coef == 0.0`,
   `latent_episode_strategy_lr is None`.
9. **Forced-`z` fraction resolves to zero at every step** for all
   relevant `global_step` values
   (`tests/test_forced_z_anneal.py` covers the resolver; v5i4 must
   resolve to `0.0`).
10. **Preference and distillation channels disabled.** R31–R34.
11. **Auxiliary prediction heads disabled.** R7–R8.
12. **Persistence enabled within documented range.** R19.
13. **Entropy maximization enabled with correct sign and reduction.**
    Canonical v5i6 requires `latent_entropy_mode == "marginal"` and
    `latent_entropy_objective == "maximize"`; conditional-entropy rows
    must be named and documented as v5i4/v5i5 comparison rows.
14. **Resampling cadence matches the paper method.** `latent_resample_every_n
    == 64`, `latent_resample_on_flag is False`.
15. **No flag-triggered resampling.** `latent_event_refresh_enabled ==
    False` and `latent_sparse_tactical_refresh_enabled == False`.
16. **Alias resolution produces an identical resolved configuration**
    across every alias.
17. **Snapshot contains no unintended scalar drift.** Diff
    `tests/preset_snapshots.json` against the prior HEAD using set-diff;
    only intended fields may change.
18. **Launch banner reports every fidelity invariant.** The audit
    banner test (R-banner) passes.
19. **Training smoke completes an update and writes telemetry.**
    `tests/test_train_ppo_smoke.py` passes.
20. **Checkpoint save/reload preserves latent configuration.**
    `read_custom_ppo_metadata` round-trips `latent_k`, `latent_z_embed_dim`,
    and the actor / critic / `q_phi` widths.

---

## 7. Open / unresolved (must be resolved before "the paper says")

These items are inherited from
[`summer-method-spec.md`](summer-method-spec.md) §13 and are recorded
here so future agents do not silently re-interpret them.

| ID  | Item                                                                                                          | Source of disagreement                                                                              | Status   |
|-----|---------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------|----------|
| O1  | `q_phi` input dimension: 19-d global summary (older doc) vs 170-d temporal context (current code)             | `Paper_experiment_alignment.md` §3 vs `rl/global_state.py::GLOBAL_STATE_DIM = 34` and `CONTEXT_STATE_DIM = 5·34 = 170` | Code authoritative; doc paragraph in Paper_experiment_alignment.md needs an update note |
| O2  | Persistence form: hard `1[z_t ≠ z_{t-1}]` (algorithm.md) vs soft `1 − p_φ(z_t = z_{t-1} | s_t)` (loss code)   | `docs/algorithm.md` vs `rl/latent_losses.py::strategy_persistence_loss`                              | Reconciled in `Paper_experiment_alignment.md` §6.1 (soft form is gradient-bearing, hard form is diagnostic) |
| O3  | `c_Z` value (`latent_strategy_ppo_coef = 0.10`) is an implementation choice; paper doesn't lock a number      | preset code                                                                                          | Documented in v5i4 docstring; pinned by tests |
| O4  | Resample cadence (`64`) is an implementation choice; paper requires "sparse"                                  | preset code                                                                                          | Documented; pinned by tests |
| O5  | Reward shaping coefficients inherited from upstream presets (`v4i1`/`v4i3`) — not strictly "task-reward only" | `apply_plan_faithful_latent_v4i1_strategic_pressure_qprobe` and ancestors                           | Open: any nonzero `env_*` or `reward_shaping_*` field should be reviewed before publishing a "task reward only" claim |
| O6  | Backwards-compat docs in `docs/algorithm.md` and `Paper_experiment_alignment.md` predate v5i6                  | older sections describe v4i3 / v3i19 as "Summer-faithful proof" or v5i4 as the operational baseline  | v5i6 is the canonical paper-faithful row; v5i4/v5i5 are conditional-entropy comparison rows and v4i3 is `SUMMER-COMPATIBLE EXTENSION` because of arc-credit |
| O7  | `H(z)` reduction: mean conditional entropy vs batch-marginal entropy                                           | original notation did not specify whether entropy is reduced before or after averaging over states    | Resolved for this repo: v5i6 is canonical and uses `latent_entropy_mode = "marginal"`; v5i4/v5i5 remain conditional-entropy comparison rows |

---

## 8. Proposed Preset Review template

Future agents must complete this template (in the PR description, the
commit message, or a top-of-file docstring on the new preset) **before**
the preset is implemented or merged.

```markdown
## Proposed Preset Review

### Identity
- Proposed name:
- Parent preset:
- Classification (one of PAPER-FAITHFUL / SUMMER-COMPATIBLE EXTENSION /
  ABLATION / DIAGNOSTIC / DEPRECATED / UNKNOWN):
- Research question (one sentence; no marketing adjectives):

### Intended delta
- Field(s) changed:
- Why this change is necessary:
- Why an existing preset cannot answer the question:

### Fidelity impact
- Actor architecture changed: YES / NO
- Router objective changed: YES / NO
- Exploration schedule changed: YES / NO
- Reward changed: YES / NO
- Supervision added: YES / NO
- Auxiliary task added: YES / NO
- Resampling changed: YES / NO

### Exact deviations from the paper-faithful preset
- None, or list each deviation explicitly with field name, old value,
  new value, and rationale:
  - <field>: <old> -> <new>; reason

### Required evidence
- Metrics (which CSV fields will demonstrate the effect):
- Evaluations (which protocol from `experiment-and-evaluation-protocol.md`):
- Baselines (which presets / random-matched / no-latent rows):
- Stopping criteria (when the run is "decisively done"):

### Required repository updates
- Preset registry entry in `rl/presets/__init__.py::PRESET_REGISTRY`:
- Classification entry in `latent-preset-registry.md`:
- Fidelity matrix line in `latent-preset-registry.md`:
- Tests added/updated:
- Snapshot regenerated (yes/no, with explicit diff):
- Audit banner update (if a new paper-faithful preset):
- Progress tracker entry in `research-progress-tracker.md`:
```

---

## 9. Single-Change Experiment Rule (required header for every preset)

A main ablation should alter one scientifically meaningful mechanism
whenever possible. Every preset's docstring must include a delta table
of the following shape, comparing against the **canonical paper-faithful
parent** (or the explicit non-paper parent named in the preset):

| Dimension          | Parent           | New preset | Intended change? |
|--------------------|------------------|------------|------------------|
| Actor conditioning | concat           | concat     | no               |
| Router task PPO    | on (`c_Z = 0.10`)| on / off   | yes / no         |
| Episode credit     | off              | off        | no               |
| Arc credit         | off              | off        | no               |
| FiLM               | off              | off        | no               |
| Forced-z schedule  | off              | off        | no               |
| Persistence        | 0.03             | same       | no               |
| Entropy            | 0.003 → 0.0002   | same       | no               |
| Resampling         | every 64         | same       | no               |
| Aux heads          | off              | off        | no               |
| Preferences        | off              | off        | no               |
| Distillation       | off              | off        | no               |

If multiple mechanisms change in a single preset, the preset must be
classified as a compound `SUMMER-COMPATIBLE EXTENSION` and the docstring
must explain why a single-change experiment was impractical (e.g.
v5i2 is "v5i1 + FiLM"; v5i3 is "v5i2 + forced-z anneal").

---

## 10. Backward compatibility

A change that touches the canonical paper-faithful preset, the actor /
critic / `q_phi` architecture, the strategy losses, or the audit banner
must:

* Keep all existing pinning tests green.
* Preserve checkpoint load semantics
  (`rl/custom_ppo/inference.py::read_custom_ppo_metadata`,
  legacy state-dict remapping in
  `rl/custom_ppo/policy.py`).
* Preserve CSV schemas (`rl/custom_ppo/csv_writers.py`,
  `rl/custom_ppo/training_telemetry.py`).
* Preserve the alias set in `rl/presets/__init__.py::PRESET_REGISTRY`.

If a breaking change is necessary, document it in
[`Paper_experiment_alignment.md`](Paper_experiment_alignment.md) §7
"Changelog" with the affected file paths and the migration path.
