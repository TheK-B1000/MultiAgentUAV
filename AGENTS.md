# AGENTS.md — Repository instructions for AI coding agents

This file is the **mandatory entry point** for any AI coding agent working in
the `MultiAgentUAV` / `AICTFProject` repository on the latent-strategy
multi-agent CTF system. Read it in full before reading anything else.

This file does not redefine the algorithm. It points at the documents that
do, and lists the rules every agent must follow before changing anything.

---

## 1. Project summary

The repository implements multi-agent reinforcement learning for an
adversarial aquatic capture-the-flag environment. The research centers on
a shared discrete latent team-strategy variable `z ∈ {1, …, K}` (typically
`K = 4`), inferred centrally during training by `q_phi(z | s_t)` and used
to condition decentralized per-agent policies `π_θ(a_i | o_i, z)`.

Training uses a custom local PPO/MAPPO-style trainer
(`rl/custom_ppo/`); the centralized critic is `V_φ(s, a, z)` (scalar value,
**not** a Q-function over counterfactual joint actions).

The default executable entry point is `rl/train_ppo.py`. Presets are
resolved through `rl/presets/__init__.py::PRESET_REGISTRY`.

---

## 2. Required reading order

Read these documents **in this order** before touching any latent-related
code, preset, training preset, evaluation script, or paper claim:

1. `AGENTS.md` (this file).
2. `AICTFProject/docs/summer-method-spec.md` — owns the scientific
   definition of the literal paper-faithful method.
3. `AICTFProject/docs/summer-fidelity-rules.md` — auditable checklist and
   classification decision tree; includes the **Proposed Preset Review
   template** future agents must complete before implementation.
4. `AICTFProject/docs/latent-preset-registry.md` — authoritative registry
   of every latent preset, every alias, every classification, every
   resolved-config delta against the paper-faithful baseline.
5. `AICTFProject/docs/experiment-and-evaluation-protocol.md` — how
   experiments must be launched, compared, and interpreted.
6. `AICTFProject/docs/research-progress-tracker.md` — current state of
   runs, open decisions, recommended next experiments.

The legacy paper-alignment document
`AICTFProject/docs/Paper_experiment_alignment.md` is also authoritative
for code↔manuscript trace and remains the source of truth for ratings of
each preset's `q_phi` gradient channels (§6 ladder, §6.7 v5i4 contract).

The literal Summer plan implementation trace is
`docs/Summer_Implementation_Plan_Implementation_Details_Trace.md` and the
algorithm sketch is `docs/algorithm.md`.

If any of those documents disagrees with `summer-method-spec.md`, the
method spec wins for fidelity classifications and the discrepancy must be
recorded in `summer-fidelity-rules.md` until resolved.

---

## 3. Canonical paper-faithful preset

The canonical operational paper-faithful preset is verified in this
repository as:

```text
function: apply_plan_faithful_latent_v5i4_end_to_end
file:     rl/presets/plan_faithful.py
run_tag:  v5i4_paper_faithful_end_to_end_OP5_OP6_OP7_1m_4v4
aliases:  v5i4
          v5i4_paper_faithful
          v5i4_end_to_end
          paper_faithful_end_to_end
          latent_v5i4_paper_faithful
          latent_v5i4_end_to_end
          plan_faithful_latent_v5i4_end_to_end
```

All seven aliases resolve to the same `PPOConfig`. This invariant is
pinned by `tests/test_v5i4_paper_faithful.py::V5i4AliasSnapshotTests`.

**Single-axis paper-faithful follow-up (v5i5).** The first paper-faithful
follow-up to v5i4 is `v5i5_paper_faithful_entropy_floor`, registered as:

```text
function: apply_plan_faithful_latent_v5i5_paper_faithful_entropy_floor
file:     rl/presets/plan_faithful.py
run_tag:  v5i5_paper_faithful_entropy_floor_OP5_OP6_OP7_1m_4v4
aliases:  v5i5
          v5i5_paper_faithful
          v5i5_paper_faithful_entropy_floor
          v5i5_entropy_floor
          paper_faithful_entropy_floor
          latent_v5i5_paper_faithful
          latent_v5i5_paper_faithful_entropy_floor
          latent_v5i5_entropy_floor
          plan_faithful_latent_v5i5_paper_faithful_entropy_floor
          plan_faithful_latent_v5i5_entropy_floor
```

The resolved-config diff between v5i4 and v5i5 is **exactly two keys**:
`latent_lam_h_end` (raised from `0.0002` to `0.001`) and `run_tag`. This
single-axis property is pinned by
`tests/test_v5i5_paper_faithful_entropy_floor.py::V5i5PresetInheritanceTests::test_v5i5_minimal_diff_vs_v5i4`.
The new entropy floor stays inside the documented Summer-plan
`[0.001, 0.01]` range, so v5i5 is also `PAPER-FAITHFUL`. The
launch-time audit banner is family-aware and prints
`[PPO] v5i5 paper-faithful audit:` for v5i5 runs (see
`rl/training/banner.py`).

**Canonical paper-faithful row (v5i6).** The current canonical
operational paper-faithful preset is
`v5i6_paper_faithful_marginal_entropy`, registered as:

```text
function: apply_plan_faithful_latent_v5i6_paper_faithful_marginal_entropy
file:     rl/presets/plan_faithful.py
run_tag:  v5i6_paper_faithful_marginal_entropy_OP5_OP6_OP7_1m_4v4
aliases:  v5i6
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

v5i6's resolved-config diff against v5i4 is exactly **three keys**
(`latent_entropy_mode`, `latent_lam_h_end`, `run_tag`) and against v5i5
exactly **two keys** (`latent_entropy_mode`, `run_tag`). The single
scientific change versus v5i5 is the entropy interpretation: mean
conditional entropy `E_s[H(q_phi(z|s))]` becomes **rollout-level**
marginal entropy `H(E_s[q_phi(z|s)])`.

> **Aggregation contract — read before touching v5i6's loss path.**
> The marginal entropy loss MUST be aggregated over the **entire**
> rollout resample subset (~1024 states for the standard rollout),
> not per-PPO-minibatch. KL is convex in `q_bar` so the per-minibatch
> aggregation is a strict upper bound on the intended rollout-level
> objective by Jensen, and the bias is closed by the gradient softening
> individual `q_phi(z|s)` toward uniform — the conditional regression
> v5i6 was meant to replace. Implementation lives in
> `rl/latent_losses.py::rollout_marginal_entropy_loss` (called once
> per PPO inner epoch from `rl/custom_ppo/ppo_updater.py`); the
> deprecated per-minibatch helper `strategy_marginal_entropy_loss` is
> retained ONLY for parity tests and must NOT be wired into a v5i6
> production path. Pinned by
> `tests/test_latent_losses.py::RolloutMarginalEntropyLossTests` (Jensen
> demo) and
> `tests/test_v5i6_paper_faithful_marginal_entropy.py::V5i6RolloutMarginalEntropyContractTests`.
> See `docs/summer-method-spec.md` §8.1 for the rationale.

**Run-tag history (`_2m_` → `_1m_`):** v5_strict_summer / v5i1 / v5i2 /
v5i3 inherited a misleading `_2m_4v4` suffix from v4i1 without ever
overriding `total_timesteps` from its `PPOConfig` default of
`1_000_000`. v5i4's preset now sets the suffix to `_1m_4v4` so the tag
agrees with the actual budget; pinned by
`tests/test_v5i4_paper_faithful.py::V5i4RunTagAndInitialOpponentConsistencyTests::test_run_tag_advertises_actual_total_timesteps_budget`.
Checkpoints, metrics CSVs, and lock files produced by v5i4 launches
*before* the tag fix carry the old `_2m_4v4` suffix on disk (see
`AICTFProject/checkpoints/4v4/ckpt_v5i4_paper_faithful_end_to_end_OP5_OP6_OP7_2m_4v4_*.zip`).
Those artifacts are valid evaluation inputs for the completed 1 M-step
paper-faithful run; only the filename string disagrees with the trainer's
reported budget. Do not retroactively rename — embedded `run_tag`
metadata in the CSV / checkpoint files would break tag-based CSV
consumers. See
`AICTFProject/docs/latent-preset-registry.md` §7 for the full
artifact-history table.

`v4i3_summer_proof` is **not** the operational paper-faithful preset for
new headline claims: it inherits the v3i19 arc-credit channel, which is
documented as a post-Summer extension (see
`AICTFProject/docs/Paper_experiment_alignment.md` §6.2). Use it as the
arc-credit row of the proof table, not as the literal paper-faithful row.

---

## 4. Mandatory rule — what "Summer-faithful" means

> **When a user asks for a Summer-faithful (or paper-faithful, plan-faithful)
> implementation, modify or inherit from the canonical paper-faithful preset
> only.** Do not inherit from FiLM, episode-credit, forced-z, preference,
> auxiliary, or curriculum presets unless the user explicitly requests an
> extension.

Specifically:

* The canonical inheritance parent for new paper-faithful work is
  `apply_plan_faithful_latent_v5i4_end_to_end` (or, if a user explicitly
  asks for a literal-strict ablation, `apply_plan_faithful_latent_v5_strict_summer`).
* It is forbidden to inherit silently from `v5i1` (episode credit), `v5i2`
  (FiLM), `v5i3` (forced-z), `v3i19`/`v4i3` (arc credit), `v3iN preference
  / AWRD / specialist` chains, or `v4i4post` (router distillation) when the
  user has asked for the paper-faithful method.
* It is forbidden to label any preset `paper_faithful`, `summer_faithful`,
  or `plan_faithful` unless every required condition in
  `summer-fidelity-rules.md` passes against the resolved configuration.

---

## 5. Mandatory rule — fix first, fork second

> Before proposing a new version (`vNext`, `improved`, `stronger`, `fixed`,
> etc.), first determine whether the requested behavior can be achieved by
> fixing the canonical implementation rather than creating another preset.

A new preset is only justified when there is a documented experimental
question that an existing preset cannot answer. Document the question
using the **Proposed Preset Review template** in `summer-fidelity-rules.md`
*before* changing code.

---

## 6. Mandatory pre-change checklist

For any change to:

* a latent preset (`rl/presets/`),
* `PPOConfig` (`rl/config/ppo_config.py`),
* the actor (`rl/networks.py::LatentConditionedActor`,
  `rl/custom_ppo/policy.py::SharedActorCentralizedCritic`),
* the centralized critic (`rl/networks.py::CentralizedCritic`),
* the strategy encoder (`rl/networks.py::StrategyEncoder`,
  `rl/latent_marl.py`),
* the strategy losses (`rl/latent_losses.py`),
* the PPO update (`rl/custom_ppo/ppo_updater.py`),
* the latent strategy state machine (`rl/custom_ppo/latent_strategy_state.py`),
* the audit banner (`rl/training/banner.py`),
* an evaluation script (`plot/eval_checkpoint.py`,
  `plot/eval_rollout.py`, `tools/q_probe*.py`,
  `tools/summer_proof_report.py`),

an agent **must** complete the following before submitting a change:

1. Read the six documents listed in §2.
2. State the intended scientific delta in plain English (no marketing
   adjectives — no "improved," "stronger," "smarter").
3. Identify the canonical parent preset / module being modified.
4. Classify the proposed change against
   `summer-fidelity-rules.md` (`PAPER-FAITHFUL`, `SUMMER-COMPATIBLE
   EXTENSION`, `ABLATION`, `DIAGNOSTIC`, `DEPRECATED`, `UNKNOWN`).
5. Produce a **resolved configuration diff** vs the parent (use
   `dataclasses.asdict(apply_preset(PPOConfig(), name))` and diff field
   by field, not from preset names).
6. Verify no unintended fields changed.
7. Confirm or update tests (see §7).
8. Update `latent-preset-registry.md` if a preset was added or its
   classification changed.
9. Update `research-progress-tracker.md` if a run was launched, finished,
   or its status changed.
10. Update `summer-method-spec.md` only if the locked paper specification
    or the canonical preset itself was intentionally revised.

---

## 7. Required tests

Any change to the canonical paper-faithful preset, to the actor /
critic / `q_phi` architecture, or to the strategy losses must keep the
existing pinning tests green. The relevant ones are:

| Test file                                            | What it pins |
|------------------------------------------------------|--------------|
| `tests/test_v5i4_paper_faithful.py`                  | v5i4 inheritance, concat-only actor, no curriculum, router task gradient ON, zero-advantage = zero policy_loss, no forbidden channels, sparse 64-step resampling, alias snapshot equality, audit banner content. |
| `tests/test_preset_resolution.py`                    | Preset registry alias resolution. |
| `tests/test_marginal_baseline.py`                    | Main-loop gating semantics (the dedicated-optimizer fix that made `lam_p`/`lam_h` actually flow into `q_phi`). |
| `tests/test_forced_z_anneal.py`                      | Forced-z resolver and resume safety (only meaningful for v5i3-class presets; v5i4 must resolve to 0 at every step). |
| `tests/test_latent_losses.py`                        | Pure-tensor checks on `strategy_ppo_loss`, `strategy_persistence_loss`, `strategy_entropy_loss`, `strategy_kl_consecutive_loss`, `strategy_aux_return_loss`, `strategy_marginal_entropy_loss` (deprecated per-minibatch path; parity-only), `rollout_marginal_entropy_loss` (canonical v5i6 path), `rollout_router_soft_diagnostics`, plus the Jensen demo (`RolloutMarginalEntropyLossTests::test_jensen_demo_per_minibatch_upper_bounds_rollout_level`). |
| `tests/test_v5i5_paper_faithful_entropy_floor.py`    | v5i5 inheritance, single-axis diff vs v5i4 (`latent_lam_h_end` + `run_tag` only), audit banner family detection, sampled-z occupancy diagnostic schema. |
| `tests/test_v5i6_paper_faithful_marginal_entropy.py` | v5i6 inheritance, single-axis diffs (vs v5i4 = 3 keys, vs v5i5 = 2 keys), `latent_entropy_mode = "marginal"` contract, audit banner reports `aggregation=rollout`, rollout-level marginal-entropy correctness on perfect-specialization and collapse cases, `router_rollout_soft_*` CSV schema. |
| `tests/test_latent_strategy_alignment.py`            | Spec-trace alignment of the latent stack. |
| `tests/test_train_ppo_smoke.py`                      | One-update training smoke. |
| `tests/test_audit_regressions.py`                    | Trainer audit banner shape / content. |

Snapshot tests live in `tests/preset_snapshots.json`. Any preset change
that touches scalar fields must regenerate the snapshot via
`tools/snapshot_presets.py`; document the diff explicitly.

If a paper-faithful preset is being created or changed, add tests
analogous to the items in `summer-fidelity-rules.md` §"Required tests."

---

## 8. Forbidden patterns

Do **not**:

1. Add a `q_phi` gradient channel that is not one of:
   * `−λ_H · H(q_phi(z | s))` (entropy maximization, paper),
   * `λ_p · L_persist` (persistence regularizer, paper, soft form
     `1 − p_φ(z_t = z_{t-1} | s_t)`),
   * `c_Z · L_strategy_PPO` (the on-policy categorical PPO term in
     `rl/latent_losses.py::strategy_ppo_loss`, paper-faithful).
2. Re-introduce `latent_strategy_aux_predict_phase_coef`,
   `latent_strategy_aux_return_head`, `latent_v3i3_event_preference_*`,
   `latent_router_distill_*`, `latent_preference_*`, `latent_awrd_*`,
   `latent_specialist_*`, or `latent_behavior_contrast_*` into a
   PAPER-FAITHFUL preset.
3. Pass `global_state`, `phase_id`, `opponent_id`, or any centralized
   feature into the actor pathway. (`rl/custom_ppo/policy.py::_assert_input_contracts`
   already enforces this; do not weaken the assertion.)
4. Set `latent_episode_strategy_lr` on a paper-faithful preset. The
   dedicated router optimizer silences the main-loop categorical PPO
   term (this is the documented mis-configuration the v5i4 audit warns
   about).
5. Rename `q_phi`, the strategy embedding, the entropy/persistence sign,
   or the categorical PPO loss without updating the audit banner, the
   pinning tests, **and** all six docs in §2.
6. Use the words "Summer-faithful," "paper-faithful," or "plan-faithful"
   as marketing labels in code or documentation. Use them only when the
   classification is verified by tests against the resolved
   configuration.
7. Interpret natural per-`z` behavior differences from rollout telemetry
   as causal evidence. Causal claims require forced-`z` matched-seed
   evaluations; see `experiment-and-evaluation-protocol.md`.
8. Interpret 0.0 or 1.0 win rates without sample counts as "decisive."
9. Modify `tests/preset_snapshots.json` without an explicit changelog
   entry in `AICTFProject/docs/Paper_experiment_alignment.md` §7.

---

## 9. Handling uncertainty

If the repository's evidence is insufficient to classify a preset, an
ancestry, an objective, or a behavior:

* Record the item under `summer-fidelity-rules.md` "Open / unresolved"
  with the specific file, line, and missing evidence.
* Mark the preset's classification as `UNKNOWN` in
  `latent-preset-registry.md`.
* Do **not** silently assume the literal paper interpretation. Do **not**
  apply a `paper_faithful` label. Ask the user.

---

## 10. Pre-change checklist (compact)

Before opening a PR that touches presets, training, or evaluation:

* [ ] Read all six docs in §2.
* [ ] Stated the scientific delta in plain English.
* [ ] Identified the canonical parent preset.
* [ ] Classified the change.
* [ ] Produced a resolved-config diff against the parent.
* [ ] Confirmed no unintended fields changed.
* [ ] Tests added/updated.
* [ ] Snapshot regenerated only if intentional.
* [ ] Registry / progress tracker updated.
* [ ] Run command, seed, opponent pool, total timesteps, agent count
      preserved across the comparison the change is meant to support.

## 11. Post-change checklist (compact)

After your change is in:

* [ ] Audit banner output reflects the new state.
* [ ] All `tests/test_v5i4_paper_faithful.py` cases still pass.
* [ ] CSV writers, evaluation tools, and inference loaders all still
      load existing checkpoints (backward compatibility).
* [ ] Any documentation paragraph that became outdated has been edited
      in the **owning** doc only — not duplicated.

---

## 12. Information ownership (avoid duplication)

Each topic has exactly one owning document. If you need to write more
than one paragraph about it elsewhere, link to the owner instead.

| Topic                                            | Owning document |
|--------------------------------------------------|-----------------|
| Scientific definition of the paper method        | `summer-method-spec.md` |
| Fidelity rules / classification / proposal form  | `summer-fidelity-rules.md` |
| Per-preset facts and aliases                     | `latent-preset-registry.md` |
| Evaluation, statistical protections, comparisons | `experiment-and-evaluation-protocol.md` |
| Current run status / next experiment             | `research-progress-tracker.md` |
| Mandatory agent behavior                         | `AGENTS.md` (this file) |
| Code↔manuscript trace                            | `AICTFProject/docs/Paper_experiment_alignment.md` |
| Spec-to-code trace                               | `docs/Summer_Implementation_Plan_Implementation_Details_Trace.md` |

---

## 13. Out-of-scope reminders

This file's purpose is process discipline, not algorithm design. Do not
edit `AGENTS.md` to redefine the method, to push a preferred experiment,
or to relax a fidelity rule. If a fidelity rule is wrong, propose the
change in `summer-method-spec.md` and `summer-fidelity-rules.md`, with
references to repository evidence.
