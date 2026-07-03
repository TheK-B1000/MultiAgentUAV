# Research Progress Tracker

**Owner:** This file is the single source of truth for *current* run
status, *open* decisions, and *recommended next* experiments. It is the
working logbook of the research effort; it is updated when a run
launches, finishes, fails, or its interpretation changes.

It is **not** the source of truth for:

* The scientific method definition →
  [`summer-method-spec.md`](summer-method-spec.md).
* Fidelity rules, classification, or proposal templates →
  [`summer-fidelity-rules.md`](summer-fidelity-rules.md).
* Per-preset facts, aliases, or resolved deltas →
  [`latent-preset-registry.md`](latent-preset-registry.md).
* Launch / eval / statistical protocols →
  [`experiment-and-evaluation-protocol.md`](experiment-and-evaluation-protocol.md).

> **Last updated:** 2026-07-03 (UTC-4)

---

## 1. Run-status legend

| Tag         | Meaning                                                                                                  |
|-------------|----------------------------------------------------------------------------------------------------------|
| `RUNNING`   | Active training job; check the terminals folder for the live tail.                                       |
| `COMPLETED` | Training finished at the configured budget; checkpoints + CSVs are on disk; eval may or may not be done. |
| `EVALUATED` | Training finished and the §4 / §5 protocol in `experiment-and-evaluation-protocol.md` has been run.       |
| `FAILED`    | Training did not reach the budget, or completed but a §7 trainer-side invariant was violated.            |
| `PLANNED`   | Preset / launch designed but not started; the *Proposed Preset Review* template must be on file.         |
| `DEFERRED`  | Designed but explicitly paused pending a decision from a parent comparison.                              |

A row is "ready to be cited in a paper claim" only when status is
`EVALUATED` and the §9 checklist in
[`experiment-and-evaluation-protocol.md`](experiment-and-evaluation-protocol.md)
is satisfied.

---

## 2. Current run status (v4 / v5 ladder, 4v4, OP5/OP6/OP7)

All rows use the §1 invariants of
[`experiment-and-evaluation-protocol.md`](experiment-and-evaluation-protocol.md)
unless otherwise noted (4v4, OP5/OP6/OP7 uniform, 1 M steps, `n_envs=32`,
`n_epochs=6`, `--seed 0`, `--device cuda`).

### 2.1 v5i4 (conditional-entropy paper-faithful interpretation) — `COMPLETED`

| Property                                | Value                                                                                                 |
|-----------------------------------------|-------------------------------------------------------------------------------------------------------|
| Preset                                  | `v5i4_paper_faithful` (conditional-entropy paper-faithful interpretation)                             |
| Classification                          | `PAPER-FAITHFUL`                                                                                       |
| Status                                  | `COMPLETED` — 1,000,000 / 1,000,000 decision steps (2 h 05 m wall).                                   |
| Run tag on disk (artifact filename)     | `v5i4_paper_faithful_end_to_end_OP5_OP6_OP7_2m_4v4`                                                   |
| Post-fix preset run tag                 | `v5i4_paper_faithful_end_to_end_OP5_OP6_OP7_1m_4v4`                                                   |
| Discrepancy                             | Run was launched before the `_2m_` → `_1m_` tag fix; artifacts retain `_2m_`. See [`latent-preset-registry.md`](latent-preset-registry.md) §7.1. |
| Final checkpoint                        | `AICTFProject/checkpoints/4v4/final_v5i4_paper_faithful_end_to_end_OP5_OP6_OP7_2m_4v4.zip`            |
| Periodic checkpoints                    | 50k stride → `ckpt_..._50000.zip` … `ckpt_..._1000000.zip`                                            |
| Final rollout WR (training)             | `64.4 %` (W=6442, L=3403, D=155, n=10 000)                                                            |
| Final ev (explained variance)           | `0.605`                                                                                                |
| Final `zH` (q_phi entropy on resample)  | `0.642` (vs `ln(K) = 1.386`); end-of-run `H/ln(K) ≈ 0.46`                                              |
| Final `z_occ`                           | `[0.092, 0.137, 0.706, 0.065]` — **z=2 dominant**                                                      |
| Final per-`z` WR (rollout, non-causal)  | `[0.677, 0.712, 0.685, 0.696]`                                                                         |
| Final `MI_z_outcome` (E3)               | `0.0041`                                                                                               |
| Final `actor_z_jsd` (sep_JSD)           | `8.33e-4` (max_JSD `0.068`)                                                                            |
| Final `actor_input_dim`                 | `164` (matches paper-faithful R14 in `summer-fidelity-rules.md`)                                       |
| Eval status                             | `PENDING` — `plot/eval_checkpoint.py` matrix has not yet been run.                                    |
| Decisive comparison rows still needed   | `no_latent_v4i3_baseline` (same-everything-except-z, **`COMPLETED`** for v4i3 budget — needs 1M run at v5i4 budget); `v5_strict_summer` at same seed; `v5i4 random-matched` eval-time row. |

**Observations from rollout telemetry only (not causal):**

* The router collapsed to a z=2-dominant occupancy by 1 M, even with
  the on-policy categorical PPO term ON. This is the same failure mode
  v5i2 / v5i3 were designed to repair (v5i3 added a forced-z anneal as
  a *coverage* fix on top of v5i2; v5i4 deliberately omits any
  curriculum to stay literal-paper-faithful).
* `zH` decayed from ~1.0 by step ~720 k to `0.64` at 1 M; the entropy
  anneal floor (`latent_lam_h_end = 0.0002`) was reached well before
  the collapse stabilized.
* Per-`z` WR spread is modest (`max − min ≈ 0.035`) but the per-`z`
  values are **not** causal evidence — see
  [`AGENTS.md`](../../AGENTS.md) §8.7.
* MI(`z`; outcome) is `O(10^-3)` — informative as a floor only.

**Next actions for v5i4:**

1. Run the eval matrix (`experiment-and-evaluation-protocol.md` §4)
   on the final checkpoint at `--episodes 300`, `--map-sets train eval`,
   `--opponents OP5_RUSHER OP6 OP7`.
2. Run the matched-schedule routing-quality control:
   `--latent-selection router` vs `--latent-selection random-matched`
   with the same `--seed`, same checkpoint, same `--episodes`.
3. Run `tools/q_probe.py` (forced-z return contrast) and
   `tools/q_probe_local_counterfactual.py` (local Q-contrast) on the
   final checkpoint.
4. Assemble the per-checkpoint Markdown report via
   `tools/summer_proof_report.py`.
5. Compare to `no_latent_v4i3_baseline` (a re-launched 1 M-step
   baseline at the same seed; the existing v4i3 baseline is at the
   v4i3 budget).
6. Use this row as the conditional-entropy reference for v5i6 comparisons
   before adding more v5i4 seeds.

### 2.2 v5_strict_summer (literal-Summer ablation) — `COMPLETED` (earlier)

| Property                                | Value                                                                                                 |
|-----------------------------------------|-------------------------------------------------------------------------------------------------------|
| Preset                                  | `v5_strict_summer`                                                                                     |
| Classification                          | `ABLATION` (literal `docs/algorithm.md` loss; no task-reward PG channel on `q_phi`)                   |
| Status                                  | `COMPLETED` (earlier; pre-dates the v5i4 launch).                                                     |
| Result interpretation                   | The literal equation alone (persistence + entropy on `q_phi`, no `L_strategy_PPO`) collapsed `q_phi` to a single z; rollout WR did not exceed the no-latent control by a paired-bootstrap-significant margin. This is the result that *motivated* v5i4 (adding the on-policy categorical PPO term). |
| Run tag on disk                         | `v5_strict_summer_OP5_OP6_OP7_2m_4v4` (legacy `_2m_` suffix; see §7.2 of the registry).                |
| Decisive comparison done                | v5_strict_summer vs `no_latent_v4i3_baseline`: not paired-bootstrap-significant. v5_strict_summer vs v5i4: pending (the v5i4 row above).                                                                                                                                                       |

### 2.3 v5i1_reward_credit_router (per-episode router PPO + dedicated AdamW) — `COMPLETED`

| Property                                | Value                                                                                                 |
|-----------------------------------------|-------------------------------------------------------------------------------------------------------|
| Preset                                  | `v5i1_reward_credit_router`                                                                            |
| Classification                          | `SUMMER-COMPATIBLE EXTENSION`                                                                          |
| Status                                  | `COMPLETED`.                                                                                          |
| Result interpretation                   | The dedicated AdamW + per-episode router PPO repaired the router collapse only partially; FiLM-less actor coverage remained the bottleneck. Motivated v5i2 (add FiLM). |

### 2.4 v5i2_stronger_z_conditioning (v5i1 + FiLM) — `COMPLETED`

| Property                                | Value                                                                                                 |
|-----------------------------------------|-------------------------------------------------------------------------------------------------------|
| Preset                                  | `v5i2_stronger_z_conditioning`                                                                         |
| Classification                          | `SUMMER-COMPATIBLE EXTENSION` (adds R11 FiLM)                                                          |
| Status                                  | `COMPLETED`; per-checkpoint files on disk through `_550000.zip`.                                       |
| Final-checkpoint observation            | `q_phi` collapsed to z=2 dominant within the first 200 k steps; z=1 reached `<5 %` occupancy by 540 k. The actor's per-`z` sensitivity grew steadily under FiLM, but only on the `z` values the router actually picked. |
| Result interpretation                   | Coverage problem (not credit-assignment); motivated v5i3 forced-z anneal.                              |

### 2.5 v5i3_balanced_warmup (v5i2 + forced-z anneal) — `COMPLETED` (partial)

| Property                                | Value                                                                                                 |
|-----------------------------------------|-------------------------------------------------------------------------------------------------------|
| Preset                                  | `v5i3_balanced_warmup`                                                                                 |
| Classification                          | `SUMMER-COMPATIBLE EXTENSION` (adds R27/R28 forced-z anneal on top of v5i2)                            |
| Status                                  | `COMPLETED` partial budget; on-disk checkpoints reach `_100000.zip` then stop (run was rotated to v5i4 once the v5i4 design was finalized). |
| Result interpretation                   | Not yet sufficient runtime to compare against v5i4; if the v5i4 forced-z-free row collapses, v5i3 is the established compound-extension fix to compare against. |

### 2.6 v4i3_summer_proof (arc-credit row) — `COMPLETED`

| Property                                | Value                                                                                                 |
|-----------------------------------------|-------------------------------------------------------------------------------------------------------|
| Preset                                  | `v4i3_summer_proof`                                                                                    |
| Classification                          | `SUMMER-COMPATIBLE EXTENSION` (carries v3i19 arc-credit; **not** literal paper-faithful — see [`AGENTS.md`](../../AGENTS.md) §3) |
| Status                                  | `COMPLETED` at `_1000000.zip` (and full periodic stride from `_50000.zip`).                            |
| Run tag on disk                         | `v4i3_summer_proof_OP5_OP6_OP7_2m_4v4` (budget-agnostic in current preset; on-disk has the historical `_2m_` suffix). |
| Role in the proof table                 | The *arc-credit row*. Not a substitute for v5i4.                                                       |

### 2.7 no_latent_v4i3_baseline (same-everything-except-z control) — `COMPLETED`

| Property                                | Value                                                                                                 |
|-----------------------------------------|-------------------------------------------------------------------------------------------------------|
| Preset                                  | `no_latent_v4i3_baseline`                                                                              |
| Classification                          | `DIAGNOSTIC` (matched no-latent control)                                                               |
| Status                                  | `COMPLETED` at the v4i3 budget; **not yet re-launched at v5i4's `--total-steps 1000000`** with `--seed 0` for the headline v5i4 comparison. |
| Required next                           | Re-launch at v5i4's exact budget and seed for the headline `v5i4` vs `no_latent` comparison.            |

---

## 3. Planned / proposed experiments

> Every row in this section must be backed by a completed *Proposed
> Preset Review* template (see
> [`summer-fidelity-rules.md`](summer-fidelity-rules.md) §8). PLANNED
> rows that have not yet had the template filed are explicitly labeled
> as such.

### 3.0 v6i9 feedforward running-mean arc-credit A/B (IMPLEMENTED, PENDING_SMOKE)

**Status:** `IMPLEMENTED, PENDING_SMOKE`. Preset committed as
`v6i9_arc_credit_running_mean_feedforward_hardpool` (aliases
`v6i9_arc_credit_feedforward`,
`plan_faithful_latent_v6i9_arc_credit_running_mean_feedforward_hardpool`),
`SUMMER-COMPATIBLE EXTENSION` (arc-credit row), parent
`v6i9_mapaware_router_feedforward_hardpool`.

**Motivation (credit audit, 2026-07-02):** the feedforward router control
routes q_phi credit through the main-loop strategy-PPO term on
`router_advantages = router_return − V_critic`. The critic overestimates
V(s_0) by ~+2.71 and, because most single-decision chunks skip the
`numel>1` advantage normalization, that constant bias survives into the
loss (chronically negative advantages, ~41% sign flips). The treatment
**replaces** that channel: `latent_strategy_ppo_coef` 0.1→0.0 (magnet
removed) plus arc credit with a detached running-mean EMA baseline that
auto-centers advantages. Resolved diff vs control = exactly 4 keys
(pinned by `tests/test_v6i9_arc_credit_feedforward.py`).

**Next actions (do NOT scale until these pass):**
1. One-update treatment smoke from the repertoire anchor with a fresh
   optimizer: `experiments/run_arc_credit_treatment_smoke.py`. Gates:
   arc-credit source active, baseline=running_mean, valid router
   decisions > 0, raw arc advantage finite, frozen actor/z hashes
   unchanged, router gradients > 0.
2. Three-update A/B mechanism test (control vs treatment, identical
   seed/budget): look for fewer all-negative decision batches, positive
   fraction moving toward balance, context-conditioned logit variation,
   argmax no longer always z1, MI(z;context) above noise.
3. Aligned episode-persistent credit audit: require Q1 negative / Q4
   positive credit and near-zero critic/global bias shift.
4. Cheap corrected ablation (learned / exact-histogram shuffled /
   uniform / fixed z2); proceed only when learned > shuffled.

### 3.0.1 v6i9 feedforward credit-patch mechanism + continuation — `EVALUATED` (context-blind collapse)

**Status:** `EVALUATED`. This is the *minimal credit patch* on the control
preset `v6i9_mapaware_router_feedforward_hardpool` (feedforward router,
35-dim team-geometry context, K=4, frozen repertoire) — **not** the
§3.0 arc-credit A/B. The patch makes the feedforward strategy-PPO update
consume `router_advantages` (erroring if absent) and applies conditional
router entropy only at decision steps (`router_ent_coef`, kept separate
from `latent_lam_h`). No context change: **credit + entropy changed,
context did not.**

**Runs (2v2, opponent pool OP8/OP9/OP10, seed 1, frozen repertoire):**

1. Wiring smoke (1 update) — all gates green (router_advantages selected,
   decision mask active, entropy subordinate, frozen repertoire hashes
   unchanged).
2. Mechanism run — 3 updates from the repertoire anchor with a fresh
   router optimizer (`final_v6i9-router-credit-mechanism-seed1_2v2.zip`).
3. Continuation — 7 more updates (base_step 1,769,472 → 2,169,472) via a
   clean resume that **preserves** the learned router + optimizer
   (`--router-reinitialize-on-load false`; behavioral-equivalence check
   PASS, argmax_diff=0; freeze intact)
   (`final_v6i9-router-credit-mechanism-cont-seed1_2v2.zip`).

**Continuation trend (updates 28→34):** `strategy_entropy` fell
monotonically `1.356 → 1.256`; **`MI_z_obs` stayed pinned at ≈0.001**
throughout; sampled occupancy concentrated `z0/z1 → z2/z3`
(`[0.24,0.21,0.28,0.28] → [0.12,0.16,0.34,0.38]`). Signature of a
**context-independent preference shift (mode-collapse toward z3)**, not
context-dependent routing.

**Cheap held-out ablation (OP8/OP10 × {map_b, map_b_split_lane_v2},
10 seeds/cell = 160 eps).** Cross-episode histogram-preserving shuffle
control added (`build_cross_episode_shuffled_mapping_from_learned_traces`).

* On the mechanism (3-update) checkpoint (base_seed 12000): eval argmax
  = z3 in 188/188 opportunities, mean-max-prob 0.300, entropy 1.376.
* On the continuation (10-update) checkpoint (fresh base_seed 14000):
  eval argmax = z3 in **177/177** opportunities across all cells;
  mean-max-prob **0.30 → 0.462**; entropy `1.226`. Returns (n=40):
  `uniform −3.231 > learned/shuffled −3.399 > fixed_z2 −3.763`. Both
  shuffle controls are byte-identical to learned (constant output ⇒
  shuffles are no-ops). Promotion gates all fail
  (`learned_beats_uniform=False`, `learned_beats_shuffled=False`,
  `proceed_to_250k=False`).

**Conclusion (decisive).** With provably healthy plumbing (correct
credit source, ~85% positive advantage fraction, entropy subordinate at
grad-ratio 11–24, repertoire frozen, resume clean) the feedforward
router trained on the **geometry-only 35-dim context collapses to a
context-independent constant z3**. More updates increase *confidence*
(max-prob 0.30→0.46) but **not context sensitivity** (`MI_z_obs≈0`
throughout; eval argmax constant). The under-training hypothesis is
refuted; this is the **Fail branch = context insufficiency**. Note the
two known ablation caveats (do not over-read the trace-based trust
gates): (a) the cross-episode shuffle grouping keyed on
`(opponent, episode_seed)` yields singleton cells, and (b) a pre-existing
trace-summary join keys episode rows on `cell_seed` vs traces on
`episode_seed` so only `episode_index=0` joins — both inflate
`same_z_sequence`; the return-based verdict is unaffected.

**Open decision (fork presented, user deferred):** (i) offline best-z
predictability probe on the real 35-dim decision-time context (cheap;
separates "context lacks signal" from "router can't extract it") before
spending compute; (ii) lift the deferral and add opponent/map identity
to the context, then retrain (the Fail-branch action); (iii) stop
feedforward routing on geometry-only context. Artifacts:
`artifacts/router_credit_mechanism_ablation_crossep/`,
`artifacts/router_credit_mechanism_cont_ablation/`,
`artifacts/v6i9_router_credit_mechanism_cont_seed1.log`.

### 3.0.2 Recurrent running-mean arc-credit A/B + collapse-visibility tooling — `EVALUATED` (mechanism repaired, router collapsed)

**Status:** `EVALUATED`. Recurrent-GRU A/B from the repertoire anchor
(2v2, OP8/OP9/OP10, seed 1, frozen repertoire, 5 updates each). Control =
`v6i9_mapaware_router_sparse_hardpool` (sparse-GAE router credit);
treatment = `v6i9_arc_credit_running_mean_hardpool` (running-mean arc
credit, main-loop strategy PPO disabled). Same GRU, same initial router +
frozen-actor hashes, fresh optimizer both arms, credit channel the only
functional difference (`compare_ab_router_credit.py` launch contract:
PASS).

**Mechanism result (treatment):** running-mean baseline works — final
`raw_adv_mean ≈ −0.09`, `positive_fraction ≈ 0.57`, per-z spread ≈ 0.067,
router gradients active. The chronic critic-bias advantage offset is
removed and the credit signal is two-sided.

**Behavioral result (fresh held-out base_seed 15000, 150 eps/condition):**
`fixed_z2 −3.09 > learned −3.58 > uniform −3.78`. Learned beats uniform
and within-episode shuffle but **loses to fixed_z2** and does not beat the
cross-episode shuffle. Root cause: the router **collapsed** — argmax = z3
in the large majority of opportunities, z0/z2 never selected. Repaired
grades did not (yet) produce better *choices*.

**Tooling fixes landed with this entry:**

1. **Cross-episode shuffle regrouping (caveat (a) in §3.0.1 fixed).**
   `build_cross_episode_shuffled_mapping_from_learned_traces` now groups
   cells by `(opponent, map)` instead of `(opponent, episode_seed)`. Under
   a unique-seed-per-episode protocol the old key produced singleton cells
   → `can_reassign=False` → a structural no-op that made the
   `learned_beats_cross_episode_shuffled` gate vacuous (delta 0.0). Map is
   threaded through the opportunity trace
   (`inference_policy.set_current_map`, set per cell in
   `eval_v6i9_router_diagnostic_ablation._run_condition`); absent map falls
   back to per-opponent grouping. Note: for a *collapsed* router the
   shuffle is still legitimately a no-op (identical signatures), which is
   now an honest collapse signal rather than a grouping artifact.
2. **Decision-point selected-z occupancy telemetry.**
   `router_selected_z_occupancy_z{0..K-1}` (+ `_max`, `_unique_count`,
   `_dominant`, `_decision_count`) computed every update in
   `_latent_rollout_stats` for both credit channels, independent of the
   entropy mode (unlike `router_rollout_soft_argmax_occupancy_*`, which
   only runs on the marginal-entropy path). Surfaced in the A/B runner and
   in `compare_ab_router_credit.py` (`router_not_collapsed` signal). Makes
   router collapse visible per-update during training, not only at the eval
   shuffle gate. Pinned by
   `tests/test_router_occupancy_and_cross_episode.py`.

### 3.0.3 v6i9 arc-credit *specialize* preset (entropy-balance) — `EVALUATED` (mechanism FAIL: global z3 collapse, MI≈0)

**5-update result (2026-07-03, recurrent, seed 1, from repertoire anchor;
`artifacts/ab_router_specialize/treatment/`).** Integrity PASS (frozen-actor
hash match, router moved `q_phi_grad` 0.041→0.073, fresh optimizer, arc
credit active, old strategy-PPO channel off, resolved entropy path correct).
Specialization/coverage FAIL:

| upd | H_marg | H_cond | MI_proxy | margin | argmax_frac |
|-----|--------|--------|----------|--------|-------------|
| 1 | 1.3845 | 1.3844 | 0.0001 | 0.065 | z1=1.00 |
| 2 | 1.3725 | 1.3722 | 0.0004 | 0.018 | z2=0.68/z3=0.32 |
| 3 | 1.3661 | 1.3649 | 0.0011 | 0.248 | z3=1.00 |
| 4 | 1.3521 | 1.3494 | 0.0027 | 0.343 | z3=1.00 |
| 5 | 1.3203 | 1.3182 | 0.0022 | 0.482 | z3=1.00 |

`MI_proxy` peaked at ~0.0027 nats (~0.2% of log 4) — `H_cond ≈ H_marg`
throughout, i.e. the router is **context-independent**. Deterministic
argmax collapsed to **z3=100%** by update 3; `q_bar` drifted z3 0.25→0.40,
z0 0.23→0.15 while `H_marg` fell 1.385→1.320 (coverage eroding). The
growing top1−top2 margin is a **global** logit bias, not contextual
confidence. This is the "false diversity → global collapse" pattern:
`latent_lam_h=0.01` marginal coverage too weak to hold the distribution
while reduced `router_ent_coef` let a global z3 preference form.

**Conclusion.** With the entropy path now correctly wired (bug in §3.0.3
fixed pre-launch) the two-axis entropy balance changed *which* latent and
increased *confidence* but not *context sensitivity* — reproducing the
§3.0.1 context-insufficiency finding on the recurrent router. No entropy
knob converts a context-independent router into a context-dependent one
when MI(z;context)≈0. Behavioral gate expected to be near-vacuous
(deterministic z3 everywhere ⇒ cross-episode shuffle likely
`can_reassign=False` / `cross_episode_gate_untestable=true`). Recommended
next: the offline best-z predictability probe on the real 35-dim
decision-time context (separate "context lacks signal" from "router can't
extract it") before adding opponent/map identity to the context.

**Learned-only preflight (`base_seed=18000`, 8 eps/cell, learned router
only, `experiments/preflight_learned_trace.py` →
`artifacts/ab_router_specialize/treatment/preflight_s18000.json`).** Ran a
cheap 48-episode learned-only trace instead of the full 900-episode
behavioral exam, to decide whether the cross-episode shuffle is even
testable. Result confirms the collapse prediction:

```text
argmax_z_histogram        : {3: 230}   (100% z3, all decisions)
distinct_z_values         : [3]
non_constant_episode_count: 0          (no episode ever switches z)
cross_episode_gate_untestable = true
```

Verdict: **STOP — do not run the full behavioral exam.** The cross-episode
shuffle is an identity permutation: every episode plays z3 at every router
opportunity, so fixed_z2 / uniform / shuffled conditions cannot prove
contextual routing (nothing to shuffle). Note a tooling lesson: the
preflight's first auto-verdict was a *false* PROCEED because
`build_cross_episode_shuffled_mapping_from_learned_traces` returned
`can_reassign=True` — but that came purely from episodes having different
*lengths* (`[3,3,3]` vs `[3,3,3,3,3]`), not different z *values*. The
preflight gate was corrected to require ≥2 distinct z **values**
(`non_constant_episode_count` / `distinct_z_values`), not length-distinct
signature tuples. This is the definitive answer for the specialize arm:
the behavioral gate is untestable; the next lever must create z-value
variation (offline best-z context probe, or context enrichment), not
another entropy-knob run.

### 3.0.3b v6i9 arc-credit *specialize* preset (entropy-balance) — original PENDING_LAUNCH notes

**Preset:** `v6i9_arc_credit_specialize_hardpool` (aliases
`v6i9_arc_credit_specialize`,
`plan_faithful_latent_v6i9_arc_credit_specialize_hardpool`), parent
`v6i9_arc_credit_running_mean_hardpool` (recurrent GRU router, running-mean
arc credit, BPTT PPO disabled). `SUMMER-COMPATIBLE EXTENSION`.

**Hypothesis:** can the router become *decisive within each context*
(lower H(z|context)) while still *using all four latents across the
context distribution* (preserve marginal coverage)? Two-axis entropy
balance on top of the repaired running-mean credit channel.

**Resolved-config diff vs the running-mean parent (exactly 6 keys, pinned
by `tests/test_v6i9_arc_credit_specialize.py`):**

```text
router_ent_coef          : 0.005 -> 0.001   (weaker conditional entropy)
latent_lam_h             : 0.0   -> 0.01     (marginal coverage weight)
latent_entropy_mode      : conditional -> marginal
latent_entropy_objective : none -> maximize
h_mode                   : conditional -> marginal  (legacy alias, kept consistent)
run_tag                  : ...specialize...
```

**Bug found and fixed before any launch (2026-07-03).** The preset
originally set only the legacy `h_mode="marginal"` field. The runtime
entropy path (`rl/custom_ppo/update/entropy_objectives.py::RolloutMarginalPrep`)
and the audit banner both key off `latent_entropy_mode`, which stayed
`"conditional"`, and the arc-credit parent had zeroed
`latent_entropy_objective` to `"none"`. Net effect of the buggy config:
the rollout-level marginal-coverage loss never engaged, and `latent_lam_h`
acted as a **conditional entropy-maximization** term (pushing q_phi toward
uniform *per context*) — the exact opposite of the intended "decisive
within each context." Fix sets `latent_entropy_mode="marginal"` and
`latent_entropy_objective="maximize"` so the rollout-level
`rollout_marginal_entropy_loss` path (AGENTS.md aggregation contract)
actually runs. Verified: resolved config now yields
`marginal-path would_apply = True`. Snapshot regenerated (additive: 6 new
arc-credit/specialize entries, 0 existing presets changed).

**Launch (5-update mechanism run, recurrent, from the repertoire anchor):**

```powershell
uv run python experiments/run_ab_router_credit.py --arm treatment `
  --preset v6i9_arc_credit_specialize_hardpool `
  --checkpoint checkpoints/2v2/final_v6i9-mapaware-repertoire-hardpool-refactor-r1-seed1_2v2.zip `
  --n-updates 5 --device cuda --seed 1 `
  --out-dir artifacts/ab_router_specialize --force
```

Mechanism success (per-update telemetry): `H_marg` high (≈log 4≈1.386),
`H_cond` falling below `H_marg`, `MI_proxy = H_marg − H_cond` rising,
`q_bar` still spread over all four z, and — via the new
`router_selected_z_occupancy_z*` telemetry — the decision-point argmax
histogram becoming context-dependent (z0/z2 used, not only z1/z3).

**Behavioral gate (fresh held-out seeds — do NOT reuse 15000):**

```powershell
uv run python experiments/eval_v6i9_router_diagnostic_ablation.py `
  --checkpoint artifacts/ab_router_specialize/treatment/final_treatment.zip `
  --anchor-checkpoint checkpoints/2v2/final_v6i9-mapaware-repertoire-hardpool-refactor-r1-seed1_2v2.zip `
  --opponents OP8 OP9 OP10 --maps map_b map_b_split_lane_v2 `
  --episodes 25 --base-seed 18000 --device cuda `
  --out-dir artifacts/ab_router_specialize/treatment/shuffle_eval_s18000
```

Promotion requires: `learned > cross_episode_shuffled`, `learned > uniform`,
`learned` closer to `fixed_z2`, and `cross_episode_gate_untestable = false`
(now emitted explicitly by the evaluator; the corrected (opponent, map)
cell grouping means a genuine reassignment/delta or an explicit
untestable flag — never a silent identity tie).

### 3.1 v5i6 — canonical marginal-entropy interpretation (IMPLEMENTED, PENDING_LAUNCH)

**Status:** `IMPLEMENTED, PENDING_LAUNCH`. Preset committed as
`v5i6_paper_faithful_marginal_entropy` (apply function
`apply_plan_faithful_latent_v5i6_paper_faithful_marginal_entropy`
in `rl/presets/plan_faithful.py`). Aliases registered in
`rl/presets/__init__.py`. Fidelity tests live in
`tests/test_v5i6_paper_faithful_marginal_entropy.py`. Snapshot
regenerated. Audit banner prints `[PPO] v5i6 paper-faithful audit:`
and reports
`entropy maximization: ON (mode=marginal, aggregation=rollout, objective=maximize)`.

**Scientific delta:** v5i6 inherits v5i4 directly and replaces mean
conditional entropy `E_s[H(q_phi(z|s))]` with **rollout-level**
marginal entropy `H(E_s[q_phi(z|s)])` aggregated over **every**
resample-decision point in the rollout (not per-PPO-minibatch — see
`summer-method-spec.md` §8.1 for the Jensen rationale). The marginal
loss is computed once per PPO inner epoch by
[`rl/latent_losses.py::rollout_marginal_entropy_loss`](../rl/latent_losses.py),
applied to the first minibatch of that epoch's `latent_loss`, and uses
the same `lambda_H` schedule as v5i5 (`0.003 -> 0.001` over
`0..300_000`) so the v5i6-vs-v5i5 comparison isolates only the
entropy-reduction interpretation. No actor, critic, sampling,
router-PPO, persistence, curriculum, label, FiLM, episode-credit,
preference, distillation, or auxiliary-head channel changes.

**Resolved diffs:** v5i4 -> v5i6 is exactly
`{latent_entropy_mode, latent_lam_h_end, run_tag}`. v5i5 -> v5i6 is
exactly `{latent_entropy_mode, run_tag}`.

**Aggregation contract (post-Jensen-bias-fix):** the marginal entropy
loss must be taken over the **full** rollout resample subset (~1024
states for the standard 32-env × 2048-step rollout with cadence 64).
The deprecated per-minibatch helper `strategy_marginal_entropy_loss`
(active in earlier v5i6 prototypes) systematically over-applied the
loss by Jensen `E_B[KL(q_bar_B || U)] >= KL(q_bar_rollout || U)` and
the gap was closed by the gradient softening individual `q_phi(z|s)`
toward uniform — the conditional-entropy regression v5i6 was meant
to replace. The current rollout-level path is pinned by
`tests/test_latent_losses.py::RolloutMarginalEntropyLossTests` and
`tests/test_v5i6_paper_faithful_marginal_entropy.py::V5i6RolloutMarginalEntropyContractTests`.

**Required evidence to declare v5i6 successful:**

1. High aggregate usage: `router_rollout_soft_marginal_entropy_nats /
   ln(K)` (and the sampled-z analogue `latent_marginal_entropy_nats /
   ln(K)` / `effective_num_latents`) remain high late in training.
2. Confident routing: `router_rollout_soft_conditional_entropy_nats`
   (per-state `H(q_phi(z|s))` averaged over the same rollout resample
   subset) stays meaningfully below the marginal value, producing
   positive `router_rollout_soft_mi_proxy_nats` rather than uniform
   per-state indecision. This is the "broad and state-specific"
   pattern; the failure mode "broad but indecisive" appears as both
   marginal and conditional approaching `ln(K)`.
3. Soft-argmax occupancy stays balanced:
   `router_rollout_soft_argmax_occupancy_max < ~0.50` and
   `router_rollout_soft_argmax_occupancy_ratio` close to `1`.
4. Forced-z evaluations show distinct behaviors under matched seeds.
5. `router` beats `random-matched` at eval time without a meaningful
   win-rate loss versus v5i4/v5i5.

**Launch command:**

```bash
python rl/train_ppo.py \
  --preset v5i6_paper_faithful \
  --total-steps 1000000 \
  --agents 4 \
  --seed 0 \
  --device cuda \
  --n-envs 32 \
  --n-epochs 6 \
  --e3-step-telemetry \
  --checkpoint-dir checkpoints/4v4 \
  --fresh-metrics-csv \
  --periodic-checkpoint-steps 50000
```

### 3.2 v5i5 — conditional entropy-floor ablation (IMPLEMENTED, PENDING_LAUNCH)

**Status:** `IMPLEMENTED, PENDING_LAUNCH`. Preset committed as
`v5i5_paper_faithful_entropy_floor` (apply function
`apply_plan_faithful_latent_v5i5_paper_faithful_entropy_floor`
in `rl/presets/plan_faithful.py`). Aliases registered in
`rl/presets/__init__.py`. Fidelity tests in
`tests/test_v5i5_paper_faithful_entropy_floor.py` (20 tests, all green).
Snapshot regenerated. Audit banner is family-aware
(`rl/training/banner.py`) and prints `[PPO] v5i5 paper-faithful audit:`
when the run tag matches `v5i5_paper_faithful*`. **Ready to launch.**
The *Proposed Preset Review* template should be filed retroactively in
[`summer-fidelity-rules.md`](summer-fidelity-rules.md) §"Open /
unresolved".

**Motivating observation (§2.1):** v5i4 collapsed to z=2-dominant
occupancy at 1 M steps (~64% on `z2`, ~7% on `z3`). The entropy schedule
`lam_h: 0.003 → 0.0002` over `0..300_000` reached its floor
`0.0002` ~700 k steps before the collapse stabilized. A higher floor
preserves `H(q_phi) ≥ ε` without changing the architecture, the
loss objective, or the resampling cadence.

**Locked single-axis design:** the only resolved field changed against
v5i4 is `latent_lam_h_end` (raised from `0.0002` to `0.001`). The
`v5i4 → v5i5` resolved-config diff is **exactly two keys**
(`latent_lam_h_end`, `run_tag`); this is enforced by
`tests.test_v5i5_paper_faithful_entropy_floor.V5i5PresetInheritanceTests.test_v5i5_minimal_diff_vs_v5i4`.
`latent_lam_h_start` stays at `0.003`, the anneal window stays
`0..300_000`, and the floor `0.001` remains *inside* the
`[0.001, 0.01]` Summer-plan range, so R20 / R21 are still satisfied.

**Classification:** `PAPER-FAITHFUL`. The change is a hyperparameter
inside the documented Summer-plan entropy range; no fidelity rule
(R1..R42 in [`summer-fidelity-rules.md`](summer-fidelity-rules.md))
flips state. Actor stays embedding-concat, router stays main-loop PG,
no forbidden channel enabled. The launch-time audit banner prints the
family-prefixed `v5i5 paper-faithful audit` block.

**New diagnostics added with v5i5 (no new gradient channel, no new
loss term):** `effective_num_latents`, `latent_marginal_entropy_nats`,
`latent_occupancy_min`, `latent_occupancy_max`,
`latent_occupancy_ratio`, `mean_strategy_duration` -- all logged to
the per-update metrics CSV.
[`rl/custom_ppo/latent_diagnostics.py`](../rl/custom_ppo/latent_diagnostics.py)
`_latent_rollout_stats` is the only function that grew. The CSV
header was extended in `rl/custom_ppo/csv_writers.py`. A schema
test (`V5i5OccupancyDiagnosticSchemaTests`) pins the new columns
plus the previously-existing per-z and per-z-per-opponent columns
so the v5i5 telemetry contract cannot regress silently.

**Required evidence to declare v5i5 successful:**

1. End-of-run `H(q_phi) / ln(K) ≥ 0.6` (vs v5i4's `0.46`).
   Equivalently: `effective_num_latents ≥ exp(0.6·ln 4) ≈ 2.30`.
2. Final `latent_occupancy_max ≤ 0.50` and
   `latent_occupancy_ratio ≤ ~5` (vs v5i4's ~9).
3. Headline WR is not worse than v5i4 by paired-bootstrap
   `Δ-CI > 0` at 95% (i.e. the entropy floor is not a free reward
   sacrifice).
4. `router` vs `random-matched` Δ at eval time is not worse than
   v5i4's.

**Required baselines:** v5i4 at the same seed/budget,
`no_latent_v4i3_baseline` at v5i4's exact budget and seed.

**Launch command:**

```bash
python rl/train_ppo.py \
  --config v5i5_paper_faithful_entropy_floor \
  --total-steps 1000000 \
  --seed 0
```

(All other knobs come from the preset; do not override
`--latent-strategy-ppo-coef`, `--latent-lam-p`, etc., or the v5i5
contract is broken.)

### 3.3 v5i7 — Summer-faithful entropy-floor split-lane row (IMPLEMENTED, PENDING_LAUNCH)

**Status:** `IMPLEMENTED, PENDING_LAUNCH`. Preset committed as
`v5i7_summer_faithful_entropy_floor_split_lane` (apply function
`apply_plan_faithful_latent_v5i7_entropy_floor_split_lane` in
`rl/presets/plan_faithful.py`). Aliases registered in
`rl/presets/__init__.py`. Fidelity tests live in
`tests/test_v5i7_entropy_floor_split_lane.py`. Snapshot regenerated.
Audit banner prints `[PPO] v5i7 paper-faithful audit:` and reports
`entropy maximization: ON (mode=conditional, aggregation=per-state, objective=maximize)`.

**Scientific delta:** v5i7 inherits v5i5 directly and changes only the
environment geometry to `map_b_split_lane`. It keeps v5i5's conditional
entropy floor (`0.003 -> 0.001`), concat-only actor, main-loop
categorical PPO term on `q_phi`, persistence, sparse 64-decision
resampling, opponent pool, no forced-z curriculum, no FiLM/adapter, no
marginal entropy, no auxiliary heads, and no extra `q_phi` gradient
channel.

**Resolved diff:** v5i5 -> v5i7 is exactly `{map_layout, run_tag}`.

**Required evidence to declare v5i7 successful:** use the v5i5 occupancy
criteria plus matched-seed forced-z and router-vs-random-matched evals on
the split-lane map. Compare only against split-lane matched controls when
making causal claims about the latent method.

**Launch command:**

```bash
python rl/train_ppo.py \
  --preset v5i7 \
  --total-steps 1000000 \
  --agents 4 \
  --seed 0 \
  --device cuda \
  --n-envs 32 \
  --checkpoint-dir checkpoints/4v4
```

### 3.4 v5i8 - Summer-faithful split-lane v2 task-pressure row (IMPLEMENTED, PENDING_LAUNCH)

**Status:** `IMPLEMENTED, PENDING_LAUNCH`. Preset committed as
`v5i8_split_lane_v2_task_pressure` (apply function
`apply_plan_faithful_latent_v5i8_split_lane_v2_task_pressure` in
`rl/presets/plan_faithful.py`). Aliases registered in
`rl/presets/__init__.py`. Fidelity tests live in
`tests/test_v5i8_split_lane_v2_task_pressure.py`. Snapshot regenerated.
Audit banner prints `[PPO] v5i8 paper-faithful audit:` and reports
`entropy maximization: ON (mode=conditional, aggregation=per-state, objective=maximize)`.

**Scientific delta:** v5i8 inherits v5i7 directly and changes only the
environment geometry to `map_b_split_lane_v2`. It keeps v5i7's v5i5
conditional entropy floor (`0.003 -> 0.001`), concat-only actor,
main-loop categorical PPO term on `q_phi`, persistence, sparse
64-decision resampling, opponent pool, no forced-z curriculum, no
FiLM/adapter, no marginal entropy, no auxiliary heads, and no extra
`q_phi` gradient channel.

**Resolved diff:** v5i7 -> v5i8 is exactly `{map_layout, run_tag}`.

**Map-side intent:** reduce wall-bump noise and make route choices more
legible. The v2 wall is narrower/shorter than v5i7's split-lane wall,
route guidance uses larger clearance around the obstacle, and OP5/OP6/OP7
stress different lane-pressure patterns through normal scripted opponent
movement. The episode CSV adds route-context counters for attack, return,
and intercept crossings so route behavior can be grouped by `latent_z`
without assigning meanings to latent IDs.

**Required evidence to declare v5i8 successful:** use the v5i7 occupancy
criteria plus lower obstacle-collision counts, nontrivial attack/return/
intercept route distributions by `latent_z`, matched-seed forced-z evals,
and router-vs-random-matched evals on the split-lane-v2 map. Compare only
against split-lane-v2 matched controls when making causal claims about the
latent method.

**Launch command:**

```bash
python rl/train_ppo.py \
  --preset v5i8 \
  --total-steps 1000000 \
  --agents 4 \
  --seed 0 \
  --device cuda \
  --n-envs 32 \
  --checkpoint-dir checkpoints/4v4
```

**Post-training forced-z evaluation command:**

```bash
python tools/v5i8_forced_z_eval.py \
  --checkpoint checkpoints/4v4/<v5i8_final>.zip \
  --metrics-csv checkpoints/4v4/<v5i8_metrics>.csv \
  --map-layout map_b_split_lane_v2 \
  --opponents OP5 OP6 OP7 \
  --episodes-per-mode 100 \
  --device cuda
```

This is the required evidence harness for the v5i8 latent-strategy claim.
It keeps training unsupervised and tests learned `z` behavior after the
checkpoint is frozen.

### 3.5 v5i9 - CSIA guided specialization extension (IMPLEMENTED, PENDING_EVIDENCE)

**Status:** `IMPLEMENTED, PENDING_EVIDENCE`. Preset committed as
`v5i9_csia_guided_specialization` (apply function
`apply_plan_faithful_latent_v5i9_csia_guided_specialization` in
`rl/presets/plan_faithful.py`). Aliases registered in
`rl/presets/__init__.py`. Focused tests live in
`tests/test_csia.py` and
`tests/test_v5i9_csia_guided_specialization.py`.

**Classification:** `SUMMER-COMPATIBLE EXTENSION`, not
`PAPER-FAITHFUL`. v5i9 inherits v5i8 but enables detached CSIA reward
feedback. Once gates pass, PPO trains on `reward_total + reward_csia`.

**Scientific delta:** v5i9 asks whether causal strategic-impact feedback
from frozen forced-z evaluation improves opponent-adaptive latent
specialization. It does not add labels, role targets, opponent ID inputs,
auxiliary heads, actor FiLM/adapters, forced-z curriculum, or a new
router optimizer.

**Resolved diff:** v5i8 -> v5i9 is exactly
`{csia_enabled, csia_reward_coef, run_tag}`.

**Required evidence before launch:** run the v5i8 forced-z harness and
save both:

```text
*_qualitative_rollout_by_z.csv
*_strategy_evidence.csv
```

**Launch command:**

```bash
python rl/train_ppo.py \
  --preset v5i9 \
  --total-steps 1000000 \
  --agents 4 \
  --seed 0 \
  --device cuda \
  --n-envs 32 \
  --checkpoint-dir checkpoints/4v4 \
  --csia-payoff-csv checkpoints/4v4/qualitative/<stem>_qualitative_rollout_by_z.csv \
  --csia-strategy-evidence-csv checkpoints/4v4/qualitative/<stem>_strategy_evidence.csv \
  --fresh-metrics-csv
```

**Success criteria:** `csia_bonus_active = 1`, gates A/B/C pass, and
post-training forced-z eval shows behavioral differences plus
opponent-dependent performance or macro-behavior differences. If v5i9
only improves win rate without forced-z behavior spread, the extension
improved performance shaping but did not prove latent strategy
specialization.

### 3.6 v5i4 multi-seed (PLANNED)

**Status:** `PLANNED`. After the v5i4 single-seed eval matrix
(§2.1) and the `no_latent_v4i3_baseline` matched-budget re-launch,
add **two more v5i4 seeds** (`--seed 1`, `--seed 2`) and two more
`no_latent_v4i3_baseline` seeds to reach the §5.4 headline minimum
of three seeds per row.

### 3.7 v5i4 random-matched eval (PLANNED, eval-time only)

**Status:** `PLANNED`, no training cost. Run
`plot/eval_checkpoint.py --latent-selection router` and
`--latent-selection random-matched` against every saved v5i4
checkpoint with identical `--seed` and identical `--episodes`. The
delta is the matched-schedule routing-quality control
([`experiment-and-evaluation-protocol.md`](experiment-and-evaluation-protocol.md) §4.2).

### 3.8 v4i4post_periodic_router_distill comparison (DEFERRED)

**Status:** `DEFERRED`. Counter-factual router distillation is the
honest next step *only* if v5i4 fails its gates (§2.1 eval matrix +
§3.2 multi-seed). If v5i4 passes the §4.2 routing-quality control with
a paired-bootstrap-significant delta, v4i4post is icing and is
deprioritized.

### 3.9 v6i10 episode-router exploration preset — `EVALUATED` (smoke PASS; 5-update mechanism HARD-STOP, MI≈0)

**Status:** `EVALUATED`. Committed at `696bfb1` (code + tests; tracker doc
follow-up `d03f7e0`). The one-update runtime smoke passed **every** gate
and the five-update mechanism run then triggered two hard-stop conditions
(MI pinned at the smoke floor; per-z advantage rankings flip randomly).
Verdict: **reject for promotion; do not run the behavioral grid.** The
35-dim geometry-only context yields no extractable routing signal — the
third independent confirmation after §3.0.1 (v6i9 continuation) and
§3.0.3 (v6i9-specialize).

**Smoke (1 update, `--load-weights-only` from anchor, run_tag
`v6i10-episode-router-explore-smoke-seed1`):** all 11 gates green — one
decision/episode (486 opportunities − 454 finalized arcs = 32 open =
one per env; `arc_mean_length=138.8`); z fixed until termination
(`strategy_switch_count=0`); behavior = `0.8·q_phi + 0.2·U` with stored
old-log-prob = behavior mixture (config `router_uniform_exploration_prob=0.2`,
`router_sampling.py:730`, unit-test pinned); all four z sampled
(`unique=4`); `latent_arc_running_mean_count=454`; marginal entropy active
(`rollout_marginal_active=1.0`, `main_loop_q_phi_grad_norm=3.2e-5>0`);
router grads nonzero (`q_phi_grad_norm=0.0253`); frozen actor+z grads/deltas
exactly 0; frozen tensor hash byte-identical to anchor (`f332687…`);
checkpoint round-trips bit-exact. Deterministic argmax already 0.909-
concentrated on one z despite near-uniform `q_bar` (the v6i9 precursor).

**Five-update mechanism (`experiments/run_ab_router_credit.py --arm
treatment`, from anchor, fresh optimizer, `source_commit=d03f7e0`,
`artifacts/v6i10_episode_router_explore/treatment/`):**

```text
              u1       u2       u3       u4       u5
mi_proxy    2.19e-5  2.13e-5  2.40e-5  2.30e-5  2.31e-5   FLAT at smoke floor
top1-top2   0.0212   0.0157   0.0124   0.0107   0.0118    SHRINKING (logits flatten)
argmax z2   0.934    0.865    0.705    0.638    0.715     (z0,z3 NEVER argmax)
best per-z  z3       z1       z3       z2       z3        ranking flips randomly
frozen hash f332687 == f332687 unchanged; router moved YES; unique z=4 each update
```

`H_marginal ≈ H_conditional ≈ ln4` every update ⇒ the router emits a
near-uniform distribution for every state; the per-episode z is therefore
effectively random (near-uniform router + 20% floor), so arc credit chases
noise and per-z ordering never stabilizes. The `argmax<0.90` guard passed
but is a **false positive**: argmax softened (0.93→0.72) because logits
flattened (indecision), not because context emerged. Plumbing verified
perfect (exact frozen hash, auto-centered arc advantage, ~85–99% positive
fraction). Artifacts: `summary.json`, `run_meta.json`, `final_treatment.zip`.

**Next lever (unchanged from §3.0.1 conclusion):** context enrichment
(opponent/map identity) or the offline best-z predictability probe on the
real 35-dim context — **not** another entropy/credit/exploration knob run.

### 3.10 v6i11 contextual Q-value return router — `EVALUATED` (three pre-run bugs fixed; 15-update diagnostic = FLAT on a VALID dataset)

**Result (2026-07-03, `artifacts/v6i11_q_router_run2_seed1/summary.json`):**
`routing_verdict = FLAT`, `promotion_status = NOT_A_CANDIDATE`,
`reliably_separating_opponents = 0/3`. The dataset is **valid, not
insufficient**: replay_size 7038, no duplicates, all z + all opponents
represented, `min_cell_arcs = 527` (≫ 20/cell bar), `return_variance = 11.08`,
`mean_arc_length ≈ 139`, `terminal_finalized_fraction = 1.0`,
`frozen_actor_ok = true`. So `FLAT` here is a genuine negative under the
tightened semantics, not a swallowed pipeline failure.

*Why FLAT (the reliability gate did its job).* Empirical row-spreads
(OP8 0.262, OP9 0.240, OP10 0.256) exceed the 0.10 magnitude threshold, but
every best-vs-second-best gap's bootstrap CI **includes zero** (OP8 gap 0.151
CI[-0.15,0.45]; OP9 0.030 CI[-0.34,0.38]; OP10 0.036 CI[-0.36,0.43]). Episode-
return std (≈ 2.6–3.9 per cell) swamps the ≈0.15–0.26 per-z mean gaps even at
~530–630 arcs/cell. Predicted-Q spread stayed tiny (0.01–0.04) — the network did
**not** invent confident spreads — and best-z agreement was 2/3 (OP8✓, OP9✗,
OP10✓), i.e. suggestive but unreliable.

*What FLAT does and does not mean here.* It does **not** re-open repertoire
diversity (already established by counterfactual actor-logit differences,
forced-z behavioural separation, and the +2.37 forced-z EPISODE oracle gap). The
key tension: the oracle gap is a **paired, matched-seed within-episode**
quantity, whereas this Q-router regresses an **unpaired between-episode**
expectation `E[return | episode-start context, z]`. Between-episode variance
(map, spawn, opponent stochasticity) dominates the per-z effect, so an unpaired
replay-mean target cannot resolve the latents at this SNR/budget — and/or
episode-start geometry is only weakly predictive of which z wins *this* episode.
FLAT = "the current unpaired Q-formulation failed to resolve the latents under
this dataset/horizon/context/budget," not "the latents don't differ."

*Consequence for the held-out gate.* Per the recommended sequence, the held-out
prospective evaluator (`experiments/eval_v6i11_q_router_heldout.py`, built) is
run only when the diagnostic is at least `WEAK_SEPARATION`. FLAT does **not**
meet that bar, so the held-out gate is **not** run and `map_id` instrumentation
is **not** warranted yet. A productive redesign (not yet actioned) would target
the paired signal directly — e.g. a per-context/per-episode baseline-subtracted
(advantage-style) target rather than a raw between-episode return mean.

**Pre-run status (retained):** 15-update diagnostic ran from the clean anchor
(seed 1, cuda, ~13 min/update ≈ 3.3 h wall). Update 1 validated coverage
(29–51 arcs/cell, balanced `count_by_z`, `arc_length ≈ 138`, `term_frac = 1.0`,
`records_after_update = 0`). Preset
`v6i11_q_router_hardpool` (aliases `v6i11_q_router`,
`plan_faithful_latent_v6i11_q_router_hardpool`), experiment
`experiments/run_v6i11_q_router.py`, external model `rl/router/q_value_router.py`.
Classification: **SUMMER-COMPATIBLE EXTENSION** — off-policy value regression
over online experienced returns, plus a 3-way opponent one-hot as an *input*
feature (not opponent-identity supervision). Targets are experienced returns
from sampled actions; no hindsight forced-z labels, no best-z labels, actor +
adapters frozen. **Not yet run.**

**Scientific delta:** replace BPTT PPO logit routing (which repeatedly turned
tiny logit biases into one-latent argmax collapse in §3.0.1/§3.9) with a
separate return-prediction model learning `context + selected z → expected
EPISODE return` from a replay buffer. Separates "estimate which latent has
higher value" (Q-router) from "execute the selected latent" (frozen actor).

**Three pre-run bugs found and fixed (2026-07-03):**

1. **Target-horizon mismatch.** The first draft inherited the cadence-32
   recurrent lineage (`strategy_interval=32`, `latent_resample_every_n=32`,
   `latent_arc_credit_min_len=32`), so each arc was a ~32-step MID-EPISODE
   segment and the target became "which z produced the best *local* arc
   return?" — NOT the episode-persistent forced-z EPISODE return validated by
   Probe A / the +2.37 oracle gap. **Fix:** re-parent to
   `v6i10_episode_router_explore_hardpool` (episode-persistent contract →
   `strategy_interval=0`, `latent_resample_every_n=0`, `min_len=1`), so
   arc == episode, `global_state_0` == episode-start context, `arc_return` ==
   total episode return. `arc_length` telemetry now printed per update to
   confirm arc ≈ episode length.

2. **Arc-extraction after drain.** The script read
   `rollout_strategy_arc_records` *after* `trainer.update()`, but
   `post_update.py` drains that buffer via `reset_arc_credit_rollout_state()`
   at the end of every update → the Q-router would have trained on **zero
   arcs** every step (silent no-op). **Fix:** extract arcs between
   `collect_rollout()` and `update()`.

3. **Opponent identity never captured (opponent one-hot always zero).** The
   rollout `arc_open` (router_sampling) and the episode-end `arc_finalize`
   (collector) both omitted `opponent_ids`, so `arc_open_opponent_id` stayed at
   its `-1` sentinel and every arc record carried `opponent_id = -1`. That
   zeroed the Q-router's opponent one-hot — collapsing the context back to
   geometry-only and defeating v6i11's premise — and forced every per-opponent
   cell to `count = 0 / mean = NaN` (an automatic `INSUFFICIENT_DATA`). A
   *second* half of the bug: the Q-router assumed OP8/9/10 → ids 8/9/10, but the
   canonical `_opponent_id_int_from_info` (`csv_writers._OPPONENT_TAG_TO_ID`,
   scheme OP_N → N-1) yields **7/8/9**, so even a threaded id would have been
   unmapped. **Fix:** (a) collector stamps the episode-end `arc_finalize` with
   `_opponent_id_int_from_info` per env (opponent is episode-constant, so the
   finalize-time value is exact for arc == episode); (b) `_OPPONENT_ID_TO_IDX`
   in the experiment and `_DEFAULT_OPPONENT_ID_TO_IDX` in the Q-router corrected
   to `{7:0, 8:1, 9:2}`; (c) `q_value_router` display labels now route through
   `_opponent_tag_from_id` so rows read OP8/OP9/OP10, not OP7/OP8/OP9. Verified
   live: update-1 `count_OP*_z*` all populated (29–51/cell), `mean_return_OP*_z*`
   real. Pinned by `V6i11OpponentContextWiringTests` (canonical one-hot rows,
   zero one-hot for -1/unmapped, default-map scheme).

**Hardening pass (2026-07-03, before trusting `summary.json`):**

* **Stable record IDs + rejection dedup.** `arc_finalize` now stamps each
  record with `env_index` + a monotonic `arc_uid`
  (`arc_credit.py`). The replay buffer dedups by identity
  `(rollout_index, env_index, arc_uid)` and **rejects** (does not insert)
  duplicates — content-hash dedup could collide two legitimate episodes.
  `push_many` returns `{inserted, duplicates_rejected, size_before,
  size_after}`.
* **Hard guards abort the run** (`check_arc_guards` → `ArcIntegrityError`)
  every update: `records_before_update > 0`, `inserted > 0`,
  `size_after > size_before`, and `records_after_update == 0` (proves the
  drain happened after we copied). A broken pipeline writes
  `routing_verdict = INVALID` and exits — it **never** emits `FLAT`.
* **Deep-copied extraction** (`copy_arc_record`) so the post-update reset
  cannot mutate the captured records; copy+push happen **before** `update()`.
* **Verdict is now 5-state** (`decide_verdict`): `INVALID` (zero arcs / dup
  contamination / horizon mismatch via terminal-finalized fraction / frozen
  actor drift), `INSUFFICIENT_DATA` (missing z, missing opponent, zero
  variance, or <20 arcs in the smallest cell), `FLAT`, `WEAK_SEPARATION`,
  `SEPARATING`. `FLAT` explicitly does **not** re-open repertoire diversity
  (proven by counterfactual logits, forced-z separation, +2.37 oracle gap);
  it means the Q-formulation failed to resolve the latents. Adding `INVALID`
  and `INSUFFICIENT_DATA` prevents `FLAT` from swallowing tooling failures.
* **Reliability gate:** an opponent separates only if row-spread ≥ threshold
  **and** the bootstrap CI on the best-vs-second-best mean-return gap excludes
  zero (`best_second_gap_ci`). Raw spread alone is insufficient.
* **Replay validity report** (`validity_report`): count-by-z, count-by-opponent,
  per-cell count/mean/std/sem, return variance, mean arc length,
  **terminal-finalized fraction** (episode-horizon check — should be ≈1.0),
  and the duplicate-rejection guard. Per-update coverage gate warns on z
  starvation by update ≥3.
* **`map × z` coverage remains NOT_INSTRUMENTED**: the arc record carries no
  `map_id` (threading it through the shared arc lifecycle touches every
  arc-credit preset). `count_by_opponent × z` is reported instead. Adding
  `map_id` is the prerequisite for a map-aware held-out grid.
* **Promotion is gated, not asserted:** a positive data verdict yields
  `promotion_status = SEPARATING_CANDIDATE` (not "wire in") and
  `heldout_gate = REQUIRED_NOT_RUN`. The decisive gate is the held-out
  prospective test (argmax-Q vs fixed-z2 / uniform / cross-episode-shuffled-Q /
  oracle; decisive = Q-router > shuffled-Q), a separate post-training step.
* Pinning tests: `tests/test_v6i11_q_router.py` (15 cases) — horizon contract,
  extraction-before-drain, zero-arc/no-insert abort, record_id dedup,
  terminal-fraction/arc-length, coverage→INSUFFICIENT_DATA, reliable
  separation→SEPARATING, noisy overlap→not SEPARATING, plus opponent-context
  wiring (canonical `{7:0,8:1,9:2}` one-hot rows, zero one-hot for -1/unmapped).

**Held-out prospective evaluator (built, not yet run):**
`experiments/eval_v6i11_q_router_heldout.py` is the decisive behavioural gate.
Matched-seed design: per `(opponent, map, seed)` held-out episode it reads the
legal t=0 context, predicts `Q(context, z)`, and runs ALL FOUR forced-z rollouts
once on fresh matched-seed envs; every condition (Q-router argmax, cross-episode
histogram-preserving shuffled-Q, uniform episode-persistent, fixed-z2, oracle)
is derived from the SAME four paired returns. Cross-episode shuffle permutes
chosen-z assignments *within* each `(opponent, map)` cell and reports
`cross_episode_gate_untestable = true` if no cell can be reassigned (all choices
identical) rather than a spurious zero-delta tie. Fresh `base_seed = 30000`,
disjoint from Probe A (42), the v6i9 diagnostic (4242), and v6i11 training
(seed 1). Decisive gate: paired `Q-router > shuffled-Q` (bootstrap CI excludes
0); then `> uniform`; then approaches/beats fixed-z2. Frozen-actor hash checked
before/after. It loads `q_router_final.pt`; run only after the diagnostic is at
least `WEAK_SEPARATION`.

**Next step:** await the running 15-update diagnostic
(`artifacts/v6i11_q_router_run2_seed1/summary.json`). If validity holds and the
verdict is at least `WEAK_SEPARATION`, run the held-out evaluator above; add
`map_id` instrumentation only before a full map-aware grid, per the recommended
sequence. Snapshot regenerated (adds the 3 v6i11 aliases only; no other preset
changed).

### 3.9-orig v6i10 episode-router exploration preset (original PENDING_SMOKE notes)

**Status:** `IMPLEMENTED, PENDING_SMOKE`. Preset committed as
`v6i10_episode_router_explore_hardpool` (aliases `v6i10`,
`v6i10_episode_router_explore`,
`latent_v6i10_episode_router_explore_hardpool`,
`plan_faithful_latent_v6i10_episode_router_explore_hardpool`),
`SUMMER-COMPATIBLE EXTENSION`, parent
`v6i9_mapaware_router_feedforward_hardpool`.

**Scientific delta:** simplify router learning to one legal initial
context, one `z`, one full episode, one return. The v6i9 repertoire
checkpoint remains the experimental anchor:
`final_v6i9-mapaware-repertoire-hardpool-refactor-r1-seed1_2v2.zip`.
Actor and z-specific repertoire parameters stay frozen through
`v6i9_training_stage = "router"` and `router_freeze_actor = True`.

**Resolved diff vs feedforward parent:** exactly
`{experiment_id, h_mode, latent_arc_credit_baseline,
latent_arc_credit_enabled, latent_arc_credit_min_len,
latent_entropy_anneal_end, latent_entropy_anneal_start,
latent_entropy_mode, latent_entropy_objective, latent_lam_h,
latent_lam_h_end, latent_lam_p, latent_resample_every_n,
latent_strategy_ppo_coef, learning_rate, router_ent_coef,
router_uniform_exploration_prob, run_tag, strategy_interval}`.

**Mechanism contract:** `latent_resample_every_n = 0`,
`strategy_interval = 0`, `latent_strategy_ppo_coef = 0.0`,
`latent_arc_credit_enabled = True`,
`latent_arc_credit_baseline = "running_mean"`,
`latent_arc_credit_min_len = 1`, `learning_rate = 1e-4`,
`router_uniform_exploration_prob = 0.20`, `router_ent_coef = 0.002`,
`latent_lam_h = latent_lam_h_end = 0.015`,
`latent_entropy_mode = "marginal"`, and
`latent_lam_p = 0.0`.

**Immediate smoke gates:** all four z sampled in the behavior policy,
router gradients positive, frozen actor/z hashes unchanged, episode
credit finite, running-mean baseline active, and behavior log-probs
computed under `0.8 * q_phi + 0.2 * Uniform` rather than raw q_phi.

**Five-update mechanism gates:** no deterministic z above 80 percent for
two consecutive updates, at least two argmax z values, high marginal
entropy, falling conditional entropy, MI proxy above noise, and no
exploding logit margin. Hard stop: one z reaches 100 percent argmax for
two consecutive updates.

---

## 4. Open decisions

| ID  | Question                                                                                                          | Owner action                                                                                                                                |
|-----|-------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------|
| D1  | Should the canonical entropy interpretation be conditional or marginal?                                           | Closed: v5i6 makes batch-marginal entropy canonical; v5i4/v5i5 remain conditional comparison rows. |
| D2  | The `_2m_` artifact-history suffix on the v5i4 in-flight run — rename or leave?                                   | Leave. Renaming changes embedded `run_tag` metadata; per [`latent-preset-registry.md`](latent-preset-registry.md) §7.1 we route by filename. |
| D3  | Re-launch `no_latent_v4i3_baseline` at v5i4's exact budget / seed, or rely on the v4i3-budget baseline already on disk? | Re-launch. The §1 invariants in `experiment-and-evaluation-protocol.md` require matched budget and seed for a headline comparison.           |
| D4  | If v5i6's routing-quality control (`router` vs `random-matched`) does **not** show a significant Δ, does the paper claim survive? | Open. Acceptable answers: (a) report the canonical row as `PAPER-FAITHFUL but inconclusive`; (b) escalate to an explicitly named extension and rerun. |
| D5  | Should `latent_strategy_ppo_coef = 0.10` (`c_Z`) be swept?                                                        | Open. Recorded as O3 in `summer-fidelity-rules.md` §7. A sweep is `SUMMER-COMPATIBLE EXTENSION` if any value ≠ `0.10` is used in a headline row. |
| D6  | Update [`Paper_experiment_alignment.md`](Paper_experiment_alignment.md) §3's `GLOBAL_STATE_DIM = 19` paragraph to match the current `GLOBAL_STATE_DIM = 34` / `CONTEXT_STATE_DIM = 170`. | Open (O1 in `summer-fidelity-rules.md` §7). Code is authoritative; doc paragraph needs an update note.                                       |

---

## 5. Closed decisions (for audit history)

| ID  | Decision                                                                                                                              | Date         | Where recorded                                                                                                                                                       |
|-----|----------------------------------------------------------------------------------------------------------------------------------------|--------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| C1  | v5i4 became the first operational paper-faithful conditional-entropy row; v4i3 is the arc-credit row, not the headline paper-faithful row. Superseded for canonical launch priority by C7. | 2026-06-15   | [`Paper_experiment_alignment.md`](Paper_experiment_alignment.md) §6.7. |
| C2  | The main-loop gate must trigger off `latent_router_optimizer is not None`, not off `latent_strategy_ppo_coef == 0`.                    | (pre-v5i4)   | [`Paper_experiment_alignment.md`](Paper_experiment_alignment.md) §6.4; `tests/test_marginal_baseline.py::MainLoopGatingTests`.                                       |
| C3  | The actor must read `z` only via `nn.Embedding(K, d_z)` concat in any paper-faithful row (FiLM / adapter / one-hot OFF).               | (with v5i4)  | [`summer-method-spec.md`](summer-method-spec.md) §5; `tests/test_v5i4_paper_faithful.py::V5i4ConcatOnlyActorTests`.                                                  |
| C4  | The forced-z resolver must be a pure function of `cfg` and the passed `global_step` (so resumes pick up the schedule correctly).       | (with v5i3)  | [`Paper_experiment_alignment.md`](Paper_experiment_alignment.md) §6.6; `tests/test_forced_z_anneal.py`.                                                              |
| C5  | v5i4's `run_tag` flips `_2m_` → `_1m_` so the tag advertises the actual `total_timesteps`. v5_strict_summer / v5i1 / v5i2 / v5i3 keep `_2m_` to preserve artifact paths. | 2026-06-15   | [`Paper_experiment_alignment.md`](Paper_experiment_alignment.md) §7; [`latent-preset-registry.md`](latent-preset-registry.md) §7; `tests/test_v5i4_paper_faithful.py::V5i4RunTagAndInitialOpponentConsistencyTests`. |
| C6  | In pool mode, the first env reset must use the first pool entry, not `cfg.fixed_opponent_tag` when the latter is out-of-pool.          | 2026-06-15   | `rl/train_ppo.py::_resolve_initial_opponent_and_phase`; `tests/test_v5i4_paper_faithful.py::V5i4RunTagAndInitialOpponentConsistencyTests`.                          |
| C7  | v5i6 is the canonical paper-faithful Summer interpretation: entropy protects the batch-marginal strategy repertoire while v5i4/v5i5 remain conditional-entropy comparison rows. | 2026-06-16   | [`summer-method-spec.md`](summer-method-spec.md) §8/§12; [`latent-preset-registry.md`](latent-preset-registry.md) §2/§6.9; `tests/test_v5i6_paper_faithful_marginal_entropy.py`. |
| C8  | v5i6's marginal entropy MUST be aggregated over the **full rollout** resample subset, not per-PPO-minibatch. The per-minibatch path is provably an upper bound on the intended objective by Jensen and the gap is closed by softening individual `q_phi(z\|s)` toward uniform — the conditional regression v5i6 was designed to replace. Implementation: [`rl/latent_losses.py::rollout_marginal_entropy_loss`](../rl/latent_losses.py) called once per PPO inner epoch from [`rl/custom_ppo/ppo_updater.py`](../rl/custom_ppo/ppo_updater.py); deprecated `strategy_marginal_entropy_loss` kept only for parity tests. | 2026-06-16 | [`summer-method-spec.md`](summer-method-spec.md) §8.1; `tests/test_latent_losses.py::RolloutMarginalEntropyLossTests` (Jensen demo); `tests/test_v5i6_paper_faithful_marginal_entropy.py::V5i6RolloutMarginalEntropyContractTests`. |

---

## 6. Recommended next experiments (priority-ordered)

### V6I2 staged gate protocol (frozen — confirmatory run pending)

| Step | Status |
|------|--------|
| v6i2 gate infrastructure + schedule clocks | **DONE** — 108+ gate/curriculum tests green |
| Short v6i2 smoke (wiring only) | **DONE** — `tests/test_v6i2_staged_integration.py` |
| Threshold calibration from v6i1 λ_cf=0.01 / 1.0 runs | **DONE** — frozen thresholds recorded |
| Freeze [`v6i2-gate-protocol-freeze.md`](v6i2-gate-protocol-freeze.md) | **DONE** — stale-aware bounded online gate fingerprint `224f1aea9ab36319` |
| Full fresh enforce confirmatory run (1.0M → up to 1.3M) | **PLANNED** with frozen fingerprint |

**2026-06-18 preset fix:** default `v6i2` previously inherited `latent_cf_coef_max = 0.01`
from v6i1 (weak CF). Preset now sets `1.0`. The in-flight
`v6i2_staged_team_intent_curriculum_OP5_OP6_OP7_1m_4v4` run at ~393k trained under
weak CF unless restarted with the fixed preset or `--latent-cf-coef-max 1.0`; treat it
as wiring/smoke evidence, not confirmatory strong-CF.

**2026-06-19 pairwise objective/gate refinement:** v6i2 now requires competence
before actor-CF separation, tracks per-pair hinge/weight telemetry, applies
persistent weak-pair weighting, and adds a worst-pair hinge term. Matched-seed
behavioral realization reports route, task-behavior, performance, and aggregate
components independently; normalized aggregation uses frozen scales and raw
component floors so route distance cannot carry a pass. Gate fingerprint remains
`224f1aea9ab36319`.

### V6I4 router ablation protocol (evaluation-only — pending promoted v6i2 checkpoint)

v6i4 is a Summer-plan-faithful, evaluation-only router-ablation protocol
over a frozen, Phase-A-promoted v6i2 checkpoint. It is currently
planned/pending. No parameters are trained or updated.

It is not a replacement training row. The checkpoint's actor, critic,
`q_phi`, latent repertoire, reward configuration, opponent pool, and
evaluation environment stay fixed; only the latent-selection rule changes
under matched seeds. The evaluator must reject pre-promotion checkpoints:
the checkpoint evidence must verify v6i2 lineage, Phase A promotion
`PASS`, gate fingerprint, promotion step, checkpoint hash, and valid
confirmatory gate lineage.

The locked comparison rows are `learned_qphi_switching`,
`uniform_episode_fixed`, `uniform_random_at_router_opportunities`,
`preselected_global_fixed_z`, `fixed_z0` through `fixed_z3`,
`qphi_initial_only_no_switch`, `shuffled_qphi_outputs`, and the
non-deployable posthoc oracle rows.
Success is return and win-rate
advantage over uniform, fixed, initial-only, and shuffled controls,
reported aggregate and per opponent. MI, entropy, occupancy, argmax
stability, and event-associated switching remain diagnostics.

### V6I6 repertoire expansion (implemented - pending anchor evidence)

v6i6 is an evidence-gated Expansion Stage E1 over v6i5, not an automatic
next launch. It activates only after forced-z and state-conditioned branch
evaluations produce a hashed anchor-validation manifest with
`verdict = "VALIDATED"`, selected `anchors`, one `expansion_target`, and
any `dormant` latents.

The implementation intentionally does not hardcode `z0`, `z1`, `z2`, or
`z3`. Training validation rejects `--preset v6i6` unless
`--v6i6-anchor-validation-manifest <path>` is supplied and the manifest
latents are disjoint and in range. E1 uses fixed-z episodes for outcome
attribution, a frozen reference critic for opportunity weights, no-op
adapter initialization, and the declared trainable scope
`target_embedding_gate_adapter_only`.

Required evidence before launch: finish forced-z evaluation, finish
state-conditioned branch evaluation, generate the anchor-validation
manifest with report hashes, run the 2k invariant smoke with the
manifest, then run the 25k birth diagnostic before any longer launch.

### V6I3 local communication (frozen contract — confirmatory run pending)

| Slice | Status |
|-------|--------|
| Spec owner doc | **DONE** — [`v6i3-local-communication-spec.md`](v6i3-local-communication-spec.md) |
| Slice 1 transport + unit tests | **DONE** — `rl/custom_ppo/communication/` |
| Slice 2 policy / rollout / PPO | **DONE** |
| Slice 3–6 phases / telemetry / corruption / gates | **DONE** |
| v6i3 preset + registry row | **DONE** — fingerprint `9ef168d941f046fb` |
| Full fresh v6i3 confirmatory run | **PLANNED** — pre-freeze v6i3 artifacts are exploratory only |

V6I3 must not modify active v6i1 runs or frozen v6i2 lineages. `communication_enabled=False` preserves v6i2 behavior.

Calibration uses v6i1/v6i2 evidence only. Confirmatory launch must consume the
frozen table unchanged (`--fresh-metrics-csv`, `confirmatory_gate_lineage_valid=True`,
gate fingerprint `9ef168d941f046fb`).

1. **Launch full v6i2 enforce confirmatory run** — not a shortened budget.
2. **Run v6i4 router ablation** on the accepted promoted v6i2 checkpoint.
3. **Launch full fresh v6i3 confirmatory run** after v6i2 lineage is accepted;
   do not reuse pre-freeze v6i3 metrics as official.
4. **Launch v5i7 seed 0 (§3.3) if the immediate target is the best
   Summer-faithful latent model on split-lane geometry.** Compare only to
   split-lane matched controls.
5. **Launch v5i6 seed 0 (§3.1) for the canonical open-map paper-faithful
   row.** Preserve the §1 invariants in
   `experiment-and-evaluation-protocol.md`.
5. **Run the v5i6 eval matrix and random-matched control.** Use
   `plot/eval_checkpoint.py --latent-selection router` and
   `--latent-selection random-matched` on the same checkpoint, seed, and
   episode budget.
6. **Run forced-z behavioral probes for v5i6/v5i7.** Use matched seeds across
   `z=0..K-1` before making causal behavior claims.
7. **Run v5i6 vs v5i4/v5i5 comparisons.** v5i6 vs v5i4 tests the full
   marginal-entropy switch; v5i6 vs v5i5 isolates entropy reduction at
   the same lambda_H floor.
8. **`no_latent_v4i3_baseline` re-launch at v5i6's exact budget /
   seed.** Closes D3 for the new headline `v5i6 vs no-latent`
   comparison.
9. **v5i6/v5i7 multi-seed.** Add seeds 1 and 2 only after seed 0 passes the
   router-quality and no-loss checks.

---

## 7. Cross-references

| Need                                              | Where to look                                                                       |
|---------------------------------------------------|-------------------------------------------------------------------------------------|
| Mandatory agent behavior                          | [`AGENTS.md`](../../AGENTS.md)                                                      |
| Scientific definition of the paper method         | [`summer-method-spec.md`](summer-method-spec.md)                                    |
| Fidelity rules / classification / proposal form   | [`summer-fidelity-rules.md`](summer-fidelity-rules.md)                              |
| Per-preset facts, aliases, deltas, run tags       | [`latent-preset-registry.md`](latent-preset-registry.md)                            |
| Launch / eval / statistical protocols             | [`experiment-and-evaluation-protocol.md`](experiment-and-evaluation-protocol.md)    |
| v6i2 frozen gate thresholds (pre-confirmatory)  | [`v6i2-gate-protocol-freeze.md`](v6i2-gate-protocol-freeze.md)                    |
| Code↔manuscript trace                             | [`Paper_experiment_alignment.md`](Paper_experiment_alignment.md)                    |
| Algorithm sketch                                  | [`../../docs/algorithm.md`](../../docs/algorithm.md)                                |
