# Summer Method Spec — canonical scientific specification

**Owner:** This file is the single source of truth for the scientific
definition of the latent team-strategy method. Other docs in the repository
must link to it rather than redefining "paper-faithful" independently.

**Scope:** This document fixes (a) the locked paper requirements,
(b) the implementation choices made by this repository, and
(c) the experiment-specific hyperparameters used in the canonical
operational run. Each section labels which of the three categories its
claims fall under.

> **Read also:**
> [`AGENTS.md`](../../AGENTS.md) ·
> [`summer-fidelity-rules.md`](summer-fidelity-rules.md) ·
> [`latent-preset-registry.md`](latent-preset-registry.md) ·
> [`experiment-and-evaluation-protocol.md`](experiment-and-evaluation-protocol.md) ·
> [`research-progress-tracker.md`](research-progress-tracker.md) ·
> [`Paper_experiment_alignment.md`](Paper_experiment_alignment.md) ·
> [`../../docs/algorithm.md`](../../docs/algorithm.md) ·
> [`../../docs/Summer_Implementation_Plan_Implementation_Details_Trace.md`](../../docs/Summer_Implementation_Plan_Implementation_Details_Trace.md)

---

## 1. Paper title and research claim

**Locked paper requirement.** The work targeted is

> *Latent Team Strategy–Aware Multi-Agent Reinforcement Learning for
> Autonomous CTF*

with the central claim that an explicit shared discrete latent
team-strategy variable, learned end-to-end from task reward, improves
multi-agent CTF coordination over a same-everything-no-latent baseline.

The two operational sub-claims this repository is designed to support
or falsify:

1. The actor's policy is non-trivially conditioned on `z` (forced-`z`
   changes behavior; latents separate behaviorally inside comparable
   contexts).
2. The learned router beats a same-architecture random-matched router on
   task return inside the comparison protocol locked in
   [`experiment-and-evaluation-protocol.md`](experiment-and-evaluation-protocol.md).

---

## 2. Dec-POMDP formulation (locked)

* `N` cooperating blue agents; per-agent local observation `o_i` (a
  channels-first grid plus a per-agent scalar vector); team reward `r_t`.
* Centralized training, decentralized execution (CTDE).
* Discrete macro action space; the implementation uses `MultiDiscrete`
  per-agent heads (`5 × 50`) — see
  [`docs/algorithm.md`](../../docs/algorithm.md) for the exact action
  space and masking rules.

The Dec-POMDP itself is not part of the latent-strategy contract; the
constraint `paper-faithful` enforces is that decentralized execution
must not see centralized state.

---

## 3. Latent strategy variable (locked)

* `z ∈ {0, 1, …, K − 1}`, **discrete**, **categorical**.
* `K` is fixed; the canonical operational value is `K = 4`
  (`PPOConfig.latent_k = 4`).
* `z` is **shared** across all `N` cooperating blue agents within an
  episode (or within a sparse refresh interval; see §7).
* `z` carries **no hard-coded semantic meaning**. No "attack," "defense,"
  "escort," "stall," or role labels are assigned to specific `z` indices.
  Latent meanings, if any, are emergent from task reward.
* `z = 0` and `z = K − 1` are not privileged.

Implementation reference:
[`rl/networks.py::LatentConditionedActor`](../rl/networks.py),
[`rl/custom_ppo/policy.py::SharedActorCentralizedCritic`](../rl/custom_ppo/policy.py),
[`rl/config/ppo_config.py::PPOConfig.latent_k`](../rl/config/ppo_config.py).

---

## 4. Strategy inference network `q_phi(z | s_t)` (locked architecture, implementation choice on input)

### Locked

* `q_phi` outputs categorical logits over `K` strategy IDs.
* `q_phi` is trained from task reward and exploration regularizers only.
  It receives **no supervised strategy targets**, **no opponent ID**,
  **no handcrafted phase label**, and **no role label**.
* The categorical strategy PPO term, when enabled, is treated as part of
  `L_MARL`, not as an auxiliary prediction task.

### Implementation choice (verified in this repository)

* `q_phi` is the `StrategyEncoder` MLP `Linear → ReLU → Linear → ReLU →
  Linear (logits)` with hidden width `128` (`PPOConfig.latent_strategy_hidden = 128`),
  no custom initialization. Source:
  [`rl/networks.py::StrategyEncoder`](../rl/networks.py),
  [`rl/latent_marl.py::StrategyEncoder`](../rl/latent_marl.py).
* `q_phi`'s input is the **temporal context vector** of dimension
  `CONTEXT_STATE_DIM = 5 · GLOBAL_STATE_DIM`. With the current
  `GLOBAL_STATE_DIM = 34` (see [`rl/global_state.py`](../rl/global_state.py)),
  this is `170`. The temporal context is `[raw_state ‖ ema_short ‖
  ema_long ‖ raw_state − ema_short ‖ raw_state − ema_long]`, computed by
  [`rl/latent_marl.py::TemporalStateTracker`](../rl/latent_marl.py) at
  centralized-training time only.
* `q_phi` is asserted to receive `CONTEXT_STATE_DIM` rows by
  [`SharedActorCentralizedCritic._assert_input_contracts`](../rl/custom_ppo/policy.py).

### Discrepancy with prior documentation

[`Paper_experiment_alignment.md`](Paper_experiment_alignment.md) §3
states `GLOBAL_STATE_DIM = 19` and that `q_phi` consumes the 19-d global
summary. The current code uses `GLOBAL_STATE_DIM = 34` and `q_phi`
consumes the 170-d **temporal** context. This document treats the code
as authoritative; the older paragraph in `Paper_experiment_alignment.md`
should be updated. It is recorded as an open item in
[`summer-fidelity-rules.md`](summer-fidelity-rules.md) §"Open /
unresolved."

---

## 5. Decentralized actor `π_θ(a_i | o_i, z)` (locked architecture)

### Locked

* The actor is shared across blue agents.
* The actor receives only:
  * local observation features (CNN over `obs_grid`),
  * per-agent scalar features (`obs_vec`),
  * a learned discrete strategy embedding for the **current** `z`.
* The actor must **not** receive: global state, opponent ID,
  handcrafted phase ID, centralized team labels, or future information.

### Implementation choice (verified in this repository)

* The literal paper-faithful conditioning mechanism is plain
  `nn.Embedding(K, d_z)` concatenated to the local CNN+vec features.
  FiLM, adapter, and one-hot conditioning are post-Summer extensions
  (see §10).
* For the canonical operational preset (`v5i6_paper_faithful_marginal_entropy`),
  the resolved actor input width is

  ```text
  CNN features (actor_cnn_feature_dim) = 128
  per-agent scalar vector              =  20
  z embedding (latent_z_embed_dim)     =  16
  -----------------------------------------
  actor_input_dim                      = 164
  ```

  This is pinned by
  [`tests/test_v5i4_paper_faithful.py::V5i4ConcatOnlyActorTests`](../tests/test_v5i4_paper_faithful.py).
  Source:
  [`rl/custom_ppo/policy.py::SharedActorCentralizedCritic._assert_input_contracts`](../rl/custom_ppo/policy.py).
* Hidden body: 256–256 ReLU MLP (`actor_hidden_dim = 256` by default).
  See [`rl/networks.py::LatentConditionedActor`](../rl/networks.py).

### Forbidden in the literal paper-faithful row

* `enable_actor_z_film = True`
* `latent_actor_z_adapter_enabled = True`
* `latent_actor_z_onehot_enabled = True`

These flags must all be `False` for any preset classified as
`PAPER-FAITHFUL`.

---

## 6. Centralized critic `V_φ(s, a, z)` (locked)

### Locked

* The critic may access global state, joint actions, and the selected
  latent strategy.
* The critic is a **scalar value function**, not a Q-function over
  counterfactual joint actions.
* Centralized information must not leak into decentralized execution.

### Implementation choice

* `CentralizedCritic` ([`rl/networks.py`](../rl/networks.py)) is a
  128–128 MLP to a scalar.
* When latent is on, the critic input is `concat(temporal_context,
  joint_action_onehot, z_onehot)`. Otherwise it is `global_state`.
* Total critic input width when latent is on:

  ```text
  CONTEXT_STATE_DIM (=170)
  + sum(action_dims)
  + latent_k (=4)
  ```

  Asserted in `SharedActorCentralizedCritic._assert_input_contracts`.

---

## 7. Resampling cadence (locked but adjustable)

### Locked

* `z` is selected at episode start.
* Mid-episode resampling, if any, must be **sparse**. Per-step resampling
  is forbidden (`PPOConfig` enforces `latent_resample_every_n != 1`).
* Event-triggered hard switching (e.g. flag-event) is not part of the
  literal method unless the paper specification is formally revised.

### Implementation choice for the canonical preset

* `latent_resample_every_n = 64` decisions
  (`PPOConfig.latent_resample_every_n = 64` for v5i4 / v5i5 / v5i6).
* `latent_resample_on_flag = False`.
* `latent_event_refresh_enabled = False`.
* `latent_sparse_tactical_refresh_enabled = False`.

This is a sparse-refresh implementation of the locked "episode start
plus optional sparse refresh" rule. See
[`Paper_experiment_alignment.md`](Paper_experiment_alignment.md) §2 for
the exact `z_resampled` / `z_persist_mask` semantics.

---

## 8. Operational objective (canonical paper-faithful row)

### Locked form

The minimized loss for the canonical operational paper-faithful row is

```text
L = L_actor_PPO
  + c_V · L_critic
  + c_Z · L_strategy_PPO
  + λ_p · L_persist
  − λ_H · H(q_bar)

q_bar(z) = E_s[q_phi(z | s)]
```

This clarifies the originally underspecified `H(z)` notation as entropy
of the aggregate strategy distribution over the router training batch.
The canonical row therefore protects the global strategy repertoire while
allowing confident state-conditioned router decisions. The prior
mean-conditional interpretation, `E_s[H(q_phi(z | s))]`, remains preserved
as the v5i4 row, with v5i5 as its higher-floor ablation.

### Component definitions and gradient paths

| Term                | Definition                                                        | Gradient into `q_phi`?                          | Source |
|---------------------|-------------------------------------------------------------------|-------------------------------------------------|--------|
| `L_actor_PPO`       | Standard clipped PPO on action log-probs.                         | No (the actor consumes a sampled `z`; gradient stops at the embedding lookup). | [`rl/ppo_core.py::ppo_policy_loss`](../rl/ppo_core.py) |
| `c_V · L_critic`    | Clipped value loss on `V_φ(s, a, z)`.                             | No.                                             | [`rl/ppo_core.py::ppo_value_loss`](../rl/ppo_core.py) |
| `c_Z · L_strategy_PPO` | Clipped categorical PPO on `q_phi`, on-policy, evaluated only on the resample subset. `c_Z = latent_strategy_ppo_coef = 0.10` for v5i4. | **Yes — this is the task-reward channel for `q_phi`.** | [`rl/latent_losses.py::strategy_ppo_loss`](../rl/latent_losses.py) |
| `λ_p · L_persist`   | Soft persistence: `1 − p_φ(z_t = z_{t-1} | s_t)`, averaged over `z_persist_mask` (mid-episode resample steps only). `λ_p = 0.03`. | Yes (regularizer).                              | [`rl/latent_losses.py::strategy_persistence_loss`](../rl/latent_losses.py) |
| `−λ_H · H(q_bar)`   | Rollout-level marginal entropy maximization over expected router probabilities at every resample-decision point in the rollout. Implemented as `KL(q_bar \|\| Uniform)` minimization where `q_bar = N⁻¹ Σ_i q_phi(z\|s_i)` is computed in a single forward pass over the **full rollout resample subset** (not per-PPO-minibatch — see §8.1). `λ_H` follows a 0.003 → 0.001 anneal over 0..300k for v5i6. | Yes (regularizer). | [`rl/latent_losses.py::rollout_marginal_entropy_loss`](../rl/latent_losses.py) |

`L_strategy_PPO` is

```text
L_strategy_PPO = − E_{t ∈ resample_mask}[
                       min( ρ_t(z) · A_t,
                            clip(ρ_t(z), 1 ± clip_range) · A_t ) ]
ρ_t(z)  = π_phi(z_t | s_t) / π_phi_old(z_t | s_t)
A_t     = GAE advantage from V_φ(s_t, a_t, z_t)
```

where the resample subset is the set of timesteps where a new `z` was
sampled (episode-start draws and 64-step refreshes). The non-resample
subset receives exactly zero gradient through this term — pinned by
[`tests/test_v5i4_paper_faithful.py::V5i4RouterTaskGradientTests`](../tests/test_v5i4_paper_faithful.py).

### Operational coefficients (v5i4 / v5i5 / v5i6)

The v5i6 row is the canonical paper-faithful operational preset. v5i4
is the preserved conditional-entropy interpretation, and v5i5 is the
higher conditional-entropy-floor ablation. All three rows keep the same
actor, critic, router PPO, persistence, resampling, opponent pool, and
no-label/no-curriculum contract.

| Field                          | v5i4 value    | v5i5 value    | v5i6 value    | Notes |
|--------------------------------|---------------|---------------|---------------|-------|
| `latent_k`                     | `4`           | `4`           | `4`           | Discrete shared latent. |
| `latent_z_embed_dim`           | `16`          | `16`          | `16`          | Embedding width. |
| `latent_resample_every_n`      | `64`          | `64`          | `64`          | Sparse refresh cadence. |
| `latent_lam_p`                 | `0.03`        | `0.03`        | `0.03`        | Persistence regularizer. |
| `latent_lam_h`                 | `0.003`       | `0.003`       | `0.003`       | Initial entropy weight. |
| `latent_lam_h_end`             | `0.0002`      | `0.001`       | `0.001`       | v5i6 uses the v5i5 entropy floor. |
| `latent_entropy_anneal_start`  | `0`           | `0`           | `0`           | Anneal window start. |
| `latent_entropy_anneal_end`    | `300_000`     | `300_000`     | `300_000`     | Anneal window end. |
| `latent_entropy_mode`          | `"conditional"` | `"conditional"` | `"marginal"` | v5i6 replaces mean conditional entropy with batch-marginal entropy. |
| `latent_entropy_objective`     | `"maximize"`  | `"maximize"`  | `"maximize"`  | Sign of the entropy term in the minimized loss. |
| `latent_strategy_ppo_coef`     | `0.10`        | `0.10`        | `0.10`        | `c_Z`. |
| `latent_episode_strategy_ppo`  | `False`       | `False`       | `False`       | v5i1 extension OFF. |
| `latent_episode_strategy_lr`   | `None`        | `None`        | `None`        | No dedicated router optimizer. |
| `latent_arc_credit_enabled`    | `False`       | `False`       | `False`       | v3i19 extension OFF. |
| `enable_actor_z_film`          | `False`       | `False`       | `False`       | v5i2 extension OFF. |
| `latent_actor_z_adapter_enabled` | `False`     | `False`       | `False`       | OFF. |
| `latent_actor_z_onehot_enabled`| `False`       | `False`       | `False`       | OFF. |
| `latent_forced_z_episode_frac` | `0.0`         | `0.0`         | `0.0`         | v5i3 curriculum OFF. |
| `latent_usage_balance_coef`    | `0.0`         | `0.0`         | `0.0`         | v5i6's marginal entropy is driven by `lambda_H`, not the legacy episode-router usage coefficient. |
| `latent_router_distill_enabled`| `False`       | `False`       | `False`       | v4i4post extension OFF. |
| `latent_strategy_aux_return_head` | `False`    | `False`       | `False`       | Auxiliary head OFF. |
| `latent_strategy_aux_predict_phase_coef` | `0.0` | `0.0`     | `0.0`         | Auxiliary head OFF. |
| `latent_v3i3_event_preference_enabled` | `False` | `False`    | `False`       | Preference learning OFF. |
| `latent_preference_coef`       | `0.0`         | `0.0`         | `0.0`         | Preference learning OFF. |

The v5i6 resolved-config diff against v5i4 is exactly
`{latent_entropy_mode, latent_lam_h_end, run_tag}`. Its diff against
v5i5 is exactly `{latent_entropy_mode, run_tag}`. These contracts are
pinned by
[`tests/test_v5i6_paper_faithful_marginal_entropy.py`](../tests/test_v5i6_paper_faithful_marginal_entropy.py).

### 8.1. Aggregation unit for the v5i6 marginal entropy loss

The marginal entropy loss

```text
L_marg = λ_H · KL( q_bar || Uniform )
q_bar(z) = N⁻¹ · Σ_{i=1..N} q_phi(z | s_i)
```

is taken over the **full set of router-decision points in the rollout**
(every `s_i` such that `z_resampled[i] = True`), not the per-PPO-minibatch
subset.

The choice is a correctness one, not a performance optimization. KL is
convex in `q_bar`, so by Jensen

```text
E_B[ KL(q_bar_B || U) ] ≥ KL( E_B[q_bar_B] || U ) = KL( q_bar_rollout || U )
```

with equality iff every minibatch's marginal `q_bar_B` equals the rollout
marginal. The per-minibatch loss is therefore a strict upper bound on the
intended objective whenever the per-minibatch marginals differ; the bias
is closed by the gradient softening individual `q_phi(z | s_i)` rows
toward uniform — the conditional-entropy regression v5i6 was designed to
replace. With the production rollout (`32 envs × 2048 steps × 1/64
resample cadence ≈ 1024` resample-decision points split across 64
PPO minibatches of ~16 resample samples each), the gap is large enough
to materially affect specialization. See
[`tests/test_latent_losses.py::RolloutMarginalEntropyLossTests::test_jensen_demo_per_minibatch_upper_bounds_rollout_level`](../tests/test_latent_losses.py)
for a 4-z, one-hot-per-minibatch demonstration where the per-minibatch
mean KL is `ln K = 1.386` while the rollout-level KL is exactly `0`.

Implementation:

* The trainer
  ([`rl/custom_ppo/ppo_updater.py`](../rl/custom_ppo/ppo_updater.py))
  gathers the rollout-level resample states once at the start of every
  PPO inner epoch, runs a single differentiable forward pass through
  `model.strategy_logits`, calls
  [`rl/latent_losses.py::rollout_marginal_entropy_loss`](../rl/latent_losses.py),
  and adds the resulting tensor to the **first minibatch's** `latent_loss`
  for that epoch. Subsequent minibatches in the same epoch contribute zero
  to the marginal term — its gradient was already realized by that
  optimizer step. With `n_epochs = 6` the marginal entropy term thus
  fires 6 times per PPO update, each time against fresh `q_phi`
  parameters, against the rollout-level `q_bar`.
* The deprecated per-minibatch helper
  [`rl/latent_losses.py::strategy_marginal_entropy_loss`](../rl/latent_losses.py)
  is retained only for parity tests and as a baseline. Production v5i6
  must not invoke it; the audit banner emits
  `entropy maximization: ON (mode=marginal, aggregation=rollout, ...)`
  on every run header.
* Telemetry (`router_rollout_soft_*` columns) reports
  `H(q_bar)`, `mean_i H(q_phi(·|s_i))`, their difference (the MI proxy),
  the per-z entries of `q_bar`, and the soft-`argmax` occupancy
  distribution from the same forward pass — all over the same rollout
  resample subset, so the three-pattern decoder
  (high marg / low cond = "broad and state-specific";
   high marg / high cond = "broad but indecisive";
   low marg / low cond = "global collapse")
  is computable from a single CSV row.

The labels for sampled-z empirical occupancy
(`latent_marginal_entropy_nats`, `latent_occupancy_min/max/ratio`,
`effective_num_latents`) are intentionally distinct from the
`router_rollout_soft_*` family above; the former are one-sample-per-state
empirical histograms over the categorical samples, the latter are soft
expectations over the differentiable router. Both are emitted every
update so cross-checks are immediate.

---

## 9. Decentralized execution boundary

* The actor body and its forward pass receive only the local CNN
  features, the local scalar vector, and the discrete `z` index.
* Global state, opponent ID, joint actions, and centralized critic
  features must never reach the actor at decision time.
* This invariant is enforced by
  [`SharedActorCentralizedCritic._assert_input_contracts`](../rl/custom_ppo/policy.py)
  (raises `AssertionError` if the actor input width changes).

---

## 10. Forbidden mechanisms (literal paper-faithful row)

A preset labeled `PAPER-FAITHFUL` must not enable any of the following.
See [`summer-fidelity-rules.md`](summer-fidelity-rules.md) for the full
machine-scannable list and decision tree.

* Reconstruction or VAE losses on `q_phi`.
* Gumbel-Softmax `z`.
* Supervised strategy labels (`latent_strategy_aux_predict_phase_coef > 0`,
  any role labels).
* Phase-prediction or return-prediction auxiliary heads
  (`latent_strategy_aux_return_head = True`,
  `latent_strategy_aux_predict_phase_coef > 0`).
* Preference learning (`latent_preference_coef > 0`,
  `latent_preference_commit_coef > 0`,
  `latent_v3i3_event_preference_enabled = True`).
* Advantage-weighted router distillation (`latent_awrd_enabled = True`).
* Episode-level external credit extension
  (`latent_episode_strategy_ppo = True` *with* a dedicated
  `latent_episode_strategy_lr`).
* Forced-z curriculum (`latent_forced_z_episode_frac > 0` or any of the
  four anneal fields set).
* Hand-designed strategic rewards or per-z handcrafted bonuses.
* Hard-coded role/strategy meanings (interpreting `z` indices as
  "attack," "defense," etc., in code or documentation).
* Hard-coded switching rules (event-triggered hard switches).
* FiLM, adapter, or one-hot actor conditioning. `nn.Embedding(K, d_z)`
  concat is the only allowed conditioning.
* Options or hierarchical sub-controllers.

---

## 11. What the method is *not*

* It is not a hierarchical RL method. There is no manager / sub-policy
  decomposition.
* It is not a contrastive representation method. There is no separation
  loss in the literal row.
* It is not a preference / RLHF method. No human or scripted teacher
  shapes `q_phi`.
* It is not a curriculum method. The literal row uses no scripted
  exploration schedule for `z`.
* `q_phi` is not a classifier; its targets are not labels. Any preset
  that adds supervised targets is, by definition, a different method.

---

## 12. Canonical preset and aliases

The canonical preset for the operational paper-faithful method is
**`v5i6_paper_faithful_marginal_entropy`**, registered in
[`rl/presets/__init__.py::PRESET_REGISTRY`](../rl/presets/__init__.py).
Its aliases (all resolve to the same `PPOConfig`):

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

The preserved conditional-entropy interpretation is
**`v5i4_paper_faithful_end_to_end`**. The higher-floor conditional
ablation is **`v5i5_paper_faithful_entropy_floor`**. In result tables,
use:

```text
v5i4  = paper-faithful conditional-entropy interpretation
v5i5  = higher conditional-entropy-floor ablation
v5i6  = canonical paper-faithful marginal-entropy interpretation
```

The literal-strict ablation (`L = L_PPO + λ_p · L_persist − λ_H · H(q_phi)`,
**no** `L_strategy_PPO`) is `v5_strict_summer` (function
`apply_plan_faithful_latent_v5_strict_summer`). It exists to test
whether the paper's literal equation alone is sufficient to train
`q_phi`. See [`latent-preset-registry.md`](latent-preset-registry.md)
for the full ladder.

---

## 13. Assumptions and unresolved ambiguities

These are the items where the paper specification, prior repository
docs, and the current code do not unambiguously agree. They are
consolidated into [`summer-fidelity-rules.md`](summer-fidelity-rules.md)
§"Open / unresolved" and must be resolved before any "the paper says"
claim is made about them.

1. **`q_phi` input dimension.** Code path uses 170-d temporal context
   (`CONTEXT_STATE_DIM = 5 · 34`). [`Paper_experiment_alignment.md`](Paper_experiment_alignment.md)
   §3 still references a 19-d global summary. Decision deferred to the
   author; the current contract is the temporal context.
2. **Persistence form.** The locked equation in `docs/algorithm.md` uses
   `1[z_t ≠ z_{t-1}]`; the implementation uses the soft form
   `1 − p_φ(z_t = z_{t-1} | s_t)` for the gradient (and the hard form
   for diagnostics). Both are consistent with the locked spec, but only
   the soft form transmits gradient into `q_phi`. See
   [`Paper_experiment_alignment.md`](Paper_experiment_alignment.md) §6.1.
3. **`c_Z` operational value.** `latent_strategy_ppo_coef = 0.10` for
   v5i4 is an implementation choice; the paper does not lock a specific
   numerical coefficient. The choice is documented inside
   `apply_plan_faithful_latent_v5i4_end_to_end`.
4. **Resample cadence.** `latent_resample_every_n = 64` is an
   implementation choice for v5i4 / v5_strict_summer / v4i3; the paper
   only requires "sparse." The choice is documented inside the preset
   and pinned by tests.
5. **Reward shaping.** Trainer-side and env-side reward shaping is
   inherited from upstream presets (`v4i1`, `v4i3`); the literal paper
   row does not specify shaping coefficients. If any shaping field is
   nonzero, that's an implementation choice that may matter for the
   "task-reward only" claim. Inspect the resolved
   `env_*` and `reward_shaping_*` fields per run.
6. **Entropy reduction.** The paper notation `H(z)` did not specify
   whether entropy is reduced per state and then averaged, or computed
   on the aggregate strategy distribution. This repository now treats
   the batch-marginal interpretation as canonical (`v5i6`) because it
   discourages global latent collapse without forcing uncertainty inside
   every state. The conditional interpretation is retained as v5i4/v5i5.

---

## 14. Cross-references

| Need                                              | Where to look |
|---------------------------------------------------|---------------|
| Mandatory agent behavior, pre-/post-change rules  | [`AGENTS.md`](../../AGENTS.md) |
| Auditable fidelity checklist & classification     | [`summer-fidelity-rules.md`](summer-fidelity-rules.md) |
| Per-preset facts, aliases, and resolved deltas    | [`latent-preset-registry.md`](latent-preset-registry.md) |
| Evaluation, statistical protections, comparisons  | [`experiment-and-evaluation-protocol.md`](experiment-and-evaluation-protocol.md) |
| Current research status                           | [`research-progress-tracker.md`](research-progress-tracker.md) |
| Code↔manuscript trace per loss / channel          | [`Paper_experiment_alignment.md`](Paper_experiment_alignment.md) |
| Original spec-to-code trace                       | [`../../docs/Summer_Implementation_Plan_Implementation_Details_Trace.md`](../../docs/Summer_Implementation_Plan_Implementation_Details_Trace.md) |
| Algorithm sketch                                  | [`../../docs/algorithm.md`](../../docs/algorithm.md) |
