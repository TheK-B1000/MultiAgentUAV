# V6I7: Summer-Faithful Recurrent Router

## Context

V6 router produces near-uniform logits and switching does not improve success. The primitives already exist (`strategy_ppo_loss`, `persistence_loss`, `marginal_entropy_loss`, `RecurrentSelectorCell`), but several are likely wired incorrectly: PPO applied at every step instead of only at opportunities; GRU hidden state stored-and-detached rather than BPTT-trained; EMA context and GRU competing; persistence calculated over held steps rather than decisions. This plan audits, fixes, and validates the wiring before another long run. V6I6 lineage is not overwritten.

**Iteration lineage:**
- **V6I7**: Recurrent router + corrected strategy-timescale credit (this document)
- **V6I8**: Marginal-coverage ablation (`latent_coverage_coef > 0`)
- **V6I9**: Causal evaluation

**Architecture:**
```
s_{0:t} → GRU (one update/env-step) → h_t
[s_t, h_t] → q_φ → z_j ~ Categorical    (sampled only at opportunity j)
z_j → π_θ^i(a_t^i | o_t^i, z_j)         (shared z, 4 agents, held between opportunities)
team reward → centralized advantage → {π_θ, q_φ, Q_ω} jointly
```

---

## Five Locked Decisions

| # | Decision | Locked value |
|---|---|---|
| 1 | Recurrent training | Sequence minibatches with BPTT; no random shuffle of router sequences |
| 2 | Router PPO & persistence scope | Applied only at actual strategy opportunity indices |
| 3 | Temporal path | GRU-only; EMA context stack disabled for q_φ |
| 4 | Resampling triggers | Fixed sparse cadence (main run); event-driven is ablation only |
| 5 | Promotion gate | Must beat best fixed-z on success (primary) with noninferiority per-opponent; forced-z behavioral separation required; history-destruction quantified; entropy/occupancy as validity checks only |

---

## Objective

```
L = L_actor_PPO
  + latent_strategy_ppo_coef * L_router_PPO       (opportunities only)
  + vf_coef * L_value
  + latent_lam_p * L_persist                       (opportunity transitions only, cross-buffer)
  - ent_coef * H_actor
  - router_entropy_coef * E_j[H(q_φ(·|s̃_j))]     (conditional; opportunities only)
```

`router_entropy_coef` is a separate coefficient from `latent_lam_h`. Coverage (`latent_coverage_coef * KL(q̄ ‖ U)`) is absent in V6I7 and introduced in V6I8.

**Smoke assertion:** `latent_loss` components are exactly:
- router PPO loss
- persistence loss
- router conditional entropy

Not present: coverage KL, phase prediction, return prediction, counterfactual separation, consecutive KL.

Starting hyperparameters:
```yaml
latent_strategy_ppo_coef: 0.10
latent_lam_p: 0.02
router_entropy_coef: 0.005        # conditional; opportunities only
latent_coverage_coef: 0.0         # V6I7: coverage absent; V6I8 will sweep this
recurrent_selector_hidden_dim: 64
recurrent_seq_len: 32
recurrent_burn_in: 8
strategy_interval: 32
latent_resample_on_flag: false
router_context_mode: "current"    # EMA disabled for q_φ
router_gae_lambda: 0.95           # GAE over strategy opportunities
```

---

## Implementation Work

### A. Sequence Minibatch Training for GRU

**First step: audit** `rl/custom_ppo/update/minibatch_updater.py`. If the current PPO update randomly shuffles transitions, the GRU receives a zero or stored-but-detached hidden state per sample and learns nothing temporal.

**New file: `rl/custom_ppo/update/sequence_minibatch.py`**

Each chunk carries:
```
global_state sequence            [chunk_len, N_env, state_dim]
episode_start_mask / done        [chunk_len, N_env]
selected_z                       [chunk_len, N_env]        index of active latent (held or freshly chosen)
router_decision_valid            [chunk_len, N_env, bool]  True only at actual strategy opportunities
critic_transition_valid          [chunk_len, N_env, bool]  True whenever a valid Q target exists
interval_end_reason              [chunk_len, N_env]        next_opportunity|terminated|truncated|buffer_cut
ended_at_router_opportunity      [chunk_len, N_env, bool]  True if boundary coincides with a router event
old router log-probability       [chunk_len, N_env]        only meaningful when router_decision_valid
old router logits                [chunk_len, N_env, K]     only meaningful when router_decision_valid
initial GRU hidden state         [N_env, d_h]
prev_router_state                [N_env, state_dim]        (cross-buffer: raw global state s_{j-1})
prev_router_hidden               [N_env, d_h]              (cross-buffer: GRU hidden h_{j-1})
prev_opportunity_valid           [N_env]                   (whether cross-buffer predecessor exists)
prev_z_selected                  [N_env]                   (index of last selected latent)
prev_episode_id                  [N_env]                   (to detect episode boundary)
rewards, critic values, bootstrap value
```

**Key invariant:** `N_router_PPO == N_actual_router_decisions`. Mid-hold continuation segments (where `router_decision_valid=False`) must contribute zero to router PPO, router entropy, opportunity count, persistence count, and old-router log-probability count. They may contribute to critic training.

**Burn-in:** Total chunk = 40 steps. Burn-in prefix = 8 (update hidden state, respect done masks, no loss, no gradients). Loss-bearing = 32. Config explicit:
```yaml
recurrent_seq_len: 32      # loss-bearing
recurrent_burn_in: 8       # warm-up prefix
```
Actor minibatch loop remains randomly shuffled. Only the router uses sequence batches.

### B. Single GRU Update Per Environment Step

`h_t ∈ R^{N_env × d_h}`, not `R^{N_env × N_agents × d_h}`. GRU called once with team-level global state. z_t broadcast to all 4 agents.

**Episode boundary resets (distinct from value bootstrap):**
```
episode_boundary = terminated OR truncated    → reset h_t, prev_opp_valid, prev_router_state
terminated (not truncated)                    → bootstrap value = 0
truncated (not terminated)                    → bootstrap normally from Q_ω
```

Partial resets: `h_t ← h_t ⊙ (1 - episode_boundary[:, None])`.

**New unit test: `tests/test_recurrent_selector.py`**
- h_t.shape == (N_env, hidden_dim) (one GRU transition per env-step, not per agent)
- Terminated env: h resets to zero; bootstrap = 0
- Truncated env: h resets to zero; bootstrap from Q_ω
- Ordinary step: implementation matches GRU cell exactly:
  ```python
  expected_h = gru_cell(global_state, previous_h)
  actual_h   = selector.update_hidden(global_state, previous_h)
  assert torch.allclose(actual_h, expected_h)
  assert not torch.allclose(actual_h, torch.zeros_like(actual_h))
  ```
- Resetting one environment must not affect others:
  ```python
  assert torch.allclose(actual_h[unaffected_envs], expected_h[unaffected_envs])
  ```
- z broadcast identical for all agents in same env

| Event | Bootstrap | GRU reset | Persistence reset |
|---|---|---|---|
| True terminal | Zero | Yes | Yes |
| Time-limit truncation | From Q_ω | Yes | Yes |
| Ordinary step | From Q_ω | No | No |

### C. Router-Value Design

Reuse the shared centralized critic `Q_ω(s_j, z_j)` — **state-only input, no GRU hidden state** — for both B0 and B1. This ensures the value estimator does not receive features available only in B1, preserving the B0/B1 comparison as a clean isolation of recurrent routing.

```
B0 router:   q_φ(z_j | s_j)             (MLP, state only)
B1 router:   q_φ(z_j | s_j, h_j)        (GRU-augmented)
Both critic: Q_ω(s_j, z_j)              (state + latent, no history)
```

**Scheduler phase in critic input (required for Markov property):** Mid-hold continuation segments make `Q_ω(s_cut, z_j)` training valid only if `s` contains enough scheduler information to distinguish `(same state, z, 1 step to next opportunity)` from `(same state, z, 25 steps to next opportunity)`. Confirm that the global state already includes one of:
- `steps_until_next_opportunity`
- `strategy_age` (steps since last decision)
- normalized interval progress `ρ_t = steps_remaining / strategy_interval`

This input must be identical for B0 and B1 — critic parity is intact as long as the phase variable is part of the shared global state and not derived from the GRU hidden state. If the existing global state does not contain an unambiguous phase variable, add one before coding begins.

**Router credit — opportunity-level GAE with state-value baseline:**

Each critic segment `m` (which may be a full interval or one side of a buffer-crossing split) accumulates its own local return:
```
R_m^seg = Σ_{k=0}^{d_m-1} γ^k r_{t_m+k}       segment return (not full interval return)
```
Critic target: `Y_m^Q = R_m^seg + γ^{d_m} · C_{m+1}^old`. Router PPO and GAE only apply to segments where `router_decision_valid = True`. Do not accumulate full interval rewards in both segment 1 and segment 2 of a buffer-crossing split — that would double-count.

```
R_j^z ≡ R_m^seg when router_decision_valid = True

V_j^old = Σ_z q_φ_old(z | x_j) · Q_ω_old(s_j, z)           state-value baseline (marginalized)
          where x_j = s_j (B0) or [s_j, h_j] (B1)

C_{j+1}^old = continuation bootstrap (see table below)

δ_j = R_j^z + γ^{d_j} · C_{j+1}^old - V_j^old              TD residual

A_j = δ_j + γ^{d_j} · λ_z · c_{j+1} · A_{j+1}              GAE backward over opportunities
```

**Critical:** subtract `V_j^old` (state-value), not `Q_ω_old(s_j, z_j)` (action-value). If `Q(s,z)` is used as the baseline, the residual `δ_j → 0` as the critic improves, and the router stops receiving a preference signal between good and bad latent choices. The marginalized `V_j^old` preserves that signal.

**Continuation bootstrap `C_{j+1}^old` depends on how the interval ended:**

| Boundary | `C_{j+1}^old` | c (GAE continues) |
|---|---|---|
| Actual next opportunity in same batch | `V_j+1^old = Σ_z q_φ_old(z\|x_{j+1}) Q_ω_old(s_{j+1},z)` | 1 |
| True terminal | 0 | 0 |
| Truncation exactly at opportunity | `V^old(s_final)` | 0 |
| Truncation mid-hold (z_j still active) | `Q_ω_old(s_final, z_j)` | 0 |
| Buffer cut exactly at opportunity | `V^old(s_cut)` | 0 |
| Buffer cut mid-hold (z_j still active) | `Q_ω_old(s_cut, z_j)` | 0 |

When the interval ends at a buffer cut mid-hold, the latent `z_j` was never released — marginalizing over a fresh router decision would assume an action that never occurred. Use the action-conditioned `Q_ω_old(s_cut, z_j)` instead. Cross-buffer GAE is not implemented in V6I7 — document this explicitly in code.

**Decision vs. continuation validity flags:**

A buffer cut mid-hold must not create a second router PPO sample for a decision that has already been logged. The rollout produces two segments from one router decision:

```
Segment 1 (steps t_j … t_cut, in buffer N):
    router_decision_valid = True     ← this is the sole PPO sample
    critic_transition_valid = True
    end_reason = buffer_cut
    ended_at_router_opportunity = False
    bootstrap = Q_old(s_cut, z_j)

Segment 2 (steps t_cut … t_{j+1}, in buffer N+1):
    router_decision_valid = False    ← continuation; no PPO, no entropy, no persistence
    critic_transition_valid = True
    selected_z = z_j                 ← same latent, still held
```

The next real router opportunity begins a new record with `router_decision_valid = True`.

**Buffer-crossing unit test:**
```python
# One strategy interval crosses a rollout boundary
assert opportunity_count == 1
assert router_ppo_sample_count == 1
assert router_entropy_sample_count == 1
assert critic_transition_count == 2

assert continuation_segment.router_decision_valid is False
assert continuation_segment.critic_transition_valid is True
assert continuation_segment.router_loss == 0.0
assert continuation_segment.entropy_loss == 0.0
assert continuation_segment.persistence_loss == 0.0
```

Old router log-probability and old router logits are only valid (and only used) when `router_decision_valid = True`.

**Critic training** uses segment notation — indexed by valid critic segments `m`, not just router decisions `j`:
```
Y_m^Q = R_m^seg + γ^{d_m} · C_{m+1}^old
L_Q   = E_{m: critic_transition_valid}[(Q_ω(s_m, z_m) - stopgrad(Y_m^Q))²]
```
Router GAE is indexed by actual decisions `j`. Critic TD learning is indexed by valid segments `m`. These are distinct clocks — the critic trains on every valid segment; PPO trains only on decision segments.

Q(s,z) estimates each latent's value; V(s,x) = Σ_z q(z|x)Q(s,z) provides the GAE baseline. These are two uses of the same critic, not two separate heads.

**Mask and bootstrap unit tests:**
```python
# True terminal
assert C == 0.0 and gae_continuation_mask == 0
# Truncation mid-hold
assert torch.allclose(C, critic(final_state, latent_id=selected_z)) and gae_continuation_mask == 0
# Buffer cut mid-hold (same latent continues)
assert torch.allclose(C, critic(cut_state, latent_id=selected_z)) and gae_continuation_mask == 0
# Actual next opportunity (router may select again)
assert torch.allclose(C, sum(old_router_probs[z] * critic(next_state, latent_id=z) for z in range(K)))
assert gae_continuation_mask == 1
```

**Advantage-direction unit test (correct setup):** To verify `A(s,z) = Q(s,z) - V(s,x)`, construct frozen returns that equal the action values. Zero reward alone does not produce this; the synthetic setup must be explicit:

```python
router_probs = torch.tensor([0.5, 0.5])
q_values     = torch.tensor([2.0, 0.0])
state_value  = (router_probs * q_values).sum()   # 1.0

# Frozen targets equal the action values (e.g., one-step with matching rewards+bootstraps)
advantage_z0 = q_values[0] - state_value   # +1.0
advantage_z1 = q_values[1] - state_value   # -1.0

assert advantage_z0 > advantage_z1
assert torch.allclose(
    torch.stack([advantage_z0, advantage_z1]),
    q_values - state_value,
)
```

This catches the exact failure mode where using `Q(s,z_j)` as the baseline produces `Q - Q = 0` for the selected latent.

**Counterfactual critic-vector test (unit test):**
```python
q_values_by_z = torch.stack(
    [critic(next_state, latent_id=z) for z in range(num_latents)], dim=-1
)
assert q_values_by_z.shape[-1] == num_latents
assert torch.isfinite(q_values_by_z).all()
```

**Lock targets before PPO epochs:** Compute all `V_j^old`, `Y_j^Q`, `δ_j`, and `A_j` once from old-policy and old-critic snapshots. Detach and normalize valid opportunity advantages. Hold fixed across all PPO update epochs.

```yaml
router_gae_lambda: 0.95
```

Log per update: `router_interval_return_mean`, `router_advantage_mean/std`, `router_advantage_explained_variance`.

Do not add a separate router critic head until the shared one is shown insufficient.

### D. Router PPO Scope Audit

Open `strategy_objectives.py:205–232` and `latent_losses.py:357–398`. Replace all uses of `opportunity_mask` with `router_decision_valid` — the canonical mask for all router-gated losses. Add assert:
```python
assert router_ppo_loss[~router_decision_valid].abs().sum() == 0
router_loss = router_ppo_loss[router_decision_valid].mean()
```
Log `router_ppo_sample_count` per update; must equal `router_decision_count`. Both must equal each other:
```python
assert router_ppo_sample_count == router_entropy_sample_count == router_decision_count
```

### E. Persistence Scope and Count — Cross-Buffer Accounting

Persistence compares consecutive router *decisions* only. The correct invariant:
```
N_persist = Σ_j 1[previous opportunity exists for same env and same episode]
```

**State carried across rollout boundaries (per env):**
```
prev_router_state    [N_env, state_dim]   raw global state s_{j-1} at last opportunity (NOT s̃; hidden stored separately)
prev_router_hidden   [N_env, d_h]         GRU hidden h_{j-1} at last opportunity (outside current BPTT chunk → already detached)
prev_z_selected      [N_env]              index of last selected latent
prev_episode_id      [N_env]              to detect episode boundary
prev_opp_valid       [N_env, bool]        whether a valid predecessor exists
```
At update time: `q_prev = q_φ(prev_router_state, prev_router_hidden)`. Do not concatenate `prev_router_hidden` into `prev_router_state` — they are fed as separate inputs to q_φ.

Reset all five on `episode_boundary = terminated OR truncated`. At rollout start, initialize from persisted cross-buffer state and pass into sequence chunk header.

**Continuation segments must not overwrite persistence state.** Only update `prev_router_*` fields when `router_decision_valid = True`:
```python
if router_decision_valid:
    update_previous_router_decision_state()
# else: leave prev_router_state, prev_router_hidden, prev_z_selected,
#       prev_episode_id, prev_opp_valid unchanged
```
Unit test:
```python
prev_before = clone(prev_router_state)
process_continuation_segment()
assert torch.equal(prev_router_state, prev_before)
assert persistence_sample_count == 0
```

**Persistence loss at update time:** recompute the previous distribution under current parameters, then detach:
```python
q_prev = stopgrad(q_φ(z | prev_router_state, prev_router_hidden))   # current params, detached
persist_loss = 1 - Σ_z q_φ(z | s̃_j) · q_prev(z)
```
Apply only where `prev_opp_valid` is true and no episode boundary separates j-1 from j. Because `prev_router_hidden` is outside the current BPTT window, it is already detached by construction; the explicit `stopgrad` makes this contract visible.

Log separately for in-buffer and cross-buffer pairs:
```
in_buffer_persist_sample_count
cross_buffer_persist_sample_count
in_buffer_expected_switch_prob
cross_buffer_expected_switch_prob
```

**Cross-buffer approximation (document in code):** At update time, `q_prev` is recomputed under the *current* q_φ parameters but using `prev_router_hidden` from a *rollout-time* GRU state. This creates a mixed snapshot: current parameters, old hidden state. Because `q_prev` is detached and serves only as the persistence target (not a gradient source), this is an acceptable V6I7 approximation. If cross-buffer persistence metrics look anomalous, the exact fix is to carry sufficient pre-boundary burn-in context and reconstruct the predecessor hidden state under current GRU weights — that complexity is deferred beyond V6I7.

### F. Disable Auxiliary Supervised Losses

```yaml
latent_strategy_aux_predict_phase_coef: 0.0
latent_strategy_aux_return_coef: 0.0
latent_cf_separation_coef: 0.0
latent_kl_consecutive: 0.0
latent_coverage_coef: 0.0
```
Assert as described in the Objective section.

### G. Router Entropy — Conditional Only (V6I7)

Router entropy is computed and trained using `router_decision_valid` — the canonical mask:
```python
router_entropy = H(q_φ(·|s̃_j))[router_decision_valid].mean()   # conditional; per-decision
entropy_loss   = -router_entropy_coef * router_entropy
```

`h_mode: "conditional"` and `router_entropy_coef: 0.005` together implement this. V6I8 adds `latent_coverage_coef > 0` to this base; the two coefficients remain separate throughout the lineage.

**Unit test for sign:**
```python
assert entropy_loss_uniform < 0.0
assert abs(entropy_loss_collapsed) < tolerance
assert entropy_loss_uniform < entropy_loss_collapsed
```
- Uniform router distribution: H ≈ log(K); entropy loss contribution maximally negative
- Collapsed router distribution: H ≈ 0; entropy loss contribution ≈ 0
- Optimizer minimizes loss, so the more-negative uniform contribution drives exploration — correct direction.

Track and log in V6I7:
```
H(q_φ(·|s̃_j))     mean conditional entropy at opportunity steps
H(q̄)              marginal entropy (logged only; not trained)
I(Z;S)             MI proxy (logged only; supporting diagnostic, not a gate)
N_eff              exp(H(q̄)) — effective latent count
```

### H. History-Destruction Controls and MI Gate

The permutation-invariant shuffle `q_φ(z|s_t, h_permuted)` is not a valid MI null because MI is permutation-invariant over complete (s,h) pairs. Use genuine history-destruction controls instead.

**Three controlled conditions — offline evaluation on matched states:**

Collect a dataset `D = {(s_t, h_t)}` from a standard normal rollout. Then evaluate the three distributions on the **same collected states** offline — do not re-run trajectories under corrupted history, as that would conflate the effect of different states visited with the effect of different hidden states:

```
q_t^normal    = q_φ(z | s_t, h_t)         standard (from rollout)
q_t^zero      = q_φ(z | s_t, 0)           same state, zeroed hidden
q_t^mismatch  = q_φ(z | s_t, h_σ(t))     same state, mismatched hidden
```

For σ (the mismatch permutation): use a **derangement** (no fixed points) matched on timestep index, episode boundary status, and opportunity validity mask. This prevents accidental self-pairs and avoids pairing wildly incompatible hidden states from different game phases.

Compute from matched-state distributions:
```
mi_normal                     I(Z;S) from {q_t^normal}
mi_zero_history               I(Z;S) from {q_t^zero}
mi_mismatched_history         I(Z;S) from {q_t^mismatch}
router_jsd_normal_vs_zero     E_t[JSD(q_t^normal, q_t^zero)]
router_jsd_normal_vs_mismatch E_t[JSD(q_t^normal, q_t^mismatch)]
```

**Separate intervention rollouts** (for `G_normal - G_mismatch`): run full episodes under normal vs. mismatched history and compare team success rates. These are distinct from the matched-state MI/JSD diagnostics above.

**History-sensitivity gate (primary, V6I7-B):**
```
router_jsd_normal_vs_mismatch > τ_h
```
τ_h set from intra-run noise floor. This directly measures whether the GRU hidden state changes router output. If B1 fails this gate, the GRU is not using temporal context and V6I8 must not launch.

**MI as supporting diagnostic (not a gate):**
```
mi_normal > mi_mismatched_history + δ_MI
```
Reported for scientific record. Does not independently promote or stop B1.

Do not use a simple global-state permutation as the null.

### I. Dynamic q_φ Input Dimension

All dimensions are resolved at runtime — do not hardcode. If the scheduler-phase audit (Section C) requires adding a phase feature to the global state, `GLOBAL_STATE_DIM` changes and the critic input layer changes too:

```python
global_state_dim = global_state_space.shape[-1]   # resolved at runtime

q_phi_input_dim = (
    global_state_dim
    if router_type == "mlp"
    else global_state_dim + recurrent_selector_hidden_dim
)

assert critic.state_input_dim == global_state_dim
assert strategy_encoder.input_dim == q_phi_input_dim
```

Save to run manifest:
```
global_state_dim
scheduler_phase_feature          (name or None)
scheduler_phase_index            (index in global_state, or None)
router_input_dim
critic_input_dim
```

If the phase feature must be added, both the critic and the router encoder input layers may change, requiring explicit fresh initialization for those layers during checkpoint loading.

**Scheduler-phase unit test (do not leave as documentation only):**
```python
state_early = make_state(steps_until_next_opportunity=25)
state_late  = make_state(steps_until_next_opportunity=1)
assert not torch.equal(
    critic.encode_state(state_early),
    critic.encode_state(state_late),
)
```
This proves the critic can distinguish residual-interval states even before training produces differing values.

Disable EMA for q_φ via `router_context_mode: "current"`.

### J. Checkpoint Initialization

V6I7 is a new run with incompatible router and potentially critic input dimensions (resolved at runtime; recorded in run manifest). Actor from compatible baseline; GRU, q_φ, and any resized critic input layers freshly initialized. Log every loaded, skipped, and newly initialized parameter by name. If warm-starting, compare against matched non-recurrent control warm-started from same checkpoint.

### K. Checkpoint Reload Verification

CPU path: bitwise where practical. GPU path: `allclose(atol=1e-5)`. Compare router logits, actor logits, critic value, next hidden state with fixed input + fixed RNG state.

---

## Behavioral Validation (H1 Gate)

Calibrate noise floor: measure JSD of same-latent action distributions across training seeds, repeated evaluations, nearby checkpoints. Set `τ_noise = Q_95(JSD(z_i^{run_a}, z_i^{run_b}))`.

**Latent alignment for cross-seed comparison:** Latent IDs are permutation-symmetric — z0 in one run may correspond to z3 in another. Before computing cross-seed JSD, align latent identities using **Hungarian matching** over a behavioral cost (e.g., matched-observation policy JSD). Compute cross-seed variability from aligned pairs. Within a single checkpoint or across repeated reloads of the same checkpoint, no alignment is necessary.

Require for ≥2 (z_i, z_j) pairs:
```
JSD(z_i, z_j) > max(0.05, τ_noise)
```
AND ≥1 trajectory-level tactical difference with paired CI excluding zero (team dispersion, attacker/defender count, flag-event frequency).

---

## Post-Hoc Response Matrix (H2 Evaluation)

Evaluation artifact only — not a training signal. Supplement per-opponent fixed-z sweeps (`router_ablation.py`) with phase-conditioned branches using existing `counterfactual_z_swap_eval.py` pattern: restore same env/RNG state, run forced z0–z3, compare outcomes. Contexts identified from existing telemetry (phase_id, flag_possession, carrier distances) — no new training labels.

---

## Experimental Ladder

### V6I7-A: Plumbing Smoke Test (50k steps)

Pass all:
- One GRU update per env step (unit test)
- Terminated env: h resets, bootstrap=0; truncated env: h resets, bootstrap from Q_ω (unit test)
- `router_ppo_sample_count == router_entropy_sample_count == router_decision_count` (assert + log)
- Persistence sample count matches cross-buffer invariant (assert + log)
- Latent loss components: exactly router PPO + persistence + conditional entropy (assert)
- GRU and q_φ gradient norms finite and nonzero
- Router conditional entropy finite; 0 ≤ H(q_φ) ≤ log(K) + tolerance; entropy loss term has correct sign (negative, pulls toward uniform)
- Coverage metrics (H(Z), I(Z;S), N_eff) all finite and non-NaN (no threshold)
- Checkpoint reload reconstructs outputs (tolerant allclose)

### V6I7-B: Routing Repair Ablation (200k steps each)

| Run | Router | Input | router_entropy_coef | latent_coverage_coef |
|---|---|---|---|---|
| B0 | MLP | s_t | 0.005 | 0.0 |
| B1 | GRU | [s_t, h_t] | 0.005 | 0.0 |

**All other hyperparameters identical between B0 and B1.** B0 receives all wiring repairs (opportunity-only PPO, action-value bootstrap, cross-buffer persistence, conditional entropy, same strategy_interval, same actor init, same seeds). The only architectural difference is `s_t` vs `[s_t, h_t]` as router input.

Select on success_rate (primary), win_margin.

**B1 proceeds to full training only if all three conditions hold:**

1. **History changes router decisions** (sensitivity):
   `JSD_normal_vs_mismatch > τ_h` (τ_h from intra-run noise floor)

2. **Correct history is not strategically harmful** (performance vs. corrupted):
   `LCB_95%(G_normal - G_mismatch) > -0.03` where `G = success_router - success_fixed`

3. **B1 is noninferior to B0** (performance vs. non-recurrent):
   `LCB_95%(Success_B1 - Success_B0) > -0.03`

All three margins are predeclared before looking at validation results.

MI comparison (`mi_normal > mi_mismatched + δ_MI`) is supporting evidence only — high MI does not by itself promote B1.

| B-stage outcome | Interpretation |
|---|---|
| B1 uses history and improves success | Temporal memory is useful |
| B1 uses history and is noninferior to B0 | Continue; evidence is preliminary |
| B1 uses history but hurts success | History is learned but strategically harmful |
| B1 does not use history (JSD gate fails) | Recurrent path adds no meaningful information |

If any gate fails, stop. If all pass, lock hyperparameters on validation seeds and proceed to full training.

### V6I7-C: Full Training

B1 configuration, only if all three B-stage gates pass: history-sensitivity (`JSD > τ_h`), correct-versus-corrupted performance (`LCB > -0.03`), and B0/B1 noninferiority (`LCB > -0.03`). ≥3 training seeds.

### V6I7-D: Confirmatory Evaluation

Untouched matched test seeds (50–100 per opponent). Full v6i4 protocol:
- best fixed z (selected on validation seeds, not test seeds: `z_best = argmax_z Success_validation(z)`, then frozen for test evaluation)
- uniform episode-fixed, uniform sparse switching
- learned recurrent q_φ (V6I7), learned non-recurrent q_φ (B0 — repaired MLP, primary comparator)
- zero-history, mismatched-history, no-switch, posthoc oracle
- historical V6 router as additional reference baseline (not the primary non-recurrent comparison)

**Promotion gate (all required):**
1. Pooled: learned recurrent q_φ beats best fixed-z; paired 95% CI lower bound on success diff > 0
2. Per-opponent noninferiority: for each opponent o, `LCB_95%(Δ_o) > -0.03` where `Δ_o = success_router_o - success_fixed_o`. Margin locked before test set is unblinded.
3. History-destruction quantified: `LCB_95%(G_normal - G_mismatched) > 0` (tightened from exploratory `-0.03` to `0` at confirmatory evaluation). Primary control is mismatched history; zero-history reported separately.
4. Forced-z behavioral separation confirmed above noise floor (H1 gate)
5. Hyperparameters locked on validation seeds; test set used once

**Outcome taxonomy:**
- Full pass: H1 + H2 + H3 (strategies exist, contextually useful, routed effectively)
- H1/H2 pass, H3 fail: repertoire exists, selector can't exploit it
- H1 fail: no meaningful repertoire
- H1 pass, H2 fail: behavioral differences not strategically useful

---

## Files to Create or Modify

| File | Action |
|---|---|
| `rl/custom_ppo/update/sequence_minibatch.py` | **New**: contiguous-chunk sampler with burn-in; cross-buffer state (prev_router_state, prev_router_hidden, prev_opp_valid, prev_episode_id) |
| `rl/custom_ppo/update/minibatch_updater.py` | **Modify**: add sequence-minibatch path for router |
| `rl/custom_ppo/update/strategy_objectives.py` | **Audit + fix**: router-decision-masked PPO (assert via `router_decision_valid`); action-value bootstrap with terminal/truncation handling; router conditional entropy at decision indices only |
| `rl/custom_ppo/update/latent_losses.py` | **Audit + fix**: cross-buffer persistence with recomputed prev dist; stopgrad explicit; log sample counts; assert no coverage loss |
| `rl/latent_marl.py:146–164` | **Verify**: RecurrentSelectorCell reset on episode_boundary (terminated OR truncated); h_t indexed by N_env not N_env×N_agents |
| `rl/custom_ppo/policy.py:286–294` | **Modify**: q_phi_input_dim derived dynamically; assert; EMA disabled for q_φ; router_entropy_coef wired separately from latent_coverage_coef |
| `rl/custom_ppo/rollout_collector.py` | **Verify + extend**: h_t once per env-step; old router log-prob stored at collection; carry cross-buffer state (5 fields); terminated vs. truncated flags distinct |
| `rl/eval/history_destruction_eval.py` | **New**: zero-history, mismatched-history conditions; compute mi_normal, mi_zero_history, mi_mismatched_history, router_jsd metrics |
| `tests/test_recurrent_selector.py` | **New**: GRU-per-env-step; terminal/truncation/ordinary reset table; cross-buffer persistence count; router conditional-entropy sign |
| V6I7 preset config | **New**: all locked hyperparameters including router_entropy_coef and latent_coverage_coef: 0.0 |
