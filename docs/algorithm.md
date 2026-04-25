# Algorithm

**Canonical spec** for the locked **IMPLEMENTATION DETAILS** (feature list, strategy encoder, resampling, losses, and optional §12) is the Word document *Summer Implementation Plan.docx* on the author’s machine (e.g. `c:\Users\K-B\Desktop\Summer Implementation Plan.docx`). The repository does not replace that file; it implements it. Any intentional departure must be updated in the trace, not left undocumented.

**Spec↔code trace:** [`Summer_Implementation_Plan_Implementation_Details_Trace.md`](Summer_Implementation_Plan_Implementation_Details_Trace.md). Vectorized vs nested training loops: [`rollout_semantics.md`](rollout_semantics.md).

## Active Variant

The default audit training path is now a local latent-strategy PPO/MAPPO-style implementation:

- Shared decentralized actor: one **MLP** (paper 256–256) on **flattened** per-agent `grid` + `vec` + shared latent embedding `z` (no CNN in the policy body; the grid is a flat local observation tensor).
- Strategy encoder: `q_phi(z | s)` is a 128-128 MLP over the structured 14-float `global_state`.
- Centralized critic: `CentralizedCritic` consumes the structured 14-float `global_state` plus joint-action one-hot features and `z_onehot` when latent strategy is enabled.
- Team reward: the environment returns one blue-team scalar reward per parallel env.
- Action space: independent categorical heads for `MultiDiscrete([5, 50] * N)`.
- Action masking: invalid logits are set to a large negative value before sampling, log-prob, and entropy.

This is MAPPO-style because the critic is centralized during training while the actor remains decentralized over per-agent observations plus the shared team strategy index. The old SB3 PPO/policy/buffer path has been removed from training; `rl.train_ppo.train_ppo` uses the local implementation only.

## Rollout Buffer

`rl.ppo_core.TensorDictRolloutBuffer` owns rollout storage. It is a named tensor registry:

| Field | Shape | Purpose |
| --- | --- | --- |
| `obs_grid` | `(T, B, N, 7, 20, 20)` | Local grid observations, flattened by the active actor MLP. |
| `obs_vec` | `(T, B, N, 18)` | Per-agent local vector features. |
| `obs_agent_mask` | `(T, B, N)` | Alive/active agent mask. |
| `obs_mask` | `(T, B, N * 55)` | Flattened action mask. |
| `global_state` | `(T, B, 14)` | CTDE global state for critic and `q_phi(z | s)`. |
| `actions` | `(T, B, 2 * N)` | Macro/target action pairs. |
| `log_probs` | `(T, B)` | Old policy log-probs for PPO ratios. |
| `values` | `(T, B)` | Old value estimates. |
| `next_values` | `(T, B)` | Bootstrap value for each transition. |
| `rewards` | `(T, B)` | Team reward. |
| `terminated` | `(T, B)` | Game-rule terminal flag. |
| `truncated` | `(T, B)` | Time-limit/reset flag. |
| `z` | `(T, B)` | Shared latent strategy index when enabled. |
| `prev_z` | `(T, B)` | Previous strategy used for persistence loss. |
| `z_log_probs` | `(T, B)` | Rollout-time `q_phi(z \| s_t)` log-prob for the chosen `z` (diagnostics; **not** used in the clipped PPO policy ratio). |
| `z_logits` | `(T, B, K)` | Rollout-time strategy logits (diagnostics, persistence/entropy, optional consecutive KL). |
| `z_resampled` | `(T, B)` | True when `z` was sampled from `q_phi` on that transition. |
| `z_persist_mask` | `(T, B)` | True only for non-initial strategy refreshes that should pay persistence loss. |
| `advantages` | `(T, B)` | Computed by GAE. |
| `returns` | `(T, B)` | `advantages + values`. |

Plain PPO ablations leave the latent fields unregistered and use the same buffer/GAE path.

## GAE

GAE is implemented in `rl.ppo_core.compute_gae`.

For each transition:

```text
delta_t = r_t + gamma * V(next_t) * (1 - terminated_t) - V(s_t)
adv_t = delta_t + gamma * lambda * (1 - (terminated_t OR truncated_t)) * adv_{t+1}
return_t = adv_t + V(s_t)
```

The important distinction is that time-limit truncations still bootstrap through `V(next_t)`, but they do not leak the recursive advantage across the auto-reset boundary.

## PPO Update

`PPOConfig` defaults below are the training numbers unless you set `use_stable_marl_ppo=True` (legacy override **not** in *Summer Implementation Plan.docx*; use CLI `--stable-marl` to enable).

Defaults:

| Hyperparameter | Default |
| --- | --- |
| `gamma` | `0.995` |
| `gae_lambda` | `0.99` |
| `clip_range` | `0.2` |
| `clip_range_vf` | `0.2` |
| `ent_coef` | `0.01` |
| `vf_coef` | `1.0` |
| `max_grad_norm` | `0.5` |
| `n_epochs` | `10` |
| `batch_size` | `512` |
| `learning_rate` | Linear decay from configured LR to `0`. |

Policy loss:

```text
ratio = exp(new_log_prob - old_log_prob)
L_pi = -mean(min(ratio * A, clip(ratio, 1-eps, 1+eps) * A))
```

Value loss:

```text
v_clipped = old_v + clip(new_v - old_v, -eps_v, eps_v)
L_v = mean(max((new_v - return)^2, (v_clipped - return)^2))
```

Advantages are normalized per minibatch. Entropy is computed from the masked categorical distributions. Approximate KL is logged as `mean((ratio - 1) - log(ratio))`.

## Latent Strategy Training

The Summer Implementation Plan's latent strategy components are wired into `CustomPPOTrainer`:

- `q_phi(z | s)`: reads the `global_state` rollout field, matching the plan’s **compact global feature vector** (see `rl/global_state.py` and the Word doc’s `global_features` list order).
- Strategy sampling: defaults to **once per episode** (plan Option A), with `latent_resample_every_n > 0` for sparse refresh every N steps (Option B), `latent_resample_on_flag` for optional event-based resampling (plan §12), and **`latent_resample_every_n == 1` disallowed** (plan: do not resample every decision).
- Policy conditioning: `nn.Embedding(K, d_z)` (`d_z` in {8, 16} per plan) concatenated to flattened per-agent local features; shared parameters across blue agents.
- Critic: the plan’s **notation** is often $Q(s,\mathbf a, z)$; the implementation is a **scalar, joint-action– and $z$–conditioned PPO value baseline** (same inputs, not a full tabular $Q$ over counterfactual actions).
- Decentralized execution: the actor only sees per-agent `grid` / `vec` and $z$ — not `global_state` (see `docs/environment.md` on how the grid is built per agent).
- **PPO clipped ratio uses action log-probs only**; `q_phi` is trained through **strategy entropy** and **persistence** (and optional consecutive KL from plan §12), not through the action PPO ratio—consistent with the plan’s **total** loss in §4 and IMPLEMENTATION §6.
- `z_resampled` records when a new discrete strategy was drawn from `q_phi`; persistence uses `1[z ≠ z_prev]` on non-initial refreshes.
- Persistence and strategy entropy are added to the PPO loss block (plan IMPLEMENTATION §6 form):

```text
L = L_PPO + lambda_p * L_persist - lambda_H * H(q_phi(z | s))
```

`L_persist` is masked off for the initial strategy sample and applies only to later refreshes. With `latent_resample_every_n = 0` and no event-based resampling, there are no later refreshes in an episode, so the persistence term is **zero** (as expected). **Minimization** of `L` with a **negative** `lambda_H * H` term on strategy entropy is equivalent to **rewarding** higher strategy entropy, matching the plan’s sign convention.

**Experiments (paper E3 / controls):** the decisive latent-vs-non-latent comparison should be **identical** PPO and environment settings with **only** latent strategy on vs off (`--no-latent-strategy`). Opponent tags and phases are **not** supervision targets for $z$.

## Strategy Diagnostics

Latent checkpoints expose the most recent `z` decision through `CustomPPOInferencePolicy.strategy_info()`. Evaluation rollouts and viewer headless evals aggregate that into per-episode columns:

- `strategy_switch_rate`: fraction of within-episode decisions where the chosen strategy id changed.
- `strategy_resample_rate`: fraction of decisions where the strategy encoder sampled/refreshed `z`.
- `strategy_unique_count`: number of distinct strategy ids used in the episode.
- `strategy_entropy_mean`: mean categorical entropy from `q_phi(z | s)`.
- `strategy_occupancy_0...K`: per-strategy episode occupancy.
- `strategy_phase_<phase>_occupancy_0...K`: per-strategy occupancy conditioned on coarse flag-state phases (`neutral`, `blue_attack`, `blue_defense`, `both_flags`).

Trainer checkpoints also persist rollout-level `last_stats` with strategy occupancy, dominant strategy, switch fraction, and resample fraction. These are diagnostics for collapse/specialization; they do not change the PPO objective.

## Experiment Telemetry

`train_ppo.py` writes two CSV streams unless `--no-metrics-csv` is passed:

- Per-update metrics: PPO losses, KL, entropy, rollout reward/return summaries, cumulative W/L/D, cumulative win rate, and strategy rollout diagnostics.
- Per-episode metrics: score, success, decision steps, opponent labels, and environment episode fields.

`plot/eval_checkpoint.py` evaluates any single checkpoint against OP3/OP4 or another scripted opponent list on one or more map splits and writes per-episode plus aggregate CSVs. `experiments/phase6_experiment_matrix.py` generates the reproducible final experiment commands for default latent, vanilla, persistence, K, sparse-refresh, OP2-trained comparison, and train-vs-held-out-map generalization variants.
