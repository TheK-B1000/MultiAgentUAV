# Algorithm

## Active Variant

The default audit training path is now a local PPO/MAPPO-style implementation:

- Shared decentralized actor: one CNN-based actor is shared across blue agents and consumes each agent's local `grid` plus local `vec`.
- Centralized critic: `CentralizedCritic` consumes the structured 14-float `global_state`.
- Team reward: the environment returns one blue-team scalar reward per parallel env.
- Action space: independent categorical heads for `MultiDiscrete([5, 50] * N)`.
- Action masking: invalid logits are set to a large negative value before sampling, log-prob, and entropy.

This is MAPPO-style because the critic is centralized during training while the actor remains decentralized over per-agent observations. The old SB3 PPO/policy/buffer path has been removed from training; `rl.train_ppo.train_ppo` uses the local implementation only.

## Rollout Buffer

`rl.ppo_core.TensorDictRolloutBuffer` owns rollout storage. It is a named tensor registry:

| Field | Shape | Purpose |
| --- | --- | --- |
| `obs_grid` | `(T, B, N, 7, 20, 20)` | CNN local visual observations. |
| `obs_vec` | `(T, B, N, 18)` | Per-agent local vector features. |
| `obs_agent_mask` | `(T, B, N)` | Alive/active agent mask. |
| `obs_mask` | `(T, B, N * 55)` | Flattened action mask. |
| `global_state` | `(T, B, 14)` | CTDE global state for critic and future `q_phi(z | s)`. |
| `actions` | `(T, B, 2 * N)` | Macro/target action pairs. |
| `log_probs` | `(T, B)` | Old policy log-probs for PPO ratios. |
| `values` | `(T, B)` | Old value estimates. |
| `next_values` | `(T, B)` | Bootstrap value for each transition. |
| `rewards` | `(T, B)` | Team reward. |
| `terminated` | `(T, B)` | Game-rule terminal flag. |
| `truncated` | `(T, B)` | Time-limit/reset flag. |
| `advantages` | `(T, B)` | Computed by GAE. |
| `returns` | `(T, B)` | `advantages + values`. |

Future latent fields such as `z`, `prev_z`, `z_logits`, or `z_resampled` are added by one `register_field(...)` call without changing sampling logic.

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

## Summer Plan Hooks

The Summer Implementation Plan's latent strategy components plug into this local path as follows:

- `q_phi(z | s)`: reads the `global_state` rollout field.
- Strategy sampling: happens during rollout collection before actor action selection.
- Policy conditioning: pass `z` through the existing policy `extra`/embedding hook when Phase 4/7 moves latent strategy onto the local trainer.
- Critic conditioning: pass joint-action one-hot plus `z` through `CentralizedCritic.forward(global_state, extra=...)`.
- Persistence and strategy entropy losses: add to the PPO loss block in `CustomPPOTrainer.update`.
