# Architecture

## Perception Contract

The active trainer treats each agent observation as mixed spatial-plus-scalar input. `SharedActorCentralizedCritic` encodes each per-agent `grid` tensor with a shared CNN, then concatenates that CNN embedding with the per-agent `vec` scalar features and, in latent mode, the shared strategy embedding.

This is an intentional professor-approved departure from the earlier flat-only sketch: `MLP(concat(CNN(grid_i), scalar_vec_i, strategy_embedding(z)))`. The compact `global_state` is never fed to the actor; it is used only by `q_phi(z | s)` and the centralized critic.

## Local Actor Trunk

Canonical active class: `AICTFProject/rl/custom_ppo.py::SharedActorCentralizedCritic`

- Input: `grid` with shape `(B, N, C, H, W)` plus `vec` with shape `(B, N, V)`.
- Per-agent visual trunk: reshape to `(B*N, C, H, W)` and run a shared `CNNEncoder`.
- Concatenation: CNN feature vector + local scalar vector features + optional shared `z` embedding.
- Actor body: `Linear -> ReLU -> Linear -> ReLU -> Linear` with 256 hidden units.
- Parameters are shared across agents.

The active local PPO actor therefore uses CNN + MLP:

```text
grid: (B, N, C, H, W)
          |
          v
shared CNNEncoder over (B*N, C, H, W)
          |
          v
concat per-agent scalar vector features
          |
          v
concat shared z embedding (latent mode)
          |
          v
shared actor body + action heads
```

## Actor And Critic Topology

The default training path is the local PPO/MAPPO-style trainer in `AICTFProject/rl/custom_ppo.py`.

- Actor: shared per-agent CNN over local `grid`, followed by an MLP over CNN features, local `vec`, and the shared strategy embedding when latent mode is enabled.
- Strategy encoder: `StrategyEncoder q_phi(z | s)` maps the 19-float global state to a categorical distribution over K team strategies.
- Critic: centralized MLP over the 19-float `global_state`; in latent mode its `extra` input is joint-action one-hot plus `z_onehot`.
- Trunks: the actor CNN/MLP and centralized critic MLP are separate, because the actor consumes local observations while the critic consumes structured CTDE state.
- Output heads: linear categorical action heads for each macro/target component.

The active trainer uses the reusable `CNNEncoder` from `AICTFProject/rl/networks.py` directly:

```python
cnn_features = self.actor_cnn(grid.reshape(batch * n_agents, c, h, w))
actor_in = torch.cat([cnn_features, scalar_vec, z_emb], dim=-1)
logits = self.actor_head(self.actor_body(actor_in))
```

`PPOPolicy.extra` remains a small reusable hook for experiments.

## Centralized Critic Hook

Canonical class: `AICTFProject/rl/networks.py::CentralizedCritic`

The centralized critic is an MLP over the structured global state, not a CNN over the map and not a flattened bundle of local observations. Its interface is:

```python
def forward(self, global_state: torch.Tensor, extra: torch.Tensor | None = None) -> torch.Tensor:
    ...
```

`extra` carries joint-action and strategy-conditioning inputs in latent mode. The default global-state dimension is `19`, preserving the original handoff fields and appending score/clock pressure for CTDE value prediction.

## Active And Dormant Paths

The old SB3 PPO/policy/buffer path has been removed from the training stack. `rl.train_ppo.train_ppo` now uses `CustomPPOTrainer`, `TensorDictRolloutBuffer`, and local PPO loss/GAE code by default and exclusively.

Latent strategy is wired into the local PPO loop. `StrategyEncoder`, the strategy embedding path, the action-conditioned critic, sparse strategy refresh, persistence loss, and strategy entropy regularization all live in `rl/custom_ppo.py`. The pure `LatentConditionedActor` in `rl/latent_marl.py` remains as a small flat-actor architecture test/reference.

Strategy telemetry is also part of the active path: `CustomPPOInferencePolicy.strategy_info()` feeds evaluation/viewer CSVs, and `CustomPPOTrainer.last_stats` records rollout-level strategy occupancy and switching diagnostics.

Final experiment tooling sits outside the trainer: `plot/eval_checkpoint.py` evaluates arbitrary checkpoints, while `experiments/phase6_experiment_matrix.py` generates the ablation/generalization command matrix. Training CSV logging is enabled by default through `PPOConfig.metrics_csv_path` and `PPOConfig.episode_csv_path`.

## Diagram

```text
                   local observation grid
                  (B, N, C, H, W)
                         |
                         v
                  shared CNNEncoder
                         |
                         v
         concat scalar vec + optional z embedding
                         |
                         v
              shared MLP actor + action heads
                         |
                         v
                  macro/target logits


             structured global state (B, 19)
                         |
                         v
        StrategyEncoder q_phi(z | s)
                         |
                         v
               shared strategy index z
                         |
                         v
       actor z embedding + critic z_onehot


             global state + joint actions + z
                         |
                         v
                  CentralizedCritic
                         |
                         v
                 centralized value
```
