# Architecture

## Perception Contract

The agent reads the field as a 2D scene. Any observation tensor carrying game-field spatial structure enters the actor through `CNNEncoder`, which consumes channels-first tensors with shape `(B, C, H, W)` and returns `(B, feature_dim)`.

Linear layers after the CNN are output projections: they turn visual features into logits or values. They are not substitutes for visual perception.

## Visual Trunk

Canonical class: `AICTFProject/rl/networks.py::CNNEncoder`

- Input: `(B, C, H, W)` float tensor.
- Convolution stack: three `Conv2d -> ReLU` blocks.
- Projection: `Flatten -> Linear(feature_dim)`.
- Default channels: `32, 64, 64`.
- Default feature dim: `512`.
- Initialization: orthogonal weights with gain `sqrt(2)`, zero biases.

The active local PPO actor in `AICTFProject/rl/custom_ppo.py::SharedActorCentralizedCritic` applies one shared `CNNEncoder` to every agent-local grid:

```text
grid: (B, N, C, H, W)
          |
          v
reshape to (B*N, C, H, W)
          |
          v
CNNEncoder
          |
          v
reshape to (B, N, D)
          |
          v
concat per-agent vector features
          |
          v
concat shared z embedding (latent mode)
          |
          v
shared actor body + action heads
```

## Actor And Critic Topology

The default training path is the local PPO/MAPPO-style trainer in `AICTFProject/rl/custom_ppo.py`.

- Actor: shared per-agent CNN policy over local `grid`, local `vec`, and the shared strategy embedding when latent mode is enabled.
- Strategy encoder: `StrategyEncoder q_phi(z | s)` maps the 14-float global state to a categorical distribution over K team strategies.
- Critic: centralized MLP over the 14-float `global_state`; in latent mode its `extra` input is joint-action one-hot plus `z_onehot`.
- Trunks: the actor CNN and centralized critic MLP are separate, because the critic consumes structured CTDE state rather than spatial observations.
- Output heads: linear categorical action heads for each macro/target component.

The standalone `PPOPolicy` in `AICTFProject/rl/networks.py` keeps the same optional-conditioning interface:

```python
def forward(self, obs: torch.Tensor, extra: torch.Tensor | None = None) -> torch.Tensor:
    features = self.cnn(obs)
    if extra is not None:
        features = torch.cat([features, extra], dim=-1)
    logits = self.actor_head(features)
    return logits
```

The active trainer implements strategy conditioning directly in `SharedActorCentralizedCritic`; `PPOPolicy.extra` remains a small reusable hook for experiments.

## Centralized Critic Hook

Canonical class: `AICTFProject/rl/networks.py::CentralizedCritic`

The centralized critic is an MLP over the structured global state, not a CNN over the map and not a flattened bundle of local observations. Its interface is:

```python
def forward(self, global_state: torch.Tensor, extra: torch.Tensor | None = None) -> torch.Tensor:
    ...
```

`extra` carries joint-action and strategy-conditioning inputs in latent mode. The default global-state dimension is `14`, matching the ICRA handoff target and the active environment `state()` contract.

## Active And Dormant Paths

The old SB3 PPO/policy/buffer path has been removed from the training stack. `rl.train_ppo.train_ppo` now uses `CustomPPOTrainer`, `TensorDictRolloutBuffer`, and local PPO loss/GAE code by default and exclusively.

Latent strategy is now wired into the local PPO loop. `StrategyEncoder`, the strategy embedding path, the action-conditioned critic, sparse strategy refresh, persistence loss, and strategy entropy regularization all live in `rl/custom_ppo.py`. The pure `LatentConditionedActor` in `rl/latent_marl.py` remains as a small architecture test/reference.

Strategy telemetry is also part of the active path: `CustomPPOInferencePolicy.strategy_info()` feeds evaluation/viewer CSVs, and `CustomPPOTrainer.last_stats` records rollout-level strategy occupancy and switching diagnostics.

Final experiment tooling sits outside the trainer: `plot/eval_checkpoint.py` evaluates arbitrary checkpoints, while `experiments/phase6_experiment_matrix.py` generates the ablation/generalization command matrix. Training CSV logging is enabled by default through `PPOConfig.metrics_csv_path` and `PPOConfig.episode_csv_path`.

## Diagram

```text
                   local visual field
                  (B, N, C, H, W)
                         |
                         v
                    CNNEncoder
                         |
                         v
                per-agent visual features
                         |
                         v
             shared actor body + action heads
                         |
                         v
                  macro/target logits


             structured global state (B, 14)
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
