# Architecture

## Perception Contract

The active Summer-plan trainer treats each agent observation as a flat local feature vector. The `grid` tensor still has channels-first spatial shape, but `SharedActorCentralizedCritic` flattens it and concatenates it with the per-agent `vec` features and, in latent mode, the shared strategy embedding.

This matches the Word implementation sketch: `MLP(concat(local_observation_i, strategy_embedding(z)))`. The compact `global_state` is never fed to the actor; it is used only by `q_phi(z | s)` and the centralized critic.

## Local Actor Trunk

Canonical active class: `AICTFProject/rl/custom_ppo.py::SharedActorCentralizedCritic`

- Input: `grid` with shape `(B, N, C, H, W)` plus `vec` with shape `(B, N, V)`.
- Per-agent flattening: `grid.reshape(B, N, C*H*W)`.
- Concatenation: flattened grid + local vector features + optional shared `z` embedding.
- Actor body: `Linear -> ReLU -> Linear -> ReLU -> Linear` with 256 hidden units.
- Parameters are shared across agents.

The active local PPO actor therefore uses an MLP, not a CNN:

```text
grid: (B, N, C, H, W)
          |
          v
flatten to (B, N, C*H*W)
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

- Actor: shared per-agent MLP over flattened local `grid`, local `vec`, and the shared strategy embedding when latent mode is enabled.
- Strategy encoder: `StrategyEncoder q_phi(z | s)` maps the 14-float global state to a categorical distribution over K team strategies.
- Critic: centralized MLP over the 14-float `global_state`; in latent mode its `extra` input is joint-action one-hot plus `z_onehot`.
- Trunks: the actor MLP and centralized critic MLP are separate, because the actor consumes local observations while the critic consumes structured CTDE state.
- Output heads: linear categorical action heads for each macro/target component.

The standalone `CNNEncoder` / `PPOPolicy` in `AICTFProject/rl/networks.py` remains available for experiments and unit coverage, but it is not the active Summer-plan training actor:

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

Latent strategy is now wired into the local PPO loop. `StrategyEncoder`, the strategy embedding path, the action-conditioned critic, sparse strategy refresh, persistence loss, and strategy entropy regularization all live in `rl/custom_ppo.py`. The pure `LatentConditionedActor` in `rl/latent_marl.py` remains as a small architecture test/reference. `CNNEncoder` remains a reusable visual-policy component, but the active plan-faithful path flattens the grid and uses an MLP.

Strategy telemetry is also part of the active path: `CustomPPOInferencePolicy.strategy_info()` feeds evaluation/viewer CSVs, and `CustomPPOTrainer.last_stats` records rollout-level strategy occupancy and switching diagnostics.

Final experiment tooling sits outside the trainer: `plot/eval_checkpoint.py` evaluates arbitrary checkpoints, while `experiments/phase6_experiment_matrix.py` generates the ablation/generalization command matrix. Training CSV logging is enabled by default through `PPOConfig.metrics_csv_path` and `PPOConfig.episode_csv_path`.

## Diagram

```text
                   local observation grid
                  (B, N, C, H, W)
                         |
                         v
                      flatten
                         |
                         v
         concat vec + optional z embedding
                         |
                         v
              shared MLP actor + action heads
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
