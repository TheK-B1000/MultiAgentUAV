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
shared actor body + action heads
```

## Actor And Critic Topology

The default training path is the local PPO/MAPPO-style trainer in `AICTFProject/rl/custom_ppo.py`.

- Actor: shared per-agent CNN policy over local `grid` and local `vec`.
- Critic: centralized MLP over the 14-float `global_state`.
- Trunks: the actor CNN and centralized critic MLP are separate, because the critic consumes structured CTDE state rather than spatial observations.
- Output heads: linear categorical action heads for each macro/target component.

The standalone `PPOPolicy` in `AICTFProject/rl/networks.py` keeps the future strategy-conditioning interface:

```python
def forward(self, obs: torch.Tensor, extra: torch.Tensor | None = None) -> torch.Tensor:
    features = self.cnn(obs)
    if extra is not None:
        features = torch.cat([features, extra], dim=-1)
    logits = self.actor_head(features)
    return logits
```

During the audit, `extra` is dormant. The future strategy embedding can be passed through this hook without rewriting the policy class.

## Centralized Critic Hook

Canonical class: `AICTFProject/rl/networks.py::CentralizedCritic`

The centralized critic is an MLP over the structured global state, not a CNN over the map and not a flattened bundle of local observations. Its interface is:

```python
def forward(self, global_state: torch.Tensor, extra: torch.Tensor | None = None) -> torch.Tensor:
    ...
```

`extra` is reserved for future joint-action and strategy-conditioning inputs. The default global-state dimension is `14`, matching the ICRA handoff target and the active environment `state()` contract.

## Active And Dormant Paths

The old SB3 PPO/policy/buffer path has been removed from the training stack. `rl.train_ppo.train_ppo` now uses `CustomPPOTrainer`, `TensorDictRolloutBuffer`, and local PPO loss/GAE code by default and exclusively.

The latent-strategy module is pure PyTorch scaffolding only during the audit: `StrategyEncoder`, `LatentConditionedActor`, and the differentiable persistence proxy remain available for the Summer/ICRA implementation, but latent training is intentionally not wired into the PPO loop yet.

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
                  CentralizedCritic
                         |
                         v
                 centralized value

Future ICRA additions:
  z embedding -> actor extra/embedding path
  joint actions + z -> CentralizedCritic.forward(..., extra=critic_extra)
```
