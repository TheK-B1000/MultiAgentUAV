# Architecture

## Perception Contract

The agent reads the field as a 2D scene. Any observation tensor that carries game-field spatial structure must enter the policy through `CNNEncoder`, which consumes channels-first tensors with shape `(B, C, H, W)` and returns `(B, feature_dim)`.

Linear layers after the CNN are output projections: they turn visual features into logits or values. They are not substitutes for visual perception.

## Visual Trunk

Canonical class: `AICTFProject/rl/networks.py::CNNEncoder`

- Input: `(B, C, H, W)` float tensor.
- Convolution stack: three `Conv2d -> ReLU` blocks.
- Projection: `Flatten -> Linear(feature_dim)`.
- Default channels: `32, 64, 64`.
- Default feature dim: `512`.
- Initialization: orthogonal weights with gain `sqrt(2)`, zero biases.

The SB3 adapter `AICTFProject/rl/ctf_cnn_extractor.py::TokenizedCombinedExtractor` applies one shared `CNNEncoder` to every agent-local grid in a tokenized observation:

```text
grid: (B, M, C, H, W)
          |
          v
reshape to (B*M, C, H, W)
          |
          v
CNNEncoder
          |
          v
reshape to (B, M, D)
          |
          v
concat per-agent vector features
          |
          v
SB3 PPO output heads
```

## Actor And Critic Topology

Current vanilla PPO training uses the SB3 `MaskedMultiInputPolicy` adapter in `AICTFProject/rl/train_ppo.py`. For this baseline, the visual trunk is shared by the policy and value paths through SB3's shared feature extractor. This is the explicit Phase 1 choice:

- Shared trunk: lower memory and aligns with the existing SB3 path.
- Actor head: linear action projection after the shared feature extractor.
- Value head: linear value projection after the shared feature extractor.

The standalone `PPOPolicy` in `AICTFProject/rl/networks.py` documents and tests the forward interface required by the upcoming ICRA work:

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

`extra` is reserved for future joint-action and strategy-conditioning inputs. The default global-state dimension is `14`, matching the ICRA handoff target. Phase 2 must still update the active environment global state, which currently reports a padded 32-dimensional vector with 18 used fields.

## Active And Dormant Paths

Vanilla PPO is the default training path during the audit. The existing latent-strategy code remains importable behind `--latent-strategy`, but it is no longer the default. This keeps the audit baseline clean while preserving the current research implementation until the archive/delete decision is made.

## Diagram

```text
                   local visual field
                    (B, C, H, W)
                         |
                         v
                    CNNEncoder
                         |
                         v
                  visual features
                         |
             +-----------+-----------+
             |                       |
             v                       v
      actor output head       value output head
       action logits             scalar value


             structured global state (B, 14)
                         |
                         v
                  CentralizedCritic
                         |
                         v
                 centralized value

Future ICRA additions:
  z embedding -> PPOPolicy.forward(..., extra=z_emb)
  joint actions + z -> CentralizedCritic.forward(..., extra=critic_extra)
```
