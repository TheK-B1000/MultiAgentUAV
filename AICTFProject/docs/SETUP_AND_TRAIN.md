# Setup And Train

The default training path is the local PPO/MAPPO-style implementation in `rl/train_ppo.py`. It no longer uses the old SB3 PPO policy, rollout buffer, or GAE path.

Latent team strategy is reserved for the Summer/ICRA implementation phase. The pure PyTorch pieces are present in `rl/latent_marl.py`, but `--latent-strategy` intentionally raises until that mechanism is wired into `rl/custom_ppo.py`.

## Install

Install PyTorch for your platform first from `https://pytorch.org`, then install project dependencies from `AICTFProject`:

```bash
cd AICTFProject
pip install -r requirements.txt
```

## Test

The supported test runner is Python `unittest`:

```bash
cd AICTFProject
python -m unittest discover -v tests
```

`pytest` is optional and is not required by the project.

## Train

Run commands from `AICTFProject`.

2v2 fixed-opponent baseline:

```bash
python rl/train_ppo.py --mode FIXED_OPPONENT --fixed-opponent OP3 --agents 2 --run-tag ppo_custom_fixed_op3_2v2
```

4v4 fixed-opponent baseline:

```bash
python rl/train_ppo.py --mode FIXED_OPPONENT --fixed-opponent OP3 --agents 4 --run-tag ppo_custom_fixed_op3_4v4
```

6v6 fixed-opponent baseline:

```bash
python rl/train_ppo.py --mode FIXED_OPPONENT --fixed-opponent OP3 --agents 6 --run-tag ppo_custom_fixed_op3_6v6
```

Short smoke run:

```bash
python rl/train_ppo.py --mode FIXED_OPPONENT --fixed-opponent OP3 --agents 2 --total-steps 128 --checkpoint-dir checkpoints/smoke --run-tag smoke_2v2
```

Checkpoints are torch checkpoints saved with the historical `.zip` suffix, for example `checkpoints/2v2/final_ppo_custom_fixed_op3_2v2.zip`.

## Algorithm Defaults

The local PPO path uses:

- CNN actor over local per-agent `grid` observations.
- Centralized critic over the 14-float `global_state`.
- Named rollout buffer with `global_state`, `terminated`, and `truncated` fields.
- GAE that bootstraps time-limit truncations but does not leak advantages across reset boundaries.
- PPO clipped policy loss, clipped value loss, entropy bonus, minibatch advantage normalization, linear LR decay, and global-norm gradient clipping.

See `docs/algorithm.md` for the full contract.

## Latent Strategy Handoff

Paper-aligned design:

| Topic | Current status |
| --- | --- |
| `StrategyEncoder q_phi(z | s)` | Implemented as a 128-128 MLP in `rl/latent_marl.py`. |
| Global state | True 14-float `GLOBAL_STATE_DIM` vector from `rl/global_state.py`. |
| Decentralized policy conditioning | `LatentConditionedActor` provides the pure PyTorch actor pattern. |
| Critic conditioning | Add joint-action one-hot and `z_onehot` through `CentralizedCritic.forward(..., extra=...)`. |
| Persistence loss | `expected_strategy_switch_penalty` is implemented. |
| Training integration | Deferred to the Summer/ICRA implementation phase inside `CustomPPOTrainer`. |

The training objective to add later is:

```text
L = L_PPO + lambda_p * L_persist - lambda_H * H(q_phi(z | s))
```

Do not resample `z` every timestep. Use once-per-episode or sparse refresh intervals such as every 20 steps.
