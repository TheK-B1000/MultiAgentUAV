# Setup And Train

The default training path is the local latent-strategy PPO/MAPPO-style implementation in `rl/train_ppo.py`. It no longer uses the old SB3 PPO policy, rollout buffer, or GAE path.

Latent team strategy is enabled by default. Use `--no-latent-strategy` for vanilla local PPO ablations.

The default run is the Summer latent implementation: `--mode FIXED_OPPONENT --fixed-opponent OP3` with latent strategy enabled. `--mode CURRICULUM` is the Jacob-style OP1 -> OP2 -> OP3 scripted-opponent curriculum baseline; it always disables latent strategy.

For **paper / ICRA alignment** (centralized value critic **V** vs **Q** wording, event-masked λ_p / λ_H, q_phi inputs, frozen eval matrix, team sizes), see `docs/Paper_experiment_alignment.md`.

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

2v2 fixed-opponent latent strategy run:

```bash
python rl/train_ppo.py --mode FIXED_OPPONENT --fixed-opponent OP3 --agents 2 --run-tag ppo_latent_fixed_op3_2v2
```

4v4 fixed-opponent latent strategy run:

```bash
python rl/train_ppo.py --mode FIXED_OPPONENT --fixed-opponent OP3 --agents 4 --run-tag ppo_latent_fixed_op3_4v4
```

6v6 fixed-opponent latent strategy run:

```bash
python rl/train_ppo.py --mode FIXED_OPPONENT --fixed-opponent OP3 --agents 6 --run-tag ppo_latent_fixed_op3_6v6
```

Vanilla local PPO ablation:

```bash
python rl/train_ppo.py --mode FIXED_OPPONENT --fixed-opponent OP3 --agents 2 --no-latent-strategy --run-tag ppo_custom_fixed_op3_2v2
```

Professor-requested baselines:

```bash
# Curriculum baseline: Jacob-style OP1->OP2->OP3, no latent strategy.
python rl/train_ppo.py --mode CURRICULUM --agents 2 --no-latent-strategy --run-tag baseline_curriculum_2v2

# No-latent baseline: Summer default opponent setting with the latent path disabled.
python rl/train_ppo.py --mode FIXED_OPPONENT --fixed-opponent OP3 --agents 2 --no-latent-strategy --run-tag baseline_no_latent_2v2
```

Short smoke run:

```bash
python rl/train_ppo.py --mode FIXED_OPPONENT --fixed-opponent OP3 --agents 2 --total-steps 128 --checkpoint-dir checkpoints/smoke --run-tag smoke_2v2
```

Domain-randomization sanity run:

```bash
python rl/train_ppo.py --mode FIXED_OPPONENT --fixed-opponent OP3 --agents 2 --total-steps 200000 --domain-randomization --run-tag dr_sanity_2v2
```

Domain randomization is episode-level and training-only by default. Current knobs sample enemy-observation position noise, enemy-detection dropout, and a blue-team speed scale from `U(1-jitter, 1)` on each reset. The speed randomization is intentionally slowdown-only because the integrator still enforces the marine speed cap. It is also asymmetric: perturbations apply to the learning blue policy, while the scripted OP3 opponent is left unperturbed. Note this as "learning-agent DR only"; symmetric DR is a future variant, not part of the current minimal implementation.

Full protocol (trajectory interpretation, `q_phi` under DR, critic expectations, no-latent stress test): **`docs/METHODOLOGY_DOMAIN_RANDOMIZATION.md`**.

Longer arc toward real boats (phases, artifacts, gap filter): **`docs/ROADMAP_BOATS.md`**.

Before a long DR run, use a 200k sanity check. If rollout WR is above roughly 50% and still climbing, keep the defaults (`noise=0.12`, `dropout=0.08`, `speed jitter=0.12`). If WR collapses near 40% and stays there, halve the DR knobs. If it matches the non-DR curve too closely, the defaults are probably too weak.

After a 1M DR checkpoint, compare robustness curves for the non-DR and DR checkpoints by sweeping noise/dropout/jitter. Keep eval-time DR off for clean baseline comparisons, then run explicit perturbation sweeps to show where each policy's WR cliff occurs. Also train a no-latent PPO with the same DR settings as an early baseline:

```bash
python rl/train_ppo.py --mode FIXED_OPPONENT --fixed-opponent OP3 --agents 2 --total-steps 200000 --domain-randomization --no-latent-strategy --run-tag dr_no_latent_sanity_2v2
```

Checkpoints are torch checkpoints saved with the historical `.zip` suffix, for example `checkpoints/2v2/final_ppo_latent_fixed_op3_2v2.zip`.

Training also writes CSV telemetry by default:

- `<run_tag>_metrics.csv`: one row per PPO update with losses, KL, rollout reward/return stats, cumulative W/L/D, win rate, and strategy diagnostics.
- `<run_tag>_episodes.csv`: one row per completed training episode with score/outcome fields for trend plots.

Use `--no-metrics-csv` to disable this, or `--metrics-csv` / `--episode-csv` to choose explicit paths.

## Algorithm Defaults

The local PPO path uses:

- CNN actor over local per-agent `grid` observations.
- Strategy encoder `q_phi(z | s)` over the 19-float `global_state`.
- Shared strategy embedding concatenated to each agent actor.
- Centralized critic over `global_state`, joint-action one-hot, and `z_onehot` in latent mode.
- Named rollout buffer with `global_state`, `terminated`, `truncated`, and latent strategy fields.
- GAE that bootstraps time-limit truncations but does not leak advantages across reset boundaries.
- PPO clipped policy loss, clipped value loss, entropy bonus, minibatch advantage normalization, linear LR decay, and global-norm gradient clipping.
- Strategy persistence and strategy entropy losses.

See `docs/algorithm.md` for the full contract.

## Latent Strategy Controls

Paper-aligned design status:

| Topic | Current status |
| --- | --- |
| `StrategyEncoder q_phi(z | s)` | Implemented as a 128-128 MLP and wired into `CustomPPOTrainer`. |
| Global state | True 19-float `GLOBAL_STATE_DIM` vector from `rl/global_state.py`. |
| Decentralized policy conditioning | Shared strategy embedding is concatenated to each agent actor input. |
| Critic conditioning | Joint-action one-hot and `z_onehot` flow through `CentralizedCritic.forward(..., extra=...)`. |
| Persistence loss | Applied only on sparse non-initial strategy refreshes. |
| Strategy entropy | Applied on strategy sampling steps to reduce collapse. |

The training objective is:

Checkpoints saved before the 19-float global-state expansion are intentionally not load-compatible with the current critic/q_phi input shape. Start a fresh run or load a checkpoint trained after that change.

```text
L = L_PPO + lambda_p * L_persist - lambda_H * H(q_phi(z | s))
```

Do not resample `z` every timestep. Use once-per-episode or sparse refresh intervals such as every 20 steps.

Useful flags:

```bash
--latent-k 4
--latent-resample-every 20
--latent-lam-p 0.02
--latent-lam-h 0.005
--latent-z-embed-dim 16
--fixed-latent-strategy --fixed-latent-id 0
--no-latent-strategy
```

## Strategy Analysis

Latent checkpoints record rollout strategy diagnostics in checkpoint `last_stats`, and evaluation CSVs include per-episode strategy fields when the policy is latent:

- `strategy_switch_rate`
- `strategy_resample_rate`
- `strategy_unique_count`
- `strategy_entropy_mean`
- `strategy_occupancy_0...K`
- `strategy_phase_<phase>_occupancy_0...K`

Generate strategy plots from any episode-level training/eval CSV:

```bash
python plot/plot_metrics.py checkpoints/2v2/ppo_latent_fixed_op3_2v2_episodes.csv --window 10
```

This writes `strategy_switch_rate_vs_episode.*`, `strategy_occupancy.*`, and `strategy_phase_occupancy.*` under `figures/` when those columns are present.

Evaluate any single checkpoint, including ablations:

```bash
python plot/eval_checkpoint.py --checkpoint checkpoints/2v2/final_ppo_latent_fixed_op3_2v2.zip --agents 2 --opponents OP3 OP4 --map-sets train eval --episodes 100
```

## Final Experiment Matrix

Generate the final-phase command matrix without launching long jobs:

```bash
python experiments/phase6_experiment_matrix.py --agents 2 4 6 --seeds 42 43 44 --steps 100000 --eval-episodes 100 --eval-map-sets train eval --out csv/phase6_commands.csv
```

The matrix trains on the `train` map split and evaluates on both `train` and held-out `eval` map splits. It covers the Summer latent default, curriculum baseline, and no-latent baseline. Add `--execute` only when you are ready to run the long training/evaluation sequence.
