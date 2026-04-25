# MARL CTF Audit Report

Date: 2026-04-24

## Phase 3 - PPO / MAPPO Algorithmic Correctness

Changed:

- Removed the old SB3 PPO/policy/rollout-buffer training path from `rl/train_ppo.py`.
- Added `rl/ppo_core.py` with local GAE, clipped PPO policy loss, clipped value loss, and a named-field rollout buffer.
- Added `rl/custom_ppo.py` with the local PPO/MAPPO-style trainer.
- Deleted the old SB3-only feature extractor and latent vector wrapper modules.
- Removed `stable-baselines3` from `requirements.txt`.
- Converted viewer, snapshot-opponent, and evaluation rollout loading to the local custom PPO checkpoint format.
- Added `docs/algorithm.md` and updated architecture/environment/setup docs to describe the local trainer.
- Training log semantics: `CustomPPOTrainer` prints cumulative W/L/D and win rate on a schedule. `mode` is `PPOConfig.mode`, `phase` is the scripted difficulty/curriculum label, and `opp` identifies the red opponent.

Verified:

- Added `tests/test_ppo.py` for GAE, policy clipping, and value clipping.
- Added `tests/test_rollout_buffer.py` for named field registration, future `z` storage, minibatch flattening, and return computation.
- Added a checkpoint inference regression test for the local PPO loader.
- Ran a 256-step custom PPO smoke with 2 envs, 32-step rollouts, and one epoch; no exceptions or NaNs.

Open risks and deferrals:

- Self-play snapshot opponents from the old training loop are not implemented in the local PPO trainer.

## Phase 4 - Latent Team Strategy Integration

Changed:

- Enabled the Summer/ICRA latent team strategy path in the local trainer by default.
- Added `StrategyEncoder q_phi(z | s)` sampling over the true 14-float global state.
- Added shared strategy embedding conditioning to the decentralized actor.
- Added action-conditioned centralized critic inputs: joint-action one-hot plus `z_onehot`.
- Added sparse strategy sampling support: default once at episode start, or every `latent_resample_every_n` decisions.
- Added masked persistence and strategy entropy losses to the PPO update.
- Updated checkpoint inference, viewer, and evaluation rollout code so latent checkpoints receive `global_state` during prediction.
- Added CLI controls: `--no-latent-strategy`, `--latent-k`, `--latent-resample-every`, `--latent-lam-p`, `--latent-lam-h`, and `--latent-z-embed-dim`.

Verified:

- Latent strategy model shape test covers actor conditioning, strategy logits/log-probs, and action-conditioned critic values.
- Latent train/save/load smoke test covers a tiny end-to-end PPO update and local inference.

Open risks and deferrals:

- The value target now uses the action-conditioned critic path for latent mode, but only short smoke tests have been run.
- Strategy interpretability plots were deferred to the next phase and are now tracked under Phase 5.
- The requested 100k-step trend smoke still has not been run.

## Phase 5 - Latent Strategy Observability

Changed:

- Added inference-time `strategy_info()` diagnostics for latent checkpoints: selected `z`, strategy probabilities, entropy, and resample flag.
- Added trainer rollout diagnostics in checkpoint `last_stats`: strategy occupancy, dominant strategy, switch count/fraction, and resample fraction.
- Added per-episode strategy columns to shared evaluation rollouts and viewer evaluation CSVs.
- Added aggregate strategy metrics to `plot_eval_metrics.py`, including CSV reload support for frozen tables.
- Added `plot_metrics.py` strategy figures for switch-rate traces and mean occupancy bars.
- Added strategy occupancy-by-phase columns/plots for post hoc interpretation by coarse flag-state phase.

Verified:

- Added a checkpoint smoke assertion that latent trainer stats include strategy occupancy/switch diagnostics.
- Added an inference regression assertion that latent checkpoint prediction exposes strategy diagnostics.
- Added an aggregate regression test for strategy switch/resample/entropy/occupancy metrics.

Open risks and deferrals:

- These are observability hooks only; no long-run interpretability analysis has been performed yet.
- The requested 100k-step trend smoke still has not been run.

## Phase 6 - Final Experiment Readiness

Changed:

- Added default training telemetry CSVs: per-update PPO/strategy stats and per-episode outcome rows.
- Added `plot/eval_checkpoint.py` for evaluating any single checkpoint/ablation against OP3, OP4, or another scripted opponent list.
- Added `experiments/phase6_experiment_matrix.py` to generate the final Summer-plan experiment commands for latent default, vanilla PPO, no-persistence, lower-K, sparse-refresh, and OP2-trained comparison runs.
- Added train/eval map-set controls so the matrix trains on the training split and evaluates on held-out maps for the Summer-plan generalization experiment.
- Added `--seed`, `--metrics-csv`, `--episode-csv`, and `--no-metrics-csv` training CLI controls.
- Updated setup/algorithm docs with the telemetry and final experiment workflow.

Verified:

- Added regression coverage for training metrics CSV output.
- Added helper tests for the final experiment command matrix and arbitrary-checkpoint CSV field handling.

Open risks and deferrals:

- This is the last implementation/refactor phase in the current audit path.
- The remaining work is empirical: run the generated 100k-step trend smoke / full experiment matrix, then collect plots and paper tables.
