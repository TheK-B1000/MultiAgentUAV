# MARL CTF Audit Report

Date: 2026-04-24

## Phase 3 - PPO / MAPPO Algorithmic Correctness

Changed:

- Removed the old SB3 PPO/policy/rollout-buffer training path from `rl/train_ppo.py`.
- Added `rl/ppo_core.py` with local GAE, clipped PPO policy loss, clipped value loss, and a named-field rollout buffer.
- Added `rl/custom_ppo.py` with the default local PPO/MAPPO-style trainer: shared decentralized CNN actor plus centralized 14-float global-state critic.
- Kept latent strategy as pure PyTorch scaffolding in `rl/latent_marl.py`: `StrategyEncoder`, `LatentConditionedActor`, and `expected_strategy_switch_penalty`.
- Deleted the old SB3-only feature extractor and latent vector wrapper modules.
- Removed `stable-baselines3` from `requirements.txt`.
- Converted viewer, snapshot-opponent, and evaluation rollout loading to the local custom PPO checkpoint format.
- Added `docs/algorithm.md` and updated architecture/environment/setup docs to describe the local trainer.
- **Training log semantics (local PPO):** `CustomPPOTrainer` prints cumulative W/L/D and win rate on a schedule. Fields mean:
  - **`mode`** — `PPOConfig.mode` (e.g. `FIXED_OPPONENT` = train vs a scripted or configured opponent, not self-play).
  - **`phase`** — Scripted difficulty / curriculum label (e.g. `OP3`), from `info["episode_result"]["phase_name"]` or `fixed_opponent_tag`, *not* a second copy of `mode`.
  - **`opp`** — Who red is for that episode, e.g. `SCRIPTED:OP3` or `SNAPSHOT:name`.
- Configure interval with `PPOConfig.episode_log_every` or CLI `--episode-log-every N` (`0` = off). Default: every 1000 completed episodes.

Verified:

- `python run_tests.py` or `python -m unittest discover -v tests` (see `AICTFProject/run_tests.py`).
- Added `tests/test_ppo.py` for GAE, policy clipping, and value clipping.
- Added `tests/test_rollout_buffer.py` for named field registration, future `z` storage, minibatch flattening, and return computation.
- Added a checkpoint inference regression test for the local PPO loader.
- Ran a 256-step custom PPO smoke with 2 envs, 32-step rollouts, and one epoch; no exceptions or NaNs.

Open risks and deferrals:

- The requested 100k-step trend smoke from the audit spec was not run in this turn; the local PPO path has only short smoke coverage so far.
- Latent strategy training is intentionally not wired into PPO yet. The Summer plan hooks are documented in `docs/algorithm.md`.
- Self-play snapshot opponents from the old training loop are not implemented in the local PPO trainer.
