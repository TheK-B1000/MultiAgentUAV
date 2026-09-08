# Vectorized rollouts vs. the plan’s nested loops

The Summer Implementation Plan sketches training as nested `for episode` / `for t` loops with `env.step` per environment. The implementation instead uses a **vectorized** `VecEnv` (`game_field_gpu.GPUCTFVecEnv`) and a single `CustomPPOTrainer.collect_rollout` loop that advances all environments in lockstep.

This is a **formal equivalence** class for PPO, not a different algorithm, provided:

1. **Independence** — each parallel environment has its own Markov state; there is no cross-env coupling in the transition kernel.
2. **Same policy** — the same `SharedActorCentralizedCritic` maps observations (and, if enabled, the shared latent strategy) to action distributions and values for every row of the batch.
3. **GAE** — `TensorDictRolloutBuffer` stores time-major trajectories and applies the same GAE/return recursion as a scalar loop, just with an extra “batch” dimension.
4. **PPO update** — minibatches are drawn from a flat pool of transition tuples `(s_t, a_t, r, s_{t+1}, …)`; shuffling is standard PPO and does not change the *expected* gradient of the supervised policy/value losses under i.i.d. or Markov data.

The difference is only **order of operations for throughput** (batch GPU sim + batched linear algebra), not the mathematical object being optimized.

**Related code:** `rl/custom_ppo.py` (`CustomPPOTrainer.collect_rollout`, `rl/ppo_core.py` for GAE).
