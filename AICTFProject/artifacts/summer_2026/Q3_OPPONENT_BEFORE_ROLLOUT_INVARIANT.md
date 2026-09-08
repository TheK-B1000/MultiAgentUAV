# Q3 invariant — opponent assigned before first PPO rollout (D3 / OPPONENT_POOL)

**Status:** `CODE_INVARIANT_PASS` for Mixed-PPO / `mode=OPPONENT_POOL` (the D3 path).

## Claim

Every gradient-bearing transition in a D3 run is collected under an opponent in
`{OP7, OP9, OP12}`. Episode-row absence of outside-pool tags is therefore
sufficient for Q3 **because** the opponent is bound before `env.step()` #1.

## Mechanism (existing code — not a new sampler)

1. **Resolve opener into the pool** — `rl/training/resolved_config.py::_resolve_initial_opponent_and_phase`

   In `OPPONENT_POOL` / `opponent_randomize` mode, if `fixed_opponent_tag` is
   not in `cfg.opponent_pool`, the opener falls back to `pool[0]`.

   For D3 (`OP7, OP9, OP12`) with G0-V5's legacy `fixed_opponent_tag=OP3`:

   ```text
   resolved initial_opponent_tag = OP7   (in pool)
   ```

2. **Apply before any rollout** — `rl/training/env_factory.py::build_training_env`

   After constructing `GPUCTFVecEnv`, before returning to the trainer:

   ```text
   set_phase(initial_phase)
   set_next_opponent("SCRIPTED", initial_opponent_tag)   # ← before learn()
   _apply_initial_opponent_params(...)
   ```

3. **First reset uses that binding** — `rl/custom_ppo/rollout/collector.py`

   First `collect_rollout` calls `env.reset()` only if `_last_obs is None`; the
   env already has the scripted opponent from step 2. Subsequent episodes are
   re-sampled by `_before_reset_indices_hook` **before** `reset_indices`, so the
   next episode never starts under a stale outside-pool tag.

4. **Pinned by tests** — `tests/test_v5i4_paper_faithful.py`

   - `test_initial_opponent_falls_back_to_pool_first_entry`
   - `test_initial_opponent_respects_explicit_in_pool_fixed_tag`

## What this does *not* cover

- **SNAPSHOT / FP path:** the 2026-08-13 FP probe showed `SCRIPTED:OP6` warmup
  rows when a snapshot pool was layered on top of a scripted G0-V5 default.
  That is why FP still needs a clean full smoke; it is **not** the D3 path.
- Mid-episode opponent switches (forbidden by construction; selection is at
  episode boundaries only).

## Preflight policy

```text
Any outside-pool episode_rows entry  →  D3_POOL_PREFLIGHT FAIL
Code invariant above                 →  Q3 closed for D3 OPPONENT_POOL
```

Do not weaken this to “contamination was negligible.”
