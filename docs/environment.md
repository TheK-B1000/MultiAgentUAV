# Environment

## API Flavor

The active training environment is a lightweight custom vector env:

- `GPUCTFVecEnv` wraps `BatchedCTFCore` for batched training.
- `GPUCTFSingleEnv` is a Gymnasium-compatible single-env adapter for tests and simple evaluation.
- The environment is not currently PettingZoo.

`GPUCTFSingleEnv.reset(seed=...)` returns `(obs, info)`. `step(action)` returns `(obs, reward, terminated, truncated, info)`. `GPUCTFVecEnv` follows the project's local vector-env API and collapses `terminated or truncated` to `done`, while preserving separate `terminated` and `truncated` booleans in each `info` dict.

## Observation Spec

Each blue-team policy observation is a `spaces.Dict`:

- `grid`: `Box(0, 1, shape=(N, 7, 20, 20), dtype=float32)`
- `vec`: `Box(-1, 1, shape=(N, 18), dtype=float32)`
- `agent_mask`: `Box(0, 1, shape=(N,), dtype=float32)`
- `mask`: `Box(0, 1, shape=(N * 55,), dtype=float32)`

`N` is `n_agents_per_team`. The spatial `grid` is built **per agent** in `BatchedCTFCore._build_grid_obs` (channel maps over a local field of view, not a dump of full-game state into the actor). With the Summer-plan policy path, that tensor is **flattened** and concatenated to `vec` and the strategy embedding; it is **not** the same as `global_state` (which is only for `q_\phi` and the centralized critic in CTDE). This preserves **decentralized execution** at the actor: per-agent local tensors + shared `z`, while global summaries stay in the critic/encoder only.

## Action Spec

The action space is `MultiDiscrete([5, 50] * N)`:

- 5 macro actions: `GO_TO`, `GRAB_MINE`, `GET_FLAG`, `PLACE_MINE`, `GO_HOME`
- 50 fixed macro targets

Illegal actions are exposed via the flattened action mask. The observation key is `mask`; the same mask is also available in `info["action_mask"]` on every step.

## Info Dict Schema

Each `info` dict contains stable keys:

- `blue_score`, `red_score`
- `decision_steps`, `sim_steps`
- `phase`, `league_mode`, `opponent_kind`, `opponent_key`, `rules_profile`
- `map_set`
- `dense_reward`, `sparse_points`
- `terminated`, `truncated`, `stalemate_truncated`
- `action_mask`
- `agent_alive`, `blue_alive`, `red_alive`
- `global_state`

Terminal vector-env infos additionally include `terminal_observation` and `episode_result`.

## Reward Table

Rewards are deterministic functions of state/action and are bounded by `tanh(raw / reward_scale)` followed by clipping to `[-reward_clip, reward_clip]`; defaults are `reward_scale=2.0`, `reward_clip=1.0`.

| Term | Sign | Default magnitude | Trigger |
| --- | --- | --- | --- |
| Win terminal reward | positive | `+1.0` before final scaling | Blue score exceeds red score at done. |
| Loss terminal penalty | negative | `-1.0` before final scaling | Red score exceeds blue score at done. |
| Draw terminal penalty | negative | `-0.5` before final scaling | Scores tied at done. |
| Flag pickup reward | positive | `+0.1` | Blue grabs red flag. |
| Flag carry-home reward | positive | `+0.5` | Blue captures red flag. |
| Enemy kill reward | positive | `+0.5` | Blue tags/eliminates red. |
| Mine placement reward | positive | `+0.2` | Blue places a mine. |
| Sparse tag/capture points | mixed | `+/-100` point scale before `/100` sparse normalization | Aquaticus-style events. |
| PBRS progress | mixed | coefficient driven | Potential-based progress toward attack/return/defense objectives. |
| Team coordination | positive | `0.02` to `0.03` defaults | Defense presence, escort, intercept shaping. |
| Spin/idle/stalemate penalties | negative | small coefficients | Low-progress or unstable behavior. |

Credit assignment is currently team-level from the trainer perspective: the reward returned by the env is one scalar per parallel environment for the blue team.

## Termination And Truncation

`terminated` means a game-rule ending, currently score limit reached by either team.

`truncated` means a time/resource ending:

- decision step limit
- simulation step limit
- stalemate trigger

The single-env Gymnasium adapter returns these separately. The vector env returns `done = terminated or truncated` and stores both booleans in `info`.

## Global State Spec

`env.state()` and `info["global_state"]` return a compact structured `np.float32` vector of shape `(14,)` for single envs and `(B, 14)` for vector envs.

Field order is locked:

1. `mean_position_blue_x`
2. `mean_position_blue_y`
3. `std_position_blue_x`
4. `std_position_blue_y`
5. `mean_position_red_x`
6. `mean_position_red_y`
7. `std_position_red_x`
8. `std_position_red_y`
9. `min_blue_to_red_flag`
10. `min_red_to_blue_flag`
11. `blue_flag_captured`
12. `red_flag_captured`
13. `avg_blue_speed`
14. `avg_red_speed`

This vector is the CTDE input for the strategy inference network and centralized critic. It is intentionally an MLP input, not a visual CNN input.

## Agent IDs

Stable IDs use the convention:

- `blue_0`, `blue_1`, ...
- `red_0`, `red_1`, ...

Adapters expose this through `AgentHandle.unique_id`; dead agents keep their index and respawn into the same identity.

## Team Size

`GPUFieldConfig(n_agents_per_team=N)` sets both blue and red team sizes. Supported experiment sizes are `N in {2, 4, 6}`. Legacy `max_blue_agents` / `max_red_agents` remain supported.

## Map Sets

`GPUFieldConfig(map_set="train" | "eval")` selects a deterministic seed range:

- `train`: seed offset `0`
- `eval`: seed offset `1_000_003`

The map geometry is fixed in this version, but spawn/opponent/randomized episode state uses disjoint deterministic streams by map set. Same seed plus same map set is reproducible; same seed across different map sets produces different initial episode state.

## Rendering

`GPUCTFSingleEnv.render(mode="rgb_array")` returns `uint8` RGB arrays with shape `(H, W, 3)`. `mode="human"` currently returns the same frame and does not open a window. The Pygame viewer remains the richer human visualization path.

## Parallelization

`GPUCTFVecEnv` stores all workers inside one torch batched core. RNG is held in a core-local `torch.Generator`, seeded by `seed + map_set_offset`. This avoids Python subprocess pickling issues while keeping worker rows independent within the batched tensor state.
