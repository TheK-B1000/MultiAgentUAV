# Canonical Agent Identity Contract

## Overview

All blue agents use **canonical slot-based keys**: `blue_0`, `blue_1`, `blue_2`, ... (zero-indexed).

These keys are:
- **Deterministic**: `blue_0` = first agent in `blue_agents` list, `blue_1` = second, etc.
- **Stable**: Keys don't change within an episode (agents don't swap slots)
- **Used consistently** for:
  - Action submission (`submit_external_actions`)
  - Reward routing (mapping reward events to agents)
  - Observation ordering (`obs["grid"][i]` corresponds to `blue_i`)
  - MARL wrapper `agent_keys`

## Implementation

### Core Module: `rl/agent_identity.py`

- **`AgentIdentity`**: Dataclass with `key`, `slot_index`, `agent`, `is_enabled`
- **`build_blue_identities()`**: Builds canonical identity list from `blue_agents`
- **`build_reward_id_map()`**: Maps internal IDs (unique_id, slot_id, side_agent_id) → slot_index
- **`route_reward_events()`**: Routes reward events using identity map, returns `(team_total, per_agent_list, dropped_count)`

### Changes Made

1. **SB3 Env (`ctf_sb3_env.py`)**:
   - Builds identities at reset: `self._blue_identities = build_blue_identities(...)`
   - Action submission uses canonical keys: `actions_by_agent[ident.key] = executed[i]`
   - Reward consumption uses identity map: `route_reward_events(events, reward_id_map, n_slots)`
   - Tracks `dropped_reward_events` per step and episode
   - Observation building uses identity ordering: `blue_ordered = [ident.agent for ident in self._blue_identities]`

2. **GameField (`game_field.py`)**:
   - `submit_external_actions()` accepts canonical `blue_i` keys
   - `_consume_external_action_for_agent()` prioritizes canonical keys, then falls back to legacy matching

3. **MARL Wrapper (`rl/marl_env.py`)**:
   - Already uses canonical keys: `_agent_keys(n)` returns `["blue_0", "blue_1", ...]`
   - Keys are used strictly for action flattening and reward splitting

4. **Episode Results (`rl/episode_result.py`)**:
   - Added `dropped_reward_events: int` field to `EpisodeSummary`
   - Tracks events that couldn't be routed to canonical slots

## Acceptance Tests

See `tests/test_agent_identity_contract.py`:

- ✅ Action keys are canonical (`blue_0`, `blue_1`, ...)
- ✅ Per-agent rewards length matches `n_blue_agents`
- ✅ `sum(per_agent_rewards)` equals team reward (within epsilon)
- ✅ `dropped_reward_events` is 0 for normal runs
- ✅ MARL wrapper obs keys match `agent_keys` exactly
- ✅ Identity map covers all canonical keys

## Usage

### In SB3 Env

```python
# Identities are built automatically at reset()
identities = env._blue_identities  # List[AgentIdentity]
reward_id_map = env._reward_id_map  # Dict[str, int]

# Action submission uses canonical keys
actions = {"blue_0": (macro_0, target_0), "blue_1": (macro_1, target_1)}
env.gf.submit_external_actions(actions)

# Reward routing uses identity map
team_total, per_agent, dropped = route_reward_events(events, reward_id_map, n_slots)
```

### In MARL Wrapper

```python
# agent_keys are canonical
env = MARLEnvWrapper(base_env)
keys = env.agent_keys  # ["blue_0", "blue_1"]

# Actions use canonical keys
actions = {key: (macro, target) for key in keys}
obs, rews, dones, infos, shared = env.step(actions)
```

## Notes

- **Backward compatibility**: `GameField.submit_external_actions()` still accepts legacy keys (unique_id, slot_id) as fallback
- **Observation ordering**: Observations are built using identity ordering, ensuring `obs["grid"][i]` corresponds to `blue_i`
- **Dropped events**: Events that can't be routed (e.g., from disabled agents) are counted in `dropped_reward_events` and logged in episode summaries
