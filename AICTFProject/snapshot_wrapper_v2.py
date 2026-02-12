"""
Snapshot wrapper v2: For 4v4/8v8 models trained with new EnvObsBuilder (14-channel obs).

This wrapper uses the same observation builder as training, ensuring compatibility
with models trained on the new multi-agent observation space.

Usage:
    wrapper = make_snapshot_wrapper_v2(snapshot_path, max_agents=4)
    game_field.set_red_policy_wrapper(wrapper)
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional

import numpy as np

try:
    from rl.env_modules.env_obs import EnvObsBuilder
    from rl.agent_identity import AgentIdentity
except ImportError:
    raise RuntimeError("snapshot_wrapper_v2 requires rl.env_modules.env_obs and rl.agent_identity")


def _load_snapshot_policy_only(path: str):
    """
    Load only the policy from an SB3 .zip (no PPO rollout buffer).
    Returns an object with .policy and .predict(obs, deterministic=True) like PPO.
    """
    import torch as th
    from stable_baselines3.common.save_util import load_from_zip_file

    try:
        from rl.train_ppo import MaskedMultiInputPolicy  # noqa: F401
    except Exception:
        pass

    data, params, _ = load_from_zip_file(path, device="cpu", load_data=True)
    if not data or "policy" not in params:
        raise ValueError(f"Invalid snapshot zip: missing data or policy params in {path!r}")

    obs_space = data["observation_space"]
    action_space = data["action_space"]
    policy_kwargs = dict(data.get("policy_kwargs", {}))
    policy_kwargs.pop("device", None)

    # Check if this is a Dict obs model (new format)
    _is_dict_obs = hasattr(obs_space, "spaces") and isinstance(getattr(obs_space, "spaces", None), dict)
    if not _is_dict_obs:
        raise ValueError(
            f"Snapshot {path!r} does not use Dict observation space. "
            "v2 wrapper requires models trained with EnvObsBuilder (Dict obs). "
            "Use make_snapshot_wrapper() for legacy Box obs models."
        )
    
    # Try to detect max_agents from obs space
    # For 4v4: grid shape is typically (4, 14, 20, 20) or similar
    detected_max_agents = None
    if "grid" in obs_space.spaces:
        grid_space = obs_space.spaces["grid"]
        if hasattr(grid_space, "shape") and len(grid_space.shape) >= 2:
            # Shape might be (max_agents, channels, H, W) or (channels*max_agents, H, W)
            # For 4v4 with 14 channels: could be (4, 14, 20, 20) or (56, 20, 20)
            if len(grid_space.shape) == 4:
                detected_max_agents = grid_space.shape[0]
            elif len(grid_space.shape) == 3:
                # (channels*max_agents, H, W) - try to infer
                total_channels = grid_space.shape[0]
                # Common: 14 channels per agent for 4v4 = 56 total
                if total_channels % 14 == 0:
                    detected_max_agents = total_channels // 14

    # Use MaskedMultiInputPolicy for Dict obs
    try:
        from rl.train_ppo import MaskedMultiInputPolicy
        policy_class = MaskedMultiInputPolicy
    except ImportError:
        from stable_baselines3 import PPO
        from stable_baselines3.common.policies import MultiInputActorCriticPolicy
        policy_class = getattr(PPO, "policy_aliases", {}).get("MultiInputPolicy") or MultiInputActorCriticPolicy

    # Remove extractor overrides to use default CombinedExtractor
    policy_kwargs.pop("features_extractor_class", None)
    policy_kwargs.pop("features_extractor_kwargs", None)

    # SB3 ActorCritic policies require lr_schedule (unused for inference)
    lr = float(data.get("learning_rate", 1.5e-4))
    lr_schedule = lambda _: lr

    policy = policy_class(obs_space, action_space, lr_schedule, **policy_kwargs)
    try:
        policy.load_state_dict(params["policy"], strict=True)
    except Exception:
        policy.load_state_dict(params["policy"], strict=False)
    policy.eval()

    class _Predictor:
        def __init__(self, pol):
            self.policy = pol

        def predict(self, obs, deterministic=True):
            obs_t, _ = self.policy.obs_to_tensor(obs)
            with th.no_grad():
                actions, _, _ = self.policy(obs_t, deterministic=bool(deterministic))
            return actions.cpu().numpy(), None

    return _Predictor(policy)


def make_snapshot_wrapper_v2(
    snapshot_path: str,
    max_agents: int = 4,
    n_macros: int = 5,
    n_targets: int = 8,
    include_mask_in_obs: bool = True,
    include_opponent_context: bool = False,
    normalize_vec: bool = False,
) -> Callable[..., Any]:
    """
    Snapshot wrapper v2: For 4v4/8v8 models trained with new EnvObsBuilder.
    
    This wrapper uses EnvObsBuilder to build observations matching the training format,
    ensuring compatibility with models trained on the new multi-agent observation space.
    
    Args:
        snapshot_path: Path to SB3 .zip checkpoint
        max_agents: Maximum number of agents per team (4 for 4v4, 8 for 8v8)
        n_macros: Number of macro actions (default: 5)
        n_targets: Number of target actions (default: 8)
        include_mask_in_obs: Whether to include action masks in observation
        include_opponent_context: Whether to include opponent context embedding
        normalize_vec: Whether to normalize vector features
    
    Returns:
        Wrapper function compatible with GameField.set_red_policy_wrapper()
    """
    path = str(snapshot_path)
    if os.path.isdir(path):
        raise ValueError("Expected SB3 PPO .zip file path, got directory")
    if not os.path.exists(path):
        alt = path + ".zip"
        if os.path.exists(alt):
            path = alt
        else:
            raise FileNotFoundError(f"Snapshot not found: {snapshot_path!r}")

    # Load policy only (no PPO rollout buffer) to avoid OOM
    # This also validates the obs space format
    model = _load_snapshot_policy_only(path)
    
    # Verify max_agents matches model's expected obs space if possible
    # (The model loading already validated Dict obs format)

    # Create observation builder matching training setup
    obs_builder = EnvObsBuilder(
        use_obs_builder=True,
        include_mask_in_obs=include_mask_in_obs,
        include_opponent_context=include_opponent_context,
        include_high_level_mode=False,
        high_level_mode=0,
        high_level_mode_onehot=False,
        normalize_vec=normalize_vec,
    )

    # Use provided n_macros and n_targets (from game_field or defaults)
    # These should match the training configuration

    def wrapper(obs, agent, game_field, base_policy=None):
        """
        Wrapper function called by GameField.decide() for each red agent.
        
        Args:
            obs: Single-agent observation (legacy format, not used)
            agent: Red agent object
            game_field: GameField instance
            base_policy: Unused
        
        Returns:
            Dict with "macro_action" and "target_action"
        """
        # Build red team identities (same structure as blue)
        red_agents = getattr(game_field, "red_agents", []) or []
        red_agents_list = [a for a in red_agents if a is not None]
        
        # Find this agent's position in the red_agents list
        agent_id = int(getattr(agent, "agent_id", 0))
        slot_idx = None
        for i, red_agent in enumerate(red_agents_list):
            if red_agent is agent or (hasattr(red_agent, "agent_id") and getattr(red_agent, "agent_id", None) == agent_id):
                slot_idx = i
                break
        
        # Fallback: use agent_id if direct match failed
        if slot_idx is None:
            slot_idx = min(agent_id, len(red_agents_list) - 1) if red_agents_list else 0
        
        # Build identities for red team (pad to max_agents for consistent obs shape)
        red_identities: List[AgentIdentity] = []
        for i in range(max_agents):
            key = f"red_{i}"
            red_agent = red_agents_list[i] if i < len(red_agents_list) else None
            ident = AgentIdentity(key=key, slot_index=i, agent=red_agent)
            red_identities.append(ident)
        
        # Build team observation using EnvObsBuilder (same as training)
        team_obs = obs_builder.build_observation(
            game_field,
            red_identities,
            max_blue_agents=max_agents,
            n_blue_agents=len(red_agents_list),
            n_macros=n_macros,
            n_targets=n_targets,
            opponent_context_id=None,
            role_macro_mask_fn=None,  # Red doesn't use role masking
        )
        
        # Get team actions from model
        action, _ = model.predict(team_obs, deterministic=True)
        
        # Parse action for this specific agent
        # Action format: [agent0_macro, agent0_tgt, agent1_macro, agent1_tgt, ...]
        action_flat = np.asarray(action).reshape(-1)
        
        # Ensure slot_idx is within valid range
        slot_idx = min(slot_idx, max_agents - 1)
        
        # Extract this agent's actions
        macro_idx = slot_idx * 2
        target_idx = slot_idx * 2 + 1
        
        if macro_idx < len(action_flat) and target_idx < len(action_flat):
            macro = int(action_flat[macro_idx]) % max(1, n_macros)
            target = int(action_flat[target_idx]) % max(1, n_targets)
        else:
            # Fallback: use first agent's actions or defaults
            if len(action_flat) >= 2:
                macro = int(action_flat[0]) % max(1, n_macros)
                target = int(action_flat[1]) % max(1, n_targets)
            else:
                macro = 0
                target = 0
        
        return {"macro_action": macro, "target_action": target}

    return wrapper


__all__ = ["make_snapshot_wrapper_v2"]
