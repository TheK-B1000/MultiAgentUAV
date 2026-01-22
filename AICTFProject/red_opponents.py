# red_opponents.py
from __future__ import annotations
from typing import Any, Callable, Dict
import os
import numpy as np

from policies import OP3RedPolicy  # you already have this


def make_species_wrapper(species_tag: str) -> Callable[..., Any]:
    """
    Returns a GameField wrapper for red that forces a playstyle.
    Wrapper signature supported by your GameField:
      wrapper(obs, agent, game_field, base_policy) -> action
    """
    tag = str(species_tag).upper()

    base = OP3RedPolicy("red")

    # --- "species" hacks (edit these to match your scripted policy knobs) ---
    # If OP3RedPolicy doesn't have these attributes, add them OR change the hack.
    if tag == "RUSHER":
        # prioritize flag, ignore mines/defense
        if hasattr(base, "mine_radius_check"):
            base.mine_radius_check = 0.1
        if hasattr(base, "defense_weight"):
            base.defense_weight = 0.0
        if hasattr(base, "flag_weight"):
            base.flag_weight = 5.0
    elif tag == "CAMPER":
        # defensive posture, high mine awareness
        if hasattr(base, "mine_radius_check"):
            base.mine_radius_check = 5.0
        if hasattr(base, "defense_weight"):
            base.defense_weight = 3.0
        if hasattr(base, "flag_weight"):
            base.flag_weight = 1.0

    def wrapper(obs, agent, game_field, base_policy=None):
        # Use the tuned scripted policy directly
        # Must return a format your GameField.decide() parser accepts.
        action_id, target = base.select_action(obs, agent, game_field)
        return {"macro_action": int(action_id), "target_action": int(target) if target is not None else 0}

    return wrapper


def make_snapshot_wrapper(snapshot_path: str) -> Callable[..., Any]:
    """
    Snapshot wrapper.
    Default implementation: load SB3 PPO from .zip and run predict() per red agent.

    This assumes the snapshot PPO was trained on SINGLE-AGENT obs (7,20,20)
    with action space MultiDiscrete([n_macros, n_targets]).
    """
    path = str(snapshot_path)
    if os.path.isdir(path):
        raise ValueError("Expected SB3 PPO .zip file path, got directory")
    if not os.path.exists(path):
        alt = path + ".zip"
        if os.path.exists(alt):
            path = alt
        else:
            raise FileNotFoundError(path)

    # Lazy import so scripted-only training doesn't require SB3 installed.
    from stable_baselines3 import PPO  # type: ignore

    model = PPO.load(path, device="cpu")

    def _model_expects_mask() -> bool:
        space = getattr(model.policy, "observation_space", None)
        if hasattr(space, "spaces") and isinstance(space.spaces, dict):
            return "mask" in space.spaces
        return False

    def _model_expects_vec() -> bool:
        space = getattr(model.policy, "observation_space", None)
        if hasattr(space, "spaces") and isinstance(space.spaces, dict):
            return "vec" in space.spaces
        return True

    def _coerce_vec(vec: np.ndarray, *, size: int = 4) -> np.ndarray:
        v = np.asarray(vec, dtype=np.float32).reshape(-1)
        if v.size == size:
            return v
        if v.size < size:
            return np.pad(v, (0, size - v.size), mode="constant")
        return v[:size]

    def _build_team_obs(game_field, side: str, obs_template: np.ndarray) -> Dict[str, np.ndarray]:
        agents = game_field.red_agents if side == "red" else game_field.blue_agents
        live = [a for a in agents if a is not None]
        while len(live) < 2:
            live.append(live[0] if live else None)

        obs_list = []
        vec_list = []
        mask_list = []

        for a in live[:2]:
            if a is None:
                obs_list.append(np.zeros_like(obs_template))
                if _model_expects_vec():
                    vec_list.append(np.zeros((4,), dtype=np.float32))
                if _model_expects_mask():
                    mask_list.append(np.ones((5,), dtype=np.float32))
                continue

            o = np.asarray(game_field.build_observation(a), dtype=np.float32)
            obs_list.append(o)

            if _model_expects_vec():
                if hasattr(game_field, "build_continuous_features"):
                    vec_list.append(_coerce_vec(game_field.build_continuous_features(a)))
                else:
                    vec_list.append(np.zeros((4,), dtype=np.float32))

            if _model_expects_mask():
                mm = np.asarray(game_field.get_macro_mask(a), dtype=np.bool_).reshape(-1)
                if mm.shape != (5,) or (not mm.any()):
                    mm = np.ones((5,), dtype=np.bool_)
                mask_list.append(mm.astype(np.float32))

        out = {"grid": np.concatenate(obs_list, axis=0).astype(np.float32)}
        if _model_expects_vec():
            out["vec"] = np.concatenate(vec_list, axis=0).astype(np.float32)
        if _model_expects_mask():
            out["mask"] = np.concatenate(mask_list, axis=0).astype(np.float32)
        return out

    def wrapper(obs, agent, game_field, base_policy=None):
        # Build a team-style dict obs matching SB3 training
        side = str(getattr(agent, "side", "red")).lower()
        obs_template = np.asarray(obs, dtype=np.float32)
        x = _build_team_obs(game_field, side=side, obs_template=obs_template)

        action, _ = model.predict(x, deterministic=True)

        # action can be array([b0_macro, b0_tgt, b1_macro, b1_tgt])
        a = np.asarray(action).reshape(-1)
        if a.size < 4:
            padded = np.zeros((4,), dtype=np.int64)
            padded[: a.size] = a
            a = padded
        elif a.size > 4:
            a = a[:4]

        idx = 0 if getattr(agent, "agent_id", 0) <= 0 else 1
        macro = int(a[idx * 2 + 0])
        tgt = int(a[idx * 2 + 1])
        return {"macro_action": macro, "target_action": tgt}

    return wrapper
