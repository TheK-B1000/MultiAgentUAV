# red_opponents.py
from __future__ import annotations
from typing import Any, Callable, Optional, Tuple
import os
import numpy as np

from policies import OP3RedPolicy  # you already have this
# If you want SB3 snapshot opponents:
from stable_baselines3 import PPO


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
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    # SB3 stores zip files; allow passing without .zip too
    if os.path.isdir(path):
        raise ValueError("Expected SB3 PPO .zip file path, got directory")

    model = PPO.load(path, device="cpu")

    def wrapper(obs, agent, game_field, base_policy=None):
        # obs from GameField is python list [7][20][20]
        x = np.asarray(obs, dtype=np.float32)

        # SB3 expects observation with batch dim removed; predict handles it
        action, _ = model.predict(x, deterministic=True)

        # action can be array([macro, tgt]) for MultiDiscrete
        try:
            macro = int(action[0])
            tgt = int(action[1]) if len(action) > 1 else 0
        except Exception:
            macro = int(action)
            tgt = 0

        return {"macro_action": macro, "target_action": tgt}

    return wrapper
