# red_opponents.py
from __future__ import annotations
from typing import Any, Callable, Dict
import os
import numpy as np

from policies import OP3RedPolicy  # you already have this

try:
    from rl.obs_builder import build_team_obs as _obs_builder_build_team_obs
except Exception:
    _obs_builder_build_team_obs = None


def _load_snapshot_policy_only(path: str):
    """
    Load only the policy from an SB3 .zip (no PPO rollout buffer).
    Avoids allocating ~175 MiB per worker when using snapshot opponents in SubprocVecEnv.
    Returns an object with .policy and .predict(obs, deterministic=True) like PPO.
    """
    import torch as th
    from stable_baselines3.common.save_util import load_from_zip_file

    # Ensure custom policy class is importable for cloudpickle when loading in worker processes
    try:
        from rl.train_ppo import MaskedMultiInputPolicy  # noqa: F401
    except Exception:
        pass

    # load_data=True so we get observation_space, action_space, policy_class from the zip.
    # (With load_data=False, data is always None and we cannot build the policy.)
    data, params, _ = load_from_zip_file(path, device="cpu", load_data=True)
    if not data or "policy" not in params:
        raise ValueError(f"Invalid snapshot zip: missing data or policy params in {path!r}")

    obs_space = data["observation_space"]
    action_space = data["action_space"]
    policy_kwargs = dict(data.get("policy_kwargs", {}))
    policy_kwargs.pop("device", None)

    policy_class = data.get("policy_class")
    if policy_class is None:
        try:
            from rl.train_ppo import MaskedMultiInputPolicy
            policy_class = MaskedMultiInputPolicy
        except Exception:
            from stable_baselines3 import PPO
            policy_class = getattr(PPO, "policy_aliases", {}).get("MlpPolicy")
            if policy_class is None:
                raise RuntimeError("Could not resolve policy_class from zip or MaskedMultiInputPolicy")
    elif isinstance(policy_class, str):
        from stable_baselines3 import PPO
        policy_class = getattr(PPO, "policy_aliases", {}).get(policy_class)
        if policy_class is None:
            try:
                from rl.train_ppo import MaskedMultiInputPolicy
                policy_class = MaskedMultiInputPolicy
            except Exception:
                raise RuntimeError(f"Unknown policy_class name in zip: {policy_class!r}")

    # SB3 ActorCritic policies require lr_schedule (unused for inference)
    lr = float(data.get("learning_rate", 1.5e-4))
    lr_schedule = lambda _: lr

    policy = policy_class(obs_space, action_space, lr_schedule, **policy_kwargs)
    policy.load_state_dict(params["policy"], strict=True)
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
        if isinstance(target, (tuple, list)) and len(target) >= 2:
            return (int(action_id), (int(target[0]), int(target[1])))
        return (int(action_id), None if target is None else int(target))

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

    # Load policy only (no PPO rollout buffer) to avoid OOM in SubprocVecEnv workers.
    model = _load_snapshot_policy_only(path)

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

    def _coerce_vec(vec: np.ndarray, *, size: int = 12) -> np.ndarray:
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

        if _obs_builder_build_team_obs is not None:
            out = _obs_builder_build_team_obs(
                game_field,
                live[:2],
                max_agents=2,
                include_mask=_model_expects_mask(),
                tokenized=False,
                vec_size_base=12,
                n_macros=5,
                n_targets=int(getattr(game_field, "num_macro_targets", 8) or 8),
            )
            return out

        # Fallback when rl.obs_builder not available
        obs_list = []
        vec_list = []
        mask_list = []

        for a in live[:2]:
            if a is None:
                obs_list.append(np.zeros_like(obs_template))
                if _model_expects_vec():
                    vec_list.append(np.zeros((12,), dtype=np.float32))
                if _model_expects_mask():
                    mm = np.ones((5,), dtype=np.float32)
                    nt = int(getattr(game_field, "num_macro_targets", 8) or 8)
                    tm = np.ones((nt,), dtype=np.float32)
                    mask_list.append(np.concatenate([mm, tm], axis=0))
                continue

            o = np.asarray(game_field.build_observation(a), dtype=np.float32)
            obs_list.append(o)

            if _model_expects_vec():
                if hasattr(game_field, "build_continuous_features"):
                    vec_list.append(_coerce_vec(game_field.build_continuous_features(a)))
                else:
                    vec_list.append(np.zeros((12,), dtype=np.float32))

            if _model_expects_mask():
                mm = np.asarray(game_field.get_macro_mask(a), dtype=np.bool_).reshape(-1)
                if mm.shape != (5,) or (not mm.any()):
                    mm = np.ones((5,), dtype=np.bool_)
                tm = np.asarray(game_field.get_target_mask(a), dtype=np.bool_).reshape(-1)
                nt = int(getattr(game_field, "num_macro_targets", 8) or 8)
                if tm.shape != (nt,) or (not tm.any()):
                    tm = np.ones((nt,), dtype=np.bool_)
                mask_list.append(np.concatenate([mm.astype(np.float32), tm.astype(np.float32)], axis=0))

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
