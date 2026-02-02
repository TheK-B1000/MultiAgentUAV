import os
import sys
import csv
import math
from typing import Optional, Tuple, Any, List, Dict

import numpy as np
import pygame as pg

# SB3/torch imported lazily when loading a PPO/HPPO model (avoids DLL errors if viewer runs scripted-only)

from viewer_game_field import ViewerGameField
from macro_actions import MacroAction
from policies import OP3RedPolicy
from config import MAP_NAME, MAP_PATH

# Match training constants (use same shapes)
from game_field import CNN_COLS, CNN_ROWS, NUM_CNN_CHANNELS, make_game_field

try:
    from rl.obs_builder import build_team_obs as _viewer_obs_builder
except Exception:
    _viewer_obs_builder = None

# ----------------------------
# MODEL PATHS (edit these)
# ----------------------------
# Point this to your Phase 1 SB3 output:
#   checkpoints_sb3/research_model_phase1.zip
# Or a snapshot:
#   checkpoints_sb3/self_play_pool/sp_snapshot_ep000050.zip
DEFAULT_PPO_MODEL_PATH = "rl/checkpoints_sb3/final_ppo_curriculum_v2.zip"
DEFAULT_HPPO_LOW_MODEL_PATH = "rl/checkpoints_sb3/hppo_low_hppo_attack_defend.zip"
DEFAULT_HPPO_HIGH_MODEL_PATH = "rl/checkpoints_sb3/hppo_high_hppo_attack_defend.zip"

# IMPORTANT: keep this order consistent with training
USED_MACROS = [
    MacroAction.GO_TO,      # 0
    MacroAction.GRAB_MINE,  # 1
    MacroAction.GET_FLAG,   # 2
    MacroAction.PLACE_MINE, # 3
    MacroAction.GO_HOME,    # 4
]
N_MACROS = len(USED_MACROS)


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def _resolve_zip_path(path: str) -> Optional[str]:
    if not path:
        return None
    if os.path.exists(path) and path.endswith(".zip"):
        return path
    if os.path.exists(path + ".zip"):
        return path + ".zip"
    if os.path.exists(path):
        # Some people pass a directory-like SB3 save prefix; SB3 expects the .zip file.
        # If it's not .zip but exists, we still try it.
        return path
    return None


class SB3TeamPPOPolicy:
    """
    Viewer-side SB3 PPO wrapper.

    SB3 model outputs indices:
      action = (b0_macro_idx, b0_target_idx, b1_macro_idx, b1_target_idx)

    Viewer internal-policy expects:
      (MacroAction, target_cell)

    So we translate indices -> enums/cells here.
    """

    def __init__(
        self,
        model_path: str,
        env: ViewerGameField,
        deterministic: bool = True,
        viewer_use_obs_builder: bool = True,
    ):
        self.model_path_raw = model_path
        self.model_path: Optional[str] = _resolve_zip_path(model_path)
        self.model_loaded: bool = False

        self.model: Optional[Any] = None
        self.deterministic = bool(deterministic)
        self._viewer_use_obs_builder = bool(viewer_use_obs_builder)

        # Targets count (fallback)
        self.n_targets = int(getattr(env, "num_macro_targets", 8) or 8)

        # Cache joint action once per sim tick
        self._cache_tick: int = -1
        self._cache_action: np.ndarray = np.array([0, 0, 0, 0], dtype=np.int64)

        if self.model_path is None:
            print(f"[CTFViewer] PPO model not found: {model_path} (or .zip)")
            return

        try:
            from stable_baselines3 import PPO as SB3PPO
            self.model = SB3PPO.load(self.model_path, device="cpu")
            self.model.policy.set_training_mode(False)
            self.model_loaded = True
            print(f"[CTFViewer] Loaded SB3 PPO model from: {self.model_path}")
        except OSError as e:
            print(f"[CTFViewer] Torch/SB3 DLL error (try: reinstall torch, or run viewer in Default mode): {e}")
            self.model_loaded = False
        except Exception as e:
            print(f"[CTFViewer] Failed to load SB3 PPO model '{self.model_path}': {e}")
            self.model_loaded = False

    def reset_cache(self) -> None:
        self._cache_tick = -1
        self._cache_action = np.array([0, 0, 0, 0], dtype=np.int64)

    def _model_expects_mask(self) -> bool:
        if self.model is None:
            return False
        space = getattr(self.model.policy, "observation_space", None)
        if hasattr(space, "spaces") and isinstance(space.spaces, dict):
            return "mask" in space.spaces
        return False

    def _model_expects_vec(self) -> bool:
        if self.model is None:
            return True
        space = getattr(self.model.policy, "observation_space", None)
        if hasattr(space, "spaces") and isinstance(space.spaces, dict):
            return "vec" in space.spaces
        return True

    def _model_vec_size(self) -> int:
        if self.model is None:
            return 0
        space = getattr(self.model.policy, "observation_space", None)
        if hasattr(space, "spaces") and isinstance(space.spaces, dict):
            vec_space = space.spaces.get("vec", None)
            if hasattr(vec_space, "shape") and vec_space.shape:
                try:
                    return int(vec_space.shape[0])
                except Exception:
                    return 0
        return 0

    def _vec_size_per_agent(self) -> int:
        total = self._model_vec_size()
        if total <= 0:
            return 12
        return max(1, int(total // 2))

    def _coerce_vec(self, vec: np.ndarray, *, size: int = 12) -> np.ndarray:
        v = np.asarray(vec, dtype=np.float32).reshape(-1)
        if v.size == size:
            return v
        if v.size < size:
            return np.pad(v, (0, size - v.size), mode="constant")
        return v[:size]

    def _build_team_obs(self, game_field: ViewerGameField, side: str) -> Dict[str, np.ndarray]:
        # In viewer we mainly use blue, but keep side for completeness
        agents = game_field.blue_agents if side == "blue" else game_field.red_agents
        live = [a for a in agents if a is not None]
        while len(live) < 2:
            live.append(live[0] if live else None)

        if self._viewer_use_obs_builder and _viewer_obs_builder is not None:
            return _viewer_obs_builder(
                game_field,
                live[:2],
                max_agents=2,
                include_mask=self._model_expects_mask(),
                tokenized=False,
                vec_size_base=12,
                n_macros=N_MACROS,
                n_targets=int(getattr(game_field, "num_macro_targets", 8) or 8),
            )

        obs_list: List[np.ndarray] = []
        vec_list: List[np.ndarray] = []
        mask_list: List[np.ndarray] = []

        for a in live[:2]:
            if a is None:
                obs_list.append(np.zeros((NUM_CNN_CHANNELS, CNN_ROWS, CNN_COLS), dtype=np.float32))
                if self._model_expects_vec():
                    vec_list.append(np.zeros((self._vec_size_per_agent(),), dtype=np.float32))
                if self._model_expects_mask():
                    mm = np.ones((N_MACROS,), dtype=np.float32)
                    tm = np.ones((int(getattr(game_field, "num_macro_targets", 8) or 8),), dtype=np.float32)
                    mask_list.append(np.concatenate([mm, tm], axis=0))
                continue

            o = np.asarray(game_field.build_observation(a), dtype=np.float32)  # [7,20,20]
            obs_list.append(o)

            if self._model_expects_vec():
                if hasattr(game_field, "build_continuous_features"):
                    vec_list.append(self._coerce_vec(game_field.build_continuous_features(a), size=self._vec_size_per_agent()))
                else:
                    vec_list.append(np.zeros((self._vec_size_per_agent(),), dtype=np.float32))

            if self._model_expects_mask():
                mm = np.asarray(game_field.get_macro_mask(a), dtype=np.bool_).reshape(-1)
                if mm.shape != (N_MACROS,) or (not mm.any()):
                    mm = np.ones((N_MACROS,), dtype=np.bool_)
                tm = np.asarray(game_field.get_target_mask(a), dtype=np.bool_).reshape(-1)
                nt = int(getattr(game_field, "num_macro_targets", 8) or 8)
                if tm.shape != (nt,) or (not tm.any()):
                    tm = np.ones((nt,), dtype=np.bool_)
                mask_list.append(np.concatenate([mm.astype(np.float32), tm.astype(np.float32)], axis=0))

        grid = np.concatenate(obs_list, axis=0).astype(np.float32)   # [14,20,20]
        out: Dict[str, np.ndarray] = {"grid": grid}
        if self._model_expects_vec():
            out["vec"] = np.concatenate(vec_list, axis=0).astype(np.float32)
        if self._model_expects_mask():
            out["mask"] = np.concatenate(mask_list, axis=0).astype(np.float32)  # [10]
        return out

    def _compute_joint_action_if_needed(self, game_field: ViewerGameField, tick: int, side: str) -> None:
        if not self.model_loaded or self.model is None:
            self._cache_tick = tick
            self._cache_action = np.array([0, 0, 0, 0], dtype=np.int64)
            return

        if self._cache_tick == tick:
            return

        obs = self._build_team_obs(game_field, side)
        act, _ = self.model.predict(obs, deterministic=self.deterministic)
        a = np.asarray(act).reshape(-1).astype(np.int64)

        # Ensure shape [4]
        if a.size < 4:
            padded = np.zeros((4,), dtype=np.int64)
            padded[: a.size] = a
            a = padded
        elif a.size > 4:
            a = a[:4]

        # Normalize ranges
        a[0] = int(a[0]) % N_MACROS
        a[2] = int(a[2]) % N_MACROS

        nt = max(1, int(getattr(game_field, "num_macro_targets", self.n_targets) or self.n_targets))
        a[1] = int(a[1]) % nt
        a[3] = int(a[3]) % nt

        self._cache_tick = tick
        self._cache_action = a

    def _resolve_target_cell(self, game_field: ViewerGameField, target_idx: int) -> Tuple[int, int]:
        """
        Convert target index -> actual grid cell.
        Falls back safely if your ViewerGameField/GameField differs.
        """
        # Preferred: your existing macro-target API
        fn = getattr(game_field, "get_macro_target", None)
        if callable(fn):
            try:
                t = fn(int(target_idx))
                if isinstance(t, (tuple, list)) and len(t) >= 2:
                    return (int(t[0]), int(t[1]))
            except Exception:
                pass

        # Alternate: maybe stored list
        mt = getattr(game_field, "macro_targets", None)
        if isinstance(mt, list) and len(mt) > 0:
            i = int(target_idx) % len(mt)
            try:
                t = mt[i]
                if isinstance(t, (tuple, list)) and len(t) >= 2:
                    return (int(t[0]), int(t[1]))
            except Exception:
                pass

        # Absolute fallback: center-ish
        cols = int(getattr(game_field, "col_count", 20) or 20)
        rows = int(getattr(game_field, "row_count", 20) or 20)
        return (max(0, cols // 2), max(0, rows // 2))

    def act_for_agent(self, agent: Any, game_field: ViewerGameField, tick: int) -> Tuple[Any, Tuple[int, int]]:
        """
        Returns what the VIEWER internal-policy pipeline expects:
          (MacroAction enum, target_cell)
        """
        side = str(getattr(agent, "side", "blue")).lower()
        self._compute_joint_action_if_needed(game_field, tick=tick, side=side)

        # Map agent_id -> [0,1]
        aid = _safe_int(getattr(agent, "agent_id", 0), 0)
        aid = 0 if aid <= 0 else 1

        macro_idx = int(self._cache_action[aid * 2 + 0]) % N_MACROS
        target_idx = int(self._cache_action[aid * 2 + 1])

        macro = USED_MACROS[macro_idx]  # <-- THIS is the crucial translation

        # Some macros can ignore target, but we still provide one
        if macro == MacroAction.PLACE_MINE:
            # place mine "here"
            try:
                ax = int(getattr(agent, "x", 0))
                ay = int(getattr(agent, "y", 0))
                return macro, (ax, ay)
            except Exception:
                return macro, self._resolve_target_cell(game_field, target_idx)

        # Default: resolve from target index
        tgt_cell = self._resolve_target_cell(game_field, target_idx)
        return macro, tgt_cell



class SB3TeamHPPOPolicy:
    """
    Viewer-side HPPO wrapper.

    Uses a high-level PPO policy to choose ATTACK/DEFEND,
    then conditions the low-level PPO on that mode.
    """

    def __init__(
        self,
        low_model_path: str,
        high_model_path: str,
        env: ViewerGameField,
        *,
        deterministic: bool = True,
        mode_interval_ticks: int = 8,
        viewer_use_obs_builder: bool = True,
    ):
        self.low_model_path_raw = low_model_path
        self.high_model_path_raw = high_model_path
        self.low_model_path = _resolve_zip_path(low_model_path)
        self.high_model_path = _resolve_zip_path(high_model_path)
        self.model_loaded = False

        self.low_model: Optional[Any] = None
        self.high_model: Optional[Any] = None
        self.deterministic = bool(deterministic)
        self.mode_interval_ticks = max(1, int(mode_interval_ticks))
        self._viewer_use_obs_builder = bool(viewer_use_obs_builder)

        # Targets count (fallback)
        self.n_targets = int(getattr(env, "num_macro_targets", 8) or 8)

        # Cache joint action + mode once per sim tick
        self._cache_tick: int = -1
        self._cache_action: np.ndarray = np.array([0, 0, 0, 0], dtype=np.int64)
        self._cache_mode: int = 0
        self._last_mode_tick: int = -1

        if self.low_model_path is None or self.high_model_path is None:
            print("[CTFViewer] HPPO models not found (low/high .zip).")
            return

        try:
            from stable_baselines3 import PPO as SB3PPO
            self.low_model = SB3PPO.load(self.low_model_path, device="cpu")
            self.low_model.policy.set_training_mode(False)
            self.high_model = SB3PPO.load(self.high_model_path, device="cpu")
            self.high_model.policy.set_training_mode(False)
            self.model_loaded = True
            print(f"[CTFViewer] Loaded HPPO low from: {self.low_model_path}")
            print(f"[CTFViewer] Loaded HPPO high from: {self.high_model_path}")
        except OSError as e:
            print(f"[CTFViewer] Torch/SB3 DLL error (try: reinstall torch, or run viewer in Default mode): {e}")
            self.model_loaded = False
        except Exception as e:
            print(f"[CTFViewer] Failed to load HPPO models: {e}")
            self.model_loaded = False

    def reset_cache(self) -> None:
        self._cache_tick = -1
        self._cache_action = np.array([0, 0, 0, 0], dtype=np.int64)
        self._cache_mode = 0
        self._last_mode_tick = -1

    def _model_expects_mask(self) -> bool:
        if self.low_model is None:
            return False
        space = getattr(self.low_model.policy, "observation_space", None)
        if hasattr(space, "spaces") and isinstance(space.spaces, dict):
            return "mask" in space.spaces
        return False

    def _model_expects_vec(self) -> bool:
        if self.low_model is None:
            return True
        space = getattr(self.low_model.policy, "observation_space", None)
        if hasattr(space, "spaces") and isinstance(space.spaces, dict):
            return "vec" in space.spaces
        return True

    def _model_vec_size(self) -> int:
        if self.low_model is None:
            return 0
        space = getattr(self.low_model.policy, "observation_space", None)
        if hasattr(space, "spaces") and isinstance(space.spaces, dict):
            vec_space = space.spaces.get("vec", None)
            if hasattr(vec_space, "shape") and vec_space.shape:
                try:
                    return int(vec_space.shape[0])
                except Exception:
                    return 0
        return 0

    def _vec_size_per_agent(self) -> int:
        total = self._model_vec_size()
        if total <= 0:
            return 12
        return max(1, int(total // 2))

    def _coerce_vec(self, vec: np.ndarray, *, size: int) -> np.ndarray:
        v = np.asarray(vec, dtype=np.float32).reshape(-1)
        if v.size == size:
            return v
        if v.size < size:
            return np.pad(v, (0, size - v.size), mode="constant")
        return v[:size]

    def _build_high_level_obs(self, game_field: ViewerGameField) -> np.ndarray:
        gm = getattr(game_field, "manager", None)
        phase_name = str(getattr(gm, "phase_name", "OP1")).upper() if gm is not None else "OP1"
        opp_tag = str(getattr(game_field, "opponent_mode", "OP3")).upper()
        vecs = []
        for agent in getattr(game_field, "blue_agents", [])[:2]:
            if agent is None:
                continue
            try:
                if hasattr(agent, "isEnabled") and (not agent.isEnabled()):
                    continue
            except Exception:
                pass
            if hasattr(game_field, "build_continuous_features"):
                vecs.append(np.asarray(game_field.build_continuous_features(agent), dtype=np.float32))

        if vecs:
            avg_vec = np.mean(np.stack(vecs, axis=0), axis=0)
        else:
            avg_vec = np.zeros((12,), dtype=np.float32)

        extras = []
        if gm is not None:
            score_limit = max(1.0, float(getattr(gm, "score_limit", 3)))
            blue_score = float(getattr(gm, "blue_score", 0)) / score_limit
            red_score = float(getattr(gm, "red_score", 0)) / score_limit
            score_diff = (float(getattr(gm, "blue_score", 0)) - float(getattr(gm, "red_score", 0))) / score_limit
            extras.extend([blue_score, red_score, score_diff])
            extras.extend([
                1.0 if bool(getattr(gm, "blue_flag_taken", False)) else 0.0,
                1.0 if bool(getattr(gm, "red_flag_taken", False)) else 0.0,
            ])
            max_time = max(1.0, float(getattr(gm, "max_time", 300.0)))
            current_time = float(getattr(gm, "current_time", max_time))
            extras.append(max(0.0, min(1.0, current_time / max_time)))

        phase_vec = np.zeros((3,), dtype=np.float32)
        phase_map = {"OP1": 0, "OP2": 1, "OP3": 2}
        if phase_name in phase_map:
            phase_vec[phase_map[phase_name]] = 1.0
        extras.extend(phase_vec.tolist())

        opp_vec = np.zeros((5,), dtype=np.float32)
        opp_map = {"OP1": 0, "OP2": 1, "OP3_EASY": 2, "OP3": 3, "OP3_HARD": 4}
        if opp_tag in opp_map:
            opp_vec[opp_map[opp_tag]] = 1.0
        extras.extend(opp_vec.tolist())

        if extras:
            out = np.concatenate([avg_vec, np.asarray(extras, dtype=np.float32)], axis=0)
        else:
            out = avg_vec

        if self.high_model is not None:
            space = getattr(self.high_model.policy, "observation_space", None)
            if hasattr(space, "shape") and space.shape:
                target = int(space.shape[0])
                if out.size < target:
                    out = np.pad(out, (0, target - out.size), mode="constant")
                elif out.size > target:
                    out = out[:target]
        return out.astype(np.float32)

    def _build_team_obs(self, game_field: ViewerGameField, side: str, mode: int) -> Dict[str, np.ndarray]:
        agents = game_field.blue_agents if side == "blue" else game_field.red_agents
        live = [a for a in agents if a is not None]
        while len(live) < 2:
            live.append(live[0] if live else None)

        if self._viewer_use_obs_builder and _viewer_obs_builder is not None:
            per_agent_size = self._vec_size_per_agent()
            base_size = 12
            extra = max(0, per_agent_size - base_size)

            def vec_append_mode(base_vec: np.ndarray) -> np.ndarray:
                v = np.asarray(base_vec, dtype=np.float32).reshape(-1)[:base_size]
                if extra == 2:
                    mode_vec = np.zeros((2,), dtype=np.float32)
                    mode_vec[max(0, min(1, int(mode)))] = 1.0
                    return np.concatenate([v, mode_vec], axis=0)
                if extra == 1:
                    return np.concatenate([v, np.asarray([float(mode)], dtype=np.float32)], axis=0)
                return v

            return _viewer_obs_builder(
                game_field,
                live[:2],
                max_agents=2,
                include_mask=self._model_expects_mask(),
                tokenized=False,
                vec_size_base=12,
                n_macros=N_MACROS,
                n_targets=int(getattr(game_field, "num_macro_targets", 8) or 8),
                vec_append_fn=vec_append_mode,
            )

        obs_list: List[np.ndarray] = []
        vec_list: List[np.ndarray] = []
        mask_list: List[np.ndarray] = []

        per_agent_size = self._vec_size_per_agent()
        base_size = 12
        extra = max(0, per_agent_size - base_size)

        for a in live[:2]:
            if a is None:
                obs_list.append(np.zeros((NUM_CNN_CHANNELS, CNN_ROWS, CNN_COLS), dtype=np.float32))
                if self._model_expects_vec():
                    vec = np.zeros((base_size,), dtype=np.float32)
                    if extra == 2:
                        mode_vec = np.zeros((2,), dtype=np.float32)
                        mode_vec[max(0, min(1, int(mode)))] = 1.0
                        vec = np.concatenate([vec, mode_vec], axis=0)
                    elif extra == 1:
                        vec = np.concatenate([vec, np.asarray([float(mode)], dtype=np.float32)], axis=0)
                    vec_list.append(self._coerce_vec(vec, size=per_agent_size))
                if self._model_expects_mask():
                    mm = np.ones((N_MACROS,), dtype=np.float32)
                    tm = np.ones((int(getattr(game_field, "num_macro_targets", 8) or 8),), dtype=np.float32)
                    mask_list.append(np.concatenate([mm, tm], axis=0))
                continue

            o = np.asarray(game_field.build_observation(a), dtype=np.float32)
            obs_list.append(o)

            if self._model_expects_vec():
                if hasattr(game_field, "build_continuous_features"):
                    vec = np.asarray(game_field.build_continuous_features(a), dtype=np.float32)
                else:
                    vec = np.zeros((base_size,), dtype=np.float32)
                vec = self._coerce_vec(vec, size=base_size)
                if extra == 2:
                    mode_vec = np.zeros((2,), dtype=np.float32)
                    mode_vec[max(0, min(1, int(mode)))] = 1.0
                    vec = np.concatenate([vec, mode_vec], axis=0)
                elif extra == 1:
                    vec = np.concatenate([vec, np.asarray([float(mode)], dtype=np.float32)], axis=0)
                vec_list.append(self._coerce_vec(vec, size=per_agent_size))

            if self._model_expects_mask():
                mm = np.asarray(game_field.get_macro_mask(a), dtype=np.bool_).reshape(-1)
                if mm.shape != (N_MACROS,) or (not mm.any()):
                    mm = np.ones((N_MACROS,), dtype=np.bool_)
                tm = np.asarray(game_field.get_target_mask(a), dtype=np.bool_).reshape(-1)
                nt = int(getattr(game_field, "num_macro_targets", 8) or 8)
                if tm.shape != (nt,) or (not tm.any()):
                    tm = np.ones((nt,), dtype=np.bool_)
                mask_list.append(np.concatenate([mm.astype(np.float32), tm.astype(np.float32)], axis=0))

        grid = np.concatenate(obs_list, axis=0).astype(np.float32)
        out: Dict[str, np.ndarray] = {"grid": grid}
        if self._model_expects_vec():
            out["vec"] = np.concatenate(vec_list, axis=0).astype(np.float32)
        if self._model_expects_mask():
            out["mask"] = np.concatenate(mask_list, axis=0).astype(np.float32)
        return out

    def _compute_mode_if_needed(self, game_field: ViewerGameField, tick: int) -> int:
        if (tick - self._last_mode_tick) < self.mode_interval_ticks:
            return int(self._cache_mode)
        if self.high_model is None:
            self._cache_mode = 0
            self._last_mode_tick = tick
            return 0
        obs = self._build_high_level_obs(game_field)
        mode, _ = self.high_model.predict(obs, deterministic=self.deterministic)
        mode = int(np.asarray(mode).reshape(-1)[0])
        self._cache_mode = max(0, min(1, mode))
        self._last_mode_tick = tick
        return int(self._cache_mode)

    def _compute_joint_action_if_needed(self, game_field: ViewerGameField, tick: int, side: str) -> None:
        if not self.model_loaded or self.low_model is None:
            self._cache_tick = tick
            self._cache_action = np.array([0, 0, 0, 0], dtype=np.int64)
            return
        if tick == self._cache_tick:
            return
        self._cache_tick = tick

        mode = self._compute_mode_if_needed(game_field, tick)
        obs = self._build_team_obs(game_field, side=side, mode=mode)
        action, _ = self.low_model.predict(obs, deterministic=self.deterministic)
        self._cache_action = np.asarray(action).reshape(-1)

    def act_for_agent(self, agent, game_field: ViewerGameField, *, tick: int) -> Tuple[MacroAction, Tuple[int, int]]:
        self._compute_joint_action_if_needed(game_field, tick, side="blue")
        idx = 0 if agent == game_field.blue_agents[0] else 2

        macro_idx = int(self._cache_action[idx]) % N_MACROS
        tgt_idx = int(self._cache_action[idx + 1]) % int(getattr(game_field, "num_macro_targets", 8) or 8)

        macro = USED_MACROS[macro_idx]
        target = game_field.get_macro_target(tgt_idx)
        return macro, target

class CTFViewer:
    def __init__(
        self,
        ppo_model_path: str = DEFAULT_PPO_MODEL_PATH,
        hppo_low_path: str = DEFAULT_HPPO_LOW_MODEL_PATH,
        hppo_high_path: str = DEFAULT_HPPO_HIGH_MODEL_PATH,
        viewer_use_obs_builder: bool = True,
    ):
        if MAP_NAME or MAP_PATH:
            base = make_game_field(map_name=MAP_NAME or None, map_path=MAP_PATH or None)
            grid = [list(row) for row in getattr(base, "grid", [[0] * 20 for _ in range(20)])]
        else:
            rows, cols = 20, 20
            grid = [[0] * cols for _ in range(rows)]

        self.game_field = ViewerGameField(grid)
        self.game_manager = self.game_field.getGameManager()
        self._set_phase_op3()

        # Ensure viewer uses internal policies
        if hasattr(self.game_field, "use_internal_policies"):
            self.game_field.use_internal_policies = True
        if hasattr(self.game_field, "set_external_control"):
            try:
                self.game_field.set_external_control("blue", False)
                self.game_field.set_external_control("red", False)
            except Exception:
                pass

        # --- Pygame setup ---
        pg.init()
        self.size = (1024, 720)
        try:
            self.screen = pg.display.set_mode(self.size, pg.SCALED | pg.DOUBLEBUF, vsync=1)
        except TypeError:
            self.screen = pg.display.set_mode(self.size, pg.SCALED | pg.DOUBLEBUF)

        pg.display.set_caption("UAV CTF Viewer | Blue: OP3/PPO vs Red: OP3 | SB3 PPO League (.zip)")
        self.clock = pg.time.Clock()
        self.font = pg.font.SysFont(None, 26)
        self.bigfont = pg.font.SysFont(None, 48)

        self.input_active = False
        self.input_text = ""

        # Sim tick counter (used for PPO action caching per update substep)
        self.sim_tick: int = 0

        # Sanity
        if getattr(self.game_field, "blue_agents", None):
            dummy_obs = self.game_field.build_observation(self.game_field.blue_agents[0])
            try:
                c = len(dummy_obs)
                h = len(dummy_obs[0])
                w = len(dummy_obs[0][0])
                print(f"[CTFViewer] Detected CNN obs shape: C={c}, H={h}, W={w}")
                print(f"[CTFViewer] num_macro_targets: {_safe_int(getattr(self.game_field, 'num_macro_targets', 0), 0)}")
            except Exception:
                print("[CTFViewer] Could not infer CNN obs shape cleanly.")
        else:
            print("[CTFViewer] No agents spawned; cannot infer obs shape.")

        # ---- Policies ----
        if hasattr(self.game_field, "policies") and isinstance(self.game_field.policies, dict):
            self.game_field.policies["red"] = OP3RedPolicy("red")

        self.blue_op3_baseline = OP3RedPolicy("blue")

        use_obs_builder = bool(viewer_use_obs_builder)
        self.blue_ppo_team = SB3TeamPPOPolicy(
            ppo_model_path, self.game_field, deterministic=True, viewer_use_obs_builder=use_obs_builder
        )
        self.blue_hppo_team = SB3TeamHPPOPolicy(
            hppo_low_path,
            hppo_high_path,
            self.game_field,
            deterministic=True,
            viewer_use_obs_builder=use_obs_builder,
        )

        # Blue policy callable (installed into game_field.policies["blue"])
        self._blue_policy_callable = None

        # Mode
        self.blue_mode: str = "DEFAULT"
        self._apply_blue_mode(self.blue_mode)
        self._reset_op3_policies()

    def _set_phase_op3(self) -> None:
        if hasattr(self.game_manager, "set_phase"):
            try:
                self.game_manager.set_phase("OP3")
            except Exception:
                pass
        if hasattr(self.game_field, "set_phase"):
            try:
                self.game_field.set_phase("OP3")
            except Exception:
                pass

    def _available_modes(self) -> List[str]:
        modes = ["DEFAULT"]
        if self.blue_ppo_team and self.blue_ppo_team.model_loaded:
            modes.append("PPO")
        if self.blue_hppo_team and self.blue_hppo_team.model_loaded:
            modes.append("HPPO")
        return modes

    def _apply_blue_mode(self, mode: str) -> None:
        if not (hasattr(self.game_field, "policies") and isinstance(self.game_field.policies, dict)):
            return

        if mode == "PPO" and self.blue_ppo_team and self.blue_ppo_team.model_loaded:
            # install a callable that slices the cached joint action
            def blue_policy(obs, agent, game_field):
                return self.blue_ppo_team.act_for_agent(agent, game_field, tick=self.sim_tick)

            self._blue_policy_callable = blue_policy
            self.game_field.policies["blue"] = self._blue_policy_callable
            self.blue_mode = "PPO"
            self.blue_ppo_team.reset_cache()
            print("[CTFViewer] Blue → PPO (SB3 .zip)")

        elif mode == "HPPO" and self.blue_hppo_team and self.blue_hppo_team.model_loaded:
            def blue_policy(obs, agent, game_field):
                return self.blue_hppo_team.act_for_agent(agent, game_field, tick=self.sim_tick)

            self._blue_policy_callable = blue_policy
            self.game_field.policies["blue"] = self._blue_policy_callable
            self.blue_mode = "HPPO"
            self.blue_hppo_team.reset_cache()
            print("[CTFViewer] Blue → HPPO (high/low)")

        else:
            self._blue_policy_callable = None
            self.game_field.policies["blue"] = self.blue_op3_baseline
            self.blue_mode = "DEFAULT"
            print("[CTFViewer] Blue → Default baseline")

    def _cycle_blue_mode(self) -> None:
        modes = self._available_modes()
        i = modes.index(self.blue_mode) if self.blue_mode in modes else 0
        nxt = modes[(i + 1) % len(modes)]
        self._apply_blue_mode(nxt)
        if self.blue_mode == "DEFAULT":
            self._reset_op3_policies()

    def _reset_op3_policies(self):
        if hasattr(self.game_field, "policies") and isinstance(self.game_field.policies, dict):
            for side in ("blue", "red"):
                pol = self.game_field.policies.get(side)
                if isinstance(pol, OP3RedPolicy) and hasattr(pol, "reset"):
                    try:
                        pol.reset()
                    except Exception:
                        pass

    # ----------------------------
    # Main loop (fixed-step sim + alpha render)
    # ----------------------------
    def run(self):
        running = True

        fixed_dt = 1.0 / 60.0
        acc = 0.0

        max_frame_dt = 1.0 / 30.0
        max_substeps = 5

        while running:
            frame_dt = self.clock.tick_busy_loop(120) / 1000.0
            if frame_dt > max_frame_dt:
                frame_dt = max_frame_dt
            acc += frame_dt

            for event in pg.event.get():
                if event.type == pg.QUIT:
                    running = False
                elif event.type == pg.KEYDOWN:
                    if self.input_active:
                        self.handle_input_key(event)
                    else:
                        self.handle_main_key(event)

            steps = 0
            while acc >= fixed_dt and steps < max_substeps:
                # Advance sim
                self.sim_tick += 1
                self.game_field.update(fixed_dt)
                acc -= fixed_dt
                steps += 1

            if steps == max_substeps:
                acc = 0.0

            alpha = acc / fixed_dt
            self.draw(alpha=alpha)
            pg.display.flip()

        pg.quit()
        sys.exit()

    # ----------------------------
    # Evaluation mode: run N episodes and collect metrics
    # ----------------------------
    def evaluate_model(
        self,
        num_episodes: int = 100,
        save_csv: Optional[str] = None,
        headless: bool = False,
        opponent: str = "OP3",
    ) -> Dict[str, Any]:
        """
        Evaluate trained model performance by running N episodes and collecting IROS-style metrics.

        Args:
            num_episodes: Number of episodes to run
            save_csv: Path to save CSV (e.g., "eval_results.csv"). If None, auto-generates name.
            headless: If True, run without display (faster). If False, show viewer.
            opponent: Red opponent tag ("OP1", "OP2", "OP3", "OP3_EASY", "OP3_HARD")

        Returns:
            Dict with summary statistics and per-episode metrics
        """
        if not headless:
            print("[Eval] Running with display (headless=False). Press ESC to stop early.")
        else:
            print(f"[Eval] Running {num_episodes} episodes headless...")

        # Ensure we're using a trained model
        if self.blue_mode == "DEFAULT":
            if self.blue_ppo_team and self.blue_ppo_team.model_loaded:
                self._apply_blue_mode("PPO")
                print("[Eval] Switched to PPO mode for evaluation")
            elif self.blue_hppo_team and self.blue_hppo_team.model_loaded:
                self._apply_blue_mode("HPPO")
                print("[Eval] Switched to HPPO mode for evaluation")
            else:
                print("[WARN] No trained model loaded! Using DEFAULT baseline.")

        # Set opponent
        if hasattr(self.game_field, "set_red_opponent"):
            try:
                self.game_field.set_red_opponent(str(opponent).upper())
            except Exception:
                pass
        self._set_phase_op3()

        fixed_dt = 1.0 / 60.0
        max_steps_per_episode = 900 * 60  # ~900 seconds at 60 Hz

        episodes = []
        episode_id = 0

        if not headless:
            if not pg.get_init():
                pg.init()
            if not hasattr(self, "screen") or self.screen is None:
                try:
                    self.screen = pg.display.set_mode(self.size, pg.SCALED | pg.DOUBLEBUF, vsync=1)
                except TypeError:
                    self.screen = pg.display.set_mode(self.size, pg.SCALED | pg.DOUBLEBUF)
            if not hasattr(self, "clock") or self.clock is None:
                self.clock = pg.time.Clock()

        try:
            for ep_idx in range(num_episodes):
                episode_id += 1
                self.game_field.reset_default()
                self._set_phase_op3()
                self.sim_tick = 0
                if self.blue_ppo_team:
                    self.blue_ppo_team.reset_cache()
                if self.blue_hppo_team:
                    self.blue_hppo_team.reset_cache()
                self._reset_op3_policies()

                step_count = 0
                running_episode = True
                while step_count < max_steps_per_episode and running_episode:
                    if not headless:
                        for event in pg.event.get():
                            if event.type == pg.QUIT or (event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE):
                                print(f"[Eval] Stopped early at episode {ep_idx + 1}/{num_episodes}")
                                running_episode = False
                                break

                    self.sim_tick += 1
                    self.game_field.update(fixed_dt)
                    step_count += 1

                    gm = self.game_manager
                    if getattr(gm, "game_over", False):
                        break

                    if not headless:
                        self.draw(alpha=1.0)
                        pg.display.flip()
                        self.clock.tick(60)

                # Collect metrics from game_manager (same structure as training)
                gm = self.game_manager
                blue_score = int(getattr(gm, "blue_score", 0))
                red_score = int(getattr(gm, "red_score", 0))
                success = 1 if blue_score > red_score else 0

                time_to_first = getattr(gm, "time_to_first_score", None)
                time_to_first_score = float(time_to_first) if time_to_first is not None else None
                time_to_game_over = getattr(gm, "time_to_game_over", None)
                time_to_game_over_sec = float(time_to_game_over) if time_to_game_over is not None else None
                if time_to_game_over_sec is None:
                    time_to_game_over_sec = float(getattr(gm, "sim_time", 0.0))

                collisions = int(getattr(gm, "collision_count_this_episode", 0))
                near_misses = int(getattr(gm, "near_miss_count_this_episode", 0))
                collision_free = 1 if collisions == 0 else 0

                dists = getattr(gm, "blue_inter_robot_distances", []) or []
                mean_inter_robot_dist = float(np.mean(dists)) if dists else None
                std_inter_robot_dist = float(np.std(dists)) if len(dists) > 1 else (0.0 if dists else None)

                visited = getattr(gm, "blue_zone_visited_cells", set()) or set()
                total_zone = int(getattr(gm, "total_blue_zone_cells", 1)) or 1
                zone_coverage = float(len(visited)) / float(total_zone) if total_zone > 0 else 0.0

                row = {
                    "episode_id": episode_id,
                    "success": success,
                    "time_to_first_score": time_to_first_score,
                    "time_to_game_over": time_to_game_over_sec,
                    "collisions_per_episode": collisions,
                    "near_misses_per_episode": near_misses,
                    "collision_free_episode": collision_free,
                    "mean_inter_robot_dist": mean_inter_robot_dist,
                    "std_inter_robot_dist": std_inter_robot_dist,
                    "zone_coverage": zone_coverage,
                    "phase_name": str(getattr(gm, "phase_name", "OP3")),
                    "opponent_kind": "SCRIPTED",
                    "scripted_tag": str(opponent).upper(),
                    "blue_score": blue_score,
                    "red_score": red_score,
                }
                episodes.append(row)

                if (ep_idx + 1) % 10 == 0:
                    wins = sum(1 for e in episodes if e["success"] == 1)
                    wr = wins / len(episodes)
                    print(f"[Eval] Episode {ep_idx + 1}/{num_episodes} | Win rate: {wr:.2%} ({wins}/{len(episodes)})")

        except KeyboardInterrupt:
            print(f"[Eval] Interrupted at episode {ep_idx + 1}/{num_episodes}")

        # Compute summary statistics
        if not episodes:
            print("[Eval] No episodes completed!")
            return {}

        wins = sum(1 for e in episodes if e["success"] == 1)
        losses = sum(1 for e in episodes if e["success"] == 0)
        win_rate = wins / len(episodes)

        time_to_first_scores = [e["time_to_first_score"] for e in episodes if e["time_to_first_score"] is not None]
        time_to_game_overs = [e["time_to_game_over"] for e in episodes if e["time_to_game_over"] is not None]
        collisions_list = [e["collisions_per_episode"] for e in episodes]
        near_misses_list = [e["near_misses_per_episode"] for e in episodes]
        collision_free_count = sum(1 for e in episodes if e["collision_free_episode"] == 1)
        mean_dists = [e["mean_inter_robot_dist"] for e in episodes if e["mean_inter_robot_dist"] is not None]
        std_dists = [e["std_inter_robot_dist"] for e in episodes if e["std_inter_robot_dist"] is not None]
        zone_coverages = [e["zone_coverage"] for e in episodes]

        summary = {
            "num_episodes": len(episodes),
            "win_rate": win_rate,
            "wins": wins,
            "losses": losses,
            "draws": len(episodes) - wins - losses,
            "mean_time_to_first_score": float(np.mean(time_to_first_scores)) if time_to_first_scores else None,
            "mean_time_to_game_over": float(np.mean(time_to_game_overs)) if time_to_game_overs else None,
            "mean_collisions_per_episode": float(np.mean(collisions_list)),
            "mean_near_misses_per_episode": float(np.mean(near_misses_list)),
            "collision_free_rate": collision_free_count / len(episodes),
            "mean_inter_robot_dist": float(np.mean(mean_dists)) if mean_dists else None,
            "mean_std_inter_robot_dist": float(np.mean(std_dists)) if std_dists else None,
            "mean_zone_coverage": float(np.mean(zone_coverages)),
        }

        # Save CSV
        if save_csv is None:
            model_name = "ppo" if self.blue_mode == "PPO" else ("hppo" if self.blue_mode == "HPPO" else "default")
            save_csv = f"eval_{model_name}_{opponent}_{len(episodes)}ep.csv"

        csv_columns = [
            "episode_id", "success", "time_to_first_score", "time_to_game_over",
            "collisions_per_episode", "near_misses_per_episode", "collision_free_episode",
            "mean_inter_robot_dist", "std_inter_robot_dist", "zone_coverage",
            "phase_name", "opponent_kind", "scripted_tag", "blue_score", "red_score",
        ]

        try:
            os.makedirs(os.path.dirname(os.path.abspath(save_csv)) or ".", exist_ok=True)
            with open(save_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=csv_columns, extrasaction="ignore")
                w.writeheader()
                for row in episodes:
                    fmt_row = {}
                    for k in csv_columns:
                        v = row.get(k)
                        if v is None:
                            fmt_row[k] = ""
                        elif isinstance(v, float):
                            fmt_row[k] = f"{v:.6g}"
                        else:
                            fmt_row[k] = str(v)
                    w.writerow(fmt_row)
            print(f"[Eval] Saved {len(episodes)} episodes to: {save_csv}")
        except Exception as exc:
            print(f"[WARN] Failed to save CSV: {exc}")

        # Print summary
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        print(f"Episodes: {summary['num_episodes']}")
        print(f"Win Rate: {summary['win_rate']:.2%} ({summary['wins']}W / {summary['losses']}L / {summary['draws']}D)")
        if summary["mean_time_to_first_score"] is not None:
            print(f"Mean Time to First Score: {summary['mean_time_to_first_score']:.2f}s")
        if summary["mean_time_to_game_over"] is not None:
            print(f"Mean Time to Game Over: {summary['mean_time_to_game_over']:.2f}s")
        print(f"Mean Collisions/Episode: {summary['mean_collisions_per_episode']:.2f}")
        print(f"Mean Near-Misses/Episode: {summary['mean_near_misses_per_episode']:.2f}")
        print(f"Collision-Free Rate: {summary['collision_free_rate']:.2%}")
        if summary["mean_inter_robot_dist"] is not None:
            print(f"Mean Inter-Robot Distance: {summary['mean_inter_robot_dist']:.2f} cells")
        print(f"Mean Zone Coverage: {summary['mean_zone_coverage']:.2%}")
        print("=" * 60)

        summary["episodes"] = episodes
        return summary

    # ----------------------------
    # Input handling
    # ----------------------------
    def handle_main_key(self, event):
        k = event.key
        if k == pg.K_F1:
            self.game_field.agents_per_team = 2
            self.game_manager.reset_game(reset_scores=True)
            self.game_field.reset_default()
            self._set_phase_op3()
            self.sim_tick = 0
            if self.blue_ppo_team:
                self.blue_ppo_team.reset_cache()
            if self.blue_hppo_team:
                self.blue_hppo_team.reset_cache()
            self._reset_op3_policies()

        elif k == pg.K_F2:
            self.input_active = True
            self.input_text = str(self.game_field.agents_per_team)

        elif k == pg.K_F3:
            self._cycle_blue_mode()

        elif k == pg.K_F4:
            self.game_field.debug_draw_ranges = not getattr(self.game_field, "debug_draw_ranges", False)

        elif k == pg.K_F5:
            self.game_field.debug_draw_mine_ranges = not getattr(self.game_field, "debug_draw_mine_ranges", False)

        elif k == pg.K_r:
            self.game_field.agents_per_team = 2
            self.game_field.reset_default()
            self._set_phase_op3()
            self.sim_tick = 0
            if self.blue_ppo_team:
                self.blue_ppo_team.reset_cache()
            if self.blue_hppo_team:
                self.blue_hppo_team.reset_cache()
            self._reset_op3_policies()

        elif k == pg.K_ESCAPE:
            pg.event.post(pg.event.Event(pg.QUIT))

    def handle_input_key(self, event):
        if event.key == pg.K_RETURN:
            try:
                n = int(self.input_text or "2")
                n = max(1, min(100, n))

                if hasattr(self.game_field, "set_agent_count_and_reset"):
                    self.game_field.set_agent_count_and_reset(n)
                else:
                    self.game_field.agents_per_team = n
                    self.game_field.reset_default()

                self._set_phase_op3()
                self.sim_tick = 0
                if self.blue_ppo_team:
                    self.blue_ppo_team.reset_cache()
                if self.blue_hppo_team:
                    self.blue_hppo_team.reset_cache()
                self._reset_op3_policies()
            except Exception as e:
                print(f"[CTFViewer] Error processing agent count: {e}")
            self.input_active = False

        elif event.key == pg.K_ESCAPE:
            self.input_active = False

        elif event.key == pg.K_BACKSPACE:
            self.input_text = self.input_text[:-1]

        elif event.unicode.isdigit():
            if len(self.input_text) < 3:
                self.input_text += event.unicode

    # ----------------------------
    # Drawing / HUD
    # ----------------------------
    def draw(self, alpha: float = 1.0):
        self.screen.fill((12, 12, 18))

        hud_h = 110
        field_rect = pg.Rect(20, hud_h + 10, self.size[0] - 20, self.size[1] - hud_h - 20)

        # IMPORTANT: ViewerGameField.draw must accept alpha
        self.game_field.draw(self.screen, field_rect, alpha=alpha)

        def txt(text, x, y, color=(230, 230, 240)):
            img = self.font.render(text, True, color)
            self.screen.blit(img, (x, y))

        gm = self.game_manager
        mode = self.blue_mode
        mode_color = (255, 255, 120) if mode == "DEFAULT" else (120, 255, 120)

        txt(
            "F1: Full Reset | F2: Set Agents | F3: Cycle Blue (Default/PPO/HPPO) | F4/F5: Debug | R: Reset",
            30, 15, (200, 200, 220)
        )

        txt(
            f"Blue Mode: {mode} | Agents: {self.game_field.agents_per_team} vs {self.game_field.agents_per_team}",
            30, 45, mode_color
        )

        txt(f"BLUE: {getattr(gm, 'blue_score', 0)}", 30, 80, (100, 180, 255))
        txt(f"RED: {getattr(gm, 'red_score', 0)}", 200, 80, (255, 100, 100))
        txt(f"Time: {int(getattr(gm, 'current_time', 0.0))}s", 380, 80, (220, 220, 255))

        right_x = self.size[0] - 460
        if self.blue_mode == "PPO" and self.blue_ppo_team and self.blue_ppo_team.model_loaded:
            ppo_name = os.path.basename(self.blue_ppo_team.model_path or "")
            txt(f"PPO(.zip): {ppo_name}", right_x, 45, (140, 240, 140))
        elif self.blue_mode == "HPPO" and self.blue_hppo_team and self.blue_hppo_team.model_loaded:
            low_name = os.path.basename(self.blue_hppo_team.low_model_path or "")
            high_name = os.path.basename(self.blue_hppo_team.high_model_path or "")
            txt(f"HPPO(.zip): {low_name} | {high_name}", right_x, 45, (140, 240, 140))
        else:
            txt("PPO/HPPO(.zip): (not loaded)", right_x, 45, (180, 180, 180))

        if self.input_active:
            overlay = pg.Surface(self.size, pg.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))

            box = pg.Rect(0, 0, 500, 200)
            box.center = self.screen.get_rect().center
            pg.draw.rect(self.screen, (40, 40, 80), box, border_radius=12)
            pg.draw.rect(self.screen, (100, 180, 255), box, width=4, border_radius=12)

            title = self.bigfont.render("Enter Agent Count (1-100)", True, (255, 255, 255))
            entry = self.bigfont.render(self.input_text or "_", True, (120, 220, 255))
            hint = self.font.render("Press Enter to confirm • Esc to cancel", True, (200, 200, 200))

            self.screen.blit(title, title.get_rect(center=(box.centerx, box.centery - 50)))
            self.screen.blit(entry, entry.get_rect(center=box.center))
            self.screen.blit(hint, hint.get_rect(center=(box.centerx, box.centery + 60)))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CTF Viewer - Interactive or Evaluation Mode")
    parser.add_argument("--eval", type=int, metavar="N", help="Run evaluation mode: run N episodes and save metrics CSV")
    parser.add_argument("--eval-csv", type=str, metavar="PATH", help="CSV output path for evaluation (default: auto-generated)")
    parser.add_argument("--headless", action="store_true", help="Run evaluation without display (faster)")
    parser.add_argument("--opponent", type=str, default="OP3", help="Red opponent for evaluation (OP1/OP2/OP3/OP3_EASY/OP3_HARD)")
    parser.add_argument("--ppo-model", type=str, help="Path to PPO model .zip file (overrides DEFAULT_PPO_MODEL_PATH)")
    parser.add_argument("--hppo-low", type=str, help="Path to HPPO low-level model .zip")
    parser.add_argument("--hppo-high", type=str, help="Path to HPPO high-level model .zip")

    args = parser.parse_args()

    viewer = CTFViewer(
        ppo_model_path=args.ppo_model or DEFAULT_PPO_MODEL_PATH,
        hppo_low_path=args.hppo_low or DEFAULT_HPPO_LOW_MODEL_PATH,
        hppo_high_path=args.hppo_high or DEFAULT_HPPO_HIGH_MODEL_PATH,
    )

    if args.eval is not None:
        # Evaluation mode
        summary = viewer.evaluate_model(
            num_episodes=args.eval,
            save_csv=args.eval_csv,
            headless=args.headless,
            opponent=args.opponent,
        )
        if not args.headless:
            print("\n[Eval] Evaluation complete. Viewer window will close.")
            viewer.run()  # Show final state
    else:
        # Interactive mode
        viewer.run()
