import os
import sys
import csv
import math
from typing import Optional, Tuple, Any, List, Dict

import numpy as np

# NumPy 1.x compatibility: models saved with NumPy 2.x reference numpy._core.*.
# On NumPy 1.x those live under numpy.core. Shim so loading works without upgrading.
if not hasattr(np, "_core"):
    import types
    _core = types.ModuleType("numpy._core")
    _core.__path__ = []  # make it a package so "import numpy._core.multiarray" works
    sys.modules["numpy._core"] = _core
    for _name in ("numeric", "multiarray", "umath"):
        try:
            _sub = __import__(f"numpy.core.{_name}", fromlist=[_name])
            setattr(_core, _name, _sub)
            sys.modules[f"numpy._core.{_name}"] = _sub
        except Exception:
            pass

import pygame as pg

# SB3/torch imported lazily when loading a PPO/HPPO model (avoids DLL errors if viewer runs scripted-only)

from viewer_game_field import ViewerGameField
from macro_actions import MacroAction
from policies import OP3RedPolicy
from config import MAP_NAME, MAP_PATH

try:
    from red_opponents import make_snapshot_wrapper
except ImportError:
    make_snapshot_wrapper = None

# Match training constants (use same shapes)
from game_field import CNN_COLS, CNN_ROWS, NUM_CNN_CHANNELS, make_game_field

try:
    from rl.obs_builder import build_team_obs as _viewer_obs_builder
except Exception:
    _viewer_obs_builder = None

# ----------------------------
# MODEL PATHS (edit these)
# ----------------------------
# Resolve relative paths from this script's directory so they work regardless of cwd
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Point this to your Phase 1 SB3 output:
#   checkpoints_sb3/research_model_phase1.zip
# Or a snapshot:
#   checkpoints_sb3/self_play_pool/sp_snapshot_ep000050.zip
DEFAULT_PPO_MODEL_PATH = "rl/checkpoints_sb3/final_ppo_league_curriculum_v3.zip"
# Baseline-specific model paths (auto-selected when testing baselines)
# Training saves to checkpoints_sb3/ under project root (run from AICTFProject)
BASELINE_MODEL_PATHS = {
    "fixed_op3": "checkpoints_sb3/final_ppo_fixed_op3.zip",
    "self_play": "checkpoints_sb3/final_ppo_selfplay.zip",
    "curriculum_no_league": "checkpoints_sb3/final_ppo_noleague.zip",
    "curriculum_league": "checkpoints_sb3/final_ppo_league_curriculum_v3.zip",
}
DEFAULT_HPPO_LOW_MODEL_PATH = "rl/checkpoints_sb3/hppo_low_hppo_attack_defend.zip"
DEFAULT_HPPO_HIGH_MODEL_PATH = "rl/checkpoints_sb3/hppo_high_hppo_attack_defend.zip"
# MAPPO (Multi-Agent PPO) .pth checkpoint from train_mappo.py
DEFAULT_MAPPO_MODEL_PATH = "rl/checkpoints_mappo/final_mappo_league_curriculum.pth"

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


def _try_paths(*candidates: str) -> Optional[str]:
    """Return first candidate that exists, or None."""
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None


def _resolve_zip_path(path: str) -> Optional[str]:
    if not path:
        return None
    candidates = [path]
    if not path.endswith(".zip"):
        candidates.append(path + ".zip")
    if not os.path.isabs(path):
        script_rel = os.path.join(_SCRIPT_DIR, path)
        candidates.append(script_rel)
        if not script_rel.endswith(".zip"):
            candidates.append(script_rel + ".zip")
    return _try_paths(*candidates)




def _resolve_pth_path(path: str) -> Optional[str]:
    """Resolve path to a MAPPO .pth checkpoint."""
    if not path:
        return None
    script_rel = os.path.join(_SCRIPT_DIR, path) if not os.path.isabs(path) else None
    return _try_paths(path, path + ".pth", script_rel, os.path.join(_SCRIPT_DIR, path + ".pth") if script_rel else None)


class MAPPOTeamPolicy:
    """
    Viewer-side wrapper for MAPPO .pth checkpoints (rl_policy.ActorCriticNet).

    Loads state_dict from train_mappo.py; runs per-agent act() with grid obs and
    central critic not used for inference. Returns (MacroAction, target_cell) per agent.
    """

    def __init__(
        self,
        model_path: str,
        env: ViewerGameField,
        deterministic: bool = True,
    ):
        self.model_path_raw = model_path
        self.model_path: Optional[str] = _resolve_pth_path(model_path)
        self.model_loaded: bool = False
        self.policy: Optional[Any] = None
        self.deterministic = bool(deterministic)
        self.n_targets = int(getattr(env, "num_macro_targets", 8) or 8)
        self._cache_tick: int = -1
        self._cache_action: np.ndarray = np.array([0, 0, 0, 0], dtype=np.int64)

        if self.model_path is None:
            print(f"[CTFViewer] MAPPO model not found: {model_path} (or .pth)")
            return

        try:
            import torch as _torch
            from rl_policy import ActorCriticNet

            n_macros = N_MACROS
            n_targets = self.n_targets
            n_agents = 2
            self.policy = ActorCriticNet(
                n_macros=n_macros,
                n_targets=n_targets,
                n_agents=n_agents,
                in_channels=NUM_CNN_CHANNELS,
                height=CNN_ROWS,
                width=CNN_COLS,
            )
            state = _torch.load(self.model_path, map_location="cpu", weights_only=True)
            self.policy.load_state_dict(state)
            self.policy.eval()
            self.model_loaded = True
            print(f"[CTFViewer] Loaded MAPPO model from: {self.model_path}")
        except Exception:
            try:
                import torch as _torch
                from rl_policy import ActorCriticNet

                n_macros = N_MACROS
                n_targets = self.n_targets
                n_agents = 2
                self.policy = ActorCriticNet(
                    n_macros=n_macros,
                    n_targets=n_targets,
                    n_agents=n_agents,
                    in_channels=NUM_CNN_CHANNELS,
                    height=CNN_ROWS,
                    width=CNN_COLS,
                )
                state = _torch.load(self.model_path, map_location="cpu")
                self.policy.load_state_dict(state)
                self.policy.eval()
                self.model_loaded = True
                print(f"[CTFViewer] Loaded MAPPO model from: {self.model_path}")
            except Exception as e2:
                print(f"[CTFViewer] Failed to load MAPPO model '{self.model_path}': {e2}")
                self.model_loaded = False

    def reset_cache(self) -> None:
        self._cache_tick = -1
        self._cache_action = np.array([0, 0, 0, 0], dtype=np.int64)

    def _macro_uses_target(self, macro: int, game_field: ViewerGameField) -> bool:
        """Check if macro action uses a target parameter."""
        from macro_actions import MacroAction
        return macro in (MacroAction.GO_TO, MacroAction.GET_FLAG, MacroAction.GRAB_MINE, MacroAction.PLACE_MINE)

    def _resolve_target_conflicts(self, actions: np.ndarray, game_field: ViewerGameField) -> np.ndarray:
        """
        Fix 6.1: Resolve target conflicts - if both agents pick same target, nearest keeps it,
        other gets next-best valid target (or GO_HOME).
        """
        a = actions.copy()
        blue_agents = getattr(game_field, "blue_agents", []) or []
        if len(blue_agents) < 2:
            return a
        
        from collections import defaultdict
        groups: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        for i in range(2):
            m, t = int(a[i * 2]), int(a[i * 2 + 1])
            if self._macro_uses_target(m, game_field):
                groups[(m, t)].append(i)
        
        for (m, t), agents in groups.items():
            if len(agents) <= 1:
                continue
            try:
                tgt_cell = game_field.get_macro_target(t)
                if tgt_cell is None or len(tgt_cell) < 2:
                    tx, ty = 0, 0
                else:
                    tx, ty = int(tgt_cell[0]), int(tgt_cell[1])
            except Exception:
                continue
            
            def dist_to_target(agent_idx: int) -> float:
                if agent_idx >= len(blue_agents):
                    return float("inf")
                agent = blue_agents[agent_idx]
                if agent is None:
                    return float("inf")
                pos = getattr(agent, "float_pos", (getattr(agent, "x", 0), getattr(agent, "y", 0)))
                dx = float(pos[0]) - tx
                dy = float(pos[1]) - ty
                return dx * dx + dy * dy
            
            agents_sorted = sorted(agents, key=dist_to_target)
            for idx, agent_idx in enumerate(agents_sorted):
                if idx == 0:
                    continue
                if agent_idx >= len(blue_agents):
                    a[agent_idx * 2] = 4  # GO_HOME
                    a[agent_idx * 2 + 1] = 0
                    continue
                agent = blue_agents[agent_idx]
                if agent is None:
                    a[agent_idx * 2] = 4
                    a[agent_idx * 2 + 1] = 0
                    continue
                try:
                    tm = np.asarray(game_field.get_target_mask(agent), dtype=np.bool_).reshape(-1)
                    if tm.shape != (self.n_targets,) or not tm.any():
                        a[agent_idx * 2] = 4
                        a[agent_idx * 2 + 1] = 0
                        continue
                    best_alt = 0
                    best_dist = float("inf")
                    for ti in range(self.n_targets):
                        if not bool(tm[ti]) or ti == t:
                            continue
                        try:
                            tc = game_field.get_macro_target(ti)
                            tcx = int(tc[0])
                            tcy = int(tc[1]) if len(tc) >= 2 else 0
                        except Exception:
                            continue
                        pos = getattr(agent, "float_pos", (getattr(agent, "x", 0), getattr(agent, "y", 0)))
                        d = (float(pos[0]) - tcx) ** 2 + (float(pos[1]) - tcy) ** 2
                        if d < best_dist:
                            best_dist = d
                            best_alt = ti
                    a[agent_idx * 2] = m
                    a[agent_idx * 2 + 1] = best_alt
                except Exception:
                    a[agent_idx * 2] = 4
                    a[agent_idx * 2 + 1] = 0
        
        return a

    def _sanitize_action_with_mask(
        self, 
        macro: int, 
        target: int, 
        game_field: ViewerGameField, 
        agent_idx: int
    ) -> Tuple[int, int]:
        """Enforce action masks: ensure macro and target are valid for this agent."""
        macro = int(macro) % N_MACROS
        target = int(target) % max(1, self.n_targets)
        
        blue_agents = getattr(game_field, "blue_agents", []) or []
        if agent_idx >= len(blue_agents):
            return macro, target
        
        agent = blue_agents[agent_idx]
        if agent is None:
            return macro, target
        
        # Enforce macro mask
        try:
            mm = np.asarray(game_field.get_macro_mask(agent), dtype=np.bool_).reshape(-1)
            if mm.shape == (N_MACROS,) and mm.any():
                if not bool(mm[macro]):
                    valid_macros = np.flatnonzero(mm)
                    macro = int(valid_macros[0]) if valid_macros.size > 0 else 0
        except Exception:
            pass
        
        # Enforce target mask if macro uses target
        try:
            from macro_actions import MacroAction
            uses_target = macro in (MacroAction.GO_TO, MacroAction.GET_FLAG, MacroAction.GRAB_MINE, MacroAction.PLACE_MINE)
            if uses_target:
                tm = np.asarray(game_field.get_target_mask(agent), dtype=np.bool_).reshape(-1)
                if tm.shape == (self.n_targets,) and tm.any():
                    if not bool(tm[target]):
                        valid_targets = np.flatnonzero(tm)
                        target = int(valid_targets[0]) if valid_targets.size > 0 else 0
        except Exception:
            pass
        
        return macro, target

    def _compute_joint_action_if_needed(self, game_field: ViewerGameField, tick: int, side: str) -> None:
        if not self.model_loaded or self.policy is None:
            self._cache_tick = tick
            self._cache_action = np.array([0, 0, 0, 0], dtype=np.int64)
            return
        if self._cache_tick == tick:
            return
        agents = getattr(game_field, "blue_agents", []) if side == "blue" else getattr(game_field, "red_agents", [])
        live = [a for a in agents if a is not None]
        # CRITICAL: Sort by agent_id to match training's canonical ordering (blue_0, blue_1)
        def _get_agent_id(agent):
            try:
                return int(getattr(agent, "agent_id", -1))
            except Exception:
                return -1
        live.sort(key=_get_agent_id)  # Sort by agent_id ascending (0, 1, 2, ...)
        while len(live) < 2:
            live.append(live[0] if live else None)
        import torch as _torch
        actions = []
        for idx, a in enumerate(live[:2]):
            if a is None:
                actions.append((0, 0))
                continue
            obs = np.asarray(game_field.build_observation(a), dtype=np.float32)
            obs_t = _torch.tensor(obs, dtype=_torch.float32).unsqueeze(0)
            with _torch.no_grad():
                out = self.policy.act(obs_t, agent=a, game_field=game_field, deterministic=self.deterministic)
            macro_idx = int(out["macro_action"][0].item()) % N_MACROS
            target_idx = int(out["target_action"][0].item()) % self.n_targets
            # Enforce masks
            macro_idx, target_idx = self._sanitize_action_with_mask(macro_idx, target_idx, game_field, idx)
            actions.append((macro_idx, target_idx))
        
        # Fix 6.1: Resolve target conflicts (training does this, viewer must too!)
        actions_array = np.array([actions[0][0], actions[0][1], actions[1][0], actions[1][1]], dtype=np.int64)
        actions_array = self._resolve_target_conflicts(actions_array, game_field)
        actions = [(actions_array[0], actions_array[1]), (actions_array[2], actions_array[3])]
        
        self._cache_action = np.array([
            actions[0][0], actions[0][1], actions[1][0], actions[1][1],
        ], dtype=np.int64)
        self._cache_tick = tick

    def _resolve_target_cell(self, game_field: ViewerGameField, target_idx: int) -> Tuple[int, int]:
        fn = getattr(game_field, "get_macro_target", None)
        if callable(fn):
            try:
                t = fn(int(target_idx))
                if isinstance(t, (tuple, list)) and len(t) >= 2:
                    return (int(t[0]), int(t[1]))
            except Exception:
                pass
        mt = getattr(game_field, "macro_targets", None)
        if isinstance(mt, list) and len(mt) > 0:
            i = int(target_idx) % len(mt)
            try:
                t = mt[i]
                if isinstance(t, (tuple, list)) and len(t) >= 2:
                    return (int(t[0]), int(t[1]))
            except Exception:
                pass
        cols = int(getattr(game_field, "col_count", 20) or 20)
        rows = int(getattr(game_field, "row_count", 20) or 20)
        return (max(0, cols // 2), max(0, rows // 2))

    def act_for_agent(self, agent: Any, game_field: ViewerGameField, tick: int) -> Tuple[Any, Tuple[int, int]]:
        """
        CRITICAL: Maps agent to correct slot index based on sorted order (matches training's canonical ordering).
        """
        side = str(getattr(agent, "side", "blue")).lower()
        self._compute_joint_action_if_needed(game_field, tick=tick, side=side)
        
        # CRITICAL: Map agent to slot index based on sorted order (not agent_id directly)
        # Training: build_blue_identities() uses list index as slot index
        # Viewer: We sort by agent_id, so find agent's position in sorted list
        blue_agents = getattr(game_field, "blue_agents", []) or []
        live = [a for a in blue_agents if a is not None]
        def _get_agent_id(agent):
            try:
                return int(getattr(agent, "agent_id", -1))
            except Exception:
                return -1
        live.sort(key=_get_agent_id)  # Sort by agent_id ascending
        
        # Find this agent's index in sorted list (0 or 1)
        slot_index = 0
        for i, a in enumerate(live[:2]):
            if a is agent or (hasattr(a, "agent_id") and hasattr(agent, "agent_id") and getattr(a, "agent_id", -1) == getattr(agent, "agent_id", -2)):
                slot_index = i
                break
        
        # Ensure slot_index is 0 or 1
        slot_index = max(0, min(1, slot_index))
        
        macro_idx = int(self._cache_action[slot_index * 2 + 0]) % N_MACROS
        target_idx = int(self._cache_action[slot_index * 2 + 1])
        macro = USED_MACROS[macro_idx]
        if macro == MacroAction.PLACE_MINE:
            try:
                ax = int(getattr(agent, "x", 0))
                ay = int(getattr(agent, "y", 0))
                return macro, (ax, ay)
            except Exception:
                return macro, self._resolve_target_cell(game_field, target_idx)
        tgt_cell = self._resolve_target_cell(game_field, target_idx)
        return macro, tgt_cell
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
        # Deprecated: always use build_team_obs() per module ownership
        # self._viewer_use_obs_builder removed - always use canonical builder

        # Targets count (fallback)
        self.n_targets = int(getattr(env, "num_macro_targets", 8) or 8)

        # Cache joint action once per sim tick
        self._cache_tick: int = -1
        self._cache_action: np.ndarray = np.array([0, 0, 0, 0], dtype=np.int64)

        if self.model_path is None:
            print(f"[CTFViewer] PPO model not found: {model_path} (or .zip)")
            self.model_loaded = False
            return

        try:
            from stable_baselines3 import PPO as SB3PPO
            self.model = SB3PPO.load(self.model_path, device="cpu")
            self.model.policy.set_training_mode(False)
            self.model_loaded = True
            # Detect observation mode and warn if mismatch
            use_tokenized = self._model_expects_tokenized()
            obs_mode = "tokenized" if use_tokenized else "legacy"
            print(f"[CTFViewer] Loaded SB3 PPO model from: {self.model_path}")
            print(f"[CTFViewer] Model observation mode: {obs_mode} (tokenized={use_tokenized})")
        except OSError as e:
            print(f"[CTFViewer] Torch/SB3 DLL error (try: reinstall torch, or run viewer in Default mode): {e}")
            self.model_loaded = False
        except FileNotFoundError as e:
            print(f"[CTFViewer] PPO model file not found: {self.model_path}")
            print(f"[CTFViewer] Error: {e}")
            self.model_loaded = False
        except Exception as e:
            print(f"[CTFViewer] Failed to load SB3 PPO model '{self.model_path}': {e}")
            import traceback
            print(f"[CTFViewer] Traceback: {traceback.format_exc()}")
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

    def _model_expects_tokenized(self) -> bool:
        """Detect if model expects tokenized observations (max_blue_agents > 2 or use_tokenized_obs=True)."""
        if self.model is None:
            return False
        space = getattr(self.model.policy, "observation_space", None)
        if not hasattr(space, "spaces") or not isinstance(space.spaces, dict):
            return False
        grid_space = space.spaces.get("grid", None)
        if grid_space is None or not hasattr(grid_space, "shape"):
            return False
        grid_shape = grid_space.shape
        # Tokenized: (M, C, H, W) where M > 1
        # Legacy: (2*C, H, W) - first dim is 2*C (even number, typically 14, 16, etc.)
        if len(grid_shape) == 4:
            # 4D shape = tokenized mode
            return True
        elif len(grid_shape) == 3:
            # 3D shape = legacy mode (2*C, H, W)
            return False
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
        """
        Build team observation using canonical build_team_obs().
        
        Per module ownership: observation building must use rl/obs_builder.py::build_team_obs().
        Manual fallback removed - if builder is unavailable, raise error.
        
        CRITICAL: Agent ordering must match training (blue_0, blue_1 by agent_id).
        Training uses build_blue_identities() which orders by agent_id ascending.
        """
        agents = game_field.blue_agents if side == "blue" else game_field.red_agents
        live = [a for a in agents if a is not None]
        
        # CRITICAL: Sort by agent_id to match training's canonical ordering (blue_0, blue_1)
        # Training uses build_blue_identities() which creates identities ordered by agent_id
        def _get_agent_id(agent):
            try:
                return int(getattr(agent, "agent_id", -1))
            except Exception:
                return -1
        live.sort(key=_get_agent_id)  # Sort by agent_id ascending (0, 1, 2, ...)
        
        while len(live) < 2:
            live.append(live[0] if live else None)

        if _viewer_obs_builder is None:
            raise RuntimeError(
                "build_team_obs() not available. Ensure rl/obs_builder.py is importable. "
                "Manual observation building is forbidden per module ownership."
            )

        # CRITICAL: Detect if model expects tokenized observations (matches training config)
        # Training uses tokenized when: use_tokenized_obs=True OR max_blue_agents > 2
        use_tokenized = self._model_expects_tokenized()
        
        # Determine max_agents from observation space if tokenized, otherwise use 2
        max_agents_for_obs = 2
        if use_tokenized and self.model is not None:
            try:
                space = getattr(self.model.policy, "observation_space", None)
                if hasattr(space, "spaces") and isinstance(space.spaces, dict):
                    grid_space = space.spaces.get("grid", None)
                    if grid_space is not None and hasattr(grid_space, "shape"):
                        grid_shape = grid_space.shape
                        if len(grid_shape) == 4:
                            max_agents_for_obs = int(grid_shape[0])  # First dim is M (max_agents)
            except Exception:
                pass  # Fallback to 2

        return _viewer_obs_builder(
            game_field,
            live[:2],
            max_agents=max_agents_for_obs,
            include_mask=self._model_expects_mask(),
            tokenized=use_tokenized,
            vec_size_base=12,
            n_macros=N_MACROS,
            n_targets=int(getattr(game_field, "num_macro_targets", 8) or 8),
        )

    def _macro_uses_target(self, macro: int, game_field: ViewerGameField) -> bool:
        """Check if macro action uses a target parameter."""
        from macro_actions import MacroAction
        return macro in (MacroAction.GO_TO, MacroAction.GET_FLAG, MacroAction.GRAB_MINE, MacroAction.PLACE_MINE)

    def _resolve_target_conflicts(self, actions: np.ndarray, game_field: ViewerGameField) -> np.ndarray:
        """
        Fix 6.1: Resolve target conflicts - if both agents pick same target, nearest keeps it,
        other gets next-best valid target (or GO_HOME).
        This matches training's EnvActionManager._resolve_target_conflicts().
        """
        a = actions.copy()
        blue_agents = getattr(game_field, "blue_agents", []) or []
        if len(blue_agents) < 2:
            return a
        
        # Group actions by (macro, target) for target-using macros
        from collections import defaultdict
        groups: Dict[Tuple[int, int], List[int]] = defaultdict(list)  # (macro, target) -> [agent_idx, ...]
        for i in range(2):  # 2 agents
            m, t = int(a[i * 2]), int(a[i * 2 + 1])
            if self._macro_uses_target(m, game_field):
                groups[(m, t)].append(i)
        
        # Resolve conflicts
        for (m, t), agents in groups.items():
            if len(agents) <= 1:
                continue
            # Sort by distance to target (nearest first)
            try:
                tgt_cell = game_field.get_macro_target(t)
                if tgt_cell is None or len(tgt_cell) < 2:
                    tx, ty = 0, 0
                else:
                    tx, ty = int(tgt_cell[0]), int(tgt_cell[1])
            except Exception:
                continue
            
            def dist_to_target(agent_idx: int) -> float:
                if agent_idx >= len(blue_agents):
                    return float("inf")
                agent = blue_agents[agent_idx]
                if agent is None:
                    return float("inf")
                pos = getattr(agent, "float_pos", (getattr(agent, "x", 0), getattr(agent, "y", 0)))
                dx = float(pos[0]) - tx
                dy = float(pos[1]) - ty
                return dx * dx + dy * dy
            
            agents_sorted = sorted(agents, key=dist_to_target)
            # Keep first; reassign rest to next-best or GO_HOME (4)
            for idx, agent_idx in enumerate(agents_sorted):
                if idx == 0:
                    continue
                if agent_idx >= len(blue_agents):
                    a[agent_idx * 2] = 4  # GO_HOME
                    a[agent_idx * 2 + 1] = 0
                    continue
                agent = blue_agents[agent_idx]
                if agent is None:
                    a[agent_idx * 2] = 4  # GO_HOME
                    a[agent_idx * 2 + 1] = 0
                    continue
                try:
                    tm = np.asarray(game_field.get_target_mask(agent), dtype=np.bool_).reshape(-1)
                    n_targets = int(getattr(game_field, "num_macro_targets", self.n_targets) or self.n_targets)
                    if tm.shape != (n_targets,) or not tm.any():
                        a[agent_idx * 2] = 4  # GO_HOME
                        a[agent_idx * 2 + 1] = 0
                        continue
                    # Exclude current contested target; pick next-best valid
                    best_alt = 0
                    best_dist = float("inf")
                    for ti in range(n_targets):
                        if not bool(tm[ti]) or ti == t:
                            continue
                        try:
                            tc = game_field.get_macro_target(ti)
                            tcx = int(tc[0])
                            tcy = int(tc[1]) if len(tc) >= 2 else 0
                        except Exception:
                            continue
                        pos = getattr(agent, "float_pos", (getattr(agent, "x", 0), getattr(agent, "y", 0)))
                        d = (float(pos[0]) - tcx) ** 2 + (float(pos[1]) - tcy) ** 2
                        if d < best_dist:
                            best_dist = d
                            best_alt = ti
                    a[agent_idx * 2] = m  # Keep same macro
                    a[agent_idx * 2 + 1] = best_alt  # Reassign target
                except Exception:
                    a[agent_idx * 2] = 4  # GO_HOME fallback
                    a[agent_idx * 2 + 1] = 0
        
        return a

    def _sanitize_action_with_mask(
        self, 
        action: np.ndarray, 
        game_field: ViewerGameField, 
        agent_idx: int
    ) -> Tuple[int, int]:
        """Enforce action masks: ensure macro and target are valid for this agent."""
        macro = int(action[0]) % N_MACROS
        target = int(action[1]) % max(1, int(getattr(game_field, "num_macro_targets", self.n_targets) or self.n_targets))
        
        blue_agents = getattr(game_field, "blue_agents", []) or []
        if agent_idx >= len(blue_agents):
            return macro, target
        
        agent = blue_agents[agent_idx]
        if agent is None:
            return macro, target
        
        # Enforce macro mask
        try:
            mm = np.asarray(game_field.get_macro_mask(agent), dtype=np.bool_).reshape(-1)
            if mm.shape == (N_MACROS,) and mm.any():
                if not bool(mm[macro]):
                    # Fallback to first valid macro (usually GO_TO=0)
                    valid_macros = np.flatnonzero(mm)
                    macro = int(valid_macros[0]) if valid_macros.size > 0 else 0
        except Exception:
            pass
        
        # Enforce target mask if macro uses target
        try:
            from macro_actions import MacroAction
            uses_target = macro in (MacroAction.GO_TO, MacroAction.GET_FLAG, MacroAction.GRAB_MINE, MacroAction.PLACE_MINE)
            if uses_target:
                tm = np.asarray(game_field.get_target_mask(agent), dtype=np.bool_).reshape(-1)
                n_targets = int(getattr(game_field, "num_macro_targets", self.n_targets) or self.n_targets)
                if tm.shape == (n_targets,) and tm.any():
                    if not bool(tm[target]):
                        # Fallback to nearest valid target
                        valid_targets = np.flatnonzero(tm)
                        if valid_targets.size > 0:
                            # Simple fallback: use first valid target
                            target = int(valid_targets[0])
                        else:
                            target = 0
        except Exception:
            pass
        
        return macro, target

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

        # Normalize ranges (before mask enforcement)
        a[0] = int(a[0]) % N_MACROS
        a[2] = int(a[2]) % N_MACROS
        nt = max(1, int(getattr(game_field, "num_macro_targets", self.n_targets) or self.n_targets))
        a[1] = int(a[1]) % nt
        a[3] = int(a[3]) % nt

        # Enforce masks: sanitize actions for each agent
        macro0, tgt0 = self._sanitize_action_with_mask(np.array([a[0], a[1]]), game_field, 0)
        macro1, tgt1 = self._sanitize_action_with_mask(np.array([a[2], a[3]]), game_field, 1)
        a[0], a[1] = macro0, tgt0
        a[2], a[3] = macro1, tgt1

        # Fix 6.1: Resolve target conflicts (training does this, viewer must too!)
        a = self._resolve_target_conflicts(a, game_field)

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
        
        CRITICAL: Maps agent to correct slot index based on sorted order (matches training's canonical ordering).
        Training uses build_blue_identities() which creates blue_0, blue_1 based on list order.
        We sort by agent_id, so agent at index 0 → slot 0, agent at index 1 → slot 1.
        """
        side = str(getattr(agent, "side", "blue")).lower()
        self._compute_joint_action_if_needed(game_field, tick=tick, side=side)

        # CRITICAL: Map agent to slot index based on sorted order (not agent_id directly)
        # Training: build_blue_identities() uses list index as slot index
        # Viewer: We sort by agent_id, so find agent's position in sorted list
        blue_agents = getattr(game_field, "blue_agents", []) or []
        live = [a for a in blue_agents if a is not None]
        def _get_agent_id(agent):
            try:
                return int(getattr(agent, "agent_id", -1))
            except Exception:
                return -1
        live.sort(key=_get_agent_id)  # Sort by agent_id ascending
        
        # Find this agent's index in sorted list (0 or 1)
        slot_index = 0
        for i, a in enumerate(live[:2]):
            if a is agent or (hasattr(a, "agent_id") and hasattr(agent, "agent_id") and getattr(a, "agent_id", -1) == getattr(agent, "agent_id", -2)):
                slot_index = i
                break
        
        # Ensure slot_index is 0 or 1
        slot_index = max(0, min(1, slot_index))

        macro_idx = int(self._cache_action[slot_index * 2 + 0]) % N_MACROS
        target_idx = int(self._cache_action[slot_index * 2 + 1])

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
        # Deprecated: always use build_team_obs() per module ownership
        # self._viewer_use_obs_builder removed - always use canonical builder

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
        """
        Build team observation with high-level mode using canonical build_team_obs().
        
        Per module ownership: observation building must use rl/obs_builder.py::build_team_obs().
        Manual fallback removed - if builder is unavailable, raise error.
        """
        agents = game_field.blue_agents if side == "blue" else game_field.red_agents
        live = [a for a in agents if a is not None]
        # CRITICAL: Sort by agent_id to match training's canonical ordering (blue_0, blue_1)
        def _get_agent_id(agent):
            try:
                return int(getattr(agent, "agent_id", -1))
            except Exception:
                return -1
        live.sort(key=_get_agent_id)  # Sort by agent_id ascending (0, 1, 2, ...)
        while len(live) < 2:
            live.append(live[0] if live else None)

        if _viewer_obs_builder is None:
            raise RuntimeError(
                "build_team_obs() not available. Ensure rl/obs_builder.py is importable. "
                "Manual observation building is forbidden per module ownership."
            )

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

    def _macro_uses_target(self, macro: int, game_field: ViewerGameField) -> bool:
        """Check if macro action uses a target parameter."""
        from macro_actions import MacroAction
        return macro in (MacroAction.GO_TO, MacroAction.GET_FLAG, MacroAction.GRAB_MINE, MacroAction.PLACE_MINE)

    def _resolve_target_conflicts(self, actions: np.ndarray, game_field: ViewerGameField) -> np.ndarray:
        """
        Fix 6.1: Resolve target conflicts - if both agents pick same target, nearest keeps it,
        other gets next-best valid target (or GO_HOME).
        """
        a = actions.copy()
        blue_agents = getattr(game_field, "blue_agents", []) or []
        if len(blue_agents) < 2:
            return a
        
        from collections import defaultdict
        groups: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        for i in range(2):
            m, t = int(a[i * 2]), int(a[i * 2 + 1])
            if self._macro_uses_target(m, game_field):
                groups[(m, t)].append(i)
        
        for (m, t), agents in groups.items():
            if len(agents) <= 1:
                continue
            try:
                tgt_cell = game_field.get_macro_target(t)
                if tgt_cell is None or len(tgt_cell) < 2:
                    tx, ty = 0, 0
                else:
                    tx, ty = int(tgt_cell[0]), int(tgt_cell[1])
            except Exception:
                continue
            
            def dist_to_target(agent_idx: int) -> float:
                if agent_idx >= len(blue_agents):
                    return float("inf")
                agent = blue_agents[agent_idx]
                if agent is None:
                    return float("inf")
                pos = getattr(agent, "float_pos", (getattr(agent, "x", 0), getattr(agent, "y", 0)))
                dx = float(pos[0]) - tx
                dy = float(pos[1]) - ty
                return dx * dx + dy * dy
            
            agents_sorted = sorted(agents, key=dist_to_target)
            for idx, agent_idx in enumerate(agents_sorted):
                if idx == 0:
                    continue
                if agent_idx >= len(blue_agents):
                    a[agent_idx * 2] = 4  # GO_HOME
                    a[agent_idx * 2 + 1] = 0
                    continue
                agent = blue_agents[agent_idx]
                if agent is None:
                    a[agent_idx * 2] = 4
                    a[agent_idx * 2 + 1] = 0
                    continue
                try:
                    tm = np.asarray(game_field.get_target_mask(agent), dtype=np.bool_).reshape(-1)
                    n_targets = int(getattr(game_field, "num_macro_targets", 8) or 8)
                    if tm.shape != (n_targets,) or not tm.any():
                        a[agent_idx * 2] = 4
                        a[agent_idx * 2 + 1] = 0
                        continue
                    best_alt = 0
                    best_dist = float("inf")
                    for ti in range(n_targets):
                        if not bool(tm[ti]) or ti == t:
                            continue
                        try:
                            tc = game_field.get_macro_target(ti)
                            tcx = int(tc[0])
                            tcy = int(tc[1]) if len(tc) >= 2 else 0
                        except Exception:
                            continue
                        pos = getattr(agent, "float_pos", (getattr(agent, "x", 0), getattr(agent, "y", 0)))
                        d = (float(pos[0]) - tcx) ** 2 + (float(pos[1]) - tcy) ** 2
                        if d < best_dist:
                            best_dist = d
                            best_alt = ti
                    a[agent_idx * 2] = m
                    a[agent_idx * 2 + 1] = best_alt
                except Exception:
                    a[agent_idx * 2] = 4
                    a[agent_idx * 2 + 1] = 0
        
        return a

    def _sanitize_action_with_mask(
        self, 
        macro: int, 
        target: int, 
        game_field: ViewerGameField, 
        agent_idx: int
    ) -> Tuple[int, int]:
        """Enforce action masks: ensure macro and target are valid for this agent."""
        macro = int(macro) % N_MACROS
        n_targets = int(getattr(game_field, "num_macro_targets", 8) or 8)
        target = int(target) % max(1, n_targets)
        
        blue_agents = getattr(game_field, "blue_agents", []) or []
        if agent_idx >= len(blue_agents):
            return macro, target
        
        agent = blue_agents[agent_idx]
        if agent is None:
            return macro, target
        
        # Enforce macro mask
        try:
            mm = np.asarray(game_field.get_macro_mask(agent), dtype=np.bool_).reshape(-1)
            if mm.shape == (N_MACROS,) and mm.any():
                if not bool(mm[macro]):
                    valid_macros = np.flatnonzero(mm)
                    macro = int(valid_macros[0]) if valid_macros.size > 0 else 0
        except Exception:
            pass
        
        # Enforce target mask if macro uses target
        try:
            from macro_actions import MacroAction
            uses_target = macro in (MacroAction.GO_TO, MacroAction.GET_FLAG, MacroAction.GRAB_MINE, MacroAction.PLACE_MINE)
            if uses_target:
                tm = np.asarray(game_field.get_target_mask(agent), dtype=np.bool_).reshape(-1)
                if tm.shape == (n_targets,) and tm.any():
                    if not bool(tm[target]):
                        valid_targets = np.flatnonzero(tm)
                        target = int(valid_targets[0]) if valid_targets.size > 0 else 0
        except Exception:
            pass
        
        return macro, target

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
        a = np.asarray(action).reshape(-1)
        
        # Ensure shape [4]
        if a.size < 4:
            padded = np.zeros((4,), dtype=np.int64)
            padded[:a.size] = a[:a.size]
            a = padded
        elif a.size > 4:
            a = a[:4]
        
        # Normalize ranges (before mask enforcement)
        a[0] = int(a[0]) % N_MACROS
        a[2] = int(a[2]) % N_MACROS
        n_targets = int(getattr(game_field, "num_macro_targets", 8) or 8)
        a[1] = int(a[1]) % n_targets
        a[3] = int(a[3]) % n_targets
        
        # Enforce masks: sanitize actions for each agent
        macro0, tgt0 = self._sanitize_action_with_mask(a[0], a[1], game_field, 0)
        macro1, tgt1 = self._sanitize_action_with_mask(a[2], a[3], game_field, 1)
        a[0], a[1] = macro0, tgt0
        a[2], a[3] = macro1, tgt1
        
        # Fix 6.1: Resolve target conflicts (training does this, viewer must too!)
        a = self._resolve_target_conflicts(a, game_field)
        
        self._cache_action = a

    def act_for_agent(self, agent, game_field: ViewerGameField, *, tick: int) -> Tuple[MacroAction, Tuple[int, int]]:
        """
        CRITICAL: Maps agent to correct slot index based on sorted order (matches training's canonical ordering).
        """
        self._compute_joint_action_if_needed(game_field, tick, side="blue")
        
        # CRITICAL: Map agent to slot index based on sorted order (not list position directly)
        # Training: build_blue_identities() uses list index as slot index, but list is sorted by agent_id
        # Viewer: We sort by agent_id, so find agent's position in sorted list
        blue_agents = getattr(game_field, "blue_agents", []) or []
        live = [a for a in blue_agents if a is not None]
        def _get_agent_id(agent):
            try:
                return int(getattr(agent, "agent_id", -1))
            except Exception:
                return -1
        live.sort(key=_get_agent_id)  # Sort by agent_id ascending
        
        # Find this agent's index in sorted list (0 or 1)
        slot_index = 0
        for i, a in enumerate(live[:2]):
            if a is agent or (hasattr(a, "agent_id") and hasattr(agent, "agent_id") and getattr(a, "agent_id", -1) == getattr(agent, "agent_id", -2)):
                slot_index = i
                break
        
        # Ensure slot_index is 0 or 1, then map to action array index (0 or 2)
        slot_index = max(0, min(1, slot_index))
        idx = slot_index * 2  # 0 or 2 (for action array indexing)

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
        mappo_model_path: str = DEFAULT_MAPPO_MODEL_PATH,
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
        self.blue_mappo_team = MAPPOTeamPolicy(
            mappo_model_path, self.game_field, deterministic=True
        )

        # Blue policy callable (installed into game_field.policies["blue"])
        self._blue_policy_callable = None

        # Mode
        self.blue_mode: str = "DEFAULT"
        self._apply_blue_mode(self.blue_mode)
        self._reset_op3_policies()
        
        # Set opponent to OP3 for testing (OP3_HARD is used during training to make agent better at OP3)
        # Testing should use standard OP3 to measure actual performance
        if hasattr(self.game_field, "set_red_opponent"):
            try:
                self.game_field.set_red_opponent("OP3")
                print("[CTFViewer] Red opponent set to OP3 (standard test opponent)")
            except Exception:
                pass

    def _set_phase_op3(self) -> None:
        """Set phase to OP3 and apply OP3 stress settings (physics enabled, etc.) to match training."""
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
        
        # CRITICAL: Apply OP3 stress settings to match training environment
        # Training uses STRESS_BY_PHASE["OP3"] which has physics_enabled=True
        # This ensures viewer matches training so agents perform well in viewer
        try:
            # Apply OP3 physics and stress settings directly (ViewerGameField doesn't use stress schedule system)
            if hasattr(self.game_field, "set_physics_enabled"):
                self.game_field.set_physics_enabled(True)  # OP3: physics enabled
            
            # Apply OP3 stress settings: current, drift, action delay, sensor noise
            if hasattr(self.game_field, "set_disturbance_config"):
                self.game_field.set_disturbance_config(
                    current_strength_cps=0.12,      # Strong ocean current (OP3)
                    drift_sigma_cells=0.03,         # Random drift (OP3)
                )
            if hasattr(self.game_field, "set_robotics_constraints"):
                self.game_field.set_robotics_constraints(
                    action_delay_steps=2,           # 2-step action delay (OP3)
                    actuation_noise_sigma=0.0,      # No actuation noise
                )
            if hasattr(self.game_field, "set_sensor_config"):
                # Get current sensor range or use default (9999.0 = unlimited range)
                sensor_range = getattr(getattr(self.game_field, "boat_cfg", None), "sensor_range_cells", 9999.0) or 9999.0
                self.game_field.set_sensor_config(
                    sensor_range_cells=float(sensor_range),  # Required positional arg: keep current range
                    sensor_noise_sigma_cells=0.2,            # High sensor noise (OP3)
                    sensor_dropout_prob=0.05,                # 5% sensor dropout (OP3)
                )
            if hasattr(self.game_field, "set_dynamics_config"):
                # OP3: full realism (not relaxed)
                self.game_field.set_dynamics_config(
                    max_speed_cps=2.2,              # Standard speed
                    max_accel_cps2=2.0,             # Standard acceleration
                    max_yaw_rate_rps=4.0,           # Standard yaw rate
                )
            # Only print once on first call (avoid spam on resets)
            if not hasattr(self, "_op3_stress_applied"):
                print("[CTFViewer] Applied OP3 stress settings (physics enabled) to match training environment")
                self._op3_stress_applied = True
        except Exception as e:
            print(f"[CTFViewer] Warning: Could not apply OP3 stress settings: {e}")
    
    def _set_phase(self, phase: str) -> None:
        """Set phase (OP1, OP2, OP3) for both game_manager and game_field."""
        phase_upper = str(phase).upper()
        if hasattr(self.game_manager, "set_phase"):
            try:
                self.game_manager.set_phase(phase_upper)
            except Exception:
                pass
        if hasattr(self.game_field, "set_phase"):
            try:
                self.game_field.set_phase(phase_upper)
            except Exception:
                pass

    def _available_modes(self) -> List[str]:
        """Cycle order: Default -> PPO -> MAPPO -> HPPO."""
        modes = ["DEFAULT"]
        if self.blue_ppo_team and self.blue_ppo_team.model_loaded:
            modes.append("PPO")
        if self.blue_mappo_team and self.blue_mappo_team.model_loaded:
            modes.append("MAPPO")
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

        elif mode == "MAPPO" and self.blue_mappo_team and self.blue_mappo_team.model_loaded:
            def blue_policy(obs, agent, game_field):
                return self.blue_mappo_team.act_for_agent(agent, game_field, tick=self.sim_tick)

            self._blue_policy_callable = blue_policy
            self.game_field.policies["blue"] = self._blue_policy_callable
            self.blue_mode = "MAPPO"
            self.blue_mappo_team.reset_cache()
            print("[CTFViewer] Blue -> MAPPO (.pth)")

        elif mode == "HPPO" and self.blue_hppo_team and self.blue_hppo_team.model_loaded:
            def blue_policy(obs, agent, game_field):
                return self.blue_hppo_team.act_for_agent(agent, game_field, tick=self.sim_tick)

            self._blue_policy_callable = blue_policy
            self.game_field.policies["blue"] = self._blue_policy_callable
            self.blue_mode = "HPPO"
            self.blue_hppo_team.reset_cache()
            print("[CTFViewer] Blue -> HPPO (high/low)")

        else:
            self._blue_policy_callable = None
            self.game_field.policies["blue"] = self.blue_op3_baseline
            self.blue_mode = "DEFAULT"
            print("[CTFViewer] Blue -> Default baseline")

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

    def set_red_learned_model(self, model_path: str) -> None:
        """
        Set red to a learned PPO model (for cross-play evaluation).
        Uses the same snapshot wrapper as training; requires red_opponents.make_snapshot_wrapper.
        """
        if make_snapshot_wrapper is None:
            raise RuntimeError("red_opponents.make_snapshot_wrapper not available")
        path = str(model_path).strip()
        if not path or not os.path.isfile(path):
            if os.path.isfile(path + ".zip"):
                path = path + ".zip"
            else:
                raise FileNotFoundError(f"Red model not found: {model_path!r}")
        if hasattr(self.game_field, "set_red_opponent"):
            self.game_field.set_red_opponent("OP3")
        if hasattr(self.game_field, "set_red_policy_wrapper"):
            self.game_field.set_red_policy_wrapper(make_snapshot_wrapper(path))

    def set_red_scripted(self, tag: str = "OP3") -> None:
        """Set red back to scripted opponent (clears learned red wrapper)."""
        if hasattr(self.game_field, "set_red_policy_wrapper"):
            self.game_field.set_red_policy_wrapper(None)
        if hasattr(self.game_field, "set_red_opponent"):
            self.game_field.set_red_opponent(str(tag).upper())

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
        opponent: str = "OP3",  # Default to OP3 for testing (OP3_HARD is training-only to improve OP3 performance)
        eval_model: Optional[str] = None,
        red_model_path: Optional[str] = None,
        episode_seeds: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate trained model performance by running N episodes and collecting IROS-style metrics.

        Args:
            num_episodes: Number of episodes to run
            save_csv: Path to save CSV (e.g., "eval_results.csv"). If None, auto-generates name.
            headless: If True, run without display (faster). If False, show viewer.
            opponent: Red opponent tag ("OP1", "OP2", "OP3", "OP3_EASY", "OP3_HARD", "NAVAL_DEFENDER", "NAVAL_RUSHER", "NAVAL_BALANCED"); ignored if red_model_path is set.
            eval_model: Force model for eval: "ppo", "mappo", "hppo". If None, auto-pick first loaded.
            red_model_path: If set, red uses this learned PPO (.zip) instead of scripted opponent (cross-play).
            episode_seeds: If set, use these seeds (one per episode) for reproducible varied initial states; len should be >= num_episodes.

        Returns:
            Dict with summary statistics and per-episode metrics
        """
        if not headless:
            print(f"[Eval] Running {num_episodes} episodes with display. Press ESC to stop early.")
        else:
            print(f"[Eval] Running {num_episodes} episodes headless...")

        # Use requested model or first loaded
        want = (eval_model or "").strip().upper()
        if want == "PPO" and self.blue_ppo_team and self.blue_ppo_team.model_loaded:
            self._apply_blue_mode("PPO")
            print("[Eval] Using PPO for evaluation")
        elif want == "MAPPO" and self.blue_mappo_team and self.blue_mappo_team.model_loaded:
            self._apply_blue_mode("MAPPO")
            print("[Eval] Using MAPPO for evaluation")
        elif want == "HPPO" and self.blue_hppo_team and self.blue_hppo_team.model_loaded:
            self._apply_blue_mode("HPPO")
            print("[Eval] Using HPPO for evaluation")
        elif self.blue_mode == "DEFAULT":
            if self.blue_ppo_team and self.blue_ppo_team.model_loaded:
                self._apply_blue_mode("PPO")
                print("[Eval] Switched to PPO mode for evaluation")
            elif self.blue_mappo_team and self.blue_mappo_team.model_loaded:
                self._apply_blue_mode("MAPPO")
                print("[Eval] Switched to MAPPO mode for evaluation")
            elif self.blue_hppo_team and self.blue_hppo_team.model_loaded:
                self._apply_blue_mode("HPPO")
                print("[Eval] Switched to HPPO mode for evaluation")
            else:
                print("[WARN] No trained model loaded! Using DEFAULT baseline.")
        if eval_model and self.blue_mode == "DEFAULT":
            print(f"[WARN] Requested model '{eval_model}' not loaded; using DEFAULT baseline.")

        # Set red: learned model (cross-play) or scripted
        if red_model_path:
            try:
                self.set_red_learned_model(red_model_path)
            except Exception as e:
                print(f"[Eval] Failed to set red to learned model {red_model_path!r}: {e}")
                self.set_red_scripted("OP3")
        else:
            opponent_upper = str(opponent).upper()
            if hasattr(self.game_field, "set_red_opponent"):
                try:
                    self.game_field.set_red_opponent(opponent_upper)
                except Exception:
                    pass
            if hasattr(self.game_field, "set_red_policy_wrapper"):
                self.game_field.set_red_policy_wrapper(None)
        # Set phase: naval opponents use OP3 (physics on); OP1/OP2 use own phase
        opponent_upper = str(opponent).upper() if not red_model_path else "OP3"
        if opponent_upper in ("OP1", "OP2"):
            self._set_phase(opponent_upper)
        elif opponent_upper in ("OP3", "OP3_EASY", "OP3_HARD"):
            phase = opponent_upper.replace("_EASY", "").replace("_HARD", "")
            self._set_phase(phase)
        elif opponent_upper in ("NAVAL_DEFENDER", "NAVAL_RUSHER", "NAVAL_BALANCED"):
            self._set_phase_op3()
        else:
            self._set_phase_op3()

        fixed_dt = 1.0 / 60.0
        # Match training: max_decision_steps (default 400 in CTFGameFieldSB3Env)
        # Training: 400 decision steps * 0.99 * decision_interval (~0.693s) = ~277 seconds
        # Viewer: sim ticks at 60 Hz, so 277s * 60 = ~16,620 sim ticks
        # CRITICAL: Training uses 0.99 * interval, so we must match that exactly
        # But also check game_over for natural termination
        decision_interval = float(getattr(self.game_field, "decision_interval_seconds", 0.7))
        # Match training exactly: training uses 0.99 * interval per step
        max_time_seconds = 400 * 0.99 * decision_interval  # = 400 * 0.693 = 277.2 seconds
        max_steps_per_episode = int(max_time_seconds * 60)  # Convert to sim ticks at 60 Hz = ~16,632 ticks
        # Ensure minimum reasonable limit (at least 60 seconds = 3600 ticks)
        max_steps_per_episode = max(3600, max_steps_per_episode)
        if not headless:
            print(f"[Eval] Episode step limit: {max_steps_per_episode} sim ticks (~{max_steps_per_episode/60:.1f}s, matches training {400} decision steps)")

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
                gf = self.game_field
                if episode_seeds is not None and ep_idx < len(episode_seeds) and hasattr(gf, "set_seed"):
                    gf.set_seed(int(episode_seeds[ep_idx]))
                self.game_field.reset_default()
                self._set_phase_op3()
                self.sim_tick = 0
                if self.blue_mappo_team:
                    self.blue_mappo_team.reset_cache()
                if self.blue_ppo_team:
                    self.blue_ppo_team.reset_cache()
                if self.blue_hppo_team:
                    self.blue_hppo_team.reset_cache()
                    self._reset_op3_policies()

                step_count = 0
                running_episode = True
                gf = self.game_field
                blue_agents_list = getattr(gf, "blue_agents", []) or []
                n_blue = len(blue_agents_list)
                attack_steps = [0] * max(1, n_blue)
                defend_steps = [0] * max(1, n_blue)

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
                    
                    # Check for natural game termination (score limit or timeout)
                    gm = self.game_manager
                    if getattr(gm, "game_over", False):
                        break

                    # Per-step: role timers (attacking vs defending)
                    if n_blue > 0 and hasattr(gf, "_zone_ranges") and hasattr(gf, "_agent_cell_pos") and hasattr(gf, "_agent_is_carrying_flag"):
                        try:
                            (blue_min_col, blue_max_col), (red_min_col, red_max_col) = gf._zone_ranges()
                            for i, agent in enumerate(blue_agents_list):
                                if i >= len(attack_steps):
                                    break
                                _en = getattr(agent, "isEnabled", True)
                                if callable(_en):
                                    _en = _en()
                                if not _en:
                                    continue
                                cell = gf._agent_cell_pos(agent)
                                col = cell[0] if isinstance(cell, (tuple, list)) and len(cell) >= 1 else 0
                                carrying = gf._agent_is_carrying_flag(agent)
                                in_opponent_half = red_min_col <= col <= red_max_col
                                attacking = in_opponent_half or carrying
                                if attacking:
                                    attack_steps[i] += 1
                                else:
                                    defend_steps[i] += 1
                        except Exception:
                            pass

                    # Check for natural game termination (score limit or timeout)
                    gm = self.game_manager
                    if getattr(gm, "game_over", False):
                        break

                    if not headless:
                        self.draw(alpha=1.0)
                        pg.display.flip()
                        self.clock.tick(60)

                # Collect metrics from game_manager (once per episode)
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

                collisions_per_tick = int(getattr(gm, "collision_count_this_episode", 0))
                collision_events = int(getattr(gm, "collision_events_this_episode", 0))  # canonical: no fallback
                near_misses = int(getattr(gm, "near_miss_count_this_episode", 0))
                collision_free = 1 if collision_events == 0 else 0  # use collision_events as canonical

                dists = getattr(gm, "blue_inter_robot_distances", []) or []
                mean_inter_robot_dist = float(np.mean(dists)) if dists else None
                std_inter_robot_dist = float(np.std(dists)) if len(dists) > 1 else (0.0 if dists else None)

                visited = getattr(gm, "blue_zone_visited_cells", set()) or set()
                total_zone = int(getattr(gm, "total_blue_zone_cells", 1)) or 1
                zone_coverage = float(len(visited)) / float(total_zone) if total_zone > 0 else 0.0

                # Per-agent role % and flag captures (use agent's active steps so attack+defend = 100%)
                total_steps_ep = max(1, step_count)
                pct_attacking_list = []
                pct_defending_list = []
                for i in range(len(attack_steps)):
                    active = attack_steps[i] + defend_steps[i]
                    if active > 0:
                        pct_attacking_list.append(100.0 * attack_steps[i] / active)
                        pct_defending_list.append(100.0 * defend_steps[i] / active)
                    else:
                        pct_attacking_list.append(0.0)
                        pct_defending_list.append(0.0)
                mean_pct_attacking = float(np.mean(pct_attacking_list)) if pct_attacking_list else 0.0
                mean_pct_defending = float(np.mean(pct_defending_list)) if pct_defending_list else 0.0
                blue_captures = list(getattr(gm, "blue_captures_this_episode", []) or [])
                if len(blue_captures) < n_blue:
                    blue_captures = blue_captures + [0] * (n_blue - len(blue_captures))
                flag_capture_variance = float(np.var(blue_captures)) if len(blue_captures) > 1 else 0.0
                mean_flag_captures = float(np.mean(blue_captures)) if blue_captures else 0.0

                # Reward per timestep (efficiency)
                blue_episode_reward = float(getattr(gm, "blue_episode_reward", 0.0) or 0.0)
                reward_per_timestep = blue_episode_reward / total_steps_ep if total_steps_ep > 0 else 0.0

                # Normalized collision / coverage (interpretable across episode lengths)
                collisions_per_100_steps = 100.0 * (collisions_per_tick / total_steps_ep) if total_steps_ep > 0 else 0.0
                near_misses_per_100_steps = 100.0 * (near_misses / total_steps_ep) if total_steps_ep > 0 else 0.0
                coverage_efficiency = zone_coverage / total_steps_ep if total_steps_ep > 0 else 0.0

                # Sanity: % attacking + % defending = 100% per agent when agent had active steps
                for i in range(len(pct_attacking_list)):
                    if attack_steps[i] + defend_steps[i] > 0:
                        _sum = pct_attacking_list[i] + pct_defending_list[i]
                        assert abs(_sum - 100.0) < 1e-6, f"attack+defend should sum to 100%, got {_sum}"

                row = {
                    "episode_id": episode_id,
                    "success": success,
                    "time_to_first_score": time_to_first_score,
                    "time_to_game_over": time_to_game_over_sec,
                    "collisions_per_episode": collisions_per_tick,
                    "collision_events_per_episode": collision_events,
                    "near_misses_per_episode": near_misses,
                    "collision_free_episode": collision_free,
                    "mean_inter_robot_dist": mean_inter_robot_dist,
                    "std_inter_robot_dist": std_inter_robot_dist,
                    "zone_coverage": zone_coverage,
                    "phase_name": str(getattr(gm, "phase_name", "OP3")),
                    "opponent_kind": "LEARNED" if red_model_path else "SCRIPTED",
                    "scripted_tag": str(opponent).upper() if not red_model_path else "",
                    "blue_score": blue_score,
                    "red_score": red_score,
                    "mean_pct_attacking": mean_pct_attacking,
                    "mean_pct_defending": mean_pct_defending,
                    "flag_capture_variance": flag_capture_variance,
                    "mean_flag_captures": mean_flag_captures,
                    "blue_episode_reward": blue_episode_reward,
                    "reward_per_timestep": reward_per_timestep,
                    "collisions_per_100_steps": collisions_per_100_steps,
                    "near_misses_per_100_steps": near_misses_per_100_steps,
                    "coverage_efficiency": coverage_efficiency,
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
        collision_events_list = [e.get("collision_events_per_episode", e["collisions_per_episode"]) for e in episodes]
        near_misses_list = [e["near_misses_per_episode"] for e in episodes]
        collision_free_count = sum(1 for e in episodes if e["collision_free_episode"] == 1)
        mean_dists = [e["mean_inter_robot_dist"] for e in episodes if e["mean_inter_robot_dist"] is not None]
        std_dists = [e["std_inter_robot_dist"] for e in episodes if e["std_inter_robot_dist"] is not None]
        zone_coverages = [e["zone_coverage"] for e in episodes]
        mean_pct_attacking_list = [e["mean_pct_attacking"] for e in episodes if "mean_pct_attacking" in e]
        mean_pct_defending_list = [e["mean_pct_defending"] for e in episodes if "mean_pct_defending" in e]
        flag_capture_var_list = [e["flag_capture_variance"] for e in episodes if "flag_capture_variance" in e]
        reward_per_step_list = [e["reward_per_timestep"] for e in episodes if "reward_per_timestep" in e]
        collisions_100_list = [e["collisions_per_100_steps"] for e in episodes if "collisions_per_100_steps" in e]
        near_misses_100_list = [e["near_misses_per_100_steps"] for e in episodes if "near_misses_per_100_steps" in e]
        coverage_eff_list = [e["coverage_efficiency"] for e in episodes if "coverage_efficiency" in e]

        summary = {
            "num_episodes": len(episodes),
            "win_rate": win_rate,
            "wins": wins,
            "losses": losses,
            "draws": len(episodes) - wins - losses,
            "mean_time_to_first_score": float(np.mean(time_to_first_scores)) if time_to_first_scores else None,
            "mean_time_to_game_over": float(np.mean(time_to_game_overs)) if time_to_game_overs else None,
            "mean_collisions_per_episode": float(np.mean(collisions_list)),
            "mean_collision_events_per_episode": float(np.mean(collision_events_list)),
            "mean_near_misses_per_episode": float(np.mean(near_misses_list)),
            "collision_free_rate": collision_free_count / len(episodes),
            "mean_inter_robot_dist": float(np.mean(mean_dists)) if mean_dists else None,
            "mean_std_inter_robot_dist": float(np.mean(std_dists)) if std_dists else None,
            "mean_zone_coverage": float(np.mean(zone_coverages)),
            "mean_pct_attacking": float(np.mean(mean_pct_attacking_list)) if mean_pct_attacking_list else None,
            "mean_pct_defending": float(np.mean(mean_pct_defending_list)) if mean_pct_defending_list else None,
            "mean_flag_capture_variance": float(np.mean(flag_capture_var_list)) if flag_capture_var_list else None,
            "mean_reward_per_timestep": float(np.mean(reward_per_step_list)) if reward_per_step_list else None,
            "mean_collisions_per_100_steps": float(np.mean(collisions_100_list)) if collisions_100_list else None,
            "mean_near_misses_per_100_steps": float(np.mean(near_misses_100_list)) if near_misses_100_list else None,
            "mean_coverage_efficiency": float(np.mean(coverage_eff_list)) if coverage_eff_list else None,
        }

        # Save CSV
        if save_csv is None:
            model_name = (
                "mappo" if self.blue_mode == "MAPPO"
                else "ppo" if self.blue_mode == "PPO"
                else "hppo" if self.blue_mode == "HPPO"
                else "default"
            )
            save_csv = os.path.join(_SCRIPT_DIR, f"eval_{model_name}_{opponent}_{len(episodes)}ep.csv")

        csv_columns = [
            "episode_id", "success", "time_to_first_score", "time_to_game_over",
            "collisions_per_episode", "collision_events_per_episode", "near_misses_per_episode", "collision_free_episode",
            "mean_inter_robot_dist", "std_inter_robot_dist", "zone_coverage",
            "mean_pct_attacking", "mean_pct_defending", "flag_capture_variance", "mean_flag_captures",
            "blue_episode_reward", "reward_per_timestep",
            "collisions_per_100_steps", "near_misses_per_100_steps", "coverage_efficiency",
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

        # Validation: coordination percentages sum to 100%
        if summary.get("mean_pct_attacking") is not None and summary.get("mean_pct_defending") is not None:
            _coord_sum = summary["mean_pct_attacking"] + summary["mean_pct_defending"]
            assert abs(_coord_sum - 100.0) < 1e-4, f"mean_pct_attacking + mean_pct_defending should ~100%, got {_coord_sum}"

        # Print summary (paper-ready sections)
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        print(f"Episodes: {summary['num_episodes']}")
        print(f"Win Rate: {summary['win_rate']:.2%} ({summary['wins']}W / {summary['losses']}L / {summary['draws']}D)")
        if summary["mean_time_to_first_score"] is not None:
            print(f"Mean Time to First Score: {summary['mean_time_to_first_score']:.2f}s")
        if summary["mean_time_to_game_over"] is not None:
            print(f"Mean Time to Game Over: {summary['mean_time_to_game_over']:.2f}s")
        print()
        print("Coordination Metrics")
        print("-" * 40)
        if summary.get("mean_pct_attacking") is not None:
            print(f"  Mean % Attacking: {summary['mean_pct_attacking']:.1f}%")
        if summary.get("mean_pct_defending") is not None:
            print(f"  Mean % Defending: {summary['mean_pct_defending']:.1f}%")
        if summary.get("mean_flag_capture_variance") is not None:
            print(f"  Flag Capture Variance: {summary['mean_flag_capture_variance']:.2f}")
        print()
        print("Efficiency Metrics")
        print("-" * 40)
        if summary.get("mean_reward_per_timestep") is not None:
            print(f"  Reward per Timestep: {summary['mean_reward_per_timestep']:.4f}")
        if summary.get("mean_collisions_per_100_steps") is not None:
            print(f"  Collisions per 100 steps: {summary['mean_collisions_per_100_steps']:.2f}")
        if summary.get("mean_near_misses_per_100_steps") is not None:
            print(f"  Near-Misses per 100 steps: {summary['mean_near_misses_per_100_steps']:.2f}")
        if summary.get("mean_coverage_efficiency") is not None:
            print(f"  Coverage efficiency (coverage/steps): {summary['mean_coverage_efficiency']:.6g}")
        print()
        print("Robustness (this run)")
        print("-" * 40)
        print(f"  Win Rate vs {opponent}: {summary['win_rate']:.0%}")
        print()
        print("Collision / Coverage (raw)")
        print("-" * 40)
        print(f"  Mean Collisions/Episode (per-tick): {summary['mean_collisions_per_episode']:.2f}")
        if "mean_collision_events_per_episode" in summary:
            print(f"  Mean Collision Events/Episode: {summary['mean_collision_events_per_episode']:.2f}")
        print(f"  Mean Near-Misses/Episode: {summary['mean_near_misses_per_episode']:.2f}")
        print(f"  Collision-Free Rate: {summary['collision_free_rate']:.2%}")
        if summary["mean_inter_robot_dist"] is not None:
            print(f"  Mean Inter-Robot Distance: {summary['mean_inter_robot_dist']:.2f} cells")
        print(f"  Mean Zone Coverage: {summary['mean_zone_coverage']:.2%}")
        print("=" * 60)

        summary["episodes"] = episodes
        return summary

    # ----------------------------
    # Fixed-opponent evaluation (Phase 3: robustness + generalization)
    # ----------------------------
    FIXED_OPPONENTS = ("OP1", "OP2", "OP3")
    TRAINING_OPPONENTS = frozenset({"OP1", "OP2"})  # held-out: OP3

    def evaluate_fixed_opponents(
        self,
        num_episodes_per_opp: int = 50,
        eval_model: Optional[str] = None,
        headless: bool = True,
        save_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run deterministic evaluation vs each fixed opponent (OP1, OP2, OP3).
        Log win rate, reward per timestep, time to first score per opponent.
        Compute generalization drop: train win rate − held-out win rate.
        """
        results_per_opp = {}
        for opp in self.FIXED_OPPONENTS:
            summary = self.evaluate_model(
                num_episodes=num_episodes_per_opp,
                save_csv=None,
                headless=headless,
                opponent=opp,
                eval_model=eval_model,
            )
            if not summary:
                results_per_opp[opp] = {"win_rate": 0.0, "mean_reward_per_timestep": None, "mean_time_to_first_score": None}
                continue
            results_per_opp[opp] = {
                "win_rate": summary["win_rate"],
                "mean_reward_per_timestep": summary.get("mean_reward_per_timestep"),
                "mean_time_to_first_score": summary.get("mean_time_to_first_score"),
                "wins": summary["wins"],
                "losses": summary["losses"],
                "draws": summary["draws"],
            }

        train_opps = [o for o in self.FIXED_OPPONENTS if o in self.TRAINING_OPPONENTS]
        held_out_opps = [o for o in self.FIXED_OPPONENTS if o not in self.TRAINING_OPPONENTS]
        train_win_rate = float(np.mean([results_per_opp[o]["win_rate"] for o in train_opps])) if train_opps else 0.0
        held_out_win_rate = float(np.mean([results_per_opp[o]["win_rate"] for o in held_out_opps])) if held_out_opps else 0.0
        generalization_drop = train_win_rate - held_out_win_rate

        print("\n" + "=" * 60)
        print("FIXED OPPONENT EVALUATION (Robustness)")
        print("=" * 60)
        for opp in self.FIXED_OPPONENTS:
            r = results_per_opp.get(opp, {})
            wr = r.get("win_rate", 0.0)
            rps = r.get("mean_reward_per_timestep")
            tfs = r.get("mean_time_to_first_score")
            w, l, d = r.get("wins", 0), r.get("losses", 0), r.get("draws", 0)
            print(f"  Opponent: {opp}  Win Rate: {wr:.0%}  (W/L/D: {w}/{l}/{d})")
            if rps is not None:
                print(f"    Mean Reward/Step: {rps:.4f}")
            if tfs is not None:
                print(f"    Mean Time to Score: {tfs:.2f}s")
        print()
        print("Generalization (training vs held-out opponents)")
        print("-" * 40)
        print(f"  Train opponents: {train_opps}  Win Rate: {train_win_rate:.0%}")
        print(f"  Held-out opponents: {held_out_opps}  Win Rate: {held_out_win_rate:.0%}")
        print(f"  Generalization Drop: {generalization_drop:.0%}")
        print("=" * 60)

        out = {
            "results_per_opponent": results_per_opp,
            "train_win_rate": train_win_rate,
            "held_out_win_rate": held_out_win_rate,
            "generalization_drop": generalization_drop,
        }
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            csv_path = os.path.join(save_dir, "fixed_opponent_summary.csv")
            try:
                with open(csv_path, "w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow(["opponent", "win_rate", "mean_reward_per_timestep", "mean_time_to_first_score", "wins", "losses", "draws"])
                    for opp in self.FIXED_OPPONENTS:
                        r = results_per_opp.get(opp, {})
                        w.writerow([
                            opp,
                            r.get("win_rate", ""),
                            r.get("mean_reward_per_timestep", ""),
                            r.get("mean_time_to_first_score", ""),
                            r.get("wins", ""),
                            r.get("losses", ""),
                            r.get("draws", ""),
                        ])
                    w.writerow([])
                    w.writerow(["train_win_rate", train_win_rate, "", "", "", "", ""])
                    w.writerow(["held_out_win_rate", held_out_win_rate, "", "", "", "", ""])
                    w.writerow(["generalization_drop", generalization_drop, "", "", "", "", ""])
                print(f"[Eval] Saved fixed-opponent summary to: {csv_path}")
            except Exception as exc:
                print(f"[WARN] Failed to save fixed-opponent CSV: {exc}")
        return out

    # ----------------------------
    # Naval opponents (held-out test: which baseline is best under naval realism?)
    # ----------------------------
    NAVAL_OPPONENTS = ("NAVAL_DEFENDER", "NAVAL_RUSHER", "NAVAL_BALANCED")

    def evaluate_naval_opponents(
        self,
        num_episodes_per_opp: int = 30,
        eval_model: Optional[str] = None,
        headless: bool = True,
        save_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run evaluation vs each naval opponent (NAVAL_DEFENDER, NAVAL_RUSHER, NAVAL_BALANCED).
        Uses OP3 stress (physics, sensors) so the test is naval-realistic. Held-out: not used in training.
        Returns win rate and metrics per opponent so you can compare baselines.
        """
        results_per_opp = {}
        for opp in self.NAVAL_OPPONENTS:
            summary = self.evaluate_model(
                num_episodes=num_episodes_per_opp,
                save_csv=None,
                headless=headless,
                opponent=opp,
                eval_model=eval_model,
            )
            if not summary:
                results_per_opp[opp] = {"win_rate": 0.0, "mean_reward_per_timestep": None, "mean_time_to_first_score": None, "wins": 0, "losses": 0, "draws": 0}
                continue
            results_per_opp[opp] = {
                "win_rate": summary["win_rate"],
                "mean_reward_per_timestep": summary.get("mean_reward_per_timestep"),
                "mean_time_to_first_score": summary.get("mean_time_to_first_score"),
                "wins": summary["wins"],
                "losses": summary["losses"],
                "draws": summary["draws"],
            }

        print("\n" + "=" * 60)
        print("NAVAL OPPONENT EVALUATION (held-out, physics on)")
        print("=" * 60)
        for opp in self.NAVAL_OPPONENTS:
            r = results_per_opp.get(opp, {})
            wr = r.get("win_rate", 0.0)
            w, l, d = r.get("wins", 0), r.get("losses", 0), r.get("draws", 0)
            print(f"  {opp}: WR {wr:.0%}  (W/L/D: {w}/{l}/{d})")
        print("=" * 60)

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            csv_path = os.path.join(save_dir, "naval_opponent_summary.csv")
            try:
                with open(csv_path, "w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow(["opponent", "win_rate", "wins", "losses", "draws"])
                    for opp in self.NAVAL_OPPONENTS:
                        r = results_per_opp.get(opp, {})
                        w.writerow([opp, r.get("win_rate", ""), r.get("wins", ""), r.get("losses", ""), r.get("draws", "")])
                print(f"[Eval] Saved naval summary to {csv_path}")
            except Exception as exc:
                print(f"[WARN] Failed to save naval CSV: {exc}")
        return {"results_per_opponent": results_per_opp}

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
            # Re-set opponent to OP3 for testing (OP3_HARD is training-only)
            if hasattr(self.game_field, "set_red_opponent"):
                try:
                    self.game_field.set_red_opponent("OP3")
                except Exception:
                    pass
            self.sim_tick = 0
            if self.blue_mappo_team:
                self.blue_mappo_team.reset_cache()
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
            # Re-set opponent to OP3 for testing (OP3_HARD is training-only)
            if hasattr(self.game_field, "set_red_opponent"):
                try:
                    self.game_field.set_red_opponent("OP3")
                except Exception:
                    pass
            self.sim_tick = 0
            if self.blue_mappo_team:
                self.blue_mappo_team.reset_cache()
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
                # Re-set opponent to OP3 for testing (OP3_HARD is training-only)
                if hasattr(self.game_field, "set_red_opponent"):
                    try:
                        self.game_field.set_red_opponent("OP3")
                    except Exception:
                        pass
                self.sim_tick = 0
                if self.blue_mappo_team:
                    self.blue_mappo_team.reset_cache()
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
            "F1: Full Reset | F2: Set Agents | F3: Cycle Blue (Default/PPO/MAPPO/HPPO) | F4/F5: Debug | R: Reset",
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
        if self.blue_mode == "MAPPO" and self.blue_mappo_team and self.blue_mappo_team.model_loaded:
            mappo_name = os.path.basename(self.blue_mappo_team.model_path or "")
            txt(f"MAPPO(.pth): {mappo_name}", right_x, 45, (140, 240, 140))
        elif self.blue_mode == "PPO" and self.blue_ppo_team and self.blue_ppo_team.model_loaded:
            ppo_name = os.path.basename(self.blue_ppo_team.model_path or "")
            txt(f"PPO(.zip): {ppo_name}", right_x, 45, (140, 240, 140))
        elif self.blue_mode == "HPPO" and self.blue_hppo_team and self.blue_hppo_team.model_loaded:
            low_name = os.path.basename(self.blue_hppo_team.low_model_path or "")
            high_name = os.path.basename(self.blue_hppo_team.high_model_path or "")
            txt(f"HPPO(.zip): {low_name} | {high_name}", right_x, 45, (140, 240, 140))
        else:
            txt("MAPPO/PPO/HPPO: (not loaded)", right_x, 45, (180, 180, 180))

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
    parser.add_argument("--eval", type=int, metavar="N", help="Run evaluation: N episodes, save metrics CSV (default: with display)")
    parser.add_argument("--eval-csv", type=str, metavar="PATH", help="CSV output path for evaluation (default: auto-generated)")
    parser.add_argument("--eval-model", type=str, choices=["ppo", "mappo", "hppo"], help="Model to evaluate (default: first loaded)")
    parser.add_argument("--headless", action="store_true", help="Run evaluation without display (faster)")
    parser.add_argument("--opponent", type=str, default="OP3", help="Red opponent for evaluation (OP1/OP2/OP3/OP3_EASY/OP3_HARD). Default: OP3 for testing (OP3_HARD is training-only to improve OP3 performance).")
    parser.add_argument("--eval-fixed-opponents", action="store_true", help="Run fixed-opponent eval vs OP1, OP2, OP3; report win rate and generalization drop")
    parser.add_argument("--eval-fixed-n", type=int, default=50, metavar="N", help="Episodes per opponent for --eval-fixed-opponents (default: 50)")
    parser.add_argument("--eval-naval", action="store_true", help="Run naval opponent eval vs NAVAL_DEFENDER, NAVAL_RUSHER, NAVAL_BALANCED (held-out, physics on)")
    parser.add_argument("--eval-naval-n", type=int, default=30, metavar="N", help="Episodes per opponent for --eval-naval (default: 30)")
    parser.add_argument("--ppo-model", type=str, help="Path to PPO model .zip file (overrides DEFAULT_PPO_MODEL_PATH)")
    parser.add_argument("--hppo-low", type=str, help="Path to HPPO low-level model .zip")
    parser.add_argument("--hppo-high", type=str, help="Path to HPPO high-level model .zip")
    parser.add_argument("--mappo-model", type=str, help="Path to MAPPO model .pth file (from train_mappo.py)")
    parser.add_argument("--baseline", type=str, choices=["fixed_op3", "self_play", "curriculum_no_league"], 
                        help="Test baseline mode: fixed_op3 (FIXED_OPPONENT OP3), self_play (SELF_PLAY), curriculum_no_league (CURRICULUM_NO_LEAGUE)")
    parser.add_argument("--test-baseline", action="store_true", 
                        help="Test baseline with display: run episodes with opponent matching baseline config")
    parser.add_argument("--test-baseline-n", type=int, default=10, metavar="N",
                        help="Number of episodes per opponent for baseline testing (default: 10, use 50 for more thorough testing)")

    args = parser.parse_args()

    # Auto-select model path based on baseline if not explicitly provided (works for --test-baseline and --eval)
    ppo_model_path = args.ppo_model
    if not ppo_model_path and args.baseline:
        baseline_key = args.baseline.lower()
        if baseline_key in BASELINE_MODEL_PATHS:
            ppo_model_path = BASELINE_MODEL_PATHS[baseline_key]
            print(f"[Baseline] Auto-selected model: {ppo_model_path} for {baseline_key.upper()} baseline")
    
    viewer = CTFViewer(
        ppo_model_path=ppo_model_path or DEFAULT_PPO_MODEL_PATH,
        hppo_low_path=args.hppo_low or DEFAULT_HPPO_LOW_MODEL_PATH,
        hppo_high_path=args.hppo_high or DEFAULT_HPPO_HIGH_MODEL_PATH,
        mappo_model_path=args.mappo_model or DEFAULT_MAPPO_MODEL_PATH,
    )

    # When running a baseline, use the loaded PPO model for Blue (not default scripted baseline)
    if args.baseline and getattr(viewer.blue_ppo_team, "model_loaded", False):
        viewer._apply_blue_mode("PPO")

    # Handle baseline testing mode: when --baseline is set, run N episodes (no need for --test-baseline)
    if args.baseline:
        baseline = args.baseline.lower()
        num_episodes = max(1, int(getattr(args, "test_baseline_n", 10)))
        print(f"\n[Baseline Test] Testing {baseline.upper()} baseline with display ({num_episodes} episodes)...")
        
        if baseline == "fixed_op3":
            # FIXED_OPPONENT(OP3): Always use OP3 opponent
            print("[Baseline] Configuration: FIXED_OPPONENT mode with OP3 opponent")
            summary = viewer.evaluate_model(
                num_episodes=num_episodes,
                save_csv=os.path.join(_SCRIPT_DIR, f"eval_baseline_fixed_op3_OP3_{num_episodes}ep.csv"),
                headless=False,  # Always show display for baseline testing
                opponent="OP3",
                eval_model=args.eval_model,
            )
            print("\n[Baseline Test] Complete. Viewer will remain open.")
            viewer.run()  # Keep viewer open after test
            
        elif baseline == "self_play":
            # SELF_PLAY: Use OP3 as base opponent (self-play uses snapshots, but for viewer we use OP3)
            print("[Baseline] Configuration: SELF_PLAY mode (using OP3 opponent for viewer)")
            summary = viewer.evaluate_model(
                num_episodes=num_episodes,
                save_csv=os.path.join(_SCRIPT_DIR, f"eval_baseline_self_play_OP3_{num_episodes}ep.csv"),
                headless=False,
                opponent="OP3",
                eval_model=args.eval_model,
            )
            print("\n[Baseline Test] Complete. Viewer will remain open.")
            viewer.run()
            
        elif baseline == "curriculum_no_league":
            # CURRICULUM_NO_LEAGUE: single eval vs red (default OP3); model was trained, just test vs OP3
            print("[Baseline] Configuration: CURRICULUM_NO_LEAGUE mode (eval vs OP3)")
            summary = viewer.evaluate_model(
                num_episodes=num_episodes,
                save_csv=os.path.join(_SCRIPT_DIR, f"eval_baseline_curriculum_no_league_OP3_{num_episodes}ep.csv"),
                headless=False,
                opponent="OP3",
                eval_model=args.eval_model,
            )
            print("\n[Baseline Test] Complete. Viewer will remain open.")
            viewer.run()
    
    elif getattr(args, "eval_fixed_opponents", False):
        # Fixed-opponent evaluation: OP1, OP2, OP3; generalization drop
        viewer.evaluate_fixed_opponents(
            num_episodes_per_opp=getattr(args, "eval_fixed_n", 50),
            eval_model=args.eval_model,
            headless=args.headless,
            save_dir=_SCRIPT_DIR,
        )
    elif getattr(args, "eval_naval", False):
        # Naval opponents (held-out): which baseline is best under naval realism?
        viewer.evaluate_naval_opponents(
            num_episodes_per_opp=getattr(args, "eval_naval_n", 30),
            eval_model=args.eval_model,
            headless=args.headless,
            save_dir=_SCRIPT_DIR,
        )
    elif args.eval is not None:
        # Evaluation mode: N episodes with metrics, optionally with display
        summary = viewer.evaluate_model(
            num_episodes=args.eval,
            save_csv=args.eval_csv,
            headless=args.headless,
            opponent=args.opponent,
            eval_model=args.eval_model,
        )
        if not args.headless:
            print("\n[Eval] Evaluation complete. Closing viewer or keep open.")
            viewer.run()  # Keep window open to see final state
    else:
        # Interactive mode
        viewer.run()
