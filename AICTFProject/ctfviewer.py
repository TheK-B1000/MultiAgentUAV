
import os
import sys
import csv
import math
from typing import Optional, Tuple, Any, List, Dict

import numpy as np

# NumPy 1.x compatibility: models saved with NumPy 2.x reference numpy._core.*.
if not hasattr(np, "_core"):
    import types
    _core = types.ModuleType("numpy._core")
    _core.__path__ = []
    sys.modules["numpy._core"] = _core
    for _name in ("numeric", "multiarray", "umath"):
        try:
            _sub = __import__(f"numpy.core.{_name}", fromlist=[_name])
            setattr(_core, _name, _sub)
            sys.modules[f"numpy._core.{_name}"] = _sub
        except Exception:
            pass

import pygame as pg

from viewer_game_field import ViewerGameField
from macro_actions import MacroAction
from policies import OP3RedPolicy
from config import MAP_NAME, MAP_PATH
from game_field import CNN_COLS, CNN_ROWS, NUM_CNN_CHANNELS, make_game_field

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
METRICS_DIR = os.path.join(_SCRIPT_DIR, "metrics")

DEFAULT_PPO_MODEL_PATH = "rl/checkpoints_sb3/final_ppo_league_curriculum_v2.zip"

USED_MACROS = [
    MacroAction.GO_TO,      # 0
    MacroAction.GRAB_MINE,  # 1
    MacroAction.GET_FLAG,   # 2
    MacroAction.PLACE_MINE, # 3
    MacroAction.GO_HOME,    # 4
]
N_MACROS = len(USED_MACROS)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def _try_paths(*candidates: str) -> Optional[str]:
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


def _build_team_obs_fallback(
    game_field: ViewerGameField,
    agents: List[Any],
    *,
    max_agents: int = 2,
    include_mask: bool = True,
    n_macros: int = N_MACROS,
    n_targets: int = 8,
) -> Dict[str, np.ndarray]:
    """
    Builds team observation in the same tokenized Dict format as GPUCTFVecEnv.
    """
    n = max(2, int(max_agents))
    nt = int(n_targets)
    while len(agents) < n:
        agents = list(agents) + [None]

    grids, vecs, masks = [], [], []
    for a in agents[:n]:
        if a is None:
            grids.append(np.zeros((NUM_CNN_CHANNELS, CNN_ROWS, CNN_COLS), dtype=np.float32))
            vecs.append(np.zeros((12,), dtype=np.float32))
            if include_mask:
                masks.append(np.zeros((n_macros + nt,), dtype=np.float32))
            continue

        grid = np.asarray(game_field.build_observation(a), dtype=np.float32)
        grids.append(grid)

        if hasattr(game_field, "build_continuous_features"):
            v = np.asarray(game_field.build_continuous_features(a), dtype=np.float32).reshape(-1)
            if v.size < 12:
                v = np.pad(v, (0, 12 - v.size))
            elif v.size > 12:
                v = v[:12]
        else:
            v = np.zeros((12,), dtype=np.float32)
        vecs.append(v)

        if include_mask:
            try:
                mm = np.asarray(game_field.get_macro_mask(a), dtype=np.bool_).reshape(-1)
                if mm.shape != (n_macros,) or not mm.any():
                    mm = np.ones((n_macros,), dtype=np.bool_)
            except Exception:
                mm = np.ones((n_macros,), dtype=np.bool_)
            try:
                tm = np.asarray(game_field.get_target_mask(a), dtype=np.bool_).reshape(-1)
                if tm.shape != (nt,) or not tm.any():
                    tm = np.ones((nt,), dtype=np.bool_)
            except Exception:
                tm = np.ones((nt,), dtype=np.bool_)
            masks.append(np.concatenate([mm.astype(np.float32), tm.astype(np.float32)]))

    agent_mask = np.zeros((n,), dtype=np.float32)
    n_live = sum(1 for a in agents[:n] if a is not None)
    agent_mask[:n_live] = 1.0

    out: Dict[str, np.ndarray] = {
        "grid": np.stack(grids).astype(np.float32),
        "vec": np.stack(vecs).astype(np.float32),
        "agent_mask": agent_mask,
    }
    if include_mask and masks:
        out["mask"] = np.concatenate(masks).astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# SB3 PPO policy wrapper
# ---------------------------------------------------------------------------
class SB3TeamPPOPolicy:
    """
    Viewer-side SB3 PPO wrapper.

    Translates SB3 action indices (macro_idx, target_idx) per agent into
    (MacroAction, target_cell) pairs that the viewer's internal policy
    pipeline expects.

    Compatible with models trained on GPUCTFVecEnv (tokenized Dict obs).
    """

    def __init__(
        self,
        model_path: str,
        env: ViewerGameField,
        deterministic: bool = True,
    ):
        self.model_path_raw = model_path
        self.model_path: Optional[str] = _resolve_zip_path(model_path)
        self.model_loaded: bool = False
        self.model: Optional[Any] = None
        self.deterministic = bool(deterministic)
        self.n_targets = int(getattr(env, "num_macro_targets", 8) or 8)
        self._cache_tick: int = -1
        self._cache_action: np.ndarray = np.array([0, 0, 0, 0], dtype=np.int64)

        if self.model_path is None:
            print(f"[CTFViewer] PPO model not found: {model_path}")
            return

        try:
            from stable_baselines3 import PPO as SB3PPO
            self.model = SB3PPO.load(self.model_path, device="cpu")
            self.model.policy.set_training_mode(False)
            self.model_loaded = True
            tok = "tokenized" if self._model_expects_tokenized() else "legacy"
            print(f"[CTFViewer] Loaded PPO model: {self.model_path}")
            print(f"[CTFViewer] Observation mode: {tok}")
        except OSError as e:
            print(f"[CTFViewer] Torch/SB3 DLL error: {e}")
        except FileNotFoundError:
            print(f"[CTFViewer] PPO model file not found: {self.model_path}")
        except Exception as e:
            print(f"[CTFViewer] Failed to load PPO model: {e}")
            import traceback
            traceback.print_exc()

    def reset_cache(self) -> None:
        self._cache_tick = -1
        self._cache_action = np.array([0, 0, 0, 0], dtype=np.int64)

    # ---- Observation space introspection ----

    def _model_expects_mask(self) -> bool:
        if self.model is None:
            return False
        space = getattr(self.model.policy, "observation_space", None)
        if hasattr(space, "spaces") and isinstance(space.spaces, dict):
            return "mask" in space.spaces
        return False

    def _model_expects_tokenized(self) -> bool:
        if self.model is None:
            return False
        space = getattr(self.model.policy, "observation_space", None)
        if not hasattr(space, "spaces") or not isinstance(space.spaces, dict):
            return False
        grid_space = space.spaces.get("grid")
        if grid_space is None or not hasattr(grid_space, "shape"):
            return False
        return len(grid_space.shape) == 4

    def _model_n_agents(self) -> int:
        if self.model is None:
            return 2
        space = getattr(self.model.policy, "observation_space", None)
        if hasattr(space, "spaces") and isinstance(space.spaces, dict):
            gs = space.spaces.get("grid")
            if gs is not None and hasattr(gs, "shape") and len(gs.shape) == 4:
                return int(gs.shape[0])
        return 2

    # ---- Observation building ----

    def _build_team_obs(self, game_field: ViewerGameField, side: str) -> Dict[str, np.ndarray]:
        agents = game_field.blue_agents if side == "blue" else game_field.red_agents
        live = [a for a in agents if a is not None]

        def _agent_id(agent: Any) -> int:
            try:
                return int(getattr(agent, "agent_id", -1))
            except Exception:
                return -1
        live.sort(key=_agent_id)
        while len(live) < 2:
            live.append(live[0] if live else None)

        use_tokenized = self._model_expects_tokenized()
        max_agents = self._model_n_agents() if use_tokenized else 2
        nt = int(getattr(game_field, "num_macro_targets", self.n_targets) or self.n_targets)

        return _build_team_obs_fallback(
            game_field,
            live[:max_agents],
            max_agents=max_agents,
            include_mask=self._model_expects_mask(),
            n_macros=N_MACROS,
            n_targets=nt,
        )

    # ---- Action mask enforcement ----

    def _macro_uses_target(self, macro: int) -> bool:
        return macro in (MacroAction.GO_TO, MacroAction.GET_FLAG,
                         MacroAction.GRAB_MINE, MacroAction.PLACE_MINE)

    def _sanitize_action_with_mask(
        self, action: np.ndarray, game_field: ViewerGameField, agent_idx: int,
    ) -> Tuple[int, int]:
        macro = int(action[0]) % N_MACROS
        nt = max(1, int(getattr(game_field, "num_macro_targets", self.n_targets) or self.n_targets))
        target = int(action[1]) % nt

        blue_agents = getattr(game_field, "blue_agents", []) or []
        if agent_idx >= len(blue_agents):
            return macro, target
        agent = blue_agents[agent_idx]
        if agent is None:
            return macro, target

        try:
            mm = np.asarray(game_field.get_macro_mask(agent), dtype=np.bool_).reshape(-1)
            if mm.shape == (N_MACROS,) and mm.any() and not bool(mm[macro]):
                valid = np.flatnonzero(mm)
                macro = int(valid[0]) if valid.size > 0 else 0
        except Exception:
            pass

        try:
            if self._macro_uses_target(macro):
                tm = np.asarray(game_field.get_target_mask(agent), dtype=np.bool_).reshape(-1)
                if tm.shape == (nt,) and tm.any() and not bool(tm[target]):
                    valid = np.flatnonzero(tm)
                    target = int(valid[0]) if valid.size > 0 else 0
        except Exception:
            pass

        return macro, target

    def _resolve_target_conflicts(self, actions: np.ndarray, game_field: ViewerGameField) -> np.ndarray:
        a = actions.copy()
        blue_agents = getattr(game_field, "blue_agents", []) or []
        if len(blue_agents) < 2:
            return a

        from collections import defaultdict
        groups: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        for i in range(2):
            m, t = int(a[i * 2]), int(a[i * 2 + 1])
            if self._macro_uses_target(m):
                groups[(m, t)].append(i)

        for (m, t), idxs in groups.items():
            if len(idxs) <= 1:
                continue
            try:
                tgt_cell = game_field.get_macro_target(t)
                if tgt_cell is None or len(tgt_cell) < 2:
                    continue
                tx, ty = int(tgt_cell[0]), int(tgt_cell[1])
            except Exception:
                continue

            def dist_to_target(ai: int) -> float:
                if ai >= len(blue_agents) or blue_agents[ai] is None:
                    return float("inf")
                pos = getattr(blue_agents[ai], "float_pos",
                              (getattr(blue_agents[ai], "x", 0), getattr(blue_agents[ai], "y", 0)))
                return (float(pos[0]) - tx) ** 2 + (float(pos[1]) - ty) ** 2

            sorted_idxs = sorted(idxs, key=dist_to_target)
            nt = max(1, int(getattr(game_field, "num_macro_targets", self.n_targets) or self.n_targets))
            for rank, ai in enumerate(sorted_idxs):
                if rank == 0:
                    continue
                if ai >= len(blue_agents) or blue_agents[ai] is None:
                    a[ai * 2], a[ai * 2 + 1] = 4, 0
                    continue
                try:
                    tm = np.asarray(game_field.get_target_mask(blue_agents[ai]), dtype=np.bool_).reshape(-1)
                    if tm.shape != (nt,) or not tm.any():
                        a[ai * 2], a[ai * 2 + 1] = 4, 0
                        continue
                    best_alt, best_d = 0, float("inf")
                    for ti in range(nt):
                        if not bool(tm[ti]) or ti == t:
                            continue
                        try:
                            tc = game_field.get_macro_target(ti)
                            tcx, tcy = int(tc[0]), int(tc[1]) if len(tc) >= 2 else 0
                        except Exception:
                            continue
                        pos = getattr(blue_agents[ai], "float_pos",
                                      (getattr(blue_agents[ai], "x", 0), getattr(blue_agents[ai], "y", 0)))
                        d = (float(pos[0]) - tcx) ** 2 + (float(pos[1]) - tcy) ** 2
                        if d < best_d:
                            best_d, best_alt = d, ti
                    a[ai * 2], a[ai * 2 + 1] = m, best_alt
                except Exception:
                    a[ai * 2], a[ai * 2 + 1] = 4, 0

        return a

    # ---- Core inference ----

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

        if a.size < 4:
            padded = np.zeros((4,), dtype=np.int64)
            padded[:a.size] = a
            a = padded
        elif a.size > 4:
            a = a[:4]

        a[0] = int(a[0]) % N_MACROS
        a[2] = int(a[2]) % N_MACROS
        nt = max(1, int(getattr(game_field, "num_macro_targets", self.n_targets) or self.n_targets))
        a[1] = int(a[1]) % nt
        a[3] = int(a[3]) % nt

        m0, t0 = self._sanitize_action_with_mask(np.array([a[0], a[1]]), game_field, 0)
        m1, t1 = self._sanitize_action_with_mask(np.array([a[2], a[3]]), game_field, 1)
        a[0], a[1], a[2], a[3] = m0, t0, m1, t1
        a = self._resolve_target_conflicts(a, game_field)

        self._cache_tick = tick
        self._cache_action = a

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
        if isinstance(mt, list) and mt:
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
        side = str(getattr(agent, "side", "blue")).lower()
        self._compute_joint_action_if_needed(game_field, tick=tick, side=side)

        blue_agents = getattr(game_field, "blue_agents", []) or []
        live = [a for a in blue_agents if a is not None]

        def _agent_id(a: Any) -> int:
            try:
                return int(getattr(a, "agent_id", -1))
            except Exception:
                return -1
        live.sort(key=_agent_id)

        slot_index = 0
        for i, a in enumerate(live[:2]):
            if a is agent or (_agent_id(a) == _agent_id(agent) and _agent_id(a) >= 0):
                slot_index = i
                break
        slot_index = max(0, min(1, slot_index))

        macro_idx = int(self._cache_action[slot_index * 2]) % N_MACROS
        target_idx = int(self._cache_action[slot_index * 2 + 1])

        macro = USED_MACROS[macro_idx]
        if macro == MacroAction.PLACE_MINE:
            try:
                return macro, (int(getattr(agent, "x", 0)), int(getattr(agent, "y", 0)))
            except Exception:
                return macro, self._resolve_target_cell(game_field, target_idx)

        return macro, self._resolve_target_cell(game_field, target_idx)


# ---------------------------------------------------------------------------
# CTFViewer  (PPO-only)
# ---------------------------------------------------------------------------
class CTFViewer:
    def __init__(
        self,
        ppo_model_path: str = DEFAULT_PPO_MODEL_PATH,
    ):
        # ---- Game field ----
        if MAP_NAME or MAP_PATH:
            base = make_game_field(map_name=MAP_NAME or None, map_path=MAP_PATH or None)
            grid = [list(row) for row in getattr(base, "grid", [[0] * 20 for _ in range(20)])]
        else:
            grid = [[0] * 20 for _ in range(20)]

        self.game_field = ViewerGameField(grid)
        self.game_manager = self.game_field.getGameManager()
        self._set_phase_op3()

        if hasattr(self.game_field, "use_internal_policies"):
            self.game_field.use_internal_policies = True
        if hasattr(self.game_field, "set_external_control"):
            try:
                self.game_field.set_external_control("blue", False)
                self.game_field.set_external_control("red", False)
            except Exception:
                pass

        # ---- Pygame ----
        pg.init()
        self.size = (1024, 720)
        try:
            self.screen = pg.display.set_mode(self.size, pg.SCALED | pg.DOUBLEBUF, vsync=1)
        except TypeError:
            self.screen = pg.display.set_mode(self.size, pg.SCALED | pg.DOUBLEBUF)
        pg.display.set_caption("UAV CTF Viewer | PPO")
        self.clock = pg.time.Clock()
        self.font = pg.font.SysFont(None, 26)
        self.bigfont = pg.font.SysFont(None, 48)

        self.input_active = False
        self.input_text = ""
        self.sim_tick: int = 0

        # Sanity-check obs shape
        if getattr(self.game_field, "blue_agents", None):
            try:
                dummy_obs = self.game_field.build_observation(self.game_field.blue_agents[0])
                c, h, w = len(dummy_obs), len(dummy_obs[0]), len(dummy_obs[0][0])
                print(f"[CTFViewer] CNN obs shape: C={c}, H={h}, W={w}")
                print(f"[CTFViewer] num_macro_targets: {_safe_int(getattr(self.game_field, 'num_macro_targets', 0))}")
            except Exception:
                print("[CTFViewer] Could not infer CNN obs shape.")
        else:
            print("[CTFViewer] No agents spawned; cannot infer obs shape.")

        # ---- Policies ----
        if hasattr(self.game_field, "policies") and isinstance(self.game_field.policies, dict):
            self.game_field.policies["red"] = OP3RedPolicy("red")
        self.blue_op3_baseline = OP3RedPolicy("blue")

        # ---- PPO model ----
        self.blue_ppo = SB3TeamPPOPolicy(ppo_model_path, self.game_field, deterministic=True)

        if self.blue_ppo.model_loaded:
            print("[CTFViewer] PPO model ready. Press F3 to toggle DEFAULT <-> PPO.")
        else:
            print("[CTFViewer] WARNING: No PPO model loaded. Running in DEFAULT (scripted) mode only.")
            resolved = _resolve_zip_path(ppo_model_path)
            if resolved is None:
                print(f"[CTFViewer] Searched for: {ppo_model_path}")
                print(f"[CTFViewer] Also checked: {os.path.join(_SCRIPT_DIR, ppo_model_path)}")

        self._blue_policy_callable = None
        self.blue_mode: str = "DEFAULT"
        self._apply_blue_mode(self.blue_mode)
        self._reset_op3_policies()

        if hasattr(self.game_field, "set_red_opponent"):
            try:
                self.game_field.set_red_opponent("OP3")
            except Exception:
                pass

    # ---- Phase helpers ----

    def _set_phase_op3(self) -> None:
        for obj in (self.game_manager, self.game_field):
            if hasattr(obj, "set_phase"):
                try:
                    obj.set_phase("OP3")
                except Exception:
                    pass

        try:
            if hasattr(self.game_field, "set_physics_enabled"):
                self.game_field.set_physics_enabled(True)
            if hasattr(self.game_field, "set_disturbance_config"):
                self.game_field.set_disturbance_config(
                    current_strength_cps=0.12, drift_sigma_cells=0.03,
                )
            if hasattr(self.game_field, "set_robotics_constraints"):
                self.game_field.set_robotics_constraints(
                    action_delay_steps=2, actuation_noise_sigma=0.0,
                )
            if hasattr(self.game_field, "set_sensor_config"):
                sr = getattr(getattr(self.game_field, "boat_cfg", None), "sensor_range_cells", 9999.0) or 9999.0
                self.game_field.set_sensor_config(
                    sensor_range_cells=float(sr),
                    sensor_noise_sigma_cells=0.2,
                    sensor_dropout_prob=0.05,
                )
            if hasattr(self.game_field, "set_dynamics_config"):
                self.game_field.set_dynamics_config(
                    max_speed_cps=2.2, max_accel_cps2=2.0, max_yaw_rate_rps=4.0,
                )
            if not hasattr(self, "_op3_stress_applied"):
                print("[CTFViewer] Applied OP3 stress settings")
                self._op3_stress_applied = True
        except Exception as e:
            print(f"[CTFViewer] Warning: Could not apply OP3 stress settings: {e}")

    # ---- Blue mode management (DEFAULT / PPO only) ----

    def _apply_blue_mode(self, mode: str) -> None:
        if not (hasattr(self.game_field, "policies") and isinstance(self.game_field.policies, dict)):
            return

        if mode == "PPO" and self.blue_ppo.model_loaded:
            def blue_policy(obs, agent, game_field):
                return self.blue_ppo.act_for_agent(agent, game_field, tick=self.sim_tick)
            self._blue_policy_callable = blue_policy
            self.game_field.policies["blue"] = self._blue_policy_callable
            self.blue_mode = "PPO"
            self.blue_ppo.reset_cache()
            print("[CTFViewer] Blue -> PPO")
        else:
            self._blue_policy_callable = None
            self.game_field.policies["blue"] = self.blue_op3_baseline
            self.blue_mode = "DEFAULT"
            print("[CTFViewer] Blue -> Default baseline")

    def _cycle_blue_mode(self) -> None:
        if self.blue_mode == "DEFAULT" and self.blue_ppo.model_loaded:
            self._apply_blue_mode("PPO")
        else:
            self._apply_blue_mode("DEFAULT")
            self._reset_op3_policies()

    def _reset_op3_policies(self) -> None:
        if hasattr(self.game_field, "policies") and isinstance(self.game_field.policies, dict):
            for pol in self.game_field.policies.values():
                if isinstance(pol, OP3RedPolicy) and hasattr(pol, "reset"):
                    try:
                        pol.reset()
                    except Exception:
                        pass

    # ---- Reset helper ----

    def _do_reset(self) -> None:
        self.game_field.agents_per_team = 2
        self.game_field.reset_default()
        self._set_phase_op3()
        if hasattr(self.game_field, "set_red_opponent"):
            try:
                self.game_field.set_red_opponent("OP3")
            except Exception:
                pass
        self.sim_tick = 0
        self.blue_ppo.reset_cache()
        self._reset_op3_policies()

    # ---- Main loop ----

    def run(self) -> None:
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
                        self._handle_input_key(event)
                    else:
                        self._handle_main_key(event)

            steps = 0
            while acc >= fixed_dt and steps < max_substeps:
                self.sim_tick += 1
                self.game_field.update(fixed_dt)
                acc -= fixed_dt
                steps += 1
            if steps == max_substeps:
                acc = 0.0

            alpha = acc / fixed_dt
            self._draw(alpha=alpha)
            pg.display.flip()

        pg.quit()
        sys.exit()

    # ---- Evaluation ----

    def evaluate_model(
        self,
        num_episodes: int = 100,
        save_csv: Optional[str] = None,
        headless: bool = False,
        opponent: str = "OP3",
    ) -> Dict[str, Any]:
        if not headless:
            print(f"[Eval] Running {num_episodes} episodes with display. Press ESC to stop early.")
        else:
            print(f"[Eval] Running {num_episodes} episodes headless...")

        if self.blue_mode == "DEFAULT" and self.blue_ppo.model_loaded:
            self._apply_blue_mode("PPO")
            print("[Eval] Switched to PPO mode for evaluation")
        elif not self.blue_ppo.model_loaded:
            print("[Eval] WARNING: No PPO model loaded. Evaluating DEFAULT baseline.")

        opponent_upper = str(opponent).upper()
        if hasattr(self.game_field, "set_red_opponent"):
            try:
                self.game_field.set_red_opponent(opponent_upper)
            except Exception:
                pass

        fixed_dt = 1.0 / 60.0
        decision_interval = float(getattr(self.game_field, "decision_interval_seconds", 0.7))
        max_time_seconds = 400 * 0.99 * decision_interval
        max_steps_per_episode = max(3600, int(max_time_seconds * 60))
        if not headless:
            print(f"[Eval] Step limit: {max_steps_per_episode} ticks (~{max_steps_per_episode / 60:.1f}s)")

        episodes: List[Dict[str, Any]] = []

        if not headless:
            if not pg.get_init():
                pg.init()

        try:
            for ep_idx in range(num_episodes):
                self.game_field.reset_default()
                self._set_phase_op3()
                self.sim_tick = 0
                self.blue_ppo.reset_cache()

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
                                running_episode = False
                                break

                    self.sim_tick += 1
                    self.game_field.update(fixed_dt)
                    step_count += 1

                    gm = self.game_manager
                    if getattr(gm, "game_over", False):
                        break

                    if n_blue > 0 and hasattr(gf, "_zone_ranges") and hasattr(gf, "_agent_cell_pos"):
                        try:
                            (_, _), (red_min_col, red_max_col) = gf._zone_ranges()
                            for i, agent in enumerate(blue_agents_list):
                                if i >= len(attack_steps):
                                    break
                                en = getattr(agent, "isEnabled", True)
                                if callable(en):
                                    en = en()
                                if not en:
                                    continue
                                cell = gf._agent_cell_pos(agent)
                                col = cell[0] if isinstance(cell, (tuple, list)) else 0
                                carrying = gf._agent_is_carrying_flag(agent) if hasattr(gf, "_agent_is_carrying_flag") else False
                                if red_min_col <= col <= red_max_col or carrying:
                                    attack_steps[i] += 1
                                else:
                                    defend_steps[i] += 1
                        except Exception:
                            pass

                    if not headless:
                        self._draw(alpha=1.0)
                        pg.display.flip()
                        self.clock.tick(60)

                # Collect per-episode metrics
                gm = self.game_manager
                blue_score = int(getattr(gm, "blue_score", 0))
                red_score = int(getattr(gm, "red_score", 0))
                success = 1 if blue_score > red_score else 0

                ttf = getattr(gm, "time_to_first_score", None)
                ttg = getattr(gm, "time_to_game_over", None)
                time_to_first_score = float(ttf) if ttf is not None else None
                time_to_game_over = float(ttg) if ttg is not None else float(getattr(gm, "sim_time", 0.0))

                collision_events = int(getattr(gm, "collision_events_this_episode", 0))
                collision_free = 1 if collision_events == 0 else 0

                total_steps = max(1, step_count)
                pct_atk = []
                pct_def = []
                for i in range(len(attack_steps)):
                    active = attack_steps[i] + defend_steps[i]
                    if active > 0:
                        pct_atk.append(100.0 * attack_steps[i] / active)
                        pct_def.append(100.0 * defend_steps[i] / active)
                    else:
                        pct_atk.append(0.0)
                        pct_def.append(0.0)

                blue_reward = float(getattr(gm, "blue_episode_reward", 0.0) or 0.0)

                row = {
                    "episode_id": ep_idx + 1,
                    "success": success,
                    "blue_score": blue_score,
                    "red_score": red_score,
                    "time_to_first_score": time_to_first_score,
                    "time_to_game_over": time_to_game_over,
                    "collision_events": collision_events,
                    "collision_free": collision_free,
                    "mean_pct_attacking": float(np.mean(pct_atk)) if pct_atk else 0.0,
                    "mean_pct_defending": float(np.mean(pct_def)) if pct_def else 0.0,
                    "blue_episode_reward": blue_reward,
                    "reward_per_timestep": blue_reward / total_steps,
                    "opponent": opponent_upper,
                }
                episodes.append(row)

                if (ep_idx + 1) % 10 == 0:
                    wins = sum(1 for e in episodes if e["success"])
                    print(f"[Eval] Episode {ep_idx + 1}/{num_episodes} | Win rate: {wins / len(episodes):.0%}")

        except KeyboardInterrupt:
            print(f"[Eval] Interrupted after {len(episodes)} episodes")

        if not episodes:
            print("[Eval] No episodes completed!")
            return {}

        wins = sum(1 for e in episodes if e["success"])
        summary = {
            "num_episodes": len(episodes),
            "win_rate": wins / len(episodes),
            "wins": wins,
            "losses": len(episodes) - wins,
            "mean_time_to_first_score": float(np.mean([e["time_to_first_score"] for e in episodes
                                                        if e["time_to_first_score"] is not None])) or None,
            "mean_time_to_game_over": float(np.mean([e["time_to_game_over"] for e in episodes
                                                      if e["time_to_game_over"] is not None])) or None,
            "mean_collision_events": float(np.mean([e["collision_events"] for e in episodes])),
            "collision_free_rate": sum(1 for e in episodes if e["collision_free"]) / len(episodes),
            "mean_pct_attacking": float(np.mean([e["mean_pct_attacking"] for e in episodes])),
            "mean_pct_defending": float(np.mean([e["mean_pct_defending"] for e in episodes])),
            "mean_reward_per_timestep": float(np.mean([e["reward_per_timestep"] for e in episodes])),
        }

        # Save CSV
        if save_csv is None:
            tag = "ppo" if self.blue_mode == "PPO" else "default"
            save_csv = os.path.join(METRICS_DIR, f"eval_{tag}_{opponent_upper}_{len(episodes)}ep.csv")
        try:
            os.makedirs(os.path.dirname(os.path.abspath(save_csv)) or ".", exist_ok=True)
            cols = list(episodes[0].keys())
            with open(save_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
                w.writeheader()
                for r in episodes:
                    w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else ("" if v is None else str(v)))
                                for k, v in r.items()})
            print(f"[Eval] Saved {len(episodes)} episodes to: {save_csv}")
        except Exception as exc:
            print(f"[Eval] Failed to save CSV: {exc}")

        # Print summary
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        print(f"  Episodes:   {summary['num_episodes']}")
        print(f"  Win Rate:   {summary['win_rate']:.0%}  ({summary['wins']}W / {summary['losses']}L)")
        if summary["mean_time_to_first_score"] is not None:
            print(f"  Avg Time to First Score: {summary['mean_time_to_first_score']:.2f}s")
        if summary["mean_time_to_game_over"] is not None:
            print(f"  Avg Game Length:         {summary['mean_time_to_game_over']:.2f}s")
        print(f"  Collision Events/Ep:     {summary['mean_collision_events']:.2f}")
        print(f"  Collision-Free Rate:     {summary['collision_free_rate']:.0%}")
        print(f"  Avg % Attacking:         {summary['mean_pct_attacking']:.1f}%")
        print(f"  Avg % Defending:         {summary['mean_pct_defending']:.1f}%")
        print(f"  Reward/Timestep:         {summary['mean_reward_per_timestep']:.4f}")
        print("=" * 60)

        summary["episodes"] = episodes
        return summary

    # ---- Input handling ----

    def _handle_main_key(self, event: Any) -> None:
        k = event.key
        if k == pg.K_F1:
            self._do_reset()
        elif k == pg.K_F2:
            self.input_active = True
            self.input_text = str(self.game_field.agents_per_team)
        elif k == pg.K_F3:
            self._cycle_blue_mode()
        elif k == pg.K_F4:
            self.game_field.debug_draw_ranges = not getattr(self.game_field, "debug_draw_ranges", False)
        elif k == pg.K_r:
            self._do_reset()
        elif k == pg.K_ESCAPE:
            pg.event.post(pg.event.Event(pg.QUIT))

    def _handle_input_key(self, event: Any) -> None:
        if event.key == pg.K_RETURN:
            try:
                n = max(1, min(100, int(self.input_text or "2")))
                if hasattr(self.game_field, "set_agent_count_and_reset"):
                    self.game_field.set_agent_count_and_reset(n)
                else:
                    self.game_field.agents_per_team = n
                    self.game_field.reset_default()
                self._set_phase_op3()
                self.sim_tick = 0
                self.blue_ppo.reset_cache()
                self._reset_op3_policies()
            except Exception as e:
                print(f"[CTFViewer] Error: {e}")
            self.input_active = False
        elif event.key == pg.K_ESCAPE:
            self.input_active = False
        elif event.key == pg.K_BACKSPACE:
            self.input_text = self.input_text[:-1]
        elif event.unicode.isdigit() and len(self.input_text) < 3:
            self.input_text += event.unicode

    # ---- Drawing / HUD ----

    def _draw(self, alpha: float = 1.0) -> None:
        self.screen.fill((12, 12, 18))

        hud_h = 90
        field_rect = pg.Rect(20, hud_h + 10, self.size[0] - 20, self.size[1] - hud_h - 20)
        self.game_field.draw(self.screen, field_rect, alpha=alpha)

        def txt(text: str, x: int, y: int, color: Tuple[int, int, int] = (230, 230, 240)) -> None:
            img = self.font.render(text, True, color)
            self.screen.blit(img, (x, y))

        gm = self.game_manager
        mode = self.blue_mode
        mode_clr = (120, 255, 120) if mode == "PPO" else (255, 255, 120)

        txt("F1/R: Reset | F2: Set Agents | F3: Toggle PPO/Default | F4: Debug Ranges", 30, 12, (200, 200, 220))

        txt(f"Blue: {mode} | Agents: {self.game_field.agents_per_team}v{self.game_field.agents_per_team}",
            30, 40, mode_clr)

        if mode == "PPO" and self.blue_ppo.model_loaded:
            name = os.path.basename(self.blue_ppo.model_path or "")
            txt(f"Model: {name}", 400, 40, (140, 240, 140))
        elif mode == "PPO":
            txt("Model: NOT LOADED", 400, 40, (255, 100, 100))

        txt(f"BLUE: {getattr(gm, 'blue_score', 0)}", 30, 68, (100, 180, 255))
        txt(f"RED: {getattr(gm, 'red_score', 0)}", 200, 68, (255, 100, 100))
        txt(f"Time: {int(getattr(gm, 'current_time', 0.0))}s", 370, 68, (220, 220, 255))

        # Input overlay
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
            hint = self.font.render("Enter to confirm | Esc to cancel", True, (200, 200, 200))

            self.screen.blit(title, title.get_rect(center=(box.centerx, box.centery - 50)))
            self.screen.blit(entry, entry.get_rect(center=box.center))
            self.screen.blit(hint, hint.get_rect(center=(box.centerx, box.centery + 60)))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CTF Viewer (PPO-only)")
    parser.add_argument("--ppo-model", type=str, default=None,
                        help=f"Path to PPO .zip (default: {DEFAULT_PPO_MODEL_PATH})")
    parser.add_argument("--eval", type=int, metavar="N",
                        help="Run N evaluation episodes and save metrics CSV")
    parser.add_argument("--eval-csv", type=str, metavar="PATH",
                        help="CSV output path (default: auto-generated)")
    parser.add_argument("--headless", action="store_true",
                        help="Run evaluation without display (faster)")
    parser.add_argument("--opponent", type=str, default="OP3",
                        help="Red opponent tag (default: OP3)")
    parser.add_argument("--agents-per-team", type=int, default=None,
                        help="Override agents per team (default: 2)")
    args = parser.parse_args()

    viewer = CTFViewer(ppo_model_path=args.ppo_model or DEFAULT_PPO_MODEL_PATH)

    if args.agents_per_team is not None:
        n = max(1, int(args.agents_per_team))
        gf = viewer.game_field
        if hasattr(gf, "set_agent_count_and_reset"):
            gf.set_agent_count_and_reset(n)
        else:
            gf.agents_per_team = n
            gf.reset_default()

    if args.eval is not None:
        summary = viewer.evaluate_model(
            num_episodes=args.eval,
            save_csv=args.eval_csv,
            headless=args.headless,
            opponent=args.opponent,
        )
        if not args.headless:
            viewer.run()
    else:
        viewer.run()
