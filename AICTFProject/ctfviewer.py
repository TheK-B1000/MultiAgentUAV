"""
CTF Viewer -- renders a single BatchedCTFCore environment in pygame.

Blue team is controlled by an SB3 PPO model (if found) or scripted DEMO.
Red team uses the scripted bot built into the GPU core.

No dependency on game_field.py, viewer_game_field.py, or policies.py.
"""

import os
import sys
import csv
import math
from typing import Optional, Tuple, Any, List, Dict

import numpy as np


def _ensure_numpy_core_compat() -> None:
    """Register numpy._core.* so models saved under NumPy 2.x load on NumPy 1.x (and vice versa)."""
    try:
        import types
        # NumPy 1.x: create numpy._core and point numpy._core.* to numpy.core.*
        if not hasattr(np, "_core"):
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
        # Ensure numpy._core.numeric is always available (unpickle may need it)
        if "numpy._core.numeric" not in sys.modules:
            _sub = None
            try:
                _sub = __import__("numpy.core.numeric", fromlist=["numeric"])
            except Exception:
                pass
            if _sub is not None:
                sys.modules["numpy._core.numeric"] = _sub
                if hasattr(np, "_core") and not hasattr(np._core, "numeric"):
                    setattr(np._core, "numeric", _sub)
    except Exception:
        pass


def _ensure_numpy_random_compat() -> None:
    """Register numpy.random._pcg64 and patch unpickler so models saved with NumPy 2.x load."""
    try:
        # Ensure _pcg64 is importable (NumPy 1.17+ has it)
        try:
            import numpy.random._pcg64 as _pcg64
            sys.modules["numpy.random._pcg64"] = _pcg64
        except Exception:
            import numpy.random
            if hasattr(numpy.random, "_pcg64"):
                sys.modules["numpy.random._pcg64"] = getattr(numpy.random, "_pcg64")

        # Patch so BitGenerator *class* (from pickle) is accepted, not only string name
        import numpy.random._pickle as _np_pickle
        from numpy.random.bit_generator import BitGenerator
        _orig_ctor = getattr(_np_pickle, "__bit_generator_ctor", None)
        if _orig_ctor is not None:

            def _bit_generator_ctor_patched(bit_generator):
                if isinstance(bit_generator, type) and issubclass(bit_generator, BitGenerator):
                    return bit_generator()
                return _orig_ctor(bit_generator)

            _np_pickle.__bit_generator_ctor = _bit_generator_ctor_patched
    except Exception:
        pass


_ensure_numpy_core_compat()
_ensure_numpy_random_compat()

import torch
import pygame as pg
from gymnasium import spaces

from game_field_gpu import (
    GPUFieldConfig,
    BatchedCTFCore,
    CNN_COLS,
    CNN_ROWS,
    NUM_CNN_CHANNELS,
)
from rl.curriculum import STRESS_BY_PHASE
from opponent_params import sample_batched_opponent_params

# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
METRICS_DIR = os.path.join(_SCRIPT_DIR, "csv")
DEFAULT_PPO_MODEL_PATH = "checkpoints_sb3/2v2/final_ppo_league_2v2.zip"
N_MACROS = 5
N_TARGETS = 50


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
        sr = os.path.join(_SCRIPT_DIR, path)
        candidates.append(sr)
        if not sr.endswith(".zip"):
            candidates.append(sr + ".zip")
    return _try_paths(*candidates)


def _make_obs_action_spaces(n_blue: int, n_macros: int = N_MACROS, n_targets: int = N_TARGETS):
    """Build observation and action spaces for GPU CTF (so SB3 can load when saved ones fail to unpickle)."""
    obs_space = spaces.Dict(
        {
            "grid": spaces.Box(low=0.0, high=1.0, shape=(n_blue, NUM_CNN_CHANNELS, CNN_ROWS, CNN_COLS), dtype=np.float32),
        "vec": spaces.Box(low=-1.0, high=1.0, shape=(n_blue, 18), dtype=np.float32),
            "agent_mask": spaces.Box(low=0.0, high=1.0, shape=(n_blue,), dtype=np.float32),
            "mask": spaces.Box(low=0.0, high=1.0, shape=(n_blue * (n_macros + n_targets),), dtype=np.float32),
        }
    )
    action_space = spaces.MultiDiscrete([n_macros, n_targets] * n_blue)
    return obs_space, action_space


# ---------------------------------------------------------------------------
# PPO wrapper  (loads SB3 model, builds obs from core, returns flat action)
# ---------------------------------------------------------------------------
class PPOController:
    """
    Wraps an SB3 PPO model trained on GPUCTFVecEnv.  Given a
    BatchedCTFCore (B=1), produces a flat int64 action tensor each tick.
    """

    def __init__(
        self,
        model_path: str,
        n_macros: int = N_MACROS,
        n_targets: int = N_TARGETS,
        deterministic: bool = True,
        device: str = "cpu",
    ):
        self.model: Optional[Any] = None
        self.model_loaded = False
        self.deterministic = deterministic
        self.n_macros = n_macros
        self.n_targets = n_targets
        self.model_path: Optional[str] = _resolve_zip_path(model_path)
        self.device = str(device)

        if self.model_path is None:
            print(f"[PPO] Model not found: {model_path}")
            return
        try:
            _ensure_numpy_core_compat()
            _ensure_numpy_random_compat()
            from stable_baselines3 import PPO as SB3PPO
            # custom_objects: avoid unpickling policy/spaces/schedules from another Python/NumPy version.
            obs_space, action_space = _make_obs_action_spaces(2, self.n_macros, self.n_targets)
            custom_objects = {
                "observation_space": obs_space,
                "action_space": action_space,
                "clip_range": 0.2,
                "lr_schedule": lambda progress_remaining: 3e-4 * progress_remaining,
            }
            try:
                from rl.train_ppo import MaskedMultiInputPolicy
                custom_objects["policy_class"] = MaskedMultiInputPolicy
            except Exception:
                from stable_baselines3.common.policies import MultiInputActorCriticPolicy
                custom_objects["policy_class"] = MultiInputActorCriticPolicy
            self.model = SB3PPO.load(
                self.model_path,
                device=self.device,
                custom_objects=custom_objects,
            )
            self.model.policy.set_training_mode(False)
            self.model_loaded = True
            print(f"[PPO] Loaded: {self.model_path} (device={self.device})")
        except Exception as exc:
            print(f"[PPO] Failed to load: {exc}")
            import traceback
            traceback.print_exc()

    def predict(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """Return flat int64 action array [n_blue * 2] from batched obs (B=1)."""
        if not self.model_loaded or self.model is None:
            n_blue = obs["agent_mask"].shape[-1] if "agent_mask" in obs else 2
            return np.zeros((n_blue * 2,), dtype=np.int64)

        # SB3 expects obs without the batch dim for single-env predict
        single = {k: v[0] if v.ndim > 1 and v.shape[0] == 1 else v
                  for k, v in obs.items()}
        act, _ = self.model.predict(single, deterministic=self.deterministic)
        return np.asarray(act).reshape(-1).astype(np.int64)


# ---------------------------------------------------------------------------
# Renderer -- draws the state of a BatchedCTFCore[0] onto a pygame Surface
# ---------------------------------------------------------------------------
class CoreRenderer:
    """Pygame renderer that reads tensor state from a BatchedCTFCore (env 0)."""

    def __init__(self, core: BatchedCTFCore):
        self.core = core

    def _t(self, tensor: torch.Tensor) -> np.ndarray:
        return tensor[0].detach().cpu().numpy()

    def draw(self, surface: pg.Surface, rect: pg.Rect) -> None:
        c = self.core
        rows, cols = c.rows, c.cols
        cw = rect.width / max(1, cols)
        ch = rect.height / max(1, rows)

        # Background
        surface.fill((20, 22, 30), rect)

        # Zone halves
        mid_x = rect.left + int(cols / 2 * cw)
        blue_band = pg.Surface((mid_x - rect.left, rect.height), pg.SRCALPHA)
        blue_band.fill((15, 45, 120, 100))
        surface.blit(blue_band, (rect.left, rect.top))
        red_band = pg.Surface((rect.right - mid_x, rect.height), pg.SRCALPHA)
        red_band.fill((120, 45, 15, 100))
        surface.blit(red_band, (mid_x, rect.top))
        pg.draw.line(surface, (190, 190, 210),
                     (mid_x, rect.top), (mid_x, rect.bottom), 2)

        # Flags
        bf = self._t(c.blue_flag_pos)
        rf = self._t(c.red_flag_pos)
        bfh = self._t(c.blue_flag_home)
        rfh = self._t(c.red_flag_home)
        bf_taken = bool(c.red_carrying[0].any().item())
        rf_taken = bool(c.blue_carrying[0].any().item())

        self._draw_flag_zone(surface, rect, cw, ch, bfh, (90, 170, 250))
        self._draw_flag_zone(surface, rect, cw, ch, rfh, (250, 120, 70))
        if not bf_taken:
            self._draw_flag_icon(surface, rect, cw, ch, bf, (90, 170, 250))
        if not rf_taken:
            self._draw_flag_icon(surface, rect, cw, ch, rf, (250, 120, 70))

        # Mine pickups (spawn points; agents GRAB_MINE to get a charge)
        if getattr(c, "pickup_active", None) is not None:
            for i in range(c.pickup_x.shape[1]):
                if c.pickup_active[0, i].item():
                    px = c.pickup_x[0, i].item()
                    py = c.pickup_y[0, i].item()
                    self._draw_mine_pickup(surface, rect, cw, ch, px, py)

        # Placed mines (trigger when enemy steps in radius)
        if getattr(c, "blue_mine_active", None) is not None:
            for i in range(c.blue_mine_x.shape[1]):
                if c.blue_mine_active[0, i].item():
                    mx = c.blue_mine_x[0, i].item()
                    my = c.blue_mine_y[0, i].item()
                    self._draw_mine(surface, rect, cw, ch, mx, my, (90, 170, 250))
        if getattr(c, "red_mine_active", None) is not None:
            for i in range(c.red_mine_x.shape[1]):
                if c.red_mine_active[0, i].item():
                    mx = c.red_mine_x[0, i].item()
                    my = c.red_mine_y[0, i].item()
                    self._draw_mine(surface, rect, cw, ch, mx, my, (250, 120, 70))

        # Agents
        bx, by = self._t(c.blue_x), self._t(c.blue_y)
        bh = self._t(c.blue_heading)
        b_alive = self._t(c.blue_alive)
        b_tagged = self._t(c.blue_tagged)
        b_carry = self._t(c.blue_carrying)

        rx, ry = self._t(c.red_x), self._t(c.red_y)
        rh = self._t(c.red_heading)
        r_alive = self._t(c.red_alive)
        r_tagged = self._t(c.red_tagged)
        r_carry = self._t(c.red_carrying)

        for i in range(bx.shape[0]):
            if not b_alive[i]:
                continue
            self._draw_agent(surface, rect, cw, ch,
                             bx[i], by[i], bh[i],
                             body=(0, 180, 255),
                             tagged=bool(b_tagged[i]),
                             carrying=bool(b_carry[i]),
                             flag_clr=(250, 120, 70))
        for i in range(rx.shape[0]):
            if not r_alive[i]:
                continue
            self._draw_agent(surface, rect, cw, ch,
                             rx[i], ry[i], rh[i],
                             body=(255, 120, 40),
                             tagged=bool(r_tagged[i]),
                             carrying=bool(r_carry[i]),
                             flag_clr=(90, 170, 250))

    # ---- helpers ----

    @staticmethod
    def _draw_mine(surface: pg.Surface, rect: pg.Rect, cw: float, ch: float, x: float, y: float, color: Tuple[int, int, int]) -> None:
        cx = rect.left + (float(x) + 0.5) * cw
        cy = rect.top + (float(y) + 0.5) * ch
        r = int(0.35 * min(cw, ch))
        pg.draw.circle(surface, color, (int(cx), int(cy)), r)
        pg.draw.circle(surface, (240, 240, 240), (int(cx), int(cy)), r, width=1)

    @staticmethod
    def _draw_mine_pickup(surface: pg.Surface, rect: pg.Rect, cw: float, ch: float, x: float, y: float) -> None:
        """Draw a mine pickup spawn (diamond) so it’s distinct from placed mines."""
        cx = rect.left + (float(x) + 0.5) * cw
        cy = rect.top + (float(y) + 0.5) * ch
        s = int(0.4 * min(cw, ch))
        pts = [(int(cx), int(cy) - s), (int(cx) + s, int(cy)), (int(cx), int(cy) + s), (int(cx) - s, int(cy))]
        pg.draw.polygon(surface, (255, 220, 80), pts)
        pg.draw.polygon(surface, (200, 180, 40), pts, width=1)

    @staticmethod
    def _draw_flag_zone(surface, rect, cw, ch, pos, color):
        cx = rect.left + (float(pos[0]) + 0.5) * cw
        cy = rect.top + (float(pos[1]) + 0.5) * ch
        r = int(1.25 * min(cw, ch))
        zone = pg.Surface((rect.width, rect.height), pg.SRCALPHA)
        local = (int(cx - rect.left), int(cy - rect.top))
        pg.draw.circle(zone, (*color, 40), local, r)
        pg.draw.circle(zone, (*color, 110), local, r, width=2)
        surface.blit(zone, rect.topleft)

    @staticmethod
    def _draw_flag_icon(surface, rect, cw, ch, pos, color):
        cx = rect.left + (float(pos[0]) + 0.5) * cw
        cy = rect.top + (float(pos[1]) + 0.5) * ch
        sz = int(0.5 * min(cw, ch))
        pg.draw.rect(surface, color,
                     pg.Rect(int(cx - sz / 2), int(cy - sz / 2), sz, sz))

    @staticmethod
    def _draw_agent(surface, rect, cw, ch, x, y, heading, *,
                    body, tagged, carrying, flag_clr):
        cx = rect.left + (float(x) + 0.5) * cw
        cy = rect.top + (float(y) + 0.5) * ch
        tri = 0.45 * min(cw, ch)

        ux, uy = math.cos(float(heading)), math.sin(float(heading))
        lx, ly = -uy, ux
        tip = (int(cx + ux * tri), int(cy + uy * tri))
        left = (int(cx - ux * tri * 0.6 + lx * tri * 0.6),
                int(cy - uy * tri * 0.6 + ly * tri * 0.6))
        right = (int(cx - ux * tri * 0.6 - lx * tri * 0.6),
                 int(cy - uy * tri * 0.6 - ly * tri * 0.6))

        pg.draw.polygon(surface, body, (tip, left, right))
        if carrying:
            fs = int(tri * 0.5)
            pg.draw.rect(surface, flag_clr,
                         pg.Rect(tip[0] - fs // 2, tip[1] - fs // 2, fs, fs))
        if tagged:
            pg.draw.polygon(surface, (245, 245, 245), (tip, left, right), width=2)


# ---------------------------------------------------------------------------
# CTFViewer
# ---------------------------------------------------------------------------
class CTFViewer:
    def __init__(self, ppo_model_path: str = DEFAULT_PPO_MODEL_PATH,
                 device: str = "cpu"):
        paper_steps = 400
        cfg = GPUFieldConfig(
            n_envs=1,
            max_blue_agents=2,
            max_red_agents=2,
            device=device,
            max_decision_steps=paper_steps,
            stalemate_max_steps=paper_steps,
            rules_profile="OURS",
            score_limit=3,
        )
        self.cfg = cfg
        self.core = BatchedCTFCore(self.cfg)
        self.cfg.max_decision_steps = paper_steps
        self.cfg.stalemate_max_steps = paper_steps
        self.core.max_steps = paper_steps
        self.core.rules_profile = "OURS"
        print(f"[Viewer] Match length: {self.core.max_steps} steps (~200 s) | rules: OURS")
        try:
            self.core.set_phase("OP3")
            self.core.set_stress_schedule(STRESS_BY_PHASE)
            self.core.set_dynamics_config({"rules_profile": "OURS", "aquaticus_profile": True})

            # Match eval/training's scripted test opponent.
            opp = sample_batched_opponent_params(
                kind="SCRIPTED",
                key="OP3",
                phase="OP3",
                n_agents=cfg.max_red_agents,
                batch_size=cfg.n_envs,
                device=cfg.device,
            )
            dyn_cfg = {}
            if "deception_prob" in opp:
                dyn_cfg["deception_prob"] = opp["deception_prob"]
            if "speed_mult" in opp:
                dyn_cfg["speed_mult"] = opp["speed_mult"]
            if "attacker_style" in opp:
                dyn_cfg["attacker_style"] = opp["attacker_style"]
            if "defender_style" in opp:
                dyn_cfg["defender_style"] = opp["defender_style"]
            if "role_switch_prob" in opp:
                dyn_cfg["role_switch_prob"] = opp["role_switch_prob"]
            if dyn_cfg:
                self.core.set_dynamics_config(dyn_cfg)
        except Exception:
            # Fall back silently if curriculum/opponent modules are unavailable; core defaults will be used.
            pass
        self.core.reset_all()

        self.renderer = CoreRenderer(self.core)

        # PPO (load model on the same device as the core)
        self.ppo = PPOController(
            ppo_model_path,
            n_macros=cfg.n_macros,
            n_targets=cfg.n_targets,
            device=device,
        )

        if self.ppo.model_loaded:
            print("[Viewer] PPO ready. F3 toggles PPO / DEMO.")
            self.blue_mode: str = "PPO"
        else:
            print("[Viewer] No PPO model. Running DEMO (scripted blue).")
            print(f"[Viewer] Searched: {ppo_model_path}")
            self.blue_mode: str = "DEMO"
            self.core.blue_scripted = True

        # Pygame
        pg.init()
        self.size = (1024, 720)
        try:
            self.screen = pg.display.set_mode(self.size, pg.SCALED | pg.DOUBLEBUF, vsync=1)
        except TypeError:
            self.screen = pg.display.set_mode(self.size, pg.SCALED | pg.DOUBLEBUF)
        pg.display.set_caption("CTF Viewer (GPU core)")
        self.clock = pg.time.Clock()
        self.font = pg.font.SysFont(None, 26)
        self.bigfont = pg.font.SysFont(None, 48)
        self._last_info: Dict[str, Any] = {}

    # ---- stepping ----

    def _get_action(self) -> torch.Tensor:
        nb = self.cfg.max_blue_agents
        if self.blue_mode == "PPO" and self.ppo.model_loaded:
            obs_np = self.core.get_obs()
            act_np = self.ppo.predict(obs_np)
        else:
            # DEMO: zeros; core uses scripted blue when blue_scripted=True
            act_np = np.zeros((nb * 2,), dtype=np.int64)
        return torch.as_tensor(act_np, dtype=torch.int64, device=self.core.device).unsqueeze(0)

    def _step_env(self) -> Dict[str, Any]:
        action = self._get_action()
        obs, rew, term, trunc, infos = self.core.step(action)
        done = np.logical_or(term, trunc) if isinstance(term, np.ndarray) else (term | trunc)
        if isinstance(done, np.ndarray):
            any_done = done.any()
        else:
            any_done = bool(done.any())
        if any_done:
            # Full reset so flags and carrying state are clean (avoids wrong-flag-after-reset bug).
            self.core.reset_all()
        info = infos[0] if infos else {}
        self._last_info = info
        return info

    # ---- main loop ----

    def run(self) -> None:
        running = True
        while running:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    running = False
                elif event.type == pg.KEYDOWN:
                    self._handle_key(event)

            # Single simulation step per frame, locked to 30 FPS for predictable speed.
            self._step_env()
            self._draw()
            pg.display.flip()
            self.clock.tick(30)

        pg.quit()

    # ---- evaluation ----

    def evaluate(self, num_episodes: int = 100, headless: bool = False) -> Dict[str, Any]:
        print(f"[Eval] {num_episodes} episodes | mode={self.blue_mode} | headless={headless}")
        episodes: List[Dict[str, Any]] = []

        for ep in range(num_episodes):
            self.core.reset_all()
            ep_reward = 0.0
            steps = 0
            max_steps = self.cfg.max_decision_steps

            for _ in range(max_steps):
                action = self._get_action()
                obs, rew, term, trunc, infos = self.core.step(action)
                r = float(rew[0]) if isinstance(rew, np.ndarray) else float(rew[0].item())
                ep_reward += r
                steps += 1

                done_val = False
                if isinstance(term, np.ndarray):
                    done_val = bool(term[0]) or bool(trunc[0])
                else:
                    done_val = bool(term[0].item()) or bool(trunc[0].item())
                if done_val:
                    break

                if not headless:
                    for event in pg.event.get():
                        if event.type == pg.QUIT or (event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE):
                            print(f"[Eval] Stopped early at episode {ep + 1}")
                            episodes.append(self._episode_row(ep, steps, ep_reward))
                            return self._summarize(episodes)
                    self._draw()
                    pg.display.flip()
                    self.clock.tick(60)

            episodes.append(self._episode_row(ep, steps, ep_reward))
            if (ep + 1) % 10 == 0:
                wins = sum(1 for e in episodes if e["success"])
                print(f"[Eval] Ep {ep + 1}/{num_episodes} | WR {wins / len(episodes):.0%}")

        return self._summarize(episodes)

    def _episode_row(self, ep: int, steps: int, ep_reward: float) -> Dict[str, Any]:
        bs = int(self.core.blue_score[0].item())
        rs = int(self.core.red_score[0].item())
        return {
            "episode": ep + 1,
            "blue_score": bs,
            "red_score": rs,
            "success": 1 if bs > rs else 0,
            "steps": steps,
            "reward": ep_reward,
            "reward_per_step": ep_reward / max(1, steps),
        }

    def _summarize(self, episodes: List[Dict]) -> Dict[str, Any]:
        if not episodes:
            print("[Eval] No episodes.")
            return {}
        wins = sum(e["success"] for e in episodes)
        n = len(episodes)
        summary = {
            "num_episodes": n,
            "win_rate": wins / n,
            "wins": wins,
            "losses": n - wins,
            "mean_reward_per_step": float(np.mean([e["reward_per_step"] for e in episodes])),
            "mean_blue_score": float(np.mean([e["blue_score"] for e in episodes])),
            "mean_red_score": float(np.mean([e["red_score"] for e in episodes])),
        }

        csv_path = os.path.join(METRICS_DIR,
                                f"eval_{self.blue_mode.lower()}_{n}ep.csv")
        try:
            os.makedirs(METRICS_DIR, exist_ok=True)
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(episodes[0].keys()))
                w.writeheader()
                w.writerows(episodes)
            print(f"[Eval] CSV saved: {csv_path}")
        except Exception as exc:
            print(f"[Eval] CSV write failed: {exc}")

        print("\n" + "=" * 50)
        print("EVALUATION SUMMARY")
        print("=" * 50)
        print(f"  Episodes:     {n}")
        print(f"  Win Rate:     {summary['win_rate']:.0%} ({wins}W / {n - wins}L)")
        print(f"  Avg Blue:     {summary['mean_blue_score']:.2f}")
        print(f"  Avg Red:      {summary['mean_red_score']:.2f}")
        print(f"  Reward/Step:  {summary['mean_reward_per_step']:.4f}")
        print("=" * 50)

        summary["episodes"] = episodes
        return summary

    # ---- input ----

    def _rebuild_core(self, agents_per_team: int) -> None:
        """Recreate the core with a new number of agents per team."""
        agents = max(1, int(agents_per_team))
        self.cfg.max_blue_agents = agents
        self.cfg.max_red_agents = agents
        self.cfg.max_decision_steps = 400
        self.cfg.stalemate_max_steps = 400
        self.core = BatchedCTFCore(self.cfg)
        self.core.max_steps = 400
        self.core.rules_profile = "OURS"
        self.core.blue_scripted = (self.blue_mode == "DEMO")
        try:
            self.core.set_phase("OP3")
            self.core.set_stress_schedule(STRESS_BY_PHASE)
            self.core.set_dynamics_config({"rules_profile": "OURS", "aquaticus_profile": True})

            opp = sample_batched_opponent_params(
                kind="SCRIPTED",
                key="OP3",
                phase="OP3",
                n_agents=agents,
                batch_size=self.cfg.n_envs,
                device=self.cfg.device,
            )
            dyn_cfg: Dict[str, Any] = {}
            if "deception_prob" in opp:
                dyn_cfg["deception_prob"] = opp["deception_prob"]
            if "speed_mult" in opp:
                dyn_cfg["speed_mult"] = opp["speed_mult"]
            if "attacker_style" in opp:
                dyn_cfg["attacker_style"] = opp["attacker_style"]
            if "defender_style" in opp:
                dyn_cfg["defender_style"] = opp["defender_style"]
            if "role_switch_prob" in opp:
                dyn_cfg["role_switch_prob"] = opp["role_switch_prob"]
            if dyn_cfg:
                self.core.set_dynamics_config(dyn_cfg)
        except Exception:
            # If curriculum/opponent code is unavailable, fall back to defaults
            pass
        self.core.reset_all()
        self.renderer = CoreRenderer(self.core)
        print(f"[Viewer] Agents per team -> {agents} v {agents}")

    def _handle_key(self, event: Any) -> None:
        k = event.key
        if k == pg.K_ESCAPE:
            pg.event.post(pg.event.Event(pg.QUIT))
        elif k in (pg.K_F1, pg.K_r):
            self.core.reset_all()
            print("[Viewer] Reset")
        elif k == pg.K_F2:
            # Cycle 2v2 -> 3v3 -> 4v4 -> 2v2
            current = int(getattr(self.cfg, "max_blue_agents", 2))
            new_agents = {2: 3, 3: 4, 4: 2}.get(current, 3)
            self._rebuild_core(new_agents)
        elif k in (pg.K_2, pg.K_3, pg.K_4):
            # Direct switch: 2 -> 2v2, 3 -> 3v3, 4 -> 4v4
            new_agents = 2 if k == pg.K_2 else (3 if k == pg.K_3 else 4)
            if new_agents != int(getattr(self.cfg, "max_blue_agents", 2)):
                self._rebuild_core(new_agents)
        elif k == pg.K_F3:
            cycle = ["PPO", "DEMO"] if self.ppo.model_loaded else ["DEMO"]
            idx = cycle.index(self.blue_mode) if self.blue_mode in cycle else 0
            self.blue_mode = cycle[(idx + 1) % len(cycle)]
            self.core.blue_scripted = (self.blue_mode == "DEMO")
            print(f"[Viewer] Blue -> {self.blue_mode}")

    # ---- drawing ----

    def _draw(self) -> None:
        self.screen.fill((12, 12, 18))
        hud_h = 80
        field_rect = pg.Rect(20, hud_h + 10,
                             self.size[0] - 40, self.size[1] - hud_h - 30)
        self.renderer.draw(self.screen, field_rect)

        def txt(s: str, x: int, y: int, c=(230, 230, 240)):
            self.screen.blit(self.font.render(s, True, c), (x, y))

        mode_clr = {
            "PPO": (120, 255, 120),
            "DEMO": (120, 200, 255),
        }.get(self.blue_mode, (230, 230, 240))
        txt("F1: Reset | F2: 2v2/3v3/4v4 (cycle) | 2/3/4: set team size | F3: PPO/Demo | ESC: Quit",
            30, 10, (200, 200, 220))
        txt(f"Blue: {self.blue_mode} | {int(self.cfg.max_blue_agents)} v {int(self.cfg.max_red_agents)}",
            30, 36, mode_clr)

        if self.blue_mode == "PPO" and self.ppo.model_loaded:
            txt(f"Model: {os.path.basename(self.ppo.model_path or '')}",
                350, 36, (140, 240, 140))

        bs = int(self.core.blue_score[0].item())
        rs = int(self.core.red_score[0].item())
        step = int(self.core.step_count[0].item())
        txt(f"BLUE: {bs}", 30, 60, (100, 180, 255))
        txt(f"RED: {rs}", 180, 60, (255, 100, 100))
        txt(f"Step: {step}/{self.core.max_steps}", 330, 60, (200, 200, 230))
        # 3 min game: 0.1 s per step -> time remaining
        sec_left = max(0, (self.core.max_steps - step)) * 0.1
        txt(f"Time: {int(sec_left // 60)}:{int(sec_left % 60):02d}", 500, 60, (200, 200, 230))


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="CTF Viewer (GPU core, PPO)")
    parser.add_argument("--ppo-model", type=str, default=None,
                        help=f"PPO .zip path (default: {DEFAULT_PPO_MODEL_PATH})")
    parser.add_argument("--eval", type=int, metavar="N",
                        help="Run N evaluation episodes")
    parser.add_argument("--headless", action="store_true",
                        help="Headless evaluation (no display)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Torch device (cpu / cuda)")
    args = parser.parse_args()

    viewer = CTFViewer(
        ppo_model_path=args.ppo_model or DEFAULT_PPO_MODEL_PATH,
        device=args.device,
    )

    if args.eval is not None:
        viewer.evaluate(num_episodes=args.eval, headless=args.headless)
        if not args.headless:
            viewer.run()
    else:
        viewer.run()
