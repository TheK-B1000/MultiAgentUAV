"""
CTF Viewer -- renders a single BatchedCTFCore environment in pygame.

Blue team is controlled by a local custom PPO checkpoint (if found) or scripted DEMO.
Red team uses the scripted bot built into the GPU core.

No dependency on game_field.py, viewer_game_field.py, or policies.py.
"""

import os
import csv
import math
from typing import Optional, Tuple, Any, List, Dict

import numpy as np

import torch
import pygame as pg
from gymnasium import spaces

from game_field_gpu import (
    GPUFieldConfig,
    BatchedCTFCore,
    CNN_COLS,
    CNN_ROWS,
    NUM_CNN_CHANNELS,
    VEC_OBS_DIM,
)
from rl.stress_schedule import STRESS_BY_PHASE
from rl.custom_ppo import load_custom_ppo_policy, read_custom_ppo_metadata

# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
METRICS_DIR = os.path.join(_SCRIPT_DIR, "csv")
DEFAULT_PPO_MODEL_PATH = "checkpoints/8v8/final_ppo_latent_fixed_op3_8v8.zip"
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


def _team_size_tag(n_agents: int) -> str:
    n = max(1, int(n_agents))
    return f"{n}v{n}"


def _candidate_model_paths_for_agents(model_path: str, n_agents: int) -> List[str]:
    """Infer sibling checkpoints for the requested team size from the currently selected model path."""
    resolved = _resolve_zip_path(model_path)
    raw = resolved or model_path or ""
    if not raw:
        return []

    team_tag = _team_size_tag(n_agents)
    dirname = os.path.dirname(raw)
    basename = os.path.basename(raw)
    stem, ext = os.path.splitext(basename)
    ext = ext or ".zip"

    candidates: List[str] = []

    # Replace both directory tag and filename suffix when the checkpoint follows the repo naming scheme.
    for src_tag in ("2v2", "3v3", "4v4", "6v6", "8v8"):
        dir_variant = raw.replace(f"\\{src_tag}\\", f"\\{team_tag}\\").replace(f"/{src_tag}/", f"/{team_tag}/")
        file_variant = os.path.join(os.path.dirname(dir_variant), os.path.basename(dir_variant).replace(src_tag, team_tag))
        candidates.append(file_variant)

    # Generic filename replacement for custom names that still embed the team tag.
    for src_tag in ("2v2", "3v3", "4v4", "6v6", "8v8"):
        if src_tag in stem:
            candidates.append(os.path.join(dirname, stem.replace(src_tag, team_tag) + ext))

    # Final fallback: default latent PPO training run (FIXED_OPPONENT OP3).
    candidates.append(os.path.join(_SCRIPT_DIR, "checkpoints", team_tag, f"final_ppo_latent_fixed_op3_{team_tag}{ext}"))

    # Deduplicate while preserving order.
    seen = set()
    ordered: List[str] = []
    for c in candidates:
        if not c or c in seen:
            continue
        seen.add(c)
        ordered.append(c)
    return ordered


def _make_obs_action_spaces(n_blue: int, n_macros: int = N_MACROS, n_targets: int = N_TARGETS):
    """Build observation and action spaces for GPU CTF custom PPO inference."""
    obs_space = spaces.Dict(
        {
            "grid": spaces.Box(low=0.0, high=1.0, shape=(n_blue, NUM_CNN_CHANNELS, CNN_ROWS, CNN_COLS), dtype=np.float32),
            "vec": spaces.Box(low=-1.0, high=1.0, shape=(n_blue, VEC_OBS_DIM), dtype=np.float32),
            "agent_mask": spaces.Box(low=0.0, high=1.0, shape=(n_blue,), dtype=np.float32),
            "mask": spaces.Box(low=0.0, high=1.0, shape=(n_blue * (n_macros + n_targets),), dtype=np.float32),
        }
    )
    action_space = spaces.MultiDiscrete([n_macros, n_targets] * n_blue)
    return obs_space, action_space


def _read_model_metadata(model_path: str) -> Dict[str, Any]:
    """Read saved custom PPO metadata so viewer/core shape matches the checkpoint."""
    resolved = _resolve_zip_path(model_path)
    if resolved is None:
        return {}

    meta: Dict[str, Any] = {"model_path": resolved}
    try:
        meta.update(read_custom_ppo_metadata(resolved))
        return meta
    except Exception:
        return meta


# ---------------------------------------------------------------------------
# PPO wrapper  (loads custom model, builds obs from core, returns flat action)
# ---------------------------------------------------------------------------
class PPOController:
    """
    Wraps a local PPO model trained on GPUCTFVecEnv. Given a
    BatchedCTFCore (B=1), produces a flat int64 action tensor each tick.
    """

    def __init__(
        self,
        model_path: str,
        n_blue: int,
        n_macros: int = N_MACROS,
        n_targets: int = N_TARGETS,
        deterministic: bool = True,
        device: str = "cpu",
        print_traceback: bool = False,
    ):
        self.model: Optional[Any] = None
        self.model_loaded = False
        self.deterministic = deterministic
        self.n_macros = n_macros
        self.n_targets = n_targets
        self.n_blue = int(n_blue)
        self.print_traceback = bool(print_traceback)
        self.model_meta = _read_model_metadata(model_path)
        self.model_path: Optional[str] = self.model_meta.get("model_path") or _resolve_zip_path(model_path)
        self.device = str(device)

        if self.model_path is None:
            print(f"[PPO] Model not found: {model_path}")
            return
        try:
            obs_space, action_space = _make_obs_action_spaces(self.n_blue, self.n_macros, self.n_targets)
            self.model = load_custom_ppo_policy(
                self.model_path,
                obs_space,
                action_space,
                device=self.device,
            )
            self.model_loaded = True
            print(f"[PPO] Loaded: {self.model_path} (device={self.device})")
        except Exception as exc:
            print(f"[PPO] Failed to load: {exc}")
            if self.print_traceback:
                import traceback
                traceback.print_exc()

    def predict(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """Return flat int64 action array [n_blue * 2] from batched obs (B=1)."""
        if not self.model_loaded or self.model is None:
            n_blue = obs["agent_mask"].shape[-1] if "agent_mask" in obs else 2
            return np.zeros((n_blue * 2,), dtype=np.int64)

        act, _ = self.model.predict(obs, deterministic=self.deterministic)
        return np.asarray(act).reshape(-1).astype(np.int64)

    def reset_strategy(self) -> None:
        if self.model is not None and hasattr(self.model, "reset_strategy"):
            self.model.reset_strategy()

    def strategy_info(self) -> Dict[str, Any]:
        if self.model is not None and hasattr(self.model, "strategy_info"):
            return dict(self.model.strategy_info())
        return {}


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
        """Draw a mine pickup spawn (diamond) so it is distinct from placed mines."""
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
                 device: str = "cpu",
                 deterministic: bool = True):
        self.device = str(device)
        self.ppo_model_path = str(ppo_model_path)
        self.deterministic = bool(deterministic)
        model_meta = _read_model_metadata(ppo_model_path)
        initial_agents = max(1, int(model_meta.get("n_blue", 4)))
        paper_steps = 400
        cfg = GPUFieldConfig(
            n_envs=1,
            max_blue_agents=initial_agents,
            max_red_agents=initial_agents,
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
            # Use the core's official opponent-selection path so every reset re-applies OP3.
            self.core.set_next_opponent("SCRIPTED", "OP3")
        except Exception:
            # Fall back silently if curriculum/opponent modules are unavailable; core defaults will be used.
            pass
        self.core.reset_all()

        self.renderer = CoreRenderer(self.core)

        # PPO (load model on the same device as the core)
        self.ppo = PPOController(
            ppo_model_path,
            n_blue=cfg.max_blue_agents,
            n_macros=cfg.n_macros,
            n_targets=cfg.n_targets,
            device=self.device,
            deterministic=self.deterministic,
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
        self._ppo_mismatch_warned = False

    def _try_reload_ppo_for_agents(self, agents_per_team: int) -> bool:
        agents = max(1, int(agents_per_team))
        for candidate in _candidate_model_paths_for_agents(self.ppo_model_path, agents):
            resolved = _resolve_zip_path(candidate)
            if resolved is None:
                continue
            meta = _read_model_metadata(resolved)
            model_agents = int(meta.get("n_blue", 0) or 0)
            if model_agents and model_agents != agents:
                continue
            controller = PPOController(
                resolved,
                n_blue=agents,
                n_macros=self.cfg.n_macros,
                n_targets=self.cfg.n_targets,
                device=self.device,
                deterministic=self.deterministic,
                print_traceback=False,
            )
            if controller.model_loaded:
                self.ppo = controller
                self.ppo_model_path = resolved
                self._ppo_mismatch_warned = False
                print(f"[Viewer] PPO model -> {agents}v{agents}: {resolved}")
                return True
        return False

    def _ppo_team_size_compatible(self, agents_per_team: Optional[int] = None) -> bool:
        agents = int(agents_per_team if agents_per_team is not None else self.cfg.max_blue_agents)
        return bool(self.ppo.model_loaded and self.ppo.n_blue == agents)

    def _set_demo_due_to_mismatch(self, agents_per_team: int) -> None:
        self.blue_mode = "DEMO"
        self.core.blue_scripted = True
        if not self._ppo_mismatch_warned:
            print(
                f"[Viewer] PPO model expects {self.ppo.n_blue}v{self.ppo.n_blue}; "
                f"switched to DEMO for {agents_per_team}v{agents_per_team}."
            )
            self._ppo_mismatch_warned = True

    # ---- stepping ----

    def _get_action(self) -> torch.Tensor:
        nb = self.cfg.max_blue_agents
        if self.blue_mode == "PPO" and self.ppo.model_loaded:
            if not self._ppo_team_size_compatible():
                self._set_demo_due_to_mismatch(nb)
                act_np = np.zeros((nb * 2,), dtype=np.int64)
                return torch.as_tensor(act_np, dtype=torch.int64, device=self.core.device).unsqueeze(0)
            obs_np = self.core.get_obs()
            obs_np["global_state"] = self.core.get_global_state()
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
            self.ppo.reset_strategy()
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
            self.ppo.reset_strategy()
            ep_reward = 0.0
            steps = 0
            max_steps = self.cfg.max_decision_steps
            strategy_state = self._new_strategy_eval_state()

            for _ in range(max_steps):
                action = self._get_action()
                self._record_strategy_eval_step(strategy_state)
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
                            episodes.append(self._episode_row(ep, steps, ep_reward, strategy_state))
                            return self._summarize(episodes)
                    self._draw()
                    pg.display.flip()
                    self.clock.tick(60)

            episodes.append(self._episode_row(ep, steps, ep_reward, strategy_state))
            if (ep + 1) % 10 == 0:
                wins = sum(1 for e in episodes if e["success"])
                print(f"[Eval] Ep {ep + 1}/{num_episodes} | WR {wins / len(episodes):.0%}")

        return self._summarize(episodes)

    def _new_strategy_eval_state(self) -> Dict[str, Any]:
        return {
            "counts": {},
            "prev": None,
            "switches": 0,
            "resamples": 0,
            "steps": 0,
            "entropy_sum": 0.0,
            "k": 0,
        }

    def _record_strategy_eval_step(self, state: Dict[str, Any]) -> None:
        if self.blue_mode != "PPO" or not self.ppo.model_loaded:
            return
        info = self.ppo.strategy_info()
        if "strategy" not in info:
            return
        strategy = int(info["strategy"])
        counts = state["counts"]
        counts[strategy] = int(counts.get(strategy, 0)) + 1
        prev = state.get("prev")
        if prev is not None and int(prev) != strategy:
            state["switches"] = int(state["switches"]) + 1
        state["prev"] = strategy
        if bool(info.get("strategy_resampled", False)):
            state["resamples"] = int(state["resamples"]) + 1
        state["steps"] = int(state["steps"]) + 1
        state["entropy_sum"] = float(state["entropy_sum"]) + float(info.get("strategy_entropy", 0.0))
        state["k"] = max(int(state.get("k", 0)), int(info.get("strategy_k", 0)))

    def _strategy_episode_fields(self, state: Dict[str, Any]) -> Dict[str, Any]:
        strategy_steps = int(state.get("steps", 0) or 0)
        if strategy_steps <= 0:
            return {}
        counts: Dict[int, int] = state["counts"]
        denom = float(max(1, strategy_steps))
        fields: Dict[str, Any] = {
            "strategy_switches": int(state.get("switches", 0)),
            "strategy_switch_rate": float(state.get("switches", 0)) / float(max(1, strategy_steps - 1)),
            "strategy_resamples": int(state.get("resamples", 0)),
            "strategy_resample_rate": float(state.get("resamples", 0)) / denom,
            "strategy_unique_count": len(counts),
            "strategy_entropy_mean": float(state.get("entropy_sum", 0.0)) / denom,
        }
        if counts:
            fields["strategy_dominant"] = max(counts.items(), key=lambda kv: kv[1])[0]
        for idx in range(int(state.get("k", 0) or 0)):
            fields[f"strategy_occupancy_{idx}"] = float(counts.get(idx, 0)) / denom
        return fields

    def _episode_row(
        self,
        ep: int,
        steps: int,
        ep_reward: float,
        strategy_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        bs = int(self.core.blue_score[0].item())
        rs = int(self.core.red_score[0].item())
        row = {
            "episode": ep + 1,
            "blue_score": bs,
            "red_score": rs,
            "success": 1 if bs > rs else 0,
            "steps": steps,
            "reward": ep_reward,
            "reward_per_step": ep_reward / max(1, steps),
        }
        if strategy_state is not None:
            row.update(self._strategy_episode_fields(strategy_state))
        return row

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
            self.core.set_next_opponent("SCRIPTED", "OP3")
        except Exception:
            # If curriculum/opponent code is unavailable, fall back to defaults
            pass
        self.core.reset_all()
        self.renderer = CoreRenderer(self.core)
        if not self._ppo_team_size_compatible(agents):
            if self._try_reload_ppo_for_agents(agents):
                self._ppo_mismatch_warned = False
                if self.blue_mode == "PPO":
                    self.core.blue_scripted = False
            elif self.blue_mode == "PPO":
                self._set_demo_due_to_mismatch(agents)
            else:
                self._ppo_mismatch_warned = False
        else:
            self._ppo_mismatch_warned = False
        print(f"[Viewer] Agents per team -> {agents} v {agents}")

    def _handle_key(self, event: Any) -> None:
        k = event.key
        if k == pg.K_ESCAPE:
            pg.event.post(pg.event.Event(pg.QUIT))
        elif k in (pg.K_F1, pg.K_r):
            self.core.reset_all()
            print("[Viewer] Reset")
        elif k == pg.K_F2:
            # Cycle 2v2 -> 3v3 -> 4v4 -> 8v8 -> 2v2
            current = int(getattr(self.cfg, "max_blue_agents", 2))
            new_agents = {2: 3, 3: 4, 4: 8, 8: 2}.get(current, 3)
            self._rebuild_core(new_agents)
        elif k in (pg.K_2, pg.K_3, pg.K_4, pg.K_8):
            # Direct switch: 2 -> 2v2, 3 -> 3v3, 4 -> 4v4, 8 -> 8v8
            new_agents = {pg.K_2: 2, pg.K_3: 3, pg.K_4: 4, pg.K_8: 8}[k]
            if new_agents != int(getattr(self.cfg, "max_blue_agents", 2)):
                self._rebuild_core(new_agents)
        elif k == pg.K_F3:
            cycle = ["PPO", "DEMO"] if self.ppo.model_loaded else ["DEMO"]
            idx = cycle.index(self.blue_mode) if self.blue_mode in cycle else 0
            requested = cycle[(idx + 1) % len(cycle)]
            if requested == "PPO" and not self._ppo_team_size_compatible():
                agents = int(self.cfg.max_blue_agents)
                if not self._try_reload_ppo_for_agents(agents):
                    print(
                        f"[Viewer] Cannot enable PPO at {agents}v{agents}; "
                        f"no matching PPO checkpoint was found."
                    )
                    self.blue_mode = "DEMO"
                    self.core.blue_scripted = True
                    return
                cycle = ["PPO", "DEMO"]
                requested = "PPO"
            else:
                self.blue_mode = requested
                self.core.blue_scripted = (self.blue_mode == "DEMO")
                if self.blue_mode == "PPO":
                    self._ppo_mismatch_warned = False
                print(f"[Viewer] Blue -> {self.blue_mode}")
        elif k == pg.K_F4:
            self.deterministic = not self.deterministic
            if self.ppo is not None:
                self.ppo.deterministic = self.deterministic
            mode = "deterministic" if self.deterministic else "stochastic"
            print(f"[Viewer] PPO inference -> {mode}")

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
        txt("F1: Reset | F2: 2v2/3v3/4v4/8v8 (cycle) | 2/3/4/8: set team size | F3: PPO/Demo | F4: det/stoch | ESC: Quit",
            30, 10, (200, 200, 220))
        txt(f"Blue: {self.blue_mode} | {int(self.cfg.max_blue_agents)} v {int(self.cfg.max_red_agents)}",
            30, 36, mode_clr)

        if self.blue_mode == "PPO" and self.ppo.model_loaded:
            infer_mode = "det" if self.deterministic else "stoch"
            txt(f"Model: {os.path.basename(self.ppo.model_path or '')} | {infer_mode}",
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
    parser.add_argument("--stochastic", action="store_true",
                        help="Use stochastic PPO actions instead of deterministic inference")
    args = parser.parse_args()

    viewer = CTFViewer(
        ppo_model_path=args.ppo_model or DEFAULT_PPO_MODEL_PATH,
        device=args.device,
        deterministic=not args.stochastic,
    )

    if args.eval is not None:
        viewer.evaluate(num_episodes=args.eval, headless=args.headless)
        if not args.headless:
            viewer.run()
    else:
        viewer.run()
