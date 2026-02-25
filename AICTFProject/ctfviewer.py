"""
CTF Viewer -- renders a single BatchedCTFCore environment in pygame.

Blue team is controlled by an SB3 PPO model (if found) or takes random
actions.  Red team uses the scripted bot built into the GPU core.

No dependency on game_field.py, viewer_game_field.py, or policies.py.
"""

import os
import sys
import csv
import math
from typing import Optional, Tuple, Any, List, Dict

import numpy as np

# NumPy 1.x compat shim for models saved under NumPy 2.x
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

import torch
import pygame as pg

from game_field_gpu import (
    GPUFieldConfig,
    BatchedCTFCore,
    CNN_COLS,
    CNN_ROWS,
    NUM_CNN_CHANNELS,
)

# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
METRICS_DIR = os.path.join(_SCRIPT_DIR, "metrics")
DEFAULT_PPO_MODEL_PATH = "rl/checkpoints_sb3/final_ppo_league_v3.zip"
N_MACROS = 5
N_TARGETS = 8


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
            from stable_baselines3 import PPO as SB3PPO
            # Load the model onto the requested device (CPU or GPU).
            self.model = SB3PPO.load(self.model_path, device=self.device)
            self.model.policy.set_training_mode(False)
            self.model_loaded = True
            print(f"[PPO] Loaded: {self.model_path} (device={self.device})")
        except Exception as exc:
            print(f"[PPO] Failed to load: {exc}")
            import traceback; traceback.print_exc()

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
        cfg = GPUFieldConfig(
            n_envs=1,
            max_blue_agents=2,
            max_red_agents=2,
            device=device,
        )
        self.core = BatchedCTFCore(cfg)
        self.cfg = cfg
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
            print("[Viewer] PPO ready. F3 toggles PPO / random.")
        else:
            print("[Viewer] No PPO model. Running random actions.")
            print(f"[Viewer] Searched: {ppo_model_path}")

        self.blue_mode: str = "PPO" if self.ppo.model_loaded else "RANDOM"

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

    # ---- stepping ----

    def _get_action(self) -> torch.Tensor:
        if self.blue_mode == "PPO" and self.ppo.model_loaded:
            obs_np = self.core.get_obs()
            act_np = self.ppo.predict(obs_np)
        else:
            nb = self.cfg.max_blue_agents
            act_np = np.stack([
                np.random.randint(0, self.cfg.n_macros, size=(nb,)),
                np.random.randint(0, self.cfg.n_targets, size=(nb,)),
            ], axis=-1).reshape(-1).astype(np.int64)
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
            if isinstance(done, np.ndarray):
                mask = torch.from_numpy(done).to(self.core.device)
            else:
                mask = done
            self.core.reset_indices(mask)
        info = infos[0] if infos else {}
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

    def _handle_key(self, event: Any) -> None:
        k = event.key
        if k == pg.K_ESCAPE:
            pg.event.post(pg.event.Event(pg.QUIT))
        elif k in (pg.K_F1, pg.K_r):
            self.core.reset_all()
            print("[Viewer] Reset")
        elif k == pg.K_F3:
            if self.blue_mode == "PPO" and self.ppo.model_loaded:
                self.blue_mode = "RANDOM"
            elif self.ppo.model_loaded:
                self.blue_mode = "PPO"
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

        mode_clr = (120, 255, 120) if self.blue_mode == "PPO" else (255, 255, 120)
        txt("F1/R: Reset | F3: Toggle PPO/Random | ESC: Quit",
            30, 10, (200, 200, 220))
        txt(f"Blue: {self.blue_mode} | 2 v 2", 30, 36, mode_clr)

        if self.blue_mode == "PPO" and self.ppo.model_loaded:
            txt(f"Model: {os.path.basename(self.ppo.model_path or '')}",
                350, 36, (140, 240, 140))

        bs = int(self.core.blue_score[0].item())
        rs = int(self.core.red_score[0].item())
        step = int(self.core.step_count[0].item())
        txt(f"BLUE: {bs}", 30, 60, (100, 180, 255))
        txt(f"RED: {rs}", 180, 60, (255, 100, 100))
        txt(f"Step: {step}/{self.core.max_steps}", 330, 60, (200, 200, 230))


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
