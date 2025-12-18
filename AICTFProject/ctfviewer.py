import os
import sys
import pygame as pg
import torch
from typing import Optional, Tuple, Any, List, Dict

from viewer_game_field import ViewerGameField
from macro_actions import MacroAction
from rl_policy import ActorCriticNet
from policies import OP3RedPolicy

# ----------------------------
# MODEL PATHS (edit these)
# ----------------------------
DEFAULT_PPO_MODEL_PATH = "checkpoints/research_model1.pth"
DEFAULT_MAPPO_MODEL_PATH = "checkpoints/research_mappo_model1.pth"

# IMPORTANT: keep this order consistent with training
USED_MACROS = [
    MacroAction.GO_TO,
    MacroAction.GRAB_MINE,
    MacroAction.GET_FLAG,
    MacroAction.PLACE_MINE,
    MacroAction.GO_HOME,
]


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def _extract_state_dict(loaded: Any) -> Dict[str, torch.Tensor]:
    """
    Supports:
      - plain state_dict
      - {"model": state_dict}
      - {"state_dict": state_dict}
    """
    if isinstance(loaded, dict):
        if "model" in loaded and isinstance(loaded["model"], dict):
            return loaded["model"]
        if "state_dict" in loaded and isinstance(loaded["state_dict"], dict):
            return loaded["state_dict"]
        # might already be a state_dict
        if any(isinstance(k, str) and "." in k for k in loaded.keys()):
            return loaded
        return loaded
    raise ValueError("Unrecognized checkpoint format (not a dict/state_dict).")


class LearnedPolicy:
    """
    Wrapper around ActorCriticNet so it can be plugged into GameField.policies["blue"].

    IMPORTANT:
      - returns (macro_idx, target_idx) NOT (macro_value, (x,y)).
        GameField.apply_macro_action should resolve target_idx -> (x,y).
    """

    def __init__(self, model_path: str, env: ViewerGameField):
        self.device = torch.device("cpu")
        self.model_path = model_path
        self.model_loaded = False

        if not getattr(env, "blue_agents", None):
            raise RuntimeError("ViewerGameField has no blue agents; cannot infer obs shape.")

        dummy_obs = env.build_observation(env.blue_agents[0])
        C = len(dummy_obs)
        H = len(dummy_obs[0])
        W = len(dummy_obs[0][0])

        n_targets = _safe_int(getattr(env, "num_macro_targets", 0), 0)
        if n_targets <= 0:
            n_targets = 50

        n_agents = _safe_int(getattr(env, "agents_per_team", 2), 2)

        self.net = ActorCriticNet(
            n_macros=len(USED_MACROS),
            n_targets=n_targets,
            in_channels=C,
            height=H,
            width=W,
            n_agents=n_agents,
        ).to(self.device)

        if not model_path or (not os.path.exists(model_path)):
            print(f"[CTFViewer] Model path not found: {model_path}")
            self.net.eval()
            return

        try:
            loaded = torch.load(model_path, map_location=self.device)
            state_dict = _extract_state_dict(loaded)
            self.net.load_state_dict(state_dict, strict=True)
            self.model_loaded = True
            print(f"[CTFViewer] Loaded model from: {model_path}")
        except Exception as e:
            print(f"[CTFViewer] Failed to load model '{model_path}': {e}")
            self.model_loaded = False

        self.net.eval()

    def __call__(self, agent, game_field):
        obs = game_field.build_observation(agent)
        macro_idx, target_idx = self.select_action(obs, agent, game_field)
        return (int(macro_idx), target_idx)

    def select_action(self, obs, agent, game_field) -> Tuple[int, Optional[int]]:
        if not self.model_loaded:
            return 0, None  # default to GO_TO idx 0

        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
            if obs_tensor.dim() == 3:
                obs_tensor = obs_tensor.unsqueeze(0)

            out = self.net.act(
                obs_tensor,
                agent=agent,
                game_field=game_field,
                deterministic=True,
            )

            macro_t = out.get("macro_action", out.get("action", out.get("macro")))
            if macro_t is None:
                return 0, None

            macro_idx = int(macro_t.reshape(-1)[0].item()) if torch.is_tensor(macro_t) else int(macro_t)
            macro_idx = max(0, min(len(USED_MACROS) - 1, macro_idx))

            target_idx = None
            if "target_action" in out:
                tgt_t = out["target_action"]
                target_idx = int(tgt_t.reshape(-1)[0].item()) if torch.is_tensor(tgt_t) else int(tgt_t)

            return macro_idx, target_idx


class CTFViewer:
    def __init__(
        self,
        ppo_model_path: str = DEFAULT_PPO_MODEL_PATH,
        mappo_model_path: str = DEFAULT_MAPPO_MODEL_PATH,
    ):
        rows, cols = 20, 20
        grid = [[0] * cols for _ in range(rows)]

        self.game_field = ViewerGameField(grid)
        self.game_manager = self.game_field.getGameManager()

        # Ensure viewer uses internal policies
        if hasattr(self.game_field, "use_internal_policies"):
            self.game_field.use_internal_policies = True
        if hasattr(self.game_field, "set_external_control"):
            try:
                self.game_field.set_external_control("blue", False)
                self.game_field.set_external_control("red", False)
            except Exception:
                pass

        # ------------- Pygame setup -------------
        pg.init()
        self.size = (1024, 720)
        try:
            self.screen = pg.display.set_mode(self.size, pg.SCALED | pg.DOUBLEBUF, vsync=1)
        except TypeError:
            self.screen = pg.display.set_mode(self.size, pg.SCALED | pg.DOUBLEBUF)
        pg.display.set_caption("UAV CTF Viewer | Blue: OP3/PPO/MAPPO vs Red: OP3 | CNN 7×20×20")
        self.clock = pg.time.Clock()
        self.font = pg.font.SysFont(None, 26)
        self.bigfont = pg.font.SysFont(None, 48)

        self.input_active = False
        self.input_text = ""

        # Sanity
        if getattr(self.game_field, "blue_agents", None):
            dummy_obs = self.game_field.build_observation(self.game_field.blue_agents[0])
            try:
                C = len(dummy_obs)
                H = len(dummy_obs[0])
                W = len(dummy_obs[0][0])
                print(f"[CTFViewer] Detected CNN obs shape: C={C}, H={H}, W={W}")
                print(f"[CTFViewer] num_macro_targets: {_safe_int(getattr(self.game_field, 'num_macro_targets', 0), 0)}")
            except Exception:
                print("[CTFViewer] Could not infer CNN obs shape cleanly.")
        else:
            print("[CTFViewer] No agents spawned; cannot infer obs shape.")

        # ---- Policies ----
        if hasattr(self.game_field, "policies") and isinstance(self.game_field.policies, dict):
            self.game_field.policies["red"] = OP3RedPolicy("red")

        self.blue_op3_baseline = OP3RedPolicy("blue")

        self.blue_ppo_policy: Optional[LearnedPolicy] = LearnedPolicy(ppo_model_path, self.game_field)
        self.blue_mappo_policy: Optional[LearnedPolicy] = LearnedPolicy(mappo_model_path, self.game_field)

        self.blue_mode: str = "OP3"
        self._apply_blue_mode(self.blue_mode)
        self._reset_op3_policies()

    def _available_modes(self) -> List[str]:
        modes = ["OP3"]
        if self.blue_ppo_policy and self.blue_ppo_policy.model_loaded:
            modes.append("PPO")
        if self.blue_mappo_policy and self.blue_mappo_policy.model_loaded:
            modes.append("MAPPO")
        return modes

    def _apply_blue_mode(self, mode: str) -> None:
        if not (hasattr(self.game_field, "policies") and isinstance(self.game_field.policies, dict)):
            return

        if mode == "PPO" and self.blue_ppo_policy and self.blue_ppo_policy.model_loaded:
            self.game_field.policies["blue"] = self.blue_ppo_policy
            self.blue_mode = "PPO"
            print("[CTFViewer] Blue → PPO model")
        elif mode == "MAPPO" and self.blue_mappo_policy and self.blue_mappo_policy.model_loaded:
            self.game_field.policies["blue"] = self.blue_mappo_policy
            self.blue_mode = "MAPPO"
            print("[CTFViewer] Blue → MAPPO model")
        else:
            self.game_field.policies["blue"] = self.blue_op3_baseline
            self.blue_mode = "OP3"
            print("[CTFViewer] Blue → OP3 baseline")

    def _cycle_blue_mode(self) -> None:
        modes = self._available_modes()
        i = modes.index(self.blue_mode) if self.blue_mode in modes else 0
        nxt = modes[(i + 1) % len(modes)]
        self._apply_blue_mode(nxt)
        if self.blue_mode == "OP3":
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

        max_frame_dt = 1.0 / 30.0  # cap big hitches
        max_substeps = 5  # avoid spiral-of-death

        while running:
            frame_dt = self.clock.tick_busy_loop(120) / 1000.0  # steadier pacing
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
                self.game_field.update(fixed_dt)  # ViewerGameField.update snapshots prev/curr
                acc -= fixed_dt
                steps += 1

            # If we fell behind badly, drop the remainder so we don't stutter for seconds
            if steps == max_substeps:
                acc = 0.0

            alpha = acc / fixed_dt  # 0..1
            self.draw(alpha=alpha)
            pg.display.flip()

        pg.quit()
        sys.exit()

    # ----------------------------
    # Input handling
    # ----------------------------
    def handle_main_key(self, event):
        k = event.key
        if k == pg.K_F1:
            self.game_field.agents_per_team = 2
            self.game_manager.reset_game(reset_scores=True)
            self.game_field.reset_default()
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
        mode_color = (255, 255, 120) if mode == "OP3" else (120, 255, 120)

        txt(
            "F1: Full Reset | F2: Set Agents | F3: Cycle Blue (OP3/PPO/MAPPO) | F4/F5: Debug | R: Reset",
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
        if self.blue_ppo_policy and self.blue_ppo_policy.model_loaded:
            txt(f"PPO: {self.blue_ppo_policy.model_path}", right_x, 45, (140, 240, 140))
        else:
            txt("PPO: (not loaded)", right_x, 45, (180, 180, 180))

        if self.blue_mappo_policy and self.blue_mappo_policy.model_loaded:
            txt(f"MAPPO: {self.blue_mappo_policy.model_path}", right_x, 70, (140, 240, 140))
        else:
            txt("MAPPO: (not loaded)", right_x, 70, (180, 180, 180))

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
    CTFViewer().run()
