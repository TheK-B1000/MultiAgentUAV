# driver.py
import sys
import math
import pygame as pg
import torch
from typing import Optional, List

from game_field import GameField
from macro_actions import MacroAction
from rl_policy import ActorCriticNet  # Your trained model class


# -------------------------------------------------------------------------- #
# Learned Policy Wrapper (unchanged logic, just cleaned up)
# -------------------------------------------------------------------------- #
class LearnedPolicy:
    """Wraps ActorCriticNet so GameField can call policy.select_action(obs)"""
    def __init__(self, obs_dim: int, n_actions: int, model_path: str = "marl_policy.pth"):
        self.device = torch.device("cpu")
        self.net = ActorCriticNet(obs_dim=obs_dim, n_actions=n_actions).to(self.device)
        self.model_loaded = False

        try:
            state = torch.load(model_path, map_location=self.device)
            state_dict = state.get("model", state) if isinstance(state, dict) else state
            self.net.load_state_dict(state_dict)
            print(f"[LearnedPolicy] Loaded '{model_path}' successfully")
            self.model_loaded = True
        except Exception as e:
            print(f"[LearnedPolicy] Failed to load '{model_path}': {e}")

        self.net.eval()

    def select_action(self, obs: List[float]) -> tuple[int, Optional[tuple[int, int]]]:
        if not self.model_loaded:
            return int(MacroAction.GO_TO), None

        with torch.no_grad():
            x = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            action_out = self.net.act(x, deterministic=True)
            action_id = int(action_out["action"][0].item())
        return action_id, None


# -------------------------------------------------------------------------- #
# Main Driver
# -------------------------------------------------------------------------- #
class Driver:
    def __init__(self):
        # Game setup
        rows, cols = 30, 40
        grid = [[0] * cols for _ in range(rows)]
        self.game_field = GameField(grid)
        self.gm = self.game_field.getGameManager()

        # Pygame init
        pg.init()
        self.size = (1280, 720)
        self.screen = pg.display.set_mode(self.size, pg.RESIZABLE)
        pg.display.set_caption("Continuous CTF – MARL Environment (Float-Based)")
        self.clock = pg.time.Clock()

        # Fonts
        self.font_small = pg.font.SysFont("consolas", 18)
        self.font_med = pg.font.SysFont("consolas", 24)
        self.font_big = pg.font.SysFont("consolas", 48)

        # Input state
        self.input_active = False
        self.input_text = ""

        # Policy setup
        self.blue_heuristic = self.game_field.policies["blue"]
        self.blue_learned: Optional[LearnedPolicy] = None
        self.use_learned_blue = False

        # Try to load learned policy
        dummy_obs = self.game_field.build_observation(self.game_field.blue_agents[0]) if self.game_field.blue_agents else [0.0] * 37
        obs_dim = len(dummy_obs)
        n_actions = len(MacroAction)

        try:
            self.blue_learned = LearnedPolicy(obs_dim=obs_dim, n_actions=n_actions)
            if self.blue_learned.model_loaded:
                self.game_field.policies["blue"] = self.blue_learned
                self.use_learned_blue = True
                print("[Driver] Blue team using LEARNED policy")
            else:
                print("[Driver] Blue team using HEURISTIC policy (model not loaded)")
        except Exception as e:
            print(f"[Driver] RL setup failed: {e}")
            print("[Driver] Blue team using HEURISTIC policy")

    # ---------------------------------------------------------------------- #
    # Main loop
    # ---------------------------------------------------------------------- #
    def run(self):
        running = True
        while running:
            dt_ms = self.clock.tick(60)
            dt_sec = dt_ms / 1000.0

            for event in pg.event.get():
                if event.type == pg.QUIT:
                    running = False
                elif event.type == pg.VIDEORESIZE:
                    self.size = event.size
                    self.screen = pg.display.set_mode(self.size, pg.RESIZABLE)
                elif event.type == pg.KEYDOWN:
                    if self.input_active:
                        self.handle_input_key(event)
                    else:
                        self.handle_hotkey(event)

            self.game_field.update(dt_sec)
            self.draw()
            pg.display.flip()

        pg.quit()
        sys.exit()

    # ---------------------------------------------------------------------- #
    # Hotkeys
    # ---------------------------------------------------------------------- #
    def handle_hotkey(self, event):
        k = event.key

        if k == pg.K_F1:
            self.gm.reset_game(reset_scores=True)
            self.game_field.reset_default()

        elif k == pg.K_r:
            self.game_field.reset_default()

        elif k == pg.K_F2:
            self.input_active = True
            self.input_text = ""

        elif k == pg.K_F3:
            self.gm.reset_game(reset_scores=True)
            self.game_field.runTestCase3()

        elif k == pg.K_F4:
            if not self.blue_learned or not self.blue_learned.model_loaded:
                print("[Driver] No learned model available!")
                return
            self.use_learned_blue = not self.use_learned_blue
            if self.use_learned_blue:
                self.game_field.policies["blue"] = self.blue_learned
                print("[Driver] Blue → LEARNED policy")
            else:
                self.game_field.policies["blue"] = self.blue_heuristic
                print("[Driver] Blue → HEURISTIC policy")

        elif k == pg.K_F5:
            self.game_field.debug_draw_ranges = not self.game_field.debug_draw_ranges

        elif k == pg.K_F6:
            self.game_field.debug_draw_mine_ranges = not self.game_field.debug_draw_mine_ranges

        elif k == pg.K_ESCAPE:
            pg.event.post(pg.event.Event(pg.QUIT))

    def handle_input_key(self, event):
        if event.key == pg.K_RETURN:
            try:
                n = int(self.input_text or "8")
                n = max(1, min(100, n))
                self.gm.reset_game(reset_scores=True)
                self.game_field.runTestCase2(n, self.gm)
                print(f"[Driver] Spawned {n} agents per team")
            except:
                pass
            self.input_active = False
            self.input_text = ""
        elif event.key == pg.K_ESCAPE:
            self.input_active = False
            self.input_text = ""
        elif event.key == pg.K_BACKSPACE:
            self.input_text = self.input_text[:-1]
        elif event.unicode.isdigit() and len(self.input_text) < 3:
            self.input_text += event.unicode

    # ---------------------------------------------------------------------- #
    # Rendering
    # ---------------------------------------------------------------------- #
    def draw(self):
        self.screen.fill((15, 15, 25))

        # Layout
        hud_height = 110
        margin = 20
        field_rect = pg.Rect(
            margin,
            hud_height + 10,
            self.size[0] - 2 * margin,
            self.size[1] - hud_height - 40
        )

        # Draw game field
        self.game_field.draw(self.screen, field_rect)

        # HUD
        self.draw_hud(hud_height)

        # Agent count input overlay
        if self.input_active:
            self.draw_input_overlay()

    def draw_hud(self, hud_h: int):
        def txt(text: str, x: int, y: int, color=(230, 230, 240), font=None):
            f = font or self.font_small
            img = f.render(text, True, color)
            self.screen.blit(img, (x, y))

        # Background semi-transparent bar
        bar = pg.Surface((self.size[0], hud_h))
        bar.set_alpha(180)
        bar.fill((20, 20, 40))
        self.screen.blit(bar, (0, 0))

        # Title + hotkeys
        txt("CONTINUOUS CTF – MARL ENVIRONMENT", 25, 12, (100, 200, 255), self.font_med)
        hotkeys = "F1 Reset • F2 Agent Count • F3 Swap Zones • R SoftReset • F4 Toggle RL • F5/F6 Debug • ESC Quit"
        txt(hotkeys, 25, 38, (180, 180, 200))

        # Policy status
        policy_name = "RL POLICY" if self.use_learned_blue else "HEURISTIC"
        policy_color = (100, 255, 100) if self.use_learned_blue else (255, 220, 100)
        txt(f"BLUE: {policy_name}", 25, 68, policy_color, self.font_med)

        # Scores & timer
        score_y = 68
        txt(f"SCORE → BLUE {self.gm.blue_score}", 300, score_y, (100, 180, 255), self.font_med)
        txt(f"RED {self.gm.red_score}", 500, score_y, (255, 100, 100), self.font_med)
        time_str = f"TIME: {max(0, int(self.gm.current_time))}s"
        txt(time_str, 700, score_y, (220, 220, 255), self.font_med)

        # Model info
        if self.blue_learned and self.blue_learned.model_loaded:
            txt("Model: marl_policy.pth", self.size[0] - 320, 15, (100, 230, 100))
            txt(f"FPS: {self.clock.get_fps():.1f}", self.size[0] - 320, 40, (200, 255, 200))

        # Debug indicators
        if self.game_field.debug_draw_ranges:
            txt("SUPPRESSION RANGES", self.size[0] - 320, 68, (100, 200, 255))
        if self.game_field.debug_draw_mine_ranges:
            txt("MINE RANGES", self.size[0] - 320, 92, (255, 150, 100))

    def draw_input_overlay(self):
        overlay = pg.Surface(self.size, pg.SRCALPHA)
        overlay.fill((0, 0, 0, 200))
        self.screen.blit(overlay, (0, 0))

        box = pg.Rect(0, 0, 480, 220)
        box.center = self.screen.get_rect().center

        pg.draw.rect(self.screen, (30, 30, 60), box, border_radius=15)
        pg.draw.rect(self.screen, (100, 180, 255), box, width=5, border_radius=15)

        title = self.font_big.render("Set Agent Count per Team", True, (255, 255, 255))
        entry = self.font_big.render(self.input_text + ("_" if int(pg.time.get_ticks() / 500) % 2 else ""), True, (120, 220, 255))
        hint = self.font_med.render("Enter number (1–100) → Press Enter", True, (200, 200, 220))

        self.screen.blit(title, title.get_rect(center=(box.centerx, box.centery - 60)))
        self.screen.blit(entry, entry.get_rect(center=(box.centerx, box.centery)))
        self.screen.blit(hint, hint.get_rect(center=(box.centerx, box.centery + 60)))


# -------------------------------------------------------------------------- #
# Entry point
# -------------------------------------------------------------------------- #
if __name__ == "__main__":
    Driver().run()