import sys
import pygame as pg
import torch
from typing import Optional, Tuple, Any, List

from game_field import GameField, MacroAction
from rl_policy import ActorCriticNet


class LearnedPolicy:
    """
    Wraps ActorCriticNet so GameField.decide() can call:
        policy.select_action(obs) -> (action_id, param)
    """
    def __init__(self, obs_dim: int, n_actions: int, model_path: Optional[str] = None):
        self.device = torch.device("cpu")  # keep simple for the driver

        # n_actions should match len(MacroAction)
        self.net = ActorCriticNet(
            obs_dim=obs_dim,
            n_actions=n_actions,
        ).to(self.device)

        self.model_loaded = False
        if model_path:
            try:
                state = torch.load(model_path, map_location=self.device)
                # support { 'model': state_dict } or plain state_dict
                state_dict = state.get("model", state) if isinstance(state, dict) else state
                self.net.load_state_dict(state_dict)
                print(f"[LearnedPolicy] Loaded model from {model_path}")
                self.model_loaded = True
            except Exception as e:
                print(f"[LearnedPolicy] Failed to load model '{model_path}': {e}")
                self.model_loaded = False

        self.net.eval()

    def select_action(self, obs: List[float]) -> Tuple[int, Optional[Tuple[int, int]]]:
        """
        Called by GameField.decide(agent):

            obs = game_field.build_observation(agent)
            action_id, param = policy.select_action(obs)

        We’re only using discrete macro-actions for now, so param is None.
        """
        if not self.model_loaded:
            # Fallback: "simple" default macro-action
            fallback_action = int(MacroAction.GO_TO)
            return fallback_action, None

        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
            if obs_tensor.dim() == 1:
                obs_tensor = obs_tensor.unsqueeze(0)  # [1, obs_dim]

            out = self.net.act(obs_tensor, deterministic=True)
            # out is a dict: { "action": Tensor, "log_prob": Tensor, "value": Tensor }
            action = out["action"]
            if isinstance(action, torch.Tensor):
                # Shape [1], so take element 0
                action = int(action[0].item())

            return action, None


class Driver:
    def __init__(self):
        rows, cols = 30, 40
        grid = [[0] * cols for _ in range(rows)]
        self.gameField = GameField(grid)
        self.gameManager = self.gameField.getGameManager()

        # ------------- Pygame setup -------------
        pg.init()
        self.size = (1024, 720)
        self.screen = pg.display.set_mode(self.size)
        pg.display.set_caption("UAV CTF – Paper-Accurate MARL Environment")
        self.clock = pg.time.Clock()
        self.font = pg.font.SysFont(None, 26)
        self.bigfont = pg.font.SysFont(None, 48)

        self.input_active = False
        self.input_text = ""

        # ------------- RL Policy Setup -------------
        if self.gameField.blue_agents:
            dummy_obs = self.gameField.build_observation(self.gameField.blue_agents[0])
        else:
            dummy_obs = [0.0] * 37  # fallback, but your build_observation returns 37

        obs_dim = len(dummy_obs)
        n_actions = len(MacroAction)

        # Keep references to the original heuristic policies
        self.blue_heuristic = self.gameField.policies["blue"]
        self.red_heuristic = self.gameField.policies["red"]

        self.blue_learned_policy: Optional[LearnedPolicy] = None
        self.use_learned_blue: bool = False

        try:
            self.blue_learned_policy = LearnedPolicy(
                obs_dim=obs_dim,
                n_actions=n_actions,
                model_path="marl_policy.pth",
            )
            if self.blue_learned_policy.model_loaded:
                self.gameField.policies["blue"] = self.blue_learned_policy
                self.use_learned_blue = True
                print("[Driver] Blue team using LEARNED policy (marl_policy.pth)")
            else:
                print("[Driver] Blue team using HEURISTIC policy (no valid model)")
        except Exception as e:
            print(f"[Driver] RL setup failed: {e}")
            print("[Driver] Blue team using HEURISTIC policy")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self):
        running = True
        while running:
            dt_ms = self.clock.tick(60)
            dt_sec = dt_ms / 1000.0

            for event in pg.event.get():
                if event.type == pg.QUIT:
                    running = False
                elif event.type == pg.KEYDOWN:
                    if self.input_active:
                        self.handle_input_key(event)
                    else:
                        self.handle_main_key(event)

            self.gameField.update(dt_sec)
            self.draw()
            pg.display.flip()

        pg.quit()
        sys.exit()

    # ------------------------------------------------------------------
    # Input handling
    # ------------------------------------------------------------------
    def handle_main_key(self, event):
        k = event.key
        if k == pg.K_F1:
            self.gameManager.reset_game(reset_scores=True)
            self.gameField.reset_default()
        elif k == pg.K_F2:
            self.input_active = True
            self.input_text = ""
        elif k == pg.K_F3:
            self.gameManager.reset_game(reset_scores=True)
            self.gameField.runTestCase3()
        elif k == pg.K_r:
            self.gameField.reset_default()
        elif k == pg.K_F4:
            # Toggle between heuristic and learned policy for blue
            if not self.blue_learned_policy or not self.blue_learned_policy.model_loaded:
                print("[Driver] No learned model available!")
                return
            self.use_learned_blue = not self.use_learned_blue
            if self.use_learned_blue:
                self.gameField.policies["blue"] = self.blue_learned_policy
                print("[Driver] Blue → LEARNED policy")
            else:
                self.gameField.policies["blue"] = self.blue_heuristic
                print("[Driver] Blue → HEURISTIC policy")
        elif k == pg.K_F5:
            self.gameField.debug_draw_ranges = not self.gameField.debug_draw_ranges
        elif k == pg.K_F6:
            self.gameField.debug_draw_mine_ranges = not self.gameField.debug_draw_mine_ranges
        elif k == pg.K_ESCAPE:
            pg.event.post(pg.event.Event(pg.QUIT))

    def handle_input_key(self, event):
        if event.key == pg.K_RETURN:
            try:
                n = int(self.input_text or "8")
                n = max(1, min(100, n))
                self.gameManager.reset_game(reset_scores=True)
                self.gameField.runTestCase2(n, self.gameManager)
            except Exception:
                pass
            self.input_active = False
        elif event.key == pg.K_ESCAPE:
            self.input_active = False
        elif event.key == pg.K_BACKSPACE:
            self.input_text = self.input_text[:-1]
        elif event.unicode.isdigit():
            if len(self.input_text) < 3:
                self.input_text += event.unicode

    # ------------------------------------------------------------------
    # Drawing / HUD
    # ------------------------------------------------------------------
    def draw(self):
        self.screen.fill((12, 12, 18))

        hud_h = 100  # Increased HUD height to fit two rows
        field_rect = pg.Rect(20, hud_h + 10, self.size[0] - 40, self.size[1] - hud_h - 30)
        self.gameField.draw(self.screen, field_rect)

        def txt(text, x, y, color=(230, 230, 240)):
            img = self.font.render(text, True, color)
            self.screen.blit(img, (x, y))

        gm = self.gameManager
        policy_name = "RL" if self.use_learned_blue else "HEURISTIC"
        policy_color = (100, 255, 100) if self.use_learned_blue else (255, 255, 100)

        # Top row: menu / help
        menu_text = (
            "F1: Reset | F2: Agent Count | F3: SwapZones | "
            "R: SoftReset | F4: ToggleRL | F5/F6: DebugDraw"
        )
        txt(menu_text, 30, 15, (200, 200, 220))

        status_text = f"Blue Policy: {policy_name}"
        txt(status_text, 30, 45, policy_color)

        # Second row: scores & time
        score_time_y = 78
        txt(f"BLUE: {gm.blue_score}", 30, score_time_y, (100, 180, 255))
        txt(f"RED: {gm.red_score}", 200, score_time_y, (255, 100, 100))
        txt(f"Time: {int(gm.current_time)}s", 380, score_time_y, (220, 220, 255))

        # Model status
        if self.blue_learned_policy and self.blue_learned_policy.model_loaded:
            txt("Model: marl_policy.pth", self.size[0] - 300, 15, (100, 220, 100))

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
            hint = self.font.render("Press Enter to confirm • Esc to cancel", True, (200, 200, 200))

            self.screen.blit(title, title.get_rect(center=(box.centerx, box.centery - 50)))
            self.screen.blit(entry, entry.get_rect(center=box.center))
            self.screen.blit(hint, hint.get_rect(center=(box.centerx, box.centery + 60)))

if __name__ == "__main__":
    Driver().run()
