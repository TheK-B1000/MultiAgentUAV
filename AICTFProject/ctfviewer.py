import sys
import pygame as pg
import torch
from typing import Optional, Tuple, Any, List

from game_field import GameField
from macro_actions import MacroAction
from rl_policy import ActorCriticNet
from policies import OP3RedPolicy


class LearnedPolicy:
    def __init__(self, obs_dim: int, n_actions: int, model_path: Optional[str] = None):
        self.device = torch.device("cpu")  # keep simple for the viewer

        # n_actions should match len(MacroAction)
        self.net = ActorCriticNet(
            obs_dim=obs_dim,
            n_actions=n_actions,
        ).to(self.device)

        self.model_loaded = False

        if model_path is None:
            model_path = "checkpoints/ctf_ppo_final.pth"

        try:
            state = torch.load(model_path, map_location=self.device)
            # support { 'model': state_dict } or plain state_dict
            if isinstance(state, dict) and "model" in state:
                state_dict = state["model"]
            else:
                state_dict = state
            self.net.load_state_dict(state_dict)
            print(f"[CTFViewer] Loaded model from {model_path}")
            self.model_loaded = True
        except Exception as e:
            print(f"[CTFViewer] Failed to load model '{model_path}': {e}")
            self.model_loaded = False

        self.net.eval()

    def __call__(self, agent, game_field) -> int:
        # Build observation exactly like your trainer did
        obs = game_field.build_observation(agent)
        action_id, _ = self.select_action(obs)
        return int(action_id)

    def select_action(self, obs: List[float]) -> Tuple[int, Optional[Tuple[int, int]]]:
        if not self.model_loaded:
            # Fallback: simple macro-action that always does something sane
            fallback_action = int(MacroAction.GO_TO_ENEMY_FLAG) if hasattr(
                MacroAction, "GO_TO_ENEMY_FLAG"
            ) else int(MacroAction.GO_TO)
            return fallback_action, None

        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
            if obs_tensor.dim() == 1:
                obs_tensor = obs_tensor.unsqueeze(0)  # [1, obs_dim]

            # ActorCriticNet.act should be the same API you used in training
            out = self.net.act(obs_tensor, deterministic=True)
            # out is a dict: { "action": Tensor, "log_prob": Tensor, "value": Tensor }
            action = out["action"]
            if isinstance(action, torch.Tensor):
                # Shape [1], so take element 0
                action = int(action[0].item())

            return action, None


class CTFViewer:
    def __init__(self):
        rows, cols = 30, 40
        grid = [[0] * cols for _ in range(rows)]
        self.game_field = GameField(grid)
        self.game_manager = self.game_field.getGameManager()

        # ------------- Pygame setup -------------
        pg.init()
        self.size = (1024, 720)
        self.screen = pg.display.set_mode(self.size)
        pg.display.set_caption("UAV CTF – Trained Agent vs OP3 Baseline")
        self.clock = pg.time.Clock()
        self.font = pg.font.SysFont(None, 26)
        self.bigfont = pg.font.SysFont(None, 48)

        self.input_active = False
        self.input_text = ""

        # ------------- RL Policy / Baseline Setup -------------
        # Sanity-check obs dimension from current GameField
        if self.game_field.blue_agents:
            dummy_obs = self.game_field.build_observation(self.game_field.blue_agents[0])
        else:
            # Updated obs builder: 17 scalars + 5x5 map = 42
            dummy_obs = [0.0] * 42
        obs_dim = len(dummy_obs)
        print(f"[CTFViewer] Detected obs_dim={obs_dim}")
        n_actions = len(MacroAction)

        # ---- OP3 baseline for BOTH sides ----
        # Red already defaults to OP3 in GameField, but we enforce it explicitly.
        if hasattr(self.game_field, "policies") and isinstance(self.game_field.policies, dict):
            self.game_field.policies["red"] = OP3RedPolicy("red")

        # Blue baseline: OP3-style policy with side="blue"
        self.blue_op3_baseline = OP3RedPolicy("blue")
        if hasattr(self.game_field, "policies") and isinstance(self.game_field.policies, dict):
            self.game_field.policies["blue"] = self.blue_op3_baseline

        # RL policy for BLUE
        self.blue_learned_policy: Optional[LearnedPolicy] = None
        self.use_learned_blue: bool = False  # start with baseline OP3

        try:
            self.blue_learned_policy = LearnedPolicy(
                obs_dim=obs_dim,
                n_actions=n_actions,
                model_path="checkpoints/ctf_ppo_final.pth",
            )
            if self.blue_learned_policy.model_loaded:
                # Initially let RL control Blue
                if hasattr(self.game_field, "policies") and isinstance(self.game_field.policies, dict):
                    self.game_field.policies["blue"] = self.blue_learned_policy
                self.use_learned_blue = True
                print("[CTFViewer] Blue team using LEARNED policy (checkpoints/ctf_ppo_final.pth)")
            else:
                # No valid model: fall back to OP3 baseline for BLUE
                if hasattr(self.game_field, "policies") and isinstance(self.game_field.policies, dict):
                    self.game_field.policies["blue"] = self.blue_op3_baseline
                self.use_learned_blue = False
                print("[CTFViewer] No valid model → Blue using OP3 baseline")
        except Exception as e:
            print(f"[CTFViewer] RL setup failed: {e}")
            # Fall back to OP3 baseline
            if hasattr(self.game_field, "policies") and isinstance(self.game_field.policies, dict):
                self.game_field.policies["blue"] = self.blue_op3_baseline
            self.use_learned_blue = False
            print("[CTFViewer] Blue using OP3 baseline")

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

            self.game_field.update(dt_sec)
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
            self.game_manager.reset_game(reset_scores=True)
            self.game_field.reset_default()
        elif k == pg.K_F2:
            self.input_active = True
            self.input_text = ""
        elif k == pg.K_F3:
            self.game_manager.reset_game(reset_scores=True)
            self.game_field.runTestCase3()
        elif k == pg.K_r:
            self.game_field.reset_default()
        elif k == pg.K_F4:
            # Toggle between OP3 baseline and learned policy for BLUE
            if not self.blue_learned_policy or not self.blue_learned_policy.model_loaded:
                print("[CTFViewer] No learned model available! Staying on OP3 baseline.")
                return

            self.use_learned_blue = not self.use_learned_blue
            if self.use_learned_blue:
                if hasattr(self.game_field, "policies") and isinstance(self.game_field.policies, dict):
                    self.game_field.policies["blue"] = self.blue_learned_policy
                print("[CTFViewer] Blue → LEARNED policy")
            else:
                if hasattr(self.game_field, "policies") and isinstance(self.game_field.policies, dict):
                    self.game_field.policies["blue"] = self.blue_op3_baseline
                print("[CTFViewer] Blue → OP3 baseline")
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
                self.game_manager.reset_game(reset_scores=True)
                self.game_field.runTestCase2(n, self.game_manager)
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
        self.game_field.draw(self.screen, field_rect)

        def txt(text, x, y, color=(230, 230, 240)):
            img = self.font.render(text, True, color)
            self.screen.blit(img, (x, y))

        gm = self.game_manager
        policy_name = "RL" if self.use_learned_blue else "OP3 BASELINE"
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
            txt("Model: checkpoints/ctf_ppo_final.pth", self.size[0] - 360, 15, (100, 220, 100))

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
    CTFViewer().run()
