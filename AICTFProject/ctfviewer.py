import sys
import pygame as pg
import torch
import numpy as np
from typing import Optional, Tuple, Any

from game_field import GameField
from macro_actions import MacroAction
from rl_policy import ActorCriticNet
from policies import OP3RedPolicy

# Match the final checkpoint from train_ppo_event.py
DEFAULT_MODEL_PATH = "checkpoints/ctf_fixed_blue_op3.pth"


class LearnedPolicy:
    """
    Wraps the trained CNN+extra dual-head ActorCriticNet so GameField can use it
    like any other Policy: policy(agent, game_field) -> (macro_idx, target_idx).
    """

    def __init__(self, model_path: Optional[str] = None):
        self.device = torch.device("cpu")  # viewer is fine on CPU

        # Must match training config: 7xHxW + extra_dim
        self.net = ActorCriticNet(
            n_macros=len(MacroAction),
            n_targets=50,
            height=30,    # matches GRID_ROWS used in training
            width=40,     # matches GRID_COLS used in training
            latent_dim=128,
            extra_dim=7,  # whatever extra_dim you used in training
        ).to(self.device)

        self.model_loaded = False
        path = model_path if model_path else DEFAULT_MODEL_PATH

        try:
            state = torch.load(path, map_location=self.device)
            # Support {'model': state_dict} or plain state_dict
            if isinstance(state, dict) and "model" in state:
                state_dict = state["model"]
            else:
                state_dict = state

            self.net.load_state_dict(state_dict)
            print(f"[LearnedPolicy] Loaded model from {path}")
            self.model_loaded = True
        except Exception as e:
            print(f"[LearnedPolicy] Failed to load model '{path}': {e}")
            self.model_loaded = False

        self.net.eval()

    # GameField calls policy(agent, game_field) -> (action_id, param)
    def __call__(self, agent, game_field) -> Tuple[int, Any]:
        # CNN-style observation: 7xHxW map + extra vector
        cnn_obs, extra_vec = game_field.build_cnn_observation(agent)
        return self.select_action(cnn_obs, extra_vec, agent, game_field)

    def select_action(
        self,
        cnn_obs: np.ndarray,
        extra_vec: np.ndarray,
        agent,
        game_field,
    ) -> Tuple[int, Any]:

        # Fallback if no model
        if not self.model_loaded:
            return int(MacroAction.GO_TO), None

        with torch.no_grad():
            # cnn_obs: [C, H, W] → [1, C, H, W]
            obs_tensor = torch.tensor(
                cnn_obs, dtype=torch.float32, device=self.device
            )
            if obs_tensor.dim() == 3:
                obs_tensor = obs_tensor.unsqueeze(0)

            # extra_vec: [extra_dim] → [1, extra_dim]
            extra_tensor = torch.tensor(
                extra_vec, dtype=torch.float32, device=self.device
            )
            if extra_tensor.dim() == 1:
                extra_tensor = extra_tensor.unsqueeze(0)

            out = self.net.act(
                obs_tensor,
                extra=extra_tensor,
                agent=agent,
                game_field=game_field,
                deterministic=True,  # deterministic for viewer/demo
            )

            macro_action = out["macro_action"]
            if isinstance(macro_action, torch.Tensor):
                macro_action = int(macro_action[0].item())

            target_action = out["target_action"]
            if isinstance(target_action, torch.Tensor):
                target_action = int(target_action[0].item())

            # macro is the MacroAction index, target is the macro_target index
            return macro_action, target_action


class CTFViewer:
    def __init__(self):
        rows, cols = 20, 20
        grid = [[0] * cols for _ in range(rows)]
        self.game_field = GameField(grid)
        self.game_manager = self.game_field.getGameManager()

        # ------------- Pygame setup -------------
        pg.init()
        self.size = (1024, 768)
        self.screen = pg.display.set_mode(self.size)
        pg.display.set_caption("UAV CTF – Learned CNN Agent vs OP3")
        self.clock = pg.time.Clock()
        self.font = pg.font.SysFont("Consolas", 18)
        self.bigfont = pg.font.SysFont(None, 48)

        self.input_active = False
        self.input_text = ""

        # ---- Sanity Check Obs Dim (CNN) ----
        if self.game_field.blue_agents:
            cnn_obs, extra_vec = self.game_field.build_observation(
                self.game_field.blue_agents[0]
            )
            print(
                f"[CTFViewer] CNN obs shape: {cnn_obs.shape}, extra_dim={extra_vec.shape[0]}"
            )

        # ---- Setup Policies ----
        # 1. Red is always OP3 (Baseline)
        if hasattr(self.game_field, "policies") and isinstance(
            self.game_field.policies, dict
        ):
            self.game_field.policies["red"] = OP3RedPolicy("red")

        # 2. Blue can be OP3 or Learned
        self.blue_op3_baseline = OP3RedPolicy("blue")
        self.blue_learned_policy: Optional[LearnedPolicy] = None
        self.use_learned_blue: bool = False

        # Try loading learned model
        self.blue_learned_policy = LearnedPolicy(model_path=DEFAULT_MODEL_PATH)

        if self.blue_learned_policy.model_loaded:
            self._set_blue_policy(use_learned=True)
            print("[CTFViewer] Default: Blue using LEARNED CNN policy")
        else:
            self._set_blue_policy(use_learned=False)
            print("[CTFViewer] Default: Blue using OP3 Baseline")

        self._reset_op3_policies()

    def _set_blue_policy(self, use_learned: bool):
        self.use_learned_blue = use_learned
        target_policy = (
            self.blue_learned_policy if use_learned else self.blue_op3_baseline
        )

        if hasattr(self.game_field, "policies") and isinstance(
            self.game_field.policies, dict
        ):
            self.game_field.policies["blue"] = target_policy

    def _reset_op3_policies(self):
        # Reset internal state of scripted policies
        if hasattr(self.game_field, "policies") and isinstance(
            self.game_field.policies, dict
        ):
            for side in ("blue", "red"):
                pol = self.game_field.policies.get(side)
                if hasattr(pol, "reset"):
                    pol.reset()

    # Main loop
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

    # Input handling
    def handle_main_key(self, event):
        k = event.key
        if k == pg.K_F1:
            # Reset
            self.game_manager.reset_game(reset_scores=True)
            self.game_field.reset_default()
            self._reset_op3_policies()

        elif k == pg.K_F2:
            # Change agent count
            self.input_active = True
            self.input_text = ""

        elif k == pg.K_F3:
            # Swap zones
            self.game_manager.reset_game(reset_scores=True)
            self.game_field.runTestCase3()
            self._reset_op3_policies()

        elif k == pg.K_r:
            # Quick soft reset
            self.game_field.reset_default()
            self._reset_op3_policies()

        elif k == pg.K_F4:
            # Toggle Blue Policy
            if not self.blue_learned_policy or not self.blue_learned_policy.model_loaded:
                print("[CTFViewer] Cannot toggle: Model not loaded.")
                return

            new_state = not self.use_learned_blue
            self._set_blue_policy(new_state)
            if not new_state:
                self._reset_op3_policies()
            print(
                f"[CTFViewer] Blue Policy switched to: "
                f"{'LEARNED' if new_state else 'OP3'}"
            )

        elif k == pg.K_F5:
            self.game_field.debug_draw_ranges = not self.game_field.debug_draw_ranges
        elif k == pg.K_F6:
            self.game_field.debug_draw_mine_ranges = not self.game_field.debug_draw_mine_ranges
        elif k == pg.K_ESCAPE:
            pg.event.post(pg.event.Event(pg.QUIT))

    def handle_input_key(self, event):
        if event.key == pg.K_RETURN:
            try:
                n = int(self.input_text or "2")
                n = max(1, min(100, n))
                self.game_manager.reset_game(reset_scores=True)
                self.game_field.runTestCase2(n, self.game_manager)
                self._reset_op3_policies()
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

    # Drawing / HUD
    def draw(self):
        self.screen.fill((20, 24, 32))

        hud_h = 100
        field_rect = pg.Rect(
            20, hud_h + 20, self.size[0] - 40, self.size[1] - hud_h - 40
        )
        self.game_field.draw(self.screen, field_rect)

        def txt(text, x, y, color=(200, 200, 200), size=None):
            f = self.font if size is None else self.bigfont
            img = f.render(text, True, color)
            self.screen.blit(img, (x, y))

        gm = self.game_manager

        # Policy Status
        pol_name = "CNN PPO" if self.use_learned_blue else "OP3 SCRIPT"
        pol_color = (
            (80, 255, 80) if self.use_learned_blue else (255, 255, 100)
        )

        # Header
        txt(
            "F1:Reset  F2:Agents  F3:SwapZones  F4:TogglePolicy  F5/F6:Debug",
            20,
            15,
            (150, 160, 180),
        )

        # Stats Row
        row_2_y = 50
        txt(f"BLUE SCORE: {gm.blue_score}", 20, row_2_y, (100, 180, 255))
        txt(f"RED SCORE:  {gm.red_score}", 200, row_2_y, (255, 100, 100))
        txt(f"TIME: {int(gm.current_time)}s", 380, row_2_y)

        # Policy Indicator
        txt(f"Blue Agent: {pol_name}", self.size[0] - 300, row_2_y, pol_color)

        # Input Overlay
        if self.input_active:
            overlay = pg.Surface(self.size, pg.SRCALPHA)
            overlay.fill((0, 0, 0, 200))
            self.screen.blit(overlay, (0, 0))

            cx, cy = self.size[0] // 2, self.size[1] // 2
            box = pg.Rect(0, 0, 400, 180)
            box.center = (cx, cy)

            pg.draw.rect(self.screen, (40, 45, 60), box, border_radius=8)
            pg.draw.rect(self.screen, (80, 90, 120), box, width=2, border_radius=8)

            txt("Agent Count (1-100):", box.x + 20, box.y + 30, (255, 255, 255))
            txt(
                self.input_text + "_",
                box.x + 20,
                box.y + 80,
                (100, 255, 255),
                size="big",
            )


if __name__ == "__main__":
    CTFViewer().run()
