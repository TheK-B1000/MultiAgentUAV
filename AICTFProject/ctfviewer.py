import sys
import pygame as pg
import torch
from typing import Optional, Tuple, Any, List

from viewer_game_field import ViewerGameField
from macro_actions import MacroAction
from rl_policy import ActorCriticNet
from policies import OP3RedPolicy

# ----------------------------
# MODEL PATHS (edit these)
# ----------------------------
DEFAULT_PPO_MODEL_PATH = "checkpoints/ctf_fixed_blue_op3.pth"
DEFAULT_MAPPO_MODEL_PATH = "checkpoints/research_mappo_model1.pth"

DEFAULT_N_TARGETS = 50

# IMPORTANT: keep this order consistent with training
USED_MACROS = [
    MacroAction.GO_TO,
    MacroAction.GRAB_MINE,
    MacroAction.GET_FLAG,
    MacroAction.PLACE_MINE,
    MacroAction.GO_HOME,
]


class LearnedPolicy:
    """
    Wrapper around ActorCriticNet so it can be plugged into GameField.policies["blue"].
    Works for PPO-trained or MAPPO-trained checkpoints (we use .act() either way).
    """

    def __init__(self, model_path: str):
        self.device = torch.device("cpu")

        self.net = ActorCriticNet(
            n_macros=len(USED_MACROS),
            n_targets=DEFAULT_N_TARGETS,
            in_channels=7,
            height=40,   # CNN_ROWS
            width=30,    # CNN_COLS
            n_agents=2,  # only matters for central critic; actor still works for any count
        ).to(self.device)

        self.model_path = model_path
        self.model_loaded = False

        try:
            state = torch.load(model_path, map_location=self.device)
            # Support {'model': state_dict} or plain state_dict
            if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
                state_dict = state["model"]
            else:
                state_dict = state

            self.net.load_state_dict(state_dict, strict=True)
            print(f"[CTFViewer] Loaded model from {model_path}")
            self.model_loaded = True
        except Exception as e:
            print(f"[CTFViewer] Failed to load model '{model_path}': {e}")
            self.model_loaded = False

        self.net.eval()

    def __call__(self, agent, game_field):
        # GameField calls this to get (macro_action_id, param)
        obs = game_field.build_observation(agent)
        macro_val, param = self.select_action(obs, agent, game_field)
        return (int(macro_val), param)

    def select_action(
        self,
        obs: Any,
        agent,
        game_field,
    ) -> Tuple[int, Optional[Tuple[int, int]]]:
        """
        obs -> tensor -> ActorCriticNet.act -> return (MacroAction.value, (x,y) or None)
        """
        if not self.model_loaded:
            return int(MacroAction.GO_TO.value), None

        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)

            # Expect [C,H,W] -> [1,C,H,W]
            if obs_tensor.dim() == 3:
                obs_tensor = obs_tensor.unsqueeze(0)
            elif obs_tensor.dim() < 3:
                # fallback (shouldn't happen with your CNN obs)
                obs_tensor = obs_tensor.unsqueeze(0)

            out = self.net.act(
                obs_tensor,
                agent=agent,
                game_field=game_field,
                deterministic=True,
            )

            # macro index in [0..n_macros-1]
            macro_t = out.get("macro_action", out.get("action"))
            macro_idx = int(macro_t[0].item()) if isinstance(macro_t, torch.Tensor) else int(macro_t)
            macro_idx = max(0, min(len(USED_MACROS) - 1, macro_idx))

            # Convert index -> MacroAction value (robust even if enum values aren't 0..4)
            macro_enum = USED_MACROS[macro_idx]
            macro_val = int(getattr(macro_enum, "value", int(macro_enum)))

            param: Optional[Tuple[int, int]] = None
            if "target_action" in out and hasattr(game_field, "get_macro_target"):
                tgt_t = out["target_action"]
                target_idx = int(tgt_t[0].item()) if isinstance(tgt_t, torch.Tensor) else int(tgt_t)
                x, y = game_field.get_macro_target(target_idx)
                param = (int(x), int(y))

            return macro_val, param


class CTFViewer:
    def __init__(
        self,
        ppo_model_path: str = DEFAULT_PPO_MODEL_PATH,
        mappo_model_path: str = DEFAULT_MAPPO_MODEL_PATH,
    ):
        # Grid dims MUST match your GameField grid dims (rows=40, cols=30)
        rows, cols = 40, 30
        grid = [[0] * cols for _ in range(rows)]

        self.game_field = ViewerGameField(grid)
        self.game_manager = self.game_field.getGameManager()

        # Make sure viewer uses internal policies (not external trainer control)
        if hasattr(self.game_field, "use_internal_policies"):
            self.game_field.use_internal_policies = True
        if hasattr(self.game_field, "set_external_control"):
            self.game_field.set_external_control("blue", False)
            self.game_field.set_external_control("red", False)

        # ------------- Pygame setup -------------
        pg.init()
        self.size = (1024, 720)
        self.screen = pg.display.set_mode(self.size)
        pg.display.set_caption("UAV CTF – Blue (Switchable) vs OP3 Red (CNN 7×40×30)")
        self.clock = pg.time.Clock()
        self.font = pg.font.SysFont(None, 26)
        self.bigfont = pg.font.SysFont(None, 48)

        self.input_active = False
        self.input_text = ""

        # Sanity: print detected observation shape
        if self.game_field.blue_agents:
            dummy_obs = self.game_field.build_observation(self.game_field.blue_agents[0])
            try:
                C = len(dummy_obs)
                H = len(dummy_obs[0])
                W = len(dummy_obs[0][0])
                print(f"[CTFViewer] Detected CNN obs shape: C={C}, H={H}, W={W}")
            except Exception:
                print("[CTFViewer] Could not infer CNN obs shape cleanly.")
        else:
            print("[CTFViewer] No agents spawned; cannot infer obs shape.")

        # ---- Policies ----
        # Red always OP3 scripted
        if hasattr(self.game_field, "policies") and isinstance(self.game_field.policies, dict):
            self.game_field.policies["red"] = OP3RedPolicy("red")

        # Blue default is OP3 baseline
        self.blue_op3_baseline = OP3RedPolicy("blue")

        # Load PPO + MAPPO models (both optional)
        self.blue_ppo_policy: Optional[LearnedPolicy] = LearnedPolicy(ppo_model_path)
        self.blue_mappo_policy: Optional[LearnedPolicy] = LearnedPolicy(mappo_model_path)

        # Mode: "OP3" | "PPO" | "MAPPO"
        self.blue_mode: str = "OP3"
        self._apply_blue_mode(self.blue_mode)

        # Ensure any OP3 policies start fresh
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
                    pol.reset()

    # ----------------------------
    # Main loop
    # ----------------------------
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

    # ----------------------------
    # Input handling
    # ----------------------------
    def handle_main_key(self, event):
        k = event.key
        if k == pg.K_F1:
            # Full reset
            self.game_field.agents_per_team = 2
            self.game_manager.reset_game(reset_scores=True)
            self.game_field.reset_default()
            self._reset_op3_policies()

        elif k == pg.K_F2:
            # Set agent count
            self.input_active = True
            self.input_text = str(self.game_field.agents_per_team)

        elif k == pg.K_F3:
            # Cycle: OP3 -> PPO -> MAPPO (only if loaded)
            self._cycle_blue_mode()

        elif k == pg.K_F4:
            # Toggle suppression range debug draw
            self.game_field.debug_draw_ranges = not self.game_field.debug_draw_ranges

        elif k == pg.K_F5:
            # Toggle mine range debug draw
            self.game_field.debug_draw_mine_ranges = not self.game_field.debug_draw_mine_ranges

        elif k == pg.K_r:
            # Quick reset (no score reset)
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
    def draw(self):
        self.screen.fill((12, 12, 18))

        hud_h = 110
        field_rect = pg.Rect(20, hud_h + 10, self.size[0] - 40, self.size[1] - hud_h - 30)
        self.game_field.draw(self.screen, field_rect)

        def txt(text, x, y, color=(230, 230, 240)):
            img = self.font.render(text, True, color)
            self.screen.blit(img, (x, y))

        gm = self.game_manager

        mode = self.blue_mode
        mode_color = (255, 255, 120) if mode == "OP3" else (120, 255, 120)

        # Top row: help
        txt(
            "F1: Full Reset | F2: Set Agent Count | F3: Cycle Blue (OP3/PPO/MAPPO) | F4/F5: Toggle Debug | R: Reset",
            30, 15, (200, 200, 220)
        )

        txt(
            f"Blue Mode: {mode} | Agents: {self.game_field.agents_per_team} vs {self.game_field.agents_per_team}",
            30, 45, mode_color
        )

        # Scores & time
        txt(f"BLUE: {gm.blue_score}", 30, 80, (100, 180, 255))
        txt(f"RED: {gm.red_score}", 200, 80, (255, 100, 100))
        txt(f"Time: {int(gm.current_time)}s", 380, 80, (220, 220, 255))

        # Model paths
        right_x = self.size[0] - 460
        if self.blue_ppo_policy and self.blue_ppo_policy.model_loaded:
            txt(f"PPO: {self.blue_ppo_policy.model_path}", right_x, 45, (140, 240, 140))
        else:
            txt("PPO: (not loaded)", right_x, 45, (180, 180, 180))

        if self.blue_mappo_policy and self.blue_mappo_policy.model_loaded:
            txt(f"MAPPO: {self.blue_mappo_policy.model_path}", right_x, 70, (140, 240, 140))
        else:
            txt("MAPPO: (not loaded)", right_x, 70, (180, 180, 180))

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
