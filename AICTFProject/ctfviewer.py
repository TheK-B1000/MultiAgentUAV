"""
Pyquaticus Viewer -- lightweight wrapper that runs the official Pyquaticus
PettingZoo environment and renders it with its built-in pygame GUI.

Aquaticus-style CTF with a fixed time horizon (match duration). The game runs
for H timesteps (max_time seconds); scoring events are tracked continuously.
Agents default to OP3-style behavior (rule-based mode switching). When an
episode ends (time up), a new episode starts automatically.

Scoring (Aquaticus spec):
  - GRAB = +1 (pick up opponent flag)
  - CAPTURE = +2 (bring opponent flag to your home side)
  - Points = grabs + 2*captures. Higher total at end of time wins; ties possible.

End condition:
  - Time limit (episode horizon) is the primary end condition: play for max_time
    (e.g. 300s / 5 min), then compare points. No early end on a score cap.

How we train (research_pyquaticus):
  - RLlib PPO on Pyquaticus; curriculum and league/self-play use configurable
    episode end (time and/or score cap) and rewards.

Usage (from project root, with pyquaticus installed in the active env):

    python AICTFProject/ctfviewer.py                    # 2v2, OP3 (hard) agents, game continues
    python AICTFProject/ctfviewer.py --team-size 4
    python AICTFProject/ctfviewer.py --mode easy       # Pav01-style (Easy Attacker / Easy Defender)
    python AICTFProject/ctfviewer.py --checkpoint PATH # Blue team = trained policy, red = OP3

To control agents with keyboard, use the upstream tests instead:

    cd pyquaticus
    python ./test/arrowkeys_test.py
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, Optional, Set

import numpy as np
import pygame as pg
from ray.rllib.algorithms.algorithm import Algorithm

# Use the official Pyquaticus env and standard config.
from pyquaticus.envs.pyquaticus import PyQuaticusEnv
from pyquaticus.config import config_dict_std, ACTION_MAP
from pyquaticus.base_policies.base_combined import Heuristic_CTF_Agent


def make_env(team_size: int = 2, render_mode: str | None = "human") -> Any:
    """
    Build a Pyquaticus 2v2 / 4v4 / 8v8 environment. Uses time-horizon mode:
    episode ends only when max_time is reached; score = GRAB(+1) + CAPTURE(+2).
    """
    cfg: Dict[str, Any] = dict(config_dict_std)
    cfg["max_time"] = 600.0  # match duration (seconds) = 10 min
    cfg["end_on_time_only"] = True  # no score cap; higher points at time end wins
    cfg["tag_on_oob"] = True
    cfg["normalize_obs"] = True
    cfg["sim_speedup_factor"] = 8
    env = PyQuaticusEnv(
        config_dict=cfg,
        render_mode=render_mode,
        reward_config=None,
        team_size=int(team_size),
        action_space="continuous",  # heuristics and optional RL policy use (speed, heading)
    )
    return env


# Paper mapping: easy = Pav01 (Easy Attacker / Easy Defender), medium = Strategy 2/3,
# hard = OP3 = Strategy 4 style (rule-based mode switching, Medium Attacker + Medium Defender).
DEFAULT_AGENT_MODE = "hard"  # OP3


def _get_learning_agent_ids(team_size: int) -> Set[str]:
    # Match training: agent_0..agent_(team_size-1) are the blue/learning side.
    return {f"agent_{i}" for i in range(int(team_size))}


def _draw_hud(env: Any, infos: Dict[str, Any]) -> None:
    """Draw a stable HUD: top row F1, F2, F3, Esc; below that points (G+2C) and time."""
    screen = getattr(env, "screen", None)
    if screen is None:
        return
    try:
        grabs = env.state.get("grabs", [0, 0])
        captures = env.state.get("captures", [0, 0])
        # Aquaticus: points = GRAB(+1) + CAPTURE(+2)
        blue_pts = int(grabs[0]) + 2 * int(captures[0])
        red_pts = int(grabs[1]) + 2 * int(captures[1])
    except (KeyError, TypeError, IndexError):
        blue_pts = red_pts = 0
    current_time = getattr(env, "current_time", 0.0)
    max_time = getattr(env, "max_time", 600.0)
    time_remaining = max(0.0, float(max_time) - float(current_time))

    # HUD bar height and full width so we overwrite the same area every time (no flash).
    w = screen.get_width()
    hud_h = 56
    hud_surf = pg.Surface((w, hud_h))
    hud_surf.set_alpha(230)
    hud_surf.fill((18, 22, 28))

    font = pg.font.SysFont(None, 26)
    key_font = pg.font.SysFont(None, 24)

    # Top row: F1, F2, F3, Esc left to right with spacing.
    menu_items = [
        ("F1", "Reset"),
        ("F2", "Agents"),
        ("F3", "Model"),
        ("Esc", "Quit"),
    ]
    x = 24
    for key, label in menu_items:
        key_s = key_font.render(key, True, (200, 220, 255))
        lbl_s = key_font.render(label, True, (180, 195, 220))
        hud_surf.blit(key_s, (x, 6))
        hud_surf.blit(lbl_s, (x, 24))
        x += max(key_s.get_width(), lbl_s.get_width()) + 48

    # Center line: points (G+2C) and time (match duration = horizon).
    score_time = f"Points (G+2C)  BLUE: {blue_pts}   RED: {red_pts}    Time: {current_time:.0f}s / {max_time:.0f}s  (left: {time_remaining:.0f}s)"
    st_surf = font.render(score_time, True, (240, 240, 248))
    st_rect = st_surf.get_rect(centerx=w // 2, bottom=hud_h - 6)
    hud_surf.blit(st_surf, st_rect)

    screen.blit(hud_surf, (0, 0))
    # Caller does one flip after this so the HUD stays visible (env's flips are suppressed).


def run(
    team_size: int = 2,
    checkpoint: Optional[str] = None,
    agent_mode: str = DEFAULT_AGENT_MODE,
    fps: int = 30,
) -> None:
    pg.init()
    _real_flip = pg.display.flip
    env = make_env(team_size=team_size, render_mode="human")
    algo: Optional[Algorithm] = None
    current_team_size = int(team_size)
    learning_agent_ids = _get_learning_agent_ids(current_team_size)
    if checkpoint:
        algo = Algorithm.from_checkpoint(checkpoint)

    # By default, if a checkpoint is provided, blue team uses the trained policy.
    use_algo_for_blue = algo is not None

    try:
        # Initial reset
        if algo is not None:
            reset_opts = None
            obs, infos = env.reset()
        else:
            reset_opts = {"normalize_obs": False, "normalize_state": False}
            obs, infos = env.reset(options=reset_opts)

        # Heuristic controllers (for all agents; blue may be overridden by RL if enabled).
        # Role bias like old viewer: alternating attacker/defender so teammates don't cluster.
        def _role_for_agent(aid: str) -> str:
            try:
                i = int(aid.split("_")[-1])
                return "attacker" if (i % 2 == 0) else "defender"
            except (ValueError, IndexError):
                return "balanced"

        policies: Dict[str, Heuristic_CTF_Agent] = {}
        for agent_id in env.agents:
            policies[agent_id] = Heuristic_CTF_Agent(
                agent_id, env, mode=agent_mode, continuous=True, role_bias=_role_for_agent(agent_id)
            )

        episode_num = 0
        running = True
        clock = pg.time.Clock()

        # Suppress pygame flips during env.step() so we can draw HUD on top and flip once (menu stays visible).
        _suppress_flip = [False]

        def _flip_noop() -> None:
            if not _suppress_flip[0]:
                _real_flip()

        pg.display.flip = _flip_noop

        # Show initial frame with score 0–0 and time 0 so HUD is visible from the start.
        _suppress_flip[0] = True
        env.render()
        _suppress_flip[0] = False
        _draw_hud(env, infos)
        _real_flip()

        while running:
            # Handle keyboard like the old viewer:
            # F1: reset episode, F2: change agents (2v2/4v4/8v8), F3: toggle model/heuristic.
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    running = False
                elif event.type == pg.KEYDOWN:
                    if event.key == pg.K_ESCAPE:
                        running = False
                    elif event.key in (pg.K_F1, pg.K_r):
                        # Reset current episode.
                        episode_num = 0
                        if reset_opts is None:
                            obs, infos = env.reset()
                        else:
                            obs, infos = env.reset(options=reset_opts)
                        print("[viewer] Reset episode.")
                        continue
                    elif event.key == pg.K_F2:
                        # Cycle team sizes: 2 -> 4 -> 8 -> 2
                        sizes = [2, 4, 8]
                        try:
                            idx = sizes.index(current_team_size)
                        except ValueError:
                            idx = 0
                        current_team_size = sizes[(idx + 1) % len(sizes)]
                        print(f"[viewer] Changing agents per team to {current_team_size} ...")
                        # Rebuild env and controllers with new team size.
                        env.close()
                        env = make_env(team_size=current_team_size, render_mode="human")
                        learning_agent_ids = _get_learning_agent_ids(current_team_size)
                        if algo is not None:
                            reset_opts = None
                            obs, infos = env.reset()
                        else:
                            reset_opts = {"normalize_obs": False, "normalize_state": False}
                            obs, infos = env.reset(options=reset_opts)
                        policies = {
                            agent_id: Heuristic_CTF_Agent(
                                agent_id, env, mode=agent_mode, continuous=True,
                                role_bias=_role_for_agent(agent_id),
                            )
                            for agent_id in env.agents
                        }
                        episode_num = 0
                        continue
                    elif event.key == pg.K_F3:
                        if algo is None:
                            print("[viewer] No checkpoint loaded; F3 has no effect.")
                        else:
                            use_algo_for_blue = not use_algo_for_blue
                            mode_str = "RL model (blue)" if use_algo_for_blue else "heuristics only"
                            print(f"[viewer] Control mode -> {mode_str}")

            # Build actions
            actions: Dict[str, Any] = {}
            for agent_id in env.agents:
                if algo is not None and use_algo_for_blue and agent_id in learning_agent_ids:
                    # Blue agents controlled by trained RL policy when enabled.
                    agent_obs = obs[agent_id]
                    action = algo.compute_single_action(
                        agent_obs, policy_id="learner_policy", explore=False
                    )
                    # Policy may be discrete (int); viewer env is continuous (speed, heading).
                    if isinstance(action, (int, np.integer)):
                        action = list(ACTION_MAP[int(action)])
                    actions[agent_id] = action
                else:
                    actions[agent_id] = policies[agent_id].compute_action(obs, infos)

            # Step environment (flips suppressed so env doesn't overwrite our HUD).
            _suppress_flip[0] = True
            obs, rewards, terminated, truncated, infos = env.step(actions)
            _suppress_flip[0] = False
            _draw_hud(env, infos)
            _real_flip()

            done = bool(terminated.get("__all__", False) or truncated.get("__all__", False))
            if done:
                episode_num += 1
                if reset_opts is None:
                    obs, infos = env.reset()
                else:
                    obs, infos = env.reset(options=reset_opts)
                print(f"Episode finished. Starting episode {episode_num + 1} ...")

            clock.tick(fps)
    finally:
        pg.display.flip = _real_flip
        env.close()
        pg.quit()
        if algo is not None:
            algo.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Pyquaticus Viewer (OP3 agents by default; game continues)")
    parser.add_argument("--team-size", type=int, default=2, choices=[1, 2, 3, 4, 5, 6, 7, 8], help="Agents per team")
    parser.add_argument(
        "--mode",
        type=str,
        default=DEFAULT_AGENT_MODE,
        choices=["easy", "medium", "hard"],
        help="Heuristic mode: easy (Pav01), medium (Strategy 2/3), hard (OP3 / Strategy 4). Default: hard.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to an RLlib checkpoint to control the blue team; red uses --mode.",
    )
    args = parser.parse_args()
    run(team_size=args.team_size, checkpoint=args.checkpoint, agent_mode=args.mode)


if __name__ == "__main__":
    main()

