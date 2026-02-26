"""
Pyquaticus Viewer -- lightweight wrapper that runs the official Pyquaticus
PettingZoo environment and renders it with its built-in pygame GUI.

This replaces the custom GPU CTF viewer; it does NOT use game_field_gpu.py.

Usage (from project root, with pyquaticus installed in the active env):

    python AICTFProject/ctfviewer.py           # 2v2, random actions
    python AICTFProject/ctfviewer.py --team-size 4

To control agents with keyboard, use the upstream tests instead:

    cd pyquaticus
    python ./test/arrowkeys_test.py
"""

from __future__ import annotations

import argparse
from typing import Any, Dict

import numpy as np

from pyquaticus import pyquaticus_v0
from pyquaticus.config import config_dict_std


def make_env(team_size: int = 2, render_mode: str | None = "human") -> Any:
    """
    Build a Pyquaticus 2v2 / 4v4 / 8v8 environment with a sensible default config.
    """
    cfg: Dict[str, Any] = dict(config_dict_std)
    # Light adjustments so episodes are closer to your Aquaticus-style runs.
    cfg["max_score"] = 3
    cfg["max_time"] = 240.0  # seconds
    cfg["tag_on_oob"] = True
    cfg["normalize_obs"] = True
    env = pyquaticus_v0.PyQuaticusEnv(
        config_dict=cfg,
        render_mode=render_mode,
        reward_config=None,
        team_size=int(team_size),
    )
    return env


def run(team_size: int = 2, fps: int = 30) -> None:
    env = make_env(team_size=team_size, render_mode="human")
    try:
        obs, infos = env.reset()
        while True:
            actions: Dict[str, Any] = {}
            for agent_id in env.agents:
                space = env.action_space(agent_id)
                actions[agent_id] = space.sample()
            obs, rewards, terminated, truncated, infos = env.step(actions)
            env.render()
            done = bool(terminated.get("__all__", False) or truncated.get("__all__", False))
            if done:
                obs, infos = env.reset()
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Pyquaticus Viewer")
    parser.add_argument("--team-size", type=int, default=2, choices=[1, 2, 3, 4, 5, 6, 7, 8], help="Agents per team")
    args = parser.parse_args()
    run(team_size=args.team_size)


if __name__ == "__main__":
    main()

