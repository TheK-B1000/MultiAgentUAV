"""
Pyquaticus Viewer -- lightweight wrapper that runs the official Pyquaticus
PettingZoo environment and renders it with its built-in pygame GUI.

Agents default to OP3-style behavior (paper: rule-based mode switching, Medium
Attacker + Medium Defender). They switch between attack and defense based on
flag possession, opponent proximity to own flag, and distance to opponent flag.
The game runs indefinitely: when an episode ends (max score or time), a new
episode starts automatically.

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
from ray.rllib.algorithms.algorithm import Algorithm

# Use the official Pyquaticus env and standard config.
from pyquaticus.envs.pyquaticus import PyQuaticusEnv
from pyquaticus.config import config_dict_std, ACTION_MAP
from pyquaticus.base_policies.base_combined import Heuristic_CTF_Agent


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
    cfg["sim_speedup_factor"] = 8  # or 12, to make motion much faster
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


def run(
    team_size: int = 2,
    checkpoint: Optional[str] = None,
    agent_mode: str = DEFAULT_AGENT_MODE,
    fps: int = 30,
) -> None:
    env = make_env(team_size=team_size, render_mode="human")
    algo: Optional[Algorithm] = None
    learning_agent_ids = _get_learning_agent_ids(team_size)
    if checkpoint:
        algo = Algorithm.from_checkpoint(checkpoint)

    try:
        if algo is not None:
            obs, infos = env.reset()
            reset_opts = None
        else:
            reset_opts = {"normalize_obs": False, "normalize_state": False}
            obs, infos = env.reset(options=reset_opts)

        policies: Dict[str, Heuristic_CTF_Agent] = {}
        for agent_id in env.agents:
            policies[agent_id] = Heuristic_CTF_Agent(
                agent_id, env, mode=agent_mode, continuous=True
            )

        episode_num = 0
        while True:
            actions: Dict[str, Any] = {}
            for agent_id in env.agents:
                if algo is not None and agent_id in learning_agent_ids:
                    agent_obs = obs[agent_id]
                    action = algo.compute_single_action(agent_obs, policy_id="learner_policy", explore=False)
                    if isinstance(action, (int, np.integer)):
                        action = list(ACTION_MAP[int(action)])
                    actions[agent_id] = action
                else:
                    actions[agent_id] = policies[agent_id].compute_action(obs, infos)

            obs, rewards, terminated, truncated, infos = env.step(actions)
            env.render()
            done = bool(terminated.get("__all__", False) or truncated.get("__all__", False))
            if done:
                episode_num += 1
                if reset_opts is None:
                    obs, infos = env.reset()
                else:
                    obs, infos = env.reset(options=reset_opts)
                # Game continues: next episode starts immediately.
                print(f"Episode finished. Starting episode {episode_num + 1} ...")
    finally:
        env.close()
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

