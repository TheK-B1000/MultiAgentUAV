"""Episode execution for the V6I9 map-awareness evaluator."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch


@dataclass(frozen=True)
class EpisodeRunnerRuntime:
    adapt_obs_for_policy: Callable[[Any, Any], Any]
    done: Callable[[Any], bool]
    first_info: Callable[[Any], Any]
    get_opponent_key: Callable[[Any], str]
    make_env: Callable[..., Any]
    model: Callable[[Any], torch.nn.Module]
    predict: Callable[[Any, Any], Any]
    reset_obs: Callable[[Any], Any]
    scores: Callable[[Any, Any], tuple[float, float]]
    set_opponent: Callable[[Any, str], str]
    unpack_step: Callable[[Any], tuple[Any, Any, Any, Any]]
    validate_opponent_name: Callable[[str], str]


def run_episode(
    *,
    runtime: EpisodeRunnerRuntime,
    policy: Any,
    policy_name: str,
    map_name: str,
    opponent: str,
    seed: int,
    device: str,
    n_agents: int,
    max_steps: int,
) -> dict[str, Any]:
    requested_opponent = runtime.validate_opponent_name(opponent)

    env = runtime.make_env(
        n_agents=n_agents,
        map_name=map_name,
        device=device,
        seed=seed,
        max_steps=max_steps,
        instrumented=True,
    )

    model = runtime.model(policy)
    was_training = model.training
    model.eval()

    try:
        resolved_before_reset = runtime.set_opponent(env, requested_opponent)
        obs = runtime.reset_obs(env.reset())
        resolved_after_reset = runtime.get_opponent_key(env)

        if resolved_after_reset != requested_opponent:
            raise RuntimeError(
                f"Opponent changed during reset: requested={requested_opponent}, "
                f"before_reset={resolved_before_reset}, after_reset={resolved_after_reset}."
            )

        last_info: Any = {}
        terminated = False

        for _ in range(max_steps + 8):
            policy_obs = runtime.adapt_obs_for_policy(obs, policy)
            action = runtime.predict(policy, policy_obs)

            obs, _, done, infos = runtime.unpack_step(env.step(action))
            last_info = runtime.first_info(infos)

            if runtime.done(done):
                terminated = True
                break

        if not terminated:
            raise RuntimeError(
                f"Episode did not terminate within {max_steps + 8} evaluator steps."
            )

        blue_score, red_score = runtime.scores(env, last_info)

        return {
            "policy": policy_name,
            "map": map_name,
            "requested_opponent": requested_opponent,
            "resolved_opponent": resolved_after_reset,
            "opponent": resolved_after_reset,
            "seed": seed,
            "blue_score": blue_score,
            "red_score": red_score,
            "win": int(blue_score > red_score),
            "loss": int(blue_score < red_score),
            "draw": int(blue_score == red_score),
            "score_margin": blue_score - red_score,
            **env.metrics(last_info),
        }
    finally:
        model.train(was_training)
        env.close()


__all__ = ["EpisodeRunnerRuntime", "run_episode"]
