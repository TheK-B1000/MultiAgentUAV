"""Matched-seed policy evaluation loop."""
from __future__ import annotations

from argparse import Namespace
from typing import Any, Callable


def matched_seed_evaluation(
    args: Namespace,
    baseline_policy: Any,
    candidate_policy: Any,
    n_agents: int,
    *,
    run_episode_fn: Callable[..., dict[str, Any]],
    validate_opponent_name: Callable[[str], str],
) -> list[dict[str, Any]]:
    """Run policies in the legacy deterministic loop order."""
    rows: list[dict[str, Any]] = []
    policies = (
        (baseline_policy, "baseline"),
        (candidate_policy, "candidate"),
    )

    total = len(args.maps) * len(args.opponents) * args.episodes * len(policies)
    completed = 0

    for map_name in args.maps:
        for opponent in args.opponents:
            requested = validate_opponent_name(opponent)

            for episode_index in range(args.episodes):
                seed = args.seed_start + episode_index

                for policy, policy_name in policies:
                    row = run_episode_fn(
                        policy=policy,
                        policy_name=policy_name,
                        map_name=map_name,
                        opponent=requested,
                        seed=seed,
                        device=args.device,
                        n_agents=n_agents,
                        max_steps=args.max_decision_steps,
                    )
                    rows.append(row)
                    completed += 1

                    print(
                        f"[eval] {completed:>4}/{total} "
                        f"policy={policy_name:9s} "
                        f"map={map_name:24s} "
                        f"requested={requested} "
                        f"resolved={row['resolved_opponent']} "
                        f"seed={seed} "
                        f"score={row['blue_score']:.0f}:"
                        f"{row['red_score']:.0f}"
                    )

    return rows


__all__ = ["matched_seed_evaluation"]
