"""Shared rollout for the O1 protocol: C1-aware episodes with policy handoff.

Used by both `run_o1_response_oracle.py` (training-time C1 panel) and
`run_o1_gates.py` (the four gates), so the two cannot disagree about what a C1
onset is or how an episode outcome is scored.

Episode mechanics are the frozen ones from ``run_g0_v2_evaluation``: same env
configuration, same ``legal_context``, same authoritative capture ledger, same
V2 rules. The only additions are C1 onset tracking and the optional handoff.

THE HANDOFF
-----------
``run_c1_episode(policy_a, policy_b=...)`` runs ``policy_a`` until
``c1_active_from_context`` first returns true, then hands control to
``policy_b`` for the remainder. With ``policy_b=None`` it is an ordinary
single-policy episode. Because the environment is seeded identically in both
cases, arm A and arm B share a byte-identical prefix up to the switch step, and
every difference afterwards is attributable to the handoff.

DEFINITION RESOLVED HERE
------------------------
``lost_after_leading`` is the EPISODE-level quantity already defined in
``run_g0_v2_evaluation.summarize_episode``:

    lead_seen at any point  AND  final blue score < final red score

The C1 confirmation also emits a *window*-level event of the same name, fired on
each red capture that follows a lead. That one is a within-episode construct
built for precursor mining. The gates are episode-level and paired on seed, so
the episode-level definition is the one that applies, and it is the existing
named field rather than a new invention.

The frozen preregistration says "lead_preserved = NOT lost_after_leading" and
cites "the exact failure C1 was confirmed against", which admits both readings.
This module resolves it to the episode-level definition. The resolution is
recorded in every report this module feeds, and it was made before any O1
training data existed.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402

from experiments.c1_context import apply_c1_scenario, c1_active_from_context  # noqa: E402

LOST_AFTER_LEADING_DEFINITION = (
    "episode-level: a lead was held at some point AND the final blue score is "
    "below the final red score (run_g0_v2_evaluation.summarize_episode)"
)


def run_c1_episode(
    policy_a: Any,
    *,
    opponent: str,
    seed: int,
    device: str,
    policy_b: Optional[Any] = None,
    inject_c1: bool = False,
    on_step: Optional[Callable[[int, dict, Any, bool], None]] = None,
) -> dict:
    """One episode, tracking C1 onset and optionally handing off at it.

    ``policy_b`` None  -> ``policy_a`` acts throughout (arm A).
    ``policy_b`` given -> ``policy_a`` acts until the first decision at which
                          C1 is active, then ``policy_b`` acts (arm B).

    ``inject_c1`` resets the episode into the C1 region. Training panels use it.
    The gates must not: see the preregistration, section 3.1.

    ``on_step(step_index, ctx, obs, is_b)`` is called before each action, for
    callers building an observation bank.
    """
    from experiments.eval_v6i9_map_awareness import (
        _adapt_obs_for_policy,
        _done,
        _predict,
        _reset_obs,
        _unpack_step,
    )
    from experiments.run_g0_v2_evaluation import (
        AGENTS,
        CANONICAL_MAP,
        EPISODE_HORIZON,
        V2_RULES,
        legal_context,
    )
    from rl.evaluation.opponent_resolution import (
        get_opponent_key as _get_opponent_key,
        set_opponent as _set_opponent,
        validate_opponent_name as _validate_opponent_name,
    )
    from gpu_env import GPUCTFVecEnv, GPUFieldConfig

    requested = _validate_opponent_name(opponent)
    cfg = GPUFieldConfig(
        n_envs=1,
        max_blue_agents=AGENTS,
        max_red_agents=AGENTS,
        map_set="train",
        map_layout=CANONICAL_MAP,
        max_decision_steps=EPISODE_HORIZON,
        aquaticus_profile=True,
        rules_profile="OURS",
        device=device,
        seed=int(seed),
        obstacle_obs_channel=True,
        tag_telemetry_enabled=True,
        **V2_RULES,
    )
    env = GPUCTFVecEnv(cfg)
    core = env.core

    models = []
    for p in (policy_a, policy_b):
        if p is None:
            continue
        m = p.model if hasattr(p, "model") else p
        models.append((m, getattr(m, "training", False)))
        if hasattr(m, "eval"):
            m.eval()

    try:
        _set_opponent(env, requested)
        obs = _reset_obs(env.reset())
        resolved = _get_opponent_key(env)
        if resolved != requested:
            raise RuntimeError(f"opponent drift: {requested} -> {resolved}")
        if inject_c1:
            apply_c1_scenario(core)
            obs = env.get_obs()
        core.drain_tag_events()

        c1_onset_step: Optional[int] = None
        c1_active_steps = 0
        steps_by_b = 0
        lead_seen = False
        captures_blue = captures_red = 0
        ledger_blue = ledger_red = 0
        obs_blue_max = obs_red_max = 0.0
        terminated = False

        for step_i in range(EPISODE_HORIZON + 8):
            ctx = legal_context(core)
            obs_blue_max = max(obs_blue_max, ctx["blue_score"])
            obs_red_max = max(obs_red_max, ctx["red_score"])
            if ctx["score_diff"] > 0:
                lead_seen = True

            active = c1_active_from_context(ctx)
            c1_active_steps += int(active)
            if active and c1_onset_step is None:
                c1_onset_step = step_i

            # The handoff is one-way and fires at the FIRST C1 decision.
            use_b = policy_b is not None and c1_onset_step is not None
            actor = policy_b if use_b else policy_a
            steps_by_b += int(use_b)

            if on_step is not None:
                on_step(step_i, ctx, obs, bool(use_b))

            action = _predict(actor, _adapt_obs_for_policy(obs, actor))
            obs, _, done, _infos = _unpack_step(env.step(action))

            for e in core.drain_tag_events():
                if e.get("event_type") != "capture_scored":
                    continue
                # The ledger is authoritative: reading core.*_score after a
                # terminal step returns the POST-RESET value.
                after = int(e.get("score_after", 0))
                if e.get("scoring_team") == "blue":
                    captures_blue += 1
                    ledger_blue = max(ledger_blue, after)
                else:
                    captures_red += 1
                    ledger_red = max(ledger_red, after)

            if _done(done):
                terminated = True
                break

        blue_score = float(max(ledger_blue, obs_blue_max))
        red_score = float(max(ledger_red, obs_red_max))
    finally:
        for m, was_training in models:
            if hasattr(m, "train"):
                m.train(was_training)
        env.close()

    lost_after_leading = int(lead_seen and blue_score < red_score)
    return {
        "opponent": requested,
        "eval_seed": int(seed),
        "episode_key": f"{requested}:{seed}",
        "blue_score": blue_score,
        "red_score": red_score,
        "score_margin": blue_score - red_score,
        "win": int(blue_score > red_score),
        "loss": int(blue_score < red_score),
        "draw": int(blue_score == red_score),
        "captures_blue": captures_blue,
        "captures_red": captures_red,
        "lead_seen": int(lead_seen),
        "lost_after_leading": lost_after_leading,
        "lead_preserved": int(lead_seen and not lost_after_leading),
        # --- C1 tracking ---------------------------------------------------
        "c1_fired": int(c1_onset_step is not None),
        "c1_onset_step": c1_onset_step,
        "c1_active_steps": c1_active_steps,
        "handoff_occurred": int(policy_b is not None and c1_onset_step is not None),
        "steps_after_handoff": steps_by_b,
        "injected": int(inject_c1),
        "terminated": terminated,
    }


def paired_bootstrap_delta(
    pairs: list[tuple[float, float]],
    *,
    resamples: int = 2000,
    seed: int = 12_345,
) -> dict:
    """Bootstrap mean(b - a) over paired episodes, resampling PAIRS.

    The pair is the cluster unit: arm A and arm B of one (opponent, seed) share
    a prefix and are not independent observations.
    """
    if len(pairs) < 2:
        return {"delta": None, "ci_low": None, "ci_high": None,
                "n_pairs": len(pairs), "excludes_zero": None,
                "insufficient_support": True}

    diffs = np.asarray([b - a for a, b in pairs], dtype=float)
    rng = np.random.default_rng(seed)
    point = float(diffs.mean())
    draws = diffs[rng.integers(0, diffs.size, (resamples, diffs.size))].mean(axis=1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {
        "delta": round(point, 4),
        "ci_low": round(float(lo), 4),
        "ci_high": round(float(hi), 4),
        "n_pairs": len(pairs),
        "excludes_zero": bool(lo > 0 or hi < 0),
        "insufficient_support": False,
    }


__all__ = [
    "LOST_AFTER_LEADING_DEFINITION",
    "paired_bootstrap_delta",
    "run_c1_episode",
]
