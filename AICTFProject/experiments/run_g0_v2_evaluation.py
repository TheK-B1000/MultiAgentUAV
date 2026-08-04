"""G0-v2 evaluation + weakness mining.

Evaluates the three preregistered 1,000,000-step policies against the admitted
OP6-OP12 mixture on map_a using HELD-OUT evaluation seeds, then mines recurring
failure modes.

Two rules shape the whole analysis:

1. The preregistered primary policy is the 1,000,000-step checkpoint. No
   intermediate checkpoint is considered, so there is no opportunity to pick a
   prettier one after seeing results.
2. Weaknesses are described only in LEGAL same-map context -- score, clock, flag
   state, tagged/cooldown state, formation, forward commitment, carrier pressure
   and escort availability. Opponent preset identity is recorded for reporting
   competence per opponent, but never used as a context feature, because the
   future router will not have access to it.

Run:  python experiments/run_g0_v2_evaluation.py
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from experiments.run_g0_v2_seed import (  # noqa: E402
    AGENTS,
    CANONICAL_MAP,
    EPISODE_HORIZON,
    G0_SEEDS,
    OPPONENTS,
    RULESET_ID,
    artifact_dir_for,
    run_tag_for,
)

# Held-out: disjoint from the training seeds (2,500,00x) by construction.
EVAL_SEED_BASE = 9_100_000
EPISODES_PER_CELL = 30
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

V2_RULES = dict(
    taggers_required=1,
    tag_min_interval_seconds=10.0,
    tag_nearest_only=True,
    tag_channel_seconds=0.0,
    suppression_attackers_required=2,
)

OUT_DIR = PROJECT_ROOT / "artifacts" / "g0_v2_evaluation"

# Forward commitment / escort geometry, in field-width fractions.
ESCORT_RADIUS_FRAC = 0.22
PRESSURE_RADIUS_FRAC = 0.18


def _np(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


# --- legal same-map context -------------------------------------------------


def legal_context(core) -> dict:
    """Same-map, opponent-agnostic state a router could legally condition on.

    Deliberately excludes opponent identity: every value here is readable from
    the field itself.
    """
    b = 0  # single-env evaluation
    blue_pos = _np(core.blue_pos)[b]          # (Nb, 2)
    red_pos = _np(core.red_pos)[b]
    blue_carry = _np(core.blue_carrying)[b].astype(bool)
    red_carry = _np(core.red_carrying)[b].astype(bool)
    blue_tagged = _np(core.blue_tagged)[b].astype(bool)
    red_tagged = _np(core.red_tagged)[b].astype(bool)
    blue_cd = _np(core.blue_tag_cooldown)[b].astype(float)
    red_cd = _np(core.red_tag_cooldown)[b].astype(float)
    blue_alive = _np(core.blue_alive)[b].astype(bool)
    red_alive = _np(core.red_alive)[b].astype(bool)
    blue_flag_pos = _np(core.blue_flag_pos)[b]
    blue_flag_home = _np(core.blue_flag_home)[b]

    cols = float(core.cols)
    rows = float(getattr(core, "rows", cols))
    mid_x = cols * 0.5

    step = int(_np(core.step_count)[b])
    horizon = int(getattr(core.cfg, "max_decision_steps", EPISODE_HORIZON))
    blue_score = float(_np(core.blue_score)[b])
    red_score = float(_np(core.red_score)[b])

    # Blue attacks toward the red half (x > mid). "Forward" = on the red side.
    blue_x = blue_pos[:, 0]
    forward = (blue_x > mid_x) & blue_alive
    n_forward = int(forward.sum())

    # Formation: lateral spread and pairwise separation, normalized.
    if blue_pos.shape[0] >= 2:
        spread = float(abs(blue_pos[0, 1] - blue_pos[1, 1]) / max(rows, 1e-6))
        separation = float(np.linalg.norm(blue_pos[0] - blue_pos[1]) / max(cols, 1e-6))
    else:
        spread = separation = 0.0

    # Carrier pressure + escort availability.
    carrier_idx = int(np.argmax(blue_carry)) if blue_carry.any() else -1
    red_tag_ready = (red_cd <= 1e-9) & red_alive & (~red_tagged)
    if carrier_idx >= 0:
        cpos = blue_pos[carrier_idx]
        red_d = np.linalg.norm(red_pos - cpos[None, :], axis=1) / max(cols, 1e-6)
        carrier_pressure = float(red_d.min())
        # Distance to the nearest defender that could actually tag right now --
        # a close defender on cooldown is not the same threat as a ready one.
        ready_d = red_d[red_tag_ready]
        nearest_ready_defender = float(ready_d.min()) if ready_d.size else float("inf")
        mates = [i for i in range(blue_pos.shape[0]) if i != carrier_idx and blue_alive[i]]
        if mates:
            mate_d = min(
                float(np.linalg.norm(blue_pos[i] - cpos) / max(cols, 1e-6)) for i in mates
            )
        else:
            mate_d = float("inf")
        escort_available = bool(mate_d <= ESCORT_RADIUS_FRAC)
        under_pressure = bool(carrier_pressure <= PRESSURE_RADIUS_FRAC)
        escort_distance = mate_d
    else:
        carrier_pressure = float("nan")
        nearest_ready_defender = float("nan")
        escort_distance = float("nan")
        escort_available = False
        under_pressure = False

    # Home threat: red holds our flag, or our flag is off its home position, or
    # a red agent is closing on it while we have nobody back.
    blue_flag_away = bool(
        float(np.linalg.norm(blue_flag_pos - blue_flag_home) / max(cols, 1e-6)) > 0.02
    )
    red_to_our_flag = np.linalg.norm(red_pos - blue_flag_pos[None, :], axis=1) / max(cols, 1e-6)
    nearest_red_to_our_flag = float(red_to_our_flag.min()) if red_to_our_flag.size else float("inf")
    blue_to_our_flag = np.linalg.norm(blue_pos - blue_flag_pos[None, :], axis=1) / max(cols, 1e-6)
    nearest_blue_to_our_flag = float(blue_to_our_flag.min()) if blue_to_our_flag.size else float("inf")
    home_threatened = bool(
        int(red_carry.sum()) > 0
        or blue_flag_away
        or (nearest_red_to_our_flag < nearest_blue_to_our_flag - 0.05)
    )

    return {
        "step": step,
        "time_remaining_frac": max(0.0, 1.0 - step / max(horizon, 1)),
        "blue_score": blue_score,
        "red_score": red_score,
        "score_diff": blue_score - red_score,
        "blue_carrying": int(blue_carry.sum()),
        "red_carrying": int(red_carry.sum()),
        "blue_tagged": int(blue_tagged.sum()),
        "red_tagged": int(red_tagged.sum()),
        "blue_cooldown_active": int((blue_cd > 1e-9).sum()),
        "agents_forward": n_forward,
        "formation_spread": spread,
        "team_separation": separation,
        "carrier_present": carrier_idx >= 0,
        "carrier_pressure": carrier_pressure,
        "carrier_under_pressure": under_pressure,
        "escort_available": escort_available,
        "escort_distance": escort_distance,
        "carrier_unescorted": bool(carrier_idx >= 0 and not escort_available),
        # --- precursor-relevant state (all legal, all same-map) -----------
        "red_tag_ready_count": int(red_tag_ready.sum()),
        "nearest_ready_defender": nearest_ready_defender,
        "defender_tag_available": bool(red_tag_ready.any()),
        "home_threatened": home_threatened,
        "our_flag_away_from_home": blue_flag_away,
        "nearest_red_to_our_flag": nearest_red_to_our_flag,
        "nearest_blue_to_our_flag": nearest_blue_to_our_flag,
        "blue_alive_count": int(blue_alive.sum()),
        "red_alive_count": int(red_alive.sum()),
    }


# --- one evaluation episode -------------------------------------------------


def run_eval_episode(policy, *, opponent: str, seed: int, device: str) -> dict:
    """One held-out episode with per-step legal-context capture."""
    # Episode mechanics come from the V6I9 evaluator, but opponent validation
    # comes from rl.evaluation.opponent_resolution: the evaluator's own
    # whitelist is scoped to OP8/OP9/OP10 and would reject the admitted mixture.
    from experiments.eval_v6i9_map_awareness import (
        _adapt_obs_for_policy,
        _done,
        _predict,
        _reset_obs,
        _unpack_step,
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
    model = policy.model if hasattr(policy, "model") else policy
    was_training = getattr(model, "training", False)
    if hasattr(model, "eval"):
        model.eval()

    try:
        _set_opponent(env, requested)
        obs = _reset_obs(env.reset())
        resolved = _get_opponent_key(env)
        if resolved != requested:
            raise RuntimeError(f"opponent drift: {requested} -> {resolved}")
        core.drain_tag_events()

        steps: list[dict] = []
        tags_against = 0      # blue agents tagged by red
        tags_for = 0          # red agents tagged by blue
        cooldown_denials = 0
        captures_blue = 0
        captures_red = 0
        drops = 0
        pickups = 0
        tagged_while_carrying = 0
        prev_carrying = 0
        ledger_blue = 0       # authoritative scores, from capture events
        ledger_red = 0
        # (kind, step_index) for every failure, so precursors can be searched
        # backward from the decision that preceded it.
        failure_events: list[tuple[str, int]] = []
        lead_seen = False

        terminated = False
        for _ in range(EPISODE_HORIZON + 8):
            ctx = legal_context(core)
            steps.append(ctx)
            step_i = len(steps) - 1
            if ctx["score_diff"] > 0:
                lead_seen = True

            action = _predict(policy, _adapt_obs_for_policy(obs, policy))
            obs, _, done, _infos = _unpack_step(env.step(action))

            captures_blue_before = captures_blue
            for e in core.drain_tag_events():
                et = e.get("event_type")
                if et == "tag_success":
                    if e.get("tagger_team") == "red":
                        tags_against += 1
                        if prev_carrying > 0:
                            tagged_while_carrying += 1
                            failure_events.append(("tagged_while_carrying", step_i))
                    else:
                        tags_for += 1
                elif et == "tag_denied":
                    cooldown_denials += 1
                elif et == "capture_scored":
                    # The ledger is authoritative. Reading core.*_score after a
                    # terminal step returns the POST-RESET value, which reports
                    # 0-0 for episodes that actually contained captures
                    # (gpu_env/_core/_rules.py::_emit_capture_events).
                    after = int(e.get("score_after", 0))
                    if e.get("scoring_team") == "blue":
                        captures_blue += 1
                        ledger_blue = max(ledger_blue, after)
                    else:
                        captures_red += 1
                        ledger_red = max(ledger_red, after)
                        failure_events.append(("capture_conceded", step_i))
                        if lead_seen:
                            failure_events.append(("lost_after_leading", step_i))

            now_carrying = int(_np(core.blue_carrying)[0].astype(bool).sum())
            # Possession events counted as transitions, so an episode with three
            # pickups contributes three -- a per-episode boolean would make the
            # conversion denominator wrong.
            if prev_carrying == 0 and now_carrying > 0:
                pickups += 1
            # A drop is losing the flag WITHOUT converting it: possession ended
            # on a step that produced no blue capture in the ledger.
            if prev_carrying > 0 and now_carrying == 0 and captures_blue == captures_blue_before:
                drops += 1
                failure_events.append(("dropped_the_flag", step_i))
            prev_carrying = now_carrying

            if _done(done):
                terminated = True
                break

        # Never trust post-terminal core state for the final score: take the
        # highest value the ledger and the pre-step observations ever agreed on.
        obs_blue = max((s["blue_score"] for s in steps), default=0.0)
        obs_red = max((s["red_score"] for s in steps), default=0.0)
        blue_score = float(max(ledger_blue, obs_blue))
        red_score = float(max(ledger_red, obs_red))
    finally:
        if hasattr(model, "train"):
            model.train(was_training)
        env.close()

    return summarize_episode(
        steps,
        failure_events=failure_events,
        opponent=requested,
        seed=seed,
        blue_score=blue_score,
        red_score=red_score,
        terminated=terminated,
        tags_against=tags_against,
        tags_for=tags_for,
        cooldown_denials=cooldown_denials,
        captures_blue=captures_blue,
        captures_red=captures_red,
        drops=drops,
        pickups=pickups,
        tagged_while_carrying=tagged_while_carrying,
    )


# Precursor search window: the decisions shortly BEFORE a failure, where a
# different coherent policy could still have chosen otherwise.
PRECURSOR_WINDOW = 30
PRECURSOR_MIN = 10


def _frac(pred, seq) -> float:
    seq = list(seq)
    if not seq:
        return 0.0
    return sum(1 for s in seq if pred(s)) / len(seq)


def _mean_finite(values) -> float:
    vals = [v for v in values if v is not None and not math.isnan(v) and not math.isinf(v)]
    return statistics.fmean(vals) if vals else float("nan")


def window_features(steps: list[dict], end: int, *, kind: str, label: str) -> dict | None:
    """Describe the legal state in the decisions leading up to ``end``.

    This is the actionable layer: everything here is observable BEFORE the
    outcome, so it can legitimately become a routing context. ``end`` is
    exclusive -- the failing decision itself is not included.
    """
    start = max(0, end - PRECURSOR_WINDOW)
    win = steps[start:end]
    if len(win) < PRECURSOR_MIN:
        return None

    carry = [s for s in win if s["carrier_present"]]
    pressures = [s["carrier_pressure"] for s in carry if not math.isnan(s["carrier_pressure"])]
    # Closing distance over the window = the threat was developing, not static.
    pressure_trend = (pressures[-1] - pressures[0]) if len(pressures) >= 2 else float("nan")

    return {
        "kind": kind,                  # "failure" or "control"
        "failure_label": label,
        "window_start": start,
        "window_end": end,             # EXCLUSIVE: the failing decision is not in the window
        "window_len": len(win),
        # --- opportunity flags: was the relevant decision even available? ---
        # Controls are matched on these so the analysis cannot rediscover that
        # carrier features are higher when a carrier exists.
        "opp_has_carrier": bool(carry),
        "opp_home_threatened": any(s["home_threatened"] for s in win),
        "opp_leading": any(s["score_diff"] > 0 for s in win),
        # --- carrier / escort fork ---------------------------------------
        "ally_carrying_frac": _frac(lambda s: s["carrier_present"], win),
        "carrier_unescorted_frac": _frac(lambda s: s["carrier_unescorted"], carry),
        "mean_escort_distance": _mean_finite(s["escort_distance"] for s in carry),
        "carrier_pressure_frac": _frac(lambda s: s["carrier_under_pressure"], carry),
        "mean_carrier_pressure": _mean_finite(pressures),
        "carrier_pressure_trend": pressure_trend,
        "carrier_pressure_increasing": bool(
            not math.isnan(pressure_trend) and pressure_trend < -0.01
        ),
        # --- cooldown fork ------------------------------------------------
        "defender_tag_available_frac": _frac(lambda s: s["defender_tag_available"], win),
        "mean_nearest_ready_defender": _mean_finite(
            s["nearest_ready_defender"] for s in carry
        ),
        "own_cooldown_bound_frac": _frac(lambda s: s["blue_cooldown_active"] > 0, win),
        # --- commitment / rotation fork -----------------------------------
        "both_forward_frac": _frac(lambda s: s["agents_forward"] >= 2, win),
        "none_forward_frac": _frac(lambda s: s["agents_forward"] == 0, win),
        "max_agents_forward": max((s["agents_forward"] for s in win), default=0),
        "mean_team_separation": _mean_finite(s["team_separation"] for s in win),
        # --- home / lead fork ---------------------------------------------
        "home_threatened_frac": _frac(lambda s: s["home_threatened"], win),
        "our_flag_away_frac": _frac(lambda s: s["our_flag_away_from_home"], win),
        "enemy_carrying_frac": _frac(lambda s: s["red_carrying"] > 0, win),
        "leading_frac": _frac(lambda s: s["score_diff"] > 0, win),
        "trailing_frac": _frac(lambda s: s["score_diff"] < 0, win),
        "time_remaining_frac_at_end": win[-1]["time_remaining_frac"],
    }


# Which control windows a given failure may legitimately be compared against.
# Without this the contrast degenerates into "carrier features are higher when
# a carrier exists", which is a tautology, not a weakness.
OPPORTUNITY_MATCH = {
    "tagged_while_carrying": "opp_has_carrier",
    "dropped_the_flag": "opp_has_carrier",
    "capture_conceded": "opp_home_threatened",
    "lost_after_leading": "opp_leading",
}


def build_windows(steps: list[dict], failure_events: list[tuple[str, int]],
                  *, episode_key: str) -> list[dict]:
    """Failure windows plus matched control windows.

    Controls matter: 'the carrier was unescorted before the failure' says
    nothing if the carrier is unescorted all the time. Control windows end at
    decisions that are NOT followed by a failure, so the comparison isolates
    what was actually different.

    ``episode_key`` is the bootstrap cluster unit -- several windows can come
    from one trajectory and are not independent samples.
    """
    windows: list[dict] = []
    failure_steps = {t for _, t in failure_events}
    for label, t in failure_events:
        w = window_features(steps, t, kind="failure", label=label)
        if w is not None:
            windows.append(w)

    # Controls: evenly spaced ends that are clear of every failure by a window.
    for end in range(PRECURSOR_WINDOW, len(steps), PRECURSOR_WINDOW):
        if any(abs(end - t) < PRECURSOR_WINDOW for t in failure_steps):
            continue
        w = window_features(steps, end, kind="control", label="none")
        if w is not None:
            windows.append(w)

    for w in windows:
        w["episode_key"] = episode_key
    return windows


def summarize_episode(steps: list[dict], **kw) -> dict:
    """Collapse per-step legal context into episode-level descriptors."""
    blue_score = kw["blue_score"]
    red_score = kw["red_score"]
    failure_events = kw.get("failure_events") or []
    carry_steps = [s for s in steps if s["carrier_present"]]

    ever_carried = bool(carry_steps)
    pressures = [s["carrier_pressure"] for s in carry_steps
                 if not math.isnan(s["carrier_pressure"])]

    # Capture latency: decisions until the first blue capture (None if never).
    first_capture = next(
        (i for i, s in enumerate(steps) if s["blue_score"] > 0), None
    )
    lead_seen = any(s["score_diff"] > 0 for s in steps)

    episode_key = f"{kw['opponent']}:{kw['seed']}"
    return {
        "_windows": build_windows(steps, failure_events, episode_key=episode_key),
        "_failure_events": [{"label": lb, "step": t} for lb, t in failure_events],
        "first_capture_step": first_capture,
        "capture_latency": first_capture if first_capture is not None else None,
        "lost_after_leading": int(lead_seen and blue_score < red_score),
        "capture_conceded": int(kw["captures_red"] > 0),
        "opponent": kw["opponent"],
        "eval_seed": kw["seed"],
        "blue_score": blue_score,
        "red_score": red_score,
        "score_margin": blue_score - red_score,
        "win": int(blue_score > red_score),
        "loss": int(blue_score < red_score),
        "draw": int(blue_score == red_score),
        "terminated": kw["terminated"],
        "episode_steps": len(steps),
        # --- event counts -------------------------------------------------
        "tags_against": kw["tags_against"],
        "tags_for": kw["tags_for"],
        "cooldown_denials": kw["cooldown_denials"],
        "captures_blue": kw["captures_blue"],
        "captures_red": kw["captures_red"],
        "drops": kw["drops"],
        "pickups": kw["pickups"],
        "tagged_while_carrying": kw["tagged_while_carrying"],
        # --- legal context descriptors ------------------------------------
        "ever_carried": int(ever_carried),
        "carried_without_scoring": int(ever_carried and blue_score <= 0),
        "carry_steps": len(carry_steps),
        "carry_frac": len(carry_steps) / max(len(steps), 1),
        "carrier_unescorted_frac": _frac(lambda s: s["carrier_unescorted"], carry_steps),
        "carrier_pressure_frac": _frac(lambda s: s["carrier_under_pressure"], carry_steps),
        "mean_carrier_pressure": (statistics.fmean(pressures) if pressures else float("nan")),
        "max_agents_forward": max((s["agents_forward"] for s in steps), default=0),
        "both_forward_frac": _frac(lambda s: s["agents_forward"] >= 2, steps),
        "none_forward_frac": _frac(lambda s: s["agents_forward"] == 0, steps),
        "mean_team_separation": statistics.fmean([s["team_separation"] for s in steps]) if steps else 0.0,
        "mean_formation_spread": statistics.fmean([s["formation_spread"] for s in steps]) if steps else 0.0,
        "blue_tagged_frac": _frac(lambda s: s["blue_tagged"] > 0, steps),
        "cooldown_active_frac": _frac(lambda s: s["blue_cooldown_active"] > 0, steps),
        "trailing_frac": _frac(lambda s: s["score_diff"] < 0, steps),
        "leading_frac": _frac(lambda s: s["score_diff"] > 0, steps),
        "conceded_first": int(
            next((s["score_diff"] < 0 for s in steps if s["score_diff"] != 0), False)
        ),
    }


# --- weakness mining --------------------------------------------------------

# Failure INDICATORS describe something that has already gone wrong. They are
# useful for finding failures but must never become routing contexts: a router
# cannot pick a strategy because the carrier will be tagged ten steps from now.
FAILURE_INDICATORS = {
    "tagged_while_carrying": lambda e: e["tagged_while_carrying"] > 0,
    "dropped_the_flag": lambda e: e["drops"] > 0,
    "lost_after_leading": lambda e: e["lost_after_leading"] == 1,
    "capture_conceded": lambda e: e["capture_conceded"] == 1,
    "carried_but_never_scored": lambda e: e["carried_without_scoring"] == 1,
    "never_carried": lambda e: e["ever_carried"] == 0,
}

# ACTIONABLE contexts are observable BEFORE the outcome, so a specialist could
# in principle be selected on them. These are the routing candidates.
ACTIONABLE_CONTEXTS = {
    "carrier_mostly_unescorted": lambda e: e["carrier_unescorted_frac"] >= 0.5,
    "carrier_often_pressured": lambda e: e["carrier_pressure_frac"] >= 0.5,
    "both_agents_forward_often": lambda e: e["both_forward_frac"] >= 0.5,
    "no_agent_forward_often": lambda e: e["none_forward_frac"] >= 0.5,
    "wide_formation": lambda e: e["mean_team_separation"] >= 0.35,
    "tight_formation": lambda e: e["mean_team_separation"] <= 0.15,
    "frequently_tagged": lambda e: e["blue_tagged_frac"] >= 0.3,
    "cooldown_bound_often": lambda e: e["cooldown_active_frac"] >= 0.5,
    "conceded_first": lambda e: e["conceded_first"] == 1,
    "trailing_most_of_episode": lambda e: e["trailing_frac"] >= 0.5,
}

CONTEXT_PREDICATES = {**ACTIONABLE_CONTEXTS, **FAILURE_INDICATORS}

# A weakness must be materially worse and not a handful of episodes.
MIN_CELL = 8
MIN_WIN_RATE_DROP = 0.15


def _wr(rows) -> float:
    rows = list(rows)
    return (sum(r["win"] for r in rows) / len(rows)) if rows else float("nan")


def _tot(rows, key: str) -> int:
    return int(sum(r[key] for r in rows))


def _ratio(numerator: int, denominator: int):
    """Aggregate ratio; null when the denominator is zero.

    Returning 0 for "no pickups ever" would read as "converted nothing", which
    is a different and much stronger claim than "never had the chance".
    """
    if denominator <= 0:
        return None
    return round(numerator / denominator, 4)


def json_safe(obj):
    """NaN/Infinity are not valid JSON -- emit null so the report stays parseable.

    NaN here means "not applicable" (e.g. carrier pressure in an episode where
    nobody ever carried), which is exactly what null conveys.
    """
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


# Precursor features examined for each recurring failure, and the decision fork
# each one would inform.
PRECURSOR_FEATURES = {
    "carrier_unescorted_frac": "escort vs continue attacking",
    "mean_escort_distance": "escort vs continue attacking",
    "carrier_pressure_increasing": "escort vs continue attacking",
    "mean_carrier_pressure": "escort vs continue attacking",
    "defender_tag_available_frac": "bait defender cooldown vs simultaneous rush",
    "mean_nearest_ready_defender": "bait defender cooldown vs simultaneous rush",
    "own_cooldown_bound_frac": "intercept vs remain forward",
    "both_forward_frac": "intercept vs remain forward",
    "none_forward_frac": "intercept vs remain forward",
    "mean_team_separation": "escort vs continue attacking",
    "home_threatened_frac": "recover our flag vs race home",
    "enemy_carrying_frac": "recover our flag vs race home",
    "our_flag_away_frac": "recover our flag vs race home",
    "leading_frac": "protect a lead vs continue pressure",
    "trailing_frac": "protect a lead vs continue pressure",
    "time_remaining_frac_at_end": "protect a lead vs continue pressure",
}

# A precursor is only actionable if it separates failure from control clearly.
MIN_WINDOW_SUPPORT = 20
MIN_PRECURSOR_SEPARATION = 0.15
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 12_345


def _numeric(v):
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float)) and not (math.isnan(v) or math.isinf(v)):
        return float(v)
    return None


def episode_clustered_ci(fail_windows: list[dict], ctrl_windows: list[dict],
                         feat: str, *, rng: np.random.Generator) -> dict:
    """Bootstrap the failure-minus-control difference, resampling EPISODES.

    Windows drawn from one episode are not independent -- several can come from
    the same trajectory. Resampling windows directly would understate the
    interval badly. The cluster unit is therefore the episode.
    """
    def by_episode(windows):
        groups: dict[str, list[float]] = defaultdict(list)
        for w in windows:
            v = _numeric(w.get(feat))
            if v is not None:
                groups[w["episode_key"]].append(v)
        return {k: v for k, v in groups.items() if v}

    f_groups, c_groups = by_episode(fail_windows), by_episode(ctrl_windows)
    f_keys, c_keys = list(f_groups), list(c_groups)
    if len(f_keys) < 2 or len(c_keys) < 2:
        return {"delta": None, "ci_low": None, "ci_high": None,
                "n_failure_episodes": len(f_keys), "n_control_episodes": len(c_keys),
                "excludes_zero": None, "insufficient_clusters": True}

    def cluster_mean(groups, keys):
        vals = [v for k in keys for v in groups[k]]
        return statistics.fmean(vals) if vals else float("nan")

    point = cluster_mean(f_groups, f_keys) - cluster_mean(c_groups, c_keys)
    deltas = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        fk = [f_keys[i] for i in rng.integers(0, len(f_keys), len(f_keys))]
        ck = [c_keys[i] for i in rng.integers(0, len(c_keys), len(c_keys))]
        d = cluster_mean(f_groups, fk) - cluster_mean(c_groups, ck)
        if not math.isnan(d):
            deltas.append(d)
    if not deltas:
        return {"delta": round(point, 4), "ci_low": None, "ci_high": None,
                "n_failure_episodes": len(f_keys), "n_control_episodes": len(c_keys),
                "excludes_zero": None, "insufficient_clusters": True}
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {
        "delta": round(point, 4),
        "ci_low": round(float(lo), 4),
        "ci_high": round(float(hi), 4),
        "n_failure_episodes": len(f_keys),
        "n_control_episodes": len(c_keys),
        "excludes_zero": bool(lo > 0 or hi < 0),
        "insufficient_clusters": False,
    }


def analyze_precursors(per_seed_windows: dict[int, list[dict]],
                       failure_labels: list[str]) -> dict:
    """For each failure, contrast the preceding state against control windows.

    Answers the question that matters for specialist training: at the moment
    before things went wrong, what was legally observable that was NOT true in
    ordinary play?
    """
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    out: dict = {}
    for label in failure_labels:
        match_key = OPPORTUNITY_MATCH.get(label)
        entry: dict = {
            "decision_forks": {},
            "per_seed": {},
            "separating_features": [],
            "opportunity_match": match_key or "none",
        }
        sep_by_feature: dict[str, list[float]] = defaultdict(list)
        ci_by_feature: dict[str, list[bool]] = defaultdict(list)

        for seed, windows in per_seed_windows.items():
            fail = [w for w in windows
                    if w["kind"] == "failure" and w["failure_label"] == label]
            ctrl = [w for w in windows if w["kind"] == "control"]
            # Opportunity matching: only compare against controls where the same
            # decision was actually available.
            if match_key:
                ctrl = [w for w in ctrl if w.get(match_key)]
            seed_entry = {
                "n_failure_windows": len(fail),
                "n_control_windows_matched": len(ctrl),
                "features": {},
            }
            if len(fail) >= MIN_WINDOW_SUPPORT and len(ctrl) >= MIN_WINDOW_SUPPORT:
                for feat in PRECURSOR_FEATURES:
                    f_mean = _mean_finite(_numeric(w.get(feat)) for w in fail)
                    c_mean = _mean_finite(_numeric(w.get(feat)) for w in ctrl)
                    if math.isnan(f_mean) or math.isnan(c_mean):
                        continue
                    delta = f_mean - c_mean
                    ci = episode_clustered_ci(fail, ctrl, feat, rng=rng)
                    big = abs(delta) >= MIN_PRECURSOR_SEPARATION
                    seed_entry["features"][feat] = {
                        "failure_mean": round(f_mean, 4),
                        "control_mean": round(c_mean, 4),
                        "delta": round(delta, 4),
                        "episode_clustered_ci": ci,
                        "separating": bool(big),
                        # Both the frozen effect-size rule AND an interval that
                        # excludes zero once episode dependence is accounted for.
                        "separating_and_ci_excludes_zero": bool(big and ci.get("excludes_zero")),
                    }
                    if big:
                        sep_by_feature[feat].append(delta)
                        ci_by_feature[feat].append(bool(ci.get("excludes_zero")))
            entry["per_seed"][str(seed)] = seed_entry

        for feat, deltas in sep_by_feature.items():
            if len(deltas) >= 2:  # reproduced across at least two training seeds
                n_ci = sum(1 for x in ci_by_feature[feat] if x)
                entry["separating_features"].append({
                    "feature": feat,
                    "decision_fork": PRECURSOR_FEATURES[feat],
                    "n_seeds": len(deltas),
                    "n_seeds_ci_excludes_zero": n_ci,
                    "mean_delta": round(statistics.fmean(deltas), 4),
                })
                entry["decision_forks"].setdefault(PRECURSOR_FEATURES[feat], []).append(feat)
        entry["separating_features"].sort(key=lambda d: -abs(d["mean_delta"]))
        entry["has_actionable_precursor"] = bool(entry["separating_features"])
        entry["has_ci_backed_precursor"] = any(
            f["n_seeds_ci_excludes_zero"] >= 2 for f in entry["separating_features"]
        )
        out[label] = entry
    return out


def mine_weaknesses(per_seed: dict[int, list[dict]]) -> dict:
    """Find context conditions under which win rate drops, per seed."""
    findings: dict[str, dict] = {}
    for name, pred in CONTEXT_PREDICATES.items():
        entry = {"per_seed": {}, "seeds_with_weakness": []}
        for seed, rows in per_seed.items():
            hit = [r for r in rows if pred(r)]
            miss = [r for r in rows if not pred(r)]
            wr_hit, wr_miss = _wr(hit), _wr(miss)
            drop = (wr_miss - wr_hit) if (hit and miss) else float("nan")
            is_weak = bool(
                len(hit) >= MIN_CELL and len(miss) >= MIN_CELL
                and not math.isnan(drop) and drop >= MIN_WIN_RATE_DROP
            )
            entry["per_seed"][str(seed)] = {
                "n_when_true": len(hit),
                "n_when_false": len(miss),
                "win_rate_when_true": round(wr_hit, 4) if hit else None,
                "win_rate_when_false": round(wr_miss, 4) if miss else None,
                "win_rate_drop": round(drop, 4) if not math.isnan(drop) else None,
                "prevalence": round(len(hit) / max(len(rows), 1), 4),
                "is_weakness": is_weak,
            }
            if is_weak:
                entry["seeds_with_weakness"].append(seed)
        entry["n_seeds_with_weakness"] = len(entry["seeds_with_weakness"])
        entry["recurring"] = len(entry["seeds_with_weakness"]) >= 2
        drops = [
            v["win_rate_drop"] for v in entry["per_seed"].values()
            if v["win_rate_drop"] is not None
        ]
        entry["mean_win_rate_drop"] = round(statistics.fmean(drops), 4) if drops else None
        findings[name] = entry
    return findings


# --- main -------------------------------------------------------------------


class ChannelResolutionError(RuntimeError):
    """Raised when a checkpoint's CNN input width cannot be established."""


def resolve_cnn_channels(payload: dict, *, context: str = "") -> int:
    """Determine the CNN input width a checkpoint was TRAINED with.

    Resolution order, most authoritative first:

      1. explicit observation-schema metadata on the checkpoint
      2. the shape of ``actor_cnn.conv.0.weight`` -- the weights themselves
      3. fail closed

    The previous implementation used ``cfg.get("cnn_channels", 0) or 8`` and
    then a try/except that retried with 7. Neither key existed, so the ``or 8``
    silently invented a width and the loader zero-expanded a 7-channel conv to
    8. It happened to be behaviourally equivalent (the new channel's weights
    were zero and the obstacle plane is zero on map_a_open), but the evaluation
    log then read ``channels=8`` as though the checkpoint had asserted it.
    A formal evaluation must know why a compatibility shim is needed rather
    than discovering the architecture through exception handling.
    """
    for key in ("cnn_channels", "num_cnn_channels", "obs_cnn_channels"):
        raw = payload.get(key, payload.get("cfg", {}).get(key))
        if raw:
            return int(raw)

    state = payload.get("model_state_dict") or {}
    for name, tensor in state.items():
        if name.endswith("actor_cnn.conv.0.weight") and hasattr(tensor, "shape"):
            if len(tensor.shape) >= 2:
                return int(tensor.shape[1])

    raise ChannelResolutionError(
        f"Cannot establish CNN input width for {context or 'checkpoint'}: no "
        "observation-schema metadata and no actor_cnn.conv.0.weight in the "
        "state dict. Refusing to guess a width for a formal evaluation."
    )


def evaluate_seed(seed: int, episodes: int, device: str) -> list[dict]:
    from rl.evaluation.checkpoint import load_policy
    from rl.custom_ppo.checkpoints.loader import read_checkpoint_payload
    from rl.ruleset_identity import ARTIFACT_IDENTITY_KEY

    ckpt = artifact_dir_for(seed) / "ckpts" / f"final_{run_tag_for(seed)}.zip"
    if not ckpt.is_file():
        raise FileNotFoundError(f"missing preregistered primary checkpoint: {ckpt}")

    payload = read_checkpoint_payload(str(ckpt), map_location="cpu")
    ai = payload.get(ARTIFACT_IDENTITY_KEY, {})
    if str(ai.get("ruleset_id")) != RULESET_ID:
        raise ValueError(f"{ckpt}: ruleset {ai.get('ruleset_id')!r} != {RULESET_ID!r}")
    if int(payload.get("global_step", 0)) < 1_000_000:
        raise ValueError(
            f"{ckpt}: global_step={payload.get('global_step')} is not the "
            "preregistered 1,000,000-step policy"
        )

    channels = resolve_cnn_channels(payload, context=str(ckpt))
    policy = load_policy(str(ckpt), device=device, num_cnn_channels=channels)

    rows: list[dict] = []
    for opp in OPPONENTS:
        for i in range(episodes):
            ev_seed = EVAL_SEED_BASE + i
            row = run_eval_episode(policy, opponent=opp, seed=ev_seed, device=device)
            row["train_seed"] = seed
            rows.append(row)
        wr = _wr([r for r in rows if r["opponent"] == opp])
        print(f"  seed {seed} vs {opp}: win_rate={wr:.3f} (n={episodes})")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=EPISODES_PER_CELL)
    ap.add_argument("--device", default=DEVICE)
    ap.add_argument("--seeds", type=int, nargs="*", default=list(G0_SEEDS))
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()
    print("=" * 78)
    print(f"G0-v2 EVALUATION  seeds={args.seeds}  opponents={OPPONENTS}")
    print(f"map={CANONICAL_MAP}  episodes/cell={args.episodes}  "
          f"held-out eval seeds {EVAL_SEED_BASE}..{EVAL_SEED_BASE + args.episodes - 1}")
    print("=" * 78)

    per_seed: dict[int, list[dict]] = {}
    for seed in args.seeds:
        print(f"\n--- training seed {seed} (1,000,000-step policy) ---")
        per_seed[seed] = evaluate_seed(seed, args.episodes, args.device)

    # --- LAYER 1: baseline competence ---------------------------------------
    competence = {}
    for seed, rows in per_seed.items():
        by_opp = {}
        for opp in OPPONENTS:
            sub = [r for r in rows if r["opponent"] == opp]
            n = max(len(sub), 1)
            lat = [r["capture_latency"] for r in sub if r["capture_latency"] is not None]
            by_opp[opp] = {
                "n": len(sub),
                "win_rate": round(_wr(sub), 4),
                "draw_rate": round(sum(r["draw"] for r in sub) / n, 4),
                "loss_rate": round(sum(r["loss"] for r in sub) / n, 4),
                "mean_margin": round(statistics.fmean([r["score_margin"] for r in sub]), 4) if sub else None,
                "capture_rate": round(sum(1 for r in sub if r["captures_blue"] > 0) / n, 4),
                "mean_captures": round(statistics.fmean([r["captures_blue"] for r in sub]), 4) if sub else None,
                "mean_captures_conceded": round(statistics.fmean([r["captures_red"] for r in sub]), 4) if sub else None,
                "median_capture_latency": round(statistics.median(lat), 1) if lat else None,
                "mean_tags_for": round(statistics.fmean([r["tags_for"] for r in sub]), 4) if sub else None,
                "mean_tags_against": round(statistics.fmean([r["tags_against"] for r in sub]), 4) if sub else None,
                "mean_cooldown_denials": round(statistics.fmean([r["cooldown_denials"] for r in sub]), 4) if sub else None,
                "pickup_rate": round(sum(r["ever_carried"] for r in sub) / n, 4),
                "mean_drops": round(statistics.fmean([r["drops"] for r in sub]), 4) if sub else None,
                "mean_pickups": round(statistics.fmean([r["pickups"] for r in sub]), 4) if sub else None,
                # Ratios are formed from AGGREGATE counts, not by averaging
                # per-episode ratios: an episode with one pickup and one capture
                # must not carry the same weight as one with ten of each.
                # Zero pickups yields null, never 0 -- "never had the flag" is
                # not the same statement as "never converted it".
                "total_blue_captures": _tot(sub, "captures_blue"),
                "total_red_captures": _tot(sub, "captures_red"),
                "total_pickups": _tot(sub, "pickups"),
                "total_drops": _tot(sub, "drops"),
                "capture_conversion": _ratio(_tot(sub, "captures_blue"), _tot(sub, "pickups")),
                "drop_per_pickup": _ratio(_tot(sub, "drops"), _tot(sub, "pickups")),
                # Explicitly BLUE minus RED, on aggregate counts.
                "net_captures": _tot(sub, "captures_blue") - _tot(sub, "captures_red"),
                "offensive_commitment": round(
                    statistics.fmean([r["both_forward_frac"] for r in sub]), 4) if sub else None,
                "defensive_commitment": round(
                    statistics.fmean([r["none_forward_frac"] for r in sub]), 4) if sub else None,
            }
        wrs = [v["win_rate"] for v in by_opp.values()]
        competence[str(seed)] = {
            "overall_win_rate": round(_wr(rows), 4),
            "per_opponent": by_opp,
            "worst_opponent": min(by_opp, key=lambda o: by_opp[o]["win_rate"]),
            "best_opponent": max(by_opp, key=lambda o: by_opp[o]["win_rate"]),
            "win_rate_spread": round(max(wrs) - min(wrs), 4),
            "opponents_above_50pct": sum(1 for w in wrs if w > 0.5),
        }

    # --- LAYER 2: recurring weakness discovery ------------------------------
    findings = mine_weaknesses(per_seed)
    recurring = {k: v for k, v in findings.items() if v["recurring"]}

    # --- LAYER 3: actionable precursor analysis -----------------------------
    per_seed_windows = {
        seed: [w for r in rows for w in r.get("_windows", [])]
        for seed, rows in per_seed.items()
    }
    # Search precursors for every failure indicator that actually occurred,
    # whether or not it also cleared the win-rate bar as an episode descriptor.
    observed_failures = sorted({
        w["failure_label"] for ws in per_seed_windows.values()
        for w in ws if w["kind"] == "failure"
    })
    precursors = analyze_precursors(per_seed_windows, observed_failures)

    all_wr = [competence[str(s)]["overall_win_rate"] for s in per_seed]
    broadly_competent = bool(
        all(w >= 0.5 for w in all_wr)
        and all(competence[str(s)]["opponents_above_50pct"] >= 5 for s in per_seed)
    )

    report = {
        "evaluation": "G0-v2",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "map": CANONICAL_MAP,
        "ruleset_id": RULESET_ID,
        "opponents": list(OPPONENTS),
        "episodes_per_cell": args.episodes,
        "held_out_eval_seeds": [EVAL_SEED_BASE, EVAL_SEED_BASE + args.episodes - 1],
        "training_seeds": list(per_seed),
        "preregistered_primary_step": 1_000_000,
        "preregistered_thresholds": {
            "min_episodes_each_side": MIN_CELL,
            "min_win_rate_drop": MIN_WIN_RATE_DROP,
            "min_seeds_reproducing": 2,
            "precursor_window_decisions": PRECURSOR_WINDOW,
            "min_window_support": MIN_WINDOW_SUPPORT,
            "min_precursor_separation": MIN_PRECURSOR_SEPARATION,
            "frozen_before_results": True,
        },
        "layer1_competence": competence,
        "broadly_competent": broadly_competent,
        "layer2_weakness_findings": findings,
        "recurring_weaknesses": sorted(
            recurring, key=lambda k: -(recurring[k]["mean_win_rate_drop"] or 0)
        ),
        "failure_indicators": sorted(FAILURE_INDICATORS),
        "actionable_contexts": sorted(ACTIONABLE_CONTEXTS),
        "layer3_precursors": precursors,
        "context_features_are_legal_same_map": True,
        "opponent_identity_used_as_context": False,
        "wall_seconds": round(time.time() - started, 2),
    }

    (OUT_DIR / "g0_v2_evaluation_report.json").write_text(
        json.dumps(json_safe(report), indent=2, default=str, allow_nan=False),
        encoding="utf-8")

    import csv
    # "_"-prefixed keys hold nested per-window structures; they belong in the
    # JSON report and the window CSV, not in the flat episode table.
    rows_all = [{k: v for k, v in r.items() if not k.startswith("_")}
                for rows in per_seed.values() for r in rows]
    if rows_all:
        with open(OUT_DIR / "episode_rows.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_all[0].keys()))
            w.writeheader()
            w.writerows(rows_all)

    win_rows = [
        {"train_seed": seed, "opponent": r["opponent"], "eval_seed": r["eval_seed"], **w}
        for seed, rows in per_seed.items() for r in rows for w in r.get("_windows", [])
    ]
    if win_rows:
        with open(OUT_DIR / "precursor_windows.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(win_rows[0].keys()))
            w.writeheader()
            w.writerows(win_rows)

    print("\n" + "=" * 78)
    print(f"broadly_competent: {broadly_competent}")
    for s in per_seed:
        c = competence[str(s)]
        print(f"  seed {s}: WR={c['overall_win_rate']:.3f} "
              f"worst={c['worst_opponent']} spread={c['win_rate_spread']:.3f}")
    print(f"\nrecurring weaknesses (>=2 seeds): {report['recurring_weaknesses'] or 'none'}")
    for label, e in precursors.items():
        if e["has_actionable_precursor"]:
            top = e["separating_features"][:3]
            print(f"  precursors for {label}:")
            for t in top:
                print(f"    {t['feature']}: delta={t['mean_delta']:+.3f} "
                      f"({t['n_seeds']} seeds) -> fork: {t['decision_fork']}")
    print(f"\nreport: {OUT_DIR / 'g0_v2_evaluation_report.json'}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
