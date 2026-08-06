"""C2 Stage 2 — lag-structured audit on the original G0-V5 discovery block.

Contract: artifacts/c2_qualification/C2_QUALIFICATION_FROZEN.json
Doc:      docs/c2-qualification-preregistration.md

Replays seeds 9400000+ with the three frozen G0-V5 policies, captures per-step
legal context, aggregates into lag bands [-30,-20), [-20,-10), [-10,0) around
each carrier-failure outcome (outcome step excluded), opportunity-matches
controls, and applies the ten frozen qualification criteria.

Stop artifacts (exactly one of):
  C2_SELECTED_CANDIDATE.json
  C2_NO_QUALIFIED_CANDIDATE.json
  C2_HEADROOM_FAIL.json   (only if a selected candidate later fails headroom —
                           under the freeze, headroom is a qualification gate,
                           so zero-qualify covers that case)

Run:  python experiments/run_c2_lag_audit.py
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pathlib
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

from experiments.run_g0_v2_evaluation import (  # noqa: E402
    OPPORTUNITY_MATCH,
    PRECURSOR_MIN,
    V2_RULES,
    _frac,
    _mean_finite,
    _numeric,
    episode_clustered_ci,
    legal_context,
    _np,
)
from experiments.run_g0_v2_seed import (  # noqa: E402
    AGENTS,
    CANONICAL_MAP,
    EPISODE_HORIZON,
    OPPONENTS,
)
from experiments.run_c2_discovery import _score_stratum  # noqa: E402
from experiments.run_c2_step_replay import (  # noqa: E402
    _json_safe,
    equivalence_gate,
)

FROZEN_PATH = PROJECT_ROOT / "artifacts" / "c2_qualification" / "C2_QUALIFICATION_FROZEN.json"
OUT_DIR = PROJECT_ROOT / "artifacts" / "c2_qualification"

G0_SEEDS = (3_200_001, 3_200_002, 3_200_003)
EVAL_SEED_BASE = 9_400_000  # original discovery block — replay, not fresh data

# Half-open bands relative to outcome t; t itself excluded by end=0 exclusive.
LAG_BANDS = (
    ("earliest", -30, -20),
    ("middle", -20, -10),
    ("latest", -10, 0),
)

CARRIER_FAILURES = ("tagged_while_carrying", "dropped_the_flag")

# Frozen audit list (criterion features_audited.must_include).
AUDIT_FEATURES = (
    "carrier_unescorted_frac",
    "mean_escort_distance",
    "defender_tag_available_frac",
    "mean_nearest_ready_defender",
    "mean_carrier_pressure",
    "carrier_pressure_increasing",
    "carrier_pressure_trend",
    "both_forward_frac",
    "none_forward_frac",
    "mean_team_separation",
    "own_cooldown_bound_frac",
    "intervention_margin",
    "mate_can_intervene",
)

EFFECT_SIZE_THR = 0.15
HEADROOM_MIN = 0.20
GATE1_EFFECT = 0.10
ACTIONABILITY_THR = 0.30
ONSET_PREVALENCE_MIN = 0.02
MIN_SUPPORT = 30
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 12_345
# "Present" for lag-decay: same sign as latest and non-trivial magnitude.
LAG_DECAY_MIN_ABS = 0.05



STAGE2_LOCK = OUT_DIR / "STAGE2_RUNNING.lock"


def _runner_commit() -> str:
    """Git commit of this file, recorded in every official output."""
    import subprocess
    try:
        out = subprocess.run(
            ["git", "log", "-1", "--format=%H", "--", str(pathlib.Path(__file__).name)],
            cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=15,
        )
        return (out.stdout or "").strip() or "unknown"
    except Exception:
        return "unknown"


def acquire_stage2_lock() -> dict:
    """Refuse to start if another Stage 2 run is active.

    Two sessions independently built this experiment once already. A lock with
    the owning PID and runner commit makes a silent double-run impossible
    rather than merely unlikely.
    """
    if STAGE2_LOCK.exists():
        try:
            held = json.loads(STAGE2_LOCK.read_text(encoding="utf-8"))
        except Exception:
            held = {"raw": STAGE2_LOCK.read_text(encoding="utf-8", errors="replace")}
        alive = False
        pid = held.get("pid")
        if isinstance(pid, int):
            try:
                os.kill(pid, 0)
                alive = True
            except OSError:
                alive = False
        if alive:
            raise RuntimeError(
                f"Stage 2 already running: {held}. Refusing to start a second "
                f"run into artifacts/c2_qualification/. Remove {STAGE2_LOCK} only "
                "if that process is genuinely dead."
            )
        print(f"[lock] stale lock from dead pid {pid}; reclaiming")
    info = {
        "pid": os.getpid(),
        "runner_commit": _runner_commit(),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    STAGE2_LOCK.parent.mkdir(parents=True, exist_ok=True)
    STAGE2_LOCK.write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


def release_stage2_lock() -> None:
    try:
        STAGE2_LOCK.unlink()
    except FileNotFoundError:
        pass


def load_frozen() -> dict:
    return json.loads(FROZEN_PATH.read_text(encoding="utf-8"))


def enrich_step(ctx: dict) -> dict:
    """Add intervention_margin / mate_can_intervene from legal_context fields."""
    out = dict(ctx)
    ed = ctx.get("escort_distance")
    nd = ctx.get("nearest_ready_defender")
    if (
        isinstance(ed, (int, float)) and isinstance(nd, (int, float))
        and not math.isnan(ed) and not math.isnan(nd)
        and not math.isinf(ed) and not math.isinf(nd)
    ):
        out["intervention_margin"] = float(nd - ed)
        out["mate_can_intervene"] = bool(ed <= nd)
    else:
        out["intervention_margin"] = float("nan")
        out["mate_can_intervene"] = False
    return out


def band_aggregate(steps: list[dict], t: int, start_off: int, end_off: int) -> dict | None:
    """Aggregate legal features on steps[t+start_off : t+end_off] (end exclusive)."""
    lo = max(0, t + start_off)
    hi = min(len(steps), t + end_off)
    if hi <= lo:
        return None
    win = steps[lo:hi]
    if len(win) < max(3, PRECURSOR_MIN // 3):
        return None

    carry = [s for s in win if s["carrier_present"]]
    pressures = [
        s["carrier_pressure"] for s in carry
        if not math.isnan(s.get("carrier_pressure", float("nan")))
    ]
    pressure_trend = (pressures[-1] - pressures[0]) if len(pressures) >= 2 else float("nan")
    margins = [s["intervention_margin"] for s in carry]

    score_ref = win[-1]["score_diff"]
    return {
        "window_start": lo,
        "window_end": hi,
        "window_len": len(win),
        "opp_has_carrier": bool(carry),
        "opp_home_threatened": any(s["home_threatened"] for s in win),
        "opp_leading": any(s["score_diff"] > 0 for s in win),
        "score_diff_at_end": score_ref,
        "score_stratum": _score_stratum(score_ref),
        "carrier_unescorted_frac": _frac(lambda s: s["carrier_unescorted"], carry),
        "mean_escort_distance": _mean_finite(s["escort_distance"] for s in carry),
        "mean_carrier_pressure": _mean_finite(pressures),
        "carrier_pressure_trend": pressure_trend,
        "carrier_pressure_increasing": bool(
            not math.isnan(pressure_trend) and pressure_trend < -0.01
        ),
        "defender_tag_available_frac": _frac(lambda s: s["defender_tag_available"], win),
        "mean_nearest_ready_defender": _mean_finite(
            s["nearest_ready_defender"] for s in carry
        ),
        "own_cooldown_bound_frac": _frac(lambda s: s["blue_cooldown_active"] > 0, win),
        "both_forward_frac": _frac(lambda s: s["agents_forward"] >= 2, win),
        "none_forward_frac": _frac(lambda s: s["agents_forward"] == 0, win),
        "mean_team_separation": _mean_finite(s["team_separation"] for s in win),
        "intervention_margin": _mean_finite(margins),
        "mate_can_intervene": _frac(lambda s: s.get("mate_can_intervene"), carry),
        "leading_frac": _frac(lambda s: s["score_diff"] > 0, win),
        "trailing_frac": _frac(lambda s: s["score_diff"] < 0, win),
    }


def replay_episode(policy, *, opponent: str, seed: int, device: str) -> dict:
    """Same mechanics as run_eval_episode, but returns steps + failure_events."""
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

    steps: list[dict] = []
    failure_events: list[tuple[str, int]] = []
    try:
        _set_opponent(env, requested)
        obs = _reset_obs(env.reset())
        if _get_opponent_key(env) != requested:
            raise RuntimeError(f"opponent drift: {requested}")
        core.drain_tag_events()

        prev_carrying = 0
        captures_blue = 0
        lead_seen = False

        for _ in range(EPISODE_HORIZON + 8):
            ctx = enrich_step(legal_context(core))
            steps.append(ctx)
            step_i = len(steps) - 1
            if ctx["score_diff"] > 0:
                lead_seen = True

            action = _predict(policy, _adapt_obs_for_policy(obs, policy))
            obs, _, done, _infos = _unpack_step(env.step(action))

            captures_blue_before = captures_blue
            for e in core.drain_tag_events():
                et = e.get("event_type")
                if et == "tag_success" and e.get("tagger_team") == "red":
                    if prev_carrying > 0:
                        failure_events.append(("tagged_while_carrying", step_i))
                elif et == "capture_scored":
                    if e.get("scoring_team") == "blue":
                        captures_blue += 1
                    else:
                        failure_events.append(("capture_conceded", step_i))
                        if lead_seen:
                            failure_events.append(("lost_after_leading", step_i))

            now_carrying = int(_np(core.blue_carrying)[0].astype(bool).sum())
            if prev_carrying > 0 and now_carrying == 0 and captures_blue == captures_blue_before:
                failure_events.append(("dropped_the_flag", step_i))
            prev_carrying = now_carrying

            if _done(done):
                break
    finally:
        if hasattr(model, "train"):
            model.train(was_training)
        env.close()

    return {
        "steps": steps,
        "failure_events": failure_events,
        "opponent": requested,
        "eval_seed": seed,
    }


def build_band_windows(ep: dict, *, train_seed: int) -> list[dict]:
    """Failure + control band rows for one episode."""
    steps = ep["steps"]
    failure_events = ep["failure_events"]
    episode_key = f"{ep['opponent']}:{ep['eval_seed']}"
    failure_steps = {t for _, t in failure_events}
    rows: list[dict] = []

    def emit(kind: str, label: str, t: int):
        for band_name, a, b in LAG_BANDS:
            feat = band_aggregate(steps, t, a, b)
            if feat is None:
                continue
            row = {
                "kind": kind,
                "failure_label": label,
                "outcome_step": t,
                "band": band_name,
                "band_start_off": a,
                "band_end_off": b,
                "episode_key": episode_key,
                "opponent": ep["opponent"],
                "eval_seed": ep["eval_seed"],
                "train_seed": train_seed,
                **feat,
            }
            rows.append(row)

    for label, t in failure_events:
        if label in CARRIER_FAILURES:
            emit("failure", label, t)

    # Controls: same spacing as the original discovery harness.
    for end in range(30, len(steps), 30):
        if any(abs(end - t) < 30 for t in failure_steps):
            continue
        emit("control", "none", end)

    return rows


def _mean_feat(windows: list[dict], feat: str) -> float | None:
    vals = [_numeric(w.get(feat)) for w in windows]
    vals = [v for v in vals if v is not None]
    return statistics.fmean(vals) if vals else None


def evaluate_cell(
    fail: list[dict],
    ctrl: list[dict],
    feat: str,
    *,
    match_key: str,
    rng: np.random.Generator,
) -> dict:
    fail_m = [w for w in fail if w.get(match_key)]
    ctrl_m = [w for w in ctrl if w.get(match_key)]
    n_f, n_c = len(fail_m), len(ctrl_m)
    mf = _mean_feat(fail_m, feat)
    mc = _mean_feat(ctrl_m, feat)
    delta = (mf - mc) if (mf is not None and mc is not None) else None
    ci = episode_clustered_ci(fail_m, ctrl_m, feat, rng=rng)
    return {
        "n_failure": n_f,
        "n_control": n_c,
        "mean_failure": None if mf is None else round(mf, 4),
        "mean_control": None if mc is None else round(mc, 4),
        "delta": None if delta is None else round(delta, 4),
        "ci_low": ci.get("ci_low"),
        "ci_high": ci.get("ci_high"),
        "excludes_zero": ci.get("excludes_zero"),
        "n_failure_episodes": ci.get("n_failure_episodes"),
        "n_control_episodes": ci.get("n_control_episodes"),
        "support_ok": n_f >= MIN_SUPPORT and n_c >= MIN_SUPPORT,
    }


def qualify_candidate(
    label: str,
    feat: str,
    per_seed_latest: dict[int, dict],
    per_seed_earliest: dict[int, dict],
    stratum_ok: dict[int, bool],
    *,
    n_episodes_per_seed: dict[int, int],
    n_onsets_per_seed: dict[int, int],
    mate_intervene_frac: dict[int, float],
    headroom_per_seed: dict[int, float],
) -> dict:
    """Apply all ten frozen criteria to one (label, feature) candidate."""
    seeds = list(G0_SEEDS)
    deltas = []
    for s in seeds:
        d = per_seed_latest[s].get("delta")
        deltas.append(d)

    # Criterion 3: same direction across all 3 (ignore None).
    signed = [d for d in deltas if d is not None]
    if len(signed) < 3:
        direction_ok = False
        common_sign = 0
    else:
        signs = [1 if d > 0 else (-1 if d < 0 else 0) for d in signed]
        direction_ok = 0 not in signs and len(set(signs)) == 1
        common_sign = signs[0] if direction_ok else 0

    # Criterion 4: |delta| >= 0.15 in >= 2/3
    effect_pass_n = sum(
        1 for d in deltas if d is not None and abs(d) >= EFFECT_SIZE_THR
    )
    effect_ok = effect_pass_n >= 2

    # Criterion 5: CI excludes zero in >= 2/3
    ci_pass_n = sum(
        1 for s in seeds if per_seed_latest[s].get("excludes_zero") is True
    )
    ci_ok = ci_pass_n >= 2

    # Criterion 6: support per policy on latest band
    support_ok = all(per_seed_latest[s].get("support_ok") for s in seeds)

    # Criterion 7: headroom
    # Use min across seeds (worst case) — niche must be trainable everywhere.
    headrooms = [headroom_per_seed[s] for s in seeds]
    headroom = min(headrooms) if headrooms else 0.0
    headroom_ok = headroom >= HEADROOM_MIN and headroom >= 2 * GATE1_EFFECT

    # Criterion 8: actionability among failure onsets
    act_fracs = [mate_intervene_frac[s] for s in seeds]
    act_mean = statistics.fmean(act_fracs) if act_fracs else 0.0
    actionability_ok = all(f >= ACTIONABILITY_THR for f in act_fracs)

    # Criterion 9a: lag decay — earliest same direction as latest, present
    lag_ok_flags = []
    for s in seeds:
        d_l = per_seed_latest[s].get("delta")
        d_e = per_seed_earliest[s].get("delta")
        if d_l is None or d_e is None or d_l == 0:
            lag_ok_flags.append(False)
            continue
        same = (d_e > 0) == (d_l > 0)
        present = abs(d_e) >= LAG_DECAY_MIN_ABS
        lag_ok_flags.append(same and present)
    # Require lag-decay on >= 2/3 policies (aligned with effect/CI bars).
    lag_ok = sum(lag_ok_flags) >= 2

    # Criterion 9b: score stratum — survives in >=1 stratum for >=2/3 policies
    stratum_ok_n = sum(1 for s in seeds if stratum_ok.get(s))
    stratum_pass = stratum_ok_n >= 2

    # Criterion 10: natural support
    prevalence_ok = all(
        (n_onsets_per_seed[s] / max(n_episodes_per_seed[s], 1)) >= ONSET_PREVALENCE_MIN
        for s in seeds
    )
    onset_count_ok = all(n_onsets_per_seed[s] >= MIN_SUPPORT for s in seeds)
    natural_ok = prevalence_ok and onset_count_ok

    # Criteria 1–2 enforced by construction.
    checks = {
        "1_temporal_precedence": True,
        "2_window_level_unit": True,
        "3_replication": direction_ok,
        "4_effect_size": effect_ok,
        "5_uncertainty": ci_ok,
        "6_support": support_ok,
        "7_headroom": headroom_ok,
        "8_actionability": actionability_ok,
        "9_lag_decay": lag_ok,
        "9_score_stratum": stratum_pass,
        "10_natural_support": natural_ok,
    }
    if not lag_ok:
        lag_class = "OUTCOME_CONTAMINATION"
    else:
        lag_class = "PASS"

    qualified = all(checks.values())
    abs_effect = statistics.fmean([abs(d) for d in signed]) if signed else 0.0
    ci_sep = statistics.fmean([
        abs(per_seed_latest[s]["delta"])
        for s in seeds
        if per_seed_latest[s].get("delta") is not None
        and per_seed_latest[s].get("excludes_zero")
    ] or [0.0])

    return {
        "failure_label": label,
        "feature": feat,
        "qualified": qualified,
        "checks": checks,
        "lag_decay_classification": lag_class,
        "replication": f"{sum(1 for _ in signed if True)}/3" if direction_ok else f"direction_fail",
        "replication_n": 3 if direction_ok else sum(1 for d in signed if d is not None and (d > 0) == (signed[0] > 0)),
        "effect_pass_n": effect_pass_n,
        "ci_pass_n": ci_pass_n,
        "common_sign": common_sign,
        "mean_abs_delta_latest": round(abs_effect, 4),
        "ci_separation_score": round(ci_sep, 4),
        "headroom": round(headroom, 4),
        "headroom_per_seed": {str(s): round(headroom_per_seed[s], 4) for s in seeds},
        "actionability_frac": round(act_mean, 4),
        "onsets_per_seed": {str(s): n_onsets_per_seed[s] for s in seeds},
        "prevalence_per_seed": {
            str(s): round(n_onsets_per_seed[s] / max(n_episodes_per_seed[s], 1), 4)
            for s in seeds
        },
        "per_seed_latest": {str(s): per_seed_latest[s] for s in seeds},
        "per_seed_earliest": {str(s): per_seed_earliest[s] for s in seeds},
        "carrier_mostly_unescorted_note": (
            "starts REJECTED; qualifies only if this audit independently passes"
            if feat == "carrier_unescorted_frac" else None
        ),
    }


def rank_key(c: dict) -> tuple:
    """Frozen ranking_order — higher is better; used with sort reverse=True."""
    return (
        c["replication_n"],
        c["mean_abs_delta_latest"],
        c["ci_separation_score"],
        statistics.fmean(c["prevalence_per_seed"].values()),
        c["headroom"],
        c["actionability_frac"],
    )


def main() -> int:
    frozen = load_frozen()
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=30,
                    help="episodes per opponent (discovery used 30)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seeds", type=int, nargs="*", default=list(G0_SEEDS))
    args = ap.parse_args()

    from experiments.run_g0_v2_evaluation import resolve_cnn_channels
    from rl.custom_ppo.checkpoints.loader import read_checkpoint_payload
    from rl.evaluation.checkpoint import load_policy

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lock_info = acquire_stage2_lock()
    started = time.time()
    print("=" * 78)
    print("C2 STAGE 2 — lag-structured audit (discovery replay)")
    print(f"runner commit: {lock_info['runner_commit'][:12]}  pid={lock_info['pid']}")
    print(f"contract: {FROZEN_PATH.relative_to(PROJECT_ROOT)}")
    print(f"policies={args.seeds}  opponents={OPPONENTS}  episodes/cell={args.episodes}")
    print(f"replay seeds {EVAL_SEED_BASE}..{EVAL_SEED_BASE + args.episodes - 1}")
    print(f"bands={list(LAG_BANDS)}  features={len(AUDIT_FEATURES)}")
    print("carrier_mostly_unescorted starts REJECTED unless independently qualified")
    print("=" * 78)
    sys.stdout.flush()

    all_windows: list[dict] = []
    n_episodes: dict[int, int] = {s: 0 for s in args.seeds}
    # REPLAY-EQUIVALENCE EVIDENCE.
    # replay_episode() reimplements the original run_eval_episode loop rather
    # than calling it, so nothing structurally guarantees these trajectories are
    # the ones the frozen discovery analysed. Rebuilding the original [-30,0)
    # windows with the ORIGINAL build_windows/window_features and comparing them
    # to precursor_windows.csv is what turns that assumption into a check. It is
    # accumulated during the replay and gated on BEFORE any analysis runs.
    rebuilt_frozen_windows: list[dict] = []

    for seed in args.seeds:
        tag = f"g0_v5_long_seed{seed}"
        ckpt = PROJECT_ROOT / "artifacts" / "g0_v5_long" / tag / "ckpts" / f"final_{tag}.zip"
        payload = read_checkpoint_payload(str(ckpt), map_location="cpu")
        gstep = int(payload.get("global_step", 0))
        if gstep < 1_000_000:
            raise ValueError(f"{ckpt}: not the preregistered 1M checkpoint ({gstep})")
        print(f"\n--- policy {seed}  ckpt={ckpt.name} step={gstep} ---")
        sys.stdout.flush()
        policy = load_policy(
            str(ckpt), device=args.device,
            num_cnn_channels=resolve_cnn_channels(payload, context=str(ckpt)),
        )
        for opp in OPPONENTS:
            for i in range(args.episodes):
                eval_seed = EVAL_SEED_BASE + i
                ep = replay_episode(policy, opponent=opp, seed=eval_seed, device=args.device)
                n_episodes[seed] += 1
                rows = build_band_windows(ep, train_seed=seed)
                all_windows.extend(rows)
                # Same trajectory, original aggregation, for the equivalence gate.
                from experiments.run_g0_v2_evaluation import build_windows as _bw

                for w in _bw(ep["steps"], ep["failure_events"],
                             episode_key=f"{opp}:{eval_seed}"):
                    rebuilt_frozen_windows.append(
                        {"train_seed": seed, "opponent": opp,
                         "eval_seed": eval_seed, **w})
            n_fail = sum(
                1 for w in all_windows
                if w["train_seed"] == seed and w["opponent"] == opp
                and w["kind"] == "failure" and w["band"] == "latest"
            )
            print(f"  vs {opp}: episodes={args.episodes}  "
                  f"carrier-failure latest-band rows so far={n_fail}")
            sys.stdout.flush()

    # Persist band windows for auditability.
    win_path = OUT_DIR / "lag_band_windows.csv"
    if all_windows:
        fields = list(all_windows[0].keys())
        with open(win_path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for row in all_windows:
                w.writerow(row)
    print(f"\nwrote {len(all_windows)} band windows -> {win_path}")
    sys.stdout.flush()

    # ---- REPLAY-EQUIVALENCE GATE ---------------------------------------
    # Mandatory, and it runs BEFORE any candidate is evaluated. If these
    # trajectories do not reconstruct the frozen discovery, every lag result
    # below would describe a different experiment than the one C2 qualification
    # is defined against -- the same class of error that invalidated three O1
    # gates. Fail closed.
    full_run = (
        int(args.episodes) == 30 and sorted(args.seeds) == sorted(G0_SEEDS)
    )
    gate = equivalence_gate(rebuilt_frozen_windows, full_run=full_run)
    gate["generated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    gate["produced_by"] = "run_c2_lag_audit.replay_episode (reimplemented loop)"
    gate["why_this_matters_here"] = (
        "replay_episode does not call run_eval_episode; it reimplements its "
        "loop. This gate is the only thing establishing that the reimplementation "
        "reproduces the frozen discovery trajectories."
    )
    gate["episodes_per_cell"] = int(args.episodes)
    gate["policy_seeds"] = list(args.seeds)
    gate["runner_commit"] = lock_info["runner_commit"]
    (OUT_DIR / "C2_REPLAY_EQUIVALENCE.json").write_text(
        json.dumps(_json_safe(gate), indent=2, default=str, allow_nan=False),
        encoding="utf-8")

    print("\n" + "-" * 78)
    print("REPLAY-EQUIVALENCE GATE")
    print(f"  rebuilt windows      : {len(rebuilt_frozen_windows):,}")
    print(f"  frozen in scope      : {gate.get('n_frozen_windows_in_replayed_scope'):,}")
    print(f"  missing / extra      : {gate.get('n_missing_in_replay')} / {gate.get('n_extra_in_replay')}")
    print(f"  values compared      : {gate.get('n_feature_values_compared'):,}")
    print(f"  mismatches           : {gate.get('n_mismatches')}")
    print(f"  VERDICT              : {'PASS' if gate.get('PASS') else 'FAIL'}")
    print("-" * 78)
    sys.stdout.flush()
    if not gate.get("PASS"):
        print("\nSTOP — replay equivalence FAILED. No candidate is evaluated.")
        print("The reimplemented replay loop does not reproduce the frozen")
        print("discovery trajectories, so any lag result would describe a")
        print("different experiment. Diagnose before re-running.")
        for m in (gate.get("first_mismatches") or [])[:5]:
            print(f"  {m}")
        for k in (gate.get("first_missing") or [])[:3]:
            print(f"  missing: {k}")
        return 5

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    candidates: list[dict] = []

    for label in CARRIER_FAILURES:
        for feat in AUDIT_FEATURES:
            per_latest: dict[int, dict] = {}
            per_earliest: dict[int, dict] = {}
            stratum_ok: dict[int, bool] = {}
            n_onsets: dict[int, int] = {}
            mate_frac: dict[int, float] = {}
            headroom: dict[int, float] = {}

            for seed in args.seeds:
                def slice_band(band: str, kind: str, lab: str = label):
                    return [
                        w for w in all_windows
                        if w["train_seed"] == seed and w["band"] == band
                        and w["kind"] == kind
                        and (w["failure_label"] == lab if kind == "failure" else True)
                    ]

                fail_l = slice_band("latest", "failure")
                ctrl_l = slice_band("latest", "control")
                fail_e = slice_band("earliest", "failure")
                ctrl_e = slice_band("earliest", "control")

                # Opportunity-match controls for carrier labels.
                match = OPPORTUNITY_MATCH[label]
                per_latest[seed] = evaluate_cell(
                    fail_l, ctrl_l, feat, match_key=match, rng=rng)
                per_earliest[seed] = evaluate_cell(
                    fail_e, ctrl_e, feat, match_key=match, rng=rng)

                n_onsets[seed] = len([w for w in fail_l if w.get(match)])
                fail_m = [w for w in fail_l if w.get(match)]
                if fail_m:
                    mate_vals = [
                        _numeric(w.get("mate_can_intervene")) for w in fail_m
                    ]
                    mate_vals = [v for v in mate_vals if v is not None]
                    mate_frac[seed] = statistics.fmean(mate_vals) if mate_vals else 0.0
                else:
                    mate_frac[seed] = 0.0

                # Headroom: failure rate among opportunity-matched latest windows.
                fail_o = [w for w in fail_l if w.get(match)]
                ctrl_o = [w for w in ctrl_l if w.get(match)]
                denom = len(fail_o) + len(ctrl_o)
                headroom[seed] = (len(fail_o) / denom) if denom else 0.0

                # Score-stratum survival on latest band.
                ok_any = False
                for stratum in ("leading", "trailing", "tied"):
                    fs = [w for w in fail_o if w.get("score_stratum") == stratum]
                    cs = [w for w in ctrl_o if w.get("score_stratum") == stratum]
                    if len(fs) < MIN_SUPPORT or len(cs) < MIN_SUPPORT:
                        continue
                    cell = evaluate_cell(
                        fs, cs, feat, match_key=match, rng=rng)
                    d = cell.get("delta")
                    d_pool = per_latest[seed].get("delta")
                    if (
                        d is not None and d_pool is not None and d_pool != 0
                        and ((d > 0) == (d_pool > 0))
                        and abs(d) >= EFFECT_SIZE_THR
                        and cell.get("excludes_zero") is True
                    ):
                        ok_any = True
                        break
                stratum_ok[seed] = ok_any

            cand = qualify_candidate(
                label, feat, per_latest, per_earliest, stratum_ok,
                n_episodes_per_seed=n_episodes,
                n_onsets_per_seed=n_onsets,
                mate_intervene_frac=mate_frac,
                headroom_per_seed=headroom,
            )
            candidates.append(cand)
            status = "QUALIFY" if cand["qualified"] else "reject"
            print(f"  [{status}] {label} / {feat}  "
                  f"|d|={cand['mean_abs_delta_latest']:.3f}  "
                  f"headroom={cand['headroom']:.3f}  "
                  f"lag={cand['lag_decay_classification']}")
            sys.stdout.flush()

    qualified = [c for c in candidates if c["qualified"]]
    qualified.sort(key=rank_key, reverse=True)

    report = {
        "evaluation": "C2 Stage 2 lag-structured qualification audit",
        "protocol": str(FROZEN_PATH.relative_to(PROJECT_ROOT)),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_seconds": round(time.time() - started, 2),
        "data_source": frozen["stage_2_data_source"],
        "replay": {
            "eval_seed_base": EVAL_SEED_BASE,
            "episodes_per_opponent": args.episodes,
            "policies": list(args.seeds),
            "opponents": list(OPPONENTS),
            "n_band_windows": len(all_windows),
        },
        "lag_bands": [{"name": n, "start": a, "end_exclusive": b} for n, a, b in LAG_BANDS],
        "n_candidates_audited": len(candidates),
        "n_qualified": len(qualified),
        "candidates": candidates,
        "ranking_order": frozen["selection"]["ranking_order"],
    }

    if not qualified:
        # Distinguish universal headroom failure from broader non-qualification.
        headroom_only = [
            c for c in candidates
            if c["checks"]["7_headroom"] is False
            and all(v for k, v in c["checks"].items() if k != "7_headroom")
        ]
        if headroom_only and len(headroom_only) == len(candidates):
            out_name = "C2_HEADROOM_FAIL.json"
            report["verdict"] = "HEADROOM_FAIL"
            report["action"] = frozen["stop_conditions"]["headroom_fail"]["action"]
        else:
            out_name = "C2_NO_QUALIFIED_CANDIDATE.json"
            report["verdict"] = "NO_QUALIFIED_CANDIDATE"
            report["action"] = frozen["stop_conditions"]["zero_candidates_qualify"]["action"]
        report["selected"] = None
    else:
        selected = qualified[0]
        out_name = "C2_SELECTED_CANDIDATE.json"
        report["verdict"] = "SELECTED"
        report["selected"] = selected
        report["action"] = (
            "C2 candidate selected under frozen criteria; "
            "a separate O2 preregistration is required before any training"
        )

    out_path = OUT_DIR / out_name
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    # Always also write the full audit table.
    (OUT_DIR / "C2_LAG_AUDIT.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )

    print("\n" + "=" * 78)
    print(f"VERDICT: {report['verdict']}")
    print(f"qualified={len(qualified)}/{len(candidates)}")
    if report.get("selected"):
        s = report["selected"]
        print(f"SELECTED: {s['failure_label']} / {s['feature']}  "
              f"|d|={s['mean_abs_delta_latest']} headroom={s['headroom']}")
    print(f"report: {out_path}")
    print(f"wall_seconds={report['wall_seconds']}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    # The lock must be released on every exit path, including the
    # equivalence-gate STOP and any exception, or a crashed run would block all
    # future Stage 2 attempts.
    try:
        _rc = main()
    finally:
        release_stage2_lock()
    raise SystemExit(_rc)
