"""C3 discovery — commitment-proximal strategic-fork detector (SUPERSEDED DRAFT).

EXECUTION STATUS
----------------
C3 methodology is DRAFT / NOT FROZEN. This runner is a superseded-draft
implementation and must not perform discovery rollouts until an explicit
authorization artifact exists:

  artifacts/c3_discovery/C3_EXECUTION_AUTHORIZATION.json

with status FROZEN_AND_AUTHORIZED and matching contract / prereg / runner
hashes. Absence of that file is the correct default. See
docs/c3-decision-proximal-preregistration.md §Execution authorization.

DISCOVERY ONLY (once authorized). This proposes nothing and confirms nothing.
Its output is a ranked list of candidate contexts; a C3 proposal is a separate,
later act, and confirming one requires a brand-new preregistration on a further
fresh seed block. C3 cannot establish latent necessity — that is owned by the
Environment-Demand Gate after an independent response oracle exists.

Run (only after authorization):
  python experiments/run_c3_decision_proximal_discovery.py --episodes 30
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import math
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from experiments.run_g0_v2_evaluation import (
    AGENTS,
    CANONICAL_MAP,
    EPISODE_HORIZON,
    V2_RULES,
    legal_context,
    _np,
)
from experiments.run_g0_v2_seed import OPPONENTS

DISCOVERY_SEED_BASE = 9_400_000
G0_SEEDS = (3_200_001, 3_200_002, 3_200_003)
OUT_DIR = PROJECT_ROOT / "artifacts" / "c3_discovery"
AUTH_PATH = OUT_DIR / "C3_EXECUTION_AUTHORIZATION.json"
CONTRACT_PATH = OUT_DIR / "C3_DISCOVERY_PREREG_FROZEN.json"
PREREG_PATH = PROJECT_ROOT / "docs" / "c3-decision-proximal-preregistration.md"

MIN_FAILURES = 20
MIN_CONTROLS = 20
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 12_345

FEATURE_NAMES = (
    'time_to_intercept',
    'relative_closing_velocity', 
    'carrier_dist_home',
    'nearest_ready_defender_dist',
    'escort_dist',
    'cooldown_remaining',
    'carrier_progress_frac',
    'pressure_trend',
    'commitment_imbalance',
    'mate_intervention_eta',
    'intercept_margin',
    'formation_spread',
)


def _score_stratum(score_diff: float) -> str:
    if score_diff > 0:
        return 'leading'
    if score_diff < 0:
        return 'trailing'
    return 'tied'


def collect_onsets(policy, *, opponent: str, seed: int, device: str, horizon: int = 30):
    """Every carrier-pressure onset in one episode, with features and outcome."""
    from experiments.eval_v6i9_map_awareness import (
        _adapt_obs_for_policy, _done, _predict, _reset_obs, _unpack_step,
    )
    from rl.evaluation.opponent_resolution import (
        get_opponent_key, set_opponent, validate_opponent_name,
    )
    from gpu_env import GPUCTFVecEnv, GPUFieldConfig
    from rl.analysis.decision_proximal_features import DecisionProximalExtractor
    
    requested = validate_opponent_name(opponent)
    cfg = GPUFieldConfig(
        n_envs=1, max_blue_agents=AGENTS, max_red_agents=AGENTS,
        map_set='train', map_layout=CANONICAL_MAP,
        max_decision_steps=EPISODE_HORIZON, aquaticus_profile=True,
        rules_profile='OURS', device=device, seed=int(seed),
        obstacle_obs_channel=True, tag_telemetry_enabled=True, **V2_RULES,
    )
    env = GPUCTFVecEnv(cfg)
    core = env.core
    model = policy.model if hasattr(policy, 'model') else policy
    was_training = getattr(model, 'training', False)
    if hasattr(model, 'eval'):
        model.eval()
    
    extractor = DecisionProximalExtractor()
    onsets = []
    tag_steps = []  # steps where blue carrier was tagged
    capture_steps = []  # steps where blue captured the flag
    
    all_features_by_step = {}  # step_i -> dict
    
    try:
        set_opponent(env, requested)
        obs = _reset_obs(env.reset())
        if get_opponent_key(env) != requested:
            raise RuntimeError('opponent drift')
        core.drain_tag_events()
        extractor.reset()
        
        for step_i in range(EPISODE_HORIZON + 8):
            features = extractor.extract(core)
            # We convert to dict for easy access later
            all_features_by_step[step_i] = dataclasses.asdict(features)
            
            for e in core.drain_tag_events():
                if e.get('event_type') == 'capture_scored' and e.get('scoring_team') == 'blue':
                    capture_steps.append(step_i)
                if e.get('event_type') == 'tagged' and e.get('tagged_team') == 'blue':
                    tag_steps.append(step_i)
            
            if features.is_carrier_pressure_onset:
                row = dataclasses.asdict(features)
                row['onset_step'] = step_i
                row['score_stratum'] = _score_stratum(features.score_diff)
                onsets.append(row)
            
            action = _predict(policy, _adapt_obs_for_policy(obs, policy))
            obs, _, done, _infos = _unpack_step(env.step(action))
            if _done(done):
                break
    finally:
        if hasattr(model, 'train'):
            model.train(was_training)
        env.close()
    
    for onset in onsets:
        t0 = onset['onset_step']
        onset['tagged_within_H'] = int(any(t0 < t <= t0 + horizon for t in tag_steps))
        onset['captured_within_H'] = int(any(t0 < t <= t0 + horizon for t in capture_steps))
        onset['failure'] = onset['tagged_within_H']  # primary failure label
    
    return onsets, all_features_by_step


def episode_clustered_delta(fail: list[dict], ctrl: list[dict], feat: str,
                            *, rng) -> dict:
    """Failure-minus-control difference, resampling EPISODES."""
    def by_ep(rows):
        g = defaultdict(list)
        for r in rows:
            v = r.get(feat)
            if v is not None and isinstance(v, (int, float)) and not (
                math.isnan(v) or math.isinf(v)
            ):
                g[r["episode_key"]].append(float(v))
        return {k: v for k, v in g.items() if v}

    f, c = by_ep(fail), by_ep(ctrl)
    fk, ck = list(f), list(c)
    if len(fk) < 2 or len(ck) < 2:
        return {"delta": None, "ci_low": None, "ci_high": None,
                "n_failure_episodes": len(fk), "n_control_episodes": len(ck),
                "excludes_zero": None, "insufficient": True}

    def m(g, keys):
        vals = [v for k in keys for v in g[k]]
        return statistics.fmean(vals) if vals else float("nan")

    point = m(f, fk) - m(c, ck)
    draws = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        a = [fk[i] for i in rng.integers(0, len(fk), len(fk))]
        b = [ck[i] for i in rng.integers(0, len(ck), len(ck))]
        d = m(f, a) - m(c, b)
        if not math.isnan(d):
            draws.append(d)
    if not draws:
        return {"delta": round(point, 4), "ci_low": None, "ci_high": None,
                "n_failure_episodes": len(fk), "n_control_episodes": len(ck),
                "excludes_zero": None, "insufficient": True}
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {
        "delta": round(point, 4),
        "ci_low": round(float(lo), 4), "ci_high": round(float(hi), 4),
        "n_failure_episodes": len(fk), "n_control_episodes": len(ck),
        "excludes_zero": bool(lo > 0 or hi < 0), "insufficient": False,
    }


def _run_stage_3(policy, device, onsets, args):
    """Counterfactual actionability gate."""
    from experiments.eval_v6i9_map_awareness import (
        _adapt_obs_for_policy, _done, _predict, _reset_obs, _unpack_step,
    )
    from rl.evaluation.opponent_resolution import (
        get_opponent_key, set_opponent, validate_opponent_name,
    )
    from gpu_env import GPUCTFVecEnv, GPUFieldConfig
    from tools.q_probe_local_counterfactual import (
        _snapshot_env, _restore_env, _snapshot_policy, _restore_policy,
    )
    from rl.analysis.counterfactual_actionability import (
        run_counterfactual_branches, compute_actionability,
    )

    # We only test failures for actionability (did alternate action avoid tag?)
    target_onsets = [o for o in onsets if o['failure']]
    if not target_onsets:
        return 0.0

    target_onsets = target_onsets[:args.max_onsets_per_policy]

    # Group by episode
    by_ep = defaultdict(list)
    for o in target_onsets:
        by_ep[(o['opponent'], o['eval_seed'])].append(o)

    actionable_count = 0
    total_tested = 0

    model = policy.model if hasattr(policy, 'model') else policy
    was_training = getattr(model, 'training', False)
    if hasattr(model, 'eval'):
        model.eval()

    for (opp, s), eps_onsets in by_ep.items():
        requested = validate_opponent_name(opp)
        cfg = GPUFieldConfig(
            n_envs=1, max_blue_agents=AGENTS, max_red_agents=AGENTS,
            map_set='train', map_layout=CANONICAL_MAP,
            max_decision_steps=EPISODE_HORIZON, aquaticus_profile=True,
            rules_profile='OURS', device=device, seed=int(s),
            obstacle_obs_channel=True, tag_telemetry_enabled=True, **V2_RULES,
        )
        env = GPUCTFVecEnv(cfg)
        core = env.core
        
        try:
            set_opponent(env, requested)
            obs = _reset_obs(env.reset())
            
            eps_onsets_sorted = sorted(eps_onsets, key=lambda x: x['onset_step'])
            next_onset_idx = 0
            
            for step_i in range(EPISODE_HORIZON + 8):
                if next_onset_idx < len(eps_onsets_sorted) and step_i == eps_onsets_sorted[next_onset_idx]['onset_step']:
                    # Reached an onset. Snapshot and branch.
                    env_snap = _snapshot_env(env)
                    pol_snap = _snapshot_policy(policy)
                    
                    branches = run_counterfactual_branches(
                        policy, env, core,
                        snapshot_env_fn=_snapshot_env, restore_env_fn=_restore_env,
                        snapshot_pol_fn=_snapshot_policy, restore_pol_fn=_restore_policy,
                        horizon=args.horizon,
                        device=device
                    )
                    
                    actionability = compute_actionability(branches)
                    if actionability['is_actionable']:
                        actionable_count += 1
                    total_tested += 1
                    
                    _restore_env(env, env_snap)
                    _restore_policy(policy, pol_snap)
                    
                    next_onset_idx += 1
                    
                if next_onset_idx >= len(eps_onsets_sorted):
                    break
                    
                action = _predict(policy, _adapt_obs_for_policy(obs, policy))
                obs, _, done, _infos = _unpack_step(env.step(action))
                if _done(done):
                    break
        finally:
            env.close()

    if hasattr(model, 'train'):
        model.train(was_training)

    return actionable_count / max(1, total_tested)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_head() -> str:
    try:
        out = subprocess.run(
            ["git", "log", "-1", "--format=%H"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        return (out.stdout or "").strip() or "unknown"
    except Exception:
        return "unknown"


def _require_c3_execution_authorization() -> dict:
    """Fail closed: no rollout until FROZEN_AND_AUTHORIZED with matching hashes."""
    if not AUTH_PATH.exists():
        raise SystemExit(
            "C3 is DRAFT / NOT FROZEN. Execution is prohibited until "
            f"{AUTH_PATH.relative_to(PROJECT_ROOT)} exists with "
            "status=FROZEN_AND_AUTHORIZED and matching "
            "c3_contract_hash / c3_prereg_commit / runner_commit. "
            "See docs/c3-decision-proximal-preregistration.md "
            "§Execution authorization."
        )
    try:
        auth = json.loads(AUTH_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"C3 authorization artifact unreadable: {AUTH_PATH}: {exc}") from exc

    status = auth.get("status")
    if status != "FROZEN_AND_AUTHORIZED":
        raise SystemExit(
            f"C3 execution refused: authorization status={status!r} "
            f"(required 'FROZEN_AND_AUTHORIZED') in {AUTH_PATH}"
        )

    if not CONTRACT_PATH.exists():
        raise SystemExit(
            "C3 execution refused: frozen machine-readable contract missing at "
            f"{CONTRACT_PATH}. Freeze C3_DISCOVERY_PREREG_FROZEN.json before authorizing."
        )

    contract_hash = _sha256_file(CONTRACT_PATH)
    expected_contract = str(auth.get("c3_contract_hash") or "")
    if not expected_contract or expected_contract != contract_hash:
        raise SystemExit(
            "C3 execution refused: c3_contract_hash mismatch.\n"
            f"  authorization: {expected_contract or '(missing)'}\n"
            f"  on-disk contract sha256: {contract_hash}"
        )

    head = _git_head()
    expected_prereg = str(auth.get("c3_prereg_commit") or "")
    expected_runner = str(auth.get("runner_commit") or "")
    if not expected_prereg or expected_prereg != head:
        raise SystemExit(
            "C3 execution refused: c3_prereg_commit mismatch "
            f"(auth={expected_prereg or '(missing)'} head={head}). "
            "Re-authorize from the freeze commit."
        )
    if not expected_runner or expected_runner != head:
        raise SystemExit(
            "C3 execution refused: runner_commit mismatch "
            f"(auth={expected_runner or '(missing)'} head={head}). "
            "Re-authorize from the freeze commit."
        )

    # Human-readable prereg must still exist; hash recorded for provenance only
    # if present in the auth artifact (optional field c3_prereg_sha256).
    if not PREREG_PATH.exists():
        raise SystemExit(f"C3 execution refused: prereg missing at {PREREG_PATH}")
    expected_prereg_sha = auth.get("c3_prereg_sha256")
    if expected_prereg_sha:
        prereg_sha = _sha256_file(PREREG_PATH)
        if str(expected_prereg_sha) != prereg_sha:
            raise SystemExit(
                "C3 execution refused: c3_prereg_sha256 mismatch.\n"
                f"  authorization: {expected_prereg_sha}\n"
                f"  on-disk prereg sha256: {prereg_sha}"
            )

    return auth


def main() -> int:
    # Hard guard FIRST — before argparse side effects that imply a live run,
    # and before any env / checkpoint / rollout work.
    auth = _require_c3_execution_authorization()

    from experiments.long_session_progress import LongSessionProgress, configure_stdio
    from experiments.run_g0_v2_evaluation import resolve_cnn_channels
    from rl.custom_ppo.checkpoints.loader import read_checkpoint_payload
    from rl.evaluation.checkpoint import load_policy

    configure_stdio()

    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', type=int, default=30)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--seeds', type=int, nargs='*', default=list(G0_SEEDS))
    ap.add_argument('--stage', type=int, default=3, help='Run up to this stage (1, 2, or 3)')
    ap.add_argument('--horizon', type=int, default=30, help='Near-term outcome horizon')
    ap.add_argument('--actionability-threshold', type=float, default=0.20)
    ap.add_argument('--min-effect', type=float, default=0.05)
    ap.add_argument('--max-onsets-per-policy', type=int, default=50)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prog = LongSessionProgress(OUT_DIR, name="C3_DISCOVERY")
    started = time.time()
    prog.log("=" * 78)
    prog.log("C3 DISCOVERY — AUTHORIZED (contract hashes verified)")
    prog.log(f"authorization: {AUTH_PATH}")
    prog.log(f"c3_contract_hash={auth.get('c3_contract_hash')}")
    prog.log(f"c3_prereg_commit={auth.get('c3_prereg_commit')}")
    prog.log(f"runner_commit={auth.get('runner_commit')}")
    prog.log(f"seeds={args.seeds}  opponents={OPPONENTS}  episodes/cell={args.episodes}")
    prog.log(f"fresh discovery seeds {DISCOVERY_SEED_BASE}..{DISCOVERY_SEED_BASE + args.episodes - 1}")
    prog.log(f"stage up to={args.stage}")
    prog.log("DISCOVERY ONLY — proposes nothing, confirms nothing")
    prog.log(f"watch: {prog.log_path}")
    prog.log(f"watch: {prog.json_path}")
    prog.log("=" * 78)

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    per_seed: dict[int, list[dict]] = {}
    time_series_by_ep = {}

    all_onsets_flat = []

    # STAGE 1: Collect & Rank
    total_eps = int(len(args.seeds) * len(OPPONENTS) * int(args.episodes))
    prog.set_phase("STAGE1", f"total_episodes={total_eps}")
    jobs = [
        (seed, opp, DISCOVERY_SEED_BASE + i)
        for seed in args.seeds
        for opp in OPPONENTS
        for i in range(args.episodes)
    ]
    rows_by_seed: dict[int, list[dict]] = {int(s): [] for s in args.seeds}
    policies: dict[int, object] = {}
    for seed in args.seeds:
        tag = f"g0_v5_long_seed{seed}"
        ckpt = PROJECT_ROOT / "artifacts" / "g0_v5_long" / tag / "ckpts" / f"final_{tag}.zip"
        payload = read_checkpoint_payload(str(ckpt), map_location="cpu")
        policies[int(seed)] = load_policy(
            str(ckpt),
            device=args.device,
            num_cnn_channels=resolve_cnn_channels(payload, context=str(ckpt)),
        )
        prog.log(f"[STAGE1] loaded policy seed={seed}")

    for seed, opp, s in prog.bar(jobs, desc="stage1_episodes", unit="ep"):
        policy = policies[int(seed)]
        onsets, series = collect_onsets(
            policy, opponent=opp, seed=s, device=args.device, horizon=args.horizon
        )
        time_series_by_ep[f"{opp}:{s}"] = series
        for p in onsets:
            p["episode_key"] = f"{opp}:{s}"
            p["opponent"] = opp
            p["eval_seed"] = s
            p["train_seed"] = seed
            rows_by_seed[int(seed)].append(p)
            all_onsets_flat.append(p)

    for seed in args.seeds:
        per_seed[int(seed)] = rows_by_seed[int(seed)]
        prog.log(f"[STAGE1] seed={seed} onsets={len(per_seed[int(seed)])}")

    stage1_report = {}
    candidates = set()
    for seed, rows in per_seed.items():
        by_stratum = {}
        for stratum in ("leading", "trailing", "tied", "ALL"):
            sub = rows if stratum == "ALL" else [r for r in rows if r["score_stratum"] == stratum]
            fail = [r for r in sub if r['failure']]
            ctrl = [r for r in sub if not r['failure']]
            
            if len(fail) < MIN_FAILURES or len(ctrl) < MIN_CONTROLS:
                continue
                
            feats = {}
            for feat in FEATURE_NAMES:
                ci = episode_clustered_delta(fail, ctrl, feat, rng=rng)
                if ci.get("excludes_zero") and abs(ci.get("delta", 0)) >= args.min_effect:
                    feats[feat] = ci
                    candidates.add(feat)
            
            by_stratum[stratum] = sorted(
                (f for f in feats.items()), key=lambda kv: -abs(kv[1]["delta"])
            )
        stage1_report[seed] = by_stratum

    prog.log(f"Stage 1 yielded {len(candidates)} CI-backed candidate features.")

    report = {
        "stage_1": stage1_report,
        "stage_2": {},
        "stage_3": {},
        "final_candidates": []
    }

    # STAGE 2: Early Temporal Qualification
    stage2_survivors = set()
    if args.stage >= 2 and candidates:
        prog.set_phase("STAGE2", f"n_candidates={len(candidates)}")
        overall_stdev = {}
        for feat in candidates:
            all_vals = []
            for r in all_onsets_flat:
                if feat in r and not math.isnan(r[feat]):
                    all_vals.append(r[feat])
            overall_stdev[feat] = statistics.stdev(all_vals) if len(all_vals) > 1 else 1e-6

        for feat in prog.bar(list(candidates), desc="stage2_features", unit="feat"):
            # check direction in early band and std change in proximity
            passed = True
            
            lag_bands = {'earliest': [], 'middle': [], 'latest': []}
            prox_changes = []
            
            for o in all_onsets_flat:
                series = time_series_by_ep[o["episode_key"]]
                t0 = o['onset_step']
                
                # Proximity test (t0-5 to t0-1)
                prox_vals = []
                for t in range(t0-5, t0):
                    if t in series:
                        prox_vals.append(series[t].get(feat, 0))
                if prox_vals:
                    prox_changes.append(max(prox_vals) - min(prox_vals))
                
                # Lag band test logic can be populated similarly
                # Just simplified passing rate for demo
                
            mean_prox = statistics.fmean(prox_changes) if prox_changes else 0
            if mean_prox < 0.20 * overall_stdev[feat]:
                passed = False
                
            if passed:
                stage2_survivors.add(feat)
                
            report["stage_2"][feat] = {
                "passed": passed,
                "mean_prox_change": mean_prox,
                "stdev_overall": overall_stdev[feat]
            }
        prog.log(f"Stage 2 yielded {len(stage2_survivors)} survivors.")

    # STAGE 3: Counterfactual Actionability
    stage3_survivors = set()
    if args.stage >= 3 and stage2_survivors:
        prog.set_phase("STAGE3", f"n_survivors={len(stage2_survivors)}")
        for seed in prog.bar(list(args.seeds), desc="stage3_policies", unit="policy"):
            policy = policies[int(seed)]
            rate = _run_stage_3(policy, args.device, per_seed[seed], args)
            report["stage_3"][seed] = {"actionability_rate": rate}
            prog.log(f"[STAGE3] seed={seed} actionability_rate={rate:.4f}")
            
            if rate >= args.actionability_threshold:
                stage3_survivors.update(stage2_survivors)
        prog.log(f"Stage 3 yielded {len(stage3_survivors)} final candidates.")

    report["final_candidates"] = list(stage3_survivors) if args.stage >= 3 else list(stage2_survivors if args.stage == 2 else candidates)

    (OUT_DIR / "C3_DISCOVERY.json").write_text(
        json.dumps(report, indent=2, default=str, allow_nan=False), encoding="utf-8")

    if all_onsets_flat:
        with open(OUT_DIR / "C3_ONSETS.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_onsets_flat[0].keys()))
            w.writeheader()
            w.writerows(all_onsets_flat)

    if report["final_candidates"]:
        (OUT_DIR / "C3_QUALIFIED_CANDIDATES.json").write_text(
            json.dumps({"ranked_candidates": report["final_candidates"]}, indent=2)
        )
    else:
        (OUT_DIR / "C3_NO_QUALIFIED_STRATEGIC_FORK.json").write_text(
            json.dumps({"result": "clean negative", "message": "No candidates passed all executed stages."}, indent=2)
        )

    prog.set_phase("COMPLETE", f"n_final={len(report['final_candidates'])}")
    prog.log("CI-backed candidates:")
    prog.log(str(report["final_candidates"]))
    prog.log(f"report: {OUT_DIR / 'C3_DISCOVERY.json'}")
    prog.log(f"progress_log: {prog.log_path}")
    prog.log(f"progress_json: {prog.json_path}")
    prog.log(f"wall: {round(time.time() - started, 1)}s")
    prog.log("=" * 78)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
