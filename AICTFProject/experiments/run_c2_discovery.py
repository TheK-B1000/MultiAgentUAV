"""C2 discovery — carrier conversion failures, at the correct temporal unit.

DISCOVERY ONLY. This proposes nothing and confirms nothing. Its output is a
ranked list of candidate contexts; a C2 proposal is a separate, later act, and
confirming one requires a brand-new preregistration on a further fresh seed
block.

WHY THIS EXISTS
---------------
The O1 postmortem (artifacts/o1_gates/O1_POSTMORTEM.json) traced three dead
gates to one mistake: a weakness confirmed as a contrast WITHIN episodes was
tested as a split BETWEEN episodes. Two structural defects follow from that,
and both are fixed here.

1. TEMPORAL ANCHOR.  The existing layer-3 analysis anchors its window at the
   FAILURE and looks back 30 decisions. For a carrier failure that window
   contains most of the carry itself, so "what preceded the failure" is
   contaminated by the failure unfolding. Here every unit is anchored at
   PICKUP -- the moment possession begins -- and the outcome is measured
   strictly afterwards. Context and outcome never share a decision.

2. THE SCORE CONFOUND.  In the G0-V5 discovery the only CI-backed separator for
   either carrier failure was ``leading_frac`` (about -0.18): carrier failures
   cluster when blue is behind. Any escort feature correlated with score state
   inherits that. Every contrast here is therefore computed WITHIN score
   strata, so a feature must separate among pickups that shared the same score
   situation.

THE UNIT
--------
    ContextOnset    a pickup: blue gains possession at decision t0
    OpportunityWindow  all pickups are opportunities by construction --
                    a carrier exists, so escort/support decisions are available
    OutcomeWindow   decisions (t0, t0+H]; did a blue capture occur?
    MatchedControl  a converted pickup in the same score stratum

    failure  = possession at t0 did not convert within H decisions
    control  = possession at t0 converted within H decisions

Features are read AT t0 only. Nothing after t0 enters a feature.

MULTIPLICITY IS EXPECTED AND DECLARED
-------------------------------------
Several features across several horizons and strata are examined. No
correction is applied, by design -- that is what makes this discovery rather
than evidence. Nothing here may be cited as a confirmed effect.

Run:  python experiments/run_c2_discovery.py --episodes 30
"""
from __future__ import annotations

import argparse
import csv
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

# Fresh block. Disjoint from 9100000+ (V6I9), 9200000+ (collapse), 9300000-2
# (panels), 9400000+ (G0-V5 discovery), 9500000+ (C1 confirmation), 9600000+
# (O1 gates), 9700000/9710000/9720000 (RESERVED by the frozen latent-birth
# protocol even though birth is dormant), 9900000+ (O1 training panels).
DISCOVERY_SEED_BASE = 9_800_000

G0_SEEDS = (3_200_001, 3_200_002, 3_200_003)
OUT_DIR = PROJECT_ROOT / "artifacts" / "c2_discovery"

# Conversion horizons examined. Reported side by side; a later confirmation
# must freeze exactly one.
HORIZONS = (30, 60, 90)

# Minimum cell sizes before a contrast is reported at all.
MIN_FAILURES = 20
MIN_CONTROLS = 20
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 12_345

# Candidate onset features. Every one is legal, same-map, opponent-agnostic and
# readable at the pickup decision.
ONSET_FEATURES = (
    "escort_distance",            # distance from carrier to nearest live mate
    "escort_available",           # mate within the escort radius
    "carrier_unescorted",         # the layer-2 candidate, now at the right unit
    "carrier_pressure",           # nearest red to the carrier
    "carrier_under_pressure",
    "nearest_ready_defender",     # nearest red that could actually tag now
    "defender_tag_available",
    "red_tag_ready_count",
    "carrier_dist_to_home",       # how far the conversion has to travel
    "agents_forward",
    "team_separation",
    "formation_spread",
    "blue_cooldown_active",
    "time_remaining_frac",
    # ACTIONABILITY proxy. A niche is only trainable if the free teammate can
    # materially affect the carrier's outcome before interception happens.
    # Legal, readable at onset: can the mate contest the carrier at least as
    # quickly as the nearest red that is actually able to tag right now?
    "mate_can_intervene",
    "intervention_margin",
)

# Headroom is now a first-class discovery output, not a later check. The O1
# failure was a niche where G0 already succeeded 99.1% of the time, leaving
# 0.9% headroom against a 10-point bar. A candidate whose natural failure rate
# is too low cannot support a specialist regardless of how well it predicts.
MIN_USEFUL_FAILURE_RATE = 0.20


def _score_stratum(score_diff: float) -> str:
    """The confound that dominated the previous discovery, made explicit."""
    if score_diff > 0:
        return "leading"
    if score_diff < 0:
        return "trailing"
    return "tied"


def collect_pickups(policy, *, opponent: str, seed: int, device: str) -> list[dict]:
    """Every pickup in one episode, with onset context and conversion outcome."""
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
        _np,
    )
    from rl.evaluation.opponent_resolution import (
        get_opponent_key as _get_opponent_key,
        set_opponent as _set_opponent,
        validate_opponent_name as _validate_opponent_name,
    )
    from gpu_env import GPUCTFVecEnv, GPUFieldConfig

    requested = _validate_opponent_name(opponent)
    cfg = GPUFieldConfig(
        n_envs=1, max_blue_agents=AGENTS, max_red_agents=AGENTS,
        map_set="train", map_layout=CANONICAL_MAP,
        max_decision_steps=EPISODE_HORIZON, aquaticus_profile=True,
        rules_profile="OURS", device=device, seed=int(seed),
        obstacle_obs_channel=True, tag_telemetry_enabled=True, **V2_RULES,
    )
    env = GPUCTFVecEnv(cfg)
    core = env.core
    model = policy.model if hasattr(policy, "model") else policy
    was_training = getattr(model, "training", False)
    if hasattr(model, "eval"):
        model.eval()

    pickups: list[dict] = []
    capture_steps: list[int] = []
    try:
        _set_opponent(env, requested)
        obs = _reset_obs(env.reset())
        if _get_opponent_key(env) != requested:
            raise RuntimeError("opponent drift")
        core.drain_tag_events()

        prev_carrying = 0
        for step_i in range(EPISODE_HORIZON + 8):
            ctx = legal_context(core)

            action = _predict(policy, _adapt_obs_for_policy(obs, policy))
            obs, _, done, _infos = _unpack_step(env.step(action))

            for e in core.drain_tag_events():
                if e.get("event_type") == "capture_scored" and e.get("scoring_team") == "blue":
                    capture_steps.append(step_i)

            now_carrying = int(_np(core.blue_carrying)[0].astype(bool).sum())
            if prev_carrying == 0 and now_carrying > 0:
                # ContextOnset. Features are read from the context BEFORE this
                # step's transition, i.e. strictly at the onset decision.
                blue_flag_home = _np(core.blue_flag_home)[0]
                blue_pos = _np(core.blue_pos)[0]
                carry = _np(core.blue_carrying)[0].astype(bool)
                cols = max(float(core.cols), 1e-6)
                if carry.any():
                    ci = int(np.argmax(carry))
                    dist_home = float(
                        np.linalg.norm(blue_pos[ci] - blue_flag_home) / cols
                    )
                else:
                    dist_home = float("nan")

                row = {
                    "onset_step": step_i,
                    "score_stratum": _score_stratum(ctx["score_diff"]),
                    "score_diff": ctx["score_diff"],
                    "carrier_dist_to_home": dist_home,
                }
                for k in ONSET_FEATURES:
                    if k in row:
                        continue
                    v = ctx.get(k)
                    row[k] = float(v) if isinstance(v, (int, float, bool)) else float("nan")

                # Actionability at onset: mate closer to the carrier than the
                # nearest red that could tag right now. Positive margin means
                # the teammate can plausibly contest the return fight.
                mate_d = float(ctx.get("escort_distance", float("nan")))
                def_d = float(ctx.get("nearest_ready_defender", float("nan")))
                if math.isnan(mate_d) or math.isinf(mate_d) or math.isnan(def_d) or math.isinf(def_d):
                    row["mate_can_intervene"] = float("nan")
                    row["intervention_margin"] = float("nan")
                else:
                    row["intervention_margin"] = def_d - mate_d
                    row["mate_can_intervene"] = float(mate_d <= def_d)
                pickups.append(row)
            prev_carrying = now_carrying

            if _done(done):
                break
    finally:
        if hasattr(model, "train"):
            model.train(was_training)
        env.close()

    # OutcomeWindow: strictly after onset, for each horizon.
    for p in pickups:
        t0 = p["onset_step"]
        for H in HORIZONS:
            p[f"converted_{H}"] = int(any(t0 < c <= t0 + H for c in capture_steps))
    return pickups


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


def analyze(rows: list[dict]) -> dict:
    """Within-stratum contrasts at every horizon, per seed."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    out: dict = {"prevalence": {}, "contrasts": {}}

    for H in HORIZONS:
        key = f"converted_{H}"
        conv = sum(r[key] for r in rows)
        fr = 1 - conv / max(len(rows), 1)
        by_stratum = {}
        for stratum in ("leading", "trailing", "tied"):
            sub = [r for r in rows if r["score_stratum"] == stratum]
            if sub:
                c = sum(r[key] for r in sub)
                by_stratum[stratum] = {
                    "n": len(sub), "failure_rate": round(1 - c / len(sub), 4)
                }
        out["prevalence"][f"H{H}"] = {
            "n_pickups": len(rows),
            "converted": conv,
            "failed": len(rows) - conv,
            "failure_rate": round(fr, 4),
            "headroom_ok": bool(fr >= MIN_USEFUL_FAILURE_RATE),
            "min_useful_failure_rate": MIN_USEFUL_FAILURE_RATE,
            "by_stratum": by_stratum,
        }

    # Actionability: of the pickups that FAILED, how often could the teammate
    # plausibly have contested? A niche where the mate could never have helped
    # offers no strategic fork, however well the feature predicts.
    act = {}
    for H in HORIZONS:
        fail = [r for r in rows if not r[f"converted_{H}"]]
        vals = [r["mate_can_intervene"] for r in fail
                if isinstance(r.get("mate_can_intervene"), float)
                and not math.isnan(r["mate_can_intervene"])]
        act[f"H{H}"] = {
            "n_failures_scored": len(vals),
            "mate_could_intervene_frac": round(statistics.fmean(vals), 4) if vals else None,
        }
    out["actionability"] = act

    for H in HORIZONS:
        key = f"converted_{H}"
        for stratum in ("leading", "trailing", "tied", "ALL"):
            sub = rows if stratum == "ALL" else [r for r in rows if r["score_stratum"] == stratum]
            fail = [r for r in sub if not r[key]]
            ctrl = [r for r in sub if r[key]]
            cell = f"H{H}/{stratum}"
            if len(fail) < MIN_FAILURES or len(ctrl) < MIN_CONTROLS:
                out["contrasts"][cell] = {
                    "n_failure": len(fail), "n_control": len(ctrl),
                    "skipped": "below minimum cell size",
                }
                continue
            feats = {}
            for feat in ONSET_FEATURES:
                ci = episode_clustered_delta(fail, ctrl, feat, rng=rng)
                if ci.get("delta") is None:
                    continue
                feats[feat] = ci
            ranked = sorted(
                (f for f in feats.items() if f[1].get("excludes_zero")),
                key=lambda kv: -abs(kv[1]["delta"]),
            )
            out["contrasts"][cell] = {
                "n_failure": len(fail), "n_control": len(ctrl),
                "features": feats,
                "ci_backed_ranked": [
                    {"feature": k, "delta": v["delta"],
                     "ci": [v["ci_low"], v["ci_high"]]} for k, v in ranked
                ],
            }
    return out


def main() -> int:
    from experiments.run_g0_v2_evaluation import resolve_cnn_channels
    from experiments.run_g0_v2_seed import OPPONENTS
    from rl.custom_ppo.checkpoints.loader import read_checkpoint_payload
    from rl.evaluation.checkpoint import load_policy

    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seeds", type=int, nargs="*", default=list(G0_SEEDS))
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()
    print("=" * 78)
    print("C2 DISCOVERY — carrier conversion, anchored at PICKUP")
    print(f"seeds={args.seeds}  opponents={OPPONENTS}  episodes/cell={args.episodes}")
    print(f"fresh discovery seeds {DISCOVERY_SEED_BASE}..{DISCOVERY_SEED_BASE + args.episodes - 1}")
    print(f"horizons={HORIZONS}  contrasts computed WITHIN score strata")
    print("DISCOVERY ONLY — proposes nothing, confirms nothing")
    print("=" * 78)

    per_seed: dict[int, list[dict]] = {}
    for seed in args.seeds:
        tag = f"g0_v5_long_seed{seed}"
        ckpt = PROJECT_ROOT / "artifacts" / "g0_v5_long" / tag / "ckpts" / f"final_{tag}.zip"
        payload = read_checkpoint_payload(str(ckpt), map_location="cpu")
        if int(payload.get("global_step", 0)) < 1_000_000:
            raise ValueError(f"{ckpt}: not the preregistered 1M checkpoint")
        policy = load_policy(str(ckpt), device=args.device,
                             num_cnn_channels=resolve_cnn_channels(payload, context=str(ckpt)))
        rows: list[dict] = []
        for opp in OPPONENTS:
            for i in range(args.episodes):
                s = DISCOVERY_SEED_BASE + i
                for p in collect_pickups(policy, opponent=opp, seed=s, device=args.device):
                    p["episode_key"] = f"{opp}:{s}"
                    p["opponent"] = opp
                    p["eval_seed"] = s
                    p["train_seed"] = seed
                    rows.append(p)
            print(f"  seed {seed} vs {opp}: {sum(1 for r in rows if r['opponent'] == opp)} pickups")
        per_seed[seed] = rows

    report = {
        "discovery": "C2 — carrier conversion failure, anchored at pickup",
        "classification": "DISCOVERY ONLY — not evidence, not a proposal",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "motivated_by": "artifacts/o1_gates/O1_POSTMORTEM.json",
        "structural_fixes": {
            "temporal_anchor": "every unit anchored at PICKUP; outcome measured strictly after; context and outcome never share a decision",
            "score_confound": "all contrasts computed WITHIN score strata, because leading_frac was the only CI-backed separator in the previous discovery",
        },
        "discovery_seed_base": DISCOVERY_SEED_BASE,
        "episodes_per_cell": args.episodes,
        "horizons": list(HORIZONS),
        "multiplicity": "several features x horizons x strata examined; NO correction applied, by design",
        "per_seed": {str(s): analyze(r) for s, r in per_seed.items()},
    }
    (OUT_DIR / "C2_DISCOVERY.json").write_text(
        json.dumps(report, indent=2, default=str, allow_nan=False), encoding="utf-8")

    flat = [r for rows in per_seed.values() for r in rows]
    if flat:
        with open(OUT_DIR / "pickups.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(flat[0].keys()))
            w.writeheader()
            w.writerows(flat)

    print("\n" + "=" * 78)
    for s, a in report["per_seed"].items():
        pv = a["prevalence"]
        print(f"seed {s}: pickups={pv['H60']['n_pickups']} "
              f"failure_rate H30={pv['H30']['failure_rate']} "
              f"H60={pv['H60']['failure_rate']} H90={pv['H90']['failure_rate']}")
    print("\nCI-backed features, H60, within-stratum (DISCOVERY — not evidence):")
    for s, a in report["per_seed"].items():
        for stratum in ("leading", "trailing", "tied"):
            cell = a["contrasts"].get(f"H60/{stratum}", {})
            ranked = cell.get("ci_backed_ranked") or []
            if ranked:
                top = ", ".join(f"{r['feature']}={r['delta']:+.3f}" for r in ranked[:3])
                print(f"  seed {s} [{stratum}] n_f={cell['n_failure']} n_c={cell['n_control']}: {top}")
            elif "skipped" in cell:
                print(f"  seed {s} [{stratum}]: {cell['skipped']} "
                      f"(n_f={cell['n_failure']}, n_c={cell['n_control']})")
    print(f"\nreport: {OUT_DIR / 'C2_DISCOVERY.json'}")
    print(f"wall: {round(time.time() - started, 1)}s")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
