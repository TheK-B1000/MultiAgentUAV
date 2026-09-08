"""C1 confirmation — home_under_threat_while_leading, on fresh held-out data.

Tests ONE preregistered feature against ONE preregistered failure, on evaluation
seeds disjoint from every prior set, using the same frozen G0-V5 checkpoints.

The protocol was frozen in artifacts/g0_v5_evaluation/C1_PROPOSAL.json BEFORE
any confirmation data existed. Nothing here selects among features, windows,
matching variables or thresholds -- all of that is fixed. A negative result,
however large, is a rejection rather than a finding, because the expected
direction was declared POSITIVE in advance.

Three concepts are kept distinct throughout, because conflating them would let
the eventual claim drift into causality:

    CONTEXT   BLUE is ahead on score and its home is threatened
    OUTCOME   lost_after_leading
    FEATURE   home_threatened_frac

What a PASS establishes is NOT "home threat causes G0 to lose". It is: among
opportunity-matched leads, greater home threat reliably predicts failure to
preserve the lead across independent evaluation samples.

Run:  python experiments/run_c1_confirmation.py
"""
from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402

import experiments.run_g0_v2_evaluation as E  # noqa: E402

# --- frozen protocol (declared before any confirmation data existed) --------

C1_NAME = "home_under_threat_while_leading"
FAILURE_PREDICTED = "lost_after_leading"
OPPORTUNITY_MATCH_KEY = "opp_leading"
PRIMARY_FEATURE = "home_threatened_frac"
EXPECTED_DIRECTION = "POSITIVE"
MIN_FAILURE_EPISODES = 30
MIN_CONTROL_EPISODES = 30
EFFECT_THRESHOLD = 0.15
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 12_345
MIN_SEEDS_PASSING = 2

POLICY_SEEDS = (3_200_001, 3_200_002, 3_200_003)
# Disjoint from: 9100000+ (V6I9 discovery), 9200000+ (collapse diagnostic),
# 9300000-2 (TASK_HEALTH panel), 9400000+ (G0-V5 discovery).
CONFIRM_SEED_BASE = 9_500_000
EPISODES_PER_CELL = 30

OUT_DIR = PROJECT_ROOT / "artifacts" / "c1_confirmation"


def _artifact_dir(seed: int) -> Path:
    return PROJECT_ROOT / "artifacts" / "g0_v5_long" / f"g0_v5_long_seed{seed}"


def _run_tag(seed: int) -> str:
    return f"g0_v5_long_seed{seed}"


def confirm_seed(policy_seed: int, episodes: int, device: str) -> dict:
    """Collect windows for one frozen policy on fresh evaluation seeds."""
    rows = E.evaluate_seed(policy_seed, episodes, device)
    windows = [w for r in rows for w in r.get("_windows", [])]

    fail = [w for w in windows
            if w["kind"] == "failure" and w["failure_label"] == FAILURE_PREDICTED]
    ctrl = [w for w in windows
            if w["kind"] == "control" and w.get(OPPORTUNITY_MATCH_KEY)]

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    ci = E.episode_clustered_ci(fail, ctrl, PRIMARY_FEATURE, rng=rng)

    f_mean = E._mean_finite(E._numeric(w.get(PRIMARY_FEATURE)) for w in fail)
    c_mean = E._mean_finite(E._numeric(w.get(PRIMARY_FEATURE)) for w in ctrl)
    delta = (f_mean - c_mean) if not (math.isnan(f_mean) or math.isnan(c_mean)) else float("nan")

    enough = (
        (ci.get("n_failure_episodes") or 0) >= MIN_FAILURE_EPISODES
        and (ci.get("n_control_episodes") or 0) >= MIN_CONTROL_EPISODES
    )
    # Direction was declared POSITIVE in advance: a negative delta is a
    # rejection regardless of magnitude.
    correct_direction = (not math.isnan(delta)) and delta > 0
    meets_effect = correct_direction and delta >= EFFECT_THRESHOLD
    excludes_zero = bool(ci.get("excludes_zero"))
    passed = bool(enough and meets_effect and excludes_zero)

    return {
        "policy_seed": policy_seed,
        "n_failure_windows": len(fail),
        "n_matched_control_windows": len(ctrl),
        "n_failure_episodes": ci.get("n_failure_episodes"),
        "n_control_episodes": ci.get("n_control_episodes"),
        "failure_mean": None if math.isnan(f_mean) else round(f_mean, 4),
        "control_mean": None if math.isnan(c_mean) else round(c_mean, 4),
        "delta": None if math.isnan(delta) else round(delta, 4),
        "ci_low": ci.get("ci_low"),
        "ci_high": ci.get("ci_high"),
        "excludes_zero": excludes_zero,
        "sufficient_support": enough,
        "correct_direction": correct_direction,
        "meets_effect_threshold": meets_effect,
        "PASS": passed,
        "overall_win_rate": round(
            sum(r["win"] for r in rows) / max(len(rows), 1), 4
        ),
        "episodes_evaluated": len(rows),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Point the frozen harness at the G0-V5 checkpoints and the FRESH seed base.
    E.artifact_dir_for = _artifact_dir
    E.run_tag_for = _run_tag
    E.EVAL_SEED_BASE = CONFIRM_SEED_BASE

    print("=" * 78)
    print("C1 CONFIRMATION — home_under_threat_while_leading")
    print("=" * 78)
    print(f"  CONTEXT : BLUE ahead on score AND home threatened")
    print(f"  OUTCOME : {FAILURE_PREDICTED}")
    print(f"  FEATURE : {PRIMARY_FEATURE}  (single, preregistered)")
    print(f"  direction declared in advance: {EXPECTED_DIRECTION}")
    print(f"  matched population: windows where {OPPORTUNITY_MATCH_KEY} is true")
    print(f"  fresh evaluation seeds: {CONFIRM_SEED_BASE}..{CONFIRM_SEED_BASE + EPISODES_PER_CELL - 1}")
    print(f"  pass: delta >= {EFFECT_THRESHOLD} AND CI excludes 0, in >= {MIN_SEEDS_PASSING}/3 policies")
    print("=" * 78)

    results = []
    for s in POLICY_SEEDS:
        print(f"\n--- policy seed {s} (frozen 1M checkpoint) ---")
        r = confirm_seed(s, EPISODES_PER_CELL, "cuda")
        results.append(r)
        print(f"  failure: {r['n_failure_episodes']} episodes / {r['n_failure_windows']} windows"
              f"  control: {r['n_control_episodes']} episodes / {r['n_matched_control_windows']} windows")
        print(f"  {PRIMARY_FEATURE}: failure={r['failure_mean']} control={r['control_mean']} "
              f"delta={r['delta']} CI=[{r['ci_low']}, {r['ci_high']}]")
        print(f"  support={r['sufficient_support']} direction={r['correct_direction']} "
              f"effect={r['meets_effect_threshold']} excl0={r['excludes_zero']} -> PASS={r['PASS']}")

    n_pass = sum(1 for r in results if r["PASS"])
    confirmed = n_pass >= MIN_SEEDS_PASSING

    report = {
        "confirmation": "C1 — " + C1_NAME,
        "verdict": "CONFIRMED" if confirmed else "REJECTED",
        "policies_passing": f"{n_pass}/3",
        "minimum_required": MIN_SEEDS_PASSING,
        "concept_separation": {
            "CONTEXT": "BLUE is ahead on score AND its home is threatened",
            "OUTCOME": FAILURE_PREDICTED,
            "FEATURE": PRIMARY_FEATURE,
            "what_a_pass_establishes": (
                "Among opportunity-matched leads, greater home threat reliably "
                "predicts failure to preserve the lead across independent "
                "evaluation samples."
            ),
            "what_a_pass_does_NOT_establish": (
                "That home threat CAUSES the loss, or that failure to rotate "
                "home is the mechanism. This is a predictive context, not a "
                "demonstrated causal mechanism."
            ),
        },
        "frozen_protocol": {
            "failure_predicted": FAILURE_PREDICTED,
            "opportunity_matched_population": OPPORTUNITY_MATCH_KEY,
            "precursor_window_decisions": E.PRECURSOR_WINDOW,
            "primary_feature": PRIMARY_FEATURE,
            "single_feature_only": True,
            "expected_effect_direction": EXPECTED_DIRECTION,
            "min_failure_episodes": MIN_FAILURE_EPISODES,
            "min_control_episodes": MIN_CONTROL_EPISODES,
            "effect_size_threshold": EFFECT_THRESHOLD,
            "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED,
                          "cluster_unit": "episode"},
            "min_policies_passing": MIN_SEEDS_PASSING,
            "frozen_before_data": True,
        },
        "data": {
            "policies": "frozen G0-V5 1M checkpoints, seeds 3200001-3 (no retraining)",
            "evaluation_seeds": [CONFIRM_SEED_BASE, CONFIRM_SEED_BASE + EPISODES_PER_CELL - 1],
            "disjoint_from": ["9100000+ V6I9 discovery", "9200000+ collapse diagnostic",
                              "9300000-9300002 TASK_HEALTH panel",
                              "9400000+ G0-V5 discovery", "all training seeds"],
            "episodes_per_policy": EPISODES_PER_CELL * 7,
        },
        "per_policy": results,
        "if_rejected": (
            "Reject cleanly. Do NOT adjust the threshold or substitute "
            "defender_tag_available_frac because it looked promising in "
            "discovery. Fall back to the preidentified runner-up "
            "carrier_mostly_unescorted with a BRAND NEW preregistered protocol "
            "on fresh data; it does not inherit this confirmation."
        ),
        "locked": ["O1", "latent_birth", "router"],
    }
    (OUT_DIR / "C1_CONFIRMATION.json").write_text(
        json.dumps(report, indent=2, default=str, allow_nan=False), encoding="utf-8")

    print("\n" + "=" * 78)
    print(f"C1 VERDICT: {report['verdict']}  ({n_pass}/3 policies passed, need {MIN_SEEDS_PASSING})")
    print(f"report: {OUT_DIR / 'C1_CONFIRMATION.json'}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
