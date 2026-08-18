"""Synthetic validation of OBSERVABILITY_V2 before it is used on real data.

The point of these cases is not coverage for its own sake. V1 failed in a
specific way -- censored episodes given a sentinel time dominated a mean -- so
the tests are built around censoring pathologies, including a reconstruction
of the exact configuration that broke V1 on the SDS_G1_4 confirmation.

Run:  python experiments/test_observability_v2.py
      pytest experiments/test_observability_v2.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.observability_v2 import (  # noqa: E402
    COMMIT_FIRST, INTENT_FIRST, UNRESOLVED, assay, classify,
)

HORIZON = 240


def ep(ti, tc):
    return {"t_intent": ti, "t_commit": tc}


# --------------------------------------------------------------------------
# 1. classification truth table, exactly as ruled
# --------------------------------------------------------------------------
def test_classification_truth_table():
    assert classify(20, 10) == COMMIT_FIRST, "commit before intent"
    assert classify(10, 20) == INTENT_FIRST, "intent before commit"
    assert classify(None, 10) == COMMIT_FIRST, "commit observed, intent never"
    assert classify(10, None) == INTENT_FIRST, "intent observed, commit never"
    assert classify(None, None) == UNRESOLVED, "neither observed"


def test_ties_go_to_intent_first():
    """If intent is readable at the same step commitment becomes due, that does
    not support precommitment uncertainty."""
    assert classify(15, 15) == INTENT_FIRST


def test_no_sentinel_arithmetic_anywhere():
    """A censored episode must not be sensitive to how late the other event is.

    Under V1 these two cases differed by ~200 in the mean; under V2 they are
    the same class, which is the entire point of the replacement."""
    assert classify(5, None) == classify(200, None) == INTENT_FIRST
    assert classify(None, 5) == classify(None, 200) == COMMIT_FIRST


# --------------------------------------------------------------------------
# 2. gate behaviour at the extremes
# --------------------------------------------------------------------------
def test_all_commit_first_passes():
    r = assay([ep(20, 10)] * 32, horizon=HORIZON)
    assert r["p_C"] == 1.0
    assert r["passes"] is True


def test_all_intent_first_fails():
    r = assay([ep(10, 20)] * 32, horizon=HORIZON)
    assert r["p_C"] == 0.0
    assert r["passes"] is False


def test_all_unresolved_fails_conservatively():
    """Neither event resolved is not evidence that commitment came first."""
    r = assay([ep(None, None)] * 32, horizon=HORIZON)
    assert r["p_C"] == 0.0
    assert r["counts"][UNRESOLVED] == 32
    assert r["passes"] is False


def test_exactly_half_fails():
    """p_C = 0.5 must fail: the gate is LCB95 > 0.5, strictly."""
    r = assay([ep(20, 10)] * 16 + [ep(10, 20)] * 16, horizon=HORIZON)
    assert r["p_C"] == 0.5
    assert r["lcb95"] <= 0.5
    assert r["passes"] is False


def test_unresolved_counts_against_the_gate():
    """Same COMMIT_FIRST count, more UNRESOLVED -> p_C must drop."""
    a = assay([ep(20, 10)] * 24 + [ep(10, 20)] * 8, horizon=HORIZON)
    b = assay([ep(20, 10)] * 24 + [ep(None, None)] * 8, horizon=HORIZON)
    assert a["p_C"] == b["p_C"] == 0.75, "unresolved must not be deleted"
    c = assay([ep(20, 10)] * 24 + [ep(None, None)] * 40, horizon=HORIZON)
    assert c["p_C"] < 0.5 and c["passes"] is False


# --------------------------------------------------------------------------
# 3. the V1 pathology, reconstructed
# --------------------------------------------------------------------------
def test_v1_pathology_does_not_reappear():
    """Reconstruct the SDS_G1_4 confirmation shape: 25 complete cases that
    mostly favour commit-first, 5 episodes where BREACH never committed, 2
    where intent never appeared.

    Under V1 the five commit-censored episodes each contributed about -230 and
    dragged the mean to -17.4 despite a +2.5 median. V2 must not be steerable
    that way: the five are simply INTENT_FIRST, worth one episode each."""
    eps = ([ep(20, 10)] * 19        # commit first, complete
           + [ep(10, 20)] * 6       # intent first, complete
           + [ep(8, None)] * 5      # intent seen, commitment never instantiated
           + [ep(None, 12)] * 2)    # commitment seen, intent never readable
    assert len(eps) == 32

    r = assay(eps, horizon=HORIZON)
    assert r["counts"][COMMIT_FIRST] == 21     # 19 complete + 2 intent-censored
    assert r["counts"][INTENT_FIRST] == 11     # 6 complete + 5 commit-censored
    assert r["counts"][UNRESOLVED] == 0

    # V1 on the same data: sentinel arithmetic, dominated by five episodes.
    miss = HORIZON + 1
    v1 = np.mean([(e["t_intent"] if e["t_intent"] is not None else miss)
                  - (e["t_commit"] if e["t_commit"] is not None else miss)
                  for e in eps])
    assert v1 < 0, "V1 should be dragged negative by the sentinel"
    assert r["p_C"] > 0.5, "V2 should not be dragged by the same episodes"
    assert abs(r["p_C"] - 21 / 32) < 1e-12


def test_single_censored_episode_cannot_flip_the_verdict():
    """One extra censored episode may move p_C by 1/n and no more. Under V1 a
    single sentinel episode could move the mean by ~7 steps at n=32."""
    base = [ep(20, 10)] * 24 + [ep(10, 20)] * 8
    r0 = assay(base, horizon=HORIZON)
    r1 = assay(base + [ep(1, None)], horizon=HORIZON)
    assert abs(r1["p_C"] - r0["p_C"]) <= 1.0 / len(base) + 1e-9


# --------------------------------------------------------------------------
# 4. telemetry is reported but never gates
# --------------------------------------------------------------------------
def test_telemetry_present_and_non_gating():
    eps = [ep(20, 10)] * 20 + [ep(8, None)] * 12
    r = assay(eps, horizon=HORIZON)
    t = r["telemetry"]
    assert t["n_both_observed"] == 20
    assert t["commitment_instantiation_rate"] == 20 / 32
    assert t["intent_observation_rate"] == 1.0
    assert t["complete_case_median_gap"] == 10.0
    # p_C is decided by ordering, not by the size of the complete-case gap
    eps_wide = [ep(2000, 10)] * 20 + [ep(8, None)] * 12
    assert assay(eps_wide, horizon=HORIZON)["p_C"] == r["p_C"]


def test_bootstrap_is_deterministic():
    eps = [ep(20, 10)] * 21 + [ep(10, 20)] * 11
    a = assay(eps, horizon=HORIZON)
    b = assay(eps, horizon=HORIZON)
    assert a["lcb95"] == b["lcb95"] and a["ucb95"] == b["ucb95"]


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
