"""OBSERVABILITY_V2 -- event-order estimator for strategic uncertainty.

Replaces OBSERVABILITY_V1 prospectively. V1 is RETIRED FOR FUTURE SEARCH but
is not rewritten: V1 is what selected SDS_G1_4 and that history stands.

Why V1 was retired
------------------
V1 computed  mean(t_intent - t_commit)  with a non-observed event encoded as
the sentinel horizon+1 = 241, and required the mean > 0.

An unobserved commitment does not mean "commitment happened at step 241". It
means only "commitment was not observed before the horizon". Averaging a
sentinel with exact observations gives censored episodes enormous artificial
leverage: in the SDS_G1_4 confirmation, five episodes in which BREACH never
committed contributed gaps near -230 each and drove the mean to -17.406, while
the censored-coded median was +2.5 and 65.6% of episodes were positive.

V1 also proved unstable as a search gate: the same construction gave +16.625
in development (n=16) and -17.406 on confirmation (n=32).

What V2 does instead
--------------------
Every episode is classified by EVENT ORDER, never by sentinel arithmetic:

    commitment observed before intent .................. COMMIT_FIRST
    intent observed before or equal to commitment ...... INTENT_FIRST
    commitment observed, intent never observed ......... COMMIT_FIRST
    intent observed, commitment never observed ......... INTENT_FIRST
    neither event observed ............................. UNRESOLVED

Ties go to INTENT_FIRST: if intent is readable at the same step commitment
becomes due, that does not support precommitment uncertainty.

The one-sided censored cases are informative, not missing. An episode where
RED's intent became readable but BLUE never committed is genuine evidence
AGAINST precommitment uncertainty, so it is retained as INTENT_FIRST rather
than deleted -- complete-case deletion would bias the estimate the other way.

Primary quantity:

    p_C = P(COMMIT_FIRST)         over ALL episodes, UNRESOLVED included

Prospective gate:

    LCB95(p_C) > 0.5

UNRESOLVED episodes count in the denominator and never in the numerator, so
they weigh against the gate. That is deliberate: an episode that resolved
neither event is not evidence that commitment came first.

Timing magnitudes are still reported -- median gap, complete-case mean,
fraction positive, instantiation rates, censoring counts -- but as MECHANISM
TELEMETRY, never as the gate. That prevents a candidate from passing because
three censor-coded episodes each contributed +230.
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np

COMMIT_FIRST = "COMMIT_FIRST"
INTENT_FIRST = "INTENT_FIRST"
UNRESOLVED = "UNRESOLVED"

P_C_FLOOR = 0.5          # prospective gate: LCB95(p_C) > 0.5
N_BOOT = 20000
ALPHA = 0.05
BOOTSTRAP_RNG = 7        # same constant used by the M1 assay and confirmation


def classify(t_intent: Optional[int], t_commit: Optional[int]) -> str:
    """Event-order class for one episode. None means 'not observed by horizon'.

    No sentinel value is substituted and no arithmetic is done on a censored
    time -- the whole point of V2 is that an unobserved event has no event time.
    """
    ti_obs = t_intent is not None
    tc_obs = t_commit is not None
    if not ti_obs and not tc_obs:
        return UNRESOLVED
    if tc_obs and not ti_obs:
        return COMMIT_FIRST
    if ti_obs and not tc_obs:
        return INTENT_FIRST
    return COMMIT_FIRST if int(t_commit) < int(t_intent) else INTENT_FIRST


def _lcb(indicator: np.ndarray, rng, n_boot: int = N_BOOT, alpha: float = ALPHA):
    """Percentile bootstrap on a 0/1 indicator. Returns (mean, lo, hi)."""
    if len(indicator) == 0:
        return float("nan"), float("nan"), float("nan")
    idx = rng.integers(0, len(indicator), size=(n_boot, len(indicator)))
    b = indicator[idx].mean(axis=1)
    lo, hi = np.percentile(b, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(indicator.mean()), float(lo), float(hi)


def assay(episodes: Iterable[dict], *, horizon: int,
          rng_seed: int = BOOTSTRAP_RNG, floor: float = P_C_FLOOR) -> dict:
    """Score OBSERVABILITY_V2 over episodes carrying 't_intent'/'t_commit'.

    Values may be None or absent, meaning the event was not observed.
    """
    eps = list(episodes)
    classes = [classify(e.get("t_intent"), e.get("t_commit")) for e in eps]
    n = len(classes)
    n_c = classes.count(COMMIT_FIRST)
    n_i = classes.count(INTENT_FIRST)
    n_u = classes.count(UNRESOLVED)

    indicator = np.array([1.0 if c == COMMIT_FIRST else 0.0 for c in classes])
    rng = np.random.default_rng(rng_seed)
    p_c, lo, hi = _lcb(indicator, rng)

    # ---- telemetry only, never gating -------------------------------
    both = [(int(e["t_intent"]), int(e["t_commit"])) for e in eps
            if e.get("t_intent") is not None and e.get("t_commit") is not None]
    gaps = np.array([ti - tc for ti, tc in both], dtype=float)
    tele = {
        "n_both_observed": len(both),
        "complete_case_mean_gap": float(gaps.mean()) if len(gaps) else None,
        "complete_case_median_gap": float(np.median(gaps)) if len(gaps) else None,
        "complete_case_frac_positive": float(np.mean(gaps > 0)) if len(gaps) else None,
        "commitment_instantiation_rate":
            float(np.mean([e.get("t_commit") is not None for e in eps])) if n else None,
        "intent_observation_rate":
            float(np.mean([e.get("t_intent") is not None for e in eps])) if n else None,
        "horizon": horizon,
        "status": "MECHANISM TELEMETRY -- descriptive only, never the gate",
    }

    return {
        "estimator": "OBSERVABILITY_V2",
        "primary": "p_C = P(COMMIT_FIRST) over all episodes, UNRESOLVED included",
        "n_episodes": n,
        "counts": {COMMIT_FIRST: n_c, INTENT_FIRST: n_i, UNRESOLVED: n_u},
        "p_C": p_c,
        "lcb95": lo,
        "ucb95": hi,
        "floor": floor,
        "passes": bool(lo > floor),
        "gate": f"LCB95(p_C) > {floor}",
        "unresolved_handling":
            "counted in the denominator, never the numerator, so they weigh "
            "against the gate",
        "bootstrap": {"n_boot": N_BOOT, "alpha": ALPHA, "rng_seed": rng_seed,
                      "procedure": "percentile bootstrap on the COMMIT_FIRST indicator"},
        "telemetry": tele,
    }


def classify_many(t_intents: Sequence[Optional[int]],
                  t_commits: Sequence[Optional[int]]) -> list[str]:
    return [classify(a, b) for a, b in zip(t_intents, t_commits)]
