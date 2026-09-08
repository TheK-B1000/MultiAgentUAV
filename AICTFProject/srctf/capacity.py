"""SRCTF affordance-capacity statistic, frozen by SRCTF_V1_ERRATUM_01/02.

    AFFORDANCE_CAPACITY_PASS  iff  r_hat >= -0.70
                              AND  q_0.025(r_bootstrap) > -0.898

r is the Pearson correlation between offensive_pressure and red_contest_fraction
across the opponent population, measured exactly as in C6 Stage 1, where the
historical 2v2 value was -0.898 across nine opponents. That figure is the
cramped-allocation-budget signature this benchmark exists to remove.

Zero variance in EITHER probe variable is AFFORDANCE_CAPACITY_FAIL, never a pass
by default and never a skipped criterion: a population in which every opponent
allocates identically has demonstrably not expanded allocation diversity.

Fully testable without any SRCTF measurement -- the statistic is content-free.
"""
from __future__ import annotations

import dataclasses

import numpy as np

HISTORICAL_R = -0.898          # C6 Stage 1, 2v2, nine opponents
POINT_THRESHOLD = -0.70        # ~half the shared variance of the baseline
RESAMPLES = 2000               # inherited from the C4/C5 contract
BOOTSTRAP_SEED = 12345
LCB_PCT = 2.5


@dataclasses.dataclass(frozen=True)
class CapacityResult:
    verdict: str
    r_hat: float | None
    r_lcb95: float | None
    n_opponents: int
    reason: str
    point_ok: bool = False
    ci_ok: bool = False

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def _pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.size < 2:
        return None
    sx, sy = float(np.std(x)), float(np.std(y))
    if sx == 0.0 or sy == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def affordance_capacity(offence: list[float], contest: list[float]) -> CapacityResult:
    """Evaluate the frozen capacity gate. Fails closed on every degenerate case."""
    x = np.asarray(offence, dtype=float)
    y = np.asarray(contest, dtype=float)
    n = int(x.size)

    if n != y.size:
        return CapacityResult("AFFORDANCE_CAPACITY_FAIL", None, None, n,
                              "offence and contest have different lengths")
    if n < 3:
        return CapacityResult("AFFORDANCE_CAPACITY_FAIL", None, None, n,
                              f"only {n} opponents; a correlation claim needs at least 3")
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        # Frozen by ERRATUM_02: degenerate statistic is a FAIL.
        return CapacityResult(
            "AFFORDANCE_CAPACITY_FAIL", None, None, n,
            "zero variance in offence or contest: every opponent allocates identically, "
            "so allocation diversity has not been expanded")

    r_hat = _pearson(x, y)
    if r_hat is None or not np.isfinite(r_hat):
        return CapacityResult("AFFORDANCE_CAPACITY_FAIL", None, None, n,
                              "Pearson r undefined")

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = []
    for _ in range(RESAMPLES):
        idx = rng.integers(0, n, n)
        r = _pearson(x[idx], y[idx])      # degenerate resamples are dropped, not imputed
        if r is not None and np.isfinite(r):
            draws.append(r)
    if len(draws) < RESAMPLES // 10:
        return CapacityResult("AFFORDANCE_CAPACITY_FAIL", round(r_hat, 4), None, n,
                              f"only {len(draws)}/{RESAMPLES} resamples were computable; "
                              f"the bootstrap fails closed rather than being waived")

    r_lcb = float(np.percentile(np.asarray(draws), LCB_PCT))
    point_ok = r_hat >= POINT_THRESHOLD
    ci_ok = r_lcb > HISTORICAL_R
    verdict = "AFFORDANCE_CAPACITY_PASS" if (point_ok and ci_ok) else "AFFORDANCE_CAPACITY_FAIL"

    if verdict == "AFFORDANCE_CAPACITY_PASS":
        reason = (f"r_hat {r_hat:.4f} >= {POINT_THRESHOLD} and "
                  f"LCB95 {r_lcb:.4f} > historical {HISTORICAL_R}")
    else:
        bits = []
        if not point_ok:
            bits.append(f"r_hat {r_hat:.4f} < {POINT_THRESHOLD}")
        if not ci_ok:
            bits.append(f"LCB95 {r_lcb:.4f} does not exceed historical {HISTORICAL_R}")
        reason = "; ".join(bits)

    return CapacityResult(verdict, round(r_hat, 4), round(r_lcb, 4), n, reason,
                          point_ok=point_ok, ci_ok=ci_ok)
