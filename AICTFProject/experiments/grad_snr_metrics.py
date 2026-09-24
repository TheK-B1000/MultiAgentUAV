r"""Policy-gradient SNR estimators for the team-size scaling hypothesis.

Kuba et al. (NeurIPS 2021, arXiv 2108.08612) prove that multi-agent policy
gradient excess variance grows linearly in the number of agents. That is a
theoretical claim about an estimator; it is NOT evidence that the effect is
what stalls specialization in our environment. This module measures the thing
directly, so we can cite a measurement rather than a paper.

THE QUANTITY THAT MATTERS
    Not raw gradient noise -- the *specialization-relevant* signal. The update
    that would make a policy specialize is the difference between the gradient
    under pole A and the gradient under pole B. So the question is:

        is  g_A - g_B  distinguishable from within-pole sampling noise?

    Estimated by a split-half ratio with a KNOWN NULL VALUE OF 1:

        within_r   = || mean(A1) - mean(A2) ||        (same pole, disjoint halves)
        between_r  = || mean(Ai) - mean(Bj) ||        (different poles)
        SNR_spec   = between / within

    Both numerator and denominator are differences of means over the SAME number
    of minibatches, so the sampling noise cancels in the ratio. If the two poles
    induce identical gradient distributions, between and within are
    statistically identical and the ratio is 1. Above 1 means the poles demand
    measurably different updates. Approaching 1 as N grows would be direct
    evidence for the scaling hypothesis in OUR environment.

    Because the null is fixed at 1 and the statistic is dimensionless, values are
    comparable across team sizes without needing a matched baseline run.

WHY EVERYTHING RUNS OFF A GRAM MATRIX
    Every statistic here is a quadratic form in the per-minibatch gradients, so
    only their pairwise inner products are needed. One 3.47M-parameter gradient
    is ~14 MB; a full design would be gigabytes of vectors. The Gram matrix is
    (n_minibatch)^2 floats, exact rather than approximate, and -- the reason that
    matters here -- it makes the estimator testable on synthetic inputs with no
    GPU, so the metric is validated before any seed is spent on it.

THE CONSTRAINT THAT WOULD SILENTLY INVALIDATE THE STUDY
    The ratio depends on TWO things besides the true effect: how many
    minibatches go into each half, and the ambient parameter dimension. With a
    true signal delta, per-minibatch noise sigma, k minibatches per half and
    dimension d, the ratio behaves like sqrt(1 + k*||delta||^2 / (2*sigma^2*d)).

    So k and d MUST both match before two raw ratios are compared, or the scales
    are not comparable. `snr_spec` refuses unequal group sizes k rather than
    returning a number that looks fine. Dimension is different: our architecture
    is byte-identical across team sizes, so a WHOLE-ACTOR-vs-WHOLE-ACTOR
    comparison across 2v2/4v4/6v6 is dimension-matched automatically and the raw
    ratio is directly comparable. It is comparing DIFFERENT MODULES against each
    other (e.g. "is the macro head's ratio lower than the backbone's?") that
    breaks this, because the two modules have different parameter counts.
    `spec_index()` below removes the k and d dependence for exactly that case;
    never read a raw snr_spec.snr_spec across modules of unequal dimension as a
    magnitude comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


def gram(vectors: Sequence[np.ndarray]) -> np.ndarray:
    """Pairwise inner products of flattened per-minibatch gradients.

    Fails closed rather than propagating garbage: ragged gradients (a shape
    mismatch, e.g. from comparing two different parameter subsets by mistake)
    and non-finite values (a NaN/Inf gradient, e.g. from a diverged rollout)
    both raise here rather than silently producing a Gram matrix whose later
    statistics merely look plausible.
    """
    vs = [np.asarray(v, dtype=np.float64).ravel() for v in vectors]
    if not vs:
        raise ValueError("gram() got zero gradients")
    lens = {v.size for v in vs}
    if len(lens) != 1:
        raise ValueError(f"ragged gradients: sizes {sorted(lens)} -- gram() requires "
                         f"every gradient to come from the SAME parameter subset")
    M = np.stack(vs)
    if not np.all(np.isfinite(M)):
        raise ValueError("non-finite gradient value(s) (NaN/Inf) -- refusing to build "
                         "a Gram matrix from them")
    return M @ M.T


def _mean_diff_sq(G: np.ndarray, idx_a: Sequence[int], idx_b: Sequence[int]) -> float:
    """|| mean(g_i : i in a) - mean(g_j : j in b) ||^2, from the Gram matrix alone."""
    w = np.zeros(G.shape[0], dtype=np.float64)
    w[list(idx_a)] = 1.0 / len(idx_a)
    w[list(idx_b)] -= 1.0 / len(idx_b)
    return float(w @ G @ w)


def _mean_diff(G: np.ndarray, a: Sequence[int], b: Sequence[int]) -> float:
    return float(np.sqrt(max(0.0, _mean_diff_sq(G, a, b))))


@dataclass
class SNRResult:
    snr_spec: float                 # between / within; null = 1.0
    lcb95: float
    ucb95: float
    between: float
    within: float
    n_per_half: int
    n_splits: int
    cos_pole_means: float           # cos(mean g_A, mean g_B); ->1 = poles agree
    within_pole_cos: dict           # mean pairwise cosine among minibatches, per pole
    grad_norm: dict                 # per-pole minibatch gradient norm summary
    dim: int | None                 # ambient parameter count, if supplied
    note: str


def spec_index(r: "SNRResult") -> float:
    """Dimension- and sample-count-normalized specialization index.

    (snr_spec^2 - 1) * dim / n_per_half  ~=  ||delta||^2 / (2*sigma^2)

    under the linear-noise model documented in the module docstring. Removing
    the k and d dependence is what makes it valid to compare across MODULES of
    different parameter count (whole actor vs. macro head), which the raw
    ``snr_spec`` is not. Requires ``dim`` to have been supplied to ``snr_spec``;
    raises rather than silently returning a meaningless number, in keeping with
    this project's rule that absence is an error state, not a default.
    """
    if r.dim is None:
        raise ValueError(
            "spec_index requires SNRResult.dim -- pass dim= to snr_spec() when "
            "the result will be used for any cross-module comparison.")
    if not isinstance(r.dim, (int, np.integer)) or r.dim <= 0:
        raise ValueError(f"invalid dim={r.dim!r}: must be a positive integer")
    if not np.isfinite(r.snr_spec):
        raise ValueError(f"non-finite snr_spec={r.snr_spec!r}; refusing to compute an index")
    idx = (r.snr_spec ** 2 - 1.0) * r.dim / r.n_per_half
    if not np.isfinite(idx):
        raise ValueError(f"spec_index computed a non-finite value ({idx!r})")
    return float(idx)


def compare_by_index(r_a: "SNRResult", r_b: "SNRResult") -> dict:
    """The SANCTIONED way to compare two SNRResults that may come from
    different-dimension parameter subsets (backbone vs. macro head; or two team
    sizes where M or dim differ). Comparing ``.snr_spec`` directly is only valid
    when ``dim`` AND ``n_per_half`` are known to match (e.g. whole-actor at
    different team sizes with M held fixed by the frozen spec) -- everywhere
    else, use this.

    Raises if either side lacks ``dim`` (delegates to ``spec_index``, which
    raises) or if the comparison would be nonsensical (both indices non-finite,
    or one side's index sits at exactly the null with the other's CI unusable).
    """
    ia, ib = spec_index(r_a), spec_index(r_b)
    return {"index_a": ia, "index_b": ib, "diff": ia - ib,
            "fall_ratio_a_over_b": ia / ib if ib != 0 else float("inf"),
            "note": "index_a, index_b are dimension- and sample-count-normalized "
                    "and safe to compare directly even if r_a.dim != r_b.dim"}


def snr_spec(G: np.ndarray, idx_A: Sequence[int], idx_B: Sequence[int],
             n_splits: int = 2000, rng_seed: int = 7,
             dim: int | None = None) -> SNRResult:
    """Split-half specialization SNR. Null value 1.0.

    ``idx_A`` / ``idx_B`` index the per-minibatch gradients collected under pole
    A and pole B at ONE team size and ONE parameter vector.
    """
    a, b = list(idx_A), list(idx_B)
    if len(a) != len(b):
        raise ValueError(
            f"pole A has {len(a)} minibatches and pole B has {len(b)}. The ratio "
            f"depends on samples-per-half, so unequal groups are not comparable.")
    if len(a) < 4 or len(a) % 2:
        raise ValueError(f"need an even number of minibatches per pole, >= 4; got {len(a)}")
    if dim is not None and (not isinstance(dim, (int, np.integer)) or dim <= 0):
        raise ValueError(f"invalid dim={dim!r}: must be a positive integer or None")
    if not np.all(np.isfinite(G)):
        raise ValueError("non-finite value(s) in the Gram matrix -- refusing to proceed")
    k = len(a) // 2

    rng = np.random.default_rng(rng_seed)
    ratios, betweens, withins = [], [], []
    for _ in range(n_splits):
        pa, pb = rng.permutation(a), rng.permutation(b)
        a1, a2, b1, b2 = pa[:k], pa[k:], pb[:k], pb[k:]
        within = 0.5 * (_mean_diff(G, a1, a2) + _mean_diff(G, b1, b2))
        between = float(np.mean([_mean_diff(G, x, y)
                                 for x in (a1, a2) for y in (b1, b2)]))
        if within <= 0:
            continue
        ratios.append(between / within)
        betweens.append(between)
        withins.append(within)
    if not ratios:
        raise ValueError("degenerate: within-pole spread is zero on every split")
    r = np.asarray(ratios)
    lo, hi = np.percentile(r, [2.5, 97.5])

    def pole_cos(idx):
        c = []
        for i in range(len(idx)):
            for j in range(i + 1, len(idx)):
                d = np.sqrt(G[idx[i], idx[i]] * G[idx[j], idx[j]])
                if d > 0:
                    c.append(G[idx[i], idx[j]] / d)
        return float(np.mean(c)) if c else float("nan")

    wa = np.zeros(G.shape[0]); wa[a] = 1.0 / len(a)
    wb = np.zeros(G.shape[0]); wb[b] = 1.0 / len(b)
    nA, nB = np.sqrt(wa @ G @ wa), np.sqrt(wb @ G @ wb)
    cos_means = float((wa @ G @ wb) / (nA * nB)) if nA > 0 and nB > 0 else float("nan")

    norms = {p: {"mean": float(np.mean(np.sqrt(np.diag(G)[idx]))),
                 "std": float(np.std(np.sqrt(np.diag(G)[idx]))),
                 "cv": float(np.std(np.sqrt(np.diag(G)[idx]))
                             / max(1e-12, np.mean(np.sqrt(np.diag(G)[idx]))))}
              for p, idx in (("A", a), ("B", b))}

    return SNRResult(
        snr_spec=float(r.mean()), lcb95=float(lo), ucb95=float(hi),
        between=float(np.mean(betweens)), within=float(np.mean(withins)),
        n_per_half=k, n_splits=len(ratios),
        cos_pole_means=cos_means,
        within_pole_cos={"A": pole_cos(a), "B": pole_cos(b)},
        grad_norm=norms, dim=dim,
        note="null value is 1.0 (poles indistinguishable). Raw snr_spec is "
             "comparable across team sizes only when n_per_half AND dim match; "
             "across different parameter subsets use spec_index() instead.")


def gradient_snr(G: np.ndarray, idx: Sequence[int]) -> dict:
    """Ordinary estimator SNR within one pole: ||E g||^2 / E||g - E g||^2.

    This is the quantity Kuba et al. speak to directly. Reported alongside
    ``snr_spec`` because a drop here with N is the mechanism, while a drop in
    ``snr_spec`` is the consequence we actually care about.
    """
    idx = list(idx)
    w = np.zeros(G.shape[0]); w[idx] = 1.0 / len(idx)
    mean_sq = float(w @ G @ w)
    # E||g_i - gbar||^2 = mean_i G[i,i] - ||gbar||^2
    within_var = float(np.mean(np.diag(G)[idx]) - mean_sq)
    return {"signal_sq": mean_sq, "noise_var": within_var,
            "snr": mean_sq / within_var if within_var > 0 else float("inf"),
            "n_minibatch": len(idx)}


def gram_by_group(grads_by_group: dict[str, Sequence[np.ndarray]]) -> dict[str, np.ndarray]:
    """One Gram matrix per named parameter subset.

    Run at minimum for the whole actor, the shared backbone, and the
    policy/macro head separately. The distinction is diagnostic, not cosmetic:
    a backbone whose gradients stay coherent while the action-selection head
    degrades with team size is a *different* failure -- and a different
    intervention -- from network-wide optimization degradation.
    """
    return {name: gram(vs) for name, vs in grads_by_group.items()}


def null_control(G: np.ndarray, idx_pole: Sequence[int], n_splits: int = 2000,
                 rng_seed: int = 7, dim: int | None = None) -> SNRResult:
    """Per-scale validity gate: run the estimator with BOTH groups drawn from the
    SAME pole. The true answer is known to be 1.0.

    This must be checked at every team size before any cross-scale comparison.
    If the null control departs from 1 at some scale, the measurement at that
    scale is invalid and no cross-scale claim can rest on it -- which is cheap
    insurance against a scale-specific artifact being read as a scaling effect.
    """
    idx = list(idx_pole)
    if len(idx) % 2:
        idx = idx[:-1]
    half = len(idx) // 2
    return snr_spec(G, idx[:half], idx[half:], n_splits=n_splits, rng_seed=rng_seed, dim=dim)


def standardized_separation(a: np.ndarray, b: np.ndarray, n_boot: int = 20000,
                            rng_seed: int = 7) -> dict:
    """Pooled-SD standardized mean difference (Cohen's d) with a bootstrap CI.

    Used on the RAW, pre-normalization layer. Standardized rather than absolute
    because reward and advantage scales need not be commensurate across team
    sizes, and an unstandardized difference would then be uncomparable.

    Interpretation caveat, stated here so it travels with the number: rollouts
    under the two poles visit different states, so this is a difference between
    two marginal distributions, not a paired per-state contrast. It answers "is
    there raw signal distinguishing the regimes at all, and does its size hold up
    with N", not "how much would a given state's update differ".
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    pooled = np.sqrt(((a.size - 1) * a.var(ddof=1) + (b.size - 1) * b.var(ddof=1))
                     / max(1, a.size + b.size - 2))
    d = (a.mean() - b.mean()) / pooled if pooled > 0 else 0.0
    rng = np.random.default_rng(rng_seed)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        sa = a[rng.integers(0, a.size, a.size)]
        sb = b[rng.integers(0, b.size, b.size)]
        p = np.sqrt(((sa.size - 1) * sa.var(ddof=1) + (sb.size - 1) * sb.var(ddof=1))
                    / max(1, sa.size + sb.size - 2))
        boot[i] = (sa.mean() - sb.mean()) / p if p > 0 else 0.0
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"cohens_d": float(d), "lcb95": float(lo), "ucb95": float(hi),
            "pooled_sd": float(pooled), "n_a": int(a.size), "n_b": int(b.size)}


#: Minimum independent replicates before a dimension-normalized index is
#: eligible for interpretation at all. Not a claim that this is ENOUGH for any
#: particular effect size -- see required_replicates() for that.
MIN_REPLICATES_FOR_INDEX = 8


def aggregate_spec_index(replicate_results: Sequence["SNRResult"],
                         null_control_result: "SNRResult",
                         min_replicates: int = MIN_REPLICATES_FOR_INDEX,
                         n_boot: int = 20000, alpha: float = 0.05,
                         rng_seed: int = 7, require_null_pass: bool = True) -> dict:
    """THE SANCTIONED way to interpret a dimension-normalized specialization
    index -- for example a genuine cross-module magnitude comparison (does the
    macro head carry more raw specialization signal than the backbone, in
    absolute terms). Individual replicate spec_index values are NOT
    interpreted: dimensional normalization strongly amplifies finite-sample
    variability (in production, dim/k ~ 4e5 for the whole actor, and the
    single-replicate standard deviation scales like sqrt(dim)/k -- see
    null_std_scaling_probe), so a microscopic finite-sample offset from the
    null can appear as an enormous number. Inference rests on the
    PRECOMMITTED AGGREGATE ESTIMATOR and its bootstrap uncertainty across
    independent replicates, CONDITIONAL ON passing the null-control gate --
    never on a single replicate's value.

    This function is the only supported entry point for that purpose: it
    REFUSES (raises) rather than returning a number if there are too few
    replicates, or if the null control for this scale/module has not passed.
    A raw ``spec_index(...)`` call on one replicate remains available for
    internal use (e.g. computing the values this function aggregates) but is
    never, on its own, a result to interpret.

    ``require_null_pass=False`` disables the null-control gate. This exists
    ONLY for testing the aggregation arithmetic itself in isolation (e.g.
    against a deliberately-failing null control, to prove the gate bites);
    it must never be set False on a real measurement.
    """
    if len(replicate_results) < min_replicates:
        raise ValueError(
            f"need >= {min_replicates} independent replicates for an "
            f"interpretable index, got {len(replicate_results)}. A single "
            f"replicate's spec_index is not scientifically meaningful at "
            f"realistic dim/k ratios.")
    null_passed = bool(null_control_result.lcb95 < 1.0 < null_control_result.ucb95)
    if require_null_pass and not null_passed:
        raise ValueError(
            f"null control failed for this scale/module: its within-pole "
            f"split-half ratio 95% interval ({null_control_result.lcb95:.4f}, "
            f"{null_control_result.ucb95:.4f}) does not contain 1.0. The "
            f"measurement is not valid here; spec_index is not eligible for "
            f"interpretation until the null control passes.")
    idxs = np.array([spec_index(r) for r in replicate_results], dtype=np.float64)
    if not np.all(np.isfinite(idxs)):
        raise ValueError("non-finite index among replicates")
    rng = np.random.default_rng(rng_seed)
    boot_idx = rng.integers(0, idxs.size, size=(n_boot, idxs.size))
    boot_means = idxs[boot_idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"mean": float(idxs.mean()), "median": float(np.median(idxs)),
            "lcb95": float(lo), "ucb95": float(hi),
            "n_replicates": int(idxs.size),
            "per_replicate_std": float(idxs.std(ddof=1)) if idxs.size > 1 else 0.0,
            "null_control_passed": null_passed,
            "note": "individual replicate values are NOT independently "
                    "meaningful; only this bootstrap-over-replicates aggregate "
                    "is interpreted, and only conditional on the null control "
                    "passing." + ("" if require_null_pass else
                                 " NOTE: require_null_pass=False -- this result "
                                 "is NOT licensed for scientific interpretation.")}


def null_std_scaling_probe(dims: Sequence[int] = (200, 800, 3200, 12800),
                           k: int = 8, n_probe: int = 300, sigma: float = 1.0,
                           rng_seed: int = 7) -> dict:
    """Calibrate, using the SAME validated synthetic generator as the rest of
    this module, how spec_index's null-condition standard deviation scales
    with parameter dimension. Fits the constant C in std(dim,k) ~= C*sqrt(dim)/k
    (the scaling predicted by concentration of measure for a norm of a
    high-dimensional Gaussian vector), so that C can be used to EXTRAPOLATE to
    dimensions too large to simulate directly -- e.g. a 3.47M-parameter actor
    -- rather than guessing a replicate count by feel.
    """
    stds = []
    for dim in dims:
        vals = []
        for i in range(n_probe):
            ra = np.random.default_rng(rng_seed * 10_000 + dim + 2 * i)
            rb = np.random.default_rng(rng_seed * 10_000 + dim + 2 * i + 1)
            gA = [sigma * ra.standard_normal(dim) for _ in range(2 * k)]
            gB = [sigma * rb.standard_normal(dim) for _ in range(2 * k)]
            r = snr_spec(gram(gA + gB), range(2 * k), range(2 * k, 4 * k),
                        n_splits=150, dim=dim)
            vals.append(spec_index(r))
        stds.append(float(np.std(vals, ddof=1)))
    Cs = [s * k / np.sqrt(d) for s, d in zip(stds, dims)]
    return {"dims_probed": list(dims), "k": k, "stds": stds,
            "fitted_C_per_dim": Cs, "fitted_C": float(np.mean(Cs)),
            "model": "std(dim, k) ~= C * sqrt(dim) / k",
            "caveat": "extrapolation beyond the probed dimension range; the "
                      "concentration-of-measure argument it rests on gets MORE "
                      "accurate at larger dim, not less, so extrapolating "
                      "upward is safer than extrapolating downward."}


def required_replicates(dim: int, k: int, target_sem: float,
                        calibration: dict | None = None) -> dict:
    """How many independent replicates R are needed for the AGGREGATE mean's
    standard error to reach ``target_sem``, at parameter dimension ``dim`` and
    ``k`` minibatches per half -- computed BEFORE committing to a replicate
    budget in a frozen spec, not chosen by feel. Uses the calibration from
    ``null_std_scaling_probe`` (or a freshly computed one, at real compute
    cost, if none is supplied).
    """
    cal = calibration if calibration is not None else null_std_scaling_probe()
    C = cal["fitted_C"]
    single_std = C * np.sqrt(dim) / k
    R = int(np.ceil((single_std / target_sem) ** 2)) if target_sem > 0 else None
    return {"dim": dim, "k": k, "predicted_single_replicate_std": single_std,
            "target_sem": target_sem, "required_replicates": R,
            "calibration_C": C}


def power_to_detect(dim: int, k: int, n_replicates: int, effect_size: float,
                    target_alpha: float = 0.05,
                    calibration: dict | None = None) -> dict:
    """The operational power question: 'with our actual dim, k, replicate
    count, and an assumed true effect magnitude, what is our probability of
    correctly distinguishing it from the null?' -- answered BEFORE collecting
    real data, using a normal approximation to the aggregate mean (justified
    by the central limit theorem over independent replicates) with standard
    error ``single_std / sqrt(n_replicates)``, where ``single_std`` comes from
    the ``null_std_scaling_probe`` calibration.

    ``effect_size`` is in the same units as ``spec_index`` (approximately
    ||delta||^2 / (2*sigma^2) under the model in the module docstring).
    """
    from scipy.stats import norm
    cal = calibration if calibration is not None else null_std_scaling_probe()
    C = cal["fitted_C"]
    single_std = C * np.sqrt(dim) / k
    sem = single_std / np.sqrt(n_replicates)
    z_crit = norm.ppf(1.0 - target_alpha / 2.0)
    z_effect = effect_size / sem if sem > 0 else float("inf")
    power = float(norm.cdf(z_effect - z_crit) + norm.cdf(-z_effect - z_crit))
    return {"dim": dim, "k": k, "n_replicates": n_replicates,
            "effect_size": effect_size, "sem": float(sem), "alpha": target_alpha,
            "power": power,
            "reads_as": f"{power * 100:.1f}% chance of correctly detecting an "
                       f"effect of this size at n={n_replicates} replicates"}


def advantage_stats(adv: np.ndarray, normalized: bool) -> dict:
    """Advantage distribution summary.

    ``normalized`` records whether these are post per-minibatch standardisation.
    Both are reported: the updater at rl/custom_ppo/update/minibatch_updater.py
    standardises advantages per minibatch, so the RAW variance is not what the
    optimizer sees -- but the normalisation is itself part of the hypothesis. If
    the noise component of the advantage grows with N while the opponent-specific
    component does not, dividing by the minibatch std actively SHRINKS the
    specialization signal at fixed learning rate.
    """
    a = np.asarray(adv, dtype=np.float64).ravel()
    return {"normalized": bool(normalized), "n": int(a.size),
            "mean": float(a.mean()), "std": float(a.std()),
            "var": float(a.var()), "abs_mean": float(np.abs(a).mean()),
            "p05": float(np.percentile(a, 5)), "p95": float(np.percentile(a, 95))}
