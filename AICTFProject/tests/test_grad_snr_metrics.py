"""Validate the policy-gradient SNR estimator on synthetic gradients.

The estimator is the whole experiment: if it does not have the null value it
claims, or cannot detect a signal we planted ourselves, then measuring it on
real rollouts tells us nothing. These tests are what licenses spending GPU time,
and they run in under a second with no GPU.
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.grad_snr_metrics import (advantage_stats, gradient_snr, gram,
                                          snr_spec, spec_index)

DIM = 400


def _pole(n, rng, sigma=1.0, mean=None, dim=DIM):
    """n minibatch gradients: a shared mean plus iid noise."""
    mu = np.zeros(dim) if mean is None else mean
    return [mu + sigma * rng.standard_normal(dim) for _ in range(n)]


def _run(gA, gB, **kw):
    G = gram(list(gA) + list(gB))
    return snr_spec(G, range(len(gA)), range(len(gA), len(gA) + len(gB)), **kw)


# ------------------------------------------------------------------ the null --
def test_identical_poles_give_ratio_one():
    """Both poles drawn from the same distribution: the estimator MUST sit at 1,
    or every cross-scale comparison is measuring its own bias."""
    rng = np.random.default_rng(0)
    r = _run(_pole(16, rng), _pole(16, rng), n_splits=500)
    assert r.lcb95 < 1.0 < r.ucb95
    assert abs(r.snr_spec - 1.0) < 0.05


def test_null_holds_across_noise_scales():
    """The null must not drift with gradient magnitude -- otherwise a scale with
    bigger gradients would look more 'specialized' for free."""
    for sigma in (0.01, 1.0, 100.0):
        rng = np.random.default_rng(1)
        r = _run(_pole(16, rng, sigma), _pole(16, rng, sigma), n_splits=400)
        assert abs(r.snr_spec - 1.0) < 0.06, f"null drifted at sigma={sigma}"


def test_null_holds_across_dimension():
    rng = np.random.default_rng(2)
    for d in (50, 2000):
        gA = [rng.standard_normal(d) for _ in range(16)]
        gB = [rng.standard_normal(d) for _ in range(16)]
        r = _run(gA, gB, n_splits=400)
        assert abs(r.snr_spec - 1.0) < 0.06, f"null drifted at dim={d}"


# --------------------------------------------------------------- the signal --
def test_planted_separation_is_detected():
    rng = np.random.default_rng(3)
    d = np.zeros(DIM); d[0] = 60.0                      # large, unambiguous
    r = _run(_pole(16, rng), _pole(16, rng, mean=d), n_splits=500)
    assert r.snr_spec > 3.0 and r.lcb95 > 1.0


def test_ratio_increases_monotonically_with_signal():
    rng = np.random.default_rng(4)
    got = []
    for mag in (0.0, 5.0, 15.0, 40.0):
        d = np.zeros(DIM); d[0] = mag
        got.append(_run(_pole(16, np.random.default_rng(5)),
                        _pole(16, np.random.default_rng(6), mean=d),
                        n_splits=400).snr_spec)
    assert got == sorted(got)
    assert abs(got[0] - 1.0) < 0.06


def test_ratio_falls_when_noise_grows_at_fixed_signal():
    """This IS the scaling hypothesis, simulated: hold the opponent-specific
    difference fixed and let per-minibatch noise grow. The specialization signal
    must become harder to distinguish."""
    d = np.zeros(DIM); d[0] = 20.0
    got = [_run(_pole(16, np.random.default_rng(7), sigma=s),
                _pole(16, np.random.default_rng(8), sigma=s, mean=d),
                n_splits=400).snr_spec
           for s in (0.5, 1.0, 2.0, 4.0)]
    assert got == sorted(got, reverse=True), f"expected monotone decline, got {got}"
    assert got[-1] < 1.5 and got[0] > 3.0


def _closed_form(k, mag, sigma, dim):
    """E[ratio] ~= sqrt(1 + k*||delta||^2 / (2 * sigma^2 * dim)).

    The `dim` term is not incidental: a half-mean difference under the null has
    norm sigma*sqrt(2*dim/k) because the noise is spread over every parameter,
    while a planted signal sits in a fixed subspace. Hence the second
    comparability precondition -- parameter DIMENSION must match across the team
    sizes being compared, exactly as samples-per-half must. Our architecture is
    byte-identical at 2v2/4v4/6v6 (3,473,592 parameters), so it does.
    """
    return np.sqrt(1.0 + k * mag**2 / (2 * sigma**2 * dim))


@pytest.mark.parametrize("k,mag,sigma", [(8, 10.0, 1.0), (8, 30.0, 1.0),
                                         (16, 10.0, 1.0), (8, 20.0, 2.0)])
def test_known_closed_form_is_recovered(k, mag, sigma):
    """Calibrated, not merely monotone."""
    d = np.zeros(DIM); d[0] = mag
    r = _run(_pole(2 * k, np.random.default_rng(9), sigma),
             _pole(2 * k, np.random.default_rng(10), sigma, mean=d), n_splits=800)
    predicted = _closed_form(k, mag, sigma, DIM)
    assert 0.85 * predicted < r.snr_spec < 1.15 * predicted, \
        f"got {r.snr_spec:.3f}, closed form {predicted:.3f}"


def test_ratio_depends_on_dimension_so_architectures_must_match():
    """Second comparability precondition, demonstrated. Identical signal and
    noise, different parameter count -> different ratio. Comparing team sizes
    with different architectures would be measuring the architecture."""
    k, mag, sigma = 8, 10.0, 1.0
    got = []
    for dim in (200, 1600):
        d = np.zeros(dim); d[0] = mag
        ra, rb = np.random.default_rng(21), np.random.default_rng(22)
        gA = [sigma * ra.standard_normal(dim) for _ in range(2 * k)]
        gB = [d + sigma * rb.standard_normal(dim) for _ in range(2 * k)]
        got.append(_run(gA, gB, n_splits=400).snr_spec)
    assert got[0] > got[1] * 1.3, f"expected dimension dependence, got {got}"


# ------------------------------------------------ the comparability guardrail --
def test_unequal_group_sizes_are_refused():
    """The ratio depends on samples-per-half. Comparing scales with different
    minibatch counts would silently invalidate the study, so it must raise."""
    rng = np.random.default_rng(11)
    with pytest.raises(ValueError, match="not comparable"):
        _run(_pole(16, rng), _pole(12, rng))


def test_too_few_or_odd_minibatches_refused():
    rng = np.random.default_rng(12)
    with pytest.raises(ValueError, match="even number"):
        _run(_pole(2, rng), _pole(2, rng))
    with pytest.raises(ValueError, match="even number"):
        _run(_pole(7, rng), _pole(7, rng))


def test_ratio_depends_on_k_so_scales_must_match_it():
    """Demonstrates WHY the guardrail exists: same underlying distributions,
    different k, materially different ratio."""
    d = np.zeros(DIM); d[0] = 8.0
    r_small = _run(_pole(8, np.random.default_rng(13)),
                   _pole(8, np.random.default_rng(14), mean=d), n_splits=400)
    r_large = _run(_pole(32, np.random.default_rng(15)),
                   _pole(32, np.random.default_rng(16), mean=d), n_splits=400)
    assert r_large.snr_spec > r_small.snr_spec * 1.3


# ------------------------------------------------------- supporting readouts --
def test_cosine_between_pole_means_tracks_alignment():
    rng = np.random.default_rng(17)
    shared = np.zeros(DIM); shared[1] = 50.0
    same = _run(_pole(16, rng, 0.5, mean=shared), _pole(16, rng, 0.5, mean=shared),
                n_splits=200)
    opposed = _run(_pole(16, rng, 0.5, mean=shared), _pole(16, rng, 0.5, mean=-shared),
                   n_splits=200)
    assert same.cos_pole_means > 0.9
    assert opposed.cos_pole_means < -0.9


def test_gradient_snr_matches_direct_computation():
    rng = np.random.default_rng(18)
    mu = np.zeros(DIM); mu[0] = 3.0
    g = _pole(24, rng, sigma=1.0, mean=mu)
    out = gradient_snr(gram(g), range(24))
    M = np.stack(g)
    gbar = M.mean(axis=0)
    assert np.isclose(out["signal_sq"], gbar @ gbar, rtol=1e-8)
    assert np.isclose(out["noise_var"], np.mean(((M - gbar) ** 2).sum(axis=1)), rtol=1e-6)


def test_gradient_snr_falls_as_noise_grows():
    mu = np.zeros(DIM); mu[0] = 3.0
    snrs = [gradient_snr(gram(_pole(24, np.random.default_rng(19), sigma=s, mean=mu)),
                         range(24))["snr"] for s in (0.5, 1.0, 2.0, 4.0)]
    assert snrs == sorted(snrs, reverse=True)


def test_advantage_stats_records_whether_normalized():
    a = np.random.default_rng(20).standard_normal(5000) * 3.0 + 1.0
    raw = advantage_stats(a, normalized=False)
    norm = advantage_stats((a - a.mean()) / a.std(), normalized=True)
    assert raw["normalized"] is False and norm["normalized"] is True
    assert abs(raw["std"] - 3.0) < 0.2
    assert abs(norm["std"] - 1.0) < 1e-6
    assert abs(norm["mean"]) < 1e-9


# ------------------------------------------- two-layer + per-module additions --
def test_null_control_returns_one_at_every_scale():
    """The per-scale validity gate: both groups from the same pole -> ratio 1."""
    from experiments.grad_snr_metrics import null_control
    for sigma in (0.2, 1.0, 5.0):
        rng = np.random.default_rng(30)
        g = _pole(32, rng, sigma)
        r = null_control(gram(g), range(32), n_splits=500)
        assert r.lcb95 < 1.0 < r.ucb95, f"null control off at sigma={sigma}"


def test_null_control_catches_a_contaminated_pole():
    """If one 'pole A' batch is secretly heterogeneous, the gate must notice --
    that is the artifact it exists to catch."""
    from experiments.grad_snr_metrics import null_control
    rng = np.random.default_rng(31)
    drift = np.zeros(DIM); drift[2] = 60.0
    g = _pole(16, rng) + _pole(16, rng, mean=drift)    # two populations in one pole
    assert null_control(gram(g), range(32), n_splits=500).lcb95 > 1.0


def test_gram_by_group_isolates_parameter_subsets():
    from experiments.grad_snr_metrics import gram_by_group
    rng = np.random.default_rng(32)
    head = np.zeros(DIM); head[0] = 40.0
    # backbone coherent across poles, head separated -- the case that matters
    gA = {"backbone": _pole(16, rng), "head": _pole(16, rng)}
    gB = {"backbone": _pole(16, rng), "head": _pole(16, rng, mean=head)}
    G = gram_by_group({k: gA[k] + gB[k] for k in gA})
    r_bb = snr_spec(G["backbone"], range(16), range(16, 32), n_splits=400)
    r_hd = snr_spec(G["head"], range(16), range(16, 32), n_splits=400)
    assert abs(r_bb.snr_spec - 1.0) < 0.08      # backbone shows no separation
    assert r_hd.lcb95 > 1.5                      # head does


# --------------------------------------------- dimension-normalized index -----
def _avg_index(dim, k, mag, sigma, n_reps, n_splits=250, seed0=0):
    """Mean spec_index over n_reps independent replicates.

    A SINGLE replicate's spec_index is amplified by dim/k (in production,
    dim/k ~ 3.47M/8 ~ 4.3e5) and can be enormous noise around the true value
    even when the estimator is correct. This is a real property of the metric,
    not a bug: it is why the frozen spec's inferential layer is the BOOTSTRAP
    MEAN over replicates, never a single replicate's number. These tests
    average over replicates for exactly that reason -- testing a single draw
    against a tight tolerance would be testing sampling noise, not the
    estimator.
    """
    d = np.zeros(dim); d[0] = mag
    vals = []
    for i in range(n_reps):
        ra = np.random.default_rng(seed0 + 2 * i)
        rb = np.random.default_rng(seed0 + 2 * i + 1)
        gA = _pole(2 * k, ra, sigma=sigma, dim=dim)
        gB = _pole(2 * k, rb, sigma=sigma, mean=d, dim=dim)
        r = snr_spec(gram(gA + gB), range(2 * k), range(2 * k, 4 * k),
                     n_splits=n_splits, dim=dim)
        vals.append(spec_index(r))
    return np.asarray(vals)


# ---- property 1: dimension invariance under the planted-signal model --------
def test_spec_index_removes_dimension_dependence_that_raw_ratio_has():
    """Direct fix for the PI's correction: raw snr_spec differs across
    differing-dimension modules with the SAME underlying signal/noise, but the
    dimension-normalized index must not (averaged over replicates -- see
    _avg_index)."""
    k, mag, sigma, n_reps = 8, 10.0, 1.0, 25
    raw_means, idx_means = [], []
    for dim in (200, 800, 3200):
        raws = []
        for i in range(n_reps):
            ra = np.random.default_rng(20_000 + i)
            rb = np.random.default_rng(30_000 + i)
            d = np.zeros(dim); d[0] = mag
            gA = _pole(2 * k, ra, sigma=sigma, dim=dim)
            gB = _pole(2 * k, rb, sigma=sigma, mean=d, dim=dim)
            r = snr_spec(gram(gA + gB), range(2 * k), range(2 * k, 4 * k),
                         n_splits=250, dim=dim)
            raws.append(r.snr_spec)
        raw_means.append(np.mean(raws))
        idx_means.append(np.mean(_avg_index(dim, k, mag, sigma, n_reps, seed0=50_000 + dim)))
    assert raw_means[0] > raw_means[1] > raw_means[2] * 1.15, \
        f"raw ratio should shrink with dim, got {raw_means}"
    spread = (max(idx_means) - min(idx_means)) / max(idx_means)
    assert spread < 0.35, f"spec_index should be roughly dimension-invariant, got {idx_means}"


def test_spec_index_dimension_invariance_holds_at_a_different_noise_scale():
    """Same property, different (mag, sigma) pair -- not a coincidence of one
    parameterisation."""
    k, mag, sigma, n_reps = 8, 6.0, 0.5, 25
    idx_means = [np.mean(_avg_index(dim, k, mag, sigma, n_reps, seed0=60_000 + dim))
                for dim in (300, 1200)]
    spread = abs(idx_means[0] - idx_means[1]) / max(abs(idx_means[0]), abs(idx_means[1]), 1e-9)
    assert spread < 0.4, f"expected rough dimension invariance, got {idx_means}"


# ---- property 2: null behaviour, any dimension or noise scale ---------------
def test_spec_index_null_is_dimension_and_noise_invariant():
    for dim, sigma in ((300, 0.3), (300, 5.0), (2000, 0.3), (2000, 5.0)):
        vals = _avg_index(dim, k=8, mag=0.0, sigma=sigma, n_reps=40, seed0=dim + int(sigma * 100))
        assert abs(vals.mean()) < 0.5, f"null drifted at dim={dim}, sigma={sigma}: {vals.mean()}"


def test_spec_index_single_replicate_is_noisy_but_the_mean_is_not():
    """States the amplification property explicitly rather than leaving it
    implicit: individual replicates disagree a lot; the mean over many is
    stable near the null. Both halves of this claim are checked."""
    vals = _avg_index(dim=1200, k=8, mag=0.0, sigma=1.0, n_reps=200, seed0=70_000)
    assert vals.std() > 1.0, "expected large single-replicate variance in this regime"
    assert abs(vals.mean()) < 0.4, "expected the many-replicate mean to sit near the null"


# ---- property 3: fail closed ------------------------------------------------
def test_spec_index_requires_dim():
    rng = np.random.default_rng(33)
    r = _run(_pole(16, rng), _pole(16, rng), n_splits=200)     # no dim supplied
    from experiments.grad_snr_metrics import spec_index
    with pytest.raises(ValueError, match="requires SNRResult.dim"):
        spec_index(r)


@pytest.mark.parametrize("bad_dim", [0, -5, 3.5, "400"])
def test_spec_index_rejects_invalid_dim(bad_dim):
    from experiments.grad_snr_metrics import spec_index
    rng = np.random.default_rng(35)
    r = _run(_pole(16, rng), _pole(16, rng), n_splits=200)
    r.dim = bad_dim
    with pytest.raises(ValueError, match="invalid dim"):
        spec_index(r)


@pytest.mark.parametrize("bad_dim", [0, -5, 3.5])
def test_snr_spec_rejects_invalid_dim_at_the_source(bad_dim):
    rng = np.random.default_rng(36)
    G = gram(_pole(16, rng) + _pole(16, rng))
    with pytest.raises(ValueError, match="invalid dim"):
        snr_spec(G, range(16), range(16, 32), n_splits=100, dim=bad_dim)


def test_gram_rejects_ragged_gradients():
    rng = np.random.default_rng(37)
    vecs = _pole(8, rng, dim=100) + _pole(8, rng, dim=50)
    with pytest.raises(ValueError, match="ragged"):
        gram(vecs)


def test_gram_rejects_non_finite_gradients():
    rng = np.random.default_rng(38)
    vecs = _pole(8, rng)
    vecs[3][0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        gram(vecs)
    vecs2 = _pole(8, np.random.default_rng(39))
    vecs2[0][1] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        gram(vecs2)


def test_gram_rejects_empty_input():
    with pytest.raises(ValueError, match="zero gradients"):
        gram([])


def test_snr_spec_rejects_non_finite_gram():
    rng = np.random.default_rng(44)
    G = gram(_pole(16, rng) + _pole(16, rng))
    G[0, 1] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        snr_spec(G, range(16), range(16, 32), n_splits=100)


def test_snr_spec_rejects_empty_groups():
    rng = np.random.default_rng(45)
    G = gram(_pole(16, rng))
    with pytest.raises(ValueError, match="even number"):
        snr_spec(G, [], [], n_splits=100)


def test_compare_by_index_raises_if_either_side_missing_dim():
    from experiments.grad_snr_metrics import compare_by_index
    rng = np.random.default_rng(46)
    with_dim = snr_spec(gram(_pole(16, rng) + _pole(16, rng)), range(16),
                       range(16, 32), n_splits=100, dim=DIM)
    without_dim = _run(_pole(16, rng), _pole(16, rng), n_splits=100)
    with pytest.raises(ValueError, match="requires SNRResult.dim"):
        compare_by_index(with_dim, without_dim)
    with pytest.raises(ValueError, match="requires SNRResult.dim"):
        compare_by_index(without_dim, with_dim)


def test_compare_by_index_works_across_different_dimensions():
    from experiments.grad_snr_metrics import compare_by_index
    rng = np.random.default_rng(47)
    d1 = np.zeros(300); d1[0] = 8.0
    d2 = np.zeros(1500); d2[0] = 8.0
    r_small = snr_spec(gram(_pole(16, rng, dim=300) + _pole(16, rng, mean=d1, dim=300)),
                       range(16), range(16, 32), n_splits=200, dim=300)
    r_big = snr_spec(gram(_pole(16, np.random.default_rng(48), dim=1500)
                         + _pole(16, np.random.default_rng(49), mean=d2, dim=1500)),
                     range(16), range(16, 32), n_splits=200, dim=1500)
    out = compare_by_index(r_small, r_big)
    assert np.isfinite(out["diff"]) and np.isfinite(out["fall_ratio_a_over_b"])


# ---- property 4: monotonicity at fixed dim, noise ---------------------------
def test_spec_index_increases_monotonically_with_signal_magnitude():
    dim, k, sigma, n_reps = 600, 8, 1.0, 20
    means = [np.mean(_avg_index(dim, k, mag, sigma, n_reps, seed0=80_000 + int(mag)))
             for mag in (0.0, 4.0, 10.0, 20.0)]
    assert means == sorted(means), f"expected monotone increase, got {means}"
    assert means[0] < 0.5 < means[-1]


def test_spec_index_still_detects_signal_where_it_exists():
    from experiments.grad_snr_metrics import spec_index
    rng = np.random.default_rng(42)
    d = np.zeros(DIM); d[0] = 50.0
    r = snr_spec(gram(_pole(16, rng) + _pole(16, rng, mean=d)),
                range(16), range(16, 32), n_splits=500, dim=DIM)
    assert spec_index(r) > 1.0


def test_null_control_carries_dim_through_every_constructor_path():
    """Every SNRResult must carry the same dimension metadata, not just the
    one produced directly by snr_spec(). null_control is a second constructor
    path (it calls snr_spec internally) and must forward dim identically."""
    from experiments.grad_snr_metrics import null_control, spec_index
    rng = np.random.default_rng(43)
    r = null_control(gram(_pole(32, rng)), range(32), n_splits=300, dim=DIM)
    assert r.dim == DIM
    val = spec_index(r)                 # must not raise; must be finite
    assert np.isfinite(val)


# ------------------------------------------- exact within-module cancellation -
def test_within_module_fall_ratio_is_identical_via_raw_delta_or_spec_index():
    """THE key structural fact that limits the blast radius of the dim/k
    amplification problem: a SAME-MODULE, SAME-k fall ratio across scales is
    IDENTICAL whether computed from spec_index or from raw (snr_spec^2-1),
    to floating-point precision, because the dim/k factor cancels exactly.
    This is why grad_snr_decision.py's routing uses raw deltas and needs no
    dimension normalization at all -- verified here, not just argued."""
    dim, k = 50_000, 8
    d = np.zeros(dim); d[0] = 12.0

    def one(seed_a, seed_b, mean_b):
        ra, rb = np.random.default_rng(seed_a), np.random.default_rng(seed_b)
        gA = [ra.standard_normal(dim) for _ in range(2 * k)]
        gB = [mean_b + rb.standard_normal(dim) for _ in range(2 * k)]
        return snr_spec(gram(gA + gB), range(2 * k), range(2 * k, 4 * k),
                        n_splits=200, dim=dim)

    r_a = one(1, 2, d)
    r_b = one(3, 4, d * 0.4)
    via_index = spec_index(r_a) / spec_index(r_b)
    via_raw = (r_a.snr_spec ** 2 - 1) / (r_b.snr_spec ** 2 - 1)
    assert np.isclose(via_index, via_raw, rtol=1e-9)


def test_cross_module_different_dim_does_NOT_cancel():
    """The contrapositive, equally important: when dim genuinely differs
    (comparing two DIFFERENT modules), spec_index and raw delta diverge --
    which is exactly why a genuine cross-module comparison needs
    aggregate_spec_index and cannot use a raw-delta ratio."""
    k, mag, sigma = 8, 10.0, 1.0
    ra, rb = np.random.default_rng(200), np.random.default_rng(201)
    d_small = np.zeros(300); d_small[0] = mag
    d_big = np.zeros(9000); d_big[0] = mag
    r_small = snr_spec(gram(_pole(2 * k, ra, sigma, dim=300)
                           + _pole(2 * k, rb, sigma, mean=d_small, dim=300)),
                       range(2 * k), range(2 * k, 4 * k), n_splits=200, dim=300)
    r_big = snr_spec(gram(_pole(2 * k, np.random.default_rng(202), sigma, dim=9000)
                         + _pole(2 * k, np.random.default_rng(203), sigma,
                                mean=d_big, dim=9000)),
                     range(2 * k), range(2 * k, 4 * k), n_splits=200, dim=9000)
    raw_ratio = (r_small.snr_spec ** 2 - 1) / (r_big.snr_spec ** 2 - 1)
    index_ratio = spec_index(r_small) / spec_index(r_big)
    assert not np.isclose(raw_ratio, index_ratio, rtol=0.5), \
        "expected these to meaningfully differ when dim is not held fixed"


# --------------------------------------------------- aggregate_spec_index ----
def test_aggregate_spec_index_refuses_too_few_replicates():
    from experiments.grad_snr_metrics import (aggregate_spec_index, null_control)
    rng = np.random.default_rng(50)
    reps = [snr_spec(gram(_pole(16, rng) + _pole(16, rng)), range(16),
                     range(16, 32), n_splits=100, dim=DIM) for _ in range(3)]
    nc = null_control(gram(_pole(32, rng)), range(32), n_splits=100, dim=DIM)
    with pytest.raises(ValueError, match="need >= "):
        aggregate_spec_index(reps, nc, min_replicates=8)


def test_aggregate_spec_index_refuses_a_failed_null_control():
    from experiments.grad_snr_metrics import aggregate_spec_index, SNRResult
    rng = np.random.default_rng(51)
    reps = [snr_spec(gram(_pole(16, rng) + _pole(16, rng)), range(16),
                     range(16, 32), n_splits=100, dim=DIM) for _ in range(10)]
    fake_failed_null = reps[0]
    fake_failed_null = SNRResult(**{**fake_failed_null.__dict__,
                                    "lcb95": 1.2, "ucb95": 1.5})  # excludes 1.0
    with pytest.raises(ValueError, match="null control failed"):
        aggregate_spec_index(reps, fake_failed_null)


def test_aggregate_spec_index_returns_stable_aggregate_under_the_null():
    """The null-control gate is a 95% CI and therefore has an inherent ~5%
    false-rejection rate under the null -- that is a feature (it is doing its
    job), not a bug to seed-hunt around. A REAL null control is computed once
    per scale from as much data as convenient, not tied to one replicate's k,
    so it is built here from a much larger sample specifically to keep this
    unit test deterministic rather than flaky."""
    from experiments.grad_snr_metrics import aggregate_spec_index, null_control
    dim, k, n_reps = 800, 8, 20
    reps = []
    for i in range(n_reps):
        ra, rb = np.random.default_rng(6000 + 2 * i), np.random.default_rng(6000 + 2 * i + 1)
        reps.append(snr_spec(gram(_pole(2 * k, ra, dim=dim) + _pole(2 * k, rb, dim=dim)),
                             range(2 * k), range(2 * k, 4 * k), n_splits=150, dim=dim))
    nc = null_control(gram(_pole(400, np.random.default_rng(7000), dim=dim)),
                      range(400), n_splits=500, dim=dim)
    out = aggregate_spec_index(reps, nc)
    assert out["n_replicates"] == n_reps
    assert out["lcb95"] < 0 < out["ucb95"] or abs(out["mean"]) < 1.0
    assert out["per_replicate_std"] > 0.1        # confirms single-replicate noise is real


def test_aggregate_spec_index_detects_a_real_signal():
    from experiments.grad_snr_metrics import aggregate_spec_index, null_control
    dim, k, n_reps = 800, 8, 20
    d = np.zeros(dim); d[0] = 40.0
    reps = []
    for i in range(n_reps):
        ra, rb = np.random.default_rng(8000 + 2 * i), np.random.default_rng(8000 + 2 * i + 1)
        reps.append(snr_spec(gram(_pole(2 * k, ra, dim=dim) + _pole(2 * k, rb, mean=d, dim=dim)),
                             range(2 * k), range(2 * k, 4 * k), n_splits=150, dim=dim))
    nc = null_control(gram(_pole(4 * k, np.random.default_rng(9000), dim=dim)),
                      range(4 * k), n_splits=300, dim=dim)
    out = aggregate_spec_index(reps, nc)
    assert out["lcb95"] > 0, f"expected a clearly positive aggregate, got {out}"


# --------------------------------------------------------- power analysis ----
def test_null_std_scaling_probe_confirms_sqrt_dim_model():
    from experiments.grad_snr_metrics import null_std_scaling_probe
    cal = null_std_scaling_probe(dims=(200, 800, 3200), k=8, n_probe=150)
    Cs = cal["fitted_C_per_dim"]
    assert max(Cs) / min(Cs) < 2.0, f"fitted C should be roughly stable, got {Cs}"


def test_required_replicates_grows_with_dimension_and_shrinks_with_target_sem():
    from experiments.grad_snr_metrics import null_std_scaling_probe, required_replicates
    cal = null_std_scaling_probe(dims=(200, 800, 3200), k=8, n_probe=150)
    r_small_dim = required_replicates(1000, 8, target_sem=1.0, calibration=cal)
    r_big_dim = required_replicates(1_000_000, 8, target_sem=1.0, calibration=cal)
    assert r_big_dim["required_replicates"] > r_small_dim["required_replicates"]
    r_loose = required_replicates(1_000_000, 8, target_sem=10.0, calibration=cal)
    assert r_loose["required_replicates"] < r_big_dim["required_replicates"]


def test_required_replicates_at_production_scale_is_documented_as_infeasible_at_R16():
    """Locks in the finding that motivated this whole detour: at the actor's
    real dimension (3,473,592) and k=8, the replicate count needed for a
    target SEM of 1.0 vastly exceeds the frozen spec's replicates_R=16. This
    is the reason cross-module magnitude comparison is kept OUT of the
    primary decision routing."""
    from experiments.grad_snr_metrics import null_std_scaling_probe, required_replicates
    cal = null_std_scaling_probe(dims=(200, 800, 3200, 12800), k=8, n_probe=200)
    out = required_replicates(3_473_592, 8, target_sem=1.0, calibration=cal)
    assert out["required_replicates"] > 1000, \
        f"expected the production case to be clearly infeasible at R=16, got {out}"
