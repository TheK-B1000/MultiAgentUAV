"""Prove the five-way (plus ambiguous) decision router before any real data
exists.

Per the PI: this must be regression-tested, not left as prose applied by eye.
Six synthetic scenarios below construct stats dicts that unambiguously satisfy
exactly one branch of experiments/grad_snr_decision.py::classify_outcome, and
one test locks the structural guarantee that makes the router safe from the
dim/k amplification problem discovered in grad_snr_metrics.py: the router
depends only on raw deltas, so changing dim alone (with every raw measurement
held fixed) cannot change the verdict.
"""

from __future__ import annotations

import copy

import pytest

from experiments.grad_snr_decision import VERDICTS, classify_outcome

THRESHOLDS = {
    "kuba_fall_supported_min": 2.0,
    "kuba_fall_refute_max": 1.3,
    "layer1_fall_weakening_min": 2.0,
    "cos_drop_interference_min": 0.15,
    "norm_cv_growth_interference_min": 1.5,
    "head_fall_min": 2.0,
    "backbone_fall_max": 1.3,
}


def _base_stats(**overrides) -> dict:
    """A scenario with NO degradation anywhere: every fall ratio is 1.0, every
    CI is tight and non-overlapping-with-null in the boring way, cosine and
    norm CV unchanged. Individual tests override just what their branch needs."""
    stats = {
        "delta_2v2": 4.0, "delta_6v6": 4.0,
        "snr_ci_2v2": (1.5, 2.0), "snr_ci_6v6": (1.5, 2.0),
        "layer1_d_2v2": 0.8, "layer1_d_6v6": 0.8,
        "within_cos_2v2": 0.9, "within_cos_6v6": 0.9,
        "norm_cv_2v2": 0.2, "norm_cv_6v6": 0.2,
        "backbone_delta_2v2": 4.0, "backbone_delta_6v6": 4.0,
        "head_delta_2v2": 4.0, "head_delta_6v6": 4.0,
    }
    stats.update(overrides)
    return stats


def test_all_verdicts_are_reachable_and_named_consistently():
    assert set(VERDICTS) == {"TASK_SIGNAL_WEAKENING", "HEAD_LOCALIZED", "SUPPORTED",
                             "GRADIENT_INTERFERENCE", "REFUTED", "AMBIGUOUS"}


# ------------------------------------------------------- the five branches ---
def test_scenario_task_signal_weakening():
    """Raw advantage separation itself collapses from 2v2 to 6v6 -- the task
    stopped demanding differentiated play, not the optimizer."""
    stats = _base_stats(layer1_d_2v2=1.2, layer1_d_6v6=0.3,      # 4x fall
                        delta_2v2=4.0, delta_6v6=0.5,             # also falls a lot
                        snr_ci_2v2=(1.5, 2.0), snr_ci_6v6=(0.1, 0.4))
    out = classify_outcome(stats, thresholds=THRESHOLDS)
    assert out["verdict"] == "TASK_SIGNAL_WEAKENING"


def test_scenario_gradient_interference():
    """Raw signal holds (layer1 stable), fall is in the inconclusive middle
    (between the refute floor and the supported threshold), but cosine
    agreement collapses -- gradients individually present but conflicting."""
    stats = _base_stats(layer1_d_2v2=0.8, layer1_d_6v6=0.75,     # stable, no weakening
                        delta_2v2=4.0, delta_6v6=2.5,             # fall = 1.6x: middle zone
                        snr_ci_2v2=(1.5, 2.0), snr_ci_6v6=(1.2, 1.6),  # CIs DO overlap
                        within_cos_2v2=0.85, within_cos_6v6=0.55)  # cosine drop 0.30 >= 0.15
    out = classify_outcome(stats, thresholds=THRESHOLDS)
    assert out["verdict"] == "GRADIENT_INTERFERENCE"


def test_scenario_gradient_interference_via_norm_cv_alone():
    """The OR condition: noise-power growth alone (no cosine drop) also
    routes to interference, per the PI's 'either... or' framing."""
    stats = _base_stats(layer1_d_2v2=0.8, layer1_d_6v6=0.75,
                        delta_2v2=4.0, delta_6v6=2.5,
                        snr_ci_2v2=(1.5, 2.0), snr_ci_6v6=(1.2, 1.6),
                        within_cos_2v2=0.85, within_cos_6v6=0.84,  # cosine barely moves
                        norm_cv_2v2=0.2, norm_cv_6v6=0.4)          # CV grew 2x >= 1.5x
    out = classify_outcome(stats, thresholds=THRESHOLDS)
    assert out["verdict"] == "GRADIENT_INTERFERENCE"


def test_scenario_supported_broad_scaling_degradation():
    """Raw signal holds, whole-actor fall is large (>=2x) and CIs at 2v2/6v6
    do not overlap -- the pattern consistent with the Kuba-motivated
    scaling hypothesis."""
    stats = _base_stats(layer1_d_2v2=0.8, layer1_d_6v6=0.7,       # stable
                        delta_2v2=4.0, delta_6v6=1.5,              # fall = 2.67x
                        snr_ci_2v2=(1.6, 2.0), snr_ci_6v6=(1.05, 1.3),  # non-overlapping
                        backbone_delta_2v2=4.0, backbone_delta_6v6=1.6,  # backbone also falls
                        head_delta_2v2=4.0, head_delta_6v6=1.6)
    out = classify_outcome(stats, thresholds=THRESHOLDS)
    assert out["verdict"] == "SUPPORTED"


def test_scenario_head_localized():
    """Whole-actor picture alone would be ambiguous, but the macro head falls
    sharply while the shared backbone barely moves -- a specific,
    architecturally localized story checked before the generic SUPPORTED test."""
    stats = _base_stats(layer1_d_2v2=0.8, layer1_d_6v6=0.75,      # stable
                        delta_2v2=4.0, delta_6v6=3.5,               # whole-actor fall small
                        snr_ci_2v2=(1.5, 2.0), snr_ci_6v6=(1.4, 1.9),  # overlapping, unremarkable
                        backbone_delta_2v2=4.0, backbone_delta_6v6=3.6,   # backbone fall 1.11x
                        head_delta_2v2=4.0, head_delta_6v6=1.2)          # head fall 3.33x
    out = classify_outcome(stats, thresholds=THRESHOLDS)
    assert out["verdict"] == "HEAD_LOCALIZED"


def test_scenario_refuted_no_signal_even_at_2v2():
    """The strongest possible refutation: the estimator finds nothing even at
    2v2, where specialization is known to have worked."""
    stats = _base_stats(delta_2v2=0.05, delta_6v6=0.04,
                        snr_ci_2v2=(0.9, 1.1), snr_ci_6v6=(0.85, 1.05))
    out = classify_outcome(stats, thresholds=THRESHOLDS)
    assert out["verdict"] == "REFUTED"
    assert any("not distinguishable from its null" in r for r in out["reasons"])


def test_scenario_refuted_fall_too_small():
    stats = _base_stats(delta_2v2=4.0, delta_6v6=3.5,     # fall = 1.14x < 1.3 floor
                        snr_ci_2v2=(1.5, 2.0), snr_ci_6v6=(1.6, 2.1))
    out = classify_outcome(stats, thresholds=THRESHOLDS)
    assert out["verdict"] == "REFUTED"


def test_scenario_ambiguous():
    """Fall lands in the inconclusive middle, CIs DO NOT overlap (so REFUTED's
    'CIs overlap' clause cannot fire either), and NEITHER interference
    signature fires: no clean story, must not be forced into one.

    Note CI non-overlap is required for a true ambiguous case: REFUTED fires
    on overlapping CIs regardless of fall magnitude (see
    test_scenario_refuted_fall_too_small's CIs, which also overlap), so the
    ambiguous zone is specifically 'fall inconclusive AND CIs non-overlapping
    AND no interference signature'."""
    stats = _base_stats(layer1_d_2v2=0.8, layer1_d_6v6=0.75,
                        delta_2v2=4.0, delta_6v6=2.5,      # fall = 1.6x: middle zone
                        snr_ci_2v2=(1.6, 2.0), snr_ci_6v6=(1.2, 1.5),  # non-overlapping
                        within_cos_2v2=0.85, within_cos_6v6=0.83,       # cosine barely moves
                        norm_cv_2v2=0.2, norm_cv_6v6=0.22)              # CV barely moves
    out = classify_outcome(stats, thresholds=THRESHOLDS)
    assert out["verdict"] == "AMBIGUOUS"


# --------------------------------------------- precedence and fail-closed ----
def test_weakening_takes_precedence_over_a_would_be_supported_pattern():
    """If layer1 also collapsed, a scenario that would otherwise look like
    SUPPORTED must route to TASK_SIGNAL_WEAKENING instead -- weakening is
    checked first because it explains away the fall without implicating the
    optimizer."""
    stats = _base_stats(layer1_d_2v2=1.2, layer1_d_6v6=0.4,       # 3x fall -- weakening
                        delta_2v2=4.0, delta_6v6=1.5,               # would satisfy SUPPORTED
                        snr_ci_2v2=(1.6, 2.0), snr_ci_6v6=(1.05, 1.3))
    out = classify_outcome(stats, thresholds=THRESHOLDS)
    assert out["verdict"] == "TASK_SIGNAL_WEAKENING"


def test_head_localized_takes_precedence_over_broad_supported():
    """When BOTH the broad SUPPORTED pattern and the head-localized pattern
    are technically satisfiable, head-localized (the more specific, more
    informative diagnosis) is checked first."""
    stats = _base_stats(layer1_d_2v2=0.8, layer1_d_6v6=0.75,
                        delta_2v2=4.0, delta_6v6=1.5,                # whole-actor: 2.67x, non-overlap
                        snr_ci_2v2=(1.6, 2.0), snr_ci_6v6=(1.05, 1.3),
                        backbone_delta_2v2=4.0, backbone_delta_6v6=3.7,  # backbone barely falls
                        head_delta_2v2=4.0, head_delta_6v6=0.9)          # head falls hard
    out = classify_outcome(stats, thresholds=THRESHOLDS)
    assert out["verdict"] == "HEAD_LOCALIZED"


def test_missing_stat_raises():
    stats = _base_stats()
    del stats["head_delta_6v6"]
    with pytest.raises(KeyError, match="missing required stats"):
        classify_outcome(stats, thresholds=THRESHOLDS)


def test_missing_threshold_raises():
    stats = _base_stats()
    th = dict(THRESHOLDS)
    del th["kuba_fall_supported_min"]
    with pytest.raises(KeyError, match="THRESHOLDS block is missing"):
        classify_outcome(stats, thresholds=th)


def test_malformed_ci_raises():
    stats = _base_stats(snr_ci_2v2=(2.0, 1.5))     # lcb95 > ucb95
    with pytest.raises(ValueError, match="lcb95 > ucb95"):
        classify_outcome(stats, thresholds=THRESHOLDS)


def test_thresholds_loaded_from_frozen_spec_have_the_right_keys():
    """The code and the frozen spec must not drift apart: this loads the REAL
    spec file's THRESHOLDS block (added specifically to be machine-readable)
    rather than a hardcoded copy."""
    from experiments.grad_snr_decision import load_thresholds
    th = load_thresholds()
    for key in THRESHOLDS:
        assert key in th, f"frozen spec THRESHOLDS missing {key}"


# --------------------------------- the dim-invariance guarantee (locked in) --
def test_classification_is_invariant_to_dim_because_router_uses_raw_deltas():
    """THE structural guarantee the PI asked to lock in: classify_outcome
    takes only raw deltas (snr_spec^2-1), never a dimension-normalized index,
    so there is no `dim` parameter anywhere in its signature or REQUIRED_STATS
    for it to depend on. This test makes that explicit and permanent: the
    exact same raw-measurement stats dict, which could have come from a
    whole-actor comparison at ANY parameter dimension, produces the SAME
    verdict regardless -- because dim literally never enters the computation
    routed through the five-way table. Protects the router from silently
    drifting back onto the amplification-prone dimension-normalized path."""
    import inspect

    from experiments.grad_snr_decision import REQUIRED_STATS, classify_outcome

    assert "dim" not in REQUIRED_STATS
    assert "dim" not in inspect.signature(classify_outcome).parameters

    stats_a = _base_stats(delta_2v2=4.0, delta_6v6=1.5,
                          snr_ci_2v2=(1.6, 2.0), snr_ci_6v6=(1.05, 1.3),
                          backbone_delta_2v2=4.0, backbone_delta_6v6=1.6,
                          head_delta_2v2=4.0, head_delta_6v6=1.6)
    # A second, independently-constructed dict with IDENTICAL raw values --
    # standing in for "the same measurement, reported from a module of a
    # different dimension". Since the router never sees dim, this must
    # classify identically to stats_a.
    stats_b = copy.deepcopy(stats_a)
    out_a = classify_outcome(stats_a, thresholds=THRESHOLDS)
    out_b = classify_outcome(stats_b, thresholds=THRESHOLDS)
    assert out_a["verdict"] == out_b["verdict"] == "SUPPORTED"
    assert out_a["measurements"] == out_b["measurements"]
