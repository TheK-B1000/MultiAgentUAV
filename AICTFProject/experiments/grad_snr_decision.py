r"""Executable decision routing for GRAD_SNR_SCALING_DIAGNOSTIC_SPEC.json.

This operationalises the frozen spec's PRECOMMITTED_INTERPRETATION as CODE, not
prose applied by eye after the numbers exist. Per the PI: "before any real
2v2/4v4/6v6 data exists, we already know what kinds of measurements map to what
conclusion." ``classify_outcome`` is tested against six synthetic scenarios --
five named branches plus an explicit ambiguous case -- entirely before a single
real gradient is collected.

Thresholds are LOADED from the frozen spec's THRESHOLDS block rather than
hardcoded a second time here, so the code and the spec cannot silently drift
apart. If a spec amendment changes a threshold, this module picks it up without
being edited.

WHY THIS ROUTING NEVER TOUCHES spec_index
    Every comparison below is a FALL RATIO -- one module's raw
    (snr_spec^2 - 1) at 2v2 divided by the SAME module's raw (snr_spec^2 - 1)
    at 6v6. Under the frozen controls (identical architecture => identical
    dim; identical M => identical k at both scales), the dimension/sample-count
    factor that ``spec_index`` introduces would be IDENTICAL in numerator and
    denominator and cancels EXACTLY -- not approximately, to floating-point
    precision (verified in tests/test_grad_snr_metrics.py::
    test_within_module_fall_ratio_is_identical_via_raw_delta_or_spec_index).
    So these fall ratios need no dimension normalization at all, and none of
    the huge-replicate-count problem documented for ``spec_index`` (module
    docstring, aggregate_spec_index) touches this routing.

    Dimension normalization is needed for exactly one thing this module does
    NOT do: comparing RAW MAGNITUDE across two DIFFERENT-dimension modules
    (e.g. "does the head carry more absolute specialization signal than the
    backbone, at one scale"). That question is real and sometimes worth
    asking, but it is a separate, explicitly exploratory computation
    (``aggregate_spec_index`` in grad_snr_metrics.py) gated by its own
    null-control and replicate-count requirements, and it is NOT wired into
    this decision routing.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEC = (ROOT / "artifacts" / "strategic_demand" / "sppo"
               / "GRAD_SNR_SCALING_DIAGNOSTIC_SPEC.json")

VERDICTS = ("TASK_SIGNAL_WEAKENING", "HEAD_LOCALIZED", "SUPPORTED",
           "GRADIENT_INTERFERENCE", "REFUTED", "AMBIGUOUS")

REQUIRED_STATS = (
    # raw (snr_spec^2 - 1), whole actor -- NOT dimension-normalized; see module
    # docstring for why the fall ratio below needs no such normalization.
    "delta_2v2", "delta_6v6",
    "snr_ci_2v2", "snr_ci_6v6",                # raw snr_spec (lcb95, ucb95), whole actor
    "layer1_d_2v2", "layer1_d_6v6",            # Cohen's d, raw advantage separation
    "within_cos_2v2", "within_cos_6v6",        # within-pole gradient cosine, whole actor
    "norm_cv_2v2", "norm_cv_6v6",              # gradient-norm coefficient of variation
    "backbone_delta_2v2", "backbone_delta_6v6",  # raw (snr_spec^2-1), shared_backbone
    "head_delta_2v2", "head_delta_6v6",          # raw (snr_spec^2-1), policy_macro_head
)


def load_thresholds(spec_path: Path | None = None) -> dict:
    p = Path(spec_path) if spec_path else DEFAULT_SPEC
    doc = json.loads(p.read_text(encoding="utf-8"))
    # THRESHOLDS lives under PRECOMMITTED_INTERPRETATION in the frozen spec
    # (that section is where the whole decision table is documented); fall
    # back to a top-level key so a future spec may promote it without
    # breaking this loader.
    th = doc.get("THRESHOLDS")
    if th is None:
        th = doc.get("PRECOMMITTED_INTERPRETATION", {}).get("THRESHOLDS")
    if th is None:
        raise KeyError(f"{p} has no THRESHOLDS block (checked top level and "
                       f"PRECOMMITTED_INTERPRETATION) -- this decision code has "
                       f"nothing to load. Add THRESHOLDS to the spec rather than "
                       f"hardcoding a value here.")
    return th


def _measurements(loc: dict) -> dict:
    keys = ("layer1_fall", "kuba_fall", "ci_non_overlap", "snr_2v2_significant",
           "backbone_fall", "head_fall", "cos_drop", "norm_cv_growth")
    return {k: (round(loc[k], 4) if isinstance(loc[k], float) else loc[k]) for k in keys}


def classify_outcome(stats: dict, thresholds: dict | None = None,
                     spec_path: Path | None = None) -> dict:
    """Route a computed cross-scale summary into exactly one of ``VERDICTS``.

    ``stats`` must carry every key in ``REQUIRED_STATS`` (see the frozen spec
    for what each one means and how it is computed from the real rollouts).
    Fails closed on a missing key rather than defaulting silently.

    PRECEDENCE (fixed, not data-dependent):
      1. TASK_SIGNAL_WEAKENING -- checked first. If the RAW pre-normalization
         signal itself weakened this much, whatever happened to the gradient
         SNR is a downstream consequence of the task, not evidence about the
         optimizer, and every other branch below would be misattributing it.
      2. HEAD_LOCALIZED -- a more specific, more informative pattern than a
         generic "SNR fell", so it is checked before the broad SUPPORTED test.
      3. SUPPORTED -- broad, reproducible fall in the whole-actor raw
         (snr_spec^2-1), with the raw signal already ruled out as the cause
         (step 1).
      4. GRADIENT_INTERFERENCE -- the fall is too small to satisfy SUPPORTED
         and not below the REFUTED floor either, but carries interference's
         specific signature (agreement dropping and/or norm variability
         growing) rather than a uniform shrink.
      5. REFUTED -- the estimator finds nothing even where specialization
         demonstrably worked (2v2), or the fall is below the refutation floor,
         or the CIs plainly overlap.
      6. AMBIGUOUS -- none of the above fired cleanly. Licenses no
         intervention; report and stop.
    """
    missing = [k for k in REQUIRED_STATS if k not in stats]
    if missing:
        raise KeyError(f"missing required stats: {missing}")
    th = thresholds if thresholds is not None else load_thresholds(spec_path)
    need = ("kuba_fall_supported_min", "kuba_fall_refute_max",
           "layer1_fall_weakening_min", "cos_drop_interference_min",
           "norm_cv_growth_interference_min", "head_fall_min", "backbone_fall_max")
    missing_th = [k for k in need if k not in th]
    if missing_th:
        raise KeyError(f"THRESHOLDS block is missing: {missing_th}")

    for k in ("snr_ci_2v2", "snr_ci_6v6"):
        v = stats[k]
        if not (isinstance(v, (list, tuple)) and len(v) == 2):
            raise ValueError(f"{k} must be a (lcb95, ucb95) pair, got {v!r}")
        if v[0] > v[1]:
            raise ValueError(f"{k} has lcb95 > ucb95: {v!r}")

    eps = 1e-9
    d2, d6 = abs(stats["layer1_d_2v2"]), abs(stats["layer1_d_6v6"])
    layer1_fall = d2 / max(d6, eps)

    # Fall ratios of raw (snr_spec^2-1). No dimension normalization: same
    # module, same frozen k at both scales, so any such factor would cancel
    # exactly -- see module docstring.
    w2, w6 = stats["delta_2v2"], stats["delta_6v6"]
    kuba_fall = (w2 / max(w6, eps)) if w2 > 0 else float("-inf")

    ci2, ci6 = stats["snr_ci_2v2"], stats["snr_ci_6v6"]
    ci_non_overlap = ci2[0] > ci6[1]
    snr_2v2_significant = ci2[0] > 1.0

    bb2, bb6 = stats["backbone_delta_2v2"], stats["backbone_delta_6v6"]
    hd2, hd6 = stats["head_delta_2v2"], stats["head_delta_6v6"]
    backbone_fall = (bb2 / max(bb6, eps)) if bb2 > 0 else float("-inf")
    head_fall = (hd2 / max(hd6, eps)) if hd2 > 0 else float("-inf")

    cos_drop = stats["within_cos_2v2"] - stats["within_cos_6v6"]
    norm_cv_growth = stats["norm_cv_6v6"] / max(stats["norm_cv_2v2"], eps)

    reasons: list[str] = []

    if layer1_fall >= th["layer1_fall_weakening_min"]:
        reasons.append(f"raw Layer-1 |Cohen's d| fell {layer1_fall:.2f}x "
                       f"(>= {th['layer1_fall_weakening_min']}x threshold)")
        return {"verdict": "TASK_SIGNAL_WEAKENING", "reasons": reasons,
                "measurements": _measurements(locals())}

    if head_fall >= th["head_fall_min"] and backbone_fall < th["backbone_fall_max"]:
        reasons.append(f"macro-head index fell {head_fall:.2f}x "
                       f"(>= {th['head_fall_min']}x) while shared-backbone fell "
                       f"only {backbone_fall:.2f}x (< {th['backbone_fall_max']}x)")
        return {"verdict": "HEAD_LOCALIZED", "reasons": reasons,
                "measurements": _measurements(locals())}

    if ci_non_overlap and kuba_fall >= th["kuba_fall_supported_min"]:
        reasons.append(f"snr_spec 95% CIs at 2v2/6v6 do not overlap and the whole-"
                       f"actor raw (snr_spec^2-1) fell {kuba_fall:.2f}x "
                       f"(>= {th['kuba_fall_supported_min']}x)")
        return {"verdict": "SUPPORTED", "reasons": reasons,
                "measurements": _measurements(locals())}

    if th["kuba_fall_refute_max"] <= kuba_fall < th["kuba_fall_supported_min"] and (
            cos_drop >= th["cos_drop_interference_min"]
            or norm_cv_growth >= th["norm_cv_growth_interference_min"]):
        reasons.append(f"kuba fall {kuba_fall:.2f}x is inconclusive alone, but "
                       f"within-pole cosine agreement dropped {cos_drop:.3f} and/or "
                       f"gradient-norm CV grew {norm_cv_growth:.2f}x")
        return {"verdict": "GRADIENT_INTERFERENCE", "reasons": reasons,
                "measurements": _measurements(locals())}

    if (not snr_2v2_significant) or kuba_fall < th["kuba_fall_refute_max"] \
            or not ci_non_overlap:
        if not snr_2v2_significant:
            reasons.append("snr_spec at 2v2 is not distinguishable from its null "
                           "value of 1.0 -- the estimator finds no signal even "
                           "where specialization demonstrably succeeded")
        if kuba_fall < th["kuba_fall_refute_max"]:
            reasons.append(f"kuba fall {kuba_fall:.2f}x is below the refutation "
                           f"floor {th['kuba_fall_refute_max']}x")
        if not ci_non_overlap:
            reasons.append("snr_spec 95% CIs at 2v2 and 6v6 overlap")
        return {"verdict": "REFUTED", "reasons": reasons,
                "measurements": _measurements(locals())}

    reasons.append("no precommitted pattern fired unambiguously")
    return {"verdict": "AMBIGUOUS", "reasons": reasons,
            "measurements": _measurements(locals())}
