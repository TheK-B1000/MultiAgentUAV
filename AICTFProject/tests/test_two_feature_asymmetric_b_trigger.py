"""Contracts for the two-feature asymmetric B-trigger calibration.

Pins the pieces that COMPOSITION_SELECTOR_TWO_FEATURE_ASYMMETRIC_REVIVAL_SPEC
relies on but that no earlier test covered:

* the r_red selection criterion is the *same* criterion that produced the
  sealed r_blue derivation (known-answer anchor against the sealed trace),
* the predeclared sign convention is a gate that aborts, not a hint that
  gets flipped to fit,
* the independent CSV re-derivation is a real check -- it agrees with the
  scoring path on honest data AND disagrees on corrupted data (negative
  control; a check that cannot fail is a false pass),
* a never-firing configuration is scored as zero B detection, which is what
  makes the degenerate-no-fire reporting rule bite.

Torch-gated: the module chain imports the CPU evaluation harness.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from experiments.calibrate_asymmetric_b_trigger import (  # noqa: E402
    run_state_machine,
    score,
    windowed,
)
from experiments.calibrate_two_feature_asymmetric_b_trigger import (  # noqa: E402
    RED_RADII,
    ROWS_CSV,
    gate_direction,
    min_misclass,
    rederive_from_csv,
)
import experiments.calibrate_two_feature_asymmetric_b_trigger as two_feature  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "strategic_demand" / "sppo"
DERIVATION = ART / "COMPOSITION_SELECTOR_FEATURE_DERIVATION_TRACE.json"
DERIVATION_ROWS = ART / "composition_selector_feature_derivation_raw_rows.csv"


def _sealed_r_blue_rows() -> tuple[np.ndarray, np.ndarray]:
    a, b = [], []
    with DERIVATION_ROWS.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            (a if row["pole"] == "A" else b).append(float(row["mean_pressure_r4"]))
    assert a and b, "sealed derivation rows are empty -- absence is an error state"
    return np.asarray(a), np.asarray(b)


def test_min_misclass_reproduces_sealed_r_blue_derivation():
    """Known-answer anchor: r_red* is chosen by this function, so it must
    reproduce the sealed r_blue numbers bit-for-bit on the sealed rows."""
    assert DERIVATION.is_file(), DERIVATION
    sealed = json.loads(DERIVATION.read_text(encoding="utf-8"))["radii"]["4.0"]
    a_vals, b_vals = _sealed_r_blue_rows()
    thr, err, a_higher = min_misclass(a_vals, b_vals)
    assert a_higher is True
    assert thr == pytest.approx(sealed["optimal_threshold"], abs=1e-12)
    assert err == pytest.approx(sealed["optimal_pooled_misclassification_rate"], abs=1e-12)


def test_all_four_red_radii_are_scored_by_the_same_criterion():
    """Every candidate radius the spec names must be scorable; a silently
    missing radius would narrow the predeclared selection set."""
    a_vals, b_vals = _sealed_r_blue_rows()
    assert RED_RADII == (4.0, 6.0, 8.0, 10.0)
    for _r in RED_RADII:
        thr, err, _ = min_misclass(a_vals, b_vals)
        assert 0.0 <= err <= 1.0 and np.isfinite(thr)


def test_direction_gate_passes_on_predeclared_orientation():
    d = {("A", 1): [2.0, 2.5, 3.0], ("B", 1): [-1.0, -0.5, -2.0]}
    stats = gate_direction(d)
    assert stats["A_mean"] > stats["B_mean"]


def test_direction_gate_aborts_when_convention_violated():
    """The convention is verified, never flipped to fit the data."""
    d = {("A", 1): [-2.0, -2.5, -3.0], ("B", 1): [1.0, 0.5, 2.0]}
    with pytest.raises(SystemExit, match="sign convention violated"):
        gate_direction(d)


def _write_csv(path: Path, series: dict[tuple[str, int], list[float]]) -> None:
    fields = ["split", "pole", "seed", "tick", "p_blue"] + [f"p_red_r{int(r)}" for r in RED_RADII]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for (pole, seed), vals in series.items():
            for t, v in enumerate(vals):
                row = {"split": "holdout", "pole": pole, "seed": seed, "tick": t, "p_blue": v}
                for r in RED_RADII:
                    row[f"p_red_r{int(r)}"] = 0.0
                w.writerow(row)


@pytest.fixture()
def synthetic_csv(tmp_path, monkeypatch):
    rng = np.random.default_rng(7)
    series: dict[tuple[str, int], list[float]] = {}
    for seed in range(4):
        series[("A", seed)] = list(rng.normal(1.5, 0.5, 60))
        series[("B", seed)] = list(rng.normal(-1.5, 0.5, 60))
    path = tmp_path / "ticks.csv"
    _write_csv(path, series)
    monkeypatch.setattr(two_feature, "ROWS_CSV", path)
    return path, series


def test_csv_rederivation_agrees_with_scoring_path(synthetic_csv):
    _path, series = synthetic_csv
    w, thr, m, d = 10, 0.0, 0.05, 4
    direct = score({k: run_state_machine(windowed(v, w), thr, m, d) for k, v in series.items()})
    check = rederive_from_csv("holdout", "p_red_r4", w, thr, m, d)
    assert set(direct) == set(check)
    for key in direct:
        assert direct[key] == pytest.approx(check[key], abs=1e-9)


def test_csv_rederivation_detects_corruption(synthetic_csv):
    """Negative control: the re-derivation must be able to FAIL. Without this,
    '6/6 exact' could mean the check compared nothing."""
    path, series = synthetic_csv
    w, thr, m, d = 10, 0.0, 0.05, 4
    direct = score({k: run_state_machine(windowed(v, w), thr, m, d) for k, v in series.items()})

    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    for row in rows:
        if row["pole"] == "B":
            row["p_blue"] = str(float(row["p_blue"]) + 5.0)
    with path.open("w", newline="", encoding="utf-8") as fh:
        wtr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wtr.writeheader()
        wtr.writerows(rows)

    corrupted = rederive_from_csv("holdout", "p_red_r4", w, thr, m, d)
    assert corrupted["B_TP_tick_rate"] != pytest.approx(direct["B_TP_tick_rate"], abs=1e-9)


def test_never_firing_config_scores_zero_b_detection():
    """The degenerate case the single-feature run hit: a threshold so low the
    machine never leaves the 2A_2D default. It satisfies the A-side budget
    vacuously and must read as zero B detection, not as a safe operating point."""
    series = {("A", 0): [1.0] * 40, ("B", 0): [-1.0] * 40}
    preds = {k: run_state_machine(windowed(v, 20), -99.0, 0.0, 4) for k, v in series.items()}
    s = score(preds)
    assert s["A_FP_tick_rate"] == 0.0
    assert s["B_TP_tick_rate"] == 0.0
    assert s["B_fraction_episodes_dominant_4A0D"] == 0.0


def test_rows_csv_target_is_inside_the_artifact_directory():
    assert ROWS_CSV.parent == ART


# --------------------------------------------------------------------------
# V2: uncertainty-aware eligibility
# (COMPOSITION_SELECTOR_TWO_FEATURE_UNCERTAINTY_AWARE_V2_SPEC.json)
# --------------------------------------------------------------------------

from experiments.calibrate_two_feature_uncertainty_aware_v2 import (  # noqa: E402
    BOOT_SEED,
    N_BOOT,
    R_RED_FROZEN,
    UCB_CAP,
    UCB_PERCENTILE,
    make_boot_index,
    ucb95_batch,
)
import experiments.calibrate_two_feature_uncertainty_aware_v2 as v2  # noqa: E402


def test_ucb_batch_equals_direct_episode_resampling():
    """The counts-matrix batching is an optimization, not a different
    estimator. It must equal a plain per-config gather-and-resample."""
    rng = np.random.default_rng(1)
    rates = rng.random((9, 16)) * 0.3
    idx = make_boot_index(16)
    fast = ucb95_batch(rates, idx)
    slow = np.array([np.percentile(r[idx].mean(axis=1), UCB_PERCENTILE) for r in rates])
    assert np.allclose(fast, slow, atol=1e-12)


def test_ucb_is_never_below_the_point_estimate():
    """A one-sided upper bound that dipped under the point estimate would make
    the eligibility rule looser than V1's, not stricter."""
    rng = np.random.default_rng(3)
    rates = rng.random((25, 16)) * 0.4
    ucb = ucb95_batch(rates, make_boot_index(16))
    assert np.all(ucb >= rates.mean(axis=1) - 1e-12)


def test_ucb_rule_is_strictly_stricter_than_the_v1_point_rule():
    """Everything eligible under UCB must have been eligible under the point
    rule. If this ever inverts, V2 would admit configs V1 rejected."""
    rng = np.random.default_rng(5)
    rates = rng.random((200, 16)) * 0.25
    ucb = ucb95_batch(rates, make_boot_index(16))
    point = rates.mean(axis=1)
    assert np.all(point[ucb <= UCB_CAP] <= UCB_CAP + 1e-12)


def test_v1_boundary_hugging_config_would_be_rejected_by_v2():
    """Known-answer anchor on the actual defect. V1's held-out Pole-A
    per-episode rates average 0.1120; a calibration config with that shape sits
    on the boundary on a point estimate and must be UCB-ineligible."""
    v1_holdout_a = np.array([0.0, 0.017, 0.017, 0.033, 0.058, 0.062, 0.067, 0.096,
                             0.104, 0.117, 0.146, 0.158, 0.167, 0.192, 0.2, 0.358])
    ucb = ucb95_batch(v1_holdout_a.reshape(1, -1), make_boot_index(16))[0]
    assert v1_holdout_a.mean() > UCB_CAP      # the point estimate that failed
    assert ucb > UCB_CAP                      # and the bound rejects it too


def test_bootstrap_is_reproducible_from_the_frozen_seed():
    assert np.array_equal(make_boot_index(16), make_boot_index(16))
    assert make_boot_index(16).shape == (N_BOOT, 16)
    assert BOOT_SEED == 20_260_919 and UCB_PERCENTILE == 95.0 and UCB_CAP == 0.10


def test_v2_never_writes_over_the_sealed_v1_rows():
    """persist_rows/rederive_from_csv address the V1 module global; V2 must
    point that global at its own file before writing."""
    assert v2.ROWS_CSV != v2.V1_ROWS
    assert v2.ROWS_CSV.name == "composition_selector_two_feature_v2_raw_ticks.csv"


def test_v2_inherits_the_frozen_statistic_and_grid():
    from experiments.calibrate_asymmetric_b_trigger import DWELL_GRID, HYST_GRID, WINDOW_GRID
    assert R_RED_FROZEN == 4.0
    assert v2.WINDOW_GRID is WINDOW_GRID
    assert v2.HYST_GRID is HYST_GRID
    assert v2.DWELL_GRID is DWELL_GRID
