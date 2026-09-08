"""Tests for stage-1 calibration evidence.

The script must measure honestly and decide nothing. Torch-free.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiments import calibrate_abstention_evidence as CAL
from rl.launch_gate import LaunchGateError


def _shard(path: Path, cells, deltas):
    """A minimal shard carrying only what stage 1 reads."""
    assert len(cells) == len(deltas)
    b_blue = np.array([max(d, 0) for d in deltas], dtype=np.int16)
    b_red = np.zeros(len(deltas), dtype=np.int16)
    a_blue = np.array([max(-d, 0) for d in deltas], dtype=np.int16)
    a_red = np.zeros(len(deltas), dtype=np.int16)
    np.savez_compressed(
        path, branch_cell=np.array(cells, dtype="U24"),
        branch_pi_B_blue=b_blue, branch_pi_B_red=b_red,
        branch_pi_A_blue=a_blue, branch_pi_A_red=a_red)


def _make_calib(tmp_path: Path, per_seed_cells=None, per_seed_deltas=None) -> Path:
    data = tmp_path / "data"
    (data / "seed_shards").mkdir(parents=True)
    for seed in CAL.CALIB_SEEDS:
        cells = per_seed_cells or CAL.CELLS[:4]
        deltas = per_seed_deltas or [1, -1, 0, 0]
        _shard(data / "seed_shards" / f"seed_{seed}.npz", cells, deltas)
    return data


# ------------------------------------------------------------------- refusal

def test_preflight_now_passes_because_the_artifacts_were_earned():
    """This test previously asserted REFUSAL, and it fired on 2026-08-29.

    That was the tripwire doing its job: it forced someone to look at why the guard
    had gone green. The answer was legitimate rather than drift --

        COLLECTION_COMPLETE  160/160, exact frozen block, written 07:28:04Z
        SUPPORT_VALIDITY     VALID, 16/16 cells, scarcest cell 59 seeds vs floor 32
        CALIB                32/32 seeds present
        FINAL                never touched

    so the assertion was flipped rather than deleted. It now pins the earned state:
    if any of those artifacts vanish or degrade, this fails and says so.
    """
    checks = CAL.preflight()
    assert all(c.passed for c in checks), [c.detail for c in checks if not c.passed]
    by_name = {c.name: c for c in checks}
    assert "16/16" in by_name["support_floor"].detail
    assert by_name["final_untouched"].passed


def test_refusal_names_each_missing_prerequisite(tmp_path):
    with pytest.raises(LaunchGateError) as exc:
        CAL.preflight(tmp_path, tmp_path / "audit.json")
    msg = str(exc.value)
    assert "collection_complete" in msg
    assert "support_floor" in msg
    assert "calib_split_complete" in msg


def test_refuses_when_calib_split_incomplete(tmp_path):
    data = _make_calib(tmp_path)
    (data / "seed_shards" / f"seed_{CAL.CALIB_LO}.npz").unlink()
    check = CAL._check_calib_present(data)
    assert not check.passed and "incomplete" in check.detail


# ------------------------------------------------------- label determinism

def test_labels_follow_the_deterministic_sign_rule(tmp_path):
    data = _make_calib(tmp_path, ["A_r0_late"] * 4, [3, -2, 0, 7])
    ev = CAL.build_evidence(CAL.collect_calib(data), minimum=1)
    lab = ev["cells"]["A_r0_late"]["labels"]
    n = len(CAL.CALIB_SEEDS)
    assert lab["b_preferred"] == 2 * n
    assert lab["a_preferred"] == 1 * n
    assert lab["not_established"] == 1 * n


def test_resolvable_mass_is_the_non_tied_fraction(tmp_path):
    data = _make_calib(tmp_path, ["A_r0_late"] * 4, [1, -1, 0, 0])
    ev = CAL.build_evidence(CAL.collect_calib(data), minimum=1)
    cell = ev["cells"]["A_r0_late"]
    assert cell["resolvable_mass"]["mean"] == pytest.approx(0.5, abs=1e-9)
    assert cell["tie_rate_point"] == pytest.approx(0.5)


def test_a_fully_tied_cell_reports_zero_resolvable_mass(tmp_path):
    """The stolen-flag reality: a cell can be almost entirely unresolvable."""
    data = _make_calib(tmp_path, ["B_r2_late"] * 4, [0, 0, 0, 0])
    ev = CAL.build_evidence(CAL.collect_calib(data), minimum=1)
    assert ev["cells"]["B_r2_late"]["resolvable_mass"]["mean"] == pytest.approx(0.0)
    assert ev["cells"]["B_r2_late"]["labels"]["not_established"] == 4 * len(CAL.CALIB_SEEDS)


# ------------------------------------------------------- seed-level clustering

def test_uncertainty_uses_seed_as_the_resampling_unit(tmp_path):
    """Per-state resampling would understate uncertainty; the CI must be wide."""
    data = _make_calib(tmp_path, ["A_r0_late"] * 4, [1, -1, 0, 0])
    ev = CAL.build_evidence(CAL.collect_calib(data), minimum=1)
    ci = ev["cells"]["A_r0_late"]["resolvable_mass"]
    assert ev["bootstrap"]["unit"] == "seed (clustered)"
    # every seed is identical here, so a seed-clustered CI must collapse to a point
    assert ci["lcb95"] == pytest.approx(ci["ucb95"])


def test_seed_counts_are_reported_per_cell(tmp_path):
    ev = CAL.build_evidence(CAL.collect_calib(_make_calib(tmp_path)), minimum=1)
    assert ev["cells"][CAL.CELLS[0]]["n_seeds"] == len(CAL.CALIB_SEEDS)


# --------------------------------------------------------- adequacy handling

def test_unfrozen_minimum_blocks_calibration_without_inventing_one(tmp_path):
    """The bar must not be chosen after seeing the counts."""
    ev = CAL.build_evidence(CAL.collect_calib(_make_calib(tmp_path)), minimum=None)
    assert ev["VERDICT"] == "MINIMUM_SUPPORT_NOT_FROZEN"
    assert ev["minimum_calib_seeds_per_cell"] is None
    assert "will not invent one" in ev["consequence"]


def test_cells_below_a_frozen_minimum_stop_calibration(tmp_path):
    data = _make_calib(tmp_path, CAL.CELLS[:4], [1, -1, 0, 0])
    ev = CAL.build_evidence(CAL.collect_calib(data), minimum=999)
    assert ev["VERDICT"] == "CALIBRATION_CANNOT_PROCEED"
    assert "do not grow the data" in ev["consequence"]


def test_empty_cells_count_as_below_minimum(tmp_path):
    """12 of 16 cells have no states in this fixture; none may be waved through."""
    ev = CAL.build_evidence(CAL.collect_calib(_make_calib(tmp_path)), minimum=1)
    assert ev["VERDICT"] == "CALIBRATION_CANNOT_PROCEED"
    assert len(ev["cells_with_no_branch_states"]) == 12


def test_all_cells_supported_yields_evidence_complete(tmp_path):
    data = _make_calib(tmp_path, CAL.CELLS, [1, -1, 0, 0] * 4)
    ev = CAL.build_evidence(CAL.collect_calib(data), minimum=1)
    assert ev["VERDICT"] == "EVIDENCE_COMPLETE"
    assert ev["cells_below_minimum"] == []


# ------------------------------------------------------ scope and provenance

def test_evidence_contains_no_threshold_of_any_kind(tmp_path):
    """Stage 1 decides nothing. Not even a suggestion."""
    ev = CAL.build_evidence(CAL.collect_calib(_make_calib(tmp_path)), minimum=1)
    blob = json.dumps(ev)
    for banned in ('"tau"', '"rho"', '"o_max"'):
        assert banned not in blob


def test_source_file_contains_no_default_threshold_values():
    """Guards against a helpful default appearing here later."""
    src = Path(CAL.__file__).read_text(encoding="utf-8")
    for banned in ("tau =", "tau=0", "rho =", "rho=0", "o_max =", "o_max=0"):
        assert banned not in src, banned


def test_evidence_declares_which_splits_were_untouched(tmp_path):
    ev = CAL.build_evidence(CAL.collect_calib(_make_calib(tmp_path)), minimum=1)
    assert ev["split"]["FIT_used"] is False
    assert ev["split"]["EVAL_used"] is False
    assert ev["split"]["FINAL_touched"] is False
    assert ev["split"]["seeds"] == [CAL.CALIB_LO, CAL.CALIB_HI, 32]


def test_only_calib_seeds_are_read(tmp_path):
    """A FIT or EVAL shard sitting alongside must not be picked up."""
    data = _make_calib(tmp_path)
    _shard(data / "seed_shards" / "seed_10700001.npz", ["A_r0_late"], [5])   # FIT
    _shard(data / "seed_shards" / "seed_10700150.npz", ["A_r0_late"], [5])   # EVAL
    per_cell = CAL.collect_calib(data)
    seeds_seen = {s for by_seed in per_cell.values() for s in by_seed}
    assert seeds_seen == set(CAL.CALIB_SEEDS)


def test_frozen_floor_is_five_and_is_read_from_the_protocol():
    """The floor lives in the protocol, not in this script's source."""
    assert CAL._minimum_support_declared() == 5


def test_a_cell_with_seeds_but_zero_resolvable_is_NOT_insufficient(tmp_path):
    """The instrument must not reject the finding it exists to detect."""
    data = _make_calib(tmp_path, CAL.CELLS, [0] * 16)          # every cell fully tied
    ev = CAL.build_evidence(CAL.collect_calib(data), minimum=5)
    assert ev["VERDICT"] == "EVIDENCE_COMPLETE"
    assert ev["cells_below_minimum"] == []
    cell = ev["cells"]["B_r2_late"]
    assert cell["resolvable_mass"]["mean"] == pytest.approx(0.0)
    assert cell["n_seeds"] >= 5


def test_all_tied_cell_is_flagged_degenerate_with_a_rule_of_three_bound(tmp_path):
    """[0,0] from a bootstrap asserts precision the protocol forbids."""
    data = _make_calib(tmp_path, ["B_r2_late"] * 2, [0, 0])
    ev = CAL.build_evidence(CAL.collect_calib(data), minimum=5)
    cell = ev["cells"]["B_r2_late"]
    assert cell["degenerate_bootstrap"] is True
    assert cell["rule_of_three_ucb95"] == pytest.approx(3.0 / len(CAL.CALIB_SEEDS))
    assert "NOT insufficient support" in cell["rule_of_three_note"]


def test_mixed_cell_is_not_flagged_degenerate(tmp_path):
    """Only a collapsed interval earns the flag."""
    data = _make_calib(tmp_path, ["A_r0_late"] * 4, [1, -1, 0, 0])
    per_cell = CAL.collect_calib(data)
    # perturb one seed so the seeds are not all identical
    per_cell["A_r0_late"][CAL.CALIB_LO] = {"b_preferred": 4, "a_preferred": 0,
                                           "not_established": 0}
    ev = CAL.build_evidence(per_cell, minimum=5)
    cell = ev["cells"]["A_r0_late"]
    assert cell["degenerate_bootstrap"] is False
    assert cell["resolvable_mass"]["lcb95"] < cell["resolvable_mass"]["ucb95"]
    assert "rule_of_three_ucb95" not in cell


def test_resolvable_cell_never_gets_a_rule_of_three_bound(tmp_path):
    """The bound is only meaningful when the count is genuinely zero."""
    data = _make_calib(tmp_path, ["A_r0_late"] * 2, [1, 1])
    cell = CAL.build_evidence(CAL.collect_calib(data), minimum=5)["cells"]["A_r0_late"]
    assert cell["degenerate_bootstrap"] is True        # identical seeds
    assert "rule_of_three_ucb95" not in cell           # but nonzero resolvable


def test_not_established_is_never_described_as_equivalence(tmp_path):
    ev = CAL.build_evidence(CAL.collect_calib(_make_calib(tmp_path)), minimum=1)
    text = ev["label_semantics"]["not_established"]
    assert "never a claim" in text and "equivalent" in text
