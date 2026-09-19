"""The 6v6 fixed-composition confirmation chooses its arm by RULE, not by hand.

derive_selection() must reproduce the composition frozen in the spec from the
sealed sweep rows, and must ABORT rather than silently pick when the rule cannot
decide (empty intersection, tie at the argmax, tie at the maximin).
"""
from __future__ import annotations

import csv

import pytest

pytest.importorskip("torch")

from experiments.run_fixed_attack_heavy_6v6_confirmation import (  # noqa: E402
    A_ARGMAX,
    ARMS,
    B_ARGMAX,
    BASELINE,
    SELECTED,
    _maximin,
    derive_selection,
)

COMPS = ["6A_0D", "5A_1D", "4A_2D", "3A_3D", "2A_4D", "1A_5D", "0A_6D"]


def test_real_sweep_reproduces_the_frozen_selection():
    sel = derive_selection()
    assert sel["selected"] == SELECTED == "5A_1D"
    assert sel["A_argmax"] == A_ARGMAX == "4A_2D"
    assert sel["B_argmax"] == B_ARGMAX == "6A_0D"
    assert sel["intersection"] == ["5A_1D", "6A_0D"]
    assert sel["n_seeds"] == 64


def test_the_two_candidates_tie_on_mean_and_maximin_breaks_it():
    """The point of the frozen rule: 5A_1D and 6A_0D tie exactly on mean win rate
    across poles, so 'best on average' cannot decide. Worst-pole win rate does."""
    sel = derive_selection()
    m = sel["mean_across_poles_win_rate"]
    assert abs(m["5A_1D"] - m["6A_0D"]) < 1e-12
    w = sel["worst_pole_win_rate"]
    assert w["5A_1D"] > w["6A_0D"]


def test_exactly_two_arms_run_and_the_argmaxes_are_derivation_record_only():
    assert ARMS == (BASELINE, SELECTED) == ("3A_3D", "5A_1D")
    assert A_ARGMAX not in ARMS and B_ARGMAX not in ARMS


def test_maximin_picks_the_higher_worst_pole_win_rate():
    wr = {"A": {"X": 0.80, "Y": 0.70}, "B": {"X": 0.80, "Y": 0.90}}
    assert _maximin(wr, {"X", "Y"}) == "X"


def test_maximin_refuses_to_break_an_exact_tie():
    wr = {"A": {"X": 0.75, "Y": 0.75}, "B": {"X": 0.90, "Y": 0.80}}
    with pytest.raises(SystemExit, match="tie at the maximin"):
        _maximin(wr, {"X", "Y"})


def test_maximin_refuses_an_empty_candidate_set():
    with pytest.raises(SystemExit):
        _maximin({"A": {}, "B": {}}, set())


def _write_sweep(path, win_fn, n=64):
    fields = ["seed", "pole", "composition", "blue_win"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for pole in "AB":
            for c in COMPS:
                for s in range(n):
                    w.writerow({"seed": s, "pole": pole, "composition": c, "blue_win": win_fn(pole, c, s)})


def test_empty_intersection_aborts_instead_of_choosing(tmp_path):
    """A wins only with 4A_2D, B wins only with 6A_0D, both perfectly separated:
    no single composition is compatible with both, so there is no candidate."""
    p = tmp_path / "sweep.csv"
    _write_sweep(p, lambda pole, c, s: int((pole == "A" and c == "4A_2D") or (pole == "B" and c == "6A_0D")))
    with pytest.raises(SystemExit, match="do not intersect"):
        derive_selection(p)


def test_tie_at_the_argmax_aborts(tmp_path):
    p = tmp_path / "sweep.csv"
    _write_sweep(p, lambda pole, c, s: int(c in ("6A_0D", "5A_1D")))
    with pytest.raises(SystemExit, match="tie at the pole-"):
        derive_selection(p)


def test_missing_rows_are_an_error_not_a_default(tmp_path):
    p = tmp_path / "empty.csv"
    p.write_text("seed,pole,composition,blue_win\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="absence is an error state"):
        derive_selection(p)


# --------------------------------------------------------------------------
# The frozen terminal interpretation, as a pure function.
# --------------------------------------------------------------------------
from experiments.run_fixed_attack_heavy_6v6_confirmation import (  # noqa: E402
    SEED_N,
    arm_identity_violations,
    decide,
)


def _ci(mean, lcb, ucb):
    return {"mean": mean, "lcb95": lcb, "ucb95": ucb, "n": SEED_N}


B_PASS = _ci(0.70, 0.60, 0.80)
B_FAIL = _ci(0.05, -0.05, 0.15)


def test_n_is_192():
    assert SEED_N == 192


def test_both_gates_pass_is_confirmed_and_promotes():
    v = decide(B_PASS, _ci(-0.05, -0.12, 0.05))
    assert v["label"] == "FIXED_ATTACK_HEAVY_6V6_CONFIRMED"
    assert v["promotion"] is True and v["routing_reopen_licensed"] is False


def test_b_failure_dominates_whatever_a_shows():
    for a in (_ci(-0.10, -0.20, 0.00),      # A great
              _ci(0.30, 0.20, 0.40)):       # A clearly harmed
        v = decide(B_FAIL, a)
        assert v["label"] == "FIXED_ATTACK_HEAVY_6V6_B_NOT_REPLICATED"
        assert v["promotion"] is False and v["routing_reopen_licensed"] is False


def test_a_failure_with_interval_spanning_zero_is_inconclusive_and_does_not_reopen_routing():
    """UCB95 > 0.10 but LCB95 <= 0: 'failed to prove safety', NOT 'proved harm'."""
    v = decide(B_PASS, _ci(0.04, -0.08, 0.14))
    assert v["label"] == "FIXED_ATTACK_HEAVY_6V6_A_SAFETY_NOT_DEMONSTRATED"
    assert v["promotion"] is False
    assert v["routing_reopen_licensed"] is False
    assert "NEW freeze" in v["note"] and "FRESH seeds" in v["note"] and "No automatic top-up" in v["note"]


def test_demonstrated_a_harm_is_the_only_branch_that_reopens_routing():
    v = decide(B_PASS, _ci(0.18, 0.06, 0.30))
    assert v["label"] == "FIXED_ATTACK_HEAVY_6V6_A_HARM_DEMONSTRATED"
    assert v["routing_reopen_licensed"] is True and v["promotion"] is False


def test_the_lcb_boundary_is_strict():
    """LCB95 exactly zero is NOT demonstrated harm."""
    v = decide(B_PASS, _ci(0.12, 0.0, 0.24))
    assert v["label"] == "FIXED_ATTACK_HEAVY_6V6_A_SAFETY_NOT_DEMONSTRATED"


def test_the_ucb_boundary_is_inclusive_as_frozen():
    """UCB95 exactly equal to tau passes, as the gate is written UCB95 <= 0.10."""
    assert decide(B_PASS, _ci(0.02, -0.06, 0.10))["label"] == "FIXED_ATTACK_HEAVY_6V6_CONFIRMED"
    assert decide(B_PASS, _ci(0.02, -0.06, 0.1000001))["label"] != "FIXED_ATTACK_HEAVY_6V6_CONFIRMED"


def test_small_positive_harm_inside_tolerance_still_passes_but_says_so():
    v = decide(B_PASS, _ci(0.05, 0.01, 0.09))
    assert v["label"] == "FIXED_ATTACK_HEAVY_6V6_CONFIRMED" and v["promotion"] is True
    assert v["note"] and "inside the tolerance" in v["note"]


def test_b_lcb_boundary_is_strict():
    assert decide(_ci(0.1, 0.0, 0.2), _ci(-0.1, -0.2, 0.0))["label"] == "FIXED_ATTACK_HEAVY_6V6_B_NOT_REPLICATED"


def test_integrity_failure_takes_precedence_over_every_scientific_label():
    for a in (_ci(-0.1, -0.2, 0.0), _ci(0.04, -0.08, 0.14), _ci(0.18, 0.06, 0.30)):
        v = decide(B_PASS, a, integrity_ok=False)
        assert v["label"] == "FIXED_ATTACK_HEAVY_6V6_ARM_IDENTITY_VIOLATED"
        assert v["promotion"] is False and v["routing_reopen_licensed"] is False
    assert decide(B_FAIL, _ci(0, -1, 1), integrity_ok=False)["label"].endswith("ARM_IDENTITY_VIOLATED")


def _row(steps, defend, attack, switches=0):
    return {"steps": steps, "defend_ticks": defend, "attack_ticks": attack, "role_switch_count": switches}


def test_arm_identity_accepts_correct_role_telemetry():
    rows = {("A", "5A_1D", 1): _row(240, 240, 1200), ("B", "3A_3D", 1): _row(240, 720, 720),
            ("A", "3A_3D", 2): _row(200, 600, 600)}       # early termination scales with steps
    assert arm_identity_violations(rows) == []


def test_arm_identity_catches_a_mislabelled_arm():
    rows = {("A", "5A_1D", 1): _row(240, 720, 720)}        # a 3A/3D episode labelled 5A/1D
    assert arm_identity_violations(rows) == ["A/5A_1D/1"]


def test_arm_identity_catches_a_role_switch():
    assert arm_identity_violations({("B", "3A_3D", 1): _row(240, 720, 720, switches=1)}) == ["B/3A_3D/1"]
