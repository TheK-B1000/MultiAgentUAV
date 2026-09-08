"""Generalising B_distinct's family names must not move the k2 numbers.

`analyze_k2_behavior_gate.b_distinct` used to iterate the hardcoded tuple
("piR", "piS") in its bootstrap branch. The O1 gate reuses the same statistic
with families G0 / O1, so the loop now reads the family names out of ``drawn``.

That is only acceptable if it is a strict generalisation. The load-bearing test
here is the randomised one: 200 trials against a verbatim copy of the old
implementation, covering repeated draws, degenerate self-pairs and missing
pairs.

The k2-audit test is a bonus and currently SKIPS: that audit was cancelled
during its first cell with zero completed cells
(``CANCELLED_DIAGNOSTIC.json``), so ``divergence_episodes.csv`` is empty. There
is no recorded k2 result for the change to move. If a future audit populates
it, the test starts running by itself.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

np = pytest.importorskip("numpy")

from experiments.analyze_k2_behavior_gate import (  # noqa: E402
    b_distinct,
    family_of,
    load,
    pair_values,
)

AUDIT_CSV = PROJECT_ROOT / "artifacts" / "k2v2_specialist_behavior_audit" / "divergence_episodes.csv"


def _hardcoded_b_distinct(pairs: dict, drawn: dict):
    """The pre-change implementation, verbatim, for comparison."""
    def look(a, b):
        return pairs.get((a, b), pairs.get((b, a)))

    within, between = [], []
    for fam in ("piR", "piS"):
        d = drawn[fam]
        for i in range(len(d)):
            for j in range(i + 1, len(d)):
                if d[i] == d[j]:
                    continue
                v = look(d[i], d[j])
                if v is not None:
                    within.append(v)
    for a in drawn["piR"]:
        for b in drawn["piS"]:
            v = look(a, b)
            if v is not None:
                between.append(v)
    if not within or not between:
        nan = float("nan")
        return nan, nan, nan
    med_b = float(np.median(between))
    q_w = float(np.quantile(within, 0.95))
    return med_b - q_w, med_b, q_w


def _same(a, b) -> bool:
    return all(
        (np.isnan(x) and np.isnan(y)) or x == pytest.approx(y, abs=1e-12)
        for x, y in zip(a, b)
    )


def test_matches_hardcoded_version_on_randomised_draws():
    rng = random.Random(20260804)
    keys = {"piR": [f"piR/s{i}" for i in range(4)], "piS": [f"piS/s{i}" for i in range(4)]}
    flat = keys["piR"] + keys["piS"]

    for _trial in range(200):
        pairs = {}
        for i in range(len(flat)):
            for j in range(i + 1, len(flat)):
                if rng.random() < 0.9:  # leave gaps so `look` returns None sometimes
                    pairs[(flat[i], flat[j])] = rng.random()
        drawn = {f: [rng.choice(v) for _ in v] for f, v in keys.items()}
        assert _same(b_distinct(pairs, drawn), _hardcoded_b_distinct(pairs, drawn))


def _audit_steps() -> list[int]:
    import csv

    if not AUDIT_CSV.is_file() or AUDIT_CSV.stat().st_size == 0:
        return []
    with open(AUDIT_CSV, newline="", encoding="utf-8") as f:
        return sorted({int(r["checkpoint_step"]) for r in csv.DictReader(f)})


def test_matches_hardcoded_version_on_the_recorded_k2_audit():
    steps = _audit_steps()
    if not steps:
        pytest.skip(
            "k2 audit CSV is empty: that run was cancelled during its first "
            "cell with zero completed cells, so there is no recorded result "
            "for this change to move"
        )

    rng = np.random.default_rng(0)
    checked = 0
    for step in steps:
        rows = load(AUDIT_CSV, step, "jsd_all_bits")
        if not rows:
            continue
        pairs = pair_values(rows, balanced=True)
        keys = sorted({k for pr in pairs for k in pr})
        fams = {f: [k for k in keys if family_of(k) == f] for f in ("piR", "piS")}
        if not (fams["piR"] and fams["piS"]):
            continue
        for _ in range(50):
            drawn = {f: [v[i] for i in rng.integers(0, len(v), len(v))]
                     for f, v in fams.items()}
            assert _same(b_distinct(pairs, drawn), _hardcoded_b_distinct(pairs, drawn))
            checked += 1
    assert checked > 0, "no comparable draws were exercised"


def test_rejects_anything_other_than_two_families():
    pairs = {("A/s1", "B/s1"): 0.5}
    with pytest.raises(ValueError, match="exactly two families"):
        b_distinct(pairs, {"A": ["A/s1"], "B": ["B/s1"], "C": ["C/s1"]})


def test_point_estimate_path_is_family_name_agnostic():
    """The drawn=None branch already used family_of(); confirm G0/O1 work."""
    pairs = {
        ("G0/s1", "G0/s2"): 0.10,
        ("O1/s1", "O1/s2"): 0.12,
        ("G0/s1", "O1/s1"): 0.40,
        ("G0/s2", "O1/s2"): 0.44,
    }
    bd, med_between, q95_within = b_distinct(pairs)
    assert med_between == pytest.approx(0.42)
    assert 0.10 <= q95_within <= 0.12
    assert bd > 0
