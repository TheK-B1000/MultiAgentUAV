"""Regression tests for the scanner's opponent-set generalization.

The refactor is infrastructure, not a methodological change, and the acceptance
rule is an equality:

    old scanner + historical roster  ==  generalized scanner + historical registry

Every test below exists to hold one half of that equality, or to keep an SRCTF
block unreachable until the reachability guards pass.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import pytest

from srctf import opponent_sets, registry

ROOT = Path(__file__).resolve().parents[1]


def test_historical_set_matches_the_source_of_truth_in_order():
    """Order matters: cell order defines the parallel merge order."""
    from experiments.run_g0_v2_seed import OPPONENTS
    assert opponent_sets.get("historical") == tuple(OPPONENTS) == opponent_sets.HISTORICAL


def test_historical_drift_fails_closed(monkeypatch):
    import experiments.run_g0_v2_seed as seed_mod
    monkeypatch.setattr(seed_mod, "OPPONENTS", ("OP7", "OP6", "OP8", "OP9",
                                                "OP10", "OP11", "OP12"))
    with pytest.raises(opponent_sets.OpponentSetError, match="drifted"):
        opponent_sets.get("historical")


def test_ordering_is_frozen_not_derived_from_a_set():
    """Registry insertion order must not be able to reorder cells."""
    assert opponent_sets.get("historical") == opponent_sets.get("historical")
    assert isinstance(opponent_sets.HISTORICAL, tuple)
    assert isinstance(opponent_sets.SRCTF, tuple)
    # a set of the same names must NOT be what determines order
    assert opponent_sets.SRCTF != tuple(sorted(registry.SRCTF_CANONICAL_OPPONENTS))


def test_cell_list_is_unchanged_by_the_refactor():
    """The (policy, opponent) cell list drives sharding and the merge order."""
    G0 = [3200001, 3200002, 3200003]
    from experiments.run_g0_v2_seed import OPPONENTS
    old = [(p, o) for p in G0 for o in list(OPPONENTS)]
    new = [(p, o) for p in G0 for o in list(opponent_sets.get("historical"))]
    assert new == old
    assert len(new) == 21


def test_unknown_set_rejected():
    with pytest.raises(opponent_sets.OpponentSetError, match="unknown opponent set"):
        opponent_sets.get("something_else")


# ------------------------------------------------------- SRCTF gating -------

def test_srctf_set_is_unreachable_while_content_is_deferred():
    """No SRCTF block may be touched before opponents exist and are admitted."""
    registry._clear_for_tests()
    with pytest.raises(opponent_sets.OpponentSetError, match="not registered"):
        opponent_sets.get("srctf")


def test_srctf_set_requires_an_admission_guard(monkeypatch):
    """Even fully registered, the set is refused without the six-condition guard."""
    registry._clear_for_tests()
    import dataclasses

    @dataclasses.dataclass(frozen=True)
    class P:
        name: str

    for n in opponent_sets.SRCTF:
        prof = P(name=n)
        registry.register(n, prof,
                          expected_profile_sha256=registry.profile_sha256(prof),
                          expected_trajectory_sha256="t" * 64)
    with pytest.raises(opponent_sets.OpponentSetError, match="admission guard"):
        opponent_sets.get("srctf")

    called = {"n": 0}

    def guard():
        called["n"] += 1

    assert opponent_sets.get("srctf", admit_guard=guard) == opponent_sets.SRCTF
    assert called["n"] == 1
    registry._clear_for_tests()


def test_a_failing_admission_guard_blocks_the_set():
    registry._clear_for_tests()
    import dataclasses

    @dataclasses.dataclass(frozen=True)
    class P:
        name: str

    for n in opponent_sets.SRCTF:
        prof = P(name=n)
        registry.register(n, prof,
                          expected_profile_sha256=registry.profile_sha256(prof),
                          expected_trajectory_sha256="t" * 64)

    def guard():
        raise registry.SRCTFRegistryError("dispatch unreachable")

    with pytest.raises(registry.SRCTFRegistryError, match="dispatch unreachable"):
        opponent_sets.get("srctf", admit_guard=guard)
    registry._clear_for_tests()


# ------------------------------------- scientific semantics unchanged -------

def test_candidate_key_still_carries_the_opponent_pair():
    """The C5 collision class must not return via the refactor."""
    import json as _json

    import experiments.run_c4_opportunity_cost as M

    frozen = _json.loads(
        (ROOT / "artifacts/c4_preregistration/C4_OPPORTUNITY_COST_FROZEN.json")
        .read_text(encoding="utf-8"))

    def mk(side, r1, r2, n=40, off=0):
        return [{"episode_key": f"{side}:{off+i}", "step": i, "classes": {"opponent": side},
                 "utilities": {"(0, 0)": r1, "(1, 1)": r2}} for i in range(n)]

    M.PARTITIONS = {"opponent": lambda c: None}
    rows = (mk("OP6", 1.0, 0.0) + mk("OP7", 0.0, 1.0, off=100)
            + mk("OP8", 0.0, 1.0, off=200) + mk("OP9", 1.0, 0.0, off=300))
    res = M.analyze({1: rows, 2: rows, 3: rows}, frozen)
    keys = sorted(res["replicated"])
    assert len(keys) == 4, "distinct opponent pairs collapsed into one candidate"
    assert all(len(k.split("|")) == 5 for k in keys), "opponent pair missing from the key"


def test_discovery_searches_pairs_but_confirmation_does_not():
    """Confirmation must accept exactly one frozen candidate and never enumerate."""
    import ast

    def called_names(path):
        """Function names actually CALLED in code -- not mentions in docstrings."""
        tree = ast.parse(path.read_text(encoding="utf-8"))
        out = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                f = node.func
                if isinstance(f, ast.Name):
                    out.add(f.id)
                elif isinstance(f, ast.Attribute):
                    out.add(f.attr)
        return out

    conf = called_names(ROOT / "experiments/run_c5_confirmation.py")
    assert "analyze" not in conf, "confirmation must never call the search routine"
    assert "combinations" not in conf, "confirmation must never enumerate pairs"
    assert "selected_candidate" in (
        ROOT / "experiments/run_c5_confirmation.py").read_text(encoding="utf-8")

    # discovery, by contrast, genuinely enumerates
    scan = called_names(ROOT / "experiments/run_c4_opportunity_cost.py")
    assert "combinations" in scan


def test_merge_is_ordered_by_cells_not_completion():
    src = (ROOT / "experiments/run_c5_parallel.py").read_text(encoding="utf-8")
    assert "for pseed, opp in cells:" in src, "merge must iterate the fixed cell list"
    assert "SHARED_KEYS" in src and "duplicate cell" in src


def test_worker_inherits_the_opponent_set():
    """A shard must be computed under the same registry as the launcher."""
    src = (ROOT / "experiments/run_c5_parallel.py").read_text(encoding="utf-8")
    assert '"--opponent-set", args.opponent_set,' in src
