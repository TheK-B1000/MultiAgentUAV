import pytest

from experiments import phase0_collect_scorer_data as P0
from experiments import phase0_full_scorer_collection as C
from experiments.phase0_full_scorer_collection import _preflight


def test_frozen_source_assignment_is_128_128_per_pole():
    seeds = range(P0.SEED_BASE, P0.SEED_BASE + P0.N_SEEDS)
    for pole in ("A", "B"):
        counts = {
            policy: sum(P0.source_policy_for(seed, pole) == policy for seed in seeds)
            for policy in ("pi_A", "pi_B")
        }
        assert counts == {"pi_A": 128, "pi_B": 128}


def test_rebuild_collector_preflight_requires_and_accepts_frozen_evidence(tmp_path, monkeypatch):
    """The frozen evidence gates accept a launch into an EMPTY output directory.

    Pointed at a temp directory, not the real one: the real collection has run, so the
    real preflight now (correctly) refuses -- that is the next test, not a failure of this.
    """
    monkeypatch.setattr(C, "FULL", tmp_path / "full_collection_rebuild_per_branch")
    _preflight()


def test_rebuild_collector_preflight_refuses_a_second_collection(tmp_path, monkeypatch):
    """One-shot guard: once COLLECTION_COMPLETE.json exists the launcher must refuse."""
    full = tmp_path / "full_collection_rebuild_per_branch"
    full.mkdir()
    (full / "COLLECTION_COMPLETE.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(C, "FULL", full)
    with pytest.raises(RuntimeError, match="already completed"):
        _preflight()


def test_frozen_collection_cardinality():
    assert P0.N_SEEDS * 4 == 1024
    assert P0.N_SEEDS * 2 * P0.BRANCHES_PER_SOURCE == 1536
    assert P0.N_TRAIN_SEEDS == 160
    assert P0.N_HELDOUT_SEEDS == 96
